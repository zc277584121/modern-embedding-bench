"""Run the research-only BRIGHT biology learned-sparse pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from mm_embed.providers import get_provider
from mm_embed.providers.bge_m3_sparse_provider import SNAPSHOT_IDENTITY as BGE_M3_SNAPSHOT_IDENTITY
from mm_embed.providers.opensearch_neural_sparse_provider import SNAPSHOT_IDENTITY as OPENSEARCH_SNAPSHOT_IDENTITY


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def stats(values: list[int]) -> dict:
    array = np.asarray(values)
    return {"count": len(values), "total": int(array.sum()), "mean": float(array.mean()),
            "p50": float(np.quantile(array, 0.50)), "p95": float(np.quantile(array, 0.95)),
            "p99": float(np.quantile(array, 0.99)), "max": int(array.max())}


def metrics(rankings: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    totals = defaultdict(float)
    for query_id, relevant in qrels.items():
        ranked = rankings[query_id]
        binary = [int(item in relevant) for item in ranked]
        dcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(binary[:10]))
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(10, len(relevant))))
        totals["ndcg@10"] += dcg / ideal
        first = next((rank + 1 for rank, value in enumerate(binary[:10]) if value), None)
        totals["mrr@10"] += 0.0 if first is None else 1.0 / first
        hits = 0
        precision_sum = 0.0
        for rank, value in enumerate(binary[:100], start=1):
            if value:
                hits += 1
                precision_sum += hits / rank
        totals["map@100"] += precision_sum / len(relevant)
        totals["recall@10"] += sum(binary[:10]) / len(relevant)
        totals["recall@100"] += sum(binary[:100]) / len(relevant)
    return {name: value / len(qrels) for name, value in totals.items()}


def load_chunk_audits(chunk_dir: Path, document_count: int, chunk_size: int) -> dict[str, float | int]:
    """Validate the complete chunk set and rebuild resumable aggregate evidence."""
    expected_starts = list(range(0, document_count, chunk_size))
    expected_names = {f"{start:06d}" for start in expected_starts}
    actual_names = {path.stem for path in chunk_dir.glob("*.npz")} | {path.stem for path in chunk_dir.glob("*.json")}
    if actual_names != expected_names:
        raise ValueError("Document chunk set is incomplete or contains unexpected chunks")
    totals: dict[str, float | int] = {"count": 0, "latency_ms": 0.0, "truncated_count": 0, "peak_vram_bytes": 0}
    for start in expected_starts:
        matrix_path = chunk_dir / f"{start:06d}.npz"
        audit_path = chunk_dir / f"{start:06d}.json"
        if not matrix_path.is_file() or not audit_path.is_file():
            raise ValueError(f"Document chunk {start} is missing its matrix or audit")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        required = {"start", "count", "latency_ms", "truncated_count", "peak_vram_bytes", "sha256"}
        if not required <= audit.keys():
            raise ValueError(f"Document chunk {start} audit is incomplete")
        expected_count = min(chunk_size, document_count - start)
        if audit["start"] != start or audit["count"] != expected_count or audit["sha256"] != sha256(matrix_path):
            raise ValueError(f"Document chunk {start} audit does not match its expected matrix")
        if audit["latency_ms"] < 0 or audit["truncated_count"] < 0 or audit["peak_vram_bytes"] < 0:
            raise ValueError(f"Document chunk {start} audit contains invalid aggregate evidence")
        totals["count"] += audit["count"]
        totals["latency_ms"] += audit["latency_ms"]
        totals["truncated_count"] += audit["truncated_count"]
        totals["peak_vram_bytes"] = max(totals["peak_vram_bytes"], audit["peak_vram_bytes"])
    if totals["count"] != document_count:
        raise ValueError("Document chunk audits do not cover the full corpus")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bge-m3", "opensearch-v3"], required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/bright-nontechnical-pilot-v0.1/biology"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((args.data / "manifest.json").read_text())
    if source_manifest["track"] != "biology" or source_manifest["publication"]["public_export_allowed"]:
        raise ValueError("Pilot requires the canonical research-only biology materialization")
    documents = read_jsonl(args.data / "corpus.jsonl")
    queries = read_jsonl(args.data / "queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for line in (args.data / "qrels/test.tsv").read_text().splitlines()[1:]:
        query_id, document_id, _ = line.split("\t")
        qrels[query_id].add(document_id)
    provider_name = "bge_m3_sparse" if args.model == "bge-m3" else "opensearch_neural_sparse"
    provider_kwargs = {"snapshot_path": args.snapshot, "device": args.device,
                       "max_length": args.max_length, "batch_size": args.batch_size}
    if args.model == "opensearch-v3":
        provider_kwargs["allow_download"] = False
    provider = get_provider(provider_name, **provider_kwargs)
    chunk_dir = args.output / "document_chunks"
    chunk_dir.mkdir(exist_ok=True)
    encode_started = time.perf_counter()
    for start in range(0, len(documents), args.chunk_size):
        path = chunk_dir / f"{start:06d}.npz"
        audit_path = chunk_dir / f"{start:06d}.json"
        if path.exists() and audit_path.exists():
            continue
        batch = documents[start:start + args.chunk_size]
        result = provider.encode_sparse_documents([item["content"] for item in batch], item_ids=[item["id"] for item in batch])
        sparse.save_npz(path, result.embeddings.values, compressed=True)
        meta = result.metadata_dict()
        audit_path.write_text(json.dumps({"start": start, "count": len(batch), "latency_ms": result.latency_ms,
                                          "truncated_count": int(meta.get("truncated_count", 0)),
                                          "peak_vram_bytes": int(result.peak_vram_bytes or 0),
                                          "sha256": sha256(path)}, sort_keys=True) + "\n")
    _ = time.perf_counter() - encode_started
    chunk_audit = load_chunk_audits(chunk_dir, len(documents), args.chunk_size)
    document_encoding_s = float(chunk_audit["latency_ms"]) / 1000.0
    truncated_documents = int(chunk_audit["truncated_count"])
    peak_vram = int(chunk_audit["peak_vram_bytes"])
    matrices = [sparse.load_npz(chunk_dir / f"{start:06d}.npz") for start in range(0, len(documents), args.chunk_size)]
    document_matrix = sparse.vstack(matrices, format="csr")
    if document_matrix.shape != (57359, provider.representation.dimensions):
        raise ValueError(f"Unexpected document matrix shape: {document_matrix.shape}")
    sparse.save_npz(args.output / "documents.npz", document_matrix, compressed=True)
    (args.output / "document_ids.json").write_text(json.dumps([item["id"] for item in documents]) + "\n")
    query_started = time.perf_counter()
    query_result = provider._encode([item["text"] for item in queries], [item["id"] for item in queries],
                                    __import__("mm_embed.providers.sparse_base", fromlist=["SparseEmbeddingRole"]).SparseEmbeddingRole.QUERY)
    query_encoding_s = time.perf_counter() - query_started
    query_matrix = query_result.embeddings.values
    sparse.save_npz(args.output / "queries.npz", query_matrix, compressed=True)
    (args.output / "query_ids.json").write_text(json.dumps([item["id"] for item in queries]) + "\n")
    peak_vram = max(peak_vram, query_result.peak_vram_bytes or 0)
    search_started = time.perf_counter()
    ranking_ids: dict[str, list[str]] = {}
    document_ids = [item["id"] for item in documents]
    ranking_path = args.output / "rankings.jsonl"
    with ranking_path.open("w", encoding="utf-8") as handle:
        for row, query in enumerate(queries):
            scores = (query_matrix[row] @ document_matrix.T).tocsr()
            score_map = dict(zip(scores.indices.tolist(), scores.data.tolist(), strict=True))
            indices = sorted(range(len(document_ids)), key=lambda index: (-score_map.get(index, 0.0), document_ids[index]))[:100]
            hits = [{"rank": rank, "document_id": document_ids[index], "score": float(score_map.get(index, 0.0))}
                    for rank, index in enumerate(indices, start=1)]
            ranking_ids[query["id"]] = [hit["document_id"] for hit in hits]
            handle.write(json.dumps({"query_id": query["id"], "hits": hits}, sort_keys=True) + "\n")
    search_s = time.perf_counter() - search_started
    files = [args.output / name for name in ("documents.npz", "queries.npz", "document_ids.json", "query_ids.json", "rankings.jsonl")]
    audit = {
        "schema_version": "1", "evidence_tier": "research_pilot", "track": "biology",
        "publication": {"publish": False, "leaderboard_publish": False, "public_export_allowed": False,
                        "gate": "closed", "classification": "research_only"},
        "model": {"id": provider.model, "revision": provider.revision, "provider": provider.name,
                  "snapshot_path": str(Path(args.snapshot).resolve()), "representation_id": provider.representation.representation_id,
                  "snapshot_identity_sha256": (BGE_M3_SNAPSHOT_IDENTITY if args.model == "bge-m3" else OPENSEARCH_SNAPSHOT_IDENTITY),
                  "vocabulary_id": provider.representation.vocabulary_id, "dimensions": provider.representation.dimensions,
                  "query_route": provider.query_route.value, "document_route": provider.document_route.value},
        "data": source_manifest, "policy": {"max_length": args.max_length, "batch_size": args.batch_size,
                  "truncation": "tokenizer truncation at fixed max_length", "empty_rows_allowed": True},
        "counts": {"documents": len(documents), "queries": len(queries), "positives": sum(map(len, qrels.values()))},
        "metrics": metrics(ranking_ids, qrels),
        "audit": {"document_truncated_count": truncated_documents,
                  "query_truncated_count": int(query_result.metadata_dict().get("truncated_count", 0)),
                  "document_empty_count": int(np.sum(np.diff(document_matrix.indptr) == 0)),
                  "query_empty_count": int(np.sum(np.diff(query_matrix.indptr) == 0)),
                  "document_non_finite_count": int(np.sum(~np.isfinite(document_matrix.data))),
                  "query_non_finite_count": int(np.sum(~np.isfinite(query_matrix.data)))},
        "nnz": {"documents": stats(np.diff(document_matrix.indptr).tolist()),
                "queries": stats(np.diff(query_matrix.indptr).tolist())},
        "timing": {"document_encoding_s": document_encoding_s, "document_throughput_per_s": len(documents) / document_encoding_s,
                   "document_latency_ms_per_item": document_encoding_s * 1000 / len(documents),
                   "query_encoding_s": query_encoding_s, "query_throughput_per_s": len(queries) / query_encoding_s,
                   "query_latency_ms_per_item": query_encoding_s * 1000 / len(queries),
                   "exact_search_s": search_s, "exact_search_latency_ms_per_query": search_s * 1000 / len(queries)},
        "resources": {"peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "peak_vram_bytes": peak_vram,
                      "csr_uncompressed_bytes": int(document_matrix.data.nbytes + document_matrix.indices.nbytes + document_matrix.indptr.nbytes + query_matrix.data.nbytes + query_matrix.indices.nbytes + query_matrix.indptr.nbytes)},
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
        "search": {"backend": "scipy_csr_exact", "exact": True, "top_k": 100, "tie_break": "document_id_ascending"},
    }
    (args.output / "manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
