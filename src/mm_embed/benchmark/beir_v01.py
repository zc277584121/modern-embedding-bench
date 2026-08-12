"""Reproducible BEIR benchmark v0.1 materialization and baselines."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import resource
import shutil
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


BENCHMARK_VERSION = "beir-three-track-v0.1"
TEXT_PROTOCOL = "title.strip() + ('\\n' if title and text else '') + text.strip()"
TOKENIZER_ID = "unicode-word-lower-v1"
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
TRACKS: dict[str, dict[str, Any]] = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "md5": "5f7d1de60b170fc8027bb7898e2efca1",
        "sha256": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "archive_bytes": 2816079,
        "docs": 5183,
        "queries": 300,
        "qrels": 339,
        "license_status": "publish_review_required",
    },
    "nfcorpus": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "md5": "a89dba18a62ef92f7d323ec890a0d38d",
        "sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "archive_bytes": 2448432,
        "docs": 3633,
        "queries": 323,
        "qrels": 12334,
        "license_status": "publish_review_required",
    },
    "fiqa": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        "md5": "17918ed23cd04fb15047f73e6c3bd9d9",
        "sha256": "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
        "archive_bytes": 17948027,
        "docs": 57638,
        "queries": 648,
        "qrels": 1706,
        "license_status": "publish_review_required",
    },
}


class BeirBenchmarkError(ValueError):
    """Fail-closed benchmark validation error."""


@dataclass(frozen=True)
class TrackData:
    name: str
    corpus: tuple[dict[str, str], ...]
    queries: tuple[dict[str, str], ...]
    qrels: dict[str, dict[str, int]]


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    count = 0
    hasher = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            content = _json_bytes(row)
            handle.write(content)
            hasher.update(content)
            count += 1
    return count, hasher.hexdigest()


def canonical_text(title: Any, text: Any) -> str:
    """Apply the model-independent document text protocol."""
    if not isinstance(title, str) or not isinstance(text, str):
        raise BeirBenchmarkError("document title and text must be strings")
    title = title.strip()
    text = text.strip()
    return title + ("\n" if title and text else "") + text


def _safe_members(archive: zipfile.ZipFile, track: str) -> dict[str, zipfile.ZipInfo]:
    expected = {
        f"{track}/corpus.jsonl",
        f"{track}/queries.jsonl",
        f"{track}/qrels/test.tsv",
    }
    members: dict[str, zipfile.ZipInfo] = {}
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
            raise BeirBenchmarkError(f"unsafe archive member: {item.filename}")
        if item.filename in expected:
            members[item.filename] = item
    if set(members) != expected:
        raise BeirBenchmarkError(f"archive is missing required files: {sorted(expected - set(members))}")
    return members


def _load_jsonl(handle: Any, role: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(handle, 1):
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BeirBenchmarkError(f"invalid {role} JSONL at line {line_number}") from exc
        if not isinstance(row, dict):
            raise BeirBenchmarkError(f"invalid {role} row at line {line_number}")
        rows.append(row)
    return rows


def load_archive(archive_path: str | Path, track: str) -> TrackData:
    """Load one fixed BEIR archive directly without trusting extracted files."""
    if track not in TRACKS:
        raise BeirBenchmarkError(f"unknown track: {track}")
    path = Path(archive_path)
    expected = TRACKS[track]
    if not path.is_file() or path.stat().st_size != expected["archive_bytes"]:
        raise BeirBenchmarkError(f"archive missing or byte size mismatch: {path}")
    if _digest(path, "md5") != expected["md5"] or _digest(path, "sha256") != expected["sha256"]:
        raise BeirBenchmarkError(f"archive hash mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        members = _safe_members(archive, track)
        corpus_raw = _load_jsonl(archive.open(members[f"{track}/corpus.jsonl"]), "corpus")
        queries_raw = _load_jsonl(archive.open(members[f"{track}/queries.jsonl"]), "queries")
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        with archive.open(members[f"{track}/qrels/test.tsv"]) as handle:
            header = handle.readline().decode("utf-8").rstrip("\r\n").split("\t")
            if header != ["query-id", "corpus-id", "score"]:
                raise BeirBenchmarkError(f"unexpected qrels header for {track}")
            for line_number, raw in enumerate(handle, 2):
                fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
                if len(fields) != 3:
                    raise BeirBenchmarkError(f"invalid qrels row at line {line_number}")
                query_id, doc_id, grade_raw = fields
                try:
                    grade = int(grade_raw)
                except ValueError as exc:
                    raise BeirBenchmarkError(f"non-integer qrel grade at line {line_number}") from exc
                if grade <= 0 or doc_id in qrels[query_id]:
                    raise BeirBenchmarkError(f"invalid or duplicate qrel at line {line_number}")
                qrels[query_id][doc_id] = grade
    corpus = []
    doc_ids: set[str] = set()
    for row in corpus_raw:
        doc_id = row.get("_id")
        if not isinstance(doc_id, str) or not doc_id or doc_id in doc_ids:
            raise BeirBenchmarkError(f"invalid or duplicate document id in {track}")
        content = canonical_text(row.get("title", ""), row.get("text", ""))
        doc_ids.add(doc_id)
        corpus.append({"id": doc_id, "title": row.get("title", "").strip(), "text": row.get("text", "").strip(), "content": content})
    query_by_id: dict[str, str] = {}
    for row in queries_raw:
        query_id, text = row.get("_id"), row.get("text")
        if not isinstance(query_id, str) or not query_id or query_id in query_by_id or not isinstance(text, str):
            raise BeirBenchmarkError(f"invalid or duplicate query in {track}")
        query_by_id[query_id] = text.strip()
    qrel_query_ids = set(qrels)
    missing_docs = sorted({doc_id for rels in qrels.values() for doc_id in rels} - doc_ids)
    missing_queries = sorted(qrel_query_ids - set(query_by_id))
    if missing_docs or missing_queries:
        raise BeirBenchmarkError(f"dangling qrels in {track}: queries={missing_queries[:3]} docs={missing_docs[:3]}")
    queries = tuple({"id": query_id, "text": query_by_id[query_id]} for query_id in sorted(qrel_query_ids))
    if len(corpus) != expected["docs"] or len(queries) != expected["queries"] or sum(map(len, qrels.values())) != expected["qrels"]:
        raise BeirBenchmarkError(f"fixed size mismatch for {track}")
    return TrackData(track, tuple(sorted(corpus, key=lambda row: row["id"])), queries, dict(qrels))


def _percentiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {key: float(value) for key, value in zip(("min", "p50", "p90", "p95", "p99", "max", "mean"), np.percentile(array, [0, 50, 90, 95, 99, 100]).tolist() + [array.mean()])}


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def audit_track(data: TrackData) -> dict[str, Any]:
    doc_tokens = [len(_tokens(row["content"])) for row in data.corpus]
    query_tokens = [len(_tokens(row["text"])) for row in data.queries]
    grades = Counter(grade for rels in data.qrels.values() for grade in rels.values())
    normalized = defaultdict(list)
    for row in data.corpus:
        normalized[" ".join(_tokens(row["content"]))].append(row["id"])
    exact_duplicate_groups = [ids for text, ids in normalized.items() if text and len(ids) > 1]
    fingerprints: dict[str, set[int]] = {}
    buckets: defaultdict[int, list[str]] = defaultdict(list)
    for row in data.corpus:
        toks = _tokens(row["content"])
        shingles = {hashlib.blake2b(" ".join(toks[i : i + 5]).encode(), digest_size=8).digest() for i in range(max(1, len(toks) - 4))}
        values = {int.from_bytes(value, "big") for value in shingles}
        fingerprints[row["id"]] = values
        for value in sorted(values)[:3]:
            buckets[value].append(row["id"])
    candidates: set[tuple[str, str]] = set()
    for ids in buckets.values():
        if len(ids) <= 25:
            candidates.update((a, b) for index, a in enumerate(ids) for b in ids[index + 1 :])
    near_pairs = []
    for left, right in sorted(candidates):
        a, b = fingerprints[left], fingerprints[right]
        similarity = len(a & b) / max(1, len(a | b))
        if similarity >= 0.9:
            near_pairs.append({"left": left, "right": right, "jaccard_5gram": round(similarity, 6)})
    query_texts = {" ".join(_tokens(row["text"])) for row in data.queries}
    corpus_texts = set(normalized)
    return {
        "track": data.name,
        "documents": len(data.corpus),
        "queries": len(data.queries),
        "qrels": sum(map(len, data.qrels.values())),
        "qrels_per_query": _percentiles([len(data.qrels[row["id"]]) for row in data.queries]),
        "grade_counts": dict(sorted(grades.items())),
        "document_token_lengths": _percentiles(doc_tokens),
        "query_token_lengths": _percentiles(query_tokens),
        "empty_documents": sum(not row["content"] for row in data.corpus),
        "empty_document_ids": [row["id"] for row in data.corpus if not row["content"]],
        "qrels_to_empty_documents": sum(
            doc_id in {row["id"] for row in data.corpus if not row["content"]}
            for rels in data.qrels.values()
            for doc_id in rels
        ),
        "empty_queries": sum(not row["text"] for row in data.queries),
        "missing_references": 0,
        "exact_duplicate_groups": exact_duplicate_groups,
        "near_duplicate_pairs": near_pairs[:1000],
        "near_duplicate_detection": "deterministic candidate buckets plus 5-token-shingle Jaccard >= 0.9; heuristic, not exhaustive",
        "exact_query_document_text_overlap": len(query_texts & corpus_texts),
        "risk_notes": [
            "BEIR test qrels are sparse and unjudged documents are not confirmed negatives.",
            "Baseline high-ranked unjudged documents are only missing-label candidates; qrels are never changed.",
            "Model training overlap is not established and results must not be described as verified zero-shot.",
        ],
    }


def materialize(cache_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Create deterministic per-track JSONL files and an audit manifest."""
    cache = Path(cache_root)
    output = Path(output_root)
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": "1",
        "benchmark_version": BENCHMARK_VERSION,
        "text_protocol": TEXT_PROTOCOL,
        "source_family": "UKP BEIR archives",
        "publish_gate": "closed_pending_dataset_specific_license_and_redistribution_review",
        "tracks": {},
    }
    for track in TRACKS:
        data = load_archive(cache / f"{track}.zip", track)
        track_dir = staging / track
        track_dir.mkdir()
        files = {}
        for name, rows in (
            ("corpus.jsonl", data.corpus),
            ("queries.jsonl", data.queries),
            ("qrels.jsonl", ({"query_id": query_id, "doc_id": doc_id, "grade": grade} for query_id in sorted(data.qrels) for doc_id, grade in sorted(data.qrels[query_id].items()))),
        ):
            count, sha256 = _write_jsonl(track_dir / name, rows)
            files[name] = {"rows": count, "bytes": (track_dir / name).stat().st_size, "sha256": sha256}
        audit = audit_track(data)
        (track_dir / "audit.json").write_bytes(_json_bytes(audit))
        files["audit.json"] = {"bytes": (track_dir / "audit.json").stat().st_size, "sha256": _digest(track_dir / "audit.json", "sha256")}
        manifest["tracks"][track] = {**TRACKS[track], "files": files}
    manifest_path = staging / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    return manifest


def load_materialized(root: str | Path, track: str) -> TrackData:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("benchmark_version") != BENCHMARK_VERSION or manifest.get("publish_gate") != "closed_pending_dataset_specific_license_and_redistribution_review":
        raise BeirBenchmarkError("invalid benchmark manifest")
    expected_files = manifest["tracks"][track]["files"]
    for name, expected in expected_files.items():
        path = root / track / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or _digest(path, "sha256") != expected["sha256"]:
            raise BeirBenchmarkError(f"materialized file validation failed: {track}/{name}")
    with (root / track / "corpus.jsonl").open(encoding="utf-8") as handle:
        corpus = tuple(json.loads(line) for line in handle)
    with (root / track / "queries.jsonl").open(encoding="utf-8") as handle:
        queries = tuple(json.loads(line) for line in handle)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (root / track / "qrels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qrels[row["query_id"]][row["doc_id"]] = row["grade"]
    return TrackData(track, corpus, queries, dict(qrels))


def evaluate_rankings(data: TrackData, rankings: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    metrics: defaultdict[str, float] = defaultdict(float)
    for query in data.queries:
        query_id = query["id"]
        ranked = [doc_id for doc_id, _ in rankings[query_id]]
        rels = data.qrels[query_id]
        gains = [rels.get(doc_id, 0) for doc_id in ranked]
        dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains[:10]))
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
        metrics["ndcg@10"] += dcg / idcg if idcg else 0.0
        binary = [gain > 0 for gain in gains]
        found = 0
        ap = 0.0
        for index, relevant in enumerate(binary[:100], 1):
            if relevant:
                found += 1
                ap += found / index
        metrics["map@100"] += ap / len(rels)
        first = next((index for index, relevant in enumerate(binary[:10], 1) if relevant), None)
        metrics["mrr@10"] += 0.0 if first is None else 1.0 / first
        metrics["recall@10"] += sum(binary[:10]) / len(rels)
        metrics["recall@100"] += sum(binary[:100]) / len(rels)
    return {key: value / len(data.queries) for key, value in metrics.items()}


def bm25_rank(data: TrackData, top_k: int = 100, k1: float = 1.2, b: float = 0.75) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Any]]:
    """Run fixed-tokenizer exact BM25 over one independent track."""
    started = time.perf_counter()
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = np.empty(len(data.corpus), dtype=np.int32)
    for index, row in enumerate(data.corpus):
        counts = Counter(_tokens(row["content"]))
        lengths[index] = sum(counts.values())
        for token, frequency in counts.items():
            postings[token].append((index, frequency))
    build_s = time.perf_counter() - started
    avgdl = float(lengths.mean())
    rankings = {}
    search_started = time.perf_counter()
    for query in data.queries:
        scores: defaultdict[int, float] = defaultdict(float)
        for token, query_frequency in Counter(_tokens(query["text"])).items():
            posting = postings.get(token, ())
            idf = math.log(1.0 + (len(data.corpus) - len(posting) + 0.5) / (len(posting) + 0.5))
            for doc_index, frequency in posting:
                denominator = frequency + k1 * (1.0 - b + b * lengths[doc_index] / avgdl)
                scores[doc_index] += query_frequency * idf * frequency * (k1 + 1.0) / denominator
        best = heapq.nlargest(top_k, scores.items(), key=lambda item: (item[1], -item[0]))
        if len(best) < top_k:
            selected = {index for index, _ in best}
            best.extend((index, 0.0) for index in range(len(data.corpus)) if index not in selected)
            best = sorted(best, key=lambda item: (-item[1], item[0]))[:top_k]
        rankings[query["id"]] = [(data.corpus[index]["id"], float(score)) for index, score in best]
    search_s = time.perf_counter() - search_started
    index_bytes = int(lengths.nbytes + sum(len(token.encode("utf-8")) + len(items) * 8 for token, items in postings.items()))
    return rankings, {"tokenizer": TOKENIZER_ID, "k1": k1, "b": b, "build_s": build_s, "search_s": search_s, "index_bytes": index_bytes}


def dense_rank(data: TrackData, model_id: str, revision: str, top_k: int = 100, query_block_size: int = 32, doc_block_size: int = 4096) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Any]]:
    """Run normalized dense exact search without a full query-document matrix."""
    from sentence_transformers import SentenceTransformer

    import torch

    model = SentenceTransformer(model_id, revision=revision, trust_remote_code=False, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    started = time.perf_counter()
    doc_embeddings = model.encode([row["content"] for row in data.corpus], batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    query_embeddings = model.encode([row["text"] for row in data.queries], batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    encode_s = time.perf_counter() - started
    rankings: dict[str, list[tuple[str, float]]] = {}
    search_started = time.perf_counter()
    max_score_block_elements = 0
    for query_start in range(0, len(data.queries), query_block_size):
        query_block = query_embeddings[query_start : query_start + query_block_size]
        heaps: list[list[tuple[float, int]]] = [[] for _ in range(len(query_block))]
        for doc_start in range(0, len(data.corpus), doc_block_size):
            scores = query_block @ doc_embeddings[doc_start : doc_start + doc_block_size].T
            max_score_block_elements = max(max_score_block_elements, scores.size)
            for row_index, row_scores in enumerate(scores):
                local = min(top_k, len(row_scores))
                indices = np.argpartition(row_scores, -local)[-local:]
                heap = heaps[row_index]
                for local_index in indices:
                    item = (float(row_scores[local_index]), doc_start + int(local_index))
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
        for offset, heap in enumerate(heaps):
            query = data.queries[query_start + offset]
            rankings[query["id"]] = [(data.corpus[index]["id"], score) for score, index in sorted(heap, key=lambda item: (-item[0], item[1]))]
    search_s = time.perf_counter() - search_started
    vram = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    return rankings, {
        "model_id": model_id,
        "revision": revision,
        "trust_remote_code": False,
        "device": device,
        "dimensions": int(doc_embeddings.shape[1]),
        "normalize_embeddings": True,
        "similarity": "exact_cosine",
        "encode_s": encode_s,
        "search_s": search_s,
        "embedding_bytes": int(doc_embeddings.nbytes + query_embeddings.nbytes),
        "query_block_size": query_block_size,
        "document_block_size": doc_block_size,
        "max_score_block_elements": max_score_block_elements,
        "full_matrix_elements": len(data.queries) * len(data.corpus),
        "peak_vram_bytes": vram,
    }


def make_audit_pack(root: str | Path, output: str | Path, rankings_by_track: dict[str, dict[str, list[tuple[str, float]]]]) -> dict[str, Any]:
    """Create a deterministic 150-query stratified review pack."""
    allocation = {"scifact": 50, "nfcorpus": 50, "fiqa": 50}
    rows = []
    for track, sample_size in allocation.items():
        data = load_materialized(root, track)
        strata: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        lengths = [len(_tokens(query["text"])) for query in data.queries]
        p33, p67 = np.percentile(lengths, [33, 67])
        for query in data.queries:
            qrel_count = len(data.qrels[query["id"]])
            density = "single" if qrel_count == 1 else ("medium" if qrel_count <= 5 else "dense")
            length = len(_tokens(query["text"]))
            length_band = "short" if length <= p33 else ("medium" if length <= p67 else "long")
            strata[(density, length_band)].append(query)
        selected = []
        keys = sorted(strata)
        cursor = 0
        while len(selected) < sample_size:
            key = keys[cursor % len(keys)]
            candidates = strata[key]
            index = cursor // len(keys)
            if index < len(candidates):
                selected.append((key, candidates[index]))
            cursor += 1
        corpus = {row["id"]: row for row in data.corpus}
        for (density, length_band), query in selected:
            ranking = rankings_by_track[track][query["id"]]
            candidates = []
            for rank, (doc_id, score) in enumerate(ranking[:10], 1):
                candidates.append({"rank": rank, "doc_id": doc_id, "score": score, "judged_grade": data.qrels[query["id"]].get(doc_id), "content": corpus[doc_id]["content"]})
            rows.append({"track": track, "query_id": query["id"], "query": query["text"], "stratum": {"qrel_density": density, "query_length": length_band}, "qrels": data.qrels[query["id"]], "baseline_top10": candidates, "review_fields": {"qrels_complete": None, "suspected_missing_labels": [], "notes": ""}})
    output_path = Path(output)
    count, sha256 = _write_jsonl(output_path, rows)
    return {"rows": count, "sha256": sha256, "allocation": allocation, "selection": "sorted query ids, round-robin over qrel-density x query-length tertile strata; no score-based query selection"}


def _write_run(path: Path, method: str, data: TrackData, rankings: dict[str, list[tuple[str, float]]], execution: dict[str, Any]) -> dict[str, Any]:
    ranking_rows = ({"query_id": query_id, "rank": rank, "doc_id": doc_id, "score": score} for query_id in sorted(rankings) for rank, (doc_id, score) in enumerate(rankings[query_id], 1))
    ranking_path = path.with_suffix(".rankings.jsonl")
    rows, ranking_sha256 = _write_jsonl(ranking_path, ranking_rows)
    result = {
        "schema_version": "1",
        "benchmark_version": BENCHMARK_VERSION,
        "track": data.name,
        "method": method,
        "queries_evaluated": len(data.queries),
        "documents_searched": len(data.corpus),
        "metrics": evaluate_rankings(data, rankings),
        "execution": {**execution, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
        "rankings": {"path": ranking_path.name, "rows": rows, "sha256": ranking_sha256, "depth": 100},
        "training_overlap": {"status": "unknown", "zero_shot_claim_allowed": False},
    }
    path.write_bytes(_json_bytes(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--cache-root", required=True)
    materialize_parser.add_argument("--output", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--data", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--method", choices=("bm25", "dense"), required=True)
    run_parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    run_parser.add_argument("--revision", default="c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
    audit_parser = subparsers.add_parser("audit-pack")
    audit_parser.add_argument("--data", required=True)
    audit_parser.add_argument("--bm25-results", required=True)
    audit_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "materialize":
        manifest = materialize(args.cache_root, args.output)
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "run":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for track in TRACKS:
            data = load_materialized(args.data, track)
            rankings, execution = bm25_rank(data) if args.method == "bm25" else dense_rank(data, args.model, args.revision)
            result = _write_run(output / f"{args.method}-{track}.json", args.method, data, rankings, execution)
            print(json.dumps({"track": track, "metrics": result["metrics"], "execution": result["execution"]}))
        return 0
    rankings_by_track = {}
    results_root = Path(args.bm25_results)
    for track in TRACKS:
        rankings: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
        with (results_root / f"bm25-{track}.rankings.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rankings[row["query_id"]].append((row["doc_id"], row["score"]))
        rankings_by_track[track] = dict(rankings)
    print(json.dumps(make_audit_pack(args.data, args.output, rankings_by_track), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
