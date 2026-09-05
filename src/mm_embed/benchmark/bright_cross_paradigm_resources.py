"""Unified frozen resource measurements for the S-009 method matrix."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub.constants import HF_HUB_CACHE

from mm_embed.benchmark import bright_learned_sparse_batch_a as sparse_a
from mm_embed.benchmark import bright_learned_sparse_batch_b as sparse_b
from mm_embed.benchmark import bright_multi_vector as multi_benchmark
from mm_embed.benchmark.bright_cross_paradigm import (
    ARTIFACT_ROOT,
    ROOT,
    STORY_ID,
    SYNTHETIC_DOCUMENTS,
    SYNTHETIC_QUERY,
    canonical_bytes,
    sha256_file,
    verify_freeze,
)
from mm_embed.benchmark.bright_cross_paradigm_package import METHODS
from mm_embed.benchmark.bright_cross_paradigm_run import RESULT_ROOT, validate_formal_cell
from mm_embed.benchmark.bright_multidomain_v02 import (
    LONG_DENSE_BATCH_SIZE,
    LONG_DENSE_MODEL_ID,
    LONG_DENSE_MODEL_REVISION,
    LONG_DENSE_SELECTED_CAP,
    LONG_DENSE_SNAPSHOT_PATH,
    _load_result,
    load_materialized,
    verify_long_dense_preflight,
)
from mm_embed.benchmark.retrieval_v01 import tokens
from mm_embed.providers.learned_sparse_inventory import BATCH_B_INVENTORY
from mm_embed.providers.real_multi_vector import INVENTORY as MULTI_INVENTORY
from mm_embed.providers.real_multi_vector import SentenceTransformerMultiVectorProvider

RESOURCE_ROOT = RESULT_ROOT / "resources"
TRACKS = ("economics", "psychology")
GPU_UUID = "GPU-435154a8-568c-35a7-ee5f-da29fbf97a39"
SPARSE_METHODS = (
    "granite-30m-sparse",
    "opensearch-doc-v2-mini",
    "opensearch-doc-v3",
    "bge-m3",
    "splade-tiny",
    "opensearch-doc-v2-distill",
    "opensearch-multilingual",
)
MULTI_METHODS = ("colbert-v2", "answerai-colbert-small", "gte-modern-colbert")
FAMILIES = {
    "bm25-unicode": "lexical",
    "tfidf-word-sublinear": "lexical",
    "bge-m3-dense": "dense",
    "all-minilm-l6-v2": "dense",
    **{method: "learned_sparse" for method in SPARSE_METHODS},
    **{method: "multi_vector" for method in MULTI_METHODS},
}
TOP_K = 100
TFIDF_REPRESENTATION_SCOPE = "complete_document_and_query_representation_plus_idf_for_tfidf"
MINILM_REPRESENTATION_SCOPE = "complete_float32_dense_document_and_query_vectors_only"
RETURN1_FAILED_CELLS = frozenset(
    {
        "bge-m3-dense:economics",
        "bge-m3-dense:psychology",
        "granite-30m-sparse:economics",
        "granite-30m-sparse:psychology",
        "opensearch-doc-v2-mini:psychology",
        "opensearch-doc-v3:psychology",
        "opensearch-doc-v2-distill:psychology",
        "opensearch-multilingual:economics",
    }
)


class ResourceMeasurementError(RuntimeError):
    """Raised when a frozen resource measurement cannot be completed exactly."""


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    identity = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return identity


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_seconds": float(np.quantile(array, 0.50)),
        "p95_seconds": float(np.quantile(array, 0.95)),
        "p99_seconds": float(np.quantile(array, 0.99)),
    }


def _hardware() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={GPU_UUID}",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    name, uuid, memory_mib, driver = (part.strip() for part in query.split(","))
    return {
        "gpu": name,
        "gpu_uuid": uuid,
        "gpu_memory_bytes": int(memory_mib) * 1024 * 1024,
        "driver": driver,
        "python": "3.12.9",
        "numpy": np.__version__,
    }


def _assert_gpu_idle() -> None:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={GPU_UUID}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if query:
        raise ResourceMeasurementError(
            f"cuda:0 is occupied at launch; process evidence sha256={hashlib.sha256(query.encode()).hexdigest()}"
        )


def _base(method: str, track: str, started: float) -> dict[str, Any]:
    return {
        "schema_version": "bright-cross-paradigm-resource-cell-v1",
        "story_id": STORY_ID,
        "cell_id": f"{method}:{track}",
        "method_key": method,
        "family": FAMILIES[method],
        "track": track,
        "status": "pass",
        "hardware": _hardware(),
        "network_bytes": 0,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "end_to_end_seconds": time.perf_counter() - started,
        "failure": None,
    }


def _order(scores: np.ndarray, document_ids: Sequence[str], excluded: set[str]) -> list[tuple[str, float]]:
    candidates = [index for index, document_id in enumerate(document_ids) if document_id not in excluded]
    ordered = sorted(candidates, key=lambda index: (-float(scores[index]), document_ids[index]))[:TOP_K]
    if len(ordered) != TOP_K or any(not math.isfinite(float(scores[index])) for index in ordered):
        raise ResourceMeasurementError("Exact search produced invalid top-100")
    return [(document_ids[index], float(scores[index])) for index in ordered]


def _accepted_ids(method: str, track: str) -> dict[str, list[str]]:
    formal_path = RESULT_ROOT / "cells" / f"{method}-{track}" / "rankings.json"
    if formal_path.is_file():
        rows = _json(formal_path)
        return {row["query_id"]: [hit["document_id"] for hit in row["hits"]] for row in rows}
    if method in {"bm25-unicode", "bge-m3-dense"}:
        source = "bm25" if method == "bm25-unicode" else "long_dense"
        _, rankings, _ = _load_result(
            ROOT / "results/bright-nontechnical-pilot-v0.2",
            ROOT / "data/bright-nontechnical-pilot-v0.2",
            source,
            track,
        )
        return {query_id: [document_id for document_id, _ in hits] for query_id, hits in rankings.items()}
    if method in SPARSE_METHODS:
        root = "bright-learned-sparse-batch-a" if method in SPARSE_METHODS[:4] else "bright-learned-sparse-batch-b"
        rows = _json(ROOT / "results" / root / method / track / "rankings.json")
        return {row["query_id"]: [hit["document_id"] for hit in row["hits"]] for row in rows}
    if method in MULTI_METHODS:
        rows = _json(ROOT / f"results/bright-multi-vector-v0.2/cells/{method}-{track}.rankings.json")
        return {row["query_id"]: [hit["document_id"] for hit in row["hits"]] for row in rows}
    root = RESULT_ROOT / "cells" / f"{method}-{track}"
    rows = _json(root / "rankings.json")
    return {row["query_id"]: [hit["document_id"] for hit in row["hits"]] for row in rows}


def _verify_rankings(method: str, track: str, rankings: Mapping[str, Sequence[tuple[str, float]]]) -> dict[str, Any]:
    observed = {query_id: [document_id for document_id, _ in hits] for query_id, hits in rankings.items()}
    accepted = _accepted_ids(method, track)
    if set(observed) != set(accepted):
        raise ResourceMeasurementError("Fresh unified query coverage differs from the formal quality cell")
    exact = 0
    overlaps = []
    for query_id in sorted(observed):
        exact += observed[query_id] == accepted[query_id]
        overlaps.append(len(set(observed[query_id]) & set(accepted[query_id])))
    return {
        "formal_queries": len(accepted),
        "exact_order_queries": exact,
        "mismatched_order_queries": len(accepted) - exact,
        "top100_overlap_min": min(overlaps),
        "top100_overlap_mean": float(np.mean(overlaps)),
        "top100_overlap_max": max(overlaps),
        "all_mismatches_full_top100_overlap": all(overlap == TOP_K for overlap in overlaps),
        "fresh_top100_ids_match": exact == len(accepted),
        "system_error": (
            "none"
            if exact == len(accepted)
            else "exact_order_mismatch_full_set_preserved"
            if all(overlap == TOP_K for overlap in overlaps)
            else "top100_set_and_order_mismatch"
        ),
    }


def _apply_verification(result: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    result["quality_verification"] = verification
    if verification["mismatched_order_queries"]:
        message = "Fresh unified top-100 order differs from the formal quality cell"
        result["status"] = "failed_closed"
        result["failure"] = {
            "type": "AggregateTop100Mismatch",
            "stage": "quality_verification",
            "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
            "stopping_condition_applied": True,
        }
    return result


def _batch_evidence(attempts: Any, used: Any) -> dict[str, Any]:
    values = list(attempts) if isinstance(attempts, (list, tuple)) else ([] if attempts is None else [attempts])
    return {
        "batch_used": used,
        "fallback": values or "not_applicable",
        "attempted_batches": values,
        "retry_count": max(0, len(values) - 1),
        "oom_count": max(0, len(values) - 1),
        "fallback_count": max(0, len(values) - 1),
    }


def _search_trials(
    rank_once: Callable[[], tuple[dict[str, list[tuple[str, float]]], list[float]]],
) -> tuple[dict[str, Any], dict[str, list[tuple[str, float]]]]:
    rank_once()
    rows = []
    pooled: list[float] = []
    reference = None
    for ordinal in range(1, 4):
        started = time.perf_counter()
        rankings, latencies = rank_once()
        wall = time.perf_counter() - started
        if reference is not None and rankings != reference:
            raise ResourceMeasurementError("Repeated exact search trials produced different rankings")
        reference = rankings
        pooled.extend(latencies)
        rows.append(
            {
                "ordinal": ordinal,
                "wall_seconds": wall,
                "queries_per_second": len(latencies) / wall,
                **_percentiles(latencies),
            }
        )
    throughput = [row["queries_per_second"] for row in rows]
    return (
        {
            "warmup_query_searches": 1,
            "trials_completed": 3,
            "measured_trials": rows,
            "median_queries_per_second": float(np.median(throughput)),
            **{f"pooled_{key}": value for key, value in _percentiles(pooled).items()},
        },
        reference or {},
    )


def _new_method_resource(method: str, track: str) -> dict[str, Any]:
    started = time.perf_counter()
    cell = validate_formal_cell(method, track)
    execution = cell["execution"]
    search = dict(execution["search"])
    search["trials_completed"] = len(search["measured_trials"])
    representation = {
        "paradigm": FAMILIES[method],
        "bytes": execution["representation_bytes"],
        "scope": (TFIDF_REPRESENTATION_SCOPE if method == "tfidf-word-sublinear" else MINILM_REPRESENTATION_SCOPE),
    }
    return {
        **_base(method, track, started),
        "identity": {
            "revision": execution.get("revision"),
            "snapshot_sha256": execution.get("snapshot_sha256"),
        },
        "gate": {"passed": True, "evidence": "frozen_gate_summary"},
        "model_load_seconds": execution.get("model_load_seconds", 0.0),
        "document_encoding": {
            "seconds": execution["document_encoding_seconds"],
            "throughput_items_per_second": execution["document_throughput_per_second"],
            "batch_used": execution.get("document_batch_size"),
            "fallback": execution.get("document_batch_fallback", "not_applicable"),
        },
        "query_encoding": {
            "seconds": execution["query_encoding_seconds"],
            "throughput_items_per_second": execution["query_throughput_per_second"],
            "batch_used": execution.get("query_batch_size"),
            "fallback": execution.get("query_batch_fallback", "not_applicable"),
        },
        "representation": representation,
        "build_seconds": execution["index_build_seconds"],
        "load_seconds": execution["index_load_seconds"],
        "search": search,
        "cuda_peak_vram_bytes": execution.get("peak_vram_bytes", 0),
        "quality_verification": {"exact_top100_replay": True, "system_error": "none"},
        "process_peak_rss_bytes": execution["process_peak_rss_bytes"],
        "end_to_end_seconds": execution["end_to_end_seconds"],
    }


def _bm25(method: str, track: str) -> dict[str, Any]:
    started = time.perf_counter()
    data = load_materialized(ROOT / "data/bright-nontechnical-pilot-v0.2", track)
    gate_scores = [len(set(tokens(SYNTHETIC_QUERY)) & set(tokens(text))) for text in SYNTHETIC_DOCUMENTS]
    if max(gate_scores) != gate_scores[0] or gate_scores.count(gate_scores[0]) != 1:
        raise ResourceMeasurementError("BM25 synthetic gate failed")
    build_started = time.perf_counter()
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = np.empty(len(data.corpus), dtype=np.int32)
    for index, row in enumerate(data.corpus):
        counts = Counter(tokens(row["content"]))
        lengths[index] = sum(counts.values())
        for token, frequency in counts.items():
            postings[token].append((index, frequency))
    build_seconds = time.perf_counter() - build_started
    document_ids = [row["id"] for row in data.corpus]
    avgdl = float(lengths.mean())

    def rank_once() -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        rankings = {}
        latencies = []
        for query in data.queries:
            query_started = time.perf_counter()
            scores = np.zeros(len(document_ids), dtype=np.float64)
            for token, query_frequency in Counter(tokens(query["text"])).items():
                posting = postings.get(token, ())
                idf = math.log(1.0 + (len(document_ids) - len(posting) + 0.5) / (len(posting) + 0.5))
                for index, frequency in posting:
                    denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * lengths[index] / avgdl)
                    scores[index] += query_frequency * idf * frequency * 2.2 / denominator
            rankings[query["id"]] = _order(scores, document_ids, set(query.get("excluded_ids", ())))
            latencies.append(time.perf_counter() - query_started)
        return rankings, latencies

    search, rankings = _search_trials(rank_once)
    verification = _verify_rankings(method, track, rankings)
    index_bytes = int(lengths.nbytes + sum(len(token.encode()) + len(rows) * 8 for token, rows in postings.items()))
    return _apply_verification(
        {
            **_base(method, track, started),
            "identity": {"tokenizer": "unicode-word-lower-v1", "k1": 1.2, "b": 0.75},
            "gate": {"passed": True, "queries": 1, "documents": 4},
            "model_load_seconds": 0.0,
            "document_encoding": {
                "seconds": build_seconds,
                "throughput_items_per_second": len(document_ids) / build_seconds,
                "batch_used": None,
                "fallback": "not_applicable",
            },
            "query_encoding": {
                "seconds": 0.0,
                "throughput_items_per_second": None,
                "batch_used": None,
                "fallback": "not_applicable",
            },
            "representation": {
                "paradigm": "inverted_index",
                "bytes": index_bytes,
                "scope": "estimated_postings_terms_frequencies_and_lengths",
            },
            "build_seconds": build_seconds,
            "load_seconds": 0.0,
            "search": search,
            "cuda_peak_vram_bytes": 0,
            "quality_verification": {"fresh_top100_ids_match": True, "system_error": "none"},
        },
        verification,
    )


def _dense(method: str, track: str) -> dict[str, Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    _assert_gpu_idle()
    verify_long_dense_preflight(ROOT / "data/bright-nontechnical-pilot-v0.2")
    load_started = time.perf_counter()
    model = SentenceTransformer(
        str(LONG_DENSE_SNAPSHOT_PATH),
        device="cuda:0",
        trust_remote_code=False,
        local_files_only=True,
    )
    model.max_seq_length = LONG_DENSE_SELECTED_CAP
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats(0)
    gate = model.encode(
        [SYNTHETIC_QUERY, *SYNTHETIC_DOCUMENTS],
        batch_size=5,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    gate_scores = gate[:1] @ gate[1:].T
    if int(np.argmax(gate_scores[0])) != 0 or not np.isfinite(gate).all():
        raise ResourceMeasurementError("BGE-M3 dense synthetic gate failed")
    data = load_materialized(ROOT / "data/bright-nontechnical-pilot-v0.2", track)
    doc_started = time.perf_counter()
    documents = model.encode(
        [row["content"] for row in data.corpus],
        batch_size=LONG_DENSE_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    document_seconds = time.perf_counter() - doc_started
    query_started = time.perf_counter()
    queries = model.encode(
        [row["text"] for row in data.queries],
        batch_size=LONG_DENSE_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    query_seconds = time.perf_counter() - query_started
    document_ids = [row["id"] for row in data.corpus]

    def rank_once() -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        rankings = {}
        latencies = []
        for query_row, query in zip(data.queries, queries, strict=True):
            query_started = time.perf_counter()
            scores = np.asarray(query @ documents.T, dtype=np.float32)
            rankings[query_row["id"]] = _order(scores, document_ids, set(query_row.get("excluded_ids", ())))
            latencies.append(time.perf_counter() - query_started)
        return rankings, latencies

    search, rankings = _search_trials(rank_once)
    verification = _verify_rankings(method, track, rankings)
    return _apply_verification(
        {
            **_base(method, track, started),
            "identity": {
                "repo_id": LONG_DENSE_MODEL_ID,
                "revision": LONG_DENSE_MODEL_REVISION,
                "max_length": LONG_DENSE_SELECTED_CAP,
                "trust_remote_code": False,
            },
            "gate": {"passed": True, "queries": 1, "documents": 4},
            "model_load_seconds": model_load_seconds,
            "document_encoding": {
                "seconds": document_seconds,
                "throughput_items_per_second": len(documents) / document_seconds,
                **_batch_evidence([LONG_DENSE_BATCH_SIZE], LONG_DENSE_BATCH_SIZE),
            },
            "query_encoding": {
                "seconds": query_seconds,
                "throughput_items_per_second": len(queries) / query_seconds,
                **_batch_evidence([LONG_DENSE_BATCH_SIZE], LONG_DENSE_BATCH_SIZE),
            },
            "representation": {
                "paradigm": "dense",
                "bytes": documents.nbytes + queries.nbytes,
                "scope": "float32_document_and_query_vectors",
            },
            "build_seconds": 0.0,
            "load_seconds": 0.0,
            "search": search,
            "cuda_peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
        },
        verification,
    )


def _sparse_provider(method: str) -> tuple[Any, dict[str, Any]]:
    if method in SPARSE_METHODS[:4]:
        snapshot_path = sparse_a._resolve_snapshot_path(method, None, None)
        audit = sparse_a._snapshot_audit(method, snapshot_path)
        provider = sparse_a._provider(method, snapshot_path, "cuda:0", 8)
        return provider, {
            "revision": sparse_a.INVENTORY[method].revision,
            "snapshot_sha256": audit["aggregate_identity_sha256"],
        }
    evidence_path = ROOT / f"results/learned-sparse-batch-b/active-v2/{method}/snapshot.json"
    evidence = _json(evidence_path)
    snapshot_path, identity = sparse_b.validate_snapshot_evidence(method, evidence)
    spec = BATCH_B_INVENTORY[method]
    provider = sparse_b.get_provider(
        spec.adapter,
        model_key=method,
        model_spec=spec,
        expected_snapshot_identity=identity,
        snapshot_path=str(snapshot_path),
        device="cuda:0",
        batch_size=8,
        max_length=spec.max_length,
        gpu_cap_bytes=sparse_b.BATCH_B_GPU_CAP_BYTES,
    )
    return provider, {"revision": spec.revision, "snapshot_sha256": evidence["aggregate_identity_sha256"]}


def _sparse(method: str, track: str) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    _assert_gpu_idle()
    load_started = time.perf_counter()
    provider, identity = _sparse_provider(method)
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats(0)
    gate_documents = provider.encode_sparse_documents(SYNTHETIC_DOCUMENTS, item_ids=[f"g{index}" for index in range(4)])
    gate_query = provider.encode_sparse_queries([SYNTHETIC_QUERY], item_ids=["gq"])
    gate_scores = np.asarray((gate_query.embeddings.values @ gate_documents.embeddings.values.T).toarray()[0])
    if int(np.argmax(gate_scores)) != 0 or gate_scores.tolist().count(float(gate_scores[0])) != 1:
        raise ResourceMeasurementError("Learned-sparse synthetic gate failed")
    data = load_materialized(ROOT / "data/bright-nontechnical-pilot-v0.2", track)
    doc_started = time.perf_counter()
    document_result = provider.encode_sparse_documents(
        [row["content"] for row in data.corpus], item_ids=[row["id"] for row in data.corpus]
    )
    document_seconds = time.perf_counter() - doc_started
    query_started = time.perf_counter()
    query_result = provider.encode_sparse_queries(
        [row["text"] for row in data.queries], item_ids=[row["id"] for row in data.queries]
    )
    query_seconds = time.perf_counter() - query_started
    documents = document_result.embeddings.values.tocsr()
    queries = query_result.embeddings.values.tocsr()
    exact_index = sparse_a.ExactSparseIndex(document_result)

    def rank_once() -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        rankings = {}
        latencies = []
        for index, query_row in enumerate(data.queries):
            query_started = time.perf_counter()
            single_query = sparse_a._result_from_csr(
                provider,
                queries[index],
                [query_row["id"]],
                sparse_a.SparseEmbeddingRole.QUERY,
            )
            result = exact_index.search(single_query, k=TOP_K)
            excluded = set(query_row.get("excluded_ids", ()))
            rankings[query_row["id"]] = [
                (hit.item_id, hit.score) for hit in result.queries[0].hits if hit.item_id not in excluded
            ][:TOP_K]
            latencies.append(time.perf_counter() - query_started)
        return rankings, latencies

    search, rankings = _search_trials(rank_once)
    verification = _verify_rankings(method, track, rankings)
    representation_bytes = sum(
        array.nbytes for matrix in (documents, queries) for array in (matrix.data, matrix.indices, matrix.indptr)
    )
    doc_meta = document_result.metadata_dict()
    query_meta = query_result.metadata_dict()
    return _apply_verification(
        {
            **_base(method, track, started),
            "identity": identity,
            "gate": {"passed": True, "queries": 1, "documents": 4},
            "model_load_seconds": model_load_seconds,
            "document_encoding": {
                "seconds": document_seconds,
                "throughput_items_per_second": len(data.corpus) / document_seconds,
                **_batch_evidence(doc_meta.get("batch_size_attempts", [8]), doc_meta.get("batch_size_used", 8)),
            },
            "query_encoding": {
                "seconds": query_seconds,
                "throughput_items_per_second": len(data.queries) / query_seconds,
                **_batch_evidence(query_meta.get("batch_size_attempts", [8]), query_meta.get("batch_size_used", 8)),
            },
            "representation": {
                "paradigm": "learned_sparse",
                "bytes": representation_bytes,
                "scope": "in_memory_float32_csr_document_and_query_values_indices_indptr",
            },
            "build_seconds": 0.0,
            "load_seconds": 0.0,
            "search": search,
            "cuda_peak_vram_bytes": max(
                int(document_result.peak_vram_bytes or 0),
                int(query_result.peak_vram_bytes or 0),
                int(torch.cuda.max_memory_allocated(0)),
            ),
        },
        verification,
    )


def _multi_snapshot(method: str) -> tuple[Path, str]:
    spec = MULTI_INVENTORY[method]
    root = Path(HF_HUB_CACHE) / f"models--{spec.repo_id.replace('/', '--')}" / "snapshots" / spec.revision
    files = {}
    for relative in spec.allowlist:
        path = root / relative
        if not path.is_file():
            raise ResourceMeasurementError("Frozen multi-vector snapshot is incomplete")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    accepted = _json(ROOT / f"results/bright-multi-vector-v0.2/representations/{method}/economics/representation.json")
    if identity != accepted["model"]["snapshot_identity_sha256"]:
        raise ResourceMeasurementError("Frozen multi-vector snapshot identity drifted")
    return root, identity


def _multi(method: str, track: str) -> dict[str, Any]:
    import torch
    from sentence_transformers.util.similarity import maxsim

    started = time.perf_counter()
    _assert_gpu_idle()
    snapshot, snapshot_identity = _multi_snapshot(method)
    load_started = time.perf_counter()
    provider = SentenceTransformerMultiVectorProvider(method, snapshot, device="cuda:0")
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats(0)
    gate_documents, _, gate_peak = provider.encode_compact(
        SYNTHETIC_DOCUMENTS, role=multi_benchmark.MultiVectorRole.DOCUMENT, batch_size=4
    )
    gate_queries, _, query_gate_peak = provider.encode_compact(
        [SYNTHETIC_QUERY], role=multi_benchmark.MultiVectorRole.QUERY, batch_size=1
    )
    gate_scores = [multi_benchmark._numpy_maxsim(gate_queries[0], document) for document in gate_documents]
    if int(np.argmax(gate_scores)) != 0 or gate_scores.count(gate_scores[0]) != 1:
        raise ResourceMeasurementError("Multi-vector synthetic gate failed")
    data = load_materialized(ROOT / "data/bright-nontechnical-pilot-v0.2", track)
    window_texts, document_indices, ordinals = multi_benchmark._windowed_corpus(data)
    doc_started = time.perf_counter()
    documents, document_execution = multi_benchmark._encode_chunks(
        provider,
        window_texts,
        role=multi_benchmark.MultiVectorRole.DOCUMENT,
        chunk_size=256,
        fallback=(32, 16, 8, 4, 2, 1),
    )
    document_seconds = time.perf_counter() - doc_started
    query_started = time.perf_counter()
    queries, query_execution = multi_benchmark._encode_chunks(
        provider,
        [row["text"] for row in data.queries],
        role=multi_benchmark.MultiVectorRole.QUERY,
        chunk_size=128,
        fallback=(64, 32, 16, 8, 4, 2, 1),
    )
    query_seconds = time.perf_counter() - query_started
    document_ids = [row["id"] for row in data.corpus]
    window_document_indices = np.asarray(document_indices, dtype=np.int32)
    window_ordinals = np.asarray(ordinals, dtype=np.int32)

    def rank_once() -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        rankings = {}
        latencies = []
        for query_row, query in zip(data.queries, queries, strict=True):
            query_started = time.perf_counter()
            window_scores = (
                maxsim([query], documents, device="cuda:0", chunk_elements=20_000_000)[0].float().cpu().numpy()
            )
            torch.cuda.synchronize(0)
            scores = np.full(len(document_ids), -np.inf, dtype=np.float32)
            winning = np.full(len(document_ids), np.iinfo(np.int32).max, dtype=np.int32)
            for window_index, score in enumerate(window_scores):
                index = int(window_document_indices[window_index])
                ordinal = int(window_ordinals[window_index])
                if score > scores[index] or (score == scores[index] and ordinal < winning[index]):
                    scores[index] = score
                    winning[index] = ordinal
            rankings[query_row["id"]] = _order(scores, document_ids, set(query_row.get("excluded_ids", ())))
            latencies.append(time.perf_counter() - query_started)
        return rankings, latencies

    search, rankings = _search_trials(rank_once)
    verification = _verify_rankings(method, track, rankings)
    representation_bytes = sum(array.nbytes for array in documents) + sum(array.nbytes for array in queries)
    representation_bytes += window_document_indices.nbytes + window_ordinals.nbytes
    return _apply_verification(
        {
            **_base(method, track, started),
            "identity": {"revision": MULTI_INVENTORY[method].revision, "snapshot_sha256": snapshot_identity},
            "gate": {"passed": True, "queries": 1, "documents": 4},
            "model_load_seconds": model_load_seconds,
            "document_encoding": {
                "seconds": document_seconds,
                "throughput_items_per_second": len(window_texts) / document_seconds,
                "batch_used": document_execution["used_batch_min"],
                "fallback": [32, 16, 8, 4, 2, 1],
                "items_are_windows": True,
            },
            "query_encoding": {
                "seconds": query_seconds,
                "throughput_items_per_second": len(queries) / query_seconds,
                "batch_used": query_execution["used_batch_min"],
                "fallback": [64, 32, 16, 8, 4, 2, 1],
            },
            "representation": {
                "paradigm": "multi_vector",
                "bytes": representation_bytes,
                "scope": "compact_float32_token_vectors_plus_window_mapping",
            },
            "build_seconds": 0.0,
            "load_seconds": 0.0,
            "search": search,
            "cuda_peak_vram_bytes": max(
                int(gate_peak or 0), int(query_gate_peak or 0), int(torch.cuda.max_memory_allocated(0))
            ),
        },
        verification,
    )


def run_cell(method: str, track: str) -> dict[str, Any]:
    """Run one isolated frozen resource cell."""
    verify_freeze()
    if method not in METHODS or track not in TRACKS:
        raise ResourceMeasurementError("Resource cell is outside the frozen matrix")
    if method not in {"bm25-unicode", "tfidf-word-sublinear"}:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ResourceMeasurementError("Neural resource cells require CUDA_VISIBLE_DEVICES=0")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise ResourceMeasurementError("Neural resource cells require explicit offline mode")
    if method in {"tfidf-word-sublinear", "all-minilm-l6-v2"}:
        result = _new_method_resource(method, track)
    elif method == "bm25-unicode":
        result = _bm25(method, track)
    elif method == "bge-m3-dense":
        result = _dense(method, track)
    elif method in SPARSE_METHODS:
        result = _sparse(method, track)
    else:
        result = _multi(method, track)
    if result["end_to_end_seconds"] > 3600:
        raise ResourceMeasurementError("Resource cell exceeded the frozen 3600-second stopping condition")
    return result


def _archive_return1_resource(method: str, track: str) -> None:
    archive = RESOURCE_ROOT / "return-1-prior"
    for folder in ("cells", "failures"):
        source = RESOURCE_ROOT / folder / f"{method}-{track}.json"
        sidecar = source.with_suffix(source.suffix + ".sha256")
        if source.is_file():
            target = archive / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
                shutil.copy2(sidecar, target.with_suffix(target.suffix + ".sha256"))
    summary = RESOURCE_ROOT / "resource-summary.json"
    summary_target = archive / "resource-summary.json"
    if summary.is_file() and not summary_target.exists():
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary, summary_target)
        shutil.copy2(
            summary.with_suffix(summary.suffix + ".sha256"),
            summary_target.with_suffix(summary_target.suffix + ".sha256"),
        )


def _unexpected_failure(method: str, track: str, started: float, error: BaseException) -> dict[str, Any]:
    return {
        **_base(method, track, started),
        "status": "failed_closed",
        "identity": {},
        "gate": {"passed": False, "stage_completed": False},
        "model_load_seconds": None,
        "document_encoding": None,
        "query_encoding": None,
        "representation": {"paradigm": FAMILIES[method], "bytes": None, "scope": "not_completed"},
        "build_seconds": None,
        "load_seconds": None,
        "search": {"warmup_query_searches": 0, "trials_completed": 0, "measured_trials": []},
        "cuda_peak_vram_bytes": None,
        "quality_verification": {
            "formal_queries": None,
            "exact_order_queries": None,
            "mismatched_order_queries": None,
            "top100_overlap_min": None,
            "top100_overlap_mean": None,
            "top100_overlap_max": None,
            "all_mismatches_full_top100_overlap": None,
            "fresh_top100_ids_match": None,
            "system_error": "measurement_aborted_before_aggregate_verification",
        },
        "failure": {
            "type": type(error).__name__,
            "stage": "uncompleted_stage",
            "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "stopping_condition_applied": True,
        },
    }


def write_cell(method: str, track: str, *, replace_failed_return1: bool = False) -> tuple[str, str]:
    path = RESOURCE_ROOT / "cells" / f"{method}-{track}.json"
    started = time.perf_counter()
    cell_id = f"{method}:{track}"
    if path.exists() or path.with_suffix(".json.sha256").exists():
        if not replace_failed_return1 or cell_id not in RETURN1_FAILED_CELLS:
            raise ResourceMeasurementError("Resource cell already exists and is immutable")
        prior = _json(path)
        if prior.get("status") != "failed_closed":
            raise ResourceMeasurementError("RETURN 1 replacement is limited to prior failed-closed cells")
        _archive_return1_resource(method, track)
    try:
        value = run_cell(method, track)
    except Exception as error:  # noqa: BLE001 - terminal evidence must survive every failure
        value = _unexpected_failure(method, track, started, error)
    value["end_to_end_seconds"] = time.perf_counter() - started
    value["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    identity = _write(path, value)
    if value["status"] == "failed_closed":
        detail = {
            "schema_version": "bright-cross-paradigm-resource-failure-detail-v2",
            "cell_id": cell_id,
            "failure": value["failure"],
            "completed_stages": {
                "gate": value["gate"],
                "model_load_seconds": value["model_load_seconds"],
                "document_encoding": value["document_encoding"],
                "query_encoding": value["query_encoding"],
                "representation": value["representation"],
                "build_seconds": value["build_seconds"],
                "load_seconds": value["load_seconds"],
                "search": value["search"],
            },
            "aggregate_top100_evidence": value["quality_verification"],
            "restricted_ids_included": False,
        }
        _write(RESOURCE_ROOT / "failures" / f"{method}-{track}.json", detail)
    else:
        stale_detail = RESOURCE_ROOT / "failures" / f"{method}-{track}.json"
        stale_sidecar = stale_detail.with_suffix(stale_detail.suffix + ".sha256")
        if stale_detail.exists():
            stale_detail.unlink()
        if stale_sidecar.exists():
            stale_sidecar.unlink()
    return identity, value["status"]


def collect() -> str:
    """Bind all terminal resource cells into a deterministic restricted summary."""
    cells = []
    for method in METHODS:
        for track in TRACKS:
            path = RESOURCE_ROOT / "cells" / f"{method}-{track}.json"
            identity = sha256_file(path)
            if path.with_suffix(".json.sha256").read_text(encoding="ascii").strip() != identity:
                raise ResourceMeasurementError("Resource cell sidecar drifted")
            cells.append(_json(path))
    value = {
        "schema_version": "bright-cross-paradigm-resource-summary-v1",
        "story_id": STORY_ID,
        "predeclaration_sha256": sha256_file(ARTIFACT_ROOT / "predeclaration.json"),
        "gate_summary_sha256": sha256_file(ARTIFACT_ROOT / "gate-summary.json"),
        "hardware": _json(ARTIFACT_ROOT / "predeclaration.json")["resource_protocol"]["hardware"],
        "cells": cells,
    }
    return _write(RESOURCE_ROOT / "resource-summary.json", value)


def _numeric_evidence(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            rows.extend(_numeric_evidence(value[key], f"{prefix}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_numeric_evidence(item, f"{prefix}/{index}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append({"path": prefix, "value": value})
    return rows


def correct_minilm_scope_return2() -> dict[str, Any]:
    """Correct only the two MiniLM scope strings and preserve pre-correction evidence."""
    archive_root = RESOURCE_ROOT / "return-2-prior/minilm-scope"
    records = []
    for track in TRACKS:
        source = RESOURCE_ROOT / "cells" / f"all-minilm-l6-v2-{track}.json"
        sidecar = source.with_suffix(source.suffix + ".sha256")
        value = _json(source)
        old_scope = value["representation"]["scope"]
        if old_scope != TFIDF_REPRESENTATION_SCOPE:
            raise ResourceMeasurementError(f"Unexpected pre-correction MiniLM scope: {track}")
        archive_root.mkdir(parents=True, exist_ok=True)
        archived = archive_root / source.name
        archived_sidecar = archived.with_suffix(archived.suffix + ".sha256")
        if archived.exists() or archived_sidecar.exists():
            raise ResourceMeasurementError("RETURN 2 pre-correction evidence already exists")
        shutil.copy2(source, archived)
        shutil.copy2(sidecar, archived_sidecar)
        before_sha256 = sha256_file(source)
        before_sidecar_sha256 = sha256_file(sidecar)
        numeric_before = hashlib.sha256(canonical_bytes(_numeric_evidence(value))).hexdigest()
        value["representation"]["scope"] = MINILM_REPRESENTATION_SCOPE
        after_sha256 = _write(source, value)
        numeric_after = hashlib.sha256(canonical_bytes(_numeric_evidence(value))).hexdigest()
        if numeric_after != numeric_before:
            raise ResourceMeasurementError("MiniLM numeric measurement evidence changed during scope correction")
        records.append(
            {
                "cell_id": f"all-minilm-l6-v2:{track}",
                "pre_correction_record": {
                    "path": archived.relative_to(ROOT).as_posix(),
                    "bytes": archived.stat().st_size,
                    "sha256": before_sha256,
                    "sidecar_sha256": before_sidecar_sha256,
                    "scope": old_scope,
                },
                "corrected_record": {
                    "path": source.relative_to(ROOT).as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": after_sha256,
                    "scope": MINILM_REPRESENTATION_SCOPE,
                },
                "measured_numeric_values_sha256_before": numeric_before,
                "measured_numeric_values_sha256_after": numeric_after,
                "numeric_values_unchanged": True,
                "only_changed_json_pointer": "/representation/scope",
            }
        )
    audit = {
        "schema_version": "bright-cross-paradigm-return2-minilm-scope-audit-v1",
        "story_id": STORY_ID,
        "reason": "Correct a cross-family representation scope label without changing measurements.",
        "records": records,
        "publication": {
            "classification": "restricted_audit_only",
            "publish": False,
            "public_export_allowed": False,
            "gate": "closed",
        },
    }
    identity = _write(archive_root / "audit.json", audit)
    return {"audit_sha256": identity, "corrected_cells": records}


def _timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("Frozen 3600-second resource-cell stopping condition reached")


def install_timeout() -> None:
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(3600)


__all__ = [
    "MINILM_REPRESENTATION_SCOPE",
    "TFIDF_REPRESENTATION_SCOPE",
    "ResourceMeasurementError",
    "collect",
    "correct_minilm_scope_return2",
    "install_timeout",
    "run_cell",
    "write_cell",
]
