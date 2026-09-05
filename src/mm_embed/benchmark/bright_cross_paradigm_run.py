"""Exact formal runs for frozen cross-paradigm baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from mm_embed.benchmark.bright_cross_paradigm import (
    ARTIFACT_ROOT,
    DATA_IDENTITY,
    METRICS,
    MINILM,
    ROOT,
    STORY_ID,
    TRACKS,
    canonical_bytes,
    minilm_snapshot_path,
    publication_policy,
    sha256_file,
    verify_freeze,
    verify_sidecar,
)
from mm_embed.benchmark.bright_multidomain_v02 import (
    LONG_DENSE_BATCH_SIZE,
    LONG_DENSE_MODEL_ID,
    LONG_DENSE_MODEL_REVISION,
    LONG_DENSE_SELECTED_CAP,
    LONG_DENSE_SNAPSHOT,
    LONG_DENSE_SNAPSHOT_PATH,
    _load_result,
    load_materialized,
    verify_long_dense_preflight,
)
from mm_embed.benchmark.retrieval_v01 import (
    aggregate_metrics,
    bootstrap_confidence_intervals,
    query_metrics,
    tokens,
    validate_track_data,
)

RESULT_ROOT = ROOT / "results/bright-cross-paradigm-v0.1"
PREDECLARATION_SHA256 = "e1f6ca4bc069ef8ffb05b5dbf673c9515ddff3403e2614e9f782abfb2c25ca30"
GATE_SUMMARY_SHA256 = "c7e823bb0462ef135b4028ca4a29d21076f9eeb0fc3653cd25b425d6f0b8f781"
BOOTSTRAP_SEED = 20_260_826
BOOTSTRAP_SAMPLES = 10_000
TOP_K = 100
SEARCH_TRIALS = 3
TFIDF_KEY = "tfidf-word-sublinear"
MINILM_KEY = "all-minilm-l6-v2"
BM25_KEY = "bm25-unicode"
BGE_KEY = "bge-m3-dense"
NEW_METHODS = (TFIDF_KEY, MINILM_KEY)
S005_REPLAY_METHODS = (BM25_KEY, BGE_KEY)
FORMAL_METHODS = (BM25_KEY, TFIDF_KEY, BGE_KEY, MINILM_KEY)


class FormalRunError(RuntimeError):
    """Fail-closed formal-run error."""


def _json_bytes(value: object) -> bytes:
    return canonical_bytes(value)


def _write_file(path: Path, payload: bytes) -> dict[str, Any]:
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(digest + "\n", encoding="ascii")
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": digest,
        "sidecar": {
            "path": sidecar.name,
            "bytes": sidecar.stat().st_size,
            "sha256": sha256_file(sidecar),
        },
    }


def _write_numpy(path: Path, value: np.ndarray) -> dict[str, Any]:
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    payload = path.read_bytes()
    return _write_file(path, payload)


def _write_sparse(path: Path, value: sparse.csr_matrix) -> dict[str, Any]:
    sparse.save_npz(path, value, compressed=False)
    payload = path.read_bytes()
    return _write_file(path, payload)


def _verify_frozen_inputs() -> dict[str, str]:
    hashes = verify_freeze()
    if hashes["predeclaration"] != PREDECLARATION_SHA256:
        raise FormalRunError("Active predeclaration identity drifted")
    gate_path = ARTIFACT_ROOT / "gate-summary.json"
    if verify_sidecar(gate_path) != GATE_SUMMARY_SHA256:
        raise FormalRunError("Gate summary identity drifted")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("all_passed") is not True:
        raise FormalRunError("Frozen bounded gates did not pass")
    return hashes


def _assert_gpu_idle() -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise FormalRunError(f"Unable to inspect GPU processes: {result.stderr.strip()}")
    target_uuid = "GPU-435154a8-568c-35a7-ee5f-da29fbf97a39"
    occupied = [line for line in result.stdout.splitlines() if line.startswith(target_uuid + ",")]
    if occupied:
        raise FormalRunError(f"Frozen cuda:0 is occupied: {occupied}")


def _rank_scores(
    scores: np.ndarray,
    *,
    document_ids: list[str],
    excluded_ids: set[str],
) -> list[tuple[str, float]]:
    if scores.shape != (len(document_ids),) or not np.isfinite(scores).all():
        raise FormalRunError("Search produced invalid scores")
    eligible = (index for index, document_id in enumerate(document_ids) if document_id not in excluded_ids)
    best = sorted(eligible, key=lambda index: (-float(scores[index]), document_ids[index]))[:TOP_K]
    if len(best) != TOP_K:
        raise FormalRunError("Search did not produce top-100")
    return [(document_ids[index], float(scores[index])) for index in best]


def _search_trials(
    *,
    query_ids: list[str],
    document_ids: list[str],
    excluded: dict[str, set[str]],
    score_one: Callable[[int], np.ndarray],
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Any]]:
    if query_ids:
        warmup = score_one(0)
        if warmup.shape != (len(document_ids),) or not np.isfinite(warmup).all():
            raise FormalRunError("Unrecorded search warmup failed")
    rankings: dict[str, list[tuple[str, float]]] = {}
    trials = []
    pooled_latencies = []
    for trial in range(SEARCH_TRIALS):
        trial_started = time.perf_counter()
        latencies = []
        current: dict[str, list[tuple[str, float]]] = {}
        for query_index, query_id in enumerate(query_ids):
            started = time.perf_counter()
            scores = score_one(query_index)
            current[query_id] = _rank_scores(
                scores,
                document_ids=document_ids,
                excluded_ids=excluded[query_id],
            )
            latencies.append(time.perf_counter() - started)
        elapsed = time.perf_counter() - trial_started
        if trial == 0:
            rankings = current
        elif current != rankings:
            raise FormalRunError("Exact search rankings or scores changed across trials")
        pooled_latencies.extend(latencies)
        trials.append(
            {
                "ordinal": trial + 1,
                "wall_seconds": elapsed,
                "queries_per_second": len(query_ids) / elapsed,
                "p50_seconds": float(np.percentile(latencies, 50)),
                "p95_seconds": float(np.percentile(latencies, 95)),
                "p99_seconds": float(np.percentile(latencies, 99)),
            }
        )
    return rankings, {
        "warmup_query_searches": 1,
        "measured_trials": trials,
        "median_queries_per_second": float(np.median([row["queries_per_second"] for row in trials])),
        "pooled_p50_seconds": float(np.percentile(pooled_latencies, 50)),
        "pooled_p95_seconds": float(np.percentile(pooled_latencies, 95)),
        "pooled_p99_seconds": float(np.percentile(pooled_latencies, 99)),
    }


def _serialize_rankings(rankings: dict[str, list[tuple[str, float]]]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": query_id,
            "hits": [
                {"document_id": document_id, "rank": rank, "score": score}
                for rank, (document_id, score) in enumerate(rankings[query_id], 1)
            ],
        }
        for query_id in sorted(rankings)
    ]


def _serialize_per_query(per_query: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    return [{"query_id": query_id, "metrics": per_query[query_id]} for query_id in sorted(per_query)]


def _truncation_counts(model: Any, data: Any) -> dict[str, int]:
    tokenizer = model.tokenizer
    original = tokenizer.model_max_length
    tokenizer.model_max_length = 10**30
    try:
        document_lengths = []
        for start in range(0, len(data.corpus), 128):
            batch = tokenizer(
                [row["content"] for row in data.corpus[start : start + 128]],
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            document_lengths.extend(map(len, batch))
        query_lengths = []
        for start in range(0, len(data.queries), 128):
            batch = tokenizer(
                [row["text"] for row in data.queries[start : start + 128]],
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            query_lengths.extend(map(len, batch))
    finally:
        tokenizer.model_max_length = original
    gold_ids = {document_id for rels in data.qrels.values() for document_id in rels}
    length_by_id = {row["id"]: length for row, length in zip(data.corpus, document_lengths, strict=True)}
    maximum = int(model.max_seq_length)
    return {
        "max_sequence_length": maximum,
        "documents_truncated": sum(length > maximum for length in document_lengths),
        "queries_truncated": sum(length > maximum for length in query_lengths),
        "gold_documents_truncated": sum(length_by_id[document_id] > maximum for document_id in gold_ids),
        "unique_gold_documents": len(gold_ids),
    }


def _encode_with_fallback(model: Any, texts: list[str], batches: list[int]) -> tuple[np.ndarray, int, float]:
    import torch

    last_error: Exception | None = None
    for batch_size in batches:
        try:
            started = time.perf_counter()
            values = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                precision="float32",
                show_progress_bar=False,
            ).astype(np.float32, copy=False)
            return values, batch_size, time.perf_counter() - started
        except torch.OutOfMemoryError as error:
            last_error = error
            torch.cuda.empty_cache()
    raise FormalRunError("MiniLM exhausted the frozen batch fallback") from last_error


def _tfidf_run(data: Any) -> tuple[Any, Any, dict[str, Any], dict[str, list[tuple[str, float]]]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float32,
    )
    document_started = time.perf_counter()
    documents = vectorizer.fit_transform([row["content"] for row in data.corpus]).tocsr()
    document_seconds = time.perf_counter() - document_started
    query_started = time.perf_counter()
    queries = vectorizer.transform([row["text"] for row in data.queries]).tocsr()
    query_seconds = time.perf_counter() - query_started
    document_ids = [row["id"] for row in data.corpus]
    query_ids = [row["id"] for row in data.queries]
    excluded = {row["id"]: set(row.get("excluded_ids", ())) for row in data.queries}
    rankings, search = _search_trials(
        query_ids=query_ids,
        document_ids=document_ids,
        excluded=excluded,
        score_one=lambda index: np.asarray((queries[index] @ documents.T).toarray()[0], dtype=np.float32),
    )
    representation_bytes = (
        sum(array.nbytes for matrix in (documents, queries) for array in (matrix.data, matrix.indices, matrix.indptr))
        + vectorizer.idf_.nbytes
    )
    execution = {
        "implementation": "sklearn.feature_extraction.text.TfidfVectorizer",
        "package": "scikit-learn==1.8.0",
        "device": "cpu",
        "document_encoding_seconds": document_seconds,
        "query_encoding_seconds": query_seconds,
        "document_throughput_per_second": len(data.corpus) / document_seconds,
        "query_throughput_per_second": len(data.queries) / query_seconds,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "document_nnz": int(documents.nnz),
        "query_nnz": int(queries.nnz),
        "representation_bytes": int(representation_bytes),
        "index_build_seconds": document_seconds,
        "index_load_seconds": 0.0,
        "network_bytes": 0,
        "batch_fallback": "not_applicable",
        "search": search,
    }
    return documents, queries, execution, rankings


def _minilm_run(data: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, list[tuple[str, float]]]]:
    import torch
    from sentence_transformers import SentenceTransformer

    _assert_gpu_idle()
    snapshot = minilm_snapshot_path()
    load_started = time.perf_counter()
    model = SentenceTransformer(
        str(snapshot),
        device="cuda:0",
        trust_remote_code=False,
        model_kwargs={"local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
    )
    load_seconds = time.perf_counter() - load_started
    if int(model.max_seq_length) != 256:
        raise FormalRunError("MiniLM max length drifted")
    truncation = _truncation_counts(model, data)
    torch.cuda.reset_peak_memory_stats(0)
    documents, document_batch, document_seconds = _encode_with_fallback(
        model,
        [row["content"] for row in data.corpus],
        [128, 64, 32, 16, 8, 4, 2, 1],
    )
    queries, query_batch, query_seconds = _encode_with_fallback(
        model,
        [row["text"] for row in data.queries],
        [256, 128, 64, 32, 16, 8, 4, 2, 1],
    )
    if documents.shape != (len(data.corpus), 384) or queries.shape != (len(data.queries), 384):
        raise FormalRunError("MiniLM formal representation shape drifted")
    if not np.isfinite(documents).all() or not np.isfinite(queries).all():
        raise FormalRunError("MiniLM produced non-finite representations")
    if not np.allclose(np.linalg.norm(documents, axis=1), 1.0, atol=1e-5):
        raise FormalRunError("MiniLM document normalization drifted")
    if not np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-5):
        raise FormalRunError("MiniLM query normalization drifted")
    document_ids = [row["id"] for row in data.corpus]
    query_ids = [row["id"] for row in data.queries]
    excluded = {row["id"]: set(row.get("excluded_ids", ())) for row in data.queries}
    rankings, search = _search_trials(
        query_ids=query_ids,
        document_ids=document_ids,
        excluded=excluded,
        score_one=lambda index: np.asarray(queries[index] @ documents.T, dtype=np.float32),
    )
    execution = {
        "model_id": MINILM["repo_id"],
        "revision": MINILM["revision"],
        "snapshot_sha256": MINILM["snapshot_sha256"],
        "device": "cuda:0",
        "dtype": "float32",
        "dimensions": 384,
        "normalize_embeddings": True,
        "trust_remote_code": False,
        "local_files_only": True,
        "model_load_seconds": load_seconds,
        "document_encoding_seconds": document_seconds,
        "query_encoding_seconds": query_seconds,
        "document_throughput_per_second": len(data.corpus) / document_seconds,
        "query_throughput_per_second": len(data.queries) / query_seconds,
        "document_batch_size": document_batch,
        "query_batch_size": query_batch,
        "document_batch_fallback": [128, 64, 32, 16, 8, 4, 2, 1],
        "query_batch_fallback": [256, 128, 64, 32, 16, 8, 4, 2, 1],
        "representation_bytes": int(documents.nbytes + queries.nbytes),
        "index_build_seconds": 0.0,
        "index_load_seconds": 0.0,
        "network_bytes": 0,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
        "truncation": truncation,
        "search": search,
    }
    return documents, queries, execution, rankings


def _bm25_run(data: Any) -> tuple[np.ndarray, None, dict[str, Any], dict[str, list[tuple[str, float]]]]:
    """Build exact BM25 scores while preserving the frozen summation order."""
    build_started = time.perf_counter()
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = np.empty(len(data.corpus), dtype=np.int32)
    for index, row in enumerate(data.corpus):
        counts = Counter(tokens(row["content"]))
        lengths[index] = sum(counts.values())
        for token, frequency in counts.items():
            postings[token].append((index, frequency))
    build_seconds = time.perf_counter() - build_started
    average_length = float(lengths.mean())
    scores = np.zeros((len(data.queries), len(data.corpus)), dtype=np.float64)
    query_started = time.perf_counter()
    for query_index, query in enumerate(data.queries):
        for token, query_frequency in Counter(tokens(query["text"])).items():
            posting = postings.get(token, ())
            inverse_document_frequency = math.log(1.0 + (len(data.corpus) - len(posting) + 0.5) / (len(posting) + 0.5))
            for document_index, frequency in posting:
                denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * lengths[document_index] / average_length)
                scores[query_index, document_index] += (
                    query_frequency * inverse_document_frequency * frequency * 2.2 / denominator
                )
    query_seconds = time.perf_counter() - query_started
    document_ids = [row["id"] for row in data.corpus]
    query_ids = [row["id"] for row in data.queries]
    excluded = {row["id"]: set(row.get("excluded_ids", ())) for row in data.queries}
    rankings, search = _search_trials(
        query_ids=query_ids,
        document_ids=document_ids,
        excluded=excluded,
        score_one=lambda index: scores[index],
    )
    index_bytes = int(
        lengths.nbytes + sum(len(token.encode("utf-8")) + len(rows) * 8 for token, rows in postings.items())
    )
    return (
        scores,
        None,
        {
            "implementation": "project in-process exact BM25",
            "tokenizer": "unicode-word-lower-v1",
            "k1": 1.2,
            "b": 0.75,
            "device": "cpu",
            "document_encoding_seconds": build_seconds,
            "query_encoding_seconds": query_seconds,
            "document_throughput_per_second": len(data.corpus) / build_seconds,
            "query_throughput_per_second": len(data.queries) / query_seconds,
            "representation_bytes": int(scores.nbytes),
            "native_index_bytes": index_bytes,
            "index_build_seconds": build_seconds,
            "index_load_seconds": 0.0,
            "network_bytes": 0,
            "batch_fallback": "not_applicable",
            "search": search,
            "recompute_basis": "saved_full_float64_score_matrix",
        },
        rankings,
    )


def _bge_run(data: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, list[tuple[str, float]]]]:
    """Rerun the accepted fixed-revision BGE-M3 dense mechanism exactly."""
    import torch
    from sentence_transformers import SentenceTransformer

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
    load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats(0)
    document_started = time.perf_counter()
    documents = model.encode(
        [row["content"] for row in data.corpus],
        batch_size=LONG_DENSE_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    document_seconds = time.perf_counter() - document_started
    query_started = time.perf_counter()
    queries = model.encode(
        [row["text"] for row in data.queries],
        batch_size=LONG_DENSE_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    query_seconds = time.perf_counter() - query_started
    if documents.shape != (len(data.corpus), 1024) or queries.shape != (len(data.queries), 1024):
        raise FormalRunError("BGE-M3 formal representation shape drifted")
    if not np.isfinite(documents).all() or not np.isfinite(queries).all():
        raise FormalRunError("BGE-M3 produced non-finite representations")
    document_ids = [row["id"] for row in data.corpus]
    query_ids = [row["id"] for row in data.queries]
    excluded = {row["id"]: set(row.get("excluded_ids", ())) for row in data.queries}
    rankings, search = _search_trials(
        query_ids=query_ids,
        document_ids=document_ids,
        excluded=excluded,
        score_one=lambda index: np.asarray(queries[index] @ documents.T, dtype=np.float32),
    )
    return (
        documents,
        queries,
        {
            "model_id": LONG_DENSE_MODEL_ID,
            "revision": LONG_DENSE_MODEL_REVISION,
            "snapshot_sha256": LONG_DENSE_SNAPSHOT["aggregate_sha256"],
            "device": "cuda:0",
            "dtype": "float32",
            "dimensions": 1024,
            "max_sequence_length": LONG_DENSE_SELECTED_CAP,
            "normalize_embeddings": True,
            "trust_remote_code": False,
            "local_files_only": True,
            "model_load_seconds": load_seconds,
            "document_encoding_seconds": document_seconds,
            "query_encoding_seconds": query_seconds,
            "document_throughput_per_second": len(data.corpus) / document_seconds,
            "query_throughput_per_second": len(data.queries) / query_seconds,
            "document_batch_size": LONG_DENSE_BATCH_SIZE,
            "query_batch_size": LONG_DENSE_BATCH_SIZE,
            "document_batch_fallback": [LONG_DENSE_BATCH_SIZE],
            "query_batch_fallback": [LONG_DENSE_BATCH_SIZE],
            "representation_bytes": int(documents.nbytes + queries.nbytes),
            "index_build_seconds": 0.0,
            "index_load_seconds": 0.0,
            "network_bytes": 0,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
            "search": search,
            "recompute_basis": "saved_normalized_float32_document_and_query_vectors",
        },
        rankings,
    )


def _accepted_anchor_comparison(
    method: str, track: str, rankings: dict[str, list[tuple[str, float]]], metrics: dict[str, float]
) -> dict[str, Any] | None:
    if method not in S005_REPLAY_METHODS:
        return None
    source_method = "bm25" if method == BM25_KEY else "long_dense"
    accepted, accepted_rankings, _ = _load_result(
        ROOT / "results/bright-nontechnical-pilot-v0.2",
        ROOT / "data/bright-nontechnical-pilot-v0.2",
        source_method,
        track,
    )
    exact = 0
    overlaps = []
    for query_id in sorted(rankings):
        observed_ids = [document_id for document_id, _ in rankings[query_id]]
        accepted_ids = [document_id for document_id, _ in accepted_rankings[query_id]]
        exact += observed_ids == accepted_ids
        overlaps.append(len(set(observed_ids) & set(accepted_ids)))
    return {
        "role": "accepted_S005_contextual_anchor_only",
        "accepted_cell_sha256": sha256_file(
            ROOT / f"results/bright-nontechnical-pilot-v0.2/{source_method}-{track}.json"
        ),
        "queries": len(rankings),
        "exact_order_queries": exact,
        "mismatched_order_queries": len(rankings) - exact,
        "top100_overlap_min": min(overlaps),
        "top100_overlap_mean": float(np.mean(overlaps)),
        "all_mismatched_queries_have_full_top100_overlap": all(overlap == TOP_K for overlap in overlaps),
        "metric_delta_rerun_minus_accepted": {
            metric: metrics[metric] - accepted["metrics"][metric] for metric in METRICS
        },
    }


def run_formal_cell(method: str, track: str, output_root: Path = RESULT_ROOT) -> dict[str, Any]:
    if method not in FORMAL_METHODS or track not in TRACKS:
        raise FormalRunError("Method or track is outside the frozen formal rerun matrix")
    _verify_frozen_inputs()
    data = load_materialized(ROOT / "data/bright-nontechnical-pilot-v0.2", track)
    validate_track_data(data)
    target = output_root / "cells" / f"{method}-{track}"
    if target.exists():
        raise FormalRunError(f"Refusing to overwrite existing formal cell: {target}")
    failure_root = output_root / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f".{method}-{track}-", dir=output_root) as temporary:
            work = Path(temporary)
            if method == BM25_KEY:
                documents, queries, execution, rankings = _bm25_run(data)
                document_entry = _write_numpy(work / "scores.npy", documents)
                query_entry = None
            elif method == TFIDF_KEY:
                documents, queries, execution, rankings = _tfidf_run(data)
                document_entry = _write_sparse(work / "documents.npz", documents)
                query_entry = _write_sparse(work / "queries.npz", queries)
            elif method == MINILM_KEY:
                documents, queries, execution, rankings = _minilm_run(data)
                document_entry = _write_numpy(work / "documents.npy", documents)
                query_entry = _write_numpy(work / "queries.npy", queries)
            else:
                documents, queries, execution, rankings = _bge_run(data)
                document_entry = _write_numpy(work / "documents.npy", documents)
                query_entry = _write_numpy(work / "queries.npy", queries)
            per_query = query_metrics(data, rankings)
            metrics = aggregate_metrics(per_query)
            intervals = bootstrap_confidence_intervals(
                per_query,
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            )
            document_ids = _write_file(
                work / "document_ids.json",
                _json_bytes([row["id"] for row in data.corpus]),
            )
            query_ids = _write_file(
                work / "query_ids.json",
                _json_bytes([row["id"] for row in data.queries]),
            )
            rankings_entry = _write_file(work / "rankings.json", _json_bytes(_serialize_rankings(rankings)))
            per_query_entry = _write_file(
                work / "per_query_metrics.json",
                _json_bytes(_serialize_per_query(per_query)),
            )
            total_seconds = time.perf_counter() - started
            execution["end_to_end_seconds"] = total_seconds
            execution["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            value = {
                "schema_version": "bright-cross-paradigm-formal-cell-v1",
                "story_id": STORY_ID,
                "finalized": True,
                "method": method,
                "family": "lexical" if method in {BM25_KEY, TFIDF_KEY} else "dense",
                "track": track,
                "predeclaration_sha256": PREDECLARATION_SHA256,
                "gate_summary_sha256": GATE_SUMMARY_SHA256,
                "data_identity": DATA_IDENTITY["tracks"][track],
                "protocol": {
                    "exact": True,
                    "candidate_pool": len(data.corpus),
                    "retrieval_unit": "canonical_passage",
                    "top_k": TOP_K,
                    "tie_policy": "score_descending_then_canonical_passage_id",
                    "metrics": list(METRICS),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                },
                "quality": {
                    "queries": len(data.queries),
                    "documents": len(data.corpus),
                    "positive_qrels": sum(map(len, data.qrels.values())),
                    "metrics": metrics,
                    "confidence_intervals": intervals,
                },
                "accepted_anchor_comparison": _accepted_anchor_comparison(method, track, rankings, metrics),
                "execution": execution,
                "restricted_recompute": {
                    "documents": document_entry,
                    "document_ids": document_ids,
                    "query_ids": query_ids,
                    "rankings": rankings_entry,
                    "per_query": per_query_entry,
                    **({"queries": query_entry} if query_entry is not None else {}),
                },
                "publication": publication_policy(),
            }
            manifest_entry = _write_file(work / "cell.json", _json_bytes(value))
            target.parent.mkdir(parents=True, exist_ok=True)
            Path(temporary).rename(target)
            return {"cell": f"{method}:{track}", "identity": manifest_entry["sha256"], "passed": True}
    except Exception as error:
        failure = {
            "schema_version": "bright-cross-paradigm-formal-failure-v1",
            "story_id": STORY_ID,
            "method": method,
            "track": track,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "protocol_weakened": False,
            "publication": publication_policy(),
        }
        failure_path = failure_root / f"{method}-{track}.json"
        if not failure_path.exists():
            _write_file(failure_path, _json_bytes(failure))
        raise


def _load_entry(root: Path, entry: dict[str, Any]) -> Any:
    path = root / entry["path"]
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise FormalRunError(f"Restricted artifact drifted: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="ascii").strip() != entry["sha256"]:
        raise FormalRunError(f"Restricted sidecar drifted: {sidecar}")
    return path


def validate_formal_cell(method: str, track: str, output_root: Path = RESULT_ROOT) -> dict[str, Any]:
    _verify_frozen_inputs()
    root = output_root / "cells" / f"{method}-{track}"
    cell_path = root / "cell.json"
    verify_sidecar(cell_path)
    value = json.loads(cell_path.read_text(encoding="utf-8"))
    if (
        value.get("finalized") is not True
        or value.get("method") != method
        or value.get("track") != track
        or value.get("predeclaration_sha256") != PREDECLARATION_SHA256
        or value.get("gate_summary_sha256") != GATE_SUMMARY_SHA256
        or value.get("publication") != publication_policy()
    ):
        raise FormalRunError(f"Formal cell contract drifted: {method}:{track}")
    restricted = value["restricted_recompute"]
    paths = {name: _load_entry(root, entry) for name, entry in restricted.items()}
    document_ids = json.loads(paths["document_ids"].read_text(encoding="utf-8"))
    query_ids = json.loads(paths["query_ids"].read_text(encoding="utf-8"))
    data = load_materialized(ROOT / "data/bright-nontechnical-pilot-v0.2", track)
    if document_ids != [row["id"] for row in data.corpus] or query_ids != [row["id"] for row in data.queries]:
        raise FormalRunError("Saved ID order drifted")
    if method == BM25_KEY:
        scores = np.load(paths["documents"], allow_pickle=False)
        if scores.shape != (len(query_ids), len(document_ids)) or not np.isfinite(scores).all():
            raise FormalRunError("Saved BM25 score matrix drifted")
        score_one = lambda index: scores[index]
    elif method == TFIDF_KEY:
        documents = sparse.load_npz(paths["documents"]).tocsr()
        queries = sparse.load_npz(paths["queries"]).tocsr()
        score_one = lambda index: np.asarray((queries[index] @ documents.T).toarray()[0], dtype=np.float32)
    else:
        documents = np.load(paths["documents"], allow_pickle=False)
        queries = np.load(paths["queries"], allow_pickle=False)
        score_one = lambda index: np.asarray(queries[index] @ documents.T, dtype=np.float32)
    excluded = {row["id"]: set(row.get("excluded_ids", ())) for row in data.queries}
    recomputed = {
        query_id: _rank_scores(score_one(index), document_ids=document_ids, excluded_ids=excluded[query_id])
        for index, query_id in enumerate(query_ids)
    }
    saved_rows = json.loads(paths["rankings"].read_text(encoding="utf-8"))
    saved = {row["query_id"]: [(hit["document_id"], hit["score"]) for hit in row["hits"]] for row in saved_rows}
    if recomputed != saved:
        raise FormalRunError("Saved top-100 cannot be recomputed from representations")
    per_query = query_metrics(data, recomputed)
    saved_per_query = {
        row["query_id"]: row["metrics"] for row in json.loads(paths["per_query"].read_text(encoding="utf-8"))
    }
    if per_query != saved_per_query or aggregate_metrics(per_query) != value["quality"]["metrics"]:
        raise FormalRunError("Saved core metrics cannot be recomputed")
    expected_files = {
        "cell.json",
        "cell.json.sha256",
        *(entry["path"] for entry in restricted.values()),
        *(entry["sidecar"]["path"] for entry in restricted.values()),
    }
    if {path.name for path in root.iterdir() if path.is_file()} != expected_files:
        raise FormalRunError("Formal cell contains unexpected or missing files")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-cell")
    run.add_argument("--method", choices=FORMAL_METHODS, required=True)
    run.add_argument("--track", choices=TRACKS, required=True)
    check = subparsers.add_parser("check-cell")
    check.add_argument("--method", choices=FORMAL_METHODS, required=True)
    check.add_argument("--track", choices=TRACKS, required=True)
    subparsers.add_parser("check-all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-cell":
        result: object = run_formal_cell(args.method, args.track)
    elif args.command == "check-cell":
        value = validate_formal_cell(args.method, args.track)
        result = {
            "cell": f"{args.method}:{args.track}",
            "identity": sha256_file(RESULT_ROOT / "cells" / f"{args.method}-{args.track}" / "cell.json"),
            "metrics": value["quality"]["metrics"],
        }
    else:
        result = {
            f"{method}:{track}": sha256_file(RESULT_ROOT / "cells" / f"{method}-{track}" / "cell.json")
            for method in FORMAL_METHODS
            for track in TRACKS
            if validate_formal_cell(method, track)
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
