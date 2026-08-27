"""Shared exact retrieval baselines and evaluation helpers."""

from __future__ import annotations

import hashlib
import heapq
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKENIZER_ID = "unicode-word-lower-v1"
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
METRIC_NAMES = ("ndcg@10", "map@100", "mrr@10", "recall@10", "recall@100")


@dataclass(frozen=True)
class TrackData:
    """One independent retrieval track under a shared text and qrel protocol."""

    name: str
    corpus: tuple[dict[str, Any], ...]
    queries: tuple[dict[str, Any], ...]
    qrels: dict[str, dict[str, int]]


def tokens(text: str) -> list[str]:
    """Tokenize text for the fixed pure-Python BM25 baseline."""
    return TOKEN_PATTERN.findall(text.lower())


def _excluded_ids(query: dict[str, Any]) -> frozenset[str]:
    values = query.get("excluded_ids", ())
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"invalid excluded_ids for query {query.get('id')!r}")
    return frozenset(values)


def validate_track_data(data: TrackData) -> None:
    """Validate the invariants needed by all exact baseline implementations."""
    document_ids = [row.get("id") for row in data.corpus]
    query_ids = [row.get("id") for row in data.queries]
    if not document_ids or any(not isinstance(value, str) or not value for value in document_ids):
        raise ValueError("corpus contains an invalid document id")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("corpus contains duplicate document ids")
    if any(not isinstance(row.get("content"), str) or not row["content"] for row in data.corpus):
        raise ValueError("corpus contains invalid document content")
    if not query_ids or any(not isinstance(value, str) or not value for value in query_ids):
        raise ValueError("queries contain an invalid query id")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("queries contain duplicate query ids")
    if any(not isinstance(row.get("text"), str) or not row["text"] for row in data.queries):
        raise ValueError("queries contain invalid text")
    if set(query_ids) != set(data.qrels):
        raise ValueError("qrels do not cover exactly the query set")
    document_id_set = set(document_ids)
    for query in data.queries:
        query_id = query["id"]
        rels = data.qrels[query_id]
        if not rels or any(not isinstance(grade, int) or grade <= 0 for grade in rels.values()):
            raise ValueError(f"query {query_id!r} has invalid qrels")
        if set(rels) - document_id_set:
            raise ValueError(f"query {query_id!r} has dangling qrels")
        excluded = _excluded_ids(query)
        if excluded - document_id_set:
            raise ValueError(f"query {query_id!r} has dangling excluded_ids")
        if excluded & set(rels):
            raise ValueError(f"query {query_id!r} has gold/excluded overlap")


def query_metrics(data: TrackData, rankings: dict[str, list[tuple[str, float]]]) -> dict[str, dict[str, float]]:
    """Compute canonical metrics separately for every query."""
    validate_track_data(data)
    expected_queries = {row["id"] for row in data.queries}
    if set(rankings) != expected_queries:
        raise ValueError("rankings do not cover exactly the query set")
    corpus_ids = {row["id"] for row in data.corpus}
    output: dict[str, dict[str, float]] = {}
    for query in data.queries:
        query_id = query["id"]
        ranked = [doc_id for doc_id, _ in rankings[query_id]]
        if len(ranked) != len(set(ranked)) or set(ranked) - corpus_ids:
            raise ValueError(f"invalid ranking for query {query_id!r}")
        if set(ranked) & _excluded_ids(query):
            raise ValueError(f"ranking contains an excluded candidate for query {query_id!r}")
        rels = data.qrels[query_id]
        gains = [rels.get(doc_id, 0) for doc_id in ranked]
        dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains[:10]))
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
        binary = [gain > 0 for gain in gains]
        found = 0
        ap = 0.0
        for index, relevant in enumerate(binary[:100], 1):
            if relevant:
                found += 1
                ap += found / index
        first = next((index for index, relevant in enumerate(binary[:10], 1) if relevant), None)
        output[query_id] = {
            "ndcg@10": dcg / idcg if idcg else 0.0,
            "map@100": ap / len(rels),
            "mrr@10": 0.0 if first is None else 1.0 / first,
            "recall@10": sum(binary[:10]) / len(rels),
            "recall@100": sum(binary[:100]) / len(rels),
        }
    return output


def aggregate_metrics(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    """Macro-average canonical metrics over queries."""
    if not per_query:
        raise ValueError("cannot aggregate an empty query metric set")
    return {
        metric: sum(values[metric] for values in per_query.values()) / len(per_query)
        for metric in METRIC_NAMES
    }


def evaluate_rankings(data: TrackData, rankings: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    """Compute canonical macro metrics for one independent track."""
    return aggregate_metrics(query_metrics(data, rankings))


def bootstrap_confidence_intervals(
    per_query: dict[str, dict[str, float]],
    *,
    samples: int = 10_000,
    seed: int = 20_260_826,
) -> dict[str, dict[str, float | int]]:
    """Return deterministic percentile bootstrap intervals over queries."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    query_ids = sorted(per_query)
    if not query_ids:
        raise ValueError("cannot bootstrap an empty query metric set")
    matrix = np.asarray([[per_query[query_id][metric] for metric in METRIC_NAMES] for query_id in query_ids])
    generator = np.random.default_rng(seed)
    estimates = np.empty((samples, len(METRIC_NAMES)), dtype=np.float64)
    for start in range(0, samples, 1_000):
        batch = min(1_000, samples - start)
        indices = generator.integers(0, len(query_ids), size=(batch, len(query_ids)))
        estimates[start : start + batch] = matrix[indices].mean(axis=1)
    bounds = np.percentile(estimates, [2.5, 97.5], axis=0)
    means = matrix.mean(axis=0)
    return {
        metric: {
            "mean": float(means[index]),
            "low": float(bounds[0, index]),
            "high": float(bounds[1, index]),
            "samples": samples,
            "seed": seed,
        }
        for index, metric in enumerate(METRIC_NAMES)
    }


def paired_bootstrap_deltas(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    *,
    samples: int = 10_000,
    seed: int = 20_260_826,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap left-minus-right metric deltas over matched queries."""
    if set(left) != set(right) or not left:
        raise ValueError("paired metric sets must cover the same non-empty queries")
    deltas = {
        query_id: {metric: left[query_id][metric] - right[query_id][metric] for metric in METRIC_NAMES}
        for query_id in left
    }
    return bootstrap_confidence_intervals(deltas, samples=samples, seed=seed)


def bm25_rank(
    data: TrackData,
    top_k: int = 100,
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Any]]:
    """Run fixed-tokenizer exact BM25 over one independent candidate pool."""
    validate_track_data(data)
    started = time.perf_counter()
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = np.empty(len(data.corpus), dtype=np.int32)
    for index, row in enumerate(data.corpus):
        counts = Counter(tokens(row["content"]))
        lengths[index] = sum(counts.values())
        for token, frequency in counts.items():
            postings[token].append((index, frequency))
    build_s = time.perf_counter() - started
    avgdl = float(lengths.mean())
    document_index = {row["id"]: index for index, row in enumerate(data.corpus)}
    rankings: dict[str, list[tuple[str, float]]] = {}
    search_started = time.perf_counter()
    for query in data.queries:
        excluded = {document_index[doc_id] for doc_id in _excluded_ids(query)}
        scores: defaultdict[int, float] = defaultdict(float)
        for token, query_frequency in Counter(tokens(query["text"])).items():
            posting = postings.get(token, ())
            idf = math.log(1.0 + (len(data.corpus) - len(posting) + 0.5) / (len(posting) + 0.5))
            for doc_index, frequency in posting:
                if doc_index in excluded:
                    continue
                denominator = frequency + k1 * (1.0 - b + b * lengths[doc_index] / avgdl)
                scores[doc_index] += query_frequency * idf * frequency * (k1 + 1.0) / denominator
        best = heapq.nlargest(top_k, scores.items(), key=lambda item: (item[1], -item[0]))
        if len(best) < top_k:
            selected = {index for index, _ in best}
            best.extend(
                (index, 0.0)
                for index in range(len(data.corpus))
                if index not in selected and index not in excluded
            )
            best = sorted(best, key=lambda item: (-item[1], item[0]))[:top_k]
        rankings[query["id"]] = [(data.corpus[index]["id"], float(score)) for index, score in best]
    search_s = time.perf_counter() - search_started
    text_bytes = sum(len(row["content"].encode("utf-8")) for row in data.corpus)
    index_bytes = int(
        lengths.nbytes
        + sum(len(token.encode("utf-8")) + len(items) * 8 for token, items in postings.items())
    )
    return rankings, {
        "tokenizer": TOKENIZER_ID,
        "k1": k1,
        "b": b,
        "build_s": build_s,
        "search_s": search_s,
        "documents_per_second": len(data.corpus) / build_s if build_s else 0.0,
        "queries_per_second": len(data.queries) / search_s if search_s else 0.0,
        "index_bytes": index_bytes,
        "input_text_bytes": text_bytes,
    }


def _truncation_counts(model: Any, data: TrackData) -> dict[str, int]:
    tokenizer = model.tokenizer
    maximum = int(model.max_seq_length)
    tokenizer_maximum = tokenizer.model_max_length
    tokenizer.model_max_length = 10**30

    def length(text: str) -> int:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False, return_attention_mask=False)
        return len(encoded["input_ids"])

    try:
        document_lengths = {row["id"]: length(row["content"]) for row in data.corpus}
        query_lengths = {row["id"]: length(row["text"]) for row in data.queries}
    finally:
        tokenizer.model_max_length = tokenizer_maximum
    gold_ids = {doc_id for rels in data.qrels.values() for doc_id in rels}
    return {
        "model_max_sequence_length": maximum,
        "document_truncated_count": sum(value > maximum for value in document_lengths.values()),
        "query_truncated_count": sum(value > maximum for value in query_lengths.values()),
        "gold_document_truncated_count": sum(document_lengths[doc_id] > maximum for doc_id in gold_ids),
        "unique_gold_documents": len(gold_ids),
    }


def dense_rank(
    data: TrackData,
    model_id: str,
    revision: str,
    top_k: int = 100,
    query_block_size: int = 32,
    doc_block_size: int = 4096,
    batch_size: int = 128,
    model_path: str | Path | None = None,
    max_sequence_length: int | None = None,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Any]]:
    """Run normalized blockwise exact cosine search over one candidate pool."""
    from sentence_transformers import SentenceTransformer

    import torch

    validate_track_data(data)
    model_source = str(model_path) if model_path is not None else model_id
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": True,
    }
    if model_path is None:
        model_kwargs["revision"] = revision
    model = SentenceTransformer(model_source, **model_kwargs)
    declared_max_sequence_length = int(model.max_seq_length)
    if max_sequence_length is not None:
        if max_sequence_length < 1 or max_sequence_length > declared_max_sequence_length:
            raise ValueError("configured max sequence length exceeds the model declaration")
        model.max_seq_length = max_sequence_length
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    truncation = _truncation_counts(model, data)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    document_started = time.perf_counter()
    document_embeddings = model.encode(
        [row["content"] for row in data.corpus],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    document_encode_s = time.perf_counter() - document_started
    query_started = time.perf_counter()
    query_embeddings = model.encode(
        [row["text"] for row in data.queries],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_encode_s = time.perf_counter() - query_started
    document_index = {row["id"]: index for index, row in enumerate(data.corpus)}
    rankings: dict[str, list[tuple[str, float]]] = {}
    search_started = time.perf_counter()
    max_score_block_elements = 0
    for query_start in range(0, len(data.queries), query_block_size):
        query_block = query_embeddings[query_start : query_start + query_block_size]
        heaps: list[list[tuple[float, int, int]]] = [[] for _ in range(len(query_block))]
        exclusions = [
            {document_index[doc_id] for doc_id in _excluded_ids(data.queries[query_start + offset])}
            for offset in range(len(query_block))
        ]
        for doc_start in range(0, len(data.corpus), doc_block_size):
            scores = query_block @ document_embeddings[doc_start : doc_start + doc_block_size].T
            max_score_block_elements = max(max_score_block_elements, scores.size)
            for row_index, row_scores in enumerate(scores):
                heap = heaps[row_index]
                excluded = exclusions[row_index]
                for local_index, score in enumerate(row_scores):
                    absolute_index = doc_start + local_index
                    if absolute_index in excluded:
                        continue
                    item = (float(score), -absolute_index, absolute_index)
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
        for offset, heap in enumerate(heaps):
            query = data.queries[query_start + offset]
            rankings[query["id"]] = [
                (data.corpus[index]["id"], score)
                for score, _, index in sorted(heap, key=lambda item: (-item[0], item[2]))
            ]
    search_s = time.perf_counter() - search_started
    peak_vram = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    return rankings, {
        "model_id": model_id,
        "revision": revision,
        "trust_remote_code": False,
        "device": device,
        "dimensions": int(document_embeddings.shape[1]),
        "normalize_embeddings": True,
        "similarity": "exact_cosine",
        "batch_size": batch_size,
        "document_encode_s": document_encode_s,
        "query_encode_s": query_encode_s,
        "encode_s": document_encode_s + query_encode_s,
        "search_s": search_s,
        "document_throughput_per_second": len(data.corpus) / document_encode_s if document_encode_s else 0.0,
        "query_throughput_per_second": len(data.queries) / query_encode_s if query_encode_s else 0.0,
        "embedding_bytes": int(document_embeddings.nbytes + query_embeddings.nbytes),
        "index_bytes": int(document_embeddings.nbytes),
        "query_block_size": query_block_size,
        "document_block_size": doc_block_size,
        "max_score_block_elements": max_score_block_elements,
        "full_matrix_elements": len(data.queries) * len(data.corpus),
        "peak_vram_bytes": peak_vram,
        "declared_max_sequence_length": declared_max_sequence_length,
        "configured_max_sequence_length": int(model.max_seq_length),
        **truncation,
    }


def snapshot_identity(repo_id: str, revision: str) -> dict[str, Any]:
    """Hash every regular file in a fixed local Hugging Face snapshot."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id, revision=revision, local_files_only=True))
    entries = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size = path.stat().st_size
        value = f"{relative}\0{size}\0{digest.hexdigest()}\n".encode()
        aggregate.update(value)
        total_bytes += size
        entries.append({"path": relative, "bytes": size, "sha256": digest.hexdigest()})
    if not entries:
        raise ValueError("fixed model snapshot is empty")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "files": len(entries),
        "bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "file_manifest": entries,
    }


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Write canonical JSONL and return its deterministic identity."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    with output.open("wb") as handle:
        for row in rows:
            content = (
                json_dumps(row) + "\n"
            ).encode("utf-8")
            handle.write(content)
            digest.update(content)
            count += 1
    return {"path": output.name, "rows": count, "bytes": output.stat().st_size, "sha256": digest.hexdigest()}


def json_dumps(value: Any) -> str:
    """Serialize one canonical JSON value."""
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
