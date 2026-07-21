"""Deterministic grouped-chunk retrieval contract smoke task."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from mm_embed.data.late_chunking_retrieval import (
    CHUNKER_VERSION,
    DATASET_VERSION,
    ChunkLayout,
    HardNegative,
    LateChunkingRetrievalFixture,
    Qrel,
    RetrievalQuery,
    load_late_chunking_retrieval_fixture,
)
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, ModalityType
from mm_embed.providers.grouped_chunks import (
    GroupedChunkEmbeddingProvider,
    GroupedChunkEmbeddingRequest,
    GroupedChunkEmbeddingResult,
    GroupedChunkGroup,
    GroupedChunkInput,
    grouped_request_fingerprint,
    grouped_result_to_dict,
    validate_grouped_chunk_result,
)
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import cosine_similarity_matrix

logger = logging.getLogger(__name__)


class LateChunkingRetrievalTask(EvalTask):
    """Fixture-only validation of grouped contextual chunk retrieval."""

    name = "late_chunking_retrieval"
    description = "No-publish grouped chunk retrieval contract smoke"
    required_modalities = {ModalityType.TEXT}

    def __init__(
        self,
        dataset_version: str = DATASET_VERSION,
        fixture_only: bool = True,
        **kwargs: Any,
    ) -> None:
        if dataset_version != DATASET_VERSION:
            raise ValueError(f"Unsupported late-chunking fixture version: {dataset_version}")
        if not fixture_only:
            raise ValueError("Late-chunking retrieval is currently fixture_only and not publishable")
        self.dataset_version = dataset_version
        self.fixture_only = fixture_only

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        model_name = getattr(provider, "model", "unknown")
        try:
            if not isinstance(provider, GroupedChunkEmbeddingProvider):
                raise TypeError("Provider does not implement the explicit grouped-chunk embedding capability")

            fixture = load_late_chunking_retrieval_fixture()
            query_inputs = [
                EmbeddingInput(
                    modality=ModalityType.TEXT,
                    content=query.text,
                    metadata={"query_id": query.query_id, "split": query.split},
                )
                for query in fixture.queries
            ]
            query_result = provider.embed(query_inputs, task_type="retrieval_query")
            query_embeddings = np.asarray(query_result.embeddings, dtype=float)
            if query_embeddings.ndim != 2 or query_embeddings.shape[0] != len(fixture.queries):
                raise ValueError("Query embedding result does not preserve query count/order")

            layout_evaluations: dict[str, dict[str, dict[str, Any]]] = {}
            serialized_primary_results: dict[str, Any] = {}
            grouped_call_details: dict[str, Any] = {}

            for layout in fixture.layouts:
                request_by_strategy: dict[str, GroupedChunkEmbeddingRequest] = {}
                result_by_strategy: dict[str, GroupedChunkEmbeddingResult] = {}
                layout_evaluations[layout.layout_id] = {}

                for strategy in ("deterministic_independent_stub", "deterministic_contextual_stub"):
                    request = _build_grouped_request(fixture, layout, strategy)
                    result = provider.embed_grouped_chunks(request)
                    try:
                        validate_grouped_chunk_result(request, result)
                    except ValueError as exc:
                        if strategy == "deterministic_contextual_stub":
                            raise ValueError(f"Unpairable context delta for {layout.layout_id}: {exc}") from exc
                        raise

                    request_by_strategy[strategy] = request
                    result_by_strategy[strategy] = result
                    chunk_ids, document_ids, chunk_embeddings = _flatten_grouped_result(result)
                    similarities = cosine_similarity_matrix(query_embeddings, chunk_embeddings)
                    metrics, per_query = _evaluate_layout(
                        fixture=fixture,
                        layout=layout,
                        chunk_ids=chunk_ids,
                        document_ids=document_ids,
                        similarities=similarities,
                    )
                    layout_evaluations[layout.layout_id][strategy] = {
                        "metrics": metrics,
                        "per_query": per_query,
                    }
                    grouped_call_details[f"{layout.layout_id}:{strategy}"] = {
                        "request_fingerprint": grouped_request_fingerprint(request),
                        "groups": len(request.groups),
                        "chunks": len(chunk_ids),
                        "latency_ms": result.latency_ms,
                        "token_usage": result.token_usage,
                        "cost_usd": result.cost_usd,
                    }

                _assert_pairable_context_results(
                    request_by_strategy["deterministic_independent_stub"],
                    result_by_strategy["deterministic_independent_stub"],
                    request_by_strategy["deterministic_contextual_stub"],
                    result_by_strategy["deterministic_contextual_stub"],
                )
                if layout.layout_id == "fixed_192_v1":
                    serialized_primary_results = {
                        "independent": grouped_result_to_dict(
                            result_by_strategy["deterministic_independent_stub"]
                        ),
                        "contextual": grouped_result_to_dict(
                            result_by_strategy["deterministic_contextual_stub"]
                        ),
                    }

            metrics = _flatten_metrics(layout_evaluations, fixture.queries)
            fixed_contextual = layout_evaluations["fixed_192_v1"]["deterministic_contextual_stub"]["metrics"]
            metrics["chunk_ndcg@10"] = fixed_contextual["chunk_ndcg@10"]

            details = {
                "dataset_version": fixture.dataset_version,
                "split": fixture.split,
                "fixture_only": fixture.fixture_only,
                "license_status": fixture.license_status,
                "not_for_publication": fixture.license_status == "not_for_publication",
                "leaderboard_publish": fixture.leaderboard_publish,
                "network": fixture.network,
                "provider_api_calls": 0,
                "model_downloads": 0,
                "n_documents": len(fixture.documents),
                "n_queries": len(fixture.queries),
                "n_context_required_queries": sum(query.context_required for query in fixture.queries),
                "n_irrelevant_context_controls": sum(not query.context_required for query in fixture.queries),
                "label_sha256": fixture.label_sha256,
                "layout_counts": {
                    layout.layout_id: {
                        "chunks": len(layout.chunks),
                        "positive_qrels": sum(qrel.layout_id == layout.layout_id for qrel in fixture.qrels),
                        "hard_negative_links": sum(
                            negative.layout_id == layout.layout_id for negative in fixture.hard_negatives
                        ),
                    }
                    for layout in fixture.layouts
                },
                "slice_counts": _slice_counts(fixture),
                "grouped_calls": grouped_call_details,
                "serialized_primary_results": serialized_primary_results,
                "query_latency_ms": query_result.latency_ms,
            }
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=model_name,
                metrics=metrics,
                details=details,
            )
        except Exception as exc:
            return EvalResult(
                task_name=self.name,
                provider_name=getattr(provider, "name", "unknown"),
                model_name=model_name,
                metrics={},
                error=str(exc),
            )


def _build_grouped_request(
    fixture: LateChunkingRetrievalFixture,
    layout: ChunkLayout,
    strategy: str,
) -> GroupedChunkEmbeddingRequest:
    chunks_by_document: dict[str, list[Any]] = defaultdict(list)
    for chunk in layout.chunks:
        chunks_by_document[chunk.document_id].append(chunk)

    groups = tuple(
        GroupedChunkGroup(
            document_id=document.document_id,
            chunks=tuple(
                GroupedChunkInput(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    text_sha256=chunk.text_sha256,
                )
                for chunk in chunks_by_document[document.document_id]
            ),
        )
        for document in fixture.documents
    )
    return GroupedChunkEmbeddingRequest(
        layout_id=layout.layout_id,
        chunker_version=CHUNKER_VERSION,
        strategy=strategy,
        groups=groups,
    )


def _flatten_grouped_result(
    result: GroupedChunkEmbeddingResult,
) -> tuple[list[str], list[str], np.ndarray]:
    chunks = [chunk for group in result.groups for chunk in group.chunks]
    return (
        [chunk.chunk_id for chunk in chunks],
        [chunk.document_id for chunk in chunks],
        np.vstack([np.asarray(chunk.embedding, dtype=float) for chunk in chunks]),
    )


def _stable_ranking(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


def _reciprocal_rank(ranked_chunk_ids: list[str], positive_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in positive_ids:
            return 1.0 / rank
    return 0.0


def _ndcg(ranked_chunk_ids: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = sum(
        (2**relevance.get(chunk_id, 0) - 1) / np.log2(rank + 2)
        for rank, chunk_id in enumerate(ranked_chunk_ids[:k])
        if relevance.get(chunk_id, 0) > 0
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / np.log2(rank + 2) for rank, grade in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def _evaluate_layout(
    *,
    fixture: LateChunkingRetrievalFixture,
    layout: ChunkLayout,
    chunk_ids: list[str],
    document_ids: list[str],
    similarities: np.ndarray,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    index_by_chunk_id = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    if len(index_by_chunk_id) != len(chunk_ids):
        raise ValueError(f"Duplicate returned chunk ids for {layout.layout_id}")

    qrels_by_query: dict[str, list[Qrel]] = defaultdict(list)
    for qrel in fixture.qrels:
        if qrel.layout_id == layout.layout_id:
            qrels_by_query[qrel.query_id].append(qrel)
    negatives_by_query: dict[str, list[HardNegative]] = defaultdict(list)
    for negative in fixture.hard_negatives:
        if negative.layout_id == layout.layout_id:
            negatives_by_query[negative.query_id].append(negative)

    per_query: dict[str, dict[str, float]] = {}
    for query_index, query in enumerate(fixture.queries):
        ranked_indices = _stable_ranking(similarities[query_index])
        ranked_chunk_ids = [chunk_ids[index] for index in ranked_indices]
        ranked_document_ids = [document_ids[index] for index in ranked_indices]
        relevance = {qrel.chunk_id: qrel.relevance for qrel in qrels_by_query[query.query_id]}
        positive_ids = set(relevance)

        unique_parents: list[str] = []
        for document_id in ranked_document_ids:
            if document_id not in unique_parents:
                unique_parents.append(document_id)

        query_metrics = {
            "chunk_recall@1": float(any(chunk_id in positive_ids for chunk_id in ranked_chunk_ids[:1])),
            "chunk_recall@5": float(any(chunk_id in positive_ids for chunk_id in ranked_chunk_ids[:5])),
            "chunk_recall@10": float(any(chunk_id in positive_ids for chunk_id in ranked_chunk_ids[:10])),
            "chunk_mrr": _reciprocal_rank(ranked_chunk_ids, positive_ids),
            "chunk_ndcg@10": _ndcg(ranked_chunk_ids, relevance, 10),
            "evidence_span_recall@1": float(any(chunk_id in positive_ids for chunk_id in ranked_chunk_ids[:1])),
            "evidence_span_recall@5": float(any(chunk_id in positive_ids for chunk_id in ranked_chunk_ids[:5])),
            "evidence_span_recall@10": float(any(chunk_id in positive_ids for chunk_id in ranked_chunk_ids[:10])),
            "parent_recall@1": float(query.gold_document_id in unique_parents[:1]),
            "parent_recall@5": float(query.gold_document_id in unique_parents[:5]),
            "boundary_failure_rate@1": float(
                query.gold_document_id in ranked_document_ids[:1]
                and not any(chunk_id in positive_ids for chunk_id in ranked_chunk_ids[:1])
            ),
            "boundary_failure_rate@5": float(
                query.gold_document_id in ranked_document_ids[:5]
                and not any(chunk_id in positive_ids for chunk_id in ranked_chunk_ids[:5])
            ),
        }

        hard_negatives = negatives_by_query[query.query_id]
        hard_pool_ids = [*positive_ids, *(negative.chunk_id for negative in hard_negatives)]
        hard_pool_ids = sorted(set(hard_pool_ids), key=lambda chunk_id: index_by_chunk_id[chunk_id])
        hard_ranked_ids = sorted(
            hard_pool_ids,
            key=lambda chunk_id: (-similarities[query_index, index_by_chunk_id[chunk_id]], index_by_chunk_id[chunk_id]),
        )
        neighbor_id = next(
            negative.chunk_id
            for negative in hard_negatives
            if negative.negative_family == "misleading_gold_parent_neighbor"
        )
        best_positive_score = max(similarities[query_index, index_by_chunk_id[chunk_id]] for chunk_id in positive_ids)
        query_metrics.update(
            {
                "hard_recall@1": float(any(chunk_id in positive_ids for chunk_id in hard_ranked_ids[:1])),
                "hard_recall@5": float(any(chunk_id in positive_ids for chunk_id in hard_ranked_ids[:5])),
                "hard_mrr": _reciprocal_rank(hard_ranked_ids, positive_ids),
                "hard_ndcg@10": _ndcg(hard_ranked_ids, relevance, 10),
                "neighbor_confusion_rate": float(
                    similarities[query_index, index_by_chunk_id[neighbor_id]] > best_positive_score
                ),
            }
        )
        per_query[query.query_id] = query_metrics

    metric_names = next(iter(per_query.values())).keys()
    metrics = {
        metric_name: float(np.mean([values[metric_name] for values in per_query.values()]))
        for metric_name in metric_names
    }
    return metrics, per_query


def _mean_paired_delta(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    metric: str,
    query_ids: list[str] | None = None,
) -> float:
    selected = query_ids or list(left)
    if set(selected) - set(left) or set(selected) - set(right):
        raise ValueError(f"Unpairable query ids for paired metric {metric}")
    return float(np.mean([left[query_id][metric] - right[query_id][metric] for query_id in selected]))


def _flatten_metrics(
    evaluations: dict[str, dict[str, dict[str, Any]]],
    queries: tuple[RetrievalQuery, ...],
) -> dict[str, float]:
    layout_prefixes = {
        "fixed_192_v1": "fixed",
        "fixed_192_overlap_48_v1": "overlap",
        "structure_adaptive_fixture_v1": "adaptive",
    }
    strategy_prefixes = {
        "deterministic_independent_stub": "independent",
        "deterministic_contextual_stub": "contextual",
    }
    metrics: dict[str, float] = {}
    for layout_id, strategies in evaluations.items():
        for strategy, evaluation in strategies.items():
            prefix = f"{layout_prefixes[layout_id]}_{strategy_prefixes[strategy]}"
            metrics.update({f"{prefix}_{name}": value for name, value in evaluation["metrics"].items()})

    fixed = evaluations["fixed_192_v1"]
    fixed_independent = fixed["deterministic_independent_stub"]["per_query"]
    fixed_contextual = fixed["deterministic_contextual_stub"]["per_query"]
    for metric in ("chunk_mrr", "chunk_ndcg@10", "hard_mrr"):
        metrics[f"context_gain_{metric}"] = _mean_paired_delta(
            fixed_contextual,
            fixed_independent,
            metric,
        )

    overlap_independent = evaluations["fixed_192_overlap_48_v1"]["deterministic_independent_stub"][
        "per_query"
    ]
    adaptive_independent = evaluations["structure_adaptive_fixture_v1"]["deterministic_independent_stub"][
        "per_query"
    ]
    for metric in ("evidence_span_recall@5", "parent_recall@5"):
        metrics[f"overlap_gain_{metric}"] = _mean_paired_delta(
            overlap_independent,
            fixed_independent,
            metric,
        )
        metrics[f"adaptive_gain_{metric}"] = _mean_paired_delta(
            adaptive_independent,
            fixed_independent,
            metric,
        )

    control_query_ids = [query.query_id for query in queries if not query.context_required]
    metrics["irrelevant_context_delta"] = _mean_paired_delta(
        fixed_contextual,
        fixed_independent,
        "chunk_mrr",
        control_query_ids,
    )
    return metrics


def _assert_pairable_context_results(
    independent_request: GroupedChunkEmbeddingRequest,
    independent_result: GroupedChunkEmbeddingResult,
    contextual_request: GroupedChunkEmbeddingRequest,
    contextual_result: GroupedChunkEmbeddingResult,
) -> None:
    def signature(request: GroupedChunkEmbeddingRequest, result: GroupedChunkEmbeddingResult) -> tuple[Any, ...]:
        request_chunks = tuple(
            (group.document_id, chunk.chunk_id, chunk.chunk_index, chunk.text)
            for group in request.groups
            for chunk in group.chunks
        )
        result_chunks = tuple(
            (group.document_id, chunk.chunk_id, chunk.chunk_index, chunk.returned_text)
            for group in result.groups
            for chunk in group.chunks
        )
        return request.layout_id, request_chunks, result_chunks

    if signature(independent_request, independent_result) != signature(contextual_request, contextual_result):
        raise ValueError("Unpairable context delta: chunk ids, text, order, or document groups differ")


def _slice_counts(fixture: LateChunkingRetrievalFixture) -> dict[str, dict[str, int]]:
    fixed_layout = next(layout for layout in fixture.layouts if layout.layout_id == "fixed_192_v1")
    span_by_id = {span.span_id: span for span in fixture.spans}
    evidence_position: Counter[str] = Counter()
    context_distance: Counter[str] = Counter()

    for query in fixture.queries:
        evidence_span = span_by_id[query.evidence_span_ids[0]]
        evidence_chunk = next(
            chunk
            for chunk in fixed_layout.chunks
            if chunk.document_id == query.gold_document_id
            and chunk.char_start <= evidence_span.char_start
            and chunk.char_end >= evidence_span.char_end
        )
        evidence_position[f"chunk_{evidence_chunk.chunk_index}"] += 1
        if not query.context_required:
            context_distance["irrelevant_context_control"] += 1
            continue
        context_span = span_by_id[query.required_context_span_ids[0]]
        context_chunk = next(
            chunk
            for chunk in fixed_layout.chunks
            if chunk.document_id == query.gold_document_id
            and chunk.char_start <= context_span.char_start
            and chunk.char_end >= context_span.char_end
        )
        context_distance[f"{evidence_chunk.chunk_index - context_chunk.chunk_index}_chunks"] += 1

    return {
        "failure_family": dict(sorted(Counter(query.family for query in fixture.queries).items())),
        "context_requirement": {
            "context_required": sum(query.context_required for query in fixture.queries),
            "irrelevant_context": sum(not query.context_required for query in fixture.queries),
        },
        "context_distance": dict(sorted(context_distance.items())),
        "evidence_position": dict(sorted(evidence_position.items())),
        "parent_length": {"768_whitespace_units": len(fixture.documents)},
        "lexical_shortcut": dict(
            sorted(Counter(str(query.lexical_shortcut_present).lower() for query in fixture.queries).items())
        ),
        "negative_family": dict(
            sorted(Counter(negative.negative_family for negative in fixture.hard_negatives).items())
        ),
        "segmentation_layout": {layout.layout_id: len(layout.chunks) for layout in fixture.layouts},
        "mapping_status": {"exact": sum(len(layout.chunks) * 2 for layout in fixture.layouts)},
    }
