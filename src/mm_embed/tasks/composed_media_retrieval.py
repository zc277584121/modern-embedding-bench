"""Deterministic composed-media contract and retrieval fixture task."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from mm_embed.data.composed_media_retrieval import (
    DATASET_VERSION,
    FIXTURE_ROOT,
    ComposedMediaHardNegative,
    ComposedMediaQrel,
    ComposedMediaQuery,
    ComposedMediaRetrievalFixture,
    fixture_tracked_bytes,
    fixture_tree_sha256,
    load_composed_media_retrieval_fixture,
)
from mm_embed.providers.base import ModalityType
from mm_embed.providers.composed_media import (
    ComposedMediaEmbeddingProvider,
    ComposedMediaEmbeddingRequest,
    ComposedMediaEmbeddingResult,
    request_fingerprint,
    validate_composed_result,
)
from mm_embed.tasks.base import EvalResult, EvalTask


@dataclass(frozen=True)
class RetrievalEvaluation:
    query_ids: tuple[str, ...]
    corpus_ids: tuple[str, ...]
    preprocessing: str
    composition_mode: str
    track_label: str
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]
    slices: dict[str, dict[str, dict[str, float | int]]]


def build_composed_request(
    *,
    items: Iterable,
    provider: str,
    model_id: str,
    model_revision: str,
    dimensions: int,
    task_route: str,
    preprocessing: str,
    composition_mode: str,
    track_label: str,
    fusion_strategy: str,
) -> ComposedMediaEmbeddingRequest:
    request = ComposedMediaEmbeddingRequest(
        items=tuple(items),
        provider=provider,
        model_id=model_id,
        model_revision=model_revision,
        dimensions=dimensions,
        task_route=task_route,
        preprocessing=preprocessing,
        composition_mode=composition_mode,
        track_label=track_label,
        fusion_strategy=fusion_strategy,
        request_sha256="",
    )
    return replace(request, request_sha256=request_fingerprint(request))


def stable_corpus_ranking(corpus_ids: list[str], scores: np.ndarray) -> list[int]:
    """Rank by descending exact score then corpus id UTF-8 bytes."""
    return sorted(range(len(corpus_ids)), key=lambda index: (-float(scores[index]), corpus_ids[index].encode("utf-8")))


def _ndcg(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = sum(
        (2**relevance.get(corpus_id, 0) - 1) / np.log2(rank + 2)
        for rank, corpus_id in enumerate(ranked_ids[:k])
        if relevance.get(corpus_id, 0) > 0
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / np.log2(rank + 2) for rank, grade in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def _average_precision(ranked_ids: list[str], positive_ids: set[str], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, corpus_id in enumerate(ranked_ids[:k], start=1):
        if corpus_id in positive_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(positive_ids), k) if positive_ids else 0.0


def _reciprocal_rank(ranked_ids: list[str], positive_ids: set[str], k: int) -> float:
    for rank, corpus_id in enumerate(ranked_ids[:k], start=1):
        if corpus_id in positive_ids:
            return 1.0 / rank
    return 0.0


def _query_metrics(ranked_ids: list[str], relevance: dict[str, int]) -> dict[str, float]:
    positive_ids = set(relevance)
    values = {
        "composed_ndcg@10": _ndcg(ranked_ids, relevance, 10),
        "composed_map@5": _average_precision(ranked_ids, positive_ids, 5),
        "composed_mrr@10": _reciprocal_rank(ranked_ids, positive_ids, 10),
        "composed_hit_rate@10": float(any(corpus_id in positive_ids for corpus_id in ranked_ids[:10])),
    }
    for k in (1, 5, 10):
        coverage = len(positive_ids.intersection(ranked_ids[:k])) / len(positive_ids)
        values[f"composed_recall@{k}"] = coverage
        values[f"positive_set_coverage@{k}"] = coverage
    return values


def _aggregate(per_query: dict[str, dict[str, float]], query_ids: Iterable[str]) -> dict[str, float]:
    selected = list(query_ids)
    if not selected:
        return {}
    return {
        metric: float(np.mean([per_query[query_id][metric] for query_id in selected]))
        for metric in next(iter(per_query.values()))
    }


def _slice_groups(
    fixture: ComposedMediaRetrievalFixture,
    track_label: str,
) -> dict[str, dict[str, set[str]]]:
    positive_count = CounterLike(qrel.query_id for qrel in fixture.qrels)
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for query in fixture.queries:
        query_id = query.query_id
        groups["query_shape"][query.shape].add(query_id)
        groups["part_count"][str(len(query.item.parts))].add(query_id)
        cardinality = "multi_positive" if positive_count[query_id] > 1 else "single_positive"
        groups["positive_cardinality"][cardinality].add(query_id)
        groups["target_modality"][query.target_modality].add(query_id)
        groups["reference_duration_bucket"][query.reference_duration_bucket].add(query_id)
        groups["fusion_track"][track_label].add(query_id)
        groups["order_sensitive_family"][query.family].add(query_id)
        groups["media_reuse"][query.media_reuse].add(query_id)
    for negative in fixture.hard_negatives:
        groups["hard_negative_family"][negative.negative_family].add(negative.query_id)
    return groups


class CounterLike(dict[str, int]):
    def __init__(self, values: Iterable[str]) -> None:
        super().__init__()
        for value in values:
            self[value] = self.get(value, 0) + 1


def _slice_metrics(
    fixture: ComposedMediaRetrievalFixture,
    per_query: dict[str, dict[str, float]],
    track_label: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    slices: dict[str, dict[str, dict[str, float | int]]] = {}
    for axis, labels in _slice_groups(fixture, track_label).items():
        slices[axis] = {}
        for label, query_ids in sorted(labels.items()):
            metrics = _aggregate(per_query, sorted(query_ids))
            slices[axis][label] = {"query_count": len(query_ids), **metrics}
    return slices


def evaluate_composed_retrieval(
    *,
    fixture: ComposedMediaRetrievalFixture,
    query_result: ComposedMediaEmbeddingResult,
    corpus_result: ComposedMediaEmbeddingResult,
    preprocessing: str,
) -> RetrievalEvaluation:
    query_ids = tuple(row.item_id for row in query_result.rows)
    corpus_ids = tuple(row.item_id for row in corpus_result.rows)
    query_matrix = np.asarray([row.embedding for row in query_result.rows], dtype=float)
    corpus_matrix = np.asarray([row.embedding for row in corpus_result.rows], dtype=float)
    if query_matrix.ndim != 2 or corpus_matrix.ndim != 2 or query_matrix.shape[1] != corpus_matrix.shape[1]:
        raise ValueError("Query and corpus embedding dimensions are incompatible")
    query_norms = np.linalg.norm(query_matrix, axis=1)
    corpus_norms = np.linalg.norm(corpus_matrix, axis=1)
    if not np.isfinite(query_matrix).all() or not np.isfinite(corpus_matrix).all():
        raise ValueError("Retrieval matrices contain non-finite values")
    if np.any(query_norms == 0) or np.any(corpus_norms == 0):
        raise ValueError("Retrieval matrices contain zero-norm vectors")
    similarities = (query_matrix / query_norms[:, None]) @ (corpus_matrix / corpus_norms[:, None]).T

    qrels_by_query: dict[str, list[ComposedMediaQrel]] = defaultdict(list)
    negatives_by_query: dict[str, list[ComposedMediaHardNegative]] = defaultdict(list)
    for qrel in fixture.qrels:
        qrels_by_query[qrel.query_id].append(qrel)
    for negative in fixture.hard_negatives:
        negatives_by_query[negative.query_id].append(negative)
    corpus_index = {corpus_id: index for index, corpus_id in enumerate(corpus_ids)}

    per_query: dict[str, dict[str, float]] = {}
    for query_index, query_id in enumerate(query_ids):
        ranking = stable_corpus_ranking(list(corpus_ids), similarities[query_index])
        ranked_ids = [corpus_ids[index] for index in ranking]
        relevance = {qrel.corpus_id: qrel.relevance for qrel in qrels_by_query[query_id]}
        values = _query_metrics(ranked_ids, relevance)

        hard_ids = sorted(
            [*relevance, *(negative.corpus_id for negative in negatives_by_query[query_id])],
            key=lambda corpus_id: corpus_id.encode("utf-8"),
        )
        hard_ranking = sorted(
            hard_ids,
            key=lambda corpus_id: (
                -float(similarities[query_index, corpus_index[corpus_id]]),
                corpus_id.encode("utf-8"),
            ),
        )
        hard_values = _query_metrics(hard_ranking, relevance)
        values.update({f"hard_pool_{name.removeprefix('composed_')}": value for name, value in hard_values.items()})
        best_positive = max(similarities[query_index, corpus_index[corpus_id]] for corpus_id in relevance)
        values["hard_negative_outrank_rate"] = float(
            any(
                similarities[query_index, corpus_index[negative.corpus_id]] > best_positive
                for negative in negatives_by_query[query_id]
            )
        )
        per_query[query_id] = values

    metrics = _aggregate(per_query, query_ids)
    return RetrievalEvaluation(
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        preprocessing=preprocessing,
        composition_mode=query_result.composition_mode,
        track_label=query_result.track_label,
        metrics=metrics,
        per_query=per_query,
        slices=_slice_metrics(fixture, per_query, query_result.track_label),
    )


def paired_fusion_deltas(
    native: RetrievalEvaluation,
    system: RetrievalEvaluation,
) -> dict[str, float]:
    """Return paired deltas only for exactly compatible ids and preprocessing."""
    if native.query_ids != system.query_ids or native.corpus_ids != system.corpus_ids:
        raise ValueError("Paired fusion diagnostics require identical ordered query/corpus ids")
    if native.preprocessing != system.preprocessing:
        raise ValueError("Paired fusion diagnostics require identical preprocessing identity")
    if native.track_label != "provider_valid_embedding" or system.track_label != "benchmark_system_fusion":
        raise ValueError("Paired fusion diagnostics require native and system-fusion tracks")
    return {
        "native_fusion_gain_ndcg@10": native.metrics["composed_ndcg@10"] - system.metrics["composed_ndcg@10"],
        "native_fusion_gain_map@5": native.metrics["composed_map@5"] - system.metrics["composed_map@5"],
        "native_fusion_gain_recall@1": native.metrics["composed_recall@1"] - system.metrics["composed_recall@1"],
    }


class ComposedMediaRetrievalTask(EvalTask):
    """Fixture-only contract, ranking, slice, and paired-delta smoke."""

    name = "composed_media_retrieval"
    description = "No-publish composed-media retrieval contract fixture"
    required_modalities = {ModalityType.TEXT, ModalityType.IMAGE, ModalityType.VIDEO, ModalityType.AUDIO}

    def __init__(self, dataset_version: str = DATASET_VERSION, fixture_only: bool = True, **kwargs: Any) -> None:
        if dataset_version != DATASET_VERSION:
            raise ValueError(f"Unsupported composed-media fixture version: {dataset_version}")
        if not fixture_only:
            raise ValueError("Composed-media retrieval is fixture_only and not publishable")
        self.dataset_version = dataset_version
        self.fixture_only = fixture_only

    def run(self, provider: Any, **kwargs: Any) -> EvalResult:
        model_name = getattr(provider, "model", "unknown")
        try:
            if not isinstance(provider, ComposedMediaEmbeddingProvider):
                raise TypeError("Provider does not implement the explicit composed-media embedding capability")
            fixture = load_composed_media_retrieval_fixture()
            dimensions = int(getattr(provider, "composed_dimensions", 16))
            provider_name = str(getattr(provider, "name", "unknown"))
            model_revision = str(getattr(provider, "model_revision", "unknown-revision"))
            preprocessing = fixture.generator_version
            query_items = [query.item for query in fixture.queries]
            corpus_items = [row.item for row in fixture.corpus]
            evaluations: dict[str, RetrievalEvaluation] = {}
            route_details: dict[str, Any] = {}

            for composition_mode, track_label, fusion_strategy in (
                ("provider_native_fusion", "provider_valid_embedding", "deterministic_native_fixture_v0"),
                ("benchmark_system_fusion", "benchmark_system_fusion", "deterministic_part_fusion_fixture_v0"),
            ):
                query_request = build_composed_request(
                    items=query_items,
                    provider=provider_name,
                    model_id=model_name,
                    model_revision=model_revision,
                    dimensions=dimensions,
                    task_route="composed_media_retrieval_query",
                    preprocessing=preprocessing,
                    composition_mode=composition_mode,
                    track_label=track_label,
                    fusion_strategy=fusion_strategy,
                )
                corpus_request = build_composed_request(
                    items=corpus_items,
                    provider=provider_name,
                    model_id=model_name,
                    model_revision=model_revision,
                    dimensions=dimensions,
                    task_route="composed_media_retrieval_corpus",
                    preprocessing=preprocessing,
                    composition_mode=composition_mode,
                    track_label=track_label,
                    fusion_strategy=fusion_strategy,
                )
                query_result = provider.embed_composed_media(query_request)
                corpus_result = provider.embed_composed_media(corpus_request)
                validate_composed_result(query_request, query_result, FIXTURE_ROOT)
                validate_composed_result(corpus_request, corpus_result, FIXTURE_ROOT)
                evaluation = evaluate_composed_retrieval(
                    fixture=fixture,
                    query_result=query_result,
                    corpus_result=corpus_result,
                    preprocessing=preprocessing,
                )
                evaluations[composition_mode] = evaluation
                route_details[composition_mode] = {
                    "track_label": track_label,
                    "score_validity": query_result.score_validity,
                    "query_request_sha256": query_request.request_sha256,
                    "query_result_sha256": query_result.result_sha256,
                    "corpus_request_sha256": corpus_request.request_sha256,
                    "corpus_result_sha256": corpus_result.result_sha256,
                    "route_evidence": query_result.route_evidence,
                }

            native = evaluations["provider_native_fusion"]
            system = evaluations["benchmark_system_fusion"]
            metrics = dict(native.metrics)
            metrics.update({f"system_{name}": value for name, value in system.metrics.items()})
            metrics.update(paired_fusion_deltas(native, system))
            details = {
                "dataset_version": fixture.dataset_version,
                "fixture_sha256": fixture.fixture_sha256,
                "fixture_tree_sha256": fixture_tree_sha256(),
                "fixture_tracked_bytes": fixture_tracked_bytes(),
                "split": fixture.split,
                "fixture_only": fixture.fixture_only,
                "publish": fixture.publish,
                "leaderboard_publish": fixture.leaderboard_publish,
                "license_status": fixture.license_status,
                "network": fixture.network,
                "provider_api_calls": fixture.provider_api_calls,
                "model_downloads": 0,
                "dataset_downloads": 0,
                "hugging_face_operations": 0,
                "n_queries": len(fixture.queries),
                "n_corpus": len(fixture.corpus),
                "n_positive_qrels": len(fixture.qrels),
                "n_hard_negatives": len(fixture.hard_negatives),
                "query_shape_counts": dict(CounterLike(query.shape for query in fixture.queries)),
                "hard_negative_family_counts": dict(CounterLike(row.negative_family for row in fixture.hard_negatives)),
                "full_corpus_metrics_authoritative": True,
                "labels": {
                    "provider_valid": "provider_valid_embedding",
                    "system_fusion": "benchmark_system_fusion",
                    "reranker": "reranker_system_only",
                    "fixture": "contract_fixture_only",
                },
                "routes": route_details,
                "slices": {
                    "provider_native_fusion": native.slices,
                    "benchmark_system_fusion": system.slices,
                },
            }
            return EvalResult(
                task_name=self.name,
                provider_name=provider_name,
                model_name=model_name,
                metrics=metrics,
                details=details,
            )
        except Exception as exc:
            return EvalResult(
                task_name=self.name,
                provider_name=str(getattr(provider, "name", "unknown")),
                model_name=model_name,
                metrics={},
                error=str(exc),
            )
