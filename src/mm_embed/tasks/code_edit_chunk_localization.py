"""Flat embedding evaluation for the invented code-edit chunk fixture."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from mm_embed.data.code_edit_chunk_localization import (
    DATASET_VERSION,
    CodeChunk,
    CodeEditChunkLocalizationFixture,
    EditQrel,
    fixture_counts,
    fixture_slice_counts,
    load_code_edit_chunk_localization_fixture,
    serialize_chunk_document,
    target_unit_ids,
)
from mm_embed.providers.base import EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import cosine_similarity_matrix

logger = logging.getLogger(__name__)


METRIC_NAMES = (
    "edit_chunk_ndcg@10",
    "edit_chunk_recall@1",
    "edit_chunk_recall@5",
    "edit_chunk_recall@10",
    "edit_chunk_recall@100",
    "edit_chunk_mrr",
    "edit_target_recall@100_lines",
    "edit_target_recall@300_lines",
    "edit_target_recall@500_lines",
    "first_edit_hit_rank",
    "file_recall@1",
    "file_recall@5",
    "file_recall@10",
    "candidate_coverage",
    "hard_mrr",
    "hard_ndcg@10",
)


def _validate_score_contract(
    score_matrix: np.ndarray,
    query_ids: Sequence[str],
    chunk_ids: Sequence[str],
    fixture: CodeEditChunkLocalizationFixture,
) -> None:
    expected_shape = (len(query_ids), len(chunk_ids))
    if score_matrix.shape != expected_shape:
        raise ValueError(f"Expected score matrix shape {expected_shape}, received {score_matrix.shape}")
    if not np.isfinite(score_matrix).all():
        raise ValueError("Score matrix must contain only finite values")
    if len(query_ids) != len(set(query_ids)) or len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Query and chunk ids must be unique")

    expected_queries = {query.query_id for query in fixture.queries}
    if set(query_ids) != expected_queries or len(query_ids) != len(fixture.queries):
        raise ValueError("Evaluation must include every fixture query exactly once")
    expected_chunks = {chunk.chunk_id for chunk in fixture.chunks}
    if set(chunk_ids) != expected_chunks or len(chunk_ids) != len(fixture.chunks):
        raise ValueError("Evaluation must rank the complete full-repository fixture corpus")


def deterministic_rankings(
    score_matrix: np.ndarray,
    chunk_ids: Sequence[str],
    fixture: CodeEditChunkLocalizationFixture | None = None,
) -> list[list[str]]:
    """Rank by score, repository, path, line start, and chunk id."""
    fixture = fixture or load_code_edit_chunk_localization_fixture()
    if score_matrix.ndim != 2 or score_matrix.shape[1] != len(chunk_ids):
        raise ValueError("Score matrix and chunk ids do not align")
    chunk_by_id = {chunk.chunk_id: chunk for chunk in fixture.chunks}
    expected_ids = set(chunk_by_id)
    if set(chunk_ids) != expected_ids or len(chunk_ids) != len(expected_ids):
        raise ValueError("Deterministic ranking requires the complete full-repository fixture corpus")
    return [
        [
            chunk_ids[index]
            for index in sorted(
                range(len(chunk_ids)),
                key=lambda index: (
                    -float(row[index]),
                    chunk_by_id[chunk_ids[index]].repository_id,
                    chunk_by_id[chunk_ids[index]].path,
                    chunk_by_id[chunk_ids[index]].line_start,
                    chunk_by_id[chunk_ids[index]].chunk_id,
                ),
            )
        ]
        for row in score_matrix
    ]


def _ndcg(ranking: Sequence[str], relevance: dict[str, int], k: int) -> float:
    dcg = sum(
        (2**relevance.get(chunk_id, 0) - 1) / np.log2(rank + 1)
        for rank, chunk_id in enumerate(ranking[:k], start=1)
        if relevance.get(chunk_id, 0) > 0
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / np.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return float(dcg / idcg) if idcg else 0.0


def _reciprocal_rank(ranking: Sequence[str], positives: set[str]) -> float:
    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in positives:
            return 1.0 / rank
    return 0.0


def _collapsed_paths(ranking: Sequence[str], chunk_by_id: dict[str, CodeChunk]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for chunk_id in ranking:
        path = chunk_by_id[chunk_id].path
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _line_budget_coverage(
    ranking: Sequence[str],
    chunk_by_id: dict[str, CodeChunk],
    units_by_chunk: dict[str, set[str]],
    all_units: set[str],
    budget: int,
) -> float:
    spent = 0
    covered: set[str] = set()
    for chunk_id in ranking:
        chunk = chunk_by_id[chunk_id]
        chunk_lines = chunk.line_end - chunk.line_start + 1
        if spent + chunk_lines > budget:
            break
        spent += chunk_lines
        covered.update(units_by_chunk.get(chunk_id, set()))
    return len(covered) / len(all_units) if all_units else 0.0


def _evaluate_queries(
    score_matrix: np.ndarray,
    query_ids: Sequence[str],
    chunk_ids: Sequence[str],
    fixture: CodeEditChunkLocalizationFixture,
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    _validate_score_contract(score_matrix, query_ids, chunk_ids, fixture)
    rankings = deterministic_rankings(score_matrix, chunk_ids, fixture)
    rankings_by_query = dict(zip(query_ids, rankings, strict=True))
    chunk_by_id = {chunk.chunk_id: chunk for chunk in fixture.chunks}

    qrels_by_query: dict[str, list[EditQrel]] = defaultdict(list)
    for qrel in fixture.qrels:
        qrels_by_query[qrel.query_id].append(qrel)
    negatives_by_query: dict[str, list[str]] = defaultdict(list)
    for negative in fixture.hard_negatives:
        negatives_by_query[negative.query_id].append(negative.chunk_id)
    targets_by_query = defaultdict(list)
    for target in fixture.patch_targets:
        targets_by_query[target.query_id].append(target)

    per_query: dict[str, dict[str, float]] = {}
    for query_id in query_ids:
        ranking = rankings_by_query[query_id]
        query_qrels = qrels_by_query[query_id]
        relevance: dict[str, int] = {}
        units_by_chunk: dict[str, set[str]] = defaultdict(set)
        for qrel in query_qrels:
            relevance[qrel.chunk_id] = max(relevance.get(qrel.chunk_id, 0), qrel.relevance)
            units_by_chunk[qrel.chunk_id].update(qrel.target_unit_ids)
        positives = set(relevance)
        all_units = {
            unit_id
            for target in targets_by_query[query_id]
            for unit_id in target_unit_ids(target)
        }
        mapped_units = {unit_id for qrel in query_qrels for unit_id in qrel.target_unit_ids}

        positive_paths = {chunk_by_id[chunk_id].path for chunk_id in positives}
        collapsed_paths = _collapsed_paths(ranking, chunk_by_id)
        first_hit = next(rank for rank, chunk_id in enumerate(ranking, start=1) if chunk_id in positives)

        hard_pool = positives | set(negatives_by_query[query_id])
        hard_ranking = [chunk_id for chunk_id in ranking if chunk_id in hard_pool]
        values = {
            "edit_chunk_ndcg@10": _ndcg(ranking, relevance, 10),
            "edit_chunk_recall@1": len(positives.intersection(ranking[:1])) / len(positives),
            "edit_chunk_recall@5": len(positives.intersection(ranking[:5])) / len(positives),
            "edit_chunk_recall@10": len(positives.intersection(ranking[:10])) / len(positives),
            "edit_chunk_recall@100": len(positives.intersection(ranking[:100])) / len(positives),
            "edit_chunk_mrr": _reciprocal_rank(ranking, positives),
            "edit_target_recall@100_lines": _line_budget_coverage(
                ranking, chunk_by_id, units_by_chunk, all_units, 100
            ),
            "edit_target_recall@300_lines": _line_budget_coverage(
                ranking, chunk_by_id, units_by_chunk, all_units, 300
            ),
            "edit_target_recall@500_lines": _line_budget_coverage(
                ranking, chunk_by_id, units_by_chunk, all_units, 500
            ),
            "first_edit_hit_rank": float(first_hit),
            "file_recall@1": len(positive_paths.intersection(collapsed_paths[:1])) / len(positive_paths),
            "file_recall@5": len(positive_paths.intersection(collapsed_paths[:5])) / len(positive_paths),
            "file_recall@10": len(positive_paths.intersection(collapsed_paths[:10])) / len(positive_paths),
            "candidate_coverage": len(mapped_units) / len(all_units),
            "hard_mrr": _reciprocal_rank(hard_ranking, positives),
            "hard_ndcg@10": _ndcg(hard_ranking, relevance, 10),
        }
        per_query[query_id] = {name: float(values[name]) for name in METRIC_NAMES}
    return per_query, rankings_by_query


def _mean_metrics(per_query: dict[str, dict[str, float]], query_ids: Iterable[str] | None = None) -> dict[str, float]:
    selected = list(query_ids) if query_ids is not None else list(per_query)
    if not selected:
        return {name: 0.0 for name in METRIC_NAMES}
    return {
        name: float(np.mean([per_query[query_id][name] for query_id in selected]))
        for name in METRIC_NAMES
    }


def evaluate_code_edit_scores(
    score_matrix: np.ndarray,
    query_ids: Sequence[str],
    chunk_ids: Sequence[str],
    fixture: CodeEditChunkLocalizationFixture | None = None,
) -> dict[str, float]:
    """Evaluate deterministic full-corpus and audited hard-pool metrics."""
    fixture = fixture or load_code_edit_chunk_localization_fixture()
    per_query, _ = _evaluate_queries(score_matrix, query_ids, chunk_ids, fixture)
    return _mean_metrics(per_query, query_ids)


def _add_slice_membership(
    groups: dict[str, dict[str, set[str]]],
    slice_name: str,
    value: str,
    query_id: str,
) -> None:
    groups.setdefault(slice_name, {}).setdefault(value, set()).add(query_id)


def _diagnostics(
    fixture: CodeEditChunkLocalizationFixture,
    per_query: dict[str, dict[str, float]],
    rankings_by_query: dict[str, list[str]],
) -> dict[str, Any]:
    query_by_id = {query.query_id: query for query in fixture.queries}
    chunk_by_id = {chunk.chunk_id: chunk for chunk in fixture.chunks}
    groups: dict[str, dict[str, set[str]]] = {}
    for query in fixture.queries:
        _add_slice_membership(groups, "edit_type", query.edit_type, query.query_id)
    for qrel in fixture.qrels:
        chunk = chunk_by_id[qrel.chunk_id]
        _add_slice_membership(groups, "file_path", qrel.changed_path, qrel.query_id)
        _add_slice_membership(groups, "path_family", chunk.path_family, qrel.query_id)
        _add_slice_membership(groups, "candidate_family", chunk.candidate_family, qrel.query_id)
        _add_slice_membership(groups, "mapping_status", qrel.mapping_status, qrel.query_id)

    slice_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    reported = (
        "edit_chunk_ndcg@10",
        "edit_chunk_mrr",
        "edit_target_recall@100_lines",
        "file_recall@5",
        "candidate_coverage",
        "hard_mrr",
    )
    for slice_name, values in groups.items():
        slice_metrics[slice_name] = {}
        for value, query_ids in sorted(values.items()):
            metrics = _mean_metrics(per_query, sorted(query_ids))
            slice_metrics[slice_name][value] = {
                "n_queries": len(query_ids),
                **{name: metrics[name] for name in reported},
            }

    hard_family_ranks: dict[str, list[float]] = defaultdict(list)
    hard_family_pool_ranks: dict[str, list[float]] = defaultdict(list)
    qrel_ids_by_query: dict[str, set[str]] = defaultdict(set)
    for qrel in fixture.qrels:
        qrel_ids_by_query[qrel.query_id].add(qrel.chunk_id)
    negatives_by_query: dict[str, list[str]] = defaultdict(list)
    for negative in fixture.hard_negatives:
        negatives_by_query[negative.query_id].append(negative.chunk_id)
    for negative in fixture.hard_negatives:
        ranking = rankings_by_query[negative.query_id]
        hard_pool = qrel_ids_by_query[negative.query_id] | set(negatives_by_query[negative.query_id])
        hard_ranking = [chunk_id for chunk_id in ranking if chunk_id in hard_pool]
        hard_family_ranks[negative.negative_family].append(float(ranking.index(negative.chunk_id) + 1))
        hard_family_pool_ranks[negative.negative_family].append(float(hard_ranking.index(negative.chunk_id) + 1))
    hard_negative_slices = {
        family: {
            "n_links": len(ranks),
            "mean_full_corpus_rank": float(np.mean(ranks)),
            "mean_hard_pool_rank": float(np.mean(hard_family_pool_ranks[family])),
        }
        for family, ranks in sorted(hard_family_ranks.items())
    }

    return {
        "per_query": {
            query_id: {
                "edit_type": query_by_id[query_id].edit_type,
                **per_query[query_id],
            }
            for query_id in sorted(per_query)
        },
        "slice_metrics": slice_metrics,
        "hard_negative_slices": hard_negative_slices,
    }


class CodeEditChunkLocalizationTask(EvalTask):
    """Issue-to-edit chunk localization over the complete invented repository."""

    name = "code_edit_chunk_localization"
    description = "Invented full-repository issue-to-edit chunk localization fixture"
    required_modalities = {ModalityType.TEXT}

    def __init__(
        self,
        dataset_version: str = DATASET_VERSION,
        fixture_only: bool = True,
        **kwargs: Any,
    ) -> None:
        if dataset_version != DATASET_VERSION:
            raise ValueError(f"Unsupported code-edit chunk dataset version: {dataset_version}")
        if not fixture_only:
            raise ValueError("Code-edit chunk localization is fixture-only and not publishable")
        self.dataset_version = dataset_version
        self.fixture_only = fixture_only

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        model_name = getattr(provider, "model", "unknown")
        try:
            fixture = load_code_edit_chunk_localization_fixture()
            query_texts = [query.text for query in fixture.queries]
            chunk_texts = [serialize_chunk_document(chunk) for chunk in fixture.chunks]
            logger.info(
                "%s: %d queries, %d full-corpus chunks across %d files",
                self.name,
                len(query_texts),
                len(chunk_texts),
                len(fixture.files),
            )

            query_result = provider.embed_text(query_texts, task_type="retrieval_query")
            document_result = provider.embed_text(chunk_texts, task_type="retrieval_document")
            if query_result.embeddings.shape[0] != len(fixture.queries):
                raise ValueError("Provider returned the wrong number of query embeddings")
            if document_result.embeddings.shape[0] != len(fixture.chunks):
                raise ValueError("Provider returned the wrong number of full-corpus document embeddings")

            scores = cosine_similarity_matrix(query_result.embeddings, document_result.embeddings)
            query_ids = [query.query_id for query in fixture.queries]
            chunk_ids = [chunk.chunk_id for chunk in fixture.chunks]
            per_query, rankings = _evaluate_queries(scores, query_ids, chunk_ids, fixture)
            metrics = _mean_metrics(per_query, query_ids)
            diagnostics = _diagnostics(fixture, per_query, rankings)
            details = {
                **fixture_counts(fixture),
                "dataset_version": fixture.dataset_version,
                "split": fixture.split,
                "fixture_only": fixture.fixture_only,
                "license_status": fixture.license_status,
                "leaderboard_publish": fixture.leaderboard_publish,
                "public_score_eligible": fixture.repository.public_score_eligible,
                "network": fixture.network,
                "fixture_provider_api_calls": fixture.provider_api_calls,
                "fixture_model_downloads": fixture.model_downloads,
                "repository_id": fixture.repository.repository_id,
                "base_commit": fixture.repository.base_commit,
                "tree_sha": fixture.repository.tree_sha,
                "repository_snapshot_sha256": fixture.repository.snapshot_sha256,
                "serialization_sha256": fixture.serialization_sha256,
                "candidate_pool_scope": "complete_invented_repository",
                "full_corpus_candidate_count": len(fixture.chunks),
                "ranked_candidate_count_per_query": len(fixture.chunks),
                "changed_path_prefilter": False,
                "gold_path_prefilter": False,
                "ranking_tie_break": "score_desc_repository_path_line_start_chunk_id",
                "unlabeled_candidate_status": "unknown_not_asserted_irrelevant",
                "query_latency_ms": query_result.latency_ms,
                "document_latency_ms": document_result.latency_ms,
                "slices": fixture_slice_counts(fixture),
                **diagnostics,
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
                provider_name=provider.name,
                model_name=model_name,
                metrics={},
                error=str(exc),
            )
