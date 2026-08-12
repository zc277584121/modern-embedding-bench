"""Exact CSR evaluation for the deterministic learned-sparse fixture."""

from __future__ import annotations

import math
import resource
import time
from typing import Any, ClassVar

from mm_embed.data.learned_sparse_retrieval_fixture import (
    DATASET_VERSION,
    load_learned_sparse_retrieval_fixture,
)
from mm_embed.data.opensearch_neural_sparse_smoke import (
    DATASET_VERSION as SMOKE_DATASET_VERSION,
)
from mm_embed.data.opensearch_neural_sparse_smoke import (
    load_opensearch_neural_sparse_smoke,
)
from mm_embed.indexes.sparse_exact import ExactSparseIndex
from mm_embed.providers.sparse_base import SparseEmbeddingProvider
from mm_embed.tasks.base import EvalResult


class LearnedSparseRetrievalTask:
    name = "learned_sparse_retrieval"
    description = "Deterministic learned-sparse retrieval contract fixture"
    required_modalities: ClassVar[set[str]] = {"text"}

    def __init__(self, *, dataset_version: str = DATASET_VERSION, fixture_only: bool = True, top_k: int = 10) -> None:
        if dataset_version not in {DATASET_VERSION, SMOKE_DATASET_VERSION} or not fixture_only:
            raise ValueError("Learned-sparse task only supports fixed no-publish fixture contracts")
        if top_k <= 0:
            raise ValueError("Learned-sparse top_k must be positive")
        self.dataset_version = dataset_version
        self.top_k = top_k

    def run(self, provider: SparseEmbeddingProvider, **kwargs: Any) -> EvalResult:
        if not isinstance(provider, SparseEmbeddingProvider):
            raise TypeError("Learned-sparse retrieval requires a SparseEmbeddingProvider")
        if self.dataset_version == SMOKE_DATASET_VERSION:
            return self._run_smoke(provider)
        fixture = load_learned_sparse_retrieval_fixture()
        documents = provider.encode_sparse_documents(
            [document.text for document in fixture.documents],
            item_ids=[document.document_id for document in fixture.documents],
        )

        if any(nnz == 0 for nnz in documents.embeddings.nnz_per_row):
            raise ValueError("Learned-sparse document rows must not be empty")
        index = ExactSparseIndex(documents)

        reciprocal_ranks: list[float] = []
        recall_at_1: list[float] = []
        ndcg_at_10: list[float] = []
        query_nnz: list[int] = []
        rankings: list[dict[str, Any]] = []
        slice_metrics: dict[str, float] = {}
        query_latency_ms = 0.0
        search_latency_ms = 0.0
        peak_vram_values = [documents.peak_vram_bytes]

        for query in fixture.queries:
            query_result = provider.encode_sparse_query(query.text, item_id=query.query_id)
            if query_result.embeddings.nnz_per_row != (len(query.activations),):
                raise ValueError("Learned-sparse query row does not match the fixed activation contract")
            ranking = index.search(query_result, k=min(self.top_k, len(fixture.documents)))
            hits = ranking.queries[0].hits
            relevant = set(query.relevant_document_ids)
            first_relevant_rank = next((hit.rank for hit in hits if hit.item_id in relevant), None)
            rr = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
            recall = float(bool(hits and hits[0].item_id in relevant))
            ndcg = 0.0 if first_relevant_rank is None or first_relevant_rank > 10 else 1.0 / math.log2(first_relevant_rank + 1)
            reciprocal_ranks.append(rr)
            recall_at_1.append(recall)
            ndcg_at_10.append(ndcg)
            query_nnz.extend(query_result.embeddings.nnz_per_row)
            query_latency_ms += query_result.latency_ms
            peak_vram_values.append(query_result.peak_vram_bytes)
            slice_metrics[query.diagnostic_slice] = recall
            rankings.append(
                {
                    "query_id": query.query_id,
                    "diagnostic_slice": query.diagnostic_slice,
                    "relevant_document_ids": list(query.relevant_document_ids),
                    "hard_negative_ids": list(query.hard_negative_ids),
                    "hits": [
                        {"rank": hit.rank, "item_id": hit.item_id, "score": hit.score}
                        for hit in hits
                    ],
                }
            )

        matrix = documents.embeddings.values
        index_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        metrics = {
            "recall@1": sum(recall_at_1) / len(recall_at_1),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "ndcg@10": sum(ndcg_at_10) / len(ndcg_at_10),
            **{f"slice_recall@1/{name}": value for name, value in sorted(slice_metrics.items())},
        }
        peak_vram_bytes = max((value for value in peak_vram_values if value is not None), default=None)
        return EvalResult(
            task_name=self.name,
            provider_name=documents.provider,
            model_name=documents.model_name,
            metrics=metrics,
            details={
                "fixture": {
                    "dataset_version": fixture.dataset_version,
                    "label_sha256": fixture.label_sha256,
                    "fixture_sha256": fixture.fixture_sha256,
                    "fixture_only": fixture.fixture_only,
                    "publish": fixture.publish,
                    "leaderboard_publish": fixture.leaderboard_publish,
                    "evidence_tier": fixture.evidence_tier,
                    "score_validity": fixture.score_validity,
                },
                "rankings": rankings,
            },
            execution={
                "retrieval_kind": "sparse_exact",
                "representation_kind": "sparse_csr",
                "provider": documents.provider,
                "model_name": documents.model_name,
                "model_revision": documents.model_revision,
                "representation_id": documents.embeddings.representation.representation_id,
                "representation_identity": documents.embeddings.representation.identity,
                "vocabulary_id": documents.embeddings.representation.vocabulary_id,
                "dimensions": documents.embeddings.dimensions,
                "query_route": documents.query_route.value,
                "document_route": documents.document_route.value,
                "query_nnz": _stats(query_nnz),
                "document_nnz": _stats(list(documents.embeddings.nnz_per_row)),
                "exact_backend": index.backend,
                "exact": True,
                "index_bytes_estimate": index_bytes,
                "encoding_latency_ms": documents.latency_ms + query_latency_ms,
                "search_latency_ms": search_latency_ms,
                "peak_memory_bytes": None,
                "peak_vram_bytes": peak_vram_bytes,
            },
        )

    def _run_smoke(self, provider: SparseEmbeddingProvider) -> EvalResult:
        fixture = load_opensearch_neural_sparse_smoke()
        documents = provider.encode_sparse_documents(
            [document.text for document in fixture.documents],
            item_ids=[document.document_id for document in fixture.documents],
        )
        if documents.embeddings.values.shape != (4, 30522):
            raise ValueError("Neural sparse smoke documents must have shape 4x30522")
        index = ExactSparseIndex(documents)
        query_spec = fixture.queries[0]
        query = provider.encode_sparse_query(query_spec.text, item_id=query_spec.query_id)
        if query.embeddings.values.shape != (1, 30522):
            raise ValueError("Neural sparse smoke query must have shape 1x30522")
        started = time.perf_counter()
        ranking = index.search(query, k=4)
        search_latency_ms = (time.perf_counter() - started) * 1000.0
        hits = ranking.queries[0].hits
        relevant_rank = next((hit.rank for hit in hits if hit.item_id in query_spec.relevant_document_ids), None)
        matrix = documents.embeddings.values
        index_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return EvalResult(
            task_name=self.name,
            provider_name=documents.provider,
            model_name=documents.model_name,
            metrics={"recall@1": float(relevant_rank == 1), "mrr": 0.0 if relevant_rank is None else 1.0 / relevant_rank},
            details={
                "fixture": {"dataset_version": fixture.dataset_version, "fixture_sha256": fixture.fixture_sha256,
                            "fixture_only": True, "publish": False, "leaderboard_publish": False,
                            "evidence_tier": "smoke", "score_validity": "smoke_only"},
                "rankings": [{"query_id": query_spec.query_id, "relevant_document_ids": list(query_spec.relevant_document_ids),
                              "relevant_rank": relevant_rank,
                              "hits": [{"rank": hit.rank, "item_id": hit.item_id, "score": hit.score} for hit in hits]}],
            },
            execution={
                "retrieval_kind": "sparse_exact", "representation_kind": "sparse_csr", "provider": documents.provider,
                "model_name": documents.model_name, "model_revision": documents.model_revision,
                "representation_id": documents.embeddings.representation.representation_id,
                "representation_identity": documents.embeddings.representation.identity,
                "vocabulary_id": documents.embeddings.representation.vocabulary_id,
                "dimensions": documents.embeddings.dimensions, "query_route": query.query_route.value,
                "document_route": documents.document_route.value, "query_nnz": _stats(list(query.embeddings.nnz_per_row)),
                "document_nnz": _stats(list(documents.embeddings.nnz_per_row)), "exact_backend": index.backend,
                "exact": True, "index_bytes_estimate": index_bytes,
                "encoding_latency_ms": documents.latency_ms + query.latency_ms,
                "search_latency_ms": search_latency_ms, "peak_memory_bytes": rss_bytes,
                "peak_vram_bytes": documents.peak_vram_bytes,
            },
        )


def _stats(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total": sum(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


__all__ = ["LearnedSparseRetrievalTask"]
