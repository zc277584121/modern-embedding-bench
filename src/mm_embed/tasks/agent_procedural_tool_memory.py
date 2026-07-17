"""Agent procedural tool-memory retrieval task."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mm_embed.data.agent_procedural_tool_memory import (
    fixture_slice_counts,
    load_agent_procedural_tool_memory_fixture,
    serialize_tool_document,
)
from mm_embed.providers.base import EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import cosine_similarity_matrix, mrr, ndcg_at_k, recall_at_k

logger = logging.getLogger(__name__)


class AgentProceduralToolMemoryTask(EvalTask):
    """Text-only query-to-tool-card retrieval for procedural agent memory."""

    name = "agent_procedural_tool_memory"
    description = "Query-to-tool-document retrieval for procedural memory in agent systems"
    required_modalities = {ModalityType.TEXT}

    def __init__(
        self,
        use_mock: bool = False,
        max_queries: int | None = None,
        hard_mode: bool = True,
        **kwargs: Any,
    ) -> None:
        self.use_mock = use_mock
        self.max_queries = max_queries
        self.hard_mode = hard_mode

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        model_name = getattr(provider, "model", "unknown")

        try:
            fixture = load_agent_procedural_tool_memory_fixture(max_queries=self.max_queries)
            queries = list(fixture.queries)
            documents = list(fixture.documents)
            doc_id_to_index = {document.doc_id: i for i, document in enumerate(documents)}

            query_texts = [query.text for query in queries]
            document_texts = [serialize_tool_document(document) for document in documents]

            logger.info(
                "Agent procedural tool memory: %d queries, %d documents",
                len(query_texts),
                len(document_texts),
            )
            query_result = provider.embed_text(query_texts, task_type="retrieval_query")
            document_result = provider.embed_text(document_texts, task_type="retrieval_document")

            sim_matrix = cosine_similarity_matrix(query_result.embeddings, document_result.embeddings)
            ground_truth = np.array(
                [doc_id_to_index[query.positive_doc_id] for query in queries],
                dtype=int,
            )

            metrics: dict[str, float] = {
                "recall@1": recall_at_k(sim_matrix, ground_truth, k=1),
                "recall@5": recall_at_k(sim_matrix, ground_truth, k=5),
                "mrr": mrr(sim_matrix, ground_truth),
                "ndcg@10": ndcg_at_k(sim_matrix, ground_truth, k=10),
            }

            hard_sim_matrix, hard_ground_truth = self._build_hard_pool_matrix(
                sim_matrix=sim_matrix,
                queries=queries,
                doc_id_to_index=doc_id_to_index,
            )
            metrics.update(
                {
                    "hard_recall@1": recall_at_k(hard_sim_matrix, hard_ground_truth, k=1),
                    "hard_recall@5": recall_at_k(hard_sim_matrix, hard_ground_truth, k=5),
                    "hard_mrr": mrr(hard_sim_matrix, hard_ground_truth),
                    "hard_ndcg@10": ndcg_at_k(hard_sim_matrix, hard_ground_truth, k=10),
                }
            )
            metrics.update(self._slice_metrics(hard_sim_matrix, hard_ground_truth, queries))

            details = {
                "n_queries": len(queries),
                "n_documents": len(documents),
                "n_qrels": len(queries),
                "n_hard_negatives": sum(len(query.hard_negative_doc_ids) for query in queries),
                "source_datasets": list(fixture.source_datasets),
                "license_audit_status": fixture.license_audit_status,
                "slices": fixture_slice_counts(tuple(queries)),
                "hard_pool_size": int(hard_sim_matrix.shape[1]),
                "query_latency_ms": query_result.latency_ms,
                "document_latency_ms": document_result.latency_ms,
            }

            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=model_name,
                metrics=metrics,
                details=details,
            )

        except Exception as e:
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=model_name,
                metrics={},
                error=str(e),
            )

    @staticmethod
    def _build_hard_pool_matrix(
        sim_matrix: np.ndarray,
        queries: list[Any],
        doc_id_to_index: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        hard_rows: list[np.ndarray] = []

        for query_index, query in enumerate(queries):
            pool_doc_ids = (query.positive_doc_id, *query.hard_negative_doc_ids)
            pool_indices = [doc_id_to_index[doc_id] for doc_id in pool_doc_ids]
            hard_rows.append(sim_matrix[query_index, pool_indices])

        return np.vstack(hard_rows), np.zeros(len(queries), dtype=int)

    @staticmethod
    def _slice_metrics(
        hard_sim_matrix: np.ndarray,
        hard_ground_truth: np.ndarray,
        queries: list[Any],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        slices = sorted({query.slice for query in queries})
        for slice_name in slices:
            indices = np.array(
                [i for i, query in enumerate(queries) if query.slice == slice_name],
                dtype=int,
            )
            if indices.size == 0:
                continue
            metrics[f"hard_mrr_{slice_name}"] = mrr(
                hard_sim_matrix[indices],
                hard_ground_truth[indices],
            )
        return metrics
