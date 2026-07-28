"""Task H: Needle-in-a-Haystack for Embeddings.

Tests whether embedding models can accurately retrieve specific facts
embedded within long documents of varying lengths.

Key insight: Unlike LLM needle-in-a-haystack (which tests generation),
this tests whether the embedding of a long document preserves enough
semantic information about specific details to be retrieved by a query.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from mm_embed.benchmark.materialization import (
    MaterializationAuthorization,
    normalize_data_mode,
    require_task_materialization_binding,
)
from mm_embed.data.mock import get_needle_haystack_data
from mm_embed.data.real_data import load_needle_haystack_real_data
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import cosine_similarity

logger = logging.getLogger(__name__)


class NeedleInHaystackTask(EvalTask):
    """Needle-in-a-Haystack for Embeddings.

    Test procedure:
    1. Create documents of varying lengths with needles inserted at various positions
    2. Batch-embed all unique queries, documents-with-needle, and documents-without-needle
    3. Compare similarity scores to determine retrieval accuracy
    4. Measure accuracy across length x position matrix
    """

    name = "needle_in_haystack"
    description = "Embedding retrieval of specific facts in long documents"
    required_modalities = {ModalityType.TEXT}

    def __init__(
        self,
        haystack_lengths: list[int] | None = None,
        needle_positions: list[float] | None = None,
        data_mode: str | None = None,
        use_mock: bool | None = None,
        materialization_binding: MaterializationAuthorization | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ):
        self.haystack_lengths = haystack_lengths or [4000, 8000, 16000, 32000]
        self.needle_positions = needle_positions or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.data_mode = normalize_data_mode(data_mode, use_mock)
        self.materialization_binding = materialization_binding
        self.use_cache = use_cache

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        model_name = getattr(provider, "model", "unknown")

        try:
            materialization = require_task_materialization_binding(
                self.name,
                self.data_mode,
                self.materialization_binding,
            )
            if self.data_mode == "fixture":
                test_cases = get_needle_haystack_data(
                    haystack_lengths=self.haystack_lengths,
                    needle_positions=self.needle_positions,
                )
            else:
                if materialization is None:
                    raise ValueError("Real needle data requires a materialization authorization")
                test_cases = load_needle_haystack_real_data(
                    materialization,
                    haystack_lengths=self.haystack_lengths,
                    needle_positions=self.needle_positions,
                )

            # Filter test cases to documents within provider's max context
            max_chars = provider.max_text_length * 4  # rough tokens-to-chars ratio
            valid_cases = [tc for tc in test_cases if len(tc["document"]) <= max_chars]

            if not valid_cases:
                return EvalResult(
                    task_name=self.name,
                    provider_name=provider.name,
                    model_name=model_name,
                    metrics={},
                    error=f"No test cases fit within provider's max context ({provider.max_text_length} tokens)",
                )

            logger.info("Needle-in-haystack: %d valid cases (%d skipped due to context limit)",
                        len(valid_cases), len(test_cases) - len(valid_cases))

            # Collect all unique texts to embed in batches
            unique_queries: dict[str, int] = {}
            unique_docs_with: dict[str, int] = {}
            unique_docs_without: dict[str, int] = {}

            query_list: list[str] = []
            doc_with_list: list[str] = []
            doc_without_list: list[str] = []

            for tc in valid_cases:
                query = tc["query"]
                doc_with = tc["document"]
                doc_without = doc_with.replace(tc["needle"], "", 1)

                if query not in unique_queries:
                    unique_queries[query] = len(query_list)
                    query_list.append(query)
                if doc_with not in unique_docs_with:
                    unique_docs_with[doc_with] = len(doc_with_list)
                    doc_with_list.append(doc_with)
                if doc_without not in unique_docs_without:
                    unique_docs_without[doc_without] = len(doc_without_list)
                    doc_without_list.append(doc_without)

            logger.info("Unique texts: %d queries, %d docs_with, %d docs_without",
                        len(query_list), len(doc_with_list), len(doc_without_list))

            # Batch embed
            logger.info("Embedding queries...")
            query_result = self._embed_texts(provider, query_list, task_type="retrieval_query")
            query_evidence = self._call_evidence("queries", query_list, query_result, "retrieval_query")

            logger.info("Embedding documents (with needle)...")
            doc_with_result = self._embed_texts(provider, doc_with_list, task_type="retrieval_document")
            doc_with_evidence = self._call_evidence(
                "documents_with_needle",
                doc_with_list,
                doc_with_result,
                "retrieval_document",
            )

            logger.info("Embedding documents (without needle)...")
            doc_without_result = self._embed_texts(provider, doc_without_list, task_type="retrieval_document")
            doc_without_evidence = self._call_evidence(
                "documents_without_needle",
                doc_without_list,
                doc_without_result,
                "retrieval_document",
            )

            call_evidence = [
                query_evidence,
                doc_with_evidence,
                doc_without_evidence,
            ]

            # Compute results
            results_by_length: dict[int, list[bool]] = defaultdict(list)
            results_by_position: dict[float, list[bool]] = defaultdict(list)
            results_by_length_position: dict[tuple[int, float], list[bool]] = defaultdict(list)

            for tc in valid_cases:
                query = tc["query"]
                doc_with = tc["document"]
                doc_without = doc_with.replace(tc["needle"], "", 1)
                length = tc["length"]
                position = tc["position"]

                q_idx = unique_queries[query]
                dw_idx = unique_docs_with[doc_with]
                dwo_idx = unique_docs_without[doc_without]

                sim_with = cosine_similarity(
                    query_result.embeddings[q_idx], doc_with_result.embeddings[dw_idx]
                )
                sim_without = cosine_similarity(
                    query_result.embeddings[q_idx], doc_without_result.embeddings[dwo_idx]
                )

                hit = sim_with > sim_without
                results_by_length[length].append(hit)
                results_by_position[position].append(hit)
                results_by_length_position[(length, position)].append(hit)

            # Compute metrics
            metrics: dict[str, float] = {}

            all_hits = [h for hits in results_by_length.values() for h in hits]
            metrics["overall_accuracy"] = sum(all_hits) / len(all_hits) if all_hits else 0.0

            for length in sorted(results_by_length.keys()):
                hits = results_by_length[length]
                metrics[f"accuracy_len_{length}"] = sum(hits) / len(hits) if hits else 0.0

            for pos in sorted(results_by_position.keys()):
                hits = results_by_position[pos]
                pos_label = f"{int(pos * 100)}pct"
                metrics[f"accuracy_pos_{pos_label}"] = sum(hits) / len(hits) if hits else 0.0

            lengths_sorted = sorted(results_by_length.keys())
            if len(lengths_sorted) >= 2:
                first_acc = sum(results_by_length[lengths_sorted[0]]) / len(results_by_length[lengths_sorted[0]])
                last_acc = sum(results_by_length[lengths_sorted[-1]]) / len(results_by_length[lengths_sorted[-1]])
                metrics["degradation_rate"] = first_acc - last_acc

            heatmap: dict[str, dict[str, float]] = {}
            for (length, pos), hits in results_by_length_position.items():
                len_key = str(length)
                pos_key = f"{int(pos * 100)}%"
                if len_key not in heatmap:
                    heatmap[len_key] = {}
                heatmap[len_key][pos_key] = sum(hits) / len(hits) if hits else 0.0

            details = {
                "data_mode": self.data_mode,
                "n_test_cases": len(valid_cases),
                "n_skipped": len(test_cases) - len(valid_cases),
                "max_provider_context": provider.max_text_length,
                "heatmap": heatmap,
                "cache_enabled": self.use_cache,
                "fresh_provider_calls": not self.use_cache,
                "input_cardinality": {
                    "queries": len(query_list),
                    "documents_with_needle": len(doc_with_list),
                    "documents_without_needle": len(doc_without_list),
                    "total": sum(item["input_count"] for item in call_evidence),
                },
                "response_cardinality": {
                    "queries": call_evidence[0]["response_count"],
                    "documents_with_needle": call_evidence[1]["response_count"],
                    "documents_without_needle": call_evidence[2]["response_count"],
                    "total": sum(item["response_count"] for item in call_evidence),
                },
                "input_character_count": sum(item["input_character_count"] for item in call_evidence),
                "embedding_dimensions": sorted({item["dimensions"] for item in call_evidence}),
                "provider_latency_ms": sum(item["latency_ms"] for item in call_evidence),
                "token_usage": self._sum_optional(call_evidence, "token_usage"),
                "cost_usd": self._sum_optional(call_evidence, "cost_usd"),
                "all_embeddings_finite": all(item["all_finite"] for item in call_evidence),
                "all_embeddings_unit_norm": all(item["all_unit_norm"] for item in call_evidence),
                "norm_range": {
                    "min": min(item["norm_min"] for item in call_evidence),
                    "max": max(item["norm_max"] for item in call_evidence),
                },
                "embedding_calls": call_evidence,
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

    def _embed_texts(
        self,
        provider: EmbeddingProvider,
        texts: list[str],
        *,
        task_type: str,
    ) -> EmbeddingResult:
        if self.use_cache:
            return provider.embed_text(texts, task_type=task_type)
        inputs = [EmbeddingInput(modality=ModalityType.TEXT, content=text) for text in texts]
        return provider.embed(inputs, task_type=task_type)

    @staticmethod
    def _call_evidence(
        label: str,
        texts: list[str],
        result: EmbeddingResult,
        task_type: str,
    ) -> dict[str, Any]:
        embeddings = np.asarray(result.embeddings, dtype=float)
        if embeddings.ndim != 2:
            raise ValueError(f"{label} response must be a 2D embedding matrix, got shape={embeddings.shape}")
        if embeddings.shape[0] != len(texts):
            raise ValueError(
                f"{label} response cardinality mismatch: expected {len(texts)}, got {embeddings.shape[0]}"
            )
        if embeddings.shape[1] != result.dimensions:
            raise ValueError(
                f"{label} response dimension mismatch: metadata={result.dimensions}, matrix={embeddings.shape[1]}"
            )

        all_finite = bool(np.isfinite(embeddings).all())
        if not all_finite:
            raise ValueError(f"{label} response contains non-finite embedding values")

        norms = np.linalg.norm(embeddings, axis=1)
        unit_norm_tolerance = 1e-3
        return {
            "label": label,
            "task_type": task_type,
            "input_count": len(texts),
            "input_character_count": sum(len(text) for text in texts),
            "response_count": int(embeddings.shape[0]),
            "dimensions": int(embeddings.shape[1]),
            "latency_ms": float(result.latency_ms),
            "token_usage": result.token_usage,
            "cost_usd": result.cost_usd,
            "cache_hit": bool(result.metadata.get("cache_hit", False)),
            "all_finite": all_finite,
            "unit_norm_tolerance": unit_norm_tolerance,
            "all_unit_norm": bool(np.all(np.abs(norms - 1.0) <= unit_norm_tolerance)),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
        }

    @staticmethod
    def _sum_optional(call_evidence: list[dict[str, Any]], key: str) -> float | int | None:
        if not call_evidence:
            return None
        total: float | int = 0
        for item in call_evidence:
            value = item.get(key)
            if value is None:
                return None
            total += value
        return total
