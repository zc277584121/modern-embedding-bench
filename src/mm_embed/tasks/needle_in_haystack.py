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

from mm_embed.data.mock import get_needle_haystack_data
from mm_embed.data.real_data import load_needle_haystack_real_data
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, ModalityType
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
        use_mock: bool = False,
        **kwargs: Any,
    ):
        self.haystack_lengths = haystack_lengths or [4000, 8000, 16000, 32000]
        self.needle_positions = needle_positions or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.use_mock = use_mock

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        model_name = getattr(provider, "model", "unknown")

        try:
            if self.use_mock:
                test_cases = get_needle_haystack_data(
                    haystack_lengths=self.haystack_lengths,
                    needle_positions=self.needle_positions,
                )
            else:
                try:
                    test_cases = load_needle_haystack_real_data(
                        haystack_lengths=self.haystack_lengths,
                        needle_positions=self.needle_positions,
                    )
                except FileNotFoundError:
                    test_cases = get_needle_haystack_data(
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
            query_result = provider.embed_text(query_list, task_type="retrieval_query")

            logger.info("Embedding documents (with needle)...")
            doc_with_result = provider.embed_text(doc_with_list, task_type="retrieval_document")

            logger.info("Embedding documents (without needle)...")
            doc_without_result = provider.embed_text(doc_without_list, task_type="retrieval_document")

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
                "n_test_cases": len(valid_cases),
                "n_skipped": len(test_cases) - len(valid_cases),
                "max_provider_context": provider.max_text_length,
                "heatmap": heatmap,
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
