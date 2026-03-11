"""Task H: Needle-in-a-Haystack for Embeddings.

Tests whether embedding models can accurately retrieve specific facts
embedded within long documents of varying lengths.

Key insight: Unlike LLM needle-in-a-haystack (which tests generation),
this tests whether the embedding of a long document preserves enough
semantic information about specific details to be retrieved by a query.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from mm_embed.data.mock import get_needle_haystack_data
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import cosine_similarity


class NeedleInHaystackTask(EvalTask):
    """Needle-in-a-Haystack for Embeddings.

    Test procedure:
    1. Create documents of varying lengths (1K, 4K, 8K, 16K, 32K chars)
    2. Insert a "needle" (specific fact) at different positions (0%, 25%, 50%, 75%, 100%)
    3. Embed the full document as a single vector
    4. Embed the query asking about the needle
    5. Compare: does the query match the needle-containing document better than
       the same haystack without the needle?
    6. Measure retrieval accuracy across length × position matrix

    This tests the "semantic density dilution" problem: as documents get longer,
    specific facts get increasingly "drowned out" in the embedding.
    """

    name = "needle_in_haystack"
    description = "Embedding retrieval of specific facts in long documents"
    required_modalities = {ModalityType.TEXT}

    def __init__(
        self,
        haystack_lengths: list[int] | None = None,
        needle_positions: list[float] | None = None,
        **kwargs: Any,
    ):
        self.haystack_lengths = haystack_lengths or [1000, 4000, 8000, 16000, 32000]
        self.needle_positions = needle_positions or [0.0, 0.25, 0.5, 0.75, 1.0]

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        try:
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
                    model_name=getattr(provider, "model", "unknown"),
                    metrics={},
                    error=f"No test cases fit within provider's max context ({provider.max_text_length} tokens)",
                )

            # Strategy: for each query, embed the document WITH the needle and WITHOUT
            # If sim(query, doc_with_needle) > sim(query, doc_without_needle), it's a hit

            # Group by (length, position)
            results_by_length: dict[int, list[bool]] = defaultdict(list)
            results_by_position: dict[float, list[bool]] = defaultdict(list)
            results_by_length_position: dict[tuple[int, float], list[bool]] = defaultdict(list)

            for tc in valid_cases:
                doc_with_needle = tc["document"]
                query = tc["query"]
                needle = tc["needle"]
                length = tc["length"]
                position = tc["position"]

                # Create document without needle
                doc_without_needle = doc_with_needle.replace(needle, "")

                # Embed query, doc_with_needle, doc_without_needle
                query_result = provider.embed_text([query], task_type="retrieval_query")
                docs_result = provider.embed_text(
                    [doc_with_needle, doc_without_needle],
                    task_type="retrieval_document",
                )

                query_emb = query_result.embeddings[0]
                doc_with_emb = docs_result.embeddings[0]
                doc_without_emb = docs_result.embeddings[1]

                sim_with = cosine_similarity(query_emb, doc_with_emb)
                sim_without = cosine_similarity(query_emb, doc_without_emb)

                hit = sim_with > sim_without

                results_by_length[length].append(hit)
                results_by_position[position].append(hit)
                results_by_length_position[(length, position)].append(hit)

            # Compute metrics
            metrics: dict[str, float] = {}

            # Overall accuracy
            all_hits = [h for hits in results_by_length.values() for h in hits]
            metrics["overall_accuracy"] = sum(all_hits) / len(all_hits) if all_hits else 0.0

            # Accuracy by document length
            for length in sorted(results_by_length.keys()):
                hits = results_by_length[length]
                metrics[f"accuracy_len_{length}"] = sum(hits) / len(hits) if hits else 0.0

            # Accuracy by needle position
            for pos in sorted(results_by_position.keys()):
                hits = results_by_position[pos]
                pos_label = f"{int(pos * 100)}pct"
                metrics[f"accuracy_pos_{pos_label}"] = sum(hits) / len(hits) if hits else 0.0

            # Compute degradation rate (accuracy drop per doubling of length)
            lengths_sorted = sorted(results_by_length.keys())
            if len(lengths_sorted) >= 2:
                first_acc = sum(results_by_length[lengths_sorted[0]]) / len(results_by_length[lengths_sorted[0]])
                last_acc = sum(results_by_length[lengths_sorted[-1]]) / len(results_by_length[lengths_sorted[-1]])
                metrics["degradation_rate"] = first_acc - last_acc

            # Build the length × position heatmap for details
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
                model_name=getattr(provider, "model", "unknown"),
                metrics=metrics,
                details=details,
            )

        except Exception as e:
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=getattr(provider, "model", "unknown"),
                metrics={},
                error=str(e),
            )
