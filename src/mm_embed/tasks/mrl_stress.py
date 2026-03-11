"""Task A: MRL (Matryoshka Representation Learning) Stress Test.

Tests whether embedding quality degrades gracefully as dimensions are reduced.
Key question: How much can we compress embeddings before similarity ranking breaks?
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mm_embed.data.mock import get_mrl_test_data
from mm_embed.providers.base import EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import cosine_similarity, dimension_retention_score


class MRLStressTask(EvalTask):
    """MRL Stress Test — evaluate embedding quality at varying dimensions.

    Test procedure:
    1. Embed text pairs at full dimensions
    2. Re-embed (or truncate) at progressively smaller dimensions:
       [full, 1024, 512, 256, 128, 64, 32]
    3. Measure:
       - Similarity preservation (Spearman rank correlation)
       - Similar/dissimilar pair discrimination (AUC)
       - Minimum viable dimension (where accuracy drops below threshold)
    """

    name = "mrl_stress"
    description = "MRL dimension reduction stress test"
    required_modalities = {ModalityType.TEXT}

    # Dimension levels to test
    DEFAULT_DIMS = [1024, 512, 256, 128, 64, 32]

    def __init__(self, dimensions: list[int] | None = None, **kwargs: Any):
        self.test_dims = dimensions or self.DEFAULT_DIMS

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        if not provider.supports_mrl:
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=getattr(provider, "model", "unknown"),
                metrics={},
                error=f"Provider {provider.name} does not support MRL (dimension reduction)",
            )

        try:
            test_data = get_mrl_test_data()
            all_texts_a = [a for a, b, _ in test_data]
            all_texts_b = [b for a, b, _ in test_data]
            labels = [sim for _, _, sim in test_data]

            # Step 1: Get full-dimensional embeddings
            full_dim = provider.default_dimensions
            result_a_full = provider.embed_text(all_texts_a, dimensions=full_dim)
            result_b_full = provider.embed_text(all_texts_b, dimensions=full_dim)

            # Compute full-dim similarities
            full_sims = np.array([
                cosine_similarity(a, b)
                for a, b in zip(result_a_full.embeddings, result_b_full.embeddings)
            ])
            full_auc = self._compute_auc(full_sims, labels)

            metrics: dict[str, float] = {
                f"auc_dim_{full_dim}": full_auc,
            }
            dim_details: dict[str, Any] = {
                "full_dim": full_dim,
                "full_auc": full_auc,
                "per_dim": {},
            }

            # Step 2: Test each reduced dimension
            for dim in self.test_dims:
                if dim >= full_dim:
                    continue

                result_a = provider.embed_text(all_texts_a, dimensions=dim)
                result_b = provider.embed_text(all_texts_b, dimensions=dim)

                # Similarity at reduced dim
                reduced_sims = np.array([
                    cosine_similarity(a, b)
                    for a, b in zip(result_a.embeddings, result_b.embeddings)
                ])

                # Metrics
                auc = self._compute_auc(reduced_sims, labels)
                retention = dimension_retention_score(
                    result_a_full.embeddings, result_a.embeddings
                )

                metrics[f"auc_dim_{dim}"] = auc
                metrics[f"retention_dim_{dim}"] = retention
                dim_details["per_dim"][dim] = {
                    "auc": auc,
                    "retention": retention,
                    "auc_drop": full_auc - auc,
                }

            # Find minimum viable dimension (AUC > 0.8)
            min_viable = full_dim
            for dim in sorted(self.test_dims):
                key = f"auc_dim_{dim}"
                if key in metrics and metrics[key] >= 0.8:
                    min_viable = dim
                    break

            metrics["min_viable_dim"] = float(min_viable)
            dim_details["min_viable_dim"] = min_viable

            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=getattr(provider, "model", "unknown"),
                metrics=metrics,
                details=dim_details,
            )

        except Exception as e:
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=getattr(provider, "model", "unknown"),
                metrics={},
                error=str(e),
            )

    @staticmethod
    def _compute_auc(similarities: np.ndarray, labels: list[bool]) -> float:
        """Compute AUC for binary classification (similar vs dissimilar)."""
        labels_arr = np.array(labels, dtype=float)

        # Sort by similarity descending
        sorted_indices = np.argsort(-similarities)
        sorted_labels = labels_arr[sorted_indices]

        # Simple AUC via trapezoidal rule
        n_pos = sorted_labels.sum()
        n_neg = len(sorted_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp = 0.0
        fp = 0.0
        auc = 0.0
        prev_fpr = 0.0

        for label in sorted_labels:
            if label == 1.0:
                tp += 1
            else:
                fp += 1
                fpr = fp / n_neg
                tpr = tp / n_pos
                auc += tpr * (fpr - prev_fpr)
                prev_fpr = fpr

        return auc
