"""Task B: Cross-Modal Symmetric Retrieval.

Tests bidirectional retrieval between text and images:
- Text -> Image: given a text query, find the matching image
- Image -> Text: given an image, find the matching text description

Uses the FULL pool of 200 images/captions for retrieval (not subsampled).
Also injects hard negative captions to increase difficulty.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mm_embed.data.mock import get_cross_modal_data
from mm_embed.data.real_data import load_cross_modal_real_data
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import (
    cosine_similarity_matrix,
    modality_gap,
    mrr,
    recall_at_k,
)


class CrossModalRetrievalTask(EvalTask):
    """Cross-Modal Symmetric Retrieval — text<->image bidirectional search.

    Test procedure:
    1. Load ALL image-text pairs (200 for real data)
    2. Embed all texts and all images separately
    3. Compute text->image and image->text similarity matrices over the FULL pool
    4. Also evaluate with hard negative captions injected into the text pool
    5. Measure R@1, R@5, MRR, modality gap, symmetry
    """

    name = "cross_modal_retrieval"
    description = "Cross-modal symmetric retrieval (text <-> image)"
    required_modalities = {ModalityType.TEXT, ModalityType.IMAGE}

    def __init__(self, use_mock: bool = False, max_samples: int | None = None, **kwargs: Any):
        self.use_mock = use_mock
        self.max_samples = max_samples

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        if not self.check_compatibility(provider):
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=getattr(provider, "model", "unknown"),
                metrics={},
                error=f"Provider {provider.name} does not support required modalities: {self.required_modalities}",
            )

        try:
            if self.use_mock:
                data = get_cross_modal_data()
            else:
                try:
                    data = load_cross_modal_real_data()
                except FileNotFoundError:
                    data = get_cross_modal_data()

            if self.max_samples and len(data) > self.max_samples:
                import random
                data = random.Random(42).sample(data, self.max_samples)

            n = len(data)

            # Collect hard negatives for harder i2t evaluation
            hard_neg_texts: list[str] = []
            for d in data:
                if hasattr(d, "hard_negatives") and d.hard_negatives:
                    hard_neg_texts.extend(d.hard_negatives)

            # ================================================================
            # Standard evaluation: n images vs n captions
            # ================================================================

            # Embed all original texts
            text_inputs = [EmbeddingInput(ModalityType.TEXT, d.text) for d in data]
            text_result = provider.embed_with_cache(text_inputs)

            # Embed all images
            image_inputs = [EmbeddingInput(ModalityType.IMAGE, d.image_bytes) for d in data]
            image_result = provider.embed_with_cache(image_inputs)

            text_embs = text_result.embeddings  # (n, dim)
            image_embs = image_result.embeddings  # (n, dim)

            # Ground truth: item i matches item i (diagonal)
            ground_truth = np.arange(n)

            # Text -> Image retrieval
            t2i_sim = cosine_similarity_matrix(text_embs, image_embs)
            t2i_r1 = recall_at_k(t2i_sim, ground_truth, k=1)
            t2i_r5 = recall_at_k(t2i_sim, ground_truth, k=5)
            t2i_r10 = recall_at_k(t2i_sim, ground_truth, k=10)
            t2i_mrr_val = mrr(t2i_sim, ground_truth)

            # Image -> Text retrieval
            i2t_sim = cosine_similarity_matrix(image_embs, text_embs)
            i2t_r1 = recall_at_k(i2t_sim, ground_truth, k=1)
            i2t_r5 = recall_at_k(i2t_sim, ground_truth, k=5)
            i2t_r10 = recall_at_k(i2t_sim, ground_truth, k=10)
            i2t_mrr_val = mrr(i2t_sim, ground_truth)

            # Modality gap
            gap = modality_gap(text_embs, image_embs)

            # Symmetry score (1.0 = perfectly symmetric)
            symmetry = 1.0 - abs(t2i_r1 - i2t_r1)

            metrics = {
                "t2i_recall@1": t2i_r1,
                "t2i_recall@5": t2i_r5,
                "t2i_recall@10": t2i_r10,
                "t2i_mrr": t2i_mrr_val,
                "i2t_recall@1": i2t_r1,
                "i2t_recall@5": i2t_r5,
                "i2t_recall@10": i2t_r10,
                "i2t_mrr": i2t_mrr_val,
                "modality_gap": gap,
                "symmetry": symmetry,
                "avg_recall@1": (t2i_r1 + i2t_r1) / 2,
            }

            # ================================================================
            # Hard negative evaluation: image -> (original + hard neg) texts
            # ================================================================
            if hard_neg_texts:
                hard_neg_inputs = [EmbeddingInput(ModalityType.TEXT, t) for t in hard_neg_texts]
                hard_neg_result = provider.embed_with_cache(hard_neg_inputs)
                hard_neg_embs = hard_neg_result.embeddings  # (n_neg, dim)

                # Augmented text pool: original captions + hard negatives
                aug_text_embs = np.vstack([text_embs, hard_neg_embs])  # (n + n_neg, dim)

                # Image -> augmented text retrieval
                # Ground truth is still the first n indices
                i2t_aug_sim = cosine_similarity_matrix(image_embs, aug_text_embs)
                i2t_hard_r1 = recall_at_k(i2t_aug_sim, ground_truth, k=1)
                i2t_hard_r5 = recall_at_k(i2t_aug_sim, ground_truth, k=5)
                i2t_hard_mrr = mrr(i2t_aug_sim, ground_truth)

                metrics["i2t_hard_recall@1"] = i2t_hard_r1
                metrics["i2t_hard_recall@5"] = i2t_hard_r5
                metrics["i2t_hard_mrr"] = i2t_hard_mrr
                metrics["n_hard_negatives"] = float(len(hard_neg_texts))
                # Primary hard metric: average of standard t2i and hard-negative i2t
                metrics["hard_avg_recall@1"] = (t2i_r1 + i2t_hard_r1) / 2

            details = {
                "n_pairs": n,
                "n_hard_negatives": len(hard_neg_texts),
                "text_latency_ms": text_result.latency_ms,
                "image_latency_ms": image_result.latency_ms,
                "dimensions": text_result.dimensions,
                "categories": [d.category for d in data],
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
