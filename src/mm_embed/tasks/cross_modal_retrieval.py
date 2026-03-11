"""Task B: Cross-Modal Symmetric Retrieval.

Tests bidirectional retrieval between text and images:
- Text → Image: given a text query, find the matching image
- Image → Text: given an image, find the matching text description

Also measures the modality gap between text and image embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mm_embed.data.mock import get_cross_modal_data
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import (
    cosine_similarity_matrix,
    modality_gap,
    mrr,
    recall_at_k,
)


class CrossModalRetrievalTask(EvalTask):
    """Cross-Modal Symmetric Retrieval — text↔image bidirectional search.

    Test procedure:
    1. Embed all texts and all images separately
    2. Compute text→image and image→text similarity matrices
    3. Measure:
       - Text→Image Recall@1, Recall@5
       - Image→Text Recall@1, Recall@5
       - MRR in both directions
       - Modality gap (L2 distance between text/image centroids)
       - Symmetry score (difference between t2i and i2t recall)
    """

    name = "cross_modal_retrieval"
    description = "Cross-modal symmetric retrieval (text ↔ image)"
    required_modalities = {ModalityType.TEXT, ModalityType.IMAGE}

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
            data = get_cross_modal_data()
            n = len(data)

            # Embed all texts
            text_inputs = [EmbeddingInput(ModalityType.TEXT, d.text) for d in data]
            text_result = provider.embed(text_inputs, task_type="retrieval_document")

            # Embed all images
            image_inputs = [EmbeddingInput(ModalityType.IMAGE, d.image_bytes) for d in data]
            image_result = provider.embed(image_inputs)

            text_embs = text_result.embeddings  # (n, dim)
            image_embs = image_result.embeddings  # (n, dim)

            # Ground truth: item i matches item i (diagonal)
            ground_truth = np.arange(n)

            # Text → Image retrieval
            t2i_sim = cosine_similarity_matrix(text_embs, image_embs)
            t2i_r1 = recall_at_k(t2i_sim, ground_truth, k=1)
            t2i_r5 = recall_at_k(t2i_sim, ground_truth, k=5)
            t2i_mrr = mrr(t2i_sim, ground_truth)

            # Image → Text retrieval
            i2t_sim = cosine_similarity_matrix(image_embs, text_embs)
            i2t_r1 = recall_at_k(i2t_sim, ground_truth, k=1)
            i2t_r5 = recall_at_k(i2t_sim, ground_truth, k=5)
            i2t_mrr = mrr(i2t_sim, ground_truth)

            # Modality gap
            gap = modality_gap(text_embs, image_embs)

            # Symmetry score (1.0 = perfectly symmetric)
            symmetry = 1.0 - abs(t2i_r1 - i2t_r1)

            metrics = {
                "t2i_recall@1": t2i_r1,
                "t2i_recall@5": t2i_r5,
                "t2i_mrr": t2i_mrr,
                "i2t_recall@1": i2t_r1,
                "i2t_recall@5": i2t_r5,
                "i2t_mrr": i2t_mrr,
                "modality_gap": gap,
                "symmetry": symmetry,
                "avg_recall@1": (t2i_r1 + i2t_r1) / 2,
            }

            details = {
                "n_pairs": n,
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
