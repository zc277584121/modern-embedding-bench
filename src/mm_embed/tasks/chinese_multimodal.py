"""Task E: Chinese Multimodal Retrieval.

Tests embedding models' ability to handle Chinese text and cross-lingual scenarios:
- Chinese text → image retrieval
- Cross-lingual alignment (Chinese query ↔ English query → same image)
- Chinese visual document understanding
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mm_embed.data.mock import (
    get_chinese_cross_lingual_pairs,
    get_chinese_text_image_data,
)
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import (
    cosine_similarity,
    cosine_similarity_matrix,
    mrr,
    recall_at_k,
)


class ChineseMultimodalTask(EvalTask):
    """Chinese Multimodal Retrieval — CJK text + image evaluation.

    Test procedure:
    1. Chinese text → image retrieval (like cross_modal but with Chinese text)
    2. Cross-lingual alignment: embed Chinese and English queries for the same concept,
       measure whether they produce similar embeddings
    3. Compute:
       - Chinese t2i Recall@1, Recall@5, MRR
       - Cross-lingual similarity (Chinese vs English query vectors)
       - Cross-lingual retrieval consistency (do both languages retrieve the same images?)
    """

    name = "chinese_multimodal"
    description = "Chinese multimodal and cross-lingual retrieval"
    required_modalities = {ModalityType.TEXT, ModalityType.IMAGE}

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        if not self.check_compatibility(provider):
            return self._run_text_only(provider)

        try:
            return self._run_full(provider)
        except Exception as e:
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=getattr(provider, "model", "unknown"),
                metrics={},
                error=str(e),
            )

    def _run_full(self, provider: EmbeddingProvider) -> EvalResult:
        """Full multimodal + cross-lingual evaluation."""
        data = get_chinese_text_image_data()
        cross_lingual = get_chinese_cross_lingual_pairs()
        n = len(data)

        # --- Part 1: Chinese text → image retrieval ---
        text_inputs = [EmbeddingInput(ModalityType.TEXT, d.text) for d in data]
        text_result = provider.embed(text_inputs, task_type="retrieval_query")

        image_inputs = [EmbeddingInput(ModalityType.IMAGE, d.image_bytes) for d in data]
        image_result = provider.embed(image_inputs)

        text_embs = text_result.embeddings
        image_embs = image_result.embeddings

        ground_truth = np.arange(n)
        t2i_sim = cosine_similarity_matrix(text_embs, image_embs)

        t2i_r1 = recall_at_k(t2i_sim, ground_truth, k=1)
        t2i_r5 = recall_at_k(t2i_sim, ground_truth, k=5)
        t2i_mrr = mrr(t2i_sim, ground_truth)

        # --- Part 2: Cross-lingual alignment ---
        zh_queries = [zh for zh, en in cross_lingual]
        en_queries = [en for zh, en in cross_lingual]

        zh_result = provider.embed_text(zh_queries, task_type="retrieval_query")
        en_result = provider.embed_text(en_queries, task_type="retrieval_query")

        # Measure pairwise Chinese-English similarity
        cross_lingual_sims = []
        for zh_emb, en_emb in zip(zh_result.embeddings, en_result.embeddings):
            sim = cosine_similarity(zh_emb, en_emb)
            cross_lingual_sims.append(sim)

        avg_cross_lingual_sim = float(np.mean(cross_lingual_sims))
        min_cross_lingual_sim = float(np.min(cross_lingual_sims))

        # --- Part 3: Cross-lingual retrieval consistency ---
        # Do Chinese and English queries retrieve the same top-3 images?
        consistency_scores = []
        for zh_emb, en_emb in zip(zh_result.embeddings, en_result.embeddings):
            zh_sims = np.array([cosine_similarity(zh_emb, ie) for ie in image_embs])
            en_sims = np.array([cosine_similarity(en_emb, ie) for ie in image_embs])
            zh_top3 = set(np.argsort(-zh_sims)[:3])
            en_top3 = set(np.argsort(-en_sims)[:3])
            overlap = len(zh_top3 & en_top3) / 3.0
            consistency_scores.append(overlap)

        avg_consistency = float(np.mean(consistency_scores))

        metrics = {
            "zh_t2i_recall@1": t2i_r1,
            "zh_t2i_recall@5": t2i_r5,
            "zh_t2i_mrr": t2i_mrr,
            "cross_lingual_similarity": avg_cross_lingual_sim,
            "cross_lingual_min_similarity": min_cross_lingual_sim,
            "cross_lingual_retrieval_consistency": avg_consistency,
        }

        details = {
            "n_zh_pairs": n,
            "n_cross_lingual_pairs": len(cross_lingual),
            "categories": [d.category for d in data],
            "per_pair_cross_lingual_sim": cross_lingual_sims,
            "per_pair_consistency": consistency_scores,
        }

        return EvalResult(
            task_name=self.name,
            provider_name=provider.name,
            model_name=getattr(provider, "model", "unknown"),
            metrics=metrics,
            details=details,
        )

    def _run_text_only(self, provider: EmbeddingProvider) -> EvalResult:
        """Text-only: just test cross-lingual alignment."""
        cross_lingual = get_chinese_cross_lingual_pairs()

        zh_queries = [zh for zh, en in cross_lingual]
        en_queries = [en for zh, en in cross_lingual]

        zh_result = provider.embed_text(zh_queries)
        en_result = provider.embed_text(en_queries)

        cross_lingual_sims = []
        for zh_emb, en_emb in zip(zh_result.embeddings, en_result.embeddings):
            sim = cosine_similarity(zh_emb, en_emb)
            cross_lingual_sims.append(sim)

        avg_sim = float(np.mean(cross_lingual_sims))
        min_sim = float(np.min(cross_lingual_sims))

        return EvalResult(
            task_name=self.name,
            provider_name=provider.name,
            model_name=getattr(provider, "model", "unknown"),
            metrics={
                "cross_lingual_similarity": avg_sim,
                "cross_lingual_min_similarity": min_sim,
            },
            details={
                "mode": "text_only",
                "n_pairs": len(cross_lingual),
                "per_pair_sim": cross_lingual_sims,
            },
        )
