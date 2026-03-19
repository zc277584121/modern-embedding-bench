"""Task D: Cross-Lingual Retrieval (Chinese <-> English).

Tests whether embedding models can match Chinese text to its English
equivalent and vice versa in a shared vector space.

Difficulty levels:
- Easy: direct translations ("我爱你" <-> "I love you")
- Medium: natural expression differences with nuance
- Hard: idioms, culturally-specific expressions, structural variations

Includes hard negative evaluation where similar-but-wrong translations
are mixed into the candidate pool.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mm_embed.data.real_data import load_crosslingual_data
from mm_embed.providers.base import EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import (
    cosine_similarity_matrix,
    mrr,
    recall_at_k,
)

logger = logging.getLogger(__name__)


class CrossLingualRetrievalTask(EvalTask):
    """Cross-Lingual Retrieval — Chinese <-> English bidirectional search.

    Test procedure:
    1. Load Chinese-English parallel pairs + hard negatives
    2. Embed all Chinese texts and all English texts separately
    3. Compute zh->en and en->zh similarity matrices
    4. Evaluate with hard negatives mixed into candidate pools
    5. Measure R@1, R@5, MRR, with and without hard negatives
    """

    name = "crosslingual_retrieval"
    description = "Cross-lingual retrieval (Chinese <-> English)"
    required_modalities = {ModalityType.TEXT}

    def __init__(self, use_mock: bool = False, **kwargs: Any):
        self.use_mock = use_mock

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        model_name = getattr(provider, "model", "unknown")

        try:
            data = load_crosslingual_data()
            n = len(data)
            logger.info("Cross-lingual retrieval: %d parallel pairs", n)

            # Separate texts
            zh_texts = [d.zh for d in data]
            en_texts = [d.en for d in data]

            # Collect all hard negatives
            hard_neg_en: list[str] = []
            hard_neg_zh: list[str] = []
            for d in data:
                hard_neg_en.extend(d.hard_negatives_en)
                hard_neg_zh.extend(d.hard_negatives_zh)

            logger.info("Hard negatives: %d EN, %d ZH", len(hard_neg_en), len(hard_neg_zh))

            # Embed all texts
            logger.info("Embedding %d Chinese texts...", n)
            zh_result = provider.embed_text(zh_texts)
            zh_embs = zh_result.embeddings  # (n, dim)

            logger.info("Embedding %d English texts...", n)
            en_result = provider.embed_text(en_texts)
            en_embs = en_result.embeddings  # (n, dim)

            # Ground truth: item i matches item i
            ground_truth = np.arange(n)

            # ============================================================
            # Standard evaluation: n Chinese vs n English
            # ============================================================
            # ZH -> EN
            zh2en_sim = cosine_similarity_matrix(zh_embs, en_embs)
            zh2en_r1 = recall_at_k(zh2en_sim, ground_truth, k=1)
            zh2en_r5 = recall_at_k(zh2en_sim, ground_truth, k=5)
            zh2en_mrr_val = mrr(zh2en_sim, ground_truth)

            # EN -> ZH
            en2zh_sim = cosine_similarity_matrix(en_embs, zh_embs)
            en2zh_r1 = recall_at_k(en2zh_sim, ground_truth, k=1)
            en2zh_r5 = recall_at_k(en2zh_sim, ground_truth, k=5)
            en2zh_mrr_val = mrr(en2zh_sim, ground_truth)

            avg_r1 = (zh2en_r1 + en2zh_r1) / 2

            # Language gap (analogous to modality gap)
            zh_centroid = zh_embs.mean(axis=0)
            en_centroid = en_embs.mean(axis=0)
            lang_gap = float(np.linalg.norm(zh_centroid - en_centroid))

            metrics: dict[str, float] = {
                "zh2en_recall@1": zh2en_r1,
                "zh2en_recall@5": zh2en_r5,
                "zh2en_mrr": zh2en_mrr_val,
                "en2zh_recall@1": en2zh_r1,
                "en2zh_recall@5": en2zh_r5,
                "en2zh_mrr": en2zh_mrr_val,
                "avg_recall@1": avg_r1,
                "language_gap": lang_gap,
            }

            # ============================================================
            # Per-difficulty breakdown
            # ============================================================
            for diff in ("easy", "medium", "hard"):
                indices = [i for i, d in enumerate(data) if d.difficulty == diff]
                if not indices:
                    continue
                idx = np.array(indices)
                # ZH->EN for this difficulty subset
                sub_zh = zh_embs[idx]
                sub_sim = cosine_similarity_matrix(sub_zh, en_embs)
                sub_gt = np.array([i for i in indices])  # correct index in full EN pool
                sub_r1 = recall_at_k(sub_sim, sub_gt, k=1)
                metrics[f"zh2en_recall@1_{diff}"] = sub_r1
                # EN->ZH for this difficulty subset
                sub_en = en_embs[idx]
                sub_sim2 = cosine_similarity_matrix(sub_en, zh_embs)
                sub_r1_2 = recall_at_k(sub_sim2, sub_gt, k=1)
                metrics[f"en2zh_recall@1_{diff}"] = sub_r1_2

            # ============================================================
            # Hard negative evaluation
            # ============================================================
            if hard_neg_en:
                logger.info("Embedding %d hard negative EN texts...", len(hard_neg_en))
                hn_en_result = provider.embed_text(hard_neg_en)
                hn_en_embs = hn_en_result.embeddings

                # Augmented EN pool: original + hard negatives
                aug_en_embs = np.vstack([en_embs, hn_en_embs])
                zh2en_hard_sim = cosine_similarity_matrix(zh_embs, aug_en_embs)
                zh2en_hard_r1 = recall_at_k(zh2en_hard_sim, ground_truth, k=1)
                zh2en_hard_r5 = recall_at_k(zh2en_hard_sim, ground_truth, k=5)
                zh2en_hard_mrr_val = mrr(zh2en_hard_sim, ground_truth)

                metrics["zh2en_hard_recall@1"] = zh2en_hard_r1
                metrics["zh2en_hard_recall@5"] = zh2en_hard_r5
                metrics["zh2en_hard_mrr"] = zh2en_hard_mrr_val
                metrics["n_hard_negatives_en"] = float(len(hard_neg_en))

            if hard_neg_zh:
                logger.info("Embedding %d hard negative ZH texts...", len(hard_neg_zh))
                hn_zh_result = provider.embed_text(hard_neg_zh)
                hn_zh_embs = hn_zh_result.embeddings

                # Augmented ZH pool: original + hard negatives
                aug_zh_embs = np.vstack([zh_embs, hn_zh_embs])
                en2zh_hard_sim = cosine_similarity_matrix(en_embs, aug_zh_embs)
                en2zh_hard_r1 = recall_at_k(en2zh_hard_sim, ground_truth, k=1)
                en2zh_hard_r5 = recall_at_k(en2zh_hard_sim, ground_truth, k=5)
                en2zh_hard_mrr_val = mrr(en2zh_hard_sim, ground_truth)

                metrics["en2zh_hard_recall@1"] = en2zh_hard_r1
                metrics["en2zh_hard_recall@5"] = en2zh_hard_r5
                metrics["en2zh_hard_mrr"] = en2zh_hard_mrr_val
                metrics["n_hard_negatives_zh"] = float(len(hard_neg_zh))

            # Combined hard metric
            if "zh2en_hard_recall@1" in metrics and "en2zh_hard_recall@1" in metrics:
                metrics["hard_avg_recall@1"] = (
                    metrics["zh2en_hard_recall@1"] + metrics["en2zh_hard_recall@1"]
                ) / 2

            details = {
                "n_pairs": n,
                "n_hard_negatives_en": len(hard_neg_en),
                "n_hard_negatives_zh": len(hard_neg_zh),
                "difficulties": {d.difficulty: 0 for d in data},
                "categories": list(set(d.category for d in data)),
            }
            for d in data:
                details["difficulties"][d.difficulty] += 1

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
