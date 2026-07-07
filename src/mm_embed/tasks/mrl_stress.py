"""Task A: MRL (Matryoshka Representation Learning) Stress Test.

Tests whether embedding quality degrades gracefully as dimensions are reduced.
Key question: How much can we compress embeddings before similarity ranking breaks?

Uses Spearman correlation between model cosine similarities and human STS-B scores
as the primary metric — this is the standard STS evaluation approach.

Optimization: Embeds all unique sentences ONCE at full dimension, then truncates
locally to test each reduced dimension. This exploits the MRL property that
lower-dimensional embeddings are just prefixes of the full embedding.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

from mm_embed.data.mock import get_mrl_test_data
from mm_embed.data.real_data import load_mrl_continuous_data
from mm_embed.providers.base import EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import cosine_similarity, dimension_retention_score

logger = logging.getLogger(__name__)


class MRLStressTask(EvalTask):
    """MRL Stress Test — evaluate embedding quality at varying dimensions.

    Test procedure:
    1. Collect all unique sentences from test pairs (ALL STS-B pairs, continuous scores)
    2. Embed them ONCE at full dimensions (batched)
    3. Truncate embeddings locally to each target dimension
    4. Measure Spearman correlation with human scores at each dimension level
    """

    name = "mrl_stress"
    description = "MRL dimension reduction stress test"
    required_modalities = {ModalityType.TEXT}

    DEFAULT_DIMS = [1024, 512, 256, 128, 64, 32]

    def __init__(
        self,
        dimensions: list[int] | None = None,
        use_mock: bool = False,
        max_samples: int | None = 150,
        hard_mode: bool = True,
        **kwargs: Any,
    ):
        self.test_dims = dimensions or self.DEFAULT_DIMS
        self.use_mock = use_mock
        self.max_samples = max_samples
        self.hard_mode = hard_mode

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        model_name = getattr(provider, "model", "unknown")

        if not provider.supports_mrl:
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=model_name,
                metrics={},
                error=f"Provider {provider.name} does not support MRL (dimension reduction)",
            )

        try:
            if self.use_mock:
                # Mock data returns binary labels; convert to continuous-like
                mock_data = get_mrl_test_data()
                test_data = [(a, b, 5.0 if sim else 0.0) for a, b, sim in mock_data]
            else:
                try:
                    test_data = load_mrl_continuous_data()
                except FileNotFoundError:
                    mock_data = get_mrl_test_data()
                    test_data = [(a, b, 5.0 if sim else 0.0) for a, b, sim in mock_data]

            # Subsample if max_samples is set
            if self.max_samples and len(test_data) > self.max_samples:
                import random
                rng = random.Random(42)

                if self.hard_mode:
                    # Biased sampling: prefer mid-range scores (1.5-4.0)
                    # which are ambiguous and discriminate model quality
                    mid = [t for t in test_data if 1.5 <= t[2] <= 4.0]
                    easy = [t for t in test_data if t[2] < 1.0 or t[2] > 4.5]
                    rest = [t for t in test_data if t not in mid and t not in easy]

                    n_total = self.max_samples
                    n_easy_target = min(max(int(n_total * 0.2), 1), n_total)
                    n_easy = min(len(easy), n_easy_target)
                    n_mid = min(len(mid), n_total - n_easy)
                    n_rest = n_total - n_easy - n_mid

                    selected = (
                        rng.sample(mid, min(n_mid, len(mid)))
                        + rng.sample(easy, min(n_easy, len(easy)))
                    )
                    if n_rest > 0 and rest:
                        selected += rng.sample(rest, min(n_rest, len(rest)))

                    test_data = selected[:n_total]
                    rng.shuffle(test_data)
                    logger.info("Hard mode: %d mid-range, %d easy, %d other",
                                min(n_mid, len(mid)), min(n_easy, len(easy)),
                                min(n_rest, len(rest)) if n_rest > 0 else 0)
                else:
                    test_data = rng.sample(test_data, self.max_samples)

            human_scores = np.array([score for _, _, score in test_data])
            logger.info("MRL stress test: %d pairs, score range [%.1f, %.1f], mean=%.2f",
                        len(test_data), human_scores.min(), human_scores.max(), human_scores.mean())

            # Collect all unique sentences and build index mapping
            unique_texts: list[str] = []
            text_to_idx: dict[str, int] = {}
            for a, b, _ in test_data:
                for t in (a, b):
                    if t not in text_to_idx:
                        text_to_idx[t] = len(unique_texts)
                        unique_texts.append(t)

            logger.info("Unique sentences to embed: %d", len(unique_texts))

            # Embed ALL unique sentences ONCE at full dimension
            full_dim = provider.default_dimensions
            logger.info("Embedding at full dimension (%d)...", full_dim)
            result_full = provider.embed_text(unique_texts, dimensions=full_dim)
            full_embeddings = result_full.embeddings  # shape: (n_unique, full_dim)

            # Lookup pair embeddings via index
            idx_a = [text_to_idx[a] for a, _, _ in test_data]
            idx_b = [text_to_idx[b] for _, b, _ in test_data]

            emb_a_full = full_embeddings[idx_a]
            emb_b_full = full_embeddings[idx_b]

            # Compute full-dim similarities and Spearman correlation
            full_sims = np.array([
                cosine_similarity(a, b) for a, b in zip(emb_a_full, emb_b_full)
            ])
            full_spearman = stats.spearmanr(full_sims, human_scores).statistic

            metrics: dict[str, float] = {
                f"spearman_dim_{full_dim}": full_spearman,
                "n_pairs": float(len(test_data)),
                "n_unique_sentences": float(len(unique_texts)),
            }
            dim_details: dict[str, Any] = {
                "full_dim": full_dim,
                "full_spearman": full_spearman,
                "per_dim": {},
            }

            # Test each reduced dimension by TRUNCATING (MRL property)
            for dim in self.test_dims:
                if dim >= full_dim:
                    continue

                logger.info("Evaluating truncated dimension %d...", dim)

                # Truncate: just take the first `dim` components
                emb_a_trunc = emb_a_full[:, :dim]
                emb_b_trunc = emb_b_full[:, :dim]

                reduced_sims = np.array([
                    cosine_similarity(a, b) for a, b in zip(emb_a_trunc, emb_b_trunc)
                ])

                spearman = stats.spearmanr(reduced_sims, human_scores).statistic
                retention = dimension_retention_score(emb_a_full, emb_a_trunc)

                metrics[f"spearman_dim_{dim}"] = spearman
                metrics[f"retention_dim_{dim}"] = retention
                dim_details["per_dim"][dim] = {
                    "spearman": spearman,
                    "retention": retention,
                    "spearman_drop": full_spearman - spearman,
                }

            # Find minimum viable dimension (Spearman > 0.7)
            min_viable = full_dim
            for dim in sorted(self.test_dims):
                key = f"spearman_dim_{dim}"
                if key in metrics and metrics[key] >= 0.7:
                    min_viable = dim
                    break

            metrics["min_viable_dim"] = float(min_viable)
            dim_details["min_viable_dim"] = min_viable

            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=model_name,
                metrics=metrics,
                details=dim_details,
            )

        except Exception as e:
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=model_name,
                metrics={},
                error=str(e),
            )
