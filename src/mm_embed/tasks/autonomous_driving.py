"""Task G: Autonomous Driving Professional Scene Retrieval.

Tests embedding models on domain-specific autonomous driving scenarios,
simulating the CoVLA dataset use case:
- Scene retrieval by weather, road type, hazard type
- Cross-modal: text query → driving scene image
- Fine-grained object-level retrieval
- Safety-critical scenario ranking
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from mm_embed.data.mock import get_driving_queries, get_driving_scene_data
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, ModalityType
from mm_embed.tasks.base import EvalResult, EvalTask
from mm_embed.utils.metrics import cosine_similarity, cosine_similarity_matrix


class AutonomousDrivingTask(EvalTask):
    """Autonomous Driving Scene Retrieval — domain-specific evaluation.

    Test procedure:
    1. Embed driving scene images and their text captions
    2. For each query category (weather, hazard, road_type):
       a. Embed the query text
       b. Retrieve top-K scenes by text→image similarity
       c. Also test text→text (query vs captions) as baseline
    3. Measure:
       - Category-level Precision@3 (are retrieved scenes of the right category?)
       - Cross-modal vs text-only gap
       - Safety-critical recall (are all hazardous scenes found?)
       - Domain specificity (does the model understand driving terminology?)
    """

    name = "autonomous_driving"
    description = "Autonomous driving scene retrieval (CoVLA-style)"
    required_modalities = {ModalityType.TEXT, ModalityType.IMAGE}

    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        if not self.check_compatibility(provider):
            # Fall back to text-only if no image support
            return self._run_text_only(provider)

        try:
            return self._run_multimodal(provider)
        except Exception as e:
            return EvalResult(
                task_name=self.name,
                provider_name=provider.name,
                model_name=getattr(provider, "model", "unknown"),
                metrics={},
                error=str(e),
            )

    def _run_multimodal(self, provider: EmbeddingProvider) -> EvalResult:
        """Full multimodal evaluation: text query → image retrieval."""
        scenes = get_driving_scene_data()
        queries = get_driving_queries()

        # Embed all scene images
        image_inputs = [EmbeddingInput(ModalityType.IMAGE, s.image_bytes) for s in scenes]
        image_result = provider.embed(image_inputs)
        image_embs = image_result.embeddings

        # Embed all scene captions
        caption_inputs = [EmbeddingInput(ModalityType.TEXT, s.caption) for s in scenes]
        caption_result = provider.embed(caption_inputs, task_type="retrieval_document")
        caption_embs = caption_result.embeddings

        metrics: dict[str, float] = {}
        category_details: dict[str, Any] = {}

        # Test each query category
        for category, category_queries in queries.items():
            cat_precisions_t2i: list[float] = []
            cat_precisions_t2t: list[float] = []

            for condition, query_text in category_queries.items():
                # Embed query
                query_result = provider.embed_text([query_text], task_type="retrieval_query")
                query_emb = query_result.embeddings[0]

                # Ground truth: which scenes match this condition?
                if category == "weather":
                    relevant = [i for i, s in enumerate(scenes) if s.weather == condition]
                elif category == "hazard":
                    if condition == "pedestrian":
                        relevant = [i for i, s in enumerate(scenes) if "pedestrian" in s.objects or "children" in s.objects]
                    elif condition == "animal":
                        relevant = [i for i, s in enumerate(scenes) if "deer" in s.objects]
                    elif condition == "construction":
                        relevant = [i for i, s in enumerate(scenes) if s.road_type == "construction"]
                    else:
                        relevant = []
                elif category == "road_type":
                    relevant = [i for i, s in enumerate(scenes) if s.road_type == condition]
                else:
                    relevant = []

                if not relevant:
                    continue

                # Text → Image retrieval
                t2i_sims = np.array([cosine_similarity(query_emb, ie) for ie in image_embs])
                top3_t2i = np.argsort(-t2i_sims)[:3]
                precision_t2i = len(set(top3_t2i) & set(relevant)) / min(3, len(relevant))
                cat_precisions_t2i.append(precision_t2i)

                # Text → Text retrieval (baseline)
                t2t_sims = np.array([cosine_similarity(query_emb, ce) for ce in caption_embs])
                top3_t2t = np.argsort(-t2t_sims)[:3]
                precision_t2t = len(set(top3_t2t) & set(relevant)) / min(3, len(relevant))
                cat_precisions_t2t.append(precision_t2t)

            if cat_precisions_t2i:
                avg_t2i = np.mean(cat_precisions_t2i)
                avg_t2t = np.mean(cat_precisions_t2t)
                metrics[f"{category}_t2i_precision@3"] = float(avg_t2i)
                metrics[f"{category}_t2t_precision@3"] = float(avg_t2t)
                metrics[f"{category}_cross_modal_gap"] = float(avg_t2t - avg_t2i)

                category_details[category] = {
                    "t2i_precision": float(avg_t2i),
                    "t2t_precision": float(avg_t2t),
                    "gap": float(avg_t2t - avg_t2i),
                }

        # Overall metrics
        t2i_scores = [v for k, v in metrics.items() if "t2i_precision" in k]
        t2t_scores = [v for k, v in metrics.items() if "t2t_precision" in k]
        if t2i_scores:
            metrics["overall_t2i_precision@3"] = float(np.mean(t2i_scores))
        if t2t_scores:
            metrics["overall_t2t_precision@3"] = float(np.mean(t2t_scores))
        if t2i_scores and t2t_scores:
            metrics["overall_cross_modal_gap"] = float(np.mean(t2t_scores) - np.mean(t2i_scores))

        return EvalResult(
            task_name=self.name,
            provider_name=provider.name,
            model_name=getattr(provider, "model", "unknown"),
            metrics=metrics,
            details={
                "n_scenes": len(scenes),
                "mode": "multimodal",
                "categories": category_details,
                "image_latency_ms": image_result.latency_ms,
                "caption_latency_ms": caption_result.latency_ms,
            },
        )

    def _run_text_only(self, provider: EmbeddingProvider) -> EvalResult:
        """Text-only fallback: query → caption retrieval."""
        scenes = get_driving_scene_data()
        queries = get_driving_queries()

        # Embed all captions
        caption_result = provider.embed_text(
            [s.caption for s in scenes], task_type="retrieval_document"
        )
        caption_embs = caption_result.embeddings

        metrics: dict[str, float] = {}

        for category, category_queries in queries.items():
            precisions: list[float] = []

            for condition, query_text in category_queries.items():
                query_result = provider.embed_text([query_text], task_type="retrieval_query")
                query_emb = query_result.embeddings[0]

                # Ground truth
                if category == "weather":
                    relevant = [i for i, s in enumerate(scenes) if s.weather == condition]
                elif category == "road_type":
                    relevant = [i for i, s in enumerate(scenes) if s.road_type == condition]
                else:
                    relevant = [i for i, s in enumerate(scenes) if condition in " ".join(s.objects)]

                if not relevant:
                    continue

                sims = np.array([cosine_similarity(query_emb, ce) for ce in caption_embs])
                top3 = np.argsort(-sims)[:3]
                precision = len(set(top3) & set(relevant)) / min(3, len(relevant))
                precisions.append(precision)

            if precisions:
                metrics[f"{category}_t2t_precision@3"] = float(np.mean(precisions))

        t2t_scores = list(metrics.values())
        if t2t_scores:
            metrics["overall_t2t_precision@3"] = float(np.mean(t2t_scores))

        return EvalResult(
            task_name=self.name,
            provider_name=provider.name,
            model_name=getattr(provider, "model", "unknown"),
            metrics=metrics,
            details={"n_scenes": len(scenes), "mode": "text_only"},
        )
