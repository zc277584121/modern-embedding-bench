"""Google Gemini provider — Gemini Embedding 2."""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType

logger = logging.getLogger(__name__)


class GeminiProvider(EmbeddingProvider):
    """Google Gemini Embedding API.

    Models:
        - gemini-embedding-2: Latest multimodal (text+image+video+audio+PDF)
        - text-embedding-004: Text-only, production stable
        - gemini-embedding-001: Text-only, MMTEB 68.4

    Pricing: ~$0.004/1K chars (extremely cheap)
    Access: Requires VPN from China mainland. Google AI Studio free tier available.
    """

    name = "gemini"
    supported_modalities = {
        ModalityType.TEXT,
        ModalityType.IMAGE,
        ModalityType.VIDEO,
        ModalityType.AUDIO,
        ModalityType.DOCUMENT,
    }
    max_text_length = 8192
    default_dimensions = 3072
    supports_mrl = True

    TASK_TYPE_MAP = {
        "retrieval_query": "RETRIEVAL_QUERY",
        "retrieval_document": "RETRIEVAL_DOCUMENT",
        "similarity": "SEMANTIC_SIMILARITY",
        "classification": "CLASSIFICATION",
        "clustering": "CLUSTERING",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-2",
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            **kwargs,
        )
        self.model = model

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        dim = dimensions or self.default_dimensions

        gemini_task_type = None
        if task_type and task_type in self.TASK_TYPE_MAP:
            gemini_task_type = self.TASK_TYPE_MAP[task_type]

        # Separate text inputs (batchable) from non-text inputs
        text_indices = [i for i, inp in enumerate(inputs) if inp.modality == ModalityType.TEXT]
        non_text_indices = [i for i, inp in enumerate(inputs) if inp.modality != ModalityType.TEXT]

        all_embeddings: list[tuple[int, list[float]]] = []
        total_latency = 0.0

        # Batch text inputs
        if text_indices:
            text_contents = [inputs[i].content for i in text_indices]
            batch_size = self.default_batch_size
            n_batches = (len(text_contents) + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(text_contents))
                batch = text_contents[start:end]

                if n_batches > 1:
                    logger.info("Embedding text batch %d/%d (%d items)...", batch_idx + 1, n_batches, len(batch))

                config_kwargs: dict[str, Any] = {"output_dimensionality": dim}
                if gemini_task_type:
                    config_kwargs["task_type"] = gemini_task_type

                embed_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "contents": batch,
                    "config": config_kwargs,
                }

                def _call(kw=embed_kwargs):
                    return client.models.embed_content(**kw)

                response, latency = self._call_with_retry(_call)
                total_latency += latency

                for j, emb_obj in enumerate(response.embeddings):
                    all_embeddings.append((text_indices[start + j], emb_obj.values))

                # Pause between batches to respect rate limits
                if batch_idx < n_batches - 1:
                    time.sleep(10.0)

        # Non-text inputs one by one (images, videos, etc. can't be batched)
        for idx_count, i in enumerate(non_text_indices):
            if idx_count > 0:
                if len(non_text_indices) > 10 and (idx_count + 1) % 10 == 0:
                    logger.info("Embedding non-text item %d/%d...", idx_count + 1, len(non_text_indices))
                time.sleep(5.0)  # throttle image requests
            content = self._build_content(inputs[i])

            config_kwargs = {"output_dimensionality": dim}
            if gemini_task_type:
                config_kwargs["task_type"] = gemini_task_type

            embed_kwargs = {
                "model": self.model,
                "contents": content,
                "config": config_kwargs,
            }

            def _call(kw=embed_kwargs):
                return client.models.embed_content(**kw)

            response, latency = self._call_with_retry(_call)
            total_latency += latency
            all_embeddings.append((i, response.embeddings[0].values))

        # Reassemble in original order
        all_embeddings.sort(key=lambda x: x[0])
        embeddings = np.array([emb for _, emb in all_embeddings])

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
        )

    def _build_content(self, inp: EmbeddingInput) -> list[Any]:
        """Build content parts for Gemini API."""
        from google.genai import types

        if inp.modality == ModalityType.TEXT:
            return [inp.content]

        if inp.modality == ModalityType.IMAGE:
            image_bytes = self._load_bytes(inp.content)
            part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            return [part]

        if inp.modality == ModalityType.VIDEO:
            video_bytes = self._load_bytes(inp.content)
            part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
            return [part]

        if inp.modality == ModalityType.AUDIO:
            audio_bytes = self._load_bytes(inp.content)
            part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
            return [part]

        if inp.modality == ModalityType.DOCUMENT:
            doc_bytes = self._load_bytes(inp.content)
            part = types.Part.from_bytes(data=doc_bytes, mime_type="application/pdf")
            return [part]

        raise ValueError(f"Unsupported modality for Gemini: {inp.modality}")

    @staticmethod
    def _load_bytes(content: str | bytes | Path) -> bytes:
        """Load content as raw bytes."""
        if isinstance(content, bytes):
            return content
        path = Path(content)
        if path.exists():
            return path.read_bytes()
        # Try base64 decode
        return base64.b64decode(content)
