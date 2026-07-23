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

GEMINI_EMBEDDING_2_MODEL = "gemini-embedding-2"
GEMINI_EMBEDDING_2_TEXT_PREFIXES = {
    "retrieval_query": "task: search result | query: {text}",
    "similarity": "task: sentence similarity | query: {text}",
    "classification": "task: classification | query: {text}",
    "clustering": "task: clustering | query: {text}",
}


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
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        dim = dimensions or self.default_dimensions
        is_gemini_embedding_2 = self.model.removeprefix("models/") == GEMINI_EMBEDDING_2_MODEL

        gemini_task_type = None
        if not is_gemini_embedding_2 and task_type and task_type in self.TASK_TYPE_MAP:
            gemini_task_type = self.TASK_TYPE_MAP[task_type]

        # Separate text inputs (batchable) from non-text inputs
        text_indices = [i for i, inp in enumerate(inputs) if inp.modality == ModalityType.TEXT]
        non_text_indices = [i for i, inp in enumerate(inputs) if inp.modality != ModalityType.TEXT]

        all_embeddings: list[tuple[int, list[float]]] = []
        total_latency = 0.0

        # Batch text inputs
        if text_indices:
            text_inputs = [inputs[i] for i in text_indices]
            batch_size = self.default_batch_size
            n_batches = (len(text_inputs) + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(text_inputs))
                batch_inputs = text_inputs[start:end]

                if is_gemini_embedding_2:
                    batch = [
                        types.Content(
                            parts=[types.Part.from_text(text=self._format_gemini_embedding_2_text(inp, task_type))]
                        )
                        for inp in batch_inputs
                    ]
                else:
                    batch = [inp.content for inp in batch_inputs]

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

                response_embeddings = self._validated_response_embeddings(response, len(batch), "text batch")
                for j, emb_obj in enumerate(response_embeddings):
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
            response_embeddings = self._validated_response_embeddings(response, 1, "non-text input")
            all_embeddings.append((i, response_embeddings[0].values))

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

    def _format_gemini_embedding_2_text(self, inp: EmbeddingInput, task_type: str | None) -> str:
        """Apply the official Gemini Embedding 2 text instruction format."""
        text = str(inp.content)
        if task_type == "retrieval_document":
            title = inp.metadata.get("title") or "none"
            return f"title: {title} | text: {text}"
        if task_type in GEMINI_EMBEDDING_2_TEXT_PREFIXES:
            return GEMINI_EMBEDDING_2_TEXT_PREFIXES[task_type].format(text=text)
        return text

    def _cache_content_for_input(self, inp: EmbeddingInput, task_type: str | None) -> Any:
        """Version Gemini 2 text cache entries by their effective request content."""
        if self.model.removeprefix("models/") == GEMINI_EMBEDDING_2_MODEL and inp.modality == ModalityType.TEXT:
            formatted_text = self._format_gemini_embedding_2_text(inp, task_type)
            return f"gemini-embedding-2-flat-content-v1\0{formatted_text}"
        return super()._cache_content_for_input(inp, task_type)

    def _validated_response_embeddings(self, response: Any, expected_count: int, request_kind: str) -> list[Any]:
        """Reject aggregated or missing rows before they can corrupt input alignment."""
        embeddings = list(response.embeddings or [])
        if len(embeddings) != expected_count:
            raise ValueError(
                f"Gemini model {self.model} returned {len(embeddings)} embeddings for "
                f"{expected_count} logical inputs in {request_kind}"
            )
        return embeddings

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
