"""Cohere provider — Cohere Embed v4."""

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


class CohereProvider(EmbeddingProvider):
    """Cohere Embed v4 API.

    Models:
        - embed-v4.0: Multimodal (text+image+interleaved), 128K context
        - embed-multilingual-v3.0: Text-only, multilingual
        - embed-english-v3.0: Text-only, English

    Pricing: $0.12/M tokens
    Access: Requires VPN from China mainland. Free trial tier available.
    """

    name = "cohere"
    supported_modalities = {ModalityType.TEXT, ModalityType.IMAGE, ModalityType.DOCUMENT}
    max_text_length = 128000
    default_dimensions = 1024
    supports_mrl = False

    INPUT_TYPE_MAP = {
        "retrieval_query": "search_query",
        "retrieval_document": "search_document",
        "classification": "classification",
        "clustering": "clustering",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-v4.0",
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key or os.environ.get("COHERE_API_KEY"), **kwargs)
        self.model = model
        if self._uses_v2_client():
            self.default_dimensions = 1536

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        import cohere

        client = cohere.ClientV2(api_key=self.api_key) if self._uses_v2_client() else cohere.Client(api_key=self.api_key)
        dim = dimensions or self.default_dimensions

        # Separate text-only vs image inputs
        has_images = any(inp.modality in (ModalityType.IMAGE, ModalityType.DOCUMENT) for inp in inputs)

        if has_images:
            return self._embed_multimodal(client, inputs, dim, task_type)
        else:
            return self._embed_text(client, inputs, dim, task_type)

    def _uses_v2_client(self) -> bool:
        return self.model.startswith("embed-v4")

    def _embed_text(
        self,
        client: Any,
        inputs: list[EmbeddingInput],
        dim: int,
        task_type: str | None,
    ) -> EmbeddingResult:
        texts = [inp.content for inp in inputs]
        input_type = self.INPUT_TYPE_MAP.get(task_type, "search_document") if task_type else "search_document"

        # Batch manually (max 96 per call) with sleep to avoid rate limits
        batch_size = 96
        all_embeddings = []
        total_latency = 0.0
        total_tokens = 0
        n_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]

            if n_batches > 1:
                logger.info("Cohere text batch %d/%d (%d items)...", batch_idx + 1, n_batches, len(batch))

            def _call(b=batch):
                kwargs: dict[str, Any] = {
                    "texts": b,
                    "model": self.model,
                    "input_type": input_type,
                    "embedding_types": ["float"],
                }
                if self._uses_v2_client():
                    if dim != self.default_dimensions:
                        kwargs["output_dimension"] = dim
                else:
                    kwargs["batching"] = False
                return client.embed(**kwargs)

            response, latency = self._call_with_retry(_call)
            total_latency += latency

            vectors = response.embeddings.float_ if self._uses_v2_client() else response.embeddings.float
            all_embeddings.extend(vectors)
            if response.meta and response.meta.billed_units:
                total_tokens += response.meta.billed_units.input_tokens or 0

            # Sleep between batches to avoid rate limit
            if batch_idx < n_batches - 1:
                time.sleep(2.0)

        embeddings = np.array(all_embeddings)
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
            token_usage=total_tokens if total_tokens > 0 else None,
        )

    def _embed_multimodal(
        self,
        client: Any,
        inputs: list[EmbeddingInput],
        dim: int,
        task_type: str | None,
    ) -> EmbeddingResult:
        """Cohere v4 multimodal embedding using images parameter."""
        input_type = self.INPUT_TYPE_MAP.get(task_type, "search_document") if task_type else "search_document"

        all_embeddings = []
        total_latency = 0.0
        total_tokens = 0

        for inp in inputs:
            if inp.modality == ModalityType.TEXT:
                def _call():
                    kwargs: dict[str, Any] = {
                        "texts": [inp.content],
                        "model": self.model,
                        "input_type": input_type,
                        "embedding_types": ["float"],
                    }
                    if self._uses_v2_client() and dim != self.default_dimensions:
                        kwargs["output_dimension"] = dim
                    return client.embed(**kwargs)
                response, latency = self._call_with_retry(_call)
                vectors = response.embeddings.float_ if self._uses_v2_client() else response.embeddings.float
                emb = vectors[0]
            else:
                # Image input
                image_b64 = self._to_base64(inp.content)
                def _call():
                    kwargs = {
                        "images": [image_b64],
                        "model": self.model,
                        "input_type": "image" if self._uses_v2_client() else input_type,
                        "embedding_types": ["float"],
                    }
                    if self._uses_v2_client() and dim != self.default_dimensions:
                        kwargs["output_dimension"] = dim
                    return client.embed(**kwargs)
                response, latency = self._call_with_retry(_call)
                vectors = response.embeddings.float_ if self._uses_v2_client() else response.embeddings.float
                emb = vectors[0]

            all_embeddings.append(emb)
            total_latency += latency
            if response.meta and response.meta.billed_units:
                total_tokens += response.meta.billed_units.input_tokens or 0

        embeddings = np.array(all_embeddings)
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
            token_usage=total_tokens if total_tokens > 0 else None,
        )

    @staticmethod
    def _to_base64(content: str | bytes | Path) -> str:
        if isinstance(content, bytes):
            return f"data:image/png;base64,{base64.b64encode(content).decode()}"
        path = Path(content)
        if path.exists():
            data = path.read_bytes()
            suffix = path.suffix.lstrip(".") or "png"
            return f"data:image/{suffix};base64,{base64.b64encode(data).decode()}"
        return content
