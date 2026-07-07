"""OpenAI provider — text-embedding-3-large/small (baseline)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType

logger = logging.getLogger(__name__)


class OpenAIProvider(EmbeddingProvider):
    """OpenAI text embedding API (used as baseline).

    Models:
        - text-embedding-3-large: 3072 dims, MMTEB 58.9 (baseline)
        - text-embedding-3-small: 1536 dims, cheaper

    Note: OpenAI does NOT support multimodal embeddings (text only).

    Pricing: $0.13/M tokens (large), $0.02/M tokens (small)
    Access: Requires VPN from China mainland, or use Azure OpenAI (has China region).
    """

    name = "openai"
    supported_modalities = {ModalityType.TEXT}
    max_text_length = 8191
    default_dimensions = 3072
    supports_mrl = True  # text-embedding-3 supports dimension reduction

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-large",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key or os.environ.get("OPENAI_API_KEY"), **kwargs)
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if model == "text-embedding-3-small":
            self.default_dimensions = 1536
        else:
            self.default_dimensions = 3072

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = OpenAI(**client_kwargs)
        dim = dimensions or self.default_dimensions

        texts = []
        for inp in inputs:
            if inp.modality != ModalityType.TEXT:
                raise ValueError(f"OpenAI only supports text embeddings, got {inp.modality}")
            texts.append(inp.content)

        # OpenAI supports batch natively (up to 2048 texts per call)
        batch_size = 2048
        all_embeddings = []
        total_latency = 0.0
        total_tokens = 0
        n_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]

            if n_batches > 1:
                logger.info("Embedding batch %d/%d (%d items)...", batch_idx + 1, n_batches, len(batch))

            embed_kwargs: dict[str, Any] = {
                "input": batch,
                "model": self.model,
            }
            if dim != self.default_dimensions:
                embed_kwargs["dimensions"] = dim

            def _call(kw=embed_kwargs):
                return client.embeddings.create(**kw)

            response, latency = self._call_with_retry(_call)
            total_latency += latency

            # Sort by index to handle out-of-order API responses
            sorted_data = sorted(response.data, key=lambda x: x.index)
            for item in sorted_data:
                all_embeddings.append(item.embedding)

            if response.usage:
                total_tokens += response.usage.total_tokens

            if batch_idx < n_batches - 1:
                time.sleep(1.0)

        embeddings = np.array(all_embeddings)
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
            token_usage=total_tokens if total_tokens > 0 else None,
        )
