"""ARK provider — Volcengine/ByteDance embedding via OpenAI-compatible API."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType

logger = logging.getLogger(__name__)


class ArkProvider(EmbeddingProvider):
    """Volcengine ARK embedding API (OpenAI-compatible interface).

    Uses the OpenAI SDK with a custom base_url pointing to Volcengine ARK.
    This is a text-only provider — images are not supported through the
    OpenAI-compatible embedding endpoint.

    Pricing: Pay-as-you-go on Volcengine ARK
    Access: China mainland direct, no VPN needed.
    Free tier: ~500K tokens — skip heavy tasks like needle_in_haystack.
    """

    name = "ark"
    supported_modalities = {ModalityType.TEXT}
    max_text_length = 8192
    default_dimensions = 2560
    supports_mrl = False

    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key or os.environ.get("ARK_API_KEY"), **kwargs)
        # Model can be an endpoint ID (ep-xxxx) or model name
        self.model = model or os.environ.get("ARK_ENDPOINT_ID", "doubao-embedding-text-240715")
        self.base_url = base_url or os.environ.get("ARK_BASE_URL", self.DEFAULT_BASE_URL)

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        texts = []
        for inp in inputs:
            if inp.modality != ModalityType.TEXT:
                raise ValueError(f"ARK provider only supports text embeddings, got {inp.modality}")
            texts.append(inp.content)

        # Batch in groups of 25 to stay safe
        batch_size = 25
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
        actual_dim = embeddings.shape[1] if len(embeddings) > 0 else self.default_dimensions
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=actual_dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
            token_usage=total_tokens if total_tokens > 0 else None,
        )
