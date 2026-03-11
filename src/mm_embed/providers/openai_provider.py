"""OpenAI provider — text-embedding-3-large/small (baseline)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType


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

        embed_kwargs: dict[str, Any] = {
            "input": texts,
            "model": self.model,
        }
        if dim != 3072:
            embed_kwargs["dimensions"] = dim

        def _call():
            return client.embeddings.create(**embed_kwargs)

        response, latency = self._timed_call(_call)

        embeddings = np.array([item.embedding for item in response.data])
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=latency,
            token_usage=response.usage.total_tokens if response.usage else None,
        )
