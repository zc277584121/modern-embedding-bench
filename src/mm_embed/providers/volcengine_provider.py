"""Volcengine (火山引擎) provider — Seed-1.6-Embedding (ByteDance)."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType


class VolcEngineProvider(EmbeddingProvider):
    """ByteDance Volcengine ARK API for Seed-1.6-Embedding.

    Models:
        - seed-1.6-embedding: Multimodal (text+image+video), MMEB-V2 #2 (76.9)
        - seed-1.6-embedding-1215: Newer checkpoint, MMEB-V2 76.9

    Pricing: Volcengine ARK pay-as-you-go
    Access: China mainland direct, no VPN needed. Requires Volcengine account.
    """

    name = "volcengine"
    supported_modalities = {ModalityType.TEXT, ModalityType.IMAGE, ModalityType.VIDEO}
    max_text_length = 32768
    default_dimensions = 1024
    supports_mrl = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "seed-1.6-embedding",
        endpoint_id: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key or os.environ.get("ARK_API_KEY"), **kwargs)
        self.model = model
        # Volcengine ARK requires an endpoint ID (model deployment)
        self.endpoint_id = endpoint_id or os.environ.get("ARK_ENDPOINT_ID", "")

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        """Embed using Volcengine ARK API.

        The ARK API uses an OpenAI-compatible interface.
        For multimodal inputs, it uses a chat-completion-style content format.
        """
        from volcenginesdkarkruntime import Ark

        client = Ark(api_key=self.api_key)
        dim = dimensions or self.default_dimensions

        all_embeddings = []
        total_latency = 0.0
        total_tokens = 0

        for inp in inputs:
            content = self._build_content(inp)

            def _call():
                return client.embeddings.create(
                    model=self.endpoint_id or self.model,
                    input=content,
                    dimensions=dim,
                )

            response, latency = self._timed_call(_call)
            total_latency += latency

            emb = response.data[0].embedding
            all_embeddings.append(emb)

            if response.usage:
                total_tokens += response.usage.total_tokens

        embeddings = np.array(all_embeddings)
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
            token_usage=total_tokens if total_tokens > 0 else None,
        )

    def _build_content(self, inp: EmbeddingInput) -> str | list[dict]:
        """Build input content for the ARK embedding API."""
        if inp.modality == ModalityType.TEXT:
            return inp.content

        if inp.modality == ModalityType.IMAGE:
            image_url = self._to_image_url(inp.content)
            return [
                {"type": "image_url", "image_url": {"url": image_url}},
            ]

        if inp.modality == ModalityType.VIDEO:
            return [
                {"type": "video_url", "video_url": {"url": str(inp.content)}},
            ]

        raise ValueError(f"Unsupported modality for Volcengine: {inp.modality}")

    @staticmethod
    def _to_image_url(content: str | bytes | Path) -> str:
        if isinstance(content, Path) or (isinstance(content, str) and Path(content).exists()):
            path = Path(content)
            data = path.read_bytes()
            suffix = path.suffix.lstrip(".") or "png"
            return f"data:image/{suffix};base64,{base64.b64encode(data).decode()}"
        if isinstance(content, bytes):
            return f"data:image/png;base64,{base64.b64encode(content).decode()}"
        return content
