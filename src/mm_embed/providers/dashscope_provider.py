"""DashScope provider — Qwen3-VL-Embedding / Qwen3-Embedding (Alibaba)."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType


class DashScopeProvider(EmbeddingProvider):
    """Alibaba DashScope API for Qwen3 embedding models.

    Models:
        - qwen3-embedding-8b: Text-only, MMTEB Mean(Task) #1 (70.6)
        - qwen3-embedding-4b: Text-only, smaller
        - qwen3-embedding-0.6b: Text-only, lightweight
        - qwen3-vl-embedding-8b: Multimodal (text+image+video), MMEB-V2 #1 (77.8)
        - qwen3-vl-embedding-2b: Multimodal, smaller

    Pricing: ~0.0007 RMB / 1K tokens (text), ~0.0007 RMB / image
    Access: China mainland direct, no VPN needed.
    """

    name = "dashscope"
    supported_modalities = {ModalityType.TEXT, ModalityType.IMAGE, ModalityType.VIDEO}
    max_text_length = 32768
    default_dimensions = 1024
    supports_mrl = True

    # Available models
    TEXT_MODELS = ["qwen3-embedding-8b", "qwen3-embedding-4b", "qwen3-embedding-0.6b"]
    VL_MODELS = ["qwen3-vl-embedding-8b", "qwen3-vl-embedding-2b"]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen3-vl-embedding-8b",
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"), **kwargs)
        self.model = model
        self._is_vl = model in self.VL_MODELS

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        import dashscope
        from dashscope import MultiModalEmbedding, TextEmbedding

        dashscope.api_key = self.api_key

        if self._is_vl:
            return self._embed_multimodal(inputs, dimensions, task_type)
        else:
            return self._embed_text_only(inputs, dimensions, task_type)

    def _embed_text_only(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        from dashscope import TextEmbedding

        texts = [inp.content for inp in inputs]
        dim = dimensions or self.default_dimensions

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "dimension": dim,
        }
        if task_type:
            # DashScope uses "query" / "document" for retrieval tasks
            call_kwargs["text_type"] = task_type

        response, latency = self._timed_call(TextEmbedding.call, **call_kwargs)

        embeddings = np.array([item["embedding"] for item in response.output["embeddings"]])
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=latency,
            token_usage=response.usage.get("total_tokens"),
        )

    def _embed_multimodal(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        """Embed multimodal inputs using Qwen3-VL-Embedding.

        DashScope multimodal embedding API accepts a list of content items,
        each being {"text": ...} or {"image": url/base64} or {"video": url}.
        """
        from dashscope import MultiModalEmbedding

        dim = dimensions or self.default_dimensions

        # Build content list for each input
        all_contents = []
        for inp in inputs:
            if inp.modality == ModalityType.TEXT:
                all_contents.append([{"text": inp.content}])
            elif inp.modality == ModalityType.IMAGE:
                image_data = self._load_image(inp.content)
                all_contents.append([{"image": image_data}])
            elif inp.modality == ModalityType.VIDEO:
                all_contents.append([{"video": str(inp.content)}])
            else:
                raise ValueError(f"Unsupported modality for DashScope VL: {inp.modality}")

        # DashScope MultiModalEmbedding processes one item at a time
        all_embeddings = []
        total_latency = 0.0
        total_tokens = 0

        for content in all_contents:
            call_kwargs: dict[str, Any] = {
                "model": self.model,
                "input": content,
                "dimension": dim,
            }

            response, latency = self._timed_call(MultiModalEmbedding.call, **call_kwargs)
            total_latency += latency

            emb = response.output["embeddings"][0]["embedding"]
            all_embeddings.append(emb)

            if hasattr(response, "usage") and response.usage:
                total_tokens += response.usage.get("total_tokens", 0)

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
    def _load_image(content: str | bytes | Path) -> str:
        """Convert image content to a format DashScope accepts (file:// URL or base64)."""
        if isinstance(content, Path) or (isinstance(content, str) and Path(content).exists()):
            return f"file://{Path(content).resolve()}"
        if isinstance(content, bytes):
            return f"data:image/png;base64,{base64.b64encode(content).decode()}"
        # Assume it's already a URL or base64 string
        return content
