"""DashScope provider — Qwen3-VL-Embedding / Qwen3-Embedding (Alibaba)."""

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


class DashScopeProvider(EmbeddingProvider):
    """Alibaba DashScope API for embedding models.

    Models:
        - text-embedding-v4: Text-only, Qwen3-Embedding series
        - text-embedding-v3: Text-only, previous generation
        - qwen3-vl-embedding: Multimodal fused embeddings (text+image+video)
        - multimodal-embedding-v1: Previous-generation multimodal embeddings

    Pricing: ~0.0007 RMB / 1K tokens (text), ~0.0007 RMB / image
    Access: China mainland direct, no VPN needed.
    """

    name = "dashscope"
    supported_modalities = {ModalityType.TEXT, ModalityType.IMAGE, ModalityType.VIDEO}
    max_text_length = 32768
    default_dimensions = 1024
    supports_mrl = True

    # Available models
    TEXT_MODELS = ["text-embedding-v4", "text-embedding-v3", "text-embedding-v2", "text-embedding-v1"]
    VL_MODELS = ["qwen3-vl-embedding", "qwen2.5-vl-embedding", "multimodal-embedding-v1"]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-v4",
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

        # DashScope text-embedding-v3 supports batch up to 10 texts
        batch_size = 10
        all_embeddings = []
        total_latency = 0.0
        total_tokens = 0
        n_batches = (len(texts) + batch_size - 1) // batch_size

        # Map task_type to DashScope text_type values ('query' or 'document')
        ds_text_type = None
        if task_type:
            text_type_map = {
                "retrieval_query": "query",
                "retrieval_document": "document",
                "query": "query",
                "document": "document",
            }
            ds_text_type = text_type_map.get(task_type)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]

            if n_batches > 1:
                logger.info("Embedding text batch %d/%d (%d items)...", batch_idx + 1, n_batches, len(batch))

            call_kwargs: dict[str, Any] = {
                "model": self.model,
                "input": batch,
                "dimension": dim,
            }
            if ds_text_type:
                call_kwargs["text_type"] = ds_text_type

            def _call(kw=call_kwargs):
                return TextEmbedding.call(**kw)

            response, latency = self._call_with_retry(_call)
            total_latency += latency

            for item in response.output["embeddings"]:
                all_embeddings.append(item["embedding"])

            if response.usage:
                total_tokens += response.usage.get("total_tokens", 0)

            if batch_idx < n_batches - 1:
                time.sleep(0.5)

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

        for i, content in enumerate(all_contents):
            if len(all_contents) > 10 and (i + 1) % 10 == 0:
                logger.info("Embedding multimodal item %d/%d...", i + 1, len(all_contents))

            call_kwargs: dict[str, Any] = {
                "model": self.model,
                "input": content,
            }
            # multimodal-embedding-v1 does NOT support the 'dimension' parameter
            if self.model != "multimodal-embedding-v1" and dim:
                call_kwargs["dimension"] = dim

            def _call(kw=call_kwargs):
                return MultiModalEmbedding.call(**kw)

            response, latency = self._call_with_retry(_call)
            total_latency += latency

            # Handle None output (often caused by 429 rate limiting)
            retry_wait = 10.0
            for retry_attempt in range(5):
                if response.output is not None:
                    break
                status = getattr(response, "status_code", "?")
                msg = getattr(response, "message", "?")
                logger.warning(
                    "Multimodal item %d/%d returned None output (status=%s, msg=%s), "
                    "retrying after %.0fs (attempt %d/5)...",
                    i + 1, len(all_contents), status, msg, retry_wait, retry_attempt + 1,
                )
                time.sleep(retry_wait)
                retry_wait = min(retry_wait * 2, 120)
                response, latency2 = self._call_with_retry(_call)
                total_latency += latency2

            if response.output is None:
                status = getattr(response, "status_code", "?")
                msg = getattr(response, "message", "?")
                raise RuntimeError(
                    f"DashScope multimodal item {i+1}/{len(all_contents)} returned None output "
                    f"after 5 retries (status={status}, msg={msg})"
                )

            emb = response.output["embeddings"][0]["embedding"]
            all_embeddings.append(emb)

            if hasattr(response, "usage") and response.usage:
                total_tokens += response.usage.get("total_tokens", 0)

            # Delay between items to avoid rate limiting
            if i < len(all_contents) - 1:
                time.sleep(0.3)

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
