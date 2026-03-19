"""Jina AI provider — Jina Embeddings v4 (multimodal) / v5-text."""

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


class JinaProvider(EmbeddingProvider):
    """Jina AI embedding API (HTTP-based, no dedicated SDK needed).

    Models:
        - jina-embeddings-v4: Multimodal (text+image+doc), ViDoRe 90.2, CC-BY-NC
        - jina-clip-v2: CLIP-style multimodal (text+image)
        - jina-embeddings-v3: Text-only, multilingual

    Pricing: Pay-per-use via Jina API
    Access: Requires VPN from China mainland. 1M free tokens on signup.
    """

    name = "jina"
    supported_modalities = {ModalityType.TEXT, ModalityType.IMAGE, ModalityType.DOCUMENT}
    max_text_length = 8192  # API limit; native 32K
    default_dimensions = 2048
    supports_mrl = True
    supports_multi_vector = True  # Late Interaction mode

    API_URL = "https://api.jina.ai/v1/embeddings"

    # Task-specific LoRA adapters for v4
    TASK_LORA_MAP = {
        "retrieval_query": "retrieval.query",
        "retrieval_document": "retrieval.passage",
        "similarity": "text-matching",
    }

    # Models that use CLIP-style format (no task/LoRA support)
    CLIP_MODELS = {"jina-clip-v2", "jina-clip-v1"}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jina-embeddings-v4",
        late_interaction: bool = False,
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key or os.environ.get("JINA_API_KEY"), **kwargs)
        self.model = model
        self.late_interaction = late_interaction
        # CLIP models have different default dimensions
        if model in self.CLIP_MODELS:
            self.default_dimensions = 1024
            self.supports_mrl = False
            self.supports_multi_vector = False

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        import httpx

        dim = dimensions or self.default_dimensions

        # Separate text and non-text inputs for batching
        text_indices = [i for i, inp in enumerate(inputs) if inp.modality == ModalityType.TEXT]
        non_text_indices = [i for i, inp in enumerate(inputs) if inp.modality != ModalityType.TEXT]

        all_embeddings: list[tuple[int, list[float]]] = []
        total_latency = 0.0
        total_tokens = 0

        # Batch text inputs (up to 100 per request)
        if text_indices:
            text_batch_size = 100
            for batch_start in range(0, len(text_indices), text_batch_size):
                batch_idx_list = text_indices[batch_start:batch_start + text_batch_size]
                batch_inputs = [self._build_input(inputs[i]) for i in batch_idx_list]

                if len(text_indices) > text_batch_size:
                    batch_num = batch_start // text_batch_size + 1
                    n_batches = (len(text_indices) + text_batch_size - 1) // text_batch_size
                    logger.info("Embedding text batch %d/%d (%d items)...", batch_num, n_batches, len(batch_inputs))

                response, latency = self._send_batch(batch_inputs, dim, task_type, httpx)
                total_latency += latency
                total_tokens += response.get("usage", {}).get("total_tokens", 0)

                # Sort by index to handle out-of-order API responses
                sorted_data = sorted(response["data"], key=lambda x: x.get("index", 0))
                for j, item in enumerate(sorted_data):
                    all_embeddings.append((batch_idx_list[j], item["embedding"]))

                if batch_start + text_batch_size < len(text_indices):
                    time.sleep(1.0)

        # Batch non-text inputs (images/docs, smaller batches due to payload size)
        if non_text_indices:
            img_batch_size = 10
            for batch_start in range(0, len(non_text_indices), img_batch_size):
                batch_idx_list = non_text_indices[batch_start:batch_start + img_batch_size]
                batch_inputs = [self._build_input(inputs[i]) for i in batch_idx_list]

                if len(non_text_indices) > img_batch_size:
                    batch_num = batch_start // img_batch_size + 1
                    n_batches = (len(non_text_indices) + img_batch_size - 1) // img_batch_size
                    logger.info("Embedding image batch %d/%d (%d items)...", batch_num, n_batches, len(batch_inputs))

                response, latency = self._send_batch(batch_inputs, dim, task_type, httpx)
                total_latency += latency
                total_tokens += response.get("usage", {}).get("total_tokens", 0)

                # Sort by index to handle out-of-order API responses
                sorted_data = sorted(response["data"], key=lambda x: x.get("index", 0))
                for j, item in enumerate(sorted_data):
                    all_embeddings.append((batch_idx_list[j], item["embedding"]))

                if batch_start + img_batch_size < len(non_text_indices):
                    time.sleep(2.0)

        # Reassemble in original order
        all_embeddings.sort(key=lambda x: x[0])
        embeddings = np.array([emb for _, emb in all_embeddings])

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
            token_usage=total_tokens if total_tokens else None,
        )

    def _send_batch(
        self,
        jina_inputs: list[dict[str, Any]],
        dim: int,
        task_type: str | None,
        httpx_module: Any,
    ) -> tuple[dict, float]:
        """Send a batch of inputs to Jina API with retry."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body: dict[str, Any] = {
            "model": self.model,
            "input": jina_inputs,
            "dimensions": dim,
        }

        # Set task for LoRA adapter selection (not for CLIP models)
        if self.model not in self.CLIP_MODELS:
            if task_type and task_type in self.TASK_LORA_MAP:
                body["task"] = self.TASK_LORA_MAP[task_type]

        # Late interaction mode (multi-vector)
        if self.late_interaction:
            body["embedding_type"] = "float"
            body["late_chunking"] = True

        def _call():
            resp = httpx_module.post(self.API_URL, json=body, headers=headers, timeout=120.0)
            resp.raise_for_status()
            return resp.json()

        return self._call_with_retry(_call)

    def _build_input(self, inp: EmbeddingInput) -> dict[str, Any]:
        """Build a single input for Jina API."""
        if inp.modality == ModalityType.TEXT:
            return {"text": inp.content}

        if inp.modality in (ModalityType.IMAGE, ModalityType.DOCUMENT):
            image_b64 = self._to_base64(inp.content)
            return {"image": image_b64}

        raise ValueError(f"Unsupported modality for Jina: {inp.modality}")

    @staticmethod
    def _to_base64(content: str | bytes | Path) -> str:
        if isinstance(content, bytes):
            return base64.b64encode(content).decode()
        path = Path(content)
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode()
        return content
