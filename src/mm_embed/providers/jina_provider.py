"""Jina AI provider — Jina Embeddings v4 (multimodal) / v5-text."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType


class JinaProvider(EmbeddingProvider):
    """Jina AI embedding API (HTTP-based, no dedicated SDK needed).

    Models:
        - jina-embeddings-v4: Multimodal (text+image+doc), ViDoRe 90.2, CC-BY-NC
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

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        import httpx

        dim = dimensions or self.default_dimensions
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build request body
        jina_inputs = []
        for inp in inputs:
            jina_inputs.append(self._build_input(inp))

        body: dict[str, Any] = {
            "model": self.model,
            "input": jina_inputs,
            "dimensions": dim,
        }

        # Set task for LoRA adapter selection
        if task_type and task_type in self.TASK_LORA_MAP:
            body["task"] = self.TASK_LORA_MAP[task_type]

        # Late interaction mode (multi-vector)
        if self.late_interaction:
            body["embedding_type"] = "float"
            body["late_chunking"] = True

        def _call():
            resp = httpx.post(self.API_URL, json=body, headers=headers, timeout=60.0)
            resp.raise_for_status()
            return resp.json()

        response, latency = self._timed_call(_call)

        embeddings = np.array([item["embedding"] for item in response["data"]])
        token_usage = response.get("usage", {}).get("total_tokens")

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=latency,
            token_usage=token_usage,
        )

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
