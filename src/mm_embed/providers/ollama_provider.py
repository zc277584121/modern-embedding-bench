"""Ollama provider — local open-source embedding models."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType

logger = logging.getLogger(__name__)


class OllamaProvider(EmbeddingProvider):
    """Ollama local embedding API for open-source models.

    Models (example):
        - nomic-embed-text: 768 dims, text-only
        - mxbai-embed-large: 1024 dims, text-only
        - dengcao/Qwen3-Embedding-8B:Q5_K_M: 4096 dims, multimodal (text+image)

    Pricing: Free (local GPU).
    """

    name = "ollama"
    max_text_length = 8192
    supports_mrl = True  # via truncation

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        # Detect dimensions and modalities on first use
        self._detected_dims: int | None = None
        self._supports_images: bool | None = None

    @property
    def default_dimensions(self) -> int:
        if self._detected_dims is None:
            self._detect_capabilities()
        return self._detected_dims  # type: ignore[return-value]

    @property
    def supported_modalities(self) -> set[ModalityType]:
        if self._supports_images is None:
            self._detect_capabilities()
        mods = {ModalityType.TEXT}
        if self._supports_images:
            mods.add(ModalityType.IMAGE)
        return mods

    def _detect_capabilities(self) -> None:
        """Probe the model to detect native dimensions and image support."""
        with httpx.Client(base_url=self.base_url, timeout=30) as client:
            resp = client.post("/api/embed", json={
                "model": self.model,
                "input": "hello",
            })
            resp.raise_for_status()
            data = resp.json()
            self._detected_dims = len(data["embeddings"][0])
            logger.info("Ollama %s: detected %d dimensions", self.model, self._detected_dims)

        # Image support detection: try with a tiny 1x1 PNG
        tiny_png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode()
        try:
            with httpx.Client(base_url=self.base_url, timeout=30) as client:
                resp = client.post("/api/embed", json={
                    "model": self.model,
                    "input": "test",
                    "images": [tiny_png],
                })
                self._supports_images = resp.status_code == 200 and "embeddings" in resp.json()
        except Exception:
            self._supports_images = False

        logger.info("Ollama %s: image support=%s", self.model, self._supports_images)

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        # Ensure capabilities are detected
        if self._detected_dims is None:
            self._detect_capabilities()

        # Separate text-only inputs from image inputs
        has_images = any(inp.modality == ModalityType.IMAGE for inp in inputs)

        if has_images:
            return self._embed_with_images(inputs, dimensions)
        else:
            return self._embed_text_batch(inputs, dimensions)

    def _embed_text_batch(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        """Embed text inputs in batches via Ollama API."""
        texts = [inp.content for inp in inputs]

        # Ollama handles batching internally, but large batches may OOM
        batch_size = 50
        all_embeddings = []
        total_latency = 0.0
        n_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]

            if n_batches > 1:
                logger.info("Embedding text batch %d/%d (%d items)...", batch_idx + 1, n_batches, len(batch))

            payload = {"model": self.model, "input": batch}

            def _call(p=payload):
                with httpx.Client(base_url=self.base_url, timeout=300) as client:
                    resp = client.post("/api/embed", json=p)
                    resp.raise_for_status()
                    return resp.json()

            data, latency = self._call_with_retry(_call)
            total_latency += latency
            all_embeddings.extend(data["embeddings"])

        embeddings = np.array(all_embeddings)

        # MRL: truncate to requested dimensions
        dim = dimensions or self._detected_dims
        if dim and dim < embeddings.shape[1]:
            embeddings = embeddings[:, :dim]
            # Re-normalize after truncation to maintain unit length
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=embeddings.shape[1],
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
        )

    def _embed_with_images(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        """Embed inputs that may include images (one at a time for image inputs)."""
        all_embeddings = []
        total_latency = 0.0

        # Collect consecutive text-only inputs for batching
        i = 0
        while i < len(inputs):
            inp = inputs[i]

            if inp.modality == ModalityType.IMAGE:
                # Image: embed one at a time with images field
                image_b64 = self._load_image_b64(inp.content)
                payload = {
                    "model": self.model,
                    "input": " ",  # placeholder text
                    "images": [image_b64],
                }

                if len(inputs) > 10 and (i + 1) % 10 == 0:
                    logger.info("Embedding image %d/%d...", i + 1, len(inputs))

                def _call(p=payload):
                    with httpx.Client(base_url=self.base_url, timeout=300) as client:
                        resp = client.post("/api/embed", json=p)
                        resp.raise_for_status()
                        return resp.json()

                data, latency = self._call_with_retry(_call)
                total_latency += latency
                all_embeddings.append(data["embeddings"][0])
                i += 1
            else:
                # Collect consecutive text inputs for batching
                text_batch = []
                while i < len(inputs) and inputs[i].modality == ModalityType.TEXT:
                    text_batch.append(inputs[i].content)
                    i += 1

                payload = {"model": self.model, "input": text_batch}

                def _call(p=payload):
                    with httpx.Client(base_url=self.base_url, timeout=300) as client:
                        resp = client.post("/api/embed", json=p)
                        resp.raise_for_status()
                        return resp.json()

                data, latency = self._call_with_retry(_call)
                total_latency += latency
                all_embeddings.extend(data["embeddings"])

        embeddings = np.array(all_embeddings)

        dim = dimensions or self._detected_dims
        if dim and dim < embeddings.shape[1]:
            embeddings = embeddings[:, :dim]
            # Re-normalize after truncation to maintain unit length
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=embeddings.shape[1],
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
        )

    @staticmethod
    def _load_image_b64(content: str | bytes | Path) -> str:
        """Convert image content to base64 string for Ollama."""
        if isinstance(content, bytes):
            return base64.b64encode(content).decode()
        path = Path(content)
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode()
        # Assume already base64
        return content if isinstance(content, str) else content.decode()
