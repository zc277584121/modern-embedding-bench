"""GeeVec embedding providers: OpenAPI and local Lite model."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType

logger = logging.getLogger(__name__)


class GeeVecAPIProvider(EmbeddingProvider):
    """GeeVec OpenAPI embedding provider."""

    name = "geevec_api"
    supported_modalities = {ModalityType.TEXT}
    max_text_length = 32768
    default_dimensions = 4096
    supports_mrl = True
    default_batch_size = 4

    API_URL = "https://www.geevec.com/openapi/v1/embeddings"
    DOMAIN_MODELS = {
        "general": "geevec-embeddings-general-1.0",
        "coding": "geevec-embeddings-coding-1.0",
        "reasoning": "geevec-embeddings-reasoning-1.0",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        domain: str = "general",
        batch_size: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 4,
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key or os.environ.get("GEE_API_KEY"), **kwargs)
        if domain not in self.DOMAIN_MODELS:
            raise ValueError(f"Unsupported GeeVec domain: {domain}")
        self.domain = domain
        self.model = model or self.DOMAIN_MODELS[domain]
        self.batch_size = batch_size or self.default_batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        import httpx

        if not self.api_key:
            raise ValueError("GEE_API_KEY is not set")

        texts: list[str] = []
        for inp in inputs:
            if inp.modality != ModalityType.TEXT:
                raise ValueError(f"GeeVec API only supports text embeddings, got {inp.modality}")
            texts.append(str(inp.content))

        all_embeddings: list[list[float]] = []
        total_latency = 0.0
        total_tokens = 0
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        with httpx.Client(timeout=self.timeout) as client:
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                batch = texts[start : start + self.batch_size]

                if n_batches > 1:
                    logger.info(
                        "Embedding GeeVec API batch %d/%d (%d items)...",
                        batch_idx + 1,
                        n_batches,
                        len(batch),
                    )

                response, latency = self._send_batch(client, batch, dimensions)
                total_latency += latency
                total_tokens += int(response.get("usage", {}).get("total_tokens", 0) or 0)

                rows = sorted(response["data"], key=lambda item: item.get("index", 0))
                all_embeddings.extend(row["embedding"] for row in rows)

                if batch_idx < n_batches - 1:
                    time.sleep(1.0)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        actual_dim = embeddings.shape[1] if embeddings.size else (dimensions or self.default_dimensions)
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=actual_dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
            token_usage=total_tokens or None,
            metadata={"domain": self.domain},
        )

    def embed_with_cache(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        """Embed with per-input cache entries to make long API runs resumable."""
        from mm_embed.cache import cache_get, cache_put, make_cache_key

        ordered_keys: list[str] = []
        ordered_embeddings: list[np.ndarray | None] = []
        misses: dict[str, EmbeddingInput] = {}
        cache_hits = 0

        for inp in inputs:
            if inp.modality != ModalityType.TEXT:
                raise ValueError(f"GeeVec API only supports text embeddings, got {inp.modality}")

            cache_key = make_cache_key(
                model_name=self.model,
                inputs_content=[inp.content],
                modalities=[inp.modality.value],
                dimensions=dimensions,
                task_type=task_type,
            )
            ordered_keys.append(cache_key)

            cached = cache_get(self.name, self.model, cache_key)
            if cached is not None:
                ordered_embeddings.append(self._single_embedding(cached))
                cache_hits += 1
            else:
                ordered_embeddings.append(None)
                misses.setdefault(cache_key, inp)

        total_latency = 0.0
        total_tokens = 0
        fetched: dict[str, np.ndarray] = {}

        if misses:
            if not self.api_key:
                raise ValueError("GEE_API_KEY is not set")

            import httpx

            miss_items = list(misses.items())
            with httpx.Client(timeout=self.timeout) as client:
                n_batches = (len(miss_items) + self.batch_size - 1) // self.batch_size
                for batch_idx in range(n_batches):
                    start = batch_idx * self.batch_size
                    batch_items = miss_items[start : start + self.batch_size]
                    batch_keys = [key for key, _ in batch_items]
                    batch_texts = [str(inp.content) for _, inp in batch_items]

                    if n_batches > 1:
                        logger.info(
                            "Embedding GeeVec API cache-miss batch %d/%d (%d items)...",
                            batch_idx + 1,
                            n_batches,
                            len(batch_texts),
                        )

                    response, latency = self._send_batch(client, batch_texts, dimensions)
                    total_latency += latency
                    total_tokens += int(response.get("usage", {}).get("total_tokens", 0) or 0)

                    rows = sorted(response["data"], key=lambda item: item.get("index", 0))
                    for key, row in zip(batch_keys, rows):
                        embedding = np.array(row["embedding"], dtype=np.float32)
                        fetched[key] = embedding
                        cache_put(self.name, self.model, key, embedding.reshape(1, -1))

                    if batch_idx < n_batches - 1:
                        time.sleep(1.0)

        final_embeddings: list[np.ndarray] = []
        for key, cached_embedding in zip(ordered_keys, ordered_embeddings):
            embedding = cached_embedding if cached_embedding is not None else fetched[key]
            final_embeddings.append(embedding)

        embeddings = np.vstack(final_embeddings).astype(np.float32) if final_embeddings else np.empty((0, 0))
        actual_dim = embeddings.shape[1] if embeddings.size else (dimensions or self.default_dimensions)
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=actual_dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
            token_usage=total_tokens or None,
            metadata={
                "domain": self.domain,
                "cache_hits": cache_hits,
                "cache_misses": len(misses),
            },
        )

    def _send_batch(
        self,
        client: Any,
        texts: list[str],
        dimensions: int | None,
    ) -> tuple[dict[str, Any], float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if dimensions is not None and dimensions != self.default_dimensions:
            body["dimensions"] = dimensions

        def _call() -> dict[str, Any]:
            response = client.post(self.API_URL, headers=headers, json=body)
            response.raise_for_status()
            return response.json()

        return self._call_with_retry(_call, max_retries=self.max_retries)

    @staticmethod
    def _single_embedding(cached: np.ndarray) -> np.ndarray:
        if cached.ndim == 2 and cached.shape[0] == 1:
            return cached[0].astype(np.float32)
        if cached.ndim == 1:
            return cached.astype(np.float32)
        raise ValueError(f"Unexpected GeeVec API single-cache shape: {cached.shape}")


class GeeVecLiteProvider(EmbeddingProvider):
    """Local GeeVec Lite provider loaded through sentence-transformers."""

    name = "geevec_lite"
    supported_modalities = {ModalityType.TEXT}
    max_text_length = 32768
    default_dimensions = 4096
    supports_mrl = True
    default_batch_size = 8

    MODEL_ID = "geevec-ai/geevec-embeddings-1.0-lite"
    DOMAINS = {"general", "coding", "reasoning"}

    def __init__(
        self,
        model: str = MODEL_ID,
        domain: str = "general",
        device: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if domain not in self.DOMAINS:
            raise ValueError(f"Unsupported GeeVec Lite domain: {domain}")
        self.model_id = model
        self.model = f"{model}::{domain}"
        self.domain = domain
        self.device = device or os.environ.get("CUDA_DEVICE")
        self.batch_size = batch_size or self.default_batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model: Any | None = None

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        texts: list[str] = []
        for inp in inputs:
            if inp.modality != ModalityType.TEXT:
                raise ValueError(f"GeeVec Lite only supports text embeddings, got {inp.modality}")
            texts.append(str(inp.content))

        model = self._load_model()
        encode_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "show_progress_bar": False,
        }
        if self.domain != "general":
            encode_kwargs["domain"] = self.domain

        start = time.perf_counter()
        embeddings = np.array(model.encode(texts, **encode_kwargs), dtype=np.float32)
        latency_ms = (time.perf_counter() - start) * 1000

        if dimensions is not None and dimensions < embeddings.shape[1]:
            embeddings = embeddings[:, :dimensions]

        actual_dim = embeddings.shape[1] if embeddings.size else (dimensions or self.default_dimensions)
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=actual_dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            metadata={"domain": self.domain},
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        self._patch_transformers_tied_weights()

        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.device:
            kwargs["device"] = self.device

        logger.info("Loading GeeVec Lite model %s...", self.model_id)
        self._model = SentenceTransformer(self.model_id, **kwargs)
        return self._model

    @staticmethod
    def _patch_transformers_tied_weights() -> None:
        from transformers import modeling_utils

        model_cls = modeling_utils.PreTrainedModel
        if getattr(model_cls, "_geevec_tied_weights_patch", False):
            return

        original_expand = model_cls.get_expanded_tied_weights_keys
        original_mark = model_cls.mark_tied_weights_as_initialized

        def patched_expand(self_model: Any, all_submodels: bool = False) -> Any:
            tied_mapping = getattr(self_model, "_tied_weights_keys", None)
            if isinstance(tied_mapping, list):
                return {}
            return original_expand(self_model, all_submodels=all_submodels)

        def patched_mark(self_model: Any, loading_info: Any) -> Any:
            if not hasattr(self_model, "all_tied_weights_keys"):
                self_model.all_tied_weights_keys = {}
            return original_mark(self_model, loading_info)

        model_cls.get_expanded_tied_weights_keys = patched_expand
        model_cls.mark_tied_weights_as_initialized = patched_mark
        model_cls._geevec_tied_weights_patch = True
