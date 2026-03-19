"""Base class for all embedding providers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"  # PDF / screenshot


@dataclass
class EmbeddingInput:
    """A single input item for embedding."""

    modality: ModalityType
    content: str | bytes | Path  # text string, base64 bytes, or file path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of an embedding call."""

    embeddings: np.ndarray  # shape: (n_items, dim) or (n_items, n_tokens, dim) for multi-vector
    dimensions: int
    model_name: str
    provider: str
    latency_ms: float
    token_usage: int | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding model providers."""

    name: str = "base"
    supported_modalities: set[ModalityType] = set()
    max_text_length: int = 8192
    default_dimensions: int = 1024
    supports_mrl: bool = False  # Matryoshka Representation Learning
    supports_multi_vector: bool = False  # Late interaction / ColBERT style
    max_requests_per_minute: int = 100
    default_batch_size: int = 100

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        """Embed a batch of inputs.

        Args:
            inputs: List of items to embed (text, images, etc.)
            dimensions: Output embedding dimensions (for MRL models)
            task_type: Task hint (e.g. "retrieval_query", "retrieval_document", "similarity")

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        ...

    def embed_with_cache(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        """Embed with disk cache. Falls back to self.embed() on miss."""
        from mm_embed.cache import cache_get, cache_put, make_cache_key

        model_name = getattr(self, "model", self.name)
        cache_key = make_cache_key(
            model_name=model_name,
            inputs_content=[inp.content for inp in inputs],
            modalities=[inp.modality.value for inp in inputs],
            dimensions=dimensions,
            task_type=task_type,
        )

        cached = cache_get(self.name, model_name, cache_key)
        if cached is not None:
            return EmbeddingResult(
                embeddings=cached,
                dimensions=cached.shape[1] if cached.ndim == 2 else 0,
                model_name=model_name,
                provider=self.name,
                latency_ms=0.0,
                metadata={"cache_hit": True},
            )

        result = self.embed(inputs, dimensions=dimensions, task_type=task_type)
        cache_put(self.name, model_name, cache_key, result.embeddings)
        return result

    def embed_text(self, texts: list[str], **kwargs: Any) -> EmbeddingResult:
        """Convenience: embed a list of text strings (with cache)."""
        inputs = [EmbeddingInput(modality=ModalityType.TEXT, content=t) for t in texts]
        return self.embed_with_cache(inputs, **kwargs)

    def embed_images(self, image_paths: list[str | Path], **kwargs: Any) -> EmbeddingResult:
        """Convenience: embed a list of images by file path (with cache)."""
        inputs = [EmbeddingInput(modality=ModalityType.IMAGE, content=Path(p)) for p in image_paths]
        return self.embed_with_cache(inputs, **kwargs)

    def _timed_call(self, func, *args, **kwargs) -> tuple[Any, float]:
        """Execute a function and return (result, latency_ms)."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def _call_with_retry(self, func, max_retries: int = 8) -> tuple[Any, float]:
        """Execute with exponential backoff on rate limit (429) and server (5xx) errors."""
        import re

        for attempt in range(max_retries + 1):
            try:
                return self._timed_call(func)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower()
                is_server_error = "500" in err_str or "502" in err_str or "503" in err_str or "UNAVAILABLE" in err_str or "INTERNAL" in err_str
                is_retryable = is_rate_limit or is_server_error
                if not is_retryable or attempt == max_retries:
                    raise
                # Try to parse retry delay from error message
                match = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
                if match:
                    wait = float(match.group(1)) + 5.0
                elif is_server_error:
                    wait = min(2 ** attempt * 5, 60)
                else:
                    wait = min(2 ** attempt * 10, 120)
                label = "Server error" if is_server_error else "Rate limited"
                logger.warning("%s (attempt %d/%d), waiting %.0fs...", label, attempt + 1, max_retries, wait)
                time.sleep(wait)
        raise RuntimeError("Unreachable")

    def health_check(self) -> bool:
        """Verify provider connectivity."""
        try:
            result = self.embed_text(["hello world"])
            return result.embeddings.shape[0] == 1
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
