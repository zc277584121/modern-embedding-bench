"""Base class for all embedding providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


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

    def embed_text(self, texts: list[str], **kwargs: Any) -> EmbeddingResult:
        """Convenience: embed a list of text strings."""
        inputs = [EmbeddingInput(modality=ModalityType.TEXT, content=t) for t in texts]
        return self.embed(inputs, **kwargs)

    def embed_images(self, image_paths: list[str | Path], **kwargs: Any) -> EmbeddingResult:
        """Convenience: embed a list of images by file path."""
        inputs = [EmbeddingInput(modality=ModalityType.IMAGE, content=Path(p)) for p in image_paths]
        return self.embed(inputs, **kwargs)

    def _timed_call(self, func, *args, **kwargs) -> tuple[Any, float]:
        """Execute a function and return (result, latency_ms)."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def health_check(self) -> bool:
        """Verify provider connectivity."""
        try:
            result = self.embed_text(["hello world"])
            return result.embeddings.shape[0] == 1
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
