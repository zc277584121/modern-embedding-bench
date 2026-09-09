"""Sentence Transformers dense embedding component."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class SentenceTransformersEmbedding:
    """Use one fixed Sentence Transformers model for documents and queries."""

    def __init__(
        self,
        *,
        model: str,
        revision: str | None = None,
        batch_size: int = 128,
        device: str | None = None,
        normalize: bool = True,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        from sentence_transformers import SentenceTransformer

        self.model_id = model
        self.revision = revision
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = SentenceTransformer(
            model,
            revision=revision,
            device=device,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        get_dimension = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        dimension = get_dimension()
        if dimension is None:
            raise ValueError(f"Could not determine embedding dimension for {model}")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, inputs: Sequence[Any]) -> np.ndarray:
        texts = list(inputs)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("SentenceTransformersEmbedding only accepts strings")
        return np.asarray(
            self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_documents(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs)

    def encode_queries(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs)
