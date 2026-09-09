"""Independent NumPy flat search used as a correctness reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from modern_ir_bench.embeddings.protocols import DenseVectors
from modern_ir_bench.retrieval.indexes._validation import (
    as_dense_matrix,
    normalize_dense_metric,
)
from modern_ir_bench.retrieval.types import SearchHit


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


@dataclass(frozen=True)
class NumpyFlatIndex:
    """Exhaustive dense search with deterministic tie-breaking."""

    metric: str = "COSINE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", normalize_dense_metric(self.metric))

    def open(self, *, dimension: int) -> NumpyFlatSession:
        return NumpyFlatSession(dimension=dimension, metric=self.metric)


class NumpyFlatSession:
    def __init__(self, *, dimension: int, metric: str) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.metric = metric
        self._ids: list[str] = []
        self._id_set: set[str] = set()
        self._blocks: list[np.ndarray] = []
        self._vectors: np.ndarray | None = None

    def add(self, ids: Sequence[str], vectors: DenseVectors) -> None:
        if self._vectors is not None:
            raise RuntimeError("Cannot add vectors after the index has been sealed")
        matrix = as_dense_matrix(vectors, dimension=self.dimension)
        normalized_ids = [str(item_id) for item_id in ids]
        if len(normalized_ids) != matrix.shape[0]:
            raise ValueError("ids and vectors must contain the same number of rows")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("ids must be unique within a batch")
        duplicates = self._id_set.intersection(normalized_ids)
        if duplicates:
            raise ValueError(f"Duplicate ids across batches: {sorted(duplicates)}")
        self._ids.extend(normalized_ids)
        self._id_set.update(normalized_ids)
        self._blocks.append(matrix)

    def seal(self) -> None:
        if self._vectors is not None:
            return
        if not self._blocks:
            raise ValueError("Cannot seal an empty dense index")
        vectors = np.concatenate(self._blocks, axis=0)
        self._vectors = _normalize(vectors) if self.metric == "COSINE" else vectors
        self._blocks.clear()

    def search(
        self,
        query_vectors: DenseVectors,
        *,
        top_k: int,
    ) -> list[list[SearchHit]]:
        if self._vectors is None:
            raise RuntimeError("The dense index must be sealed before search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        queries = as_dense_matrix(query_vectors, dimension=self.dimension)
        if self.metric == "COSINE":
            scores = _normalize(queries) @ self._vectors.T
        elif self.metric == "IP":
            scores = queries @ self._vectors.T
        else:
            differences = queries[:, None, :] - self._vectors[None, :, :]
            scores = -np.sum(differences * differences, axis=2)

        limit = min(top_k, len(self._ids))
        output: list[list[SearchHit]] = []
        for row in scores:
            ranked = sorted(
                zip(self._ids, row.tolist(), strict=True),
                key=lambda item: (-item[1], item[0]),
            )[:limit]
            output.append([SearchHit(id=item_id, score=float(score)) for item_id, score in ranked])
        return output

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "numpy-flat",
            "metric": self.metric,
            "dimension": self.dimension,
            "count": len(self._ids),
            "exact": True,
        }

    def close(self) -> None:
        self._ids.clear()
        self._id_set.clear()
        self._blocks.clear()
        self._vectors = None
