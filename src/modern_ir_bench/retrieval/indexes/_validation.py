"""Shared validation for dense index backends."""

from __future__ import annotations

import numpy as np

from modern_ir_bench.embeddings.protocols import DenseVectors

SUPPORTED_DENSE_METRICS = frozenset({"COSINE", "IP", "L2"})


def normalize_dense_metric(metric: str) -> str:
    normalized = metric.upper()
    if normalized not in SUPPORTED_DENSE_METRICS:
        raise ValueError(f"Unsupported dense metric: {metric}")
    return normalized


def as_dense_matrix(vectors: DenseVectors, *, dimension: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"Expected a two-dimensional vector batch with dimension {dimension}")
    if not np.isfinite(matrix).all():
        raise ValueError("Vectors must contain only finite values")
    return matrix
