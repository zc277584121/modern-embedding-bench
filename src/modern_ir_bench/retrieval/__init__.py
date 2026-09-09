"""Reusable retrieval components that remain internal to Solutions."""

from modern_ir_bench.retrieval.dense import DenseRetrievalSolution
from modern_ir_bench.retrieval.types import (
    MappedResourceSource,
    RetrievalResource,
    SearchHit,
    SearchSession,
)

__all__ = [
    "DenseRetrievalSolution",
    "MappedResourceSource",
    "RetrievalResource",
    "SearchHit",
    "SearchSession",
]
