"""Provider-neutral contracts for variable-length multi-vector embeddings."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class MultiVectorRole(str, Enum):
    QUERY = "query"
    DOCUMENT = "document"


class MultiVectorRoute(str, Enum):
    QUERY = "query_late_interaction"
    DOCUMENT = "document_late_interaction"


@dataclass(frozen=True)
class MultiVectorRepresentation:
    representation_id: str
    dimensions: int

    def __post_init__(self) -> None:
        if not self.representation_id or self.dimensions <= 0:
            raise ValueError("Multi-vector representation identity and dimensions must be valid")

    @property
    def identity(self) -> str:
        return hashlib.sha256(f"{self.representation_id}:{self.dimensions}".encode()).hexdigest()


@dataclass(frozen=True)
class MultiVectorBatch:
    values: np.ndarray
    mask: np.ndarray
    item_ids: tuple[str, ...]
    passage_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    representation: MultiVectorRepresentation

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.mask)
        if values.ndim != 3 or mask.ndim != 2 or values.shape[:2] != mask.shape:
            raise ValueError("Multi-vector values and mask shapes are incompatible")
        if values.shape[2] != self.representation.dimensions:
            raise ValueError("Multi-vector dimensions do not match the representation")
        if not np.issubdtype(values.dtype, np.floating) or not np.all(np.isfinite(values)):
            raise ValueError("Multi-vector values must be finite floating-point numbers")
        if mask.dtype != np.bool_:
            raise TypeError("Multi-vector mask must use boolean dtype")
        count = values.shape[0]
        if not all(len(items) == count for items in (self.item_ids, self.passage_ids, self.document_ids)):
            raise ValueError("Multi-vector ids must align with batch rows")
        for ids, label in ((self.item_ids, "item"), (self.passage_ids, "passage")):
            if any(not value for value in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"Multi-vector {label} ids must be non-empty and unique")
        if any(not value for value in self.document_ids):
            raise ValueError("Multi-vector document ids must be non-empty")
        if len(set(zip(self.passage_ids, self.document_ids))) != len(self.passage_ids):
            raise ValueError("Multi-vector passage/document identities must be unique")
        if np.any(mask.sum(axis=1) == 0):
            raise ValueError("Multi-vector rows require at least one valid token")
        if np.any(values[~mask] != 0):
            raise ValueError("Multi-vector masked padding must be zero")
        values = values.copy(); mask = mask.copy()
        values.flags.writeable = False; mask.flags.writeable = False
        object.__setattr__(self, "values", values); object.__setattr__(self, "mask", mask)

    @property
    def valid_token_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.mask.sum(axis=1))


@dataclass(frozen=True)
class MultiVectorResult:
    embeddings: MultiVectorBatch
    role: MultiVectorRole
    route: MultiVectorRoute
    provider: str
    model_name: str
    model_revision: str
    latency_ms: float
    peak_vram_bytes: int | None = None

    def __post_init__(self) -> None:
        expected = MultiVectorRoute.QUERY if self.role is MultiVectorRole.QUERY else MultiVectorRoute.DOCUMENT
        if self.route is not expected:
            raise ValueError("Multi-vector role and route are incompatible")
        if not self.provider or not self.model_name or not self.model_revision:
            raise ValueError("Multi-vector provider identity must be complete")
        if not np.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("Multi-vector latency must be finite and non-negative")


@runtime_checkable
class MultiVectorProvider(Protocol):
    name: str
    model: str
    revision: str
    representation: MultiVectorRepresentation
    query_route: MultiVectorRoute
    document_route: MultiVectorRoute

    def encode_multi_vector_query(self, text: str, *, item_id: str) -> MultiVectorResult: ...
    def encode_multi_vector_passages(self, texts: Sequence[str], *, passage_ids: Sequence[str], document_ids: Sequence[str]) -> MultiVectorResult: ...

    def encode_multi_vector_queries(self, texts: Sequence[str], *, item_ids: Sequence[str]) -> MultiVectorResult: ...
