"""Provider-neutral contracts for explicit sparse CSR embeddings."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import sparse


class SparseEncodingRoute(str, Enum):
    """How a provider produces one side of a sparse retrieval representation."""

    NEURAL = "neural"
    STATIC_LOOKUP = "static_lookup"
    TOKENIZER_IDF = "tokenizer_idf"
    NONE = "none"


class SparseEmbeddingRole(str, Enum):
    """Whether a sparse batch represents queries or documents."""

    QUERY = "query"
    DOCUMENT = "document"


@dataclass(frozen=True)
class SparseRepresentation:
    """Identity of a sparse coordinate space and its vocabulary."""

    representation_id: str
    vocabulary_id: str
    dimensions: int

    def __post_init__(self) -> None:
        if not self.representation_id:
            raise ValueError("Sparse representation_id must not be empty")
        if not self.vocabulary_id:
            raise ValueError("Sparse vocabulary_id must not be empty")
        if self.dimensions <= 0:
            raise ValueError("Sparse representation dimensions must be positive")

    @property
    def identity(self) -> str:
        """Return a compact stable identity for compatibility checks and logs."""
        payload = {
            "dimensions": self.dimensions,
            "representation_id": self.representation_id,
            "vocabulary_id": self.vocabulary_id,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class SparseEmbeddingBatch:
    """An immutable, row-aligned sparse embedding batch backed by CSR."""

    _data_bytes: bytes = field(repr=False, compare=False)
    _indices_bytes: bytes = field(repr=False, compare=False)
    _indptr_bytes: bytes = field(repr=False, compare=False)
    _data_dtype: str = field(repr=False)
    _shape: tuple[int, int] = field(repr=False)
    item_ids: tuple[str, ...]
    representation: SparseRepresentation
    fingerprint: str = field(init=False)

    def __init__(
        self,
        values: sparse.csr_matrix,
        item_ids: Sequence[str],
        representation: SparseRepresentation,
    ) -> None:
        if not sparse.isspmatrix_csr(values):
            raise TypeError("Sparse embedding values must be a scipy.sparse.csr_matrix")

        normalized_item_ids = tuple(item_ids)
        if any(not isinstance(item_id, str) or not item_id for item_id in normalized_item_ids):
            raise ValueError("Sparse item ids must be non-empty strings")
        if len(set(normalized_item_ids)) != len(normalized_item_ids):
            raise ValueError("Sparse item ids must be unique")

        normalized_values = values.copy()
        normalized_values.sum_duplicates()
        normalized_values.sort_indices()
        normalized_values.eliminate_zeros()
        if normalized_values.shape[0] != len(normalized_item_ids):
            raise ValueError("Sparse item id count must match CSR row count")
        if normalized_values.shape[1] != representation.dimensions:
            raise ValueError("Sparse CSR dimensions do not match the representation")
        if not np.issubdtype(normalized_values.dtype, np.floating):
            raise TypeError("Sparse embedding values must use a floating-point dtype")
        if not np.all(np.isfinite(normalized_values.data)):
            raise ValueError("Sparse embedding values must be finite")

        indices = np.asarray(normalized_values.indices, dtype=np.int64)
        indptr = np.asarray(normalized_values.indptr, dtype=np.int64)
        object.__setattr__(self, "_data_bytes", normalized_values.data.tobytes(order="C"))
        object.__setattr__(self, "_indices_bytes", indices.tobytes(order="C"))
        object.__setattr__(self, "_indptr_bytes", indptr.tobytes(order="C"))
        object.__setattr__(self, "_data_dtype", normalized_values.data.dtype.str)
        object.__setattr__(self, "_shape", normalized_values.shape)
        object.__setattr__(self, "item_ids", normalized_item_ids)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(
            self,
            "fingerprint",
            _batch_fingerprint(normalized_values, normalized_item_ids, representation),
        )

    @property
    def values(self) -> sparse.csr_matrix:
        """Return a detached read-only CSR snapshot of the immutable batch."""
        data = np.frombuffer(self._data_bytes, dtype=np.dtype(self._data_dtype)).copy()
        indices = np.frombuffer(self._indices_bytes, dtype=np.int64).copy()
        indptr = np.frombuffer(self._indptr_bytes, dtype=np.int64).copy()
        values = sparse.csr_matrix((data, indices, indptr), shape=self._shape, copy=False)
        values.data.flags.writeable = False
        values.indices.flags.writeable = False
        values.indptr.flags.writeable = False
        return values

    @property
    def dimensions(self) -> int:
        return self.representation.dimensions

    @property
    def dtype(self) -> str:
        return np.dtype(self._data_dtype).name

    @property
    def nnz_total(self) -> int:
        return len(self._data_bytes) // np.dtype(self._data_dtype).itemsize

    @property
    def nnz_per_row(self) -> tuple[int, ...]:
        indptr = np.frombuffer(self._indptr_bytes, dtype=np.int64)
        return tuple(int(value) for value in np.diff(indptr))


@dataclass(frozen=True)
class SparseEmbeddingResult:
    """Auditable sparse provider output kept separate from dense results."""

    embeddings: SparseEmbeddingBatch
    role: SparseEmbeddingRole
    model_name: str
    provider: str
    model_revision: str
    query_route: SparseEncodingRoute
    document_route: SparseEncodingRoute
    latency_ms: float
    token_usage: int | None = None
    device: str | None = None
    peak_vram_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, SparseEmbeddingRole):
            raise ValueError("Sparse result role must be a SparseEmbeddingRole")
        if not isinstance(self.query_route, SparseEncodingRoute):
            raise ValueError("Sparse query_route must be a SparseEncodingRoute")
        if not isinstance(self.document_route, SparseEncodingRoute):
            raise ValueError("Sparse document_route must be a SparseEncodingRoute")
        if self.role is SparseEmbeddingRole.QUERY and self.query_route is SparseEncodingRoute.NONE:
            raise ValueError("A sparse query result requires a query encoding route")
        if self.role is SparseEmbeddingRole.DOCUMENT and self.document_route is SparseEncodingRoute.NONE:
            raise ValueError("A sparse document result requires a document encoding route")
        for field_name, value in (
            ("model_name", self.model_name),
            ("provider", self.provider),
            ("model_revision", self.model_revision),
        ):
            if not value:
                raise ValueError(f"Sparse result {field_name} must not be empty")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("Sparse result latency_ms must be finite and non-negative")
        if self.token_usage is not None and self.token_usage < 0:
            raise ValueError("Sparse result token_usage must be non-negative")
        if self.peak_vram_bytes is not None and self.peak_vram_bytes < 0:
            raise ValueError("Sparse result peak_vram_bytes must be non-negative")

        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("Sparse metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "fingerprint", _result_fingerprint(self, metadata))

    def metadata_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of result metadata."""
        metadata = _json_ready(self.metadata)
        if not isinstance(metadata, dict):
            raise RuntimeError("Frozen sparse metadata did not decode to an object")
        return metadata


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    """Minimal provider capability for distinct sparse query/document routes."""

    name: str
    model: str
    revision: str
    representation: SparseRepresentation
    query_route: SparseEncodingRoute
    document_route: SparseEncodingRoute

    def encode_sparse_query(self, text: str, *, item_id: str) -> SparseEmbeddingResult:
        """Encode one query without using the dense provider path."""
        ...

    def encode_sparse_documents(
        self,
        texts: Sequence[str],
        *,
        item_ids: Sequence[str],
    ) -> SparseEmbeddingResult:
        """Encode ordered documents without using the dense provider path."""
        ...


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("Sparse metadata must be JSON-compatible and finite") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Sparse metadata object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Sparse metadata numbers must be finite")
        return value
    raise TypeError("Sparse metadata must contain only JSON-compatible values")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _update_array_hash(digest: Any, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    header = {"dtype": contiguous.dtype.str, "shape": contiguous.shape}
    digest.update(_canonical_json(header).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))


def _batch_fingerprint(
    values: sparse.csr_matrix,
    item_ids: tuple[str, ...],
    representation: SparseRepresentation,
) -> str:
    digest = hashlib.sha256()
    header = {
        "dtype": values.dtype.name,
        "item_ids": item_ids,
        "representation": {
            "dimensions": representation.dimensions,
            "representation_id": representation.representation_id,
            "vocabulary_id": representation.vocabulary_id,
        },
        "shape": values.shape,
    }
    digest.update(_canonical_json(header).encode("utf-8"))
    _update_array_hash(digest, np.asarray(values.indptr, dtype=np.int64))
    _update_array_hash(digest, np.asarray(values.indices, dtype=np.int64))
    _update_array_hash(digest, values.data)
    return digest.hexdigest()


def _result_fingerprint(result: SparseEmbeddingResult, metadata: Mapping[str, Any]) -> str:
    payload = {
        "batch_fingerprint": result.embeddings.fingerprint,
        "device": result.device,
        "document_route": result.document_route.value,
        "latency_ms": result.latency_ms,
        "metadata": metadata,
        "model_name": result.model_name,
        "model_revision": result.model_revision,
        "peak_vram_bytes": result.peak_vram_bytes,
        "provider": result.provider,
        "query_route": result.query_route.value,
        "role": result.role.value,
        "token_usage": result.token_usage,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "SparseEmbeddingBatch",
    "SparseEmbeddingProvider",
    "SparseEmbeddingResult",
    "SparseEmbeddingRole",
    "SparseEncodingRoute",
    "SparseRepresentation",
]
