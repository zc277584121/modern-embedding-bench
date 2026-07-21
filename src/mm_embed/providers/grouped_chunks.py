"""Explicit provider-neutral contract for ordered grouped chunk embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


MAPPING_STATUSES = {"exact", "normalized_exact", "span_mapped", "unmapped"}


@dataclass(frozen=True)
class GroupedChunkInput:
    """Canonical benchmark-owned chunk supplied to a grouped provider."""

    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    text_sha256: str


@dataclass(frozen=True)
class GroupedChunkGroup:
    """One ordered parent-document group."""

    document_id: str
    chunks: tuple[GroupedChunkInput, ...]


@dataclass(frozen=True)
class GroupedChunkEmbeddingRequest:
    """Grouped request whose identity includes boundaries and strategy."""

    layout_id: str
    chunker_version: str
    strategy: str
    groups: tuple[GroupedChunkGroup, ...]
    task_type: str = "retrieval_document"


@dataclass(frozen=True)
class GroupedChunkEmbedding:
    """One returned embedding with canonical and provider identity."""

    document_id: str
    chunk_id: str
    chunk_index: int
    char_start: int
    char_end: int
    text_sha256: str
    returned_text: str
    mapping_status: str
    embedding: np.ndarray
    provider_chunk_id: str | None = None


@dataclass(frozen=True)
class GroupedChunkEmbeddingGroup:
    """Returned embeddings for one ordered parent-document group."""

    document_id: str
    chunks: tuple[GroupedChunkEmbedding, ...]


@dataclass(frozen=True)
class GroupedChunkEmbeddingResult:
    """Auditable grouped result kept separate from the flat provider result."""

    layout_id: str
    strategy: str
    groups: tuple[GroupedChunkEmbeddingGroup, ...]
    dimensions: int
    model_name: str
    provider: str
    latency_ms: float
    token_usage: int | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] | None = None


@runtime_checkable
class GroupedChunkEmbeddingProvider(Protocol):
    """Small explicit capability for context-preserving chunk embedding."""

    def embed_grouped_chunks(self, request: GroupedChunkEmbeddingRequest) -> GroupedChunkEmbeddingResult:
        """Embed ordered chunks without crossing parent-document boundaries."""
        ...


def grouped_request_fingerprint(request: GroupedChunkEmbeddingRequest) -> str:
    """Hash ordered group boundaries, identities, text, and strategy for caching."""
    payload = {
        "layout_id": request.layout_id,
        "chunker_version": request.chunker_version,
        "strategy": request.strategy,
        "task_type": request.task_type,
        "groups": [
            {
                "document_id": group.document_id,
                "chunks": [
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "text_sha256": chunk.text_sha256,
                    }
                    for chunk in group.chunks
                ],
            }
            for group in request.groups
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_grouped_chunk_result(
    request: GroupedChunkEmbeddingRequest,
    result: GroupedChunkEmbeddingResult,
) -> None:
    """Reject flattened, reordered, cross-document, or identity-losing results."""
    if result.layout_id != request.layout_id or result.strategy != request.strategy:
        raise ValueError("Grouped result layout/strategy does not match its request")
    if result.dimensions <= 0:
        raise ValueError("Grouped result dimensions must be positive")
    if len(result.groups) != len(request.groups):
        raise ValueError("Grouped result flattened or changed document group count")

    seen_chunk_ids: set[str] = set()
    for request_group, result_group in zip(request.groups, result.groups, strict=True):
        if result_group.document_id != request_group.document_id:
            raise ValueError("Grouped result reordered or crossed document groups")
        if len(result_group.chunks) != len(request_group.chunks):
            raise ValueError(f"Grouped result changed chunk count for {request_group.document_id}")

        for request_chunk, result_chunk in zip(request_group.chunks, result_group.chunks, strict=True):
            if result_chunk.document_id != request_group.document_id:
                raise ValueError(f"Grouped result crossed document boundary for {request_chunk.chunk_id}")
            if result_chunk.chunk_id != request_chunk.chunk_id:
                raise ValueError(f"Grouped result lost or reordered chunk id {request_chunk.chunk_id}")
            if result_chunk.chunk_index != request_chunk.chunk_index:
                raise ValueError(f"Grouped result changed chunk index for {request_chunk.chunk_id}")
            if (result_chunk.char_start, result_chunk.char_end) != (
                request_chunk.char_start,
                request_chunk.char_end,
            ):
                raise ValueError(f"Grouped result changed canonical offsets for {request_chunk.chunk_id}")
            if result_chunk.text_sha256 != request_chunk.text_sha256:
                raise ValueError(f"Grouped result changed canonical text hash for {request_chunk.chunk_id}")
            if result_chunk.returned_text != request_chunk.text:
                raise ValueError(f"Grouped result lost canonical chunk text for {request_chunk.chunk_id}")
            if result_chunk.mapping_status not in MAPPING_STATUSES:
                raise ValueError(f"Unknown mapping status for {request_chunk.chunk_id}")
            if result_chunk.mapping_status != "exact":
                raise ValueError(f"Fixture primary track requires exact mapping for {request_chunk.chunk_id}")
            if result_chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate grouped result chunk id: {result_chunk.chunk_id}")
            seen_chunk_ids.add(result_chunk.chunk_id)
            embedding = np.asarray(result_chunk.embedding)
            if embedding.ndim != 1 or embedding.shape[0] != result.dimensions:
                raise ValueError(f"Invalid embedding shape for {request_chunk.chunk_id}")
            if not np.all(np.isfinite(embedding)):
                raise ValueError(f"Non-finite embedding for {request_chunk.chunk_id}")


def grouped_result_to_dict(result: GroupedChunkEmbeddingResult) -> dict[str, Any]:
    """Serialize a grouped result while retaining returned text and ids."""
    return {
        "layout_id": result.layout_id,
        "strategy": result.strategy,
        "dimensions": result.dimensions,
        "model_name": result.model_name,
        "provider": result.provider,
        "latency_ms": result.latency_ms,
        "token_usage": result.token_usage,
        "cost_usd": result.cost_usd,
        "metadata": dict(result.metadata or {}),
        "groups": [
            {
                "document_id": group.document_id,
                "chunks": [
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "text_sha256": chunk.text_sha256,
                        "returned_text": chunk.returned_text,
                        "mapping_status": chunk.mapping_status,
                        "embedding": np.asarray(chunk.embedding).tolist(),
                        "provider_chunk_id": chunk.provider_chunk_id,
                    }
                    for chunk in group.chunks
                ],
            }
            for group in result.groups
        ],
    }


def grouped_result_from_dict(data: dict[str, Any]) -> GroupedChunkEmbeddingResult:
    """Deserialize a grouped result for deterministic round-trip tests."""
    return GroupedChunkEmbeddingResult(
        layout_id=str(data["layout_id"]),
        strategy=str(data["strategy"]),
        dimensions=int(data["dimensions"]),
        model_name=str(data["model_name"]),
        provider=str(data["provider"]),
        latency_ms=float(data["latency_ms"]),
        token_usage=data.get("token_usage"),
        cost_usd=data.get("cost_usd"),
        metadata=dict(data.get("metadata") or {}),
        groups=tuple(
            GroupedChunkEmbeddingGroup(
                document_id=str(group["document_id"]),
                chunks=tuple(
                    GroupedChunkEmbedding(
                        document_id=str(chunk["document_id"]),
                        chunk_id=str(chunk["chunk_id"]),
                        chunk_index=int(chunk["chunk_index"]),
                        char_start=int(chunk["char_start"]),
                        char_end=int(chunk["char_end"]),
                        text_sha256=str(chunk["text_sha256"]),
                        returned_text=str(chunk["returned_text"]),
                        mapping_status=str(chunk["mapping_status"]),
                        embedding=np.asarray(chunk["embedding"], dtype=float),
                        provider_chunk_id=chunk.get("provider_chunk_id"),
                    )
                    for chunk in group["chunks"]
                ),
            )
            for group in data["groups"]
        ),
    )
