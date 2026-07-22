"""NPZ-plus-JSON persistence for explicit sparse embedding results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scipy import sparse

from mm_embed.providers.sparse_base import (
    SparseEmbeddingBatch,
    SparseEmbeddingResult,
    SparseEmbeddingRole,
    SparseEncodingRoute,
    SparseRepresentation,
)


SPARSE_ARTIFACT_SCHEMA_VERSION = "1.0"


def save_sparse_result(
    path_prefix: str | Path,
    result: SparseEmbeddingResult,
) -> tuple[Path, Path]:
    """Save a sparse result as ``<prefix>.npz`` plus ``<prefix>.json``."""
    matrix_path, manifest_path = _artifact_paths(path_prefix)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(matrix_path, result.embeddings.values, compressed=True)
    manifest = _manifest_for_result(matrix_path, result)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return matrix_path, manifest_path


def load_sparse_result(path_prefix: str | Path) -> SparseEmbeddingResult:
    """Load and validate a sparse result and its logical integrity fingerprint."""
    matrix_path, manifest_path = _artifact_paths(path_prefix)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read sparse manifest: {manifest_path}") from exc

    if manifest.get("schema_version") != SPARSE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported sparse artifact schema version")
    if manifest.get("matrix_file") != matrix_path.name:
        raise ValueError("Sparse manifest matrix_file does not match its path prefix")

    try:
        values = sparse.load_npz(matrix_path)
    except Exception as exc:
        raise ValueError(f"Unable to read sparse matrix artifact: {matrix_path}") from exc

    batch_data = _require_mapping(manifest, "batch")
    representation_data = _require_mapping(batch_data, "representation")
    representation = SparseRepresentation(
        representation_id=str(representation_data["representation_id"]),
        vocabulary_id=str(representation_data["vocabulary_id"]),
        dimensions=int(representation_data["dimensions"]),
    )
    batch = SparseEmbeddingBatch(
        values=values,
        item_ids=tuple(str(item_id) for item_id in batch_data["item_ids"]),
        representation=representation,
    )
    expected_batch = _batch_manifest(batch)
    if batch_data != expected_batch:
        raise ValueError("Sparse manifest batch metadata does not match the CSR artifact")

    result_data = _require_mapping(manifest, "result")
    result = SparseEmbeddingResult(
        embeddings=batch,
        role=SparseEmbeddingRole(result_data["role"]),
        model_name=str(result_data["model_name"]),
        provider=str(result_data["provider"]),
        model_revision=str(result_data["model_revision"]),
        query_route=SparseEncodingRoute(result_data["query_route"]),
        document_route=SparseEncodingRoute(result_data["document_route"]),
        latency_ms=float(result_data["latency_ms"]),
        token_usage=_optional_int(result_data.get("token_usage")),
        device=_optional_str(result_data.get("device")),
        peak_vram_bytes=_optional_int(result_data.get("peak_vram_bytes")),
        metadata=dict(result_data.get("metadata") or {}),
    )
    if result.fingerprint != manifest.get("fingerprint"):
        raise ValueError("Sparse artifact fingerprint mismatch")
    return result


def _artifact_paths(path_prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(path_prefix)
    return prefix.with_suffix(".npz"), prefix.with_suffix(".json")


def _manifest_for_result(matrix_path: Path, result: SparseEmbeddingResult) -> dict[str, Any]:
    return {
        "schema_version": SPARSE_ARTIFACT_SCHEMA_VERSION,
        "matrix_file": matrix_path.name,
        "batch": _batch_manifest(result.embeddings),
        "result": {
            "role": result.role.value,
            "model_name": result.model_name,
            "provider": result.provider,
            "model_revision": result.model_revision,
            "query_route": result.query_route.value,
            "document_route": result.document_route.value,
            "latency_ms": result.latency_ms,
            "token_usage": result.token_usage,
            "device": result.device,
            "peak_vram_bytes": result.peak_vram_bytes,
            "metadata": result.metadata_dict(),
        },
        "fingerprint": result.fingerprint,
    }


def _batch_manifest(batch: SparseEmbeddingBatch) -> dict[str, Any]:
    return {
        "item_ids": list(batch.item_ids),
        "shape": list(batch.values.shape),
        "dimensions": batch.dimensions,
        "dtype": batch.dtype,
        "nnz_total": batch.nnz_total,
        "nnz_per_row": list(batch.nnz_per_row),
        "representation": {
            "representation_id": batch.representation.representation_id,
            "vocabulary_id": batch.representation.vocabulary_id,
            "dimensions": batch.representation.dimensions,
            "identity": batch.representation.identity,
        },
        "fingerprint": batch.fingerprint,
    }


def _require_mapping(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Sparse manifest field {key!r} must be an object")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = ["SPARSE_ARTIFACT_SCHEMA_VERSION", "load_sparse_result", "save_sparse_result"]
