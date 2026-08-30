"""Fail-closed Sentence Transformers SparseEncoder adapter with CSR-only output."""

from __future__ import annotations

import resource
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from mm_embed.providers.learned_sparse_inventory import INVENTORY, LearnedSparseSpec
from mm_embed.providers.snapshot_identity import verify_snapshot_identity
from mm_embed.providers.sparse_base import (
    SparseEmbeddingBatch,
    SparseEmbeddingResult,
    SparseEmbeddingRole,
    SparseEncodingRoute,
    SparseRepresentation,
)

GPU_GATE_CAP_BYTES = int(10.5 * 1024**3)
ROUTES = {route.value: route for route in SparseEncodingRoute}


class SentenceTransformersSparseProvider:
    """Run a pinned standard SparseEncoder without dense materialization."""

    name = "sentence_transformers_sparse"

    def __init__(
        self,
        *,
        model_key: str,
        snapshot_path: str | None = None,
        cache_root: str | None = None,
        allow_download: bool = False,
        device: str = "cuda:0",
        batch_size: int = 8,
        max_length: int | None = None,
        gpu_cap_bytes: int = GPU_GATE_CAP_BYTES,
        encoder_factory: Callable[..., Any] | None = None,
        model_spec: LearnedSparseSpec | None = None,
        expected_snapshot_identity: Mapping[str, str] | None = None,
    ) -> None:
        if model_spec is None:
            try:
                spec = INVENTORY[model_key]
            except KeyError as exc:
                raise ValueError(f"Unknown learned-sparse model key: {model_key}") from exc
        else:
            spec = model_spec
            if spec.key != model_key:
                raise ValueError("model_spec key does not match model_key")
        if spec.adapter != self.name or spec.status != "selected":
            raise ValueError(f"{model_key} is not a selected generic SparseEncoder model")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected_max_length = spec.max_length if max_length is None else max_length
        if selected_max_length != spec.max_length:
            raise ValueError(f"{model_key} max_length is fixed at {spec.max_length}")
        frozen_identity = dict(expected_snapshot_identity or spec.identity)
        if not frozen_identity:
            raise ValueError(f"{model_key} has no externally frozen snapshot identity")

        self.spec = spec
        self.model = spec.repo_id
        self.revision = spec.revision
        self.representation = SparseRepresentation(
            f"{spec.key}-csr-v1",
            spec.vocabulary_id,
            spec.dimensions,
        )
        self.query_route = ROUTES[spec.query_route]
        self.document_route = ROUTES[spec.document_route]
        self.max_length = selected_max_length
        self.batch_size = batch_size
        self.gpu_cap_bytes = gpu_cap_bytes
        if allow_download:
            raise ValueError("Model downloads must use the bounded resolver before provider construction")
        if snapshot_path is None:
            from mm_embed.providers.learned_sparse_inventory import BoundedSnapshotResolver

            resolved = BoundedSnapshotResolver(cache_root=cache_root).resolve_local(spec)
            self._snapshot_path = Path(resolved["snapshot_path"])
            self.hub_cache_dir = str(resolved["cache"]["hub_cache_dir"])
        else:
            self._snapshot_path = Path(snapshot_path).expanduser().resolve()
            self.hub_cache_dir = None
        verify_snapshot_identity(self._snapshot_path, frozen_identity, label=spec.key)

        if encoder_factory is None:
            try:
                from sentence_transformers import SparseEncoder
            except ImportError as exc:
                raise ImportError("Install the local extra to use SentenceTransformersSparseProvider") from exc
            encoder_factory = SparseEncoder
        self._encoder = encoder_factory(
            str(self._snapshot_path),
            trust_remote_code=False,
            local_files_only=True,
            device=device,
        )
        self._encoder.max_seq_length = self.max_length
        self.device = str(getattr(self._encoder, "device", device))

    def _token_lengths(self, texts: Sequence[str]) -> tuple[list[int], int]:
        tokenizer = getattr(self._encoder, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("SparseEncoder does not expose a tokenizer for truncation audit")
        encoded = tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            return_length=True,
            verbose=False,
        )
        lengths = [int(value) for value in encoded["length"]]
        return lengths, sum(int(value > self.max_length) for value in lengths)

    def _encode(
        self,
        texts: Sequence[str],
        item_ids: Sequence[str],
        role: SparseEmbeddingRole,
    ) -> SparseEmbeddingResult:
        if not texts or len(texts) != len(item_ids):
            raise ValueError("Sparse text and id batches must have the same non-zero length")
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("Sparse input texts must be non-empty strings")
        import torch

        lengths, truncated_count = self._token_lengths(texts)
        method = self._encoder.encode_query if role is SparseEmbeddingRole.QUERY else self._encoder.encode_document
        pruning = self.spec.query_pruning if role is SparseEmbeddingRole.QUERY else self.spec.document_pruning
        started = time.perf_counter()
        cpu_started = time.process_time()
        attempted_batch_sizes: list[int] = []
        active_batch_size = self.batch_size
        while True:
            attempted_batch_sizes.append(active_batch_size)
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
            try:
                encoded = method(
                    list(texts),
                    convert_to_sparse_tensor=True,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=active_batch_size,
                    max_active_dims=pruning,
                    save_to_cpu=True,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if not self.device.startswith("cuda") or active_batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                active_batch_size = max(1, active_batch_size // 2)

        matrix = to_csr(encoded, len(texts), self.representation.dimensions)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        cpu_time_s = time.process_time() - cpu_started
        peak_vram = int(torch.cuda.max_memory_allocated(self.device)) if self.device.startswith("cuda") else 0
        if peak_vram > self.gpu_cap_bytes:
            raise ValueError(f"Sparse encoding peak VRAM {peak_vram} exceeds cap {self.gpu_cap_bytes}")
        return SparseEmbeddingResult(
            embeddings=SparseEmbeddingBatch(matrix, item_ids, self.representation),
            role=role,
            model_name=self.model,
            provider=self.name,
            model_revision=self.revision,
            query_route=self.query_route,
            document_route=self.document_route,
            latency_ms=elapsed_ms,
            device=self.device,
            peak_vram_bytes=peak_vram,
            metadata={
                "backend": "sentence_transformers_sparse_encoder",
                "batch_size_requested": self.batch_size,
                "batch_size_used": active_batch_size,
                "batch_size_attempts": attempted_batch_sizes,
                "cpu_time_s": cpu_time_s,
                "input_count": len(texts),
                "max_length": self.max_length,
                "max_observed_tokens": max(lengths),
                "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "pruning_max_active_dims": pruning,
                "route": role.value,
                "snapshot_path": str(self._snapshot_path),
                "tokenizer_class": type(self._encoder.tokenizer).__name__,
                "truncated_count": truncated_count,
                "trust_remote_code": False,
            },
        )

    def encode_sparse_queries(self, texts: Sequence[str], *, item_ids: Sequence[str]) -> SparseEmbeddingResult:
        """Encode a public ordered query batch through the model query route."""
        return self._encode(texts, item_ids, SparseEmbeddingRole.QUERY)

    def encode_sparse_query(self, text: str, *, item_id: str) -> SparseEmbeddingResult:
        return self.encode_sparse_queries((text,), item_ids=(item_id,))

    def encode_sparse_documents(self, texts: Sequence[str], *, item_ids: Sequence[str]) -> SparseEmbeddingResult:
        """Encode ordered documents through the model document route."""
        return self._encode(texts, item_ids, SparseEmbeddingRole.DOCUMENT)


def to_csr(encoded: object, rows: int, dimensions: int) -> sparse.csr_matrix:
    """Convert only sparse outputs to canonical finite non-negative CSR."""
    if sparse.isspmatrix_csr(encoded):
        matrix = encoded.copy()
    elif sparse.issparse(encoded):
        matrix = encoded.tocsr(copy=True)
    elif getattr(encoded, "is_sparse", False):
        tensor = encoded.coalesce()
        if tensor.ndim != 2:
            raise ValueError("SparseEncoder output must be a two-dimensional sparse tensor")
        indices = tensor.indices().detach().cpu().numpy()
        values = tensor.values().detach().float().cpu().numpy()
        matrix = sparse.csr_matrix((values, (indices[0], indices[1])), shape=(rows, dimensions))
    else:
        raise TypeError("SparseEncoder output must be sparse; dense materialization is forbidden")
    if matrix.shape != (rows, dimensions):
        raise ValueError(f"SparseEncoder shape {matrix.shape} does not match {(rows, dimensions)}")
    matrix = matrix.astype(np.float32, copy=False)
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("SparseEncoder output contains non-finite values")
    if np.any(matrix.data < 0):
        raise ValueError("SparseEncoder output contains negative values")
    return matrix


__all__ = ["GPU_GATE_CAP_BYTES", "SentenceTransformersSparseProvider", "to_csr"]
