"""Sentence Transformers SparseEncoder adapter for OpenSearch neural sparse v3."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import sparse

from mm_embed.providers.sparse_base import (
    SparseEmbeddingBatch,
    SparseEmbeddingResult,
    SparseEmbeddingRole,
    SparseEncodingRoute,
    SparseRepresentation,
)

MODEL_ID = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
REVISION = "babf71f3c48695e2e53a978208e8aba48335e3c0"
DIMENSIONS = 30522
VOCABULARY_ID = "sentencepiece-wordpiece-vocab-30522"
REPRESENTATION_ID = "opensearch-neural-sparse-v3-distill-csr-v1"
MAX_SNAPSHOT_BYTES = 300 * 1024 * 1024


class OpenSearchNeuralSparseProvider:
    """Explicit query/document routing through ``SparseEncoder``."""

    name = "opensearch_neural_sparse"
    model = MODEL_ID
    revision = REVISION
    representation = SparseRepresentation(REPRESENTATION_ID, VOCABULARY_ID, DIMENSIONS)
    query_route = SparseEncodingRoute.STATIC_LOOKUP
    document_route = SparseEncodingRoute.DOCUMENT_EXPANSION

    def __init__(self, *, model: str = MODEL_ID, revision: str = REVISION,
                 snapshot_path: str | None = None, cache_dir: str | None = None,
                 allow_download: bool = True, device: str | None = None) -> None:
        if model != MODEL_ID or revision != REVISION:
            raise ValueError(f"This adapter requires model {MODEL_ID!r} at revision {REVISION!r}")
        self.model = model
        self.revision = revision
        self._snapshot_path = Path(snapshot_path).expanduser() if snapshot_path else None
        if self._snapshot_path is not None:
            self._check_snapshot(self._snapshot_path)
        try:
            from sentence_transformers import SparseEncoder
        except ImportError as exc:
            raise ImportError("Install the local extra to use OpenSearchNeuralSparseProvider") from exc
        load_kwargs = {"trust_remote_code": False}
        if cache_dir:
            load_kwargs["cache_folder"] = cache_dir
        if device:
            load_kwargs["device"] = device
        if not self._snapshot_path and not allow_download:
            raise ValueError("snapshot_path is required when allow_download=False")
        if self._snapshot_path is None:
            from huggingface_hub import snapshot_download

            snapshot = snapshot_download(
                repo_id=model,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=[
                    "config.json", "config_sentence_transformers.json", "document_1_SpladePooling/*",
                    "idf.json", "model.safetensors", "modules.json", "query_0_SparseStaticEmbedding/*",
                    "query_token_weights.txt", "router_config.json", "special_tokens_map.json", "tokenizer.json",
                    "tokenizer_config.json", "vocab.txt",
                ],
                ignore_patterns=["README.md", ".gitattributes"],
            )
            self._snapshot_path = Path(snapshot)
        self._check_snapshot(self._snapshot_path)
        self._encoder = SparseEncoder(str(self._snapshot_path), **load_kwargs)
        self.device = str(getattr(self._encoder, "device", device or "cpu"))

    @staticmethod
    def _check_snapshot(path: Path) -> None:
        if not path.is_dir():
            raise ValueError(f"snapshot_path must be a directory: {path}")
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        if total > MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot exceeds the 300 MB allow-list limit")

    def _encode(self, texts: Sequence[str], item_ids: Sequence[str], role: SparseEmbeddingRole) -> SparseEmbeddingResult:
        started = time.perf_counter()
        method = self._encoder.encode_query if role is SparseEmbeddingRole.QUERY else self._encoder.encode_document
        encoded = method(list(texts), convert_to_sparse_tensor=True, show_progress_bar=False)
        matrix = _to_csr(encoded, len(texts), self.representation.dimensions)
        elapsed = (time.perf_counter() - started) * 1000.0
        return SparseEmbeddingResult(
            embeddings=SparseEmbeddingBatch(matrix, item_ids, self.representation), role=role,
            model_name=self.model, provider=self.name, model_revision=self.revision,
            query_route=self.query_route, document_route=self.document_route,
            latency_ms=elapsed, device=self.device, peak_vram_bytes=0,
            metadata={"trust_remote_code": False, "snapshot_path": str(self._snapshot_path) if self._snapshot_path else None,
                      "route": role.value, "backend": "sentence_transformers_sparse_encoder"},
        )

    def encode_sparse_query(self, text: str, *, item_id: str) -> SparseEmbeddingResult:
        return self._encode((text,), (item_id,), SparseEmbeddingRole.QUERY)

    def encode_sparse_documents(self, texts: Sequence[str], *, item_ids: Sequence[str]) -> SparseEmbeddingResult:
        if len(texts) != len(item_ids):
            raise ValueError("Sparse document text and id counts must match")
        return self._encode(texts, item_ids, SparseEmbeddingRole.DOCUMENT)


def _to_csr(encoded: object, rows: int, dimensions: int) -> sparse.csr_matrix:
    if sparse.isspmatrix_csr(encoded):
        matrix = encoded
    elif sparse.issparse(encoded):
        matrix = encoded.tocsr()
    elif getattr(encoded, "is_sparse", False):
        tensor = encoded.coalesce()
        indices = tensor.indices().detach().cpu().numpy()
        values = tensor.values().detach().cpu().numpy()
        matrix = sparse.csr_matrix((values, (indices[0], indices[1])), shape=(rows, dimensions))
    else:
        raise TypeError("SparseEncoder output must be sparse; dense materialization is forbidden")
    if matrix.shape != (rows, dimensions):
        raise ValueError(f"SparseEncoder shape {matrix.shape} does not match {(rows, dimensions)}")
    matrix = matrix.astype(np.float32, copy=False)
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("SparseEncoder output contains non-finite values")
    return matrix


__all__ = ["DIMENSIONS", "MODEL_ID", "REVISION", "OpenSearchNeuralSparseProvider"]
