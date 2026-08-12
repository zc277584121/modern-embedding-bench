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
from mm_embed.providers.snapshot_identity import verify_snapshot_identity

MODEL_ID = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
REVISION = "babf71f3c48695e2e53a978208e8aba48335e3c0"
DIMENSIONS = 30522
VOCABULARY_ID = "sentencepiece-wordpiece-vocab-30522"
REPRESENTATION_ID = "opensearch-neural-sparse-v3-distill-csr-v1"
MAX_SNAPSHOT_BYTES = 300 * 1024 * 1024
SNAPSHOT_IDENTITY = {
    "config.json": "ee97780493e7d0a3b7b788ea98f3391e6be6b0b379921b465ca55bfdd0d9cbe3",
    "config_sentence_transformers.json": "44e5b5295415281c3f61341f0670bf6752ce1d85880380faeebae3752889d718",
    "document_1_SpladePooling/config.json": "d7a84fb418b94887421b277c5407175e476f0fa629037f4f5db540e98db682f3",
    "idf.json": "da23a1c0b9252776cc8c6d70fd14723e218f484d489cd9027ac6e4065d5b9edd",
    "model.safetensors": "83a3cc9757876b8590aac53f4f6685012f89d7fb4bbeb540815a54d325f7f70a",
    "modules.json": "2e12f8a5fbc625578d7bbaee4c18b748ccd82f0d4549c3d952638693ce4058cf",
    "query_0_SparseStaticEmbedding/config.json": "f1aa3269d4139c461e2d3b7b8f1570f507bc59529cffb004c6a3702da4f5087a",
    "query_0_SparseStaticEmbedding/model.safetensors": "711ec64837a7962d2ae106996079782b7ee87860089a0b2348bf7cb840f252d3",
    "query_0_SparseStaticEmbedding/special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    "query_0_SparseStaticEmbedding/tokenizer.json": "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854",
    "query_0_SparseStaticEmbedding/tokenizer_config.json": "74fe3555a2e4e56248a4f64333c7891f07acc7f0276c01a40764defc9564a839",
    "query_0_SparseStaticEmbedding/vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "query_token_weights.txt": "79c55cf7c5c7e1680d6332225aa58dda9b71445cf02e21b348b97c3ee5c81f07",
    "router_config.json": "c373414574b65ea6612a43c16a17d8a5942e91ccde1cd1433e41619af02a8264",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer.json": "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854",
    "tokenizer_config.json": "77128cb6e818fc57e5d75f305bdc6b7ebf8c7e91228ec9d7f33f4fcb493bc0a7",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


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
                 allow_download: bool = True, device: str | None = None,
                 max_length: int = 512, batch_size: int = 32) -> None:
        if model != MODEL_ID or revision != REVISION:
            raise ValueError(f"This adapter requires model {MODEL_ID!r} at revision {REVISION!r}")
        self.model = model
        self.revision = revision
        if max_length <= 2 or batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")
        self.max_length = max_length
        self.batch_size = batch_size
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
        self._encoder.max_seq_length = max_length
        self.device = str(getattr(self._encoder, "device", device or "cpu"))

    @staticmethod
    def _check_snapshot(path: Path) -> None:
        verify_snapshot_identity(path, SNAPSHOT_IDENTITY, label="OpenSearch neural sparse v3")
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        if total > MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot exceeds the 300 MB allow-list limit")

    def _encode(self, texts: Sequence[str], item_ids: Sequence[str], role: SparseEmbeddingRole) -> SparseEmbeddingResult:
        import torch

        started = time.perf_counter()
        method = self._encoder.encode_query if role is SparseEmbeddingRole.QUERY else self._encoder.encode_document
        truncated_count = 0
        tokenizer = getattr(self._encoder, "tokenizer", None)
        if tokenizer is not None:
            lengths = tokenizer(list(texts), add_special_tokens=True, truncation=False, return_length=True)["length"]
            truncated_count = sum(int(length > self.max_length) for length in lengths)
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        encoded = method(
            list(texts), convert_to_sparse_tensor=True, show_progress_bar=False, batch_size=self.batch_size,
        )
        matrix = _to_csr(encoded, len(texts), self.representation.dimensions)
        elapsed = (time.perf_counter() - started) * 1000.0
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(self.device)) if self.device.startswith("cuda") else 0
        return SparseEmbeddingResult(
            embeddings=SparseEmbeddingBatch(matrix, item_ids, self.representation), role=role,
            model_name=self.model, provider=self.name, model_revision=self.revision,
            query_route=self.query_route, document_route=self.document_route,
            latency_ms=elapsed, device=self.device, peak_vram_bytes=peak_vram_bytes,
            metadata={"trust_remote_code": False, "snapshot_path": str(self._snapshot_path) if self._snapshot_path else None,
                      "route": role.value, "backend": "sentence_transformers_sparse_encoder",
                      "max_length": self.max_length, "batch_size": self.batch_size,
                      "truncated_count": truncated_count, "input_count": len(texts)},
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
