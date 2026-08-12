"""Sparse-only BGE-M3 adapter with pinned local snapshot loading."""

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

MODEL_ID = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DIMENSIONS = 250002
VOCABULARY_ID = "bge-m3-xlm-roberta-vocab-250002"
REPRESENTATION_ID = "bge-m3-contextual-lexical-csr-v1"
MAX_SNAPSHOT_BYTES = 3 * 1024**3
SNAPSHOT_IDENTITY = {
    "1_Pooling/config.json": "e54c164a07274f2eb45bb724f54a79d1efcc90c41573887cd9a29aeee0597352",
    "colbert_linear.pt": "19bfbae397c2b7524158c919d0e9b19393c5639d098f0a66932c91ed8f5f9abb",
    "config.json": "26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa",
    "config_sentence_transformers.json": "1eef72430e7194a1e59680e635aed81ffa083f05668dbc5bb1c56c04c0999c38",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "pytorch_model.bin": "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
    "sentence_bert_config.json": "eb9b44b13c0f52a3b3685c3b1cbdea1ba8b04bea123b98f61610048940776eb1",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "sparse_linear.pt": "45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    "tokenizer.json": "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
    "tokenizer_config.json": "a62b2b6784f990259fddef5f16388693a8043be4f69179e6a5257eeb3f9abac4",
}


class BGEM3SparseProvider:
    """Encode both routes as max-pooled contextual token weights."""

    name = "bge_m3_sparse"
    model = MODEL_ID
    revision = REVISION
    representation = SparseRepresentation(REPRESENTATION_ID, VOCABULARY_ID, DIMENSIONS)
    query_route = SparseEncodingRoute.NEURAL
    document_route = SparseEncodingRoute.NEURAL

    def __init__(
        self,
        *,
        model: str = MODEL_ID,
        revision: str = REVISION,
        snapshot_path: str,
        device: str = "cuda:0",
        max_length: int = 512,
        batch_size: int = 8,
        use_fp16: bool = True,
    ) -> None:
        if model != MODEL_ID or revision != REVISION:
            raise ValueError(f"This adapter requires model {MODEL_ID!r} at revision {REVISION!r}")
        if max_length <= 2 or batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")
        self.model, self.revision = model, revision
        self.max_length, self.batch_size = max_length, batch_size
        self._snapshot_path = Path(snapshot_path).expanduser().resolve()
        self._check_snapshot(self._snapshot_path)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self._snapshot_path, local_files_only=True)
        self._encoder = AutoModel.from_pretrained(self._snapshot_path, local_files_only=True, trust_remote_code=False)
        self._sparse_linear = torch.nn.Linear(self._encoder.config.hidden_size, 1)
        state = torch.load(self._snapshot_path / "sparse_linear.pt", map_location="cpu", weights_only=True)
        self._sparse_linear.load_state_dict(state)
        self.device = device
        dtype = torch.float16 if use_fp16 and device.startswith("cuda") else torch.float32
        self._encoder = self._encoder.to(device=device, dtype=dtype).eval()
        self._sparse_linear = self._sparse_linear.to(device=device, dtype=dtype).eval()
        self._unused_ids = {
            value for value in (
                self._tokenizer.cls_token_id, self._tokenizer.eos_token_id,
                self._tokenizer.pad_token_id, self._tokenizer.unk_token_id,
            ) if value is not None
        }

    @staticmethod
    def _check_snapshot(path: Path) -> None:
        verify_snapshot_identity(path, SNAPSHOT_IDENTITY, label="BGE-M3")
        total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        if total > MAX_SNAPSHOT_BYTES:
            raise ValueError("BGE-M3 snapshot exceeds the 3 GiB allow-list limit")

    def _encode(self, texts: Sequence[str], item_ids: Sequence[str], role: SparseEmbeddingRole) -> SparseEmbeddingResult:
        torch = self._torch
        started = time.perf_counter()
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        truncated = 0
        peak_vram = 0
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            for offset in range(0, len(texts), self.batch_size):
                batch = list(texts[offset:offset + self.batch_size])
                full_lengths = self._tokenizer(batch, add_special_tokens=True, truncation=False, return_length=True)["length"]
                truncated += sum(int(length > self.max_length) for length in full_lengths)
                tokens = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length,
                    return_tensors="pt", return_token_type_ids=False,
                ).to(self.device)
                hidden = self._encoder(**tokens, return_dict=True).last_hidden_state
                weights = torch.relu(self._sparse_linear(hidden)).squeeze(-1).float().cpu().numpy()
                token_ids = tokens["input_ids"].cpu().numpy()
                masks = tokens["attention_mask"].cpu().numpy()
                for local_row, (ids, values, mask) in enumerate(zip(token_ids, weights, masks, strict=True)):
                    maxima: dict[int, float] = {}
                    for token_id, value, active in zip(ids, values, mask, strict=True):
                        token_id, value = int(token_id), float(value)
                        if active and token_id not in self._unused_ids and value > maxima.get(token_id, 0.0):
                            maxima[token_id] = value
                    row = offset + local_row
                    rows.extend([row] * len(maxima)); cols.extend(maxima); data.extend(maxima.values())
        if self.device.startswith("cuda"):
            peak_vram = int(torch.cuda.max_memory_allocated(self.device))
        matrix = sparse.csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(len(texts), DIMENSIONS))
        elapsed = (time.perf_counter() - started) * 1000.0
        return SparseEmbeddingResult(
            embeddings=SparseEmbeddingBatch(matrix, item_ids, self.representation), role=role,
            model_name=self.model, provider=self.name, model_revision=self.revision,
            query_route=self.query_route, document_route=self.document_route,
            latency_ms=elapsed, device=self.device, peak_vram_bytes=peak_vram,
            metadata={"backend": "transformers_sparse_only", "snapshot_path": str(self._snapshot_path),
                      "max_length": self.max_length, "batch_size": self.batch_size,
                      "truncated_count": truncated, "input_count": len(texts), "trust_remote_code": False},
        )

    def encode_sparse_query(self, text: str, *, item_id: str) -> SparseEmbeddingResult:
        return self._encode((text,), (item_id,), SparseEmbeddingRole.QUERY)

    def encode_sparse_documents(self, texts: Sequence[str], *, item_ids: Sequence[str]) -> SparseEmbeddingResult:
        if len(texts) != len(item_ids):
            raise ValueError("Sparse document text and id counts must match")
        return self._encode(texts, item_ids, SparseEmbeddingRole.DOCUMENT)


__all__ = ["BGEM3SparseProvider", "DIMENSIONS", "MODEL_ID", "REVISION"]
