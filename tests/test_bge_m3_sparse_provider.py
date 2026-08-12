from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import mm_embed.providers.bge_m3_sparse_provider as provider_module
from mm_embed.providers.bge_m3_sparse_provider import BGEM3SparseProvider, DIMENSIONS
from mm_embed.providers.snapshot_identity import file_sha256
from mm_embed.providers.sparse_base import SparseEmbeddingRole


def test_bge_m3_sparse_snapshot_rejects_forged_revision_directory(tmp_path: Path) -> None:
    snapshot = tmp_path / provider_module.REVISION
    snapshot.mkdir()
    for name in provider_module.SNAPSHOT_IDENTITY:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("forged", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        BGEM3SparseProvider._check_snapshot(snapshot)


def test_bge_m3_sparse_snapshot_accepts_content_identical_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "copied-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("pinned", encoding="utf-8")
    monkeypatch.setattr(provider_module, "SNAPSHOT_IDENTITY", {"config.json": file_sha256(snapshot / "config.json")})
    BGEM3SparseProvider._check_snapshot(snapshot)


def test_bge_m3_sparse_adapter_emits_finite_csr_without_dense_conversion() -> None:
    provider = object.__new__(BGEM3SparseProvider)
    provider.model = "BAAI/bge-m3"
    provider.revision = "5617a9f61b028005a4858fdac845db406aefb181"
    provider.device = "cpu"
    provider.max_length = 8
    provider.batch_size = 2
    provider._snapshot_path = Path("/pinned")
    provider._unused_ids = {0, 1, 2, 3}
    provider._torch = __import__("torch")

    class Tokens(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            if kwargs.get("return_length"):
                return {"length": [4 for _ in texts]}
            torch = provider._torch
            return Tokens(input_ids=torch.tensor([[0, 7, 7, 2] for _ in texts]),
                          attention_mask=torch.ones((len(texts), 4), dtype=torch.long))

    class Encoder:
        def __call__(self, **kwargs):
            torch = provider._torch
            return type("Output", (), {"last_hidden_state": torch.ones((*kwargs["input_ids"].shape, 1))})

    provider._tokenizer = Tokenizer()
    provider._encoder = Encoder()
    provider._sparse_linear = lambda hidden: hidden
    result = provider._encode(["a", "b"], ["a", "b"], SparseEmbeddingRole.DOCUMENT)
    assert sparse.isspmatrix_csr(result.embeddings.values)
    assert result.embeddings.values.shape == (2, DIMENSIONS)
    assert np.all(np.isfinite(result.embeddings.values.data))
    assert result.embeddings.nnz_per_row == (1, 1)
