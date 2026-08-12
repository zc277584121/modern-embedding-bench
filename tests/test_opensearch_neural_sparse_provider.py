from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import mm_embed.providers.opensearch_neural_sparse_provider as provider_module
from mm_embed.providers.opensearch_neural_sparse_provider import (
    DIMENSIONS,
    MODEL_ID,
    REVISION,
    OpenSearchNeuralSparseProvider,
    _to_csr,
)
from mm_embed.providers.sparse_base import SparseEmbeddingRole, SparseEncodingRoute


def _forbid_dense_conversion(*args: object, **kwargs: object) -> None:
    raise AssertionError("Dense materialization is forbidden")


class FakeSparseEncoder:
    def __init__(self, source: str, **kwargs: object) -> None:
        self.source = source
        self.kwargs = kwargs
        self.device = "cpu"
        self.calls: list[str] = []

    def encode_query(self, texts: list[str], **kwargs: object) -> sparse.csr_matrix:
        self.calls.append("query")
        return sparse.csr_matrix(([1.0], ([0], [7])), shape=(1, DIMENSIONS), dtype=np.float32)

    def encode_document(self, texts: list[str], **kwargs: object) -> sparse.csr_matrix:
        self.calls.append("document")
        rows = np.arange(len(texts))
        return sparse.csr_matrix((np.ones(len(texts)), (rows, rows + 7)), shape=(len(texts), DIMENSIONS), dtype=np.float32)


def _provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OpenSearchNeuralSparseProvider:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    module = types.ModuleType("sentence_transformers")
    module.SparseEncoder = FakeSparseEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return OpenSearchNeuralSparseProvider(snapshot_path=str(snapshot), allow_download=False)


def test_adapter_pins_revision_routes_and_uses_csr_without_dense_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparse.csr_matrix, "toarray", _forbid_dense_conversion)
    monkeypatch.setattr(sparse.csr_matrix, "todense", _forbid_dense_conversion)
    provider = _provider(tmp_path, monkeypatch)
    query = provider.encode_sparse_query("query", item_id="q")
    documents = provider.encode_sparse_documents(["a", "b", "c", "d"], item_ids=["a", "b", "c", "d"])

    assert provider.model == MODEL_ID and provider.revision == REVISION
    assert query.role is SparseEmbeddingRole.QUERY
    assert query.embeddings.values.shape == (1, DIMENSIONS)
    assert documents.embeddings.values.shape == (4, DIMENSIONS)
    assert query.embeddings.representation == documents.embeddings.representation
    assert query.query_route is SparseEncodingRoute.STATIC_LOOKUP
    assert query.document_route is SparseEncodingRoute.DOCUMENT_EXPANSION
    assert documents.query_route is SparseEncodingRoute.STATIC_LOOKUP
    assert documents.document_route is SparseEncodingRoute.DOCUMENT_EXPANSION
    assert provider._encoder.calls == ["query", "document"]
    assert provider._encoder.kwargs["trust_remote_code"] is False


def test_adapter_rejects_revision_drift_oversized_snapshot_and_dense_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="requires model"):
        OpenSearchNeuralSparseProvider(revision="main")
    snapshot = tmp_path / "oversized"
    snapshot.mkdir()
    file_path = snapshot / "model.safetensors"
    file_path.touch()
    file_path.write_bytes(b"x")
    monkeypatch.setattr(provider_module, "MAX_SNAPSHOT_BYTES", 0)
    with pytest.raises(ValueError, match="300 MB"):
        OpenSearchNeuralSparseProvider._check_snapshot(snapshot)
    with pytest.raises(TypeError, match="dense materialization"):
        _to_csr(np.zeros((1, DIMENSIONS), dtype=np.float32), 1, DIMENSIONS)


def test_adapter_rejects_shape_nonfinite_and_document_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="does not match"):
        _to_csr(sparse.csr_matrix((1, 7), dtype=np.float32), 1, DIMENSIONS)
    bad = sparse.csr_matrix(([np.inf], ([0], [0])), shape=(1, DIMENSIONS), dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        _to_csr(bad, 1, DIMENSIONS)
    provider = _provider(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="counts must match"):
        provider.encode_sparse_documents(["a"], item_ids=[])
