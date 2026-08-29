from __future__ import annotations

import shutil
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
from mm_embed.providers.snapshot_identity import file_sha256


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
    monkeypatch.setattr(provider_module, "SNAPSHOT_IDENTITY", {"config.json": file_sha256(snapshot / "config.json")})
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
    monkeypatch.setattr(provider_module, "SNAPSHOT_IDENTITY", {"model.safetensors": file_sha256(file_path)})
    with pytest.raises(ValueError, match="300 MB"):
        OpenSearchNeuralSparseProvider._check_snapshot(snapshot)
    with pytest.raises(TypeError, match="dense materialization"):
        _to_csr(np.zeros((1, DIMENSIONS), dtype=np.float32), 1, DIMENSIONS)


def test_adapter_rejects_forged_snapshot_named_as_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / REVISION
    snapshot.mkdir()
    for name in provider_module.SNAPSHOT_IDENTITY:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("forged", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        OpenSearchNeuralSparseProvider._check_snapshot(snapshot)


def test_snapshot_identity_accepts_identical_copy_and_rejects_query_route_tampering(tmp_path: Path) -> None:
    source = (
        Path.home()
        / ".cache/huggingface/hub/models--opensearch-project--opensearch-neural-sparse-encoding-doc-v3-distill"
        / "snapshots"
        / REVISION
    )
    if not source.is_dir():
        pytest.skip("pinned OpenSearch snapshot is not available")

    copied = tmp_path / "snapshot-copy"
    shutil.copytree(source, copied, symlinks=False)
    cache_lock = copied / ".cache" / "huggingface" / "download.lock"
    cache_lock.parent.mkdir(parents=True)
    cache_lock.write_text("layout-local cache metadata", encoding="utf-8")
    OpenSearchNeuralSparseProvider._check_snapshot(copied)

    targets = (
        "query_token_weights.txt",
        "idf.json",
        "query_0_SparseStaticEmbedding/model.safetensors",
    )
    for relative in targets:
        tampered = tmp_path / relative.replace("/", "-")
        shutil.copytree(copied, tampered)
        with (tampered / relative).open("ab") as handle:
            handle.write(b"tampered")
        with pytest.raises(ValueError, match="identity mismatch"):
            OpenSearchNeuralSparseProvider._check_snapshot(tampered)


def test_snapshot_identity_rejects_added_behavior_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = snapshot / "config.json"
    config.write_text("{}", encoding="utf-8")
    extra = snapshot / "new_behavior.json"
    extra.write_text("{}", encoding="utf-8")
    expected = {"config.json": file_sha256(config)}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(provider_module, "SNAPSHOT_IDENTITY", expected)
        with pytest.raises(ValueError, match="unexpected identity files"):
            OpenSearchNeuralSparseProvider._check_snapshot(snapshot)


def test_adapter_rejects_shape_nonfinite_and_document_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="does not match"):
        _to_csr(sparse.csr_matrix((1, 7), dtype=np.float32), 1, DIMENSIONS)
    bad = sparse.csr_matrix(([np.inf], ([0], [0])), shape=(1, DIMENSIONS), dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        _to_csr(bad, 1, DIMENSIONS)
    negative = sparse.csr_matrix(([-1.0], ([0], [0])), shape=(1, DIMENSIONS), dtype=np.float32)
    with pytest.raises(ValueError, match="negative"):
        _to_csr(negative, 1, DIMENSIONS)
    provider = _provider(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="counts must match"):
        provider.encode_sparse_documents(["a"], item_ids=[])


def test_csr_canonicalization_sums_duplicates_and_eliminates_zeros() -> None:
    matrix = sparse.csr_matrix(
        (
            np.asarray([1.25, 2.75, 0.0], dtype=np.float32),
            np.asarray([7, 7, 9], dtype=np.int32),
            np.asarray([0, 3], dtype=np.int32),
        ),
        shape=(1, DIMENSIONS),
    )
    result = _to_csr(matrix, 1, DIMENSIONS)
    assert result.has_canonical_format
    assert result.nnz == 1
    assert result[0, 7] == 4.0


def test_anchor_uses_deterministic_oom_fallback_and_enforces_gpu_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    provider = _provider(tmp_path, monkeypatch)
    provider.device = "cuda:0"
    provider.batch_size = 8
    provider.gpu_cap_bytes = 1000
    provider._encoder.tokenizer = lambda texts, **kwargs: {"length": [4 for _ in texts]}
    attempts: list[int] = []

    def encode(texts, **kwargs):
        attempts.append(kwargs["batch_size"])
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return sparse.csr_matrix(([1.0], ([0], [7])), shape=(1, DIMENSIONS), dtype=np.float32)

    provider._encoder.encode_query = encode
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 900)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = provider.encode_sparse_query("query", item_id="q")
    assert attempts == [8, 4]
    assert result.metadata_dict()["batch_size_requested"] == 8
    assert result.metadata_dict()["batch_size_attempts"] == [8, 4]
    assert result.metadata_dict()["batch_size_used"] == 4

    provider.gpu_cap_bytes = 800
    with pytest.raises(ValueError, match="exceeds cap"):
        provider.encode_sparse_query("query", item_id="q")
