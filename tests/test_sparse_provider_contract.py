from __future__ import annotations

import json
import socket
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from mm_embed.indexes.sparse_exact import ExactSparseIndex
from mm_embed.providers.base import EmbeddingResult
from mm_embed.providers.sparse_base import (
    SparseEmbeddingBatch,
    SparseEmbeddingProvider,
    SparseEmbeddingResult,
    SparseEmbeddingRole,
    SparseEncodingRoute,
    SparseRepresentation,
)
from mm_embed.sparse_cache import load_sparse_result, save_sparse_result


class DeterministicSparseProvider:
    name = "deterministic-sparse-local"
    model = "fixture-token-counts"
    revision = "fixture-v1"
    representation = SparseRepresentation(
        representation_id="fixture-token-counts-v1",
        vocabulary_id="fixture-vocabulary-v1",
        dimensions=6,
    )
    query_route = SparseEncodingRoute.STATIC_LOOKUP
    document_route = SparseEncodingRoute.TOKENIZER_IDF
    vocabulary = {
        "alpha": 0,
        "beta": 1,
        "gamma": 2,
        "delta": 3,
        "epsilon": 4,
        "zeta": 5,
    }

    def encode_sparse_query(self, text: str, *, item_id: str) -> SparseEmbeddingResult:
        return self._encode((text,), (item_id,), SparseEmbeddingRole.QUERY)

    def encode_sparse_documents(
        self,
        texts: Sequence[str],
        *,
        item_ids: Sequence[str],
    ) -> SparseEmbeddingResult:
        return self._encode(tuple(texts), tuple(item_ids), SparseEmbeddingRole.DOCUMENT)

    def _encode(
        self,
        texts: tuple[str, ...],
        item_ids: tuple[str, ...],
        role: SparseEmbeddingRole,
    ) -> SparseEmbeddingResult:
        if len(texts) != len(item_ids):
            raise ValueError("Text and item id counts must match")

        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []
        for row_index, text in enumerate(texts):
            counts = Counter(token for token in text.split() if token in self.vocabulary)
            for token, count in sorted(counts.items(), key=lambda item: self.vocabulary[item[0]]):
                row_indices.append(row_index)
                column_indices.append(self.vocabulary[token])
                values.append(float(count))

        matrix = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), (row_indices, column_indices)),
            shape=(len(texts), self.representation.dimensions),
            dtype=np.float32,
        )
        batch = SparseEmbeddingBatch(
            values=matrix,
            item_ids=item_ids,
            representation=self.representation,
        )
        return SparseEmbeddingResult(
            embeddings=batch,
            role=role,
            model_name=self.model,
            provider=self.name,
            model_revision=self.revision,
            query_route=self.query_route,
            document_route=self.document_route,
            latency_ms=0.0,
            token_usage=0,
            device="cpu",
            peak_vram_bytes=0,
            metadata={"activation": "fixture_token_counts", "network": "forbidden"},
        )


def _documents(provider: DeterministicSparseProvider) -> SparseEmbeddingResult:
    return provider.encode_sparse_documents(
        ("alpha delta", "beta beta", "alpha gamma", "unknown"),
        item_ids=("doc-c", "doc-b", "doc-a", "doc-d"),
    )


def _forbid_dense_conversion(*args: object, **kwargs: object) -> None:
    raise AssertionError("Sparse contract must not materialize dense vocabulary vectors")


def test_sparse_provider_contract_scores_in_csr_without_dense_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparse.csr_matrix, "toarray", _forbid_dense_conversion)
    monkeypatch.setattr(sparse.csr_matrix, "todense", _forbid_dense_conversion)
    provider = DeterministicSparseProvider()

    assert isinstance(provider, SparseEmbeddingProvider)
    documents = _documents(provider)
    query = provider.encode_sparse_query("alpha", item_id="query-alpha")

    assert sparse.isspmatrix_csr(documents.embeddings.values)
    assert sparse.isspmatrix_csr(query.embeddings.values)
    assert documents.embeddings.item_ids == ("doc-c", "doc-b", "doc-a", "doc-d")
    assert documents.embeddings.dimensions == 6
    assert documents.embeddings.dtype == "float32"
    assert documents.embeddings.nnz_total == 5
    assert documents.embeddings.nnz_per_row == (2, 1, 2, 0)
    assert documents.embeddings.values.data.flags.writeable is False
    assert documents.embeddings.values.indices.flags.writeable is False
    assert documents.embeddings.values.indptr.flags.writeable is False
    assert documents.embeddings.representation.vocabulary_id == "fixture-vocabulary-v1"
    assert documents.query_route is SparseEncodingRoute.STATIC_LOOKUP
    assert documents.document_route is SparseEncodingRoute.TOKENIZER_IDF
    assert documents.model_name == "fixture-token-counts"
    assert documents.provider == "deterministic-sparse-local"
    assert documents.model_revision == "fixture-v1"
    assert documents.fingerprint == _documents(provider).fingerprint

    result = ExactSparseIndex(documents).search(query, k=4)

    assert result.backend == "scipy_csr_exact"
    assert result.exact is True
    assert result.document_count == 4
    assert result.representation == provider.representation
    assert result.query_route is SparseEncodingRoute.STATIC_LOOKUP
    assert result.document_route is SparseEncodingRoute.TOKENIZER_IDF
    assert [hit.item_id for hit in result.queries[0].hits] == ["doc-a", "doc-c", "doc-b", "doc-d"]
    assert [hit.rank for hit in result.queries[0].hits] == [1, 2, 3, 4]
    assert [hit.score for hit in result.queries[0].hits] == [1.0, 1.0, 0.0, 0.0]


def test_sparse_batch_exposes_defensive_csr_snapshots_without_stale_fingerprints() -> None:
    batch = _documents(DeterministicSparseProvider()).embeddings
    fingerprint = batch.fingerprint
    original_data = batch.values.data.copy()

    exposed = batch.values
    exposed.data = exposed.data + np.float32(10.0)

    assert batch.fingerprint == fingerprint
    np.testing.assert_array_equal(batch.values.data, original_data)
    assert not np.array_equal(exposed.data, batch.values.data)
    rebuilt = SparseEmbeddingBatch(
        values=batch.values,
        item_ids=batch.item_ids,
        representation=batch.representation,
    )
    assert rebuilt.fingerprint == batch.fingerprint


def test_sparse_result_deep_freezes_metadata_without_stale_fingerprints() -> None:
    documents = _documents(DeterministicSparseProvider())
    source_metadata = {
        "audit": {
            "labels": ["fixture", "sparse"],
            "route": {"query": "static_lookup", "document": "tokenizer_idf"},
        }
    }
    result = replace(documents, metadata=source_metadata)
    fingerprint = result.fingerprint

    source_metadata["audit"]["labels"].append("mutated")
    source_metadata["audit"]["route"]["query"] = "mutated"
    exported = result.metadata_dict()
    exported["audit"]["labels"].append("exported-mutation")

    assert result.metadata_dict() == {
        "audit": {
            "labels": ["fixture", "sparse"],
            "route": {"document": "tokenizer_idf", "query": "static_lookup"},
        }
    }
    assert result.fingerprint == fingerprint
    assert replace(result, metadata=result.metadata_dict()).fingerprint == fingerprint
    with pytest.raises(TypeError):
        result.metadata["audit"]["route"]["query"] = "mutated"
    with pytest.raises(TypeError):
        result.metadata["audit"]["labels"][0] = "mutated"


def test_sparse_npz_manifest_round_trip_preserves_identity_and_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparse.csr_matrix, "toarray", _forbid_dense_conversion)
    monkeypatch.setattr(sparse.csr_matrix, "todense", _forbid_dense_conversion)
    original = _documents(DeterministicSparseProvider())
    prefix = tmp_path / "documents"

    matrix_path, manifest_path = save_sparse_result(prefix, original)
    restored = load_sparse_result(prefix)

    assert matrix_path.name == "documents.npz"
    assert manifest_path.name == "documents.json"
    assert sparse.isspmatrix_csr(restored.embeddings.values)
    assert restored.embeddings.item_ids == original.embeddings.item_ids
    assert restored.embeddings.representation == original.embeddings.representation
    assert restored.embeddings.dtype == original.embeddings.dtype
    assert restored.embeddings.nnz_per_row == original.embeddings.nnz_per_row
    assert restored.metadata_dict() == original.metadata_dict()
    assert restored.embeddings.fingerprint == original.embeddings.fingerprint
    assert restored.fingerprint == original.fingerprint
    np.testing.assert_array_equal(restored.embeddings.values.indptr, original.embeddings.values.indptr)
    np.testing.assert_array_equal(restored.embeddings.values.indices, original.embeddings.values.indices)
    np.testing.assert_array_equal(restored.embeddings.values.data, original.embeddings.values.data)

    tampered_matrix = restored.embeddings.values.copy()
    tampered_matrix.data[0] += np.float32(0.5)
    sparse.save_npz(matrix_path, tampered_matrix)
    with pytest.raises(ValueError, match="metadata|fingerprint"):
        load_sparse_result(prefix)

    second_prefix = tmp_path / "manifest-tamper"
    _, second_manifest_path = save_sparse_result(second_prefix, original)
    manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["provider"] = "tampered-provider"
    second_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_sparse_result(second_prefix)


@pytest.mark.parametrize(
    ("values", "item_ids", "representation", "error"),
    [
        (
            sparse.coo_matrix(np.ones((1, 3), dtype=np.float32)),
            ("item",),
            SparseRepresentation("rep", "vocab", 3),
            "csr_matrix",
        ),
        (
            sparse.csr_matrix(np.ones((2, 3), dtype=np.float32)),
            ("item",),
            SparseRepresentation("rep", "vocab", 3),
            "row count",
        ),
        (
            sparse.csr_matrix(np.ones((1, 3), dtype=np.float32)),
            ("item",),
            SparseRepresentation("rep", "vocab", 4),
            "dimensions",
        ),
        (
            sparse.csr_matrix(np.ones((2, 3), dtype=np.float32)),
            ("duplicate", "duplicate"),
            SparseRepresentation("rep", "vocab", 3),
            "unique",
        ),
        (
            sparse.csr_matrix(np.ones((1, 3), dtype=np.int64)),
            ("item",),
            SparseRepresentation("rep", "vocab", 3),
            "floating-point",
        ),
        (
            sparse.csr_matrix(np.asarray([[np.inf, 0.0, 0.0]], dtype=np.float32)),
            ("item",),
            SparseRepresentation("rep", "vocab", 3),
            "finite",
        ),
    ],
)
def test_sparse_batch_rejects_malformed_shape_ids_dimensions_and_values(
    values: sparse.spmatrix,
    item_ids: tuple[str, ...],
    representation: SparseRepresentation,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        SparseEmbeddingBatch(values=values, item_ids=item_ids, representation=representation)  # type: ignore[arg-type]


def test_sparse_result_rejects_malformed_roles_and_routes() -> None:
    batch = _documents(DeterministicSparseProvider()).embeddings

    with pytest.raises(ValueError, match="query_route"):
        SparseEmbeddingResult(
            embeddings=batch,
            role=SparseEmbeddingRole.QUERY,
            model_name="model",
            provider="provider",
            model_revision="revision",
            query_route="static_lookup",  # type: ignore[arg-type]
            document_route=SparseEncodingRoute.TOKENIZER_IDF,
            latency_ms=0.0,
        )
    with pytest.raises(ValueError, match="requires a query encoding route"):
        SparseEmbeddingResult(
            embeddings=batch,
            role=SparseEmbeddingRole.QUERY,
            model_name="model",
            provider="provider",
            model_revision="revision",
            query_route=SparseEncodingRoute.NONE,
            document_route=SparseEncodingRoute.TOKENIZER_IDF,
            latency_ms=0.0,
        )


def test_exact_sparse_index_rejects_incompatible_dimensions_and_representation() -> None:
    provider = DeterministicSparseProvider()
    index = ExactSparseIndex(_documents(provider))

    dimension_mismatch = SparseEmbeddingResult(
        embeddings=SparseEmbeddingBatch(
            values=sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 7), dtype=np.float32),
            item_ids=("query",),
            representation=SparseRepresentation("fixture-token-counts-v1", "fixture-vocabulary-v1", 7),
        ),
        role=SparseEmbeddingRole.QUERY,
        model_name=provider.model,
        provider=provider.name,
        model_revision=provider.revision,
        query_route=provider.query_route,
        document_route=provider.document_route,
        latency_ms=0.0,
    )
    with pytest.raises(ValueError, match="dimensions"):
        index.search(dimension_mismatch)

    identity_mismatch = SparseEmbeddingResult(
        embeddings=SparseEmbeddingBatch(
            values=sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 6), dtype=np.float32),
            item_ids=("query",),
            representation=SparseRepresentation("other-representation", "fixture-vocabulary-v1", 6),
        ),
        role=SparseEmbeddingRole.QUERY,
        model_name=provider.model,
        provider=provider.name,
        model_revision=provider.revision,
        query_route=provider.query_route,
        document_route=provider.document_route,
        latency_ms=0.0,
    )
    with pytest.raises(ValueError, match="representation identities"):
        index.search(identity_mismatch)


def test_sparse_contract_is_zero_network_and_separate_from_dense_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("Network access is forbidden for the sparse contract test")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    provider = DeterministicSparseProvider()
    documents = _documents(provider)
    query = provider.encode_sparse_query("beta", item_id="query-beta")
    ranking = ExactSparseIndex(documents).search(query, k=1)
    save_sparse_result(tmp_path / "zero-network", documents)
    restored = load_sparse_result(tmp_path / "zero-network")

    assert ranking.queries[0].hits[0].item_id == "doc-b"
    assert restored.fingerprint == documents.fingerprint
    dense = EmbeddingResult(
        embeddings=np.zeros((1, 3), dtype=np.float32),
        dimensions=3,
        model_name="dense-fixture",
        provider="dense-fixture",
        latency_ms=0.0,
    )
    assert isinstance(dense.embeddings, np.ndarray)
    assert not isinstance(dense, SparseEmbeddingResult)
