"""Zero-network sparse provider for the learned-sparse contract fixture."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse

from mm_embed.data.learned_sparse_retrieval_fixture import (
    DATASET_VERSION,
    DIMENSIONS,
    REPRESENTATION_ID,
    VOCABULARY_ID,
    load_learned_sparse_retrieval_fixture,
)
from mm_embed.providers.sparse_base import (
    SparseEmbeddingBatch,
    SparseEmbeddingResult,
    SparseEmbeddingRole,
    SparseEncodingRoute,
    SparseRepresentation,
)


class DeterministicSparseFixtureProvider:
    """Encode fixture ids through fixed, role-specific sparse activations."""

    name = "deterministic_sparse_fixture"
    model = "learned-sparse-contract-fixture"
    revision = "fixture-v1"
    representation = SparseRepresentation(REPRESENTATION_ID, VOCABULARY_ID, DIMENSIONS)
    query_route = SparseEncodingRoute.STATIC_LOOKUP
    document_route = SparseEncodingRoute.DOCUMENT_EXPANSION

    def __init__(
        self,
        *,
        model: str = model,
        revision: str = revision,
        dataset_version: str = DATASET_VERSION,
        document_route: str | SparseEncodingRoute = document_route,
    ) -> None:
        if dataset_version != DATASET_VERSION:
            raise ValueError(f"Unsupported learned-sparse fixture version: {dataset_version}")
        self.model = model
        self.revision = revision
        self.document_route = SparseEncodingRoute(document_route)
        if self.document_route not in {
            SparseEncodingRoute.DOCUMENT_EXPANSION,
            SparseEncodingRoute.STATIC_LOOKUP,
        }:
            raise ValueError("Sparse fixture only supports document-expansion or static-lookup document routes")
        fixture = load_learned_sparse_retrieval_fixture()
        self._query_rows = {query.query_id: query.activations for query in fixture.queries}
        self._query_text_ids = {query.text: query.query_id for query in fixture.queries}
        self._document_rows = {document.document_id: document.activations for document in fixture.documents}
        self._static_document_rows = {
            document.document_id: document.static_lookup_activations or document.activations
            for document in fixture.documents
        }
        self._document_text_ids = {document.text: document.document_id for document in fixture.documents}

    def encode_sparse_query(self, text: str, *, item_id: str) -> SparseEmbeddingResult:
        expected_id = self._query_text_ids.get(text)
        if expected_id != item_id:
            raise ValueError("Sparse fixture query text/id binding is unknown or mismatched")
        return self._encode((item_id,), SparseEmbeddingRole.QUERY)

    def encode_sparse_documents(
        self,
        texts: Sequence[str],
        *,
        item_ids: Sequence[str],
    ) -> SparseEmbeddingResult:
        if len(texts) != len(item_ids):
            raise ValueError("Sparse fixture document text and id counts must match")
        for text, item_id in zip(texts, item_ids, strict=True):
            if self._document_text_ids.get(text) != item_id:
                raise ValueError("Sparse fixture document text/id binding is unknown or mismatched")
        return self._encode(tuple(item_ids), SparseEmbeddingRole.DOCUMENT)

    def _encode(self, item_ids: tuple[str, ...], role: SparseEmbeddingRole) -> SparseEmbeddingResult:
        if role is SparseEmbeddingRole.QUERY:
            rows = self._query_rows
        elif self.document_route is SparseEncodingRoute.DOCUMENT_EXPANSION:
            rows = self._document_rows
        else:
            rows = self._static_document_rows
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []
        for row_index, item_id in enumerate(item_ids):
            try:
                activations = rows[item_id]
            except KeyError as exc:
                raise ValueError(f"Unknown sparse fixture item id: {item_id}") from exc
            for coordinate, value in activations:
                row_indices.append(row_index)
                column_indices.append(coordinate)
                values.append(value)
        matrix = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), (row_indices, column_indices)),
            shape=(len(item_ids), self.representation.dimensions),
            dtype=np.float32,
        )
        return SparseEmbeddingResult(
            embeddings=SparseEmbeddingBatch(matrix, item_ids, self.representation),
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
            metadata={
                "dataset_version": DATASET_VERSION,
                "fixture_only": True,
                "network": "forbidden",
                "route": role.value,
            },
        )


__all__ = ["DeterministicSparseFixtureProvider"]
