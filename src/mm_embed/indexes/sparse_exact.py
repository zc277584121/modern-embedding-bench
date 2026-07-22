"""Deterministic exact dot-product search over sparse CSR embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_embed.providers.sparse_base import (
    SparseEmbeddingResult,
    SparseEmbeddingRole,
    SparseEncodingRoute,
    SparseRepresentation,
)


@dataclass(frozen=True)
class SparseSearchHit:
    """One exact sparse retrieval hit with its raw dot-product score."""

    rank: int
    item_id: str
    score: float


@dataclass(frozen=True)
class SparseQueryRanking:
    """Deterministic ranked hits for one query row."""

    query_id: str
    hits: tuple[SparseSearchHit, ...]


@dataclass(frozen=True)
class SparseIndexResult:
    """Ranked output and compatibility identity from an exact sparse index."""

    queries: tuple[SparseQueryRanking, ...]
    backend: str
    exact: bool
    document_count: int
    representation: SparseRepresentation
    query_route: SparseEncodingRoute
    document_route: SparseEncodingRoute


class ExactSparseIndex:
    """Small reference index that scores directly in CSR form."""

    backend = "scipy_csr_exact"

    def __init__(self, documents: SparseEmbeddingResult) -> None:
        if documents.role is not SparseEmbeddingRole.DOCUMENT:
            raise ValueError("Exact sparse index requires a document result")
        if not documents.embeddings.item_ids:
            raise ValueError("Exact sparse index requires at least one document")
        self._documents = documents

    @property
    def document_count(self) -> int:
        return len(self._documents.embeddings.item_ids)

    @property
    def representation(self) -> SparseRepresentation:
        return self._documents.embeddings.representation

    def search(self, queries: SparseEmbeddingResult, *, k: int = 10) -> SparseIndexResult:
        """Return exact dot-product rankings with item-id tie breaking."""
        if queries.role is not SparseEmbeddingRole.QUERY:
            raise ValueError("Exact sparse search requires a query result")
        if k <= 0:
            raise ValueError("Sparse search k must be positive")
        if queries.embeddings.dimensions != self._documents.embeddings.dimensions:
            raise ValueError("Sparse query and index dimensions do not match")
        if queries.embeddings.representation != self._documents.embeddings.representation:
            raise ValueError("Sparse query and index representation identities do not match")

        scores = (queries.embeddings.values @ self._documents.embeddings.values.T).tocsr()
        scores.sum_duplicates()
        scores.eliminate_zeros()
        if not np.all(np.isfinite(scores.data)):
            raise ValueError("Sparse dot-product scores must be finite")

        document_ids = self._documents.embeddings.item_ids
        limit = min(k, len(document_ids))
        rankings: list[SparseQueryRanking] = []
        for row_index, query_id in enumerate(queries.embeddings.item_ids):
            start = scores.indptr[row_index]
            end = scores.indptr[row_index + 1]
            score_by_index = {
                int(document_index): float(score)
                for document_index, score in zip(
                    scores.indices[start:end],
                    scores.data[start:end],
                    strict=True,
                )
            }
            ranked_indices = sorted(
                range(len(document_ids)),
                key=lambda index: (-score_by_index.get(index, 0.0), document_ids[index]),
            )[:limit]
            hits = tuple(
                SparseSearchHit(
                    rank=rank,
                    item_id=document_ids[document_index],
                    score=score_by_index.get(document_index, 0.0),
                )
                for rank, document_index in enumerate(ranked_indices, start=1)
            )
            rankings.append(SparseQueryRanking(query_id=query_id, hits=hits))

        return SparseIndexResult(
            queries=tuple(rankings),
            backend=self.backend,
            exact=True,
            document_count=len(document_ids),
            representation=self.representation,
            query_route=queries.query_route,
            document_route=self._documents.document_route,
        )


__all__ = ["ExactSparseIndex", "SparseIndexResult", "SparseQueryRanking", "SparseSearchHit"]
