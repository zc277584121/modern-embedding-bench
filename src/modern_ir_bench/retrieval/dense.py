"""Composable dense retrieval Solution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.embeddings.protocols import DenseEmbedding
from modern_ir_bench.retrieval.indexes.protocols import DenseIndex, DenseIndexSession
from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit


def _batches(
    resources: Iterable[RetrievalResource[Any]],
    batch_size: int,
) -> Iterable[list[RetrievalResource[Any]]]:
    batch: list[RetrievalResource[Any]] = []
    for resource in resources:
        batch.append(resource)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass(frozen=True, kw_only=True)
class DenseRetrievalSolution(Solution):
    """Dense embedding plus a replaceable index backend."""

    embedding: DenseEmbedding
    index: DenseIndex
    document_batch_size: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.document_batch_size < 1:
            raise ValueError("document_batch_size must be positive")

    def prepare(
        self,
        resources: Iterable[RetrievalResource[Any]],
    ) -> DenseRetrievalSession:
        index_session = self.index.open(dimension=self.embedding.dimension)
        try:
            count = 0
            for batch in _batches(resources, self.document_batch_size):
                vectors = np.asarray(
                    self.embedding.encode_documents([resource.value for resource in batch]),
                    dtype=np.float32,
                )
                index_session.add([resource.id for resource in batch], vectors)
                count += len(batch)
            if count == 0:
                raise ValueError("A dense retrieval Solution requires at least one resource")
            index_session.seal()
        except Exception:
            index_session.close()
            raise
        return DenseRetrievalSession(
            embedding=self.embedding,
            index=index_session,
        )


class DenseRetrievalSession:
    def __init__(
        self,
        *,
        embedding: DenseEmbedding,
        index: DenseIndexSession,
    ) -> None:
        self.embedding = embedding
        self.index = index

    def search_batch(
        self,
        queries: Sequence[Any],
        *,
        top_k: int,
    ) -> list[list[SearchHit]]:
        vectors = np.asarray(self.embedding.encode_queries(queries), dtype=np.float32)
        return self.index.search(vectors, top_k=top_k)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.index.metadata

    def close(self) -> None:
        self.index.close()
