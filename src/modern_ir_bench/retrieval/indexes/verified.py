"""Shadow an index with an independent exact implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from modern_ir_bench.embeddings.protocols import DenseVectors
from modern_ir_bench.retrieval.indexes.protocols import DenseIndex, DenseIndexSession
from modern_ir_bench.retrieval.types import SearchHit


@dataclass(frozen=True)
class DenseIndexAudit:
    """One primary-versus-oracle top-k comparison."""

    query_offset: int
    index_recall: float
    primary_ids: tuple[str, ...]
    oracle_ids: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedDenseIndex:
    """Return primary results while retaining exact-search audit evidence."""

    primary: DenseIndex
    oracle: DenseIndex

    def open(self, *, dimension: int) -> VerifiedDenseSession:
        return VerifiedDenseSession(
            primary=self.primary.open(dimension=dimension),
            oracle=self.oracle.open(dimension=dimension),
        )


class VerifiedDenseSession:
    def __init__(
        self,
        *,
        primary: DenseIndexSession,
        oracle: DenseIndexSession,
    ) -> None:
        self.primary = primary
        self.oracle = oracle
        self.audits: list[DenseIndexAudit] = []

    def add(self, ids: Sequence[str], vectors: DenseVectors) -> None:
        self.primary.add(ids, vectors)
        self.oracle.add(ids, vectors)

    def seal(self) -> None:
        self.primary.seal()
        self.oracle.seal()

    def search(
        self,
        query_vectors: DenseVectors,
        *,
        top_k: int,
    ) -> list[list[SearchHit]]:
        primary_results = self.primary.search(query_vectors, top_k=top_k)
        oracle_results = self.oracle.search(query_vectors, top_k=top_k)
        if len(primary_results) != len(oracle_results):
            raise RuntimeError("Primary and oracle indexes returned different batch sizes")

        offset = len(self.audits)
        for index, (primary_hits, oracle_hits) in enumerate(
            zip(primary_results, oracle_results, strict=True)
        ):
            primary_ids = tuple(hit.id for hit in primary_hits)
            oracle_ids = tuple(hit.id for hit in oracle_hits)
            denominator = len(oracle_ids)
            recall = len(set(primary_ids).intersection(oracle_ids)) / denominator if denominator else 1.0
            self.audits.append(
                DenseIndexAudit(
                    query_offset=offset + index,
                    index_recall=recall,
                    primary_ids=primary_ids,
                    oracle_ids=oracle_ids,
                )
            )
        return primary_results

    @property
    def mean_index_recall(self) -> float | None:
        if not self.audits:
            return None
        return fmean(audit.index_recall for audit in self.audits)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "verified-dense",
            "primary": dict(self.primary.metadata),
            "oracle": dict(self.oracle.metadata),
            "audit_queries": len(self.audits),
            "mean_index_recall": self.mean_index_recall,
        }

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.oracle.close()
