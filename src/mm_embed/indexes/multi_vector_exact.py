"""Bounded exact MaxSim scoring for multi-vector representations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mm_embed.providers.multi_vector_base import MultiVectorResult, MultiVectorRole


@dataclass(frozen=True)
class MaxSimHit:
    rank: int
    document_id: str
    passage_id: str
    score: float


class ExactMaxSimIndex:
    backend = "numpy_exact_maxsim"

    def __init__(self, documents: MultiVectorResult, *, aggregation: str = "max_passage", max_cells: int = 1_000_000):
        if documents.role is not MultiVectorRole.DOCUMENT:
            raise ValueError("Exact MaxSim index requires document representations")
        if aggregation != "max_passage":
            raise ValueError("Unsupported multi-vector aggregation")
        self.documents = documents; self.aggregation = aggregation; self.max_cells = max_cells

    def search(self, query: MultiVectorResult, *, k: int) -> list[MaxSimHit]:
        if query.role is not MultiVectorRole.QUERY or query.embeddings.values.shape[0] != 1:
            raise ValueError("Exact MaxSim search requires one query row")
        if query.embeddings.representation != self.documents.embeddings.representation:
            raise ValueError("Multi-vector representation identity or dimensions drifted")
        q = query.embeddings.values[0, query.embeddings.mask[0]]
        passage_scores = []
        for row, (passage_id, document_id) in enumerate(zip(self.documents.embeddings.passage_ids, self.documents.embeddings.document_ids, strict=True)):
            d = self.documents.embeddings.values[row, self.documents.embeddings.mask[row]]
            if q.shape[0] * d.shape[0] > self.max_cells:
                raise ValueError("Exact MaxSim cell bound exceeded")
            score = float((q @ d.T).max(axis=1).sum())
            passage_scores.append((score, document_id, passage_id))
        best = {}
        for score, document_id, passage_id in passage_scores:
            current = best.get(document_id)
            if current is None or score > current[0] or (score == current[0] and passage_id < current[1]):
                best[document_id] = (score, passage_id)
        ordered = sorted(((score, doc, passage) for doc, (score, passage) in best.items()), key=lambda x: (-x[0], x[1], x[2]))
        return [MaxSimHit(i + 1, doc, passage, score) for i, (score, doc, passage) in enumerate(ordered[:k])]

    def search_batch(self, queries: MultiVectorResult, *, k: int) -> list[list[MaxSimHit]]:
        if queries.role is not MultiVectorRole.QUERY:
            raise ValueError("Exact MaxSim batch search requires query representations")
        results = []
        for row in range(queries.embeddings.values.shape[0]):
            batch = queries.embeddings
            one = type(batch)(batch.values[row:row + 1], batch.mask[row:row + 1], (batch.item_ids[row],), (batch.passage_ids[row],), (batch.document_ids[row],), batch.representation)
            results.append(self.search(type(queries)(one, queries.role, queries.route, queries.provider, queries.model_name, queries.model_revision, queries.latency_ms, queries.peak_vram_bytes), k=k))
        return results
