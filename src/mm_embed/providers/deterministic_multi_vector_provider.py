from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from mm_embed.data.multi_vector_fixture import DIMENSIONS, REPRESENTATION_ID, load_multi_vector_fixture
from mm_embed.providers.multi_vector_base import *

class DeterministicMultiVectorFixtureProvider:
    name = "deterministic_multi_vector_fixture"; model = "multi-vector-contract-fixture"; revision = "fixture-v1"
    representation = MultiVectorRepresentation(REPRESENTATION_ID, DIMENSIONS)
    query_route = MultiVectorRoute.QUERY; document_route = MultiVectorRoute.DOCUMENT
    def __init__(self, **kwargs):
        docs, queries, _, _, _ = load_multi_vector_fixture(); self._docs = {d.text:d for d in docs}; self._queries = {q.text:q for q in queries}
    def encode_multi_vector_query(self, text, *, item_id):
        q = self._queries.get(text)
        if q is None or q.query_id != item_id: raise ValueError("Unknown or mismatched multi-vector query")
        return self._result((q.vectors,), (item_id,), (item_id,), (item_id,), MultiVectorRole.QUERY, self.query_route)
    def encode_multi_vector_passages(self, texts: Sequence[str], *, passage_ids: Sequence[str], document_ids: Sequence[str]):
        if len(texts) != len(passage_ids) or len(texts) != len(document_ids): raise ValueError("Passage ids must align")
        rows = [self._docs.get(t) for t in texts]
        if any(r is None or r.passage_id != p or r.document_id != d for r,p,d in zip(rows, passage_ids, document_ids, strict=True)): raise ValueError("Unknown or mismatched multi-vector passage")
        return self._result([r.vectors for r in rows], tuple(passage_ids), tuple(passage_ids), tuple(document_ids), MultiVectorRole.DOCUMENT, self.document_route)
    def encode_multi_vector_queries(self, texts: Sequence[str], *, item_ids: Sequence[str]):
        if len(texts) != len(item_ids): raise ValueError("Query ids must align")
        rows = [self._queries.get(t) for t in texts]
        if any(r is None or r.query_id != i for r, i in zip(rows, item_ids, strict=True)):
            raise ValueError("Unknown or mismatched multi-vector query")
        return self._result([r.vectors for r in rows], tuple(item_ids), tuple(item_ids), tuple(item_ids), MultiVectorRole.QUERY, self.query_route)
    def _result(self, rows, item_ids, passage_ids, document_ids, role, route):
        max_tokens = max(len(row) for row in rows)
        arr = np.zeros((len(rows), max_tokens, DIMENSIONS), dtype=np.float32)
        mask = np.zeros((len(rows), max_tokens), dtype=bool)
        for i, row in enumerate(rows):
            arr[i, :len(row)] = row
            mask[i, :len(row)] = True
        return MultiVectorResult(MultiVectorBatch(arr, mask, item_ids, passage_ids, document_ids, self.representation), role, route, self.name, self.model, self.revision, 0.0, 0)
