"""Fixed one-query/four-document no-publish neural sparse smoke fixture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

DATASET_VERSION = "opensearch-neural-sparse-smoke-v0"


@dataclass(frozen=True)
class SmokeDocument:
    document_id: str
    text: str


@dataclass(frozen=True)
class SmokeQuery:
    query_id: str
    text: str
    relevant_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class NeuralSparseSmokeFixture:
    dataset_version: str
    fixture_only: bool
    publish: bool
    leaderboard_publish: bool
    evidence_tier: str
    score_validity: str
    documents: tuple[SmokeDocument, ...]
    queries: tuple[SmokeQuery, ...]
    fixture_sha256: str


def load_opensearch_neural_sparse_smoke() -> NeuralSparseSmokeFixture:
    documents = (
        SmokeDocument("doc-cache-gold", "A result cache stores computed responses so repeated requests avoid recomputation."),
        SmokeDocument("doc-cache-hard", "Exponential backoff delays repeated requests after transient failures."),
        SmokeDocument("doc-index", "An inverted index maps terms to documents for efficient lexical retrieval."),
        SmokeDocument("doc-queue", "A durable message queue preserves events until consumers acknowledge them."),
    )
    queries = (SmokeQuery("query-cache", "How can repeated calls avoid recomputing the same result?", ("doc-cache-gold",)),)
    payload = json.dumps(
        {"documents": [asdict(item) for item in documents], "queries": [asdict(item) for item in queries]},
        sort_keys=True, separators=(",", ":"),
    )
    return NeuralSparseSmokeFixture(
        dataset_version=DATASET_VERSION, fixture_only=True, publish=False, leaderboard_publish=False,
        evidence_tier="smoke", score_validity="smoke_only", documents=documents, queries=queries,
        fixture_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


__all__ = ["DATASET_VERSION", "load_opensearch_neural_sparse_smoke"]
