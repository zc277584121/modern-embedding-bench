"""Deterministic multi-vector retrieval fixture."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json

DATASET_VERSION = "multi-vector-retrieval-fixture-v0"
REPRESENTATION_ID = "multi-vector-fixture-v1"
DIMENSIONS = 4

@dataclass(frozen=True)
class MVDocument:
    document_id: str
    passage_id: str
    text: str
    vectors: tuple[tuple[float, ...], ...]
    diagnostic_slice: str

@dataclass(frozen=True)
class MVQuery:
    query_id: str
    text: str
    vectors: tuple[tuple[float, ...], ...]
    diagnostic_slice: str

@dataclass(frozen=True)
class MVQrel:
    query_id: str
    document_id: str

@dataclass(frozen=True)
class MVHardNegative:
    query_id: str
    document_id: str
    negative_family: str

def load_multi_vector_fixture():
    docs = (
        MVDocument("doc-evidence", "p-evidence", "fine grained evidence", ((1,0,0,0),(0,1,0,0)), "fine_grained_evidence"),
        MVDocument("doc-evidence", "p-evidence-detail", "independent supporting detail", ((0,0,1,0),(0,1,0,0)), "fine_grained_evidence"),
        MVDocument("doc-long", "p-long", "long document local match", ((0,0,1,0),(0,0,0,1)), "long_document_local_match"),
        MVDocument("doc-entity", "p-entity", "entity attribute binding", ((1,0,0,0),(0,0,0,1)), "entity_attribute_binding"),
        MVDocument("doc-compose", "p-compose", "compositional constraints", ((0,1,0,0),(0,0,1,0.25)), "compositional_constraints"),
        MVDocument("doc-code", "p-code", "code tool docs", ((0,0,1,0),(0,1,0,-0.25)), "code_tool_docs"),
        MVDocument("doc-mean-trap", "p-mean-trap", "mean pooled distractor", ((0.8,0.8,0,0),), "mean_pooling_trap"),
    )
    queries = (
        MVQuery("q-evidence", "evidence", ((1,0,0,0),(0,1,0,0)), "fine_grained_evidence"),
        MVQuery("q-long", "local", ((0,0,1,0),(0,0,0,1)), "long_document_local_match"),
        MVQuery("q-entity", "binding", ((1,0,0,0),(0,0,0,1)), "entity_attribute_binding"),
        MVQuery("q-compose", "compose", ((0,1,0,0),(0,0,1,0.25)), "compositional_constraints"),
        MVQuery("q-code", "code", ((0,0,1,0),(0,1,0,-0.25)), "code_tool_docs"),
    )
    qrels = tuple(MVQrel(query_id, document_id) for query_id, document_id in (
        ("q-evidence", "doc-evidence"), ("q-long", "doc-long"), ("q-entity", "doc-entity"),
        ("q-compose", "doc-compose"), ("q-code", "doc-code"),
    ))
    hard_negatives = tuple(MVHardNegative(query_id, document_id, "authored_near_neighbor") for query_id, document_id in (
        ("q-evidence", "doc-long"), ("q-evidence", "doc-entity"), ("q-long", "doc-code"),
        ("q-long", "doc-compose"), ("q-entity", "doc-compose"), ("q-entity", "doc-code"),
        ("q-compose", "doc-entity"), ("q-compose", "doc-code"), ("q-code", "doc-long"),
        ("q-code", "doc-compose"),
    ))
    doc_ids = [d.document_id for d in docs]
    passage_ids = [d.passage_id for d in docs]
    query_ids_list = [q.query_id for q in queries]
    if len(set(passage_ids)) != len(passage_ids):
        raise ValueError("Fixture passage ids must be unique")
    if len(set(query_ids_list)) != len(query_ids_list):
        raise ValueError("Fixture query ids must be unique")
    document_passage_keys = [(d.document_id, d.passage_id) for d in docs]
    if len(set(document_passage_keys)) != len(document_passage_keys):
        raise ValueError("Fixture document/passage identities must be unique")
    qrel_rows = [(row.query_id, row.document_id) for row in qrels]
    hard_negative_rows = [(row.query_id, row.document_id) for row in hard_negatives]
    if len(set(qrel_rows)) != len(qrel_rows):
        raise ValueError("Fixture qrels must be unique")
    if len(set(hard_negative_rows)) != len(hard_negative_rows):
        raise ValueError("Fixture hard negatives must be unique")
    qrel_keys = {(row.query_id, row.document_id) for row in qrels}
    hard_negative_keys = {(row.query_id, row.document_id) for row in hard_negatives}
    if qrel_keys & hard_negative_keys:
        raise ValueError("Fixture qrels and hard negatives must not overlap")
    query_ids = {q.query_id for q in queries}
    if any(query_id not in query_ids or document_id not in doc_ids for query_id, document_id in qrel_keys | hard_negative_keys):
        raise ValueError("Fixture qrels reference unknown documents")
    if {row.query_id for row in qrels} != query_ids:
        raise ValueError("Every fixture query must have an independent qrel")
    if {row.query_id for row in hard_negatives} != query_ids:
        raise ValueError("Every fixture query must have an independent hard negative")
    payload = json.dumps({"docs": [d.__dict__ for d in docs], "queries": [q.__dict__ for q in queries], "qrels": [q.__dict__ for q in qrels], "hard_negatives": [n.__dict__ for n in hard_negatives]}, sort_keys=True, default=list)
    return docs, queries, qrels, hard_negatives, hashlib.sha256(payload.encode()).hexdigest()
