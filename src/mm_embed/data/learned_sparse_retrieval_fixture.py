"""Deterministic learned-sparse retrieval contract fixture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any


DATASET_VERSION = "learned-sparse-retrieval-fixture-v0"
REPRESENTATION_ID = "learned-sparse-fixture-representation-v1"
VOCABULARY_ID = "learned-sparse-fixture-vocabulary-v1"
DIMENSIONS = 32
LONG_DOCUMENT_MIN_WORDS = 300
DIAGNOSTIC_SLICES = (
    "exact_entity_product_name",
    "code_tool_identifier",
    "long_tail_term",
    "lexical_mismatch_document_expansion",
    "long_document_local_evidence",
)


@dataclass(frozen=True)
class SparseFixtureDocument:
    document_id: str
    text: str
    diagnostic_slice: str
    activations: tuple[tuple[int, float], ...]
    static_lookup_activations: tuple[tuple[int, float], ...] | None = None


@dataclass(frozen=True)
class SparseFixtureQuery:
    query_id: str
    text: str
    diagnostic_slice: str
    relevant_document_ids: tuple[str, ...]
    hard_negative_ids: tuple[str, ...]
    activations: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class LearnedSparseRetrievalFixture:
    dataset_version: str
    fixture_only: bool
    publish: bool
    leaderboard_publish: bool
    evidence_tier: str
    score_validity: str
    network: str
    representation_id: str
    vocabulary_id: str
    dimensions: int
    documents: tuple[SparseFixtureDocument, ...]
    queries: tuple[SparseFixtureQuery, ...]
    label_sha256: str
    fixture_sha256: str


_LONG_DOCUMENT_PREFIX = """
This operations handbook describes the routine inspection cycle for the North Annex control room. Operators begin each shift by confirming the wall clock, signing the paper log, checking the ventilation indicator, and comparing the cabinet seals with the previous shift record. The handbook then explains how to inspect power supplies, verify cable labels, review temperature history, and record harmless cosmetic damage without interrupting service. Each checklist item uses the same neutral terminology so that technicians can compare notes across facilities and across quarterly maintenance windows.

The monitoring chapter covers ordinary status lights, fan noise, dust filters, spare fuse inventory, grounding straps, and the sequence used to acknowledge noncritical alerts. It reminds readers that an alert should be copied into the incident worksheet before any control is touched. It also describes how two operators confirm panel names aloud, how portable meters are checked against a reference source, and how temporary labels are removed after maintenance. These procedures provide shared background and do not identify the control needed for any particular subsystem.

The communications chapter describes shift handoff, radio checks, escalation contacts, visitor access, and the storage of printed diagrams. During handoff, the outgoing operator reviews open work orders, deferred cleaning tasks, weather notices, and planned generator exercises. The incoming operator confirms that doors latch correctly, emergency lighting remains available, and the maintenance cart contains insulated gloves, a flashlight, blank tags, and current forms. None of these general steps changes the state of a relay or names the subsystem addressed by the local troubleshooting evidence.

The preventive maintenance chapter lists monthly tasks for cabinet hinges, terminal covers, filter housings, indicator lenses, and document holders. It explains that technicians should photograph existing conditions, avoid overtightening fasteners, preserve wire routing, and return unused parts to labeled bins. A separate paragraph covers housekeeping around raised floors, including removal of packaging, verification of clear walkways, and inspection for moisture. The text is intentionally repetitive because the same safety and documentation habits apply throughout the facility.

The troubleshooting chapter begins with general precautions. Operators must identify the affected subsystem, compare the event time with the monitoring log, confirm that no authorized test is underway, and ask a second operator to read the panel designation. They should not infer a corrective action from cabinet color, nearby equipment, or an old handwritten note. Instead, they must locate the precise local instruction embedded in the relevant equipment section and record the action exactly as written.
""".strip()

_LONG_DOCUMENT_SUFFIX = """
After applying a local instruction, operators observe the status display for one full cycle, record the outcome, and leave unrelated controls unchanged. If the expected state does not return, they stop and escalate rather than improvising. The closing checklist requires both operators to sign the worksheet, restore protective covers, return tools, and attach the event excerpt to the shift log.

The appendix contains ordinary reference material about abbreviations, cabinet numbering, approved pens, archive folders, inspection intervals, and replacement label stock. It repeats that only the equipment-specific sentence in the troubleshooting section determines the requested action. General background, nearby examples, and visually similar controls are not substitutes for that local evidence.
""".strip()


def _long_document_text(local_evidence: str) -> str:
    return f"{_LONG_DOCUMENT_PREFIX}\n\nLOCAL EQUIPMENT EVIDENCE: {local_evidence}\n\n{_LONG_DOCUMENT_SUFFIX}"


def _documents() -> tuple[SparseFixtureDocument, ...]:
    return (
        SparseFixtureDocument("doc-entity-gold", "Orchid Matrix XJ-42 calibration guide.", DIAGNOSTIC_SLICES[0], ((0, 3.0), (20, 0.5))),
        SparseFixtureDocument("doc-entity-hard", "Orchid Matrix XJ-24 calibration guide.", DIAGNOSTIC_SLICES[0], ((1, 3.0), (20, 0.5))),
        SparseFixtureDocument("doc-code-gold", "Use uv sync --frozen before running the workspace checks.", DIAGNOSTIC_SLICES[1], ((2, 3.0), (21, 0.5))),
        SparseFixtureDocument("doc-code-hard", "Use uv lock before reviewing dependency changes.", DIAGNOSTIC_SLICES[1], ((3, 3.0), (21, 0.5))),
        SparseFixtureDocument("doc-tail-gold", "The zygomaticomaxillary complex fracture protocol is documented here.", DIAGNOSTIC_SLICES[2], ((4, 3.0), (22, 0.5))),
        SparseFixtureDocument("doc-tail-hard", "A general maxillofacial fracture protocol is documented here.", DIAGNOSTIC_SLICES[2], ((5, 3.0), (22, 0.5))),
        SparseFixtureDocument(
            "doc-expansion-gold",
            "The service keeps requests briefly so repeated calls avoid recomputation.",
            DIAGNOSTIC_SLICES[3],
            ((6, 3.0), (23, 0.5)),
            ((10, 3.0), (23, 0.5)),
        ),
        SparseFixtureDocument(
            "doc-expansion-hard",
            "The request result cache setting is disabled while failed calls use exponential backoff.",
            DIAGNOSTIC_SLICES[3],
            ((7, 3.0), (23, 0.5)),
            ((6, 3.0), (23, 0.5)),
        ),
        SparseFixtureDocument(
            "doc-long-gold",
            _long_document_text("The amber relay resets subsystem Kappa."),
            DIAGNOSTIC_SLICES[4],
            ((8, 3.0), (24, 0.5)),
        ),
        SparseFixtureDocument(
            "doc-long-hard",
            _long_document_text("The amber relay tests subsystem Lambda."),
            DIAGNOSTIC_SLICES[4],
            ((9, 3.0), (24, 0.5)),
        ),
        SparseFixtureDocument("doc-tie-a", "Unrelated deterministic tie control A.", "tie_control", ((31, 1.0),)),
        SparseFixtureDocument("doc-tie-b", "Unrelated deterministic tie control B.", "tie_control", ((31, 1.0),)),
    )


def _queries() -> tuple[SparseFixtureQuery, ...]:
    return (
        SparseFixtureQuery("query-entity", "Orchid Matrix XJ-42", DIAGNOSTIC_SLICES[0], ("doc-entity-gold",), ("doc-entity-hard",), ((0, 1.0),)),
        SparseFixtureQuery("query-code", "Which command installs the locked uv workspace?", DIAGNOSTIC_SLICES[1], ("doc-code-gold",), ("doc-code-hard",), ((2, 1.0),)),
        SparseFixtureQuery("query-tail", "zygomaticomaxillary complex", DIAGNOSTIC_SLICES[2], ("doc-tail-gold",), ("doc-tail-hard",), ((4, 1.0),)),
        SparseFixtureQuery("query-expansion", "request result cache", DIAGNOSTIC_SLICES[3], ("doc-expansion-gold",), ("doc-expansion-hard",), ((6, 1.0),)),
        SparseFixtureQuery("query-long", "What resets subsystem Kappa?", DIAGNOSTIC_SLICES[4], ("doc-long-gold",), ("doc-long-hard",), ((8, 1.0),)),
        SparseFixtureQuery("query-tie", "deterministic tie control", "tie_control", ("doc-tie-a",), ("doc-tie-b",), ((31, 1.0),)),
    )


def _label_digest(queries: tuple[SparseFixtureQuery, ...]) -> str:
    labels = [
        {
            "query_id": query.query_id,
            "diagnostic_slice": query.diagnostic_slice,
            "relevant_document_ids": query.relevant_document_ids,
            "hard_negative_ids": query.hard_negative_ids,
        }
        for query in queries
    ]
    payload = json.dumps(labels, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fixture_digest(
    documents: tuple[SparseFixtureDocument, ...],
    queries: tuple[SparseFixtureQuery, ...],
) -> str:
    payload = json.dumps(
        {"documents": [asdict(document) for document in documents], "queries": [asdict(query) for query in queries]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_learned_sparse_retrieval_fixture() -> LearnedSparseRetrievalFixture:
    documents = _documents()
    queries = _queries()
    fixture = LearnedSparseRetrievalFixture(
        dataset_version=DATASET_VERSION,
        fixture_only=True,
        publish=False,
        leaderboard_publish=False,
        evidence_tier="fixture",
        score_validity="contract_only",
        network="forbidden",
        representation_id=REPRESENTATION_ID,
        vocabulary_id=VOCABULARY_ID,
        dimensions=DIMENSIONS,
        documents=documents,
        queries=queries,
        label_sha256=_label_digest(queries),
        fixture_sha256=_fixture_digest(documents, queries),
    )
    validate_learned_sparse_retrieval_fixture(fixture)
    return fixture


def validate_learned_sparse_retrieval_fixture(fixture: LearnedSparseRetrievalFixture) -> None:
    if (
        fixture.dataset_version != DATASET_VERSION
        or not fixture.fixture_only
        or fixture.publish
        or fixture.leaderboard_publish
        or fixture.evidence_tier != "fixture"
        or fixture.score_validity != "contract_only"
        or fixture.network != "forbidden"
    ):
        raise ValueError("Learned-sparse fixture publication contract changed")
    if (
        fixture.representation_id != REPRESENTATION_ID
        or fixture.vocabulary_id != VOCABULARY_ID
        or fixture.dimensions != DIMENSIONS
    ):
        raise ValueError("Learned-sparse fixture representation identity changed")
    document_ids = [document.document_id for document in fixture.documents]
    query_ids = [query.query_id for query in fixture.queries]
    if len(document_ids) != len(set(document_ids)) or len(query_ids) != len(set(query_ids)):
        raise ValueError("Learned-sparse fixture ids must be unique")
    if {query.diagnostic_slice for query in fixture.queries if query.diagnostic_slice != "tie_control"} != set(DIAGNOSTIC_SLICES):
        raise ValueError("Learned-sparse fixture diagnostic slices changed")
    known_documents = set(document_ids)
    for item in (*fixture.documents, *fixture.queries):
        if not item.activations:
            raise ValueError("Learned-sparse fixture rows must not be empty")
        coordinates = [coordinate for coordinate, _ in item.activations]
        values = [value for _, value in item.activations]
        if len(coordinates) != len(set(coordinates)) or any(coordinate < 0 or coordinate >= DIMENSIONS for coordinate in coordinates):
            raise ValueError("Learned-sparse fixture activation coordinates are invalid")
        if any(value <= 0 for value in values):
            raise ValueError("Learned-sparse fixture activation values must be positive")
    for query in fixture.queries:
        if not query.relevant_document_ids or not query.hard_negative_ids:
            raise ValueError("Every learned-sparse query requires fixed qrels and hard negatives")
        if set(query.relevant_document_ids) & set(query.hard_negative_ids):
            raise ValueError("Learned-sparse qrels and hard negatives must be disjoint")
        if not set((*query.relevant_document_ids, *query.hard_negative_ids)).issubset(known_documents):
            raise ValueError("Learned-sparse labels reference an unknown document")
    if fixture.label_sha256 != _label_digest(fixture.queries):
        raise ValueError("Learned-sparse fixture label digest changed")
    if fixture.fixture_sha256 != _fixture_digest(fixture.documents, fixture.queries):
        raise ValueError("Learned-sparse fixture content digest changed")
    long_documents = {
        document.document_id: document
        for document in fixture.documents
        if document.diagnostic_slice == "long_document_local_evidence"
    }
    if set(long_documents) != {"doc-long-gold", "doc-long-hard"}:
        raise ValueError("Learned-sparse long-document pair changed")
    gold_text = long_documents["doc-long-gold"].text
    hard_text = long_documents["doc-long-hard"].text
    if min(len(gold_text.split()), len(hard_text.split())) < LONG_DOCUMENT_MIN_WORDS:
        raise ValueError("Learned-sparse long-document texts are too short")
    if _LONG_DOCUMENT_PREFIX not in gold_text or _LONG_DOCUMENT_PREFIX not in hard_text:
        raise ValueError("Learned-sparse long documents must share their background")
    if _LONG_DOCUMENT_SUFFIX not in gold_text or _LONG_DOCUMENT_SUFFIX not in hard_text:
        raise ValueError("Learned-sparse long documents must share their trailing background")
    for text, evidence in (
        (gold_text, "The amber relay resets subsystem Kappa."),
        (hard_text, "The amber relay tests subsystem Lambda."),
    ):
        evidence_position = text.index(evidence) / len(text)
        if not 0.6 < evidence_position < 0.9:
            raise ValueError("Learned-sparse local evidence must appear late inside the long document")


def fixture_to_dict(fixture: LearnedSparseRetrievalFixture) -> dict[str, Any]:
    return asdict(fixture)


def fixture_with_publication(
    fixture: LearnedSparseRetrievalFixture,
    *,
    fixture_only: bool | None = None,
    publish: bool | None = None,
    leaderboard_publish: bool | None = None,
) -> LearnedSparseRetrievalFixture:
    return replace(
        fixture,
        fixture_only=fixture.fixture_only if fixture_only is None else fixture_only,
        publish=fixture.publish if publish is None else publish,
        leaderboard_publish=(
            fixture.leaderboard_publish if leaderboard_publish is None else leaderboard_publish
        ),
    )


__all__ = [
    "DATASET_VERSION",
    "DIAGNOSTIC_SLICES",
    "DIMENSIONS",
    "LONG_DOCUMENT_MIN_WORDS",
    "LearnedSparseRetrievalFixture",
    "REPRESENTATION_ID",
    "VOCABULARY_ID",
    "fixture_to_dict",
    "fixture_with_publication",
    "load_learned_sparse_retrieval_fixture",
    "validate_learned_sparse_retrieval_fixture",
]
