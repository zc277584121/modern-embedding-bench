"""Deterministic no-publish fixture for grouped late-chunking retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable


DATASET_VERSION = "late-chunking-retrieval-fixture-v0"
NORMALIZATION = "unicode-nfc-lf-v1"
SOURCE_TYPE = "self_created_fixture"
LICENSE_STATUS = "not_for_publication"
SPLIT = "fixture_only"
CHUNKER_VERSION = "fixture-whitespace-v1"
LAYOUT_IDS = (
    "fixed_192_v1",
    "fixed_192_overlap_48_v1",
    "structure_adaptive_fixture_v1",
)
FAILURE_FAMILIES = (
    "split_antecedent_reference",
    "ambiguous_local_policy",
    "definition_example_separation",
    "section_title_dependence",
)


@dataclass(frozen=True)
class ParentDocument:
    """Normalized parent document owned by the fixture."""

    document_id: str
    split: str
    title: str
    text: str
    source_type: str
    source_revision: str
    normalization: str
    license_status: str
    sha256: str
    section_token_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EvidenceSpan:
    """Canonical half-open character span into a normalized parent."""

    span_id: str
    document_id: str
    role: str
    char_start: int
    char_end: int
    text: str
    text_sha256: str


@dataclass(frozen=True)
class RetrievalQuery:
    """Fixture query with explicit context and evidence identity."""

    query_id: str
    split: str
    text: str
    family: str
    gold_document_id: str
    pair_document_id: str
    evidence_span_ids: tuple[str, ...]
    required_context_span_ids: tuple[str, ...]
    context_required: bool
    evidence_slot: str
    lexical_shortcut_present: bool = False


@dataclass(frozen=True)
class RetrievalChunk:
    """Canonical retrieval unit for one layout."""

    layout_id: str
    document_id: str
    chunk_id: str
    chunk_index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    text: str
    chunker_version: str
    text_sha256: str


@dataclass(frozen=True)
class ChunkLayout:
    """Ordered document groups for one segmentation policy."""

    layout_id: str
    chunks: tuple[RetrievalChunk, ...]


@dataclass(frozen=True)
class Qrel:
    """Generated relevance label for a canonical chunk."""

    query_id: str
    layout_id: str
    chunk_id: str
    relevance: int
    evidence_span_id: str


@dataclass(frozen=True)
class HardNegative:
    """Curated deterministic hard-negative link."""

    query_id: str
    layout_id: str
    chunk_id: str
    negative_family: str
    reason: str
    false_negative_review: str


@dataclass(frozen=True)
class LateChunkingRetrievalFixture:
    """Complete deterministic fixture and its generated labels."""

    dataset_version: str
    split: str
    fixture_only: bool
    license_status: str
    leaderboard_publish: bool
    network: str
    documents: tuple[ParentDocument, ...]
    spans: tuple[EvidenceSpan, ...]
    queries: tuple[RetrievalQuery, ...]
    layouts: tuple[ChunkLayout, ...]
    qrels: tuple[Qrel, ...]
    hard_negatives: tuple[HardNegative, ...]
    label_sha256: str


_DOCUMENT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "document_id": "doc_aster_relay",
        "title": "Aster Relay Operations",
        "family": "split_antecedent_reference",
        "pair_document_id": "doc_boreal_relay",
        "scope_a": "In Aster Relay the runner alias identifies the intake reconciler component.",
        "evidence_a": "The runner suspends after the amber seven signal appears.",
        "query_a": "Which signal suspends the Aster intake reconciler?",
        "context_b": "The checksum clerk alias refers to the Aster manifest verifier.",
        "evidence_b": "For Aster Relay the manifest verifier retries exactly two times.",
        "query_b": "How many retries does the Aster manifest verifier use?",
        "control_b": True,
    },
    {
        "document_id": "doc_boreal_relay",
        "title": "Boreal Relay Operations",
        "family": "split_antecedent_reference",
        "pair_document_id": "doc_aster_relay",
        "scope_a": "In Boreal Relay the runner alias identifies the export reconciler component.",
        "evidence_a": "The runner suspends after the cobalt eight signal appears.",
        "query_a": "Which signal suspends the Boreal export reconciler?",
        "context_b": "The checksum clerk alias refers to the Boreal manifest verifier.",
        "evidence_b": "The clerk retries exactly four times before sending a review notice.",
        "query_b": "How many retries does the Boreal manifest verifier use?",
        "control_b": False,
    },
    {
        "document_id": "doc_cinder_archive",
        "title": "Cinder Archive Policy",
        "family": "ambiguous_local_policy",
        "pair_document_id": "doc_delta_archive",
        "scope_a": "This policy governs Cinder archive imports in the northern region.",
        "evidence_a": "The worker waits twelve minutes before the second attempt.",
        "query_a": "What wait applies to northern Cinder archive imports?",
        "context_b": "Cinder deletion exports are assigned to the silver retention class.",
        "evidence_b": "For Cinder deletion exports the retention period is thirty days.",
        "query_b": "What is the retention period for Cinder deletion exports?",
        "control_b": True,
    },
    {
        "document_id": "doc_delta_archive",
        "title": "Delta Archive Policy",
        "family": "ambiguous_local_policy",
        "pair_document_id": "doc_cinder_archive",
        "scope_a": "This policy governs Delta archive imports in the southern region.",
        "evidence_a": "The worker waits twenty minutes before the second attempt.",
        "query_a": "What wait applies to southern Delta archive imports?",
        "context_b": "Delta deletion exports are assigned to the bronze retention class.",
        "evidence_b": "The retention period is forty five days after confirmation.",
        "query_b": "What is the retention period for Delta deletion exports?",
        "control_b": False,
    },
    {
        "document_id": "doc_ember_ledger",
        "title": "Ember Ledger Definitions",
        "family": "definition_example_separation",
        "pair_document_id": "doc_fjord_ledger",
        "scope_a": "Within Ember Ledger a quiet cycle means no signed entries for six hours.",
        "evidence_a": "This example therefore enters review at the next checkpoint.",
        "query_a": "When does the Ember quiet-cycle example enter review?",
        "context_b": "An Ember warm exception is a correction received during reconciliation.",
        "evidence_b": "For Ember Ledger a warm exception remains open for nine hours.",
        "query_b": "How long does an Ember warm exception remain open?",
        "control_b": True,
    },
    {
        "document_id": "doc_fjord_ledger",
        "title": "Fjord Ledger Definitions",
        "family": "definition_example_separation",
        "pair_document_id": "doc_ember_ledger",
        "scope_a": "Within Fjord Ledger a quiet cycle means no signed entries for ten hours.",
        "evidence_a": "This example therefore enters review after the nightly close.",
        "query_a": "When does the Fjord quiet-cycle example enter review?",
        "context_b": "A Fjord warm exception is a correction received after reconciliation.",
        "evidence_b": "The exception remains open for fourteen hours before escalation.",
        "query_b": "How long does a Fjord warm exception remain open?",
        "control_b": False,
    },
    {
        "document_id": "doc_glint_console",
        "title": "Glint Console Procedures",
        "family": "section_title_dependence",
        "pair_document_id": "doc_harbor_console",
        "scope_a": "Section Restore Preview applies only to Glint Console snapshot previews.",
        "evidence_a": "The operation requires the violet approval channel before execution.",
        "query_a": "Which approval channel applies to Glint snapshot previews?",
        "context_b": "Section Export Cleanup covers Glint Console temporary export bundles.",
        "evidence_b": "Glint temporary export bundles are removed after eighteen hours.",
        "query_b": "When are Glint temporary export bundles removed?",
        "control_b": True,
    },
    {
        "document_id": "doc_harbor_console",
        "title": "Harbor Console Procedures",
        "family": "section_title_dependence",
        "pair_document_id": "doc_glint_console",
        "scope_a": "Section Restore Preview applies only to Harbor Console snapshot previews.",
        "evidence_a": "The operation requires the orange approval channel before execution.",
        "query_a": "Which approval channel applies to Harbor snapshot previews?",
        "context_b": "Section Export Cleanup covers Harbor Console temporary export bundles.",
        "evidence_b": "The bundles are removed after twenty six hours by the cleanup worker.",
        "query_b": "When are Harbor temporary export bundles removed?",
        "control_b": False,
    },
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Apply the fixture's authoritative Unicode and newline normalization."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _insert_phrase(tokens: list[str], start: int, phrase: str) -> tuple[int, int]:
    phrase_tokens = phrase.split()
    end = start + len(phrase_tokens)
    tokens[start:end] = phrase_tokens
    return start, end


def _token_offsets(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))


def _char_range(offsets: tuple[tuple[int, int], ...], token_start: int, token_end: int) -> tuple[int, int]:
    return offsets[token_start][0], offsets[token_end - 1][1]


def _build_documents_and_items() -> tuple[
    tuple[ParentDocument, ...],
    tuple[EvidenceSpan, ...],
    tuple[RetrievalQuery, ...],
]:
    documents: list[ParentDocument] = []
    spans: list[EvidenceSpan] = []
    queries: list[RetrievalQuery] = []

    for document_index, spec in enumerate(_DOCUMENT_SPECS):
        document_id = str(spec["document_id"])
        slug = document_id.removeprefix("doc_")
        tokens = [f"{slug}_unit_{index:04d}" for index in range(768)]
        section_ranges = (
            ((0, 168), (168, 360), (360, 552), (552, 768))
            if document_index % 2 == 0
            else ((0, 176), (176, 368), (368, 560), (560, 768))
        )
        for section_index, (start, _) in enumerate(section_ranges):
            _insert_phrase(tokens, start, f"Section {section_index + 1} {spec['title']}")

        phrase_ranges = {
            "context_a": _insert_phrase(tokens, 40, str(spec["scope_a"])),
            "evidence_a": _insert_phrase(tokens, 300, str(spec["evidence_a"])),
            "context_b": _insert_phrase(tokens, 340, str(spec["context_b"])),
            "evidence_b": _insert_phrase(tokens, 520, str(spec["evidence_b"])),
        }
        _insert_phrase(
            tokens,
            250,
            "A nearby example mentions urgent retry approval but applies to a different operation.",
        )
        _insert_phrase(tokens, 650, "A closing note mentions archival timing but is not an answer to either query.")

        text = normalize_text(" ".join(tokens))
        offsets = _token_offsets(text)
        document = ParentDocument(
            document_id=document_id,
            split=SPLIT,
            title=str(spec["title"]),
            text=text,
            source_type=SOURCE_TYPE,
            source_revision=DATASET_VERSION,
            normalization=NORMALIZATION,
            license_status=LICENSE_STATUS,
            sha256=_sha256(text),
            section_token_ranges=section_ranges,
        )
        documents.append(document)

        for slot in ("a", "b"):
            evidence_token_start, evidence_token_end = phrase_ranges[f"evidence_{slot}"]
            evidence_char_start, evidence_char_end = _char_range(offsets, evidence_token_start, evidence_token_end)
            evidence_text = text[evidence_char_start:evidence_char_end]
            evidence_span_id = f"span_{slug}_{slot}_evidence"
            spans.append(
                EvidenceSpan(
                    span_id=evidence_span_id,
                    document_id=document_id,
                    role="minimal_evidence",
                    char_start=evidence_char_start,
                    char_end=evidence_char_end,
                    text=evidence_text,
                    text_sha256=_sha256(evidence_text),
                )
            )

            is_control = slot == "b" and bool(spec["control_b"])
            context_span_ids: tuple[str, ...] = ()
            if not is_control:
                context_token_start, context_token_end = phrase_ranges[f"context_{slot}"]
                context_char_start, context_char_end = _char_range(offsets, context_token_start, context_token_end)
                context_text = text[context_char_start:context_char_end]
                context_span_id = f"span_{slug}_{slot}_context"
                spans.append(
                    EvidenceSpan(
                        span_id=context_span_id,
                        document_id=document_id,
                        role="required_context",
                        char_start=context_char_start,
                        char_end=context_char_end,
                        text=context_text,
                        text_sha256=_sha256(context_text),
                    )
                )
                context_span_ids = (context_span_id,)

            queries.append(
                RetrievalQuery(
                    query_id=f"q_{slug}_{slot}",
                    split=SPLIT,
                    text=str(spec[f"query_{slot}"]),
                    family=str(spec["family"]),
                    gold_document_id=document_id,
                    pair_document_id=str(spec["pair_document_id"]),
                    evidence_span_ids=(evidence_span_id,),
                    required_context_span_ids=context_span_ids,
                    context_required=not is_control,
                    evidence_slot=slot,
                )
            )

    return tuple(documents), tuple(spans), tuple(queries)


def _layout_token_ranges(document: ParentDocument, layout_id: str) -> tuple[tuple[int, int], ...]:
    if layout_id == "fixed_192_v1":
        return ((0, 192), (192, 384), (384, 576), (576, 768))
    if layout_id == "fixed_192_overlap_48_v1":
        return ((0, 192), (144, 336), (288, 480), (432, 624), (576, 768))
    if layout_id == "structure_adaptive_fixture_v1":
        return document.section_token_ranges
    raise KeyError(f"Unknown layout id: {layout_id}")


def _build_layouts(documents: tuple[ParentDocument, ...]) -> tuple[ChunkLayout, ...]:
    layouts: list[ChunkLayout] = []
    for layout_id in LAYOUT_IDS:
        chunks: list[RetrievalChunk] = []
        for document in documents:
            offsets = _token_offsets(document.text)
            for chunk_index, (token_start, token_end) in enumerate(_layout_token_ranges(document, layout_id)):
                char_start, char_end = _char_range(offsets, token_start, token_end)
                text = document.text[char_start:char_end]
                chunks.append(
                    RetrievalChunk(
                        layout_id=layout_id,
                        document_id=document.document_id,
                        chunk_id=f"{layout_id}:{document.document_id}:{chunk_index:04d}",
                        chunk_index=chunk_index,
                        token_start=token_start,
                        token_end=token_end,
                        char_start=char_start,
                        char_end=char_end,
                        text=text,
                        chunker_version=CHUNKER_VERSION,
                        text_sha256=_sha256(text),
                    )
                )
        layouts.append(ChunkLayout(layout_id=layout_id, chunks=tuple(chunks)))
    return tuple(layouts)


def _chunks_covering_span(layout: ChunkLayout, span: EvidenceSpan) -> tuple[RetrievalChunk, ...]:
    return tuple(
        chunk
        for chunk in layout.chunks
        if chunk.document_id == span.document_id
        and chunk.char_start <= span.char_start
        and chunk.char_end >= span.char_end
    )


def generate_qrels(
    queries: tuple[RetrievalQuery, ...],
    spans: tuple[EvidenceSpan, ...],
    layouts: tuple[ChunkLayout, ...],
) -> tuple[Qrel, ...]:
    """Generate relevance labels from canonical evidence containment."""
    span_by_id = {span.span_id: span for span in spans}
    qrels: list[Qrel] = []
    for layout in layouts:
        for query in queries:
            for evidence_span_id in query.evidence_span_ids:
                span = span_by_id[evidence_span_id]
                qrels.extend(
                    Qrel(
                        query_id=query.query_id,
                        layout_id=layout.layout_id,
                        chunk_id=chunk.chunk_id,
                        relevance=2,
                        evidence_span_id=evidence_span_id,
                    )
                    for chunk in _chunks_covering_span(layout, span)
                )
    return tuple(qrels)


def generate_hard_negatives(
    queries: tuple[RetrievalQuery, ...],
    spans: tuple[EvidenceSpan, ...],
    layouts: tuple[ChunkLayout, ...],
) -> tuple[HardNegative, ...]:
    """Generate three predeclared hard-negative families for every query/layout."""
    query_by_doc_slot = {(query.gold_document_id, query.evidence_slot): query for query in queries}
    span_by_id = {span.span_id: span for span in spans}
    negatives: list[HardNegative] = []

    for layout in layouts:
        chunks_by_document: dict[str, list[RetrievalChunk]] = defaultdict(list)
        for chunk in layout.chunks:
            chunks_by_document[chunk.document_id].append(chunk)

        for query in queries:
            positive_span = span_by_id[query.evidence_span_ids[0]]
            positive_chunks = _chunks_covering_span(layout, positive_span)
            positive_indices = {chunk.chunk_index for chunk in positive_chunks}

            pair_query = query_by_doc_slot[(query.pair_document_id, query.evidence_slot)]
            pair_span = span_by_id[pair_query.evidence_span_ids[0]]
            pair_positive_chunks = _chunks_covering_span(layout, pair_span)
            collision_chunk = pair_positive_chunks[0]

            neighbor_candidates = [
                chunk
                for chunk in chunks_by_document[query.gold_document_id]
                if chunk.chunk_index not in positive_indices
            ]
            neighbor_chunk = min(
                neighbor_candidates,
                key=lambda chunk: (
                    min(abs(chunk.chunk_index - index) for index in positive_indices),
                    chunk.chunk_index,
                ),
            )

            family_candidates = [
                chunk
                for chunk in chunks_by_document[query.pair_document_id]
                if chunk.chunk_id != collision_chunk.chunk_id
            ]
            family_chunk = max(family_candidates, key=lambda chunk: chunk.chunk_index)

            negatives.extend(
                (
                    HardNegative(
                        query_id=query.query_id,
                        layout_id=layout.layout_id,
                        chunk_id=collision_chunk.chunk_id,
                        negative_family="cross_document_local_text_collision",
                        reason="paired document uses nearly identical local wording under a different parent scope",
                        false_negative_review="pass",
                    ),
                    HardNegative(
                        query_id=query.query_id,
                        layout_id=layout.layout_id,
                        chunk_id=neighbor_chunk.chunk_id,
                        negative_family="misleading_gold_parent_neighbor",
                        reason=(
                            "neighboring gold-parent chunk mentions related operations "
                            "but does not contain the evidence"
                        ),
                        false_negative_review="pass",
                    ),
                    HardNegative(
                        query_id=query.query_id,
                        layout_id=layout.layout_id,
                        chunk_id=family_chunk.chunk_id,
                        negative_family="same_family_scope_collision",
                        reason="same failure family and paired scope, but the chunk cannot answer the query",
                        false_negative_review="pass",
                    ),
                )
            )
    return tuple(negatives)


def _label_digest(qrels: Iterable[Qrel], hard_negatives: Iterable[HardNegative]) -> str:
    payload = {
        "qrels": [asdict(qrel) for qrel in qrels],
        "hard_negatives": [asdict(negative) for negative in hard_negatives],
    }
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def fixture_to_dict(fixture: LateChunkingRetrievalFixture) -> dict[str, Any]:
    """Serialize the fixture into a deterministic JSON-safe mapping."""
    return asdict(fixture)


def load_late_chunking_retrieval_fixture() -> LateChunkingRetrievalFixture:
    """Build and validate the deterministic no-network fixture."""
    documents, spans, queries = _build_documents_and_items()
    layouts = _build_layouts(documents)
    qrels = generate_qrels(queries, spans, layouts)
    hard_negatives = generate_hard_negatives(queries, spans, layouts)
    fixture = LateChunkingRetrievalFixture(
        dataset_version=DATASET_VERSION,
        split=SPLIT,
        fixture_only=True,
        license_status=LICENSE_STATUS,
        leaderboard_publish=False,
        network="forbidden",
        documents=documents,
        spans=spans,
        queries=queries,
        layouts=layouts,
        qrels=qrels,
        hard_negatives=hard_negatives,
        label_sha256=_label_digest(qrels, hard_negatives),
    )
    validate_late_chunking_retrieval_fixture(fixture)
    return fixture


def validate_late_chunking_retrieval_fixture(fixture: LateChunkingRetrievalFixture) -> None:
    """Reject identity, grouping, label, and publication-contract drift."""
    if (
        fixture.dataset_version != DATASET_VERSION
        or fixture.split != SPLIT
        or not fixture.fixture_only
        or fixture.license_status != LICENSE_STATUS
        or fixture.leaderboard_publish
        or fixture.network != "forbidden"
    ):
        raise ValueError("Late-chunking fixture publication or provenance contract changed")

    if len(fixture.documents) != 8 or len(fixture.queries) != 16:
        raise ValueError("Late-chunking fixture must contain exactly 8 documents and 16 queries")
    if len({document.document_id for document in fixture.documents}) != len(fixture.documents):
        raise ValueError("Duplicate parent document ids")
    if len({query.query_id for query in fixture.queries}) != len(fixture.queries):
        raise ValueError("Duplicate query ids")
    if Counter(query.family for query in fixture.queries) != Counter({family: 4 for family in FAILURE_FAMILIES}):
        raise ValueError("Failure-family labels are not the deterministic 4-by-4 fixture")
    if sum(query.context_required for query in fixture.queries) != 12:
        raise ValueError("Fixture must contain 12 context-required queries")
    if sum(not query.context_required for query in fixture.queries) != 4:
        raise ValueError("Fixture must contain 4 irrelevant-context controls")

    document_by_id = {document.document_id: document for document in fixture.documents}
    for document in fixture.documents:
        if document.split != SPLIT or document.license_status != LICENSE_STATUS:
            raise ValueError(f"Invalid split/license for {document.document_id}")
        if document.text != normalize_text(document.text) or len(document.text.split()) != 768:
            raise ValueError(f"Invalid normalization or token count for {document.document_id}")
        if document.sha256 != _sha256(document.text):
            raise ValueError(f"Invalid parent hash for {document.document_id}")

    span_by_id = {span.span_id: span for span in fixture.spans}
    if len(span_by_id) != len(fixture.spans):
        raise ValueError("Duplicate span ids")
    for span in fixture.spans:
        document = document_by_id.get(span.document_id)
        if document is None:
            raise ValueError(f"Unknown parent for span {span.span_id}")
        reconstructed = document.text[span.char_start:span.char_end]
        if reconstructed != span.text or span.text_sha256 != _sha256(reconstructed):
            raise ValueError(f"Span text/hash does not reconstruct for {span.span_id}")

    for query in fixture.queries:
        if query.split != SPLIT or query.gold_document_id not in document_by_id:
            raise ValueError(f"Invalid query parent/split for {query.query_id}")
        if query.pair_document_id not in document_by_id or query.pair_document_id == query.gold_document_id:
            raise ValueError(f"Invalid paired parent for {query.query_id}")
        if len(query.evidence_span_ids) != 1:
            raise ValueError(f"{query.query_id} must have exactly one minimal evidence span")
        linked_spans = [span_by_id[span_id] for span_id in (*query.evidence_span_ids, *query.required_context_span_ids)]
        if any(span.document_id != query.gold_document_id for span in linked_spans):
            raise ValueError(f"Cross-document span link for {query.query_id}")
        if query.context_required != bool(query.required_context_span_ids):
            raise ValueError(f"Context-required flag/span mismatch for {query.query_id}")

    layout_by_id = {layout.layout_id: layout for layout in fixture.layouts}
    if tuple(layout_by_id) != LAYOUT_IDS:
        raise ValueError("Layout ids or order changed")
    expected_chunk_counts = {
        "fixed_192_v1": 32,
        "fixed_192_overlap_48_v1": 40,
        "structure_adaptive_fixture_v1": 32,
    }
    for layout in fixture.layouts:
        if len(layout.chunks) != expected_chunk_counts[layout.layout_id]:
            raise ValueError(f"Unexpected chunk count for {layout.layout_id}")
        seen_chunk_ids: set[str] = set()
        grouped: dict[str, list[RetrievalChunk]] = defaultdict(list)
        for chunk in layout.chunks:
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk id: {chunk.chunk_id}")
            seen_chunk_ids.add(chunk.chunk_id)
            grouped[chunk.document_id].append(chunk)
            document = document_by_id[chunk.document_id]
            if chunk.layout_id != layout.layout_id:
                raise ValueError(f"Cross-layout chunk: {chunk.chunk_id}")
            if chunk.chunk_id != f"{layout.layout_id}:{chunk.document_id}:{chunk.chunk_index:04d}":
                raise ValueError(f"Noncanonical chunk id: {chunk.chunk_id}")
            if document.text[chunk.char_start:chunk.char_end] != chunk.text:
                raise ValueError(f"Chunk text does not reconstruct: {chunk.chunk_id}")
            if chunk.text_sha256 != _sha256(chunk.text):
                raise ValueError(f"Chunk hash mismatch: {chunk.chunk_id}")
        for document in fixture.documents:
            chunks = grouped[document.document_id]
            if [chunk.chunk_index for chunk in chunks] != list(range(len(chunks))):
                raise ValueError(f"Chunk order changed for {layout.layout_id}/{document.document_id}")
            actual_ranges = tuple((chunk.token_start, chunk.token_end) for chunk in chunks)
            if actual_ranges != _layout_token_ranges(document, layout.layout_id):
                raise ValueError(f"Chunk boundaries changed for {layout.layout_id}/{document.document_id}")

    expected_qrels = generate_qrels(fixture.queries, fixture.spans, fixture.layouts)
    if fixture.qrels != expected_qrels:
        raise ValueError("Qrels do not reproduce from canonical evidence containment")
    qrel_counts = Counter(qrel.layout_id for qrel in fixture.qrels)
    if qrel_counts != Counter({"fixed_192_v1": 16, "fixed_192_overlap_48_v1": 24, "structure_adaptive_fixture_v1": 16}):
        raise ValueError("Qrel counts do not preserve the eight overlap-created positives")

    fixed_layout = layout_by_id["fixed_192_v1"]
    for query in fixture.queries:
        if not query.context_required:
            continue
        evidence_span = span_by_id[query.evidence_span_ids[0]]
        context_span = span_by_id[query.required_context_span_ids[0]]
        evidence_chunk = _chunks_covering_span(fixed_layout, evidence_span)
        context_chunk = _chunks_covering_span(fixed_layout, context_span)
        if len(evidence_chunk) != 1 or len(context_chunk) != 1:
            raise ValueError(f"Fixed layout must map required spans uniquely for {query.query_id}")
        if evidence_chunk[0].chunk_id == context_chunk[0].chunk_id:
            raise ValueError(f"Context dependency is not cross-chunk for {query.query_id}")

    expected_negatives = generate_hard_negatives(fixture.queries, fixture.spans, fixture.layouts)
    if fixture.hard_negatives != expected_negatives:
        raise ValueError("Hard-negative labels do not reproduce deterministically")
    negative_groups: dict[tuple[str, str], list[HardNegative]] = defaultdict(list)
    for negative in fixture.hard_negatives:
        negative_groups[(negative.query_id, negative.layout_id)].append(negative)
    qrel_keys = {(qrel.query_id, qrel.layout_id, qrel.chunk_id) for qrel in fixture.qrels}
    for key, negatives in negative_groups.items():
        if len(negatives) != 3 or len({negative.chunk_id for negative in negatives}) != 3:
            raise ValueError(f"Expected three unique hard negatives for {key}")
        if {negative.negative_family for negative in negatives} != {
            "cross_document_local_text_collision",
            "misleading_gold_parent_neighbor",
            "same_family_scope_collision",
        }:
            raise ValueError(f"Hard-negative families changed for {key}")
        if any((negative.query_id, negative.layout_id, negative.chunk_id) in qrel_keys for negative in negatives):
            raise ValueError(f"Positive chunk mislabeled as hard negative for {key}")
        if any(negative.false_negative_review != "pass" for negative in negatives):
            raise ValueError(f"Unreviewed hard negative for {key}")

    expected_digest = _label_digest(fixture.qrels, fixture.hard_negatives)
    if fixture.label_sha256 != expected_digest:
        raise ValueError("Fixture label digest changed")


def fixture_with_qrels(
    fixture: LateChunkingRetrievalFixture,
    qrels: tuple[Qrel, ...],
) -> LateChunkingRetrievalFixture:
    """Return a fixture copy for negative validation tests."""
    return replace(fixture, qrels=qrels, label_sha256=_label_digest(qrels, fixture.hard_negatives))
