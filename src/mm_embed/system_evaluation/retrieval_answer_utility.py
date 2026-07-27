"""Deterministic no-network contract fixture for retrieval answer utility."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Sequence


FAMILY = "retrieval_answer_utility"
EVALUATION_LEVEL = "system"
EVALUATION_MODE = "answer_utility"
SUBJECT_KIND = "retrieval_answer_system"
DATASET_VERSION = "retrieval-answer-utility-fixture-v0"
SCHEMA_VERSION = "retrieval-answer-utility-system-fixture-v0"
SOURCE_KIND = "local_invented_fixture"
LICENSE_STATUS = "local_invented_not_for_publication"
SPLIT = "fixture_only"
EVIDENCE_TIER = "fixture"
PRIMARY_METRIC = "answer_accuracy"
JUDGE_REVISION = "deterministic-typed-answer-judge-v0"
NETWORK = "forbidden"
ANSWER_TYPES = frozenset({"entity_id", "entity_id_set", "boolean", "integer", "number"})
STATUSES = frozenset({"ok", "invalid_answer", "timeout", "error", "refused"})
ENTITY_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9-]*$")
FIXTURE_VALUE_PATTERN = re.compile(
    r"\[fixture-(entity-id|entity-id-set|boolean|integer|number)=([^\]]+)\]",
    flags=re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "which",
    }
)


class ContractValidationError(ValueError):
    """Validation failure with a stable machine-readable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    text: str
    answer_type: str
    required_citation_count_min: int
    split: str = SPLIT
    source_kind: str = SOURCE_KIND
    source_revision: str = DATASET_VERSION


@dataclass(frozen=True)
class CorpusRecord:
    doc_id: str
    title: str
    text: str
    media_type: str
    source_kind: str
    source_revision: str
    content_sha256: str
    license_status: str


@dataclass(frozen=True)
class AnswerRecord:
    query_id: str
    answer_type: str
    canonical_answer: Any
    accepted_answers: tuple[Any, ...]
    normalizer: str
    numeric_abs_tolerance: float | None = None
    numeric_rel_tolerance: float | None = None


@dataclass(frozen=True)
class QrelRecord:
    query_id: str
    doc_id: str
    relevance: int
    support_kind: str
    required_for_complete_support: bool


@dataclass(frozen=True)
class RetrievalAnswerUtilityFixture:
    dataset_id: str
    dataset_version: str
    schema_version: str
    serialization_version: str
    split: str
    source_kind: str
    license_status: str
    privacy_review: str
    leakage_review: str
    publish: bool
    evidence_tier: str
    fixture_only: bool
    network: str
    file_sha256: dict[str, str]
    queries: tuple[QueryRecord, ...]
    corpus: tuple[CorpusRecord, ...]
    answers: tuple[AnswerRecord, ...]
    qrels: tuple[QrelRecord, ...]
    expected_retrieval_rankings: dict[str, tuple[str, ...]]
    expected_system_metrics: dict[str, dict[str, Any]]
    bundle_hash_contract: str
    task_manifest_payload_sha256: str
    bundle_sha256: str


@dataclass(frozen=True)
class UsageRecord:
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int
    online_cost_usd: float | None
    cost_complete: bool


@dataclass(frozen=True)
class TraceEvent:
    event: str
    component: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class RetrievedDocument:
    doc_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class SystemOutput:
    query_id: str
    answer: Any
    answer_type: str
    cited_doc_ids: tuple[str, ...]
    retrieved: tuple[RetrievedDocument, ...]
    context_doc_ids: tuple[str, ...]
    usage: UsageRecord
    trace: tuple[TraceEvent, ...]
    online_latency_ms: float
    status: str
    error: str | None


@dataclass(frozen=True)
class SystemManifest:
    system_id: str
    system_revision: str
    bracket: str
    evaluation_level: str
    mode: str
    subject_kind: str
    components: dict[str, Any]
    execution: dict[str, Any]


@dataclass(frozen=True)
class JudgeResult:
    query_id: str
    reason_code: str
    answer_valid: bool
    answer_correct: bool
    citation_precision: float
    required_citation_recall: float
    citation_f1: float
    trace_complete: bool
    cost_complete: bool


class FixtureSystem(Protocol):
    manifest: SystemManifest

    def answer(self, query: QueryRecord, fixture: RetrievalAnswerUtilityFixture) -> SystemOutput: ...


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "benchmark" / "fixtures" / DATASET_VERSION
DATA_FILES = ("answers.jsonl", "corpus.jsonl", "qrels.jsonl", "queries.jsonl")
SERIALIZATION_VERSION = "canonical-jsonl-sort-keys-v0"
BUNDLE_HASH_CONTRACT = "sha256(canonical-json({file_sha256,task_manifest_payload_sha256}))"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def serialize_fixture(fixture: RetrievalAnswerUtilityFixture) -> str:
    """Serialize the complete immutable fixture deterministically."""
    return _canonical_json(asdict(fixture))


def _manifest_payload(fixture: RetrievalAnswerUtilityFixture, *, include_integrity: bool) -> dict[str, Any]:
    payload = {
        "bundle_hash_contract": fixture.bundle_hash_contract,
        "dataset_id": fixture.dataset_id,
        "dataset_version": fixture.dataset_version,
        "evidence_tier": fixture.evidence_tier,
        "expected_retrieval_rankings": fixture.expected_retrieval_rankings,
        "expected_system_metrics": fixture.expected_system_metrics,
        "file_sha256": fixture.file_sha256,
        "fixture_only": fixture.fixture_only,
        "leakage_review": fixture.leakage_review,
        "license_status": fixture.license_status,
        "network": fixture.network,
        "privacy_review": fixture.privacy_review,
        "publish": fixture.publish,
        "schema_version": fixture.schema_version,
        "serialization_version": fixture.serialization_version,
        "source_kind": fixture.source_kind,
        "split": fixture.split,
    }
    if include_integrity:
        payload["task_manifest_payload_sha256"] = fixture.task_manifest_payload_sha256
        payload["bundle_sha256"] = fixture.bundle_sha256
    return payload


def _bundle_payload(file_sha256: dict[str, str], task_manifest_payload_sha256: str) -> dict[str, Any]:
    return {
        "file_sha256": file_sha256,
        "task_manifest_payload_sha256": task_manifest_payload_sha256,
    }


def _tokens(value: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(value.casefold()) if token not in STOPWORDS}


def token_overlap_ranking(
    query: QueryRecord,
    corpus: Sequence[CorpusRecord],
) -> tuple[RetrievedDocument, ...]:
    """Rank every document by fixed unique-token overlap and doc-id tie-breaking."""
    query_tokens = _tokens(query.text)
    scored = [
        (float(len(query_tokens & _tokens(f"{document.title} {document.text}"))), document.doc_id)
        for document in corpus
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(
        RetrievedDocument(doc_id=doc_id, rank=rank, score=score)
        for rank, (score, doc_id) in enumerate(scored, start=1)
    )


def _read_canonical_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    if not path.is_file():
        raise ContractValidationError("missing_fixture_file", f"Tracked fixture file is missing: {path.name}")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ContractValidationError("noncanonical_bytes", f"{path.name} must be non-empty and newline terminated")
    try:
        values = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractValidationError("malformed_fixture_file", f"Cannot parse {path.name}") from error
    if not all(isinstance(value, dict) for value in values):
        raise ContractValidationError("malformed_fixture_file", f"{path.name} must contain JSON objects")
    canonical = "".join(f"{_canonical_json(value)}\n" for value in values).encode("utf-8")
    if canonical != raw:
        raise ContractValidationError("noncanonical_bytes", f"{path.name} is not canonical JSONL")
    return values, raw


def _read_canonical_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise ContractValidationError("missing_fixture_file", "Tracked task_manifest.json is missing")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractValidationError("malformed_fixture_file", "Cannot parse task_manifest.json") from error
    if not isinstance(value, dict):
        raise ContractValidationError("malformed_fixture_file", "task_manifest.json must contain one object")
    if raw != f"{_canonical_json(value)}\n".encode("utf-8"):
        raise ContractValidationError("noncanonical_bytes", "task_manifest.json is not canonical JSON")
    return value, raw


def _answer_from_dict(value: dict[str, Any]) -> AnswerRecord:
    canonical_answer = value["canonical_answer"]
    accepted_answers = value["accepted_answers"]
    if value["answer_type"] == "entity_id_set":
        canonical_answer = tuple(canonical_answer)
        accepted_answers = tuple(tuple(candidate) for candidate in accepted_answers)
    else:
        accepted_answers = tuple(accepted_answers)
    return AnswerRecord(
        query_id=value["query_id"],
        answer_type=value["answer_type"],
        canonical_answer=canonical_answer,
        accepted_answers=accepted_answers,
        normalizer=value["normalizer"],
        numeric_abs_tolerance=value["numeric_abs_tolerance"],
        numeric_rel_tolerance=value["numeric_rel_tolerance"],
    )


def _fixture_from_files(root: Path) -> RetrievalAnswerUtilityFixture:
    manifest, _ = _read_canonical_manifest(root / "task_manifest.json")
    raw_files: dict[str, bytes] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for name in DATA_FILES:
        records[name], raw_files[name] = _read_canonical_jsonl(root / name)

    file_sha256 = manifest.get("file_sha256")
    if not isinstance(file_sha256, dict) or set(file_sha256) != set(DATA_FILES):
        raise ContractValidationError("file_hash_metadata", "Manifest must hash every canonical JSONL file")
    for name in DATA_FILES:
        expected = file_sha256[name]
        observed = hashlib.sha256(raw_files[name]).hexdigest()
        if expected != observed:
            raise ContractValidationError("file_hash_mismatch", f"SHA256 mismatch for {name}")

    manifest_without_integrity = dict(manifest)
    bundle_sha256 = manifest_without_integrity.pop("bundle_sha256", None)
    task_manifest_payload_sha256 = manifest_without_integrity.pop("task_manifest_payload_sha256", None)
    observed_manifest_payload = _sha256_text(_canonical_json(manifest_without_integrity))
    if task_manifest_payload_sha256 != observed_manifest_payload:
        raise ContractValidationError("manifest_hash", "Task manifest payload hash does not reproduce")
    observed_bundle = _sha256_text(_canonical_json(_bundle_payload(file_sha256, task_manifest_payload_sha256)))
    if bundle_sha256 != observed_bundle:
        raise ContractValidationError("bundle_hash", "Complete fixture bundle hash does not reproduce")

    try:
        fixture = RetrievalAnswerUtilityFixture(
            dataset_id=manifest["dataset_id"],
            dataset_version=manifest["dataset_version"],
            schema_version=manifest["schema_version"],
            serialization_version=manifest["serialization_version"],
            split=manifest["split"],
            source_kind=manifest["source_kind"],
            license_status=manifest["license_status"],
            privacy_review=manifest["privacy_review"],
            leakage_review=manifest["leakage_review"],
            publish=manifest["publish"],
            evidence_tier=manifest["evidence_tier"],
            fixture_only=manifest["fixture_only"],
            network=manifest["network"],
            file_sha256=dict(file_sha256),
            queries=tuple(QueryRecord(**value) for value in records["queries.jsonl"]),
            corpus=tuple(CorpusRecord(**value) for value in records["corpus.jsonl"]),
            answers=tuple(_answer_from_dict(value) for value in records["answers.jsonl"]),
            qrels=tuple(QrelRecord(**value) for value in records["qrels.jsonl"]),
            expected_retrieval_rankings={
                query_id: tuple(ranking)
                for query_id, ranking in manifest["expected_retrieval_rankings"].items()
            },
            expected_system_metrics=manifest["expected_system_metrics"],
            bundle_hash_contract=manifest["bundle_hash_contract"],
            task_manifest_payload_sha256=task_manifest_payload_sha256,
            bundle_sha256=bundle_sha256,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractValidationError("malformed_fixture_file", "Fixture record shape is invalid") from error
    return fixture


def _require_unique(values: Sequence[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise ContractValidationError("duplicate_id", f"Duplicate {label}")


def validate_fixture(fixture: RetrievalAnswerUtilityFixture) -> None:
    """Validate fixture shape, hashes, qrels, provenance, and answer leakage boundaries."""
    if (
        fixture.dataset_id != DATASET_VERSION
        or fixture.dataset_version != DATASET_VERSION
        or fixture.schema_version != SCHEMA_VERSION
        or fixture.serialization_version != SERIALIZATION_VERSION
        or fixture.split != SPLIT
        or fixture.source_kind != SOURCE_KIND
        or fixture.license_status != LICENSE_STATUS
        or fixture.privacy_review != "local_invented_no_personal_data"
        or fixture.leakage_review != "query_text_checked_against_answers_and_doc_ids"
        or fixture.publish
        or fixture.evidence_tier != EVIDENCE_TIER
        or not fixture.fixture_only
        or fixture.network != NETWORK
        or fixture.bundle_hash_contract != BUNDLE_HASH_CONTRACT
    ):
        raise ContractValidationError("fixture_boundary", "Fixture publication or provenance boundary is invalid")
    if set(fixture.file_sha256) != set(DATA_FILES) or any(
        not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in fixture.file_sha256.values()
    ):
        raise ContractValidationError("file_hash_metadata", "Fixture file hash metadata is incomplete or malformed")
    if not re.fullmatch(r"[0-9a-f]{64}", fixture.task_manifest_payload_sha256):
        raise ContractValidationError("manifest_hash", "Task manifest payload hash metadata is malformed")
    if (
        len(fixture.queries) != 6
        or len(fixture.corpus) != 12
        or len(fixture.answers) != 6
        or len(fixture.qrels) != 8
    ):
        raise ContractValidationError(
            "fixture_shape",
            "Fixture must contain 6 queries, 12 documents, 6 answers, and 8 qrels",
        )

    query_ids = [query.query_id for query in fixture.queries]
    doc_ids = [document.doc_id for document in fixture.corpus]
    answer_query_ids = [answer.query_id for answer in fixture.answers]
    qrel_keys = [(qrel.query_id, qrel.doc_id) for qrel in fixture.qrels]
    _require_unique(query_ids, "query id")
    _require_unique(doc_ids, "document id")
    _require_unique(answer_query_ids, "answer query id")
    _require_unique(qrel_keys, "qrel")
    if (
        query_ids != sorted(query_ids)
        or doc_ids != sorted(doc_ids)
        or answer_query_ids != sorted(answer_query_ids)
        or qrel_keys != sorted(qrel_keys)
    ):
        raise ContractValidationError("noncanonical_order", "Fixture records must use canonical id order")
    queries_by_id = {query.query_id: query for query in fixture.queries}
    documents_by_id = {document.doc_id: document for document in fixture.corpus}
    answers_by_id = {answer.query_id: answer for answer in fixture.answers}

    required_counts: dict[str, int] = {query_id: 0 for query_id in query_ids}
    for qrel in fixture.qrels:
        if qrel.query_id not in queries_by_id or qrel.doc_id not in documents_by_id:
            raise ContractValidationError("invalid_qrel_reference", "Qrel references an unknown query or document")
        if qrel.relevance != 1 or qrel.support_kind not in {"direct", "partial"}:
            raise ContractValidationError("invalid_qrel", "Qrels must use binary relevance and declared support kinds")
        if qrel.required_for_complete_support:
            required_counts[qrel.query_id] += 1
    if sorted(required_counts.values()) != [1, 1, 1, 1, 2, 2]:
        raise ContractValidationError(
            "invalid_qrel_counts",
            "Fixture must have four single-evidence and two two-evidence answers",
        )

    for query in fixture.queries:
        if (
            query.answer_type not in ANSWER_TYPES
            or query.required_citation_count_min != required_counts[query.query_id]
        ):
            raise ContractValidationError(
                "invalid_query_contract",
                f"Invalid answer or citation contract for {query.query_id}",
            )
        answer = answers_by_id.get(query.query_id)
        if answer is None or answer.answer_type != query.answer_type:
            raise ContractValidationError("missing_answer", f"Missing typed answer for {query.query_id}")
        normalized_query = query.text.casefold()
        leaked_values = [str(answer.canonical_answer), *doc_ids]
        if any(value.casefold() in normalized_query for value in leaked_values):
            raise ContractValidationError("answer_leakage", f"Query {query.query_id} leaks an answer or document id")

    for document in fixture.corpus:
        if (
            document.media_type != "text/plain"
            or document.source_kind != SOURCE_KIND
            or document.source_revision != DATASET_VERSION
            or document.license_status != LICENSE_STATUS
            or document.content_sha256 != _sha256_text(document.text)
        ):
            raise ContractValidationError("document_hash", f"Invalid local corpus record {document.doc_id}")

    expected_rankings = {
        query.query_id: tuple(item.doc_id for item in token_overlap_ranking(query, fixture.corpus))
        for query in fixture.queries
    }
    if fixture.expected_retrieval_rankings != expected_rankings:
        raise ContractValidationError("ranking_drift", "Checked-in token-overlap rankings do not reproduce")
    if set(fixture.expected_system_metrics) != {
        "closed_book_constant",
        "oracle_structured_lookup",
        "token_overlap_retrieval",
    }:
        raise ContractValidationError("metric_table", "Checked-in expected metric table is incomplete")
    expected_manifest_digest = _sha256_text(_canonical_json(_manifest_payload(fixture, include_integrity=False)))
    if fixture.task_manifest_payload_sha256 != expected_manifest_digest:
        raise ContractValidationError("manifest_hash", "Task manifest payload hash does not reproduce")
    expected_bundle_digest = _sha256_text(
        _canonical_json(_bundle_payload(fixture.file_sha256, fixture.task_manifest_payload_sha256))
    )
    if fixture.bundle_sha256 != expected_bundle_digest:
        raise ContractValidationError("bundle_hash", "Fixture bundle hash does not reproduce")


def load_retrieval_answer_utility_fixture(
    root: str | Path = FIXTURE_ROOT,
) -> RetrievalAnswerUtilityFixture:
    """Load and validate the tracked deterministic local no-publish fixture bytes."""
    fixture = _fixture_from_files(Path(root))
    validate_fixture(fixture)
    return fixture


def parse_typed_answer(answer: Any, answer_type: str) -> Any:
    """Parse one output using only the declared deterministic v0 answer type."""
    if answer_type == "entity_id":
        if not isinstance(answer, str) or not ENTITY_ID_PATTERN.fullmatch(answer.strip()):
            raise ContractValidationError("malformed_answer", "Entity answers must be uppercase identifiers")
        return answer.strip()
    if answer_type == "entity_id_set":
        if not isinstance(answer, (list, tuple)) or not answer:
            raise ContractValidationError("malformed_answer", "Entity set answers must be non-empty sequences")
        values = tuple(str(value).strip() for value in answer)
        if len(values) != len(set(values)) or any(not ENTITY_ID_PATTERN.fullmatch(value) for value in values):
            raise ContractValidationError("malformed_answer", "Entity set answers contain invalid or duplicate ids")
        return tuple(sorted(values))
    if answer_type == "boolean":
        if not isinstance(answer, bool):
            raise ContractValidationError("malformed_answer", "Boolean answers must be JSON booleans")
        return answer
    if answer_type == "integer":
        if isinstance(answer, bool) or not isinstance(answer, int):
            raise ContractValidationError("malformed_answer", "Integer answers must be JSON integers")
        return answer
    if answer_type == "number":
        if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(float(answer)):
            raise ContractValidationError("malformed_answer", "Number answers must be finite JSON numbers")
        return float(answer)
    raise ContractValidationError("unknown_answer_type", f"Unsupported answer type: {answer_type}")


def validate_usage(usage: UsageRecord) -> None:
    """Validate local token, call, and cost completeness accounting."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.llm_calls,
    )):
        raise ContractValidationError("malformed_usage", "Usage counters must be non-negative integers")
    if usage.online_cost_usd is not None and (
        isinstance(usage.online_cost_usd, bool)
        or not isinstance(usage.online_cost_usd, (int, float))
        or not math.isfinite(float(usage.online_cost_usd))
        or usage.online_cost_usd < 0
    ):
        raise ContractValidationError("malformed_usage", "Online cost must be null or a finite non-negative number")
    if usage.cost_complete and usage.online_cost_usd is None:
        raise ContractValidationError("missing_cost", "Complete cost accounting cannot omit online cost")


def validate_trace(output: SystemOutput) -> bool:
    """Require a valid trace for success or an explicit failure record for non-success."""
    if output.trace:
        previous_end = 0.0
        for event in output.trace:
            if (
                not event.event
                or not event.component
                or event.start_ms < 0
                or event.end_ms < event.start_ms
                or event.start_ms < previous_end
            ):
                raise ContractValidationError(
                    "malformed_trace",
                    "Trace events must be named, ordered, and non-negative",
                )
            previous_end = event.end_ms
        if output.online_latency_ms < output.trace[-1].end_ms:
            raise ContractValidationError("malformed_trace", "Online latency cannot end before the trace")
        return True
    if output.status != "ok" and output.error:
        return True
    reason = "missing_trace" if output.status == "ok" else "missing_failure_record"
    raise ContractValidationError(reason, "Every attempt requires a trace or explicit failure record")


def validate_citations(output: SystemOutput, fixture: RetrievalAnswerUtilityFixture) -> None:
    """Validate citation and context ids against the pinned local corpus."""
    known_ids = {document.doc_id for document in fixture.corpus}
    referenced_ids = [
        *output.cited_doc_ids,
        *output.context_doc_ids,
        *(item.doc_id for item in output.retrieved),
    ]
    if any(doc_id not in known_ids for doc_id in referenced_ids):
        raise ContractValidationError("unknown_citation", "Output references an unknown corpus document")
    if len(output.cited_doc_ids) != len(set(output.cited_doc_ids)):
        raise ContractValidationError("duplicate_citation", "Citations must not contain duplicates")
    if len(output.context_doc_ids) != len(set(output.context_doc_ids)):
        raise ContractValidationError("duplicate_context", "Context ids must not contain duplicates")
    if any(doc_id not in output.context_doc_ids for doc_id in output.cited_doc_ids):
        raise ContractValidationError("citation_not_in_context", "Every citation must identify a context document")
    if output.retrieved:
        expected_ranks = list(range(1, len(output.retrieved) + 1))
        if [item.rank for item in output.retrieved] != expected_ranks:
            raise ContractValidationError("malformed_retrieval", "Retrieved documents must have contiguous ranks")


def validate_system_output(output: SystemOutput, fixture: RetrievalAnswerUtilityFixture) -> None:
    """Validate a complete system-level attempt without invoking a model or network service."""
    queries_by_id = {query.query_id: query for query in fixture.queries}
    query = queries_by_id.get(output.query_id)
    if query is None:
        raise ContractValidationError("unknown_query", f"Unknown query id: {output.query_id}")
    if output.answer_type != query.answer_type:
        raise ContractValidationError("answer_type_mismatch", "Output answer type does not match the query contract")
    if output.status not in STATUSES:
        raise ContractValidationError("invalid_status", f"Unsupported system status: {output.status}")
    if output.online_latency_ms < 0 or not math.isfinite(output.online_latency_ms):
        raise ContractValidationError("malformed_latency", "Online latency must be finite and non-negative")
    if output.status == "ok" and output.error is not None:
        raise ContractValidationError("invalid_failure_record", "Successful outputs cannot carry an error")
    if output.status != "ok" and not output.error:
        raise ContractValidationError("missing_failure_record", "Non-success outputs require an explicit error record")
    validate_usage(output.usage)
    validate_citations(output, fixture)
    validate_trace(output)


def _citation_scores(output: SystemOutput, fixture: RetrievalAnswerUtilityFixture) -> tuple[float, float, float]:
    relevant = {
        qrel.doc_id
        for qrel in fixture.qrels
        if qrel.query_id == output.query_id and qrel.relevance > 0
    }
    required = {
        qrel.doc_id
        for qrel in fixture.qrels
        if qrel.query_id == output.query_id and qrel.required_for_complete_support
    }
    cited = set(output.cited_doc_ids)
    precision = len(cited & relevant) / len(cited) if cited else 0.0
    recall = len(cited & required) / len(required)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _answers_match(parsed: Any, answer: AnswerRecord) -> tuple[bool, bool]:
    if answer.answer_type == "entity_id":
        normalized = parsed.casefold().strip()
        canonical = str(answer.canonical_answer).casefold().strip()
        accepted = {str(value).casefold().strip() for value in answer.accepted_answers}
        return normalized in accepted, normalized != canonical and normalized in accepted
    if answer.answer_type == "entity_id_set":
        normalized = tuple(sorted(value.casefold().strip() for value in parsed))
        canonical = tuple(sorted(value.casefold().strip() for value in answer.canonical_answer))
        accepted = {
            tuple(sorted(value.casefold().strip() for value in candidate))
            for candidate in answer.accepted_answers
        }
        return normalized in accepted, normalized != canonical and normalized in accepted
    if answer.answer_type == "number":
        target = float(answer.canonical_answer)
        return math.isclose(
            parsed,
            target,
            abs_tol=answer.numeric_abs_tolerance or 0.0,
            rel_tol=answer.numeric_rel_tolerance or 0.0,
        ), False
    return parsed in answer.accepted_answers, parsed != answer.canonical_answer and parsed in answer.accepted_answers


def judge_output(output: SystemOutput, fixture: RetrievalAnswerUtilityFixture) -> JudgeResult:
    """Judge typed correctness and citations with stable reason-code precedence."""
    try:
        trace_complete = validate_trace(output)
    except ContractValidationError:
        trace_complete = False
    try:
        validate_usage(output.usage)
        cost_complete = output.usage.cost_complete and output.usage.online_cost_usd is not None
    except ContractValidationError:
        cost_complete = False
    try:
        validate_system_output(output, fixture)
    except ContractValidationError as error:
        return JudgeResult(
            output.query_id,
            error.reason_code,
            False,
            False,
            0.0,
            0.0,
            0.0,
            trace_complete,
            cost_complete,
        )

    precision, recall, f1 = _citation_scores(output, fixture)
    if output.status != "ok":
        reason = "timeout" if output.status == "timeout" else "system_error"
        return JudgeResult(
            output.query_id,
            reason,
            False,
            False,
            precision,
            recall,
            f1,
            trace_complete,
            cost_complete,
        )

    try:
        parsed = parse_typed_answer(output.answer, output.answer_type)
    except ContractValidationError as error:
        return JudgeResult(
            output.query_id,
            error.reason_code,
            False,
            False,
            precision,
            recall,
            f1,
            trace_complete,
            cost_complete,
        )

    answer = next(answer for answer in fixture.answers if answer.query_id == output.query_id)
    correct, accepted_alias = _answers_match(parsed, answer)
    if not correct:
        reason = {
            "entity_id_set": "set_mismatch",
            "number": "numeric_out_of_tolerance",
        }.get(answer.answer_type, "answer_mismatch")
    elif recall < 1.0:
        reason = "missing_required_citation"
    elif precision < 1.0:
        reason = "irrelevant_citation"
    else:
        reason = "accepted_alias" if accepted_alias else "exact_match"
    return JudgeResult(
        output.query_id,
        reason,
        True,
        correct,
        precision,
        recall,
        f1,
        trace_complete,
        cost_complete,
    )


def _manifest(
    system_id: str,
    bracket: str,
    components: dict[str, Any],
) -> SystemManifest:
    return SystemManifest(
        system_id=system_id,
        system_revision=f"{system_id}-v0",
        bracket=bracket,
        evaluation_level=EVALUATION_LEVEL,
        mode=EVALUATION_MODE,
        subject_kind=SUBJECT_KIND,
        components=components,
        execution={
            "concurrency": 1,
            "timeout_s": 1,
            "max_retries": 0,
            "cache_policy": "disabled",
            "network": NETWORK,
        },
    )


def _local_usage(cost: float | None = 0.0, *, complete: bool = True) -> UsageRecord:
    return UsageRecord(0, 0, 0, cost, complete)


def _parse_documents(query: QueryRecord, documents: Sequence[CorpusRecord]) -> tuple[Any, tuple[str, ...]]:
    expected_marker = query.answer_type.replace("_", "-")
    values: list[str] = []
    citations: list[str] = []
    for document in documents:
        document_values = [
            value.strip()
            for marker_type, value in FIXTURE_VALUE_PATTERN.findall(document.text)
            if marker_type.casefold() == expected_marker
        ]
        if document_values:
            values.extend(document_values)
            citations.append(document.doc_id)
    if query.answer_type == "entity_id":
        answer: Any = values[0] if len(values) == 1 else "UNKNOWN"
    elif query.answer_type == "entity_id_set":
        answer = tuple(sorted(values))
    elif query.answer_type == "boolean":
        answer = values[0].casefold() == "true" if len(values) == 1 else "UNKNOWN"
    elif query.answer_type == "integer":
        answer = int(values[0]) if len(values) == 1 else "UNKNOWN"
    else:
        answer = sum(float(value) for value in values) if values else "UNKNOWN"
    return answer, tuple(citations)


class ClosedBookConstant:
    """Pure-local closed-book anchor that returns UNKNOWN for every query."""

    manifest = _manifest(
        "closed_book_constant",
        "closed_book",
        {"generator": {"kind": "local_constant", "revision": "unknown-v0"}},
    )

    def answer(self, query: QueryRecord, fixture: RetrievalAnswerUtilityFixture) -> SystemOutput:
        del fixture
        return SystemOutput(
            query_id=query.query_id,
            answer="UNKNOWN",
            answer_type=query.answer_type,
            cited_doc_ids=(),
            retrieved=(),
            context_doc_ids=(),
            usage=_local_usage(),
            trace=(TraceEvent("constant", "local_constant", 0.0, 1.0),),
            online_latency_ms=1.0,
            status="ok",
            error=None,
        )


class OracleStructuredLookup:
    """Pure-local oracle that parses exactly the required gold evidence documents."""

    manifest = _manifest(
        "oracle_structured_lookup",
        "oracle_context",
        {"context": {"kind": "gold_qrels", "ordering": "doc_id"}, "generator": {"kind": "local_parser"}},
    )

    def answer(self, query: QueryRecord, fixture: RetrievalAnswerUtilityFixture) -> SystemOutput:
        document_by_id = {document.doc_id: document for document in fixture.corpus}
        context_ids = tuple(
            sorted(
                qrel.doc_id
                for qrel in fixture.qrels
                if qrel.query_id == query.query_id and qrel.required_for_complete_support
            )
        )
        context = tuple(document_by_id[doc_id] for doc_id in context_ids)
        answer, citations = _parse_documents(query, context)
        latency = 1.0 + len(context)
        return SystemOutput(
            query_id=query.query_id,
            answer=answer,
            answer_type=query.answer_type,
            cited_doc_ids=citations,
            retrieved=(),
            context_doc_ids=context_ids,
            usage=_local_usage(),
            trace=(
                TraceEvent("context", "oracle_context", 0.0, 1.0),
                TraceEvent("parse", "local_parser", 1.0, latency),
            ),
            online_latency_ms=latency,
            status="ok",
            error=None,
        )


class TokenOverlapRetrieval:
    """Pure-local top-two token-overlap retrieval followed by the oracle parser."""

    manifest = _manifest(
        "token_overlap_retrieval",
        "retrieval",
        {
            "retriever": {"kind": "unique_token_overlap", "top_k": 2, "tie_break": "doc_id_ascending"},
            "generator": {"kind": "local_parser"},
        },
    )

    def answer(self, query: QueryRecord, fixture: RetrievalAnswerUtilityFixture) -> SystemOutput:
        ranking = token_overlap_ranking(query, fixture.corpus)
        context_ids = tuple(item.doc_id for item in ranking[:2])
        document_by_id = {document.doc_id: document for document in fixture.corpus}
        context = tuple(document_by_id[doc_id] for doc_id in context_ids)
        answer, citations = _parse_documents(query, context)
        return SystemOutput(
            query_id=query.query_id,
            answer=answer,
            answer_type=query.answer_type,
            cited_doc_ids=citations,
            retrieved=ranking,
            context_doc_ids=context_ids,
            usage=_local_usage(),
            trace=(
                TraceEvent("retrieve", "token_overlap", 0.0, 2.0),
                TraceEvent("parse", "local_parser", 2.0, 4.0),
            ),
            online_latency_ms=4.0,
            status="ok",
            error=None,
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def aggregate_system_metrics(
    outputs: Sequence[SystemOutput],
    judgments: Sequence[JudgeResult],
    fixture: RetrievalAnswerUtilityFixture,
) -> dict[str, Any]:
    attempted = len(outputs)
    known_costs = [
        float(output.usage.online_cost_usd)
        for output in outputs
        if output.usage.online_cost_usd is not None
    ]
    cost_complete_count = sum(judgment.cost_complete for judgment in judgments)
    trace_complete_count = sum(judgment.trace_complete for judgment in judgments)
    citation_complete_count = sum(judgment.required_citation_recall == 1.0 for judgment in judgments)
    latencies = [output.online_latency_ms for output in outputs]
    metrics: dict[str, Any] = {
        "attempted_count": attempted,
        "ok_count": sum(output.status == "ok" for output in outputs),
        "failure_count": sum(output.status != "ok" for output in outputs),
        "answer_correct_count": sum(judgment.answer_correct for judgment in judgments),
        "answer_accuracy": _mean([float(judgment.answer_correct) for judgment in judgments]),
        "citation_precision": _mean([judgment.citation_precision for judgment in judgments]),
        "required_citation_recall": _mean([judgment.required_citation_recall for judgment in judgments]),
        "citation_f1": _mean([judgment.citation_f1 for judgment in judgments]),
        "required_citation_complete_count": citation_complete_count,
        "missing_required_citation_count": attempted - citation_complete_count,
        "valid_answer_count": sum(judgment.answer_valid for judgment in judgments),
        "valid_answer_rate": _mean([float(judgment.answer_valid) for judgment in judgments]),
        "trace_complete_count": trace_complete_count,
        "trace_missing_count": attempted - trace_complete_count,
        "trace_complete_rate": trace_complete_count / attempted,
        "cost_known_count": len(known_costs),
        "cost_missing_count": attempted - len(known_costs),
        "cost_complete_count": cost_complete_count,
        "cost_complete_rate": cost_complete_count / attempted,
        "known_online_cost_usd_total": sum(known_costs),
        "known_online_cost_usd_per_attempted_query": sum(known_costs) / attempted,
        "total_online_cost_usd": sum(known_costs) if cost_complete_count == attempted else None,
        "mean_online_cost_usd": _mean(known_costs) if cost_complete_count == attempted else None,
        "cost_comparison_eligible": cost_complete_count == attempted,
        "online_latency_ms_mean": _mean(latencies),
        "online_latency_ms_median": _median(latencies),
        "online_latency_ms_p95": _percentile_nearest_rank(latencies, 0.95),
        "timeout_rate": _mean([float(output.status == "timeout") for output in outputs]),
        "error_rate": _mean([float(output.status in {"error", "invalid_answer"}) for output in outputs]),
        "judge_cost_usd": 0.0,
    }
    if any(output.retrieved for output in outputs):
        recalls = []
        for output in outputs:
            required = {
                qrel.doc_id
                for qrel in fixture.qrels
                if qrel.query_id == output.query_id and qrel.required_for_complete_support
            }
            top_two = {item.doc_id for item in output.retrieved[:2]}
            recalls.append(len(required & top_two) / len(required))
        metrics["ranking_recall@2"] = _mean(recalls)
    return metrics


def _manifest_to_dict(manifest: SystemManifest) -> dict[str, Any]:
    return asdict(manifest)


def evaluate_system(
    system: FixtureSystem,
    fixture: RetrievalAnswerUtilityFixture | None = None,
    *,
    started_at: str = "2026-07-27T00:00:00Z",
    completed_at: str = "2026-07-27T00:00:00Z",
) -> dict[str, Any]:
    """Evaluate one local system and construct a separately discriminated system result."""
    fixture = fixture or load_retrieval_answer_utility_fixture()
    manifest = system.manifest
    if (
        manifest.evaluation_level != EVALUATION_LEVEL
        or manifest.mode != EVALUATION_MODE
        or manifest.subject_kind != SUBJECT_KIND
    ):
        raise ContractValidationError(
            "system_discriminator",
            "System manifests must use system-level answer discriminators",
        )
    outputs = tuple(system.answer(query, fixture) for query in fixture.queries)
    judgments = tuple(judge_output(output, fixture) for output in outputs)
    metrics = aggregate_system_metrics(outputs, judgments, fixture)
    manifest_payload = _manifest_to_dict(manifest)
    manifest_sha256 = _sha256_text(_canonical_json(manifest_payload))
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": {
            "family": FAMILY,
            "level": EVALUATION_LEVEL,
            "mode": EVALUATION_MODE,
            "leaderboard_surface": "system",
        },
        "subject": {
            "kind": SUBJECT_KIND,
            "id": manifest.system_id,
            "manifest_sha256": manifest_sha256,
            "manifest": manifest_payload,
        },
        "run": {
            "id": f"{DATASET_VERSION}:{manifest.system_id}",
            "publish": False,
            "evidence_tier": EVIDENCE_TIER,
        },
        "timestamps": {
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_s": 0.0,
        },
        "task": {
            "id": DATASET_VERSION,
            "display_name": "Retrieval answer utility fixture",
            "primary_metric": PRIMARY_METRIC,
            "publish": False,
            "evidence_tier": EVIDENCE_TIER,
        },
        "metrics": metrics,
        "resource_usage": {
            "online_system_cost_usd": metrics["total_online_cost_usd"],
            "known_online_system_cost_usd": metrics["known_online_cost_usd_total"],
            "cost_known_count": metrics["cost_known_count"],
            "cost_missing_count": metrics["cost_missing_count"],
            "cost_complete_rate": metrics["cost_complete_rate"],
            "judge_cost_usd": 0.0,
        },
        "details": {
            "fixture_only": True,
            "publish": False,
            "evidence_tier": EVIDENCE_TIER,
            "network": NETWORK,
            "provider_api_calls": 0,
            "model_inference_calls": 0,
            "model_downloads": 0,
            "fixture_bundle_sha256": fixture.bundle_sha256,
            "judge_revision": JUDGE_REVISION,
            "outputs": [asdict(output) for output in outputs],
            "per_query": [asdict(judgment) for judgment in judgments],
        },
        "error": None,
    }


def evaluate_fixture_brackets(
    fixture: RetrievalAnswerUtilityFixture | None = None,
) -> dict[str, Any]:
    """Evaluate all three local brackets and return decomposed cross-bracket diagnostics."""
    fixture = fixture or load_retrieval_answer_utility_fixture()
    systems: tuple[FixtureSystem, ...] = (
        ClosedBookConstant(),
        OracleStructuredLookup(),
        TokenOverlapRetrieval(),
    )
    runs = {system.manifest.system_id: evaluate_system(system, fixture) for system in systems}
    closed = runs["closed_book_constant"]
    oracle = runs["oracle_structured_lookup"]
    retrieval = runs["token_overlap_retrieval"]
    oracle_by_query = {item["query_id"]: item for item in oracle["details"]["per_query"]}
    retrieval_by_query = {item["query_id"]: item for item in retrieval["details"]["per_query"]}
    retrieval_outputs = {item["query_id"]: item for item in retrieval["details"]["outputs"]}
    required_by_query = {
        query.query_id: {
            qrel.doc_id
            for qrel in fixture.qrels
            if qrel.query_id == query.query_id and qrel.required_for_complete_support
        }
        for query in fixture.queries
    }
    oracle_answerable = {
        query_id for query_id, result in oracle_by_query.items() if result["answer_correct"]
    }
    retrieval_failures = {
        query_id
        for query_id in oracle_answerable
        if not retrieval_by_query[query_id]["answer_correct"]
    }
    generation_failures = 0
    for query_id, output in retrieval_outputs.items():
        if required_by_query[query_id].issubset(set(output["context_doc_ids"])) and not retrieval_by_query[query_id][
            "answer_correct"
        ]:
            generation_failures += 1
    diagnostics = {
        "retrieval_minus_closed_book_answer_accuracy": (
            retrieval["metrics"]["answer_accuracy"] - closed["metrics"]["answer_accuracy"]
        ),
        "oracle_minus_retrieval_answer_accuracy": (
            oracle["metrics"]["answer_accuracy"] - retrieval["metrics"]["answer_accuracy"]
        ),
        "retrieval_required_citation_recall": retrieval["metrics"]["required_citation_recall"],
        "retrieval_ranking_recall@2": retrieval["metrics"]["ranking_recall@2"],
        "answerable_with_oracle_count": len(oracle_answerable),
        "retrieval_failures_among_oracle_answerable": len(retrieval_failures),
        "generation_failures_with_required_context": generation_failures,
    }
    return {"runs": runs, "diagnostics": diagnostics}


def normalize_result_timestamps(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize run timestamps before byte-for-byte repeatability comparisons."""
    normalized = copy.deepcopy(record)
    timestamps = normalized.get("timestamps")
    if isinstance(timestamps, dict):
        for key in ("started_at", "completed_at"):
            if key in timestamps:
                timestamps[key] = "<normalized>"
        if "duration_s" in timestamps:
            timestamps["duration_s"] = 0.0
    return normalized


def serialize_result(record: dict[str, Any], *, normalize_timestamps: bool = False) -> str:
    """Serialize one system result canonically, optionally normalizing timestamps."""
    payload = normalize_result_timestamps(record) if normalize_timestamps else record
    return _canonical_json(payload)
