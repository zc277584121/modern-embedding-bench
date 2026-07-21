"""Invented no-publish fixture for agent skill routing diagnostics."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


SOURCE_KIND = "local_invented_fixture"
SOURCE_REVISION = "agent-skill-routing-fixture-v0"
LICENSE_STATUS = "local_invented_sanitized_no_external_sources"
SPLIT = "fixture_only"
PUBLIC_SCORE_ELIGIBLE = False
REVIEW_STATUS = "reviewed_against_full_query"


@dataclass(frozen=True)
class SkillDocument:
    """An invented skill card used as a retrieval document."""

    skill_id: str
    family_id: str
    name: str
    description: str
    body: str
    source_kind: str
    source_id: str
    source_revision: str
    license_status: str
    public_score_eligible: bool


@dataclass(frozen=True)
class SkillQuery:
    """An invented routing query."""

    query_id: str
    text: str
    split: str
    slice: str
    language: str
    source_kind: str
    source_id: str
    source_revision: str
    license_status: str
    public_score_eligible: bool


@dataclass(frozen=True)
class SkillQrel:
    """A binary query-to-skill relevance judgment."""

    query_id: str
    skill_id: str
    relevance: int
    source_kind: str = SOURCE_KIND
    source_revision: str = SOURCE_REVISION
    license_status: str = LICENSE_STATUS
    public_score_eligible: bool = PUBLIC_SCORE_ELIGIBLE


@dataclass(frozen=True)
class SkillSet:
    """A query-conditioned compatible or rejected skill set."""

    query_id: str
    set_id: str
    skill_ids: tuple[str, ...]
    label: str
    reason_code: str
    review_status: str
    source_kind: str = SOURCE_KIND
    source_revision: str = SOURCE_REVISION
    license_status: str = LICENSE_STATUS
    public_score_eligible: bool = PUBLIC_SCORE_ELIGIBLE


@dataclass(frozen=True)
class EvidenceRecord:
    """Local contract evidence for one invented risk relation."""

    evidence_id: str
    query_id: str
    statement: str
    source_kind: str = SOURCE_KIND
    source_revision: str = SOURCE_REVISION
    license_status: str = LICENSE_STATUS
    public_score_eligible: bool = PUBLIC_SCORE_ELIGIBLE


@dataclass(frozen=True)
class RiskPair:
    """A query-specific helpful and risky same-family sibling pair."""

    query_id: str
    family_id: str
    helpful_skill_id: str
    risky_skill_id: str
    risk_type: str
    admission_basis: str
    evidence_id: str
    source_kind: str = SOURCE_KIND
    source_revision: str = SOURCE_REVISION
    license_status: str = LICENSE_STATUS
    public_score_eligible: bool = PUBLIC_SCORE_ELIGIBLE


@dataclass(frozen=True)
class HardNegative:
    """An audited query-specific hard negative."""

    query_id: str
    skill_id: str
    negative_role: str
    reason: str
    review_status: str
    source_kind: str = SOURCE_KIND
    source_revision: str = SOURCE_REVISION
    license_status: str = LICENSE_STATUS
    public_score_eligible: bool = PUBLIC_SCORE_ELIGIBLE


@dataclass(frozen=True)
class AgentSkillRoutingFixture:
    """Complete shared fixture for compatibility and risk tasks."""

    skills: tuple[SkillDocument, ...]
    queries: tuple[SkillQuery, ...]
    qrels: tuple[SkillQrel, ...]
    compatible_sets: tuple[SkillSet, ...]
    rejected_sets: tuple[SkillSet, ...]
    evidence: tuple[EvidenceRecord, ...]
    risk_pairs: tuple[RiskPair, ...]
    hard_negatives: tuple[HardNegative, ...]
    split: str
    source_kind: str
    source_revision: str
    license_status: str
    public_score_eligible: bool


def _skill(
    skill_id: str,
    family_id: str,
    name: str,
    description: str,
    body: str,
    source_id: str,
) -> SkillDocument:
    return SkillDocument(
        skill_id=skill_id,
        family_id=family_id,
        name=name,
        description=description,
        body=body,
        source_kind=SOURCE_KIND,
        source_id=source_id,
        source_revision=SOURCE_REVISION,
        license_status=LICENSE_STATUS,
        public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
    )


SKILLS: tuple[SkillDocument, ...] = (
    _skill(
        "schema_contract_diff",
        "schema_change",
        "Schema Contract Diff",
        "Compares two structured data contracts and reports field-level changes.",
        "Inputs: earlier contract and later contract. Returns added, removed, and changed field rules.",
        "skill_schema_001",
    ),
    _skill(
        "schema_sample_mapper",
        "schema_change",
        "Schema Sample Mapper",
        "Checks a sample record against a selected structured data contract.",
        "Inputs: one sample record and one contract. Returns field mappings and validation mismatches.",
        "skill_schema_002",
    ),
    _skill(
        "schema_doc_outline",
        "schema_change",
        "Schema Doc Outline",
        "Creates a documentation outline from a structured data contract.",
        "Inputs: one contract. Returns headings and explanatory prompts without validating sample records.",
        "skill_schema_003",
    ),
    _skill(
        "log_redaction_review",
        "log_diagnostics",
        "Log Redaction Review",
        "Finds credential-like fields in invented diagnostic logs and replaces their values.",
        "Inputs: diagnostic lines and redaction policy. Returns sanitized lines plus a replacement count.",
        "skill_log_001",
    ),
    _skill(
        "error_signature_cluster",
        "log_diagnostics",
        "Error Signature Cluster",
        "Groups sanitized diagnostic errors by normalized signature.",
        "Inputs: sanitized diagnostic lines. Returns signature groups and occurrence counts.",
        "skill_log_002",
    ),
    _skill(
        "log_volume_rollup",
        "log_diagnostics",
        "Log Volume Rollup",
        "Aggregates diagnostic line volume by invented component and time bucket.",
        "Inputs: diagnostic lines and bucket width. Returns volume totals without signature grouping.",
        "skill_log_003",
    ),
    _skill(
        "archive_day_lookup",
        "archive_lookup",
        "Archive Day Lookup",
        "Checks for an archived snapshot on one requested calendar date.",
        "Inputs: page label, target date, exact-or-prior mode. Returns one date-scoped result.",
        "skill_archive_001",
    ),
    _skill(
        "archive_range_list",
        "archive_lookup",
        "Archive Range List",
        "Lists archived snapshots across a requested date interval.",
        "Inputs: page label, start date, and end date. Returns an interval listing.",
        "skill_archive_002",
    ),
    _skill(
        "page_freshness_score",
        "archive_lookup",
        "Page Freshness Score",
        "Estimates how recently an invented public page changed.",
        "Inputs: page label and lookback window. Returns a recency score rather than snapshot existence.",
        "skill_archive_003",
    ),
    _skill(
        "dependency_license_matrix",
        "dependency_audit",
        "Dependency License Matrix",
        "Builds a package version and license compatibility matrix for an invented policy.",
        "Inputs: package inventory and policy rules. Returns per-package license compatibility findings.",
        "skill_dependency_001",
    ),
    _skill(
        "package_age_report",
        "dependency_audit",
        "Package Age Report",
        "Summarizes how long invented package versions have remained unchanged.",
        "Inputs: package inventory and reference date. Returns age buckets without license review.",
        "skill_dependency_002",
    ),
    _skill(
        "dependency_update_window",
        "dependency_audit",
        "Dependency Update Window",
        "Plans maintenance windows for invented dependency updates.",
        "Inputs: package inventory and maintenance constraints. Returns proposed update windows.",
        "skill_dependency_003",
    ),
)


QUERIES: tuple[SkillQuery, ...] = (
    SkillQuery(
        query_id="q_compat_schema_validate",
        text="Compare two invented data contracts, then check one sample record against the newer rules.",
        split=SPLIT,
        slice="compatible_set",
        language="en",
        source_kind=SOURCE_KIND,
        source_id="query_fixture_001",
        source_revision=SOURCE_REVISION,
        license_status=LICENSE_STATUS,
        public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
    ),
    SkillQuery(
        query_id="q_compat_log_triage",
        text=(
            "Replace credential-like values in invented diagnostic lines, "
            "then group the remaining errors by signature."
        ),
        split=SPLIT,
        slice="compatible_set",
        language="en",
        source_kind=SOURCE_KIND,
        source_id="query_fixture_002",
        source_revision=SOURCE_REVISION,
        license_status=LICENSE_STATUS,
        public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
    ),
    SkillQuery(
        query_id="q_risk_archive_exact_day",
        text="Check whether the invented public page had a snapshot on 2031-04-12; do not return an interval listing.",
        split=SPLIT,
        slice="same_capability_risk",
        language="en",
        source_kind=SOURCE_KIND,
        source_id="query_fixture_003",
        source_revision=SOURCE_REVISION,
        license_status=LICENSE_STATUS,
        public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
    ),
    SkillQuery(
        query_id="q_risk_dependency_policy",
        text="Audit invented package names, versions, licenses, and compatibility with the Cedar Harbor policy.",
        split=SPLIT,
        slice="same_capability_risk",
        language="en",
        source_kind=SOURCE_KIND,
        source_id="query_fixture_004",
        source_revision=SOURCE_REVISION,
        license_status=LICENSE_STATUS,
        public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
    ),
)


QRELS: tuple[SkillQrel, ...] = (
    SkillQrel("q_compat_schema_validate", "schema_contract_diff", 1),
    SkillQrel("q_compat_schema_validate", "schema_sample_mapper", 1),
    SkillQrel("q_compat_log_triage", "log_redaction_review", 1),
    SkillQrel("q_compat_log_triage", "error_signature_cluster", 1),
    SkillQrel("q_risk_archive_exact_day", "archive_day_lookup", 1),
    SkillQrel("q_risk_dependency_policy", "dependency_license_matrix", 1),
)


COMPATIBLE_SETS: tuple[SkillSet, ...] = (
    SkillSet(
        "q_compat_schema_validate",
        "set_schema_gold",
        ("schema_contract_diff", "schema_sample_mapper"),
        "compatible",
        "ordered_complementary_operations",
        REVIEW_STATUS,
    ),
    SkillSet(
        "q_compat_log_triage",
        "set_log_gold",
        ("log_redaction_review", "error_signature_cluster"),
        "compatible",
        "ordered_complementary_operations",
        REVIEW_STATUS,
    ),
)


REJECTED_SETS: tuple[SkillSet, ...] = (
    SkillSet(
        "q_compat_schema_validate",
        "set_schema_rejected_outline",
        ("schema_contract_diff", "schema_doc_outline"),
        "rejected",
        "wrong_second_operation",
        REVIEW_STATUS,
    ),
    SkillSet(
        "q_compat_log_triage",
        "set_log_rejected_volume",
        ("log_volume_rollup", "error_signature_cluster"),
        "rejected",
        "wrong_first_operation",
        REVIEW_STATUS,
    ),
)


EVIDENCE: tuple[EvidenceRecord, ...] = (
    EvidenceRecord(
        "evidence_archive_exact_day_v0",
        "q_risk_archive_exact_day",
        "The query asks for one date-scoped result and explicitly rejects an interval listing.",
    ),
    EvidenceRecord(
        "evidence_dependency_policy_v0",
        "q_risk_dependency_policy",
        "The query requires license-policy compatibility, which a version-age summary does not provide.",
    ),
)


RISK_PAIRS: tuple[RiskPair, ...] = (
    RiskPair(
        "q_risk_archive_exact_day",
        "archive_lookup",
        "archive_day_lookup",
        "archive_range_list",
        "output_scope_mismatch",
        "invented_contract_review",
        "evidence_archive_exact_day_v0",
    ),
    RiskPair(
        "q_risk_dependency_policy",
        "dependency_audit",
        "dependency_license_matrix",
        "package_age_report",
        "required_analysis_missing",
        "invented_contract_review",
        "evidence_dependency_policy_v0",
    ),
)


HARD_NEGATIVES: tuple[HardNegative, ...] = (
    HardNegative(
        "q_compat_schema_validate",
        "schema_doc_outline",
        "same_family_wrong_operation",
        "Uses the same contract vocabulary but produces documentation instead of checking the sample record.",
        REVIEW_STATUS,
    ),
    HardNegative(
        "q_compat_log_triage",
        "log_volume_rollup",
        "same_family_wrong_operation",
        "Uses the same diagnostic lines but aggregates volume instead of replacing credential-like values.",
        REVIEW_STATUS,
    ),
    HardNegative(
        "q_risk_archive_exact_day",
        "page_freshness_score",
        "same_family_non_pair",
        "Uses the same page vocabulary but measures recency of change, not snapshot existence.",
        REVIEW_STATUS,
    ),
    HardNegative(
        "q_risk_dependency_policy",
        "dependency_update_window",
        "same_family_non_pair",
        "Uses the same package inventory but plans maintenance instead of evaluating license compatibility.",
        REVIEW_STATUS,
    ),
)


FIXTURE = AgentSkillRoutingFixture(
    skills=SKILLS,
    queries=QUERIES,
    qrels=QRELS,
    compatible_sets=COMPATIBLE_SETS,
    rejected_sets=REJECTED_SETS,
    evidence=EVIDENCE,
    risk_pairs=RISK_PAIRS,
    hard_negatives=HARD_NEGATIVES,
    split=SPLIT,
    source_kind=SOURCE_KIND,
    source_revision=SOURCE_REVISION,
    license_status=LICENSE_STATUS,
    public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
)


def serialize_skill_document(skill: SkillDocument) -> str:
    """Serialize a skill card deterministically for document embedding."""
    return f"Name: {skill.name}\nDescription: {skill.description}\nBody: {skill.body}"


def fixture_to_dict(fixture: AgentSkillRoutingFixture) -> dict[str, Any]:
    """Convert the fixture to a JSON-compatible deterministic mapping."""
    return asdict(fixture)


def serialize_fixture(fixture: AgentSkillRoutingFixture) -> str:
    """Serialize the complete fixture deterministically."""
    return json.dumps(fixture_to_dict(fixture), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fixture_slice_counts(queries: tuple[SkillQuery, ...]) -> dict[str, int]:
    """Count fixture queries by slice."""
    return dict(sorted(Counter(query.slice for query in queries).items()))


def fixture_counts(fixture: AgentSkillRoutingFixture) -> dict[str, int]:
    """Return auditable record counts for result details."""
    return {
        "n_skills": len(fixture.skills),
        "n_queries": len(fixture.queries),
        "n_qrels": len(fixture.qrels),
        "n_compatible_sets": len(fixture.compatible_sets),
        "n_rejected_sets": len(fixture.rejected_sets),
        "n_risk_pairs": len(fixture.risk_pairs),
        "n_hard_negatives": len(fixture.hard_negatives),
    }


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} values")


def _all_records(fixture: AgentSkillRoutingFixture) -> tuple[Any, ...]:
    return (
        *fixture.skills,
        *fixture.queries,
        *fixture.qrels,
        *fixture.compatible_sets,
        *fixture.rejected_sets,
        *fixture.evidence,
        *fixture.risk_pairs,
        *fixture.hard_negatives,
    )


def validate_agent_skill_routing_fixture(fixture: AgentSkillRoutingFixture) -> None:
    """Validate fixture shape, references, provenance, and leakage constraints."""
    if fixture_counts(fixture) != {
        "n_skills": 12,
        "n_queries": 4,
        "n_qrels": 6,
        "n_compatible_sets": 2,
        "n_rejected_sets": 2,
        "n_risk_pairs": 2,
        "n_hard_negatives": 4,
    }:
        raise ValueError("Unexpected agent skill routing fixture shape")
    if (
        fixture.split != SPLIT
        or fixture.source_kind != SOURCE_KIND
        or fixture.source_revision != SOURCE_REVISION
        or fixture.license_status != LICENSE_STATUS
        or fixture.public_score_eligible
    ):
        raise ValueError("Unexpected fixture provenance or publication status")

    skill_ids = [skill.skill_id for skill in fixture.skills]
    query_ids = [query.query_id for query in fixture.queries]
    set_ids = [skill_set.set_id for skill_set in (*fixture.compatible_sets, *fixture.rejected_sets)]
    evidence_ids = [record.evidence_id for record in fixture.evidence]
    source_ids = [record.source_id for record in (*fixture.skills, *fixture.queries)]
    _require_unique(skill_ids, "skill id")
    _require_unique(query_ids, "query id")
    _require_unique(set_ids, "set id")
    _require_unique(evidence_ids, "evidence id")
    _require_unique(source_ids, "source id")
    _require_unique([(qrel.query_id, qrel.skill_id) for qrel in fixture.qrels], "qrel")

    skills_by_id = {skill.skill_id: skill for skill in fixture.skills}
    queries_by_id = {query.query_id: query for query in fixture.queries}
    evidence_by_id = {record.evidence_id: record for record in fixture.evidence}
    family_ids = {skill.family_id for skill in fixture.skills}
    if family_ids != {"schema_change", "log_diagnostics", "archive_lookup", "dependency_audit"}:
        raise ValueError("Unexpected or unresolved capability families")

    for record in _all_records(fixture):
        if getattr(record, "source_kind") != SOURCE_KIND:
            raise ValueError("Every fixture record must use the fixed local source kind")
        if getattr(record, "source_revision") != SOURCE_REVISION:
            raise ValueError("Every fixture record must use the fixed source revision")
        if getattr(record, "license_status") != LICENSE_STATUS:
            raise ValueError("Every fixture record must use the fixed local license status")
        if getattr(record, "public_score_eligible"):
            raise ValueError("Every fixture record must be ineligible for public scoring")

    if any(query.split != SPLIT or query.language != "en" for query in fixture.queries):
        raise ValueError("Every query must use the fixed fixture split and language")
    if fixture_slice_counts(fixture.queries) != {"compatible_set": 2, "same_capability_risk": 2}:
        raise ValueError("Unexpected query slices")

    positive_ids: dict[str, set[str]] = {query_id: set() for query_id in query_ids}
    for qrel in fixture.qrels:
        if qrel.query_id not in queries_by_id or qrel.skill_id not in skills_by_id or qrel.relevance != 1:
            raise ValueError("Invalid qrel")
        positive_ids[qrel.query_id].add(qrel.skill_id)

    compatibility_ids = {query.query_id for query in fixture.queries if query.slice == "compatible_set"}
    risk_ids = {query.query_id for query in fixture.queries if query.slice == "same_capability_risk"}
    for query_id in compatibility_ids:
        if len(positive_ids[query_id]) < 2:
            raise ValueError("Every compatibility query must have at least two positives")
    for query_id in risk_ids:
        if len(positive_ids[query_id]) != 1:
            raise ValueError("Every risk query must have exactly one helpful qrel")

    compatible_by_query = {skill_set.query_id: skill_set for skill_set in fixture.compatible_sets}
    rejected_by_query: dict[str, list[SkillSet]] = {query_id: [] for query_id in compatibility_ids}
    for skill_set in (*fixture.compatible_sets, *fixture.rejected_sets):
        if skill_set.query_id not in compatibility_ids:
            raise ValueError("Skill sets must reference compatibility queries")
        if not set(skill_set.skill_ids).issubset(skills_by_id):
            raise ValueError("Skill set references an unknown skill")
        if len(skill_set.skill_ids) != len(set(skill_set.skill_ids)):
            raise ValueError("Skill set contains duplicate members")
        if skill_set.review_status != REVIEW_STATUS:
            raise ValueError("Skill set review record is missing")
        if skill_set.label == "rejected":
            rejected_by_query[skill_set.query_id].append(skill_set)
    if set(compatible_by_query) != compatibility_ids:
        raise ValueError("Every compatibility query must have one compatible set")
    for query_id, skill_set in compatible_by_query.items():
        if skill_set.label != "compatible" or set(skill_set.skill_ids) != positive_ids[query_id]:
            raise ValueError("Compatible set must match the query qrels")
    if any(not rejected_by_query[query_id] for query_id in compatibility_ids):
        raise ValueError("Every compatibility query must have a rejected set")

    pair_by_query: dict[str, RiskPair] = {}
    for pair in fixture.risk_pairs:
        if pair.query_id in pair_by_query or pair.query_id not in risk_ids:
            raise ValueError("Every risk query must have exactly one risk pair")
        if pair.family_id not in family_ids or pair.helpful_skill_id == pair.risky_skill_id:
            raise ValueError("Invalid risk pair identity")
        helpful = skills_by_id.get(pair.helpful_skill_id)
        risky = skills_by_id.get(pair.risky_skill_id)
        evidence = evidence_by_id.get(pair.evidence_id)
        if not helpful or not risky or helpful.family_id != pair.family_id or risky.family_id != pair.family_id:
            raise ValueError("Risk pair siblings must resolve to the same family")
        if positive_ids[pair.query_id] != {pair.helpful_skill_id}:
            raise ValueError("Risk-pair helpful skill must match the qrel")
        if not evidence or evidence.query_id != pair.query_id:
            raise ValueError("Risk-pair evidence must be unique and resolvable")
        pair_by_query[pair.query_id] = pair
    if set(pair_by_query) != risk_ids:
        raise ValueError("Every risk query must have exactly one risk pair")

    hard_negative_keys: set[tuple[str, str]] = set()
    for negative in fixture.hard_negatives:
        key = (negative.query_id, negative.skill_id)
        if key in hard_negative_keys or negative.query_id not in queries_by_id or negative.skill_id not in skills_by_id:
            raise ValueError("Invalid hard negative")
        if negative.skill_id in positive_ids[negative.query_id] or negative.review_status != REVIEW_STATUS:
            raise ValueError("Hard negative conflicts with a positive or lacks review")
        pair = pair_by_query.get(negative.query_id)
        if pair and negative.skill_id in {pair.helpful_skill_id, pair.risky_skill_id}:
            raise ValueError("Risk-pair siblings cannot be their own hard negatives")
        hard_negative_keys.add(key)
    if {negative.query_id for negative in fixture.hard_negatives} != set(query_ids):
        raise ValueError("Every query must have a reviewed hard negative")

    normalized_skills = [_normalized_text(serialize_skill_document(skill)) for skill in fixture.skills]
    normalized_queries = [_normalized_text(query.text) for query in fixture.queries]
    _require_unique(normalized_skills, "normalized skill text")
    _require_unique(normalized_queries, "normalized query text")
    normalized_names = [_normalized_text(skill.name) for skill in fixture.skills]
    if any(name in query for name in normalized_names for query in normalized_queries):
        raise ValueError("Query leaks an exact normalized skill name")

    serialized = serialize_fixture(fixture).lower()
    prohibited_fragments = (
        "http://",
        "https://",
        "www.",
        "github",
        "huggingface",
        "@",
        "api_key",
        "access_token",
        "bearer ",
        "password=",
    )
    if any(fragment in serialized for fragment in prohibited_fragments):
        raise ValueError("Fixture contains an external location or credential-like value")


def load_agent_skill_routing_fixture() -> AgentSkillRoutingFixture:
    """Load and validate the deterministic local fixture."""
    validate_agent_skill_routing_fixture(FIXTURE)
    return FIXTURE
