"""Deterministic contracts for the restricted BRIGHT label-quality audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from mm_embed.benchmark.retrieval_v01 import json_dumps, write_jsonl

AUDIT_ID = "bright-label-quality-audit-v0.1"
SCHEMA_VERSION = "1"
TRACKS = ("economics", "psychology")
QUERY_LENGTH_STRATA = ("short", "medium", "long")
REVIEW_PLAN_SHA256 = "93848a826c12acf9cb6fed50d71dd9490be7baebfcea1122dd4fffb4881fdc4e"
AUDIT_PACK_SHA256 = "9549525cbd35e0a3c520ad16b5961b3f5e8658ff7cb4b5061e9edd39bc96c956"
REVIEW_PLAN_ROWS = 120
AUDIT_PACK_ROWS = 120
GOLD_ITEMS = 1_235
CANDIDATE_ITEMS = 2_804
DOUBLE_REVIEW_PER_TRACK_AND_LENGTH = 4
DOUBLE_REVIEW_QUERIES = 24
DOUBLE_REVIEW_SALT = f"{AUDIT_ID}-double-review-v1"
UNSUPPORTED_GOLD_THRESHOLD = 0.05
MISSING_POSITIVE_QUERY_THRESHOLD = 0.10
MINIMUM_KAPPA = 0.70
SCHEMA_NAMES = (
    "bright-label-audit-adjudication-v01.schema.json",
    "bright-label-audit-annotation-v01.schema.json",
    "bright-label-audit-blind-input-v01.schema.json",
    "bright-label-audit-frozen-manifest-v01.schema.json",
    "bright-label-audit-predeclaration-v01.schema.json",
    "bright-label-audit-safe-aggregate-v01.schema.json",
    "bright-label-audit-selection-v01.schema.json",
    "bright-label-audit-validator-access-v01.schema.json",
    "bright-label-audit-validator-seal-v01.schema.json",
)

SUPPORT_LABELS = frozenset({"supports", "partially_supports", "does_not_support", "uncertain"})
AMBIGUITY_LABELS = frozenset({"unambiguous", "ambiguous", "uncertain"})
RELEVANCE_LABELS = frozenset({"relevant", "not_relevant", "uncertain"})
BINARY_RELEVANCE_LABELS = frozenset({"relevant", "not_relevant"})
TERNARY_LABELS = frozenset({"yes", "no", "uncertain"})
MISSING_LABELS = frozenset({"credible", "not_credible", "not_applicable", "uncertain"})
RISK_LABELS = frozenset({"none", "low", "moderate", "high", "uncertain"})
CONFIDENCE_LABELS = frozenset({"high", "medium", "low"})
DECISION_STATUSES = frozenset({"decided", "uncertain", "abstain"})


class BrightLabelAuditError(ValueError):
    """A stable fail-closed label-audit validation error."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256((json_dumps(value) + "\n").encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(value) + "\n", encoding="utf-8")
    return {"rows": 1, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise BrightLabelAuditError(f"non-object JSONL row at line {line_number}")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise BrightLabelAuditError(f"missing or invalid JSONL input: {path.name}") from error
    return tuple(rows)


def _schema_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "schemas" / name


@cache
def _load_schema(name: str) -> dict[str, Any]:
    schema = json.loads(_schema_path(name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


@cache
def _schema_registry() -> Registry:
    registry = Registry()
    for path in sorted(_schema_path(".").glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        if schema_id := schema.get("$id"):
            registry = registry.with_resource(schema_id, Resource.from_contents(schema))
    return registry


def _validate_schema(value: Any, name: str, subject: str) -> None:
    errors = sorted(
        Draft202012Validator(
            _load_schema(name),
            registry=_schema_registry(),
            format_checker=FormatChecker(),
        ).iter_errors(value),
        key=lambda item: list(item.path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.path) or "<root>"
        raise BrightLabelAuditError(f"invalid {subject} at {location}: {first.message}")


def _identity_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    values = sorted(f"{row['track']}\0{row['query_id']}" for row in rows)
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _validate_source_shapes(
    plan: Sequence[dict[str, Any]],
    pack: Sequence[dict[str, Any]],
    *,
    expected_rows: int,
    expected_gold_items: int,
    expected_candidate_items: int,
) -> None:
    if len(plan) != expected_rows or len(pack) != expected_rows:
        raise BrightLabelAuditError("frozen source row count mismatch")
    plan_keys = [(row.get("track"), row.get("query_id")) for row in plan]
    pack_keys = [(row.get("track"), row.get("query_id")) for row in pack]
    if len(set(plan_keys)) != expected_rows or plan_keys != pack_keys:
        raise BrightLabelAuditError("review plan and audit pack query identities do not match exactly")
    for plan_row, pack_row in zip(plan, pack, strict=True):
        if plan_row != {key: pack_row[key] for key in plan_row}:
            raise BrightLabelAuditError("audit pack does not preserve its review-plan row")
        if plan_row.get("track") not in TRACKS:
            raise BrightLabelAuditError("source row contains an unknown track")
        if plan_row.get("strata", {}).get("query_length") not in QUERY_LENGTH_STRATA:
            raise BrightLabelAuditError("source row contains an unknown query-length stratum")
        gold = pack_row.get("gold_documents")
        candidates = pack_row.get("baseline_top10_union")
        if not isinstance(gold, list) or not isinstance(candidates, list):
            raise BrightLabelAuditError("source row is missing gold or candidate items")
        gold_ids = [item.get("document_id") for item in gold]
        candidate_ids = [item.get("document_id") for item in candidates]
        if len(gold_ids) != len(set(gold_ids)) or len(candidate_ids) != len(set(candidate_ids)):
            raise BrightLabelAuditError("source row contains duplicate item identities")
    if sum(len(row["gold_documents"]) for row in pack) != expected_gold_items:
        raise BrightLabelAuditError("frozen gold-item count mismatch")
    if sum(len(row["baseline_top10_union"]) for row in pack) != expected_candidate_items:
        raise BrightLabelAuditError("frozen candidate-item count mismatch")


def verify_frozen_inputs(
    review_plan_path: str | Path,
    audit_pack_path: str | Path,
    *,
    review_plan_sha256: str = REVIEW_PLAN_SHA256,
    audit_pack_sha256: str = AUDIT_PACK_SHA256,
    expected_rows: int = REVIEW_PLAN_ROWS,
    expected_gold_items: int = GOLD_ITEMS,
    expected_candidate_items: int = CANDIDATE_ITEMS,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Validate frozen byte identities, query bindings, and complete item counts."""
    plan_path = Path(review_plan_path)
    pack_path = Path(audit_pack_path)
    if _sha256_file(plan_path) != review_plan_sha256:
        raise BrightLabelAuditError("review plan identity mismatch")
    if _sha256_file(pack_path) != audit_pack_sha256:
        raise BrightLabelAuditError("audit pack identity mismatch")
    plan = _read_jsonl(plan_path)
    pack = _read_jsonl(pack_path)
    _validate_source_shapes(
        plan,
        pack,
        expected_rows=expected_rows,
        expected_gold_items=expected_gold_items,
        expected_candidate_items=expected_candidate_items,
    )
    return plan, pack


def select_double_review(plan: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Select four queries per track and query-length stratum without model scores."""
    selected = []
    for track in TRACKS:
        for stratum in QUERY_LENGTH_STRATA:
            candidates = [row for row in plan if row["track"] == track and row["strata"]["query_length"] == stratum]
            candidates.sort(
                key=lambda row: (
                    hashlib.sha256(f"{DOUBLE_REVIEW_SALT}\0{track}\0{stratum}\0{row['query_id']}".encode()).hexdigest(),
                    row["query_id"],
                )
            )
            if len(candidates) < DOUBLE_REVIEW_PER_TRACK_AND_LENGTH:
                raise BrightLabelAuditError("insufficient rows for frozen double-review stratum")
            selected.extend(candidates[:DOUBLE_REVIEW_PER_TRACK_AND_LENGTH])
    return tuple(sorted(selected, key=lambda row: (row["track"], row["query_id"])))


def _blind_row(source_row: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for item in source_row["baseline_top10_union"]:
        candidates.append({"document_id": item["document_id"], "content": item["content"]})
    candidates.sort(
        key=lambda item: hashlib.sha256(
            f"{AUDIT_ID}\0blind-order\0{source_row['track']}\0{source_row['query_id']}\0{item['document_id']}".encode()
        ).hexdigest()
    )
    value = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "track": source_row["track"],
        "query_id": source_row["query_id"],
        "query": source_row["query"],
        "source_row_sha256": _sha256_json(source_row),
        "gold_documents": [
            {"document_id": item["document_id"], "content": item["content"], "grade": item["grade"]}
            for item in source_row["gold_documents"]
        ],
        "candidate_documents": candidates,
    }
    _validate_schema(value, "bright-label-audit-blind-input-v01.schema.json", "blind input row")
    return value


def freeze_audit(
    review_plan_path: str | Path,
    audit_pack_path: str | Path,
    rubric_path: str | Path,
    restricted_output: str | Path,
    predeclaration_output: str | Path,
) -> dict[str, Any]:
    """Freeze restricted blind inputs and a publication-safe predeclaration."""
    plan, pack = verify_frozen_inputs(review_plan_path, audit_pack_path)
    rubric = Path(rubric_path)
    output = Path(restricted_output)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_double_review(plan)
    selected_keys = {(row["track"], row["query_id"]) for row in selected}
    blind_rows = tuple(_blind_row(row) for row in pack)
    core_rows = tuple(row for row in blind_rows if (row["track"], row["query_id"]) in selected_keys)
    selection_rows = tuple(
        {
            "track": row["track"],
            "query_id": row["query_id"],
            "query_length_stratum": row["strata"]["query_length"],
            "source_plan_row_sha256": _sha256_json(row),
        }
        for row in selected
    )
    primary_entry = write_jsonl(output / "primary" / "input.jsonl", blind_rows)
    core_entry = write_jsonl(output / "validator-blind" / "core-input.jsonl", core_rows)
    selection_entry = write_jsonl(output / "control" / "double-review-selection.jsonl", selection_rows)
    validator_access = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "phase": "independent_core_review",
        "core_input_sha256": core_entry["sha256"],
        "core_input_rows": DOUBLE_REVIEW_QUERIES,
        "rubric_sha256": _sha256_file(rubric),
        "annotation_schema_sha256": _sha256_file(_schema_path("bright-label-audit-annotation-v01.schema.json")),
        "allowed_input_roles": ["validator_core_input", "frozen_rubric", "annotation_schema"],
        "primary_input_allowed": False,
        "primary_annotations_allowed": False,
        "core_created_before_primary_annotations": True,
        "comparison_allowed_before_validator_seal": False,
    }
    validator_access_entry = _write_json(output / "validator-blind" / "access-manifest.json", validator_access)
    restricted_manifest = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "source": {
            "review_plan_sha256": REVIEW_PLAN_SHA256,
            "audit_pack_sha256": AUDIT_PACK_SHA256,
        },
        "rubric_sha256": _sha256_file(rubric),
        "double_review": {
            "queries": DOUBLE_REVIEW_QUERIES,
            "selection_identity_sha256": _identity_rows(selected),
            "selection_artifact": selection_entry,
            "selection_salt": DOUBLE_REVIEW_SALT,
            "per_track_and_query_length_stratum": DOUBLE_REVIEW_PER_TRACK_AND_LENGTH,
        },
        "primary_input": primary_entry,
        "validator_core_input": core_entry,
        "validator_access_manifest": validator_access_entry,
        "contains_restricted_text_and_identifiers": True,
        "publication_allowed": False,
    }
    manifest_entry = _write_json(output / "frozen-input-manifest.json", restricted_manifest)
    schema_identities = {name: _sha256_file(_schema_path(name)) for name in SCHEMA_NAMES}
    predeclaration = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "story_id": "S-20260905-001",
        "phase": "pre_label",
        "source_identities": {
            "review_plan_sha256": REVIEW_PLAN_SHA256,
            "audit_pack_sha256": AUDIT_PACK_SHA256,
            "review_plan_rows": REVIEW_PLAN_ROWS,
            "audit_pack_rows": AUDIT_PACK_ROWS,
            "gold_items": GOLD_ITEMS,
            "candidate_items": CANDIDATE_ITEMS,
        },
        "frozen_artifact_identities": {
            "frozen_input_manifest_sha256": manifest_entry["sha256"],
            "primary_input_sha256": primary_entry["sha256"],
            "validator_core_input_sha256": core_entry["sha256"],
            "validator_access_manifest_sha256": validator_access_entry["sha256"],
            "double_review_selection_sha256": selection_entry["sha256"],
        },
        "rubric_sha256": restricted_manifest["rubric_sha256"],
        "schema_identities": schema_identities,
        "double_review": {
            "queries": DOUBLE_REVIEW_QUERIES,
            "selection_identity_sha256": restricted_manifest["double_review"]["selection_identity_sha256"],
            "allocation": {
                "per_track": 12,
                "per_track_and_query_length_stratum": DOUBLE_REVIEW_PER_TRACK_AND_LENGTH,
            },
            "selection_score_used": False,
            "agreement_metric": "cohen_kappa_binary_candidate_relevance",
        },
        "thresholds": {
            "unsupported_or_ambiguous_gold_fraction_exceeds": UNSUPPORTED_GOLD_THRESHOLD,
            "credible_missing_positive_query_fraction_exceeds": MISSING_POSITIVE_QUERY_THRESHOLD,
            "binary_candidate_relevance_kappa_below": MINIMUM_KAPPA,
        },
        "judgment_origin_allowed": ["agent_judged", "llm_assisted"],
        "human_gold_standard_claim_allowed": False,
        "official_qrels_modified": False,
        "restricted_artifacts_tracked": False,
        "safe_aggregate_contains_restricted_text_or_identifiers": False,
        "publication": {"classification": "research_only", "leaderboard": "closed", "export": "closed"},
    }
    _validate_schema(predeclaration, "bright-label-audit-predeclaration-v01.schema.json", "predeclaration")
    predeclaration_path = Path(predeclaration_output)
    predeclaration_entry = _write_json(predeclaration_path, predeclaration)
    predeclaration_path.with_suffix(predeclaration_path.suffix + ".sha256").write_text(
        predeclaration_entry["sha256"] + "\n",
        encoding="utf-8",
    )
    return {
        "restricted_manifest": manifest_entry,
        "predeclaration": predeclaration_entry,
        "double_review_selection_identity_sha256": predeclaration["double_review"]["selection_identity_sha256"],
    }


def _validate_decision_status(item: dict[str, Any], uncertain: bool) -> None:
    status = item["decision_status"]
    confidence = item["confidence"]
    if status == "decided" and uncertain:
        raise BrightLabelAuditError("decided judgment contains an uncertain label")
    if status in {"uncertain", "abstain"} and (not uncertain or confidence != "low"):
        raise BrightLabelAuditError("uncertain or abstained judgment must use uncertain labels and low confidence")
    if status == "abstain" and not item["notes"].strip():
        raise BrightLabelAuditError("abstained judgment requires a rationale")


def _validate_provenance(provenance: dict[str, Any], row: dict[str, Any], input_sha256: str) -> None:
    if provenance["input_artifact_sha256"] != input_sha256:
        raise BrightLabelAuditError("judgment provenance input artifact identity mismatch")
    if provenance["input_row_sha256"] != row["input_row_sha256"]:
        raise BrightLabelAuditError("judgment provenance input row identity mismatch")
    if provenance["rubric_sha256"] != row["rubric_sha256"]:
        raise BrightLabelAuditError("judgment provenance rubric identity mismatch")


def validate_annotations(
    annotation_path: str | Path,
    input_path: str | Path,
    *,
    expected_role: str,
) -> tuple[dict[str, Any], ...]:
    """Validate complete annotations against a source-only input artifact."""
    annotations = _read_jsonl(Path(annotation_path))
    inputs = _read_jsonl(Path(input_path))
    input_sha256 = _sha256_file(Path(input_path))
    input_by_key = {(row["track"], row["query_id"]): row for row in inputs}
    annotation_by_key = {(row.get("track"), row.get("query_id")): row for row in annotations}
    if len(annotation_by_key) != len(annotations) or set(annotation_by_key) != set(input_by_key):
        raise BrightLabelAuditError("annotation query coverage does not match its input artifact")
    for key, row in annotation_by_key.items():
        _validate_schema(row, "bright-label-audit-annotation-v01.schema.json", "annotation row")
        if row["review_role"] != expected_role:
            raise BrightLabelAuditError("annotation review role mismatch")
        source = input_by_key[key]
        expected_agent_role = "executor_primary" if expected_role == "primary" else "independent_validator"
        source_row_sha256 = _sha256_json(source)
        if row["input_row_sha256"] != source_row_sha256:
            raise BrightLabelAuditError("annotation input row identity mismatch")
        if [item["document_id"] for item in row["gold_reviews"]] != [
            item["document_id"] for item in source["gold_documents"]
        ]:
            raise BrightLabelAuditError("gold annotation coverage or order mismatch")
        if {item["document_id"] for item in row["candidate_reviews"]} != {
            item["document_id"] for item in source["candidate_documents"]
        } or len(row["candidate_reviews"]) != len(source["candidate_documents"]):
            raise BrightLabelAuditError("candidate annotation coverage mismatch")
        for item in row["gold_reviews"]:
            _validate_decision_status(item, "uncertain" in {item["support"], item["ambiguity"]})
            _validate_provenance(item["provenance"], row, input_sha256)
            if item["provenance"]["agent_role"] != expected_agent_role:
                raise BrightLabelAuditError("judgment agent role does not match annotation role")
        for item in row["candidate_reviews"]:
            uncertain = "uncertain" in {
                item["relevance"],
                item["likely_hard_negative"],
                item["suspected_missing_positive"],
                item["sensitive_information_risk"],
            }
            _validate_decision_status(item, uncertain)
            _validate_provenance(item["provenance"], row, input_sha256)
            if item["provenance"]["agent_role"] != expected_agent_role:
                raise BrightLabelAuditError("judgment agent role does not match annotation role")
            if item["suspected_missing_positive"] == "credible" and item["relevance"] != "relevant":
                raise BrightLabelAuditError("credible missing positive must be relevant")
            if item["likely_hard_negative"] == "yes" and item["relevance"] != "not_relevant":
                raise BrightLabelAuditError("likely hard negative must be not relevant")
            is_existing_positive = item["document_id"] in {
                source_item["document_id"] for source_item in source["gold_documents"]
            }
            if is_existing_positive and item["suspected_missing_positive"] != "not_applicable":
                raise BrightLabelAuditError("existing positive cannot be marked as a missing positive")
            if not is_existing_positive and item["suspected_missing_positive"] == "not_applicable":
                raise BrightLabelAuditError("unjudged candidate requires a missing-positive judgment")
            if item["sensitive_information_risk"] == "none" and item["sensitive_information_types"]:
                raise BrightLabelAuditError("no sensitive-information risk requires an empty type list")
            if (
                item["sensitive_information_risk"] in {"low", "moderate", "high"}
                and not item["sensitive_information_types"]
            ):
                raise BrightLabelAuditError("identified sensitive-information risk requires a type")
        query_review = row["query_review"]
        query_uncertain = "uncertain" in {
            query_review["credible_missing_positive"],
            query_review["likely_hard_negative_present"],
            query_review["qrels_completeness"],
            query_review["answerability"],
            query_review["label_incompleteness_risk"],
            query_review["sensitive_information_risk"],
        }
        _validate_decision_status(query_review, query_uncertain)
        _validate_provenance(query_review["provenance"], row, input_sha256)
        if query_review["provenance"]["agent_role"] != expected_agent_role:
            raise BrightLabelAuditError("judgment agent role does not match annotation role")
        credible_ids = sorted(
            item["document_id"] for item in row["candidate_reviews"] if item["suspected_missing_positive"] == "credible"
        )
        hard_ids = sorted(
            item["document_id"] for item in row["candidate_reviews"] if item["likely_hard_negative"] == "yes"
        )
        if sorted(query_review["credible_missing_positive_ids"]) != credible_ids:
            raise BrightLabelAuditError("query missing-positive IDs do not match item judgments")
        if sorted(query_review["likely_hard_negative_ids"]) != hard_ids:
            raise BrightLabelAuditError("query hard-negative IDs do not match item judgments")
        expected_missing = "yes" if credible_ids else "no"
        expected_hard = "yes" if hard_ids else "no"
        if query_review["credible_missing_positive"] not in {expected_missing, "uncertain"}:
            raise BrightLabelAuditError("query missing-positive summary contradicts item judgments")
        if query_review["likely_hard_negative_present"] not in {expected_hard, "uncertain"}:
            raise BrightLabelAuditError("query hard-negative summary contradicts item judgments")
        if credible_ids and query_review["qrels_completeness"] == "complete_enough":
            raise BrightLabelAuditError("credible missing positive contradicts complete-enough qrels")
    return tuple(annotation_by_key[key] for key in sorted(annotation_by_key))


def prepare_validator_supplement(
    primary_annotation_path: str | Path,
    primary_input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create source-only query rows for every primary uncertain or low-confidence item."""
    primary = validate_annotations(primary_annotation_path, primary_input_path, expected_role="primary")
    source = _read_jsonl(Path(primary_input_path))
    source_by_key = {(row["track"], row["query_id"]): row for row in source}
    keys = set()
    for row in primary:
        judgments = [*row["gold_reviews"], *row["candidate_reviews"], row["query_review"]]
        if any(item["confidence"] == "low" or item["decision_status"] != "decided" for item in judgments):
            keys.add((row["track"], row["query_id"]))
    rows = tuple(source_by_key[key] for key in sorted(keys))
    entry = write_jsonl(output_path, rows)
    return {**entry, "queries": len(rows), "selection_reason": "primary_low_confidence_or_unresolved"}


def seal_validator_annotations(
    validator_annotation_path: str | Path,
    validator_input_path: str | Path,
    access_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal independent Validator labels before primary/Validator comparison."""
    annotations = validate_annotations(
        validator_annotation_path,
        validator_input_path,
        expected_role="validator",
    )
    access_path = Path(access_manifest_path)
    access = json.loads(access_path.read_text(encoding="utf-8"))
    expected_access_keys = {
        "schema_version",
        "audit_id",
        "phase",
        "core_input_sha256",
        "core_input_rows",
        "rubric_sha256",
        "annotation_schema_sha256",
        "allowed_input_roles",
        "primary_input_allowed",
        "primary_annotations_allowed",
        "core_created_before_primary_annotations",
        "comparison_allowed_before_validator_seal",
    }
    if set(access) != expected_access_keys:
        raise BrightLabelAuditError("Validator access manifest fields do not match the fixed contract")
    if access["phase"] != "independent_core_review":
        raise BrightLabelAuditError("Validator access manifest phase mismatch")
    if access["core_input_sha256"] != _sha256_file(Path(validator_input_path)):
        raise BrightLabelAuditError("Validator core input does not match its access manifest")
    if access["core_input_rows"] != len(annotations):
        raise BrightLabelAuditError("Validator core row count does not match its access manifest")
    if access["annotation_schema_sha256"] != _sha256_file(
        _schema_path("bright-label-audit-annotation-v01.schema.json")
    ):
        raise BrightLabelAuditError("Validator annotation schema does not match its access manifest")
    if (
        access["primary_input_allowed"] is not False
        or access["primary_annotations_allowed"] is not False
        or access["core_created_before_primary_annotations"] is not True
        or access["comparison_allowed_before_validator_seal"] is not False
    ):
        raise BrightLabelAuditError("Validator access manifest does not enforce blind review")
    if any(row["rubric_sha256"] != access["rubric_sha256"] for row in annotations):
        raise BrightLabelAuditError("Validator annotations do not use the access-manifest rubric")
    annotation_path = Path(validator_annotation_path)
    seal = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "phase": "validator_core_sealed_before_comparison",
        "validator_input_sha256": _sha256_file(Path(validator_input_path)),
        "validator_annotations_sha256": _sha256_file(annotation_path),
        "validator_annotation_rows": len(annotations),
        "validator_annotation_bytes": annotation_path.stat().st_size,
        "access_manifest_sha256": _sha256_file(access_path),
        "rubric_sha256": access["rubric_sha256"],
        "comparison_unlocked": True,
    }
    _validate_schema(seal, "bright-label-audit-validator-seal-v01.schema.json", "Validator seal")
    _write_json(Path(output_path), seal)
    return seal


def cohen_kappa(primary: Sequence[str], validator: Sequence[str]) -> dict[str, Any]:
    """Compute unweighted Cohen's kappa for two binary label sequences."""
    if len(primary) != len(validator) or not primary:
        raise BrightLabelAuditError("kappa requires non-empty paired labels")
    if any(value not in BINARY_RELEVANCE_LABELS for value in [*primary, *validator]):
        raise BrightLabelAuditError("kappa accepts only binary relevance labels")
    pairs = Counter(zip(primary, validator, strict=True))
    total = len(primary)
    observed = sum(left == right for left, right in zip(primary, validator, strict=True)) / total
    primary_positive = sum(value == "relevant" for value in primary) / total
    validator_positive = sum(value == "relevant" for value in validator) / total
    expected = primary_positive * validator_positive + (1 - primary_positive) * (1 - validator_positive)
    kappa = None if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)
    return {
        "pairs": total,
        "pair_counts": {f"{left}__{right}": count for (left, right), count in sorted(pairs.items())},
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
    }


def _counter(values: Iterable[str], allowed: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts.get(key, 0) for key in sorted(allowed)}


def _legacy_recompute_safe_aggregate(
    primary_annotation_path: str | Path,
    primary_input_path: str | Path,
    validator_annotation_path: str | Path | None,
    validator_core_input_path: str | Path,
    selection_path: str | Path,
    predeclaration_path: str | Path,
    validator_seal_path: str | Path | None,
) -> dict[str, Any]:
    """Recompute irreversible counts, agreement, and fail-closed audit gates."""
    primary = validate_annotations(primary_annotation_path, primary_input_path, expected_role="primary")
    selection = _read_jsonl(Path(selection_path))
    predeclaration = json.loads(Path(predeclaration_path).read_text(encoding="utf-8"))
    _validate_schema(predeclaration, "bright-label-audit-predeclaration-v01.schema.json", "predeclaration")
    if predeclaration["source_identities"]["review_plan_sha256"] != REVIEW_PLAN_SHA256:
        raise BrightLabelAuditError("predeclaration review-plan identity mismatch")
    if predeclaration["source_identities"]["audit_pack_sha256"] != AUDIT_PACK_SHA256:
        raise BrightLabelAuditError("predeclaration audit-pack identity mismatch")
    for name in SCHEMA_NAMES:
        if predeclaration["schema_identities"][name] != _sha256_file(_schema_path(name)):
            raise BrightLabelAuditError("current schema does not match the frozen predeclaration")
    if len(selection) != DOUBLE_REVIEW_QUERIES:
        raise BrightLabelAuditError("double-review selection row count mismatch")
    selection_identity = _identity_rows(selection)
    if selection_identity != predeclaration["double_review"]["selection_identity_sha256"]:
        raise BrightLabelAuditError("double-review selection does not match the frozen predeclaration")
    if any(row["rubric_sha256"] != predeclaration["rubric_sha256"] for row in primary):
        raise BrightLabelAuditError("primary annotations do not use the frozen rubric")
    gold = [item for row in primary for item in row["gold_reviews"]]
    candidates = [item for row in primary for item in row["candidate_reviews"]]
    queries = [row["query_review"] for row in primary]
    unsupported = sum(item["support"] != "supports" or item["ambiguity"] != "unambiguous" for item in gold)
    missing_queries = sum(item["credible_missing_positive"] == "yes" for item in queries)
    unresolved_primary = sum(
        item["decision_status"] != "decided"
        for row in primary
        for item in [*row["gold_reviews"], *row["candidate_reviews"], row["query_review"]]
    )
    agreement: dict[str, Any] = {
        "status": "pending",
        "double_review_queries": DOUBLE_REVIEW_QUERIES,
        "binary_pairs": 0,
        "excluded_unresolved_pairs": 0,
        "kappa": None,
    }
    if validator_annotation_path is not None:
        if validator_seal_path is None:
            raise BrightLabelAuditError("Validator annotations require a pre-comparison seal")
        seal = json.loads(Path(validator_seal_path).read_text(encoding="utf-8"))
        _validate_schema(seal, "bright-label-audit-validator-seal-v01.schema.json", "Validator seal")
        if seal["validator_input_sha256"] != _sha256_file(Path(validator_core_input_path)):
            raise BrightLabelAuditError("Validator seal input identity mismatch")
        if seal["validator_annotations_sha256"] != _sha256_file(Path(validator_annotation_path)):
            raise BrightLabelAuditError("Validator seal annotation identity mismatch")
        if seal["rubric_sha256"] != predeclaration["rubric_sha256"]:
            raise BrightLabelAuditError("Validator seal rubric identity mismatch")
        validator = validate_annotations(
            validator_annotation_path,
            validator_core_input_path,
            expected_role="validator",
        )
        primary_by_key = {(row["track"], row["query_id"]): row for row in primary}
        validator_by_key = {(row["track"], row["query_id"]): row for row in validator}
        selected_keys = {(row["track"], row["query_id"]) for row in selection}
        if set(validator_by_key) != selected_keys:
            raise BrightLabelAuditError("validator annotations do not match the frozen double-review selection")
        if any(row["rubric_sha256"] != predeclaration["rubric_sha256"] for row in validator):
            raise BrightLabelAuditError("Validator annotations do not use the frozen rubric")
        primary_sessions = {
            item["provenance"]["session_id"]
            for row in primary
            for item in [*row["gold_reviews"], *row["candidate_reviews"], row["query_review"]]
        }
        validator_sessions = {
            item["provenance"]["session_id"]
            for row in validator
            for item in [*row["gold_reviews"], *row["candidate_reviews"], row["query_review"]]
        }
        if primary_sessions & validator_sessions:
            raise BrightLabelAuditError("primary and Validator sessions are not independent")
        primary_labels = []
        validator_labels = []
        unresolved = 0
        for key in sorted(selected_keys):
            left = {item["document_id"]: item["relevance"] for item in primary_by_key[key]["candidate_reviews"]}
            right = {item["document_id"]: item["relevance"] for item in validator_by_key[key]["candidate_reviews"]}
            if set(left) != set(right):
                raise BrightLabelAuditError("paired candidate identities do not match")
            for document_id in sorted(left):
                if left[document_id] in BINARY_RELEVANCE_LABELS and right[document_id] in BINARY_RELEVANCE_LABELS:
                    primary_labels.append(left[document_id])
                    validator_labels.append(right[document_id])
                else:
                    unresolved += 1
        computed = cohen_kappa(primary_labels, validator_labels) if primary_labels else None
        agreement_fail_closed = unresolved > 0 or computed is None or computed["kappa"] is None
        agreement = {
            "status": "fail_closed" if agreement_fail_closed else "complete",
            "double_review_queries": len(selected_keys),
            "binary_pairs": len(primary_labels),
            "excluded_unresolved_pairs": unresolved,
            "observed_agreement": None if computed is None else computed["observed_agreement"],
            "expected_agreement": None if computed is None else computed["expected_agreement"],
            "kappa": None if computed is None else computed["kappa"],
        }
    unsupported_fraction = unsupported / len(gold)
    missing_fraction = missing_queries / len(queries)
    kappa = agreement["kappa"]
    gates = {
        "unsupported_or_ambiguous_gold": {
            "threshold": UNSUPPORTED_GOLD_THRESHOLD,
            "comparison": "exceeds",
            "numerator": unsupported,
            "denominator": len(gold),
            "value": unsupported_fraction,
            "status": "fail" if unsupported_fraction > UNSUPPORTED_GOLD_THRESHOLD else "pass",
        },
        "credible_missing_positive_queries": {
            "threshold": MISSING_POSITIVE_QUERY_THRESHOLD,
            "comparison": "exceeds",
            "numerator": missing_queries,
            "denominator": len(queries),
            "value": missing_fraction,
            "status": "fail" if missing_fraction > MISSING_POSITIVE_QUERY_THRESHOLD else "pass",
        },
        "binary_candidate_relevance_kappa": {
            "threshold": MINIMUM_KAPPA,
            "comparison": "below",
            "value": kappa,
            "status": (
                "pending"
                if agreement["status"] == "pending"
                else "fail_closed"
                if agreement["status"] == "fail_closed"
                else "fail"
                if kappa < MINIMUM_KAPPA
                else "pass"
            ),
        },
        "unresolved_judgments": {
            "threshold": 0,
            "comparison": "exceeds",
            "value": unresolved_primary + agreement["excluded_unresolved_pairs"],
            "status": "fail_closed" if unresolved_primary + agreement["excluded_unresolved_pairs"] else "pass",
        },
    }
    gate_statuses = {gate["status"] for gate in gates.values()}
    if gate_statuses == {"pass"}:
        outcome = "pass"
    elif gate_statuses & {"fail", "fail_closed"}:
        outcome = "fail_closed"
    else:
        outcome = "pending"
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "source_identities": {
            "review_plan_sha256": REVIEW_PLAN_SHA256,
            "audit_pack_sha256": AUDIT_PACK_SHA256,
            "double_review_selection_identity_sha256": selection_identity,
            "rubric_sha256": predeclaration["rubric_sha256"],
        },
        "coverage": {
            "queries": len(primary),
            "gold_items": len(gold),
            "candidate_items": len(candidates),
            "tracks": _counter((row["track"] for row in primary), TRACKS),
        },
        "gold": {
            "support": _counter((item["support"] for item in gold), SUPPORT_LABELS),
            "ambiguity": _counter((item["ambiguity"] for item in gold), AMBIGUITY_LABELS),
            "confidence": _counter((item["confidence"] for item in gold), CONFIDENCE_LABELS),
            "decision_status": _counter((item["decision_status"] for item in gold), DECISION_STATUSES),
        },
        "candidates": {
            "relevance": _counter((item["relevance"] for item in candidates), RELEVANCE_LABELS),
            "likely_hard_negative": _counter((item["likely_hard_negative"] for item in candidates), TERNARY_LABELS),
            "suspected_missing_positive": _counter(
                (item["suspected_missing_positive"] for item in candidates), MISSING_LABELS
            ),
            "sensitive_information_risk": _counter(
                (item["sensitive_information_risk"] for item in candidates), RISK_LABELS
            ),
            "confidence": _counter((item["confidence"] for item in candidates), CONFIDENCE_LABELS),
            "decision_status": _counter((item["decision_status"] for item in candidates), DECISION_STATUSES),
        },
        "queries": {
            "credible_missing_positive": _counter(
                (item["credible_missing_positive"] for item in queries), TERNARY_LABELS
            ),
            "likely_hard_negative_present": _counter(
                (item["likely_hard_negative_present"] for item in queries), TERNARY_LABELS
            ),
            "qrels_completeness": _counter(
                (item["qrels_completeness"] for item in queries), {"complete_enough", "incomplete", "uncertain"}
            ),
            "answerability": _counter(
                (item["answerability"] for item in queries), {"answerable", "no_answer", "uncertain"}
            ),
            "label_incompleteness_risk": _counter((item["label_incompleteness_risk"] for item in queries), RISK_LABELS),
            "sensitive_information_risk": _counter(
                (item["sensitive_information_risk"] for item in queries), RISK_LABELS
            ),
        },
        "agreement": agreement,
        "gates": gates,
        "outcome": outcome,
        "official_qrels_modified": False,
        "judgment_standard": "agent_label_audit_not_human_gold",
        "publication": {"classification": "research_only", "leaderboard": "closed", "export": "closed"},
        "restricted_text_identifiers_rankings_paths_or_reviewer_identity_included": False,
    }
    _validate_schema(aggregate, "bright-label-audit-safe-aggregate-v01.schema.json", "safe aggregate")
    return aggregate


def bind_completed_predeclaration(**kwargs: Any) -> dict[str, Any]:
    """Bind the safe predeclaration to the completed restricted evidence graph."""
    from mm_embed.benchmark.bright_label_audit_completion import bind_completed_predeclaration as bind

    return bind(**kwargs)


def recompute_safe_aggregate(**kwargs: Any) -> dict[str, Any]:
    """Recompute the final safe aggregate from every bound restricted artifact."""
    from mm_embed.benchmark.bright_label_audit_completion import recompute_bound_safe_aggregate

    return recompute_bound_safe_aggregate(**kwargs)


def validate_completed_audit(**kwargs: Any) -> dict[str, Any]:
    """Validate the completed evidence graph without writing an aggregate."""
    from mm_embed.benchmark.bright_label_audit_completion import validate_completed_audit as validate

    return validate(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--review-plan", required=True)
    verify.add_argument("--audit-pack", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--review-plan", required=True)
    freeze.add_argument("--audit-pack", required=True)
    freeze.add_argument("--rubric", required=True)
    freeze.add_argument("--restricted-output", required=True)
    freeze.add_argument("--predeclaration-output", required=True)
    validate = subparsers.add_parser("validate-annotations")
    validate.add_argument("--annotations", required=True)
    validate.add_argument("--input", required=True)
    validate.add_argument("--role", choices=("primary", "validator"), required=True)
    supplement = subparsers.add_parser("prepare-validator-supplement")
    supplement.add_argument("--primary-annotations", required=True)
    supplement.add_argument("--primary-input", required=True)
    supplement.add_argument("--output", required=True)
    seal = subparsers.add_parser("seal-validator")
    seal.add_argument("--annotations", required=True)
    seal.add_argument("--input", required=True)
    seal.add_argument("--access-manifest", required=True)
    seal.add_argument("--output", required=True)
    bind = subparsers.add_parser("bind-completed-predeclaration")
    validate_completion = subparsers.add_parser("validate-completion")
    recompute = subparsers.add_parser("recompute")
    completion_arguments = (
        ("--predeclaration", "predeclaration_path"),
        ("--review-plan", "review_plan_path"),
        ("--audit-pack", "audit_pack_path"),
        ("--rubric", "rubric_path"),
        ("--primary-input", "primary_input_path"),
        ("--primary-annotations", "primary_annotation_path"),
        ("--validator-core-input", "validator_core_input_path"),
        ("--validator-core-annotations", "validator_core_annotation_path"),
        ("--validator-supplement-input", "validator_supplement_input_path"),
        ("--validator-supplement-annotations", "validator_supplement_annotation_path"),
        ("--selection", "selection_path"),
        ("--frozen-manifest", "frozen_manifest_path"),
        ("--validator-access-manifest", "validator_access_manifest_path"),
        ("--validator-seal", "validator_seal_path"),
        ("--adjudications", "adjudication_path"),
    )
    for subparser in (bind, validate_completion, recompute):
        for flag, destination in completion_arguments:
            subparser.add_argument(flag, dest=destination, required=True)
    recompute.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify-inputs":
        plan, pack = verify_frozen_inputs(args.review_plan, args.audit_pack)
        result = {
            "review_plan_rows": len(plan),
            "audit_pack_rows": len(pack),
            "gold_items": sum(len(row["gold_documents"]) for row in pack),
            "candidate_items": sum(len(row["baseline_top10_union"]) for row in pack),
        }
    elif args.command == "freeze":
        result = freeze_audit(
            args.review_plan,
            args.audit_pack,
            args.rubric,
            args.restricted_output,
            args.predeclaration_output,
        )
    elif args.command == "validate-annotations":
        rows = validate_annotations(args.annotations, args.input, expected_role=args.role)
        result = {"rows": len(rows), "valid": True}
    elif args.command == "prepare-validator-supplement":
        result = prepare_validator_supplement(args.primary_annotations, args.primary_input, args.output)
    elif args.command == "seal-validator":
        result = seal_validator_annotations(args.annotations, args.input, args.access_manifest, args.output)
    elif args.command == "bind-completed-predeclaration":
        result = bind_completed_predeclaration(
            **{destination: getattr(args, destination) for _, destination in completion_arguments}
        )
    elif args.command == "validate-completion":
        result = validate_completed_audit(
            **{destination: getattr(args, destination) for _, destination in completion_arguments}
        )
    else:
        result = recompute_safe_aggregate(
            **{destination: getattr(args, destination) for _, destination in completion_arguments}
        )
        output_path = Path(args.output)
        entry = _write_json(output_path, result)
        output_path.with_suffix(output_path.suffix + ".sha256").write_text(
            entry["sha256"] + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
