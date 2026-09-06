"""Fail-closed completion graph for the restricted BRIGHT label-quality audit."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from mm_embed.benchmark import bright_label_audit as contract

GOLD_CONFLICT_FIELDS = ("support", "ambiguity", "confidence", "decision_status")
CANDIDATE_CONFLICT_FIELDS = (
    "relevance",
    "likely_hard_negative",
    "suspected_missing_positive",
    "sensitive_information_risk",
    "sensitive_information_types",
    "confidence",
    "decision_status",
)
QUERY_CONFLICT_FIELDS = (
    "credible_missing_positive",
    "credible_missing_positive_ids",
    "likely_hard_negative_present",
    "likely_hard_negative_ids",
    "qrels_completeness",
    "answerability",
    "label_incompleteness_risk",
    "sensitive_information_risk",
    "confidence",
    "decision_status",
)
CONFLICT_FIELDS = {
    "gold": GOLD_CONFLICT_FIELDS,
    "candidate": CANDIDATE_CONFLICT_FIELDS,
    "query": QUERY_CONFLICT_FIELDS,
}


def _read_object(path: Path, subject: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise contract.BrightLabelAuditError(f"missing or invalid {subject}") from error
    if not isinstance(value, dict):
        raise contract.BrightLabelAuditError(f"{subject} must be a JSON object")
    return value


def _validate_blind_rows(rows: tuple[dict[str, Any], ...], subject: str) -> None:
    keys = []
    for row in rows:
        contract._validate_schema(row, "bright-label-audit-blind-input-v01.schema.json", subject)
        keys.append((row["track"], row["query_id"]))
    if len(keys) != len(set(keys)):
        raise contract.BrightLabelAuditError(f"{subject} contains duplicate query identities")


def _artifact_entry(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": contract._sha256_file(path)}


def _assert_manifest_entry(
    actual: dict[str, Any],
    path: Path,
    rows: int,
    subject: str,
    *,
    expected_name: str | None,
) -> None:
    expected = _artifact_entry(path, rows)
    for field, value in expected.items():
        if actual.get(field) != value:
            raise contract.BrightLabelAuditError(f"{subject} {field} identity mismatch")
    if expected_name is not None and actual.get("path") != expected_name:
        raise contract.BrightLabelAuditError(f"{subject} logical filename mismatch")
    if expected_name is None and "path" in actual:
        raise contract.BrightLabelAuditError(f"{subject} contains an unexpected logical filename")


def _expected_selection(plan: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "track": row["track"],
            "query_id": row["query_id"],
            "query_length_stratum": row["strata"]["query_length"],
            "source_plan_row_sha256": contract._sha256_json(row),
        }
        for row in contract.select_double_review(plan)
    )


def _derive_supplement(
    primary: tuple[dict[str, Any], ...],
    primary_input: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    source_by_key = {(row["track"], row["query_id"]): row for row in primary_input}
    selected = set()
    for row in primary:
        judgments = [*row["gold_reviews"], *row["candidate_reviews"], row["query_review"]]
        if any(item["confidence"] == "low" or item["decision_status"] != "decided" for item in judgments):
            selected.add((row["track"], row["query_id"]))
    return tuple(source_by_key[key] for key in sorted(selected))


def _validate_frozen_graph(
    *,
    review_plan_path: Path,
    audit_pack_path: Path,
    rubric_path: Path,
    primary_input_path: Path,
    validator_core_input_path: Path,
    selection_path: Path,
    frozen_manifest_path: Path,
    validator_access_manifest_path: Path,
) -> dict[str, Any]:
    plan, pack = contract.verify_frozen_inputs(review_plan_path, audit_pack_path)
    rubric_sha256 = contract._sha256_file(rubric_path)

    primary_input = contract._read_jsonl(primary_input_path)
    _validate_blind_rows(primary_input, "primary blind input row")
    expected_primary = tuple(contract._blind_row(row) for row in pack)
    if primary_input != expected_primary:
        raise contract.BrightLabelAuditError("primary input does not exactly derive from the frozen audit pack")

    selection = contract._read_jsonl(selection_path)
    for row in selection:
        contract._validate_schema(row, "bright-label-audit-selection-v01.schema.json", "selection row")
    expected_selection = _expected_selection(plan)
    if selection != expected_selection:
        raise contract.BrightLabelAuditError("double-review selection does not exactly derive from the frozen plan")

    selected_keys = {(row["track"], row["query_id"]) for row in selection}
    validator_core_input = contract._read_jsonl(validator_core_input_path)
    _validate_blind_rows(validator_core_input, "Validator core blind input row")
    expected_core = tuple(row for row in expected_primary if (row["track"], row["query_id"]) in selected_keys)
    if validator_core_input != expected_core:
        raise contract.BrightLabelAuditError("Validator core input does not exactly match the frozen selection")

    frozen_manifest = _read_object(frozen_manifest_path, "frozen-input manifest")
    contract._validate_schema(
        frozen_manifest,
        "bright-label-audit-frozen-manifest-v01.schema.json",
        "frozen-input manifest",
    )
    if frozen_manifest["rubric_sha256"] != rubric_sha256:
        raise contract.BrightLabelAuditError("frozen manifest rubric identity mismatch")
    _assert_manifest_entry(
        frozen_manifest["primary_input"],
        primary_input_path,
        contract.REVIEW_PLAN_ROWS,
        "primary input",
        expected_name="input.jsonl",
    )
    _assert_manifest_entry(
        frozen_manifest["validator_core_input"],
        validator_core_input_path,
        contract.DOUBLE_REVIEW_QUERIES,
        "Validator core input",
        expected_name="core-input.jsonl",
    )
    _assert_manifest_entry(
        frozen_manifest["double_review"]["selection_artifact"],
        selection_path,
        contract.DOUBLE_REVIEW_QUERIES,
        "double-review selection",
        expected_name="double-review-selection.jsonl",
    )
    selection_identity = contract._identity_rows(selection)
    if frozen_manifest["double_review"]["selection_identity_sha256"] != selection_identity:
        raise contract.BrightLabelAuditError("frozen manifest selection identity mismatch")

    access = _read_object(validator_access_manifest_path, "Validator access manifest")
    contract._validate_schema(
        access,
        "bright-label-audit-validator-access-v01.schema.json",
        "Validator access manifest",
    )
    if access["core_input_sha256"] != contract._sha256_file(validator_core_input_path):
        raise contract.BrightLabelAuditError("Validator access core-input identity mismatch")
    if access["rubric_sha256"] != rubric_sha256:
        raise contract.BrightLabelAuditError("Validator access rubric identity mismatch")
    annotation_schema_sha256 = contract._sha256_file(
        contract._schema_path("bright-label-audit-annotation-v01.schema.json")
    )
    if access["annotation_schema_sha256"] != annotation_schema_sha256:
        raise contract.BrightLabelAuditError("Validator access annotation-schema identity mismatch")
    _assert_manifest_entry(
        frozen_manifest["validator_access_manifest"],
        validator_access_manifest_path,
        1,
        "Validator access manifest",
        expected_name=None,
    )

    return {
        "plan": plan,
        "pack": pack,
        "primary_input": primary_input,
        "validator_core_input": validator_core_input,
        "selection": selection,
        "selected_keys": selected_keys,
        "selection_identity": selection_identity,
        "rubric_sha256": rubric_sha256,
        "frozen_manifest": frozen_manifest,
        "access": access,
    }


def _unit_map(row: dict[str, Any]) -> dict[tuple[str, str | None], dict[str, Any]]:
    result = {("gold", item["document_id"]): item for item in row["gold_reviews"]}
    result.update({("candidate", item["document_id"]): item for item in row["candidate_reviews"]})
    result[("query", None)] = row["query_review"]
    return result


def _unit_key(
    track: str,
    query_id: str,
    unit_type: str,
    document_id: str | None,
) -> tuple[str, str, str, str | None]:
    return track, query_id, unit_type, document_id


def _unresolved(rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> int:
    return sum(
        item["decision_status"] != "decided"
        for row in rows
        for item in [*row["gold_reviews"], *row["candidate_reviews"], row["query_review"]]
    )


def _validate_effective_rows(rows: list[dict[str, Any]], primary_input: tuple[dict[str, Any], ...]) -> None:
    source_by_key = {(row["track"], row["query_id"]): row for row in primary_input}
    for row in rows:
        source = source_by_key[row["track"], row["query_id"]]
        gold_ids = {item["document_id"] for item in source["gold_documents"]}
        for item in row["gold_reviews"]:
            contract._validate_decision_status(item, "uncertain" in {item["support"], item["ambiguity"]})
        for item in row["candidate_reviews"]:
            uncertain = "uncertain" in {
                item["relevance"],
                item["likely_hard_negative"],
                item["suspected_missing_positive"],
                item["sensitive_information_risk"],
            }
            contract._validate_decision_status(item, uncertain)
            if item["suspected_missing_positive"] == "credible" and item["relevance"] != "relevant":
                raise contract.BrightLabelAuditError("effective credible missing positive must be relevant")
            if item["likely_hard_negative"] == "yes" and item["relevance"] != "not_relevant":
                raise contract.BrightLabelAuditError("effective likely hard negative must be not relevant")
            existing_positive = item["document_id"] in gold_ids
            if existing_positive != (item["suspected_missing_positive"] == "not_applicable"):
                raise contract.BrightLabelAuditError("effective missing-positive applicability mismatch")
            if item["sensitive_information_risk"] == "none" and item["sensitive_information_types"]:
                raise contract.BrightLabelAuditError("effective no-risk judgment has sensitive-information types")
            if (
                item["sensitive_information_risk"] in {"low", "moderate", "high"}
                and not item["sensitive_information_types"]
            ):
                raise contract.BrightLabelAuditError("effective risk judgment is missing a sensitive-information type")
        query = row["query_review"]
        query_uncertain = "uncertain" in {
            query["credible_missing_positive"],
            query["likely_hard_negative_present"],
            query["qrels_completeness"],
            query["answerability"],
            query["label_incompleteness_risk"],
            query["sensitive_information_risk"],
        }
        contract._validate_decision_status(query, query_uncertain)
        credible_ids = sorted(
            item["document_id"] for item in row["candidate_reviews"] if item["suspected_missing_positive"] == "credible"
        )
        hard_ids = sorted(
            item["document_id"] for item in row["candidate_reviews"] if item["likely_hard_negative"] == "yes"
        )
        if sorted(query["credible_missing_positive_ids"]) != credible_ids:
            raise contract.BrightLabelAuditError("effective query missing-positive IDs contradict item judgments")
        if sorted(query["likely_hard_negative_ids"]) != hard_ids:
            raise contract.BrightLabelAuditError("effective query hard-negative IDs contradict item judgments")
        expected_missing = "yes" if credible_ids else "no"
        expected_hard = "yes" if hard_ids else "no"
        if query["credible_missing_positive"] not in {expected_missing, "uncertain"}:
            raise contract.BrightLabelAuditError("effective query missing-positive summary contradicts items")
        if query["likely_hard_negative_present"] not in {expected_hard, "uncertain"}:
            raise contract.BrightLabelAuditError("effective query hard-negative summary contradicts items")
        if credible_ids and query["qrels_completeness"] == "complete_enough":
            raise contract.BrightLabelAuditError("effective missing positive contradicts complete-enough qrels")


def _validate_adjudications(
    *,
    primary: tuple[dict[str, Any], ...],
    primary_input: tuple[dict[str, Any], ...],
    primary_annotation_path: Path,
    validator_core: tuple[dict[str, Any], ...],
    validator_core_input: tuple[dict[str, Any], ...],
    validator_core_input_path: Path,
    validator_core_annotation_path: Path,
    validator_supplement: tuple[dict[str, Any], ...],
    validator_supplement_input: tuple[dict[str, Any], ...],
    validator_supplement_input_path: Path,
    validator_supplement_annotation_path: Path,
    validator_seal_path: Path,
    adjudication_path: Path,
    rubric_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], list[dict[str, Any]]]:
    primary_by_key = {(row["track"], row["query_id"]): row for row in primary}
    core_by_key = {(row["track"], row["query_id"]): row for row in validator_core}
    core_input_by_key = {(row["track"], row["query_id"]): row for row in validator_core_input}
    supplement_by_key = {(row["track"], row["query_id"]): row for row in validator_supplement}
    supplement_input_by_key = {(row["track"], row["query_id"]): row for row in validator_supplement_input}
    if set(core_by_key) & set(supplement_by_key):
        raise contract.BrightLabelAuditError("Validator core and supplement query sets overlap")

    expected: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
    for scope, validator_by_key, source_by_key in (
        ("core_conflict", core_by_key, core_input_by_key),
        ("supplement_low_confidence", supplement_by_key, supplement_input_by_key),
    ):
        for query_key, validator_row in validator_by_key.items():
            primary_row = primary_by_key[query_key]
            left_units = _unit_map(primary_row)
            right_units = _unit_map(validator_row)
            if set(left_units) != set(right_units):
                raise contract.BrightLabelAuditError("primary and Validator judgment unit coverage differs")
            for (unit_type, document_id), left in left_units.items():
                right = right_units[unit_type, document_id]
                conflicting_fields = [field for field in CONFLICT_FIELDS[unit_type] if left[field] != right[field]]
                trigger_reasons = []
                if conflicting_fields:
                    trigger_reasons.append("field_disagreement")
                if left["confidence"] == "low" or left["decision_status"] != "decided":
                    trigger_reasons.append("primary_low_confidence_or_unresolved")
                if not trigger_reasons:
                    continue
                key = _unit_key(query_key[0], query_key[1], unit_type, document_id)
                if key in expected:
                    raise contract.BrightLabelAuditError("duplicate adjudication trigger identity")
                expected[key] = {
                    "review_scope": scope,
                    "trigger_reasons": trigger_reasons,
                    "conflicting_fields": conflicting_fields,
                    "primary_judgment": left,
                    "validator_judgment": right,
                    "source_input": source_by_key[query_key],
                    "primary_row": primary_row,
                    "validator_row": validator_row,
                }

    decisions = contract._read_jsonl(adjudication_path)
    decision_by_key: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
    primary_sha256 = contract._sha256_file(primary_annotation_path)
    core_sha256 = contract._sha256_file(validator_core_annotation_path)
    supplement_sha256 = contract._sha256_file(validator_supplement_annotation_path)
    seal_sha256 = contract._sha256_file(validator_seal_path)
    prompt_recipes = set()
    output_schemas = set()
    sampling_identities = set()
    query_request_identities: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in decisions:
        contract._validate_schema(row, "bright-label-audit-adjudication-v01.schema.json", "adjudication row")
        key = _unit_key(row["track"], row["query_id"], row["unit_type"], row["document_id"])
        if key in decision_by_key:
            raise contract.BrightLabelAuditError("duplicate adjudication decision identity")
        decision_by_key[key] = row
        expected_row = expected.get(key)
        if expected_row is None:
            raise contract.BrightLabelAuditError("adjudication decision has no deterministic trigger")
        for field in (
            "review_scope",
            "trigger_reasons",
            "conflicting_fields",
            "primary_judgment",
            "validator_judgment",
        ):
            if row[field] != expected_row[field]:
                raise contract.BrightLabelAuditError(f"adjudication {field} does not match its trigger")
        adjudicated = row["adjudicated_judgment"]
        if row["unit_type"] != "query" and adjudicated["document_id"] != row["document_id"]:
            raise contract.BrightLabelAuditError("adjudicated document identity mismatch")
        provenance = row["adjudication_provenance"]
        if adjudicated["provenance"] != provenance:
            raise contract.BrightLabelAuditError("adjudicated judgment does not preserve its provenance")
        validator_annotation_sha256 = core_sha256 if row["review_scope"] == "core_conflict" else supplement_sha256
        expected_provenance = {
            "rubric_sha256": rubric_sha256,
            "review_scope": row["review_scope"],
            "source_input_artifact_sha256": (
                contract._sha256_file(validator_core_input_path)
                if row["review_scope"] == "core_conflict"
                else contract._sha256_file(validator_supplement_input_path)
            ),
            "source_input_row_sha256": contract._sha256_json(expected_row["source_input"]),
            "primary_annotations_sha256": primary_sha256,
            "primary_annotation_row_sha256": contract._sha256_json(expected_row["primary_row"]),
            "validator_annotations_sha256": validator_annotation_sha256,
            "validator_annotation_row_sha256": contract._sha256_json(expected_row["validator_row"]),
            "validator_core_seal_sha256": seal_sha256,
        }
        for field, value in expected_provenance.items():
            if provenance[field] != value:
                raise contract.BrightLabelAuditError(f"adjudication provenance {field} mismatch")
        if provenance["sampling_sha256"] != contract._sha256_json(provenance["sampling"]):
            raise contract.BrightLabelAuditError("adjudication sampling identity mismatch")
        prompt_recipes.add(provenance["prompt_recipe_sha256"])
        output_schemas.add(provenance["output_schema_sha256"])
        sampling_identities.add(provenance["sampling_sha256"])
        query_key = (row["review_scope"], row["track"], row["query_id"])
        request_identity = (provenance["input_bundle_sha256"], provenance["request_prompt_sha256"])
        if query_key in query_request_identities and query_request_identities[query_key] != request_identity:
            raise contract.BrightLabelAuditError("adjudication query request identity is inconsistent")
        query_request_identities[query_key] = request_identity
    if set(decision_by_key) != set(expected):
        raise contract.BrightLabelAuditError("adjudication decisions do not exactly cover the trigger set")
    if len(prompt_recipes) != 1 or len(output_schemas) != 1 or len(sampling_identities) != 1:
        raise contract.BrightLabelAuditError("adjudication execution identities are inconsistent")
    effective_by_key = {key: copy.deepcopy(row) for key, row in primary_by_key.items()}
    for key in sorted(decision_by_key):
        row = decision_by_key[key]
        effective = effective_by_key[row["track"], row["query_id"]]
        if row["unit_type"] == "query":
            effective["query_review"] = copy.deepcopy(row["adjudicated_judgment"])
            continue
        collection_name = "gold_reviews" if row["unit_type"] == "gold" else "candidate_reviews"
        index = next(
            index for index, item in enumerate(effective[collection_name]) if item["document_id"] == row["document_id"]
        )
        effective[collection_name][index] = copy.deepcopy(row["adjudicated_judgment"])
    effective_rows = [effective_by_key[key] for key in sorted(effective_by_key)]
    _validate_effective_rows(effective_rows, primary_input)
    return decisions, effective_rows


def _validate_seal(
    seal_path: Path,
    core_input_path: Path,
    core_annotation_path: Path,
    access_manifest_path: Path,
    rubric_sha256: str,
) -> dict[str, Any]:
    seal = _read_object(seal_path, "Validator core seal")
    contract._validate_schema(seal, "bright-label-audit-validator-seal-v01.schema.json", "Validator core seal")
    expected = {
        "validator_input_sha256": contract._sha256_file(core_input_path),
        "validator_annotations_sha256": contract._sha256_file(core_annotation_path),
        "validator_annotation_rows": contract.DOUBLE_REVIEW_QUERIES,
        "validator_annotation_bytes": core_annotation_path.stat().st_size,
        "access_manifest_sha256": contract._sha256_file(access_manifest_path),
        "rubric_sha256": rubric_sha256,
    }
    if any(seal[field] != value for field, value in expected.items()):
        raise contract.BrightLabelAuditError("Validator core seal binding mismatch")
    return seal


def _validate_predeclaration(
    predeclaration_path: Path,
    paths: dict[str, Path],
    graph: dict[str, Any],
) -> dict[str, Any]:
    predeclaration = _read_object(predeclaration_path, "predeclaration")
    contract._validate_schema(
        predeclaration,
        "bright-label-audit-predeclaration-v01.schema.json",
        "predeclaration",
    )
    if predeclaration["phase"] != "completed_evidence_bound":
        raise contract.BrightLabelAuditError("aggregate requires a completed evidence-bound predeclaration")
    if predeclaration["rubric_sha256"] != graph["rubric_sha256"]:
        raise contract.BrightLabelAuditError("predeclaration rubric identity mismatch")
    for name in contract.SCHEMA_NAMES:
        if predeclaration["schema_identities"][name] != contract._sha256_file(contract._schema_path(name)):
            raise contract.BrightLabelAuditError("current schema does not match the frozen predeclaration")
    frozen_expected = {
        "frozen_input_manifest_sha256": contract._sha256_file(paths["frozen_manifest"]),
        "primary_input_sha256": contract._sha256_file(paths["primary_input"]),
        "validator_core_input_sha256": contract._sha256_file(paths["validator_core_input"]),
        "validator_access_manifest_sha256": contract._sha256_file(paths["validator_access_manifest"]),
        "double_review_selection_sha256": contract._sha256_file(paths["selection"]),
    }
    completion_expected = {
        "primary_annotations_sha256": contract._sha256_file(paths["primary_annotations"]),
        "validator_core_annotations_sha256": contract._sha256_file(paths["validator_core_annotations"]),
        "validator_supplement_input_sha256": contract._sha256_file(paths["validator_supplement_input"]),
        "validator_supplement_annotations_sha256": contract._sha256_file(paths["validator_supplement_annotations"]),
        "validator_core_seal_sha256": contract._sha256_file(paths["validator_seal"]),
        "adjudication_decisions_sha256": contract._sha256_file(paths["adjudications"]),
    }
    if predeclaration["frozen_artifact_identities"] != frozen_expected:
        raise contract.BrightLabelAuditError("predeclaration frozen-artifact identity mismatch")
    if predeclaration["completion_artifact_identities"] != completion_expected:
        raise contract.BrightLabelAuditError("predeclaration completion-artifact identity mismatch")
    if predeclaration["double_review"]["selection_identity_sha256"] != graph["selection_identity"]:
        raise contract.BrightLabelAuditError("predeclaration selection identity mismatch")
    return predeclaration


def _paths(**values: str | Path) -> dict[str, Path]:
    return {name: Path(value) for name, value in values.items()}


def _validate_completion_evidence(paths: dict[str, Path]) -> dict[str, Any]:
    graph = _validate_frozen_graph(
        review_plan_path=paths["review_plan"],
        audit_pack_path=paths["audit_pack"],
        rubric_path=paths["rubric"],
        primary_input_path=paths["primary_input"],
        validator_core_input_path=paths["validator_core_input"],
        selection_path=paths["selection"],
        frozen_manifest_path=paths["frozen_manifest"],
        validator_access_manifest_path=paths["validator_access_manifest"],
    )
    primary = contract.validate_annotations(
        paths["primary_annotations"], paths["primary_input"], expected_role="primary"
    )
    validator_core = contract.validate_annotations(
        paths["validator_core_annotations"],
        paths["validator_core_input"],
        expected_role="validator",
    )
    core_by_key = {(row["track"], row["query_id"]) for row in validator_core}
    if core_by_key != graph["selected_keys"]:
        raise contract.BrightLabelAuditError("Validator core annotations do not match the frozen selection")

    supplement_input = contract._read_jsonl(paths["validator_supplement_input"])
    _validate_blind_rows(supplement_input, "Validator supplement blind input row")
    if supplement_input != _derive_supplement(primary, graph["primary_input"]):
        raise contract.BrightLabelAuditError("Validator supplement does not exactly cover primary low-confidence units")
    validator_supplement = contract.validate_annotations(
        paths["validator_supplement_annotations"],
        paths["validator_supplement_input"],
        expected_role="validator",
    )
    primary_sessions = {
        item["provenance"]["session_id"]
        for row in primary
        for item in [*row["gold_reviews"], *row["candidate_reviews"], row["query_review"]]
    }
    validator_sessions = {
        item["provenance"]["session_id"]
        for row in [*validator_core, *validator_supplement]
        for item in [*row["gold_reviews"], *row["candidate_reviews"], row["query_review"]]
    }
    if primary_sessions & validator_sessions:
        raise contract.BrightLabelAuditError("primary and Validator sessions are not independent")
    _validate_seal(
        paths["validator_seal"],
        paths["validator_core_input"],
        paths["validator_core_annotations"],
        paths["validator_access_manifest"],
        graph["rubric_sha256"],
    )
    decisions, effective = _validate_adjudications(
        primary=primary,
        primary_input=graph["primary_input"],
        primary_annotation_path=paths["primary_annotations"],
        validator_core=validator_core,
        validator_core_input=graph["validator_core_input"],
        validator_core_input_path=paths["validator_core_input"],
        validator_core_annotation_path=paths["validator_core_annotations"],
        validator_supplement=validator_supplement,
        validator_supplement_input=supplement_input,
        validator_supplement_input_path=paths["validator_supplement_input"],
        validator_supplement_annotation_path=paths["validator_supplement_annotations"],
        validator_seal_path=paths["validator_seal"],
        adjudication_path=paths["adjudications"],
        rubric_sha256=graph["rubric_sha256"],
    )
    return {
        **graph,
        "primary": primary,
        "validator_core": validator_core,
        "supplement_input": supplement_input,
        "validator_supplement": validator_supplement,
        "decisions": decisions,
        "effective": effective,
    }


def bind_completed_predeclaration(
    *,
    predeclaration_path: str | Path,
    review_plan_path: str | Path,
    audit_pack_path: str | Path,
    rubric_path: str | Path,
    primary_input_path: str | Path,
    primary_annotation_path: str | Path,
    validator_core_input_path: str | Path,
    validator_core_annotation_path: str | Path,
    validator_supplement_input_path: str | Path,
    validator_supplement_annotation_path: str | Path,
    selection_path: str | Path,
    frozen_manifest_path: str | Path,
    validator_access_manifest_path: str | Path,
    validator_seal_path: str | Path,
    adjudication_path: str | Path,
) -> dict[str, Any]:
    """Bind a safe predeclaration to the immutable completed restricted evidence."""
    paths = _paths(
        review_plan=review_plan_path,
        audit_pack=audit_pack_path,
        rubric=rubric_path,
        primary_input=primary_input_path,
        primary_annotations=primary_annotation_path,
        validator_core_input=validator_core_input_path,
        validator_core_annotations=validator_core_annotation_path,
        validator_supplement_input=validator_supplement_input_path,
        validator_supplement_annotations=validator_supplement_annotation_path,
        selection=selection_path,
        frozen_manifest=frozen_manifest_path,
        validator_access_manifest=validator_access_manifest_path,
        validator_seal=validator_seal_path,
        adjudications=adjudication_path,
    )
    evidence = _validate_completion_evidence(paths)
    predeclaration_file = Path(predeclaration_path)
    value = _read_object(predeclaration_file, "predeclaration")
    value["phase"] = "completed_evidence_bound"
    value["frozen_artifact_identities"] = {
        "frozen_input_manifest_sha256": contract._sha256_file(paths["frozen_manifest"]),
        "primary_input_sha256": contract._sha256_file(paths["primary_input"]),
        "validator_core_input_sha256": contract._sha256_file(paths["validator_core_input"]),
        "validator_access_manifest_sha256": contract._sha256_file(paths["validator_access_manifest"]),
        "double_review_selection_sha256": contract._sha256_file(paths["selection"]),
    }
    value["completion_artifact_identities"] = {
        "primary_annotations_sha256": contract._sha256_file(paths["primary_annotations"]),
        "validator_core_annotations_sha256": contract._sha256_file(paths["validator_core_annotations"]),
        "validator_supplement_input_sha256": contract._sha256_file(paths["validator_supplement_input"]),
        "validator_supplement_annotations_sha256": contract._sha256_file(paths["validator_supplement_annotations"]),
        "validator_core_seal_sha256": contract._sha256_file(paths["validator_seal"]),
        "adjudication_decisions_sha256": contract._sha256_file(paths["adjudications"]),
    }
    value["schema_identities"] = {
        name: contract._sha256_file(contract._schema_path(name)) for name in contract.SCHEMA_NAMES
    }
    if value["rubric_sha256"] != evidence["rubric_sha256"]:
        raise contract.BrightLabelAuditError("predeclaration rubric differs from completed evidence")
    if value["double_review"]["selection_identity_sha256"] != evidence["selection_identity"]:
        raise contract.BrightLabelAuditError("predeclaration selection differs from completed evidence")
    contract._validate_schema(
        value,
        "bright-label-audit-predeclaration-v01.schema.json",
        "completed predeclaration",
    )
    entry = contract._write_json(predeclaration_file, value)
    predeclaration_file.with_suffix(predeclaration_file.suffix + ".sha256").write_text(
        entry["sha256"] + "\n",
        encoding="utf-8",
    )
    return entry


def recompute_bound_safe_aggregate(
    *,
    predeclaration_path: str | Path,
    review_plan_path: str | Path,
    audit_pack_path: str | Path,
    rubric_path: str | Path,
    primary_input_path: str | Path,
    primary_annotation_path: str | Path,
    validator_core_input_path: str | Path,
    validator_core_annotation_path: str | Path,
    validator_supplement_input_path: str | Path,
    validator_supplement_annotation_path: str | Path,
    selection_path: str | Path,
    frozen_manifest_path: str | Path,
    validator_access_manifest_path: str | Path,
    validator_seal_path: str | Path,
    adjudication_path: str | Path,
) -> dict[str, Any]:
    """Validate the entire evidence graph and recompute adjudicated safe counts."""
    paths = _paths(
        review_plan=review_plan_path,
        audit_pack=audit_pack_path,
        rubric=rubric_path,
        primary_input=primary_input_path,
        primary_annotations=primary_annotation_path,
        validator_core_input=validator_core_input_path,
        validator_core_annotations=validator_core_annotation_path,
        validator_supplement_input=validator_supplement_input_path,
        validator_supplement_annotations=validator_supplement_annotation_path,
        selection=selection_path,
        frozen_manifest=frozen_manifest_path,
        validator_access_manifest=validator_access_manifest_path,
        validator_seal=validator_seal_path,
        adjudications=adjudication_path,
    )
    evidence = _validate_completion_evidence(paths)
    predeclaration = _validate_predeclaration(Path(predeclaration_path), paths, evidence)
    effective = evidence["effective"]
    gold = [item for row in effective for item in row["gold_reviews"]]
    candidates = [item for row in effective for item in row["candidate_reviews"]]
    queries = [row["query_review"] for row in effective]

    primary_by_key = {(row["track"], row["query_id"]): row for row in evidence["primary"]}
    validator_by_key = {(row["track"], row["query_id"]): row for row in evidence["validator_core"]}
    primary_labels = []
    validator_labels = []
    excluded_pairs = 0
    for key in sorted(evidence["selected_keys"]):
        left = {item["document_id"]: item["relevance"] for item in primary_by_key[key]["candidate_reviews"]}
        right = {item["document_id"]: item["relevance"] for item in validator_by_key[key]["candidate_reviews"]}
        if set(left) != set(right):
            raise contract.BrightLabelAuditError("paired candidate identities do not match")
        for document_id in sorted(left):
            if (
                left[document_id] in contract.BINARY_RELEVANCE_LABELS
                and right[document_id] in contract.BINARY_RELEVANCE_LABELS
            ):
                primary_labels.append(left[document_id])
                validator_labels.append(right[document_id])
            else:
                excluded_pairs += 1
    computed = contract.cohen_kappa(primary_labels, validator_labels) if primary_labels else None
    agreement_fail_closed = excluded_pairs > 0 or computed is None or computed["kappa"] is None
    agreement = {
        "status": "fail_closed" if agreement_fail_closed else "complete",
        "double_review_queries": len(evidence["selected_keys"]),
        "binary_pairs": len(primary_labels),
        "excluded_unresolved_pairs": excluded_pairs,
        "observed_agreement": None if computed is None else computed["observed_agreement"],
        "expected_agreement": None if computed is None else computed["expected_agreement"],
        "kappa": None if computed is None else computed["kappa"],
    }

    unresolved_breakdown = {
        "primary": _unresolved(evidence["primary"]),
        "validator_core": _unresolved(evidence["validator_core"]),
        "validator_supplement": _unresolved(evidence["validator_supplement"]),
        "adjudicated": sum(
            row["adjudicated_judgment"]["decision_status"] != "decided" for row in evidence["decisions"]
        ),
        "effective_final": _unresolved(effective),
    }
    unresolved_history = sum(
        unresolved_breakdown[name] for name in ("primary", "validator_core", "validator_supplement", "adjudicated")
    )
    unsupported = sum(item["support"] != "supports" or item["ambiguity"] != "unambiguous" for item in gold)
    missing_queries = sum(item["credible_missing_positive"] == "yes" for item in queries)
    unsupported_fraction = unsupported / len(gold)
    missing_fraction = missing_queries / len(queries)
    kappa = agreement["kappa"]
    gates = {
        "unsupported_or_ambiguous_gold": {
            "threshold": contract.UNSUPPORTED_GOLD_THRESHOLD,
            "comparison": "exceeds",
            "numerator": unsupported,
            "denominator": len(gold),
            "value": unsupported_fraction,
            "status": "fail" if unsupported_fraction > contract.UNSUPPORTED_GOLD_THRESHOLD else "pass",
        },
        "credible_missing_positive_queries": {
            "threshold": contract.MISSING_POSITIVE_QUERY_THRESHOLD,
            "comparison": "exceeds",
            "numerator": missing_queries,
            "denominator": len(queries),
            "value": missing_fraction,
            "status": "fail" if missing_fraction > contract.MISSING_POSITIVE_QUERY_THRESHOLD else "pass",
        },
        "binary_candidate_relevance_kappa": {
            "threshold": contract.MINIMUM_KAPPA,
            "comparison": "below",
            "value": kappa,
            "status": (
                "fail_closed"
                if agreement["status"] == "fail_closed"
                else "fail"
                if kappa < contract.MINIMUM_KAPPA
                else "pass"
            ),
        },
        "unresolved_judgments": {
            "threshold": 0,
            "comparison": "exceeds",
            "value": unresolved_history,
            "breakdown": unresolved_breakdown,
            "status": "fail_closed" if unresolved_history else "pass",
        },
    }
    gate_statuses = {gate["status"] for gate in gates.values()}
    outcome = "pass" if gate_statuses == {"pass"} else "fail_closed"
    aggregate = {
        "schema_version": contract.SCHEMA_VERSION,
        "audit_id": contract.AUDIT_ID,
        "source_identities": {
            "review_plan_sha256": contract.REVIEW_PLAN_SHA256,
            "audit_pack_sha256": contract.AUDIT_PACK_SHA256,
            "rubric_sha256": evidence["rubric_sha256"],
            "predeclaration_sha256": contract._sha256_file(Path(predeclaration_path)),
            "frozen_input_manifest_sha256": contract._sha256_file(paths["frozen_manifest"]),
            "primary_input_sha256": contract._sha256_file(paths["primary_input"]),
            "primary_annotations_sha256": contract._sha256_file(paths["primary_annotations"]),
            "validator_core_input_sha256": contract._sha256_file(paths["validator_core_input"]),
            "validator_core_annotations_sha256": contract._sha256_file(paths["validator_core_annotations"]),
            "validator_core_seal_sha256": contract._sha256_file(paths["validator_seal"]),
            "validator_supplement_input_sha256": contract._sha256_file(paths["validator_supplement_input"]),
            "validator_supplement_annotations_sha256": contract._sha256_file(paths["validator_supplement_annotations"]),
            "double_review_selection_sha256": contract._sha256_file(paths["selection"]),
            "double_review_selection_identity_sha256": evidence["selection_identity"],
            "adjudication_decisions_sha256": contract._sha256_file(paths["adjudications"]),
        },
        "coverage": {
            "queries": len(effective),
            "gold_items": len(gold),
            "candidate_items": len(candidates),
            "tracks": contract._counter((row["track"] for row in effective), contract.TRACKS),
        },
        "gold": {
            "support": contract._counter((item["support"] for item in gold), contract.SUPPORT_LABELS),
            "ambiguity": contract._counter((item["ambiguity"] for item in gold), contract.AMBIGUITY_LABELS),
            "confidence": contract._counter((item["confidence"] for item in gold), contract.CONFIDENCE_LABELS),
            "decision_status": contract._counter(
                (item["decision_status"] for item in gold), contract.DECISION_STATUSES
            ),
        },
        "candidates": {
            "relevance": contract._counter((item["relevance"] for item in candidates), contract.RELEVANCE_LABELS),
            "likely_hard_negative": contract._counter(
                (item["likely_hard_negative"] for item in candidates), contract.TERNARY_LABELS
            ),
            "suspected_missing_positive": contract._counter(
                (item["suspected_missing_positive"] for item in candidates), contract.MISSING_LABELS
            ),
            "sensitive_information_risk": contract._counter(
                (item["sensitive_information_risk"] for item in candidates), contract.RISK_LABELS
            ),
            "confidence": contract._counter((item["confidence"] for item in candidates), contract.CONFIDENCE_LABELS),
            "decision_status": contract._counter(
                (item["decision_status"] for item in candidates), contract.DECISION_STATUSES
            ),
        },
        "queries": {
            "credible_missing_positive": contract._counter(
                (item["credible_missing_positive"] for item in queries), contract.TERNARY_LABELS
            ),
            "likely_hard_negative_present": contract._counter(
                (item["likely_hard_negative_present"] for item in queries), contract.TERNARY_LABELS
            ),
            "qrels_completeness": contract._counter(
                (item["qrels_completeness"] for item in queries), {"complete_enough", "incomplete", "uncertain"}
            ),
            "answerability": contract._counter(
                (item["answerability"] for item in queries), {"answerable", "no_answer", "uncertain"}
            ),
            "label_incompleteness_risk": contract._counter(
                (item["label_incompleteness_risk"] for item in queries), contract.RISK_LABELS
            ),
            "sensitive_information_risk": contract._counter(
                (item["sensitive_information_risk"] for item in queries), contract.RISK_LABELS
            ),
            "confidence": contract._counter((item["confidence"] for item in queries), contract.CONFIDENCE_LABELS),
            "decision_status": contract._counter(
                (item["decision_status"] for item in queries), contract.DECISION_STATUSES
            ),
        },
        "adjudication": {
            "decisions": len(evidence["decisions"]),
            "review_scope": contract._counter(
                (row["review_scope"] for row in evidence["decisions"]),
                {"core_conflict", "supplement_low_confidence"},
            ),
            "unit_type": contract._counter(
                (row["unit_type"] for row in evidence["decisions"]),
                {"gold", "candidate", "query"},
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
    if predeclaration["official_qrels_modified"] is not False:
        raise contract.BrightLabelAuditError("predeclaration qrels boundary is not closed")
    contract._validate_schema(aggregate, "bright-label-audit-safe-aggregate-v01.schema.json", "safe aggregate")
    return aggregate


def validate_completed_audit(**kwargs: Any) -> dict[str, Any]:
    """Validate every completion artifact and return publication-safe coverage evidence."""
    aggregate = recompute_bound_safe_aggregate(**kwargs)
    return {
        "valid": True,
        "coverage": aggregate["coverage"],
        "adjudication": aggregate["adjudication"],
        "unresolved_judgments": aggregate["gates"]["unresolved_judgments"],
        "outcome": aggregate["outcome"],
    }
