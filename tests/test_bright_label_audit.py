import hashlib
import json
from pathlib import Path

import pytest

from mm_embed.benchmark import bright_label_audit as audit
from mm_embed.hf_publish.export import _copy_benchmark_data, _restricted_bright_object_reason

REVIEW_PLAN = Path("data/bright-nontechnical-pilot-v0.2/review-plan-120.jsonl")
AUDIT_PACK = Path("results/bright-nontechnical-pilot-v0.2/audit/audit-pack-120.jsonl")
RUBRIC = Path("benchmark/research/bright_label_quality_audit_protocol_20260906.md")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def _provenance(row: dict, input_sha256: str, rubric_sha256: str, role: str) -> dict:
    return {
        "judgment_origin": "agent_judged",
        "agent_role": "executor_primary" if role == "primary" else "independent_validator",
        "provider": "test-provider",
        "model": "test-model",
        "model_revision": "test-revision",
        "session_id": f"test-{role}-session",
        "round": 1,
        "judged_at": "2026-09-06T13:00:00Z",
        "rubric_sha256": rubric_sha256,
        "input_artifact_sha256": input_sha256,
        "input_row_sha256": audit._sha256_json(row),
    }


def _annotation_rows(input_path: Path, rubric_sha256: str, role: str) -> list[dict]:
    inputs = audit._read_jsonl(input_path)
    input_sha256 = _file_sha256(input_path)
    rows = []
    for source in inputs:
        provenance = _provenance(source, input_sha256, rubric_sha256, role)
        gold_ids = {item["document_id"] for item in source["gold_documents"]}
        gold = [
            {
                "document_id": item["document_id"],
                "support": "supports",
                "ambiguity": "unambiguous",
                "confidence": "high",
                "decision_status": "decided",
                "notes": "",
                "provenance": provenance,
            }
            for item in source["gold_documents"]
        ]
        candidates = []
        for item in source["candidate_documents"]:
            existing_positive = item["document_id"] in gold_ids
            candidates.append(
                {
                    "document_id": item["document_id"],
                    "relevance": "relevant" if existing_positive else "not_relevant",
                    "likely_hard_negative": "no",
                    "suspected_missing_positive": "not_applicable" if existing_positive else "not_credible",
                    "sensitive_information_risk": "none",
                    "sensitive_information_types": [],
                    "confidence": "high",
                    "decision_status": "decided",
                    "notes": "",
                    "provenance": provenance,
                }
            )
        rows.append(
            {
                "schema_version": "1",
                "audit_id": audit.AUDIT_ID,
                "review_role": role,
                "track": source["track"],
                "query_id": source["query_id"],
                "input_row_sha256": audit._sha256_json(source),
                "rubric_sha256": rubric_sha256,
                "gold_reviews": gold,
                "candidate_reviews": candidates,
                "query_review": {
                    "credible_missing_positive": "no",
                    "credible_missing_positive_ids": [],
                    "likely_hard_negative_present": "no",
                    "likely_hard_negative_ids": [],
                    "qrels_completeness": "complete_enough",
                    "answerability": "answerable",
                    "label_incompleteness_risk": "none",
                    "sensitive_information_risk": "none",
                    "confidence": "high",
                    "decision_status": "decided",
                    "notes": "",
                    "provenance": provenance,
                },
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    audit.write_jsonl(path, rows)


@pytest.fixture()
def frozen(tmp_path: Path) -> dict[str, Path]:
    restricted = tmp_path / "restricted"
    predeclaration = tmp_path / "predeclaration.json"
    audit.freeze_audit(REVIEW_PLAN, AUDIT_PACK, RUBRIC, restricted, predeclaration)
    return {"restricted": restricted, "predeclaration": predeclaration}


def test_frozen_source_identities_and_counts_are_exact() -> None:
    plan, pack = audit.verify_frozen_inputs(REVIEW_PLAN, AUDIT_PACK)
    assert len(plan) == len(pack) == 120
    assert sum(len(row["gold_documents"]) for row in pack) == 1_235
    assert sum(len(row["baseline_top10_union"]) for row in pack) == 2_804


def test_double_review_selection_is_score_blind_stratified_and_stable(frozen: dict[str, Path]) -> None:
    restricted = frozen["restricted"]
    selection = audit._read_jsonl(restricted / "control" / "double-review-selection.jsonl")
    predeclaration = json.loads(frozen["predeclaration"].read_text())
    cells = {
        (track, stratum): sum(row["track"] == track and row["query_length_stratum"] == stratum for row in selection)
        for track in audit.TRACKS
        for stratum in audit.QUERY_LENGTH_STRATA
    }
    assert len(selection) == 24
    assert set(cells.values()) == {4}
    assert audit._identity_rows(selection) == predeclaration["double_review"]["selection_identity_sha256"]
    assert predeclaration["double_review"]["selection_score_used"] is False
    assert predeclaration["schema_identities"] == {
        name: _file_sha256(audit._schema_path(name)) for name in audit.SCHEMA_NAMES
    }


def test_blind_inputs_remove_rank_score_status_and_review_fields(frozen: dict[str, Path]) -> None:
    restricted = frozen["restricted"]
    primary = audit._read_jsonl(restricted / "primary" / "input.jsonl")
    validator = audit._read_jsonl(restricted / "validator-blind" / "core-input.jsonl")
    access = json.loads((restricted / "validator-blind" / "access-manifest.json").read_text())
    forbidden = {"baseline_positions", "score", "rank", "review", "review_summary", "label_status", "judged_grade"}
    assert len(primary) == 120
    assert len(validator) == 24
    assert not (set(_all_keys(primary)) & forbidden)
    assert not (set(_all_keys(validator)) & forbidden)
    assert access["primary_input_allowed"] is False
    assert access["primary_annotations_allowed"] is False
    assert access["comparison_allowed_before_validator_seal"] is False
    assert access["core_input_rows"] == 24
    assert access["annotation_schema_sha256"] == _file_sha256(
        audit._schema_path("bright-label-audit-annotation-v01.schema.json")
    )


@pytest.mark.parametrize(
    "relative_source",
    [Path("primary/input.jsonl"), Path("control/double-review-selection.jsonl")],
)
def test_public_export_rejects_renamed_restricted_audit_rows(
    frozen: dict[str, Path],
    tmp_path: Path,
    relative_source: Path,
) -> None:
    attack = tmp_path / "neutral"
    attack.mkdir()
    (attack / "payload.data").write_bytes((frozen["restricted"] / relative_source).read_bytes())
    with pytest.raises(ValueError, match="public export denied"):
        _copy_benchmark_data(attack, tmp_path / "export", include_images=False)


def test_public_export_classifier_allows_safe_predeclaration(frozen: dict[str, Path]) -> None:
    predeclaration = json.loads(frozen["predeclaration"].read_text())
    assert _restricted_bright_object_reason(predeclaration, "bright-nontechnical-pilot-v0.2") is None


def test_kappa_is_undefined_for_single_class_and_must_fail_closed() -> None:
    result = audit.cohen_kappa(["not_relevant", "not_relevant"], ["not_relevant", "not_relevant"])
    assert result["observed_agreement"] == 1.0
    assert result["expected_agreement"] == 1.0
    assert result["kappa"] is None


@pytest.mark.parametrize("mutation", ["input-identity", "missing-positive", "uncertain-confidence"])
def test_annotation_validation_fails_closed_on_semantic_mutations(
    frozen: dict[str, Path],
    mutation: str,
) -> None:
    restricted = frozen["restricted"]
    rows = _annotation_rows(
        restricted / "validator-blind" / "core-input.jsonl",
        _file_sha256(RUBRIC),
        "validator",
    )
    if mutation == "input-identity":
        rows[0]["input_row_sha256"] = "0" * 64
    elif mutation == "missing-positive":
        rows[0]["candidate_reviews"][0]["suspected_missing_positive"] = "credible"
    else:
        rows[0]["candidate_reviews"][0]["relevance"] = "uncertain"
        rows[0]["candidate_reviews"][0]["decision_status"] = "uncertain"
        rows[0]["candidate_reviews"][0]["confidence"] = "high"
    path = restricted / "mutated.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(audit.BrightLabelAuditError):
        audit.validate_annotations(
            path,
            restricted / "validator-blind" / "core-input.jsonl",
            expected_role="validator",
        )


def test_validator_supplement_contains_only_source_rows_for_low_confidence_queries(frozen: dict[str, Path]) -> None:
    restricted = frozen["restricted"]
    rows = _annotation_rows(restricted / "primary" / "input.jsonl", _file_sha256(RUBRIC), "primary")
    gold_ids = {item["document_id"] for item in rows[0]["gold_reviews"]}
    target = next(item for item in rows[0]["candidate_reviews"] if item["document_id"] not in gold_ids)
    target.update(
        {
            "relevance": "uncertain",
            "likely_hard_negative": "uncertain",
            "suspected_missing_positive": "uncertain",
            "sensitive_information_risk": "uncertain",
            "confidence": "low",
            "decision_status": "uncertain",
        }
    )
    rows[0]["query_review"].update(
        {
            "credible_missing_positive": "uncertain",
            "likely_hard_negative_present": "uncertain",
            "qrels_completeness": "uncertain",
            "answerability": "uncertain",
            "label_incompleteness_risk": "uncertain",
            "sensitive_information_risk": "uncertain",
            "confidence": "low",
            "decision_status": "uncertain",
        }
    )
    primary_path = restricted / "primary" / "annotations.jsonl"
    supplement_path = restricted / "validator-blind" / "supplement-input.jsonl"
    _write_jsonl(primary_path, rows)
    result = audit.prepare_validator_supplement(
        primary_path,
        restricted / "primary" / "input.jsonl",
        supplement_path,
    )
    supplement = audit._read_jsonl(supplement_path)
    assert result["queries"] == 1
    assert len(supplement) == 1
    assert "review_role" not in set(_all_keys(supplement))
    assert "relevance" not in set(_all_keys(supplement))
