import copy
import json
from pathlib import Path

import pytest

from mm_embed.benchmark import bright_qrels_remediation as remediation

CONTROL = Path("benchmark/artifacts/bright-qrels-remediation-v0.1/control-freeze.json")


def _control() -> dict:
    return json.loads(CONTROL.read_text(encoding="utf-8"))


def _provenance(control: dict, input_row_sha256: str, source_annotation_sha256: str) -> dict:
    audit = control["source_identities"]["audit"]
    return {
        "judgment_origin": "agent_judged",
        "agent_role": "executor_primary",
        "session_id": "synthetic-test-session",
        "round": 1,
        "judged_at": "2026-09-06T18:18:37Z",
        "rubric_sha256": audit["rubric_sha256"],
        "input_artifact_sha256": audit["audit_pack"]["sha256"],
        "input_row_sha256": input_row_sha256,
        "source_annotation_sha256": source_annotation_sha256,
    }


def _fixture() -> tuple[dict, dict, dict, dict]:
    control = _control()
    source_sha256 = control["source_identities"]["audit"]["primary_annotations"]["sha256"]
    inventory = {}
    dispositions = []
    official = {}
    qrels_rows = []
    for track, suffix in (("economics", "e"), ("psychology", "p")):
        query_id = f"query-{suffix}"
        input_row_sha256 = ("1" if suffix == "e" else "2") * 64
        units = (
            ("query", None, "clear", "retain_query", "document_level_support"),
            ("gold", f"gold-{suffix}", "clear", "retain_official_positive", "direct_supporting_span"),
            ("candidate", f"candidate-{suffix}", "unjudged", "no_change_unjudged", "contextual_or_topical_only"),
        )
        for unit_type, document_id, status, disposition, evidence_level in units:
            key = (unit_type, track, query_id, document_id)
            inventory[key] = {
                "source_judgment_status": status,
                "input_row_sha256": input_row_sha256,
                "source_annotation_sha256": source_sha256,
            }
            dispositions.append(
                {
                    "unit_type": unit_type,
                    "track": track,
                    "query_id": query_id,
                    "document_id": document_id,
                    "source_judgment_status": status,
                    "taxonomy": ["no_qrels_defect"],
                    "disposition": disposition,
                    "evidence_level": evidence_level,
                    "supporting_spans": [],
                    "rationale": "Synthetic control fixture.",
                    "provenance": _provenance(control, input_row_sha256, source_sha256),
                }
            )
        official[(track, query_id, f"gold-{suffix}")] = 1
        qrels_rows.append(
            {
                "track": track,
                "query_id": query_id,
                "document_id": f"gold-{suffix}",
                "grade": 1,
                "operation": "retain",
            }
        )
    proposal = {
        "schema_version": "1",
        "proposal_id": "bright-qrels-remediation-proposal-v0.1",
        "version": "0.1",
        "control_sha256": CONTROL.with_suffix(".json.sha256").read_text(encoding="utf-8").strip(),
        "status": "inactive",
        "activation": {
            "default_activation": False,
            "registry_integration": False,
            "runner_integration": False,
        },
        "publication": {
            "classification": "restricted_no_publish",
            "public_export_allowed": False,
            "leaderboard_allowed": False,
            "publish_allowed": False,
        },
        "source_identities": copy.deepcopy(control["source_identities"]),
        "dispositions": dispositions,
        "proposed_qrels_rows": qrels_rows,
        "conclusion": {
            "tracks": {
                "economics": "remediable_with_existing_data",
                "psychology": "remediable_with_existing_data",
            },
            "overall": "remediable_with_existing_data",
        },
    }
    return proposal, control, inventory, official


def _validate_fixture(proposal: dict, control: dict, inventory: dict, official: dict) -> dict:
    return remediation._validate_proposal_payload(proposal, control, inventory, official)


def test_real_control_freeze_and_immutable_baselines_validate() -> None:
    result = remediation.validate_control(CONTROL)
    assert result["coverage"] == {
        "queries": 120,
        "gold_items": 1_235,
        "candidate_items": 2_804,
        "tracks": {"economics": 60, "psychology": 60},
        "outcome": "fail_closed",
    }
    assert result["proposal_status"] == "not_created"
    assert result["publication"] == "closed"


def test_minimal_synthetic_inactive_proposal_validates() -> None:
    proposal, control, inventory, official = _fixture()
    result = _validate_fixture(proposal, control, inventory, official)
    assert result["overall"] == "remediable_with_existing_data"


def test_missing_provenance_fails_closed() -> None:
    proposal, control, inventory, official = _fixture()
    del proposal["dispositions"][0]["provenance"]["session_id"]
    with pytest.raises(remediation.BrightQrelsRemediationError, match="provenance"):
        _validate_fixture(proposal, control, inventory, official)


def test_unknown_identity_fails_closed() -> None:
    proposal, control, inventory, official = _fixture()
    proposal["dispositions"][2]["document_id"] = "unknown-document"
    with pytest.raises(remediation.BrightQrelsRemediationError, match="unknown disposition identity"):
        _validate_fixture(proposal, control, inventory, official)


def test_duplicate_or_conflicting_disposition_fails_closed() -> None:
    proposal, control, inventory, official = _fixture()
    conflicting = copy.deepcopy(proposal["dispositions"][2])
    conflicting["disposition"] = "abstain_uncertain"
    proposal["dispositions"].append(conflicting)
    with pytest.raises(remediation.BrightQrelsRemediationError, match="duplicate or conflicting"):
        _validate_fixture(proposal, control, inventory, official)


def test_source_identity_drift_fails_closed() -> None:
    proposal, control, inventory, official = _fixture()
    proposal["source_identities"]["audit"]["audit_pack"]["sha256"] = "0" * 64
    with pytest.raises(remediation.BrightQrelsRemediationError, match="source identities drifted"):
        _validate_fixture(proposal, control, inventory, official)


def test_default_activation_fails_closed() -> None:
    proposal, control, inventory, official = _fixture()
    proposal["activation"]["default_activation"] = True
    with pytest.raises(remediation.BrightQrelsRemediationError, match="default_activation"):
        _validate_fixture(proposal, control, inventory, official)


def test_proposal_outside_restricted_root_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(remediation.BrightQrelsRemediationError, match="frozen restricted root|exact frozen lexical"):
        remediation._assert_restricted_proposal_path(Path.cwd(), tmp_path / "proposal.json", "results/restricted")


def test_unjudged_candidate_cannot_become_a_decided_nonpositive() -> None:
    proposal, control, inventory, official = _fixture()
    proposal["dispositions"][2]["disposition"] = "not_applicable"
    with pytest.raises(remediation.BrightQrelsRemediationError, match="unjudged evidence"):
        _validate_fixture(proposal, control, inventory, official)


def test_ambiguous_source_cannot_become_a_decided_disposition() -> None:
    proposal, control, inventory, official = _fixture()
    row = proposal["dispositions"][1]
    key = remediation._unit_key(row)
    inventory[key]["source_judgment_status"] = "ambiguous"
    row["source_judgment_status"] = "ambiguous"
    row["taxonomy"] = ["gold_ambiguity"]
    row["disposition"] = "quarantine_official_positive"
    row["evidence_level"] = "conflicting"
    with pytest.raises(remediation.BrightQrelsRemediationError, match="ambiguous or uncertain"):
        _validate_fixture(proposal, control, inventory, official)


def test_negative_qrels_grade_fails_closed() -> None:
    proposal, control, inventory, official = _fixture()
    proposal["proposed_qrels_rows"][0]["grade"] = 0
    with pytest.raises(remediation.BrightQrelsRemediationError, match="grade"):
        _validate_fixture(proposal, control, inventory, official)


def test_undeclared_negative_label_field_fails_closed() -> None:
    proposal, control, inventory, official = _fixture()
    proposal["dispositions"][2]["proposed_relevance"] = "not_relevant"
    with pytest.raises(remediation.BrightQrelsRemediationError, match="proposed_relevance"):
        _validate_fixture(proposal, control, inventory, official)


def test_declared_conclusion_cannot_override_frozen_derivation() -> None:
    proposal, control, inventory, official = _fixture()
    proposal["conclusion"]["overall"] = "insufficient_existing_evidence"
    with pytest.raises(remediation.BrightQrelsRemediationError, match="three-state derivation"):
        _validate_fixture(proposal, control, inventory, official)


def test_control_sidecar_detects_byte_drift(tmp_path: Path) -> None:
    copied = tmp_path / "control-freeze.json"
    copied.write_text(CONTROL.read_text(encoding="utf-8") + " ", encoding="utf-8")
    copied.with_suffix(".json.sha256").write_text(
        CONTROL.with_suffix(".json.sha256").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(remediation.BrightQrelsRemediationError, match="control identity mismatch"):
        remediation.validate_control(copied)


def test_official_qrels_byte_drift_fails_closed(tmp_path: Path) -> None:
    source = Path("data/bright-nontechnical-pilot-v0.2/economics/qrels.jsonl")
    drifted = tmp_path / "qrels.jsonl"
    drifted.write_text(source.read_text(encoding="utf-8").rstrip("\n") + " \n", encoding="utf-8")
    with pytest.raises(remediation.BrightQrelsRemediationError, match="official qrels identity drift"):
        remediation.validate_control(CONTROL, path_overrides={"economics_qrels": drifted})
