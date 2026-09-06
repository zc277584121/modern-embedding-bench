import copy
import hashlib
import json
from pathlib import Path

import pytest

from mm_embed.benchmark import bright_label_audit as audit
from mm_embed.benchmark import bright_label_audit_completion as completion
from mm_embed.hf_publish.export import _copy_benchmark_data, _restricted_bright_object_reason

RESTRICTED = Path("results/bright-label-quality-audit-v0.1/restricted")
ARTIFACTS = Path("benchmark/artifacts/bright-label-quality-audit-v0.1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    audit.write_jsonl(path, rows)


def _kwargs() -> dict[str, Path]:
    return {
        "predeclaration_path": ARTIFACTS / "predeclaration.json",
        "review_plan_path": Path("data/bright-nontechnical-pilot-v0.2/review-plan-120.jsonl"),
        "audit_pack_path": Path("results/bright-nontechnical-pilot-v0.2/audit/audit-pack-120.jsonl"),
        "rubric_path": Path("benchmark/research/bright_label_quality_audit_protocol_20260906.md"),
        "primary_input_path": RESTRICTED / "primary/input.jsonl",
        "primary_annotation_path": RESTRICTED / "primary/annotations.jsonl",
        "validator_core_input_path": RESTRICTED / "validator-blind/core-input.jsonl",
        "validator_core_annotation_path": RESTRICTED / "validator-blind/core-annotations.jsonl",
        "validator_supplement_input_path": RESTRICTED / "validator-blind/supplement-input.jsonl",
        "validator_supplement_annotation_path": RESTRICTED / "validator-blind/supplement-annotations.jsonl",
        "selection_path": RESTRICTED / "control/double-review-selection.jsonl",
        "frozen_manifest_path": RESTRICTED / "frozen-input-manifest.json",
        "validator_access_manifest_path": RESTRICTED / "validator-blind/access-manifest.json",
        "validator_seal_path": RESTRICTED / "validator-blind/core-seal.json",
        "adjudication_path": RESTRICTED / "adjudication/decisions.jsonl",
    }


def _internal_paths(kwargs: dict[str, Path]) -> dict[str, Path]:
    return completion._paths(
        review_plan=kwargs["review_plan_path"],
        audit_pack=kwargs["audit_pack_path"],
        rubric=kwargs["rubric_path"],
        primary_input=kwargs["primary_input_path"],
        primary_annotations=kwargs["primary_annotation_path"],
        validator_core_input=kwargs["validator_core_input_path"],
        validator_core_annotations=kwargs["validator_core_annotation_path"],
        validator_supplement_input=kwargs["validator_supplement_input_path"],
        validator_supplement_annotations=kwargs["validator_supplement_annotation_path"],
        selection=kwargs["selection_path"],
        frozen_manifest=kwargs["frozen_manifest_path"],
        validator_access_manifest=kwargs["validator_access_manifest_path"],
        validator_seal=kwargs["validator_seal_path"],
        adjudications=kwargs["adjudication_path"],
    )


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_validator_owned_evidence_hashes_are_immutable() -> None:
    assert _sha256(RESTRICTED / "primary/annotations.jsonl") == (
        "4f18deca204c8bcbfd269ecc216f130abbd6a5630a5c245870cfa55a341b0356"
    )
    assert _sha256(RESTRICTED / "validator-blind/core-annotations.jsonl") == (
        "bbe93a3af20f64d6d240a82b481fba7262fca2fbfa9369e966d5c802eb0318ff"
    )
    assert _sha256(RESTRICTED / "validator-blind/supplement-annotations.jsonl") == (
        "8e975641f0d083f972a71286f9782413c75557a5c85627bc35026e5d8940359f"
    )
    assert _sha256(RESTRICTED / "validator-blind/core-seal.json") == (
        "613a197bf183305157aadd68d4a5db3c5dda81d4e204bfce9009b0e556dbb11f"
    )
    assert _sha256(RESTRICTED / "adjudication/decisions.jsonl") == (
        "2be3f846b5ee6dc3e7efc11625f3fa813bb31312e14808a871715cab9a74db39"
    )


def test_completed_recompute_consumes_adjudication_and_counts_unresolved() -> None:
    result = audit.recompute_safe_aggregate(**_kwargs())
    assert result["coverage"] == {
        "queries": 120,
        "gold_items": 1_235,
        "candidate_items": 2_804,
        "tracks": {"economics": 60, "psychology": 60},
    }
    assert result["adjudication"] == {
        "decisions": 241,
        "review_scope": {"core_conflict": 205, "supplement_low_confidence": 36},
        "unit_type": {"candidate": 133, "gold": 85, "query": 23},
    }
    assert result["gates"]["unresolved_judgments"] == {
        "threshold": 0,
        "comparison": "exceeds",
        "value": 1,
        "breakdown": {
            "primary": 0,
            "validator_core": 0,
            "validator_supplement": 0,
            "adjudicated": 1,
            "effective_final": 1,
        },
        "status": "fail_closed",
    }
    assert result["queries"]["decision_status"] == {"abstain": 0, "decided": 119, "uncertain": 1}
    assert result["outcome"] == "fail_closed"
    forbidden = {
        "query_id",
        "document_id",
        "content",
        "notes",
        "session_id",
        "model",
        "provider",
        "path",
        "rank",
    }
    assert not (set(_all_keys(result)) & forbidden)
    aggregate_text = json.dumps(result, sort_keys=True)
    assert all(row["query"] not in aggregate_text for row in _read_jsonl(_kwargs()["primary_input_path"]))
    assert _restricted_bright_object_reason(result, "bright-nontechnical-pilot-v0.2") is None


def test_synchronized_primary_text_and_provenance_drift_fails_closed(tmp_path: Path) -> None:
    kwargs = _kwargs()
    inputs = _read_jsonl(kwargs["primary_input_path"])
    annotations = _read_jsonl(kwargs["primary_annotation_path"])
    inputs[0]["query"] += " drift"
    input_path = tmp_path / "primary-input.jsonl"
    _write_jsonl(input_path, inputs)
    input_sha256 = _sha256(input_path)
    input_row_sha256 = audit._sha256_json(inputs[0])
    annotations[0]["input_row_sha256"] = input_row_sha256
    for index, row in enumerate(annotations):
        for judgment in [*row["gold_reviews"], *row["candidate_reviews"], row["query_review"]]:
            judgment["provenance"]["input_artifact_sha256"] = input_sha256
            if index == 0:
                judgment["provenance"]["input_row_sha256"] = input_row_sha256
    annotation_path = tmp_path / "primary-annotations.jsonl"
    _write_jsonl(annotation_path, annotations)
    kwargs["primary_input_path"] = input_path
    kwargs["primary_annotation_path"] = annotation_path
    with pytest.raises(audit.BrightLabelAuditError, match="exactly derive"):
        audit.recompute_safe_aggregate(**kwargs)


@pytest.mark.parametrize("field", ["query_length_stratum", "source_plan_row_sha256"])
def test_selection_metadata_drift_fails_closed(tmp_path: Path, field: str) -> None:
    kwargs = _kwargs()
    rows = _read_jsonl(kwargs["selection_path"])
    rows[0][field] = "long" if field == "query_length_stratum" else "0" * 64
    path = tmp_path / "selection.jsonl"
    _write_jsonl(path, rows)
    kwargs["selection_path"] = path
    with pytest.raises(audit.BrightLabelAuditError, match="selection does not exactly derive"):
        audit.recompute_safe_aggregate(**kwargs)


@pytest.mark.parametrize("scope", ["validator_core_input_path", "validator_supplement_input_path"])
def test_validator_source_content_drift_fails_closed(tmp_path: Path, scope: str) -> None:
    kwargs = _kwargs()
    rows = _read_jsonl(kwargs[scope])
    rows[0]["query"] += " drift"
    path = tmp_path / "input.jsonl"
    _write_jsonl(path, rows)
    kwargs[scope] = path
    expected = "core input" if scope == "validator_core_input_path" else "supplement"
    with pytest.raises(audit.BrightLabelAuditError, match=expected):
        audit.recompute_safe_aggregate(**kwargs)


def test_frozen_manifest_access_logical_path_drift_fails_closed(tmp_path: Path) -> None:
    kwargs = _kwargs()
    manifest = json.loads(kwargs["frozen_manifest_path"].read_text())
    manifest["validator_access_manifest"]["path"] = "renamed-access.json"
    path = tmp_path / "frozen-input-manifest.json"
    path.write_text(json.dumps(manifest))
    kwargs["frozen_manifest_path"] = path
    with pytest.raises(audit.BrightLabelAuditError, match="invalid frozen-input manifest"):
        audit.recompute_safe_aggregate(**kwargs)


def test_resealed_validator_gold_uncertain_cannot_be_ignored(tmp_path: Path) -> None:
    kwargs = _kwargs()
    rows = _read_jsonl(kwargs["validator_core_annotation_path"])
    judgment = rows[0]["gold_reviews"][0]
    judgment["support"] = "uncertain"
    judgment["confidence"] = "low"
    judgment["decision_status"] = "uncertain"
    annotation_path = tmp_path / "core-annotations.jsonl"
    _write_jsonl(annotation_path, rows)
    seal_path = tmp_path / "core-seal.json"
    audit.seal_validator_annotations(
        annotation_path,
        kwargs["validator_core_input_path"],
        kwargs["validator_access_manifest_path"],
        seal_path,
    )
    kwargs["validator_core_annotation_path"] = annotation_path
    kwargs["validator_seal_path"] = seal_path
    with pytest.raises(audit.BrightLabelAuditError, match="adjudication"):
        completion._validate_completion_evidence(_internal_paths(kwargs))


@pytest.mark.parametrize("attack", ["unknown", "missing", "duplicate", "omitted", "prior", "seal"])
def test_adjudication_contract_rejects_integrity_attacks(tmp_path: Path, attack: str) -> None:
    kwargs = _kwargs()
    rows = _read_jsonl(kwargs["adjudication_path"])
    if attack == "unknown":
        rows[0]["unexpected"] = True
    elif attack == "missing":
        del rows[0]["trigger_reasons"]
    elif attack == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif attack == "omitted":
        rows.pop()
    elif attack == "prior":
        rows[0]["primary_judgment"]["notes"] += " drift"
    else:
        rows[0]["adjudication_provenance"]["validator_core_seal_sha256"] = "0" * 64
        rows[0]["adjudicated_judgment"]["provenance"]["validator_core_seal_sha256"] = "0" * 64
    path = tmp_path / "decisions.jsonl"
    _write_jsonl(path, rows)
    kwargs["adjudication_path"] = path
    with pytest.raises(audit.BrightLabelAuditError):
        completion._validate_completion_evidence(_internal_paths(kwargs))


def test_renamed_adjudication_is_rejected_by_real_copy_path(tmp_path: Path) -> None:
    attack = tmp_path / "neutral"
    attack.mkdir()
    first_row = (RESTRICTED / "adjudication/decisions.jsonl").read_text().splitlines()[0]
    payload = json.loads(first_row)
    assert (
        _restricted_bright_object_reason(payload, "bright-nontechnical-pilot-v0.2")
        == "BRIGHT label-audit adjudication decisions"
    )
    (attack / "payload.data").write_text(first_row + "\n")
    with pytest.raises(ValueError, match="public export denied.*payload[.]data"):
        _copy_benchmark_data(attack, tmp_path / "export", include_images=False)
