"""Tests for the deterministic learned-sparse phase-one synthesis."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from mm_embed.benchmark import learned_sparse_phase_one as phase_one

REPO_ROOT = Path(__file__).parents[1]
ARTIFACT_ROOT = REPO_ROOT / "benchmark/artifacts/learned-sparse-phase-one-v0.1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk(value: object):
    yield value
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _walk(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk(nested)


def test_deterministic_check_matches_saved_outputs() -> None:
    result = phase_one.check_outputs(REPO_ROOT)
    assert result["status"] == "pass"
    assert result["model_loaded"] is False
    assert result["milvus_connected"] is False
    assert result["output_count"] == 6


def test_all_output_schemas_validate() -> None:
    bindings = {
        "summary.json": "learned-sparse-phase-one-summary-v01.schema.json",
        "evidence-map.json": "learned-sparse-phase-one-evidence-map-v01.schema.json",
        "claim-support-matrix.json": "learned-sparse-phase-one-claim-support-v01.schema.json",
        "manifest.json": "learned-sparse-phase-one-manifest-v01.schema.json",
    }
    for artifact_name, schema_name in bindings.items():
        schema = _load(REPO_ROOT / "schemas" / schema_name)
        value = _load(ARTIFACT_ROOT / artifact_name)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)


def test_accepted_input_authentication_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_id = "s005-summary"
    original = phase_one.ACCEPTED_FILES[artifact_id]
    monkeypatch.setitem(
        phase_one.ACCEPTED_FILES,
        artifact_id,
        (original[0], original[1], "0" * 64),
    )
    with pytest.raises(phase_one.PhaseOneError, match="hash mismatch"):
        phase_one.authenticate_inputs(REPO_ROOT)


def test_milestone_coverage_is_fail_closed() -> None:
    values = phase_one.load_inputs(REPO_ROOT)
    changed = copy.deepcopy(values)
    changed["system_summary"]["coverage"]["raw_trial_files_recomputed"] -= 1
    with pytest.raises(phase_one.PhaseOneError, match="coverage mismatch"):
        phase_one._validate_milestone_counts(changed)


def test_tracks_metrics_and_uncertainty_remain_separate() -> None:
    summary = _load(ARTIFACT_ROOT / "summary.json")
    assert summary["quality"]["track_aggregation"] == "none"
    assert set(summary["quality"]["tracks"]) == {"economics", "psychology"}
    assert summary["quality"]["tracks"]["economics"]["point_order_ndcg_at_10"][0] == ("opensearch-doc-v2-distill")
    assert summary["quality"]["tracks"]["psychology"]["point_order_ndcg_at_10"][0] == "bge-m3"
    for track in phase_one.TRACKS:
        uncertainty = summary["quality"]["paired_uncertainty"]["tracks"][track]
        assert uncertainty["pairs"] == 21
        assert uncertainty["metric_comparisons"] == 105
        assert uncertainty["bootstrap_samples_each"] == 10_000
        assert sum(uncertainty["verdict_counts"].values()) == 105


def test_slice_and_case_boundaries_are_explicit() -> None:
    summary = _load(ARTIFACT_ROOT / "summary.json")
    assert summary["cases"] == {
        "counts": {"per_track_disagreement": 6, "per_track_failure": 6, "total": 24},
        "opaque_only": True,
    }
    economics_bins = [row for dimension in summary["slices"]["economics"] for row in dimension["bins"]]
    assert any(row["n"] == 8 and row["small_sample"] for row in economics_bins)
    assert all(row["inference"] == "point_estimates_only_no_win_claim" for row in economics_bins if row["n"] < 10)


def test_anchor_fairness_and_system_layering() -> None:
    summary = _load(ARTIFACT_ROOT / "summary.json")
    assert all(summary["anchors"]["fairness_checks"].values())
    assert summary["anchors"]["role"] == "limited_contextual_anchor"
    assert summary["system"]["coverage"]["native_cells"] == 18
    assert summary["system"]["coverage"]["system_100k_cells"] == 9
    assert summary["system"]["coverage"]["system_1m_cells"] == 9
    assert summary["system"]["coverage"]["raw_trial_files_recomputed"] == 1296
    assert summary["system"]["coverage"]["warmup_files_validated"] == 432
    assert summary["system"]["correctness"]["native_fp16_aware_tie_recall"] == 1.0
    assert summary["system"]["correctness"]["system_only_near_tie_is_quality_evidence"] is False
    assert summary["system"]["qps"]["crosses_one"] is True
    assert summary["system"]["persisted_index"]["sindi_larger_pairs"] == 12


def test_claims_and_recommendations_are_fully_linked() -> None:
    summary = _load(ARTIFACT_ROOT / "summary.json")
    claims = _load(ARTIFACT_ROOT / "claim-support-matrix.json")
    evidence = _load(ARTIFACT_ROOT / "evidence-map.json")
    claim_by_id = {row["claim_id"]: row for row in claims["claims"]}
    assert len(claim_by_id) == len(claims["claims"])
    assert {row["support"] for row in claims["claims"]} == {
        "supported",
        "limited_anchor",
        "unsupported",
    }
    assert claim_by_id["C-UNIVERSAL-MODEL"]["support"] == "unsupported"
    assert claim_by_id["C-UNIVERSAL-SINDI"]["support"] == "unsupported"
    assert claim_by_id["C-MULTILINGUAL-GAP"]["support"] == "unsupported"
    assert claim_by_id["C-PRODUCTION"]["support"] == "unsupported"
    assert {row["claim_id"] for row in evidence["claims"]} == set(claim_by_id)
    for recommendation in summary["recommendations"]:
        assert set(recommendation["support_claims"]) <= set(claim_by_id)
        assert recommendation["limitations"]
    for table in evidence["tables"]:
        assert set(table["claim_ids"]) <= set(claim_by_id)
        assert set(table["artifact_refs"]) <= set(evidence["tracked_artifacts"])
    assert {table["table_id"] for table in evidence["tables"]} == {
        "T-DATA",
        "T-MODEL-PORTFOLIO",
        "T-QUALITY-ECONOMICS",
        "T-QUALITY-PSYCHOLOGY",
        "T-ANCHORS",
        "T-ENCODING-EXACT",
        "T-SLICES",
        "T-SYSTEM-COVERAGE",
        "T-SYSTEM-RESULTS",
        "T-CLAIM-BOUNDARY",
    }
    assert {row["recommendation_id"] for row in evidence["recommendations"]} == {
        row["id"] for row in summary["recommendations"]
    }
    for recommendation in evidence["recommendations"]:
        assert set(recommendation["claim_ids"]) <= set(claim_by_id)
        assert recommendation["sources"]
        assert recommendation["limitations"]


def test_public_artifacts_do_not_expose_private_paths_or_restricted_rows() -> None:
    private_path_markers = (
        "/" + "home/",
        "/" + "data1/",
        "/" + "data2/",
        "results" + "/",
        "data/" + "bright-nontechnical",
    )
    forbidden_keys = {
        "query_id",
        "document_id",
        "query_ids",
        "document_ids",
        "rankings",
        "source_text",
        "raw_text",
    }
    for path in [
        *ARTIFACT_ROOT.glob("*.json"),
        REPO_ROOT / "benchmark/research/learned_sparse_phase_one_20260904.md",
    ]:
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in private_path_markers)
        if path.suffix == ".json":
            value = json.loads(text)
            assert forbidden_keys.isdisjoint(item for item in _walk(value) if isinstance(item, str))
    publication = _load(ARTIFACT_ROOT / "summary.json")["publication"]
    assert publication["gate"] == "closed"
    assert publication["contains_source_text"] is False
    assert publication["contains_canonical_ids"] is False
    assert publication["contains_raw_rankings"] is False
    assert publication["contains_private_paths"] is False


def test_manifest_and_sidecar_bind_exact_output_bytes() -> None:
    manifest_path = ARTIFACT_ROOT / "manifest.json"
    manifest = _load(manifest_path)
    for entry in manifest["outputs"].values():
        path = REPO_ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert phase_one.sha256_file(path) == entry["sha256"]
    assert (ARTIFACT_ROOT / "manifest.sha256").read_text(encoding="ascii") == (
        phase_one.sha256_file(manifest_path) + "\n"
    )
