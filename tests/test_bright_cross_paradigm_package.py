from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

from jsonschema import Draft202012Validator

from mm_embed.benchmark.bright_cross_paradigm import sha256_file
from mm_embed.benchmark.bright_cross_paradigm_package import (
    METHODS,
    MINILM_REPRESENTATION_SCOPE,
    TFIDF_REPRESENTATION_SCOPE,
    candidate_paths,
    validate_machine_contracts,
)

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "benchmark/artifacts/bright-cross-paradigm-results-v0.1"


def load(name: str) -> dict:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def test_candidate_matrix_pairwise_slices_and_cases_are_complete() -> None:
    summary = load("summary.json")
    assert summary["matrix"] == {"methods": 14, "tracks": 2, "cells": 28, "complete": True}
    pairwise = load("pairwise.json")
    expected_pairs = set(itertools.combinations(sorted(METHODS), 2))
    for track in ("economics", "psychology"):
        assert {(row["left"], row["right"]) for row in pairwise["tracks"][track]} == expected_pairs
    slices = load("slices.json")
    assert sum(len(row["bins"]) for rows in slices["tracks"].values() for row in rows) == 26
    assert all(
        len(row["pairwise"]) in {0, 91} for rows in slices["tracks"].values() for item in rows for row in item["bins"]
    )
    cases = load("cases.json")
    assert cases["counts"] == {"total": 24, "per_track_disagreement": 6, "per_track_failure": 6}


def test_resource_failures_are_explicit_and_not_zero_cost() -> None:
    resources = load("resources.json")
    assert resources["coverage"]["planned"] == 28
    assert resources["coverage"]["passed"] + resources["coverage"]["failed_closed"] == 28
    assert resources["coverage"]["complete"] is True
    failures = [row for row in resources["cells"] if row["status"] == "failed_closed"]
    assert failures
    assert all(row["end_to_end_seconds"] > 0 for row in failures)
    assert all(row["failure"]["stopping_condition_applied"] is True for row in failures)
    assert all(row["document_encoding"] is not None for row in failures)
    assert all(row["query_encoding"] is not None for row in failures)
    assert all(row["representation"]["bytes"] > 0 for row in failures)
    assert all(row["search"]["trials_completed"] == 3 for row in failures)
    assert all(row["quality_verification"]["mismatched_order_queries"] > 0 for row in failures)


def test_lexical_and_dense_resource_scopes_are_family_specific() -> None:
    resources = load("resources.json")
    by_cell = {row["cell_id"]: row for row in resources["cells"]}
    for track in ("economics", "psychology"):
        assert by_cell[f"tfidf-word-sublinear:{track}"]["representation"]["scope"] == (TFIDF_REPRESENTATION_SCOPE)
        assert by_cell[f"all-minilm-l6-v2:{track}"]["representation"]["scope"] == (MINILM_REPRESENTATION_SCOPE)


def test_candidate_artifacts_match_schemas_and_sidecars() -> None:
    manifest = load("manifest.json")
    Draft202012Validator(
        json.loads((ROOT / "schemas/bright-cross-paradigm-manifest-v01.schema.json").read_text())
    ).validate(manifest)
    for relative, identity in manifest["tracked_files"].items():
        path = PACKAGE / relative if "/" not in relative else ROOT / relative
        assert path.stat().st_size == identity["bytes"]
        assert sha256_file(path) == identity["sha256"]
    for path in PACKAGE.glob("*.json"):
        assert path.with_suffix(".json.sha256").read_text(encoding="ascii").strip() == sha256_file(path)


def test_cross_file_contracts_and_negative_mutations() -> None:
    files = {
        name: load(name)
        for name in (
            "summary.json",
            "pairwise.json",
            "slices.json",
            "cases.json",
            "lineage.json",
            "resources.json",
            "descriptive-sensitivity.json",
            "scenario-guidance.json",
        )
    }
    validate_machine_contracts(files)
    result = subprocess.run(
        ["uv", "run", "--python", "3.12", "python", "scripts/check_bright_cross_paradigm_mutations.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["mutations_rejected"] == 25


def test_candidate_paths_exclude_restricted_and_protected_material() -> None:
    paths = candidate_paths()
    assert paths == tuple(sorted(paths))
    assert len(paths) == len(set(paths))
    assert not any(path.startswith("results/") for path in paths)
    assert not any("milvus-sindi-system" in path for path in paths)
    assert "pyproject.toml" not in paths
    assert "uv.lock" not in paths
