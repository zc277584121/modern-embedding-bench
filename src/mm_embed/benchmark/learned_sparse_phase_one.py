"""Deterministically derive the learned-sparse phase-one evidence package."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


class PhaseOneError(ValueError):
    """Raised when an accepted input or derived output fails closed."""


PHASE_ID = "learned-sparse-phase-one-v0.1"
STORY_ID = "S-20260814-011"
TRACKS = ("economics", "psychology")
METRICS = ("ndcg@10", "map@100", "mrr@10", "recall@10", "recall@100")
PUBLICATION = {
    "classification": "research_only",
    "gate": "closed",
    "public_export_allowed": False,
    "leaderboard_publish": False,
    "contains_source_text": False,
    "contains_canonical_ids": False,
    "contains_raw_rankings": False,
    "contains_private_paths": False,
}

ACCEPTED_COMMITS = {
    "S-20260814-005": {
        "commit": "10573c1f24975272bd68b61c6bece2ac1a25681e",
        "tree": "9fb8d6c0db9f8a09efbb2aefced2bf5a09929dd0",
    },
    "S-20260814-006": {
        "commit": "9283a47a49a5d6e6623badf7139ea3ae33360ef4",
        "tree": "dcffee28bba7f9df38db67989008daa5edbd12cf",
    },
    "S-20260814-008": {
        "commit": "74635817770023b70b1e9f9e97b3d83d9d11f987",
        "tree": "dc4a28d1805c82de5d4e2397fa163275a671638c",
    },
    "S-20260814-010": {
        "commit": "bad3248b0b704e9f4ddee3333aed1208d4e2b64e",
        "tree": "8b93654ffa89672525dea7928ebbb6467dc1b303",
    },
}

ACCEPTED_FILES = {
    "s005-manifest": (
        "S-20260814-005",
        "benchmark/artifacts/bright-nontechnical-pilot-v0.2/manifest.json",
        "6b48e45d4f8332a5386729898845c1a62b02b2fd1481b4665d501947a082fd86",
    ),
    "s005-summary": (
        "S-20260814-005",
        "benchmark/artifacts/bright-nontechnical-pilot-v0.2/summary.json",
        "7aa2f3561263d2f8d82026244841304b26b725caff185b4a21a8a868c610c249",
    ),
    "s005-report": (
        "S-20260814-005",
        "benchmark/research/bright_multidomain_pilot_v02.md",
        "6314edf84419bfd98dbc3a0651d9de5cae39e1f1922da012f3d673ff58c6fa8b",
    ),
    "s005-materialization-schema": (
        "S-20260814-005",
        "schemas/bright-multidomain-materialization-v02.schema.json",
        "9c9ab271140ef61c9e997af8f19f11387e269a1046c58c84840151823d4aad4a",
    ),
    "s005-result-schema": (
        "S-20260814-005",
        "schemas/bright-multidomain-result-v02.schema.json",
        "7219d66bb584b9f1054c389af9aeac6de8a5faff885be58250b1ec84045f8ff2",
    ),
    "s006-inventory": (
        "S-20260814-006",
        "benchmark/artifacts/bright-learned-sparse-batch-a-v0.1/inventory.json",
        "a4b1ff3c9e5f4e308c9c64a9a8c7e0d3dc207372b0e0429c4c2abd72a8a7f303",
    ),
    "s006-manifest": (
        "S-20260814-006",
        "benchmark/artifacts/bright-learned-sparse-batch-a-v0.1/manifest.json",
        "068dd07e3b080bd6f39ece28793e225893732b92907f1e65ca1e69e9f69df362",
    ),
    "s006-summary": (
        "S-20260814-006",
        "benchmark/artifacts/bright-learned-sparse-batch-a-v0.1/summary.json",
        "904000f96b4290904b014bdd50043f8651bb4b42c19cc5b88fa71874ef991716",
    ),
    "s006-report": (
        "S-20260814-006",
        "benchmark/research/learned_sparse_batch_a_20260828.md",
        "ad233f499322f3e5a55d1cf0453983cbe0969615fe35cea262b2df0af55a4734",
    ),
    "s006-manifest-schema": (
        "S-20260814-006",
        "schemas/bright-learned-sparse-artifact-manifest-v01.schema.json",
        "d08a0be1ff86379d0b2359938f54a2cb70a62576421db449e7589f0c377b22c6",
    ),
    "s006-summary-schema": (
        "S-20260814-006",
        "schemas/bright-learned-sparse-summary-v01.schema.json",
        "84338230341dfecf629e38eac26ede87532f4e218f9753859aa4c5df269cd73e",
    ),
    "s006-inventory-schema": (
        "S-20260814-006",
        "schemas/learned-sparse-inventory-v01.schema.json",
        "941610063e60ecff5d0ac687a7eae60f7f5e7e03319f89f51ae571f5b36d3d18",
    ),
    "s008-predeclaration": (
        "S-20260814-008",
        "benchmark/artifacts/bright-learned-sparse-batch-b-v0.1/predeclaration.json",
        "2d30ac69b58e7bbd217550418619bab085e2c8b73e1453ee02e0d237827a32ef",
    ),
    "s008-input-contract": (
        "S-20260814-008",
        "benchmark/contracts/bright-learned-sparse-main-v0.1-inputs.json",
        "18ec21848358fd19a454ed2f7362e1eb6117d6c508ca6f30cc9980d2642d1dc6",
    ),
    "s008-cases": (
        "S-20260814-008",
        "benchmark/artifacts/bright-learned-sparse-main-v0.1/cases.json",
        "28da5092c727d62502159972a0bb660fc0807fe642e051c2aa935491f353f46b",
    ),
    "s008-manifest": (
        "S-20260814-008",
        "benchmark/artifacts/bright-learned-sparse-main-v0.1/manifest.json",
        "3accbed7b2a9961eedff20e601375a23c50167cfc7d8e77adbf91399aed91f47",
    ),
    "s008-pairwise": (
        "S-20260814-008",
        "benchmark/artifacts/bright-learned-sparse-main-v0.1/pairwise.json",
        "f2933b1d4979e8ea37c13b3f68ec3c3f21ba985cd4b8d6bef9bf6ebe736fe74d",
    ),
    "s008-resources": (
        "S-20260814-008",
        "benchmark/artifacts/bright-learned-sparse-main-v0.1/resources.json",
        "ad0e7b4d2a0ba9bc91899fc91aa181b412eafebf56cb54b84cfeb3d889c9964f",
    ),
    "s008-slices": (
        "S-20260814-008",
        "benchmark/artifacts/bright-learned-sparse-main-v0.1/slices.json",
        "d0ce5981dea7ba5c510dc9c68f54356cec0a37b050d559429ac0fa2f2cac410b",
    ),
    "s008-summary": (
        "S-20260814-008",
        "benchmark/artifacts/bright-learned-sparse-main-v0.1/summary.json",
        "80e57e7648047753d21639660d89aa3676c084f5c8107f1a538a4cfbb4598a79",
    ),
    "s008-report": (
        "S-20260814-008",
        "benchmark/research/learned_sparse_main_20260829.md",
        "3388f028445fdaa2e989f7753e2614af4fe7cedaaf2842e995294a1640339389",
    ),
    "s008-input-schema": (
        "S-20260814-008",
        "schemas/bright-learned-sparse-main-inputs-v01.schema.json",
        "7722c17d0520c74f05653a4b844de5813c8cb211f30c81d2bff0677f0c5be3e9",
    ),
    "s008-manifest-schema": (
        "S-20260814-008",
        "schemas/bright-learned-sparse-main-manifest-v01.schema.json",
        "926316c1a4ec8d3dacaa315b1f15378940ed4aa60b30b7b8505ce922df2e98e1",
    ),
    "s008-pairwise-schema": (
        "S-20260814-008",
        "schemas/bright-learned-sparse-main-pairwise-v01.schema.json",
        "662c1babb578ca14c544139471d7075dcad4a0225060a5d99169ec8212225bfc",
    ),
    "s008-resources-schema": (
        "S-20260814-008",
        "schemas/bright-learned-sparse-main-resources-v01.schema.json",
        "1c5f0b98bd143d276235e3d807c169a9761bf4f1bb001ff46f3449c8cadeef0f",
    ),
    "s008-slices-schema": (
        "S-20260814-008",
        "schemas/bright-learned-sparse-main-slices-v01.schema.json",
        "78de44bdef250efc166023818541d98194fea618811f79c22dbe0da13786dfdf",
    ),
    "s008-summary-schema": (
        "S-20260814-008",
        "schemas/bright-learned-sparse-main-summary-v01.schema.json",
        "0536958ecb55de28e1de38a31e227d82b3bad29a90276389ada6d0cab9762865",
    ),
    "s008-cases-schema": (
        "S-20260814-008",
        "schemas/bright-learned-sparse-main-cases-v01.schema.json",
        "06131b06668b66b80b7ffa2cde0d9bcc258394b9bd56f2113c91b0c33ada19ed",
    ),
    "s010-predeclaration": (
        "S-20260814-010",
        "benchmark/artifacts/milvus-sindi-system-v0.1/predeclaration.json",
        "d1c3e3c9c2ad1e7e4b79bf37b0088e480058196a570f56d21ccc1a4bd2ff47d1",
    ),
    "s010-source-attestation": (
        "S-20260814-010",
        "benchmark/artifacts/milvus-sindi-system-v0.1/source-attestation.json",
        "dce7e8c088954bfc5d689cea1017f13b088dc417532d5b5a9a312abe6cf46f65",
    ),
    "s010-summary": (
        "S-20260814-010",
        "benchmark/artifacts/milvus-sindi-system-v0.1/formal-summary.json",
        "d2735fec05c7d49bfcdc767bad37e59d2a00fae2aafa885d061bb82139c761d3",
    ),
    "s010-report": (
        "S-20260814-010",
        "benchmark/research/milvus_sindi_system_report_20260904.md",
        "e4d6be9987bc0ede99e6cd81b645b38080135b1410bb380aff85b8ba3ca28e03",
    ),
    "s010-predeclaration-schema": (
        "S-20260814-010",
        "schemas/milvus-sindi-system-predeclaration-v01.schema.json",
        "e28b7b239d489b25e584c7b52380c8e1b90c37264086b918d871559a07e57cb0",
    ),
    "s010-source-schema": (
        "S-20260814-010",
        "schemas/milvus-sindi-system-source-attestation-v01.schema.json",
        "f5cadf99d35ce1eafe8da4ed8ac9e85fb53976136eb5599e267e1ee5a91996bb",
    ),
    "s010-summary-schema": (
        "S-20260814-010",
        "schemas/milvus-sindi-formal-summary-v01.schema.json",
        "c3900de5294cd2a67e68839f4d3fd53fd630926edc55bd83cc477d2b7f081e2f",
    ),
}

PRIVATE_EVIDENCE = {
    "s005-materialization-manifest": {
        "story_id": "S-20260814-005",
        "sha256": "8174b0c01a32e977cf0c5d89522356c485196b0aa0571251cfe1595c5579cb1f",
    },
    "s005-review-plan": {
        "story_id": "S-20260814-005",
        "sha256": "93848a826c12acf9cb6fed50d71dd9490be7baebfcea1122dd4fffb4881fdc4e",
    },
    "s005-audit-pack": {
        "story_id": "S-20260814-005",
        "sha256": "9549525cbd35e0a3c520ad16b5961b3f5e8658ff7cb4b5061e9edd39bc96c956",
    },
    "s008-raw-manifest-set": {
        "story_id": "S-20260814-008",
        "sha256": "ac91bc9a62775eace06350ebbd31bd0785210efef0325aded70b90451833043d",
    },
    "s008-private-package-replay": {
        "story_id": "S-20260814-008",
        "sha256": "9e1eb574351d2fe63b0be775ffbc31ed57658b653bbce21589c2d0668c2c4ae2",
    },
    "s010-native-run-manifest": {
        "story_id": "S-20260814-010",
        "sha256": "a6982b48d6be8651f8060a8667a1a0c4aec7487b893032a464b198f77b179ab2",
    },
    "s010-100k-run-manifest": {
        "story_id": "S-20260814-010",
        "sha256": "29418222dddc6fec61c0609338f142a864fe700d8ce3284bdbec3f3bf94f617a",
    },
    "s010-1m-run-manifest": {
        "story_id": "S-20260814-010",
        "sha256": "87c152bcf45780ec2d97e751684efef083e029edf9b54f0b1dee6d5f8fa9e9ac",
    },
    "s010-formal-file-manifest": {
        "story_id": "S-20260814-010",
        "sha256": "63c44107e4c07b80ef089a9a527a6a5c11e3305d3b8ef2cec42e895e6c5f8726",
    },
    "s010-csr-exact-catalog": {
        "story_id": "S-20260814-010",
        "sha256": "bb16544edd6469fcb6a4d8fdc54298b61a5209d9754a6cd4c3a3d45b00437840",
    },
}

DERIVATION_PROTOCOL = {
    "accepted_inputs_only": True,
    "model_loaded": False,
    "milvus_connected": False,
    "track_aggregation": "none",
    "uncertainty_rule": "support a pairwise direction only when the paired 95% interval excludes zero",
    "small_slice_rule": "n<10 is descriptive only",
    "anchor_rule": "BM25 and dense are contextual anchors only after shared data, candidate, qrel, and metric identity checks",
    "system_only_rule": "system-only workloads support system scaling and cost claims, never quality claims",
}


def json_bytes(value: Any) -> bytes:
    """Return stable, human-readable JSON bytes."""
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise PhaseOneError(f"invalid JSON input: {path}") from error


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo_root, check=False, text=True, capture_output=True)
    if result.returncode:
        raise PhaseOneError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def authenticate_inputs(repo_root: Path) -> dict[str, Any]:
    """Verify accepted commits, trees, ancestry, tracked state, and file bytes."""
    accepted = {}
    for story_id, identity in ACCEPTED_COMMITS.items():
        commit = identity["commit"]
        tree = _git(repo_root, "show", "-s", "--format=%T", commit)
        if tree != identity["tree"]:
            raise PhaseOneError(f"accepted tree mismatch for {story_id}")
        result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=repo_root, check=False)
        if result.returncode:
            raise PhaseOneError(f"accepted commit is not an ancestor of HEAD: {story_id}")
        accepted[story_id] = {**identity, "ancestor_of_head": True}

    files = {}
    for artifact_id, (story_id, relative_path, expected_sha256) in ACCEPTED_FILES.items():
        path = repo_root / relative_path
        if not path.is_file():
            raise PhaseOneError(f"accepted input is missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise PhaseOneError(f"accepted input hash mismatch: {relative_path}")
        if _git(repo_root, "ls-files", "--error-unmatch", relative_path) != relative_path:
            raise PhaseOneError(f"accepted input is not tracked: {relative_path}")
        files[artifact_id] = {
            "story_id": story_id,
            "path": relative_path,
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return {"commits": accepted, "files": files}


def _validate_source_schemas(repo_root: Path, values: Mapping[str, Any]) -> None:
    bindings = (
        ("s006-manifest-schema", values["batch_a_manifest"]),
        ("s008-input-schema", values["s008_input"]),
        ("s008-manifest-schema", values["s008_manifest"]),
        ("s008-pairwise-schema", values["pairwise"]),
        ("s008-resources-schema", values["resources"]),
        ("s008-slices-schema", values["slices"]),
        ("s008-summary-schema", values["main_summary"]),
        ("s008-cases-schema", values["cases"]),
        ("s010-predeclaration-schema", values["s010_predeclaration"]),
        ("s010-source-schema", values["source_attestation"]),
        ("s010-summary-schema", values["system_summary"]),
    )
    for schema_id, value in bindings:
        schema_path = repo_root / ACCEPTED_FILES[schema_id][1]
        schema = _load_json(schema_path)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)


def load_inputs(repo_root: Path) -> dict[str, Any]:
    """Authenticate and load the accepted aggregate inputs."""
    authentication = authenticate_inputs(repo_root)
    values = {
        "pilot": _load_json(repo_root / ACCEPTED_FILES["s005-summary"][1]),
        "inventory": _load_json(repo_root / ACCEPTED_FILES["s006-inventory"][1]),
        "batch_a_manifest": _load_json(repo_root / ACCEPTED_FILES["s006-manifest"][1]),
        "s008_input": _load_json(repo_root / ACCEPTED_FILES["s008-input-contract"][1]),
        "s008_manifest": _load_json(repo_root / ACCEPTED_FILES["s008-manifest"][1]),
        "main_summary": _load_json(repo_root / ACCEPTED_FILES["s008-summary"][1]),
        "pairwise": _load_json(repo_root / ACCEPTED_FILES["s008-pairwise"][1]),
        "resources": _load_json(repo_root / ACCEPTED_FILES["s008-resources"][1]),
        "slices": _load_json(repo_root / ACCEPTED_FILES["s008-slices"][1]),
        "cases": _load_json(repo_root / ACCEPTED_FILES["s008-cases"][1]),
        "s010_predeclaration": _load_json(repo_root / ACCEPTED_FILES["s010-predeclaration"][1]),
        "source_attestation": _load_json(repo_root / ACCEPTED_FILES["s010-source-attestation"][1]),
        "system_summary": _load_json(repo_root / ACCEPTED_FILES["s010-summary"][1]),
        "system_report": (repo_root / ACCEPTED_FILES["s010-report"][1]).read_text(encoding="utf-8"),
        "authentication": authentication,
    }
    _validate_source_schemas(repo_root, values)
    _validate_cross_source_identity(values)
    _validate_milestone_counts(values)
    return values


def _validate_cross_source_identity(values: Mapping[str, Any]) -> None:
    pilot = values["pilot"]
    manifest = values["s008_manifest"]
    source = values["source_attestation"]
    system = values["system_summary"]
    canonical = pilot["artifact_identities"]["materialization_manifest_sha256"]
    if canonical != manifest["bindings"]["canonical_data_sha256"]:
        raise PhaseOneError("S-005 and S-008 canonical data identities do not match")
    if source["main_manifest_sha256"] != ACCEPTED_FILES["s008-manifest"][2]:
        raise PhaseOneError("S-010 source attestation does not bind the accepted S-008 manifest")
    if source["commit"] != ACCEPTED_COMMITS["S-20260814-008"]["commit"]:
        raise PhaseOneError("S-010 source commit does not bind accepted S-008")
    if system["source_attestation_sha256"] != ACCEPTED_FILES["s010-source-attestation"][2]:
        raise PhaseOneError("S-010 aggregate does not bind its accepted source attestation")
    for value in (values["main_summary"], values["pairwise"], values["resources"], values["slices"], values["cases"]):
        if value["publication"] != PUBLICATION | {
            "contains_source_text": False,
            "contains_canonical_ids": False,
            "contains_raw_rankings": False,
            "contains_private_paths": False,
        }:
            # Accepted packages use a smaller publication object; validate only the shared closed flags below.
            publication = value["publication"]
            if publication.get("classification") != "research_only" or publication.get("gate") != "closed":
                raise PhaseOneError("accepted learned-sparse package publication gate is not closed")
    if system["model_loaded"] is not False or system["publication_gate"] != "closed":
        raise PhaseOneError("accepted system aggregate violates the no-model or publication boundary")


def _validate_milestone_counts(values: Mapping[str, Any]) -> None:
    pilot = values["pilot"]
    main = values["main_summary"]
    pairwise = values["pairwise"]
    resources = values["resources"]
    slices = values["slices"]
    cases = values["cases"]
    system = values["system_summary"]

    if {key: pilot["scope"][key] for key in ("queries", "documents", "positive_qrels")} != {
        "queries": 204,
        "documents": 15_000,
        "positive_qrels": 1_492,
    }:
        raise PhaseOneError("S-005 aggregate counts do not match the accepted milestone")

    model_sets = [{row["model_key"] for row in main["tracks"][track]["rows"]} for track in TRACKS]
    resource_models = {row["model_key"] for row in resources["models"]}
    if any(len(models) != 7 for models in model_sets) or len(resource_models) != 7:
        raise PhaseOneError("S-006/S-008 model coverage does not match the accepted 7 x 2 matrix")
    if not (model_sets[0] == model_sets[1] == resource_models):
        raise PhaseOneError("S-006/S-008 model identities differ across tracks or resources")
    for model in resources["models"]:
        track_rows = {row["track"]: row for row in model["tracks"]}
        if set(track_rows) != set(TRACKS):
            raise PhaseOneError(f"S-008 resource track coverage mismatch for {model['model_key']}")
        for track, row in track_rows.items():
            if (
                row["document_nnz"]["count"] != pilot["tracks"][track]["documents"]
                or row["query_nnz"]["count"] != pilot["tracks"][track]["queries"]
            ):
                raise PhaseOneError(f"S-008 retrieval-unit count mismatch for {model['model_key']}:{track}")
    for track in TRACKS:
        rows = main["tracks"][track]["rows"]
        if any(set(row["metrics"]) != set(METRICS) for row in rows):
            raise PhaseOneError(f"S-008 metric coverage mismatch for {track}")
        pairs = pairwise["tracks"][track]
        if len(pairs) != 21 or any(set(row["metrics"]) != set(METRICS) for row in pairs):
            raise PhaseOneError(f"S-008 pairwise coverage mismatch for {track}")
        if any(metric["samples"] != 10_000 for row in pairs for metric in row["metrics"].values()):
            raise PhaseOneError(f"S-008 bootstrap sample count mismatch for {track}")
        dimensions = slices["tracks"][track]
        if len(dimensions) != 4 or sum(len(dimension["bins"]) for dimension in dimensions) != 13:
            raise PhaseOneError(f"S-008 slice coverage mismatch for {track}")
    if cases["counts"] != {"per_track_disagreement": 6, "per_track_failure": 6, "total": 24}:
        raise PhaseOneError("S-008 opaque-case coverage mismatch")

    coverage = system["coverage"]
    expected_coverage = {
        "native_cells": 18,
        "system_100k_cells": 9,
        "system_1m_cells": 9,
        "raw_trial_files_recomputed": 1_296,
        "warmup_files_validated": 432,
    }
    if any(coverage[key] != value for key, value in expected_coverage.items()) or len(system["costs"]) != 36:
        raise PhaseOneError("S-010 cell, trial, warmup, or cost coverage mismatch")


def _pairwise_summary(pairwise: Mapping[str, Any]) -> dict[str, Any]:
    tracks = {}
    for track in TRACKS:
        verdicts = Counter()
        significant = []
        for pair in pairwise["tracks"][track]:
            for metric, interval in pair["metrics"].items():
                verdicts[interval["verdict"]] += 1
                if interval["verdict"] != "tie_or_uncertain":
                    winner = pair["left"] if interval["verdict"] == "left_wins" else pair["right"]
                    loser = pair["right"] if interval["verdict"] == "left_wins" else pair["left"]
                    significant.append(
                        {
                            "metric": metric,
                            "winner": winner,
                            "loser": loser,
                            "mean_delta": abs(interval["mean"]),
                            "ci_low": min(abs(interval["low"]), abs(interval["high"])),
                            "ci_high": max(abs(interval["low"]), abs(interval["high"])),
                            "samples": interval["samples"],
                            "n": pair["n"],
                        }
                    )
        tracks[track] = {
            "pairs": len(pairwise["tracks"][track]),
            "metric_comparisons": len(pairwise["tracks"][track]) * len(METRICS),
            "bootstrap_samples_each": 10_000,
            "verdict_counts": dict(sorted(verdicts.items())),
            "significant": sorted(significant, key=lambda row: (row["metric"], row["winner"], row["loser"])),
        }
    return {"method": pairwise["method"], "tracks": tracks}


def _quality_summary(main: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for track in TRACKS:
        rows = main["tracks"][track]["rows"]
        output[track] = {
            "rows": sorted(rows, key=lambda row: row["model_key"]),
            "point_order_ndcg_at_10": list(main["tracks"][track]["primary_order"]),
        }
    return output


def _resource_summary(resources: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in resources["models"]:
        tracks = {}
        for track in model["tracks"]:
            tracks[track["track"]] = {
                "document_nnz_mean": track["document_nnz"]["mean"],
                "query_nnz_mean": track["query_nnz"]["mean"],
                "document_throughput_items_per_second": track["encoding"]["document_throughput_items_per_second"],
                "query_throughput_items_per_second": track["encoding"]["query_throughput_items_per_second"],
                "exact_search_seconds": track["exact_search_seconds"],
                "document_csr_bytes": track["representation"]["document_csr_bytes"],
                "query_csr_bytes": track["representation"]["query_csr_bytes"],
                "document_peak_vram_bytes": track["encoding"]["document_peak_vram_bytes"],
                "observed_peak_rss_bytes": max(
                    track["encoding"]["document_observed_rss_bytes"],
                    track["encoding"]["query_observed_rss_bytes"],
                ),
                "rss_scope": track["encoding"]["rss_scope"],
            }
        rows.append(
            {
                key: model[key]
                for key in (
                    "model_key",
                    "mechanism",
                    "query_route",
                    "document_route",
                    "dimensions",
                    "revision",
                    "license_boundary",
                    "parameter_count",
                    "snapshot_bytes",
                    "snapshot_bytes_kind",
                )
            }
            | {"tracks": tracks}
        )
    return sorted(rows, key=lambda row: row["model_key"])


def _slice_summary(slices: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for track in TRACKS:
        dimensions = []
        for dimension in slices["tracks"][track]:
            bins = [
                {
                    "label": row["bin"],
                    "n": row["n"],
                    "inference": row["inference"],
                    "small_sample": 0 < row["n"] < 10,
                    "empty": row["n"] == 0,
                }
                for row in dimension["bins"]
            ]
            dimensions.append({"dimension": dimension["key"], "bins": bins})
        output[track] = dimensions
    return output


def _anchor_summary(
    pilot: Mapping[str, Any],
    main: Mapping[str, Any],
    batch_a_manifest: Mapping[str, Any],
    manifest: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    pilot_metric_sets = [
        set(pilot["tracks"][track][method]["metrics"]) for track in TRACKS for method in ("bm25", "long_dense")
    ]
    canonical_sha256 = pilot["artifact_identities"]["materialization_manifest_sha256"]
    source_revisions = {
        pilot["tracks"][track]["governance_audit"]["provenance_and_time"]["revision"] for track in TRACKS
    }
    selection_hashes = {pilot["tracks"][track]["selection_ids_sha256"] for track in TRACKS}
    fairness = {
        "canonical_data_identity_equal": canonical_sha256
        == batch_a_manifest["protocol"]["materialization_manifest_sha256"]
        == manifest["bindings"]["canonical_data_sha256"],
        "source_revision_equal": source_revisions == {batch_a_manifest["protocol"]["source_revision"]},
        "tracks_equal": tuple(pilot["scope"]["tracks"]) == TRACKS == tuple(sorted(main["tracks"])),
        "candidate_pools_are_track_specific": pilot["scope"]["independent_candidate_pools"] is True,
        "same_retrieval_unit_chunking_and_selection": len(selection_hashes) == len(TRACKS)
        and all(len(value) == 64 for value in selection_hashes)
        and all(
            track_row["document_nnz"]["count"] == pilot["tracks"][track_row["track"]]["documents"]
            and track_row["query_nnz"]["count"] == pilot["tracks"][track_row["track"]]["queries"]
            for model in resources["models"]
            for track_row in model["tracks"]
        ),
        "same_quality_metrics": all(metric_set == set(METRICS) for metric_set in pilot_metric_sets)
        and manifest["protocol"]["metrics"] == list(METRICS)
        and all(set(row["metrics"]) == set(METRICS) for track in TRACKS for row in main["tracks"][track]["rows"]),
        "same_top_100_depth": batch_a_manifest["protocol"]["search"]["top_k"] == 100
        and all(metric.endswith(("@10", "@100")) for metric in METRICS),
        "same_grade_1_qrels_and_candidate_filtering": all(
            pilot["tracks"][track]["governance_audit"]["candidate_policy"]["candidates_per_query"]
            == pilot["tracks"][track]["documents"]
            and pilot["tracks"][track]["governance_audit"]["grade_counts"]
            == {"1": pilot["tracks"][track]["positive_qrels"]}
            and pilot["tracks"][track]["governance_audit"]["candidate_policy"]["excluded_ids"]
            == "filter before ranking and metrics; never negative qrels"
            for track in TRACKS
        ),
    }
    if not all(fairness.values()):
        raise PhaseOneError("BM25/dense anchor fairness identity is incomplete")
    tracks = {}
    for track in TRACKS:
        tracks[track] = {
            "bm25": pilot["tracks"][track]["bm25"]["metrics"],
            "bge_m3_long_dense": pilot["tracks"][track]["long_dense"]["metrics"],
            "bge_m3_dense_minus_bm25_paired": pilot["tracks"][track]["long_dense_minus_bm25_paired_bootstrap"],
        }
    return {
        "role": "limited_contextual_anchor",
        "fairness_checks": fairness,
        "tracks": tracks,
        "limitations": [
            "This is not the multi-baseline cross-paradigm study reserved for S-20260814-009.",
            "The dense BGE-M3 route uses a predeclared 1,024-token cap while learned-sparse cells use 512 tokens.",
            "Training overlap remains unknown, so the anchor supports no verified zero-shot claim.",
        ],
    }


def _system_summary(system: Mapping[str, Any], accepted_report: str) -> dict[str, Any]:
    costs = [row for row in system["costs"] if row["comparator"] in {"SINDI", "DAAT_MAXSCORE"}]
    paired_costs = {}
    for row in costs:
        key = (row["tier"], row["model_profile"], row["track"], row["scale"])
        paired_costs.setdefault(key, {})[row["comparator"]] = row["persisted_index_bytes"]
    ratios = []
    for key, values in sorted(paired_costs.items(), key=str):
        if set(values) != {"SINDI", "DAAT_MAXSCORE"}:
            raise PhaseOneError(f"unpaired system cost row: {key}")
        ratios.append(values["SINDI"] / values["DAAT_MAXSCORE"])
    if len(ratios) != 12 or not all(value > 1 for value in ratios):
        raise PhaseOneError("accepted system cost matrix does not contain 12 larger SINDI indexes")
    native = next(row for row in system["strata"] if row["name"] == "native")
    tie_match = re.search(
        r"minimum strict recall was ([0-9.]+) and tie-aware Recall@K was ([0-9.]+).*?deltas were at most "
        r"([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)",
        accepted_report,
    )
    if tie_match is None:
        raise PhaseOneError("accepted S-010 report no longer exposes the native fp16-aware tie result")
    persisted_minimum_strict, tie_recall, maximum_score_delta = map(float, tie_match.groups())
    if persisted_minimum_strict != 0.9 or tie_recall != 1.0:
        raise PhaseOneError("accepted native fp16-aware correctness result changed")
    return {
        "deployment": system["deployment"],
        "coverage": system["coverage"],
        "correctness": {
            "native_minimum_strict_recall": min(
                native["minimum_sindi_strict_recall"], native["minimum_daat_maxscore_strict_recall"]
            ),
            "native_persisted_fp16_minimum_strict_recall": persisted_minimum_strict,
            "native_fp16_aware_tie_recall": tie_recall,
            "native_fp16_aware_maximum_score_delta": maximum_score_delta,
            "all_tiers_minimum_strict_recall": min(
                system["aggregate"]["minimum_sindi_strict_recall"],
                system["aggregate"]["minimum_daat_maxscore_strict_recall"],
            ),
            "system_only_near_tie_is_quality_evidence": False,
        },
        "qps": {
            "sindi_over_daat_median": system["aggregate"]["median_sindi_over_daat_qps_ratio"],
            "sindi_over_daat_min": system["aggregate"]["minimum_sindi_over_daat_qps_ratio"],
            "sindi_over_daat_max": system["aggregate"]["maximum_sindi_over_daat_qps_ratio"],
            "crosses_one": system["aggregate"]["minimum_sindi_over_daat_qps_ratio"]
            < 1
            < system["aggregate"]["maximum_sindi_over_daat_qps_ratio"],
            "strata": [
                {
                    "name": row["name"],
                    "median_sindi_over_daat": row["median_sindi_over_daat_qps_ratio"],
                    "minimum_sindi_over_daat": row["minimum_sindi_over_daat_qps_ratio"],
                    "maximum_sindi_over_daat": row["maximum_sindi_over_daat_qps_ratio"],
                    "median_sindi_warm_over_cold": row["median_sindi_warm_over_cold_qps_ratio"],
                    "median_daat_warm_over_cold": row["median_daat_warm_over_cold_qps_ratio"],
                }
                for row in system["strata"]
            ],
        },
        "persisted_index": {
            "pairs": len(ratios),
            "sindi_larger_pairs": sum(value > 1 for value in ratios),
            "sindi_over_daat_ratio_min": min(ratios),
            "sindi_over_daat_ratio_max": max(ratios),
        },
        "boundaries": [
            "Native results cover three predeclared model profiles on economics and psychology; they do not cover all seven models.",
            "The 100k and 1M workloads are deterministic system-only expansions and cannot support model-quality claims.",
            "Cold means collection release and reload; host page caches were not dropped.",
            "The result is specific to the pinned Milvus 3.0.0 build, vector index version 10, hardware allocation, and tested matrix.",
            "A pass status means evidence and execution checks passed, not that an algorithm or model was good.",
        ],
    }


def _recommendations() -> list[dict[str, Any]]:
    return [
        {
            "id": "REC-QUALITY",
            "scenario": "Quality-first selection",
            "guidance": "Shortlist by track and target metric, then require a paired interval that excludes zero before claiming an advantage. Economics and psychology point orders differ, so do not collapse them.",
            "support_claims": ["C-QUALITY", "C-UNCERTAINTY"],
            "limitations": ["No universal model winner", "Training overlap is unknown", "Only two English tracks"],
        },
        {
            "id": "REC-LIGHTWEIGHT",
            "scenario": "Lightweight or encoding-throughput priority",
            "guidance": "Use snapshot size, document/query throughput, CSR bytes, and paired quality together. SPLADE-Tiny is the smallest and fastest-document option measured here, but its standalone license evidence is incomplete and its quality trade-offs are track/metric specific.",
            "support_claims": ["C-RESOURCES", "C-LICENSE"],
            "limitations": ["Encoding-observed RSS is not whole-cell peak", "Local research-only use for SPLADE-Tiny"],
        },
        {
            "id": "REC-STATIC",
            "scenario": "Static-query lookup with document expansion",
            "guidance": "Compare the three OpenSearch static-query routes when inexpensive query-side execution matters, but budget document-side expansion and index density separately. The English measurements do not establish multilingual behavior.",
            "support_claims": ["C-ROUTING", "C-RESOURCES"],
            "limitations": ["No production SLA", "No multilingual effectiveness evidence"],
        },
        {
            "id": "REC-MULTILINGUAL",
            "scenario": "Multilingual candidate selection",
            "guidance": "Keep OpenSearch multilingual as a candidate because its architecture is multilingual; treat every effectiveness number here as English-only evidence and schedule a multilingual track before deployment claims.",
            "support_claims": ["C-MULTILINGUAL-GAP"],
            "limitations": ["No multilingual track", "No verified zero-shot evidence"],
        },
        {
            "id": "REC-EXACT",
            "scenario": "Exact retrieval and quality evaluation",
            "guidance": "Keep SciPy CSR exact inner product as the quality ground truth and simplest 7,500-document maintenance path. Its system cost is a CPU implementation anchor, not a universal production baseline.",
            "support_claims": ["C-EXACT", "C-SYSTEM"],
            "limitations": ["Small native corpus", "No production availability or network layer"],
        },
        {
            "id": "REC-MILVUS",
            "scenario": "Milvus deployment choice",
            "guidance": "Benchmark SINDI and DAAT_MAXSCORE on the intended workload rather than selecting from the algorithm name. The accepted matrix crosses over by workload/configuration, and SINDI serialized indexes were larger in every paired cell.",
            "support_claims": ["C-SYSTEM", "C-INDEX-BYTES"],
            "limitations": [
                "Pinned Milvus 3.0.0 build only",
                "System-only scaling is not quality evidence",
                "No production SLA",
            ],
        },
    ]


def build_summary(values: Mapping[str, Any]) -> dict[str, Any]:
    pilot = values["pilot"]
    main = values["main_summary"]
    result = {
        "schema_version": "learned-sparse-phase-one-summary-v1",
        "phase_id": PHASE_ID,
        "story_id": STORY_ID,
        "publication": PUBLICATION,
        "data": {
            "source": pilot["source"],
            "selection": pilot["selection"],
            "scope": pilot["scope"],
            "tracks": [
                {
                    "track": track,
                    "queries": pilot["tracks"][track]["queries"],
                    "documents": pilot["tracks"][track]["documents"],
                    "positive_qrels": pilot["tracks"][track]["positive_qrels"],
                    "qrel_density": pilot["tracks"][track]["governance_audit"]["qrel_density"],
                    "single_positive_queries": pilot["tracks"][track]["governance_audit"]["single_positive_queries"],
                }
                for track in TRACKS
            ],
            "governance": {
                "research_only": True,
                "public_export_allowed": False,
                "training_overlap": "unknown_not_zero_shot_verified",
                "manual_review_plan_queries": pilot["review_plan"]["rows"],
                "review_completed": False,
                "unjudged_are_negatives": False,
            },
        },
        "quality": {
            "metrics": list(METRICS),
            "models": 7,
            "tracks": _quality_summary(main),
            "paired_uncertainty": _pairwise_summary(values["pairwise"]),
            "track_aggregation": "none",
        },
        "anchors": _anchor_summary(
            pilot,
            main,
            values["batch_a_manifest"],
            values["s008_manifest"],
            values["resources"],
        ),
        "models": _resource_summary(values["resources"]),
        "slices": _slice_summary(values["slices"]),
        "cases": {"counts": values["cases"]["counts"], "opaque_only": True},
        "system": _system_summary(values["system_summary"], values["system_report"]),
        "recommendations": _recommendations(),
        "remaining_gaps": [
            "No complete cross-paradigm comparison with multiple representative dense, BM25, and multi-vector baselines.",
            "No multilingual effectiveness or verified zero-shot evaluation.",
            "No production SLA, failure-domain, cost-of-operations, or current-version Milvus validation.",
            "No document-level upstream redistribution-rights clearance; publication gate remains closed.",
            "The fixed pilot covers 204 queries and 15,000 passages in two English tracks, below the longer-term 3-5-domain target.",
            "The planned 120-query manual audit has not been represented as completed evidence.",
        ],
        "derivation": {
            "protocol_sha256": sha256_bytes(json_bytes(DERIVATION_PROTOCOL)),
            "accepted_inputs_only": True,
            "model_loaded": False,
            "milvus_connected": False,
        },
    }
    return result


def build_claim_matrix() -> dict[str, Any]:
    claims = [
        (
            "C-DATA",
            "supported",
            "The fixed research pilot contains 204 queries, 15,000 passages, and 1,492 grade-1 qrels across independent economics and psychology tracks.",
            ["s005-summary"],
            "Fixed v0.2 pilot only",
        ),
        (
            "C-QUALITY",
            "supported",
            "Seven learned-sparse models have per-track results for five quality metrics on the shared pilot.",
            ["s008-summary"],
            "Same-paradigm, two-track comparison",
        ),
        (
            "C-UNCERTAINTY",
            "supported",
            "Pairwise statements require query-aligned 10,000-sample intervals that exclude zero; intervals crossing zero remain uncertain.",
            ["s008-pairwise"],
            "Per track and metric only",
        ),
        (
            "C-SLICES",
            "supported",
            "Four predeclared slice dimensions and 13 bins per track are available; bins with n<10 are descriptive only.",
            ["s008-slices", "s008-predeclaration"],
            "No post-hoc semantic categories",
        ),
        (
            "C-CASES",
            "supported",
            "The tracked package contains 24 opaque disagreement/failure cases without canonical identifiers or source text.",
            ["s008-cases"],
            "Mechanism-level triage only",
        ),
        (
            "C-ROUTING",
            "supported",
            "The portfolio covers neural/neural and static-query/document-expansion routes with distinct sparsity and encoding profiles.",
            ["s008-resources", "s006-inventory"],
            "Measured at the frozen revisions and protocol",
        ),
        (
            "C-RESOURCES",
            "supported",
            "Snapshot, nnz, encoding throughput, CSR bytes, exact-search time, VRAM, and encoding-observed RSS can inform bounded maintenance choices.",
            ["s008-resources"],
            "RSS is not a whole-cell peak; no production SLA",
        ),
        (
            "C-EXACT",
            "supported",
            "SciPy CSR exact sparse inner product is the quality ground truth for the fixed pilot.",
            ["s006-manifest", "s008-manifest", "s010-source-attestation"],
            "Small native corpus and CPU implementation",
        ),
        (
            "C-SYSTEM",
            "supported",
            "On the pinned Milvus 3.0.0 matrix, SINDI and DAAT_MAXSCORE exhibit workload/configuration-dependent QPS crossovers.",
            ["s010-summary"],
            "Pinned build, hardware, and matrix only",
        ),
        (
            "C-INDEX-BYTES",
            "supported",
            "SINDI serialized indexes are larger in all 12 paired accepted system cells.",
            ["s010-summary"],
            "Serialized index bytes, not total cost",
        ),
        (
            "C-LICENSE",
            "supported",
            "SPLADE-Tiny remains local research-only because standalone license evidence is incomplete; the dataset publication gate is closed.",
            ["s008-resources", "s005-summary"],
            "No weight or dataset redistribution authorization",
        ),
        (
            "C-ANCHORS",
            "limited_anchor",
            "BM25 and BGE-M3 long-dense results provide contextual anchors on the same raw pilot, candidates, qrels, and metrics.",
            ["s005-summary", "s006-manifest", "s008-manifest"],
            "Not the complete S-009 cross-paradigm comparison; dense uses a different token cap",
        ),
        (
            "C-MULTILINGUAL-GAP",
            "unsupported",
            "The measured results establish multilingual effectiveness or verified zero-shot behavior.",
            ["s008-summary"],
            "Both tracks are English and training overlap is unknown",
        ),
        (
            "C-UNIVERSAL-MODEL",
            "unsupported",
            "One model is the universal learned-sparse winner.",
            ["s008-summary", "s008-pairwise"],
            "Track/metric orders differ and many intervals cross zero",
        ),
        (
            "C-UNIVERSAL-SINDI",
            "unsupported",
            "SINDI is universally faster or smaller than DAAT_MAXSCORE.",
            ["s010-summary"],
            "QPS ratios cross one and every measured SINDI index is larger",
        ),
        (
            "C-SYSTEM-QUALITY",
            "unsupported",
            "The 100k/1M generated workloads demonstrate model-quality gains.",
            ["s010-summary"],
            "System-only near-tie workloads are quality-ineligible",
        ),
        (
            "C-PRODUCTION",
            "unsupported",
            "The phase-one results establish a production SLA or deployment recommendation valid across systems.",
            ["s010-summary"],
            "No production environment or SLA study",
        ),
        (
            "C-CROSS-TRACK",
            "unsupported",
            "A cross-track micro-average or total ordering is valid.",
            ["s008-summary"],
            "Independent candidate pools and explicit no-aggregation policy",
        ),
    ]
    return {
        "schema_version": "learned-sparse-claim-support-v1",
        "phase_id": PHASE_ID,
        "publication": PUBLICATION,
        "claims": [
            {"claim_id": claim_id, "support": support, "claim": claim, "evidence_refs": refs, "scope_or_reason": scope}
            for claim_id, support, claim, refs, scope in claims
        ],
    }


def build_evidence_map(authentication: Mapping[str, Any], claim_matrix: Mapping[str, Any]) -> dict[str, Any]:
    files = authentication["files"]
    locators = {
        "C-DATA": [("s005-summary", "/scope"), ("s005-summary", "/tracks")],
        "C-QUALITY": [("s008-summary", "/tracks")],
        "C-UNCERTAINTY": [("s008-pairwise", "/method"), ("s008-pairwise", "/tracks")],
        "C-SLICES": [("s008-slices", "/tracks"), ("s008-predeclaration", "/slices")],
        "C-CASES": [("s008-cases", "/counts")],
        "C-ROUTING": [("s006-inventory", "/models"), ("s008-resources", "/models")],
        "C-RESOURCES": [("s008-resources", "/models")],
        "C-EXACT": [
            ("s006-manifest", "/protocol/search"),
            ("s008-manifest", "/protocol"),
            ("s010-source-attestation", "/csr_exact_catalog_sha256"),
        ],
        "C-SYSTEM": [
            ("s010-summary", "/coverage"),
            ("s010-summary", "/deployment"),
            ("s010-summary", "/aggregate"),
            ("s010-summary", "/strata"),
            ("s010-report", "/lines/26-27"),
        ],
        "C-INDEX-BYTES": [("s010-summary", "/costs")],
        "C-LICENSE": [("s008-resources", "/models"), ("s005-summary", "/publication")],
        "C-ANCHORS": [
            ("s005-summary", "/artifact_identities/materialization_manifest_sha256"),
            ("s005-summary", "/tracks"),
            ("s006-manifest", "/protocol"),
            ("s008-manifest", "/bindings/canonical_data_sha256"),
            ("s008-manifest", "/protocol/metrics"),
        ],
        "C-MULTILINGUAL-GAP": [
            ("s005-summary", "/source"),
            ("s005-summary", "/tracks/*/governance_audit/provenance_and_time/training_overlap_status"),
            ("s008-summary", "/publication"),
        ],
        "C-UNIVERSAL-MODEL": [("s008-summary", "/tracks"), ("s008-pairwise", "/tracks")],
        "C-UNIVERSAL-SINDI": [("s010-summary", "/aggregate"), ("s010-summary", "/costs")],
        "C-SYSTEM-QUALITY": [("s010-summary", "/quality_claims_allowed_for_system_only")],
        "C-PRODUCTION": [("s010-summary", "/deployment"), ("s010-report", "/lines/67-71")],
        "C-CROSS-TRACK": [("s008-summary", "/track_aggregation")],
    }
    claims = []
    by_id = {row["claim_id"]: row for row in claim_matrix["claims"]}
    for claim_id in sorted(by_id):
        claims.append(
            {
                "claim_id": claim_id,
                "support": by_id[claim_id]["support"],
                "sources": [
                    {
                        "artifact_id": artifact_id,
                        "story_id": files[artifact_id]["story_id"],
                        "accepted_commit": ACCEPTED_COMMITS[files[artifact_id]["story_id"]]["commit"],
                        "artifact_sha256": files[artifact_id]["sha256"],
                        "locator": locator,
                    }
                    for artifact_id, locator in locators[claim_id]
                ],
                "scope_or_reason": by_id[claim_id]["scope_or_reason"],
            }
        )
    all_claim_ids = [row["claim_id"] for row in claim_matrix["claims"]]
    all_artifact_refs = sorted({artifact_id for rows in locators.values() for artifact_id, _ in rows})
    table_map = [
        {
            "table_id": "T-DATA",
            "report_section": "Data contract and governance",
            "claim_ids": ["C-DATA"],
            "artifact_refs": ["s005-summary"],
        },
        {
            "table_id": "T-QUALITY-ECONOMICS",
            "report_section": "Economics quality",
            "claim_ids": ["C-QUALITY", "C-UNCERTAINTY"],
            "artifact_refs": ["s008-summary", "s008-pairwise"],
        },
        {
            "table_id": "T-QUALITY-PSYCHOLOGY",
            "report_section": "Psychology quality",
            "claim_ids": ["C-QUALITY", "C-UNCERTAINTY"],
            "artifact_refs": ["s008-summary", "s008-pairwise"],
        },
        {
            "table_id": "T-ANCHORS",
            "report_section": "Limited contextual anchors",
            "claim_ids": ["C-ANCHORS"],
            "artifact_refs": ["s005-summary", "s006-manifest", "s008-manifest"],
        },
        {
            "table_id": "T-MODEL-PORTFOLIO",
            "report_section": "Model routes and resource envelope",
            "claim_ids": ["C-ROUTING", "C-RESOURCES"],
            "artifact_refs": ["s006-inventory", "s008-resources"],
        },
        {
            "table_id": "T-ENCODING-EXACT",
            "report_section": "Encoding, sparsity, and exact retrieval",
            "claim_ids": ["C-RESOURCES", "C-EXACT"],
            "artifact_refs": ["s006-manifest", "s008-resources"],
        },
        {
            "table_id": "T-SLICES",
            "report_section": "Predeclared slices",
            "claim_ids": ["C-SLICES"],
            "artifact_refs": ["s008-slices", "s008-predeclaration"],
        },
        {
            "table_id": "T-SYSTEM-COVERAGE",
            "report_section": "Sparse-system evidence: coverage",
            "claim_ids": ["C-EXACT", "C-SYSTEM", "C-SYSTEM-QUALITY"],
            "artifact_refs": ["s010-summary", "s010-report"],
        },
        {
            "table_id": "T-SYSTEM-RESULTS",
            "report_section": "Sparse-system evidence: QPS and serialized index results",
            "claim_ids": ["C-SYSTEM", "C-INDEX-BYTES", "C-UNIVERSAL-SINDI"],
            "artifact_refs": ["s010-summary"],
        },
        {
            "table_id": "T-CLAIM-BOUNDARY",
            "report_section": "Claim boundary matrix",
            "claim_ids": all_claim_ids,
            "artifact_refs": all_artifact_refs,
        },
    ]
    recommendations = []
    for recommendation in _recommendations():
        source_rows = {
            (source["artifact_id"], source["artifact_sha256"], source["accepted_commit"], source["locator"]): source
            for claim_id in recommendation["support_claims"]
            for source in next(row["sources"] for row in claims if row["claim_id"] == claim_id)
        }
        recommendations.append(
            {
                "recommendation_id": recommendation["id"],
                "claim_ids": recommendation["support_claims"],
                "sources": [source_rows[key] for key in sorted(source_rows)],
                "limitations": recommendation["limitations"],
            }
        )
    return {
        "schema_version": "learned-sparse-evidence-map-v1",
        "phase_id": PHASE_ID,
        "publication": PUBLICATION,
        "accepted_commits": authentication["commits"],
        "tracked_artifacts": files,
        "private_research_evidence": {
            key: value | {"classification": "private_research_only", "path_exposed": False}
            for key, value in sorted(PRIVATE_EVIDENCE.items())
        },
        "claims": claims,
        "tables": table_map,
        "recommendations": recommendations,
        "derivation_protocol": DERIVATION_PROTOCOL,
        "derivation_protocol_sha256": sha256_bytes(json_bytes(DERIVATION_PROTOCOL)),
    }


def _fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def render_report(summary: Mapping[str, Any], claim_matrix: Mapping[str, Any]) -> str:
    lines = [
        "# Learned Sparse Retrieval: Phase-One Synthesis",
        "",
        "## Decision summary",
        "",
        "Phase one establishes a reproducible same-paradigm learned-sparse comparison on two English research tracks, plus a separate sparse-system experiment. It does not establish a universal model winner, a cross-paradigm ranking, multilingual or verified zero-shot effectiveness, a production SLA, or a universal SINDI advantage. The publication gate remains closed.",
        "",
        "The most defensible maintenance rule is conditional selection: choose the track and metric first, require paired uncertainty for quality claims, then price the model route, encoding cost, sparsity, exact-search footprint, and deployment system separately. [C-QUALITY, C-UNCERTAINTY, C-RESOURCES, C-SYSTEM]",
        "",
        "## Evidence and publication boundary",
        "",
        "This report is deterministically derived from the accepted commits for S-20260814-005, S-20260814-006, S-20260814-008, and S-20260814-010. The evidence map binds every table and material recommendation to a tracked SHA-256 identity and accepted commit; private evidence appears only as non-path identities. No model is loaded and no Milvus connection is made by the derivation command.",
        "",
        "The underlying BRIGHT passages, queries, qrels, canonical identifiers, raw rankings, per-query rows, and review mappings remain restricted. Repository-level licensing does not establish redistribution rights for every linked upstream document. All outputs are `research_only`, with public export and leaderboard publication disabled. [C-LICENSE]",
        "",
        "## Data contract and governance",
        "",
        "The source is `xlangai/BRIGHT` at revision `3066d29c9651a576c8aba4832d249807b181ecae`, loaded without remote code. Economics and psychology retain independent candidate pools. Every query and positive passage is retained; remaining passages are selected by a score-blind, salted exact-content-hash policy. `gold_ids` supply grade-1 qrels, excluded IDs are filters rather than negative labels, and unjudged passages remain unjudged. [C-DATA]",
        "",
        "| Track | Queries | Passages | Positive qrels | Qrel density | Single-positive queries |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["data"]["tracks"]:
        lines.append(
            f"| {row['track']} | {row['queries']} | {row['documents']:,} | {row['positive_qrels']} | {row['qrel_density']:.8f} | {row['single_positive_queries']} |"
        )
    lines += [
        "",
        "The combined package therefore contains 204 queries, 15,000 passages, and 1,492 positive qrels. A 120-query stratified manual-review plan was frozen before model execution, but this phase does not represent that review as completed. Training overlap is unknown, and no verified zero-shot claim is allowed. [C-DATA, C-MULTILINGUAL-GAP]",
        "",
        "## Model portfolio and routing",
        "",
        "The seven-model matrix spans neural query/document expansion, contextual lexical projection, and static query lookup paired with neural document expansion. Revisions are fixed; route differences matter operationally because static-query models move more computation to document ingestion while neural-query models retain learned query encoding. [C-ROUTING]",
        "",
        "| Model | Query route | Document route | Dimensions | Snapshot MiB | Economics doc/query nnz | Psychology doc/query nnz |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in summary["models"]:
        econ = model["tracks"]["economics"]
        psych = model["tracks"]["psychology"]
        lines.append(
            f"| {model['model_key']} | {model['query_route']} | {model['document_route']} | {model['dimensions']:,} | {model['snapshot_bytes'] / 2**20:.1f} | "
            f"{econ['document_nnz_mean']:.1f}/{econ['query_nnz_mean']:.1f} | {psych['document_nnz_mean']:.1f}/{psych['query_nnz_mean']:.1f} |"
        )
    lines += [
        "",
        "SPLADE-Tiny is the smallest frozen snapshot (about 17.8 MiB), but its MIT statement is frozen only from model-card/API evidence and the repository lacks a standalone license file. It remains local research-only and is not a redistribution recommendation. [C-LICENSE]",
        "",
        "## Same-paradigm quality evidence",
        "",
        "All learned-sparse rows use the same per-track corpus, queries, qrels, candidate policy, maximum length 512, exact SciPy CSR inner product, top-100 depth, and five metrics. Point estimates are useful for shortlisting, but a point leader is not automatically a statistically supported winner. [C-QUALITY, C-UNCERTAINTY]",
        "",
    ]
    for track in TRACKS:
        lines += [
            f"### {track.title()}",
            "",
            "| Model | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        rows = sorted(
            summary["quality"]["tracks"][track]["rows"], key=lambda row: row["metrics"]["ndcg@10"], reverse=True
        )
        for row in rows:
            metric = row["metrics"]
            lines.append(
                f"| {row['model_key']} | {_fmt(metric['ndcg@10'], 6)} | {_fmt(metric['map@100'], 6)} | {_fmt(metric['mrr@10'], 6)} | {_fmt(metric['recall@10'], 6)} | {_fmt(metric['recall@100'], 6)} |"
            )
        uncertainty = summary["quality"]["paired_uncertainty"]["tracks"][track]
        significant = uncertainty["verdict_counts"].get("left_wins", 0) + uncertainty["verdict_counts"].get(
            "right_wins", 0
        )
        lines += [
            "",
            f"The package evaluates 21 pairs × 5 metrics = {uncertainty['metric_comparisons']} query-aligned comparisons, each with 10,000 bootstrap samples. {significant} intervals exclude zero; the remaining {uncertainty['verdict_counts'].get('tie_or_uncertain', 0)} are explicitly uncertain. The nDCG@10 point order shown above is track-specific, not an overall ranking. [C-UNCERTAINTY]",
            "",
        ]
    lines += [
        "## Limited BM25 and dense anchors",
        "",
        "S-005 BM25 and BGE-M3 long-dense numbers are retained only as contextual anchors. The source revision, canonical passage/query retrieval units without re-segmentation, independent per-track candidate pools, qrels, exclusion filtering, top-100 depth, and five metrics are identity-linked to the learned-sparse package. Encoder processing batch sizes differ but do not redefine retrieval units. This supports bounded same-pilot context, but not the complete multi-baseline cross-paradigm comparison reserved for S-20260814-009. Dense uses a predeclared 1,024-token cap, whereas learned-sparse uses 512 tokens. [C-ANCHORS]",
        "",
        "| Track | Method | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for track in TRACKS:
        for method, label in (("bm25", "BM25"), ("bge_m3_long_dense", "BGE-M3 long dense")):
            metrics = summary["anchors"]["tracks"][track][method]
            lines.append(
                f"| {track} | {label} | {_fmt(metrics['ndcg@10'])} | {_fmt(metrics['map@100'])} | {_fmt(metrics['mrr@10'])} | {_fmt(metrics['recall@10'])} | {_fmt(metrics['recall@100'])} |"
            )
    lines += [
        "",
        "Economics does not establish a top-10 dense-versus-BM25 difference because the paired nDCG@10 interval crosses zero. Psychology has five dense-minus-BM25 intervals excluding zero. These statements remain anchor-specific and do not rank paradigms generally. [C-ANCHORS]",
        "",
        "## Encoding, sparsity, and exact retrieval",
        "",
        "| Model | Track | Doc items/s | Query items/s | Doc CSR MiB | Exact search s | Doc peak VRAM MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in summary["models"]:
        for track in TRACKS:
            row = model["tracks"][track]
            lines.append(
                f"| {model['model_key']} | {track} | {row['document_throughput_items_per_second']:.1f} | {row['query_throughput_items_per_second']:.1f} | {row['document_csr_bytes'] / 2**20:.2f} | {row['exact_search_seconds']:.3f} | {row['document_peak_vram_bytes'] / 2**20:.1f} |"
            )
    lines += [
        "",
        "SPLADE-Tiny has the highest measured document throughput in both tracks, while the static-query routes provide high query throughput. BGE-M3 sparse has the smallest document CSR footprint among the seven. These are bounded encoding/exact measurements, not end-to-end production throughput. Every RSS value is encoding-observed rather than a whole-cell peak. [C-RESOURCES]",
        "",
        "## Predeclared slices and opaque cases",
        "",
        "The package freezes query length, positive lexical overlap, positive-document length, and positive-qrel-density slices before Batch-B scores. It reports 13 bins per track. Empty bins are NA, and any bin with n<10 is descriptive only; no bin is merged after results. [C-SLICES]",
        "",
        "| Track | Slice | Bin sizes |",
        "| --- | --- | --- |",
    ]
    for track in TRACKS:
        for dimension in summary["slices"][track]:
            bins = ", ".join(f"{row['label']}: n={row['n']}" for row in dimension["bins"])
            lines.append(f"| {track} | {dimension['dimension']} | {bins} |")
    lines += [
        "",
        "The tracked case set contains six disagreements and six shared failures per track, 24 cases total. Only opaque case labels, mechanism-level summaries, and non-reversible hashes are exposed; the canonical mapping and source text remain private. These cases diagnose where mechanisms disagree or jointly fail, but do not create new semantic slice labels. [C-CASES]",
        "",
        "## Sparse-system evidence",
        "",
        "The system layer is separate from model quality. It uses saved CSR inputs with `model_loaded=false` and compares exact SciPy CSR, Milvus SINDI, and Milvus DAAT_MAXSCORE. The server is Milvus 3.0.0, PyMilvus 3.0.1, vector index version 10, fixed CPU/memory/PID limits, explicit sparse IP algorithms, `drop_ratio_build=0`, and `drop_ratio_search=0`. Searches require serialized indexes, sealed segments, zero growing rows, and no active or queued compaction. [C-EXACT, C-SYSTEM]",
        "",
        "| Tier | Cells | Models/tracks | Purpose | Quality eligible |",
        "| --- | ---: | --- | --- | --- |",
        "| Native | 18 | 3 model profiles × 2 tracks × 3 comparators | Correctness and bounded system behavior | No; model quality remains in the original quality package |",
        "| System-only 100k | 9 | 3 profiles × 3 comparators | Scaling and cost | No |",
        "| System-only 1M | 9 | 3 profiles × 3 comparators | Scaling and cost | No |",
        "",
        f"The validator recomputed {summary['system']['coverage']['raw_trial_files_recomputed']:,} measured trials and validated {summary['system']['coverage']['warmup_files_validated']} warmups. Native raw strict Recall@K reaches a minimum of {summary['system']['correctness']['native_minimum_strict_recall']:.1f}; after matching the persisted float16-document/float32-query representation, tie-aware Recall@K is 1.0 and the maximum score delta is {summary['system']['correctness']['native_fp16_aware_maximum_score_delta']:.14f}. The system-only strict minimum of {summary['system']['correctness']['all_tiers_minimum_strict_recall']:.1f} occurs at synthetic near-tie boundaries and is not model-quality evidence. [C-SYSTEM, C-SYSTEM-QUALITY]",
        "",
        "| System result | Value | Interpretation |",
        "| --- | ---: | --- |",
        f"| Median SINDI/DAAT QPS ratio | {summary['system']['qps']['sindi_over_daat_median']:.4f} | Near parity in the pooled tested configurations |",
        f"| Observed ratio range | {summary['system']['qps']['sindi_over_daat_min']:.4f}–{summary['system']['qps']['sindi_over_daat_max']:.4f} | Crosses 1.0; workload/configuration crossover |",
        f"| SINDI-larger serialized-index pairs | {summary['system']['persisted_index']['sindi_larger_pairs']}/{summary['system']['persisted_index']['pairs']} | SINDI used more serialized index bytes in every pair |",
        f"| Serialized-index ratio range | {summary['system']['persisted_index']['sindi_over_daat_ratio_min']:.6f}–{summary['system']['persisted_index']['sindi_over_daat_ratio_max']:.6f} | Does not include all runtime or operational cost |",
        "",
        "The maximum QPS ratio comes from a synthetic 1M OpenSearch-multilingual K100/concurrency-1 DAAT tail collapse; other configurations show parity or DAAT advantages. Warm-over-cold medians exceed 1.0 by stratum, but individual reversals remain. Cold runs reload collections without dropping host page caches. There is no universal algorithm, cache, scale, or production claim. [C-SYSTEM, C-UNIVERSAL-SINDI, C-PRODUCTION]",
        "",
        "## Scenario-based maintenance guide",
        "",
    ]
    for recommendation in summary["recommendations"]:
        lines += [
            f"### {recommendation['scenario']}",
            "",
            recommendation["guidance"] + " [" + ", ".join(recommendation["support_claims"]) + "]",
            "",
            "Limits: " + "; ".join(recommendation["limitations"]) + ".",
            "",
        ]
    lines += [
        "## Claim boundary matrix",
        "",
        "| Claim ID | Support | Statement | Scope or reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in claim_matrix["claims"]:
        lines.append(f"| {row['claim_id']} | {row['support']} | {row['claim']} | {row['scope_or_reason']} |")
    lines += [
        "",
        "## Remaining evidence gaps",
        "",
    ]
    lines.extend(f"- {gap}" for gap in summary["remaining_gaps"])
    lines += [
        "",
        "These gaps are inputs to the next milestone route review. They do not authorize S-007 data generation, S-009 cross-paradigm experiments, S-012 multi-vector work, publication, or any new benchmark/index run.",
        "",
        "## Deterministic verification",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES='' uv run python scripts/learned_sparse_phase_one_report.py check",
        "```",
        "",
        "The command authenticates accepted commit ancestry and tracked file hashes, validates source and output schemas, recomputes every derived table and claim object in memory, and compares exact bytes. It neither loads a model nor connects to Milvus.",
    ]
    return "\n".join(lines) + "\n"


def build_outputs(repo_root: Path) -> dict[str, bytes]:
    values = load_inputs(repo_root)
    summary = build_summary(values)
    claims = build_claim_matrix()
    evidence = build_evidence_map(values["authentication"], claims)
    report = render_report(summary, claims).encode()
    return {
        "summary.json": json_bytes(summary),
        "evidence-map.json": json_bytes(evidence),
        "claim-support-matrix.json": json_bytes(claims),
        "report": report,
    }


def _output_manifest(outputs: Mapping[str, bytes]) -> dict[str, Any]:
    names = {
        "summary.json": "benchmark/artifacts/learned-sparse-phase-one-v0.1/summary.json",
        "evidence-map.json": "benchmark/artifacts/learned-sparse-phase-one-v0.1/evidence-map.json",
        "claim-support-matrix.json": "benchmark/artifacts/learned-sparse-phase-one-v0.1/claim-support-matrix.json",
        "report": "benchmark/research/learned_sparse_phase_one_20260904.md",
    }
    return {
        "schema_version": "learned-sparse-phase-one-manifest-v1",
        "phase_id": PHASE_ID,
        "story_id": STORY_ID,
        "publication": PUBLICATION,
        "accepted_commits": ACCEPTED_COMMITS,
        "derivation_protocol_sha256": sha256_bytes(json_bytes(DERIVATION_PROTOCOL)),
        "outputs": {
            key: {"path": names[key], "bytes": len(payload), "sha256": sha256_bytes(payload)}
            for key, payload in outputs.items()
        },
    }


def _validate_outputs(repo_root: Path, outputs: Mapping[str, bytes], manifest: Mapping[str, Any]) -> None:
    schema_bindings = {
        "summary.json": "schemas/learned-sparse-phase-one-summary-v01.schema.json",
        "evidence-map.json": "schemas/learned-sparse-phase-one-evidence-map-v01.schema.json",
        "claim-support-matrix.json": "schemas/learned-sparse-phase-one-claim-support-v01.schema.json",
        "manifest.json": "schemas/learned-sparse-phase-one-manifest-v01.schema.json",
    }
    values = {key: json.loads(payload) for key, payload in outputs.items() if key != "report"}
    values["manifest.json"] = manifest
    for key, schema_path in schema_bindings.items():
        schema = _load_json(repo_root / schema_path)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(values[key])


def write_outputs(repo_root: Path) -> dict[str, Any]:
    outputs = build_outputs(repo_root)
    manifest = _output_manifest(outputs)
    _validate_outputs(repo_root, outputs, manifest)
    artifact_root = repo_root / "benchmark/artifacts/learned-sparse-phase-one-v0.1"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for key in ("summary.json", "evidence-map.json", "claim-support-matrix.json"):
        (artifact_root / key).write_bytes(outputs[key])
    report_path = repo_root / manifest["outputs"]["report"]["path"]
    report_path.write_bytes(outputs["report"])
    manifest_bytes = json_bytes(manifest)
    (artifact_root / "manifest.json").write_bytes(manifest_bytes)
    (artifact_root / "manifest.sha256").write_text(sha256_bytes(manifest_bytes) + "\n", encoding="ascii")
    return manifest


def check_outputs(repo_root: Path) -> dict[str, Any]:
    outputs = build_outputs(repo_root)
    expected_manifest = _output_manifest(outputs)
    _validate_outputs(repo_root, outputs, expected_manifest)
    artifact_root = repo_root / "benchmark/artifacts/learned-sparse-phase-one-v0.1"
    expected = {
        artifact_root / "summary.json": outputs["summary.json"],
        artifact_root / "evidence-map.json": outputs["evidence-map.json"],
        artifact_root / "claim-support-matrix.json": outputs["claim-support-matrix.json"],
        repo_root / expected_manifest["outputs"]["report"]["path"]: outputs["report"],
        artifact_root / "manifest.json": json_bytes(expected_manifest),
    }
    for path, content in expected.items():
        if not path.is_file() or path.read_bytes() != content:
            raise PhaseOneError(f"derived output differs from deterministic rebuild: {path.relative_to(repo_root)}")
    manifest_bytes = expected[artifact_root / "manifest.json"]
    sidecar = artifact_root / "manifest.sha256"
    if sidecar.read_text(encoding="ascii") != sha256_bytes(manifest_bytes) + "\n":
        raise PhaseOneError("phase-one manifest sidecar mismatch")
    return {
        "status": "pass",
        "phase_id": PHASE_ID,
        "model_loaded": False,
        "milvus_connected": False,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "output_count": len(expected) + 1,
    }
