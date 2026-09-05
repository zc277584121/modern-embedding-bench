#!/usr/bin/env python3
"""Run deterministic negative mutation checks for the S-009 machine contracts."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

from jsonschema import Draft202012Validator

from mm_embed.benchmark.bright_cross_paradigm_package import (
    PACKAGE_ROOT,
    ROOT,
    SCHEMAS,
    CrossParadigmPackageError,
    validate_machine_contracts,
)


def load_candidate() -> dict[str, Any]:
    return {name: json.loads((PACKAGE_ROOT / name).read_text(encoding="utf-8")) for name in SCHEMAS}


def validate(files: dict[str, Any]) -> None:
    for name, schema_name in SCHEMAS.items():
        schema = json.loads((ROOT / "schemas" / schema_name).read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(files[name])
    validate_machine_contracts(files)


def expect_rejected(files: dict[str, Any], mutate: Callable[[dict[str, Any]], None], name: str) -> str:
    candidate = copy.deepcopy(files)
    mutate(candidate)
    try:
        validate(candidate)
    except Exception:  # noqa: BLE001 - every declared contract rejection is acceptable
        return name
    raise CrossParadigmPackageError(f"Mutation was not rejected: {name}")


def remove_failed_resource_evidence(value: dict[str, Any]) -> None:
    row = next(item for item in value["resources.json"]["cells"] if item["status"] == "failed_closed")
    row["failure"] = None


def set_minilm_scope(value: dict[str, Any], scope: str) -> None:
    row = next(item for item in value["resources.json"]["cells"] if item["cell_id"] == "all-minilm-l6-v2:economics")
    row["representation"]["scope"] = scope


def run_mutations() -> list[str]:
    files = load_candidate()
    validate(files)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("matrix_method_count", lambda value: value["summary.json"]["matrix"].__setitem__("methods", 13)),
        ("summary_method_coverage", lambda value: value["summary.json"]["tracks"]["economics"]["methods"].pop()),
        (
            "summary_metric_set",
            lambda value: value["summary.json"]["tracks"]["economics"]["methods"][0]["metrics"].pop("ndcg@10"),
        ),
        ("summary_track_set", lambda value: value["summary.json"]["tracks"].pop("psychology")),
        (
            "pair_metric_nonempty",
            lambda value: value["pairwise.json"]["tracks"]["economics"][0].__setitem__("metrics", {}),
        ),
        (
            "pair_metric_exact",
            lambda value: value["pairwise.json"]["tracks"]["psychology"][0]["metrics"].pop("recall@100"),
        ),
        ("pair_count", lambda value: value["pairwise.json"]["tracks"]["economics"].pop()),
        (
            "resource_terminal_status",
            lambda value: value["resources.json"]["cells"][0].__setitem__("status", "running"),
        ),
        (
            "resource_pass_failure",
            lambda value: value["resources.json"]["cells"][0].__setitem__("failure", {"type": "x"}),
        ),
        (
            "resource_pass_representation",
            lambda value: value["resources.json"]["cells"][0]["representation"].__setitem__("bytes", None),
        ),
        ("resource_failure_evidence", remove_failed_resource_evidence),
        (
            "minilm_cross_family_scope",
            lambda value: set_minilm_scope(value, "complete_document_and_query_representation_plus_idf_for_tfidf"),
        ),
        (
            "minilm_nonsense_scope",
            lambda value: set_minilm_scope(value, "nonsense_cross_family_scope"),
        ),
        ("lineage_cell_count", lambda value: value["lineage.json"]["cells"].pop()),
        ("lineage_anchor_count", lambda value: value["lineage.json"]["contextual_anchors"].pop()),
        (
            "guidance_empty_order",
            lambda value: value["scenario-guidance.json"]["scenarios"][0].__setitem__("ordered_methods", []),
        ),
        (
            "guidance_empty_resources",
            lambda value: value["scenario-guidance.json"]["scenarios"][0].__setitem__(
                "successful_resource_tradeoffs", []
            ),
        ),
    ]
    for artifact in SCHEMAS:
        mutations.append(
            (
                f"publication_gate:{artifact}",
                lambda value, artifact=artifact: value[artifact]["publication"].__setitem__("gate", "open"),
            )
        )
    return [expect_rejected(files, mutation, name) for name, mutation in mutations]


def main() -> None:
    rejected = run_mutations()
    print(json.dumps({"status": "pass", "mutations_rejected": len(rejected), "names": rejected}, sort_keys=True))


if __name__ == "__main__":
    main()
