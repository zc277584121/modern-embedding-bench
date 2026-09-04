#!/usr/bin/env python3
"""Operate the frozen S-20260814-012 real multi-vector benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_embed.benchmark.bright_multi_vector import (
    ACTIVE_PREDECLARATION_SHA256,
    build_gate_summary,
    encode_formal_model,
    replay_formal_cell,
    run_gate,
    validate_predeclaration,
)
from mm_embed.benchmark.bright_multi_vector_package import (
    collect_cells,
    compute_outputs,
    make_manifest,
    write_or_check_package,
)
from mm_embed.providers.real_multi_vector import SELECTED_KEYS

DEFAULT_PREDECLARATION = Path("benchmark/artifacts/bright-multi-vector-v0.2/predeclaration.json")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-freeze")
    check.add_argument("--predeclaration", type=Path, default=DEFAULT_PREDECLARATION)
    check.add_argument("--expected-sha256", default=ACTIVE_PREDECLARATION_SHA256)
    gate = commands.add_parser("gate")
    gate.add_argument("--model", choices=SELECTED_KEYS, required=True)
    gate.add_argument("--predeclaration", type=Path, default=DEFAULT_PREDECLARATION)
    gate.add_argument("--predeclaration-sha256", default=ACTIVE_PREDECLARATION_SHA256)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--device", default="cuda:0")
    summary = commands.add_parser("gate-summary")
    summary.add_argument("--gates", type=Path, nargs=3, required=True)
    summary.add_argument("--output", type=Path, required=True)
    encode = commands.add_parser("encode-model")
    encode.add_argument("--model", choices=SELECTED_KEYS, required=True)
    encode.add_argument("--predeclaration", type=Path, default=DEFAULT_PREDECLARATION)
    encode.add_argument("--predeclaration-sha256", default=ACTIVE_PREDECLARATION_SHA256)
    encode.add_argument(
        "--gate-summary",
        type=Path,
        default=Path("benchmark/artifacts/bright-multi-vector-v0.2/gate-summary.json"),
    )
    encode.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    encode.add_argument("--output-root", type=Path, default=Path("results/bright-multi-vector-v0.2/representations"))
    encode.add_argument("--device", default="cuda:0")
    replay = commands.add_parser("replay-cell")
    replay.add_argument("--model", choices=SELECTED_KEYS, required=True)
    replay.add_argument("--track", choices=("economics", "psychology"), required=True)
    replay.add_argument("--predeclaration", type=Path, default=DEFAULT_PREDECLARATION)
    replay.add_argument("--predeclaration-sha256", default=ACTIVE_PREDECLARATION_SHA256)
    replay.add_argument(
        "--gate-summary",
        type=Path,
        default=Path("benchmark/artifacts/bright-multi-vector-v0.2/gate-summary.json"),
    )
    replay.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    replay.add_argument(
        "--representation-root",
        type=Path,
        default=Path("results/bright-multi-vector-v0.2/representations"),
    )
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--device", default="cuda:0")
    for command in ("build-package", "check-package"):
        package = commands.add_parser(command)
        package.add_argument("--predeclaration", type=Path, default=DEFAULT_PREDECLARATION)
        package.add_argument(
            "--inventory",
            type=Path,
            default=Path("benchmark/artifacts/bright-multi-vector-v0.1/inventory.json"),
        )
        package.add_argument(
            "--gate-summary",
            type=Path,
            default=Path("benchmark/artifacts/bright-multi-vector-v0.2/gate-summary.json"),
        )
        package.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
        package.add_argument("--cell-root", type=Path, default=Path("results/bright-multi-vector-v0.2/cells"))
        package.add_argument(
            "--package-root", type=Path, default=Path("benchmark/artifacts/bright-multi-vector-results-v0.2")
        )
        package.add_argument(
            "--report", type=Path, default=Path("benchmark/research/bright_multi_vector_v02_20260904.md")
        )
        package.add_argument(
            "--local-mapping", type=Path, default=Path("results/bright-multi-vector-v0.2/case-mapping.json")
        )
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "check-freeze":
        result = validate_predeclaration(args.predeclaration, args.expected_sha256)
        print(json.dumps({"status": "valid", "story_id": result["story_id"], "models": len(result["models"])}))
    elif args.command == "gate":
        result = run_gate(
            args.model,
            predeclaration_path=args.predeclaration,
            predeclaration_sha256=args.predeclaration_sha256,
            output_path=args.output,
            device=args.device,
        )
        print(json.dumps({"model": args.model, "passed": result["passed"]}))
    elif args.command == "gate-summary":
        result = build_gate_summary(args.gates, args.output)
        print(json.dumps({"models": len(result["models"]), "formal_scoring_authorized": True}))
    elif args.command == "encode-model":
        result = encode_formal_model(
            args.model,
            predeclaration_path=args.predeclaration,
            predeclaration_sha256=args.predeclaration_sha256,
            gate_summary_path=args.gate_summary,
            data_root=args.data_root,
            output_root=args.output_root,
            device=args.device,
        )
        print(json.dumps({"model": args.model, "tracks": sorted(result["tracks"]), "finalized": True}))
    elif args.command == "replay-cell":
        result = replay_formal_cell(
            args.model,
            args.track,
            predeclaration_path=args.predeclaration,
            predeclaration_sha256=args.predeclaration_sha256,
            gate_summary_path=args.gate_summary,
            data_root=args.data_root,
            representation_root=args.representation_root,
            output_path=args.output,
            device=args.device,
        )
        print(json.dumps({"model": args.model, "track": args.track, "metrics": result["quality"]["metrics"]}))
    elif args.command in {"build-package", "check-package"}:
        cells, predeclaration, representation_inputs = collect_cells(
            cell_root=args.cell_root,
            predeclaration_path=args.predeclaration,
            gate_summary_path=args.gate_summary,
            data_root=args.data_root,
        )
        gate_summary = json.loads(args.gate_summary.read_text(encoding="utf-8"))
        files, evidence = compute_outputs(
            cells=cells,
            predecl=predeclaration,
            gate_summary=gate_summary,
            data_root=args.data_root,
        )
        manifest = make_manifest(
            files=files,
            cells=cells,
            predecl_path=args.predeclaration,
            inventory_path=args.inventory,
            gate_summary_path=args.gate_summary,
            representation_inputs=representation_inputs,
            local_mapping=evidence["local_mapping"],
            privacy=evidence["privacy"],
        )
        identity = write_or_check_package(
            package_root=args.package_root,
            report_path=args.report,
            local_mapping_path=args.local_mapping,
            files=files,
            manifest=manifest,
            local_mapping=evidence["local_mapping"],
            check_only=args.command == "check-package",
        )
        print(
            json.dumps(
                {
                    "status": "pass" if args.command == "check-package" else "candidate_built",
                    "manifest_sha256": identity,
                    "model_loaded": False,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
