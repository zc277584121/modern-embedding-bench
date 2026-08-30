"""CLI for Batch-B freeze, readiness, formal execution, and CPU replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_embed.benchmark.bright_learned_sparse_batch_b import (
    download_snapshot,
    plan_snapshot,
    replay_formal_cell,
    run_formal_cell,
    run_gate,
    validate_batch_a_readonly,
    validate_evidence_file,
    validate_hub_metadata,
    validate_predeclaration,
    validate_supersession,
    write_evidence,
)
from mm_embed.providers.learned_sparse_inventory import BATCH_B_SELECTED_KEYS


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predeclaration", type=Path, required=True)
    parser.add_argument("--expected-predeclaration-sha256", required=True)
    parser.add_argument(
        "--superseded-predeclaration",
        type=Path,
        default=Path(
            "benchmark/artifacts/bright-learned-sparse-batch-b-v0.1/predeclaration-v1-superseded.json"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-freeze")
    for name in ("metadata", "download-plan"):
        child = commands.add_parser(name)
        child.add_argument("--model", choices=BATCH_B_SELECTED_KEYS, required=True)
        child.add_argument("--cache-root", type=Path)
        child.add_argument("--output", type=Path, required=True)
        if name == "download-plan":
            child.add_argument("--metadata", type=Path, required=True)
            child.add_argument("--expected-metadata-sha256", required=True)
    download = commands.add_parser("download")
    download.add_argument("--model", choices=BATCH_B_SELECTED_KEYS, required=True)
    download.add_argument("--plan", type=Path, required=True)
    download.add_argument("--expected-plan-sha256", required=True)
    download.add_argument("--cache-root", type=Path)
    download.add_argument("--output", type=Path, required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--model", choices=BATCH_B_SELECTED_KEYS, required=True)
    gate.add_argument("--snapshot-evidence", type=Path, required=True)
    gate.add_argument("--expected-snapshot-evidence-sha256", required=True)
    gate.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--device", default="cuda:0")
    run = commands.add_parser("run")
    run.add_argument("--model", choices=BATCH_B_SELECTED_KEYS, required=True)
    run.add_argument("--track", choices=("economics", "psychology"), required=True)
    run.add_argument("--snapshot-evidence", type=Path, required=True)
    run.add_argument("--expected-snapshot-evidence-sha256", required=True)
    run.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--expected-manifest-sha256")
    replay = commands.add_parser("replay")
    replay.add_argument("--result", type=Path, required=True)
    replay.add_argument("--snapshot-evidence", type=Path, required=True)
    replay.add_argument("--expected-snapshot-evidence-sha256", required=True)
    replay.add_argument("--expected-manifest-sha256", required=True)
    replay.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen = validate_predeclaration(
        args.predeclaration, expected_sha256=args.expected_predeclaration_sha256
    )
    validate_supersession(frozen, args.superseded_predeclaration)
    validate_batch_a_readonly(frozen, args.repo_root)
    if args.command == "validate-freeze":
        value = {"status": "valid", "predeclaration_sha256": args.expected_predeclaration_sha256}
    elif args.command == "metadata":
        value = validate_hub_metadata(
            args.model, frozen, predeclaration_sha256=args.expected_predeclaration_sha256
        )
        identity = write_evidence(args.output, value)
        value = {**value, "evidence_sha256": identity}
    elif args.command == "download-plan":
        metadata = validate_evidence_file(
            args.metadata, expected_sha256=args.expected_metadata_sha256, label="metadata evidence"
        )
        value = plan_snapshot(
            args.model,
            metadata=metadata,
            metadata_sha256=args.expected_metadata_sha256,
            predeclaration_sha256=args.expected_predeclaration_sha256,
            cache_root=args.cache_root,
        )
        identity = write_evidence(args.output, value)
        value = {**value, "evidence_sha256": identity}
    elif args.command == "download":
        plan = validate_evidence_file(
            args.plan, expected_sha256=args.expected_plan_sha256, label="download plan"
        )
        value = download_snapshot(args.model, plan, cache_root=args.cache_root)
        identity = write_evidence(args.output, value)
        value = {**value, "evidence_sha256": identity}
    elif args.command == "gate":
        snapshot = validate_evidence_file(
            args.snapshot_evidence,
            expected_sha256=args.expected_snapshot_evidence_sha256,
            label="snapshot evidence",
        )
        value = run_gate(
            model_key=args.model,
            predeclaration=frozen,
            snapshot_evidence=snapshot,
            data_root=args.data_root,
            output_path=args.output,
            device=args.device,
            batch_size=8,
            predeclaration_sha256=args.expected_predeclaration_sha256,
            snapshot_evidence_sha256=args.expected_snapshot_evidence_sha256,
        )
        identity = write_evidence(args.output, value)
        value = {**value, "evidence_sha256": identity}
    elif args.command == "run":
        snapshot = validate_evidence_file(
            args.snapshot_evidence,
            expected_sha256=args.expected_snapshot_evidence_sha256,
            label="snapshot evidence",
        )
        value = run_formal_cell(
            model_key=args.model,
            track=args.track,
            predeclaration=frozen,
            predeclaration_sha256=args.expected_predeclaration_sha256,
            snapshot_evidence=snapshot,
            snapshot_evidence_sha256=args.expected_snapshot_evidence_sha256,
            data_root=args.data_root,
            output_root=args.output,
            device=args.device,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        value = {**value, "manifest_sha256": (args.output / "manifest.sha256").read_text().strip()}
    else:
        snapshot = validate_evidence_file(
            args.snapshot_evidence,
            expected_sha256=args.expected_snapshot_evidence_sha256,
            label="snapshot evidence",
        )
        replayed = replay_formal_cell(
            result_root=args.result,
            data_root=args.data_root,
            predeclaration=frozen,
            predeclaration_sha256=args.expected_predeclaration_sha256,
            snapshot_evidence=snapshot,
            snapshot_evidence_sha256=args.expected_snapshot_evidence_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        identity = write_evidence(args.output, replayed["evidence"])
        value = {**replayed["evidence"], "evidence_sha256": identity}
    _print(value)


if __name__ == "__main__":
    main()
