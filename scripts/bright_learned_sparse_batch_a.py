"""Command-line entry point for BRIGHT learned-sparse Batch A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_embed.benchmark.bright_learned_sparse_batch_a import (
    build_artifact_manifest,
    resolve_selected,
    run_gate,
    run_track,
    tracked_aggregate,
    validate_artifact_manifest,
    validate_raw_result,
    validate_raw_results,
    write_failure_cases,
    write_inventory,
)
from mm_embed.providers.learned_sparse_inventory import ANCHOR_KEYS, SELECTED_KEYS


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--output", type=Path)
    for command in ("download-plan", "download"):
        child = subparsers.add_parser(command)
        child.add_argument("--cache-root", type=Path)
        child.add_argument("--output", type=Path)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--model", choices=SELECTED_KEYS, required=True)
    gate.add_argument("--snapshot", type=Path)
    gate.add_argument("--cache-root", type=Path)
    gate.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--device", default="cuda:0")
    gate.add_argument("--batch-size", type=int, default=8)
    run = subparsers.add_parser("run")
    run.add_argument("--model", choices=SELECTED_KEYS + ANCHOR_KEYS, required=True)
    run.add_argument("--track", choices=("economics", "psychology"), required=True)
    run.add_argument("--snapshot", type=Path)
    run.add_argument("--cache-root", type=Path)
    run.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--chunk-size", type=int, default=256)
    run.add_argument("--expected-manifest-sha256")
    run.add_argument("--allow-incomplete-resume", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--result", type=Path, required=True)
    validate.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    validate.add_argument("--expected-manifest-sha256", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--result", type=Path, action="append", required=True)
    aggregate.add_argument("--expected-manifest-sha256", action="append", required=True)
    aggregate.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    aggregate.add_argument("--output", type=Path, required=True)
    failure_cases = subparsers.add_parser("failure-cases")
    failure_cases.add_argument("--result", type=Path, action="append", required=True)
    failure_cases.add_argument("--expected-manifest-sha256", action="append", required=True)
    failure_cases.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    failure_cases.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--result", type=Path, action="append", required=True)
    finalize.add_argument("--expected-manifest-sha256", action="append")
    finalize.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    finalize.add_argument("--artifact-root", type=Path, required=True)
    finalize.add_argument("--report", type=Path, required=True)
    finalize.add_argument("--expected-existing-manifest-sha256")
    validate_package = subparsers.add_parser("validate-package")
    validate_package.add_argument("--result", type=Path, action="append", required=True)
    validate_package.add_argument("--data-root", type=Path, default=Path("data/bright-nontechnical-pilot-v0.2"))
    validate_package.add_argument("--manifest", type=Path, required=True)
    validate_package.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()

    if args.command == "inventory":
        value = write_inventory(args.output)
    elif args.command in {"download-plan", "download"}:
        value = resolve_selected(args.cache_root, download=args.command == "download")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "gate":
        value = run_gate(
            model_key=args.model, snapshot_path=args.snapshot, data_root=args.data_root,
            cache_root=args.cache_root, output_path=args.output, device=args.device, batch_size=args.batch_size,
        )
    elif args.command == "run":
        value = run_track(
            model_key=args.model, track=args.track, snapshot_path=args.snapshot,
            cache_root=args.cache_root, data_root=args.data_root, output_root=args.output, device=args.device,
            batch_size=args.batch_size, chunk_size=args.chunk_size,
            expected_manifest_sha256=args.expected_manifest_sha256,
            allow_incomplete_resume=args.allow_incomplete_resume,
        )
    elif args.command == "validate":
        value = validate_raw_result(
            args.result,
            args.data_root,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    elif args.command == "aggregate":
        _, manifests = validate_raw_results(
            args.result, args.data_root, args.expected_manifest_sha256
        )
        value = tracked_aggregate(manifests)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "failure-cases":
        value = write_failure_cases(
            args.result,
            args.data_root,
            args.output,
            args.expected_manifest_sha256,
        )
    elif args.command == "finalize":
        value = build_artifact_manifest(
            artifact_root=args.artifact_root,
            report_path=args.report,
            result_roots=args.result,
            data_root=args.data_root,
            expected_manifest_sha256es=args.expected_manifest_sha256,
            expected_existing_manifest_sha256=args.expected_existing_manifest_sha256,
        )
    else:
        value = validate_artifact_manifest(
            manifest_path=args.manifest,
            result_roots=args.result,
            data_root=args.data_root,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    _print(value)


if __name__ == "__main__":
    main()
