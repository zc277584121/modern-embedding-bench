#!/usr/bin/env python3
"""Build or replay the fail-closed BRIGHT learned-sparse aggregate package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_embed.benchmark import bright_learned_sparse_batch_b as batch_b
from mm_embed.benchmark.bright_learned_sparse_package import (
    PACKAGE_PROTOCOL,
    PUBLICATION,
    REPLAY_SCHEMA,
    canonical_sha256,
    collect_authenticated_cells,
    compute_outputs,
    file_sha256,
    formatted_json_bytes,
    load_input_contract,
    package_manifest,
    validate_complete_package,
    write_package,
    _load_json,
    _validate_schema,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--input-contract", required=True)
    root.add_argument("--expected-input-contract-sha256", required=True)
    root.add_argument("--repo-root", default=".")
    root.add_argument("--data-root", default="data/bright-nontechnical-pilot-v0.2")
    root.add_argument("--batch-a-raw-root", default="results/bright-learned-sparse-batch-a")
    root.add_argument("--batch-b-raw-root", default="results/bright-learned-sparse-batch-b")
    root.add_argument("--batch-b-replay-root", default="results/bright-learned-sparse-batch-b-replay")
    root.add_argument("--snapshot-evidence-root", default="results/learned-sparse-batch-b/active-v2")
    root.add_argument("--package-root", default="benchmark/artifacts/bright-learned-sparse-main-v0.1")
    root.add_argument("--report", default="benchmark/research/learned_sparse_main_20260829.md")
    root.add_argument("--local-mapping", default="results/bright-learned-sparse-main-v0.1/case-mapping.json")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("preview")
    build = commands.add_parser("build")
    build.add_argument("--expected-package-manifest-sha256", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--expected-package-manifest-sha256", required=True)
    replay.add_argument("--output", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    contract = load_input_contract(args.input_contract, args.expected_input_contract_sha256)
    cells = collect_authenticated_cells(
        contract=contract,
        repo_root=args.repo_root,
        data_root=args.data_root,
        batch_a_raw_root=args.batch_a_raw_root,
        batch_b_raw_root=args.batch_b_raw_root,
        batch_b_replay_root=args.batch_b_replay_root,
        snapshot_evidence_root=args.snapshot_evidence_root,
    )
    predecl = batch_b.validate_predeclaration(
        Path(args.repo_root) / "benchmark/artifacts/bright-learned-sparse-batch-b-v0.1/predeclaration.json",
        expected_sha256=contract["active_predeclaration_sha256"],
    )
    batch_a_inventory = _load_json(
        Path(args.repo_root) / "benchmark/artifacts/bright-learned-sparse-batch-a-v0.1/inventory.json"
    )
    files, evidence = compute_outputs(
        cells=cells,
        data_root=args.data_root,
        predeclaration=predecl,
        batch_a_inventory=batch_a_inventory,
    )
    local_mapping_sha = canonical_sha256(evidence["local_mapping"])
    manifest = package_manifest(
        files=files,
        contract=contract,
        input_contract_sha256=args.expected_input_contract_sha256,
        local_mapping_sha256=local_mapping_sha,
        privacy=evidence["privacy"],
    )
    manifest_payload = formatted_json_bytes(manifest)
    manifest_sha = __import__("hashlib").sha256(manifest_payload).hexdigest()
    if args.command == "preview":
        print(json.dumps({
            "status": "preview",
            "package_manifest_sha256": manifest_sha,
            "local_mapping_sha256": local_mapping_sha,
            "tracked_files": manifest["tracked_files"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "build":
        identity = write_package(
            package_root=args.package_root,
            report_path=args.report,
            local_mapping_path=args.local_mapping,
            files=files,
            manifest=manifest,
            local_mapping=evidence["local_mapping"],
            expected_package_sha256=args.expected_package_manifest_sha256,
        )
        print(json.dumps({"status": "built", "package_manifest_sha256": identity}, sort_keys=True))
        return
    validate_complete_package(
        package_root=args.package_root,
        report_path=args.report,
        expected_package_sha256=args.expected_package_manifest_sha256,
        expected_files=files,
        expected_manifest=manifest,
    )
    saved_local = _load_json(args.local_mapping)
    if saved_local != evidence["local_mapping"] or canonical_sha256(saved_local) != local_mapping_sha:
        raise ValueError("Local research-only case mapping cannot be rebuilt from authenticated raw cells")
    replay = {
        "schema_version": "bright-learned-sparse-main-replay-v1",
        "status": "pass",
        "cpu_only": True,
        "model_loaded": False,
        "package_manifest_sha256": args.expected_package_manifest_sha256,
        "package_protocol_sha256": canonical_sha256(PACKAGE_PROTOCOL),
        "input_contract_sha256": args.expected_input_contract_sha256,
        "rebuilt_tracked_files_sha256": canonical_sha256(manifest["tracked_files"]),
        "local_mapping_sha256": local_mapping_sha,
        "publication": PUBLICATION,
    }
    _validate_schema(replay, REPLAY_SCHEMA)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(formatted_json_bytes(replay))
    output.with_suffix(output.suffix + ".sha256").write_text(file_sha256(output) + "\n", encoding="ascii")
    print(json.dumps(replay, sort_keys=True))


if __name__ == "__main__":
    main()
