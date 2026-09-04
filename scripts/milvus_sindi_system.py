#!/usr/bin/env python3
"""Fail-closed evidence CLI for the Milvus SINDI system benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_embed.benchmark.milvus_sindi_fixture import (
    attach_compaction_evidence,
    capture_deployment_evidence,
    capture_recovery_evidence,
    finalize_fixture_evidence,
    run_sparse_fixture,
    validate_fixture_evidence,
    validate_recovery_evidence,
)
from mm_embed.benchmark.milvus_sindi_formal import (
    audit_native_sources,
    audit_one_million_cap,
    generate_system_workload,
    prepare_native_run,
    run_native_matrix,
    run_system_matrix,
    validate_formal_runtime,
)
from mm_embed.benchmark.milvus_sindi_results import (
    build_candidate_manifest,
    build_file_manifest,
    build_final_isolation_evidence,
    build_public_summary,
    validate_and_aggregate,
    validate_run_manifests,
    write_markdown_report,
)
from mm_embed.benchmark.milvus_sindi_system import (
    build_csr_exact_catalog,
    capture_baseline,
    capture_registry_manifests,
    compare_isolation_baselines,
    compare_recovery_baselines,
    validate_baseline,
    validate_deployment_evidence,
    validate_formal_gate,
    validate_predeclaration,
    validate_source_attestation,
    write_identity_artifact,
)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("capture-baseline")
    baseline.add_argument("--output", type=Path, required=True)

    registry = commands.add_parser("capture-registry-manifests")
    registry.add_argument("--output", type=Path, required=True)

    deployment = commands.add_parser("capture-deployment")
    deployment.add_argument("--output", type=Path, required=True)
    deployment.add_argument("--predeclaration", type=Path, required=True)
    deployment.add_argument("--predeclaration-sha256", required=True)

    deployment_check = commands.add_parser("validate-deployment")
    deployment_check.add_argument("--deployment-evidence", type=Path, required=True)
    deployment_check.add_argument("--expected-sha256", required=True)

    fixture = commands.add_parser("run-fixture")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--predeclaration", type=Path, required=True)
    fixture.add_argument("--predeclaration-sha256", required=True)
    fixture.add_argument("--deployment-evidence", type=Path, required=True)
    fixture.add_argument("--deployment-evidence-sha256", required=True)

    fixture_finalize = commands.add_parser("finalize-fixture")
    fixture_finalize.add_argument("--fixture-evidence", type=Path, required=True)
    fixture_finalize.add_argument("--expected-sha256", required=True)

    fixture_check = commands.add_parser("validate-fixture")
    fixture_check.add_argument("--fixture-evidence", type=Path, required=True)
    fixture_check.add_argument("--expected-sha256", required=True)

    fixture_compaction = commands.add_parser("attach-compaction-evidence")
    fixture_compaction.add_argument("--fixture-evidence", type=Path, required=True)
    fixture_compaction.add_argument("--expected-sha256", required=True)

    recovery = commands.add_parser("capture-recovery")
    recovery.add_argument("--output", type=Path, required=True)
    recovery.add_argument("--predeclaration", type=Path, required=True)
    recovery.add_argument("--predeclaration-sha256", required=True)
    recovery.add_argument("--fixture-evidence", type=Path, required=True)
    recovery.add_argument("--fixture-evidence-sha256", required=True)
    recovery.add_argument("--hardened-deployment", type=Path, required=True)
    recovery.add_argument("--hardened-deployment-sha256", required=True)
    recovery.add_argument("--before-stat", type=Path, required=True)
    recovery.add_argument("--after-stat", type=Path, required=True)
    recovery.add_argument("--prehardening-baseline-sha256", required=True)
    recovery.add_argument("--stopped-baseline-sha256", required=True)

    recovery_check = commands.add_parser("validate-recovery")
    recovery_check.add_argument("--recovery-evidence", type=Path, required=True)
    recovery_check.add_argument("--expected-sha256", required=True)

    isolation = commands.add_parser("compare-isolation")
    isolation.add_argument("--prelaunch-manifest", type=Path, required=True)
    isolation.add_argument("--prelaunch-sha256", required=True)
    isolation.add_argument("--postfixture-manifest", type=Path, required=True)
    isolation.add_argument("--postfixture-sha256", required=True)
    isolation.add_argument("--output", type=Path, required=True)

    recovery_isolation = commands.add_parser("compare-recovery-isolation")
    recovery_isolation.add_argument("--prehardening-manifest", type=Path, required=True)
    recovery_isolation.add_argument("--prehardening-sha256", required=True)
    recovery_isolation.add_argument("--stopped-manifest", type=Path, required=True)
    recovery_isolation.add_argument("--stopped-sha256", required=True)
    recovery_isolation.add_argument("--posthardening-manifest", type=Path, required=True)
    recovery_isolation.add_argument("--posthardening-sha256", required=True)
    recovery_isolation.add_argument("--output", type=Path, required=True)

    baseline_check = commands.add_parser("validate-baseline")
    baseline_check.add_argument("--manifest", type=Path, required=True)
    baseline_check.add_argument("--expected-sha256", required=True)
    baseline_check.add_argument("--predeclaration", type=Path, required=True)
    baseline_check.add_argument("--predeclaration-sha256", required=True)

    catalog = commands.add_parser("build-source-catalog")
    catalog.add_argument("--input-contract", type=Path, required=True)
    catalog.add_argument("--batch-a-root", type=Path, default=Path("results/bright-learned-sparse-batch-a"))
    catalog.add_argument("--batch-b-root", type=Path, default=Path("results/bright-learned-sparse-batch-b"))
    catalog.add_argument("--output", type=Path, required=True)

    predeclare = commands.add_parser("validate-predeclaration")
    predeclare.add_argument("--predeclaration", type=Path, required=True)
    predeclare.add_argument("--expected-sha256", required=True)

    source = commands.add_parser("validate-source-attestation")
    source.add_argument("--attestation", type=Path, required=True)
    source.add_argument("--expected-sha256", required=True)

    gate = commands.add_parser("formal-gate")
    gate.add_argument("--predeclaration", type=Path, required=True)
    gate.add_argument("--predeclaration-sha256", required=True)
    gate.add_argument("--source-attestation", type=Path, required=True)
    gate.add_argument("--source-attestation-sha256", required=True)
    gate.add_argument("--baseline-manifest", type=Path, required=True)
    gate.add_argument("--baseline-manifest-sha256", required=True)
    gate.add_argument("--fixture-evidence", type=Path, required=True)
    gate.add_argument("--fixture-evidence-sha256", required=True)
    gate.add_argument("--deployment-evidence", type=Path, required=True)
    gate.add_argument("--deployment-evidence-sha256", required=True)
    gate.add_argument("--postfixture-baseline-manifest", type=Path, required=True)
    gate.add_argument("--postfixture-baseline-sha256", required=True)
    gate.add_argument("--isolation-regression", type=Path, required=True)
    gate.add_argument("--isolation-regression-sha256", required=True)
    gate.add_argument("--hardened-deployment-evidence", type=Path, required=True)
    gate.add_argument("--hardened-deployment-evidence-sha256", required=True)
    gate.add_argument("--recovery-evidence", type=Path, required=True)
    gate.add_argument("--recovery-evidence-sha256", required=True)
    gate.add_argument("--prehardening-baseline-manifest", type=Path, required=True)
    gate.add_argument("--prehardening-baseline-sha256", required=True)
    gate.add_argument("--stopped-baseline-manifest", type=Path, required=True)
    gate.add_argument("--stopped-baseline-sha256", required=True)
    gate.add_argument("--posthardening-baseline-manifest", type=Path, required=True)
    gate.add_argument("--posthardening-baseline-sha256", required=True)
    gate.add_argument("--recovery-isolation", type=Path, required=True)
    gate.add_argument("--recovery-isolation-sha256", required=True)
    gate.add_argument("--output", type=Path, required=True)

    commands.add_parser("validate-formal-runtime")

    formal_source = commands.add_parser("audit-formal-sources")
    formal_source.add_argument("--output", type=Path, required=True)

    native_prepare = commands.add_parser("prepare-native-run")
    native_prepare.add_argument("--output-root", type=Path, required=True)

    native_run = commands.add_parser("run-native-matrix")
    native_run.add_argument("--output-root", type=Path, required=True)

    system_generate = commands.add_parser("generate-system-workload")
    system_generate.add_argument("--output-root", type=Path, required=True)
    system_generate.add_argument("--model", required=True)
    system_generate.add_argument("--scale", type=int, choices=(100_000, 1_000_000), required=True)

    system_run = commands.add_parser("run-system-matrix")
    system_run.add_argument("--output-root", type=Path, required=True)
    system_run.add_argument("--scale", type=int, choices=(100_000, 1_000_000), required=True)

    cap = commands.add_parser("audit-one-million-cap")
    cap.add_argument("--output-root", type=Path, required=True)

    formal_validate = commands.add_parser("validate-formal-results")
    formal_validate.add_argument("--output-root", type=Path, required=True)
    formal_validate.add_argument("--output", type=Path, required=True)

    run_manifest_validate = commands.add_parser("validate-formal-run-manifests")
    run_manifest_validate.add_argument("--output-root", type=Path, required=True)
    run_manifest_validate.add_argument("--output", type=Path, required=True)

    formal_manifest = commands.add_parser("build-formal-manifest")
    formal_manifest.add_argument("--output-root", type=Path, required=True)
    formal_manifest.add_argument("--output", type=Path, required=True)

    formal_summary = commands.add_parser("build-formal-summary")
    formal_summary.add_argument("--recomputed-evidence", type=Path, required=True)
    formal_summary.add_argument("--recomputed-sha256", required=True)
    formal_summary.add_argument("--output", type=Path, required=True)

    formal_report = commands.add_parser("write-formal-report")
    formal_report.add_argument("--summary", type=Path, required=True)
    formal_report.add_argument("--summary-sha256", required=True)
    formal_report.add_argument("--output", type=Path, required=True)

    final_isolation = commands.add_parser("build-final-isolation")
    final_isolation.add_argument("--pre-manifest", type=Path, required=True)
    final_isolation.add_argument("--pre-sha256", required=True)
    final_isolation.add_argument("--post-manifest", type=Path, required=True)
    final_isolation.add_argument("--post-sha256", required=True)
    final_isolation.add_argument("--output", type=Path, required=True)

    candidate = commands.add_parser("build-validator-candidate")
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--files", type=Path, nargs="+", required=True)

    args = parser.parse_args()
    if args.command == "validate-formal-runtime":
        value = validate_formal_runtime()
    elif args.command == "audit-formal-sources":
        evidence, identity = audit_native_sources(args.output)
        value = {
            "status": evidence["status"],
            "model_loaded": evidence["model_loaded"],
            "cell_count": len(evidence["cells"]),
            "source_audit_sha256": identity,
        }
    elif args.command == "prepare-native-run":
        evidence, identity = prepare_native_run(args.output_root)
        value = {
            "status": evidence["status"],
            "cell_count": len(evidence["cells"]),
            "cell_order_sha256": identity,
        }
    elif args.command == "run-native-matrix":
        value = run_native_matrix(args.output_root)
    elif args.command == "generate-system-workload":
        evidence, identity = generate_system_workload(args.output_root, model=args.model, scale=args.scale)
        value = {
            "status": evidence["status"],
            "model_profile": evidence["model_profile"],
            "scale": evidence["scale"],
            "generated_input_bytes": evidence["generated_input_bytes"],
            "manifest_sha256": identity,
        }
    elif args.command == "run-system-matrix":
        value = run_system_matrix(args.output_root, scale=args.scale)
    elif args.command == "audit-one-million-cap":
        evidence, identity = audit_one_million_cap(args.output_root)
        value = {
            "status": evidence["status"],
            "decision": evidence["decision"],
            "checks": evidence["checks"],
            "cap_evidence_sha256": identity,
        }
    elif args.command == "validate-formal-results":
        evidence, identity = validate_and_aggregate(args.output_root, output=args.output)
        value = {
            "status": evidence["status"],
            "cell_count": evidence["cell_count"],
            "raw_trial_files_recomputed": evidence["raw_trial_files_recomputed"],
            "recomputed_evidence_sha256": identity,
        }
    elif args.command == "validate-formal-run-manifests":
        evidence, identity = validate_run_manifests(args.output_root, output=args.output)
        value = {
            "status": evidence["status"],
            "run_count": len(evidence["runs"]),
            "cell_count": evidence["total_cell_count"],
            "resolved_missing_inline_sha256_count": sum(run["missing_inline_sha256_count"] for run in evidence["runs"]),
            "run_manifest_integrity_sha256": identity,
        }
    elif args.command == "build-formal-manifest":
        evidence, identity = build_file_manifest(args.output_root, output=args.output)
        value = {
            "status": evidence["status"],
            "file_count": evidence["file_count"],
            "total_bytes": evidence["total_bytes"],
            "formal_manifest_sha256": identity,
        }
    elif args.command == "build-formal-summary":
        evidence, identity = build_public_summary(
            args.recomputed_evidence,
            recomputed_sha256=args.recomputed_sha256,
            output=args.output,
        )
        value = {
            "status": evidence["status"],
            "coverage": evidence["coverage"],
            "aggregate": evidence["aggregate"],
            "formal_summary_sha256": identity,
        }
    elif args.command == "write-formal-report":
        evidence, identity = write_markdown_report(
            args.summary,
            summary_sha256=args.summary_sha256,
            output=args.output,
        )
        value = {**evidence, "report_sha256": identity}
    elif args.command == "build-final-isolation":
        evidence, identity = build_final_isolation_evidence(
            pre_manifest=args.pre_manifest,
            pre_sha256=args.pre_sha256,
            post_manifest=args.post_manifest,
            post_sha256=args.post_sha256,
            output=args.output,
        )
        value = {
            "status": evidence["status"],
            "checks": evidence["checks"],
            "ambient_changed_container_count": len(evidence["non_story_ambient_drift"]["changed_containers"]),
            "final_isolation_sha256": identity,
        }
    elif args.command == "build-validator-candidate":
        evidence, identity = build_candidate_manifest(args.files, output=args.output)
        value = {
            "status": evidence["status"],
            "candidate_file_count": evidence["candidate_file_count"],
            "candidate_set_sha256": evidence["candidate_set_sha256"],
            "candidate_manifest_sha256": identity,
        }
    elif args.command == "capture-baseline":
        manifest, identity = capture_baseline(args.output)
        value = {
            "status": manifest["status"],
            "baseline_manifest_sha256": identity,
            "captured_at_utc": manifest["captured_at_utc"],
            "related_container_count": manifest["related_container_count"],
        }
    elif args.command == "capture-registry-manifests":
        manifest, identity = capture_registry_manifests(args.output)
        value = {
            "status": manifest["status"],
            "registry_manifest_sha256": identity,
            "captured_at_utc": manifest["captured_at_utc"],
            "index_digest": manifest["index"]["digest"],
            "child_digest": manifest["child"]["digest"],
        }
    elif args.command == "capture-deployment":
        evidence, identity = capture_deployment_evidence(
            args.output,
            predeclaration_path=args.predeclaration,
            predeclaration_sha256=args.predeclaration_sha256,
        )
        value = {
            "status": evidence["status"],
            "deployment_evidence_sha256": identity,
            "container_id": evidence["container_id"],
            "image_id": evidence["image_id"],
        }
    elif args.command == "validate-deployment":
        evidence = validate_deployment_evidence(args.deployment_evidence, args.expected_sha256)
        value = {
            "status": "valid",
            "deployment_evidence_sha256": args.expected_sha256,
            "container_id": evidence["container_id"],
            "image_id": evidence["image_id"],
        }
    elif args.command == "run-fixture":
        evidence, identity = run_sparse_fixture(
            args.output,
            predeclaration_path=args.predeclaration,
            predeclaration_sha256=args.predeclaration_sha256,
            deployment_evidence_path=args.deployment_evidence,
            deployment_evidence_sha256=args.deployment_evidence_sha256,
        )
        value = {
            "status": evidence["status"],
            "fixture_evidence_sha256": identity,
            "algorithms": list(evidence["algorithms"]),
        }
    elif args.command == "finalize-fixture":
        evidence, identity = finalize_fixture_evidence(args.fixture_evidence, args.expected_sha256)
        value = {
            "status": evidence["status"],
            "fixture_evidence_sha256": identity,
            "sealed_indexed_segments_only": evidence["validation"]["sealed_indexed_segments_only"],
            "algorithms": {
                name: row["reported_algorithm"] for name, row in evidence["validation"]["algorithms"].items()
            },
        }
    elif args.command == "validate-fixture":
        evidence = validate_fixture_evidence(args.fixture_evidence, args.expected_sha256)
        value = {
            "status": "valid",
            "fixture_evidence_sha256": args.expected_sha256,
            "sealed_indexed_segments_only": evidence["validation"]["sealed_indexed_segments_only"],
        }
    elif args.command == "attach-compaction-evidence":
        evidence, identity = attach_compaction_evidence(args.fixture_evidence, args.expected_sha256)
        value = {
            "status": evidence["status"],
            "fixture_evidence_sha256": identity,
            "compaction": evidence["validation"]["compaction"],
        }
    elif args.command == "capture-recovery":
        evidence, identity = capture_recovery_evidence(
            args.output,
            predeclaration_path=args.predeclaration,
            predeclaration_sha256=args.predeclaration_sha256,
            fixture_evidence_path=args.fixture_evidence,
            fixture_evidence_sha256=args.fixture_evidence_sha256,
            hardened_deployment_path=args.hardened_deployment,
            hardened_deployment_sha256=args.hardened_deployment_sha256,
            before_stat_path=args.before_stat,
            after_stat_path=args.after_stat,
            prehardening_baseline_sha256=args.prehardening_baseline_sha256,
            stopped_baseline_sha256=args.stopped_baseline_sha256,
        )
        value = {
            "status": evidence["status"],
            "recovery_evidence_sha256": identity,
            "runtime_root_mode": evidence["runtime_root_mode"],
            "algorithms": {
                name: row["validation"]["reported_algorithm"] for name, row in evidence["algorithms"].items()
            },
        }
    elif args.command == "validate-recovery":
        evidence = validate_recovery_evidence(args.recovery_evidence, args.expected_sha256)
        value = {
            "status": "valid",
            "recovery_evidence_sha256": args.expected_sha256,
            "runtime_root_mode": evidence["runtime_root_mode"],
        }
    elif args.command == "compare-isolation":
        evidence, identity = compare_isolation_baselines(
            prelaunch_manifest_path=args.prelaunch_manifest,
            prelaunch_manifest_sha256=args.prelaunch_sha256,
            postfixture_manifest_path=args.postfixture_manifest,
            postfixture_manifest_sha256=args.postfixture_sha256,
            output=args.output,
        )
        value = {
            "status": evidence["status"],
            "isolation_regression_sha256": identity,
            "checks": evidence["checks"],
        }
    elif args.command == "compare-recovery-isolation":
        evidence, identity = compare_recovery_baselines(
            prehardening_manifest_path=args.prehardening_manifest,
            prehardening_manifest_sha256=args.prehardening_sha256,
            stopped_manifest_path=args.stopped_manifest,
            stopped_manifest_sha256=args.stopped_sha256,
            posthardening_manifest_path=args.posthardening_manifest,
            posthardening_manifest_sha256=args.posthardening_sha256,
            output=args.output,
        )
        value = {
            "status": evidence["status"],
            "recovery_isolation_sha256": identity,
            "checks": evidence["checks"],
        }
    elif args.command == "validate-baseline":
        predeclaration = validate_predeclaration(args.predeclaration, args.predeclaration_sha256)
        manifest = validate_baseline(args.manifest, args.expected_sha256, predeclaration)
        value = {
            "status": "valid",
            "baseline_manifest_sha256": args.expected_sha256,
            "captured_at_utc": manifest["captured_at_utc"],
            "related_container_count": manifest["related_container_count"],
        }
    elif args.command == "build-source-catalog":
        value = build_csr_exact_catalog(
            input_contract_path=args.input_contract,
            batch_a_root=args.batch_a_root,
            batch_b_root=args.batch_b_root,
        )
        value = {"status": "pass", "catalog_sha256": write_identity_artifact(args.output, value)}
    elif args.command == "validate-predeclaration":
        validated = validate_predeclaration(args.predeclaration, args.expected_sha256)
        value = {"status": "valid", "schema_version": validated["schema_version"], "sha256": args.expected_sha256}
    elif args.command == "validate-source-attestation":
        validated = validate_source_attestation(args.attestation, args.expected_sha256)
        value = {"status": "valid", "raw_cells": validated["raw_cells"], "sha256": args.expected_sha256}
    else:
        value = validate_formal_gate(
            predeclaration_path=args.predeclaration,
            predeclaration_sha256=args.predeclaration_sha256,
            source_attestation_path=args.source_attestation,
            source_attestation_sha256=args.source_attestation_sha256,
            baseline_manifest_path=args.baseline_manifest,
            baseline_manifest_sha256=args.baseline_manifest_sha256,
            fixture_evidence_path=args.fixture_evidence,
            fixture_evidence_sha256=args.fixture_evidence_sha256,
            deployment_evidence_path=args.deployment_evidence,
            deployment_evidence_sha256=args.deployment_evidence_sha256,
            postfixture_manifest_path=args.postfixture_baseline_manifest,
            postfixture_manifest_sha256=args.postfixture_baseline_sha256,
            isolation_regression_path=args.isolation_regression,
            isolation_regression_sha256=args.isolation_regression_sha256,
            hardened_deployment_evidence_path=args.hardened_deployment_evidence,
            hardened_deployment_evidence_sha256=args.hardened_deployment_evidence_sha256,
            recovery_evidence_path=args.recovery_evidence,
            recovery_evidence_sha256=args.recovery_evidence_sha256,
            prehardening_manifest_path=args.prehardening_baseline_manifest,
            prehardening_manifest_sha256=args.prehardening_baseline_sha256,
            stopped_manifest_path=args.stopped_baseline_manifest,
            stopped_manifest_sha256=args.stopped_baseline_sha256,
            posthardening_manifest_path=args.posthardening_baseline_manifest,
            posthardening_manifest_sha256=args.posthardening_baseline_sha256,
            recovery_isolation_path=args.recovery_isolation,
            recovery_isolation_sha256=args.recovery_isolation_sha256,
        )
        value["formal_gate_sha256"] = write_identity_artifact(args.output, value)
    _print(value)


if __name__ == "__main__":
    main()
