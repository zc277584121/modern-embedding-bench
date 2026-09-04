#!/usr/bin/env python3
"""Exercise real S-20260814-012 tamper rejection in a reflink copy-on-write tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from mm_embed.providers.real_multi_vector import file_sha256


def formatted(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def flip_last_byte(path: Path) -> int:
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))
    return original[0]


def restore_last_byte(path: Path, value: int) -> None:
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(bytes([value]))


def make_shadow(source: Path, target: Path, private_copies: set[str]) -> None:
    """Build an isolated copy-on-write shadow using links plus private mutation copies."""
    target.mkdir(parents=True)
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        target_path = target / relative
        if source_path.is_dir():
            target_path.mkdir()
        elif relative.as_posix() in private_copies:
            target_path.write_bytes(source_path.read_bytes())
        else:
            target_path.symlink_to(source_path.resolve())


def expect_rejection(command: list[str], diagnostic: str) -> dict[str, object]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    combined = result.stdout + result.stderr
    if result.returncode == 0 or diagnostic not in combined:
        raise RuntimeError(f"Expected tamper rejection was not observed: {diagnostic}")
    return {"status": "rejected", "exit_code": result.returncode, "diagnostic": diagnostic}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/bright-multi-vector-v0.2/tamper-evidence.json"))
    args = parser.parse_args()
    repo = Path.cwd()
    source_results = repo / "results/bright-multi-vector-v0.2"
    source_package = repo / "benchmark/artifacts/bright-multi-vector-results-v0.2"
    source_report = repo / "benchmark/research/bright_multi_vector_v02_20260904.md"
    if not source_package.is_dir():
        raise RuntimeError("Build the candidate package before running real tamper checks")
    with tempfile.TemporaryDirectory(prefix=".bright-mv-tamper-", dir=repo / "results") as temporary:
        root = Path(temporary)
        copied_results = root / "results"
        copied_package = root / "package"
        copied_report = root / "report.md"
        copied_local = root / "case-mapping.json"
        copied_results.mkdir()
        make_shadow(
            source_results / "representations",
            copied_results / "representations",
            {"colbert-v2/economics/documents.values.npy"},
        )
        make_shadow(
            source_results / "cells",
            copied_results / "cells",
            {"colbert-v2-economics.json"},
        )
        make_shadow(source_package, copied_package, {"summary.json"})
        copied_report.symlink_to(source_report.resolve())
        copied_report.with_suffix(".md.sha256").symlink_to(source_report.with_suffix(".md.sha256").resolve())
        copied_local.symlink_to((source_results / "case-mapping.json").resolve())
        copied_local.with_suffix(".json.sha256").symlink_to((source_results / "case-mapping.json.sha256").resolve())

        check = [
            sys.executable,
            "scripts/bright_multi_vector.py",
            "check-package",
            "--cell-root",
            str(copied_results / "cells"),
            "--package-root",
            str(copied_package),
            "--report",
            str(copied_report),
            "--local-mapping",
            str(copied_local),
        ]
        attacks: dict[str, object] = {}
        representation = copied_results / "representations/colbert-v2/economics/documents.values.npy"
        original = flip_last_byte(representation)
        try:
            attacks["representation_byte"] = expect_rejection(check, "Saved representation file identity drifted")
        finally:
            restore_last_byte(representation, original)

        cell = copied_results / "cells/colbert-v2-economics.json"
        original = flip_last_byte(cell)
        try:
            attacks["formal_cell_byte"] = expect_rejection(check, "Formal cell identity drifted")
        finally:
            restore_last_byte(cell, original)

        summary = copied_package / "summary.json"
        original = flip_last_byte(summary)
        try:
            attacks["package_byte"] = expect_rejection(check, "Tracked package drifted: summary.json")
        finally:
            restore_last_byte(summary, original)

        original = flip_last_byte(representation)
        try:
            attacks["resume_representation_byte"] = expect_rejection(
                [
                    sys.executable,
                    "scripts/bright_multi_vector.py",
                    "encode-model",
                    "--model",
                    "colbert-v2",
                    "--output-root",
                    str(copied_results / "representations"),
                ],
                "Saved representation file identity drifted",
            )
        finally:
            restore_last_byte(representation, original)

    evidence = {
        "schema_version": "bright-real-multi-vector-tamper-evidence-v1",
        "story_id": "S-20260814-012",
        "copy_mode": "isolated_symlink_shadow_with_private_mutation_copies",
        "source_package_manifest_sha256": file_sha256(source_package / "manifest.json"),
        "attacks": attacks,
        "real_sources_modified": False,
        "publication_gate": "closed",
    }
    payload = formatted(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {"attacks": len(attacks), "sha256": hashlib.sha256(payload).hexdigest(), "status": "pass"}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
