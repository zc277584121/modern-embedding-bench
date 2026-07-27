"""Local fixture-only export for system evaluation records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mm_embed.system_evaluation.result_schema import validate_system_result
from mm_embed.system_evaluation.retrieval_answer_utility import (
    DATASET_VERSION,
    EVALUATION_LEVEL,
    EVALUATION_MODE,
    FAMILY,
    NETWORK,
    SCHEMA_VERSION,
    ContractValidationError,
    evaluate_fixture_brackets,
    load_retrieval_answer_utility_fixture,
)


DEFAULT_SYSTEM_EXPORT_ROOT = Path("dist/system-evaluation") / DATASET_VERSION
SYSTEM_RESULTS_FILENAME = "retrieval-answer-utility.system-results.fixture-only.jsonl"
SYSTEM_EXPORT_MANIFEST_FILENAME = "retrieval-answer-utility.system-export.fixture-only.json"
SYSTEM_EXPORT_OWNED_FILENAMES = frozenset({SYSTEM_RESULTS_FILENAME, SYSTEM_EXPORT_MANIFEST_FILENAME})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prepare_output_directory(output: Path) -> None:
    if output.is_symlink():
        raise ContractValidationError(
            "system_export_boundary",
            "System export output must be a real directory",
        )
    if not output.exists():
        output.mkdir(parents=True)
        return
    if not output.is_dir():
        raise ContractValidationError(
            "system_export_boundary",
            "System export output must be a real directory",
        )

    entries = sorted(output.iterdir(), key=lambda path: path.name)
    unexpected = [entry.name for entry in entries if entry.name not in SYSTEM_EXPORT_OWNED_FILENAMES]
    if unexpected:
        names = ", ".join(unexpected)
        raise ContractValidationError(
            "system_export_boundary",
            f"System export directory contains unowned entries: {names}",
        )
    invalid_owned = [entry.name for entry in entries if entry.is_symlink() or not entry.is_file()]
    if invalid_owned:
        names = ", ".join(invalid_owned)
        raise ContractValidationError(
            "system_export_boundary",
            f"System export owned paths must be regular files: {names}",
        )


def export_retrieval_answer_utility_fixture(
    output_dir: str | Path = DEFAULT_SYSTEM_EXPORT_ROOT,
) -> Path:
    """Write the three validated fixture runs to a system-only local directory."""
    fixture = load_retrieval_answer_utility_fixture()
    runs = evaluate_fixture_brackets(fixture)["runs"]
    records = [runs[system_id] for system_id in sorted(runs)]
    for record in records:
        validate_system_result(record, fixture)

    results_bytes = "".join(f"{_canonical_json(record)}\n" for record in records).encode("utf-8")
    manifest = {
        "export_kind": "system_evaluation_fixture",
        "schema_version": SCHEMA_VERSION,
        "evaluation": {
            "family": FAMILY,
            "level": EVALUATION_LEVEL,
            "mode": EVALUATION_MODE,
            "leaderboard_surface": "system",
        },
        "fixture": {
            "dataset_id": DATASET_VERSION,
            "bundle_sha256": fixture.bundle_sha256,
            "fixture_only": True,
            "publish": False,
            "network": NETWORK,
        },
        "record_count": len(records),
        "record_order": [record["subject"]["id"] for record in records],
        "files": {
            SYSTEM_RESULTS_FILENAME: {
                "sha256": _sha256_bytes(results_bytes),
                "records": len(records),
            }
        },
        "publish": False,
    }
    manifest_bytes = f"{_canonical_json(manifest)}\n".encode("utf-8")

    output = Path(output_dir)
    _prepare_output_directory(output)
    (output / SYSTEM_RESULTS_FILENAME).write_bytes(results_bytes)
    (output / SYSTEM_EXPORT_MANIFEST_FILENAME).write_bytes(manifest_bytes)
    return output
