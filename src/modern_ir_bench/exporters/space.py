"""Export a run report for the read-only Hugging Face Space."""

from __future__ import annotations

import json
from pathlib import Path

from modern_ir_bench.core.result import RunReport


def build_space_payload(
    report: RunReport,
    *,
    release: str,
    notice: str,
) -> dict[str, object]:
    return {
        "benchmark": {
            "title": "Modern IR Bench",
            "release": release,
            "notice": notice,
        },
        "tasks": sorted(report.tasks.values(), key=lambda item: item["id"]),
        "solutions": sorted(report.solutions.values(), key=lambda item: item["id"]),
        "results": report.records,
    }


def write_space_results(
    report: RunReport,
    path: Path,
    *,
    release: str,
    notice: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_space_payload(report, release=release, notice=notice),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
