"""Print a compact summary of GeeVec result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geevec"
KEY_METRICS = [
    "spearman_dim_4096",
    "spearman_dim_1024",
    "spearman_dim_512",
    "min_viable_dim",
    "overall_accuracy",
    "avg_recall@1",
    "hard_avg_recall@1",
    "zh2en_recall@1",
    "en2zh_recall@1",
    "cross_lingual_similarity",
    "cross_lingual_min_similarity",
    "overall_t2t_precision@3",
    "weather_t2t_precision@3",
    "road_type_t2t_precision@3",
    "hazard_t2t_precision@3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Result JSON files. Defaults to results/geevec/*.json.")
    return parser.parse_args()


def load_entries(paths: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if isinstance(row, dict):
                row["_source"] = str(path)
                entries.append(row)
    return entries


def metric_summary(metrics: dict[str, Any]) -> str:
    parts = []
    for key in KEY_METRICS:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "-"


def main() -> None:
    args = parse_args()
    paths = args.paths or sorted(RESULTS_DIR.glob("*.json"))
    if not paths:
        print(f"No result JSON files found under {RESULTS_DIR}")
        return

    entries = load_entries(paths)
    if not entries:
        print("No result entries found.")
        return

    print(f"{'provider':<12} {'domain':<10} {'task':<24} {'status':<8} metrics")
    print("-" * 100)
    for row in entries:
        provider = row.get("provider", "?")
        domain = row.get("domain", "-")
        task = row.get("task", "?")
        status = "ERROR" if row.get("error") else "OK"
        metrics = row.get("metrics") or {}
        print(f"{provider:<12} {domain:<10} {task:<24} {status:<8} {metric_summary(metrics)}")


if __name__ == "__main__":
    main()
