"""Build leaderboard tables from v2 result JSONL files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from mm_embed.benchmark.registry import BenchmarkCatalog, load_catalog
from mm_embed.benchmark.results import load_jsonl


def primary_metric_value(record: dict[str, Any], catalog: BenchmarkCatalog | None = None) -> float | None:
    """Return the record's primary metric value, if available."""
    metrics = record.get("metrics") or {}
    task_info = record.get("task") or {}
    metric = task_info.get("primary_metric")
    if not metric and catalog:
        task_id = task_info.get("id")
        if task_id in catalog.tasks:
            metric = catalog.tasks[task_id].primary_metric
    if metric and metric in metrics:
        return metrics[metric]
    return _fallback_metric(task_info.get("id"), metrics)


def _fallback_metric(task_id: str | None, metrics: dict[str, Any]) -> float | None:
    if not metrics:
        return None
    if task_id == "mrl_stress":
        spearman_keys = sorted(
            [key for key in metrics if key.startswith("spearman_dim_")],
            key=lambda key: int(key.rsplit("_", 1)[-1]),
        )
        if spearman_keys:
            return metrics[spearman_keys[0]]
    for key in ("hard_avg_recall@1", "avg_recall@1", "overall_accuracy"):
        if key in metrics:
            return metrics[key]
    return None


def build_leaderboard(records: list[dict[str, Any]], catalog: BenchmarkCatalog | None = None) -> list[dict[str, Any]]:
    """Build a flat leaderboard table, one row per successful model-task result."""
    rows = []
    for record in records:
        if record.get("error"):
            continue
        if (record.get("run") or {}).get("publish") is False:
            continue
        task_info = record.get("task") or {}
        task_id = task_info.get("id")
        if task_info.get("publish") is False:
            continue
        if task_info.get("leaderboard_publish") is False:
            continue
        if catalog and task_id in catalog.tasks and not catalog.tasks[task_id].publish:
            continue
        if catalog and task_id in catalog.tasks and not catalog.tasks[task_id].leaderboard_publish:
            continue
        value = primary_metric_value(record, catalog)
        if value is None:
            continue
        model = record.get("model") or {}
        task = task_info
        metric = task.get("primary_metric")
        if not metric and catalog and task.get("id") in catalog.tasks:
            metric = catalog.tasks[task["id"]].primary_metric
        rows.append({
            "task_id": task.get("id"),
            "task": task.get("display_name") or task.get("id"),
            "model_id": model.get("id"),
            "model": model.get("display_name") or model.get("id"),
            "provider": model.get("provider"),
            "primary_metric": metric,
            "score": value,
            "run_id": (record.get("run") or {}).get("id"),
            "duration_s": (record.get("timestamps") or {}).get("duration_s"),
        })
    rows.sort(key=lambda row: (row["task_id"] or "", -float(row["score"])))
    return rows


def write_leaderboard_csv(rows: list[dict[str, Any]], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_id", "task", "model_id", "model", "provider", "primary_metric", "score", "run_id", "duration_s"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_from_file(
    results_path: str | Path,
    output: str | Path,
    benchmark_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    catalog = load_catalog(benchmark_root)
    rows = build_leaderboard(load_jsonl(results_path), catalog)
    write_leaderboard_csv(rows, output)
    return rows
