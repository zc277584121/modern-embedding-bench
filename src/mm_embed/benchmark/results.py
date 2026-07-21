"""Result records for benchmark v2."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.benchmark.registry import ModelSpec, RunManifest, RunTask, TaskSpec
from mm_embed.tasks.base import EvalResult

RESULT_SCHEMA_VERSION = "2.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def current_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    """Convert numpy/dataclass/path objects to JSON-safe values."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sanitized_provider_kwargs(model: ModelSpec) -> dict[str, Any]:
    """Keep model identity fields while avoiding accidental secret leakage."""
    hidden = {"api_key", "token", "password", "secret"}
    clean: dict[str, Any] = {}
    for key, value in model.provider_kwargs.items():
        if key.lower() in hidden:
            clean[key] = "<redacted>"
        else:
            clean[key] = value
    return clean


def make_result_record(
    *,
    run: RunManifest,
    model: ModelSpec,
    task: TaskSpec,
    run_task: RunTask,
    result: EvalResult,
    started_at: str,
    finished_at: str,
    duration_s: float,
    error: str | None = None,
) -> dict[str, Any]:
    """Build the canonical JSONL record for a model-task evaluation."""
    return json_safe({
        "schema_version": RESULT_SCHEMA_VERSION,
        "run": {
            "id": run.id,
            "description": run.description,
            "metadata": run.metadata,
            "git_sha": current_git_sha(),
        },
        "timestamps": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": round(duration_s, 3),
        },
        "model": {
            "id": model.id,
            "display_name": model.display_name,
            "provider": model.provider,
            "provider_kwargs": sanitized_provider_kwargs(model),
            "modalities": model.modalities,
            "dimensions": model.dimensions,
            "max_text_length": model.max_text_length,
            "supports_mrl": model.supports_mrl,
            "access": model.access,
            "source": model.source,
            "tags": model.tags,
        },
        "task": {
            "id": task.id,
            "display_name": task.display_name,
            "task": task.task,
            "dataset_version": task.dataset_version,
            "primary_metric": task.primary_metric,
            "metric_direction": task.metric_direction,
            "publish": task.publish,
            "kwargs": run_task.kwargs,
            "tags": task.tags,
        },
        "provider_result": {
            "provider": result.provider_name,
            "model_name": result.model_name,
        },
        "metrics": result.metrics,
        "details": result.details,
        "error": error if error is not None else result.error,
    })


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True))
        f.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def import_legacy_result_file(path: str | Path, output: str | Path) -> int:
    """Convert a legacy results/*.json file into loose v2 JSONL records."""
    input_path = Path(path)
    with open(input_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Legacy result file must contain a JSON list: {input_path}")

    count = 0
    for row in rows:
        model_name = normalize_legacy_model_name(row.get("model") or row.get("provider") or "unknown")
        record = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run": {
                "id": f"legacy:{input_path.stem}",
                "description": "Imported legacy result",
                "metadata": {"legacy_source": str(input_path)},
                "git_sha": None,
            },
            "timestamps": {
                "started_at": None,
                "finished_at": None,
                "duration_s": row.get("elapsed_s"),
            },
            "model": {
                "id": model_name,
                "display_name": model_name,
                "provider": row.get("provider") or "unknown",
                "provider_kwargs": {},
                "modalities": [],
                "dimensions": None,
                "max_text_length": None,
                "supports_mrl": None,
                "access": "legacy",
                "source": None,
                "tags": ["legacy"],
            },
            "task": {
                "id": row.get("task") or "unknown",
                "display_name": row.get("task") or "unknown",
                "task": row.get("task") or "unknown",
                "dataset_version": "legacy",
                "primary_metric": None,
                "metric_direction": "higher",
                "kwargs": {},
                "tags": ["legacy"],
            },
            "provider_result": {
                "provider": row.get("provider"),
                "model_name": model_name,
            },
            "metrics": row.get("metrics") or {},
            "details": row.get("details") or {},
            "error": row.get("error"),
        }
        append_jsonl(output, record)
        count += 1
    return count


def normalize_legacy_model_name(value: Any) -> str:
    """Avoid publishing local absolute paths while preserving public model ids."""
    name = str(value or "unknown")
    if os.path.isabs(name):
        return Path(name).name or "unknown"
    return name
