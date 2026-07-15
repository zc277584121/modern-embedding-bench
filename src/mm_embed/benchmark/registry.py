"""Data-backed benchmark registry.

The v2 benchmark layer keeps model/task/run definitions in YAML so new models
can be reviewed and added without editing Python code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "benchmark"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


@dataclass(frozen=True)
class ModelSpec:
    """A reviewable model entry from benchmark/models/*.yaml."""

    id: str
    display_name: str
    provider: str
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=list)
    dimensions: int | None = None
    max_text_length: int | None = None
    supports_mrl: bool = False
    access: str = "unknown"
    api_key_env: str | None = None
    status: str = "active"
    publish: bool = True
    priority: int = 100
    tags: list[str] = field(default_factory=list)
    source: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_file: Path) -> "ModelSpec":
        required = ("id", "display_name", "provider")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Model entry in {source_file} is missing: {', '.join(missing)}")
        return cls(
            id=str(data["id"]),
            display_name=str(data["display_name"]),
            provider=str(data["provider"]),
            provider_kwargs=dict(data.get("provider_kwargs") or {}),
            modalities=list(data.get("modalities") or []),
            dimensions=data.get("dimensions"),
            max_text_length=data.get("max_text_length"),
            supports_mrl=bool(data.get("supports_mrl", False)),
            access=str(data.get("access", "unknown")),
            api_key_env=data.get("api_key_env"),
            status=str(data.get("status", "active")),
            publish=bool(data.get("publish", True)),
            priority=int(data.get("priority", 100)),
            tags=list(data.get("tags") or []),
            source=data.get("source"),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class TaskSpec:
    """A reviewable task entry from benchmark/tasks/*.yaml."""

    id: str
    display_name: str
    task: str
    description: str
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    required_modalities: list[str] = field(default_factory=list)
    primary_metric: str | None = None
    metric_direction: str = "higher"
    dataset_version: str = "unknown"
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_file: Path) -> "TaskSpec":
        required = ("id", "display_name", "task", "description")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Task entry in {source_file} is missing: {', '.join(missing)}")
        return cls(
            id=str(data["id"]),
            display_name=str(data["display_name"]),
            task=str(data["task"]),
            description=str(data["description"]),
            default_kwargs=dict(data.get("default_kwargs") or {}),
            required_modalities=list(data.get("required_modalities") or []),
            primary_metric=data.get("primary_metric"),
            metric_direction=str(data.get("metric_direction", "higher")),
            dataset_version=str(data.get("dataset_version", "unknown")),
            tags=list(data.get("tags") or []),
        )


@dataclass(frozen=True)
class RunTask:
    """A task selected by a run manifest."""

    id: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | dict[str, Any]) -> "RunTask":
        if isinstance(value, str):
            return cls(id=value)
        if isinstance(value, dict):
            task_id = value.get("id") or value.get("task")
            if not task_id:
                raise ValueError(f"Run task entry is missing id/task: {value}")
            return cls(id=str(task_id), kwargs=dict(value.get("kwargs") or {}))
        raise ValueError(f"Unsupported run task entry: {value!r}")


@dataclass(frozen=True)
class RunManifest:
    """A concrete benchmark run plan."""

    id: str
    description: str = ""
    model_ids: list[str] = field(default_factory=list)
    tasks: list[RunTask] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCatalog:
    """Loaded model and task registries."""

    root: Path
    models: dict[str, ModelSpec]
    tasks: dict[str, TaskSpec]

    def require_model(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.models))
            raise KeyError(f"Unknown model spec '{model_id}'. Available: {available}") from exc

    def require_task(self, task_id: str) -> TaskSpec:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.tasks))
            raise KeyError(f"Unknown task spec '{task_id}'. Available: {available}") from exc


def load_catalog(root: str | Path | None = None) -> BenchmarkCatalog:
    """Load all model and task specs under a benchmark root."""
    benchmark_root = Path(root) if root else DEFAULT_BENCHMARK_ROOT
    model_files = sorted((benchmark_root / "models").glob("*.yaml"))
    task_files = sorted((benchmark_root / "tasks").glob("*.yaml"))

    models: dict[str, ModelSpec] = {}
    for path in model_files:
        for row in _read_yaml(path).get("models", []):
            spec = ModelSpec.from_dict(row, path)
            if spec.id in models:
                raise ValueError(f"Duplicate model id '{spec.id}' in {path}")
            models[spec.id] = spec

    tasks: dict[str, TaskSpec] = {}
    for path in task_files:
        for row in _read_yaml(path).get("tasks", []):
            spec = TaskSpec.from_dict(row, path)
            if spec.id in tasks:
                raise ValueError(f"Duplicate task id '{spec.id}' in {path}")
            tasks[spec.id] = spec

    return BenchmarkCatalog(root=benchmark_root, models=models, tasks=tasks)


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load a benchmark run manifest."""
    run_path = Path(path)
    data = _read_yaml(run_path)
    if "id" not in data:
        raise ValueError(f"Run manifest {run_path} is missing id")
    return RunManifest(
        id=str(data["id"]),
        description=str(data.get("description", "")),
        model_ids=[str(item) for item in data.get("models", [])],
        tasks=[RunTask.from_value(item) for item in data.get("tasks", [])],
        metadata=dict(data.get("metadata") or {}),
    )
