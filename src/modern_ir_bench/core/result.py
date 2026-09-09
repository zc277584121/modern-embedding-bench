"""Long-form benchmark results independent of presentation layers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from datasets import Dataset

from modern_ir_bench.core.metric import MetricValue
from modern_ir_bench.core.provenance import RunProvenance


class RunReport:
    """Task observations plus long-form aggregate measurements."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self.observations: dict[str, Dataset] = {}
        self.tasks: dict[str, dict[str, str]] = {}
        self.solutions: dict[str, dict[str, str]] = {}

    def add_evaluation(
        self,
        *,
        task_id: str,
        task_title: str,
        task_description: str,
        task_version: str,
        dataset_id: str,
        dataset_version: str,
        solution_id: str,
        solution_title: str,
        solution_description: str,
        primary_metric_id: str,
        metrics: Iterable[MetricValue],
        observations: Dataset,
        provenance: RunProvenance,
        status: str,
    ) -> None:
        self.tasks[task_id] = {
            "id": task_id,
            "title": task_title,
            "description": task_description,
            "version": task_version,
            "primary_metric": primary_metric_id,
        }
        self.solutions[solution_id] = {
            "id": solution_id,
            "title": solution_title,
            "description": solution_description,
        }
        observation_key = f"{task_id}__{dataset_id}__{solution_id}"
        self.observations[observation_key] = observations

        for metric in metrics:
            self.records.append(
                {
                    "task_id": task_id,
                    "task_version": task_version,
                    "dataset_id": dataset_id,
                    "dataset_version": dataset_version,
                    "solution_id": solution_id,
                    "metric_id": metric.metric_id,
                    "metric_label": metric.label,
                    "value": round(metric.value, 8),
                    "primary": metric.metric_id == primary_metric_id,
                    "status": status,
                    "source_commit": provenance.source_commit,
                    "source_module": provenance.source_module,
                    "source_path": provenance.source_path,
                    "source_line": provenance.source_line,
                    "source_url": provenance.source_url,
                    "created_at": provenance.created_at,
                    "observation_key": observation_key,
                }
            )

    def extend(self, other: RunReport) -> None:
        self.records.extend(other.records)
        self.observations.update(other.observations)
        self.tasks.update(other.tasks)
        self.solutions.update(other.solutions)

    def write_observations(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for key, observations in self.observations.items():
            observations.to_parquet(directory / f"{key}.parquet")
