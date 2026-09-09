"""Task orchestration without a global dataset or solution protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from datasets import Dataset

from modern_ir_bench.core.metric import MetricSet
from modern_ir_bench.core.provenance import RunProvenance
from modern_ir_bench.core.result import RunReport
from modern_ir_bench.core.solution import Solution


class Task(ABC):
    """A modern IR evaluation problem with task-specific semantics."""

    def __init__(
        self,
        *,
        id: str,
        title: str,
        description: str,
        version: str,
        datasets: Mapping[str, Any],
        dataset_versions: Mapping[str, str],
        metrics: MetricSet,
    ) -> None:
        if set(datasets) != set(dataset_versions):
            raise ValueError("Each dataset must have exactly one dataset version")
        self.id = id
        self.title = title
        self.description = description
        self.version = version
        self.datasets = dict(datasets)
        self.dataset_versions = dict(dataset_versions)
        self.metrics = metrics

    def run(
        self,
        solutions: Sequence[Solution],
        *,
        provenance: RunProvenance,
        status: str = "official",
    ) -> RunReport:
        report = RunReport()
        for dataset_id, dataset in self.datasets.items():
            self.validate_dataset(dataset)
            for solution in solutions:
                observations = self.evaluate(dataset=dataset, solution=solution)
                report.add_evaluation(
                    task_id=self.id,
                    task_title=self.title,
                    task_description=self.description,
                    task_version=self.version,
                    dataset_id=dataset_id,
                    dataset_version=self.dataset_versions[dataset_id],
                    solution_id=solution.id,
                    solution_title=solution.title,
                    solution_description=solution.description,
                    primary_metric_id=self.metrics.primary.id,
                    metrics=self.metrics.evaluate(observations),
                    observations=observations,
                    provenance=provenance,
                    status=status,
                )
        return report

    @abstractmethod
    def validate_dataset(self, dataset: Any) -> None:
        """Validate the task-specific dataset contract."""

    @abstractmethod
    def evaluate(self, *, dataset: Any, solution: Solution) -> Dataset:
        """Run one solution and return task-specific observations."""
