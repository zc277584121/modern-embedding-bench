"""Metric contracts and metric collections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from datasets import Dataset


@dataclass(frozen=True)
class MetricValue:
    """One aggregate metric value."""

    metric_id: str
    label: str
    value: float


class Metric(ABC):
    """A scorer over task-produced observations."""

    id: str
    label: str
    required_columns: frozenset[str] = frozenset()

    def __call__(self, observations: Dataset) -> MetricValue:
        missing = self.required_columns.difference(observations.column_names)
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"Metric {self.id} requires missing columns: {columns}")
        return MetricValue(
            metric_id=self.id,
            label=self.label,
            value=float(self.compute(observations)),
        )

    @abstractmethod
    def compute(self, observations: Dataset) -> float:
        """Compute an aggregate score from observations."""


class MetricSet:
    """Ordered task metrics with one primary metric."""

    def __init__(
        self,
        *,
        primary: Metric,
        secondary: Iterable[Metric] = (),
    ) -> None:
        self.primary = primary
        self._metrics = (primary, *tuple(secondary))
        ids = [metric.id for metric in self._metrics]
        if len(ids) != len(set(ids)):
            raise ValueError("Metric ids must be unique within a task")

    def __iter__(self):
        return iter(self._metrics)

    def evaluate(self, observations: Dataset) -> list[MetricValue]:
        return [metric(observations) for metric in self._metrics]
