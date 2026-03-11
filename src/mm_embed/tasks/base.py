"""Base class for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mm_embed.providers.base import EmbeddingProvider


@dataclass
class EvalResult:
    """Result of a single evaluation run."""

    task_name: str
    provider_name: str
    model_name: str
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        lines = [f"[{self.task_name}] {self.provider_name}/{self.model_name}"]
        if self.error:
            lines.append(f"  ERROR: {self.error}")
        else:
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)


class EvalTask(ABC):
    """Abstract base class for evaluation tasks."""

    name: str = "base_task"
    description: str = ""
    required_modalities: set = set()

    @abstractmethod
    def run(self, provider: EmbeddingProvider, **kwargs: Any) -> EvalResult:
        """Run the evaluation task against a provider.

        Args:
            provider: The embedding provider to evaluate
            **kwargs: Task-specific parameters

        Returns:
            EvalResult with metrics and details
        """
        ...

    def check_compatibility(self, provider: EmbeddingProvider) -> bool:
        """Check if the provider supports the modalities this task requires."""
        return self.required_modalities.issubset(provider.supported_modalities)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
