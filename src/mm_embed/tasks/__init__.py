"""Evaluation tasks."""

from mm_embed.tasks.base import EvalTask, EvalResult
from mm_embed.tasks.registry import TASK_REGISTRY, get_task

__all__ = ["EvalTask", "EvalResult", "TASK_REGISTRY", "get_task"]
