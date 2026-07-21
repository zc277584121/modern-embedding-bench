"""Task registry — lazy-load tasks."""

from __future__ import annotations

from typing import Any

from mm_embed.tasks.base import EvalTask

# Registry: name -> (module_path, class_name)
TASK_REGISTRY: dict[str, tuple[str, str]] = {
    "mrl_stress": ("mm_embed.tasks.mrl_stress", "MRLStressTask"),
    "cross_modal_retrieval": ("mm_embed.tasks.cross_modal_retrieval", "CrossModalRetrievalTask"),
    "needle_in_haystack": ("mm_embed.tasks.needle_in_haystack", "NeedleInHaystackTask"),
    "crosslingual_retrieval": ("mm_embed.tasks.crosslingual_retrieval", "CrossLingualRetrievalTask"),
    "autonomous_driving": ("mm_embed.tasks.autonomous_driving", "AutonomousDrivingTask"),
    "chinese_multimodal": ("mm_embed.tasks.chinese_multimodal", "ChineseMultimodalTask"),
    "agent_procedural_tool_memory": (
        "mm_embed.tasks.agent_procedural_tool_memory",
        "AgentProceduralToolMemoryTask",
    ),
    "agent_skill_compatible_set_retrieval": (
        "mm_embed.tasks.agent_skill_routing",
        "AgentSkillCompatibleSetRetrievalTask",
    ),
    "agent_skill_same_capability_risk": (
        "mm_embed.tasks.agent_skill_routing",
        "AgentSkillSameCapabilityRiskTask",
    ),
    "late_chunking_retrieval": (
        "mm_embed.tasks.late_chunking_retrieval",
        "LateChunkingRetrievalTask",
    ),
}


def get_task(name: str, **kwargs: Any) -> EvalTask:
    """Instantiate a task by name."""
    if name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise KeyError(f"Unknown task '{name}'. Available: {available}")

    module_path, class_name = TASK_REGISTRY[name]

    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
