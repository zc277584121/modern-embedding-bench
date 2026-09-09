"""Built-in modern IR tasks."""

from modern_ir_bench.tasks.agent_memory import AgentMemoryRetrieval
from modern_ir_bench.tasks.code_localization import CodeLocalization
from modern_ir_bench.tasks.ranked_retrieval import RankedRetrieval

__all__ = ["AgentMemoryRetrieval", "CodeLocalization", "RankedRetrieval"]
