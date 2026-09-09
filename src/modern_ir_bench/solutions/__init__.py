"""Reference and demonstration solutions."""

from modern_ir_bench.retrieval import DenseRetrievalSolution
from modern_ir_bench.solutions.text import (
    BM25Solution,
    CharacterNGramSolution,
    HybridSolution,
)

__all__ = [
    "BM25Solution",
    "CharacterNGramSolution",
    "DenseRetrievalSolution",
    "HybridSolution",
]
