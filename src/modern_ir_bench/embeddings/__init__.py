"""Embedding component contracts and implementations."""

from modern_ir_bench.embeddings.protocols import DenseEmbedding
from modern_ir_bench.embeddings.sentence_transformers import (
    SentenceTransformersEmbedding,
)

__all__ = ["DenseEmbedding", "SentenceTransformersEmbedding"]
