"""Dense index backends and independent verification."""

from modern_ir_bench.retrieval.indexes.flat import NumpyFlatIndex
from modern_ir_bench.retrieval.indexes.milvus import (
    MilvusDenseIndex,
    MilvusLite,
    MilvusServer,
    ZillizCloud,
)
from modern_ir_bench.retrieval.indexes.protocols import DenseIndex, DenseIndexSession
from modern_ir_bench.retrieval.indexes.verified import (
    DenseIndexAudit,
    VerifiedDenseIndex,
)

__all__ = [
    "DenseIndex",
    "DenseIndexAudit",
    "DenseIndexSession",
    "MilvusDenseIndex",
    "MilvusLite",
    "MilvusServer",
    "NumpyFlatIndex",
    "VerifiedDenseIndex",
    "ZillizCloud",
]
