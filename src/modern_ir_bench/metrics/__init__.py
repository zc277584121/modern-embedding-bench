"""Built-in metrics."""

from modern_ir_bench.metrics.ranking import (
    NDCG,
    AveragePrecision,
    MeanReciprocalRank,
    Recall,
    Success,
)

__all__ = [
    "NDCG",
    "AveragePrecision",
    "MeanReciprocalRank",
    "Recall",
    "Success",
]
