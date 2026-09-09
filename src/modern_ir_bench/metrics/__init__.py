"""Built-in metrics."""

from modern_ir_bench.metrics.ranking import (
    NDCG,
    MeanReciprocalRank,
    Recall,
    Success,
)

__all__ = ["NDCG", "MeanReciprocalRank", "Recall", "Success"]
