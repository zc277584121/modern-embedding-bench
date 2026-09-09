from datasets import Dataset

from modern_ir_bench.metrics import (
    NDCG,
    AveragePrecision,
    MeanReciprocalRank,
    Recall,
    Success,
)


def observations() -> Dataset:
    return Dataset.from_list(
        [
            {
                "expected_ids": ["a"],
                "ranked_ids": ["a", "b", "c"],
            },
            {
                "expected_ids": ["d"],
                "ranked_ids": ["x", "d", "y"],
            },
        ]
    )


def test_ranking_metrics() -> None:
    data = observations()

    assert Recall(2)(data).value == 1.0
    assert AveragePrecision(3)(data).value == 0.75
    assert Success(1)(data).value == 0.5
    assert MeanReciprocalRank(3)(data).value == 0.75
    assert 0.8 < NDCG(3)(data).value < 0.82


def test_ndcg_uses_optional_graded_relevance() -> None:
    data = Dataset.from_list(
        [
            {
                "expected_ids": ["high", "low"],
                "expected_relevance": [2, 1],
                "ranked_ids": ["low", "high"],
            }
        ]
    )

    assert 0.79 < NDCG(2)(data).value < 0.80
