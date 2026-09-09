"""Ranking metrics over expected_ids and ranked_ids observations."""

from __future__ import annotations

import math

from datasets import Dataset

from modern_ir_bench.core.metric import Metric


class RankingMetric(Metric):
    required_columns = frozenset({"expected_ids", "ranked_ids"})

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k


class Recall(RankingMetric):
    def __init__(self, k: int) -> None:
        super().__init__(k)
        self.id = f"recall@{k}"
        self.label = f"Recall@{k}"

    def compute(self, observations: Dataset) -> float:
        values = []
        for row in observations:
            expected = set(row["expected_ids"])
            retrieved = set(row["ranked_ids"][: self.k])
            values.append(len(expected & retrieved) / len(expected))
        return sum(values) / len(values)


class Success(RankingMetric):
    def __init__(self, k: int) -> None:
        super().__init__(k)
        self.id = f"success@{k}"
        self.label = f"Success@{k}"

    def compute(self, observations: Dataset) -> float:
        values = []
        for row in observations:
            expected = set(row["expected_ids"])
            retrieved = set(row["ranked_ids"][: self.k])
            values.append(float(bool(expected & retrieved)))
        return sum(values) / len(values)


class MeanReciprocalRank(RankingMetric):
    def __init__(self, k: int) -> None:
        super().__init__(k)
        self.id = f"mrr@{k}"
        self.label = f"MRR@{k}"

    def compute(self, observations: Dataset) -> float:
        values = []
        for row in observations:
            expected = set(row["expected_ids"])
            reciprocal_rank = 0.0
            for rank, item_id in enumerate(row["ranked_ids"][: self.k], start=1):
                if item_id in expected:
                    reciprocal_rank = 1.0 / rank
                    break
            values.append(reciprocal_rank)
        return sum(values) / len(values)


class NDCG(RankingMetric):
    def __init__(self, k: int) -> None:
        super().__init__(k)
        self.id = f"ndcg@{k}"
        self.label = f"NDCG@{k}"

    def compute(self, observations: Dataset) -> float:
        values = []
        for row in observations:
            expected = set(row["expected_ids"])
            gains = [1.0 if item_id in expected else 0.0 for item_id in row["ranked_ids"][: self.k]]
            dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
            ideal_count = min(len(expected), self.k)
            ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            values.append(dcg / ideal)
        return sum(values) / len(values)
