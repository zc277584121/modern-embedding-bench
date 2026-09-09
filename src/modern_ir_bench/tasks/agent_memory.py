"""Agent memory retrieval task."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import Dataset, Features, Sequence, Value

from modern_ir_bench.solution import Solution
from modern_ir_bench.task import Task


class AgentMemoryRetrieval(Task):
    """Retrieve the memories needed to answer a later user request."""

    dataset_features: ClassVar[dict[str, Features]] = {
        "memories": Features(
            {
                "memory_id": Value("string"),
                "content": Value("string"),
            }
        ),
        "queries": Features(
            {
                "query_id": Value("string"),
                "query": Value("string"),
                "relevant_memory_ids": Sequence(Value("string")),
            }
        ),
    }

    def __init__(self, *, top_k: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def validate_dataset(self, dataset: Any) -> None:
        if set(dataset) != set(self.dataset_features):
            raise ValueError("Agent memory data requires memories and queries tables")
        for table_name, expected_features in self.dataset_features.items():
            table = dataset[table_name]
            if table.features != expected_features:
                raise ValueError(f"Unexpected features for {table_name}: {table.features}")
        memory_ids = set(dataset["memories"]["memory_id"])
        query_ids = dataset["queries"]["query_id"]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique")
        for relevant_ids in dataset["queries"]["relevant_memory_ids"]:
            if not relevant_ids or not set(relevant_ids).issubset(memory_ids):
                raise ValueError("Each query must reference existing memories")

    def evaluate(self, *, dataset: Any, solution: Solution) -> Dataset:
        searcher = solution.prepare(
            dataset["memories"],
            id_field="memory_id",
            text_field="content",
        )
        observations = []
        for batch in self.runtime.batch_rows(dataset["queries"]):
            results = searcher.search_batch(
                [row["query"] for row in batch],
                top_k=self.top_k,
            )
            for row, hits in zip(batch, results, strict=True):
                observations.append(
                    {
                        "query_id": row["query_id"],
                        "expected_ids": row["relevant_memory_ids"],
                        "ranked_ids": [hit["id"] for hit in hits],
                        "scores": [hit["score"] for hit in hits],
                    }
                )
        return Dataset.from_list(observations)
