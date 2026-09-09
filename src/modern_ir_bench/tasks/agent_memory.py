"""Agent memory retrieval task."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import Dataset, Features, Sequence, Value

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.core.task import Task
from modern_ir_bench.retrieval import MappedResourceSource


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

    def __init__(
        self,
        *,
        top_k: int = 3,
        query_batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        self.top_k = top_k
        self.query_batch_size = query_batch_size

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
        resources = MappedResourceSource(
            dataset["memories"],
            id_of=lambda row: row["memory_id"],
            value_of=lambda row: row["content"],
        )
        searcher = solution.prepare(resources)
        try:
            observations = []
            for batch in dataset["queries"].iter(batch_size=self.query_batch_size):
                results = searcher.search_batch(
                    batch["query"],
                    top_k=self.top_k,
                )
                for query_id, relevant_ids, hits in zip(
                    batch["query_id"],
                    batch["relevant_memory_ids"],
                    results,
                    strict=True,
                ):
                    observations.append(
                        {
                            "query_id": query_id,
                            "expected_ids": relevant_ids,
                            "ranked_ids": [hit.id for hit in hits],
                            "scores": [hit.score for hit in hits],
                        }
                    )
        finally:
            searcher.close()
        return Dataset.from_list(observations)
