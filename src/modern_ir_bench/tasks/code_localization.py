"""Repository issue-to-file localization task."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import Dataset, Features, Sequence, Value

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.core.task import Task
from modern_ir_bench.retrieval import MappedResourceSource


class CodeLocalization(Task):
    """Retrieve files that need to be inspected for a repository issue."""

    dataset_features: ClassVar[dict[str, Features]] = {
        "files": Features(
            {
                "file_id": Value("string"),
                "source": Value("string"),
            }
        ),
        "issues": Features(
            {
                "issue_id": Value("string"),
                "issue": Value("string"),
                "target_file_ids": Sequence(Value("string")),
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
            raise ValueError("Code localization data requires files and issues tables")
        for table_name, expected_features in self.dataset_features.items():
            table = dataset[table_name]
            if table.features != expected_features:
                raise ValueError(f"Unexpected features for {table_name}: {table.features}")
        file_ids = set(dataset["files"]["file_id"])
        issue_ids = dataset["issues"]["issue_id"]
        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError("issue_id values must be unique")
        for target_ids in dataset["issues"]["target_file_ids"]:
            if not target_ids or not set(target_ids).issubset(file_ids):
                raise ValueError("Each issue must reference existing files")

    def evaluate(self, *, dataset: Any, solution: Solution) -> Dataset:
        resources = MappedResourceSource(
            dataset["files"],
            id_of=lambda row: row["file_id"],
            value_of=lambda row: row["source"],
        )
        searcher = solution.prepare(resources)
        try:
            observations = []
            for batch in dataset["issues"].iter(batch_size=self.query_batch_size):
                results = searcher.search_batch(
                    batch["issue"],
                    top_k=self.top_k,
                )
                for issue_id, target_ids, hits in zip(
                    batch["issue_id"],
                    batch["target_file_ids"],
                    results,
                    strict=True,
                ):
                    observations.append(
                        {
                            "issue_id": issue_id,
                            "expected_ids": target_ids,
                            "ranked_ids": [hit.id for hit in hits],
                            "scores": [hit.score for hit in hits],
                        }
                    )
        finally:
            searcher.close()
        return Dataset.from_list(observations)
