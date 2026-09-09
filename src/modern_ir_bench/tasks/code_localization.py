"""Repository issue-to-file localization task."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import Dataset, Features, Sequence, Value

from modern_ir_bench.solution import Solution
from modern_ir_bench.task import Task


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

    def __init__(self, *, top_k: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

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
        searcher = solution.prepare(
            dataset["files"],
            id_field="file_id",
            text_field="source",
        )
        observations = []
        for batch in self.runtime.batch_rows(dataset["issues"]):
            results = searcher.search_batch(
                [row["issue"] for row in batch],
                top_k=self.top_k,
            )
            for row, hits in zip(batch, results, strict=True):
                observations.append(
                    {
                        "issue_id": row["issue_id"],
                        "expected_ids": row["target_file_ids"],
                        "ranked_ids": [hit["id"] for hit in hits],
                        "scores": [hit["score"] for hit in hits],
                    }
                )
        return Dataset.from_list(observations)
