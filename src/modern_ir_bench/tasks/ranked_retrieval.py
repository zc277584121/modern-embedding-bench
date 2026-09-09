"""Ranked retrieval over a task-owned three-table dataset contract."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

from datasets import Dataset, Features, Value

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.core.task import Task
from modern_ir_bench.retrieval import MappedResourceSource


class RankedRetrieval(Task):
    """Retrieve relevant documents from one independent candidate pool."""

    dataset_features: ClassVar[dict[str, Features]] = {
        "documents": Features(
            {
                "document_id": Value("string"),
                "content": Value("string"),
            }
        ),
        "queries": Features(
            {
                "query_id": Value("string"),
                "query": Value("string"),
            }
        ),
        "qrels": Features(
            {
                "query_id": Value("string"),
                "document_id": Value("string"),
                "relevance": Value("int32"),
            }
        ),
    }

    def __init__(
        self,
        *,
        top_k: int = 100,
        query_batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        self.top_k = top_k
        self.query_batch_size = query_batch_size

    def validate_dataset(self, dataset: Any) -> None:
        if set(dataset) != set(self.dataset_features):
            raise ValueError("Ranked retrieval data requires documents, queries, and qrels tables")
        for table_name, expected_features in self.dataset_features.items():
            table = dataset[table_name]
            if table.features != expected_features:
                raise ValueError(f"Unexpected features for {table_name}: {table.features}")

        document_ids = dataset["documents"]["document_id"]
        query_ids = dataset["queries"]["query_id"]
        if not document_ids or len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be non-empty and unique")
        if not query_ids or len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be non-empty and unique")
        if any(not value for value in dataset["documents"]["content"]):
            raise ValueError("document content must be non-empty")
        if any(not value for value in dataset["queries"]["query"]):
            raise ValueError("query text must be non-empty")

        document_id_set = set(document_ids)
        query_id_set = set(query_ids)
        qrel_pairs: set[tuple[str, str]] = set()
        covered_queries: set[str] = set()
        for row in dataset["qrels"]:
            pair = (row["query_id"], row["document_id"])
            if pair in qrel_pairs:
                raise ValueError(f"Duplicate qrel: {pair}")
            if row["query_id"] not in query_id_set or row["document_id"] not in document_id_set:
                raise ValueError(f"Dangling qrel: {pair}")
            if row["relevance"] <= 0:
                raise ValueError(f"Qrel relevance must be positive: {pair}")
            qrel_pairs.add(pair)
            covered_queries.add(row["query_id"])
        if covered_queries != query_id_set:
            raise ValueError("Every query must have at least one positive qrel")

    def evaluate(self, *, dataset: Any, solution: Solution) -> Dataset:
        expected: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for row in dataset["qrels"]:
            expected[row["query_id"]].append((row["document_id"], row["relevance"]))

        resources = MappedResourceSource(
            dataset["documents"],
            id_of=lambda row: row["document_id"],
            value_of=lambda row: row["content"],
        )
        document_ids = set(dataset["documents"]["document_id"])
        searcher = solution.prepare(resources)
        observations = []
        try:
            for batch in dataset["queries"].iter(batch_size=self.query_batch_size):
                results = searcher.search_batch(batch["query"], top_k=self.top_k)
                if len(results) != len(batch["query_id"]):
                    raise RuntimeError("Solution returned a different number of rankings than queries")
                for query_id, hits in zip(batch["query_id"], results, strict=True):
                    ranked_ids = [hit.id for hit in hits]
                    if len(ranked_ids) != len(set(ranked_ids)):
                        raise RuntimeError(f"Solution returned duplicate hits for query {query_id}")
                    if not set(ranked_ids).issubset(document_ids):
                        raise RuntimeError(f"Solution returned unknown hits for query {query_id}")
                    qrels = expected[query_id]
                    observations.append(
                        {
                            "query_id": query_id,
                            "expected_ids": [document_id for document_id, _ in qrels],
                            "expected_relevance": [relevance for _, relevance in qrels],
                            "ranked_ids": ranked_ids,
                            "scores": [hit.score for hit in hits],
                        }
                    )
        finally:
            searcher.close()
        return Dataset.from_list(observations)
