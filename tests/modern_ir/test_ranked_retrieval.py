from __future__ import annotations

from datasets import Dataset, DatasetDict

from modern_ir_bench import MetricSet, RunProvenance
from modern_ir_bench.metrics import NDCG, AveragePrecision, Recall
from modern_ir_bench.solutions import BM25Solution
from modern_ir_bench.tasks import RankedRetrieval


def dataset() -> DatasetDict:
    return DatasetDict(
        documents=Dataset.from_dict(
            {
                "document_id": ["d1", "d2", "d3"],
                "content": ["alpha beta", "gamma delta", "alpha gamma"],
            },
            features=RankedRetrieval.dataset_features["documents"],
        ),
        queries=Dataset.from_dict(
            {
                "query_id": ["q1", "q2"],
                "query": ["alpha beta", "gamma delta"],
            },
            features=RankedRetrieval.dataset_features["queries"],
        ),
        qrels=Dataset.from_dict(
            {
                "query_id": ["q1", "q2"],
                "document_id": ["d1", "d2"],
                "relevance": [1, 1],
            },
            features=RankedRetrieval.dataset_features["qrels"],
        ),
    )


def provenance() -> RunProvenance:
    return RunProvenance(
        source_commit="0123456789abcdef",
        source_module="tests.modern_ir.test_ranked_retrieval",
        source_path="tests/modern_ir/test_ranked_retrieval.py",
        source_line=1,
        repository_url="https://github.com/example/modern-ir-bench",
        created_at="2026-09-09T00:00:00+00:00",
    )


def test_ranked_retrieval_uses_three_table_hf_dataset_contract() -> None:
    task = RankedRetrieval(
        id="ranked-retrieval-test",
        title="Ranked Retrieval Test",
        description="Test task",
        version="1",
        datasets={"tiny": dataset()},
        dataset_versions={"tiny": "1"},
        metrics=MetricSet(
            primary=NDCG(k=2),
            secondary=[AveragePrecision(k=2), Recall(k=2)],
        ),
        top_k=2,
        query_batch_size=1,
    )
    report = task.run(
        [BM25Solution(id="bm25", title="BM25")],
        provenance=provenance(),
        status="test",
    )

    assert {record["value"] for record in report.records} == {1.0}
    observation = report.observations["ranked-retrieval-test__tiny__bm25"]
    assert observation.column_names == [
        "query_id",
        "expected_ids",
        "expected_relevance",
        "ranked_ids",
        "scores",
    ]
