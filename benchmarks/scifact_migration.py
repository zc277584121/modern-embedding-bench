"""Reproduce the legacy SciFact baselines through the Modern IR Bench framework.

This is migration evidence only. SciFact is a public BEIR dataset and these
measurements must not be added to the official Modern IR Bench leaderboard.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from modern_ir_bench import MetricSet, RunProvenance
from modern_ir_bench.embeddings import SentenceTransformersEmbedding
from modern_ir_bench.exporters import write_space_results
from modern_ir_bench.metrics import (
    NDCG,
    AveragePrecision,
    MeanReciprocalRank,
    Recall,
)
from modern_ir_bench.retrieval import DenseRetrievalSolution
from modern_ir_bench.retrieval.indexes import NumpyFlatIndex, VerifiedDenseIndex
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex, MilvusLite
from modern_ir_bench.solutions import BM25Solution
from modern_ir_bench.tasks import RankedRetrieval

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data" / "beir-three-track-v0.1"
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "migration-validation" / "scifact"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DATASET_VERSION = "beir-three-track-v0.1-scifact"
UNICODE_WORD_PATTERN = re.compile(r"(?u)\b\w+\b")

EXPECTED_METRICS = {
    "bm25-unicode-word": {
        "ndcg@10": 0.6634477520519494,
        "map@100": 0.6231540925452861,
        "mrr@10": 0.6308915343915344,
        "recall@10": 0.7876111111111112,
        "recall@100": 0.8825555555555555,
    },
    "minilm-milvus-lite": {
        "ndcg@10": 0.6450816521455774,
        "map@100": 0.6030740834806814,
        "mrr@10": 0.6047248677248678,
        "recall@10": 0.7833333333333333,
        "recall@100": 0.925,
    },
}


def unicode_word_lower(text: str) -> list[str]:
    """Legacy tokenizer kept explicit so the BM25 result is reproducible."""
    return UNICODE_WORD_PATTERN.findall(text.lower())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_path(root: Path, manifest: dict[str, Any], name: str) -> Path:
    expected = manifest["tracks"]["scifact"]["files"][name]
    path = root / "scifact" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing migration source data: {path}")
    if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
        raise ValueError(f"Migration source data failed identity validation: {path}")
    return path


def load_scifact_dataset(root: Path = DATA_ROOT) -> DatasetDict:
    """Adapt the fixed legacy materialization into the Task's HF Dataset contract."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_version") != "beir-three-track-v0.1":
        raise ValueError("Unexpected BEIR migration manifest version")

    documents = (
        Dataset.from_json(str(_validated_path(root, manifest, "corpus.jsonl")))
        .select_columns(["id", "content"])
        .rename_column("id", "document_id")
        .cast(RankedRetrieval.dataset_features["documents"])
    )
    queries = (
        Dataset.from_json(str(_validated_path(root, manifest, "queries.jsonl")))
        .select_columns(["id", "text"])
        .rename_columns({"id": "query_id", "text": "query"})
        .cast(RankedRetrieval.dataset_features["queries"])
    )
    qrels = (
        Dataset.from_json(str(_validated_path(root, manifest, "qrels.jsonl")))
        .rename_columns({"doc_id": "document_id", "grade": "relevance"})
        .cast(RankedRetrieval.dataset_features["qrels"])
    )
    return DatasetDict(documents=documents, queries=queries, qrels=qrels)


def build_task() -> RankedRetrieval:
    return RankedRetrieval(
        id="scifact-retrieval-migration",
        title="SciFact Retrieval (Migration Validation)",
        description="Legacy public data used only to verify the new framework against fixed historical results.",
        version="0.1.0-migration",
        datasets={"scifact": load_scifact_dataset()},
        dataset_versions={"scifact": DATASET_VERSION},
        metrics=MetricSet(
            primary=NDCG(k=10),
            secondary=[
                AveragePrecision(k=100),
                MeanReciprocalRank(k=10),
                Recall(k=10),
                Recall(k=100),
            ],
        ),
        top_k=100,
        query_batch_size=32,
    )


def build_solutions():
    bm25 = BM25Solution(
        id="bm25-unicode-word",
        title="BM25 (Unicode Word)",
        description="Exact legacy BM25 protocol with k1=1.2 and b=0.75.",
        k1=1.2,
        b=0.75,
        tokenizer=unicode_word_lower,
        tokenizer_id="unicode-word-lower-v1",
    )
    embedding = SentenceTransformersEmbedding(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        batch_size=128,
        local_files_only=True,
        trust_remote_code=False,
    )
    dense = DenseRetrievalSolution(
        id="minilm-milvus-lite",
        title="MiniLM + Milvus Lite FLAT",
        description="Fixed MiniLM snapshot indexed by Milvus Lite and shadowed by exact NumPy search.",
        embedding=embedding,
        index=VerifiedDenseIndex(
            primary=MilvusDenseIndex(
                target=MilvusLite(ARTIFACT_ROOT / "milvus.db"),
                metric="COSINE",
                index_type="FLAT",
                collection_prefix="scifact_migration",
            ),
            oracle=NumpyFlatIndex(metric="COSINE"),
            minimum_index_recall=1.0,
        ),
        document_batch_size=128,
    )
    return [bm25, dense]


def assert_historical_regression(records: list[dict[str, object]], tolerance: float = 1e-6) -> None:
    actual = {(str(record["solution_id"]), str(record["metric_id"])): float(record["value"]) for record in records}
    failures = []
    for solution_id, metrics in EXPECTED_METRICS.items():
        for metric_id, expected in metrics.items():
            value = actual[(solution_id, metric_id)]
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
                failures.append(f"{solution_id} {metric_id}: expected {expected:.8f}, got {value:.8f}")
    if failures:
        raise AssertionError("Historical regression failed:\n" + "\n".join(failures))


def _print_results(records: list[dict[str, object]]) -> None:
    metrics = ["ndcg@10", "map@100", "mrr@10", "recall@10", "recall@100"]
    by_solution = {
        solution_id: {
            str(record["metric_id"]): float(record["value"])
            for record in records
            if record["solution_id"] == solution_id
        }
        for solution_id in EXPECTED_METRICS
    }
    print("solution\t" + "\t".join(metrics))
    for solution_id, values in by_solution.items():
        print(solution_id + "\t" + "\t".join(f"{values[metric]:.8f}" for metric in metrics))


def main() -> None:
    provenance = RunProvenance.capture(
        source_module="benchmarks.scifact_migration",
        source_path="benchmarks/scifact_migration.py",
        source_line=inspect.getsourcelines(main)[1],
    )
    report = build_task().run(
        build_solutions(),
        provenance=provenance,
        status="migration-validation",
    )
    assert_historical_regression(report.records)
    write_space_results(
        report,
        ARTIFACT_ROOT / "results.json",
        release="scifact-migration-validation",
        notice="Public legacy data for framework migration validation only. Not an official leaderboard result.",
    )
    report.write_observations(ARTIFACT_ROOT / "observations")
    _print_results(report.records)
    print(f"Historical regression passed; artifacts written to {ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()
