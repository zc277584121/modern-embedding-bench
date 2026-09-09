"""Build the public framework preview from deterministic synthetic data."""

from __future__ import annotations

import inspect
from pathlib import Path

from datasets import Dataset

from modern_ir_bench import MetricSet, RunProvenance, RunReport
from modern_ir_bench.exporters import write_space_results
from modern_ir_bench.metrics import NDCG, MeanReciprocalRank, Recall, Success
from modern_ir_bench.solutions import (
    BM25Solution,
    CharacterNGramSolution,
    HybridSolution,
)
from modern_ir_bench.tasks import AgentMemoryRetrieval, CodeLocalization

RELEASE = "framework-preview-2026-09"
NOTICE = "Synthetic demo data for validating the framework and Space. Not a scientific benchmark result."


def adapt_memory_dataset(raw: dict[str, Dataset]) -> dict[str, Dataset]:
    return {
        "memories": (
            raw["store"]
            .select_columns(["record_key", "body"])
            .rename_columns({"record_key": "memory_id", "body": "content"})
            .cast(AgentMemoryRetrieval.dataset_features["memories"])
        ),
        "queries": (
            raw["requests"]
            .select_columns(["request_key", "request", "needed_records"])
            .rename_columns(
                {
                    "request_key": "query_id",
                    "request": "query",
                    "needed_records": "relevant_memory_ids",
                }
            )
            .cast(AgentMemoryRetrieval.dataset_features["queries"])
        ),
    }


def adapt_code_dataset(raw: dict[str, Dataset]) -> dict[str, Dataset]:
    return {
        "files": (
            raw["repository"]
            .select_columns(["path", "content"])
            .rename_columns({"path": "file_id", "content": "source"})
            .cast(CodeLocalization.dataset_features["files"])
        ),
        "issues": (
            raw["tickets"]
            .select_columns(["ticket", "description", "changed_paths"])
            .rename_columns(
                {
                    "ticket": "issue_id",
                    "description": "issue",
                    "changed_paths": "target_file_ids",
                }
            )
            .cast(CodeLocalization.dataset_features["issues"])
        ),
    }


def memory_data() -> dict[str, Dataset]:
    raw = {
        "store": Dataset.from_dict(
            {
                "record_key": ["m-auth", "m-cache", "m-db", "m-k8s", "m-python", "m-billing"],
                "body": [
                    "OAuth refresh tokens are rotated after every successful refresh. Reusing an old token returns invalid_grant.",
                    "Redis eviction uses allkeys-lru in production. Cache keys for user profiles start with profile:v2.",
                    "PostgreSQL reports table queries require the composite index on tenant_id and created_at.",
                    "Kubernetes readiness checks use /health/ready while liveness checks use /health/live.",
                    "Python background jobs use asyncio TaskGroup so sibling tasks are cancelled together after a failure.",
                    "Billing invoices are retried for three days before an account enters the past_due state.",
                ],
            }
        ),
        "requests": Dataset.from_dict(
            {
                "request_key": ["q1", "q2", "q3", "q4", "q5"],
                "request": [
                    "Why does reusing my OAuth refresh token return invalid_grant?",
                    "Which Kubernetes endpoint should the readiness probe call?",
                    "What database index should I add for tenant report queries ordered by creation time?",
                    "How are sibling async Python jobs cancelled when one task fails?",
                    "When does an unpaid customer account become past due?",
                ],
                "needed_records": [
                    ["m-auth"],
                    ["m-k8s"],
                    ["m-db"],
                    ["m-python"],
                    ["m-billing"],
                ],
            }
        ),
    }
    return adapt_memory_dataset(raw)


def code_data() -> dict[str, Dataset]:
    raw = {
        "repository": Dataset.from_dict(
            {
                "path": [
                    "auth/token_store.py",
                    "cache/profile_cache.py",
                    "database/report_queries.py",
                    "deploy/health_routes.py",
                    "workers/task_group.py",
                    "billing/invoice_retry.py",
                ],
                "content": [
                    "Rotate OAuth refresh tokens and reject a previously consumed token with invalid_grant.",
                    "Build Redis profile:v2 keys and apply user profile cache invalidation.",
                    "Query tenant reports by tenant_id and created_at using the reporting database index.",
                    "Expose Kubernetes /health/ready and /health/live endpoints for probes.",
                    "Run Python asyncio background workers inside a TaskGroup and cancel sibling jobs on failure.",
                    "Retry unpaid invoices and transition customer accounts to past_due after the retry window.",
                ],
            }
        ),
        "tickets": Dataset.from_dict(
            {
                "ticket": ["issue-17", "issue-23", "issue-41", "issue-58", "issue-72"],
                "description": [
                    "The readiness probe returns 404 after the health endpoint refactor.",
                    "Report list requests are slow when filtering by tenant and creation date.",
                    "A refresh credential can be consumed twice during concurrent OAuth requests.",
                    "One failed async worker leaves its sibling background jobs running.",
                    "User profile updates continue returning stale values from Redis.",
                ],
                "changed_paths": [
                    ["deploy/health_routes.py"],
                    ["database/report_queries.py"],
                    ["auth/token_store.py"],
                    ["workers/task_group.py"],
                    ["cache/profile_cache.py"],
                ],
            }
        ),
    }
    return adapt_code_dataset(raw)


def build_solutions():
    bm25 = BM25Solution(
        id="bm25",
        title="BM25",
        description="Deterministic lexical BM25 baseline.",
    )
    character = CharacterNGramSolution(
        id="char-ngram",
        title="Character N-Gram",
        description="Character trigram cosine baseline.",
        size=3,
    )
    hybrid = HybridSolution(
        id="hybrid-rrf",
        title="Hybrid RRF",
        description="Reciprocal-rank fusion over BM25 and character n-grams.",
        primary=bm25,
        secondary=character,
        alpha=0.65,
    )
    return [bm25, character, hybrid]


def build_tasks():
    return [
        AgentMemoryRetrieval(
            id="agent-memory-retrieval",
            title="Agent Memory Retrieval",
            description="Retrieve prior memories needed to answer a later request.",
            version="0.1.0",
            datasets={"synthetic-memory": memory_data()},
            dataset_versions={"synthetic-memory": "2026-09-v1"},
            metrics=MetricSet(
                primary=NDCG(k=3),
                secondary=[Recall(k=3), MeanReciprocalRank(k=3)],
            ),
            top_k=3,
            query_batch_size=2,
        ),
        CodeLocalization(
            id="code-localization",
            title="Code Localization",
            description="Retrieve repository files that should be inspected for an issue.",
            version="0.1.0",
            datasets={"synthetic-repository": code_data()},
            dataset_versions={"synthetic-repository": "2026-09-v1"},
            metrics=MetricSet(
                primary=Recall(k=2),
                secondary=[Success(k=1), MeanReciprocalRank(k=3)],
            ),
            top_k=3,
            query_batch_size=2,
        ),
    ]


def main() -> None:
    provenance = RunProvenance.capture(
        source_module="benchmarks.mock_showcase",
        source_path="benchmarks/mock_showcase.py",
        source_line=inspect.getsourcelines(main)[1],
    )
    report = RunReport()
    solutions = build_solutions()
    for task in build_tasks():
        report.extend(
            task.run(
                solutions,
                provenance=provenance,
                status="demo",
            )
        )

    write_space_results(
        report,
        Path("space/data/results.json"),
        release=RELEASE,
        notice=NOTICE,
    )
    report.write_observations(Path("artifacts/framework-preview/observations"))
    print(f"Wrote {len(report.records)} measurements for {len(report.tasks)} tasks")


if __name__ == "__main__":
    main()
