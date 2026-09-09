from __future__ import annotations

from datasets import IterableDataset

from benchmarks.mock_showcase import build_solutions, build_tasks
from modern_ir_bench import RunProvenance, Runtime


def provenance() -> RunProvenance:
    return RunProvenance(
        source_commit="0123456789abcdef",
        source_module="benchmarks.mock_showcase",
        source_path="benchmarks/mock_showcase.py",
        source_line=1,
        repository_url="https://github.com/example/modern-ir-bench",
        created_at="2026-09-09T00:00:00+00:00",
    )


def test_two_tasks_produce_long_form_results() -> None:
    solutions = build_solutions()
    reports = [task.run(solutions, provenance=provenance(), status="demo") for task in build_tasks()]

    assert len(reports) == 2
    assert all(len(report.records) == 9 for report in reports)
    assert all(record["source_commit"] == "0123456789abcdef" for report in reports for record in report.records)
    assert {record["status"] for report in reports for record in report.records} == {"demo"}


def test_solution_is_reused_across_different_task_contracts() -> None:
    solution = build_solutions()[0]
    tasks = build_tasks()

    memory_report = tasks[0].run([solution], provenance=provenance())
    code_report = tasks[1].run([solution], provenance=provenance())

    assert memory_report.records[0]["task_id"] == "agent-memory-retrieval"
    assert code_report.records[0]["task_id"] == "code-localization"
    assert memory_report.records[0]["dataset_id"] != code_report.records[0]["dataset_id"]


def test_runtime_batches_hugging_face_iterable_dataset() -> None:
    stream = IterableDataset.from_generator(lambda: ({"value": value} for value in range(5)))

    batches = list(Runtime(query_batch_size=2).batch_rows(stream))

    assert [[row["value"] for row in batch] for batch in batches] == [
        [0, 1],
        [2, 3],
        [4],
    ]
