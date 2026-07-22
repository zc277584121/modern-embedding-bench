from __future__ import annotations

import csv
import json

from mm_embed.benchmark.leaderboard import build_leaderboard
from mm_embed.benchmark.registry import (
    BenchmarkCatalog,
    ModelSpec,
    RunManifest,
    RunTask,
    TaskSpec,
    load_catalog,
    load_run_manifest,
)
from mm_embed.benchmark.results import import_legacy_result_file, load_jsonl, normalize_legacy_model_name
from mm_embed.benchmark.runner import BenchmarkRunner
from mm_embed.hf_publish.export import export_dataset_repo, export_space_repo


def test_default_catalog_and_run_manifests_load() -> None:
    catalog = load_catalog()

    assert "openai-text-embedding-3-large" in catalog.models
    assert "mrl_stress" in catalog.tasks
    assert catalog.models["geevec-lite-general"].publish is False
    assert catalog.models["geevec-api-general"].publish is False

    for path in (
        "benchmark/runs/api-coverage-smoke.yaml",
        "benchmark/runs/api-modern-smoke.yaml",
        "benchmark/runs/openai-smoke.yaml",
        "benchmark/runs/local-smoke.yaml",
        "benchmark/runs/core-text-standard.yaml",
        "benchmark/runs/late-chunking-retrieval-local-smoke.yaml",
    ):
        manifest = load_run_manifest(path)
        for model_id in manifest.model_ids:
            catalog.require_model(model_id)
        for run_task in manifest.tasks:
            catalog.require_task(run_task.id)


def test_leaderboard_backfills_primary_metric_from_catalog() -> None:
    catalog = load_catalog()
    records = [
        {
            "run": {"id": "legacy:test"},
            "timestamps": {"duration_s": 1.2},
            "model": {"id": "model-a", "display_name": "Model A", "provider": "provider-a"},
            "task": {"id": "crosslingual_retrieval", "display_name": "Crosslingual", "primary_metric": None},
            "metrics": {"hard_avg_recall@1": 0.75},
            "error": None,
        }
    ]

    rows = build_leaderboard(records, catalog)

    assert rows == [
        {
            "task_id": "crosslingual_retrieval",
            "task": "Crosslingual",
            "model_id": "model-a",
            "model": "Model A",
            "provider": "provider-a",
            "primary_metric": "hard_avg_recall@1",
            "score": 0.75,
            "run_id": "legacy:test",
            "duration_s": 1.2,
        }
    ]


def test_unpublished_fixture_task_is_excluded_from_public_outputs(tmp_path) -> None:
    catalog = load_catalog()
    record = {
        "run": {"id": "late-chunking-retrieval-local-smoke"},
        "timestamps": {"duration_s": 0.0},
        "model": {
            "id": "openai-text-embedding-3-large",
            "display_name": "OpenAI",
            "provider": "openai",
        },
        "provider_result": {"provider": "deterministic-grouped-local", "model_name": "fixture-label-test-double"},
        "task": {
            "id": "late_chunking_retrieval",
            "display_name": "Context-aware chunk retrieval fixture",
            "primary_metric": "chunk_ndcg@10",
            "publish": False,
        },
        "metrics": {"chunk_ndcg@10": 1.0},
        "error": None,
    }

    assert build_leaderboard([record], catalog) == []

    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(record) + "\n", encoding="utf-8")
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s\n"
        "late_chunking_retrieval,Fixture,openai-text-embedding-3-large,OpenAI,openai,chunk_ndcg@10,1.0,fixture,0.0\n",
        encoding="utf-8",
    )

    output = export_dataset_repo(
        output_dir=tmp_path / "dataset",
        results_path=results,
        leaderboard_path=leaderboard,
    )

    assert "late_chunking_retrieval" not in (output / "tasks.jsonl").read_text(encoding="utf-8")
    assert (output / "results" / "latest.jsonl").read_text(encoding="utf-8") == ""
    assert "late_chunking_retrieval" not in (output / "leaderboards" / "latest.csv").read_text(encoding="utf-8")
    assert not (output / "runs" / "late-chunking-retrieval-local-smoke.yaml").exists()


def test_import_legacy_results_to_jsonl(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.json"
    output_path = tmp_path / "imported.jsonl"
    legacy_path.write_text(
        json.dumps(
            [
                {
                    "provider": "openai",
                    "model": "text-embedding-3-large",
                    "task": "needle_in_haystack",
                    "metrics": {"overall_accuracy": 1.0},
                    "details": {"n_test_cases": 30},
                    "error": None,
                    "elapsed_s": 2.5,
                }
            ]
        ),
        encoding="utf-8",
    )

    count = import_legacy_result_file(legacy_path, output_path)
    records = load_jsonl(output_path)

    assert count == 1
    assert records[0]["schema_version"] == "2.0"
    assert records[0]["run"]["id"] == "legacy:legacy"
    assert records[0]["model"]["id"] == "text-embedding-3-large"
    assert records[0]["task"]["id"] == "needle_in_haystack"
    assert records[0]["metrics"]["overall_accuracy"] == 1.0


def test_normalize_legacy_model_name_keeps_public_ids() -> None:
    assert normalize_legacy_model_name("/data2/models/Qwen3-VL-Embedding-2B") == "Qwen3-VL-Embedding-2B"
    assert normalize_legacy_model_name("BAAI/bge-m3") == "BAAI/bge-m3"


def test_runner_overwrite_replaces_existing_jsonl(tmp_path) -> None:
    output = tmp_path / "results.jsonl"
    output.write_text('{"stale": true}\n', encoding="utf-8")

    catalog = BenchmarkCatalog(
        root=tmp_path,
        models={
            "inactive-model": ModelSpec(
                id="inactive-model",
                display_name="Inactive Model",
                provider="missing_provider",
                status="disabled",
            )
        },
        tasks={
            "dummy_task": TaskSpec(
                id="dummy_task",
                display_name="Dummy Task",
                task="dummy_task",
                description="Task that should not run when model preflight fails.",
            )
        },
    )
    manifest = RunManifest(id="overwrite-test", model_ids=["inactive-model"], tasks=[RunTask(id="dummy_task")])

    records = BenchmarkRunner(catalog=catalog, output=output, overwrite=True).run_manifest(manifest)
    written = load_jsonl(output)

    assert len(records) == 1
    assert len(written) == 1
    assert "stale" not in written[0]
    assert written[0]["error"] == "model status is disabled"


def test_export_hf_dataset_skips_cache_artifacts(tmp_path) -> None:
    output = export_dataset_repo(output_dir=tmp_path / "dataset", include_data=True)

    exported_files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]

    assert "models.jsonl" in exported_files
    assert "tasks.jsonl" in exported_files
    assert not any("embedding_cache" in path for path in exported_files)
    assert not any(path.endswith(".npy") for path in exported_files)


def test_export_hf_dataset_filters_unpublished_models(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    records = [
        {
            "run": {"id": "test"},
            "timestamps": {"duration_s": 1.0},
            "model": {"id": "openai-text-embedding-3-large", "display_name": "OpenAI", "provider": "openai"},
            "provider_result": {"provider": "openai", "model_name": "text-embedding-3-large"},
            "task": {"id": "needle_in_haystack", "display_name": "Needle", "primary_metric": "overall_accuracy"},
            "metrics": {"overall_accuracy": 0.9},
            "error": None,
        },
        {
            "run": {"id": "test"},
            "timestamps": {"duration_s": 1.0},
            "model": {"id": "geevec-api-general", "display_name": "Preview Model", "provider": "geevec_api"},
            "provider_result": {"provider": "geevec_api", "model_name": "preview"},
            "task": {"id": "needle_in_haystack", "display_name": "Needle", "primary_metric": "overall_accuracy"},
            "metrics": {"overall_accuracy": 1.0},
            "error": None,
        },
    ]
    results.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "\n".join(
            [
                "task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s",
                "needle_in_haystack,Needle,openai-text-embedding-3-large,OpenAI,openai,overall_accuracy,0.9,test,1.0",
                "needle_in_haystack,Needle,geevec-api-general,Preview,geevec_api,overall_accuracy,1.0,test,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = export_dataset_repo(
        output_dir=tmp_path / "dataset",
        results_path=results,
        leaderboard_path=leaderboard,
    )

    for path in (
        output / "models.jsonl",
        output / "results" / "latest.jsonl",
        output / "results" / "latest-successful.jsonl",
        output / "leaderboards" / "latest.csv",
    ):
        exported = path.read_text(encoding="utf-8").lower()
        assert "geevec" not in exported
        assert "preview model" not in exported

    assert len(load_jsonl(output / "results" / "latest.jsonl")) == 1


def test_export_hf_dataset_marks_leaderboard_provenance_and_latest(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    records = [
        {
            "run": {
                "id": "legacy:baseline",
                "description": "Imported legacy result",
                "metadata": {"legacy_source": "legacy/results/baseline.json"},
                "git_sha": None,
            },
            "timestamps": {"duration_s": 1.0},
            "model": {
                "id": "openai-text-embedding-3-large",
                "display_name": "OpenAI",
                "provider": "openai",
                "access": "legacy",
                "tags": ["legacy"],
            },
            "provider_result": {"provider": "openai", "model_name": "text-embedding-3-large"},
            "task": {
                "id": "needle_in_haystack",
                "display_name": "Needle",
                "primary_metric": "overall_accuracy",
                "tags": ["legacy"],
            },
            "metrics": {"overall_accuracy": 0.8},
            "error": None,
        },
        {
            "run": {
                "id": "openai-smoke",
                "description": "OpenAI smoke benchmark",
                "metadata": {},
                "git_sha": "abc123",
            },
            "timestamps": {"duration_s": 0.9},
            "model": {
                "id": "openai-text-embedding-3-large",
                "display_name": "OpenAI",
                "provider": "openai",
                "access": "api",
                "tags": ["smoke"],
            },
            "provider_result": {"provider": "openai", "model_name": "text-embedding-3-large"},
            "task": {
                "id": "needle_in_haystack",
                "display_name": "Needle",
                "primary_metric": "overall_accuracy",
                "tags": [],
            },
            "metrics": {"overall_accuracy": 0.9},
            "error": None,
        },
    ]
    results.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "\n".join(
            [
                "task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s",
                ",".join(
                    [
                        "needle_in_haystack",
                        "Needle",
                        "openai-text-embedding-3-large",
                        "OpenAI",
                        "openai",
                        "overall_accuracy",
                        "0.8",
                        "legacy:baseline",
                        "1.0",
                    ]
                ),
                ",".join(
                    [
                        "needle_in_haystack",
                        "Needle",
                        "openai-text-embedding-3-large",
                        "OpenAI",
                        "openai",
                        "overall_accuracy",
                        "0.9",
                        "openai-smoke",
                        "0.9",
                    ]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = export_dataset_repo(
        output_dir=tmp_path / "dataset",
        results_path=results,
        leaderboard_path=leaderboard,
    )

    with open(output / "leaderboards" / "latest.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames[:9] == [
        "task_id",
        "task",
        "model_id",
        "model",
        "provider",
        "primary_metric",
        "score",
        "run_id",
        "duration_s",
    ]
    assert reader.fieldnames[9:] == [
        "evidence_tier",
        "evidence_source",
        "task_model_duplicate_count",
        "task_model_run_rank",
        "is_latest_for_task_model",
    ]
    assert [row["evidence_tier"] for row in rows] == ["legacy", "smoke"]
    assert rows[0]["evidence_source"] == "legacy/results/baseline.json"
    assert rows[1]["evidence_source"] == "abc123"
    assert [row["task_model_duplicate_count"] for row in rows] == ["2", "2"]
    assert [row["task_model_run_rank"] for row in rows] == ["1", "2"]
    assert [row["is_latest_for_task_model"] for row in rows] == ["false", "true"]

    manifest_text = (output / "export_manifest.yaml").read_text(encoding="utf-8")
    assert "duplicate_task_model_repeats: 1" in manifest_text
    assert "legacy: 1" in manifest_text
    assert "smoke: 1" in manifest_text


def _load_generated_space_app(output, monkeypatch):
    import sys
    import types

    class DataFrame:
        def __init__(self, rows=None, columns=None):
            source_rows = list(rows or [])
            self._rows = (
                [{column: row.get(column) for column in columns} for row in source_rows]
                if columns is not None
                else source_rows
            )
            self.index = range(len(self._rows))

        @property
        def empty(self):
            return not self._rows

        def to_dict(self, orient):
            assert orient == "records"
            return [dict(row) for row in self._rows]

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = DataFrame
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.setitem(sys.modules, "gradio", types.ModuleType("gradio"))
    namespace = {"__name__": "generated_space_app"}
    monkeypatch.chdir(output)
    app_path = output / "app.py"
    exec(compile(app_path.read_text(encoding="utf-8"), str(app_path), "exec"), namespace)
    return namespace


def test_export_hf_space_bundles_current_evidence_view(tmp_path, monkeypatch) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s,evidence_tier,"
        "evidence_source,task_model_duplicate_count,task_model_run_rank,is_latest_for_task_model\n"
        "needle_in_haystack,Needle,model-a,Model A,openai,overall_accuracy,0.4,legacy:a,1.0,legacy,old,2,1,false\n"
        "needle_in_haystack,Needle,model-a,Model A,openai,overall_accuracy,0.8,benchmark:a,0.8,benchmark,new,2,2,true\n"
        "needle_in_haystack,Needle,model-b,Model B,openai,overall_accuracy,0.7,benchmark:b,0.7,,new,1,1,true\n"
        "needle_in_haystack,Needle,model-c,Model C,local,overall_accuracy,0.9,legacy:c,0.6,legacy,new,1,1,true\n"
        "mrl_stress,MRL,model-a,Model A,openai,spearman,0.6,smoke:mrl,0.5,smoke,new,1,1,true\n"
        "late_chunking_retrieval,Fixture,fixture-model,Fixture Model,local,chunk_ndcg@10,1.0,fixture,0.0,"
        "smoke,fixture,1,1,true\n",
        encoding="utf-8",
    )

    output = export_space_repo(output_dir=tmp_path / "space", bundled_leaderboard=leaderboard)

    assert (output / "README.md").exists()
    assert (output / "app.py").exists()
    compile((output / "app.py").read_text(encoding="utf-8"), str(output / "app.py"), "exec")
    bundled_text = (output / "leaderboard.csv").read_text(encoding="utf-8")
    assert "legacy:a" in bundled_text
    assert "benchmark:a" in bundled_text
    assert "late_chunking_retrieval" not in bundled_text

    app = _load_generated_space_app(output, monkeypatch)
    assert app["LATEST_MARKERS_AVAILABLE"] is True
    assert app["DEFAULT_LATEST_ONLY"] is True
    assert app["EVIDENCE_TIERS"] == ["All evidence tiers", "benchmark", "legacy", "smoke", "unknown"]

    current_rows = app["filtered_rows"](
        "needle_in_haystack", "All providers", "All evidence tiers", "", app["DEFAULT_LATEST_ONLY"]
    )
    current_keys = [app["task_model_key"](row) for row in current_rows]
    assert len(current_keys) == len(set(current_keys))
    current = app["table_from_rows"](current_rows, 30).to_dict("records")
    assert len(current) == 3
    assert all(row["is_latest_for_task_model"] == "true" for row in current)

    history = app["render_table"](
        "needle_in_haystack", "All providers", "All evidence tiers", "", False, 30
    ).to_dict("records")
    assert len(history) == 4
    assert [row["run_id"] for row in history if row["model"] == "Model A"] == ["benchmark:a", "legacy:a"]

    legacy = app["render_table"](
        "needle_in_haystack", "All providers", "legacy", "", False, 30
    ).to_dict("records")
    assert {row["run_id"] for row in legacy} == {"legacy:a", "legacy:c"}
    assert all(row["evidence_tier"] == "legacy" for row in legacy)

    top_openai_legacy = app["render_table"](
        "needle_in_haystack", "openai", "legacy", "model", False, 1
    ).to_dict("records")
    assert [row["run_id"] for row in top_openai_legacy] == ["legacy:a"]
    assert top_openai_legacy[0]["rank"] == 1

    empty_note, empty_table = app["render"](
        "needle_in_haystack", "All providers", "smoke", "", False, 30
    )
    assert empty_table.empty
    assert "No rows match the selected filters." in empty_note
    assert "all historical rows" in empty_note


def test_export_hf_space_single_evidence_tier_ui_is_neutral(tmp_path, monkeypatch) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "task_id,model_id,model,provider,score,evidence_tier,is_latest_for_task_model\n"
        "needle_in_haystack,model-a,Model A,openai,0.8,legacy,true\n",
        encoding="utf-8",
    )

    output = export_space_repo(output_dir=tmp_path / "space", bundled_leaderboard=leaderboard)
    app = _load_generated_space_app(output, monkeypatch)

    assert app["EVIDENCE_TIERS"] == ["All evidence tiers", "legacy"]
    assert app["evidence_summary"]() == "legacy=1"
