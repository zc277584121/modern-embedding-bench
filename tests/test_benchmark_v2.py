from __future__ import annotations

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
        "benchmark/runs/openai-smoke.yaml",
        "benchmark/runs/local-smoke.yaml",
        "benchmark/runs/core-text-standard.yaml",
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


def test_export_hf_space_bundles_leaderboard(tmp_path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text("task_id,model,score\nmrl,model-a,1.0\n", encoding="utf-8")

    output = export_space_repo(output_dir=tmp_path / "space", bundled_leaderboard=leaderboard)

    assert (output / "README.md").exists()
    assert (output / "app.py").exists()
    compile((output / "app.py").read_text(encoding="utf-8"), str(output / "app.py"), "exec")
    assert (output / "leaderboard.csv").read_text(encoding="utf-8").startswith("task_id,model,score")
