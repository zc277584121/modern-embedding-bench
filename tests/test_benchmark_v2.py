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

    expected_tiers = {
        "benchmark/runs/api-coverage-smoke.yaml": "smoke",
        "benchmark/runs/api-modern-smoke.yaml": "smoke",
        "benchmark/runs/openai-smoke.yaml": "smoke",
        "benchmark/runs/local-smoke.yaml": "smoke",
        "benchmark/runs/core-text-standard.yaml": "benchmark",
        "benchmark/runs/late-chunking-retrieval-local-smoke.yaml": "fixture",
        "benchmark/runs/composed-media-retrieval-local-smoke.yaml": "fixture",
    }
    for path, expected_tier in expected_tiers.items():
        manifest = load_run_manifest(path)
        assert manifest.evidence_tier == expected_tier
        for model_id in manifest.model_ids:
            catalog.require_model(model_id)
        for run_task in manifest.tasks:
            catalog.require_task(run_task.id)

    fixture_manifest = load_run_manifest("benchmark/runs/late-chunking-retrieval-local-smoke.yaml")
    assert fixture_manifest.publish is False


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


def test_leaderboard_excludes_explicitly_unpublished_runs_but_keeps_historical_defaults() -> None:
    catalog = load_catalog()
    base_record = {
        "timestamps": {"duration_s": 1.2},
        "model": {"id": "model-a", "display_name": "Model A", "provider": "provider-a"},
        "task": {
            "id": "crosslingual_retrieval",
            "display_name": "Crosslingual",
            "primary_metric": "hard_avg_recall@1",
        },
        "metrics": {"hard_avg_recall@1": 0.75},
        "error": None,
    }
    records = [
        {**base_record, "run": {"id": "historical-default"}},
        {**base_record, "run": {"id": "private-run", "publish": False}},
    ]

    rows = build_leaderboard(records, catalog)

    assert [row["run_id"] for row in rows] == ["historical-default"]


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


def test_public_task_catalog_excludes_invented_and_mock_only_contracts(tmp_path) -> None:
    catalog = load_catalog()
    expected_private_contracts = {
        "agent_procedural_tool_memory": {
            "required_modalities": ["text"],
            "primary_metric": "hard_mrr",
            "dataset_version": "agent-procedural-tool-memory-fixture-v0",
            "classification_tag": "fixture-only",
        },
        "autonomous_driving": {
            "required_modalities": ["text", "image"],
            "primary_metric": "overall_t2i_precision@3",
            "dataset_version": "autonomous-driving-mock-fixture-v0",
            "classification_tag": "blocked-real-source",
        },
        "chinese_multimodal": {
            "required_modalities": ["text", "image"],
            "primary_metric": "zh_t2i_recall@1",
            "dataset_version": "chinese-multimodal-mock-fixture-v0",
            "classification_tag": "blocked-real-source",
        },
    }

    for task_id, expected in expected_private_contracts.items():
        spec = catalog.require_task(task_id)
        assert spec.publish is False
        assert spec.leaderboard_publish is False
        assert spec.required_modalities == expected["required_modalities"]
        assert spec.primary_metric == expected["primary_metric"]
        assert spec.dataset_version == expected["dataset_version"]
        assert {expected["classification_tag"], "no-publish"}.issubset(spec.tags)

    dataset_output = export_dataset_repo(output_dir=tmp_path / "dataset")
    space_output = export_space_repo(output_dir=tmp_path / "space")
    public_dataset_tasks = {
        row["id"] for row in load_jsonl(dataset_output / "tasks.jsonl")
    }
    public_space_tasks = {
        row["id"] for row in load_jsonl(space_output / "tasks.jsonl")
    }

    assert public_dataset_tasks == public_space_tasks
    assert public_dataset_tasks == {
        "cross_modal_retrieval",
        "crosslingual_retrieval",
        "mrl_stress",
        "needle_in_haystack",
    }
    assert public_dataset_tasks.isdisjoint(expected_private_contracts)


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
    assert records[0]["run"]["publish"] is True
    assert records[0]["run"]["evidence_tier"] == "legacy"
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
    manifest = RunManifest(
        id="overwrite-test",
        model_ids=["inactive-model"],
        tasks=[RunTask(id="dummy_task")],
        metadata={"scope": "contract-test"},
        publish=False,
        evidence_tier="smoke",
    )

    records = BenchmarkRunner(catalog=catalog, output=output, overwrite=True).run_manifest(manifest)
    written = load_jsonl(output)

    assert len(records) == 1
    assert len(written) == 1
    assert "stale" not in written[0]
    assert written[0]["error"] == "model status is disabled"
    assert written[0]["run"]["publish"] is False
    assert written[0]["run"]["evidence_tier"] == "smoke"
    assert written[0]["run"]["metadata"] == {"scope": "contract-test"}
    assert "scope" not in written[0]


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


def test_export_hf_dataset_filters_unpublished_runs_in_both_leaderboard_paths(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    records = [
        {
            "run": {"id": "public-smoke", "evidence_tier": "smoke"},
            "timestamps": {"duration_s": 1.0},
            "model": {"id": "model-public", "display_name": "Public", "provider": "openai"},
            "provider_result": {"provider": "openai", "model_name": "model-public"},
            "task": {
                "id": "needle_in_haystack",
                "display_name": "Needle",
                "primary_metric": "overall_accuracy",
            },
            "metrics": {"overall_accuracy": 0.9},
            "error": None,
        },
        {
            "run": {"id": "private-fixture", "publish": False, "evidence_tier": "fixture"},
            "timestamps": {"duration_s": 0.5},
            "model": {"id": "model-private-run", "display_name": "Private Run", "provider": "openai"},
            "provider_result": {"provider": "openai", "model_name": "model-private-run"},
            "task": {
                "id": "needle_in_haystack",
                "display_name": "Needle",
                "primary_metric": "overall_accuracy",
            },
            "metrics": {"overall_accuracy": 1.0},
            "error": None,
        },
        {
            "run": {"id": "public-failed", "publish": True, "evidence_tier": "benchmark"},
            "timestamps": {"duration_s": 0.2},
            "model": {"id": "model-failed", "display_name": "Failed", "provider": "openai"},
            "provider_result": {"provider": "openai", "model_name": "model-failed"},
            "task": {
                "id": "needle_in_haystack",
                "display_name": "Needle",
                "primary_metric": "overall_accuracy",
            },
            "metrics": {},
            "error": "provider timeout",
        },
    ]
    results.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s\n"
        "needle_in_haystack,Needle,model-public,Public,openai,overall_accuracy,0.9,public-smoke,1.0\n"
        "needle_in_haystack,Needle,model-private-run,Private Run,openai,overall_accuracy,1.0,private-fixture,0.5\n",
        encoding="utf-8",
    )

    for name, leaderboard_path in (("precomputed", leaderboard), ("results-only", None)):
        output = export_dataset_repo(
            output_dir=tmp_path / name,
            results_path=results,
            leaderboard_path=leaderboard_path,
        )
        exported_records = load_jsonl(output / "results" / "latest.jsonl")
        successful_records = load_jsonl(output / "results" / "latest-successful.jsonl")
        with open(output / "leaderboards" / "latest.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        assert [record["run"]["id"] for record in exported_records] == ["public-smoke", "public-failed"]
        assert [record["run"]["id"] for record in successful_records] == ["public-smoke"]
        assert [row["run_id"] for row in rows] == ["public-smoke"]
        assert rows[0]["evidence_tier"] == "smoke"


def test_explicit_evidence_metadata_wins_over_text_and_row_inference(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    records = [
        {
            "run": {
                "id": "legacy-smoke-name",
                "description": "Smoke legacy wording",
                "metadata": {"legacy_source": "legacy/conflict.json"},
                "evidence_tier": "benchmark",
            },
            "timestamps": {"duration_s": 1.0},
            "model": {
                "id": "model-explicit",
                "display_name": "Explicit",
                "provider": "openai",
                "access": "legacy",
                "tags": ["smoke"],
            },
            "provider_result": {"provider": "openai", "model_name": "model-explicit"},
            "task": {
                "id": "needle_in_haystack",
                "display_name": "Needle",
                "primary_metric": "overall_accuracy",
                "tags": ["legacy"],
            },
            "metrics": {"overall_accuracy": 0.9},
            "error": None,
        },
        {
            "run": {"id": "smoke-inferred"},
            "timestamps": {"duration_s": 1.0},
            "model": {"id": "model-row", "display_name": "Row", "provider": "openai", "tags": ["smoke"]},
            "provider_result": {"provider": "openai", "model_name": "model-row"},
            "task": {
                "id": "needle_in_haystack",
                "display_name": "Needle",
                "primary_metric": "overall_accuracy",
            },
            "metrics": {"overall_accuracy": 0.8},
            "error": None,
        },
        {
            "run": {"id": "legacy:historical", "metadata": {"legacy_source": "legacy/old.json"}},
            "timestamps": {"duration_s": 1.0},
            "model": {"id": "model-legacy", "display_name": "Legacy", "provider": "openai"},
            "provider_result": {"provider": "openai", "model_name": "model-legacy"},
            "task": {
                "id": "needle_in_haystack",
                "display_name": "Needle",
                "primary_metric": "overall_accuracy",
            },
            "metrics": {"overall_accuracy": 0.7},
            "error": None,
        },
    ]
    results.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s,evidence_tier\n"
        "needle_in_haystack,Needle,model-explicit,Explicit,openai,overall_accuracy,0.9,legacy-smoke-name,1.0,fixture\n"
        "needle_in_haystack,Needle,model-row,Row,openai,overall_accuracy,0.8,smoke-inferred,1.0,fixture\n"
        "needle_in_haystack,Needle,model-legacy,Legacy,openai,overall_accuracy,0.7,legacy:historical,1.0,\n",
        encoding="utf-8",
    )

    output = export_dataset_repo(
        output_dir=tmp_path / "dataset",
        results_path=results,
        leaderboard_path=leaderboard,
    )
    with open(output / "leaderboards" / "latest.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert [row["evidence_tier"] for row in rows] == ["benchmark", "fixture", "legacy"]


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
        "crosslingual_retrieval,Crosslingual,model-a,Model A,openai,hard_avg_recall@1,0.5,benchmark:cross,"
        "0.4,benchmark,new,1,1,true\n"
        "cross_modal_retrieval,Cross-modal,model-a,Model A,openai,hard_avg_recall@1,0.3,legacy:modal,0.3,"
        "legacy,new,1,1,true\n"
        "late_chunking_retrieval,Fixture,fixture-model,Fixture Model,local,chunk_ndcg@10,1.0,fixture,0.0,"
        "smoke,fixture,1,1,true\n",
        encoding="utf-8",
    )

    output = export_space_repo(
        output_dir=tmp_path / "space",
        dataset_repo_id="example/modern-embedding-bench",
        bundled_leaderboard=leaderboard,
    )

    assert (output / "README.md").exists()
    assert (output / "app.py").exists()
    assert (output / "models.jsonl").exists()
    assert (output / "tasks.jsonl").exists()
    compile((output / "app.py").read_text(encoding="utf-8"), str(output / "app.py"), "exec")
    bundled_text = (output / "leaderboard.csv").read_text(encoding="utf-8")
    assert "legacy:a" in bundled_text
    assert "benchmark:a" in bundled_text
    assert "late_chunking_retrieval" not in bundled_text

    task_specs = [json.loads(line) for line in (output / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    declared_tasks = sorted(task["id"] for task in task_specs)
    assert declared_tasks == [
        "cross_modal_retrieval",
        "crosslingual_retrieval",
        "mrl_stress",
        "needle_in_haystack",
    ]
    assert all(task["publish"] is True and task["leaderboard_publish"] is True for task in task_specs)
    assert all("default_kwargs" not in task and "task" not in task and "tags" not in task for task in task_specs)

    model_specs = [json.loads(line) for line in (output / "models.jsonl").read_text(encoding="utf-8").splitlines()]
    allowed_model_fields = {
        "access",
        "dimensions",
        "display_name",
        "id",
        "max_text_length",
        "modalities",
        "provider",
        "source",
        "status",
        "supports_mrl",
    }
    assert model_specs
    assert all(set(model) == allowed_model_fields for model in model_specs)
    assert all(model["provider"] not in {"geevec_api", "geevec_lite"} for model in model_specs)
    assert all("geevec" not in " ".join(str(value).lower() for value in model.values()) for model in model_specs)
    assert all("provider_kwargs" not in model and "api_key_env" not in model for model in model_specs)

    import sys
    import types

    download_attempts = []
    huggingface_hub = types.ModuleType("huggingface_hub")

    def unavailable_download(*, repo_id, repo_type, filename):
        download_attempts.append((repo_id, repo_type, filename))
        raise RuntimeError("offline test")

    huggingface_hub.hf_hub_download = unavailable_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    app = _load_generated_space_app(output, monkeypatch)
    assert [attempt[2] for attempt in download_attempts] == [
        "tasks.jsonl",
        "models.jsonl",
        "leaderboards/latest.csv",
    ]
    assert app["TASKS"] == [
        "cross_modal_retrieval",
        "crosslingual_retrieval",
        "mrl_stress",
        "needle_in_haystack",
    ]
    assert app["DECLARED_TASKS"] == declared_tasks
    assert app["LATEST_MARKERS_AVAILABLE"] is True
    assert app["DEFAULT_LATEST_ONLY"] is True
    assert app["EVIDENCE_TIERS"] == ["All evidence tiers", "benchmark", "legacy", "smoke", "unknown"]
    assert "Declared public tasks: **4**" in app["summary_markdown"]()
    assert "Tasks with score rows: **4**" in app["summary_markdown"]()
    assert "Declared public models: **{}**".format(len(model_specs)) in app["summary_markdown"]()

    model_catalog = app["model_catalog_table"]("All catalog providers", "").to_dict("records")
    assert [row["display_name"] for row in model_catalog] == sorted(
        (row["display_name"] for row in model_catalog), key=str.lower
    )
    assert all(row["declaration"] == "declared in public model catalog" for row in model_catalog)
    assert all(row["evaluation_evidence"] == "declared only - no public score rows" for row in model_catalog)
    assert "model-a" not in {row["id"] for row in model_catalog}
    model_note, _ = app["render_model_catalog"]("All catalog providers", "")
    assert "not a model ranking" in model_note
    assert "does not imply quality" in model_note
    assert "Score-only legacy identities remain available" in model_note

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

    coverage = app["coverage_table"]("All providers", "All evidence tiers", "").to_dict("records")
    assert [(row["model"], row["covered_tasks"]) for row in coverage] == [
        ("Model A", 4),
        ("Model B", 1),
        ("Model C", 1),
    ]
    assert "score" not in coverage[0]
    model_a = coverage[0]
    assert model_a["needle_in_haystack"] == "evaluated (benchmark)"
    assert model_a["mrl_stress"] == "evaluated (smoke)"
    assert model_a["crosslingual_retrieval"] == "evaluated (benchmark)"
    assert model_a["cross_modal_retrieval"] == "evaluated (legacy)"
    model_b = next(row for row in coverage if row["model"] == "Model B")
    assert model_b["needle_in_haystack"] == "evaluated (unknown)"
    assert model_b["mrl_stress"] == "not evaluated"
    assert model_b["crosslingual_retrieval"] == "not evaluated"
    assert model_b["cross_modal_retrieval"] == "not evaluated"
    smoke_coverage = app["coverage_table"]("All providers", "smoke", "").to_dict("records")
    assert [row["model"] for row in smoke_coverage] == ["Model A"]
    assert smoke_coverage[0]["covered_tasks"] == 4
    assert smoke_coverage[0]["needle_in_haystack"] == "evaluated (benchmark)"

    openai_coverage = app["coverage_table"]("openai", "All evidence tiers", "").to_dict("records")
    assert [row["model"] for row in openai_coverage] == ["Model A", "Model B"]
    crosslingual_coverage = app["coverage_table"](
        "All providers", "All evidence tiers", "crosslingual"
    ).to_dict("records")
    assert [row["model"] for row in crosslingual_coverage] == ["Model A"]

    coverage_note, _ = app["render_coverage"]("All providers", "smoke", "")
    assert "task declaration alone is not evaluation evidence" in coverage_note
    assert "scores are neither compared nor averaged across tasks" in coverage_note
    assert "not model quality" in coverage_note
    assert "not evaluated" in coverage_note

    manifest_text = (output / "export_manifest.yaml").read_text(encoding="utf-8")
    assert "models.jsonl" in manifest_text
    assert "tasks.jsonl" in manifest_text
    assert "declared_public_models:" in manifest_text
    assert "declared_public_tasks: 4" in manifest_text


def test_export_hf_space_loads_remote_public_task_catalog(tmp_path, monkeypatch) -> None:
    local_leaderboard = tmp_path / "local-leaderboard.csv"
    local_leaderboard.write_text(
        "task_id,model_id,model,provider,score,evidence_tier,is_latest_for_task_model\n"
        "mrl_stress,local-model,Local Model,local,0.5,smoke,true\n",
        encoding="utf-8",
    )
    output = export_space_repo(
        output_dir=tmp_path / "space",
        dataset_repo_id="example/remote-dataset",
        bundled_leaderboard=local_leaderboard,
    )

    remote_leaderboard = tmp_path / "remote-leaderboard.csv"
    remote_leaderboard.write_text(
        "task_id,model_id,model,provider,score,evidence_tier,is_latest_for_task_model\n"
        "needle_in_haystack,remote-model,Remote Model,remote,0.8,benchmark,true\n"
        "needle_in_haystack,legacy-only,Legacy Only,legacy,0.6,legacy,true\n"
        "late_chunking_retrieval,fixture-model,Fixture Model,local,1.0,fixture,true\n",
        encoding="utf-8",
    )
    remote_models = tmp_path / "remote-models.jsonl"
    remote_models.write_text(
        "\n".join(
            json.dumps(model)
            for model in [
                {
                    "id": "remote-model",
                    "display_name": "Remote Model",
                    "provider": "remote",
                    "modalities": ["text"],
                    "dimensions": 768,
                    "max_text_length": 8192,
                    "supports_mrl": True,
                    "access": "api",
                    "status": "active",
                    "source": "https://example.invalid/remote-model",
                    "publish": True,
                    "provider_kwargs": {"token": "must-not-leak"},
                    "api_key_env": "REMOTE_API_KEY",
                    "notes": "internal note",
                    "priority": 1,
                },
                {
                    "id": "declared-only",
                    "display_name": "Declared Only",
                    "provider": "remote",
                    "modalities": ["text", "image"],
                    "dimensions": 1024,
                    "max_text_length": 4096,
                    "supports_mrl": False,
                    "access": "weights",
                    "status": "active",
                    "source": "https://example.invalid/declared-only",
                    "publish": True,
                },
                {
                    "id": "private-model",
                    "display_name": "Private Model",
                    "provider": "remote",
                    "publish": False,
                },
                {
                    "id": "third-party-geevec-copy",
                    "display_name": "Third-party Copy",
                    "provider": "remote",
                    "publish": True,
                },
                {
                    "id": "hidden-provider-model",
                    "display_name": "Hidden Provider Model",
                    "provider": "geevec_lite",
                    "publish": True,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    remote_tasks = tmp_path / "remote-tasks.jsonl"
    remote_tasks.write_text(
        "\n".join(
            json.dumps(task)
            for task in [
                {
                    "id": "needle_in_haystack",
                    "display_name": "Remote Needle",
                    "description": "Remote scored task.",
                    "primary_metric": "overall_accuracy",
                    "publish": True,
                    "leaderboard_publish": True,
                },
                {
                    "id": "autonomous_driving",
                    "display_name": "Remote Driving",
                    "description": "Remote declared task without rows.",
                    "primary_metric": "avg_recall@1",
                    "publish": True,
                    "leaderboard_publish": True,
                },
                {
                    "id": "late_chunking_retrieval",
                    "display_name": "Fixture",
                    "description": "Private fixture.",
                    "primary_metric": "chunk_ndcg@10",
                    "publish": False,
                    "leaderboard_publish": False,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import sys
    import types

    huggingface_hub = types.ModuleType("huggingface_hub")

    def local_download(*, repo_id, repo_type, filename):
        assert repo_id == "example/remote-dataset"
        assert repo_type == "dataset"
        if filename == "tasks.jsonl":
            return str(remote_tasks)
        if filename == "models.jsonl":
            return str(remote_models)
        return str(remote_leaderboard)

    huggingface_hub.hf_hub_download = local_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    app = _load_generated_space_app(output, monkeypatch)

    assert app["TASKS"] == ["needle_in_haystack"]
    assert app["DECLARED_TASKS"] == ["autonomous_driving", "needle_in_haystack"]
    assert [row["model"] for row in app["ROWS"]] == ["Remote Model", "Legacy Only"]
    assert [model["id"] for model in app["MODEL_SPECS"]] == ["remote-model", "declared-only"]
    assert all(
        set(model)
        == {
            "access",
            "dimensions",
            "display_name",
            "id",
            "max_text_length",
            "modalities",
            "provider",
            "source",
            "status",
            "supports_mrl",
        }
        for model in app["MODEL_SPECS"]
    )
    remote_catalog = app["model_catalog_table"]("All catalog providers", "").to_dict("records")
    assert [(row["id"], row["evaluation_evidence"]) for row in remote_catalog] == [
        ("declared-only", "declared only - no public score rows"),
        ("remote-model", "public score rows available"),
    ]
    assert "legacy-only" not in {row["id"] for row in remote_catalog}
    coverage = app["coverage_table"]("All providers", "All evidence tiers", "").to_dict("records")
    assert coverage == [
        {
            "model": "Legacy Only",
            "provider": "legacy",
            "covered_tasks": 1,
            "autonomous_driving": "not evaluated",
            "needle_in_haystack": "evaluated (legacy)",
        },
        {
            "model": "Remote Model",
            "provider": "remote",
            "covered_tasks": 1,
            "autonomous_driving": "not evaluated",
            "needle_in_haystack": "evaluated (benchmark)",
        }
    ]


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
