from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mm_embed.benchmark.leaderboard import build_leaderboard
from mm_embed.benchmark.registry import (
    BenchmarkCatalog,
    EvaluationSourceClaim,
    ModelSpec,
    NegativeTrainingClaim,
    ReviewSpec,
    RunManifest,
    RunTask,
    TaskSpec,
    TrainingSourceClaim,
    load_catalog,
    load_run_manifest,
)
from mm_embed.benchmark.results import (
    import_legacy_result_file,
    load_jsonl,
    make_result_record,
    normalize_legacy_model_name,
)
from mm_embed.benchmark.training_overlap import (
    RELATIONSHIP_EFFECTS,
    RelationshipRegistry,
    assess_training_overlap,
    legacy_unknown_assessment,
    load_relationship_registry,
    validate_assessment_snapshot,
    validate_assessment_registry_binding,
    validate_catalog_contract,
)
from mm_embed.benchmark.runner import BenchmarkRunner
from mm_embed.hf_publish.export import export_dataset_repo, export_space_repo
from mm_embed.tasks.base import EvalResult


def test_default_catalog_and_run_manifests_load() -> None:
    catalog = load_catalog()

    assert "openai-text-embedding-3-large" in catalog.models
    assert "mrl_stress" in catalog.tasks
    assert catalog.models["geevec-lite-general"].publish is False
    assert catalog.models["geevec-api-general"].publish is False
    assert catalog.models["openai-text-embedding-3-large"].training_data.disclosure == "unknown"
    assert catalog.tasks["needle_in_haystack"].evaluation_sources.disclosure == "unknown"
    relationships = load_relationship_registry()
    assert relationships.sources == {}
    assert relationships.relationships == ()

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
            "data_overlap_status": "unknown",
            "task_training_status": "unknown",
            "zero_shot_status": "unknown",
            "overlap_reason_codes": "legacy_missing_contract",
            "overlap_relationship_registry_revision": "legacy",
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
    (tmp_path / "training_overlap_relationships.yaml").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "revision": "runner-test.1",
                "sources": [],
                "relationships": [],
            }
        ),
        encoding="utf-8",
    )

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
    assert written[0]["training_overlap_contract_version"] == "1"
    assert written[0]["training_overlap"]["zero_shot_status"] == "unknown"


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
                "metadata": {"results_path": "/home/example/private/results.jsonl"},
                "git_sha": "abc123",
            },
            "timestamps": {
                "duration_s": 0.9,
                "started_at": "2026-07-27T10:17:56+00:00",
                "finished_at": "2026-07-27T10:18:04+00:00",
            },
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
                "dataset_version": "needle-v1",
                "primary_metric": "overall_accuracy",
                "tags": [],
            },
            "metrics": {"overall_accuracy": 0.9},
            "details": {
                "input_cardinality": {"queries": 10, "documents": 99, "total": 109},
                "token_usage": 89430,
                "provider_latency_ms": 7357.978,
                "cost_usd": None,
                "fresh_provider_calls": True,
                "cache_enabled": False,
                "embedding_calls": [{"raw_prompt": "must-not-leak"}],
                "provider_kwargs": {"api_key_env": "MUST_NOT_LEAK"},
            },
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
        "run_started_at",
        "run_finished_at",
        "dataset_version",
        "input_count_total",
        "token_usage",
        "provider_latency_ms",
        "cost_usd",
        "fresh_provider_calls",
        "cache_enabled",
        "data_overlap_status",
        "task_training_status",
        "zero_shot_status",
        "overlap_reason_codes",
        "overlap_relationship_registry_revision",
    ]
    assert [row["evidence_tier"] for row in rows] == ["legacy", "smoke"]
    assert rows[0]["evidence_source"] == "legacy/results/baseline.json"
    assert rows[1]["evidence_source"] == "abc123"
    assert [row["task_model_duplicate_count"] for row in rows] == ["2", "2"]
    assert [row["task_model_run_rank"] for row in rows] == ["1", "2"]
    assert [row["is_latest_for_task_model"] for row in rows] == ["false", "true"]
    assert rows[0]["input_count_total"] == ""
    assert rows[1]["run_started_at"] == "2026-07-27T10:17:56+00:00"
    assert rows[1]["run_finished_at"] == "2026-07-27T10:18:04+00:00"
    assert rows[1]["dataset_version"] == "needle-v1"
    assert rows[1]["input_count_total"] == "109"
    assert rows[1]["token_usage"] == "89430"
    assert rows[1]["provider_latency_ms"] == "7357.978"
    assert rows[1]["cost_usd"] == ""
    assert rows[1]["fresh_provider_calls"] == "true"
    assert rows[1]["cache_enabled"] == "false"
    leaderboard_text = (output / "leaderboards" / "latest.csv").read_text(encoding="utf-8")
    assert "must-not-leak" not in leaderboard_text
    assert "MUST_NOT_LEAK" not in leaderboard_text
    assert "/home/example/private/results.jsonl" not in leaderboard_text

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
        "training_data",
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
    assert all(row["training_disclosure"] == "unknown" for row in model_catalog)
    assert all(row["lineage_disclosure"] == "unknown" for row in model_catalog)
    assert all(row["review_state"] == "pending" for row in model_catalog)
    assert all(row["model_revision"] == "unknown" for row in model_catalog)
    assert all(row["training_source_count"] == 0 for row in model_catalog)
    assert all(row["lineage_parent_count"] == 0 for row in model_catalog)
    assert "model-a" not in {row["id"] for row in model_catalog}
    model_note, _ = app["render_model_catalog"]("All catalog providers", "")
    assert "not a model ranking" in model_note
    assert "does not imply quality" in model_note
    assert "do not imply zero-shot evaluation or absence of training overlap" in model_note
    assert "missing, empty, incomplete, unresolved, or stale" in model_note
    assert "Score-only legacy identities remain available" in model_note

    task_catalog = app["task_catalog_table"](
        "All evaluation-source disclosures", "All catalog review states", ""
    ).to_dict("records")
    assert [row["display_name"] for row in task_catalog] == sorted(
        (row["display_name"] for row in task_catalog), key=str.lower
    )
    assert all(row["declaration"] == "declared in public task catalog" for row in task_catalog)
    assert all(row["evaluation_evidence"] == "public score rows available" for row in task_catalog)
    assert all(row["evaluation_source_disclosure"] == "unknown" for row in task_catalog)
    assert all(row["review_state"] == "pending" for row in task_catalog)
    assert all(row["source_count"] == 0 for row in task_catalog)
    assert next(row for row in task_catalog if row["id"] == "cross_modal_retrieval") == {
        "display_name": "Text-image retrieval",
        "id": "cross_modal_retrieval",
        "description": "Bidirectional COCO text-image retrieval with hard negative captions.",
        "required_modalities": "text, image",
        "primary_metric": "hard_avg_recall@1",
        "metric_direction": "higher",
        "dataset_version": "cross-modal-coco-hard-v1",
        "evaluation_source_disclosure": "unknown",
        "review_state": "pending",
        "source_count": 0,
        "declaration": "declared in public task catalog",
        "evaluation_evidence": "public score rows available",
    }
    assert [row["id"] for row in app["task_catalog_table"](
        "unknown", "pending", "needle"
    ).to_dict("records")] == ["needle_in_haystack"]
    task_note, _ = app["render_task_catalog"](
        "All evaluation-source disclosures", "All catalog review states", ""
    )
    assert "not a task ranking" in task_note
    assert "does not imply task quality, a global ranking" in task_note
    assert "zero-shot status, or absence of training overlap" in task_note
    assert "missing, empty, incomplete, unresolved, or stale" in task_note

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
                    "training_data": {
                        "disclosure": "complete",
                        "lineage_disclosure": "complete",
                        "model_revision": "remote-revision-1",
                        "source_ids": ["training-source-b", "training-source-a"],
                        "adapted_from": ["base-model"],
                        "review_state": "approved",
                        "private_notes": "must-not-leak",
                    },
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
                    "training_data": {
                        "disclosure": "",
                        "lineage_disclosure": "stale",
                        "model_revision": "",
                        "source_ids": [],
                        "adapted_from": [],
                        "review_state": "unresolved",
                    },
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
                    "required_modalities": ["text"],
                    "primary_metric": "overall_accuracy",
                    "metric_direction": "higher",
                    "dataset_version": "remote-needle-v1",
                    "publish": True,
                    "leaderboard_publish": True,
                    "evaluation_sources": {
                        "disclosure": "complete",
                        "sources": [
                            {
                                "source_id": "remote-evaluation-source",
                                "source_revision": "source-revision-1",
                                "config": "public",
                                "split": "test",
                                "transformation_id": "transform-1",
                                "private_notes": "must-not-leak",
                            }
                        ],
                        "review_state": "approved",
                        "private_notes": "must-not-leak",
                    },
                },
                {
                    "id": "autonomous_driving",
                    "display_name": "Remote Driving",
                    "description": "Remote declared task without rows.",
                    "required_modalities": ["text", "image"],
                    "primary_metric": "avg_recall@1",
                    "metric_direction": "higher",
                    "dataset_version": "remote-driving-v1",
                    "publish": True,
                    "leaderboard_publish": True,
                    "evaluation_sources": {
                        "disclosure": "stale",
                        "sources": [],
                        "review_state": "unresolved",
                    },
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
            "training_data",
        }
        for model in app["MODEL_SPECS"]
    )
    remote_catalog = app["model_catalog_table"]("All catalog providers", "").to_dict("records")
    assert [(row["id"], row["evaluation_evidence"]) for row in remote_catalog] == [
        ("declared-only", "declared only - no public score rows"),
        ("remote-model", "public score rows available"),
    ]
    assert next(row for row in remote_catalog if row["id"] == "remote-model") == {
        "display_name": "Remote Model",
        "id": "remote-model",
        "provider": "remote",
        "modalities": "text",
        "dimensions": 768,
        "max_text_length": 8192,
        "supports_mrl": "yes",
        "access": "api",
        "status": "active",
        "training_disclosure": "complete",
        "lineage_disclosure": "complete",
        "review_state": "approved",
        "model_revision": "remote-revision-1",
        "training_source_count": 2,
        "lineage_parent_count": 1,
        "declaration": "declared in public model catalog",
        "evaluation_evidence": "public score rows available",
        "source": "https://example.invalid/remote-model",
    }
    declared_only_model = next(row for row in remote_catalog if row["id"] == "declared-only")
    assert declared_only_model["training_disclosure"] == "unknown"
    assert declared_only_model["lineage_disclosure"] == "unknown"
    assert declared_only_model["review_state"] == "pending"
    assert declared_only_model["model_revision"] == "unknown"
    assert "must-not-leak" not in json.dumps(remote_catalog)
    assert "legacy-only" not in {row["id"] for row in remote_catalog}

    remote_task_catalog = app["task_catalog_table"](
        "All evaluation-source disclosures", "All catalog review states", ""
    ).to_dict("records")
    assert [(row["id"], row["evaluation_evidence"]) for row in remote_task_catalog] == [
        ("autonomous_driving", "declared only - no public score rows"),
        ("needle_in_haystack", "public score rows available"),
    ]
    remote_task = next(row for row in remote_task_catalog if row["id"] == "needle_in_haystack")
    assert remote_task["required_modalities"] == "text"
    assert remote_task["metric_direction"] == "higher"
    assert remote_task["dataset_version"] == "remote-needle-v1"
    assert remote_task["evaluation_source_disclosure"] == "complete"
    assert remote_task["review_state"] == "approved"
    assert remote_task["source_count"] == 1
    declared_only_task = next(row for row in remote_task_catalog if row["id"] == "autonomous_driving")
    assert declared_only_task["evaluation_source_disclosure"] == "unknown"
    assert declared_only_task["review_state"] == "pending"
    assert declared_only_task["source_count"] == 0
    assert "must-not-leak" not in json.dumps(remote_task_catalog)
    assert [row["id"] for row in app["task_catalog_table"](
        "complete", "approved", "remote"
    ).to_dict("records")] == ["needle_in_haystack"]
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


def test_export_hf_space_operational_evidence_is_unranked(tmp_path, monkeypatch) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s,evidence_tier,"
        "evidence_source,task_model_duplicate_count,task_model_run_rank,is_latest_for_task_model,"
        "run_started_at,run_finished_at,dataset_version,input_count_total,token_usage,provider_latency_ms,"
        "cost_usd,fresh_provider_calls,cache_enabled\n"
        "needle_in_haystack,Needle,model-z,Model Z,voyage,overall_accuracy,1.0,smoke:z,10.332,smoke,"
        "sha-z,1,1,true,2026-07-27T10:18:04+00:00,2026-07-27T10:18:15+00:00,needle-v1,109,90505,"
        "4667.493,,true,false\n"
        "needle_in_haystack,Needle,model-a,Model A,openai,overall_accuracy,0.1,smoke:a,8.046,smoke,"
        "sha-a,1,1,true,2026-07-27T10:17:56+00:00,2026-07-27T10:18:04+00:00,needle-v1,109,89430,"
        "7357.979,,true,false\n",
        encoding="utf-8",
    )

    output = export_space_repo(output_dir=tmp_path / "space", bundled_leaderboard=leaderboard)
    app = _load_generated_space_app(output, monkeypatch)

    note, table = app["render_operational"](
        "All providers", "All evidence tiers", "", False, 50
    )
    rows = table.to_dict("records")
    assert [row["model"] for row in rows] == ["Model A", "Model Z"]
    assert "score" not in rows[0]
    assert "rank" not in rows[0]
    assert rows[0]["task_id"] == "needle_in_haystack"
    assert rows[0]["evidence_tier"] == "smoke"
    assert rows[0]["run_id"] == "smoke:a"
    assert rows[0]["evidence_source"] == "sha-a"
    assert rows[0]["dataset_version"] == "needle-v1"
    assert rows[0]["input_count_total"] == "109"
    assert rows[0]["token_usage"] == "89430"
    assert rows[0]["provider_latency_ms"] == "7357.979"
    assert rows[0]["cost_usd"] == ""
    assert rows[0]["fresh_provider_calls"] == "true"
    assert rows[0]["cache_enabled"] == "false"
    assert "non-ranking per-run operational evidence" in note
    assert "not comparable across providers, routes, hardware, batching, task sizes, or cache states" in note
    assert "workload evidence, not quality" in note
    assert "Missing values mean unreported, not zero" in note
    assert "no price, throughput, normalized efficiency, energy, or CO2 value is inferred" in note


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


def _contract_provenance(evidence_revision: str = "evidence-r1") -> dict:
    return {
        "urls": ["https://example.invalid/evidence"],
        "evidence_revision": evidence_revision,
        "reviewed_at": "2026-07-28",
        "reviewed_by": "reviewer",
    }


def _contract_source(
    source_id: str,
    *,
    revision: str = "source-r1",
    canonical: bool = True,
    private_notes: str | None = None,
) -> dict:
    return {
        "id": source_id,
        "canonical": canonical,
        "locator": {"authority": "local", "revision": revision},
        "public_provenance": {
            "urls": [f"https://example.invalid/{source_id}"],
            "reviewed_at": "2026-07-28",
            "reviewed_by": "reviewer",
        },
        "review": {"state": "approved", "private_notes": private_notes},
    }


def _contract_relationship(
    relationship_id: str,
    subject: str,
    predicate: str,
    object_: str,
    *,
    subject_revision: str = "source-r1",
    object_revision: str = "source-r1",
    applies_to: dict | None = None,
    private_notes: str | None = None,
) -> dict:
    data_overlap, task_training, transitive = RELATIONSHIP_EFFECTS[predicate]
    applicability = {
        "subject_revision": subject_revision,
        "object_revision": object_revision,
        **(applies_to or {}),
    }
    return {
        "id": relationship_id,
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "effect": {
            "data_overlap": data_overlap,
            "task_training": task_training,
            "transitive": transitive,
        },
        "applies_to": applicability,
        "public_provenance": {
            "urls": [f"https://example.invalid/{relationship_id}"],
            "reviewed_at": "2026-07-28",
            "reviewed_by": "reviewer",
        },
        "review": {"state": "approved", "private_notes": private_notes},
    }


def _contract_registry(
    sources: list[dict],
    relationships: list[dict] | None = None,
    *,
    revision: str = "test.1",
) -> RelationshipRegistry:
    return RelationshipRegistry.from_dict(
        {
            "schema_version": "1",
            "revision": revision,
            "sources": sources,
            "relationships": relationships or [],
        }
    )


def _contract_model(
    model_id: str,
    *,
    source_claims: list[tuple[str, str]] | None = None,
    negative_claims: list[tuple[str, str]] | None = None,
    adapted_from: list[str] | None = None,
    disclosure: str = "partial",
    lineage_disclosure: str = "complete",
    reviewed: bool = True,
    evidence_revision: str = "model-evidence-r1",
    model_revision: str = "model-r1",
    private_notes: str | None = None,
) -> ModelSpec:
    training_data = {
        "disclosure": disclosure,
        "source_claims": [
            {
                "source_id": source_id,
                "relation": "trained_on",
                "scope": "material_samples",
                "source_revision": source_revision,
            }
            for source_id, source_revision in source_claims or []
        ],
        "negative_claims": [
            {
                "source_id": source_id,
                "relation": "not_trained_on",
                "scope": "material_samples",
                "source_revision": source_revision,
            }
            for source_id, source_revision in negative_claims or []
        ],
        "adapted_from": adapted_from or [],
        "lineage_disclosure": lineage_disclosure,
        "model_revision": model_revision,
        "public_provenance": _contract_provenance(evidence_revision),
        "review": {
            "state": "approved" if reviewed else "pending",
            "private_notes": private_notes,
        },
    }
    return ModelSpec.from_dict(
        {
            "id": model_id,
            "display_name": model_id,
            "provider": "invented",
            "training_data": training_data,
        },
        Path("invented-model.yaml"),
    )


def _contract_task(
    source_id: str,
    *,
    source_revision: str = "source-r1",
    dataset_version: str = "task-dataset-r1",
    evidence_revision: str = "task-evidence-r1",
    private_notes: str | None = None,
) -> TaskSpec:
    return TaskSpec.from_dict(
        {
            "id": "invented_task",
            "display_name": "Invented task",
            "task": "invented_task",
            "description": "Invented contract fixture.",
            "dataset_version": dataset_version,
            "evaluation_sources": {
                "disclosure": "complete",
                "sources": [
                    {
                        "source_id": source_id,
                        "usage": "evaluation",
                        "source_revision": source_revision,
                        "config": "default",
                        "split": "test",
                        "transformation_id": "invented-transform-r1",
                    }
                ],
                "public_provenance": _contract_provenance(evidence_revision),
                "review": {"state": "approved", "private_notes": private_notes},
            },
        },
        Path("invented-task.yaml"),
    )


def _contract_assessment(
    model: ModelSpec,
    task: TaskSpec,
    registry: RelationshipRegistry,
    *other_models: ModelSpec,
) -> tuple[dict, BenchmarkCatalog]:
    models = {item.id: item for item in (model, *other_models)}
    catalog = BenchmarkCatalog(root=Path("."), models=models, tasks={task.id: task})
    assessment = assess_training_overlap(
        model=model,
        task=task,
        catalog=catalog,
        relationship_registry=registry,
        assessed_at="2026-07-28T00:00:00Z",
    )
    return assessment, catalog


def _contract_result(
    run_id: str,
    model_id: str,
    score: float,
    *,
    assessment: dict | None = None,
    contract_version: bool = True,
    provider: str = "invented",
) -> dict:
    record = {
        "schema_version": "2.0",
        "run": {"id": run_id, "publish": True, "evidence_tier": "benchmark"},
        "timestamps": {"duration_s": 1.0},
        "model": {"id": model_id, "display_name": model_id, "provider": provider},
        "provider_result": {"provider": provider, "model_name": model_id},
        "task": {
            "id": "needle_in_haystack",
            "display_name": "Needle",
            "primary_metric": "overall_accuracy",
            "dataset_version": "needle-v1",
        },
        "metrics": {"overall_accuracy": score},
        "details": {},
        "error": None,
    }
    if contract_version:
        record["training_overlap_contract_version"] = "1"
    if assessment is not None:
        record["training_overlap"] = assessment
    return record


def test_training_overlap_unknown_is_not_zero_shot_and_filter_excludes_it(tmp_path, monkeypatch) -> None:
    registry = _contract_registry([_contract_source("local:eval")])
    task = _contract_task("local:eval")
    model = ModelSpec(id="missing-metadata", display_name="Missing metadata", provider="invented")
    assessment, _catalog = _contract_assessment(model, task, registry)

    assert assessment["data_overlap_status"] == "unknown"
    assert assessment["task_training_status"] == "unknown"
    assert assessment["zero_shot_status"] == "unknown"

    reviewed = dict(legacy_unknown_assessment())
    reviewed.update(
        {
            "relationship_registry_revision": "invented.1",
            "data_overlap_status": "declared_none",
            "task_training_status": "declared_none",
            "zero_shot_status": "reviewed_yes",
            "reason_codes": ["complete_reviewed_non_overlap"],
        }
    )
    leaderboard = tmp_path / "leaderboard.csv"
    rows = build_leaderboard(
        [
            _contract_result("unknown", "unknown-model", 0.9, assessment=assessment),
            _contract_result("reviewed", "reviewed-model", 0.8, assessment=reviewed),
        ]
    )
    with open(leaderboard, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = export_space_repo(output_dir=tmp_path / "space", bundled_leaderboard=leaderboard)
    app = _load_generated_space_app(output, monkeypatch)
    filtered = app["filtered_rows"](
        "needle_in_haystack",
        "All providers",
        "All evidence tiers",
        "",
        False,
        app["ALL_DATA_OVERLAP_STATUSES"],
        app["ALL_TASK_TRAINING_STATUSES"],
        True,
    )
    assert [row["run_id"] for row in filtered] == ["reviewed"]
    app_source = (output / "app.py").read_text(encoding="utf-8")
    assert 'label="Reviewed zero-shot only"' in app_source
    assert "Unknown means unreported, incomplete, unresolved, or stale" in app_source


def test_training_overlap_explicit_empty_requires_complete_review() -> None:
    registry = _contract_registry([_contract_source("local:eval")])
    task = _contract_task("local:eval")
    incomplete = _contract_model("incomplete", source_claims=[], disclosure="complete", reviewed=False)
    incomplete_catalog = BenchmarkCatalog(
        root=Path("."), models={incomplete.id: incomplete}, tasks={task.id: task}
    )
    with pytest.raises(ValueError, match="unreviewed complete empty"):
        validate_catalog_contract(incomplete_catalog, registry)

    reviewed = _contract_model("reviewed", source_claims=[], disclosure="complete")
    assessment, catalog = _contract_assessment(reviewed, task, registry)
    validate_catalog_contract(catalog, registry)
    assert assessment["data_overlap_status"] == "declared_none"
    assert assessment["task_training_status"] == "declared_none"
    assert assessment["zero_shot_status"] == "reviewed_yes"


def test_training_overlap_ambiguous_alias_fails_validation() -> None:
    sources = [
        _contract_source("local:alias", canonical=False),
        _contract_source("local:canonical-a"),
        _contract_source("local:canonical-b"),
    ]
    relationships = [
        _contract_relationship("alias-a", "local:alias", "alias_of", "local:canonical-a"),
        _contract_relationship("alias-b", "local:alias", "alias_of", "local:canonical-b"),
    ]
    with pytest.raises(ValueError, match="Ambiguous alias component"):
        _contract_registry(sources, relationships)


def test_training_overlap_unresolved_base_lineage_is_unknown() -> None:
    registry = _contract_registry([_contract_source("local:eval")])
    task = _contract_task("local:eval")
    child = _contract_model(
        "child",
        source_claims=[],
        disclosure="complete",
        adapted_from=["missing-parent"],
    )
    assessment, _catalog = _contract_assessment(child, task, registry)
    assert assessment["zero_shot_status"] == "unknown"
    assert "unresolved_model_lineage" in assessment["reason_codes"]


def test_training_overlap_positive_ancestor_propagates_to_descendant() -> None:
    registry = _contract_registry(
        [_contract_source("local:train"), _contract_source("local:eval")],
        [_contract_relationship("adapted", "local:train", "sampled_from", "local:eval")],
    )
    task = _contract_task("local:eval")
    base = _contract_model("base", source_claims=[("local:train", "source-r1")])
    child = _contract_model("child", source_claims=[], disclosure="complete", adapted_from=["base"])
    assessment, _catalog = _contract_assessment(child, task, registry, base)
    assert assessment["data_overlap_status"] == "adapted"
    assert assessment["zero_shot_status"] == "no"
    assert assessment["matched_model_ids"] == ["base"]


def test_training_overlap_multi_parent_positive_dominates_and_unresolved_prevents_negative() -> None:
    registry = _contract_registry([_contract_source("local:eval")])
    task = _contract_task("local:eval")
    overlapping = _contract_model("overlapping", source_claims=[("local:eval", "source-r1")])
    clean = _contract_model("clean", source_claims=[], disclosure="complete")
    child = _contract_model(
        "child",
        source_claims=[],
        disclosure="complete",
        adapted_from=["overlapping", "clean"],
    )
    positive, _catalog = _contract_assessment(child, task, registry, overlapping, clean)
    assert positive["data_overlap_status"] == "exact"
    assert positive["zero_shot_status"] == "no"

    unresolved = _contract_model(
        "unresolved-child",
        source_claims=[],
        disclosure="complete",
        adapted_from=["clean", "missing-parent"],
    )
    unknown, _catalog = _contract_assessment(unresolved, task, registry, clean)
    assert unknown["zero_shot_status"] == "unknown"
    assert "unresolved_model_lineage" in unknown["reason_codes"]


def test_training_overlap_transitive_material_relationship_detects_adapted_overlap() -> None:
    registry = _contract_registry(
        [_contract_source("local:a"), _contract_source("local:b"), _contract_source("local:c")],
        [
            _contract_relationship("a-from-b", "local:a", "sampled_from", "local:b"),
            _contract_relationship("b-from-c", "local:b", "translated_from", "local:c"),
        ],
    )
    model = _contract_model("model", source_claims=[("local:a", "source-r1")])
    task = _contract_task("local:c")
    assessment, _catalog = _contract_assessment(model, task, registry)
    assert assessment["data_overlap_status"] == "adapted"
    assert assessment["relationship_ids"] == ["a-from-b", "b-from-c"]


def test_training_overlap_similar_task_relationship_is_not_transitive() -> None:
    registry = _contract_registry(
        [_contract_source("local:a"), _contract_source("local:b"), _contract_source("local:c")],
        [
            _contract_relationship("a-sim-b", "local:a", "similar_task_to", "local:b"),
            _contract_relationship("b-sim-c", "local:b", "similar_task_to", "local:c"),
        ],
    )
    model = _contract_model("model", source_claims=[("local:a", "source-r1")])
    task = _contract_task("local:c")
    assessment, _catalog = _contract_assessment(model, task, registry)
    assert assessment["data_overlap_status"] == "unknown"
    assert assessment["task_training_status"] == "unknown"
    assert assessment["zero_shot_status"] == "unknown"


def test_training_overlap_stale_mapping_yields_unknown() -> None:
    registry = _contract_registry(
        [_contract_source("local:train"), _contract_source("local:eval")],
        [
            _contract_relationship(
                "stale-adaptation",
                "local:train",
                "sampled_from",
                "local:eval",
                applies_to={"task_dataset_version": "old-task-version"},
            )
        ],
    )
    model = _contract_model("model", source_claims=[("local:train", "source-r1")])
    task = _contract_task("local:eval", dataset_version="new-task-version")
    assessment, _catalog = _contract_assessment(model, task, registry)
    assert assessment["zero_shot_status"] == "unknown"
    assert {"stale_relationship", "stale_task_evidence"}.issubset(assessment["reason_codes"])


def test_training_overlap_conflicting_claims_fail_validation() -> None:
    registry = _contract_registry([_contract_source("local:eval")])
    task = _contract_task("local:eval")
    model = _contract_model(
        "conflict",
        source_claims=[("local:eval", "source-r1")],
        negative_claims=[("local:eval", "source-r1")],
    )
    catalog = BenchmarkCatalog(root=Path("."), models={model.id: model}, tasks={task.id: task})
    with pytest.raises(ValueError, match="conflicting positive and negative"):
        validate_catalog_contract(catalog, registry)


def test_training_overlap_private_notes_never_enter_public_artifacts(tmp_path) -> None:
    sentinel = "PRIVATE_OVERLAP_SENTINEL_7B5C2E19"
    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "models").mkdir(parents=True)
    (benchmark_root / "tasks").mkdir(parents=True)
    registry_data = {
        "schema_version": "1",
        "revision": "private-test.1",
        "sources": [_contract_source("local:eval", private_notes=sentinel)],
        "relationships": [],
    }
    (benchmark_root / "training_overlap_relationships.yaml").write_text(
        json.dumps(registry_data), encoding="utf-8"
    )
    model = _contract_model(
        "public-model",
        source_claims=[("local:eval", "source-r1")],
        private_notes=sentinel,
    )
    task = _contract_task("local:eval", private_notes=sentinel)
    model_row = {
        "models": [
            {
                "id": model.id,
                "display_name": model.display_name,
                "provider": model.provider,
                "provider_kwargs": {"secret": sentinel},
                "training_data": {
                    "disclosure": "partial",
                    "source_claims": [
                        {
                            "source_id": "local:eval",
                            "source_revision": "source-r1",
                            "relation": "trained_on",
                            "scope": "material_samples",
                        }
                    ],
                    "negative_claims": [],
                    "adapted_from": [],
                    "lineage_disclosure": "complete",
                    "model_revision": "model-r1",
                    "public_provenance": _contract_provenance("model-evidence-r1"),
                    "review": {"state": "approved", "private_notes": sentinel},
                },
            }
        ]
    }
    task_row = {
        "tasks": [
            {
                "id": task.id,
                "display_name": task.display_name,
                "task": task.task,
                "description": task.description,
                "dataset_version": task.dataset_version,
                "primary_metric": "overall_accuracy",
                "evaluation_sources": {
                    "disclosure": "complete",
                    "sources": [
                        {
                            "source_id": "local:eval",
                            "source_revision": "source-r1",
                            "usage": "evaluation",
                            "config": "default",
                            "split": "test",
                            "transformation_id": "invented-transform-r1",
                        }
                    ],
                    "public_provenance": _contract_provenance("task-evidence-r1"),
                    "review": {"state": "approved", "private_notes": sentinel},
                },
            }
        ]
    }
    (benchmark_root / "models" / "core.yaml").write_text(json.dumps(model_row), encoding="utf-8")
    (benchmark_root / "tasks" / "core.yaml").write_text(json.dumps(task_row), encoding="utf-8")
    registry = load_relationship_registry(benchmark_root / "training_overlap_relationships.yaml")
    assessment, _catalog = _contract_assessment(model, task, registry)
    record = _contract_result("public-run", model.id, 0.5, assessment=assessment)
    record["model"]["training_data"] = {"review": {"private_notes": sentinel}}
    record["task"]["evaluation_sources"] = {"review": {"private_notes": sentinel}}
    record["details"] = {"raw_source_payload": sentinel}
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = export_dataset_repo(
        output_dir=tmp_path / "dataset",
        benchmark_root=benchmark_root,
        results_path=results,
    )
    space = export_space_repo(
        output_dir=tmp_path / "space",
        benchmark_root=benchmark_root,
        bundled_leaderboard=dataset / "leaderboards" / "latest.csv",
    )
    compile((space / "app.py").read_text(encoding="utf-8"), str(space / "app.py"), "exec")
    for output in (dataset, space):
        for path in output.rglob("*"):
            if path.is_file():
                assert sentinel not in path.read_text(encoding="utf-8")


def test_public_result_sanitizer_preserves_safe_baseline_evidence(tmp_path) -> None:
    safe_record = {
        "schema_version": "2.0",
        "evaluation_level": "embedding",
        "evaluation": {
            "level": "embedding",
            "mode": "ranking",
            "leaderboard_surface": "embedding",
            "diagnostics": {"folds": ["fold-a", "fold-b"], "complete": True},
        },
        "subject": {
            "kind": "embedding_model",
            "metadata": {"family": "invented-family", "revision_label": "public-r1"},
        },
        "run": {
            "id": "baseline-safe-run",
            "description": "Safe baseline compatibility fixture",
            "metadata": {
                "suite": "compatibility",
                "relative_results_path": "results/baseline-safe.jsonl",
                "nested": {"attempt": 2, "labels": ["safe", "baseline"]},
            },
            "publish": True,
            "evidence_tier": "benchmark",
            "git_sha": "abc123",
        },
        "timestamps": {
            "started_at": "2026-07-28T00:00:00Z",
            "finished_at": "2026-07-28T00:00:30Z",
            "duration_s": 30.0,
        },
        "model": {
            "id": "baseline-safe-model",
            "display_name": "Baseline Safe Model",
            "provider": "invented",
            "provider_kwargs": {
                "model": "invented-safe-v1",
                "dimensions": 768,
                "encoding_format": "float",
                "endpoint": "https://example.invalid/v1/embeddings",
                "nested": {"batch_size": 32, "normalize": True},
            },
            "modalities": ["text"],
            "dimensions": 768,
            "access": "api",
            "tags": ["baseline", "safe"],
        },
        "task": {
            "id": "needle_in_haystack",
            "display_name": "Needle",
            "task": "needle_in_haystack",
            "dataset_version": "needle-v1",
            "primary_metric": "overall_accuracy",
            "metric_direction": "higher",
            "kwargs": {
                "haystack_lengths": [4000, 8000],
                "needle_positions": [0.25, 0.75],
                "nested": {"diagnostic_mode": "strict", "max_cases": 12},
            },
            "tags": ["long-context", "text"],
        },
        "provider_result": {
            "provider": "invented",
            "model_name": "invented-safe-v1",
            "request_count": 4,
            "batch_sizes": [3, 3, 3, 3],
        },
        "metrics": {
            "overall_accuracy": 0.75,
            "by_length": {"4000": 0.8, "8000": 0.7},
        },
        "details": {
            "input_cardinality": {"queries": 12, "documents": 48, "total": 60},
            "task_diagnostics": {
                "per_query": [
                    {"query_id": "query/1", "correct": True, "rank": 1},
                    {"query_id": "query/2", "correct": False, "rank": 3},
                ],
                "buckets": {"short": {"count": 6, "accuracy": 0.8}},
            },
            "cache": {"enabled": False, "hits": 0},
            "provider_latency_ms": 120.5,
            "embedding_calls": [
                {"request_id": "request-1", "batch_size": 3, "latency_ms": 12.5},
                {"request_id": "request-2", "batch_size": 3, "latency_ms": 11.0},
            ],
            "notes": ["safe diagnostic note", "retry not required"],
        },
        "error": "provider timeout after 30 seconds",
        "custom_public_evidence": {
            "attempts": [
                {"id": "attempt-1", "status": "timeout"},
                {"id": "attempt-2", "status": "complete"},
            ]
        },
    }
    record = json.loads(json.dumps(safe_record))
    private_sentinel = "PRIVATE_NOTES_SENTINEL_8D81E4"
    secret_value = "sk-secretvalue1234567890"
    unsafe_path = "/home/reviewer/private/results.jsonl"
    record["run"]["metadata"].update(
        {
            "api_key_env": "INVENTED_API_KEY",
            "authorization": "Bearer private-token-123456",
            "diagnostic_note": "api_key=private-value-123456",
            "unsafe_path": unsafe_path,
            "secret_copy": secret_value,
        }
    )
    record["model"]["provider_kwargs"].update(
        {
            "api_key": secret_value,
            "api_key_env": "INVENTED_API_KEY",
            "credential_copy": secret_value,
            "local_model_path": "/data2/private/model",
        }
    )
    record["model"]["training_data"] = {
        "review": {"state": "approved", "private_notes": private_sentinel}
    }
    record["task"]["kwargs"].update(
        {
            "password": "private-password",
            "local_cache": unsafe_path,
            "secret_value_copy": secret_value,
        }
    )
    record["task"]["evaluation_sources"] = {
        "review": {"state": "approved", "private_notes": private_sentinel}
    }
    record["provider_result"]["token"] = secret_value
    record["details"].update(
        {
            "raw_source_payload": private_sentinel,
            "review": {"private_notes": private_sentinel},
            "artifact_path": "/data2/private/artifact.json",
            "credential_note": secret_value,
        }
    )
    record["details"]["embedding_calls"][0].update(
        {"raw_prompt": private_sentinel, "api_key_env": "INVENTED_API_KEY"}
    )
    record["private_notes"] = private_sentinel

    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = export_dataset_repo(output_dir=tmp_path / "dataset", results_path=results)
    exported = load_jsonl(output / "results" / "latest.jsonl")

    expected = json.loads(json.dumps(safe_record))
    expected["training_overlap"] = legacy_unknown_assessment()
    assert exported == [expected]
    assert exported[0]["model"]["provider_kwargs"] == safe_record["model"]["provider_kwargs"]
    assert exported[0]["task"]["kwargs"] == safe_record["task"]["kwargs"]
    assert exported[0]["run"]["metadata"] == safe_record["run"]["metadata"]
    assert exported[0]["details"] == safe_record["details"]
    assert exported[0]["error"] == safe_record["error"]

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file()
    )
    for forbidden in (private_sentinel, secret_value, unsafe_path, "INVENTED_API_KEY"):
        assert forbidden not in public_text
    public_result_text = json.dumps(exported[0], sort_keys=True)
    assert "training_data" not in public_result_text
    assert "evaluation_sources" not in public_result_text


def test_training_overlap_legacy_rows_export_unknown_without_recomputation(tmp_path) -> None:
    records = [
        _contract_result("legacy-high", "model-high", 0.9, contract_version=False),
        _contract_result("legacy-low", "model-low", 0.4, contract_version=False),
    ]
    rows = build_leaderboard(records)
    assert [row["score"] for row in rows] == [0.9, 0.4]
    assert all(row["zero_shot_status"] == "unknown" for row in rows)
    assert all(row["overlap_reason_codes"] == "legacy_missing_contract" for row in rows)

    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    output = export_dataset_repo(output_dir=tmp_path / "dataset", results_path=results)
    exported_records = load_jsonl(output / "results" / "latest.jsonl")
    assert all(record["training_overlap"]["reason_codes"] == ["legacy_missing_contract"] for record in exported_records)
    with open(output / "leaderboards" / "latest.csv", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        exported_rows = list(reader)
    assert reader.fieldnames[-5:] == [
        "data_overlap_status",
        "task_training_status",
        "zero_shot_status",
        "overlap_reason_codes",
        "overlap_relationship_registry_revision",
    ]
    assert [row["score"] for row in exported_rows] == ["0.9", "0.4"]


def test_training_overlap_status_changes_preserve_score_order_and_provenance(tmp_path) -> None:
    unknown = legacy_unknown_assessment()
    reviewed = dict(unknown)
    reviewed.update(
        {
            "relationship_registry_revision": "invented.1",
            "data_overlap_status": "declared_none",
            "task_training_status": "declared_none",
            "zero_shot_status": "reviewed_yes",
            "reason_codes": ["complete_reviewed_non_overlap"],
        }
    )
    exact = dict(unknown)
    exact.update(
        {
            "relationship_registry_revision": "invented.1",
            "data_overlap_status": "exact",
            "task_training_status": "same_task",
            "zero_shot_status": "no",
            "reason_codes": ["exact_source_match", "same_task_exposure"],
        }
    )
    records_a = [
        _contract_result("run-low", "same-model", 0.4, assessment=reviewed),
        _contract_result("run-high", "same-model", 0.9, assessment=exact),
    ]
    records_b = [
        _contract_result("run-low", "same-model", 0.4, assessment=exact),
        _contract_result("run-high", "same-model", 0.9, assessment=reviewed),
    ]
    comparable = []
    for name, records in (("a", records_a), ("b", records_b)):
        results = tmp_path / f"{name}.jsonl"
        results.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        output = export_dataset_repo(output_dir=tmp_path / name, results_path=results)
        with open(output / "leaderboards" / "latest.csv", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        comparable.append(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "data_overlap_status",
                        "task_training_status",
                        "zero_shot_status",
                        "overlap_reason_codes",
                        "overlap_relationship_registry_revision",
                    }
                }
                for row in rows
            ]
        )
        assert [row["score"] for row in rows] == ["0.9", "0.4"]
        assert [row["task_model_duplicate_count"] for row in rows] == ["2", "2"]
        assert {row["task_model_run_rank"] for row in rows} == {"1", "2"}
        assert {row["is_latest_for_task_model"] for row in rows} == {"false", "true"}
        assert all(row["evidence_tier"] == "benchmark" for row in rows)
    assert comparable[0] == comparable[1]


def test_training_overlap_does_not_use_heuristic_matching() -> None:
    registry = _contract_registry(
        [
            _contract_source("local:CaseSensitive"),
            _contract_source("local:casesensitive"),
            _contract_source("local:path/shared-name"),
            _contract_source("other:shared-name"),
        ]
    )
    task = _contract_task("local:casesensitive")
    model = _contract_model("model-local:casesensitive", source_claims=[("local:CaseSensitive", "source-r1")])
    assessment, _catalog = _contract_assessment(model, task, registry)
    assert assessment["data_overlap_status"] == "unknown"
    assert assessment["task_training_status"] == "unknown"
    assert assessment["matched_training_source_ids"] == []


def test_training_overlap_post_contract_publication_requires_assessment(tmp_path) -> None:
    missing = _contract_result("missing", "model", 0.5, assessment=None)
    results = tmp_path / "missing.jsonl"
    results.write_text(json.dumps(missing) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing training_overlap"):
        export_dataset_repo(output_dir=tmp_path / "rejected", results_path=results)

    explicit_unknown = legacy_unknown_assessment()
    explicit_unknown.update(
        {
            "relationship_registry_revision": "invented.1",
            "reason_codes": ["model_training_disclosure_unknown"],
        }
    )
    allowed = _contract_result("allowed", "model", 0.5, assessment=explicit_unknown)
    results.write_text(json.dumps(allowed) + "\n", encoding="utf-8")
    output = export_dataset_repo(output_dir=tmp_path / "allowed", results_path=results)
    assert load_jsonl(output / "results" / "latest.jsonl")[0]["training_overlap"]["zero_shot_status"] == "unknown"


def test_training_overlap_relationship_digest_invalidates_same_revision_snapshot() -> None:
    sources = [_contract_source("local:a"), _contract_source("local:b")]
    registry_a = _contract_registry(
        sources,
        [_contract_relationship("related", "local:a", "same_task_as", "local:b")],
        revision="digest-test.1",
    )
    registry_b = _contract_registry(
        sources,
        [_contract_relationship("related", "local:a", "similar_task_to", "local:b")],
        revision="digest-test.1",
    )
    model = _contract_model("model", source_claims=[("local:a", "source-r1")])
    task = _contract_task("local:b")
    assessment, _catalog = _contract_assessment(model, task, registry_a)
    assert registry_a.sha256 != registry_b.sha256
    with pytest.raises(ValueError, match="digest is stale"):
        validate_assessment_registry_binding(assessment, registry_b)


def test_make_result_record_uses_catalog_root_relationship_registry(tmp_path) -> None:
    benchmark_root = tmp_path / "custom-benchmark"
    benchmark_root.mkdir()
    registry_data = {
        "schema_version": "1",
        "revision": "custom-root.7",
        "sources": [_contract_source("local:custom")],
        "relationships": [],
    }
    relationship_path = benchmark_root / "training_overlap_relationships.yaml"
    relationship_path.write_text(json.dumps(registry_data), encoding="utf-8")
    model = _contract_model("custom-model", source_claims=[("local:custom", "source-r1")])
    task = _contract_task("local:custom")
    catalog = BenchmarkCatalog(root=benchmark_root, models={model.id: model}, tasks={task.id: task})

    record = make_result_record(
        run=RunManifest(id="custom-root-run"),
        model=model,
        task=task,
        run_task=RunTask(id=task.id),
        result=EvalResult(
            task_name=task.task,
            provider_name="invented",
            model_name=model.id,
            metrics={"overall_accuracy": 1.0},
        ),
        started_at="2026-07-28T00:00:00Z",
        finished_at="2026-07-28T00:00:01Z",
        duration_s=1.0,
        catalog=catalog,
    )

    custom_registry = load_relationship_registry(relationship_path)
    assert record["training_overlap"]["relationship_registry_revision"] == "custom-root.7"
    assert record["training_overlap"]["relationship_registry_sha256"] == custom_registry.sha256
    assert record["training_overlap"]["data_overlap_status"] == "exact"


def test_registry_parsing_rejects_unsupported_declaration_semantics() -> None:
    base_model = {
        "id": "invalid-model",
        "display_name": "Invalid model",
        "provider": "invented",
        "training_data": {
            "source_claims": [{"source_id": "local:source", "source_revision": "source-r1"}],
            "negative_claims": [],
            "public_provenance": _contract_provenance(),
            "review": {"state": "approved"},
        },
    }
    invalid_model_cases = [
        ("source_claims", "relation", "pretrained_on", "training source relation"),
        ("source_claims", "scope", "metadata_only", "training source scope"),
        ("negative_claims", "relation", "never_seen", "negative training relation"),
        ("negative_claims", "scope", "task_family", "negative training scope"),
    ]
    for claim_group, key, value, message in invalid_model_cases:
        payload = json.loads(json.dumps(base_model))
        payload["training_data"]["source_claims"] = []
        payload["training_data"]["negative_claims"] = []
        payload["training_data"][claim_group] = [
            {"source_id": "local:source", "source_revision": "source-r1", key: value}
        ]
        with pytest.raises(ValueError, match=message):
            ModelSpec.from_dict(payload, Path("invalid-model.yaml"))

    invalid_review = json.loads(json.dumps(base_model))
    invalid_review["training_data"]["review"]["state"] = "trusted"
    with pytest.raises(ValueError, match="Unsupported review state"):
        ModelSpec.from_dict(invalid_review, Path("invalid-model.yaml"))

    base_task = {
        "id": "invalid-task",
        "display_name": "Invalid task",
        "task": "invalid_task",
        "description": "Invalid fixture.",
        "evaluation_sources": {
            "sources": [{"source_id": "local:source", "usage": "evaluation"}],
            "review": {"state": "approved"},
        },
    }
    invalid_usage = json.loads(json.dumps(base_task))
    invalid_usage["evaluation_sources"]["sources"][0]["usage"] = "training"
    with pytest.raises(ValueError, match="evaluation source usage"):
        TaskSpec.from_dict(invalid_usage, Path("invalid-task.yaml"))

    invalid_task_review = json.loads(json.dumps(base_task))
    invalid_task_review["evaluation_sources"]["review"]["state"] = "trusted"
    with pytest.raises(ValueError, match="Unsupported review state"):
        TaskSpec.from_dict(invalid_task_review, Path("invalid-task.yaml"))


def test_assessment_rejects_directly_constructed_unsupported_semantics() -> None:
    registry = _contract_registry([_contract_source("local:eval")])
    task = _contract_task("local:eval")
    model = _contract_model("model", source_claims=[("local:eval", "source-r1")])
    invalid_training = replace(
        model.training_data,
        source_claims=[
            TrainingSourceClaim(
                source_id="local:eval",
                relation="pretrained_on",
                scope="material_samples",
                source_revision="source-r1",
            )
        ],
    )
    invalid_model = replace(model, training_data=invalid_training)
    catalog = BenchmarkCatalog(root=Path("."), models={invalid_model.id: invalid_model}, tasks={task.id: task})
    with pytest.raises(ValueError, match="unsupported training relation"):
        assess_training_overlap(
            model=invalid_model,
            task=task,
            catalog=catalog,
            relationship_registry=registry,
            assessed_at="2026-07-28T00:00:00Z",
        )

    invalid_evaluation = replace(
        task.evaluation_sources,
        sources=[
            EvaluationSourceClaim(
                source_id="local:eval",
                usage="training",
                source_revision="source-r1",
            )
        ],
    )
    invalid_task = replace(task, evaluation_sources=invalid_evaluation)
    catalog = BenchmarkCatalog(root=Path("."), models={model.id: model}, tasks={invalid_task.id: invalid_task})
    with pytest.raises(ValueError, match="unsupported evaluation usage"):
        assess_training_overlap(
            model=model,
            task=invalid_task,
            catalog=catalog,
            relationship_registry=registry,
            assessed_at="2026-07-28T00:00:00Z",
        )

    invalid_review_model = replace(
        model,
        training_data=replace(model.training_data, review=ReviewSpec(state="trusted")),
    )
    catalog = BenchmarkCatalog(
        root=Path("."), models={invalid_review_model.id: invalid_review_model}, tasks={task.id: task}
    )
    with pytest.raises(ValueError, match="unsupported review state"):
        validate_catalog_contract(catalog, registry)

    invalid_negative_model = replace(
        model,
        training_data=replace(
            model.training_data,
            source_claims=[],
            negative_claims=[
                NegativeTrainingClaim(
                    source_id="local:eval",
                    relation="not_trained_on",
                    scope="task_family",
                    source_revision="source-r1",
                )
            ],
        ),
    )
    catalog = BenchmarkCatalog(
        root=Path("."), models={invalid_negative_model.id: invalid_negative_model}, tasks={task.id: task}
    )
    with pytest.raises(ValueError, match="unsupported negative scope"):
        validate_catalog_contract(catalog, registry)


def test_relationship_registry_rejects_unsupported_review_states() -> None:
    invalid_source = _contract_source("local:source")
    invalid_source["review"]["state"] = "trusted"
    with pytest.raises(ValueError, match="Source 'local:source' has unsupported review state"):
        _contract_registry([invalid_source])

    relationship = _contract_relationship("related", "local:a", "same_task_as", "local:b")
    relationship["review"]["state"] = "trusted"
    with pytest.raises(ValueError, match="Relationship 'related' has unsupported review state"):
        _contract_registry([_contract_source("local:a"), _contract_source("local:b")], [relationship])


@pytest.mark.parametrize("predicate", ["same_task_as", "similar_task_to"])
def test_alias_component_does_not_inherit_unrelated_task_edge(predicate) -> None:
    registry = _contract_registry(
        [
            _contract_source("local:alias", canonical=False),
            _contract_source("local:canonical"),
            _contract_source("local:unrelated"),
        ],
        [
            _contract_relationship("alias", "local:alias", "alias_of", "local:canonical"),
            _contract_relationship("unrelated-edge", "local:canonical", predicate, "local:unrelated"),
        ],
    )
    model = _contract_model("model", source_claims=[("local:alias", "source-r1")])
    task = _contract_task("local:canonical")
    assessment, _catalog = _contract_assessment(model, task, registry)

    assert assessment["data_overlap_status"] == "exact"
    assert assessment["task_training_status"] == "unknown"
    assert assessment["zero_shot_status"] == "no"
    assert assessment["relationship_ids"] == ["alias"]


@pytest.mark.parametrize(
    ("data_status", "task_status", "zero_shot_status"),
    [
        ("exact", "unknown", "reviewed_yes"),
        ("unknown", "unknown", "no"),
        ("declared_none", "declared_none", "unknown"),
        ("unknown", "similar_task", "reviewed_yes"),
    ],
)
def test_frozen_assessment_rejects_inconsistent_zero_shot_status(
    data_status,
    task_status,
    zero_shot_status,
) -> None:
    snapshot = legacy_unknown_assessment()
    snapshot.update(
        {
            "data_overlap_status": data_status,
            "task_training_status": task_status,
            "zero_shot_status": zero_shot_status,
        }
    )
    with pytest.raises(ValueError, match="zero_shot_status is inconsistent"):
        validate_assessment_snapshot(snapshot)


def test_public_export_rejects_inconsistent_frozen_assessment(tmp_path) -> None:
    snapshot = legacy_unknown_assessment()
    snapshot.update(
        {
            "relationship_registry_revision": "invented.1",
            "data_overlap_status": "exact",
            "task_training_status": "unknown",
            "zero_shot_status": "reviewed_yes",
            "reason_codes": ["exact_source_match"],
        }
    )
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(_contract_result("invalid", "model", 0.5, assessment=snapshot)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="zero_shot_status is inconsistent"):
        export_dataset_repo(output_dir=tmp_path / "dataset", results_path=results)
