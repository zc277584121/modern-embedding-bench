from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from mm_embed.benchmark.leaderboard import build_leaderboard
from mm_embed.benchmark.registry import load_catalog
from mm_embed.data.agent_skill_routing_fixture import (
    LICENSE_STATUS,
    SOURCE_KIND,
    SOURCE_REVISION,
    fixture_counts,
    fixture_slice_counts,
    fixture_to_dict,
    load_agent_skill_routing_fixture,
    serialize_fixture,
    serialize_skill_document,
    validate_agent_skill_routing_fixture,
)
from mm_embed.hf_publish.export import export_dataset_repo, export_space_repo
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType
from mm_embed.tasks.agent_skill_routing import evaluate_compatibility_scores, evaluate_risk_scores
from mm_embed.tasks.registry import get_task


class ShapeOnlyTextProvider(EmbeddingProvider):
    name = "agent-skill-routing-shape-only-local"
    model = "fixture-shape-test-double"
    supported_modalities = {ModalityType.TEXT}

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str | None, int]] = []

    def embed_with_cache(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        return self.embed(inputs, dimensions=dimensions, task_type=task_type)

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        self.calls.append((task_type, len(inputs)))
        dim = dimensions or 4
        embeddings = np.zeros((len(inputs), dim), dtype=float)
        for row in range(len(inputs)):
            embeddings[row, row % dim] = 1.0
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=0.0,
        )


def _score_matrix(query_count: int, skill_ids: list[str], rows: list[dict[str, float]]) -> np.ndarray:
    matrix = np.full((query_count, len(skill_ids)), -1.0, dtype=float)
    skill_index = {skill_id: index for index, skill_id in enumerate(skill_ids)}
    for row_index, scores in enumerate(rows):
        for skill_id, score in scores.items():
            matrix[row_index, skill_index[skill_id]] = score
    return matrix


def test_agent_skill_routing_fixture_shape_validation_and_serialization() -> None:
    first = load_agent_skill_routing_fixture()
    second = load_agent_skill_routing_fixture()

    assert fixture_counts(first) == {
        "n_skills": 12,
        "n_queries": 4,
        "n_qrels": 6,
        "n_compatible_sets": 2,
        "n_rejected_sets": 2,
        "n_risk_pairs": 2,
        "n_hard_negatives": 4,
    }
    assert fixture_slice_counts(first.queries) == {"compatible_set": 2, "same_capability_risk": 2}
    assert first.source_kind == SOURCE_KIND
    assert first.source_revision == SOURCE_REVISION
    assert first.license_status == LICENSE_STATUS
    assert first.public_score_eligible is False
    assert serialize_fixture(first) == serialize_fixture(second)
    assert json.dumps(fixture_to_dict(first), sort_keys=True) == json.dumps(fixture_to_dict(second), sort_keys=True)

    serialized_documents = "\n".join(serialize_skill_document(skill) for skill in first.skills).lower()
    assert "http://" not in serialized_documents
    assert "https://" not in serialized_documents
    assert "github" not in serialized_documents
    assert "huggingface" not in serialized_documents
    assert "@" not in serialized_documents

    corrupted_skill = replace(first.skills[0], public_score_eligible=True)
    with pytest.raises(ValueError, match="ineligible for public scoring"):
        validate_agent_skill_routing_fixture(replace(first, skills=(corrupted_skill, *first.skills[1:])))


def test_compatibility_metrics_use_exact_multi_positive_definitions_and_skill_id_ties() -> None:
    fixture = load_agent_skill_routing_fixture()
    query_ids = ["q_compat_schema_validate", "q_compat_log_triage"]
    skill_ids = [skill.skill_id for skill in fixture.skills]
    scores = _score_matrix(
        2,
        skill_ids,
        [
            {
                "schema_contract_diff": 1.0,
                "schema_doc_outline": 0.8,
                "schema_sample_mapper": 0.8,
            },
            {
                "log_volume_rollup": 1.0,
                "error_signature_cluster": 0.8,
                "log_redaction_review": 0.7,
            },
        ],
    )

    metrics = evaluate_compatibility_scores(scores, query_ids, skill_ids, fixture)

    assert metrics == pytest.approx(
        {
            "recall@1": 0.25,
            "complete_set@1": 0.0,
            "ndcg@1": 0.5,
            "recall@3": 1.0,
            "complete_set@3": 1.0,
            "ndcg@3": 0.8065735963827292,
            "recall@5": 1.0,
            "complete_set@5": 1.0,
            "ndcg@5": 0.8065735963827292,
            "mrr": 0.75,
            "set_compat": 0.0,
            "rejected_set_exposure@3": 1.0,
            "rejected_set_exposure@5": 1.0,
        }
    )


def test_risk_metrics_keep_helpful_retrieval_hsr_and_safe_helpful_decomposed() -> None:
    fixture = load_agent_skill_routing_fixture()
    query_ids = ["q_risk_archive_exact_day", "q_risk_dependency_policy"]
    skill_ids = [skill.skill_id for skill in fixture.skills]
    scores = _score_matrix(
        2,
        skill_ids,
        [
            {
                "archive_day_lookup": 1.0,
                "page_freshness_score": 0.9,
                "schema_doc_outline": 0.8,
                "archive_range_list": 0.7,
            },
            {
                "package_age_report": 1.0,
                "dependency_update_window": 0.9,
                "dependency_license_matrix": 0.8,
            },
        ],
    )

    metrics = evaluate_risk_scores(scores, query_ids, skill_ids, fixture)

    assert metrics == pytest.approx(
        {
            "helpful_recall@1": 0.5,
            "helpful_recall@3": 1.0,
            "helpful_ndcg@3": 0.75,
            "hsr@1": 0.5,
            "hsr@3": 0.5,
            "helpful_over_risky_win_rate": 0.5,
            "safe_helpful@3": 0.5,
        }
    )
    assert {"safe_helpful@3", "helpful_recall@3", "hsr@3"}.issubset(metrics)


def test_risk_metrics_expose_ties_empty_retrieval_and_sibling_heavy_failure() -> None:
    fixture = load_agent_skill_routing_fixture()
    skill_ids = [skill.skill_id for skill in fixture.skills]

    tied_scores = _score_matrix(
        1,
        skill_ids,
        [
            {
                "page_freshness_score": 0.9,
                "schema_doc_outline": 0.8,
                "archive_day_lookup": 0.5,
                "archive_range_list": 0.5,
            }
        ],
    )
    tied = evaluate_risk_scores(tied_scores, ["q_risk_archive_exact_day"], skill_ids, fixture)
    assert tied["helpful_recall@3"] == 1.0
    assert tied["hsr@3"] == 0.0
    assert tied["safe_helpful@3"] == 1.0
    assert tied["helpful_over_risky_win_rate"] == 0.0

    misleading_scores = _score_matrix(
        2,
        skill_ids,
        [
            {"schema_doc_outline": 0.9, "log_volume_rollup": 0.8, "package_age_report": 0.7},
            {
                "package_age_report": 0.9,
                "dependency_update_window": 0.8,
                "schema_doc_outline": 0.7,
                "dependency_license_matrix": 0.6,
            },
        ],
    )
    misleading = evaluate_risk_scores(
        misleading_scores,
        ["q_risk_archive_exact_day", "q_risk_dependency_policy"],
        skill_ids,
        fixture,
    )
    assert misleading["helpful_recall@3"] == 0.0
    assert misleading["hsr@3"] == 0.5
    assert misleading["safe_helpful@3"] == 0.0


@pytest.mark.parametrize(
    ("task_id", "primary_metric", "slice_name"),
    [
        ("agent_skill_compatible_set_retrieval", "set_compat", "compatible_set"),
        ("agent_skill_same_capability_risk", "safe_helpful@3", "same_capability_risk"),
    ],
)
def test_agent_skill_tasks_catalog_registry_routing_and_auditable_details(
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
    primary_metric: str,
    slice_name: str,
) -> None:
    def fail_on_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network access is forbidden for the agent skill routing fixture")

    monkeypatch.setattr("socket.socket", fail_on_network)
    catalog = load_catalog()
    spec = catalog.tasks[task_id]
    assert spec.primary_metric == primary_metric
    assert spec.required_modalities == ["text"]
    assert spec.dataset_version == SOURCE_REVISION
    assert spec.publish is False
    assert {"fixture-only", "no-publish"}.issubset(spec.tags)

    task = get_task(task_id, **spec.default_kwargs)
    provider = ShapeOnlyTextProvider()
    result = task.run(provider)

    assert result.passed, result.error
    assert provider.calls == [("retrieval_query", 2), ("retrieval_document", 12)]
    assert primary_metric in result.metrics
    if task_id == "agent_skill_same_capability_risk":
        assert {"safe_helpful@3", "helpful_recall@3", "hsr@3"}.issubset(result.metrics)
    assert result.details == {
        "n_skills": 12,
        "n_queries": 4,
        "n_qrels": 6,
        "n_compatible_sets": 2,
        "n_rejected_sets": 2,
        "n_risk_pairs": 2,
        "n_hard_negatives": 4,
        "n_evidence_records": 2,
        "n_evaluated_queries": 2,
        "evaluated_slice": slice_name,
        "slices": {"compatible_set": 2, "same_capability_risk": 2},
        "split": "fixture_only",
        "fixture_only": True,
        "source_kind": SOURCE_KIND,
        "source_revision": SOURCE_REVISION,
        "license_status": LICENSE_STATUS,
        "public_score_eligible": False,
        "query_latency_ms": 0.0,
        "document_latency_ms": 0.0,
        "ranking_tie_break": "skill_id_ascending",
    }


def test_unpublished_agent_skill_tasks_are_excluded_from_dataset_leaderboard_and_space(tmp_path) -> None:
    catalog = load_catalog()
    task_ids = ["agent_skill_compatible_set_retrieval", "agent_skill_same_capability_risk"]
    records = []
    leaderboard_lines = ["task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s"]
    for task_id in task_ids:
        spec = catalog.tasks[task_id]
        record = {
            "run": {"id": "agent-skill-routing-fixture"},
            "timestamps": {"duration_s": 0.0},
            "model": {"id": "fixture-model", "display_name": "Fixture Model", "provider": "local"},
            "provider_result": {"provider": "local", "model_name": "fixture-model"},
            "task": {
                "id": task_id,
                "display_name": spec.display_name,
                "primary_metric": spec.primary_metric,
                "publish": False,
            },
            "metrics": {spec.primary_metric: 1.0},
            "error": None,
        }
        records.append(record)
        leaderboard_lines.append(
            f"{task_id},Fixture,fixture-model,Fixture Model,local,{spec.primary_metric},1.0,fixture,0.0"
        )

    assert build_leaderboard(records, catalog) == []

    results_path = tmp_path / "results.jsonl"
    results_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    leaderboard_path = tmp_path / "leaderboard.csv"
    leaderboard_path.write_text("\n".join(leaderboard_lines) + "\n", encoding="utf-8")

    dataset_output = export_dataset_repo(
        output_dir=tmp_path / "dataset",
        results_path=results_path,
        leaderboard_path=leaderboard_path,
    )
    space_output = export_space_repo(
        output_dir=tmp_path / "space",
        bundled_leaderboard=leaderboard_path,
    )

    exported_texts = [
        (dataset_output / "tasks.jsonl").read_text(encoding="utf-8"),
        (dataset_output / "results" / "latest.jsonl").read_text(encoding="utf-8"),
        (dataset_output / "leaderboards" / "latest.csv").read_text(encoding="utf-8"),
        (space_output / "leaderboard.csv").read_text(encoding="utf-8"),
    ]
    for task_id in task_ids:
        assert all(task_id not in text for text in exported_texts)
