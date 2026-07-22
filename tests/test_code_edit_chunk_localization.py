from __future__ import annotations

import json
import socket
from dataclasses import replace

import numpy as np
import pytest

from mm_embed.benchmark.leaderboard import build_leaderboard
from mm_embed.benchmark.registry import load_catalog
from mm_embed.data.code_edit_chunk_localization import (
    DATASET_VERSION,
    fixture_counts,
    fixture_slice_counts,
    load_code_edit_chunk_localization_fixture,
    serialize_fixture,
    validate_code_edit_chunk_localization_fixture,
)
from mm_embed.hf_publish.export import export_dataset_repo, export_space_repo
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType
from mm_embed.tasks.code_edit_chunk_localization import (
    deterministic_rankings,
    evaluate_code_edit_scores,
)
from mm_embed.tasks.registry import get_task


class LocalFullCorpusProvider(EmbeddingProvider):
    name = "code-edit-fixture-local"
    model = "deterministic-shape-test-double"
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
        width = dimensions or 5
        embeddings = np.zeros((len(inputs), width), dtype=float)
        for row in range(len(inputs)):
            embeddings[row, row % width] = 1.0
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=width,
            model_name=self.model,
            provider=self.name,
            latency_ms=0.0,
        )


def _score_matrix(
    query_ids: list[str],
    chunk_ids: list[str],
    rankings: dict[str, list[str]],
) -> np.ndarray:
    matrix = np.full((len(query_ids), len(chunk_ids)), -100.0, dtype=float)
    chunk_index = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    for row, query_id in enumerate(query_ids):
        ordered = rankings[query_id]
        for position, chunk_id in enumerate(ordered):
            matrix[row, chunk_index[chunk_id]] = float(len(ordered) - position)
    return matrix


def test_fixture_is_full_repository_deterministic_and_maps_all_edit_types() -> None:
    first = load_code_edit_chunk_localization_fixture()
    second = load_code_edit_chunk_localization_fixture()

    assert fixture_counts(first) == {
        "n_files": 8,
        "n_queries": 3,
        "n_chunks": 24,
        "n_patch_targets": 4,
        "n_qrels": 4,
        "n_hard_negatives": 9,
        "n_provenance_records": 3,
    }
    assert first.fixture_only is True
    assert first.leaderboard_publish is False
    assert first.network == "forbidden"
    assert first.provider_api_calls == 0
    assert first.model_downloads == 0
    assert first.repository.public_score_eligible is False
    assert first.repository.public_redistribution is False

    assert serialize_fixture(first).encode("utf-8") == serialize_fixture(second).encode("utf-8")
    assert first.serialization_sha256 == second.serialization_sha256
    assert len(first.serialization_sha256) == 64
    serialized = serialize_fixture(first).lower()
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert "github" not in serialized
    assert "huggingface" not in serialized

    chunks_by_path = {path: [] for path in (file.path for file in first.files)}
    for chunk in first.chunks:
        chunks_by_path[chunk.path].append(chunk)
    assert all(chunks_by_path.values())
    assert {chunk.candidate_family for chunk in first.chunks}.issuperset(
        {"ast_function", "ast_class", "module_preamble", "line_fallback"}
    )
    assert {file.path_family for file in first.files} == {
        "implementation",
        "test",
        "configuration",
        "documentation",
    }

    targets_by_type = {}
    for target in first.patch_targets:
        targets_by_type.setdefault(target.patch_change_type, []).append(target)
    assert set(targets_by_type) == {"replacement", "deletion", "insertion_only"}
    assert len(targets_by_type["replacement"]) == 2
    assert targets_by_type["deletion"][0].preimage_line_start is not None
    assert targets_by_type["insertion_only"][0].insertion_anchor_after_line is not None

    assert all(qrel.relevance == 2 and qrel.mapping_status == "exact" for qrel in first.qrels)
    assert {qrel.label_family for qrel in first.qrels} == {
        "modified_preimage_line",
        "deleted_preimage_line",
        "insert_anchor_containing_chunk",
    }
    assert all(qrel.target_unit_ids for qrel in first.qrels)
    positive_keys = {(qrel.query_id, qrel.chunk_id) for qrel in first.qrels}
    assert all(
        (negative.query_id, negative.chunk_id) not in positive_keys
        and negative.source_chunk_id
        and negative.reason
        and negative.false_negative_review == "pass"
        and negative.review_metadata
        for negative in first.hard_negatives
    )

    slices = fixture_slice_counts(first)
    assert slices["edit_type"] == {"deletion": 1, "insertion_only": 1, "replacement": 1}
    assert slices["qrel_mapping_status"] == {"exact": 4}
    assert slices["hard_negative_family"] == {
        "documentation_code_collision": 3,
        "error_message_collision": 1,
        "same_file_neighbor": 2,
        "same_subsystem_path": 1,
        "same_symbol_family": 2,
    }


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("file_hash", "File hash mismatch"),
        ("qrel_mapping", "Qrel mapping does not reproduce"),
        ("public_provenance", "Invalid provenance"),
        ("hard_overlap", "Hard negative overlaps"),
        ("serialization", "serialization hash is unstable"),
        ("repository_public", "cannot be public-score eligible"),
    ],
)
def test_fixture_validation_rejects_contract_corruption(corruption: str, message: str) -> None:
    fixture = load_code_edit_chunk_localization_fixture()

    if corruption == "file_hash":
        corrupted_file = replace(fixture.files[0], text_sha256="0" * 64)
        corrupted = replace(fixture, files=(corrupted_file, *fixture.files[1:]))
    elif corruption == "qrel_mapping":
        corrupted_qrel = replace(fixture.qrels[0], chunk_id=fixture.qrels[1].chunk_id)
        corrupted = replace(fixture, qrels=(corrupted_qrel, *fixture.qrels[1:]))
    elif corruption == "public_provenance":
        corrupted_record = replace(fixture.provenance[0], public_score_eligible=True)
        corrupted = replace(fixture, provenance=(corrupted_record, *fixture.provenance[1:]))
    elif corruption == "hard_overlap":
        positive = next(qrel for qrel in fixture.qrels if qrel.query_id == fixture.hard_negatives[0].query_id)
        corrupted_negative = replace(fixture.hard_negatives[0], chunk_id=positive.chunk_id)
        corrupted = replace(fixture, hard_negatives=(corrupted_negative, *fixture.hard_negatives[1:]))
    elif corruption == "serialization":
        corrupted = replace(fixture, serialization_sha256="0" * 64)
    else:
        corrupted_repository = replace(fixture.repository, public_score_eligible=True)
        corrupted = replace(fixture, repository=corrupted_repository)

    with pytest.raises(ValueError, match=message):
        validate_code_edit_chunk_localization_fixture(corrupted)


def test_exact_full_corpus_metrics_and_tie_breaking() -> None:
    fixture = load_code_edit_chunk_localization_fixture()
    query_ids = [query.query_id for query in fixture.queries]
    chunk_ids = [chunk.chunk_id for chunk in reversed(fixture.chunks)]
    positives = {
        query_id: [qrel.chunk_id for qrel in fixture.qrels if qrel.query_id == query_id]
        for query_id in query_ids
    }
    negatives = {
        query_id: [negative.chunk_id for negative in fixture.hard_negatives if negative.query_id == query_id]
        for query_id in query_ids
    }
    q_retry, q_delete, q_insert = query_ids
    retry_impl = next(
        chunk_id
        for chunk_id in positives[q_retry]
        if next(chunk for chunk in fixture.chunks if chunk.chunk_id == chunk_id).path.startswith("src/")
    )
    retry_test = next(chunk_id for chunk_id in positives[q_retry] if chunk_id != retry_impl)
    rankings = {
        q_retry: [retry_impl, negatives[q_retry][1], negatives[q_retry][2], retry_test],
        q_delete: [negatives[q_delete][0], positives[q_delete][0]],
        q_insert: [negatives[q_insert][0], negatives[q_insert][2], positives[q_insert][0]],
    }
    scores = _score_matrix(query_ids, chunk_ids, rankings)

    metrics = evaluate_code_edit_scores(scores, query_ids, chunk_ids, fixture)

    two_positive_rank_1_4_ndcg = (1.0 + 1.0 / np.log2(5)) / (1.0 + 1.0 / np.log2(3))
    expected_ndcg = (two_positive_rank_1_4_ndcg + 1.0 / np.log2(3) + 0.5) / 3.0
    assert metrics == pytest.approx(
        {
            "edit_chunk_ndcg@10": expected_ndcg,
            "edit_chunk_recall@1": 1.0 / 6.0,
            "edit_chunk_recall@5": 1.0,
            "edit_chunk_recall@10": 1.0,
            "edit_chunk_recall@100": 1.0,
            "edit_chunk_mrr": (1.0 + 0.5 + 1.0 / 3.0) / 3.0,
            "edit_target_recall@100_lines": 1.0,
            "edit_target_recall@300_lines": 1.0,
            "edit_target_recall@500_lines": 1.0,
            "first_edit_hit_rank": 2.0,
            "file_recall@1": 0.5,
            "file_recall@5": 1.0,
            "file_recall@10": 1.0,
            "candidate_coverage": 1.0,
            "hard_mrr": (1.0 + 0.5 + 1.0 / 3.0) / 3.0,
            "hard_ndcg@10": expected_ndcg,
        }
    )

    tied_scores = np.zeros((1, len(chunk_ids)), dtype=float)
    tied = deterministic_rankings(tied_scores, chunk_ids, fixture)[0]
    expected_tie_order = [
        chunk.chunk_id
        for chunk in sorted(
            fixture.chunks,
            key=lambda chunk: (
                chunk.repository_id,
                chunk.path,
                chunk.line_start,
                chunk.chunk_id,
            ),
        )
    ]
    assert tied == expected_tie_order

    with pytest.raises(ValueError, match="complete full-repository fixture corpus"):
        evaluate_code_edit_scores(scores[:, :-1], query_ids, chunk_ids[:-1], fixture)


def test_task_uses_flat_provider_full_corpus_and_emits_diagnostic_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("Network access is forbidden for the invented fixture test")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    provider = LocalFullCorpusProvider()
    result = get_task("code_edit_chunk_localization").run(provider)

    assert result.passed, result.error
    assert provider.calls == [("retrieval_query", 3), ("retrieval_document", 24)]
    assert "edit_chunk_ndcg@10" in result.metrics
    assert result.details["candidate_pool_scope"] == "complete_invented_repository"
    assert result.details["full_corpus_candidate_count"] == 24
    assert result.details["ranked_candidate_count_per_query"] == 24
    assert result.details["changed_path_prefilter"] is False
    assert result.details["gold_path_prefilter"] is False
    assert result.details["fixture_only"] is True
    assert result.details["leaderboard_publish"] is False
    assert result.details["public_score_eligible"] is False
    assert result.details["network"] == "forbidden"
    assert result.details["fixture_provider_api_calls"] == 0
    assert result.details["fixture_model_downloads"] == 0
    assert set(result.details["slice_metrics"]) == {
        "edit_type",
        "file_path",
        "path_family",
        "candidate_family",
        "mapping_status",
    }
    assert result.details["hard_negative_slices"]
    expected_query_ids = {
        query.query_id for query in load_code_edit_chunk_localization_fixture().queries
    }
    assert set(result.details["per_query"]) == expected_query_ids


def test_catalog_registry_and_public_exports_exclude_code_edit_fixture(tmp_path) -> None:
    catalog = load_catalog()
    spec = catalog.tasks["code_edit_chunk_localization"]

    assert spec.task == "code_edit_chunk_localization"
    assert spec.primary_metric == "edit_chunk_ndcg@10"
    assert spec.dataset_version == DATASET_VERSION
    assert spec.required_modalities == ["text"]
    assert spec.publish is False
    assert {"fixture-only", "no-publish"}.issubset(spec.tags)
    assert get_task("code_edit_chunk_localization", **spec.default_kwargs).name == "code_edit_chunk_localization"

    record = {
        "run": {"id": "code-edit-fixture", "publish": False, "evidence_tier": "fixture"},
        "timestamps": {"duration_s": 0.0},
        "model": {"id": "fixture-model", "display_name": "Fixture Model", "provider": "local"},
        "provider_result": {"provider": "local", "model_name": "fixture-model"},
        "task": {
            "id": spec.id,
            "display_name": spec.display_name,
            "primary_metric": spec.primary_metric,
            "publish": False,
        },
        "metrics": {"edit_chunk_ndcg@10": 1.0},
        "error": None,
    }
    assert build_leaderboard([record], catalog) == []

    results_path = tmp_path / "results.jsonl"
    results_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    leaderboard_path = tmp_path / "leaderboard.csv"
    leaderboard_path.write_text(
        "task_id,task,model_id,model,provider,primary_metric,score,run_id,duration_s\n"
        "code_edit_chunk_localization,Fixture,fixture-model,Fixture Model,local,"
        "edit_chunk_ndcg@10,1.0,code-edit-fixture,0.0\n",
        encoding="utf-8",
    )

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
    assert all("code_edit_chunk_localization" not in text for text in exported_texts)
