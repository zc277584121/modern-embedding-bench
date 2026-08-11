from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from scipy import sparse

from mm_embed.benchmark.leaderboard import build_leaderboard
from mm_embed.benchmark.registry import ModelSpec, TaskSpec, load_catalog, load_run_manifest
from mm_embed.benchmark.results import load_jsonl
from mm_embed.benchmark.runner import BenchmarkRunner
from mm_embed.data.learned_sparse_retrieval_fixture import (
    DIAGNOSTIC_SLICES,
    LONG_DOCUMENT_MIN_WORDS,
    fixture_with_publication,
    load_learned_sparse_retrieval_fixture,
    validate_learned_sparse_retrieval_fixture,
)
from mm_embed.hf_publish.export import export_dataset_repo, export_space_repo
from mm_embed.indexes.sparse_exact import ExactSparseIndex
from mm_embed.providers import get_provider
from mm_embed.providers.deterministic_sparse_provider import DeterministicSparseFixtureProvider
from mm_embed.providers.sparse_base import SparseEmbeddingRole, SparseEncodingRoute


RUN_PATH = Path("benchmark/runs/learned-sparse-retrieval-fixture.yaml")


def _forbid_dense_conversion(*args: object, **kwargs: object) -> None:
    raise AssertionError("Sparse fixture execution must not materialize dense vectors")


def _semantic_record(record: dict) -> dict:
    value = copy.deepcopy(record)
    value["timestamps"] = {"started_at": None, "finished_at": None, "duration_s": None}
    value["training_overlap"]["assessed_at"] = None
    return value


def test_fixture_has_fixed_five_slice_labels_and_long_local_evidence() -> None:
    fixture = load_learned_sparse_retrieval_fixture()

    assert fixture.fixture_only is True
    assert fixture.publish is False
    assert fixture.leaderboard_publish is False
    assert fixture.evidence_tier == "fixture"
    assert fixture.score_validity == "contract_only"
    assert {query.diagnostic_slice for query in fixture.queries}.issuperset(DIAGNOSTIC_SLICES)
    assert all(query.relevant_document_ids and query.hard_negative_ids for query in fixture.queries)

    query = next(query for query in fixture.queries if query.diagnostic_slice == "lexical_mismatch_document_expansion")
    document = next(document for document in fixture.documents if document.document_id == query.relevant_document_ids[0])
    assert set(query.text.lower().split()).isdisjoint(document.text.lower().split())
    assert query.activations[0][0] == document.activations[0][0]

    long_query = next(query for query in fixture.queries if query.diagnostic_slice == "long_document_local_evidence")
    long_gold = next(document for document in fixture.documents if document.document_id == long_query.relevant_document_ids[0])
    long_hard = next(document for document in fixture.documents if document.document_id == long_query.hard_negative_ids[0])
    assert len(long_gold.text.split()) >= LONG_DOCUMENT_MIN_WORDS
    assert len(long_hard.text.split()) >= LONG_DOCUMENT_MIN_WORDS
    gold_evidence = "The amber relay resets subsystem Kappa."
    hard_evidence = "The amber relay tests subsystem Lambda."
    assert 0.6 < long_gold.text.index(gold_evidence) / len(long_gold.text) < 0.9
    assert 0.6 < long_hard.text.index(hard_evidence) / len(long_hard.text) < 0.9
    assert long_gold.text.replace(gold_evidence, hard_evidence) == long_hard.text


def test_lexical_mismatch_rank_one_depends_on_document_expansion_route() -> None:
    fixture = load_learned_sparse_retrieval_fixture()
    query_spec = next(query for query in fixture.queries if query.diagnostic_slice == "lexical_mismatch_document_expansion")
    document_texts = [document.text for document in fixture.documents]
    document_ids = [document.document_id for document in fixture.documents]

    expanded_provider = DeterministicSparseFixtureProvider()
    expanded_documents = expanded_provider.encode_sparse_documents(document_texts, item_ids=document_ids)
    expanded_query = expanded_provider.encode_sparse_query(query_spec.text, item_id=query_spec.query_id)
    expanded_hits = ExactSparseIndex(expanded_documents).search(expanded_query, k=2).queries[0].hits
    assert expanded_documents.document_route is SparseEncodingRoute.DOCUMENT_EXPANSION
    assert expanded_hits[0].item_id == query_spec.relevant_document_ids[0]

    ablated_provider = DeterministicSparseFixtureProvider(document_route=SparseEncodingRoute.STATIC_LOOKUP)
    ablated_documents = ablated_provider.encode_sparse_documents(document_texts, item_ids=document_ids)
    ablated_query = ablated_provider.encode_sparse_query(query_spec.text, item_id=query_spec.query_id)
    ablated_hits = ExactSparseIndex(ablated_documents).search(
        ablated_query,
        k=len(fixture.documents),
    ).queries[0].hits
    assert ablated_documents.document_route is SparseEncodingRoute.STATIC_LOOKUP
    assert ablated_hits[0].item_id == query_spec.hard_negative_ids[0]
    assert next(hit.rank for hit in ablated_hits if hit.item_id == query_spec.relevant_document_ids[0]) != 1


@pytest.mark.parametrize(
    "fixture",
    [
        fixture_with_publication(load_learned_sparse_retrieval_fixture(), fixture_only=False),
        fixture_with_publication(load_learned_sparse_retrieval_fixture(), publish=True),
        fixture_with_publication(load_learned_sparse_retrieval_fixture(), leaderboard_publish=True),
    ],
)
def test_fixture_rejects_publication_flag_mutations(fixture) -> None:
    with pytest.raises(ValueError, match="publication contract"):
        validate_learned_sparse_retrieval_fixture(fixture)


def test_sparse_registry_contract_and_manifest_are_explicit_and_schema_valid() -> None:
    catalog = load_catalog()
    model = catalog.require_model("learned-sparse-contract-fixture")
    task = catalog.require_task("learned_sparse_retrieval")
    manifest = load_run_manifest(RUN_PATH)

    assert model.representation_kind == "sparse_csr"
    assert model.model_revision == "fixture-v1"
    assert model.vocabulary_id == "learned-sparse-fixture-vocabulary-v1"
    assert model.query_route == "static_lookup"
    assert model.document_route == "document_expansion"
    assert model.publish is False
    assert task.execution_kind == "sparse_exact"
    assert task.fixture_only is True
    assert task.score_validity == "contract_only"
    assert task.publish is False and task.leaderboard_publish is False
    assert manifest.publish is False and manifest.evidence_tier == "fixture"

    for schema_path, document_path in (
        (Path("schemas/model.schema.json"), Path("benchmark/models/core.yaml")),
        (Path("schemas/task.schema.json"), Path("benchmark/tasks/core.yaml")),
        (Path("schemas/run.schema.json"), RUN_PATH),
    ):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        document = yaml.safe_load(document_path.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(document)


def test_sparse_registry_rejects_identity_and_fixture_publication_drift(tmp_path: Path) -> None:
    sparse_model = {
        "id": "broken",
        "display_name": "Broken",
        "provider": "deterministic_sparse_fixture",
        "representation_kind": "sparse_csr",
        "model_revision": "fixture-v1",
        "representation_id": "rep",
        "dimensions": 32,
        "query_route": "static_lookup",
        "document_route": "tokenizer_idf",
    }
    with pytest.raises(ValueError, match="vocabulary_id"):
        ModelSpec.from_dict(sparse_model, tmp_path / "models.yaml")

    published_model = {
        **sparse_model,
        "vocabulary_id": "vocab",
        "publish": True,
    }
    with pytest.raises(ValueError, match="disable publication"):
        ModelSpec.from_dict(published_model, tmp_path / "models.yaml")

    with pytest.raises(ValueError, match="disable public"):
        TaskSpec.from_dict(
            {
                "id": "broken",
                "display_name": "Broken",
                "task": "learned_sparse_retrieval",
                "description": "Broken fixture publication contract.",
                "execution_kind": "sparse_exact",
                "fixture_only": True,
                "publish": True,
                "leaderboard_publish": False,
            },
            tmp_path / "tasks.yaml",
        )


def test_runner_executes_sparse_csr_path_without_dense_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparse.csr_matrix, "toarray", _forbid_dense_conversion)
    monkeypatch.setattr(sparse.csr_matrix, "todense", _forbid_dense_conversion)
    output = tmp_path / "sparse.jsonl"
    runner, manifest = BenchmarkRunner.from_manifest_path(RUN_PATH, output=output, overwrite=True)

    records = runner.run_manifest(manifest)

    assert records == load_jsonl(output)
    assert len(records) == 1
    record = records[0]
    assert record["error"] is None
    assert record["run"]["publish"] is False
    assert record["run"]["evidence_tier"] == "fixture"
    assert record["model"]["representation_kind"] == "sparse_csr"
    assert record["task"]["execution_kind"] == "sparse_exact"
    assert record["task"]["fixture_only"] is True
    assert record["task"]["score_validity"] == "contract_only"
    assert record["metrics"]["recall@1"] == 1.0
    assert all(record["metrics"][f"slice_recall@1/{name}"] == 1.0 for name in DIAGNOSTIC_SLICES)
    assert record["execution"]["representation_kind"] == "sparse_csr"
    assert record["execution"]["query_route"] == "static_lookup"
    assert record["execution"]["document_route"] == "document_expansion"
    assert record["execution"]["exact_backend"] == "scipy_csr_exact"
    assert record["execution"]["query_nnz"]["total"] == 6
    assert record["execution"]["document_nnz"]["total"] == 22
    assert record["execution"]["index_bytes_estimate"] > 0
    assert record["details"]["fixture"]["fixture_sha256"] == load_learned_sparse_retrieval_fixture().fixture_sha256
    serialized = output.read_text(encoding="utf-8")
    assert '"indptr"' not in serialized and '"indices"' not in serialized

    schema = json.loads(Path("schemas/result.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(record)


def test_exact_sparse_fixture_ties_use_item_id_and_route_mutation_is_rejected() -> None:
    fixture = load_learned_sparse_retrieval_fixture()
    provider = get_provider("deterministic_sparse_fixture")
    documents = provider.encode_sparse_documents(
        [document.text for document in fixture.documents],
        item_ids=[document.document_id for document in fixture.documents],
    )
    tie_query_spec = next(query for query in fixture.queries if query.query_id == "query-tie")
    query = provider.encode_sparse_query(tie_query_spec.text, item_id=tie_query_spec.query_id)
    ranking = ExactSparseIndex(documents).search(query, k=2)

    assert [hit.item_id for hit in ranking.queries[0].hits] == ["doc-tie-a", "doc-tie-b"]
    mutated = copy.copy(query)
    object.__setattr__(mutated, "role", SparseEmbeddingRole.DOCUMENT)
    with pytest.raises(ValueError, match="query result"):
        ExactSparseIndex(documents).search(mutated)
    route_mutation = replace(query, document_route=SparseEncodingRoute.NEURAL)
    with pytest.raises(ValueError, match="document route"):
        ExactSparseIndex(documents).search(route_mutation)


def test_fixture_run_is_semantically_deterministic_and_isolated_from_publication(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_runner, manifest = BenchmarkRunner.from_manifest_path(RUN_PATH, output=first_path, overwrite=True)
    second_runner, second_manifest = BenchmarkRunner.from_manifest_path(RUN_PATH, output=second_path, overwrite=True)
    first = first_runner.run_manifest(manifest)[0]
    second = second_runner.run_manifest(second_manifest)[0]

    assert _semantic_record(first) == _semantic_record(second)
    catalog = load_catalog()
    assert build_leaderboard([first], catalog) == []

    dataset = export_dataset_repo(output_dir=tmp_path / "dataset", results_path=first_path)
    space = export_space_repo(output_dir=tmp_path / "space")
    for output in (dataset, space):
        assert "learned-sparse-contract-fixture" not in (output / "models.jsonl").read_text(encoding="utf-8")
        assert "learned_sparse_retrieval" not in (output / "tasks.jsonl").read_text(encoding="utf-8")
    assert "learned_sparse_retrieval" not in (dataset / "results" / "latest.jsonl").read_text(encoding="utf-8")


def test_checked_in_fixture_manifest_matches_runtime_contract() -> None:
    fixture = load_learned_sparse_retrieval_fixture()
    manifest = json.loads(
        Path("benchmark/fixtures/learned-sparse-retrieval-fixture-v0/fixture.json").read_text(encoding="utf-8")
    )

    assert manifest["dataset_version"] == fixture.dataset_version
    assert manifest["fixture_only"] == fixture.fixture_only
    assert manifest["publish"] == fixture.publish
    assert manifest["leaderboard_publish"] == fixture.leaderboard_publish
    assert manifest["label_sha256"] == fixture.label_sha256
    assert manifest["fixture_sha256"] == fixture.fixture_sha256
    assert manifest["diagnostic_slices"] == list(DIAGNOSTIC_SLICES)
