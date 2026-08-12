import copy
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from mm_embed.benchmark.registry import load_catalog, load_run_manifest
from mm_embed.benchmark.runner import BenchmarkRunner
from mm_embed.data.multi_vector_fixture import load_multi_vector_fixture
from mm_embed.indexes.multi_vector_exact import ExactMaxSimIndex
from mm_embed.providers.deterministic_multi_vector_provider import DeterministicMultiVectorFixtureProvider
from mm_embed.providers.multi_vector_base import MultiVectorBatch, MultiVectorRepresentation, MultiVectorResult, MultiVectorRole, MultiVectorRoute


def test_contract_rejects_mask_empty_nonfinite_padding_and_duplicate_ids():
    rep = MultiVectorRepresentation("test", 2)
    good = np.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    with pytest.raises(ValueError, match="at least one"):
        MultiVectorBatch(good, np.zeros((1, 2), bool), ("i",), ("p",), ("d",), rep)
    bad = good.copy(); bad[0,0,0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        MultiVectorBatch(bad, np.array([[True, False]]), ("i",), ("p",), ("d",), rep)
    bad = good.copy(); bad[0,1,0] = 1
    with pytest.raises(ValueError, match="padding"):
        MultiVectorBatch(bad, np.array([[True, False]]), ("i",), ("p",), ("d",), rep)
    with pytest.raises(ValueError, match="unique"):
        MultiVectorBatch(np.ones((2,1,2),np.float32), np.ones((2,1),bool), ("i","i"), ("p","q"), ("d","e"), rep)
    with pytest.raises(ValueError, match="shapes"):
        MultiVectorBatch(good, np.ones((1, 3), bool), ("i",), ("p",), ("d",), rep)
    with pytest.raises(TypeError, match="boolean"):
        MultiVectorBatch(good, np.ones((1, 2), np.int8), ("i",), ("p",), ("d",), rep)
    with pytest.raises(ValueError, match="unique"):
        MultiVectorBatch(np.ones((2,1,2), np.float32), np.ones((2,1), bool), ("i", "j"), ("p", "p"), ("d", "e"), rep)


def test_contract_rejects_role_route_swap_and_identity_drift():
    provider = DeterministicMultiVectorFixtureProvider(); docs, queries, *_ = load_multi_vector_fixture()
    query = provider.encode_multi_vector_query(queries[0].text, item_id=queries[0].query_id)
    with pytest.raises(ValueError, match="role and route"):
        replace(query, route=MultiVectorRoute.DOCUMENT)
    encoded = provider.encode_multi_vector_passages([d.text for d in docs], passage_ids=[d.passage_id for d in docs], document_ids=[d.document_id for d in docs])
    index = ExactMaxSimIndex(encoded)
    drift = replace(query, embeddings=replace(query.embeddings, representation=MultiVectorRepresentation("drift", 4)))
    with pytest.raises(ValueError, match="identity"):
        index.search(drift, k=2)
    with pytest.raises(ValueError, match="aggregation"):
        ExactMaxSimIndex(encoded, aggregation="mean")


def test_exact_maxsim_gold_rank_one_while_mean_pooling_is_not():
    provider = DeterministicMultiVectorFixtureProvider(); docs, queries, *_ = load_multi_vector_fixture()
    encoded = provider.encode_multi_vector_passages([d.text for d in docs], passage_ids=[d.passage_id for d in docs], document_ids=[d.document_id for d in docs])
    query = provider.encode_multi_vector_query(queries[0].text, item_id=queries[0].query_id)
    assert ExactMaxSimIndex(encoded).search(query, k=6)[0].document_id == "doc-evidence"
    qmean = query.embeddings.values[0, query.embeddings.mask[0]].mean(0)
    scores = np.asarray([encoded.embeddings.values[i, encoded.embeddings.mask[i]].mean(0) @ qmean for i in range(len(docs))])
    assert encoded.embeddings.document_ids[int(np.argmax(scores))] != "doc-evidence"


def test_multi_passage_grouping_and_duplicate_document_distinction():
    provider = DeterministicMultiVectorFixtureProvider(); docs, queries, *_ = load_multi_vector_fixture()
    encoded = provider.encode_multi_vector_passages([d.text for d in docs], passage_ids=[d.passage_id for d in docs], document_ids=[d.document_id for d in docs])
    assert encoded.embeddings.document_ids.count("doc-evidence") == 2
    hit = ExactMaxSimIndex(encoded).search(provider.encode_multi_vector_query("evidence", item_id="q-evidence"), k=1)[0]
    assert hit.document_id == "doc-evidence" and hit.passage_id in {"p-evidence", "p-evidence-detail"}
    rep = encoded.embeddings.representation
    rows = np.ones((2, 1, rep.dimensions), dtype=np.float32)
    mask = np.ones((2, 1), dtype=bool)
    with pytest.raises(ValueError, match="item ids.*unique"):
        MultiVectorBatch(rows, mask, ("duplicate-document", "duplicate-document"), ("p-one", "p-two"), ("doc-x", "doc-x"), rep)
    grouped = MultiVectorBatch(rows, mask, ("p-one", "p-two"), ("p-one", "p-two"), ("doc-x", "doc-x"), rep)
    assert grouped.document_ids == ("doc-x", "doc-x")


def test_provider_rejects_cross_document_passage_leakage():
    provider = DeterministicMultiVectorFixtureProvider(); docs, *_ = load_multi_vector_fixture()
    with pytest.raises(ValueError, match="mismatched multi-vector passage"):
        provider.encode_multi_vector_passages([docs[0].text], passage_ids=[docs[0].passage_id], document_ids=["doc-long"])


def test_fixture_has_independent_non_overlapping_qrels_and_hard_negatives():
    _, queries, qrels, hard_negatives, _ = load_multi_vector_fixture()
    assert {row.query_id for row in qrels} == {query.query_id for query in queries}
    assert {(row.query_id, row.document_id) for row in qrels}.isdisjoint(
        {(row.query_id, row.document_id) for row in hard_negatives}
    )


@pytest.mark.parametrize("query_id", ["q-evidence", "q-long", "q-entity", "q-compose", "q-code"])
def test_each_query_ranks_its_qrel_above_hard_negatives(query_id):
    provider = DeterministicMultiVectorFixtureProvider()
    docs, queries, qrels, hard_negatives, _ = load_multi_vector_fixture()
    encoded = provider.encode_multi_vector_passages(
        [document.text for document in docs],
        passage_ids=[document.passage_id for document in docs],
        document_ids=[document.document_id for document in docs],
    )
    query = next(row for row in queries if row.query_id == query_id)
    hits = ExactMaxSimIndex(encoded).search(
        provider.encode_multi_vector_query(query.text, item_id=query.query_id),
        k=len({document.document_id for document in docs}),
    )
    expected = next(row.document_id for row in qrels if row.query_id == query_id)
    scores = {hit.document_id: hit.score for hit in hits}
    negatives = [row.document_id for row in hard_negatives if row.query_id == query_id]
    assert hits[0].document_id == expected
    assert all(scores[negative] < scores[expected] for negative in negatives)


def test_batch_single_consistency_and_padding_mutation_rejection():
    provider = DeterministicMultiVectorFixtureProvider(); docs, queries, *_ = load_multi_vector_fixture()
    encoded = provider.encode_multi_vector_passages([d.text for d in docs], passage_ids=[d.passage_id for d in docs], document_ids=[d.document_id for d in docs])
    index = ExactMaxSimIndex(encoded)
    single = index.search(provider.encode_multi_vector_query("evidence", item_id="q-evidence"), k=3)
    batch = provider.encode_multi_vector_queries(["evidence", "local"], item_ids=["q-evidence", "q-long"])
    assert [(h.document_id, h.score) for h in single] == [(h.document_id, h.score) for h in index.search_batch(batch, k=3)[0]]
    padded = MultiVectorBatch(
        np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=np.float32),
        np.array([[True, False], [True, True]], dtype=bool),
        ("q-one", "q-two"),
        ("q-one", "q-two"),
        ("q-one", "q-two"),
        encoded.embeddings.representation,
    )
    values = padded.values.copy(); values[~padded.mask] = 12345.0
    with pytest.raises(ValueError, match="padding"):
        MultiVectorBatch(values, padded.mask, padded.item_ids, padded.passage_ids, padded.document_ids, padded.representation)


def test_max_passage_tie_breaks_by_smallest_passage_id():
    rep = MultiVectorRepresentation("tie", 2)
    document_batch = MultiVectorBatch(
        np.ones((2, 1, 2), dtype=np.float32),
        np.ones((2, 1), dtype=bool),
        ("item-z", "item-a"),
        ("passage-z", "passage-a"),
        ("doc-tie", "doc-tie"),
        rep,
    )
    documents = MultiVectorResult(
        document_batch,
        MultiVectorRole.DOCUMENT,
        MultiVectorRoute.DOCUMENT,
        "test",
        "test",
        "v1",
        0.0,
    )
    query_batch = MultiVectorBatch(
        np.ones((1, 1, 2), dtype=np.float32),
        np.ones((1, 1), dtype=bool),
        ("query",),
        ("query",),
        ("query",),
        rep,
    )
    query = MultiVectorResult(
        query_batch,
        MultiVectorRole.QUERY,
        MultiVectorRoute.QUERY,
        "test",
        "test",
        "v1",
        0.0,
    )
    assert ExactMaxSimIndex(documents).search(query, k=1)[0].passage_id == "passage-a"


def test_manifest_is_deterministic_schema_ready_and_no_publish(tmp_path):
    manifest = load_run_manifest("benchmark/runs/multi-vector-retrieval-fixture.yaml")
    catalog = load_catalog(); assert not catalog.require_model(manifest.model_ids[0]).publish
    outputs=[]
    for name in ("one.jsonl", "two.jsonl"):
        path=tmp_path/name; record=BenchmarkRunner(catalog,path,overwrite=True).run_manifest(manifest)[0]; outputs.append(record)
        assert record["run"]["publish"] is False and record["task"]["leaderboard_publish"] is False
        assert record["metrics"]["recall@1"] == 1.0
        assert "values" not in json.dumps(record)
        model = record["model"]
        execution = record["execution"]
        assert execution["provider"] == model["provider"] == record["provider_result"]["provider"]
        assert execution["model_name"] == model["id"] == record["provider_result"]["model_name"]
        assert execution["model_revision"] == model["model_revision"]
        assert execution["representation_id"] == model["representation_id"]
        assert execution["dimensions"] == model["dimensions"]
        assert execution["query_route"] == model["query_route"]
        assert execution["document_route"] == model["document_route"]
        assert execution["exact"] is True
    def stable(record):
        record=json.loads(json.dumps(record)); record.pop("timestamps"); record["run"].pop("git_sha",None); return record
    assert stable(outputs[0]) == stable(outputs[1])


@pytest.mark.parametrize(
    "field",
    [
        "provider", "model_name", "model_revision", "representation_id", "representation_identity",
        "query_route", "document_route", "exact_backend", "aggregation",
    ],
)
def test_multi_vector_execution_identity_fields_are_schema_required(tmp_path, field):
    manifest = load_run_manifest("benchmark/runs/multi-vector-retrieval-fixture.yaml")
    record = BenchmarkRunner(load_catalog(), tmp_path / "result.jsonl", overwrite=True).run_manifest(manifest)[0]
    schema = json.loads(Path("schemas/result.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    validator.validate(record)
    broken = copy.deepcopy(record)
    del broken["execution"][field]
    assert list(validator.iter_errors(broken)), field


def test_multi_vector_empty_execution_is_schema_invalid(tmp_path):
    manifest = load_run_manifest("benchmark/runs/multi-vector-retrieval-fixture.yaml")
    record = BenchmarkRunner(load_catalog(), tmp_path / "result.jsonl", overwrite=True).run_manifest(manifest)[0]
    schema = json.loads(Path("schemas/result.schema.json").read_text(encoding="utf-8"))
    broken = copy.deepcopy(record)
    broken["execution"] = {}
    assert list(Draft202012Validator(schema).iter_errors(broken))
