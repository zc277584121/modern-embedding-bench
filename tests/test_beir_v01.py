import copy
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from mm_embed.benchmark.beir_v01 import BeirBenchmarkError, TrackData, canonical_text, evaluate_rankings


def test_canonical_text_protocol() -> None:
    assert canonical_text(" Title ", " Body ") == "Title\nBody"
    assert canonical_text("", " Body ") == "Body"
    with pytest.raises(BeirBenchmarkError):
        canonical_text(None, "text")


def test_metric_definitions() -> None:
    data = TrackData(
        "scifact",
        ({"id": "d1", "content": "one"}, {"id": "d2", "content": "two"}),
        ({"id": "q1", "text": "one"},),
        {"q1": {"d1": 1}},
    )
    metrics = evaluate_rankings(data, {"q1": [("d2", 2.0), ("d1", 1.0)]})
    assert metrics["mrr@10"] == 0.5
    assert metrics["map@100"] == 0.5
    assert metrics["recall@10"] == 1.0
    assert metrics["ndcg@10"] == pytest.approx(1 / np.log2(3))


RESULTS_ROOT = Path("results/beir-three-track-v0.1")
RESULT_SCHEMA = json.loads(Path("schemas/beir-benchmark-result-v01.schema.json").read_text())
RESULT_VALIDATOR = Draft202012Validator(RESULT_SCHEMA)


def _result(method: str, track: str) -> dict:
    return json.loads((RESULTS_ROOT / f"{method}-{track}.json").read_text())


def test_result_schema_and_all_results_are_valid() -> None:
    Draft202012Validator.check_schema(RESULT_SCHEMA)
    for method in ("bm25", "dense"):
        for track in ("scifact", "nfcorpus", "fiqa"):
            RESULT_VALIDATOR.validate(_result(method, track))


def _swap_execution_method(result: dict) -> None:
    other_method = "dense" if result["method"] == "bm25" else "bm25"
    result["execution"] = copy.deepcopy(_result(other_method, result["track"])["execution"])


@pytest.mark.parametrize(
    "mutation",
    [
        _swap_execution_method,
        lambda result: result["execution"].__setitem__("junk", True),
        lambda result: result.__setitem__("queries_evaluated", result["queries_evaluated"] + 1),
        lambda result: result.__setitem__("documents_searched", result["documents_searched"] + 1),
        lambda result: result["rankings"].__setitem__("rows", result["rankings"]["rows"] + 1),
        lambda result: result["rankings"].__setitem__("path", "../bm25-scifact.rankings.jsonl"),
        lambda result: result["rankings"].__setitem__("path", "dense-fiqa.rankings.jsonl"),
        lambda result: result["execution"].pop(next(iter(result["execution"]))),
        lambda result: result["execution"].__setitem__("query_block_size", 31),
    ],
    ids=[
        "execution-method-swap",
        "additional-property",
        "queries",
        "documents",
        "rows",
        "path-traversal",
        "cross-run-path",
        "missing-execution-field",
        "wrong-fixed-constant",
    ],
)
def test_result_schema_rejects_mutations(mutation) -> None:
    result = copy.deepcopy(_result("dense", "scifact"))
    mutation(result)
    assert not RESULT_VALIDATOR.is_valid(result)


def test_bm25_fixed_constants_and_missing_field_are_rejected() -> None:
    result = _result("bm25", "nfcorpus")
    for field, value in (("tokenizer", "other"), ("k1", 1.3), ("b", 0.7)):
        mutated = copy.deepcopy(result)
        mutated["execution"][field] = value
        assert not RESULT_VALIDATOR.is_valid(mutated)
    mutated = copy.deepcopy(result)
    mutated["execution"].pop("index_bytes")
    assert not RESULT_VALIDATOR.is_valid(mutated)


def test_result_metrics_match_summary() -> None:
    summary = json.loads(Path("benchmark/artifacts/beir-three-track-v0.1/summary.json").read_text())
    for method in ("bm25", "dense"):
        for track in ("scifact", "nfcorpus", "fiqa"):
            assert _result(method, track)["metrics"] == summary["tracks"][track][method]


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape", "bad")
    from mm_embed.benchmark.beir_v01 import _safe_members

    with zipfile.ZipFile(archive) as handle, pytest.raises(BeirBenchmarkError, match="unsafe archive"):
        _safe_members(handle, "scifact")


def test_blockwise_exact_scores_match_brute_matrix() -> None:
    rng = np.random.default_rng(17)
    queries = rng.normal(size=(7, 11)).astype(np.float32)
    documents = rng.normal(size=(23, 11)).astype(np.float32)
    brute = queries @ documents.T
    blockwise = np.empty_like(brute)
    for query_start in range(0, len(queries), 3):
        for document_start in range(0, len(documents), 5):
            blockwise[query_start : query_start + 3, document_start : document_start + 5] = (
                queries[query_start : query_start + 3] @ documents[document_start : document_start + 5].T
            )
    np.testing.assert_allclose(blockwise, brute, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.argsort(-blockwise, axis=1)[:, :10], np.argsort(-brute, axis=1)[:, :10])
