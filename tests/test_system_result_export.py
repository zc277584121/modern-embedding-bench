from __future__ import annotations

import copy
import csv
import hashlib
import json
import socket

import pytest

from mm_embed.benchmark.leaderboard import build_leaderboard
from mm_embed.benchmark.results import is_embedding_result_record, load_jsonl
from mm_embed.hf_publish.export import export_dataset_repo, export_space_repo
from mm_embed.system_evaluation import (
    SYSTEM_RESULT_SCHEMA_PATH,
    TokenOverlapRetrieval,
    evaluate_fixture_brackets,
    evaluate_system,
    export_retrieval_answer_utility_fixture,
    validate_system_result,
)
from mm_embed.system_evaluation.export import SYSTEM_EXPORT_MANIFEST_FILENAME, SYSTEM_RESULTS_FILENAME
from mm_embed.system_evaluation.retrieval_answer_utility import ContractValidationError


def _accepted_record() -> dict:
    return copy.deepcopy(evaluate_system(TokenOverlapRetrieval()))


def _publishable_looking_system_record() -> dict:
    record = _accepted_record()
    record["run"]["publish"] = True
    record["task"]["publish"] = True
    record["task"]["leaderboard_publish"] = True
    record["model"] = {
        "id": "publishable-looking-system",
        "display_name": "Publishable Looking System",
        "provider": "local",
    }
    record["provider_result"] = {"provider": "local", "model_name": "publishable-looking-system"}
    return record


def test_dedicated_system_schema_and_all_fixture_runs_validate() -> None:
    schema = json.loads(SYSTEM_RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    runs = evaluate_fixture_brackets()["runs"]

    assert SYSTEM_RESULT_SCHEMA_PATH.name == "system_result.schema.json"
    assert schema["properties"]["evaluation"]["properties"]["level"]["const"] == "system"
    assert schema["properties"]["run"]["properties"]["publish"]["const"] is False
    assert sorted(runs) == ["closed_book_constant", "oracle_structured_lookup", "token_overlap_retrieval"]
    for record in runs.values():
        validate_system_result(record)


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (lambda record: record["evaluation"].pop("level"), "system_discriminator"),
        (lambda record: record["subject"].__setitem__("kind", "embedding_model"), "system_discriminator"),
        (lambda record: record["run"].__setitem__("publish", True), "system_publication_boundary"),
        (lambda record: record["metrics"].__setitem__("cost_complete_rate", 0.5), "system_cost_completeness"),
        (lambda record: record["details"]["outputs"][0].pop("trace"), "system_output_shape"),
    ],
)
def test_system_result_validation_rejects_invalid_boundaries_with_stable_reasons(
    mutate,
    reason_code: str,
) -> None:
    record = _accepted_record()
    mutate(record)

    with pytest.raises(ContractValidationError) as error:
        validate_system_result(record)

    assert error.value.reason_code == reason_code


def test_system_result_validation_rejects_incomplete_cost_and_malformed_trace() -> None:
    incomplete_cost = _accepted_record()
    incomplete_cost["details"]["outputs"][0]["usage"]["cost_complete"] = False

    with pytest.raises(ContractValidationError) as incomplete_error:
        validate_system_result(incomplete_cost)
    assert incomplete_error.value.reason_code == "system_cost_completeness"

    missing_cost = _accepted_record()
    missing_cost["details"]["outputs"][0]["usage"]["online_cost_usd"] = None

    with pytest.raises(ContractValidationError) as cost_error:
        validate_system_result(missing_cost)
    assert cost_error.value.reason_code == "missing_cost"

    malformed_trace = _accepted_record()
    malformed_trace["details"]["outputs"][0]["trace"][0]["end_ms"] = -1.0

    with pytest.raises(ContractValidationError) as trace_error:
        validate_system_result(malformed_trace)
    assert trace_error.value.reason_code == "malformed_trace"


def test_system_fixture_export_is_deterministic_validated_and_no_publish(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("System fixture export must not access the network")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    first = export_retrieval_answer_utility_fixture(tmp_path / "first-system-export")
    first_results = (first / SYSTEM_RESULTS_FILENAME).read_bytes()
    first_manifest = (first / SYSTEM_EXPORT_MANIFEST_FILENAME).read_bytes()
    rerun = export_retrieval_answer_utility_fixture(first)
    second = export_retrieval_answer_utility_fixture(tmp_path / "second-system-export")

    assert rerun == first
    assert sorted(path.name for path in first.iterdir()) == [
        SYSTEM_EXPORT_MANIFEST_FILENAME,
        SYSTEM_RESULTS_FILENAME,
    ]
    assert (rerun / SYSTEM_RESULTS_FILENAME).read_bytes() == first_results
    assert (rerun / SYSTEM_EXPORT_MANIFEST_FILENAME).read_bytes() == first_manifest
    second_results = (second / SYSTEM_RESULTS_FILENAME).read_bytes()
    second_manifest = (second / SYSTEM_EXPORT_MANIFEST_FILENAME).read_bytes()
    assert first_results == second_results
    assert first_manifest == second_manifest

    records = load_jsonl(first / SYSTEM_RESULTS_FILENAME)
    manifest = json.loads(first_manifest)
    assert [record["subject"]["id"] for record in records] == [
        "closed_book_constant",
        "oracle_structured_lookup",
        "token_overlap_retrieval",
    ]
    assert manifest["record_count"] == 3
    assert manifest["publish"] is False
    assert manifest["fixture"]["fixture_only"] is True
    assert manifest["fixture"]["publish"] is False
    assert manifest["files"][SYSTEM_RESULTS_FILENAME]["sha256"] == hashlib.sha256(first_results).hexdigest()
    for record in records:
        validate_system_result(record)


@pytest.mark.parametrize("unexpected_kind", ["file", "directory"])
def test_system_fixture_export_fails_closed_on_unowned_entries(tmp_path, unexpected_kind: str) -> None:
    output = tmp_path / "system-export"
    output.mkdir()
    results_path = output / SYSTEM_RESULTS_FILENAME
    manifest_path = output / SYSTEM_EXPORT_MANIFEST_FILENAME
    results_path.write_bytes(b"owned-results-sentinel\n")
    manifest_path.write_bytes(b"owned-manifest-sentinel\n")
    unexpected = output / "stale-embedding-data"
    if unexpected_kind == "file":
        unexpected.write_bytes(b"do-not-delete\n")
    else:
        unexpected.mkdir()

    with pytest.raises(ContractValidationError) as error:
        export_retrieval_answer_utility_fixture(output)

    assert error.value.reason_code == "system_export_boundary"
    assert "stale-embedding-data" in str(error.value)
    assert results_path.read_bytes() == b"owned-results-sentinel\n"
    assert manifest_path.read_bytes() == b"owned-manifest-sentinel\n"
    assert unexpected.exists()
    if unexpected_kind == "file":
        assert unexpected.read_bytes() == b"do-not-delete\n"


def test_publishable_looking_system_record_isolated_from_embedding_and_hf_outputs(tmp_path) -> None:
    record = _publishable_looking_system_record()
    results = tmp_path / "adversarial-system-results.jsonl"
    results.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    assert is_embedding_result_record(record) is False
    assert build_leaderboard([record]) == []

    dataset = export_dataset_repo(output_dir=tmp_path / "dataset", results_path=results)
    assert load_jsonl(dataset / "results" / "latest.jsonl") == []
    with open(dataset / "leaderboards" / "latest.csv", encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == []

    space = export_space_repo(
        output_dir=tmp_path / "space",
        bundled_leaderboard=dataset / "leaderboards" / "latest.csv",
    )
    with open(space / "leaderboard.csv", encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == []
