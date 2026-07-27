from __future__ import annotations

import copy
import hashlib
import json
import shutil
import socket
from dataclasses import replace

import pytest

from mm_embed.benchmark.leaderboard import build_leaderboard
from mm_embed.system_evaluation.retrieval_answer_utility import (
    DATASET_VERSION,
    DATA_FILES,
    EVALUATION_LEVEL,
    EVALUATION_MODE,
    FIXTURE_ROOT,
    SUBJECT_KIND,
    ClosedBookConstant,
    ContractValidationError,
    OracleStructuredLookup,
    QueryRecord,
    RetrievalAnswerUtilityFixture,
    SystemOutput,
    TokenOverlapRetrieval,
    UsageRecord,
    evaluate_fixture_brackets,
    evaluate_system,
    judge_output,
    load_retrieval_answer_utility_fixture,
    serialize_fixture,
    serialize_result,
    token_overlap_ranking,
    validate_fixture,
    validate_system_output,
    validate_usage,
)


def test_fixture_is_exact_local_invented_and_deterministic() -> None:
    first = load_retrieval_answer_utility_fixture()
    second = load_retrieval_answer_utility_fixture()

    assert len(first.queries) == 6
    assert len(first.corpus) == 12
    assert len(first.answers) == 6
    required_counts = {
        query.query_id: sum(
            qrel.query_id == query.query_id and qrel.required_for_complete_support
            for qrel in first.qrels
        )
        for query in first.queries
    }
    assert sorted(required_counts.values()) == [1, 1, 1, 1, 2, 2]
    assert first.publish is False
    assert first.evidence_tier == "fixture"
    assert first.fixture_only is True
    assert first.network == "forbidden"
    assert all(query.source_kind == "local_invented_fixture" for query in first.queries)
    assert all(document.source_kind == "local_invented_fixture" for document in first.corpus)
    assert all(len(document.content_sha256) == 64 for document in first.corpus)
    assert serialize_fixture(first) == serialize_fixture(second)
    assert first.bundle_sha256 == second.bundle_sha256
    assert "http://" not in serialize_fixture(first).lower()
    assert "https://" not in serialize_fixture(first).lower()


def test_tracked_bundle_files_hashes_and_non_self_referential_bundle_are_exact() -> None:
    fixture = load_retrieval_answer_utility_fixture()
    expected_files = {*DATA_FILES, "task_manifest.json"}

    assert {path.name for path in FIXTURE_ROOT.iterdir() if path.is_file()} == expected_files
    for name in DATA_FILES:
        path = FIXTURE_ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture.file_sha256[name]

    manifest_path = FIXTURE_ROOT / "task_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest_path.read_bytes() == (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    bundle_sha256 = manifest.pop("bundle_sha256")
    task_manifest_payload_sha256 = manifest.pop("task_manifest_payload_sha256")
    observed_manifest_payload = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert observed_manifest_payload == task_manifest_payload_sha256 == fixture.task_manifest_payload_sha256
    bundle_payload = {
        "file_sha256": fixture.file_sha256,
        "task_manifest_payload_sha256": task_manifest_payload_sha256,
    }
    observed_bundle = hashlib.sha256(
        json.dumps(bundle_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert observed_bundle == bundle_sha256 == fixture.bundle_sha256


@pytest.mark.parametrize(
    "tamper",
    ["file_bytes", "file_hash_metadata", "manifest_hash", "bundle_hash"],
)
def test_loader_rejects_tampered_tracked_bundle(tmp_path, tamper: str) -> None:
    root = tmp_path / DATASET_VERSION
    shutil.copytree(FIXTURE_ROOT, root)
    manifest_path = root / "task_manifest.json"

    if tamper == "file_bytes":
        queries_path = root / "queries.jsonl"
        queries_path.write_text(
            queries_path.read_text(encoding="utf-8").replace("amber archive", "amber archival", 1),
            encoding="utf-8",
        )
        expected_reason = "file_hash_mismatch"
    else:
        manifest = json.loads(manifest_path.read_bytes())
        if tamper == "file_hash_metadata":
            manifest["file_sha256"]["queries.jsonl"] = "0" * 64
            expected_reason = "file_hash_mismatch"
        elif tamper == "manifest_hash":
            manifest["task_manifest_payload_sha256"] = "0" * 64
            expected_reason = "manifest_hash"
        else:
            manifest["bundle_sha256"] = "0" * 64
            expected_reason = "bundle_hash"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ContractValidationError) as error:
        load_retrieval_answer_utility_fixture(root)
    assert error.value.reason_code == expected_reason


def test_fixture_validation_rejects_hash_qrel_and_publication_corruption() -> None:
    fixture = load_retrieval_answer_utility_fixture()

    bad_document = replace(fixture.corpus[0], content_sha256="0" * 64)
    with pytest.raises(ContractValidationError, match="Invalid local corpus record"):
        validate_fixture(replace(fixture, corpus=(bad_document, *fixture.corpus[1:])))

    bad_qrel = replace(fixture.qrels[0], doc_id="doc_unknown")
    with pytest.raises(ContractValidationError, match="unknown query or document"):
        validate_fixture(replace(fixture, qrels=(bad_qrel, *fixture.qrels[1:])))

    with pytest.raises(ContractValidationError, match="canonical id order") as order_error:
        validate_fixture(replace(fixture, queries=tuple(reversed(fixture.queries))))
    assert order_error.value.reason_code == "noncanonical_order"

    with pytest.raises(ContractValidationError, match="publication or provenance boundary"):
        validate_fixture(replace(fixture, publish=True))


def test_checked_in_token_overlap_rankings_reproduce_exactly() -> None:
    fixture = load_retrieval_answer_utility_fixture()

    observed = {
        query.query_id: tuple(item.doc_id for item in token_overlap_ranking(query, fixture.corpus))
        for query in fixture.queries
    }

    assert observed == fixture.expected_retrieval_rankings
    assert all(len(ranking) == 12 for ranking in observed.values())
    assert observed["q_bronze_capacity"][:2] == ("doc_bronze_primary", "doc_bronze_reserve")
    assert observed["q_indigo_services"][:2] == ("doc_indigo_north", "doc_indigo_south")


def test_three_local_systems_match_exact_metrics_and_bracket_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("Network access is forbidden for the retrieval answer utility fixture")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    fixture = load_retrieval_answer_utility_fixture()
    result = evaluate_fixture_brackets(fixture)

    assert result["diagnostics"] == {
        "retrieval_minus_closed_book_answer_accuracy": 1.0,
        "oracle_minus_retrieval_answer_accuracy": 0.0,
        "retrieval_required_citation_recall": 1.0,
        "retrieval_ranking_recall@2": 1.0,
        "answerable_with_oracle_count": 6,
        "retrieval_failures_among_oracle_answerable": 0,
        "generation_failures_with_required_context": 0,
    }
    for system_id, expected in fixture.expected_system_metrics.items():
        metrics = result["runs"][system_id]["metrics"]
        for metric, value in expected.items():
            if isinstance(value, float):
                assert metrics[metric] == pytest.approx(value)
            else:
                assert metrics[metric] == value

    closed = result["runs"]["closed_book_constant"]
    oracle = result["runs"]["oracle_structured_lookup"]
    retrieval = result["runs"]["token_overlap_retrieval"]
    assert closed["metrics"]["answer_correct_count"] == 0
    assert closed["metrics"]["valid_answer_count"] == 2
    assert closed["metrics"]["missing_required_citation_count"] == 6
    assert oracle["metrics"]["answer_correct_count"] == 6
    assert retrieval["metrics"]["answer_correct_count"] == 6
    assert {item["reason_code"] for item in oracle["details"]["per_query"]} == {"exact_match"}
    assert {item["reason_code"] for item in retrieval["details"]["per_query"]} == {"exact_match"}


def test_system_records_have_explicit_separate_discriminators() -> None:
    record = evaluate_system(TokenOverlapRetrieval())

    assert record["schema_version"] == "retrieval-answer-utility-system-fixture-v0"
    assert record["evaluation"] == {
        "family": "retrieval_answer_utility",
        "level": EVALUATION_LEVEL,
        "mode": EVALUATION_MODE,
        "leaderboard_surface": "system",
    }
    assert record["subject"]["kind"] == SUBJECT_KIND
    assert record["subject"]["id"] == "token_overlap_retrieval"
    assert len(record["subject"]["manifest_sha256"]) == 64
    assert record["run"]["publish"] is False
    assert record["run"]["evidence_tier"] == "fixture"
    assert record["task"]["id"] == DATASET_VERSION
    assert record["task"]["publish"] is False
    assert "model" not in record
    assert "provider_result" not in record


def test_unknown_and_missing_citations_have_deterministic_reason_codes() -> None:
    fixture = load_retrieval_answer_utility_fixture()
    query = next(query for query in fixture.queries if query.query_id == "q_bronze_capacity")
    output = OracleStructuredLookup().answer(query, fixture)

    unknown = replace(output, cited_doc_ids=("doc_unknown",), context_doc_ids=("doc_unknown",))
    assert judge_output(unknown, fixture).reason_code == "unknown_citation"

    missing = replace(output, cited_doc_ids=("doc_bronze_primary",))
    judged = judge_output(missing, fixture)
    assert judged.reason_code == "missing_required_citation"
    assert judged.answer_correct is True
    assert judged.citation_precision == 1.0
    assert judged.required_citation_recall == 0.5
    assert judged.citation_f1 == pytest.approx(2.0 / 3.0)


def test_malformed_typed_answers_score_zero_with_reason_code() -> None:
    fixture = load_retrieval_answer_utility_fixture()
    query = next(query for query in fixture.queries if query.query_id == "q_cobalt_flag")
    output = OracleStructuredLookup().answer(query, fixture)

    judged = judge_output(replace(output, answer="true"), fixture)

    assert judged.reason_code == "malformed_answer"
    assert judged.answer_valid is False
    assert judged.answer_correct is False


class _PartialCostRetrieval:
    manifest = TokenOverlapRetrieval.manifest

    def answer(self, query: QueryRecord, fixture: RetrievalAnswerUtilityFixture) -> SystemOutput:
        output = TokenOverlapRetrieval().answer(query, fixture)
        if output.query_id == "q_amber_policy":
            return replace(output, usage=replace(output.usage, online_cost_usd=None, cost_complete=False))
        return output


def test_missing_cost_uses_partial_fields_and_blocks_complete_comparison() -> None:
    record = evaluate_system(_PartialCostRetrieval())
    metrics = record["metrics"]

    assert metrics["cost_known_count"] == 5
    assert metrics["cost_missing_count"] == 1
    assert metrics["cost_complete_count"] == 5
    assert metrics["cost_complete_rate"] == pytest.approx(5.0 / 6.0)
    assert metrics["known_online_cost_usd_total"] == 0.0
    assert metrics["known_online_cost_usd_per_attempted_query"] == 0.0
    assert metrics["total_online_cost_usd"] is None
    assert metrics["mean_online_cost_usd"] is None
    assert metrics["cost_comparison_eligible"] is False

    with pytest.raises(ContractValidationError, match="cannot omit online cost") as error:
        validate_usage(UsageRecord(0, 0, 0, None, True))
    assert error.value.reason_code == "missing_cost"


def test_trace_or_explicit_failure_record_is_required() -> None:
    fixture = load_retrieval_answer_utility_fixture()
    query = fixture.queries[0]
    output = OracleStructuredLookup().answer(query, fixture)

    with pytest.raises(ContractValidationError, match="requires a trace") as missing_trace:
        validate_system_output(replace(output, trace=()), fixture)
    assert missing_trace.value.reason_code == "missing_trace"

    explicit_failure = replace(output, answer=None, trace=(), status="error", error="explicit local failure")
    validate_system_output(explicit_failure, fixture)
    judged = judge_output(explicit_failure, fixture)
    assert judged.reason_code == "system_error"
    assert judged.trace_complete is True

    with pytest.raises(ContractValidationError, match="explicit error record") as missing_failure:
        validate_system_output(replace(explicit_failure, error=None), fixture)
    assert missing_failure.value.reason_code == "missing_failure_record"


def test_repeatability_is_byte_stable_after_documented_timestamp_normalization() -> None:
    first = evaluate_system(
        TokenOverlapRetrieval(),
        started_at="2026-07-27T01:00:00Z",
        completed_at="2026-07-27T01:00:01Z",
    )
    second = evaluate_system(
        TokenOverlapRetrieval(),
        started_at="2026-07-27T02:00:00Z",
        completed_at="2026-07-27T02:00:01Z",
    )

    assert serialize_result(first) != serialize_result(second)
    assert serialize_result(first, normalize_timestamps=True) == serialize_result(second, normalize_timestamps=True)


def test_system_level_record_cannot_enter_embedding_leaderboard_when_publishable_looking() -> None:
    record = copy.deepcopy(evaluate_system(TokenOverlapRetrieval()))
    record["run"]["publish"] = True
    record["task"]["publish"] = True
    record["model"] = {
        "id": "publishable-looking-system",
        "display_name": "Publishable Looking System",
        "provider": "local",
    }
    record["provider_result"] = {"provider": "local", "model_name": "publishable-looking-system"}
    assert record["task"]["primary_metric"] in record["metrics"]

    assert build_leaderboard([record]) == []
