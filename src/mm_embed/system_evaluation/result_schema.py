"""Deterministic validation for fixture-only system result records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mm_embed.system_evaluation.retrieval_answer_utility import (
    DATASET_VERSION,
    EVALUATION_LEVEL,
    EVALUATION_MODE,
    EVIDENCE_TIER,
    FAMILY,
    JUDGE_REVISION,
    NETWORK,
    PRIMARY_METRIC,
    SCHEMA_VERSION,
    SUBJECT_KIND,
    ContractValidationError,
    RetrievedDocument,
    RetrievalAnswerUtilityFixture,
    SystemOutput,
    TraceEvent,
    UsageRecord,
    aggregate_system_metrics,
    judge_output,
    load_retrieval_answer_utility_fixture,
    validate_system_output,
)


SYSTEM_RESULT_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "system_result.schema.json"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "evaluation",
    "subject",
    "run",
    "timestamps",
    "task",
    "metrics",
    "resource_usage",
    "details",
    "error",
}
_OUTPUT_FIELDS = {
    "query_id",
    "answer",
    "answer_type",
    "cited_doc_ids",
    "retrieved",
    "context_doc_ids",
    "usage",
    "trace",
    "online_latency_ms",
    "status",
    "error",
}
_DETAIL_FIELDS = {
    "fixture_only",
    "publish",
    "evidence_tier",
    "network",
    "provider_api_calls",
    "model_inference_calls",
    "model_downloads",
    "fixture_bundle_sha256",
    "judge_revision",
    "outputs",
    "per_query",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: Any, *, reason: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(reason, f"{label} must be an object")
    return value


def _exact_fields(value: dict[str, Any], fields: set[str], *, reason: str, label: str) -> None:
    if set(value) != fields:
        raise ContractValidationError(reason, f"{label} fields are incomplete or unsupported")


def _non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_number(value: Any, *, non_negative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    return not non_negative or value >= 0


def _string_list(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _parse_output(value: Any) -> SystemOutput:
    output = _mapping(value, reason="system_output_shape", label="System output")
    _exact_fields(output, _OUTPUT_FIELDS, reason="system_output_shape", label="System output")

    cited_doc_ids = _string_list(output["cited_doc_ids"])
    context_doc_ids = _string_list(output["context_doc_ids"])
    if cited_doc_ids is None or context_doc_ids is None:
        raise ContractValidationError("system_output_shape", "Citation and context ids must be string arrays")

    usage_value = _mapping(output["usage"], reason="system_output_shape", label="Output usage")
    _exact_fields(
        usage_value,
        {"prompt_tokens", "completion_tokens", "llm_calls", "online_cost_usd", "cost_complete"},
        reason="system_output_shape",
        label="Output usage",
    )
    if (
        not _non_negative_int(usage_value["prompt_tokens"])
        or not _non_negative_int(usage_value["completion_tokens"])
        or not _non_negative_int(usage_value["llm_calls"])
        or not isinstance(usage_value["cost_complete"], bool)
    ):
        raise ContractValidationError("system_output_shape", "Usage counters and completeness must be typed")
    online_cost_usd = usage_value["online_cost_usd"]
    if online_cost_usd is not None and not _finite_number(online_cost_usd, non_negative=True):
        raise ContractValidationError("system_output_shape", "Online cost must be null or non-negative")
    usage = UsageRecord(
        prompt_tokens=usage_value["prompt_tokens"],
        completion_tokens=usage_value["completion_tokens"],
        llm_calls=usage_value["llm_calls"],
        online_cost_usd=online_cost_usd,
        cost_complete=usage_value["cost_complete"],
    )

    retrieved_value = output["retrieved"]
    if not isinstance(retrieved_value, (list, tuple)):
        raise ContractValidationError("system_output_shape", "Retrieved documents must be an array")
    retrieved: list[RetrievedDocument] = []
    for item in retrieved_value:
        item = _mapping(item, reason="system_output_shape", label="Retrieved document")
        _exact_fields(item, {"doc_id", "rank", "score"}, reason="system_output_shape", label="Retrieved document")
        if (
            not isinstance(item["doc_id"], str)
            or not _non_negative_int(item["rank"])
            or item["rank"] == 0
            or not _finite_number(item["score"])
        ):
            raise ContractValidationError("system_output_shape", "Retrieved document fields must be typed")
        retrieved.append(RetrievedDocument(item["doc_id"], item["rank"], float(item["score"])))

    trace_value = output["trace"]
    if not isinstance(trace_value, (list, tuple)):
        raise ContractValidationError("system_output_shape", "Trace must be an array")
    trace: list[TraceEvent] = []
    for item in trace_value:
        item = _mapping(item, reason="system_output_shape", label="Trace event")
        _exact_fields(
            item,
            {"event", "component", "start_ms", "end_ms"},
            reason="system_output_shape",
            label="Trace event",
        )
        if (
            not isinstance(item["event"], str)
            or not isinstance(item["component"], str)
            or not _finite_number(item["start_ms"])
            or not _finite_number(item["end_ms"])
        ):
            raise ContractValidationError("system_output_shape", "Trace event fields must be typed")
        trace.append(TraceEvent(item["event"], item["component"], float(item["start_ms"]), float(item["end_ms"])))

    if (
        not isinstance(output["query_id"], str)
        or not isinstance(output["answer_type"], str)
        or not isinstance(output["status"], str)
        or (output["error"] is not None and not isinstance(output["error"], str))
        or not _finite_number(output["online_latency_ms"], non_negative=True)
    ):
        raise ContractValidationError("system_output_shape", "System output scalar fields must be typed")
    return SystemOutput(
        query_id=output["query_id"],
        answer=output["answer"],
        answer_type=output["answer_type"],
        cited_doc_ids=cited_doc_ids,
        retrieved=tuple(retrieved),
        context_doc_ids=context_doc_ids,
        usage=usage,
        trace=tuple(trace),
        online_latency_ms=float(output["online_latency_ms"]),
        status=output["status"],
        error=output["error"],
    )


def _validate_discriminators(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = _mapping(record.get("evaluation"), reason="system_discriminator", label="Evaluation")
    subject = _mapping(record.get("subject"), reason="system_discriminator", label="Subject")
    if record.get("schema_version") != SCHEMA_VERSION or evaluation != {
        "family": FAMILY,
        "level": EVALUATION_LEVEL,
        "mode": EVALUATION_MODE,
        "leaderboard_surface": "system",
    }:
        raise ContractValidationError("system_discriminator", "System evaluation discriminators are incomplete")
    if subject.get("kind") != SUBJECT_KIND or not isinstance(subject.get("id"), str) or not subject["id"]:
        raise ContractValidationError("system_discriminator", "System subject discriminator is invalid")
    return evaluation, subject


def _validate_manifest(subject: dict[str, Any]) -> None:
    _exact_fields(subject, {"kind", "id", "manifest_sha256", "manifest"}, reason="system_manifest", label="Subject")
    manifest = _mapping(subject["manifest"], reason="system_manifest", label="System manifest")
    _exact_fields(
        manifest,
        {
            "system_id",
            "system_revision",
            "bracket",
            "evaluation_level",
            "mode",
            "subject_kind",
            "components",
            "execution",
        },
        reason="system_manifest",
        label="System manifest",
    )
    execution = _mapping(manifest["execution"], reason="system_manifest", label="System execution")
    _exact_fields(
        execution,
        {"concurrency", "timeout_s", "max_retries", "cache_policy", "network"},
        reason="system_manifest",
        label="System execution",
    )
    if (
        manifest.get("system_id") != subject["id"]
        or manifest.get("evaluation_level") != EVALUATION_LEVEL
        or manifest.get("mode") != EVALUATION_MODE
        or manifest.get("subject_kind") != SUBJECT_KIND
        or not isinstance(manifest.get("system_revision"), str)
        or not manifest["system_revision"]
        or not isinstance(manifest.get("bracket"), str)
        or not manifest["bracket"]
        or not isinstance(manifest.get("components"), dict)
        or not _non_negative_int(execution.get("concurrency"))
        or not _finite_number(execution.get("timeout_s"), non_negative=True)
        or not _non_negative_int(execution.get("max_retries"))
        or not isinstance(execution.get("cache_policy"), str)
        or not execution["cache_policy"]
        or execution.get("network") != NETWORK
    ):
        raise ContractValidationError("system_manifest", "System manifest boundary is invalid")
    digest = subject["manifest_sha256"]
    if not isinstance(digest, str) or not _HEX_SHA256.fullmatch(digest):
        raise ContractValidationError("system_manifest_hash", "System manifest hash is malformed")
    if digest != _sha256_text(_canonical_json(manifest)):
        raise ContractValidationError("system_manifest_hash", "System manifest hash does not reproduce")


def _validate_publication_boundary(record: dict[str, Any], subject_id: str) -> None:
    run = _mapping(record.get("run"), reason="system_publication_boundary", label="Run")
    task = _mapping(record.get("task"), reason="system_publication_boundary", label="Task")
    details = _mapping(record.get("details"), reason="system_publication_boundary", label="Details")
    _exact_fields(
        details,
        _DETAIL_FIELDS,
        reason="system_publication_boundary",
        label="System details",
    )
    if run != {
        "id": f"{DATASET_VERSION}:{subject_id}",
        "publish": False,
        "evidence_tier": EVIDENCE_TIER,
    }:
        raise ContractValidationError(
            "system_publication_boundary",
            "System run must remain fixture-only and unpublished",
        )
    if task != {
        "id": DATASET_VERSION,
        "display_name": "Retrieval answer utility fixture",
        "primary_metric": PRIMARY_METRIC,
        "publish": False,
        "evidence_tier": EVIDENCE_TIER,
    }:
        raise ContractValidationError(
            "system_publication_boundary",
            "System task must remain fixture-only and unpublished",
        )
    if (
        details.get("fixture_only") is not True
        or details.get("publish") is not False
        or details.get("evidence_tier") != EVIDENCE_TIER
        or details.get("network") != NETWORK
        or details.get("provider_api_calls") != 0
        or details.get("model_inference_calls") != 0
        or details.get("model_downloads") != 0
    ):
        raise ContractValidationError("system_publication_boundary", "System details exceed the local fixture boundary")


def _validate_timestamps(record: dict[str, Any]) -> None:
    timestamps = _mapping(record.get("timestamps"), reason="system_result_shape", label="Timestamps")
    _exact_fields(
        timestamps,
        {"started_at", "completed_at", "duration_s"},
        reason="system_result_shape",
        label="Timestamps",
    )
    if (
        not isinstance(timestamps["started_at"], str)
        or not isinstance(timestamps["completed_at"], str)
        or not _finite_number(timestamps["duration_s"], non_negative=True)
    ):
        raise ContractValidationError("system_result_shape", "System timestamps must be typed")


def _validate_cost_and_metrics(
    record: dict[str, Any],
    outputs: tuple[SystemOutput, ...],
    fixture: RetrievalAnswerUtilityFixture,
) -> None:
    metrics = _mapping(record.get("metrics"), reason="system_metric_shape", label="Metrics")
    if any(
        value is not None and not isinstance(value, bool) and not _finite_number(value)
        for value in metrics.values()
    ):
        raise ContractValidationError("system_metric_shape", "System metrics must contain finite typed values")
    judgments = tuple(judge_output(output, fixture) for output in outputs)
    expected_metrics = aggregate_system_metrics(outputs, judgments, fixture)
    expected_local_usage = UsageRecord(0, 0, 0, 0.0, True)
    if any(output.usage != expected_local_usage for output in outputs):
        raise ContractValidationError(
            "system_cost_completeness",
            "Accepted local fixture results require complete zero-cost usage",
        )
    cost_fields = {
        "cost_known_count",
        "cost_missing_count",
        "cost_complete_count",
        "cost_complete_rate",
        "known_online_cost_usd_total",
        "known_online_cost_usd_per_attempted_query",
        "total_online_cost_usd",
        "mean_online_cost_usd",
        "cost_comparison_eligible",
    }
    if any(metrics.get(field) != expected_metrics[field] for field in cost_fields):
        raise ContractValidationError("system_cost_completeness", "System cost completeness does not reproduce")
    if metrics != expected_metrics:
        raise ContractValidationError("system_metric_mismatch", "System aggregate metrics do not reproduce")

    resource_usage = _mapping(record.get("resource_usage"), reason="system_cost_completeness", label="Resource usage")
    expected_usage = {
        "online_system_cost_usd": metrics["total_online_cost_usd"],
        "known_online_system_cost_usd": metrics["known_online_cost_usd_total"],
        "cost_known_count": metrics["cost_known_count"],
        "cost_missing_count": metrics["cost_missing_count"],
        "cost_complete_rate": metrics["cost_complete_rate"],
        "judge_cost_usd": 0.0,
    }
    if resource_usage != expected_usage:
        raise ContractValidationError("system_cost_completeness", "System resource usage does not match metrics")

    per_query = record["details"].get("per_query")
    expected_per_query = [asdict(judgment) for judgment in judgments]
    if per_query != expected_per_query:
        raise ContractValidationError("system_judgment_mismatch", "Per-query judgments do not reproduce")


def validate_system_result(
    record: dict[str, Any],
    fixture: RetrievalAnswerUtilityFixture | None = None,
) -> None:
    """Validate one system-only fixture result without external dependencies."""
    if not isinstance(record, dict):
        raise ContractValidationError("system_result_shape", "System result must be an object")
    _exact_fields(record, _TOP_LEVEL_FIELDS, reason="system_result_shape", label="System result")
    _, subject = _validate_discriminators(record)
    _validate_manifest(subject)
    _validate_publication_boundary(record, subject["id"])
    _validate_timestamps(record)
    if record["error"] is not None:
        raise ContractValidationError("system_result_shape", "Accepted fixture results cannot contain a run error")

    fixture = fixture or load_retrieval_answer_utility_fixture()
    details = record["details"]
    if details.get("judge_revision") != JUDGE_REVISION:
        raise ContractValidationError("system_result_shape", "System judge revision is invalid")
    if details.get("fixture_bundle_sha256") != fixture.bundle_sha256:
        raise ContractValidationError("system_fixture_evidence", "Fixture bundle hash does not match")
    output_values = details.get("outputs")
    if not isinstance(output_values, list):
        raise ContractValidationError("system_output_shape", "System outputs must be an array")
    outputs = tuple(_parse_output(value) for value in output_values)
    expected_query_ids = [query.query_id for query in fixture.queries]
    if [output.query_id for output in outputs] != expected_query_ids:
        raise ContractValidationError(
            "system_output_shape",
            "System outputs must cover fixture queries in canonical order",
        )
    for output in outputs:
        validate_system_output(output, fixture)
    if not isinstance(details.get("per_query"), list) or len(details["per_query"]) != len(outputs):
        raise ContractValidationError("system_output_shape", "Per-query judgments must match system outputs")
    _validate_cost_and_metrics(record, outputs, fixture)
