"""Export benchmark artifacts into Hugging Face-ready folders."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from mm_embed.benchmark.leaderboard import TRAINING_OVERLAP_FIELDNAMES, build_leaderboard, primary_metric_value
from mm_embed.benchmark.registry import (
    BenchmarkCatalog,
    ModelSpec,
    TaskSpec,
    load_catalog,
    load_run_manifest,
    normalize_evidence_tier,
)
from mm_embed.benchmark.results import is_embedding_result_record, json_safe, load_jsonl
from mm_embed.benchmark.training_overlap import (
    load_relationship_registry,
    public_assessment_for_record,
    public_model_training_projection,
    public_task_source_projection,
    validate_assessment_registry_binding,
)


DEFAULT_EXPORT_ROOT = Path("dist/huggingface")
PUBLIC_EXCLUDED_PROVIDERS = {"geevec_api", "geevec_lite"}
PUBLIC_EXCLUDED_MARKERS = ("geevec",)
LEADERBOARD_BASE_FIELDNAMES = [
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
LEADERBOARD_PROVENANCE_FIELDNAMES = [
    "evidence_tier",
    "evidence_source",
    "task_model_duplicate_count",
    "task_model_run_rank",
    "is_latest_for_task_model",
]
LEADERBOARD_OPERATIONAL_FIELDNAMES = [
    "run_started_at",
    "run_finished_at",
    "dataset_version",
    "input_count_total",
    "token_usage",
    "provider_latency_ms",
    "cost_usd",
    "fresh_provider_calls",
    "cache_enabled",
]
LEADERBOARD_FIELDNAMES = (
    LEADERBOARD_BASE_FIELDNAMES
    + LEADERBOARD_PROVENANCE_FIELDNAMES
    + LEADERBOARD_OPERATIONAL_FIELDNAMES
    + TRAINING_OVERLAP_FIELDNAMES
)

TRAINING_OVERLAP_WARNING = (
    "Training-overlap status is interpretation evidence for this model revision, "
    "task source, and reviewed relationship-table revision. Unknown means "
    "unreported, incomplete, unresolved, or stale; it does not mean zero-shot. "
    "Status does not change the task score or ranking."
)
PUBLIC_TRAINING_OVERLAP_FIELDS = (
    "schema_version",
    "relationship_registry_revision",
    "relationship_registry_sha256",
    "model_revision",
    "model_training_evidence_revision",
    "task_dataset_version",
    "task_source_evidence_revision",
    "data_overlap_status",
    "task_training_status",
    "zero_shot_status",
    "matched_model_ids",
    "matched_training_source_ids",
    "matched_evaluation_source_ids",
    "relationship_ids",
    "reason_codes",
    "assessed_at",
)
_DROP_PUBLIC_VALUE = object()
_NON_PUBLIC_RESULT_KEYS = {
    "api_key_env",
    "environment",
    "environment_variables",
    "evaluation_sources",
    "private_notes",
    "raw_prompt",
    "raw_provider_payload",
    "raw_request",
    "raw_response",
    "raw_source_payload",
    "request_body",
    "request_headers",
    "response_body",
    "response_headers",
    "review",
    "review_notes",
    "reviewer_notes",
    "training_data",
}
_SECRET_RESULT_KEYS = {
    "access_key",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
    "set_cookie",
    "token",
}
_SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+[a-z0-9._~+/=-]{8,}|(?:sk|hf|gh[pousr])_[a-z0-9_-]{8,}|"
    r"sk-[a-z0-9_-]{12,}|github_pat_[a-z0-9_]{12,}|AKIA[0-9A-Z]{12,}|AIza[0-9A-Za-z_-]{20,})"
)
_INLINE_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|secret|credential)\s*[:=]\s*\S{4,}"
)
_SECRET_ENV_VALUE_PATTERN = re.compile(
    r"^[A-Z][A-Z0-9_]*(?:API_KEY|ACCESS_KEY|PRIVATE_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)$"
)
_EMBEDDED_PRIVATE_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'(=])(?:~[/\\]|/(?:home|root|Users|data\d*|mnt|tmp|workspace)(?:[/\\]|$)|"
    r"[A-Za-z]:[/\\]|\\\\)"
)
_CREDENTIAL_URL_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@\s:]+:[^/@\s]+@", re.IGNORECASE)

TASK_NOTES = {
    "mrl_stress": {
        "label": "MRL compression stress",
        "summary": "Semantic stability when embeddings are truncated to smaller dimensions.",
        "metric": "Spearman correlation at the configured low dimension.",
    },
    "crosslingual_retrieval": {
        "label": "Chinese-English retrieval",
        "summary": "Bidirectional technical retrieval with hard negatives across Chinese and English.",
        "metric": "Hard-negative average recall@1.",
    },
    "needle_in_haystack": {
        "label": "Long-document needle retrieval",
        "summary": "Retrieving facts inserted at different positions in long documents.",
        "metric": "Overall accuracy across length and position buckets.",
    },
    "cross_modal_retrieval": {
        "label": "Text-image retrieval",
        "summary": "COCO-style text-image matching with hard negative captions.",
        "metric": "Hard-negative average recall@1.",
    },
}


def _reset_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(json_safe(data), f, sort_keys=False, allow_unicode=True)


def _is_public_model(model: ModelSpec) -> bool:
    return (
        model.publish
        and model.provider.lower() not in PUBLIC_EXCLUDED_PROVIDERS
        and not _contains_excluded_marker(model.id, model.display_name, model.provider)
    )


def _is_public_task(task: TaskSpec) -> bool:
    return task.publish and task.leaderboard_publish


def _space_task_catalog_rows(catalog: BenchmarkCatalog) -> list[dict[str, Any]]:
    rows = []
    for task in sorted(catalog.tasks.values(), key=lambda item: item.id):
        if not _is_public_task(task):
            continue
        rows.append(
            {
                "id": task.id,
                "display_name": task.display_name,
                "description": task.description,
                "primary_metric": task.primary_metric,
                "metric_direction": task.metric_direction,
                "dataset_version": task.dataset_version,
                "required_modalities": task.required_modalities,
                "publish": True,
                "leaderboard_publish": True,
                "evaluation_sources": public_task_source_projection(task),
            }
        )
    return rows


def _contains_excluded_marker(*values: Any) -> bool:
    for value in values:
        text = str(value or "").lower()
        if any(marker in text for marker in PUBLIC_EXCLUDED_MARKERS):
            return True
    return False


def _space_model_catalog_rows(catalog: BenchmarkCatalog) -> list[dict[str, Any]]:
    rows = []
    for model in catalog.models.values():
        if not _is_public_model(model):
            continue
        rows.append(
            {
                "id": model.id,
                "display_name": model.display_name,
                "provider": model.provider,
                "modalities": model.modalities,
                "dimensions": model.dimensions,
                "max_text_length": model.max_text_length,
                "supports_mrl": model.supports_mrl,
                "access": model.access,
                "status": model.status,
                "source": model.source,
                "training_data": public_model_training_projection(model),
            }
        )
    rows.sort(key=lambda row: (row["display_name"].lower(), row["provider"].lower(), row["id"].lower()))
    return rows


def _public_records(
    records: list[dict[str, Any]],
    private_model_ids: set[str],
    private_task_ids: set[str],
) -> list[dict[str, Any]]:
    public = []
    for record in records:
        if not is_embedding_result_record(record):
            continue
        run = record.get("run") or {}
        model = record.get("model") or {}
        provider_result = record.get("provider_result") or {}
        task = record.get("task") or {}
        providers = {str(model.get("provider") or ""), str(provider_result.get("provider") or "")}
        model_ids = {str(model.get("id") or ""), str(provider_result.get("model_name") or "")}
        if run.get("publish") is False:
            continue
        if private_model_ids.intersection(model_ids):
            continue
        if PUBLIC_EXCLUDED_PROVIDERS.intersection(providers):
            continue
        if str(task.get("id") or "") in private_task_ids or task.get("publish") is False:
            continue
        if _contains_excluded_marker(
            model.get("id"),
            model.get("display_name"),
            model.get("provider"),
            provider_result.get("provider"),
            provider_result.get("model_name"),
        ):
            continue
        public.append(_public_result_record(record))
    return public


def _dataset_model_catalog_rows(catalog: BenchmarkCatalog) -> list[dict[str, Any]]:
    return [
        {
            "id": model.id,
            "display_name": model.display_name,
            "provider": model.provider,
            "modalities": model.modalities,
            "dimensions": model.dimensions,
            "max_text_length": model.max_text_length,
            "supports_mrl": model.supports_mrl,
            "access": model.access,
            "status": model.status,
            "source": model.source,
            "tags": model.tags,
            "training_data": public_model_training_projection(model),
        }
        for model in catalog.models.values()
        if _is_public_model(model)
    ]


def _dataset_task_catalog_rows(catalog: BenchmarkCatalog) -> list[dict[str, Any]]:
    return [
        {
            "id": task.id,
            "display_name": task.display_name,
            "description": task.description,
            "required_modalities": task.required_modalities,
            "primary_metric": task.primary_metric,
            "metric_direction": task.metric_direction,
            "dataset_version": task.dataset_version,
            "publish": True,
            "leaderboard_publish": True,
            "tags": task.tags,
            "evaluation_sources": public_task_source_projection(task),
        }
        for task in catalog.tasks.values()
        if _is_public_task(task)
    ]


def _public_result_record(record: dict[str, Any]) -> dict[str, Any]:
    """Preserve existing public evidence while recursively removing unsafe values."""
    overlap = public_assessment_for_record(record)
    source_record = {key: value for key, value in record.items() if key != "training_overlap"}
    sanitized = _sanitize_public_result_value(source_record)
    if not isinstance(sanitized, dict):
        raise ValueError("Public result sanitizer did not produce a result mapping")
    public_record = sanitized
    public_record["training_overlap"] = {
        key: overlap.get(key)
        for key in PUBLIC_TRAINING_OVERLAP_FIELDS
    }
    original_error = record.get("error")
    if original_error and not isinstance(public_record.get("error"), str):
        public_record["error"] = "Evaluation failed"
    return public_record


def _sanitize_public_result_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_non_public_result_key(key):
                continue
            public_item = _sanitize_public_result_value(item)
            if public_item is not _DROP_PUBLIC_VALUE:
                sanitized[key] = public_item
        return sanitized
    if isinstance(value, (list, tuple)):
        sanitized_items = []
        for item in value:
            public_item = _sanitize_public_result_value(item)
            if public_item is not _DROP_PUBLIC_VALUE:
                sanitized_items.append(public_item)
        return sanitized_items
    if isinstance(value, str):
        return _DROP_PUBLIC_VALUE if _is_unsafe_public_result_text(value) else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _DROP_PUBLIC_VALUE


def _is_non_public_result_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized in _NON_PUBLIC_RESULT_KEYS or normalized in _SECRET_RESULT_KEYS:
        return True
    if normalized in {"token_count", "token_usage", "input_tokens", "output_tokens", "total_tokens"}:
        return False
    if normalized.startswith(("private_", "reviewer_", "internal_only_")):
        return True
    if normalized.startswith(
        (
            "access_key_",
            "access_token_",
            "api_key_",
            "auth_token_",
            "authorization_",
            "credential_",
            "password_",
            "private_key_",
            "secret_",
        )
    ):
        return True
    return normalized.endswith(
        (
            "_api_key",
            "_access_key",
            "_auth_token",
            "_credential",
            "_credentials",
            "_password",
            "_private_key",
            "_refresh_token",
            "_secret",
            "_secret_key",
            "_token",
        )
    )


def _is_unsafe_public_result_text(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if Path(text).is_absolute() or text.startswith(("~/", "~\\", "$HOME/", "${HOME}/")):
        return True
    if text.lower().startswith("file://"):
        return True
    if _EMBEDDED_PRIVATE_PATH_PATTERN.search(text):
        return True
    if (
        _SECRET_VALUE_PATTERN.search(text)
        or _SECRET_ENV_VALUE_PATTERN.fullmatch(text)
        or _INLINE_SECRET_ASSIGNMENT_PATTERN.search(text)
    ):
        return True
    return bool(_CREDENTIAL_URL_PATTERN.match(text))


def _read_leaderboard_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or LEADERBOARD_FIELDNAMES)


def _write_leaderboard_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    requested_fields = list(fieldnames or LEADERBOARD_FIELDNAMES)
    ordered_fields = [field for field in requested_fields if field not in TRAINING_OVERLAP_FIELDNAMES]
    extras = sorted({key for row in rows for key in row if key not in requested_fields})
    final_fields = ordered_fields + extras + TRAINING_OVERLAP_FIELDNAMES
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _leaderboard_fieldnames_with_provenance(fieldnames: list[str] | None) -> list[str]:
    fields = [
        field
        for field in (fieldnames or LEADERBOARD_BASE_FIELDNAMES)
        if field not in TRAINING_OVERLAP_FIELDNAMES
    ]
    for field in LEADERBOARD_PROVENANCE_FIELDNAMES + LEADERBOARD_OPERATIONAL_FIELDNAMES + TRAINING_OVERLAP_FIELDNAMES:
        if field not in fields:
            fields.append(field)
    return fields


def _normalize_overlap_row(row: dict[str, Any]) -> None:
    row.setdefault("data_overlap_status", "unknown")
    row.setdefault("task_training_status", "unknown")
    row.setdefault("zero_shot_status", "unknown")
    row.setdefault("overlap_reason_codes", "legacy_missing_contract")
    row.setdefault("overlap_relationship_registry_revision", "legacy")


def _public_leaderboard_rows(
    rows: list[dict[str, Any]],
    private_model_ids: set[str],
    private_task_ids: set[str],
    result_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    unpublished_run_keys = {
        _record_leaderboard_key(record)
        for record in result_records or []
        if (record.get("run") or {}).get("publish") is False
    }
    public = []
    for row in rows:
        provider = str(row.get("provider") or "")
        model_id = str(row.get("model_id") or "")
        model = str(row.get("model") or "")
        task_id = str(row.get("task_id") or "")
        if _leaderboard_row_key(row) in unpublished_run_keys:
            continue
        if provider in PUBLIC_EXCLUDED_PROVIDERS:
            continue
        if model_id in private_model_ids:
            continue
        if task_id in private_task_ids:
            continue
        if _contains_excluded_marker(provider, model_id, model):
            continue
        _normalize_overlap_row(row)
        public.append(row)
    return public


def _leaderboard_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("task_id") or ""),
        str(row.get("model_id") or row.get("model") or ""),
        str(row.get("run_id") or ""),
    )


def _leaderboard_group_key(row: dict[str, Any]) -> tuple[str, str]:
    task_id, model_id, _ = _leaderboard_row_key(row)
    return task_id, model_id


def _record_leaderboard_key(record: dict[str, Any]) -> tuple[str, str, str]:
    task = record.get("task") or {}
    model = record.get("model") or {}
    run = record.get("run") or {}
    return (
        str(task.get("id") or ""),
        str(model.get("id") or model.get("display_name") or ""),
        str(run.get("id") or ""),
    )


def _explicit_record_evidence_tier(record: dict[str, Any]) -> str | None:
    run = record.get("run") or {}
    metadata = run.get("metadata") or {}
    for value in (
        run.get("evidence_tier"),
        record.get("evidence_tier"),
        metadata.get("evidence_tier"),
        metadata.get("tier"),
    ):
        if value is not None and str(value).strip():
            return normalize_evidence_tier(value)
    return None


def _record_evidence_tier(record: dict[str, Any]) -> str:
    explicit_tier = _explicit_record_evidence_tier(record)
    if explicit_tier is not None:
        return explicit_tier

    run = record.get("run") or {}
    model = record.get("model") or {}
    task = record.get("task") or {}
    metadata = run.get("metadata") or {}
    tags = [*(model.get("tags") or []), *(task.get("tags") or [])]
    text = " ".join(
        str(value or "").lower()
        for value in (
            run.get("id"),
            run.get("description"),
            model.get("access"),
            metadata.get("legacy_source"),
            *tags,
        )
    )
    if metadata.get("legacy_source") or "legacy" in text:
        return "legacy"
    if "smoke" in text:
        return "smoke"
    return "benchmark"


def _row_evidence_tier(row: dict[str, Any]) -> str:
    if row.get("evidence_tier") is not None and str(row["evidence_tier"]).strip():
        return normalize_evidence_tier(row["evidence_tier"])
    run_id = str(row.get("run_id") or "").lower()
    if run_id.startswith("legacy:") or "legacy" in run_id:
        return "legacy"
    if "smoke" in run_id:
        return "smoke"
    return "unknown"


def _record_evidence_source(record: dict[str, Any]) -> str:
    run = record.get("run") or {}
    metadata = run.get("metadata") or {}
    for key in ("legacy_source", "source", "results_path"):
        value = _safe_public_text(metadata.get(key), max_length=512)
        if value:
            return value
    return _safe_public_text(run.get("git_sha")) or _safe_public_text(run.get("id")) or ""


def _safe_public_text(value: Any, *, max_length: int = 128) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > max_length or any(char in text for char in "\x00\r\n"):
        return None
    if text.startswith(("/", "\\\\")):
        return None
    if text.startswith("~/") or text[0] in "=+-@":
        return None
    if len(text) >= 3 and text[1] == ":" and text[2] in {"/", "\\"}:
        return None
    return text


def _safe_nonnegative_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return value


def _safe_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _record_operational_evidence(record: dict[str, Any]) -> dict[str, Any]:
    timestamps = record.get("timestamps") if isinstance(record.get("timestamps"), dict) else {}
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    input_cardinality = (
        details.get("input_cardinality") if isinstance(details.get("input_cardinality"), dict) else {}
    )
    candidates = {
        "run_started_at": _safe_public_text(timestamps.get("started_at")),
        "run_finished_at": _safe_public_text(timestamps.get("finished_at")),
        "dataset_version": _safe_public_text(task.get("dataset_version")),
        "input_count_total": _safe_nonnegative_int(input_cardinality.get("total")),
        "token_usage": _safe_nonnegative_int(details.get("token_usage")),
        "provider_latency_ms": _safe_nonnegative_number(details.get("provider_latency_ms")),
        "cost_usd": _safe_nonnegative_number(details.get("cost_usd")),
        "fresh_provider_calls": (
            str(details["fresh_provider_calls"]).lower()
            if isinstance(details.get("fresh_provider_calls"), bool)
            else None
        ),
        "cache_enabled": (
            str(details["cache_enabled"]).lower() if isinstance(details.get("cache_enabled"), bool) else None
        ),
    }
    return {key: value for key, value in candidates.items() if value is not None}


def _leaderboard_provenance_by_key(
    records: list[dict[str, Any]],
    catalog: BenchmarkCatalog,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    provenance: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.get("error") or primary_metric_value(record, catalog) is None:
            continue
        explicit_tier = _explicit_record_evidence_tier(record)
        provenance[_record_leaderboard_key(record)].append(
            {
                "_source_index": index,
                "evidence_tier": _record_evidence_tier(record),
                "evidence_tier_explicit": explicit_tier is not None,
                "evidence_source": _record_evidence_source(record),
                **_record_operational_evidence(record),
                **_record_overlap_evidence(record),
            }
        )
    return provenance


def _enrich_leaderboard_rows(
    rows: list[dict[str, Any]],
    *,
    result_records: list[dict[str, Any]],
    catalog: BenchmarkCatalog,
) -> list[dict[str, Any]]:
    enriched = [dict(row) for row in rows]
    provenance = _leaderboard_provenance_by_key(result_records, catalog) if result_records else {}

    for index, row in enumerate(enriched):
        matches = provenance.get(_leaderboard_row_key(row)) or []
        match = matches.pop(0) if matches else {}
        if result_records:
            for field in LEADERBOARD_OPERATIONAL_FIELDNAMES:
                row.pop(field, None)
        row["_source_index"] = match.get("_source_index", index)
        explicit_row_tier = (
            normalize_evidence_tier(row["evidence_tier"])
            if row.get("evidence_tier") is not None and str(row["evidence_tier"]).strip()
            else None
        )
        if match.get("evidence_tier_explicit"):
            row["evidence_tier"] = match["evidence_tier"]
        elif explicit_row_tier is not None:
            row["evidence_tier"] = explicit_row_tier
        else:
            row["evidence_tier"] = match.get("evidence_tier") or _row_evidence_tier(row)
        row["evidence_source"] = match.get("evidence_source") or row.get("evidence_source") or ""
        for field in LEADERBOARD_OPERATIONAL_FIELDNAMES:
            if field in match:
                row[field] = match[field]
        for field in TRAINING_OVERLAP_FIELDNAMES:
            if field in match:
                row[field] = match[field]
        _normalize_overlap_row(row)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[_leaderboard_group_key(row)].append(row)

    for group_rows in groups.values():
        group_rows.sort(key=lambda row: int(row.get("_source_index") or 0))
        group_size = len(group_rows)
        for rank, row in enumerate(group_rows, start=1):
            row["task_model_duplicate_count"] = group_size
            row["task_model_run_rank"] = rank
            row["is_latest_for_task_model"] = str(rank == group_size).lower()
            row.pop("_source_index", None)

    return enriched


def _record_overlap_evidence(record: dict[str, Any]) -> dict[str, Any]:
    overlap = public_assessment_for_record(record)
    return {
        "data_overlap_status": overlap["data_overlap_status"],
        "task_training_status": overlap["task_training_status"],
        "zero_shot_status": overlap["zero_shot_status"],
        "overlap_reason_codes": ";".join(overlap["reason_codes"]),
        "overlap_relationship_registry_revision": overlap["relationship_registry_revision"],
    }


def _result_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    failed = sum(1 for record in records if record.get("error"))
    total = len(records)
    return {
        "total": total,
        "successful": total - failed,
        "failed": failed,
    }


def _leaderboard_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = Counter(
        _leaderboard_group_key(row)
        for row in rows
        if row.get("task_id") and (row.get("model_id") or row.get("model"))
    )
    evidence_tiers = Counter(str(row.get("evidence_tier") or "unknown") for row in rows)
    return {
        "rows": len(rows),
        "tasks": len({row.get("task_id") for row in rows if row.get("task_id")}),
        "providers": len({row.get("provider") for row in rows if row.get("provider")}),
        "task_model_pairs": len(groups),
        "duplicate_task_model_repeats": sum(count - 1 for count in groups.values() if count > 1),
        "latest_task_model_rows": sum(1 for row in rows if str(row.get("is_latest_for_task_model")).lower() == "true"),
        "evidence_tiers": dict(sorted(evidence_tiers.items())),
    }


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _write_dataset_card(
    path: Path,
    *,
    catalog: BenchmarkCatalog,
    public_models: list,
    public_tasks: list,
    excluded_private_models: int,
    result_stats: dict[str, int],
    leaderboard_stats: dict[str, Any],
    include_data: bool,
) -> None:
    task_rows = []
    for task in sorted(public_tasks, key=lambda item: item.id):
        note = TASK_NOTES.get(task.id, {})
        task_rows.append(
            "| {id} | {name} | {metric} | {summary} |".format(
                id=task.id,
                name=note.get("label") or task.display_name,
                metric=task.primary_metric or "",
                summary=note.get("summary") or task.description,
            )
        )

    result_note = (
        f"{result_stats['total']} records, {result_stats['successful']} successful, "
        f"{result_stats['failed']} failed"
        if result_stats["total"]
        else "No result file bundled yet."
    )
    data_note = "Bundled JSONL benchmark inputs are included." if include_data else "Benchmark inputs are not bundled."
    evidence_note = _format_counts(leaderboard_stats["evidence_tiers"])
    path.write_text(
        f"""---
license: mit
tags:
  - benchmark
  - embeddings
  - retrieval
  - multimodal
  - rag
pretty_name: Modern Embedding Bench
---

# Modern Embedding Bench

Modern Embedding Bench evaluates embedding models on practical retrieval tasks
that show up in current AI systems but are often under-covered by broad
leaderboards. The focus is on agent memory, tool and document retrieval,
long-context RAG, cross-lingual technical retrieval, coding-oriented retrieval,
and multimodal search rather than a single aggregate score.

The companion leaderboard Space is available at:
https://huggingface.co/spaces/zc277584121/modern-embedding-bench-leaderboard

The source code is available at:
https://github.com/zc277584121/modern-embedding-bench

## Contents

- `models.jsonl`: registered model specs
- `tasks.jsonl`: registered task specs and primary metrics
- `runs/`: run manifests used to produce results
- `results/latest.jsonl`: canonical v2 result records
- `leaderboards/latest.csv`: flat leaderboard table derived from result records
- `benchmark_data/`: optional benchmark input data exported from the local repo

## Current Public Export

- Registry model specs: {len(catalog.models)}
- Public model specs exported: {len(public_models)}
- Excluded private or preview model specs: {excluded_private_models}
- Public task specs exported: {len(public_tasks)}
- Result records: {result_note}
- Leaderboard rows: {leaderboard_stats["rows"]}
- Tasks with leaderboard rows: {leaderboard_stats["tasks"]}
- Providers with leaderboard rows: {leaderboard_stats["providers"]}
- Unique task/model leaderboard pairs: {leaderboard_stats["task_model_pairs"]}
- Duplicate task/model repeats kept for inspection: {leaderboard_stats["duplicate_task_model_repeats"]}
- Latest task/model marker rows: {leaderboard_stats["latest_task_model_rows"]}
- Evidence tiers: {evidence_note}
- Data: {data_note}

## Tasks

| ID | Name | Primary metric | What it probes |
| --- | --- | --- | --- |
{chr(10).join(task_rows)}

## Result Format

Each line in `results/latest.jsonl` is one model-task run. Important fields:

- `run`: run id, description, metadata, publication intent, normalized evidence tier, and git sha when available
- `model`: model id, display name, provider, modalities, dimensions, and tags
- `task`: task id, dataset version, primary metric, and task kwargs
- `metrics`: task-specific metric dictionary
- `details`: diagnostic details for deeper analysis
- `error`: error text for failed runs, otherwise `null`

## Training-overlap interpretation

{TRAINING_OVERLAP_WARNING}

The leaderboard appends `data_overlap_status`, `task_training_status`,
`zero_shot_status`, `overlap_reason_codes`, and
`overlap_relationship_registry_revision` after all score, provenance, and
operational fields. `Reviewed zero-shot` means the strict combined reviewed
status; unknown rows remain visible and are not treated as zero-shot.

## Leaderboard Provenance

`leaderboards/latest.csv` keeps every public row, including historical duplicate
runs for the same `task_id` and `model_id`. The first columns remain compatible
with older CSV readers, and provenance columns are appended:

- `evidence_tier`: `legacy`, `smoke`, `benchmark`, `fixture`, or `unknown`
- `evidence_source`: legacy source file, git sha, or run id when available
- `task_model_duplicate_count`: rows kept for the same task/model pair
- `task_model_run_rank`: 1-based order for that task/model pair
- `is_latest_for_task_model`: `true` for the latest exported row in that pair

Optional operational-evidence columns are projected only from explicit public
result fields: `run_started_at`, `run_finished_at`, `dataset_version`,
`input_count_total`, `token_usage`, `provider_latency_ms`, `cost_usd`,
`fresh_provider_calls`, and `cache_enabled`. Missing values remain blank; in
particular, missing cost is not converted to zero or estimated.

Latest markers are computed from the order of `results/latest.jsonl` when result
records are available, otherwise from CSV row order. Use
`is_latest_for_task_model=true` to inspect one current row per task/model pair
without losing the full historical trail.

## Usage

Install and inspect the registry:

```bash
uv sync
uv run modern-embed-bench benchmark models
uv run modern-embed-bench benchmark tasks
```

Run a small OpenAI smoke benchmark:

```bash
uv run modern-embed-bench benchmark run \\
  --manifest benchmark/runs/openai-smoke.yaml \\
  --output results/openai-smoke.jsonl \\
  --overwrite

uv run modern-embed-bench benchmark leaderboard \\
  --results results/openai-smoke.jsonl \\
  --output results/openai-smoke-leaderboard.csv
```

## Notes and Limitations

- Rows imported from legacy runs are published for continuity and should be read
  as historical baseline evidence, not as a fully normalized one-shot run.
- Explicitly unpublished runs are excluded from public result and leaderboard
  artifacts; historical records without a publication field remain public.
- Scores are task-specific. Avoid comparing scores across tasks as if they were
  one global ranking.
- Duration and provider latency are run observations, not comparable performance
  metrics across providers, routes, hardware, batching, task sizes, or cache
  states. Token and input counts describe workload, not model quality.
- No price, throughput, efficiency, energy, or CO2 values are inferred from the
  optional operational evidence.
- Some preview or private-in-progress model results are intentionally excluded
  from the public export until they are ready for publication.
- Image binaries are not bundled by default; `cross_modal` metadata is included
  separately from the source image files.
""",
        encoding="utf-8",
    )


def _write_export_manifest(path: Path, *, kind: str, files: list[str], metadata: dict[str, Any]) -> None:
    _write_yaml(
        path,
        {
            "kind": kind,
            "files": sorted(files),
            "metadata": metadata,
        },
    )


def export_dataset_repo(
    *,
    output_dir: str | Path = DEFAULT_EXPORT_ROOT / "dataset",
    benchmark_root: str | Path | None = None,
    results_path: str | Path | None = None,
    leaderboard_path: str | Path | None = None,
    include_data: bool = False,
    include_images: bool = False,
    clean: bool = True,
) -> Path:
    """Create a Hugging Face Dataset repo folder."""
    output = Path(output_dir)
    _reset_dir(output, clean=clean)
    catalog = load_catalog(benchmark_root)

    public_models = [model for model in catalog.models.values() if _is_public_model(model)]
    private_model_ids = {model.id for model in catalog.models.values() if not _is_public_model(model)}
    public_tasks = [task for task in catalog.tasks.values() if _is_public_task(task)]
    private_task_ids = {
        task.id for task in catalog.tasks.values() if not task.publish or not task.leaderboard_publish
    }
    result_records: list[dict[str, Any]] = []
    public_records: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []

    _write_jsonl(output / "models.jsonl", _dataset_model_catalog_rows(catalog))
    _write_jsonl(output / "tasks.jsonl", _dataset_task_catalog_rows(catalog))

    run_dir = catalog.root / "runs"
    if run_dir.exists():
        for run_file in sorted(run_dir.glob("*.yaml")):
            run_manifest = load_run_manifest(run_file)
            if run_manifest.publish and all(run_task.id not in private_task_ids for run_task in run_manifest.tasks):
                _copy_if_exists(run_file, output / "runs" / run_file.name)

    if results_path:
        result_src = Path(results_path)
        result_records = load_jsonl(result_src)
        public_records = _public_records(result_records, private_model_ids, private_task_ids)
        relationship_registry = load_relationship_registry(catalog.root / "training_overlap_relationships.yaml")
        for public_record in public_records:
            validate_assessment_registry_binding(public_record["training_overlap"], relationship_registry)
        _write_jsonl(output / "results" / "latest.jsonl", public_records)
        _write_jsonl(output / "results" / "latest-successful.jsonl", [r for r in public_records if not r.get("error")])

    if leaderboard_path:
        rows, fieldnames = _read_leaderboard_csv(Path(leaderboard_path))
        leaderboard_rows = _public_leaderboard_rows(
            rows,
            private_model_ids,
            private_task_ids,
            result_records=result_records,
        )
        leaderboard_rows = _enrich_leaderboard_rows(leaderboard_rows, result_records=public_records, catalog=catalog)
        _write_leaderboard_csv_rows(
            output / "leaderboards" / "latest.csv",
            leaderboard_rows,
            _leaderboard_fieldnames_with_provenance(fieldnames),
        )
    elif results_path:
        leaderboard_rows = build_leaderboard(public_records, catalog)
        leaderboard_rows = _enrich_leaderboard_rows(leaderboard_rows, result_records=public_records, catalog=catalog)
        _write_leaderboard_csv_rows(
            output / "leaderboards" / "latest.csv",
            leaderboard_rows,
            LEADERBOARD_FIELDNAMES,
        )

    if include_data:
        _copy_benchmark_data(Path("data"), output / "benchmark_data", include_images=include_images)
    else:
        (output / "benchmark_data").mkdir(parents=True, exist_ok=True)
        (output / "benchmark_data" / "README.md").write_text(
            "Benchmark input data was not exported. Re-run with --include-data to bundle JSONL data.\n",
            encoding="utf-8",
        )

    _write_dataset_card(
        output / "README.md",
        catalog=catalog,
        public_models=public_models,
        public_tasks=public_tasks,
        excluded_private_models=len(private_model_ids),
        result_stats=_result_stats(public_records),
        leaderboard_stats=_leaderboard_stats(leaderboard_rows),
        include_data=include_data,
    )

    files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]
    _write_export_manifest(
        output / "export_manifest.yaml",
        kind="hf_dataset",
        files=files,
        metadata={
            "models": len(public_models),
            "registry_models": len(catalog.models),
            "tasks": len(catalog.tasks),
            "excluded_private_models": len(private_model_ids),
            "include_data": include_data,
            "include_images": include_images,
            "leaderboard": _leaderboard_stats(leaderboard_rows),
        },
    )
    return output


def _copy_benchmark_data(data_root: Path, output: Path, *, include_images: bool) -> None:
    if not data_root.exists():
        output.mkdir(parents=True, exist_ok=True)
        (output / "README.md").write_text("Local data directory was not found.\n", encoding="utf-8")
        return

    for path in sorted(data_root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(data_root)
        if _should_skip_data_file(rel, include_images=include_images):
            continue
        _copy_if_exists(path, output / rel)

    note = [
        "# Benchmark Data Export",
        "",
        "This folder mirrors selected local benchmark input data.",
        "",
        "Embedding caches, tool caches, numpy arrays, and temporary files are intentionally skipped.",
    ]
    if not include_images:
        note.append("Image files were skipped. Re-run with `--include-images` to bundle image assets.")
    (output / "README.md").write_text("\n".join(note) + "\n", encoding="utf-8")


def _should_skip_data_file(rel: Path, *, include_images: bool) -> bool:
    parts = set(rel.parts)
    suffix = rel.suffix.lower()
    if "embedding_cache" in parts or ".cache" in parts:
        return True
    if suffix in {".npy", ".npz", ".tmp", ".pyc"}:
        return True
    if any(part == "__pycache__" for part in rel.parts):
        return True
    if suffix in {".jpg", ".jpeg", ".png", ".webp"} and not include_images:
        return True
    return False


def export_space_repo(
    *,
    output_dir: str | Path = DEFAULT_EXPORT_ROOT / "space",
    dataset_repo_id: str | None = None,
    bundled_leaderboard: str | Path | None = None,
    benchmark_root: str | Path | None = None,
    clean: bool = True,
) -> Path:
    """Create a Hugging Face Gradio Space folder."""
    output = Path(output_dir)
    _reset_dir(output, clean=clean)
    catalog = load_catalog(benchmark_root)
    public_model_rows = _space_model_catalog_rows(catalog)
    public_task_rows = _space_task_catalog_rows(catalog)

    (output / "README.md").write_text(_space_readme(dataset_repo_id), encoding="utf-8")
    (output / "requirements.txt").write_text("gradio>=5.0\npandas>=2.0\nhuggingface_hub>=0.30\n", encoding="utf-8")
    (output / "app.py").write_text(_space_app_source(dataset_repo_id), encoding="utf-8")
    _write_jsonl(output / "models.jsonl", public_model_rows)
    _write_jsonl(output / "tasks.jsonl", public_task_rows)

    if bundled_leaderboard:
        rows, fieldnames = _read_leaderboard_csv(Path(bundled_leaderboard))
        private_task_ids = {
            task.id for task in catalog.tasks.values() if not _is_public_task(task)
        }
        public_rows = _public_leaderboard_rows(rows, set(), private_task_ids)
        _write_leaderboard_csv_rows(
            output / "leaderboard.csv",
            public_rows,
            _leaderboard_fieldnames_with_provenance(fieldnames),
        )
    else:
        _write_empty_leaderboard(output / "leaderboard.csv")

    files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]
    _write_export_manifest(
        output / "export_manifest.yaml",
        kind="hf_space",
        files=files,
        metadata={
            "dataset_repo_id": dataset_repo_id,
            "declared_public_models": len(public_model_rows),
            "declared_public_tasks": len(public_task_rows),
        },
    )
    return output


def _write_empty_leaderboard(path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LEADERBOARD_FIELDNAMES)


def _space_readme(dataset_repo_id: str | None) -> str:
    dataset_line = f"DATASET_REPO_ID: {dataset_repo_id}" if dataset_repo_id else "DATASET_REPO_ID: optional"
    return f"""---
title: Modern Embedding Bench
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Modern Embedding Bench Leaderboard

This Gradio Space renders task-specific leaderboard views from a benchmark Dataset repo.

{dataset_line}
"""


def _space_app_source(dataset_repo_id: str | None) -> str:
    default_repo = dataset_repo_id or ""
    source = '''from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd

DEFAULT_DATASET_REPO_ID = __DATASET_REPO_ID__
LEADERBOARD_FILE = "leaderboards/latest.csv"
MODEL_CATALOG_FILE = "models.jsonl"
TASK_CATALOG_FILE = "tasks.jsonl"
STATIC_TASK_DETAILS = __TASK_DETAILS__
PUBLIC_EXCLUDED_PROVIDERS = {"geevec_api", "geevec_lite"}
PUBLIC_EXCLUDED_MARKERS = ("geevec",)
TRAINING_OVERLAP_WARNING = __TRAINING_OVERLAP_WARNING__
ALL_DATA_OVERLAP_STATUSES = "All data-overlap statuses"
ALL_TASK_TRAINING_STATUSES = "All task-training statuses"
REVIEWED_ZERO_SHOT_ONLY = "Reviewed zero-shot only"
ALL_TASK_SOURCE_DISCLOSURES = "All evaluation-source disclosures"
ALL_CATALOG_REVIEW_STATES = "All catalog review states"
OVERLAP_STATUS_LABELS = {
    "exact": "Known exact training overlap",
    "adapted": "Known adapted training overlap",
    "same_task": "Same-task training exposure",
    "similar_task": "Similar-task training exposure",
    "declared_none": "Reviewed no declared overlap",
    "unknown": "Unknown - no zero-shot claim",
}


def dataset_file(filename, bundled_filename, label):
    dataset_repo_id = os.environ.get("DATASET_REPO_ID") or DEFAULT_DATASET_REPO_ID
    local_path = Path(bundled_filename)
    if dataset_repo_id:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=dataset_repo_id,
                repo_type="dataset",
                filename=filename,
            )
            local_path = Path(downloaded)
        except Exception as exc:
            print("Could not load {} from {}: {}. Falling back to bundled data.".format(label, dataset_repo_id, exc))
    if not local_path.exists():
        return None
    return local_path


def load_rows():
    local_path = dataset_file(LEADERBOARD_FILE, "leaderboard.csv", "leaderboard")
    if local_path is None:
        return []
    with open(local_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def enabled(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalized_catalog_status(value, allowed, default):
    status = str(value or "").strip().lower()
    return status if status in allowed else default


def bounded_unique_count(values, key=None, limit=100):
    if not isinstance(values, list):
        return 0
    unique = set()
    for value in values:
        if key is not None:
            value = value.get(key) if isinstance(value, dict) else None
        text = str(value or "").strip()
        if not text:
            continue
        unique.add(text)
        if len(unique) > limit:
            return "{}+".format(limit)
    return len(unique)


def catalog_text(value, default="unknown"):
    text = str(value or "").strip()
    return text or default


def contains_excluded_marker(*values):
    for value in values:
        text = str(value or "").lower()
        if any(marker in text for marker in PUBLIC_EXCLUDED_MARKERS):
            return True
    return False


def load_model_specs():
    local_path = dataset_file(MODEL_CATALOG_FILE, "models.jsonl", "model catalog")
    if local_path is None:
        return []
    models = {}
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                model = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(model, dict) or not model.get("id"):
                continue
            provider = str(model.get("provider") or "")
            if not enabled(model.get("publish"), True):
                continue
            if provider.lower() in PUBLIC_EXCLUDED_PROVIDERS:
                continue
            if contains_excluded_marker(model.get("id"), model.get("display_name"), provider):
                continue
            modalities = model.get("modalities") or []
            if not isinstance(modalities, list):
                modalities = []
            model_id = str(model["id"])
            training = model.get("training_data") if isinstance(model.get("training_data"), dict) else {}
            models[model_id] = {
                "id": model_id,
                "display_name": str(model.get("display_name") or model_id),
                "provider": provider,
                "modalities": [str(value) for value in modalities if str(value).strip()],
                "dimensions": model.get("dimensions"),
                "max_text_length": model.get("max_text_length"),
                "supports_mrl": enabled(model.get("supports_mrl"), False),
                "access": str(model.get("access") or "unknown"),
                "status": str(model.get("status") or "unknown"),
                "source": str(model.get("source") or ""),
                "training_data": {
                    "disclosure": normalized_catalog_status(
                        training.get("disclosure"), {"unknown", "partial", "complete"}, "unknown"
                    ),
                    "lineage_disclosure": normalized_catalog_status(
                        training.get("lineage_disclosure"), {"unknown", "complete"}, "unknown"
                    ),
                    "model_revision": catalog_text(training.get("model_revision")),
                    "source_ids": [str(value) for value in training.get("source_ids") or []],
                    "adapted_from": [str(value) for value in training.get("adapted_from") or []],
                    "evidence_revision": training.get("evidence_revision"),
                    "evidence_urls": [str(value) for value in training.get("evidence_urls") or []],
                    "review_state": normalized_catalog_status(
                        training.get("review_state"), {"pending", "approved", "rejected"}, "pending"
                    ),
                },
            }
    return list(models.values())


def load_task_specs():
    local_path = dataset_file(TASK_CATALOG_FILE, "tasks.jsonl", "task catalog")
    if local_path is None:
        return []
    tasks = []
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(task, dict) or not task.get("id"):
                continue
            if not enabled(task.get("publish"), True):
                continue
            if not enabled(task.get("leaderboard_publish"), enabled(task.get("publish"), True)):
                continue
            evaluation = task.get("evaluation_sources") if isinstance(task.get("evaluation_sources"), dict) else {}
            required_modalities = task.get("required_modalities") or []
            if not isinstance(required_modalities, list):
                required_modalities = []
            tasks.append({
                "id": str(task["id"]),
                "display_name": str(task.get("display_name") or task["id"]),
                "description": str(task.get("description") or ""),
                "required_modalities": [
                    str(value) for value in required_modalities if str(value).strip()
                ],
                "primary_metric": str(task.get("primary_metric") or ""),
                "metric_direction": str(task.get("metric_direction") or "higher"),
                "dataset_version": str(task.get("dataset_version") or "unknown"),
                "publish": True,
                "leaderboard_publish": True,
                "evaluation_sources": {
                    "disclosure": normalized_catalog_status(
                        evaluation.get("disclosure"), {"unknown", "complete"}, "unknown"
                    ),
                    "sources": [
                        {
                            "source_id": str(source.get("source_id") or ""),
                            "source_revision": source.get("source_revision"),
                            "config": source.get("config"),
                            "split": source.get("split"),
                            "transformation_id": source.get("transformation_id"),
                        }
                        for source in evaluation.get("sources") or []
                        if isinstance(source, dict)
                    ],
                    "evidence_revision": evaluation.get("evidence_revision"),
                    "evidence_urls": [str(value) for value in evaluation.get("evidence_urls") or []],
                    "review_state": normalized_catalog_status(
                        evaluation.get("review_state"), {"pending", "approved", "rejected"}, "pending"
                    ),
                },
            })
    return tasks


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truthy(value):
    return str(value or "").lower() in {"1", "true", "yes", "y"}


def task_model_key(row):
    return (row.get("task_id") or "", row.get("model_id") or row.get("model") or "")


def coverage_model_key(row):
    return (row.get("model_id") or row.get("model") or "", row.get("provider") or "")


def evidence_tier_value(row):
    return str(row.get("evidence_tier") or "").strip() or "unknown"


def evidence_summary():
    tiers = {}
    for row in ROWS:
        tier = evidence_tier_value(row)
        tiers[tier] = tiers.get(tier, 0) + 1
    if not tiers:
        return "none"
    return ", ".join("{}={}".format(key, tiers[key]) for key in sorted(tiers))


TASK_SPECS = load_task_specs()
MODEL_SPECS = load_model_specs()
PUBLIC_TASK_IDS = {str(task.get("id") or "") for task in TASK_SPECS if task.get("id")}
RAW_ROWS = load_rows()
ROWS = [row for row in RAW_ROWS if row.get("task_id") in PUBLIC_TASK_IDS] if PUBLIC_TASK_IDS else RAW_ROWS
ALL_PROVIDERS = "All providers"
ALL_MODEL_PROVIDERS = "All catalog providers"
ALL_EVIDENCE_TIERS = "All evidence tiers"
NOT_EVALUATED = "not evaluated"
TASKS = sorted(
    {
        row.get("task_id", "")
        for row in ROWS
        if row.get("task_id") and as_float(row.get("score")) is not None
    }
)
DECLARED_TASKS = sorted(PUBLIC_TASK_IDS) if PUBLIC_TASK_IDS else TASKS
PROVIDERS = [ALL_PROVIDERS] + sorted({row.get("provider", "") for row in ROWS if row.get("provider")})
MODEL_PROVIDERS = [ALL_MODEL_PROVIDERS] + sorted(
    {model.get("provider", "") for model in MODEL_SPECS if model.get("provider")}
)
TASK_SOURCE_DISCLOSURES = [ALL_TASK_SOURCE_DISCLOSURES] + sorted(
    {
        task.get("evaluation_sources", {}).get("disclosure", "unknown")
        for task in TASK_SPECS
    }
)
CATALOG_REVIEW_STATES = [ALL_CATALOG_REVIEW_STATES] + sorted(
    {
        task.get("evaluation_sources", {}).get("review_state", "pending")
        for task in TASK_SPECS
    }
)
EVIDENCE_TIERS = [ALL_EVIDENCE_TIERS] + sorted({evidence_tier_value(row) for row in ROWS})
DATA_OVERLAP_STATUSES = [ALL_DATA_OVERLAP_STATUSES] + sorted(
    {str(row.get("data_overlap_status") or "unknown") for row in ROWS}
)
TASK_TRAINING_STATUSES = [ALL_TASK_TRAINING_STATUSES] + sorted(
    {str(row.get("task_training_status") or "unknown") for row in ROWS}
)
LATEST_MARKERS_AVAILABLE = any(str(row.get("is_latest_for_task_model") or "").strip() for row in ROWS)
DEFAULT_LATEST_ONLY = LATEST_MARKERS_AVAILABLE
PROVENANCE_COLUMNS = [
    "evidence_tier",
    "evidence_source",
    "task_model_duplicate_count",
    "task_model_run_rank",
    "is_latest_for_task_model",
    "data_overlap_status",
    "task_training_status",
    "zero_shot_status",
    "overlap_reason_codes",
    "overlap_relationship_registry_revision",
]
OPERATIONAL_COLUMNS = [
    "task_id",
    "model",
    "model_id",
    "provider",
    "evidence_tier",
    "run_id",
    "evidence_source",
    "run_started_at",
    "run_finished_at",
    "dataset_version",
    "duration_s",
    "provider_latency_ms",
    "input_count_total",
    "token_usage",
    "cost_usd",
    "fresh_provider_calls",
    "cache_enabled",
    "is_latest_for_task_model",
]
OPERATIONAL_DETAIL_FIELDS = [
    "provider_latency_ms",
    "input_count_total",
    "token_usage",
    "cost_usd",
    "fresh_provider_calls",
    "cache_enabled",
]


def build_task_details():
    details = {}
    for task in TASK_SPECS:
        task_id = str(task.get("id") or "")
        if not task_id:
            continue
        details[task_id] = {
            "label": task.get("display_name") or task_id,
            "summary": task.get("description") or "",
            "metric": task.get("primary_metric") or "",
        }
    for task_id, static_details in STATIC_TASK_DETAILS.items():
        if task_id in details or task_id in TASKS:
            details.setdefault(task_id, {}).update(static_details)
    return details


TASK_DETAILS = build_task_details()


def summary_markdown():
    providers = len({row.get("provider") for row in ROWS if row.get("provider")})
    models = len({row.get("model_id") or row.get("model") for row in ROWS if row.get("model_id") or row.get("model")})
    group_counts = {}
    for row in ROWS:
        if row.get("task_id"):
            key = task_model_key(row)
            group_counts[key] = group_counts.get(key, 0) + 1
    task_model_pairs = len(group_counts)
    duplicate_repeats = sum(max(count - 1, 0) for count in group_counts.values())
    latest_rows = sum(1 for row in ROWS if truthy(row.get("is_latest_for_task_model")))
    latest_note = str(latest_rows) if LATEST_MARKERS_AVAILABLE else "unavailable"
    return (
        "Rows: **{}** | Declared public models: **{}** | Model identities with score rows: **{}** | "
        "Declared public tasks: **{}** | Tasks with score rows: **{}** | Providers with score rows: **{}**  \\n"
        "Task/model pairs: **{}** | Duplicate repeats: **{}** | Latest markers: **{}** | Evidence: **{}**"
    ).format(
        len(ROWS),
        len(MODEL_SPECS),
        models,
        len(DECLARED_TASKS),
        len(TASKS),
        providers,
        task_model_pairs,
        duplicate_repeats,
        latest_note,
        evidence_summary(),
    )


def task_markdown(task_id):
    details = TASK_DETAILS.get(task_id or "", {})
    if not details:
        return "Scores are task-specific and should not be averaged into a global ranking.  \\n{}".format(
            TRAINING_OVERLAP_WARNING
        )
    return "**{}**  \\n{}  \\nPrimary signal: `{}`  \\n{}".format(
        details.get("label", task_id),
        details.get("summary", ""),
        details.get("metric", ""),
        TRAINING_OVERLAP_WARNING,
    )


def filtered_rows(
    task_id,
    provider,
    evidence_tier,
    query,
    latest_only,
    data_overlap_status=ALL_DATA_OVERLAP_STATUSES,
    task_training_status=ALL_TASK_TRAINING_STATUSES,
    reviewed_zero_shot_only=False,
):
    filtered = []
    for source_row in ROWS:
        if source_row.get("task_id") != task_id:
            continue
        row = source_row.copy()
        row["evidence_tier"] = evidence_tier_value(row)
        filtered.append(row)
    if provider != ALL_PROVIDERS:
        filtered = [row for row in filtered if row.get("provider") == provider]
    if evidence_tier != ALL_EVIDENCE_TIERS:
        filtered = [row for row in filtered if row.get("evidence_tier") == evidence_tier]
    if data_overlap_status != ALL_DATA_OVERLAP_STATUSES:
        filtered = [
            row for row in filtered if str(row.get("data_overlap_status") or "unknown") == data_overlap_status
        ]
    if task_training_status != ALL_TASK_TRAINING_STATUSES:
        filtered = [
            row for row in filtered if str(row.get("task_training_status") or "unknown") == task_training_status
        ]
    if reviewed_zero_shot_only:
        filtered = [row for row in filtered if row.get("zero_shot_status") == "reviewed_yes"]
    if latest_only and LATEST_MARKERS_AVAILABLE:
        filtered = [row for row in filtered if truthy(row.get("is_latest_for_task_model"))]
    query = (query or "").strip().lower()
    if query:
        filtered = [
            row
            for row in filtered
            if query in " ".join(
                str(row.get(key) or "").lower()
                for key in (
                    "model",
                    "model_id",
                    "provider",
                    "run_id",
                    "primary_metric",
                    "evidence_tier",
                    "evidence_source",
                    "data_overlap_status",
                    "task_training_status",
                    "zero_shot_status",
                    "overlap_reason_codes",
                )
            )
        ]
    filtered.sort(
        key=lambda row: as_float(row.get("score")) if as_float(row.get("score")) is not None else float("-inf"),
        reverse=True,
    )
    return filtered


def coverage_source_rows():
    candidates = [
        row
        for row in ROWS
        if as_float(row.get("score")) is not None
        and (not LATEST_MARKERS_AVAILABLE or truthy(row.get("is_latest_for_task_model")))
    ]
    selected = {}
    for source_row in candidates:
        task_id, model_id = task_model_key(source_row)
        if not task_id or not model_id:
            continue
        row = source_row.copy()
        row["evidence_tier"] = evidence_tier_value(row)
        selected[(task_id,) + coverage_model_key(row)] = row
    return list(selected.values())


COVERAGE_SOURCE_ROWS = coverage_source_rows()
COVERAGE_EVIDENCE_TIERS = [ALL_EVIDENCE_TIERS] + sorted(
    {evidence_tier_value(row) for row in COVERAGE_SOURCE_ROWS}
)


def coverage_model_rows():
    grouped = {}
    for row in COVERAGE_SOURCE_ROWS:
        model_id, provider = coverage_model_key(row)
        key = (model_id, provider)
        if key not in grouped:
            grouped[key] = {
                "model": row.get("model") or model_id,
                "model_id": model_id,
                "provider": provider,
                "tasks": {},
            }
        grouped[key]["tasks"][row.get("task_id")] = evidence_tier_value(row)
    return list(grouped.values())


def filtered_coverage_rows(provider, evidence_tier, query):
    filtered = []
    query = (query or "").strip().lower()
    for group in coverage_model_rows():
        if provider != ALL_PROVIDERS and group["provider"] != provider:
            continue
        tiers = set(group["tasks"].values())
        if evidence_tier != ALL_EVIDENCE_TIERS and evidence_tier not in tiers:
            continue
        search_values = [group["model"], group["model_id"], group["provider"]]
        search_values.extend(
            "{} {}".format(task_id, tier) for task_id, tier in sorted(group["tasks"].items())
        )
        if query and query not in " ".join(str(value or "").lower() for value in search_values):
            continue
        visible = {
            "model": group["model"],
            "provider": group["provider"],
            "covered_tasks": len(group["tasks"]),
        }
        for task_id in DECLARED_TASKS:
            tier = group["tasks"].get(task_id)
            visible[task_id] = "evaluated ({})".format(tier) if tier else NOT_EVALUATED
        filtered.append(visible)
    filtered.sort(key=lambda row: (row["model"].lower(), row["provider"].lower()))
    return filtered


def coverage_table(provider, evidence_tier, query):
    columns = ["model", "provider", "covered_tasks"] + DECLARED_TASKS
    return pd.DataFrame(filtered_coverage_rows(provider, evidence_tier, query), columns=columns)


def coverage_markdown(provider, evidence_tier, query, matching_count):
    source_note = (
        "current task/model markers"
        if LATEST_MARKERS_AVAILABLE
        else "the last available row per task/model because current markers are unavailable"
    )
    filters = []
    if provider != ALL_PROVIDERS:
        filters.append("provider={}".format(provider))
    if evidence_tier != ALL_EVIDENCE_TIERS:
        filters.append("at least one current cell with evidence={}".format(evidence_tier))
    if (query or "").strip():
        filters.append('search="{}"'.format((query or "").strip()))
    filter_note = " | ".join(filters) if filters else "no provider, evidence, or search filter"
    return (
        "Coverage columns come from the public task catalog, while cell status is derived only from **{}**; "
        "task declaration alone is not evaluation evidence, and scores are neither compared nor averaged across tasks. "
        "Missing cells are labeled **{}**. Evidence tiers describe the evidence behind a present cell, not model "
        "quality. Rows are alphabetized by model, not ranked by coverage or performance.  \\n"
        "Showing **{}** model/provider rows | View: **{}**"
    ).format(source_note, NOT_EVALUATED, matching_count, filter_note)


def render_coverage(provider, evidence_tier, query):
    table = coverage_table(provider, evidence_tier, query)
    return coverage_markdown(provider, evidence_tier, query, len(table.index)), table


def model_ids_with_score_rows():
    return {
        str(row.get("model_id") or "")
        for row in ROWS
        if row.get("model_id") and as_float(row.get("score")) is not None
    }


def task_ids_with_score_rows():
    return {
        str(row.get("task_id") or "")
        for row in ROWS
        if row.get("task_id") and as_float(row.get("score")) is not None
    }


def filtered_task_catalog_rows(source_disclosure, review_state, query):
    evaluated_task_ids = task_ids_with_score_rows()
    query = (query or "").strip().lower()
    filtered = []
    for source_task in TASK_SPECS:
        evaluation = source_task.get("evaluation_sources") or {}
        if (
            source_disclosure != ALL_TASK_SOURCE_DISCLOSURES
            and evaluation.get("disclosure") != source_disclosure
        ):
            continue
        if review_state != ALL_CATALOG_REVIEW_STATES and evaluation.get("review_state") != review_state:
            continue
        task = {
            "display_name": source_task.get("display_name"),
            "id": source_task.get("id"),
            "description": source_task.get("description"),
            "required_modalities": ", ".join(source_task.get("required_modalities") or []),
            "primary_metric": source_task.get("primary_metric"),
            "metric_direction": source_task.get("metric_direction"),
            "dataset_version": source_task.get("dataset_version"),
            "evaluation_source_disclosure": evaluation.get("disclosure", "unknown"),
            "review_state": evaluation.get("review_state", "pending"),
            "source_count": bounded_unique_count(evaluation.get("sources"), key="source_id"),
            "declaration": "declared in public task catalog",
            "evaluation_evidence": (
                "public score rows available"
                if source_task.get("id") in evaluated_task_ids
                else "declared only - no public score rows"
            ),
        }
        if query and query not in " ".join(str(value or "").lower() for value in task.values()):
            continue
        filtered.append(task)
    filtered.sort(
        key=lambda task: (
            str(task.get("display_name") or "").lower(),
            str(task.get("id") or "").lower(),
        )
    )
    return filtered


def task_catalog_table(source_disclosure, review_state, query):
    columns = [
        "display_name",
        "id",
        "description",
        "required_modalities",
        "primary_metric",
        "metric_direction",
        "dataset_version",
        "evaluation_source_disclosure",
        "review_state",
        "source_count",
        "declaration",
        "evaluation_evidence",
    ]
    return pd.DataFrame(filtered_task_catalog_rows(source_disclosure, review_state, query), columns=columns)


def task_catalog_markdown(source_disclosure, review_state, query, matching_count):
    filters = []
    if source_disclosure != ALL_TASK_SOURCE_DISCLOSURES:
        filters.append("evaluation source disclosure={}".format(source_disclosure))
    if review_state != ALL_CATALOG_REVIEW_STATES:
        filters.append("review state={}".format(review_state))
    if (query or "").strip():
        filters.append('search="{}"'.format((query or "").strip()))
    filter_note = " | ".join(filters) if filters else "no disclosure, review, or search filter"
    return (
        "This is an alphabetical catalog of public task registry declarations, **not a task ranking**. "
        "`public score rows available` means at least one numeric public leaderboard row has the same declared "
        "task id; it does not imply task quality, a global ranking, current coverage, reviewed source completeness, "
        "zero-shot status, or absence of training overlap. Evaluation-source disclosure, review state, and bounded "
        "source count are declared metadata only. Unknown or pending means evidence may be missing, empty, "
        "incomplete, unresolved, or stale; it is not a reviewed negative claim.  \\n"
        "Showing **{}** declared tasks | View: **{}**"
    ).format(matching_count, filter_note)


def render_task_catalog(source_disclosure, review_state, query):
    table = task_catalog_table(source_disclosure, review_state, query)
    return task_catalog_markdown(source_disclosure, review_state, query, len(table.index)), table


def filtered_model_catalog_rows(provider, query):
    evaluated_model_ids = model_ids_with_score_rows()
    query = (query or "").strip().lower()
    filtered = []
    for source_model in MODEL_SPECS:
        if provider != ALL_MODEL_PROVIDERS and source_model.get("provider") != provider:
            continue
        model = source_model.copy()
        model["modalities"] = ", ".join(model.get("modalities") or [])
        model["supports_mrl"] = "yes" if model.get("supports_mrl") else "no"
        training = model.pop("training_data", {})
        model["training_disclosure"] = training.get("disclosure", "unknown")
        model["lineage_disclosure"] = training.get("lineage_disclosure", "unknown")
        model["review_state"] = training.get("review_state", "pending")
        model["model_revision"] = catalog_text(training.get("model_revision"))
        model["training_source_count"] = bounded_unique_count(training.get("source_ids"))
        model["lineage_parent_count"] = bounded_unique_count(training.get("adapted_from"))
        model["declaration"] = "declared in public model catalog"
        model["evaluation_evidence"] = (
            "public score rows available"
            if model.get("id") in evaluated_model_ids
            else "declared only - no public score rows"
        )
        if query and query not in " ".join(str(value or "").lower() for value in model.values()):
            continue
        filtered.append(model)
    filtered.sort(
        key=lambda model: (
            str(model.get("display_name") or "").lower(),
            str(model.get("provider") or "").lower(),
            str(model.get("id") or "").lower(),
        )
    )
    return filtered


def model_catalog_table(provider, query):
    columns = [
        "display_name",
        "id",
        "provider",
        "modalities",
        "dimensions",
        "max_text_length",
        "supports_mrl",
        "access",
        "status",
        "training_disclosure",
        "lineage_disclosure",
        "review_state",
        "model_revision",
        "training_source_count",
        "lineage_parent_count",
        "declaration",
        "evaluation_evidence",
        "source",
    ]
    return pd.DataFrame(filtered_model_catalog_rows(provider, query), columns=columns)


def model_catalog_markdown(provider, query, matching_count):
    filters = []
    if provider != ALL_MODEL_PROVIDERS:
        filters.append("provider={}".format(provider))
    if (query or "").strip():
        filters.append('search="{}"'.format((query or "").strip()))
    filter_note = " | ".join(filters) if filters else "no provider or search filter"
    return (
        "This is an alphabetical catalog of public registry declarations, **not a model ranking**. "
        "`public score rows available` means at least one numeric public leaderboard row has the same declared "
        "model id; it does not imply quality, complete task coverage, recency, or account availability. Registry "
        "status, dimensions, maximum text length, MRL support, access, training disclosure, lineage disclosure, "
        "review state, revision, and bounded counts are declared metadata only. Complete or approved declarations "
        "do not imply zero-shot evaluation or absence of training overlap. Unknown or pending means evidence may be "
        "missing, empty, incomplete, unresolved, or stale; it is not a reviewed negative claim. Score-only "
        "legacy identities remain available in the Task leaderboard and Coverage views and are not promoted to "
        "registry declarations here.  \\n"
        "Showing **{}** declared models | View: **{}**"
    ).format(matching_count, filter_note)


def render_model_catalog(provider, query):
    table = model_catalog_table(provider, query)
    return model_catalog_markdown(provider, query, len(table.index)), table


def reported(row, key):
    value = row.get(key)
    return value is not None and str(value).strip() != ""


def operational_filtered_rows(provider, evidence_tier, query, latest_only):
    filtered = []
    query = (query or "").strip().lower()
    for source_row in ROWS:
        row = source_row.copy()
        row["evidence_tier"] = evidence_tier_value(row)
        if provider != ALL_PROVIDERS and row.get("provider") != provider:
            continue
        if evidence_tier != ALL_EVIDENCE_TIERS and row.get("evidence_tier") != evidence_tier:
            continue
        if latest_only and LATEST_MARKERS_AVAILABLE and not truthy(row.get("is_latest_for_task_model")):
            continue
        if query and query not in " ".join(
            str(row.get(key) or "").lower()
            for key in (
                "task_id",
                "task",
                "model",
                "model_id",
                "provider",
                "evidence_tier",
                "run_id",
                "evidence_source",
                "dataset_version",
            )
        ):
            continue
        filtered.append(row)
    filtered.sort(
        key=lambda row: (
            str(row.get("task_id") or "").lower(),
            str(row.get("model") or row.get("model_id") or "").lower(),
            str(row.get("provider") or "").lower(),
            str(row.get("run_started_at") or ""),
            str(row.get("run_id") or ""),
        )
    )
    return filtered


def operational_table(provider, evidence_tier, query, latest_only, top_n):
    filtered = operational_filtered_rows(provider, evidence_tier, query, latest_only)
    return pd.DataFrame(filtered[: int(top_n or 50)], columns=OPERATIONAL_COLUMNS)


def operational_markdown(provider, evidence_tier, query, latest_only, matching_count, shown_count):
    filters = ["current marked rows only" if latest_only and LATEST_MARKERS_AVAILABLE else "all run rows"]
    if provider != ALL_PROVIDERS:
        filters.append("provider={}".format(provider))
    if evidence_tier != ALL_EVIDENCE_TIERS:
        filters.append("evidence={}".format(evidence_tier))
    if (query or "").strip():
        filters.append('search="{}"'.format((query or "").strip()))
    filtered = operational_filtered_rows(provider, evidence_tier, query, latest_only)
    detailed_count = sum(
        1
        for row in filtered
        if any(reported(row, field) for field in OPERATIONAL_DETAIL_FIELDS)
    )
    return (
        "This is **non-ranking per-run operational evidence**. Rows are alphabetized by task and model; no "
        "operational field affects leaderboard scores, sorting, or latest-row selection. Latency and duration are "
        "not comparable across providers, routes, hardware, batching, task sizes, or cache states. Token and input "
        "counts are workload evidence, not quality. Missing values mean unreported, not zero. Cost is shown only "
        "when explicitly reported; no price, throughput, normalized efficiency, energy, or CO2 value is inferred.  \\n"
        "Showing **{}** of **{}** matching rows; **{}** report at least one detailed operational field | View: **{}**"
    ).format(shown_count, matching_count, detailed_count, " | ".join(filters))


def render_operational(provider, evidence_tier, query, latest_only, top_n):
    filtered = operational_filtered_rows(provider, evidence_tier, query, latest_only)
    table = pd.DataFrame(filtered[: int(top_n or 50)], columns=OPERATIONAL_COLUMNS)
    note = operational_markdown(provider, evidence_tier, query, latest_only, len(filtered), len(table.index))
    return note, table


def table_from_rows(filtered, top_n):
    filtered = filtered[: int(top_n or 50)]
    for index, row in enumerate(filtered, start=1):
        row["rank"] = index
        score = as_float(row.get("score"))
        row["score"] = round(score, 6) if score is not None else row.get("score")
        if row.get("data_overlap_status") in {"exact", "adapted"}:
            status_key = row.get("data_overlap_status")
        elif row.get("task_training_status") in {"same_task", "similar_task"}:
            status_key = row.get("task_training_status")
        elif row.get("zero_shot_status") == "reviewed_yes":
            status_key = "declared_none"
        else:
            status_key = "unknown"
        row["overlap_interpretation"] = OVERLAP_STATUS_LABELS[status_key]
    columns = ["rank", "model", "provider", "score", "primary_metric", "run_id", "duration_s"]
    columns.extend(column for column in PROVENANCE_COLUMNS if any(row.get(column) for row in filtered))
    columns.append("overlap_interpretation")
    return pd.DataFrame(filtered, columns=columns)


def render_table(
    task_id,
    provider,
    evidence_tier,
    query,
    latest_only,
    top_n,
    data_overlap_status=ALL_DATA_OVERLAP_STATUSES,
    task_training_status=ALL_TASK_TRAINING_STATUSES,
    reviewed_zero_shot_only=False,
):
    return table_from_rows(
        filtered_rows(
            task_id,
            provider,
            evidence_tier,
            query,
            latest_only,
            data_overlap_status,
            task_training_status,
            reviewed_zero_shot_only,
        ),
        top_n,
    )


def view_markdown(
    provider,
    evidence_tier,
    query,
    latest_only,
    matching_count,
    shown_count,
    data_overlap_status=ALL_DATA_OVERLAP_STATUSES,
    task_training_status=ALL_TASK_TRAINING_STATUSES,
    reviewed_zero_shot_only=False,
):
    if latest_only and LATEST_MARKERS_AVAILABLE:
        history_state = "current marked rows only"
        history_hint = "Uncheck the current-row filter to inspect all historical rows."
    elif latest_only:
        history_state = "all historical rows; latest markers unavailable"
        history_hint = "This snapshot does not provide current-row markers."
    else:
        history_state = "all historical rows"
        history_hint = "Enable the current-row filter to return to one marked row per task/model pair."
    filters = [history_state]
    if provider != ALL_PROVIDERS:
        filters.append("provider={}".format(provider))
    if evidence_tier != ALL_EVIDENCE_TIERS:
        filters.append("evidence={}".format(evidence_tier))
    if data_overlap_status != ALL_DATA_OVERLAP_STATUSES:
        filters.append("data_overlap={}".format(data_overlap_status))
    if task_training_status != ALL_TASK_TRAINING_STATUSES:
        filters.append("task_training={}".format(task_training_status))
    if reviewed_zero_shot_only:
        filters.append(REVIEWED_ZERO_SHOT_ONLY)
    if (query or "").strip():
        filters.append('search="{}"'.format((query or "").strip()))
    if matching_count:
        status = "Showing **{}** of **{}** matching rows.".format(shown_count, matching_count)
    else:
        status = "No rows match the selected filters."
    return "{}  \\nView: **{}**  \\n{}".format(status, " | ".join(filters), history_hint)


def render(
    task_id,
    provider,
    evidence_tier,
    query,
    latest_only,
    top_n,
    data_overlap_status=ALL_DATA_OVERLAP_STATUSES,
    task_training_status=ALL_TASK_TRAINING_STATUSES,
    reviewed_zero_shot_only=False,
):
    if not task_id:
        return "No leaderboard rows are available.", pd.DataFrame()
    filtered = filtered_rows(
        task_id,
        provider,
        evidence_tier,
        query,
        latest_only,
        data_overlap_status,
        task_training_status,
        reviewed_zero_shot_only,
    )
    table = table_from_rows(filtered, top_n)
    note = view_markdown(
        provider,
        evidence_tier,
        query,
        latest_only,
        len(filtered),
        len(table.index),
        data_overlap_status,
        task_training_status,
        reviewed_zero_shot_only,
    )
    return "{}\\n\\n{}".format(task_markdown(task_id), note), table


def main():
    default_task = TASKS[0] if TASKS else None
    default_evidence_tier = ALL_EVIDENCE_TIERS
    initial_note, initial_table = (
        render(default_task, ALL_PROVIDERS, default_evidence_tier, "", DEFAULT_LATEST_ONLY, 30)
        if default_task
        else ("No leaderboard rows are available.", pd.DataFrame())
    )
    initial_coverage_note, initial_coverage_table = render_coverage(
        ALL_PROVIDERS, default_evidence_tier, ""
    )
    initial_task_catalog_note, initial_task_catalog_table = render_task_catalog(
        ALL_TASK_SOURCE_DISCLOSURES, ALL_CATALOG_REVIEW_STATES, ""
    )
    initial_model_note, initial_model_table = render_model_catalog(ALL_MODEL_PROVIDERS, "")
    initial_operational_note, initial_operational_table = render_operational(
        ALL_PROVIDERS, default_evidence_tier, "", False, 50
    )
    with gr.Blocks(title="Modern Embedding Bench") as demo:
        gr.Markdown("# Modern Embedding Bench")
        gr.Markdown(summary_markdown())
        with gr.Tab("Task leaderboard"):
            task_note = gr.Markdown(initial_note)

            with gr.Row():
                task = gr.Dropdown(choices=TASKS, value=default_task, label="Task")
                provider = gr.Dropdown(choices=PROVIDERS, value=ALL_PROVIDERS, label="Provider")
                evidence_tier = gr.Dropdown(
                    choices=EVIDENCE_TIERS,
                    value=default_evidence_tier,
                    label="Evidence tier available in this snapshot",
                )
                top_n = gr.Slider(5, 100, value=30, step=5, label="Rows")
            latest_only = gr.Checkbox(
                value=DEFAULT_LATEST_ONLY,
                label="Current marked row per task/model only",
                interactive=LATEST_MARKERS_AVAILABLE,
            )
            with gr.Row():
                data_overlap_status = gr.Dropdown(
                    choices=DATA_OVERLAP_STATUSES,
                    value=ALL_DATA_OVERLAP_STATUSES,
                    label="Data overlap status",
                )
                task_training_status = gr.Dropdown(
                    choices=TASK_TRAINING_STATUSES,
                    value=ALL_TASK_TRAINING_STATUSES,
                    label="Task training status",
                )
            reviewed_zero_shot_only = gr.Checkbox(
                value=False,
                label="Reviewed zero-shot only",
            )
            query = gr.Textbox(label="Search", placeholder="Filter by model, provider, run, or metric")
            table = gr.Dataframe(
                value=initial_table,
                label="Leaderboard",
                interactive=False,
            )
            controls = [
                task,
                provider,
                evidence_tier,
                query,
                latest_only,
                top_n,
                data_overlap_status,
                task_training_status,
                reviewed_zero_shot_only,
            ]
            task.change(render, inputs=controls, outputs=[task_note, table])
            provider.change(render, inputs=controls, outputs=[task_note, table])
            evidence_tier.change(render, inputs=controls, outputs=[task_note, table])
            query.change(render, inputs=controls, outputs=[task_note, table])
            latest_only.change(render, inputs=controls, outputs=[task_note, table])
            top_n.change(render, inputs=controls, outputs=[task_note, table])
            data_overlap_status.change(render, inputs=controls, outputs=[task_note, table])
            task_training_status.change(render, inputs=controls, outputs=[task_note, table])
            reviewed_zero_shot_only.change(render, inputs=controls, outputs=[task_note, table])

        with gr.Tab("Operational evidence"):
            operational_note = gr.Markdown(initial_operational_note)
            with gr.Row():
                operational_provider = gr.Dropdown(
                    choices=PROVIDERS,
                    value=ALL_PROVIDERS,
                    label="Provider",
                )
                operational_evidence = gr.Dropdown(
                    choices=EVIDENCE_TIERS,
                    value=default_evidence_tier,
                    label="Evidence tier",
                )
                operational_top_n = gr.Slider(10, 250, value=50, step=10, label="Rows")
            operational_latest_only = gr.Checkbox(
                value=False,
                label="Current marked row per task/model only",
                interactive=LATEST_MARKERS_AVAILABLE,
            )
            operational_query = gr.Textbox(
                label="Search",
                placeholder="Filter by task, model, provider, run, provenance, or dataset version",
            )
            operational_table_view = gr.Dataframe(
                value=initial_operational_table,
                label="Operational evidence (unranked)",
                interactive=False,
            )
            operational_controls = [
                operational_provider,
                operational_evidence,
                operational_query,
                operational_latest_only,
                operational_top_n,
            ]
            operational_provider.change(
                render_operational,
                inputs=operational_controls,
                outputs=[operational_note, operational_table_view],
            )
            operational_evidence.change(
                render_operational,
                inputs=operational_controls,
                outputs=[operational_note, operational_table_view],
            )
            operational_query.change(
                render_operational,
                inputs=operational_controls,
                outputs=[operational_note, operational_table_view],
            )
            operational_latest_only.change(
                render_operational,
                inputs=operational_controls,
                outputs=[operational_note, operational_table_view],
            )
            operational_top_n.change(
                render_operational,
                inputs=operational_controls,
                outputs=[operational_note, operational_table_view],
            )

        with gr.Tab("Task catalog"):
            task_catalog_note = gr.Markdown(initial_task_catalog_note)
            with gr.Row():
                task_source_disclosure = gr.Dropdown(
                    choices=TASK_SOURCE_DISCLOSURES,
                    value=ALL_TASK_SOURCE_DISCLOSURES,
                    label="Evaluation-source disclosure",
                )
                task_review_state = gr.Dropdown(
                    choices=CATALOG_REVIEW_STATES,
                    value=ALL_CATALOG_REVIEW_STATES,
                    label="Review state",
                )
            task_catalog_query = gr.Textbox(
                label="Search",
                placeholder="Filter declared task metadata",
            )
            task_catalog = gr.Dataframe(
                value=initial_task_catalog_table,
                label="Declared task catalog (unranked)",
                interactive=False,
            )
            task_catalog_controls = [task_source_disclosure, task_review_state, task_catalog_query]
            task_source_disclosure.change(
                render_task_catalog,
                inputs=task_catalog_controls,
                outputs=[task_catalog_note, task_catalog],
            )
            task_review_state.change(
                render_task_catalog,
                inputs=task_catalog_controls,
                outputs=[task_catalog_note, task_catalog],
            )
            task_catalog_query.change(
                render_task_catalog,
                inputs=task_catalog_controls,
                outputs=[task_catalog_note, task_catalog],
            )

        with gr.Tab("Model catalog"):
            model_note = gr.Markdown(initial_model_note)
            model_provider = gr.Dropdown(
                choices=MODEL_PROVIDERS,
                value=ALL_MODEL_PROVIDERS,
                label="Declared provider",
            )
            model_query = gr.Textbox(
                label="Search",
                placeholder="Filter declared model metadata",
            )
            model_table = gr.Dataframe(
                value=initial_model_table,
                label="Declared model catalog (unranked)",
                interactive=False,
            )
            model_controls = [model_provider, model_query]
            model_provider.change(
                render_model_catalog,
                inputs=model_controls,
                outputs=[model_note, model_table],
            )
            model_query.change(
                render_model_catalog,
                inputs=model_controls,
                outputs=[model_note, model_table],
            )

        with gr.Tab("Coverage"):
            coverage_note = gr.Markdown(initial_coverage_note)
            with gr.Row():
                coverage_provider = gr.Dropdown(choices=PROVIDERS, value=ALL_PROVIDERS, label="Provider")
                coverage_evidence = gr.Dropdown(
                    choices=COVERAGE_EVIDENCE_TIERS,
                    value=default_evidence_tier,
                    label="Has at least one cell with evidence tier",
                )
            coverage_query = gr.Textbox(
                label="Search",
                placeholder="Filter by model, provider, evaluated task, or evidence tier",
            )
            coverage = gr.Dataframe(
                value=initial_coverage_table,
                label="Current model/task coverage",
                interactive=False,
            )
            coverage_controls = [coverage_provider, coverage_evidence, coverage_query]
            coverage_provider.change(render_coverage, inputs=coverage_controls, outputs=[coverage_note, coverage])
            coverage_evidence.change(render_coverage, inputs=coverage_controls, outputs=[coverage_note, coverage])
            coverage_query.change(render_coverage, inputs=coverage_controls, outputs=[coverage_note, coverage])
    demo.launch()


if __name__ == "__main__":
    main()
'''
    return (
        source.replace("__DATASET_REPO_ID__", repr(default_repo))
        .replace("__TASK_DETAILS__", repr(TASK_NOTES))
        .replace("__TRAINING_OVERLAP_WARNING__", repr(TRAINING_OVERLAP_WARNING))
    )
