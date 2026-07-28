"""Fail-closed materialization contracts for public benchmark data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mm_embed.benchmark.registry import TaskSpec


DATA_SOURCE_CONTRACT_VERSION = "1"
MATERIALIZATION_SCHEMA_VERSION = "1"
PUBLIC_MATERIALIZATION_TASK_IDS = frozenset(
    {
        "mrl_stress",
        "crosslingual_retrieval",
        "needle_in_haystack",
        "cross_modal_retrieval",
    }
)
DATA_MODES = frozenset({"real", "fixture"})
HEX_DIGEST_LENGTH = 64
PUBLIC_TASK_PAYLOAD_ROLES = {
    "mrl_stress": frozenset({"mrl_stsb_pairs"}),
    "crosslingual_retrieval": frozenset({"crosslingual_pairs"}),
    "needle_in_haystack": frozenset({"needle_haystacks", "needle_facts"}),
    "cross_modal_retrieval": frozenset({"cross_modal_metadata"}),
}
_AUTHORIZATION_TOKEN = object()


class MaterializationContractError(ValueError):
    """A stable, public-safe materialization validation error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreparedDataSource:
    """Execution-time source binding or a fail-closed validation error."""

    snapshot: dict[str, Any]
    authorization: MaterializationAuthorization | None = None
    error: str | None = None


@dataclass(frozen=True)
class AuthorizedMaterializationFile:
    """One exact manifest-declared file authorized for runtime reads."""

    path: str
    bytes: int
    sha256: str
    role: str | None = None
    rows: int | None = None


@dataclass(frozen=True)
class MaterializationAuthorization:
    """Private runtime authorization derived from a validated manifest."""

    snapshot: dict[str, Any]
    repository_root: Path
    payload_files: tuple[AuthorizedMaterializationFile, ...]
    assets: tuple[AuthorizedMaterializationFile, ...]
    _validation_token: object = field(repr=False, compare=False)

    @property
    def asset_paths(self) -> tuple[str, ...]:
        return tuple(asset.path for asset in self.assets)

    def read_payload_text(self, role: str) -> str:
        item = self._payload_file(role)
        try:
            return self._verified_bytes(item).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MaterializationContractError(
                "invalid_payload",
                f"authorized payload is not UTF-8: {item.path}",
            ) from exc

    def _payload_file(self, role: str) -> AuthorizedMaterializationFile:
        matches = [item for item in self.payload_files if item.role == role]
        if len(matches) != 1:
            raise MaterializationContractError(
                "payload_role_mismatch",
                f"validated materialization does not authorize exactly one payload for role '{role}'",
            )
        return matches[0]

    def resolve_payload_reference(self, payload_role: str, reference: Any) -> str:
        if not isinstance(reference, str) or not reference.strip():
            raise MaterializationContractError("asset_reference_mismatch", "asset reference must be a string")
        reference_path = Path(reference)
        if reference_path.is_absolute() or ".." in reference_path.parts:
            raise MaterializationContractError(
                "asset_reference_mismatch",
                "asset reference must be relative to its manifest-declared payload",
            )
        payload = self._payload_file(payload_role)
        payload_parent = Path(payload.path).parent
        resolved = _repository_path(self.repository_root, str(payload_parent / reference_path))
        return resolved.relative_to(self.repository_root.resolve()).as_posix()

    def read_asset(self, relative_path: str) -> bytes:
        matches = [item for item in self.assets if item.path == relative_path]
        if len(matches) != 1:
            raise MaterializationContractError(
                "unmanifested_asset",
                f"asset is not authorized by the validated materialization: {relative_path}",
            )
        return self._verified_bytes(matches[0])

    def _verified_bytes(self, item: AuthorizedMaterializationFile) -> bytes:
        path = _repository_path(self.repository_root, item.path)
        if not path.is_file():
            raise MaterializationContractError("missing_payload", f"authorized file is missing: {item.path}")
        content = path.read_bytes()
        if len(content) != item.bytes:
            raise MaterializationContractError("payload_size_mismatch", f"authorized file size changed: {item.path}")
        if _sha256_bytes(content) != item.sha256:
            raise MaterializationContractError("payload_hash_mismatch", f"authorized file hash changed: {item.path}")
        return content


def normalize_data_mode(data_mode: Any = None, use_mock: Any = None) -> str:
    """Resolve an explicit real/fixture mode, including the legacy mock flag."""
    explicit_mode = None if data_mode is None else str(data_mode).strip().lower()
    if explicit_mode == "mock":
        explicit_mode = "fixture"
    if explicit_mode is not None and explicit_mode not in DATA_MODES:
        raise MaterializationContractError(
            "invalid_data_mode",
            f"data_mode must be one of {sorted(DATA_MODES)}, got {data_mode!r}",
        )
    if use_mock is not None and not isinstance(use_mock, bool):
        raise MaterializationContractError("invalid_data_mode", "use_mock must be a boolean when provided")
    legacy_mode = None if use_mock is None else ("fixture" if use_mock else "real")
    if explicit_mode is not None and legacy_mode is not None and explicit_mode != legacy_mode:
        raise MaterializationContractError(
            "conflicting_data_mode",
            f"data_mode={explicit_mode!r} conflicts with use_mock={use_mock!r}",
        )
    resolved = explicit_mode or legacy_mode
    if resolved is None:
        raise MaterializationContractError(
            "missing_data_mode",
            "data_mode must be explicitly set to 'real' or 'fixture'",
        )
    return resolved


def prepare_data_source_contract(
    task: TaskSpec,
    task_kwargs: dict[str, Any],
    benchmark_root: str | Path,
) -> PreparedDataSource | None:
    """Prepare the immutable source snapshot used by a public task result."""
    if task.id not in PUBLIC_MATERIALIZATION_TASK_IDS:
        return None
    try:
        data_mode = normalize_data_mode(task_kwargs.get("data_mode"), task_kwargs.get("use_mock"))
    except MaterializationContractError as exc:
        return PreparedDataSource(
            snapshot=invalid_data_source_snapshot(task, data_mode="unknown", reason_code=exc.code),
            error=str(exc),
        )

    if data_mode == "fixture":
        return PreparedDataSource(snapshot=fixture_data_source_snapshot(task))

    try:
        authorization = validate_materialization_authorization(task, benchmark_root=benchmark_root)
    except MaterializationContractError as exc:
        return PreparedDataSource(
            snapshot=invalid_data_source_snapshot(task, data_mode="real", reason_code=exc.code),
            error=str(exc),
        )
    return PreparedDataSource(snapshot=authorization.snapshot, authorization=authorization)


def validate_materialization_manifest(
    task: TaskSpec,
    *,
    benchmark_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a manifest and return its public result-level snapshot."""
    return validate_materialization_authorization(
        task,
        benchmark_root=benchmark_root,
        manifest_path=manifest_path,
    ).snapshot


def validate_materialization_authorization(
    task: TaskSpec,
    *,
    benchmark_root: str | Path,
    manifest_path: str | Path | None = None,
) -> MaterializationAuthorization:
    """Validate a manifest and return its exact private runtime authorization."""
    benchmark_dir = Path(benchmark_root)
    repository_root = benchmark_dir.parent
    if manifest_path is None:
        path = benchmark_dir / "data_manifests" / task.id / f"{task.dataset_version}.json"
    else:
        path = Path(manifest_path)
        if not path.is_absolute():
            path = benchmark_dir / path
    if not path.is_file():
        raise MaterializationContractError(
            "missing_manifest",
            f"materialization manifest is missing for task '{task.id}' and dataset '{task.dataset_version}'",
        )

    raw_manifest = path.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationContractError("invalid_manifest", "materialization manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise MaterializationContractError("invalid_manifest", "materialization manifest must be an object")

    _validate_manifest_structure(manifest)
    _validate_task_source_binding(task, manifest)
    payload_files, row_count, ordered_rows_sha256, duplicate_rows = _validate_payload_files(
        manifest["materialization"]["files"],
        repository_root,
    )
    assets, asset_count, asset_manifest_sha256 = _validate_assets(
        manifest["materialization"]["assets"],
        repository_root,
    )
    overlapping_paths = {item.path for item in payload_files}.intersection(item.path for item in assets)
    if overlapping_paths:
        raise MaterializationContractError(
            "duplicate_path",
            f"paths cannot be both payloads and assets: {sorted(overlapping_paths)}",
        )
    _validate_transformation_code(manifest["transformation"], repository_root)

    materialization = manifest["materialization"]
    validation = manifest["validation"]
    if materialization["row_count"] != row_count:
        raise MaterializationContractError("row_count_mismatch", "manifest row_count does not match payload rows")
    if materialization["ordered_row_manifest_sha256"] != ordered_rows_sha256:
        raise MaterializationContractError(
            "row_identity_mismatch",
            "ordered row manifest digest does not match payload rows",
        )
    if materialization["asset_count"] != asset_count:
        raise MaterializationContractError("asset_count_mismatch", "manifest asset_count does not match assets")
    if materialization["asset_manifest_sha256"] != asset_manifest_sha256:
        raise MaterializationContractError("asset_identity_mismatch", "asset manifest digest does not match assets")
    if validation["exact_duplicate_rows"] != duplicate_rows:
        raise MaterializationContractError(
            "duplicate_count_mismatch",
            "manifest duplicate-row count does not match payload rows",
        )

    snapshot = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "task_id": task.id,
        "dataset_version": task.dataset_version,
        "manifest_revision": manifest["manifest_revision"],
        "manifest_sha256": _sha256_bytes(raw_manifest),
        "data_mode": "real",
        "validation_status": "validated",
        "source_ids": sorted(source["source_id"] for source in manifest["sources"]),
        "transformation_id": manifest["transformation"]["transformation_id"],
        "row_count": row_count,
        "asset_count": asset_count,
        "reason_codes": [],
    }
    validate_data_source_snapshot(snapshot)
    authorization = MaterializationAuthorization(
        snapshot=snapshot,
        repository_root=repository_root.resolve(),
        payload_files=payload_files,
        assets=assets,
        _validation_token=_AUTHORIZATION_TOKEN,
    )
    _validate_task_runtime_authorization(task, authorization)
    return authorization


def fixture_data_source_snapshot(task: TaskSpec) -> dict[str, Any]:
    """Return the explicit source snapshot for an invented fixture path."""
    snapshot = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "task_id": task.id,
        "dataset_version": task.dataset_version,
        "manifest_revision": None,
        "manifest_sha256": None,
        "data_mode": "fixture",
        "validation_status": "fixture",
        "source_ids": [],
        "transformation_id": None,
        "row_count": None,
        "asset_count": None,
        "reason_codes": ["fixture_data"],
    }
    validate_data_source_snapshot(snapshot)
    return snapshot


def invalid_data_source_snapshot(task: TaskSpec, *, data_mode: str, reason_code: str) -> dict[str, Any]:
    """Return a non-publishable snapshot for a failed materialization check."""
    snapshot = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "task_id": task.id,
        "dataset_version": task.dataset_version,
        "manifest_revision": None,
        "manifest_sha256": None,
        "data_mode": data_mode,
        "validation_status": "invalid",
        "source_ids": [],
        "transformation_id": None,
        "row_count": None,
        "asset_count": None,
        "reason_codes": [reason_code],
    }
    validate_data_source_snapshot(snapshot)
    return snapshot


def legacy_unknown_data_source_contract(record: dict[str, Any]) -> dict[str, Any]:
    """Freeze pre-contract rows as unknown without consulting current files."""
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    snapshot = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "task_id": str(task.get("id") or "unknown"),
        "dataset_version": str(task.get("dataset_version") or "unknown"),
        "manifest_revision": None,
        "manifest_sha256": None,
        "data_mode": "unknown",
        "validation_status": "unknown",
        "source_ids": [],
        "transformation_id": None,
        "row_count": None,
        "asset_count": None,
        "reason_codes": ["legacy_missing_source_contract"],
    }
    validate_data_source_snapshot(snapshot)
    return snapshot


def public_data_source_contract_for_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a frozen source snapshot, or normalize a pre-contract row."""
    snapshot = record.get("data_source_contract")
    contract_version = record.get("data_source_contract_version")
    if snapshot is None:
        if contract_version == DATA_SOURCE_CONTRACT_VERSION:
            raise ValueError("Post-contract public result is missing data_source_contract")
        return legacy_unknown_data_source_contract(record)
    if not isinstance(snapshot, dict):
        raise ValueError("data_source_contract must be an object")
    validate_data_source_snapshot(snapshot)
    if contract_version != DATA_SOURCE_CONTRACT_VERSION and snapshot["validation_status"] != "unknown":
        raise ValueError("Non-legacy data_source_contract is missing its contract version")
    return dict(snapshot)


def validate_public_data_source_record(record: dict[str, Any]) -> dict[str, Any]:
    """Reject public post-contract rows without a valid real-data binding."""
    snapshot = public_data_source_contract_for_record(record)
    if record.get("data_source_contract_version") != DATA_SOURCE_CONTRACT_VERSION:
        return snapshot
    run = record.get("run") if isinstance(record.get("run"), dict) else {}
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    if snapshot["task_id"] != task.get("id") or snapshot["dataset_version"] != task.get("dataset_version"):
        raise ValueError("Post-contract data_source_contract does not match the result task identity")
    if snapshot["validation_status"] in {"validated", "fixture"}:
        kwargs = task.get("kwargs") if isinstance(task.get("kwargs"), dict) else {}
        try:
            declared_mode = normalize_data_mode(kwargs.get("data_mode"), kwargs.get("use_mock"))
        except MaterializationContractError as exc:
            raise ValueError("Post-contract public result lacks an explicit data mode") from exc
        if declared_mode != snapshot["data_mode"]:
            raise ValueError("Post-contract result data mode does not match its source snapshot")
    if run.get("publish") is False:
        return snapshot
    if snapshot["data_mode"] == "fixture":
        raise ValueError("Post-contract public result uses fixture data")
    if snapshot["data_mode"] != "real" or snapshot["validation_status"] != "validated":
        raise ValueError("Post-contract public result lacks a valid real-data materialization binding")
    return snapshot


def validate_data_source_snapshot(snapshot: dict[str, Any]) -> None:
    """Validate the frozen result-level source snapshot only."""
    required = {
        "schema_version",
        "task_id",
        "dataset_version",
        "manifest_revision",
        "manifest_sha256",
        "data_mode",
        "validation_status",
        "source_ids",
        "transformation_id",
        "row_count",
        "asset_count",
        "reason_codes",
    }
    _require_exact_keys(snapshot, required, set(), "data_source_contract")
    if snapshot["schema_version"] != MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("Unsupported data-source contract schema version")
    _require_nonempty_string(snapshot["task_id"], "data_source_contract.task_id")
    _require_nonempty_string(snapshot["dataset_version"], "data_source_contract.dataset_version")
    if snapshot["data_mode"] not in {"real", "fixture", "unknown"}:
        raise ValueError("Unsupported data_source_contract data_mode")
    if snapshot["validation_status"] not in {"validated", "fixture", "invalid", "unknown"}:
        raise ValueError("Unsupported data_source_contract validation_status")
    _require_string_list(snapshot["source_ids"], "data_source_contract.source_ids", unique=True)
    _require_string_list(snapshot["reason_codes"], "data_source_contract.reason_codes", unique=True)

    status = snapshot["validation_status"]
    if status == "validated":
        if snapshot["data_mode"] != "real":
            raise ValueError("Validated data_source_contract must use real data")
        _require_nonempty_string(snapshot["manifest_revision"], "data_source_contract.manifest_revision")
        _require_sha256(snapshot["manifest_sha256"], "data_source_contract.manifest_sha256")
        _require_nonempty_string(snapshot["transformation_id"], "data_source_contract.transformation_id")
        _require_nonnegative_int(snapshot["row_count"], "data_source_contract.row_count")
        _require_nonnegative_int(snapshot["asset_count"], "data_source_contract.asset_count")
        if not snapshot["source_ids"]:
            raise ValueError("Validated data_source_contract requires source_ids")
        if snapshot["reason_codes"]:
            raise ValueError("Validated data_source_contract cannot contain reason codes")
        return

    if any(
        snapshot[key] is not None
        for key in ("manifest_revision", "manifest_sha256", "transformation_id", "row_count", "asset_count")
    ):
        raise ValueError("Non-validated data_source_contract cannot claim materialization identity")
    if snapshot["source_ids"]:
        raise ValueError("Non-validated data_source_contract cannot claim source_ids")
    if not snapshot["reason_codes"]:
        raise ValueError("Non-validated data_source_contract requires a reason code")
    if status == "fixture" and snapshot["data_mode"] != "fixture":
        raise ValueError("Fixture data_source_contract must use fixture data")
    if status == "unknown" and snapshot["data_mode"] != "unknown":
        raise ValueError("Unknown data_source_contract must use unknown data mode")


def require_task_materialization_binding(
    task_id: str,
    data_mode: str,
    binding: MaterializationAuthorization | None,
) -> MaterializationAuthorization | None:
    """Prevent direct real task execution without a validator-issued binding."""
    if data_mode == "fixture":
        return None
    if (
        not isinstance(binding, MaterializationAuthorization)
        or binding._validation_token is not _AUTHORIZATION_TOKEN
    ):
        raise MaterializationContractError(
            "missing_runtime_binding",
            f"real data for task '{task_id}' requires a validated materialization binding",
        )
    validate_data_source_snapshot(binding.snapshot)
    if (
        binding.snapshot["task_id"] != task_id
        or binding.snapshot["data_mode"] != "real"
        or binding.snapshot["validation_status"] != "validated"
    ):
        raise MaterializationContractError(
            "invalid_runtime_binding",
            f"real data for task '{task_id}' has an invalid materialization binding",
        )
    return binding


def ordered_jsonl_manifest_sha256(paths: list[Path]) -> tuple[int, str, int]:
    """Return row count, ordered canonical-row digest, and duplicate count."""
    row_digests: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MaterializationContractError(
                        "invalid_payload",
                        f"payload JSONL is invalid at {path.name}:{line_number}",
                    ) from exc
                canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                row_digests.append(_sha256_bytes(canonical))
    digest = _sha256_bytes("\n".join(row_digests).encode())
    return len(row_digests), digest, len(row_digests) - len(set(row_digests))


def asset_manifest_sha256(assets: list[dict[str, Any]]) -> str | None:
    """Return the canonical digest for an ordered asset identity list."""
    if not assets:
        return None
    canonical = json.dumps(assets, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(canonical)


def _validate_manifest_structure(manifest: dict[str, Any]) -> None:
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "task_id",
            "dataset_version",
            "manifest_revision",
            "created_at",
            "sources",
            "transformation",
            "materialization",
            "validation",
            "review",
        },
        set(),
        "manifest",
    )
    if manifest["schema_version"] != MATERIALIZATION_SCHEMA_VERSION:
        raise MaterializationContractError("schema_version_mismatch", "unsupported materialization schema version")
    for key in ("task_id", "dataset_version", "manifest_revision"):
        _require_nonempty_string(manifest[key], f"manifest.{key}")
    _require_rfc3339(manifest["created_at"], "manifest.created_at")

    sources = manifest["sources"]
    if not isinstance(sources, list) or not sources:
        raise MaterializationContractError("invalid_manifest", "manifest.sources must be a non-empty array")
    source_ids: list[str] = []
    generated_roles = False
    for index, source in enumerate(sources):
        label = f"manifest.sources[{index}]"
        if not isinstance(source, dict):
            raise MaterializationContractError("invalid_manifest", f"{label} must be an object")
        _require_exact_keys(
            source,
            {
                "source_id",
                "source_revision",
                "config",
                "split",
                "role",
                "provenance_urls",
                "license_id",
                "authorship",
                "attribution",
                "redistribution",
                "privacy_review",
                "publication_status",
            },
            set(),
            label,
        )
        _require_nonempty_string(source["source_id"], f"{label}.source_id")
        _require_nonempty_string(source["source_revision"], f"{label}.source_revision")
        _require_optional_string(source["config"], f"{label}.config")
        _require_optional_string(source["split"], f"{label}.split")
        _require_nonempty_string(source["role"], f"{label}.role")
        _require_string_list(source["provenance_urls"], f"{label}.provenance_urls", nonempty=True, unique=True)
        for key in ("license_id", "authorship", "attribution"):
            _require_nonempty_string(source[key], f"{label}.{key}")
        if source["redistribution"] not in {"allowed", "restricted", "not_reviewed"}:
            raise MaterializationContractError("invalid_manifest", f"{label}.redistribution is invalid")
        if source["privacy_review"] not in {"approved", "restricted", "not_recorded"}:
            raise MaterializationContractError("invalid_manifest", f"{label}.privacy_review is invalid")
        if source["publication_status"] not in {"public", "private", "unknown"}:
            raise MaterializationContractError("invalid_manifest", f"{label}.publication_status is invalid")
        source_ids.append(source["source_id"])
        generated_roles = generated_roles or source["role"].startswith("generated")
    if len(source_ids) != len(set(source_ids)):
        raise MaterializationContractError("source_mismatch", "manifest source IDs must be unique")

    transformation = manifest["transformation"]
    if not isinstance(transformation, dict):
        raise MaterializationContractError("invalid_manifest", "manifest.transformation must be an object")
    _require_exact_keys(
        transformation,
        {
            "transformation_id",
            "code_commit",
            "code_path",
            "code_sha256",
            "parameters",
            "seed",
            "determinism",
            "generator",
        },
        set(),
        "manifest.transformation",
    )
    for key in ("transformation_id", "code_commit", "code_path"):
        _require_nonempty_string(transformation[key], f"manifest.transformation.{key}")
    _require_sha256(transformation["code_sha256"], "manifest.transformation.code_sha256")
    if not isinstance(transformation["parameters"], dict):
        raise MaterializationContractError("invalid_manifest", "transformation.parameters must be an object")
    if transformation["seed"] is not None and (
        isinstance(transformation["seed"], bool) or not isinstance(transformation["seed"], int)
    ):
        raise MaterializationContractError("invalid_manifest", "transformation.seed must be an integer or null")
    if transformation["determinism"] not in {"deterministic", "generator_logged", "non_reproducible"}:
        raise MaterializationContractError("invalid_manifest", "transformation.determinism is invalid")
    _validate_generator(
        transformation["generator"],
        required=generated_roles or transformation["determinism"] != "deterministic",
    )

    materialization = manifest["materialization"]
    if not isinstance(materialization, dict):
        raise MaterializationContractError("invalid_manifest", "manifest.materialization must be an object")
    _require_exact_keys(
        materialization,
        {
            "row_identity_scheme",
            "row_count",
            "ordered_row_manifest_sha256",
            "asset_count",
            "asset_manifest_sha256",
            "files",
            "assets",
        },
        set(),
        "manifest.materialization",
    )
    _require_nonempty_string(materialization["row_identity_scheme"], "materialization.row_identity_scheme")
    _require_nonnegative_int(materialization["row_count"], "materialization.row_count")
    _require_sha256(materialization["ordered_row_manifest_sha256"], "materialization.ordered_row_manifest_sha256")
    _require_nonnegative_int(materialization["asset_count"], "materialization.asset_count")
    if materialization["asset_manifest_sha256"] is not None:
        _require_sha256(materialization["asset_manifest_sha256"], "materialization.asset_manifest_sha256")
    if not isinstance(materialization["files"], list) or not materialization["files"]:
        raise MaterializationContractError("invalid_manifest", "materialization.files must be a non-empty array")
    if not isinstance(materialization["assets"], list):
        raise MaterializationContractError("invalid_manifest", "materialization.assets must be an array")
    for kind, entries, keys in (
        ("files", materialization["files"], {"role", "path", "bytes", "rows", "sha256"}),
        ("assets", materialization["assets"], {"path", "bytes", "sha256"}),
    ):
        for index, entry in enumerate(entries):
            label = f"materialization.{kind}[{index}]"
            if not isinstance(entry, dict):
                raise MaterializationContractError("invalid_manifest", f"{label} must be an object")
            _require_exact_keys(entry, keys, set(), label)
            if kind == "files":
                _require_nonempty_string(entry["role"], f"{label}.role")
            _require_safe_payload_path(entry["path"], f"{label}.path")
            _require_nonnegative_int(entry["bytes"], f"{label}.bytes")
            if kind == "files":
                _require_nonnegative_int(entry["rows"], f"{label}.rows")
            _require_sha256(entry["sha256"], f"{label}.sha256")

    validation = manifest["validation"]
    if not isinstance(validation, dict):
        raise MaterializationContractError("invalid_manifest", "manifest.validation must be an object")
    _require_exact_keys(
        validation,
        {
            "required_files_complete",
            "missing_assets",
            "orphan_assets",
            "exact_duplicate_rows",
            "source_binding_complete",
        },
        set(),
        "manifest.validation",
    )
    if validation["required_files_complete"] is not True or validation["source_binding_complete"] is not True:
        raise MaterializationContractError("partial_materialization", "manifest completeness checks are not approved")
    for key in ("missing_assets", "orphan_assets", "exact_duplicate_rows"):
        _require_nonnegative_int(validation[key], f"manifest.validation.{key}")
    if validation["missing_assets"] or validation["orphan_assets"]:
        raise MaterializationContractError("partial_materialization", "manifest records missing or orphan assets")

    review = manifest["review"]
    if not isinstance(review, dict):
        raise MaterializationContractError("invalid_manifest", "manifest.review must be an object")
    _require_exact_keys(review, {"state", "reviewed_at", "reviewed_by"}, set(), "manifest.review")
    if review["state"] != "approved":
        raise MaterializationContractError("manifest_not_approved", "materialization manifest is not approved")
    _require_rfc3339(review["reviewed_at"], "manifest.review.reviewed_at")
    _require_nonempty_string(review["reviewed_by"], "manifest.review.reviewed_by")


def _validate_task_source_binding(task: TaskSpec, manifest: dict[str, Any]) -> None:
    if manifest["task_id"] != task.id:
        raise MaterializationContractError("task_mismatch", "manifest task_id does not match task registry")
    if manifest["dataset_version"] != task.dataset_version:
        raise MaterializationContractError(
            "dataset_version_mismatch",
            "manifest dataset_version does not match task registry",
        )
    evaluation = task.evaluation_sources
    provenance = evaluation.public_provenance
    if (
        evaluation.disclosure != "complete"
        or evaluation.review.state != "approved"
        or not evaluation.sources
        or not provenance.evidence_revision
        or not provenance.reviewed_at
        or not provenance.reviewed_by
        or not provenance.urls
    ):
        raise MaterializationContractError(
            "task_source_declaration_incomplete",
            "task evaluation-source declaration is not complete and approved",
        )
    expected_tuples = {
        (source.source_id, source.source_revision, source.config, source.split)
        for source in evaluation.sources
    }
    manifest_tuples = {
        (source["source_id"], source["source_revision"], source["config"], source["split"])
        for source in manifest["sources"]
    }
    if expected_tuples != manifest_tuples:
        raise MaterializationContractError("source_mismatch", "manifest sources do not match task registry")
    transformation_ids = {source.transformation_id for source in evaluation.sources}
    if None in transformation_ids or transformation_ids != {manifest["transformation"]["transformation_id"]}:
        raise MaterializationContractError(
            "transformation_mismatch",
            "manifest transformation does not match task registry",
        )
    if manifest["transformation"]["parameters"] != task.materialization.transformation_parameters:
        raise MaterializationContractError(
            "parameters_mismatch",
            "manifest transformation parameters do not match task registry",
        )


def _validate_task_runtime_authorization(
    task: TaskSpec,
    authorization: MaterializationAuthorization,
) -> None:
    required_roles = PUBLIC_TASK_PAYLOAD_ROLES.get(task.id)
    declared_roles = {item.role for item in authorization.payload_files}
    if required_roles is not None and declared_roles != required_roles:
        raise MaterializationContractError(
            "payload_role_mismatch",
            f"task '{task.id}' requires payload roles {sorted(required_roles)}, got {sorted(declared_roles)}",
        )
    if task.id != "cross_modal_retrieval":
        return
    metadata_text = authorization.read_payload_text("cross_modal_metadata")
    references: list[str] = []
    for line_number, line in enumerate(metadata_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MaterializationContractError(
                "invalid_payload",
                f"cross-modal metadata is invalid at row {line_number}",
            ) from exc
        if not isinstance(row, dict) or "image_path" not in row:
            raise MaterializationContractError(
                "asset_reference_mismatch",
                f"cross-modal metadata row {line_number} is missing image_path",
            )
        references.append(
            authorization.resolve_payload_reference("cross_modal_metadata", row["image_path"])
        )
    if tuple(references) != authorization.asset_paths:
        raise MaterializationContractError(
            "asset_reference_mismatch",
            "cross-modal metadata references do not exactly match manifest assets",
        )


def _validate_generator(generator: Any, *, required: bool) -> None:
    if not required:
        if generator is not None:
            raise MaterializationContractError(
                "generator_metadata_mismatch",
                "deterministic materialization must not include generator metadata",
            )
        return
    if not isinstance(generator, dict):
        raise MaterializationContractError(
            "incomplete_generator_metadata",
            "generator-backed materialization requires generator metadata",
        )
    required_keys = {
        "provider",
        "model_revision",
        "model_revision_unavailable_reason",
        "model_alias",
        "prompt_sha256",
        "parameters",
        "requested_seed",
        "nondeterminism_reason",
        "request_ids",
        "response_ids",
        "generation_timestamps",
        "ordered_output_sha256",
    }
    if set(generator) != required_keys:
        raise MaterializationContractError(
            "incomplete_generator_metadata",
            "generator metadata keys are incomplete or unsupported",
        )
    _require_nonempty_string(generator["provider"], "generator.provider")
    _require_optional_string(generator["model_revision"], "generator.model_revision")
    _require_optional_string(
        generator["model_revision_unavailable_reason"],
        "generator.model_revision_unavailable_reason",
    )
    if not generator["model_revision"] and not generator["model_revision_unavailable_reason"]:
        raise MaterializationContractError(
            "incomplete_generator_metadata",
            "generator requires an immutable model revision or an explicit unavailable reason",
        )
    _require_nonempty_string(generator["model_alias"], "generator.model_alias")
    _require_sha256(generator["prompt_sha256"], "generator.prompt_sha256")
    if not isinstance(generator["parameters"], dict) or not generator["parameters"]:
        raise MaterializationContractError(
            "incomplete_generator_metadata",
            "generator.parameters must be a non-empty object",
        )
    if generator["requested_seed"] is not None and (
        isinstance(generator["requested_seed"], bool) or not isinstance(generator["requested_seed"], int)
    ):
        raise MaterializationContractError("incomplete_generator_metadata", "generator.requested_seed is invalid")
    _require_optional_string(generator["nondeterminism_reason"], "generator.nondeterminism_reason")
    if generator["requested_seed"] is None and not generator["nondeterminism_reason"]:
        raise MaterializationContractError(
            "incomplete_generator_metadata",
            "generator without a requested seed requires a nondeterminism reason",
        )
    for key in ("request_ids", "response_ids", "generation_timestamps"):
        _require_string_list(generator[key], f"generator.{key}", nonempty=True, unique=True)
    if not (
        len(generator["request_ids"])
        == len(generator["response_ids"])
        == len(generator["generation_timestamps"])
    ):
        raise MaterializationContractError(
            "incomplete_generator_metadata",
            "generator request, response, and timestamp counts must match",
        )
    for value in generator["generation_timestamps"]:
        _require_rfc3339(value, "generator.generation_timestamps")
    _require_sha256(generator["ordered_output_sha256"], "generator.ordered_output_sha256")


def _validate_payload_files(
    entries: list[dict[str, Any]],
    repository_root: Path,
) -> tuple[tuple[AuthorizedMaterializationFile, ...], int, str, int]:
    paths: list[Path] = []
    authorized: list[AuthorizedMaterializationFile] = []
    declared_rows = 0
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for entry in entries:
        relative = entry["path"]
        role = entry["role"]
        if relative in seen_paths:
            raise MaterializationContractError("duplicate_path", f"duplicate payload path '{relative}'")
        if role in seen_roles:
            raise MaterializationContractError("payload_role_mismatch", f"duplicate payload role '{role}'")
        seen_paths.add(relative)
        seen_roles.add(role)
        path = _repository_path(repository_root, relative)
        if not path.is_file():
            raise MaterializationContractError("missing_payload", f"required payload is missing: {relative}")
        if path.stat().st_size != entry["bytes"]:
            raise MaterializationContractError("payload_size_mismatch", f"payload size mismatch: {relative}")
        if _sha256_path(path) != entry["sha256"]:
            raise MaterializationContractError("payload_hash_mismatch", f"payload hash mismatch: {relative}")
        if path.suffix != ".jsonl" and entry["rows"] != 0:
            raise MaterializationContractError(
                "row_count_mismatch",
                f"non-JSONL payload must declare zero rows: {relative}",
            )
        if path.suffix == ".jsonl":
            rows, _, _ = ordered_jsonl_manifest_sha256([path])
            if rows != entry["rows"]:
                raise MaterializationContractError("row_count_mismatch", f"payload row count mismatch: {relative}")
            declared_rows += entry["rows"]
            paths.append(path)
        authorized.append(
            AuthorizedMaterializationFile(
                role=role,
                path=relative,
                bytes=entry["bytes"],
                rows=entry["rows"],
                sha256=entry["sha256"],
            )
        )
    actual_rows, digest, duplicates = ordered_jsonl_manifest_sha256(paths)
    if actual_rows != declared_rows:
        raise MaterializationContractError("row_count_mismatch", "payload row count is inconsistent")
    return tuple(authorized), actual_rows, digest, duplicates


def _validate_assets(
    entries: list[dict[str, Any]],
    repository_root: Path,
) -> tuple[tuple[AuthorizedMaterializationFile, ...], int, str | None]:
    validated: list[dict[str, Any]] = []
    authorized: list[AuthorizedMaterializationFile] = []
    seen: set[str] = set()
    for entry in entries:
        relative = entry["path"]
        if relative in seen:
            raise MaterializationContractError("duplicate_path", f"duplicate asset path '{relative}'")
        seen.add(relative)
        path = _repository_path(repository_root, relative)
        if not path.is_file():
            raise MaterializationContractError("missing_asset", f"required asset is missing: {relative}")
        size = path.stat().st_size
        digest = _sha256_path(path)
        if size != entry["bytes"]:
            raise MaterializationContractError("asset_size_mismatch", f"asset size mismatch: {relative}")
        if digest != entry["sha256"]:
            raise MaterializationContractError("asset_hash_mismatch", f"asset hash mismatch: {relative}")
        validated.append({"path": relative, "bytes": size, "sha256": digest})
        authorized.append(
            AuthorizedMaterializationFile(
                path=relative,
                bytes=size,
                sha256=digest,
            )
        )
    return tuple(authorized), len(validated), asset_manifest_sha256(validated)


def _validate_transformation_code(transformation: dict[str, Any], repository_root: Path) -> None:
    relative = transformation["code_path"]
    _require_safe_repository_path(relative, "transformation.code_path")
    path = _repository_path(repository_root, relative)
    if not path.is_file():
        raise MaterializationContractError("missing_transformation_code", "transformation code path is missing")
    if _sha256_path(path) != transformation["code_sha256"]:
        raise MaterializationContractError("transformation_hash_mismatch", "transformation code hash is stale")


def _require_exact_keys(value: dict[str, Any], required: set[str], optional: set[str], label: str) -> None:
    missing = required - set(value)
    unexpected = set(value) - required - optional
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)}")
        raise MaterializationContractError("invalid_manifest", f"{label} has invalid keys ({', '.join(details)})")


def _require_nonempty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MaterializationContractError("invalid_manifest", f"{label} must be a non-empty string")


def _require_optional_string(value: Any, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise MaterializationContractError("invalid_manifest", f"{label} must be a non-empty string or null")


def _require_string_list(
    value: Any,
    label: str,
    *,
    nonempty: bool = False,
    unique: bool = False,
) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise MaterializationContractError("invalid_manifest", f"{label} must be an array of non-empty strings")
    if nonempty and not value:
        raise MaterializationContractError("invalid_manifest", f"{label} must not be empty")
    if unique and len(value) != len(set(value)):
        raise MaterializationContractError("invalid_manifest", f"{label} values must be unique")


def _require_nonnegative_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MaterializationContractError("invalid_manifest", f"{label} must be a non-negative integer")


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != HEX_DIGEST_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise MaterializationContractError("invalid_manifest", f"{label} must be a lowercase SHA256 digest")


def _require_rfc3339(value: Any, label: str) -> None:
    _require_nonempty_string(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MaterializationContractError("invalid_manifest", f"{label} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise MaterializationContractError("invalid_manifest", f"{label} must include a timezone")


def _require_safe_payload_path(value: Any, label: str) -> None:
    _require_safe_repository_path(value, label)
    if Path(value).parts[0] != "data":
        raise MaterializationContractError("invalid_manifest", f"{label} must be under data/")


def _require_safe_repository_path(value: Any, label: str) -> None:
    _require_nonempty_string(value, label)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise MaterializationContractError("invalid_manifest", f"{label} must be repository-relative")


def _repository_path(repository_root: Path, relative: str) -> Path:
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise MaterializationContractError("invalid_manifest", "manifest path escapes the repository root")
    return path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
