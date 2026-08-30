"""Fail-closed Batch-B freeze, readiness, formal execution, and CPU replay contracts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from huggingface_hub import HfApi
from jsonschema import Draft202012Validator
from scipy import sparse

import mm_embed.benchmark.bright_learned_sparse_batch_a as batch_a
from mm_embed.benchmark.bright_learned_sparse_batch_a import canonical_sha256, ids_sha256
from mm_embed.benchmark.bright_multidomain_v02 import load_materialized, validate_materialization
from mm_embed.benchmark.retrieval_v01 import aggregate_metrics, bootstrap_confidence_intervals, query_metrics
from mm_embed.indexes.sparse_exact import ExactSparseIndex
from mm_embed.providers import get_provider
from mm_embed.providers.learned_sparse_inventory import (
    BATCH_B_DOWNLOAD_CAP_BYTES,
    BATCH_B_GPU_CAP_BYTES,
    BATCH_B_INVENTORY,
    BATCH_B_SELECTED_KEYS,
    STORY_DISK_CAP_BYTES,
    BoundedSnapshotResolver,
    LearnedSparseSpec,
    SnapshotPolicyError,
    aggregate_snapshot_identity,
)
from mm_embed.providers.snapshot_identity import file_sha256, snapshot_identity, verify_snapshot_identity

PREDECLARATION_SCHEMA = "bright-learned-sparse-batch-b-predeclaration-v02.schema.json"
SUPERSEDED_PREDECLARATION_SCHEMA = "bright-learned-sparse-batch-b-predeclaration-v01.schema.json"
GATE_SCHEMA = "bright-learned-sparse-batch-b-gate-v01.schema.json"
FORMAL_RESULT_SCHEMA = "bright-learned-sparse-batch-b-result-v01.schema.json"
FORMAL_REPLAY_SCHEMA = "bright-learned-sparse-batch-b-replay-v01.schema.json"
ACTIVE_PREDECLARATION_SHA256 = "2d30ac69b58e7bbd217550418619bab085e2c8b73e1453ee02e0d237827a32ef"
PUBLICATION = {
    "publish": False,
    "leaderboard_publish": False,
    "public_export_allowed": False,
    "gate": "closed",
    "classification": "research_only",
}
GATE_WALL_CAP_SECONDS = 600.0
EXPECTED_FALLBACK = [8, 4, 2, 1]
FORMAL_CHUNK_SIZE = 256
FORMAL_BATCH_SIZE = 8
FORMAL_TOP_K = 100
FORMAL_BOOTSTRAP_SAMPLES = 10_000
FORMAL_BOOTSTRAP_SEED = 20_260_826
FORMAL_TRACKS = ("economics", "psychology")
MECHANISMS = {
    "splade-tiny": "lightweight neural SPLADE masked-language-model expansion",
    "opensearch-doc-v2-distill": "distilled neural SPLADE query and document expansion",
    "opensearch-multilingual": "static sparse query lookup plus multilingual neural document expansion",
}


class BatchBError(ValueError):
    """Raised when Batch-B evidence violates a frozen boundary."""


def _schema_path(name: str) -> Path:
    return Path(__file__).parents[3] / "schemas" / name


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchBError(f"Missing or invalid JSON artifact: {path}") from exc


def _validate_schema(value: Any, name: str) -> None:
    try:
        Draft202012Validator(_load_json(_schema_path(name))).validate(value)
    except Exception as exc:
        raise BatchBError(f"{name} validation failed: {exc}") from exc


def _require_sha256(value: str | None, label: str) -> str:
    if value is None or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise BatchBError(f"Missing or invalid externally supplied {label}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def write_evidence(path: str | Path, value: Mapping[str, Any]) -> str:
    """Write research-only evidence and a content identity sidecar."""
    assert_research_only(value)
    output = Path(path)
    _write_json(output, value)
    identity = file_sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return identity


def validate_predeclaration(path: str | Path, *, expected_sha256: str | None) -> dict[str, Any]:
    """Validate schema, embedded payload identity, sidecar, and external identity."""
    artifact_path = Path(path)
    expected = _require_sha256(expected_sha256, "predeclaration SHA256")
    if expected != ACTIVE_PREDECLARATION_SHA256:
        raise BatchBError("Expected predeclaration identity is not the unique active revision")
    actual = file_sha256(artifact_path)
    if actual != expected:
        raise BatchBError("Predeclaration does not match its externally supplied identity")
    sidecar = artifact_path.with_suffix(".sha256")
    try:
        sidecar_value = sidecar.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise BatchBError("Predeclaration identity sidecar is missing") from exc
    if sidecar_value != actual:
        raise BatchBError("Predeclaration sidecar identity mismatch")
    value = _load_json(artifact_path)
    _validate_schema(value, PREDECLARATION_SCHEMA)
    payload = dict(value)
    identity = payload.pop("identity")
    if canonical_sha256(payload) != identity["payload_sha256"]:
        raise BatchBError("Predeclaration embedded payload identity mismatch")
    if tuple(row["key"] for row in value["models"]) != BATCH_B_SELECTED_KEYS:
        raise BatchBError("Predeclaration model order drifted")
    for row in value["models"]:
        spec = BATCH_B_INVENTORY[row["key"]]
        expected_contract = {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "license": spec.license,
            "gated": spec.gated,
            "trust_remote_code": spec.trust_remote_code,
            "allowlist": list(spec.allowlist),
            "query_route": spec.query_route,
            "document_route": spec.document_route,
            "vocabulary_id": spec.vocabulary_id,
            "dimensions": spec.dimensions,
            "max_length": spec.max_length,
            "download_cap_bytes": spec.download_cap_bytes,
            "estimated_snapshot_bytes": spec.estimated_snapshot_bytes,
        }
        observed = {key: row[key] for key in expected_contract}
        if observed != expected_contract:
            raise BatchBError(f"Predeclaration executable model contract drifted for {spec.key}")
    return value


def validate_superseded_predeclaration(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    """Validate the archived v1 readiness declaration without treating it as active."""
    artifact_path = Path(path)
    expected = _require_sha256(expected_sha256, "superseded predeclaration SHA256")
    if file_sha256(artifact_path) != expected:
        raise BatchBError("Superseded predeclaration identity drifted")
    try:
        sidecar = artifact_path.with_suffix(".sha256").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise BatchBError("Superseded predeclaration sidecar is missing") from exc
    if sidecar != expected:
        raise BatchBError("Superseded predeclaration sidecar identity mismatch")
    value = _load_json(artifact_path)
    _validate_schema(value, SUPERSEDED_PREDECLARATION_SCHEMA)
    payload = dict(value)
    identity = payload.pop("identity")
    if canonical_sha256(payload) != identity["payload_sha256"]:
        raise BatchBError("Superseded predeclaration payload identity mismatch")
    return value


def _business_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "created_at",
        "formal_scores_observed",
        "identity",
        "schema_version",
        "superseded_readiness_evidence",
        "supersedes_sha256",
        "supersession_reason",
    }
    return {key: item for key, item in value.items() if key not in excluded}


def validate_supersession(active: Mapping[str, Any], superseded_path: str | Path) -> dict[str, Any]:
    """Require exact business-contract equality across the audit-only revision."""
    old = validate_superseded_predeclaration(
        superseded_path, expected_sha256=active["supersedes_sha256"]
    )
    if _business_contract(active) != _business_contract(old):
        raise BatchBError("Active predeclaration changed frozen business fields during supersession")
    if active["formal_scores_observed"] is not False:
        raise BatchBError("Supersession must precede all formal Batch-B scoring")
    return old


def validate_batch_a_readonly(predeclaration: Mapping[str, Any], repo_root: str | Path) -> None:
    """Fail closed if any frozen Batch-A package or raw identity anchor drifts."""
    root = Path(repo_root)
    binding = predeclaration["bindings"]["batch_a"]
    manifest_path = root / binding["tracked_manifest_path"]
    if file_sha256(manifest_path) != binding["tracked_manifest_sha256"]:
        raise BatchBError("Batch-A tracked package manifest drifted")
    manifest = _load_json(manifest_path)
    if manifest["protocol"]["identity_sha256"] != binding["protocol_identity_sha256"]:
        raise BatchBError("Batch-A protocol identity drifted")
    observed_raw = {
        f"{row['model_key']}:{row['track']}": row["manifest_sha256"] for row in manifest["raw_results"]
    }
    if observed_raw != binding["raw_manifest_sha256"]:
        raise BatchBError("Batch-A raw manifest identity anchors drifted")


def validate_canonical_binding(predeclaration: Mapping[str, Any], data_root: str | Path) -> dict[str, Any]:
    materialization = validate_materialization(data_root)
    expected = predeclaration["bindings"]["canonical_data"]
    manifest_path = Path(data_root) / "manifest.json"
    if file_sha256(manifest_path) != expected["manifest_sha256"]:
        raise BatchBError("Canonical data identity drifted")
    if materialization["source"]["revision"] != expected["source_revision"]:
        raise BatchBError("Canonical source revision drifted")
    return materialization


def _api_license(info: Any) -> str | None:
    card_data = getattr(info, "card_data", None)
    if card_data is None:
        return None
    if hasattr(card_data, "to_dict"):
        card_data = card_data.to_dict()
    return card_data.get("license") if isinstance(card_data, dict) else None


def validate_hub_metadata(
    model_key: str,
    predeclaration: Mapping[str, Any],
    *,
    api: Any | None = None,
    predeclaration_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate immutable Hub API metadata without loading model behavior."""
    if model_key not in BATCH_B_SELECTED_KEYS:
        raise BatchBError("Model is outside the frozen Batch-B order")
    spec = BATCH_B_INVENTORY[model_key]
    api = api or HfApi()
    info = api.model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
    if str(info.sha) != spec.revision:
        raise BatchBError(f"{model_key} Hub revision drifted")
    if bool(getattr(info, "private", False)) or bool(getattr(info, "gated", False)):
        raise BatchBError(f"{model_key} became private or gated")
    observed_license = _api_license(info)
    accepted = predeclaration["model_evidence"][model_key]["accepted_api_licenses"]
    if observed_license not in accepted:
        raise BatchBError(f"{model_key} Hub license metadata drifted: {observed_license!r}")
    siblings = {row.rfilename: row for row in info.siblings}
    missing = sorted(set(spec.allowlist) - set(siblings))
    if missing:
        raise BatchBError(f"{model_key} Hub file set drifted; missing={missing}")
    files = []
    for name in spec.allowlist:
        row = siblings[name]
        size = getattr(row, "size", None)
        if size is None:
            raise BatchBError(f"{model_key} Hub API omitted file size for {name}")
        files.append({"path": name, "bytes": int(size)})
    declared = sum(row["bytes"] for row in files)
    if declared != spec.estimated_snapshot_bytes or declared > int(spec.download_cap_bytes or 0):
        raise BatchBError(f"{model_key} Hub allowlist bytes drifted: {declared}")
    return {
        "schema_version": "learned-sparse-batch-b-metadata-v1",
        "publication": PUBLICATION,
        "predeclaration_sha256": _require_sha256(predeclaration_sha256, "predeclaration SHA256"),
        "model_key": model_key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "private": False,
        "gated": False,
        "api_license": observed_license,
        "files": sorted(files, key=lambda row: row["path"]),
        "declared_bytes": declared,
        "repository_file_count": len(siblings),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def validate_evidence_file(path: str | Path, *, expected_sha256: str | None, label: str) -> dict[str, Any]:
    evidence_path = Path(path)
    expected = _require_sha256(expected_sha256, f"{label} SHA256")
    actual = file_sha256(evidence_path)
    if actual != expected:
        raise BatchBError(f"{label} does not match its externally supplied identity")
    sidecar_path = evidence_path.with_suffix(evidence_path.suffix + ".sha256")
    try:
        sidecar = sidecar_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise BatchBError(f"{label} identity sidecar is missing") from exc
    if sidecar != actual:
        raise BatchBError(f"{label} identity sidecar mismatch")
    value = _load_json(evidence_path)
    if value.get("publication") != PUBLICATION:
        raise BatchBError(f"{label} publication boundary drifted")
    return value


def plan_snapshot(
    model_key: str,
    *,
    metadata: Mapping[str, Any],
    metadata_sha256: str,
    predeclaration_sha256: str,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    spec = BATCH_B_INVENTORY[model_key]
    plan = BoundedSnapshotResolver(cache_root=cache_root).plan(spec)
    expected_files = [{"path": row["path"], "bytes": row["bytes"]} for row in plan["files"]]
    if metadata.get("model_key") != model_key or metadata.get("revision") != spec.revision:
        raise BatchBError("Metadata evidence does not match the download plan")
    if metadata.get("files") != expected_files or metadata.get("declared_bytes") != plan["declared_bytes"]:
        raise BatchBError("Hub metadata file bytes drifted from the download plan")
    return {
        "schema_version": "learned-sparse-batch-b-download-plan-v1",
        "publication": PUBLICATION,
        "predeclaration_sha256": _require_sha256(predeclaration_sha256, "predeclaration SHA256"),
        "metadata_sha256": _require_sha256(metadata_sha256, "metadata SHA256"),
        **plan,
    }


def download_snapshot(
    model_key: str,
    plan: Mapping[str, Any],
    *,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    spec = BATCH_B_INVENTORY[model_key]
    if plan.get("publication") not in (None, PUBLICATION):
        raise BatchBError("Download plan publication boundary drifted")
    resolver_plan = {key: value for key, value in plan.items() if key not in {"schema_version", "publication"}}
    resolved = BoundedSnapshotResolver(cache_root=cache_root).resolve(spec, plan=resolver_plan)
    return {
        "schema_version": "learned-sparse-batch-b-snapshot-v1",
        "publication": PUBLICATION,
        "predeclaration_sha256": plan["predeclaration_sha256"],
        "metadata_sha256": plan["metadata_sha256"],
        **resolved,
    }


def validate_snapshot_evidence(model_key: str, evidence: Mapping[str, Any]) -> tuple[Path, dict[str, str]]:
    spec = BATCH_B_INVENTORY[model_key]
    if evidence.get("model_key", evidence.get("key")) != model_key:
        raise BatchBError("Snapshot evidence model mismatch")
    if evidence.get("revision") != spec.revision or evidence.get("publication") != PUBLICATION:
        raise BatchBError("Snapshot evidence boundary drifted")
    path = Path(evidence["snapshot_path"])
    identity = dict(evidence["identity"])
    verify_snapshot_identity(path, identity, label=model_key)
    if set(identity) != set(spec.allowlist):
        raise BatchBError("Snapshot evidence allowlist drifted")
    if aggregate_snapshot_identity(identity) != evidence["aggregate_identity_sha256"]:
        raise BatchBError("Snapshot aggregate identity drifted")
    actual_bytes = sum((path / name).stat().st_size for name in identity)
    if actual_bytes != evidence["actual_bytes"] or actual_bytes > int(spec.download_cap_bytes or 0):
        raise BatchBError("Snapshot byte identity drifted")
    return path, identity


def _validate_csr(matrix: sparse.csr_matrix, *, rows: int, dimensions: int) -> None:
    if not sparse.isspmatrix_csr(matrix) or matrix.shape != (rows, dimensions):
        raise BatchBError("Gate output must be canonical CSR with the frozen shape")
    if not matrix.has_sorted_indices or not matrix.has_canonical_format:
        raise BatchBError("Gate CSR is not canonical")
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise BatchBError("Gate CSR must be finite and nonnegative")


def _audit_result(result: Any, requested_batch: int) -> dict[str, Any]:
    metadata = result.metadata_dict()
    attempts = [int(value) for value in metadata["batch_size_attempts"]]
    if attempts != [value for value in EXPECTED_FALLBACK if value <= requested_batch][: len(attempts)]:
        raise BatchBError(f"Unexpected batch fallback sequence: {attempts}")
    return {
        "shape": list(result.embeddings.values.shape),
        "nnz": list(result.embeddings.nnz_per_row),
        "finite": True,
        "nonnegative": True,
        "latency_ms": result.latency_ms,
        "cpu_time_s": float(metadata["cpu_time_s"]),
        "peak_rss_bytes": int(metadata["peak_ram_bytes"]),
        "peak_vram_bytes": int(result.peak_vram_bytes or 0),
        "batch_size_requested": int(metadata["batch_size_requested"]),
        "batch_size_attempted": attempts,
        "batch_size_used": int(metadata["batch_size_used"]),
        "truncated_count": int(metadata["truncated_count"]),
        "max_observed_tokens": int(metadata["max_observed_tokens"]),
        "tokenizer_class": str(metadata["tokenizer_class"]),
    }


def run_gate(
    *,
    model_key: str,
    predeclaration: Mapping[str, Any],
    snapshot_evidence: Mapping[str, Any],
    data_root: str | Path,
    output_path: str | Path,
    device: str = "cuda:0",
    batch_size: int = 8,
    predeclaration_sha256: str | None = None,
    snapshot_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one canonical query and four documents without computing track metrics."""
    started = time.monotonic()
    if model_key not in BATCH_B_SELECTED_KEYS or batch_size != 8:
        raise BatchBError("Gate model order and requested batch size are frozen")
    materialization = validate_canonical_binding(predeclaration, data_root)
    snapshot_path, identity = validate_snapshot_evidence(model_key, snapshot_evidence)
    spec = BATCH_B_INVENTORY[model_key]
    data = load_materialized(data_root, "economics")
    query = sorted(data.queries, key=lambda row: row["id"])[0]
    relevant_id = sorted(data.qrels[query["id"]])[0]
    excluded = set(query.get("excluded_ids", ()))
    rows_by_id = {row["id"]: row for row in data.corpus}
    documents = [rows_by_id[relevant_id]]
    documents.extend(
        row for row in sorted(data.corpus, key=lambda row: row["id"])
        if row["id"] != relevant_id and row["id"] not in excluded
    )
    documents = documents[:4]
    provider = get_provider(
        spec.adapter,
        model_key=model_key,
        model_spec=spec,
        expected_snapshot_identity=identity,
        snapshot_path=str(snapshot_path),
        device=device,
        batch_size=batch_size,
        max_length=spec.max_length,
        gpu_cap_bytes=BATCH_B_GPU_CAP_BYTES,
    )
    document_result = provider.encode_sparse_documents(
        [row["content"] for row in documents], item_ids=[row["id"] for row in documents]
    )
    query_result = provider.encode_sparse_queries([query["text"]], item_ids=[query["id"]])
    _validate_csr(document_result.embeddings.values, rows=4, dimensions=spec.dimensions)
    _validate_csr(query_result.embeddings.values, rows=1, dimensions=spec.dimensions)
    if not document_result.embeddings.nnz_total or not query_result.embeddings.nnz_total:
        raise BatchBError("Gate produced an empty sparse side")
    ranking = ExactSparseIndex(document_result).search(query_result, k=4)
    query_audit = _audit_result(query_result, batch_size)
    document_audit = _audit_result(document_result, batch_size)
    peak_vram = max(query_audit["peak_vram_bytes"], document_audit["peak_vram_bytes"])
    if peak_vram > BATCH_B_GPU_CAP_BYTES:
        raise BatchBError("Gate exceeded the frozen VRAM cap")
    wall_time = time.monotonic() - started
    if wall_time > GATE_WALL_CAP_SECONDS:
        raise BatchBError("Gate exceeded the ten-minute wall-time cap")
    gate = {
        "schema_version": "learned-sparse-batch-b-gate-v1",
        "publication": PUBLICATION,
        "status": "go",
        "bindings": {
            "predeclaration_sha256": _require_sha256(predeclaration_sha256, "predeclaration SHA256"),
            "snapshot_evidence_sha256": _require_sha256(snapshot_evidence_sha256, "snapshot evidence SHA256"),
        },
        "model": {
            "key": model_key,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "license": spec.license,
            "query_route": spec.query_route,
            "document_route": spec.document_route,
            "dimensions": spec.dimensions,
            "max_length": spec.max_length,
        },
        "snapshot": {
            "actual_bytes": snapshot_evidence["actual_bytes"],
            "aggregate_identity_sha256": snapshot_evidence["aggregate_identity_sha256"],
            "files": identity,
        },
        "data": {
            "dataset_version": predeclaration["bindings"]["canonical_data"]["dataset_version"],
            "track": "economics",
            "query_count": 1,
            "document_count": 4,
            "query_ids_sha256": ids_sha256([query["id"]]),
            "document_ids_sha256": ids_sha256([row["id"] for row in documents]),
            "manifest_sha256": predeclaration["bindings"]["canonical_data"]["manifest_sha256"],
            "source_revision": materialization["source"]["revision"],
        },
        "query": query_audit,
        "documents": document_audit,
        "search": {
            "backend": ranking.backend,
            "exact": ranking.exact,
            "ranking_fingerprint": canonical_sha256(
                [[hit.rank, hit.item_id, hit.score] for hit in ranking.queries[0].hits]
            ),
        },
        "wall_time_s": wall_time,
    }
    _validate_schema(gate, GATE_SCHEMA)
    _write_json(Path(output_path), gate)
    return gate


def formal_protocol() -> dict[str, Any]:
    """Return the model-independent active formal execution protocol."""
    protocol = {
        "schema_version": "bright-learned-sparse-batch-b-protocol-v1",
        "active_predeclaration_sha256": ACTIVE_PREDECLARATION_SHA256,
        "dataset_version": "bright-nontechnical-pilot-v0.2",
        "canonical_data_manifest_sha256": "8174b0c01a32e977cf0c5d89522356c485196b0aa0571251cfe1595c5579cb1f",
        "tracks": list(FORMAL_TRACKS),
        "track_aggregation": "independent_macro_average_no_micro_average",
        "max_length": 512,
        "requested_batch_size": FORMAL_BATCH_SIZE,
        "batch_fallback": EXPECTED_FALLBACK,
        "document_chunk_size": FORMAL_CHUNK_SIZE,
        "representation": "canonical_scipy_csr_float32_finite_nonnegative",
        "dense_materialization": False,
        "search": {
            "backend": "scipy_csr_exact",
            "similarity": "inner_product",
            "top_k": FORMAL_TOP_K,
            "tie_break": "document_id_ascending",
        },
        "uncertainty": {
            "method": "query_level_bootstrap",
            "samples": FORMAL_BOOTSTRAP_SAMPLES,
            "seed": FORMAL_BOOTSTRAP_SEED,
        },
        "publication": PUBLICATION,
    }
    return {**protocol, "identity_sha256": canonical_sha256(protocol)}


def _formal_model_contract(
    model_key: str,
    snapshot_evidence: Mapping[str, Any],
    snapshot_evidence_sha256: str,
) -> dict[str, Any]:
    spec = BATCH_B_INVENTORY[model_key]
    return {
        "key": model_key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "license": spec.license,
        "provider": spec.adapter,
        "mechanism": MECHANISMS[model_key],
        "snapshot_evidence_sha256": snapshot_evidence_sha256,
        "snapshot_identity_sha256": snapshot_evidence["aggregate_identity_sha256"],
        "vocabulary_id": spec.vocabulary_id,
        "dimensions": spec.dimensions,
        "query_route": spec.query_route,
        "document_route": spec.document_route,
    }


def _formal_data_contract(
    materialization: Mapping[str, Any], data_root: Path, track: str, data: Any
) -> dict[str, Any]:
    return {
        "dataset_version": "bright-nontechnical-pilot-v0.2",
        "track": track,
        "documents": len(data.corpus),
        "queries": len(data.queries),
        "qrels": sum(len(values) for values in data.qrels.values()),
        "materialization_manifest_sha256": file_sha256(data_root / "manifest.json"),
        "source_revision": materialization["source"]["revision"],
        "selection_ids_sha256": materialization["tracks"][track]["selection_ids_sha256"],
        "corpus_sha256": materialization["tracks"][track]["files"]["corpus.jsonl"]["sha256"],
        "queries_sha256": materialization["tracks"][track]["files"]["queries.jsonl"]["sha256"],
        "qrels_sha256": materialization["tracks"][track]["files"]["qrels.jsonl"]["sha256"],
    }


def _formal_chunk_contract(model_key: str, track: str, snapshot_identity: str) -> dict[str, Any]:
    spec = BATCH_B_INVENTORY[model_key]
    return {
        "schema_version": "bright-learned-sparse-batch-b-chunk-v1",
        "protocol_identity_sha256": formal_protocol()["identity_sha256"],
        "active_predeclaration_sha256": ACTIVE_PREDECLARATION_SHA256,
        "dataset_version": "bright-nontechnical-pilot-v0.2",
        "model_key": model_key,
        "model_revision": spec.revision,
        "snapshot_identity_sha256": snapshot_identity,
        "track": track,
        "dimensions": spec.dimensions,
        "max_length": spec.max_length,
        "query_route": spec.query_route,
        "document_route": spec.document_route,
        "query_pruning": spec.query_pruning,
        "document_pruning": spec.document_pruning,
        "chunk_size": FORMAL_CHUNK_SIZE,
        "batch_size": FORMAL_BATCH_SIZE,
        "batch_fallback": EXPECTED_FALLBACK,
        "trust_remote_code": False,
    }


def _csr_memory_bytes(matrix: sparse.csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _formal_result(
    model: Mapping[str, Any], matrix: sparse.csr_matrix, item_ids: list[str], *, query: bool
) -> Any:
    role = batch_a.SparseEmbeddingRole.QUERY if query else batch_a.SparseEmbeddingRole.DOCUMENT
    return batch_a._result_from_manifest(dict(model), matrix, item_ids, role)


def _validate_formal_csr(matrix: sparse.csr_matrix, *, rows: int, dimensions: int, label: str) -> None:
    if not sparse.isspmatrix_csr(matrix) or matrix.shape != (rows, dimensions):
        raise BatchBError(f"{label} is not CSR with the frozen shape")
    if not matrix.has_sorted_indices or not matrix.has_canonical_format:
        raise BatchBError(f"{label} CSR is not canonical")
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise BatchBError(f"{label} CSR is not finite and nonnegative")


def _fallback_evidence(chunk_root: Path, query_metadata: Mapping[str, Any]) -> dict[str, Any]:
    chunks = []
    for path in sorted(chunk_root.glob("*.json")):
        audit = _load_json(path)
        attempts = [int(value) for value in audit.get("batch_size_attempts", [audit["batch_size_used"]])]
        if attempts != EXPECTED_FALLBACK[: len(attempts)]:
            raise BatchBError(f"Document chunk fallback order drifted: {path.name}")
        if len(attempts) > 1:
            chunks.append({"offset": audit["start"], "attempts": attempts, "used": audit["batch_size_used"]})
    query_attempts = [int(value) for value in query_metadata["batch_size_attempts"]]
    if query_attempts != EXPECTED_FALLBACK[: len(query_attempts)]:
        raise BatchBError("Query fallback order drifted")
    return {
        "document_fallback_chunks": chunks,
        "query_attempts": query_attempts,
        "query_used": int(query_metadata["batch_size_used"]),
        "batch1_oom": False,
        "terminal_failure": None,
    }


def run_formal_cell(
    *,
    model_key: str,
    track: str,
    predeclaration: Mapping[str, Any],
    predeclaration_sha256: str,
    snapshot_evidence: Mapping[str, Any],
    snapshot_evidence_sha256: str,
    data_root: str | Path,
    output_root: str | Path,
    device: str = "cuda:0",
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute a fresh formal cell or authenticate a complete finalized cell."""
    if predeclaration_sha256 != ACTIVE_PREDECLARATION_SHA256:
        raise BatchBError("Formal execution requires the unique active predeclaration identity")
    if model_key not in BATCH_B_SELECTED_KEYS or track not in FORMAL_TRACKS:
        raise BatchBError("Unsupported formal Batch-B cell")
    output = Path(output_root)
    manifest_path = output / "manifest.json"
    sidecar_path = output / "manifest.sha256"
    if manifest_path.exists() or sidecar_path.exists():
        if not manifest_path.is_file() or not sidecar_path.is_file():
            raise BatchBError("Finalized formal cell has an incomplete manifest identity pair")
        replayed = replay_formal_cell(
            result_root=output,
            data_root=data_root,
            predeclaration=predeclaration,
            predeclaration_sha256=predeclaration_sha256,
            snapshot_evidence=snapshot_evidence,
            snapshot_evidence_sha256=snapshot_evidence_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            require_cpu_only=True,
        )
        return replayed["manifest"]
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise BatchBError("Incomplete formal output is not externally authenticated and cannot resume")

    started = time.monotonic()
    materialization = validate_canonical_binding(predeclaration, data_root)
    data_path = Path(data_root)
    data = load_materialized(data_path, track)
    snapshot_path, identity = validate_snapshot_evidence(model_key, snapshot_evidence)
    if snapshot_evidence.get("predeclaration_sha256") != ACTIVE_PREDECLARATION_SHA256:
        raise BatchBError("Snapshot evidence is not bound to the active predeclaration")
    spec = BATCH_B_INVENTORY[model_key]
    provider = get_provider(
        spec.adapter,
        model_key=model_key,
        model_spec=spec,
        expected_snapshot_identity=identity,
        snapshot_path=str(snapshot_path),
        device=device,
        batch_size=FORMAL_BATCH_SIZE,
        max_length=spec.max_length,
        gpu_cap_bytes=BATCH_B_GPU_CAP_BYTES,
    )
    output.mkdir(parents=True, exist_ok=False)
    contract = _formal_chunk_contract(model_key, track, snapshot_evidence["aggregate_identity_sha256"])
    encoded_documents = batch_a.encode_document_chunks(
        provider, data.corpus, output / "document_chunks", contract
    )
    document_matrix = encoded_documents["matrix"]
    document_ids = list(encoded_documents["item_ids"])
    _validate_formal_csr(document_matrix, rows=len(data.corpus), dimensions=spec.dimensions, label="documents")
    document_result = batch_a._result_from_csr(
        provider, document_matrix, document_ids, batch_a.SparseEmbeddingRole.DOCUMENT
    )
    query_result = provider.encode_sparse_queries(
        [row["text"] for row in data.queries], item_ids=[row["id"] for row in data.queries]
    )
    query_matrix = query_result.embeddings.values
    query_ids = list(query_result.embeddings.item_ids)
    _validate_formal_csr(query_matrix, rows=len(data.queries), dimensions=spec.dimensions, label="queries")

    sparse.save_npz(output / "documents.npz", document_matrix, compressed=True)
    sparse.save_npz(output / "queries.npz", query_matrix, compressed=True)
    _write_json(output / "document_ids.json", document_ids)
    _write_json(output / "query_ids.json", query_ids)
    search_started = time.perf_counter()
    rankings = batch_a._rank(document_result, query_result, data)
    exact_search_s = time.perf_counter() - search_started
    per_query = query_metrics(data, rankings)
    _write_json(output / "rankings.json", batch_a._raw_ranking_rows(rankings))
    _write_json(output / "per_query_metrics.json", per_query)
    query_metadata = query_result.metadata_dict()
    fallback = _fallback_evidence(output / "document_chunks", query_metadata)
    document_audit = dict(encoded_documents["audit"])
    query_audit = {
        "count": len(query_ids),
        "latency_ms": query_result.latency_ms,
        "cpu_time_s": float(query_metadata["cpu_time_s"]),
        "truncated_count": int(query_metadata["truncated_count"]),
        "max_observed_tokens": int(query_metadata["max_observed_tokens"]),
        "encoding_observed_peak_rss_bytes": int(query_metadata["peak_ram_bytes"]),
        "peak_vram_bytes": int(query_result.peak_vram_bytes or 0),
        "tokenizer_class": str(query_metadata["tokenizer_class"]),
        "batch_size_requested": FORMAL_BATCH_SIZE,
        "batch_size_attempted": [int(value) for value in query_metadata["batch_size_attempts"]],
        "batch_size_used": int(query_metadata["batch_size_used"]),
        "throughput_items_per_second": len(query_ids) / (query_result.latency_ms / 1000.0),
    }
    document_audit["encoding_observed_peak_rss_bytes"] = document_audit.pop("peak_ram_bytes")
    document_audit["throughput_items_per_second"] = len(document_ids) / (document_audit["latency_ms"] / 1000.0)
    artifacts = (
        output / "documents.npz",
        output / "queries.npz",
        output / "document_ids.json",
        output / "query_ids.json",
        output / "rankings.json",
        output / "per_query_metrics.json",
        *sorted((output / "document_chunks").glob("*")),
    )
    protocol = formal_protocol()
    model = _formal_model_contract(model_key, snapshot_evidence, snapshot_evidence_sha256)
    manifest = {
        "schema_version": "bright-learned-sparse-batch-b-result-v1",
        "publication": PUBLICATION,
        "bindings": {
            "active_predeclaration_sha256": ACTIVE_PREDECLARATION_SHA256,
            "canonical_data_manifest_sha256": predeclaration["bindings"]["canonical_data"]["manifest_sha256"],
            "snapshot_evidence_sha256": snapshot_evidence_sha256,
            "snapshot_identity_sha256": snapshot_evidence["aggregate_identity_sha256"],
            "protocol_identity_sha256": protocol["identity_sha256"],
        },
        "model": model,
        "data": _formal_data_contract(materialization, data_path, track, data),
        "protocol": protocol,
        "config": contract,
        "metrics": aggregate_metrics(per_query),
        "confidence_intervals": bootstrap_confidence_intervals(
            per_query, samples=FORMAL_BOOTSTRAP_SAMPLES, seed=FORMAL_BOOTSTRAP_SEED
        ),
        "nnz": {
            "documents": batch_a._nnz_stats(document_matrix),
            "queries": batch_a._nnz_stats(query_matrix),
        },
        "representations": {
            "documents": {
                "shape": list(document_matrix.shape), "nnz": int(document_matrix.nnz),
                "item_ids_sha256": ids_sha256(document_ids), "csr_memory_bytes": _csr_memory_bytes(document_matrix),
                "npz_bytes": (output / "documents.npz").stat().st_size,
                "index_backend": "scipy_csr_exact", "index_memory_bytes": _csr_memory_bytes(document_matrix),
            },
            "queries": {
                "shape": list(query_matrix.shape), "nnz": int(query_matrix.nnz),
                "item_ids_sha256": ids_sha256(query_ids), "csr_memory_bytes": _csr_memory_bytes(query_matrix),
                "npz_bytes": (output / "queries.npz").stat().st_size,
            },
        },
        "resources": {
            "documents": document_audit,
            "queries": query_audit,
            "exact_search_s": exact_search_s,
            "gpu_wall_s": time.monotonic() - started,
            "fallback_oom": fallback,
        },
        "search": {
            "backend": "scipy_csr_exact", "exact": True, "similarity": "inner_product",
            "top_k": FORMAL_TOP_K, "tie_break": "document_id_ascending",
        },
        "artifacts": {
            path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in artifacts
        },
    }
    _validate_schema(manifest, FORMAL_RESULT_SCHEMA)
    _write_json(manifest_path, manifest)
    sidecar_path.write_text(file_sha256(manifest_path) + "\n", encoding="ascii")
    return manifest


def replay_formal_cell(
    *,
    result_root: str | Path,
    data_root: str | Path,
    predeclaration: Mapping[str, Any],
    predeclaration_sha256: str,
    snapshot_evidence: Mapping[str, Any],
    snapshot_evidence_sha256: str,
    expected_manifest_sha256: str | None,
    require_cpu_only: bool = True,
) -> dict[str, Any]:
    """Replay exact retrieval and metrics using saved CSR only, never a model."""
    if require_cpu_only and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise BatchBError("Formal replay requires CUDA_VISIBLE_DEVICES='' for CPU-only execution")
    expected = _require_sha256(expected_manifest_sha256, "raw manifest SHA256")
    if predeclaration_sha256 != ACTIVE_PREDECLARATION_SHA256:
        raise BatchBError("Replay predeclaration identity is not active")
    root = Path(result_root)
    manifest_path = root / "manifest.json"
    actual = file_sha256(manifest_path)
    if actual != expected:
        raise BatchBError("Raw manifest does not match the externally supplied identity")
    try:
        sidecar = (root / "manifest.sha256").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise BatchBError("Raw manifest sidecar is missing") from exc
    if sidecar != actual:
        raise BatchBError("Raw manifest sidecar identity mismatch")
    manifest = _load_json(manifest_path)
    _validate_schema(manifest, FORMAL_RESULT_SCHEMA)
    model_key = manifest["model"]["key"]
    track = manifest["data"]["track"]
    spec = BATCH_B_INVENTORY[model_key]
    if snapshot_evidence.get("key") != model_key:
        raise BatchBError("Replay snapshot evidence model mismatch")
    if aggregate_snapshot_identity(dict(snapshot_evidence["identity"])) != snapshot_evidence["aggregate_identity_sha256"]:
        raise BatchBError("Replay snapshot evidence aggregate identity drifted")
    protocol = formal_protocol()
    expected_bindings = {
        "active_predeclaration_sha256": ACTIVE_PREDECLARATION_SHA256,
        "canonical_data_manifest_sha256": predeclaration["bindings"]["canonical_data"]["manifest_sha256"],
        "snapshot_evidence_sha256": snapshot_evidence_sha256,
        "snapshot_identity_sha256": snapshot_evidence["aggregate_identity_sha256"],
        "protocol_identity_sha256": protocol["identity_sha256"],
    }
    if manifest["bindings"] != expected_bindings or manifest["protocol"] != protocol:
        raise BatchBError("Formal raw binding or protocol identity drifted")
    if manifest["model"] != _formal_model_contract(model_key, snapshot_evidence, snapshot_evidence_sha256):
        raise BatchBError("Formal raw model contract drifted")
    materialization = validate_canonical_binding(predeclaration, data_root)
    data_path = Path(data_root)
    data = load_materialized(data_path, track)
    if manifest["data"] != _formal_data_contract(materialization, data_path, track, data):
        raise BatchBError("Formal raw canonical data contract drifted")
    if manifest["config"] != _formal_chunk_contract(model_key, track, snapshot_evidence["aggregate_identity_sha256"]):
        raise BatchBError("Formal raw encoding contract drifted")
    observed_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path not in {manifest_path, root / "manifest.sha256"}
    }
    if observed_files != set(manifest["artifacts"]):
        raise BatchBError("Formal raw artifact file set drifted")
    for name, identity in manifest["artifacts"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise BatchBError("Formal raw artifact path is unsafe")
        path = root / name
        if path.stat().st_size != identity["bytes"] or file_sha256(path) != identity["sha256"]:
            raise BatchBError(f"Formal raw artifact identity mismatch: {name}")
    document_ids = _load_json(root / "document_ids.json")
    query_ids = _load_json(root / "query_ids.json")
    if document_ids != [row["id"] for row in data.corpus] or query_ids != [row["id"] for row in data.queries]:
        raise BatchBError("Formal raw canonical item ID order drifted")
    documents = sparse.load_npz(root / "documents.npz").tocsr()
    queries = sparse.load_npz(root / "queries.npz").tocsr()
    _validate_formal_csr(documents, rows=len(document_ids), dimensions=spec.dimensions, label="replay documents")
    _validate_formal_csr(queries, rows=len(query_ids), dimensions=spec.dimensions, label="replay queries")
    document_result = _formal_result(manifest["model"], documents, document_ids, query=False)
    query_result = _formal_result(manifest["model"], queries, query_ids, query=True)
    rankings = batch_a._rank(document_result, query_result, data)
    if _load_json(root / "rankings.json") != batch_a._raw_ranking_rows(rankings):
        raise BatchBError("Formal raw rankings do not match exact CPU replay")
    per_query = query_metrics(data, rankings)
    if _load_json(root / "per_query_metrics.json") != per_query:
        raise BatchBError("Formal raw per-query metrics do not match CPU replay")
    metrics = aggregate_metrics(per_query)
    confidence = bootstrap_confidence_intervals(
        per_query, samples=FORMAL_BOOTSTRAP_SAMPLES, seed=FORMAL_BOOTSTRAP_SEED
    )
    if manifest["metrics"] != metrics or manifest["confidence_intervals"] != confidence:
        raise BatchBError("Formal raw aggregate metrics or confidence intervals drifted")
    if manifest["nnz"] != {"documents": batch_a._nnz_stats(documents), "queries": batch_a._nnz_stats(queries)}:
        raise BatchBError("Formal raw sparsity statistics drifted")
    expected_representations = {
        "documents": {
            "shape": list(documents.shape), "nnz": int(documents.nnz),
            "item_ids_sha256": ids_sha256(document_ids), "csr_memory_bytes": _csr_memory_bytes(documents),
            "npz_bytes": (root / "documents.npz").stat().st_size,
            "index_backend": "scipy_csr_exact", "index_memory_bytes": _csr_memory_bytes(documents),
        },
        "queries": {
            "shape": list(queries.shape), "nnz": int(queries.nnz),
            "item_ids_sha256": ids_sha256(query_ids), "csr_memory_bytes": _csr_memory_bytes(queries),
            "npz_bytes": (root / "queries.npz").stat().st_size,
        },
    }
    if manifest["representations"] != expected_representations:
        raise BatchBError("Formal raw representation resource identity drifted")
    evidence = {
        "schema_version": "bright-learned-sparse-batch-b-replay-v1",
        "publication": PUBLICATION,
        "status": "pass",
        "cpu_only": require_cpu_only,
        "model_loaded": False,
        "raw_manifest_sha256": actual,
        "active_predeclaration_sha256": ACTIVE_PREDECLARATION_SHA256,
        "snapshot_evidence_sha256": snapshot_evidence_sha256,
        "protocol_identity_sha256": protocol["identity_sha256"],
        "model_key": model_key,
        "track": track,
        "artifact_map_sha256": canonical_sha256(manifest["artifacts"]),
        "metrics_sha256": canonical_sha256({"per_query": per_query, "aggregate": metrics, "confidence": confidence}),
        "metrics": metrics,
    }
    _validate_schema(evidence, FORMAL_REPLAY_SCHEMA)
    return {"manifest": manifest, "evidence": evidence}


def refuse_incomplete_resume(output_root: str | Path) -> None:
    """Reject any non-empty formal output unless a future authenticated manifest path is used."""
    root = Path(output_root)
    if root.exists() and any(path.is_file() for path in root.rglob("*")):
        raise BatchBError("Incomplete Batch-B output is not externally authenticated")


def assert_research_only(value: Mapping[str, Any]) -> None:
    if value.get("publication") != PUBLICATION:
        raise BatchBError("Batch-B evidence is not research-only/no-publish")


__all__ = [
    "BatchBError",
    "assert_research_only",
    "download_snapshot",
    "plan_snapshot",
    "formal_protocol",
    "refuse_incomplete_resume",
    "replay_formal_cell",
    "run_gate",
    "run_formal_cell",
    "validate_batch_a_readonly",
    "validate_canonical_binding",
    "validate_evidence_file",
    "validate_hub_metadata",
    "validate_predeclaration",
    "validate_snapshot_evidence",
    "validate_superseded_predeclaration",
    "validate_supersession",
    "write_evidence",
]
