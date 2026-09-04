"""Formal research-only BRIGHT real multi-vector benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import re
import resource
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.benchmark.bright_multidomain_v02 import load_materialized, validate_materialization
from mm_embed.benchmark.retrieval_v01 import (
    aggregate_metrics,
    bootstrap_confidence_intervals,
    query_metrics,
)
from mm_embed.indexes.multi_vector_exact import ExactMaxSimIndex
from mm_embed.providers.multi_vector_base import MultiVectorRole
from mm_embed.providers.real_multi_vector import (
    INVENTORY,
    SELECTED_KEYS,
    SentenceTransformerMultiVectorProvider,
    file_sha256,
    resolve_snapshot,
    validate_hub_metadata,
)

ACTIVE_PREDECLARATION_SHA256 = "2ef6bbc245d9f196d71adcd8498575b14ecd650c29242fb9d0eb284704c16f05"
ORIGINAL_PRE_SCORE_PREDECLARATION_SHA256 = "ec9c100e8470e6dd754fe87388dbd5bf9562ab62fdf12d6daac25ba8a92c761f"
CANONICAL_DATA_MANIFEST_SHA256 = "8174b0c01a32e977cf0c5d89522356c485196b0aa0571251cfe1595c5579cb1f"
TRACKS = ("economics", "psychology")
TOP_K = 100
WORD_PATTERN = re.compile(r"(?u)\b\w+\b")
WINDOW_WORDS = 128
STRIDE_WORDS = 96
PUBLICATION = {
    "classification": "research_only",
    "gate": "closed",
    "publish": False,
    "leaderboard_publish": False,
    "public_export_allowed": False,
}
GATE_QUERY = "Which biological process lets green plants convert light energy into stored chemical energy?"
GATE_DOCUMENTS = (
    "Photosynthesis lets green plants use light to form energy-rich sugars from carbon dioxide and water.",
    "Cellular respiration releases usable energy by breaking down sugars in cells.",
    "Evaporation changes liquid water into water vapor when heat is supplied.",
    "Fermentation is an anaerobic pathway that converts sugars into other compounds.",
)


class BrightMultiVectorError(ValueError):
    """Raised when a frozen multi-vector benchmark invariant is violated."""


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def write_json_with_hash(path: str | Path, value: Mapping[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value)
    output.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def load_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BrightMultiVectorError(f"Missing or invalid JSON artifact: {path}") from exc


def validate_predeclaration(path: str | Path, expected_sha256: str | None) -> dict[str, Any]:
    """Validate the immutable pre-score declaration against an external identity."""
    if expected_sha256 != ACTIVE_PREDECLARATION_SHA256:
        raise BrightMultiVectorError("Expected predeclaration is not the unique active frozen identity")
    artifact = Path(path)
    if file_sha256(artifact) != expected_sha256:
        raise BrightMultiVectorError("Predeclaration content drifted")
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    if sidecar.read_text(encoding="ascii").strip() != expected_sha256:
        raise BrightMultiVectorError("Predeclaration sidecar drifted")
    value = load_json(artifact)
    if value.get("schema_version") != "bright-real-multi-vector-predeclaration-v2":
        raise BrightMultiVectorError("Active predeclaration is not the chronology-corrected v2 artifact")
    chronology_binding = value.get("chronology", {})
    if (
        value.get("formal_scores_observed") is not True
        or chronology_binding.get("issued_after_prior_scores_observed") is not True
        or chronology_binding.get("model_or_protocol_reselection") is not False
        or chronology_binding.get("original_pre_score_predeclaration_sha256")
        != ORIGINAL_PRE_SCORE_PREDECLARATION_SHA256
    ):
        raise BrightMultiVectorError("Chronology reissue does not transparently bind the pre-score contract")
    chronology_path = artifact.with_name("chronology.json")
    chronology_sha256 = file_sha256(chronology_path)
    if (
        chronology_sha256 != chronology_binding.get("chronology_attestation_sha256")
        or chronology_path.with_suffix(".json.sha256").read_text(encoding="ascii").strip() != chronology_sha256
    ):
        raise BrightMultiVectorError("Chronology attestation identity drifted")
    chronology = load_json(chronology_path)
    original_path = artifact.parent.parent / "bright-multi-vector-v0.1" / "predeclaration.json"
    if (
        file_sha256(original_path) != ORIGINAL_PRE_SCORE_PREDECLARATION_SHA256
        or original_path.with_suffix(".json.sha256").read_text(encoding="ascii").strip()
        != ORIGINAL_PRE_SCORE_PREDECLARATION_SHA256
        or chronology["original_pre_score_artifact"]["sha256"] != ORIGINAL_PRE_SCORE_PREDECLARATION_SHA256
        or chronology["original_pre_score_artifact"]["formal_scores_observed"] is not False
        or chronology["correction"]["models_changed"] is not False
        or chronology["correction"]["protocol_changed"] is not False
    ):
        raise BrightMultiVectorError("Original pre-score identity or chronology correction drifted")
    original = load_json(original_path)
    semantic = dict(value)
    semantic.pop("created_at")
    semantic.pop("chronology")
    semantic["schema_version"] = "bright-real-multi-vector-predeclaration-v1"
    semantic["formal_scores_observed"] = False
    if (
        hashlib.sha256(canonical_bytes(semantic)).hexdigest()
        != chronology["original_pre_score_artifact"]["semantic_contract_sha256"]
    ):
        raise BrightMultiVectorError("Reissued model/protocol contract differs from the pre-score artifact")
    original_frozen = datetime.fromisoformat(
        chronology["original_pre_score_artifact"]["observed_filesystem_mtime"].replace("Z", "+00:00")
    )
    first_gate = datetime.fromisoformat(chronology["first_gate"]["observed_filesystem_mtime"].replace("Z", "+00:00"))
    reissued = datetime.fromisoformat(value["created_at"].replace("Z", "+00:00"))
    if original_frozen >= first_gate or reissued.tzinfo != UTC or reissued.timestamp() > artifact.stat().st_mtime + 1.0:
        raise BrightMultiVectorError("Predeclaration filesystem chronology is inconsistent")
    if original.get("formal_scores_observed") is not False:
        raise BrightMultiVectorError("Original pre-score declaration was not score blind")
    if tuple(row["key"] for row in value.get("models", [])) != SELECTED_KEYS:
        raise BrightMultiVectorError("Selected model order drifted")
    for row in value["models"]:
        spec = INVENTORY[row["key"]]
        required = {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "license": spec.license,
            "gated": spec.gated,
            "private": spec.private,
            "dimensions": spec.dimensions,
            "query_length": spec.query_length,
            "document_length": spec.document_length,
            "query_route": spec.query_route,
            "document_route": spec.document_route,
            "trust_remote_code": False,
            "download_cap_bytes": spec.download_cap_bytes,
        }
        if {key: row[key] for key in required} != required:
            raise BrightMultiVectorError(f"Frozen model contract drifted for {spec.key}")
    return value


def validate_canonical_data(predeclaration: Mapping[str, Any], data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root)
    expected = predeclaration["bindings"]["canonical_data"]
    if file_sha256(root / "manifest.json") != expected["manifest_sha256"]:
        raise BrightMultiVectorError("Canonical data manifest drifted")
    manifest = validate_materialization(root)
    if manifest["source"]["revision"] != expected["source_revision"]:
        raise BrightMultiVectorError("Canonical source revision drifted")
    return manifest


def passage_windows(text: str) -> list[str]:
    """Split a canonical passage into deterministic overlapping source substrings."""
    matches = list(WORD_PATTERN.finditer(text))
    if len(matches) <= WINDOW_WORDS:
        return [text]
    starts = list(range(0, len(matches) - WINDOW_WORDS + 1, STRIDE_WORDS))
    tail = len(matches) - WINDOW_WORDS
    if starts[-1] != tail:
        starts.append(tail)
    spans = []
    for start in starts:
        begin = matches[start].start()
        end = matches[start + WINDOW_WORDS - 1].end()
        span = text[begin:end]
        if not spans or span != spans[-1]:
            spans.append(span)
    return spans


def _numpy_maxsim(query: np.ndarray, document: np.ndarray) -> float:
    return float(np.max(query @ document.T, axis=1).sum(dtype=np.float32))


def run_gate(
    model_key: str,
    *,
    predeclaration_path: str | Path,
    predeclaration_sha256: str,
    output_path: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run the required one-query/four-document gate for one frozen checkpoint."""
    started = time.perf_counter()
    predeclaration = validate_predeclaration(predeclaration_path, predeclaration_sha256)
    if model_key not in SELECTED_KEYS:
        raise BrightMultiVectorError("Gate model is outside the frozen selected set")
    hub = validate_hub_metadata(model_key)
    snapshot_path, snapshot = resolve_snapshot(model_key)
    provider = SentenceTransformerMultiVectorProvider(model_key, snapshot_path, device=device, batch_size=4)
    documents = provider.encode_multi_vector_passages(
        GATE_DOCUMENTS,
        passage_ids=tuple(f"gate-p{index}" for index in range(4)),
        document_ids=tuple(f"gate-d{index}" for index in range(4)),
    )
    query = provider.encode_multi_vector_query(GATE_QUERY, item_id="gate-q0")
    index = ExactMaxSimIndex(documents, aggregation="max_passage", max_cells=100_000)
    hits = index.search(query, k=4)
    q = query.embeddings.values[0, query.embeddings.mask[0]]
    independent = [
        _numpy_maxsim(q, documents.embeddings.values[row, documents.embeddings.mask[row]]) for row in range(4)
    ]
    project_scores = {int(hit.document_id.removeprefix("gate-d")): hit.score for hit in hits}
    max_delta = max(abs(project_scores[index] - independent[index]) for index in range(4))
    if max_delta > 1e-5:
        raise BrightMultiVectorError("Project and independent NumPy MaxSim disagree")
    if hits[0].document_id != "gate-d0" or not (hits[0].score > hits[1].score):
        raise BrightMultiVectorError("Synthetic gate positive is not uniquely rank one")
    if documents.embeddings.mask.dtype != np.bool_ or not np.any(~documents.embeddings.mask):
        raise BrightMultiVectorError("Gate did not exercise explicit boolean padding")
    if np.any(documents.embeddings.values[~documents.embeddings.mask] != 0):
        raise BrightMultiVectorError("Adapter padding is not zero")
    elapsed = time.perf_counter() - started
    if elapsed > predeclaration["gate"]["wall_cap_seconds_per_model"]:
        raise BrightMultiVectorError("Gate exceeded the frozen wall cap")
    evidence = {
        "schema_version": "bright-real-multi-vector-gate-v1",
        "story_id": "S-20260814-012",
        "predeclaration_sha256": predeclaration_sha256,
        "model": {
            "key": model_key,
            "repo_id": INVENTORY[model_key].repo_id,
            "revision": INVENTORY[model_key].revision,
            "license": hub["license"],
            "gated": hub["gated"],
            "private": hub["private"],
            "trust_remote_code": False,
            "dimension": INVENTORY[model_key].dimensions,
        },
        "snapshot": {key: value for key, value in snapshot.items() if key != "cache_root"},
        "loaded_semantics": provider.semantic_evidence,
        "gate": {
            "queries": 1,
            "documents": 4,
            "query_token_vectors": int(query.embeddings.mask.sum()),
            "document_token_vectors": [int(value) for value in documents.embeddings.mask.sum(axis=1)],
            "padding_mask_false": int((~documents.embeddings.mask).sum()),
            "positive_rank": 1,
            "positive_margin": float(hits[0].score - hits[1].score),
            "maxsim_independent_max_abs_delta": max_delta,
            "all_vectors_finite": True,
            "all_scores_finite": all(math.isfinite(value) for value in independent),
            "wall_seconds": elapsed,
            "peak_vram_bytes": max(query.peak_vram_bytes or 0, documents.peak_vram_bytes or 0),
        },
        "publication": PUBLICATION,
        "passed": True,
    }
    write_json_with_hash(output_path, evidence)
    return evidence


def validate_gate(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    artifact = Path(path)
    actual = file_sha256(artifact)
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256").read_text(encoding="ascii").strip()
    if sidecar != actual or (expected_sha256 is not None and expected_sha256 != actual):
        raise BrightMultiVectorError("Gate identity drifted")
    value = load_json(artifact)
    if value.get("predeclaration_sha256") != ACTIVE_PREDECLARATION_SHA256 or value.get("passed") is not True:
        raise BrightMultiVectorError("Gate is not a passed result bound to the active freeze")
    key = value["model"]["key"]
    spec = INVENTORY[key]
    if value["model"]["revision"] != spec.revision or value["model"]["dimension"] != spec.dimensions:
        raise BrightMultiVectorError("Gate model identity drifted")
    return value


def build_gate_summary(gate_paths: Sequence[str | Path], output_path: str | Path) -> dict[str, Any]:
    gates = [validate_gate(path) for path in gate_paths]
    if tuple(value["model"]["key"] for value in gates) != SELECTED_KEYS:
        raise BrightMultiVectorError("Gate set does not cover the frozen model order")
    summary = {
        "schema_version": "bright-real-multi-vector-gate-summary-v1",
        "story_id": "S-20260814-012",
        "predeclaration_sha256": ACTIVE_PREDECLARATION_SHA256,
        "models": [
            {
                "key": value["model"]["key"],
                "repo_id": value["model"]["repo_id"],
                "revision": value["model"]["revision"],
                "license": value["model"]["license"],
                "dimension": value["model"]["dimension"],
                "snapshot_identity_sha256": value["snapshot"]["snapshot_identity_sha256"],
                "snapshot_bytes": value["snapshot"]["snapshot_bytes"],
                "incremental_download_bytes": value["snapshot"]["incremental_download_bytes"],
                "query_token_vectors": value["gate"]["query_token_vectors"],
                "document_token_vectors": value["gate"]["document_token_vectors"],
                "positive_rank": value["gate"]["positive_rank"],
                "positive_margin": value["gate"]["positive_margin"],
                "maxsim_max_abs_delta": value["gate"]["maxsim_independent_max_abs_delta"],
                "peak_vram_bytes": value["gate"]["peak_vram_bytes"],
                "wall_seconds": value["gate"]["wall_seconds"],
                "passed": True,
            }
            for value in gates
        ],
        "formal_scoring_authorized": True,
        "publication": PUBLICATION,
    }
    write_json_with_hash(output_path, summary)
    return summary


def process_rss_bytes() -> int:
    """Return Linux ru_maxrss in bytes for the current process."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise BrightMultiVectorError("Latency/vector statistics require finite observations")
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _encode_chunks(
    provider: SentenceTransformerMultiVectorProvider,
    texts: Sequence[str],
    *,
    role: MultiVectorRole,
    chunk_size: int,
    fallback: Sequence[int],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Encode bounded chunks with a semantics-preserving OOM batch fallback."""
    import torch

    arrays: list[np.ndarray] = []
    chunk_amortized_ms: list[float] = []
    chunk_wall_ms: list[float] = []
    used_batches: list[int] = []
    attempted_batches: list[int] = []
    oom_failures = 0
    retry_count = 0
    peak_vram = 0
    total_ms = 0.0
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        last_error: Exception | None = None
        for batch in fallback:
            attempted_batches.append(batch)
            try:
                encoded, elapsed_ms, observed_peak = provider.encode_compact(chunk, role=role, batch_size=batch)
                arrays.extend(encoded)
                chunk_amortized_ms.append(elapsed_ms / len(chunk))
                chunk_wall_ms.append(elapsed_ms)
                used_batches.append(batch)
                peak_vram = max(peak_vram, observed_peak or 0)
                total_ms += elapsed_ms
                break
            except torch.OutOfMemoryError as exc:
                last_error = exc
                oom_failures += 1
                retry_count += 1
                torch.cuda.empty_cache()
        else:
            raise BrightMultiVectorError("All frozen encoding batch fallbacks exhausted") from last_error
    if len(arrays) != len(texts):
        raise BrightMultiVectorError("Encoding did not cover every input exactly once")
    return arrays, {
        "items": len(texts),
        "chunks": len(chunk_amortized_ms),
        "wall_ms": total_ms,
        "throughput_per_second": len(texts) / (total_ms / 1000.0),
        "chunk_amortized_per_item_ms": _percentiles(chunk_amortized_ms),
        "chunk_wall_ms": _percentiles(chunk_wall_ms),
        "latency_scope": "Each observation is one chunk wall time divided by items in that chunk; these are not individually timed items.",
        "requested_batch": int(fallback[0]),
        "used_batch_min": min(used_batches),
        "fallback_triggered": min(used_batches) != fallback[0],
        "attempt_counts": {
            str(batch): attempted_batches.count(batch) for batch in fallback if batch in attempted_batches
        },
        "successful_batch_counts": {
            str(batch): used_batches.count(batch) for batch in fallback if batch in used_batches
        },
        "retry_count": retry_count,
        "failure": {
            "oom_count": oom_failures,
            "other_failure_count": 0,
            "terminal_failure": None,
        },
        "peak_vram_bytes": peak_vram,
    }


def _save_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, value, allow_pickle=False)
    digest = file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(digest + "\n", encoding="ascii")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "sidecar": {"path": sidecar.name, "bytes": sidecar.stat().st_size, "sha256": file_sha256(sidecar)},
    }


def _save_json(path: Path, value: object) -> dict[str, Any]:
    data = canonical_bytes(value)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(digest + "\n", encoding="ascii")
    return {
        "path": path.name,
        "bytes": len(data),
        "sha256": digest,
        "sidecar": {"path": sidecar.name, "bytes": sidecar.stat().st_size, "sha256": file_sha256(sidecar)},
    }


def _save_ragged(root: Path, prefix: str, arrays: Sequence[np.ndarray]) -> dict[str, Any]:
    if not arrays:
        raise BrightMultiVectorError("Cannot save an empty representation")
    dimensions = {array.shape[1] for array in arrays}
    if len(dimensions) != 1:
        raise BrightMultiVectorError("Ragged representation dimensions drifted")
    counts = np.asarray([array.shape[0] for array in arrays], dtype=np.int64)
    offsets = np.empty(len(arrays) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    values = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    mask = np.ones(values.shape[0], dtype=np.bool_)
    if not np.all(np.isfinite(values)) or np.any(counts <= 0):
        raise BrightMultiVectorError("Representation contains invalid vectors")
    return {
        "values": _save_array(root / f"{prefix}.values.npy", values),
        "offsets": _save_array(root / f"{prefix}.offsets.npy", offsets),
        "mask": _save_array(root / f"{prefix}.mask.npy", mask),
        "items": len(arrays),
        "token_vectors": int(values.shape[0]),
        "dimensions": int(values.shape[1]),
        "token_count": _percentiles(counts.astype(np.float64)),
    }


def _windowed_corpus(data: Any) -> tuple[list[str], list[int], list[int]]:
    texts: list[str] = []
    document_indices: list[int] = []
    ordinals: list[int] = []
    for document_index, row in enumerate(data.corpus):
        for ordinal, text in enumerate(passage_windows(row["content"])):
            texts.append(text)
            document_indices.append(document_index)
            ordinals.append(ordinal)
    return texts, document_indices, ordinals


def validate_gate_summary(path: str | Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("predeclaration_sha256") != ACTIVE_PREDECLARATION_SHA256:
        raise BrightMultiVectorError("Gate summary is not bound to the active freeze")
    if value.get("formal_scoring_authorized") is not True:
        raise BrightMultiVectorError("Gate summary does not authorize formal scoring")
    if tuple(row["key"] for row in value.get("models", [])) != SELECTED_KEYS:
        raise BrightMultiVectorError("Gate summary model order drifted")
    sidecar = Path(path).with_suffix(Path(path).suffix + ".sha256")
    if sidecar.read_text(encoding="ascii").strip() != file_sha256(Path(path)):
        raise BrightMultiVectorError("Gate summary sidecar drifted")
    return value


def encode_formal_model(
    model_key: str,
    *,
    predeclaration_path: str | Path,
    predeclaration_sha256: str,
    gate_summary_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Encode both formal tracks and save complete replayable token vectors."""
    predeclaration = validate_predeclaration(predeclaration_path, predeclaration_sha256)
    validate_gate_summary(gate_summary_path)
    canonical = validate_canonical_data(predeclaration, data_root)
    if model_key not in SELECTED_KEYS:
        raise BrightMultiVectorError("Formal model is outside the frozen selected set")
    gate_summary_sha256 = file_sha256(Path(gate_summary_path))
    model_root = Path(output_root) / model_key
    if model_root.exists():
        if not any(model_root.iterdir()):
            raise BrightMultiVectorError("Incomplete empty model encoding cannot be resumed")
        validated = validate_model_encoding(
            model_root,
            model_key=model_key,
            predeclaration_sha256=predeclaration_sha256,
            gate_summary_sha256=gate_summary_sha256,
            data_root=data_root,
        )
        return validated["manifest"]
    snapshot_path, snapshot = resolve_snapshot(model_key)
    provider = SentenceTransformerMultiVectorProvider(model_key, snapshot_path, device=device)
    model_root.mkdir(parents=True, exist_ok=False)
    track_entries: dict[str, Any] = {}
    started = time.perf_counter()
    for track in TRACKS:
        track_started = time.perf_counter()
        data = load_materialized(data_root, track)
        window_texts, document_indices, ordinals = _windowed_corpus(data)
        query_texts = [row["text"] for row in data.queries]
        document_arrays, document_execution = _encode_chunks(
            provider,
            window_texts,
            role=MultiVectorRole.DOCUMENT,
            chunk_size=256,
            fallback=(32, 16, 8, 4, 2, 1),
        )
        query_arrays, query_execution = _encode_chunks(
            provider,
            query_texts,
            role=MultiVectorRole.QUERY,
            chunk_size=128,
            fallback=(64, 32, 16, 8, 4, 2, 1),
        )
        track_root = model_root / track
        track_root.mkdir(parents=True, exist_ok=False)
        documents = _save_ragged(track_root, "documents", document_arrays)
        queries = _save_ragged(track_root, "queries", query_arrays)
        window_counts = np.bincount(np.asarray(document_indices), minlength=len(data.corpus)).astype(float)
        files = {
            "document_indices": _save_array(
                track_root / "window_document_indices.npy",
                np.asarray(document_indices, dtype=np.int32),
            ),
            "window_ordinals": _save_array(track_root / "window_ordinals.npy", np.asarray(ordinals, dtype=np.int32)),
            "document_ids": _save_json(track_root / "document_ids.json", [row["id"] for row in data.corpus]),
            "query_ids": _save_json(track_root / "query_ids.json", [row["id"] for row in data.queries]),
        }
        entry = {
            "schema_version": "bright-real-multi-vector-representation-v1",
            "story_id": "S-20260814-012",
            "predeclaration_sha256": predeclaration_sha256,
            "gate_summary_sha256": file_sha256(Path(gate_summary_path)),
            "model": {
                "key": model_key,
                "repo_id": provider.spec.repo_id,
                "revision": provider.spec.revision,
                "dimension": provider.spec.dimensions,
                "snapshot_identity_sha256": snapshot["snapshot_identity_sha256"],
                "snapshot_bytes": snapshot["snapshot_bytes"],
                "loaded_semantics": provider.semantic_evidence,
            },
            "data": {
                "track": track,
                "documents": len(data.corpus),
                "queries": len(data.queries),
                "qrels": sum(len(value) for value in data.qrels.values()),
                "materialization_manifest_sha256": CANONICAL_DATA_MANIFEST_SHA256,
                "selection_ids_sha256": canonical["tracks"][track]["selection_ids_sha256"],
            },
            "windowing": {
                "window_words": WINDOW_WORDS,
                "stride_words": STRIDE_WORDS,
                "windows": len(window_texts),
                "multi_window_documents": int(np.count_nonzero(window_counts > 1)),
                "windows_per_document": _percentiles(window_counts),
                "aggregation": "max_passage",
            },
            "representations": {"documents": documents, "queries": queries, "files": files},
            "encoding": {
                "documents": document_execution,
                "queries": query_execution,
                "process_maxrss_bytes": process_rss_bytes(),
                "track_wall_seconds": time.perf_counter() - track_started,
            },
            "publication": PUBLICATION,
            "finalized": True,
        }
        identity = write_json_with_hash(track_root / "representation.json", entry)
        track_entries[track] = {"representation_sha256": identity, "path": str(track_root)}
    model_manifest = {
        "schema_version": "bright-real-multi-vector-model-encoding-v1",
        "story_id": "S-20260814-012",
        "predeclaration_sha256": predeclaration_sha256,
        "gate_summary_sha256": gate_summary_sha256,
        "model_key": model_key,
        "tracks": track_entries,
        "model_wall_seconds": time.perf_counter() - started,
        "publication": PUBLICATION,
        "finalized": True,
    }
    write_json_with_hash(model_root / "encoding-manifest.json", model_manifest)
    return validate_model_encoding(
        model_root,
        model_key=model_key,
        predeclaration_sha256=predeclaration_sha256,
        gate_summary_sha256=gate_summary_sha256,
        data_root=data_root,
    )["manifest"]


def _validate_saved_entry(root: Path, entry: Mapping[str, Any]) -> Path:
    relative = Path(str(entry["path"]))
    if relative.is_absolute() or relative.name != str(entry["path"]) or ".." in relative.parts:
        raise BrightMultiVectorError("Saved representation contains an unsafe path")
    path = root / relative
    if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
        raise BrightMultiVectorError("Saved representation file identity drifted")
    sidecar_entry = entry.get("sidecar")
    if not isinstance(sidecar_entry, Mapping):
        raise BrightMultiVectorError("Saved representation file lacks an authenticated sidecar")
    sidecar = root / str(sidecar_entry["path"])
    if (
        sidecar.name != sidecar_entry["path"]
        or sidecar.stat().st_size != sidecar_entry["bytes"]
        or file_sha256(sidecar) != sidecar_entry["sha256"]
        or sidecar.read_text(encoding="ascii").strip() != entry["sha256"]
    ):
        raise BrightMultiVectorError("Saved representation sidecar identity drifted")
    return path


def _load_ragged(root: Path, section: Mapping[str, Any]) -> list[np.ndarray]:
    values_entry = section["values"]
    offsets_entry = section["offsets"]
    mask_entry = section["mask"]
    for entry in (values_entry, offsets_entry, mask_entry):
        _validate_saved_entry(root, entry)
    values = np.load(root / values_entry["path"], mmap_mode="r", allow_pickle=False)
    offsets = np.load(root / offsets_entry["path"], allow_pickle=False)
    mask = np.load(root / mask_entry["path"], mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float32 or offsets.dtype != np.int64 or mask.dtype != np.bool_:
        raise BrightMultiVectorError("Saved representation dtype drifted")
    if len(mask) != len(values) or not np.all(mask) or offsets[0] != 0 or offsets[-1] != len(values):
        raise BrightMultiVectorError("Saved compact mask/offset contract drifted")
    if np.any(np.diff(offsets) <= 0) or not np.all(np.isfinite(values)):
        raise BrightMultiVectorError("Saved representation contains empty or non-finite vectors")
    counts = np.diff(offsets).astype(np.float64)
    if (
        section["items"] != len(counts)
        or section["token_vectors"] != len(values)
        or section["dimensions"] != values.shape[1]
        or section["token_count"] != _percentiles(counts)
    ):
        raise BrightMultiVectorError("Saved representation statistics drifted")
    return [np.asarray(values[offsets[i] : offsets[i + 1]]) for i in range(len(offsets) - 1)]


def validate_representation_artifact(
    root: str | Path,
    *,
    model_key: str,
    track: str,
    predeclaration_sha256: str,
    gate_summary_sha256: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate one complete replayable representation without loading a model."""
    representation_root = Path(root)
    manifest_path = representation_root / "representation.json"
    manifest_sha256 = file_sha256(manifest_path)
    sidecar_path = manifest_path.with_suffix(".json.sha256")
    if sidecar_path.read_text(encoding="ascii").strip() != manifest_sha256:
        raise BrightMultiVectorError("Representation manifest sidecar drifted")
    value = load_json(manifest_path)
    if (
        value.get("schema_version") != "bright-real-multi-vector-representation-v1"
        or value.get("finalized") is not True
        or value.get("publication") != PUBLICATION
        or value.get("predeclaration_sha256") != predeclaration_sha256
        or value.get("gate_summary_sha256") != gate_summary_sha256
        or value["model"]["key"] != model_key
        or value["model"]["revision"] != INVENTORY[model_key].revision
        or value["model"]["dimension"] != INVENTORY[model_key].dimensions
        or value["data"]["track"] != track
    ):
        raise BrightMultiVectorError("Representation manifest contract drifted")
    documents = _load_ragged(representation_root, value["representations"]["documents"])
    queries = _load_ragged(representation_root, value["representations"]["queries"])
    files = value["representations"]["files"]
    paths = {key: _validate_saved_entry(representation_root, entry) for key, entry in files.items()}
    document_indices = np.load(paths["document_indices"], allow_pickle=False)
    ordinals = np.load(paths["window_ordinals"], allow_pickle=False)
    document_ids = load_json(paths["document_ids"])
    query_ids = load_json(paths["query_ids"])
    if document_indices.dtype != np.int32 or ordinals.dtype != np.int32:
        raise BrightMultiVectorError("Window mapping dtype drifted")
    if (
        document_indices.ndim != 1
        or ordinals.ndim != 1
        or len(document_indices) != len(documents)
        or len(ordinals) != len(documents)
        or len(document_ids) != value["data"]["documents"]
        or len(query_ids) != value["data"]["queries"]
        or len(queries) != len(query_ids)
        or len(document_ids) != len(set(document_ids))
        or len(query_ids) != len(set(query_ids))
        or np.any(document_indices < 0)
        or np.any(document_indices >= len(document_ids))
        or np.any(ordinals < 0)
    ):
        raise BrightMultiVectorError("Representation mapping shape or identity coverage drifted")
    for document_index in range(len(document_ids)):
        observed = ordinals[document_indices == document_index]
        if not np.array_equal(observed, np.arange(len(observed), dtype=np.int32)):
            raise BrightMultiVectorError("Window ordinals are not contiguous per canonical passage")
    if data_root is not None:
        data = load_materialized(data_root, track)
        if document_ids != [row["id"] for row in data.corpus] or query_ids != [row["id"] for row in data.queries]:
            raise BrightMultiVectorError("Saved canonical ID order drifted")
    component_entries = [
        value["representations"][role][kind]
        for role in ("documents", "queries")
        for kind in ("values", "offsets", "mask")
    ] + list(files.values())
    expected_files = {"representation.json", "representation.json.sha256"}
    for entry in component_entries:
        expected_files.add(str(entry["path"]))
        expected_files.add(str(entry["sidecar"]["path"]))
    observed_files = {path.name for path in representation_root.iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise BrightMultiVectorError("Representation directory contains an unexpected or missing file")
    token_array_bytes = sum(
        int(value["representations"][role][kind]["bytes"])
        for role in ("documents", "queries")
        for kind in ("values", "offsets", "mask")
    )
    mapping_identity_bytes = sum(int(entry["bytes"]) for entry in files.values())
    sidecar_bytes = sidecar_path.stat().st_size + sum(int(entry["sidecar"]["bytes"]) for entry in component_entries)
    storage = {
        "token_array_bytes": token_array_bytes,
        "mapping_identity_bytes": mapping_identity_bytes,
        "representation_manifest_bytes": manifest_path.stat().st_size,
        "complete_replayable_non_sidecar_bytes": token_array_bytes
        + mapping_identity_bytes
        + manifest_path.stat().st_size,
        "sidecar_bytes": sidecar_bytes,
        "complete_replayable_with_sidecars_bytes": token_array_bytes
        + mapping_identity_bytes
        + manifest_path.stat().st_size
        + sidecar_bytes,
    }
    return {
        "manifest": value,
        "manifest_sha256": manifest_sha256,
        "manifest_sidecar": {
            "path": sidecar_path.name,
            "bytes": sidecar_path.stat().st_size,
            "sha256": file_sha256(sidecar_path),
        },
        "storage": storage,
        "files": {
            entry["path"]: {
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "sidecar": dict(entry["sidecar"]),
            }
            for entry in component_entries
        },
    }


def validate_model_encoding(
    model_root: str | Path,
    *,
    model_key: str,
    predeclaration_sha256: str,
    gate_summary_sha256: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate one finalized two-track encoding and every component identity."""
    root = Path(model_root)
    manifest_path = root / "encoding-manifest.json"
    manifest_sha256 = file_sha256(manifest_path)
    sidecar = manifest_path.with_suffix(".json.sha256")
    if sidecar.read_text(encoding="ascii").strip() != manifest_sha256:
        raise BrightMultiVectorError("Encoding manifest sidecar drifted")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != "bright-real-multi-vector-model-encoding-v1"
        or manifest.get("finalized") is not True
        or manifest.get("publication") != PUBLICATION
        or manifest.get("predeclaration_sha256") != predeclaration_sha256
        or manifest.get("gate_summary_sha256") != gate_summary_sha256
        or manifest.get("model_key") != model_key
        or set(manifest.get("tracks", {})) != set(TRACKS)
    ):
        raise BrightMultiVectorError("Encoding manifest contract drifted")
    representations = {}
    for track in TRACKS:
        evidence = validate_representation_artifact(
            root / track,
            model_key=model_key,
            track=track,
            predeclaration_sha256=predeclaration_sha256,
            gate_summary_sha256=gate_summary_sha256,
            data_root=data_root,
        )
        if manifest["tracks"][track]["representation_sha256"] != evidence["manifest_sha256"]:
            raise BrightMultiVectorError("Encoding manifest representation binding drifted")
        representations[track] = evidence
    if {path.name for path in root.iterdir() if path.is_file()} != {
        "encoding-manifest.json",
        "encoding-manifest.json.sha256",
    }:
        raise BrightMultiVectorError("Model encoding root contains an unexpected or missing file")
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "manifest_sidecar_sha256": file_sha256(sidecar),
        "representations": representations,
    }


def replay_formal_cell(
    model_key: str,
    track: str,
    *,
    predeclaration_path: str | Path,
    predeclaration_sha256: str,
    gate_summary_path: str | Path,
    data_root: str | Path,
    representation_root: str | Path,
    output_path: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Recompute one formal cell from saved token vectors without loading a model."""
    import torch
    from sentence_transformers.util.similarity import maxsim

    predeclaration = validate_predeclaration(predeclaration_path, predeclaration_sha256)
    validate_gate_summary(gate_summary_path)
    validate_canonical_data(predeclaration, data_root)
    if model_key not in SELECTED_KEYS or track not in TRACKS:
        raise BrightMultiVectorError("Formal cell is outside the frozen matrix")
    model_root = Path(representation_root) / model_key
    encoding = validate_model_encoding(
        model_root,
        model_key=model_key,
        predeclaration_sha256=predeclaration_sha256,
        gate_summary_sha256=file_sha256(Path(gate_summary_path)),
        data_root=data_root,
    )
    root = model_root / track
    representation_evidence = encoding["representations"][track]
    representation = representation_evidence["manifest"]
    representation_sha256 = representation_evidence["manifest_sha256"]
    load_started = time.perf_counter()
    documents = _load_ragged(root, representation["representations"]["documents"])
    queries = _load_ragged(root, representation["representations"]["queries"])
    files = representation["representations"]["files"]
    for entry in files.values():
        _validate_saved_entry(root, entry)
    window_document_indices = np.load(root / files["document_indices"]["path"], allow_pickle=False)
    window_ordinals = np.load(root / files["window_ordinals"]["path"], allow_pickle=False)
    document_ids = load_json(root / files["document_ids"]["path"])
    query_ids = load_json(root / files["query_ids"]["path"])
    if len(documents) != len(window_document_indices) or len(documents) != len(window_ordinals):
        raise BrightMultiVectorError("Window representations and mappings are misaligned")
    data = load_materialized(data_root, track)
    if document_ids != [row["id"] for row in data.corpus] or query_ids != [row["id"] for row in data.queries]:
        raise BrightMultiVectorError("Saved canonical ID order drifted")
    if len(queries) != len(query_ids):
        raise BrightMultiVectorError("Saved query representation coverage drifted")
    build_load_seconds = time.perf_counter() - load_started
    document_index = {value: index for index, value in enumerate(document_ids)}
    scoring_device = torch.device(device)
    torch.cuda.set_device(scoring_device)
    torch.cuda.reset_peak_memory_stats(scoring_device)
    warmup_started = time.perf_counter()
    warmup = maxsim([queries[0]], documents[: min(256, len(documents))], device=device, chunk_elements=20_000_000)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    if not torch.isfinite(warmup).all():
        raise BrightMultiVectorError("Warmup produced non-finite scores")
    rankings: dict[str, list[tuple[str, float]]] = {}
    latency_ms: list[float] = []
    search_started = time.perf_counter()
    for query_id, query in zip(query_ids, queries, strict=True):
        started = time.perf_counter()
        window_scores = (
            maxsim([query], documents, device=scoring_device, chunk_elements=20_000_000)[0].float().cpu().numpy()
        )
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not np.all(np.isfinite(window_scores)):
            raise BrightMultiVectorError("Exact MaxSim produced non-finite scores")
        document_scores = np.full(len(document_ids), -np.inf, dtype=np.float32)
        winning_ordinals = np.full(len(document_ids), np.iinfo(np.int32).max, dtype=np.int32)
        for window_index, score in enumerate(window_scores):
            index = int(window_document_indices[window_index])
            ordinal = int(window_ordinals[window_index])
            if score > document_scores[index] or (
                score == document_scores[index] and ordinal < winning_ordinals[index]
            ):
                document_scores[index] = score
                winning_ordinals[index] = ordinal
        query_row = next(row for row in data.queries if row["id"] == query_id)
        for excluded in query_row.get("excluded_ids", ()):
            document_scores[document_index[excluded]] = -np.inf
        ordered = sorted(
            range(len(document_ids)), key=lambda index: (-float(document_scores[index]), document_ids[index])
        )[:TOP_K]
        if len(ordered) != TOP_K or any(not math.isfinite(float(document_scores[index])) for index in ordered):
            raise BrightMultiVectorError("Formal ranking does not contain 100 finite candidates")
        rankings[query_id] = [(document_ids[index], float(document_scores[index])) for index in ordered]
        latency_ms.append(elapsed_ms)
    search_seconds = time.perf_counter() - search_started
    per_query = query_metrics(data, rankings)
    metrics = aggregate_metrics(per_query)
    uncertainty = bootstrap_confidence_intervals(per_query, samples=10_000, seed=20_260_826)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranking_file = output.with_name(output.stem + ".rankings.json")
    per_query_file = output.with_name(output.stem + ".per-query.json")
    ranking_entry = _save_json(
        ranking_file,
        [
            {"query_id": query_id, "hits": [{"document_id": doc, "score": score} for doc, score in hits]}
            for query_id, hits in rankings.items()
        ],
    )
    per_query_entry = _save_json(
        per_query_file,
        [{"query_id": query_id, "metrics": values} for query_id, values in per_query.items()],
    )
    storage = representation_evidence["storage"]
    cell = {
        "schema_version": "bright-real-multi-vector-cell-v1",
        "story_id": "S-20260814-012",
        "predeclaration_sha256": predeclaration_sha256,
        "gate_summary_sha256": file_sha256(Path(gate_summary_path)),
        "model": representation["model"],
        "track": representation["data"],
        "protocol": {
            "candidate_policy": predeclaration["protocol"]["candidate_policy"],
            "windowing": predeclaration["protocol"]["windowing"],
            "aggregation": "max_passage",
            "score": predeclaration["protocol"]["score"],
            "tie_policy": predeclaration["protocol"]["tie_policy"],
            "top_k": TOP_K,
            "dtype": "float32",
            "exact": True,
            "approximate_index_used": False,
        },
        "representation": {
            "manifest_sha256": representation_sha256,
            "encoding_manifest_sha256": encoding["manifest_sha256"],
            "storage": storage,
            "query_token_vectors": representation["representations"]["queries"]["token_vectors"],
            "document_token_vectors": representation["representations"]["documents"]["token_vectors"],
            "query_token_count": representation["representations"]["queries"]["token_count"],
            "document_token_count": representation["representations"]["documents"]["token_count"],
            "windows": representation["windowing"]["windows"],
            "windows_per_document": representation["windowing"]["windows_per_document"],
            "mask": "explicit compact boolean valid mask; padded adapter gate separately verified false/zero rows",
        },
        "quality": {"metrics": metrics, "uncertainty": uncertainty},
        "execution": {
            "model_loaded": False,
            "encoding": representation["encoding"],
            "build_load_seconds": build_load_seconds,
            "warmups": 1,
            "warmup_seconds": warmup_seconds,
            "measured_queries": len(query_ids),
            "exact_search_seconds": search_seconds,
            "exact_search_latency_ms": _percentiles(latency_ms),
            "query_throughput_per_second": len(query_ids) / search_seconds,
            "end_to_end_seconds": representation["encoding"]["track_wall_seconds"] + search_seconds,
            "process_maxrss_bytes": process_rss_bytes(),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(scoring_device)),
            "device": device,
            "batch_fallback": {
                "documents": representation["encoding"]["documents"],
                "queries": representation["encoding"]["queries"],
            },
        },
        "restricted_recompute": {
            "rankings": ranking_entry,
            "per_query": per_query_entry,
            "representation_root": str(root),
        },
        "publication": PUBLICATION,
        "finalized": True,
    }
    write_json_with_hash(output, cell)
    return cell
