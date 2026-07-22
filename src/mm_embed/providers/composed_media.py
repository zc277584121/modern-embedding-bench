"""Provider-neutral contract for one embedding per composed media item."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


PART_SCHEMA_VERSION = "composed-media-part-v0"
ITEM_SCHEMA_VERSION = "composed-media-item-v0"
REQUEST_SCHEMA_VERSION = "composed-media-request-v0"
RESULT_SCHEMA_VERSION = "composed-media-result-v0"
NORMALIZATION = "unicode-nfc-lf-v1"
BINARY_NORMALIZATION = "binary-exact-v1"

COMPOSITION_MODES = {"provider_native_fusion", "benchmark_system_fusion"}
TRACK_LABELS = {"provider_valid_embedding", "benchmark_system_fusion"}
SCORE_VALIDITY_LABELS = {
    "provider_valid_embedding",
    "benchmark_system_fusion",
    "reranker_system_only",
    "contract_fixture_only",
}
PART_MODALITIES = {"text", "image", "audio", "video", "document"}


@dataclass(frozen=True)
class ComposedMediaPart:
    """One ordered part of a logical media item."""

    part_id: str
    part_index: int
    modality: str
    payload: dict[str, Any]
    mime_type: str
    content_sha256: str
    byte_length: int
    normalization: str
    media: dict[str, Any] | None
    provenance_id: str
    part_sha256: str
    schema_version: str = PART_SCHEMA_VERSION


@dataclass(frozen=True)
class ComposedMediaItem:
    """One logical item that must produce exactly one embedding."""

    item_id: str
    role: str
    parts: tuple[ComposedMediaPart, ...]
    instruction: str | None
    provenance_id: str
    item_sha256: str
    schema_version: str = ITEM_SCHEMA_VERSION


@dataclass(frozen=True)
class ComposedMediaEmbeddingRequest:
    """Auditable composed request with a cache-safe identity."""

    items: tuple[ComposedMediaItem, ...]
    provider: str
    model_id: str
    model_revision: str
    dimensions: int
    task_route: str
    preprocessing: str
    composition_mode: str
    track_label: str
    fusion_strategy: str
    request_sha256: str
    schema_version: str = REQUEST_SCHEMA_VERSION


@dataclass(frozen=True)
class ComposedMediaEmbeddingRow:
    """One returned vector with complete request and route identity."""

    item_id: str
    item_sha256: str
    request_sha256: str
    provider: str
    model_id: str
    model_revision: str
    composition_mode: str
    track_label: str
    dimensions: int
    route_evidence: dict[str, Any]
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class ComposedMediaEmbeddingResult:
    """Ordered one-row-per-item response for a composed request."""

    request_sha256: str
    rows: tuple[ComposedMediaEmbeddingRow, ...]
    dimensions: int
    provider: str
    model_id: str
    model_revision: str
    composition_mode: str
    track_label: str
    score_validity: str
    route_evidence: dict[str, Any]
    latency_ms: float
    result_sha256: str
    schema_version: str = RESULT_SCHEMA_VERSION


@runtime_checkable
class ComposedMediaEmbeddingProvider(Protocol):
    """Separate capability for ordered heterogeneous logical items."""

    def embed_composed_media(
        self,
        request: ComposedMediaEmbeddingRequest,
    ) -> ComposedMediaEmbeddingResult:
        """Return exactly one embedding per logical item in request order."""
        ...


def normalize_text(text: str) -> str:
    """Normalize authored text before hashing or serialization."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes with array order preserved."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or normalize_text(value) != value:
        raise ValueError(f"{field_name} must be a non-empty normalized UTF-8 string")
    value.encode("utf-8")


def part_to_dict(part: ComposedMediaPart, *, include_fingerprint: bool = True) -> dict[str, Any]:
    payload = {
        "schema_version": part.schema_version,
        "part_id": part.part_id,
        "part_index": part.part_index,
        "modality": part.modality,
        "payload": part.payload,
        "mime_type": part.mime_type,
        "content_sha256": part.content_sha256,
        "byte_length": part.byte_length,
        "normalization": part.normalization,
        "media": part.media,
        "provenance_id": part.provenance_id,
    }
    if include_fingerprint:
        payload["part_sha256"] = part.part_sha256
    return payload


def part_from_dict(value: dict[str, Any]) -> ComposedMediaPart:
    return ComposedMediaPart(
        schema_version=str(value["schema_version"]),
        part_id=str(value["part_id"]),
        part_index=int(value["part_index"]),
        modality=str(value["modality"]),
        payload=dict(value["payload"]),
        mime_type=str(value["mime_type"]),
        content_sha256=str(value["content_sha256"]),
        byte_length=int(value["byte_length"]),
        normalization=str(value["normalization"]),
        media=dict(value["media"]) if value.get("media") is not None else None,
        provenance_id=str(value["provenance_id"]),
        part_sha256=str(value["part_sha256"]),
    )


def part_fingerprint(part: ComposedMediaPart) -> str:
    domain = PART_SCHEMA_VERSION.encode("utf-8") + b"\0"
    return sha256_bytes(domain + canonical_json_bytes(part_to_dict(part, include_fingerprint=False)))


def item_to_dict(item: ComposedMediaItem, *, include_fingerprint: bool = True) -> dict[str, Any]:
    payload = {
        "schema_version": item.schema_version,
        "item_id": item.item_id,
        "role": item.role,
        "parts": [part_to_dict(part) for part in item.parts],
        "instruction": item.instruction,
        "provenance_id": item.provenance_id,
    }
    if include_fingerprint:
        payload["item_sha256"] = item.item_sha256
    return payload


def item_from_dict(value: dict[str, Any]) -> ComposedMediaItem:
    return ComposedMediaItem(
        schema_version=str(value["schema_version"]),
        item_id=str(value["item_id"]),
        role=str(value["role"]),
        parts=tuple(part_from_dict(part) for part in value["parts"]),
        instruction=value.get("instruction"),
        provenance_id=str(value["provenance_id"]),
        item_sha256=str(value["item_sha256"]),
    )


def item_fingerprint(item: ComposedMediaItem) -> str:
    domain = ITEM_SCHEMA_VERSION.encode("utf-8") + b"\0"
    return sha256_bytes(domain + canonical_json_bytes(item_to_dict(item, include_fingerprint=False)))


def request_to_dict(
    request: ComposedMediaEmbeddingRequest,
    *,
    include_fingerprint: bool = True,
) -> dict[str, Any]:
    payload = {
        "schema_version": request.schema_version,
        "items": [item_to_dict(item) for item in request.items],
        "provider": request.provider,
        "model_id": request.model_id,
        "model_revision": request.model_revision,
        "dimensions": request.dimensions,
        "task_route": request.task_route,
        "preprocessing": request.preprocessing,
        "composition_mode": request.composition_mode,
        "track_label": request.track_label,
        "fusion_strategy": request.fusion_strategy,
    }
    if include_fingerprint:
        payload["request_sha256"] = request.request_sha256
    return payload


def request_fingerprint(request: ComposedMediaEmbeddingRequest) -> str:
    domain = REQUEST_SCHEMA_VERSION.encode("utf-8") + b"\0"
    return sha256_bytes(domain + canonical_json_bytes(request_to_dict(request, include_fingerprint=False)))


def row_to_dict(row: ComposedMediaEmbeddingRow) -> dict[str, Any]:
    return {
        "item_id": row.item_id,
        "item_sha256": row.item_sha256,
        "request_sha256": row.request_sha256,
        "provider": row.provider,
        "model_id": row.model_id,
        "model_revision": row.model_revision,
        "composition_mode": row.composition_mode,
        "track_label": row.track_label,
        "dimensions": row.dimensions,
        "route_evidence": row.route_evidence,
        "embedding": list(row.embedding),
    }


def result_to_dict(
    result: ComposedMediaEmbeddingResult,
    *,
    include_fingerprint: bool = True,
) -> dict[str, Any]:
    payload = {
        "schema_version": result.schema_version,
        "request_sha256": result.request_sha256,
        "rows": [row_to_dict(row) for row in result.rows],
        "dimensions": result.dimensions,
        "provider": result.provider,
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "composition_mode": result.composition_mode,
        "track_label": result.track_label,
        "score_validity": result.score_validity,
        "route_evidence": result.route_evidence,
        "latency_ms": result.latency_ms,
    }
    if include_fingerprint:
        payload["result_sha256"] = result.result_sha256
    return payload


def result_fingerprint(result: ComposedMediaEmbeddingResult) -> str:
    domain = RESULT_SCHEMA_VERSION.encode("utf-8") + b"\0"
    return sha256_bytes(domain + canonical_json_bytes(result_to_dict(result, include_fingerprint=False)))


def _resolve_asset(asset_root: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts or not relative_path:
        raise ValueError(f"Media path must be repository-relative: {relative_path!r}")
    root = asset_root.resolve()
    resolved = (root / rel).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Media path escapes fixture root: {relative_path!r}")
    if not resolved.is_file():
        raise ValueError(f"Tracked media is missing: {relative_path}")
    return resolved


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG asset: {path.name}")
    return struct.unpack(">II", data[16:24])


def _validate_route_evidence(value: dict[str, Any]) -> None:
    forbidden = ("secret", "token", "password", "api_key", "authorization")
    for key, item in value.items():
        if any(marker in str(key).lower() for marker in forbidden):
            raise ValueError(f"Route evidence contains a secret-bearing field: {key}")
        if isinstance(item, dict):
            _validate_route_evidence(item)
        elif isinstance(item, list):
            if any(isinstance(entry, (dict, list, tuple, set)) for entry in item):
                raise ValueError("Route evidence lists must contain scalar values")
        elif not isinstance(item, (str, int, bool, float, type(None))):
            raise ValueError("Route evidence must contain JSON scalar values")


def validate_composed_part(part: ComposedMediaPart, asset_root: Path) -> None:
    """Reject changed content, metadata, indexes, and non-portable paths."""
    if part.schema_version != PART_SCHEMA_VERSION:
        raise ValueError("Unsupported composed-media part schema")
    _require_identifier(part.part_id, "part_id")
    _require_identifier(part.provenance_id, "provenance_id")
    if part.modality not in PART_MODALITIES:
        raise ValueError(f"Unsupported composed-media modality: {part.modality}")
    if not isinstance(part.part_index, int) or part.part_index < 0:
        raise ValueError("part_index must be a non-negative integer")
    if not isinstance(part.byte_length, int) or part.byte_length < 0:
        raise ValueError("byte_length must be a non-negative integer")

    kind = part.payload.get("kind")
    if part.modality == "text":
        text = part.payload.get("text")
        if kind != "inline_text" or not isinstance(text, str):
            raise ValueError("Text parts require an inline_text payload")
        normalized = normalize_text(text)
        content = normalized.encode("utf-8")
        if text != normalized or part.normalization != NORMALIZATION:
            raise ValueError("Text part normalization is invalid")
        if part.mime_type != "text/plain; charset=utf-8" or part.media is not None:
            raise ValueError("Text part MIME/media metadata is invalid")
    else:
        relative_path = part.payload.get("path")
        if not isinstance(relative_path, str):
            raise ValueError("Media parts require a relative path payload")
        path = _resolve_asset(asset_root, relative_path)
        content = path.read_bytes()
        if part.normalization != BINARY_NORMALIZATION or not isinstance(part.media, dict):
            raise ValueError("Media normalization/metadata is invalid")

        if part.modality == "image":
            if kind != "tracked_file" or part.mime_type != "image/png":
                raise ValueError("Image parts require tracked PNG files")
            width, height = _png_dimensions(path)
            if part.media != {"width_px": width, "height_px": height, "representation": "png"}:
                raise ValueError("Image dimensions or representation changed")
        elif part.modality == "audio":
            if kind != "tracked_file" or part.mime_type != "audio/wav":
                raise ValueError("Audio parts require tracked WAV files")
            with wave.open(str(path), "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                frame_count = handle.getnframes()
            duration_ms = frame_count * 1000 // sample_rate
            expected = {
                "duration_ms": duration_ms,
                "sample_rate_hz": sample_rate,
                "channels": channels,
                "sample_width_bytes": sample_width,
                "representation": "pcm_s16le_wav",
            }
            if part.media != expected:
                raise ValueError("Audio duration or PCM metadata changed")
        elif part.modality == "video":
            if kind != "frame_manifest" or part.mime_type != "application/vnd.mm-embed.frame-manifest+json":
                raise ValueError("Video parts require canonical frame manifests")
            manifest = json.loads(content)
            if canonical_json_bytes(manifest) != content:
                raise ValueError("Video manifest is not canonical JSON")
            frames = manifest.get("frames") or []
            if [frame.get("frame_index") for frame in frames] != list(range(len(frames))):
                raise ValueError("Video frame indexes are reordered or non-contiguous")
            for frame in frames:
                frame_path = _resolve_asset(asset_root, str(frame.get("path") or ""))
                frame_bytes = frame_path.read_bytes()
                if frame.get("byte_length") != len(frame_bytes) or frame.get("sha256") != sha256_bytes(frame_bytes):
                    raise ValueError("Video frame content identity changed")
                if _png_dimensions(frame_path) != (manifest.get("width_px"), manifest.get("height_px")):
                    raise ValueError("Video frame dimensions changed")
            expected = {
                "duration_ms": manifest.get("duration_ms"),
                "frame_count": manifest.get("frame_count"),
                "width_px": manifest.get("width_px"),
                "height_px": manifest.get("height_px"),
                "fps_numerator": manifest.get("fps_numerator"),
                "fps_denominator": manifest.get("fps_denominator"),
                "color_space": manifest.get("color_space"),
                "representation": "ordered_png_frames",
            }
            if manifest.get("frame_count") != len(frames) or part.media != expected:
                raise ValueError("Video manifest metadata changed")
        elif part.modality == "document":
            raise ValueError("The v0 fixture does not define a document representation")

    if part.byte_length != len(content) or part.content_sha256 != sha256_bytes(content):
        raise ValueError("Part content hash or byte length changed")
    if part.part_sha256 != part_fingerprint(part):
        raise ValueError("Part fingerprint changed")


def validate_composed_item(item: ComposedMediaItem, asset_root: Path) -> None:
    """Validate item grouping, ordered parts, ids, and item identity."""
    if item.schema_version != ITEM_SCHEMA_VERSION:
        raise ValueError("Unsupported composed-media item schema")
    _require_identifier(item.item_id, "item_id")
    _require_identifier(item.provenance_id, "provenance_id")
    if item.role not in {"query", "corpus"}:
        raise ValueError("Composed item role must be query or corpus")
    if item.instruction is not None and normalize_text(item.instruction) != item.instruction:
        raise ValueError("Item instruction is not normalized")
    if not item.parts:
        raise ValueError("Composed items require at least one part")
    if [part.part_index for part in item.parts] != list(range(len(item.parts))):
        raise ValueError("Part indexes must be contiguous and match array order")
    part_ids = [part.part_id for part in item.parts]
    if len(set(part_ids)) != len(part_ids):
        raise ValueError("Duplicate part ids are invalid")
    for index, part in enumerate(item.parts):
        if part.part_id != f"{item.item_id}:p{index:02d}":
            raise ValueError("Part is attached to the wrong logical item")
        if part.provenance_id != item.provenance_id:
            raise ValueError("Part and item provenance ids differ")
        validate_composed_part(part, asset_root)
    if item.item_sha256 != item_fingerprint(item):
        raise ValueError("Item fingerprint changed")


def validate_composed_request(request: ComposedMediaEmbeddingRequest, asset_root: Path) -> None:
    """Validate routing labels, ordered items, dimensions, and request identity."""
    if request.schema_version != REQUEST_SCHEMA_VERSION:
        raise ValueError("Unsupported composed-media request schema")
    for field_name in ("provider", "model_id", "model_revision", "task_route", "preprocessing", "fusion_strategy"):
        _require_identifier(getattr(request, field_name), field_name)
    if request.dimensions <= 0:
        raise ValueError("Request dimensions must be positive")
    if request.composition_mode not in COMPOSITION_MODES:
        raise ValueError("Unsupported composition mode")
    if request.track_label not in TRACK_LABELS:
        raise ValueError("Unsupported embedding track label")
    expected_track = (
        "provider_valid_embedding"
        if request.composition_mode == "provider_native_fusion"
        else "benchmark_system_fusion"
    )
    if request.track_label != expected_track:
        raise ValueError("Composition mode and track label are incompatible")
    if not request.items:
        raise ValueError("Composed requests require at least one logical item")
    item_ids = [item.item_id for item in request.items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Duplicate logical item ids are invalid")
    for item in request.items:
        validate_composed_item(item, asset_root)
    if request.request_sha256 != request_fingerprint(request):
        raise ValueError("Request fingerprint changed")


def validate_composed_result(
    request: ComposedMediaEmbeddingRequest,
    result: ComposedMediaEmbeddingResult,
    asset_root: Path,
) -> None:
    """Reject flattened, reordered, mislabeled, or invalid composed results."""
    validate_composed_request(request, asset_root)
    if result.schema_version != RESULT_SCHEMA_VERSION:
        raise ValueError("Unsupported composed-media result schema")
    if result.request_sha256 != request.request_sha256:
        raise ValueError("Result lost the request fingerprint")
    header_fields = ("dimensions", "provider", "model_id", "model_revision", "composition_mode", "track_label")
    for field_name in header_fields:
        if getattr(result, field_name) != getattr(request, field_name):
            raise ValueError(f"Result {field_name} does not match the request")
    if result.score_validity not in SCORE_VALIDITY_LABELS:
        raise ValueError("Unsupported score-validity label")
    if result.score_validity == "reranker_system_only":
        raise ValueError("Reranker scores cannot be labeled as embeddings")
    if result.score_validity != "contract_fixture_only" and result.score_validity != request.track_label:
        raise ValueError("Result score validity is incompatible with its route")
    _validate_route_evidence(result.route_evidence)
    if not math.isfinite(result.latency_ms) or result.latency_ms < 0:
        raise ValueError("Result latency must be finite and non-negative")
    if len(result.rows) != len(request.items):
        raise ValueError("Result cardinality must equal logical-item cardinality")

    for item, row in zip(request.items, result.rows):
        if row.item_id != item.item_id or row.item_sha256 != item.item_sha256:
            raise ValueError("Result item ids or fingerprints are reordered/lost")
        if row.request_sha256 != request.request_sha256:
            raise ValueError("Result row lost the request fingerprint")
        for field_name in header_fields:
            if getattr(row, field_name) != getattr(request, field_name):
                raise ValueError(f"Result row {field_name} does not match the request")
        if row.route_evidence != result.route_evidence:
            raise ValueError("Result row route evidence differs from the result header")
        _validate_route_evidence(row.route_evidence)
        if len(row.embedding) != request.dimensions:
            raise ValueError("Embedding dimensions are inconsistent")
        if not all(math.isfinite(float(value)) for value in row.embedding):
            raise ValueError("Embedding contains NaN or infinity")
        if math.sqrt(sum(float(value) ** 2 for value in row.embedding)) == 0.0:
            raise ValueError("Embedding vectors must be non-zero")
    if result.result_sha256 != result_fingerprint(result):
        raise ValueError("Result fingerprint changed")
