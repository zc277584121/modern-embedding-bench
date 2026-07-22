"""Deterministic project-owned composed-media retrieval fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import wave
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mm_embed.providers.composed_media import (
    BINARY_NORMALIZATION,
    NORMALIZATION,
    ComposedMediaItem,
    ComposedMediaPart,
    canonical_json_bytes,
    item_fingerprint,
    item_from_dict,
    item_to_dict,
    normalize_text,
    part_fingerprint,
    sha256_bytes,
    validate_composed_item,
)


DATASET_VERSION = "composed-media-retrieval-fixture-v0"
GENERATOR_VERSION = "composed-media-fixture-generator-v0"
PROVENANCE_ID = "prov_fixture_v0"
FIXTURE_SCHEMA_VERSION = "composed-media-retrieval-fixture-v0"
FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "benchmark" / "fixtures" / DATASET_VERSION

QUERY_SHAPES = (
    "text_image_to_video",
    "text_video_to_video",
    "source_audio_text_to_audio",
)
HARD_NEGATIVE_FAMILIES = (
    "text_match_media_mismatch",
    "media_neighbor_text_contradiction",
    "temporal_neighbor",
    "difference_direction_inversion",
    "part_order_swap",
    "cross_family_collision",
    "duplicate_near_duplicate_leakage",
)


@dataclass(frozen=True)
class ComposedMediaQuery:
    query_id: str
    item: ComposedMediaItem
    split: str
    shape: str
    family: str
    target_modality: str
    provider_valid_required: bool
    reference_duration_bucket: str
    media_reuse: str
    order_sensitive: bool


@dataclass(frozen=True)
class ComposedMediaCorpusItem:
    corpus_id: str
    item: ComposedMediaItem
    split: str
    shape: str


@dataclass(frozen=True)
class ComposedMediaQrel:
    query_id: str
    corpus_id: str
    relevance: int
    judgment: str
    provenance_id: str


@dataclass(frozen=True)
class ComposedMediaHardNegative:
    query_id: str
    corpus_id: str
    negative_family: str
    reason: str
    review_status: str


@dataclass(frozen=True)
class ComposedMediaRetrievalFixture:
    schema_version: str
    dataset_version: str
    generator_version: str
    split: str
    fixture_only: bool
    publish: bool
    leaderboard_publish: bool
    license_status: str
    network: str
    provider_api_calls: int
    provenance: dict[str, Any]
    queries: tuple[ComposedMediaQuery, ...]
    corpus: tuple[ComposedMediaCorpusItem, ...]
    qrels: tuple[ComposedMediaQrel, ...]
    hard_negatives: tuple[ComposedMediaHardNegative, ...]
    fixture_sha256: str


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _png_bytes(video_index: int, frame_index: int) -> bytes:
    width = height = 64
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            moving = 96 if (x - frame_index * 3 - video_index * 5) % 64 < 12 else 0
            rows.extend(
                (
                    (x * 3 + video_index * 29 + frame_index * 7 + moving) % 256,
                    (y * 5 + video_index * 43 + frame_index * 11) % 256,
                    ((x + y) * 2 + video_index * 61 + frame_index * 13) % 256,
                )
            )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _write_wav(path: Path, audio_index: int) -> None:
    sample_rate = 8000
    sample_count = sample_rate * 2
    periods = (61, 73, 89, 109, 137, 173)
    period = periods[audio_index]
    samples = bytearray()
    for sample_index in range(sample_count):
        saw = ((sample_index % period) * 18000 // period) - 9000
        pulse = 2600 if (sample_index // (period * 2)) % 2 == 0 else -2600
        value = max(-32768, min(32767, saw + pulse + audio_index * 350))
        samples.extend(struct.pack("<h", value))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(samples))


def _make_part(
    *,
    item_id: str,
    part_index: int,
    modality: str,
    payload: dict[str, Any],
    mime_type: str,
    content: bytes,
    normalization: str,
    media: dict[str, Any] | None,
) -> ComposedMediaPart:
    part = ComposedMediaPart(
        part_id=f"{item_id}:p{part_index:02d}",
        part_index=part_index,
        modality=modality,
        payload=payload,
        mime_type=mime_type,
        content_sha256=sha256_bytes(content),
        byte_length=len(content),
        normalization=normalization,
        media=media,
        provenance_id=PROVENANCE_ID,
        part_sha256="",
    )
    return replace(part, part_sha256=part_fingerprint(part))


def _text_part(item_id: str, part_index: int, text: str) -> ComposedMediaPart:
    normalized = normalize_text(text)
    return _make_part(
        item_id=item_id,
        part_index=part_index,
        modality="text",
        payload={"kind": "inline_text", "text": normalized},
        mime_type="text/plain; charset=utf-8",
        content=normalized.encode("utf-8"),
        normalization=NORMALIZATION,
        media=None,
    )


def _media_part(
    root: Path,
    *,
    item_id: str,
    part_index: int,
    modality: str,
    relative_path: str,
    mime_type: str,
    media: dict[str, Any],
) -> ComposedMediaPart:
    kind = "frame_manifest" if modality == "video" else "tracked_file"
    return _make_part(
        item_id=item_id,
        part_index=part_index,
        modality=modality,
        payload={"kind": kind, "path": relative_path},
        mime_type=mime_type,
        content=(root / relative_path).read_bytes(),
        normalization=BINARY_NORMALIZATION,
        media=media,
    )


def _make_item(item_id: str, role: str, parts: tuple[ComposedMediaPart, ...]) -> ComposedMediaItem:
    item = ComposedMediaItem(
        item_id=item_id,
        role=role,
        parts=parts,
        instruction="Retrieve the target media item." if role == "query" else None,
        provenance_id=PROVENANCE_ID,
        item_sha256="",
    )
    return replace(item, item_sha256=item_fingerprint(item))


def _video_media() -> dict[str, Any]:
    return {
        "duration_ms": 2000,
        "frame_count": 16,
        "width_px": 64,
        "height_px": 64,
        "fps_numerator": 8,
        "fps_denominator": 1,
        "color_space": "srgb",
        "representation": "ordered_png_frames",
    }


def _audio_media() -> dict[str, Any]:
    return {
        "duration_ms": 2000,
        "sample_rate_hz": 8000,
        "channels": 1,
        "sample_width_bytes": 2,
        "representation": "pcm_s16le_wav",
    }


def _image_media() -> dict[str, Any]:
    return {"width_px": 64, "height_px": 64, "representation": "png"}


_QUERY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "query_id": "q_ti_0001",
        "shape": "text_image_to_video",
        "family": "temporal_successor",
        "reference": "stills/still_00.png",
        "text": "Retrieve the clip immediately after the striped reference state, with the bright block moving upward.",
        "reuse": "distinct_reference",
    },
    {
        "query_id": "q_ti_0002",
        "shape": "text_image_to_video",
        "family": "attribute_transfer",
        "reference": "stills/still_01.png",
        "text": "Keep the reference motion family but retrieve the warmer state with the block shifted to the right.",
        "reuse": "distinct_reference",
    },
    {
        "query_id": "q_ti_0003",
        "shape": "text_image_to_video",
        "family": "temporal_predecessor",
        "reference": "stills/still_02.png",
        "text": "Retrieve the clip just before the pictured state while preserving its diagonal texture.",
        "reuse": "distinct_reference",
    },
    {
        "query_id": "q_ti_0004",
        "shape": "text_image_to_video",
        "family": "motion_contrast",
        "reference": "stills/still_03.png",
        "text": "Find the related clip whose moving block travels in the opposite direction from the reference.",
        "reuse": "distinct_reference",
    },
    {
        "query_id": "q_tv_0001",
        "shape": "text_video_to_video",
        "family": "temporal_successor",
        "reference": "videos/video_00/manifest.json",
        "text": "Continue the reference sequence into the next authored motion phase.",
        "reuse": "reused_reference",
    },
    {
        "query_id": "q_tv_0002",
        "shape": "text_video_to_video",
        "family": "motion_inversion",
        "reference": "videos/video_00/manifest.json",
        "text": "Retrieve the sequence with the reference motion reversed while retaining its cadence.",
        "reuse": "reused_reference",
    },
    {
        "query_id": "q_tv_0003",
        "shape": "text_video_to_video",
        "family": "color_transfer",
        "reference": "videos/video_01/manifest.json",
        "text": "Find the matching motion family rendered with the cooler authored palette.",
        "reuse": "distinct_reference",
    },
    {
        "query_id": "q_tv_0004",
        "shape": "text_video_to_video",
        "family": "cadence_change",
        "reference": "videos/video_02/manifest.json",
        "text": "Retrieve the related sequence with the same direction but a shifted visual cadence.",
        "reuse": "distinct_reference",
    },
    {
        "query_id": "q_at_0001",
        "shape": "source_audio_text_to_audio",
        "family": "pitch_direction",
        "reference": "audio/audio_00.wav",
        "text": "Retrieve the authored sound that raises the source contour while preserving its pulse pattern.",
        "reuse": "reused_reference",
    },
    {
        "query_id": "q_at_0002",
        "shape": "source_audio_text_to_audio",
        "family": "rhythm_transfer",
        "reference": "audio/audio_00.wav",
        "text": "Keep the source contour but retrieve the denser pulse transformation.",
        "reuse": "reused_reference",
    },
    {
        "query_id": "q_at_0003",
        "shape": "source_audio_text_to_audio",
        "family": "difference_direction",
        "reference": "audio/audio_01.wav",
        "text": "Retrieve the target after applying the described upward difference, not its inverse.",
        "reuse": "distinct_reference",
    },
    {
        "query_id": "q_at_0004",
        "shape": "source_audio_text_to_audio",
        "family": "timbre_contrast",
        "reference": "audio/audio_02.wav",
        "text": "Find the related authored sound with a sharper pulse edge and the same duration.",
        "reuse": "distinct_reference",
    },
)


_POSITIVE_PLAN: dict[str, tuple[tuple[str, int], ...]] = {
    "q_ti_0001": (("video_corpus_04", 2),),
    "q_ti_0002": (("video_corpus_01", 2), ("video_corpus_05", 1)),
    "q_ti_0003": (("video_corpus_03", 2),),
    "q_ti_0004": (("video_corpus_00", 2),),
    "q_tv_0001": (("video_corpus_02", 2),),
    "q_tv_0002": (("video_corpus_05", 2), ("video_corpus_01", 1)),
    "q_tv_0003": (("video_corpus_00", 2),),
    "q_tv_0004": (("video_corpus_03", 2), ("video_corpus_04", 1)),
    "q_at_0001": (("audio_corpus_03", 2),),
    "q_at_0002": (("audio_corpus_00", 2), ("audio_corpus_05", 1)),
    "q_at_0003": (("audio_corpus_04", 2),),
    "q_at_0004": (("audio_corpus_01", 2),),
}


_NEGATIVE_PLAN: dict[str, tuple[tuple[str, str], ...]] = {
    "q_ti_0001": (
        ("video_corpus_02", "text_match_media_mismatch"),
        ("video_corpus_03", "temporal_neighbor"),
        ("video_corpus_05", "cross_family_collision"),
    ),
    "q_ti_0002": (
        ("video_corpus_00", "media_neighbor_text_contradiction"),
        ("video_corpus_02", "part_order_swap"),
        ("video_corpus_03", "duplicate_near_duplicate_leakage"),
    ),
    "q_ti_0003": (
        ("video_corpus_02", "temporal_neighbor"),
        ("video_corpus_04", "text_match_media_mismatch"),
        ("video_corpus_00", "cross_family_collision"),
    ),
    "q_ti_0004": (
        ("video_corpus_01", "media_neighbor_text_contradiction"),
        ("video_corpus_05", "part_order_swap"),
        ("video_corpus_02", "duplicate_near_duplicate_leakage"),
    ),
    "q_tv_0001": (
        ("video_corpus_01", "temporal_neighbor"),
        ("video_corpus_03", "media_neighbor_text_contradiction"),
        ("video_corpus_05", "cross_family_collision"),
    ),
    "q_tv_0002": (
        ("video_corpus_04", "temporal_neighbor"),
        ("video_corpus_00", "text_match_media_mismatch"),
        ("video_corpus_02", "part_order_swap"),
    ),
    "q_tv_0003": (
        ("video_corpus_01", "temporal_neighbor"),
        ("video_corpus_02", "duplicate_near_duplicate_leakage"),
        ("video_corpus_04", "cross_family_collision"),
    ),
    "q_tv_0004": (
        ("video_corpus_02", "media_neighbor_text_contradiction"),
        ("video_corpus_05", "text_match_media_mismatch"),
        ("video_corpus_00", "part_order_swap"),
    ),
    "q_at_0001": (
        ("audio_corpus_02", "difference_direction_inversion"),
        ("audio_corpus_04", "media_neighbor_text_contradiction"),
        ("audio_corpus_00", "cross_family_collision"),
    ),
    "q_at_0002": (
        ("audio_corpus_01", "difference_direction_inversion"),
        ("audio_corpus_02", "part_order_swap"),
        ("audio_corpus_04", "duplicate_near_duplicate_leakage"),
    ),
    "q_at_0003": (
        ("audio_corpus_03", "difference_direction_inversion"),
        ("audio_corpus_05", "text_match_media_mismatch"),
        ("audio_corpus_01", "cross_family_collision"),
    ),
    "q_at_0004": (
        ("audio_corpus_00", "difference_direction_inversion"),
        ("audio_corpus_02", "media_neighbor_text_contradiction"),
        ("audio_corpus_05", "part_order_swap"),
    ),
}


def _build_fixture(root: Path) -> ComposedMediaRetrievalFixture:
    queries: list[ComposedMediaQuery] = []
    for spec in _QUERY_SPECS:
        query_id = str(spec["query_id"])
        shape = str(spec["shape"])
        reference = str(spec["reference"])
        if shape == "text_image_to_video":
            parts = (
                _text_part(query_id, 0, str(spec["text"])),
                _media_part(
                    root,
                    item_id=query_id,
                    part_index=1,
                    modality="image",
                    relative_path=reference,
                    mime_type="image/png",
                    media=_image_media(),
                ),
            )
            target_modality = "video"
            duration_bucket = "still_image"
        elif shape == "text_video_to_video":
            parts = (
                _text_part(query_id, 0, str(spec["text"])),
                _media_part(
                    root,
                    item_id=query_id,
                    part_index=1,
                    modality="video",
                    relative_path=reference,
                    mime_type="application/vnd.mm-embed.frame-manifest+json",
                    media=_video_media(),
                ),
            )
            target_modality = "video"
            duration_bucket = "two_seconds"
        else:
            parts = (
                _media_part(
                    root,
                    item_id=query_id,
                    part_index=0,
                    modality="audio",
                    relative_path=reference,
                    mime_type="audio/wav",
                    media=_audio_media(),
                ),
                _text_part(query_id, 1, str(spec["text"])),
            )
            target_modality = "audio"
            duration_bucket = "two_seconds"
        queries.append(
            ComposedMediaQuery(
                query_id=query_id,
                item=_make_item(query_id, "query", parts),
                split="fixture_only",
                shape=shape,
                family=str(spec["family"]),
                target_modality=target_modality,
                provider_valid_required=True,
                reference_duration_bucket=duration_bucket,
                media_reuse=str(spec["reuse"]),
                order_sensitive=True,
            )
        )

    corpus: list[ComposedMediaCorpusItem] = []
    for index in range(6):
        corpus_id = f"video_corpus_{index:02d}"
        part = _media_part(
            root,
            item_id=corpus_id,
            part_index=0,
            modality="video",
            relative_path=f"videos/video_{index:02d}/manifest.json",
            mime_type="application/vnd.mm-embed.frame-manifest+json",
            media=_video_media(),
        )
        corpus.append(
            ComposedMediaCorpusItem(
                corpus_id,
                _make_item(corpus_id, "corpus", (part,)),
                "fixture_only",
                "video",
            )
        )
    for index in range(6):
        corpus_id = f"audio_corpus_{index:02d}"
        part = _media_part(
            root,
            item_id=corpus_id,
            part_index=0,
            modality="audio",
            relative_path=f"audio/audio_{index:02d}.wav",
            mime_type="audio/wav",
            media=_audio_media(),
        )
        corpus.append(
            ComposedMediaCorpusItem(
                corpus_id,
                _make_item(corpus_id, "corpus", (part,)),
                "fixture_only",
                "audio",
            )
        )

    qrels = tuple(
        ComposedMediaQrel(
            query_id=query_id,
            corpus_id=corpus_id,
            relevance=relevance,
            judgment="exact_target" if relevance == 2 else "valid_secondary",
            provenance_id=PROVENANCE_ID,
        )
        for query_id, positives in _POSITIVE_PLAN.items()
        for corpus_id, relevance in positives
    )
    hard_negatives = tuple(
        ComposedMediaHardNegative(
            query_id=query_id,
            corpus_id=corpus_id,
            negative_family=family,
            reason=f"Authored and reviewed {family.replace('_', ' ')} distractor.",
            review_status="reviewed_not_relevant",
        )
        for query_id, negatives in _NEGATIVE_PLAN.items()
        for corpus_id, family in negatives
    )
    fixture = ComposedMediaRetrievalFixture(
        schema_version=FIXTURE_SCHEMA_VERSION,
        dataset_version=DATASET_VERSION,
        generator_version=GENERATOR_VERSION,
        split="fixture_only",
        fixture_only=True,
        publish=False,
        leaderboard_publish=False,
        license_status="not_for_publication",
        network="forbidden",
        provider_api_calls=0,
        provenance={
            "provenance_id": PROVENANCE_ID,
            "source_kind": "self_created_fixture",
            "source_uri": None,
            "source_revision": DATASET_VERSION,
            "license_id": "not_for_publication",
            "derivation": "deterministic_project_owned_generator",
            "transform_version": GENERATOR_VERSION,
            "network_required": False,
        },
        queries=tuple(queries),
        corpus=tuple(corpus),
        qrels=qrels,
        hard_negatives=hard_negatives,
        fixture_sha256="",
    )
    return replace(fixture, fixture_sha256=fixture_fingerprint(fixture))


def _query_to_dict(query: ComposedMediaQuery) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "item": item_to_dict(query.item),
        "split": query.split,
        "shape": query.shape,
        "family": query.family,
        "target_modality": query.target_modality,
        "provider_valid_required": query.provider_valid_required,
        "reference_duration_bucket": query.reference_duration_bucket,
        "media_reuse": query.media_reuse,
        "order_sensitive": query.order_sensitive,
    }


def fixture_to_dict(
    fixture: ComposedMediaRetrievalFixture,
    *,
    include_fingerprint: bool = True,
) -> dict[str, Any]:
    payload = {
        "schema_version": fixture.schema_version,
        "dataset_version": fixture.dataset_version,
        "generator_version": fixture.generator_version,
        "split": fixture.split,
        "fixture_only": fixture.fixture_only,
        "publish": fixture.publish,
        "leaderboard_publish": fixture.leaderboard_publish,
        "license_status": fixture.license_status,
        "network": fixture.network,
        "provider_api_calls": fixture.provider_api_calls,
        "provenance": fixture.provenance,
        "queries": [_query_to_dict(query) for query in fixture.queries],
        "corpus": [
            {
                "corpus_id": row.corpus_id,
                "item": item_to_dict(row.item),
                "split": row.split,
                "shape": row.shape,
            }
            for row in fixture.corpus
        ],
        "qrels": [qrel.__dict__ for qrel in fixture.qrels],
        "hard_negatives": [negative.__dict__ for negative in fixture.hard_negatives],
    }
    if include_fingerprint:
        payload["fixture_sha256"] = fixture.fixture_sha256
    return payload


def fixture_fingerprint(fixture: ComposedMediaRetrievalFixture) -> str:
    return sha256_bytes(
        FIXTURE_SCHEMA_VERSION.encode("utf-8")
        + b"\0"
        + canonical_json_bytes(fixture_to_dict(fixture, include_fingerprint=False))
    )


def fixture_from_dict(value: dict[str, Any]) -> ComposedMediaRetrievalFixture:
    return ComposedMediaRetrievalFixture(
        schema_version=str(value["schema_version"]),
        dataset_version=str(value["dataset_version"]),
        generator_version=str(value["generator_version"]),
        split=str(value["split"]),
        fixture_only=bool(value["fixture_only"]),
        publish=bool(value["publish"]),
        leaderboard_publish=bool(value["leaderboard_publish"]),
        license_status=str(value["license_status"]),
        network=str(value["network"]),
        provider_api_calls=int(value["provider_api_calls"]),
        provenance=dict(value["provenance"]),
        queries=tuple(
            ComposedMediaQuery(
                query_id=str(row["query_id"]),
                item=item_from_dict(row["item"]),
                split=str(row["split"]),
                shape=str(row["shape"]),
                family=str(row["family"]),
                target_modality=str(row["target_modality"]),
                provider_valid_required=bool(row["provider_valid_required"]),
                reference_duration_bucket=str(row["reference_duration_bucket"]),
                media_reuse=str(row["media_reuse"]),
                order_sensitive=bool(row["order_sensitive"]),
            )
            for row in value["queries"]
        ),
        corpus=tuple(
            ComposedMediaCorpusItem(
                corpus_id=str(row["corpus_id"]),
                item=item_from_dict(row["item"]),
                split=str(row["split"]),
                shape=str(row["shape"]),
            )
            for row in value["corpus"]
        ),
        qrels=tuple(ComposedMediaQrel(**row) for row in value["qrels"]),
        hard_negatives=tuple(ComposedMediaHardNegative(**row) for row in value["hard_negatives"]),
        fixture_sha256=str(value["fixture_sha256"]),
    )


def validate_composed_media_retrieval_fixture(
    fixture: ComposedMediaRetrievalFixture,
    root: Path = FIXTURE_ROOT,
) -> None:
    """Validate exact fixture shape, assets, qrels, negatives, and publication gates."""
    expected_header = (
        fixture.schema_version == FIXTURE_SCHEMA_VERSION
        and fixture.dataset_version == DATASET_VERSION
        and fixture.generator_version == GENERATOR_VERSION
        and fixture.split == "fixture_only"
        and fixture.fixture_only is True
        and fixture.publish is False
        and fixture.leaderboard_publish is False
        and fixture.license_status == "not_for_publication"
        and fixture.network == "forbidden"
        and fixture.provider_api_calls == 0
    )
    if not expected_header:
        raise ValueError("Fixture publication or identity gates changed")
    if (
        fixture.provenance.get("network_required") is not False
        or fixture.provenance.get("license_id") != "not_for_publication"
    ):
        raise ValueError("Fixture provenance is not safe for fixture-only use")
    if fixture.fixture_sha256 != fixture_fingerprint(fixture):
        raise ValueError("Fixture fingerprint changed")

    query_shape_counts = Counter(query.shape for query in fixture.queries)
    expected_query_shape_counts = Counter({shape: 4 for shape in QUERY_SHAPES})
    if len(fixture.queries) != 12 or query_shape_counts != expected_query_shape_counts:
        raise ValueError("Fixture must contain exactly four queries for each composed shape")
    if len(fixture.corpus) != 12 or Counter(row.shape for row in fixture.corpus) != {"video": 6, "audio": 6}:
        raise ValueError("Fixture corpus must contain six videos and six audio clips")

    query_ids = [query.query_id for query in fixture.queries]
    corpus_ids = [row.corpus_id for row in fixture.corpus]
    if len(set(query_ids)) != len(query_ids) or len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("Duplicate query or corpus ids are invalid")
    for query in fixture.queries:
        if query.item.item_id != query.query_id or query.item.role != "query":
            raise ValueError("Query item identity changed")
        expected_modalities = {
            "text_image_to_video": ["text", "image"],
            "text_video_to_video": ["text", "video"],
            "source_audio_text_to_audio": ["audio", "text"],
        }[query.shape]
        if [part.modality for part in query.item.parts] != expected_modalities:
            raise ValueError("Composed query part order changed")
        validate_composed_item(query.item, root)
    for row in fixture.corpus:
        if row.item.item_id != row.corpus_id or row.item.role != "corpus" or len(row.item.parts) != 1:
            raise ValueError("Corpus item identity/cardinality changed")
        if row.item.parts[0].modality != row.shape:
            raise ValueError("Corpus media modality changed")
        validate_composed_item(row.item, root)

    qrel_keys = [(qrel.query_id, qrel.corpus_id) for qrel in fixture.qrels]
    if len(qrel_keys) != len(set(qrel_keys)) or len(fixture.qrels) != 16:
        raise ValueError("Fixture must contain 16 unique positive graded qrels")
    qrels_by_query: dict[str, list[ComposedMediaQrel]] = defaultdict(list)
    for qrel in fixture.qrels:
        if qrel.query_id not in query_ids or qrel.corpus_id not in corpus_ids:
            raise ValueError("Qrel references an unknown query or corpus id")
        if not isinstance(qrel.relevance, int) or qrel.relevance <= 0:
            raise ValueError("Fixture qrels must be positive integer grades")
        qrels_by_query[qrel.query_id].append(qrel)
    cardinalities = Counter(len(qrels_by_query[query_id]) for query_id in query_ids)
    if cardinalities != {1: 8, 2: 4}:
        raise ValueError("Fixture must have eight single-positive and four multi-positive queries")
    for query_id, rows in qrels_by_query.items():
        grades = sorted((row.relevance for row in rows), reverse=True)
        if grades not in ([2], [2, 1]):
            raise ValueError(f"Invalid graded qrels for {query_id}")

    negatives_by_query: dict[str, list[ComposedMediaHardNegative]] = defaultdict(list)
    seen_negative_keys: set[tuple[str, str]] = set()
    for negative in fixture.hard_negatives:
        key = (negative.query_id, negative.corpus_id)
        if key in seen_negative_keys or negative.query_id not in query_ids or negative.corpus_id not in corpus_ids:
            raise ValueError("Hard negative ids must be unique and known")
        seen_negative_keys.add(key)
        if key in set(qrel_keys):
            raise ValueError("A hard negative cannot also be a positive")
        if negative.negative_family not in HARD_NEGATIVE_FAMILIES or negative.review_status != "reviewed_not_relevant":
            raise ValueError("Hard negative family or review status changed")
        negatives_by_query[negative.query_id].append(negative)
    if len(fixture.hard_negatives) != 36 or any(len(negatives_by_query[query_id]) != 3 for query_id in query_ids):
        raise ValueError("Every query must have exactly three reviewed hard negatives")
    if set(negative.negative_family for negative in fixture.hard_negatives) != set(HARD_NEGATIVE_FAMILIES):
        raise ValueError("Hard negatives must cover all seven required families")

    query_by_id = {query.query_id: query for query in fixture.queries}
    corpus_by_id = {row.corpus_id: row for row in fixture.corpus}
    for query_id, rows in [*qrels_by_query.items(), *negatives_by_query.items()]:
        target = query_by_id[query_id].target_modality
        if any(corpus_by_id[row.corpus_id].shape != target for row in rows):
            raise ValueError("Qrels or hard negatives cross the target modality pool")

    manifests = sorted(root.glob("videos/video_*/manifest.json"))
    frames = sorted(root.glob("videos/video_*/*.png"))
    wavs = sorted(root.glob("audio/*.wav"))
    stills = sorted(root.glob("stills/*.png"))
    if (len(manifests), len(frames), len(wavs), len(stills)) != (6, 96, 6, 6):
        raise ValueError("Tracked media counts do not match the accepted fixture shape")
    for index, still in enumerate(stills):
        source = root / f"videos/video_{index:02d}/frame_{(index * 3) % 16:02d}.png"
        if still.read_bytes() != source.read_bytes():
            raise ValueError("Still references must reuse the declared authored video frames")


def generate_composed_media_retrieval_fixture(output_dir: str | Path) -> ComposedMediaRetrievalFixture:
    """Generate the canonical fixture without network, codecs, or external data."""
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    for video_index in range(6):
        frames: list[dict[str, Any]] = []
        for frame_index in range(16):
            relative_path = f"videos/video_{video_index:02d}/frame_{frame_index:02d}.png"
            data = _png_bytes(video_index, frame_index)
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            frames.append(
                {
                    "frame_index": frame_index,
                    "path": relative_path,
                    "sha256": sha256_bytes(data),
                    "byte_length": len(data),
                }
            )
        manifest = {
            "schema_version": "ordered-png-video-manifest-v0",
            "video_id": f"video_{video_index:02d}",
            "fps_numerator": 8,
            "fps_denominator": 1,
            "duration_ms": 2000,
            "frame_count": 16,
            "width_px": 64,
            "height_px": 64,
            "color_space": "srgb",
            "frames": frames,
        }
        (root / f"videos/video_{video_index:02d}/manifest.json").write_bytes(canonical_json_bytes(manifest))

        source_frame = root / f"videos/video_{video_index:02d}/frame_{(video_index * 3) % 16:02d}.png"
        still = root / f"stills/still_{video_index:02d}.png"
        still.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_frame, still)

    for audio_index in range(6):
        _write_wav(root / f"audio/audio_{audio_index:02d}.wav", audio_index)

    fixture = _build_fixture(root)
    validate_composed_media_retrieval_fixture(fixture, root)
    (root / "fixture.json").write_bytes(canonical_json_bytes(fixture_to_dict(fixture)))
    return fixture


def load_composed_media_retrieval_fixture(
    root: str | Path = FIXTURE_ROOT,
) -> ComposedMediaRetrievalFixture:
    fixture_root = Path(root)
    fixture_path = fixture_root / "fixture.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Tracked composed-media fixture is missing: {fixture_path}")
    fixture = fixture_from_dict(json.loads(fixture_path.read_bytes()))
    validate_composed_media_retrieval_fixture(fixture, fixture_root)
    return fixture


def fixture_tree_sha256(root: str | Path = FIXTURE_ROOT) -> str:
    fixture_root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in fixture_root.rglob("*") if item.is_file()):
        relative = path.relative_to(fixture_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def fixture_tracked_bytes(root: str | Path = FIXTURE_ROOT) -> int:
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(FIXTURE_ROOT))
    args = parser.parse_args()
    fixture = generate_composed_media_retrieval_fixture(args.output)
    print(f"fixture_sha256={fixture.fixture_sha256}")
    print(f"tree_sha256={fixture_tree_sha256(args.output)}")
    print(f"tracked_bytes={fixture_tracked_bytes(args.output)}")


if __name__ == "__main__":
    main()
