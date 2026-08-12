"""Bounded, deterministic BRIGHT source preflight and materialization."""

from __future__ import annotations

import hashlib
import json
import argparse
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator


HF_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
HF_DATASET = "xlangai/BRIGHT"
BENCHMARK_VERSION = "bright-nontechnical-pilot-v0.1"
NO_EXCLUDED_IDS_SENTINEL = "N/A"
TRACKS = ("biology", "economics", "psychology")
ROLES = ("documents", "examples")
MAX_TOTAL_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
EXPECTED: dict[str, dict[str, dict[str, Any]]] = {
    "biology": {
        "documents": {
            "bytes": 11046045,
            "sha256": "8516d0c233f9c34e9eb6922b56e8a1698e5a6f6e504a9499fcd511cdd5741670",
            "rows": 57359,
        },
        "examples": {
            "bytes": 200655,
            "sha256": "6e105c4f09d9a70b8a20ed6a4d0e386823a5545151df41b3f0e64eb5c5987829",
            "rows": 103,
        },
    },
    "economics": {
        "documents": {
            "bytes": 10969621,
            "sha256": "f3ba8a0fbc9a9aed07b4970cc686e32cfefcd06d6922402587adf871f006394c",
            "rows": 50220,
        },
        "examples": {
            "bytes": 219518,
            "sha256": "2a79f0f3a881c7c03a258cf8ef8ac2db1ca9080963252d9a020bb45a264aa037",
            "rows": 103,
        },
    },
    "psychology": {
        "documents": {
            "bytes": 11430533,
            "sha256": "085d381739cb24b4227dfaf577f39d0adcad8b7b1ae74be028ac239d37be3c1d",
            "rows": 52835,
        },
        "examples": {
            "bytes": 183889,
            "sha256": "404e7dff2a4528419df0bdc162541e92138e35b78918d82d3a04ade5b8f7876b",
            "rows": 101,
        },
    },
}
DOCUMENT_SCHEMA = {"id": "string", "content": "string"}
EXAMPLE_SCHEMA = {
    "query": "string",
    "reasoning": "string",
    "id": "string",
    "excluded_ids": "list<element: string>",
    "gold_ids_long": "list<element: string>",
    "gold_ids": "list<element: string>",
    "gold_answer": "string",
}
MATERIALIZED_FILE_KEYS = frozenset({"corpus.jsonl", "queries.jsonl", "qrels/test.tsv", "audit.json"})
MATERIALIZABLE_TRACKS = ("biology",)
CANONICAL_MATERIALIZATION: dict[str, dict[str, dict[str, Any]]] = {
    "biology": {
        "corpus.jsonl": {
            "path": "corpus.jsonl",
            "rows": 57359,
            "bytes": 22991898,
            "sha256": "80414ed1384102adde0168c5f4ef3e3e51536c276c49b97c9660ef19b1606a7b",
        },
        "queries.jsonl": {
            "path": "queries.jsonl",
            "rows": 103,
            "bytes": 65972,
            "sha256": "e125db608e452ed254d3f6b8ce0c7be7604ef1e3049442cc533c18d455e90f72",
        },
        "qrels/test.tsv": {
            "path": "qrels/test.tsv",
            "rows": 372,
            "bytes": 18307,
            "sha256": "3a8425f7d93fff080238fa2f16443d0b4645f799a415831b2a771baec455f7a0",
        },
        "audit.json": {
            "path": "audit.json",
            "bytes": 1134,
            "sha256": "14fff3c99df731a9657828a43b45e669d3082305f656c5883fbb37ad4d797854",
        },
    },
}


class BrightMaterializationError(ValueError):
    """A stable fail-closed BRIGHT validation error."""


@dataclass(frozen=True)
class TrackRows:
    track: str
    documents: tuple[dict[str, str], ...]
    examples: tuple[dict[str, Any], ...]


def source_url(track: str, role: str) -> str:
    _validate_track_role(track, role)
    return (
        f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/"
        f"{role}/{track}-00000-of-00001.parquet"
    )


def _validate_track_role(track: str, role: str) -> None:
    if track not in TRACKS or role not in ROLES:
        raise BrightMaterializationError(f"unsupported BRIGHT source: {track}/{role}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[3] / "schemas" / "bright-materialization-v01.schema.json"


def _safe_materialized_path(output: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise BrightMaterializationError(f"unsafe materialized path: {relative!r}")
    resolved = (output / candidate).resolve()
    try:
        resolved.relative_to(output.resolve())
    except ValueError as error:
        raise BrightMaterializationError(f"materialized path escapes output root: {relative!r}") from error
    return resolved


def _data_rows(path: Path, relative: str) -> int | None:
    if relative.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if relative == "qrels/test.tsv":
        with path.open("r", encoding="utf-8") as handle:
            if handle.readline() != "query-id\tcorpus-id\tscore\n":
                raise BrightMaterializationError("invalid qrels header")
            return sum(1 for line in handle if line.strip())
    return None


def _read_jsonl_objects(path: Path, *, fields: dict[str, type], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise BrightMaterializationError(f"blank {label} row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BrightMaterializationError(
                    f"invalid {label} JSON at line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict) or set(row) != set(fields):
                raise BrightMaterializationError(f"invalid {label} fields at line {line_number}")
            for field, expected_type in fields.items():
                if not isinstance(row[field], expected_type):
                    raise BrightMaterializationError(
                        f"invalid {label} {field} type at line {line_number}"
                    )
            rows.append(row)
    return rows


def _validate_materialized_content(output: Path) -> None:
    corpus = _read_jsonl_objects(
        output / "corpus.jsonl",
        fields={"id": str, "content": str},
        label="corpus",
    )
    corpus_ids: set[str] = set()
    for row in corpus:
        if not row["id"] or row["id"] in corpus_ids:
            raise BrightMaterializationError("corpus ids must be non-empty and unique")
        if not row["content"] or row["content"] != row["content"].strip():
            raise BrightMaterializationError(f"invalid corpus content: {row['id']!r}")
        corpus_ids.add(row["id"])

    queries = _read_jsonl_objects(
        output / "queries.jsonl",
        fields={"id": str, "text": str, "excluded_ids": list, "gold_ids_long": list},
        label="query",
    )
    query_ids: set[str] = set()
    for row in queries:
        if not row["id"] or row["id"] in query_ids:
            raise BrightMaterializationError("query ids must be non-empty and unique")
        if not row["text"] or row["text"] != row["text"].strip():
            raise BrightMaterializationError(f"invalid query text: {row['id']!r}")
        for field in ("excluded_ids", "gold_ids_long"):
            values = row[field]
            if any(not isinstance(value, str) or not value for value in values):
                raise BrightMaterializationError(f"invalid query {field}: {row['id']!r}")
            if len(values) != len(set(values)):
                raise BrightMaterializationError(f"duplicate query {field}: {row['id']!r}")
        if set(row["excluded_ids"]) - corpus_ids:
            raise BrightMaterializationError(f"dangling excluded passage id: {row['id']!r}")
        query_ids.add(row["id"])

    qrels: list[tuple[str, str, int]] = []
    pairs: set[tuple[str, str]] = set()
    with (output / "qrels/test.tsv").open("r", encoding="utf-8", newline="") as handle:
        if handle.readline() != "query-id\tcorpus-id\tscore\n":
            raise BrightMaterializationError("invalid qrels header")
        for line_number, line in enumerate(handle, 2):
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 3 or any(not value for value in columns):
                raise BrightMaterializationError(f"malformed qrels row at line {line_number}")
            query_id, document_id, grade_text = columns
            try:
                grade = int(grade_text)
            except ValueError as error:
                raise BrightMaterializationError(f"invalid qrel grade at line {line_number}") from error
            if str(grade) != grade_text or grade != 1:
                raise BrightMaterializationError(f"illegal qrel grade at line {line_number}")
            if query_id not in query_ids or document_id not in corpus_ids:
                raise BrightMaterializationError(f"dangling qrel at line {line_number}")
            pair = (query_id, document_id)
            if pair in pairs:
                raise BrightMaterializationError(f"duplicate qrel pair at line {line_number}")
            pairs.add(pair)
            qrels.append((query_id, document_id, grade))

    gold_by_query: dict[str, list[str]] = {query_id: [] for query_id in query_ids}
    for query_id, document_id, _grade in qrels:
        gold_by_query[query_id].append(document_id)
    reconstructed = TrackRows(
        track=json.loads((output / "manifest.json").read_text(encoding="utf-8"))["track"],
        documents=tuple(corpus),
        examples=tuple(
            {
                "id": row["id"],
                "query": row["text"],
                "gold_ids": gold_by_query[row["id"]],
                "excluded_ids": row["excluded_ids"],
                "gold_ids_long": row["gold_ids_long"],
            }
            for row in queries
        ),
    )
    for row in reconstructed.examples:
        if not row["gold_ids"]:
            raise BrightMaterializationError(f"query has no positive qrels: {row['id']!r}")
        if set(row["gold_ids"]) & set(row["excluded_ids"]):
            raise BrightMaterializationError(f"gold/excluded overlap: {row['id']!r}")
    try:
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BrightMaterializationError(f"invalid audit JSON: {error}") from error
    if not isinstance(audit, dict) or audit != audit_track(reconstructed):
        raise BrightMaterializationError("audit does not match materialized content")


def validate_materialization(output_root: str | Path) -> dict[str, Any]:
    """Validate a BRIGHT materialization and every byte named by its manifest."""
    output = Path(output_root)
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise BrightMaterializationError(f"missing regular manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = json.loads(_schema_path().read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(manifest)
    except (OSError, json.JSONDecodeError) as error:
        raise BrightMaterializationError(f"invalid materialization manifest: {error}") from error
    except Exception as error:
        raise BrightMaterializationError(f"manifest schema validation failed: {error}") from error

    track = manifest["track"]
    if track not in MATERIALIZABLE_TRACKS:
        raise BrightMaterializationError(f"track is not approved for complete materialization: {track}")
    expected_sources = [
        {"role": role, "url": source_url(track, role), **EXPECTED[track][role]}
        for role in ROLES
    ]
    if manifest["source"]["files"] != expected_sources:
        raise BrightMaterializationError("source identity does not match the pinned BRIGHT contract")
    file_entries = manifest["materialization"]["files"]
    if set(file_entries) != MATERIALIZED_FILE_KEYS:
        raise BrightMaterializationError("materialized file set does not match the BRIGHT contract")
    canonical_files = CANONICAL_MATERIALIZATION[track]
    if file_entries != canonical_files:
        raise BrightMaterializationError("materialized file identity does not match the pinned BRIGHT contract")
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != MATERIALIZED_FILE_KEYS | {"manifest.json"}:
        raise BrightMaterializationError("output tree contains missing or unexpected files")
    for key in sorted(MATERIALIZED_FILE_KEYS):
        entry = file_entries[key]
        if entry["path"] != key:
            raise BrightMaterializationError(f"manifest path mismatch for {key}")
        path = _safe_materialized_path(output, entry["path"])
        if not path.is_file() or path.is_symlink():
            raise BrightMaterializationError(f"missing regular materialized file: {key}")
        if path.stat().st_size != entry["bytes"]:
            raise BrightMaterializationError(f"byte count mismatch for {key}")
        if _sha256(path) != entry["sha256"]:
            raise BrightMaterializationError(f"SHA256 mismatch for {key}")
        rows = _data_rows(path, key)
        if rows is not None and rows != entry["rows"]:
            raise BrightMaterializationError(f"row count mismatch for {key}")
    _validate_materialized_content(output)
    return manifest


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            count += 1
    return {"path": path.name, "rows": count, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def download_sources(cache_root: str | Path, tracks: Iterable[str] = TRACKS) -> dict[str, Any]:
    """Download only pinned target Parquet files under strict byte limits."""
    cache = Path(cache_root)
    selected = tuple(tracks)
    for track in selected:
        if track not in TRACKS:
            raise BrightMaterializationError(f"unsupported BRIGHT track: {track}")
    expected_total = sum(EXPECTED[track][role]["bytes"] for track in selected for role in ROLES)
    if expected_total > MAX_TOTAL_DOWNLOAD_BYTES:
        raise BrightMaterializationError("download plan exceeds total byte limit")
    files = []
    for track in selected:
        for role in ROLES:
            expected = EXPECTED[track][role]
            if expected["bytes"] > MAX_FILE_BYTES:
                raise BrightMaterializationError("download plan exceeds per-file byte limit")
            destination = cache / role / f"{track}-00000-of-00001.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            valid_cached_file = (
                destination.is_file()
                and destination.stat().st_size == expected["bytes"]
                and _sha256(destination) == expected["sha256"]
            )
            if not valid_cached_file:
                temporary = destination.with_suffix(".parquet.part")
                if temporary.exists():
                    temporary.unlink()
                with (
                    urllib.request.urlopen(source_url(track, role), timeout=60) as response,
                    temporary.open("wb") as output,
                ):
                    copied = 0
                    while chunk := response.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > expected["bytes"] or copied > MAX_FILE_BYTES:
                            raise BrightMaterializationError(f"source exceeded byte limit: {track}/{role}")
                        output.write(chunk)
                temporary.replace(destination)
            if (
                destination.stat().st_size != expected["bytes"]
                or _sha256(destination) != expected["sha256"]
            ):
                raise BrightMaterializationError(f"source identity mismatch: {track}/{role}")
            files.append({"track": track, "role": role, "path": destination.as_posix(), **expected})
    return {"revision": HF_REVISION, "expected_total_bytes": expected_total, "files": files}


def _parquet_file(cache_root: str | Path, track: str, role: str) -> Path:
    _validate_track_role(track, role)
    path = Path(cache_root) / role / f"{track}-00000-of-00001.parquet"
    expected = EXPECTED[track][role]
    if not path.is_file() or path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
        raise BrightMaterializationError(f"missing or invalid pinned source: {track}/{role}")
    return path


def _schema_map(schema: Any) -> dict[str, str]:
    return {field.name: str(field.type) for field in schema}


def preflight_track(cache_root: str | Path, track: str, sample_rows: int = 8) -> dict[str, Any]:
    """Read real Parquet metadata, schemas, and a bounded first batch."""
    import pyarrow.parquet as pq

    if not 1 <= sample_rows <= 128:
        raise BrightMaterializationError("sample_rows must be between 1 and 128")
    report: dict[str, Any] = {"track": track, "sample_rows_requested": sample_rows, "files": {}}
    for role, required_schema in (("documents", DOCUMENT_SCHEMA), ("examples", EXAMPLE_SCHEMA)):
        path = _parquet_file(cache_root, track, role)
        parquet = pq.ParquetFile(path)
        schema = _schema_map(parquet.schema_arrow)
        if schema != required_schema:
            raise BrightMaterializationError(f"unexpected Parquet schema: {track}/{role}: {schema}")
        if parquet.metadata.num_rows != EXPECTED[track][role]["rows"]:
            raise BrightMaterializationError(f"unexpected Parquet row count: {track}/{role}")
        batch = next(parquet.iter_batches(batch_size=sample_rows))
        rows = batch.to_pylist()
        if not rows:
            raise BrightMaterializationError(f"empty bounded preflight: {track}/{role}")
        report["files"][role] = {
            "url": source_url(track, role),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "created_by": parquet.metadata.created_by,
            "schema": schema,
            "sample_rows_read": len(rows),
            "sample_ids_sha256": hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest(),
        }
    return report


def load_track(cache_root: str | Path, track: str) -> TrackRows:
    """Load and validate a complete pinned track with no remote code execution."""
    import pyarrow.parquet as pq

    preflight_track(cache_root, track)
    documents_raw = pq.read_table(_parquet_file(cache_root, track, "documents")).to_pylist()
    examples_raw = pq.read_table(_parquet_file(cache_root, track, "examples")).to_pylist()
    documents: list[dict[str, str]] = []
    document_ids: set[str] = set()
    for row in documents_raw:
        doc_id, content = row.get("id"), row.get("content")
        if not isinstance(doc_id, str) or not doc_id or doc_id in document_ids:
            raise BrightMaterializationError(f"invalid or duplicate document id in {track}")
        if not isinstance(content, str) or not content.strip():
            raise BrightMaterializationError(f"empty document content in {track}: {doc_id}")
        document_ids.add(doc_id)
        documents.append({"id": doc_id, "content": content.strip()})
    examples: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for row in examples_raw:
        query_id, query = row.get("id"), row.get("query")
        if not isinstance(query_id, str) or not query_id or query_id in query_ids:
            raise BrightMaterializationError(f"invalid or duplicate query id in {track}")
        if not isinstance(query, str) or not query.strip():
            raise BrightMaterializationError(f"empty query in {track}: {query_id}")
        gold = row.get("gold_ids")
        excluded = row.get("excluded_ids")
        gold_long = row.get("gold_ids_long")
        for label, values in (("gold_ids", gold), ("excluded_ids", excluded), ("gold_ids_long", gold_long)):
            if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
                raise BrightMaterializationError(f"invalid {label} in {track}: {query_id}")
            if len(values) != len(set(values)):
                raise BrightMaterializationError(f"duplicate {label} in {track}: {query_id}")
        if excluded == [NO_EXCLUDED_IDS_SENTINEL]:
            excluded = []
        elif NO_EXCLUDED_IDS_SENTINEL in excluded:
            raise BrightMaterializationError(f"mixed excluded_ids sentinel in {track}: {query_id}")
        if not gold:
            raise BrightMaterializationError(f"empty gold set in {track}: {query_id}")
        dangling_gold = sorted(set(gold) - document_ids)
        dangling_excluded = sorted(set(excluded) - document_ids)
        if dangling_gold or dangling_excluded:
            raise BrightMaterializationError(
                f"dangling passage ids in {track}/{query_id}: gold={dangling_gold[:3]} excluded={dangling_excluded[:3]}"
            )
        if set(gold) & set(excluded):
            raise BrightMaterializationError(f"gold/excluded overlap in {track}: {query_id}")
        query_ids.add(query_id)
        examples.append({
            "id": query_id,
            "query": query.strip(),
            "gold_ids": list(gold),
            "excluded_ids": list(excluded),
            "gold_ids_long": list(gold_long),
        })
    return TrackRows(
        track,
        tuple(sorted(documents, key=lambda row: row["id"])),
        tuple(sorted(examples, key=lambda row: row["id"])),
    )


def _percentiles(values: list[int]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    keys = ("min", "p50", "p90", "p95", "p99", "max")
    result = dict(zip(keys, (float(value) for value in np.percentile(array, [0, 50, 90, 95, 99, 100]))))
    result["mean"] = float(array.mean())
    return result


def audit_track(data: TrackRows) -> dict[str, Any]:
    gold_counts = [len(row["gold_ids"]) for row in data.examples]
    excluded_counts = [len(row["excluded_ids"]) for row in data.examples]
    long_counts = [len(row["gold_ids_long"]) for row in data.examples]
    gold_ids = [doc_id for row in data.examples for doc_id in row["gold_ids"]]
    excluded_ids = [doc_id for row in data.examples for doc_id in row["excluded_ids"]]
    return {
        "track": data.track,
        "documents": len(data.documents),
        "queries": len(data.examples),
        "positive_qrels": len(gold_ids),
        "grade_counts": {"1": len(gold_ids)},
        "gold_per_query": _percentiles(gold_counts),
        "excluded_per_query": _percentiles(excluded_counts),
        "gold_long_per_query": _percentiles(long_counts),
        "unique_gold_passage_ids": len(set(gold_ids)),
        "unique_excluded_passage_ids": len(set(excluded_ids)),
        "gold_excluded_overlap": len(set(gold_ids) & set(excluded_ids)),
        "duplicate_document_content_groups": sum(
            count > 1 for count in Counter(row["content"] for row in data.documents).values()
        ),
        "label_contract": {
            "gold_ids": "positive qrels with grade 1",
            "excluded_ids": "evaluation-time candidate filtering only; never negative qrels",
            "gold_ids_long": "long-document identifiers retained as query metadata; never passage qrels",
            "excluded_ids_source_sentinel": (
                f"{NO_EXCLUDED_IDS_SENTINEL!r} is normalized to an empty list only when it is the sole value"
            ),
        },
        "contamination": {
            "benchmark_first_published": "2024-07-16",
            "training_overlap_status": "unknown_not_zero_shot_verified",
            "risk": "public benchmark queries and labels may have entered later training or evaluation pipelines",
        },
    }


def materialize_track(cache_root: str | Path, output_root: str | Path, track: str = "biology") -> dict[str, Any]:
    """Create canonical runner-compatible files for one complete track."""
    if track not in MATERIALIZABLE_TRACKS:
        raise BrightMaterializationError(f"track is not approved for complete materialization: {track}")
    data = load_track(cache_root, track)
    output = Path(output_root)
    staging = output.with_name(f"{output.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    files = {}
    files["corpus.jsonl"] = _write_jsonl(staging / "corpus.jsonl", data.documents)
    files["queries.jsonl"] = _write_jsonl(
        staging / "queries.jsonl",
        (
            {
                "id": row["id"],
                "text": row["query"],
                "excluded_ids": row["excluded_ids"],
                "gold_ids_long": row["gold_ids_long"],
            }
            for row in data.examples
        ),
    )
    qrel_rows = [
        (row["id"], doc_id, 1)
        for row in data.examples
        for doc_id in sorted(row["gold_ids"])
    ]
    qrels_path = staging / "qrels" / "test.tsv"
    qrels_path.parent.mkdir()
    with qrels_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, doc_id, grade in qrel_rows:
            if grade != 1:
                raise BrightMaterializationError("illegal qrel grade")
            handle.write(f"{query_id}\t{doc_id}\t{grade}\n")
    files["qrels/test.tsv"] = {
        "path": "qrels/test.tsv",
        "rows": len(qrel_rows),
        "bytes": qrels_path.stat().st_size,
        "sha256": _sha256(qrels_path),
    }
    audit = audit_track(data)
    audit_path = staging / "audit.json"
    audit_path.write_bytes(_canonical_json(audit))
    files["audit.json"] = {"path": "audit.json", "bytes": audit_path.stat().st_size, "sha256": _sha256(audit_path)}
    source_files = [
        {"role": role, "url": source_url(track, role), **EXPECTED[track][role]}
        for role in ROLES
    ]
    manifest = {
        "schema_version": "1",
        "benchmark_version": BENCHMARK_VERSION,
        "track": track,
        "source": {
            "dataset": HF_DATASET,
            "revision": HF_REVISION,
            "trust_remote_code": False,
            "files": source_files,
        },
        "materialization": {
            "deterministic": True,
            "ordering": "lexicographic query/document ids; lexicographic gold ids per query",
            "text_protocol": "content.strip()",
            "files": files,
        },
        "runner_adapter": {
            "corpus": {"id": "id", "text": "content"},
            "queries": {"id": "id", "text": "text"},
            "qrels": {"query_id": "query-id", "document_id": "corpus-id", "grade": "score"},
            "excluded_ids_policy": "filter candidates before ranking and metric computation",
        },
        "publication": {
            "classification": "research_only",
            "public_export_allowed": False,
            "gate": "closed_pending_upstream_web_content_rights_review",
            "reason": (
                "The repository-level CC-BY-4.0 declaration does not establish that every copied upstream "
                "web page was sublicensable for public redistribution."
            ),
        },
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    validate_materialization(staging)
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--cache-root", required=True)
    download.add_argument("--tracks", nargs="+", choices=TRACKS, default=list(TRACKS))
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--cache-root", required=True)
    preflight.add_argument("--tracks", nargs="+", choices=TRACKS, default=list(TRACKS))
    preflight.add_argument("--sample-rows", type=int, default=8)
    preflight.add_argument("--output")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--cache-root", required=True)
    materialize.add_argument("--track", choices=MATERIALIZABLE_TRACKS, default="biology")
    materialize.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "download":
        result = download_sources(args.cache_root, args.tracks)
    elif args.command == "preflight":
        result = {
            "schema_version": "1",
            "benchmark_version": BENCHMARK_VERSION,
            "revision": HF_REVISION,
            "trust_remote_code": False,
            "byte_limits": {"total": MAX_TOTAL_DOWNLOAD_BYTES, "per_file": MAX_FILE_BYTES},
            "tracks": {track: preflight_track(args.cache_root, track, args.sample_rows) for track in args.tracks},
        }
        if args.output:
            Path(args.output).write_bytes(_canonical_json(result))
    else:
        result = materialize_track(args.cache_root, args.output, args.track)
    print(json.dumps(result, indent=2, sort_keys=True))
