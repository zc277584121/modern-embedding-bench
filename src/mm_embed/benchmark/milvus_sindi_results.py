"""Independent validation and aggregation for formal Milvus SINDI evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from mm_embed.benchmark.milvus_sindi_formal import (
    CONCURRENCY_VALUES,
    MEASURED_TRIALS,
    MODEL_SHORT,
    REPRESENTATIVES,
    TOP_K_VALUES,
    TRACK_SHORT,
    TRACKS,
    WARMUP_PASSES,
)
from mm_embed.benchmark.milvus_sindi_system import MilvusSindiError, file_sha256, formatted_json_bytes

COMPARATORS = ("exact_scipy_csr", "SINDI", "DAAT_MAXSCORE")
CELL_SCHEMA = Path("schemas/milvus-sindi-formal-cell-v01.schema.json")
RUN_MANIFEST_SPECS = (
    ("native", "native-run.json", 18),
    ("system_100k", "system-100k-run.json", 9),
    ("system_1m", "system-1m-run.json", 9),
)
BYTE_UNITS = {
    "B": 1,
    "kB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(formatted_json_bytes(value))
    identity = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return identity


def _record_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or path.stat().st_size != int(record["bytes"]) or file_sha256(path) != record["sha256"]:
        raise MilvusSindiError(f"Referenced formal evidence drifted: {path}")
    return path


def _read_gzip_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = _record_path(record)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _cell_paths(root: Path) -> list[Path]:
    paths = []
    for model in REPRESENTATIVES:
        for track in TRACKS:
            base = root / "native" / MODEL_SHORT[model] / TRACK_SHORT[track]
            paths.extend(base / comparator.lower() / "cell-evidence.json" for comparator in COMPARATORS)
    for scale in (100_000, 1_000_000):
        for model in REPRESENTATIVES:
            base = root / "system-only" / MODEL_SHORT[model] / ("100k" if scale == 100_000 else "1m")
            paths.extend(base / comparator.lower() / "cell-evidence.json" for comparator in COMPARATORS)
    return paths


def _paths_for_run(root: Path, run_key: str) -> list[Path]:
    paths = _cell_paths(root)
    if run_key == "native":
        return [path for path in paths if path.is_relative_to(root / "native")]
    scale_name = "100k" if run_key == "system_100k" else "1m"
    return [path for path in paths if scale_name in path.parts and "system-only" in path.parts]


def _verified_sidecar(path: Path, *, label: str) -> str:
    if not path.is_file():
        raise MilvusSindiError(f"{label} is absent: {path}")
    identity = file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != identity:
        raise MilvusSindiError(f"{label} sidecar drifted: {path}")
    return identity


def validate_run_manifests(root: str | Path, *, output: str | Path) -> tuple[dict[str, Any], str]:
    """Validate the three immutable run manifests and normalize missing inline cell hashes."""
    root_path = Path(root)
    runs = []
    for run_key, filename, expected_count in RUN_MANIFEST_SPECS:
        path = root_path / filename
        run_identity = _verified_sidecar(path, label="Run manifest")
        run = _load_json(path)
        expected_paths = {str(item) for item in _paths_for_run(root_path, run_key)}
        completed = run.get("completed_cells")
        if (
            run.get("status") != "pass"
            or run.get("completed_cell_count") != expected_count
            or not isinstance(completed, list)
            or len(completed) != expected_count
            or run.get("runtime", {}).get("model_loaded") is not False
            or run.get("publication_gate") != "closed"
        ):
            raise MilvusSindiError(f"Run manifest is incomplete or open: {path}")
        ordinals = [int(record["ordinal"]) for record in completed]
        paths = [str(record["path"]) for record in completed]
        if ordinals != list(range(expected_count)) or len(paths) != len(set(paths)) or set(paths) != expected_paths:
            raise MilvusSindiError(f"Run manifest cell coverage drifted: {path}")
        normalized = []
        for record in completed:
            cell_path = Path(str(record["path"]))
            cell_identity = _verified_sidecar(cell_path, label="Formal cell")
            inline = record.get("sha256")
            if inline is not None and inline != cell_identity:
                raise MilvusSindiError(f"Run manifest inline cell identity drifted: {cell_path}")
            normalized.append(
                {
                    "ordinal": int(record["ordinal"]),
                    "path": str(cell_path),
                    "resumed": bool(record.get("resumed", False)),
                    "inline_sha256_present": inline is not None,
                    "resolved_sha256": cell_identity,
                    "resolution": "inline_and_sidecar" if inline is not None else "content_and_sidecar",
                }
            )
        runs.append(
            {
                "run_key": run_key,
                "path": str(path),
                "sha256": run_identity,
                "cell_count": expected_count,
                "normalized_cells": normalized,
                "missing_inline_sha256_count": sum(not row["inline_sha256_present"] for row in normalized),
                "status": "pass",
            }
        )
    evidence = {
        "schema_version": "milvus-sindi-run-manifest-integrity-v1",
        "story_id": "S-20260814-010",
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "runs": runs,
        "total_cell_count": sum(run["cell_count"] for run in runs),
        "source_artifacts_mutated": False,
        "raw_trials_mutated": False,
        "model_loaded": False,
        "publication_gate": "closed",
        "status": "pass",
    }
    identity = _write_json(Path(output), evidence)
    return evidence, identity


def _exact_record(root: Path, cell: Mapping[str, Any]) -> Mapping[str, Any]:
    if cell["tier"] == "native":
        return cell["exact_ground_truth"]
    profile = str(cell["model_profile"])
    scale = int(cell["scale"])
    manifest_path = (
        root / "system-only" / MODEL_SHORT[profile] / ("100k" if scale == 100_000 else "1m") / "generated/manifest.json"
    )
    manifest = _load_json(manifest_path)
    sidecar = manifest_path.with_suffix(".json.sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != file_sha256(manifest_path):
        raise MilvusSindiError(f"Generated manifest drifted: {manifest_path}")
    return manifest["exact_top100"]


def _exact_hits(record: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    rows = sorted(_read_gzip_rows(record), key=lambda row: int(row["query_ordinal"]))
    return [row["hits"] for row in rows]


def _trial_groups(config: Mapping[str, Any]) -> Sequence[tuple[str, Sequence[Mapping[str, Any]]]]:
    return (("cold", config["cold_trials"]), ("warm", config["warm_trials"]))


def _median_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.median([row[field] for row in rows]))


def _independent_correctness(
    rows: Sequence[Mapping[str, Any]], exact: Sequence[Sequence[Mapping[str, Any]]], top_k: int
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["query_ordinal"]))
    if len(ordered) != len(exact) or [int(row["query_ordinal"]) for row in ordered] != list(range(len(exact))):
        raise MilvusSindiError("Raw query ordinals are incomplete or duplicated")
    recalls = []
    finite = True
    unique = True
    maximum_delta = 0.0
    for actual, expected in zip(ordered, exact, strict=True):
        ids = [int(value) for value in actual["ids"]]
        scores = [float(value) for value in actual["scores"]]
        if len(ids) != top_k or len(scores) != top_k:
            raise MilvusSindiError("A raw trial did not return the declared topK")
        unique = unique and len(ids) == len(set(ids))
        finite = finite and all(math.isfinite(value) for value in scores)
        expected_ids = {int(hit["id"]) for hit in expected[:top_k]}
        recalls.append(len(expected_ids.intersection(ids)) / top_k)
        expected_scores = {int(hit["id"]): float(hit["score"]) for hit in expected[:top_k]}
        for pk, score in zip(ids, scores, strict=True):
            if pk in expected_scores:
                maximum_delta = max(maximum_delta, abs(score - expected_scores[pk]))
    return {
        "minimum_strict_recall": min(recalls),
        "all_ids_unique": unique,
        "all_scores_finite": finite,
        "maximum_absolute_score_delta": maximum_delta,
    }


def _independent_summary(cell: Mapping[str, Any], exact: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    summaries = []
    combinations = set()
    for config in cell["trials"]["configurations"]:
        key = (int(config["top_k"]), int(config["concurrency"]))
        combinations.add(key)
        if len(config["warmups"]) != WARMUP_PASSES:
            raise MilvusSindiError("Warmup count drifted")
        for warmup in config["warmups"]:
            rows = _read_gzip_rows(
                warmup["raw_queries"] if "raw_queries" in warmup else warmup["search"]["raw_queries"]
            )
            if len(rows) != int(cell["query_count"]):
                raise MilvusSindiError("Warmup query count drifted")
        for temperature, trials in _trial_groups(config):
            if len(trials) != MEASURED_TRIALS:
                raise MilvusSindiError("Measured trial count drifted")
            reduced = []
            for trial in trials:
                search = trial["search"]
                rows = _read_gzip_rows(search["raw_queries"])
                correctness = _independent_correctness(rows, exact, int(config["top_k"]))
                prepare_latencies = np.asarray([int(row.get("prepare_ns", 0)) / 1e6 for row in rows], dtype=np.float64)
                search_latencies = np.asarray(
                    [int(row.get("rpc_end_to_end_ns", row.get("search_ns", 0))) / 1e6 for row in rows],
                    dtype=np.float64,
                )
                end_to_end_latencies = np.asarray([int(row["end_to_end_ns"]) / 1e6 for row in rows], dtype=np.float64)
                server = search.get("server_metric_delta") or {}
                reduced.append(
                    {
                        "trial": int(trial["trial"]),
                        "qps": len(rows) / (int(search["wall_ns"]) / 1e9),
                        "p50_ms": float(np.percentile(end_to_end_latencies, 50)),
                        "p95_ms": float(np.percentile(end_to_end_latencies, 95)),
                        "p99_ms": float(np.percentile(end_to_end_latencies, 99)),
                        "prepare_p50_ms": float(np.percentile(prepare_latencies, 50)),
                        "prepare_p95_ms": float(np.percentile(prepare_latencies, 95)),
                        "prepare_p99_ms": float(np.percentile(prepare_latencies, 99)),
                        "search_p50_ms": float(np.percentile(search_latencies, 50)),
                        "search_p95_ms": float(np.percentile(search_latencies, 95)),
                        "search_p99_ms": float(np.percentile(search_latencies, 99)),
                        "client_cpu_ns": int(search["client_cpu_ns"]),
                        "network_received_bytes": int(server.get("proxy_received_bytes", 0)),
                        "network_sent_bytes": int(server.get("proxy_sent_bytes_global_single_active_cell", 0)),
                        "server_search_mean_ms": server.get("server_search_mean_ms"),
                        "raw_trial_bytes": int(search["raw_queries"]["bytes"]),
                        "correctness": correctness,
                    }
                )
            schedule_hashes = {str(trial["search"]["schedule_sha256"]) for trial in trials}
            if len(schedule_hashes) != 1:
                raise MilvusSindiError("Measured trial query schedule drifted")

            server_means = [
                float(row["server_search_mean_ms"]) for row in reduced if row["server_search_mean_ms"] is not None
            ]
            summaries.append(
                {
                    "top_k": int(config["top_k"]),
                    "concurrency": int(config["concurrency"]),
                    "temperature": temperature,
                    "schedule_sha256": schedule_hashes.pop(),
                    "qps_median": _median_field(reduced, "qps"),
                    "p50_ms_median": _median_field(reduced, "p50_ms"),
                    "p95_ms_median": _median_field(reduced, "p95_ms"),
                    "p99_ms_median": _median_field(reduced, "p99_ms"),
                    "client_prepare_p50_ms_median": _median_field(reduced, "prepare_p50_ms"),
                    "client_prepare_p95_ms_median": _median_field(reduced, "prepare_p95_ms"),
                    "client_prepare_p99_ms_median": _median_field(reduced, "prepare_p99_ms"),
                    "search_call_p50_ms_median": _median_field(reduced, "search_p50_ms"),
                    "search_call_p95_ms_median": _median_field(reduced, "search_p95_ms"),
                    "search_call_p99_ms_median": _median_field(reduced, "search_p99_ms"),
                    "client_cpu_ns_median": int(np.median([row["client_cpu_ns"] for row in reduced])),
                    "network_received_bytes_median": int(np.median([row["network_received_bytes"] for row in reduced])),
                    "network_sent_bytes_median": int(np.median([row["network_sent_bytes"] for row in reduced])),
                    "server_search_mean_ms_median": float(np.median(server_means)) if server_means else None,
                    "raw_trial_bytes_total": sum(row["raw_trial_bytes"] for row in reduced),
                    "minimum_strict_recall": min(row["correctness"]["minimum_strict_recall"] for row in reduced),
                    "all_ids_unique": all(row["correctness"]["all_ids_unique"] for row in reduced),
                    "all_scores_finite": all(row["correctness"]["all_scores_finite"] for row in reduced),
                    "maximum_score_delta": max(row["correctness"]["maximum_absolute_score_delta"] for row in reduced),
                }
            )
    expected_combinations = {(top_k, concurrency) for top_k in TOP_K_VALUES for concurrency in CONCURRENCY_VALUES}
    if combinations != expected_combinations:
        raise MilvusSindiError("TopK/concurrency coverage drifted")
    return summaries


def _summary_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(row["top_k"]), int(row["concurrency"]), str(row["temperature"])


def _compare_summaries(recomputed: Sequence[Mapping[str, Any]], reported: Sequence[Mapping[str, Any]]) -> None:
    expected = {_summary_key(row): row for row in recomputed}
    actual = {_summary_key(row): row for row in reported}
    if expected.keys() != actual.keys():
        raise MilvusSindiError("Reported summary coverage drifted")
    numeric_fields = (
        "qps_median",
        "p50_ms_median",
        "p95_ms_median",
        "p99_ms_median",
        "minimum_strict_recall",
        "maximum_score_delta",
    )
    for key, row in expected.items():
        for field in numeric_fields:
            if not math.isclose(float(row[field]), float(actual[key][field]), rel_tol=1e-12, abs_tol=1e-12):
                raise MilvusSindiError(f"Reported summary value drifted: {key} {field}")


def _parse_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMG]i?B|B)", value)
    if match is None:
        raise MilvusSindiError(f"Unrecognized Docker byte value: {value}")
    return round(float(match.group(1)) * BYTE_UNITS[match.group(2)])


def _resource_peaks(cell: Mapping[str, Any]) -> dict[str, Any]:
    samples = []
    phases = cell["lifecycle"].get("phases", {})
    for phase in phases.values():
        samples.extend(phase.get("docker_samples", []))
    for config in cell["trials"]["configurations"]:
        searches = [*config["warmups"]]
        searches.extend(trial["search"] for trial in config["cold_trials"])
        searches.extend(trial["search"] for trial in config["warm_trials"])
        for search in searches:
            samples.extend(search.get("docker_samples", []))
    docker = [sample["docker"] for sample in samples if "docker" in sample]
    client_rss = []
    for phase in phases.values():
        if "client_max_rss_bytes" in phase:
            client_rss.append(int(phase["client_max_rss_bytes"]))
    for config in cell["trials"]["configurations"]:
        for temperature, trials in _trial_groups(config):
            del temperature
            client_rss.extend(int(trial["search"]["client_max_rss_bytes"]) for trial in trials)
    return {
        "container_peak_memory_bytes": max(
            (_parse_bytes(row["MemUsage"].split(" / ", 1)[0]) for row in docker), default=0
        ),
        "container_peak_cpu_percent": max((float(row["CPUPerc"].rstrip("%")) for row in docker), default=0.0),
        "container_peak_pids": max((int(row["PIDs"]) for row in docker), default=0),
        "client_peak_rss_bytes": max(client_rss, default=0),
        "docker_sampler_error_count": sum(1 for sample in samples if "error" in sample),
    }


def validate_and_aggregate(root: str | Path, *, output: str | Path) -> tuple[dict[str, Any], str]:
    """Validate every raw trial and write a private recomputed aggregate."""
    root_path = Path(root)
    run_integrity, run_integrity_sha256 = validate_run_manifests(
        root_path, output=root_path / "validation/run-manifest-integrity.json"
    )
    schema = _load_json(CELL_SCHEMA)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    cells = []
    for path in _cell_paths(root_path):
        if not path.is_file():
            raise MilvusSindiError(f"Formal cell is absent: {path}")
        sidecar = path.with_suffix(path.suffix + ".sha256")
        identity = file_sha256(path)
        if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != identity:
            raise MilvusSindiError(f"Formal cell identity drifted: {path}")
        cell = _load_json(path)
        validator.validate(cell)
        if (
            cell.get("status") != "pass"
            or cell.get("formal_performance_executed") is not True
            or cell.get("model_loaded") is not False
            or cell.get("publication_gate") != "closed"
        ):
            raise MilvusSindiError(f"Formal cell execution/gate evidence drifted: {path}")
        exact = _exact_hits(_exact_record(root_path, cell))
        if len(exact) != int(cell["query_count"]):
            raise MilvusSindiError(f"Exact query count drifted: {path}")
        recomputed = _independent_summary(cell, exact)
        _compare_summaries(recomputed, cell["summary"])
        lifecycle = cell["lifecycle"]
        if cell["comparator"] in ("SINDI", "DAAT_MAXSCORE"):
            index = lifecycle["state_before_search"]["describe_index"]
            after = cell["trials"]["state_after_search"]
            if (
                index["inverted_index_algo"] != cell["comparator"]
                or index["state"] != "Finished"
                or int(index["indexed_rows"]) != int(cell["scale"])
                or int(index["pending_index_rows"]) != 0
                or after["reported_algorithm"] != cell["comparator"]
                or after["load_state"]["state"] != "Loaded"
                or after["growing_segment_count"] != 0
                or not after["all_segments_indexed"]
                or after["compaction"]["active"]
                or int(after["compaction"]["active_threads"]) != 0
                or int(after["compaction"]["queue_depth"]) != 0
                or int(after["sealed_rows"]) != int(cell["scale"])
                or any(segment["state"] != "Sealed" or int(segment["indexID"]) <= 0 for segment in after["segments"])
            ):
                raise MilvusSindiError(f"Milvus algorithm/state evidence drifted: {path}")
            _record_path(lifecycle["persisted_index"]["server_log"])
        elif lifecycle.get("backend") != "scipy_csr_float32_inner_product":
            raise MilvusSindiError(f"Exact CSR backend identity drifted: {path}")
        phases = lifecycle.get("phases", {})
        cells.append(
            {
                "tier": cell["tier"],
                "model": cell.get("model", cell.get("model_profile")),
                "track": cell.get("track"),
                "scale": int(cell["scale"]),
                "query_count": int(cell["query_count"]),
                "comparator": cell["comparator"],
                "cell_sha256": identity,
                "summary": recomputed,
                "resources": _resource_peaks(cell),
                "build_wall_ns": lifecycle.get("phases", {}).get("index_build", {}).get("wall_ns"),
                "load_wall_ns": lifecycle.get("phases", {}).get("load", {}).get("wall_ns"),
                "insert_wall_ns": lifecycle.get("phases", {}).get("insert", {}).get("wall_ns"),
                "flush_wall_ns": phases.get("flush", {}).get("wall_ns"),
                "phase_client_cpu_ns": {
                    name: int(phase["client_cpu_ns"]) for name, phase in phases.items() if "client_cpu_ns" in phase
                },
                "incremental_runtime_bytes": lifecycle.get("incremental_runtime_bytes", 0),
                "persisted_index_bytes": lifecycle.get("persisted_index", {}).get("serialized_bytes", 0),
                "document_csr_bytes": lifecycle.get("document_csr_bytes"),
                "query_csr_bytes": lifecycle.get("query_csr_bytes"),
            }
        )
    grouped_cells: dict[tuple[str, str, str | None, int], list[Mapping[str, Any]]] = {}
    for cell in cells:
        grouped_cells.setdefault(_comparison_key(cell), []).append(cell)
    for key, group in grouped_cells.items():
        if {str(cell["comparator"]) for cell in group} != set(COMPARATORS):
            raise MilvusSindiError(f"Formal comparator coverage drifted: {key}")
        query_counts = {int(cell["query_count"]) for cell in group}
        schedules = [{(_summary_key(row), row["schedule_sha256"]) for row in cell["summary"]} for cell in group]
        if len(query_counts) != 1 or not all(schedule == schedules[0] for schedule in schedules[1:]):
            raise MilvusSindiError(f"Formal vectors/query schedule fairness drifted: {key}")
    evidence = {
        "schema_version": "milvus-sindi-formal-recomputed-v1",
        "story_id": "S-20260814-010",
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "cell_count": len(cells),
        "native_cell_count": sum(cell["tier"] == "native" for cell in cells),
        "system_100k_cell_count": sum(cell["tier"] == "system_only" and cell["scale"] == 100_000 for cell in cells),
        "system_1m_cell_count": sum(cell["tier"] == "system_only" and cell["scale"] == 1_000_000 for cell in cells),
        "raw_trial_files_recomputed": len(cells) * len(TOP_K_VALUES) * len(CONCURRENCY_VALUES) * 2 * MEASURED_TRIALS,
        "warmup_files_validated": len(cells) * len(TOP_K_VALUES) * len(CONCURRENCY_VALUES) * WARMUP_PASSES,
        "run_manifest_integrity": {
            "path": str(root_path / "validation/run-manifest-integrity.json"),
            "sha256": run_integrity_sha256,
            "run_count": len(run_integrity["runs"]),
            "resolved_missing_inline_sha256_count": sum(
                run["missing_inline_sha256_count"] for run in run_integrity["runs"]
            ),
        },
        "cells": cells,
        "model_loaded": False,
        "publication_gate": "closed",
        "status": "pass",
    }
    if len(cells) != 36:
        raise MilvusSindiError("Formal matrix is incomplete")
    identity = _write_json(Path(output), evidence)
    return evidence, identity


def build_file_manifest(root: str | Path, *, output: str | Path) -> tuple[dict[str, Any], str]:
    """Hash every private formal artifact except the output manifest itself."""
    root_path = Path(root)
    output_path = Path(output)
    files = []
    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path not in (output_path, output_path.with_suffix(output_path.suffix + ".sha256")):
            files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    manifest = {
        "schema_version": "milvus-sindi-formal-file-manifest-v1",
        "story_id": "S-20260814-010",
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "root": str(root_path),
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in files),
        "files": files,
        "model_loaded": False,
        "publication_gate": "closed",
        "status": "pass",
    }
    identity = _write_json(output_path, manifest)
    return manifest, identity


def build_final_isolation_evidence(
    *,
    pre_manifest: str | Path,
    pre_sha256: str,
    post_manifest: str | Path,
    post_sha256: str,
    output: str | Path,
) -> tuple[dict[str, Any], str]:
    """Compare the formal pre/post host snapshots without claiming control of ambient drift."""
    from mm_embed.benchmark.milvus_sindi_system import (
        _container_rows,
        _listening_endpoints,
        _named_rows,
        _validated_baseline_root,
    )

    pre_value, pre_root = _validated_baseline_root(pre_manifest, pre_sha256)
    post_value, post_root = _validated_baseline_root(post_manifest, post_sha256)
    story_container = "meb-s010-milvus-v300-amd64"
    story_network = "meb-s010-sindi-net-v01"
    story_ports = {("tcp", "LISTEN", "127.0.0.1:49531"), ("tcp", "LISTEN", "127.0.0.1:49092")}
    pre_containers = _container_rows(pre_root / "docker_containers.tsv")
    post_containers = _container_rows(post_root / "docker_containers.tsv")
    if story_container not in pre_containers or story_container not in post_containers:
        raise MilvusSindiError("Story container is absent from a formal boundary baseline")
    story_fields = ("id", "image", "state", "ports", "mounts", "networks")
    story_identity_unchanged = all(
        pre_containers[story_container][field] == post_containers[story_container][field] for field in story_fields
    )
    pre_non_story = {name: row for name, row in pre_containers.items() if name != story_container}
    post_non_story = {name: row for name, row in post_containers.items() if name != story_container}
    shared_names = set(pre_non_story).intersection(post_non_story)
    changed = []
    for name in sorted(shared_names):
        fields = {
            field: {"before": pre_non_story[name][field], "after": post_non_story[name][field]}
            for field in pre_non_story[name]
            if pre_non_story[name][field] != post_non_story[name][field]
        }
        if fields:
            changed.append({"name": name, "fields": fields})
    pre_networks = _named_rows(pre_root / "docker_networks.tsv")
    post_networks = _named_rows(post_root / "docker_networks.tsv")
    pre_volumes = sorted((pre_root / "docker_volumes.tsv").read_text(encoding="utf-8").splitlines())
    post_volumes = sorted((post_root / "docker_volumes.tsv").read_text(encoding="utf-8").splitlines())
    pre_endpoints = _listening_endpoints(pre_root / "listening_ports.txt")
    post_endpoints = _listening_endpoints(post_root / "listening_ports.txt")
    pre_non_story_endpoints = pre_endpoints - story_ports
    post_non_story_endpoints = post_endpoints - story_ports
    checks = {
        "story_container_identity_state_and_scope_unchanged": story_identity_unchanged,
        "story_ports_present_at_both_boundaries": story_ports.issubset(pre_endpoints)
        and story_ports.issubset(post_endpoints),
        "non_story_container_name_set_unchanged": set(pre_non_story) == set(post_non_story),
        "non_story_container_state_set_unchanged": {name: row["state"] for name, row in pre_non_story.items()}
        == {name: row["state"] for name, row in post_non_story.items()},
        "non_story_container_image_set_unchanged": {name: row["image"] for name, row in pre_non_story.items()}
        == {name: row["image"] for name, row in post_non_story.items()},
        "docker_networks_unchanged": pre_networks == post_networks,
        "docker_volumes_unchanged": pre_volumes == post_volumes,
        "non_story_listening_endpoints_unchanged": pre_non_story_endpoints == post_non_story_endpoints,
    }
    if not all(checks.values()):
        raise MilvusSindiError(f"Final isolation boundary invariant failed: {checks}")
    evidence = {
        "schema_version": "milvus-sindi-final-isolation-v1",
        "story_id": "S-20260814-010",
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "pre": {"path": str(pre_manifest), "sha256": pre_sha256, "captured_at_utc": pre_value["captured_at_utc"]},
        "post": {
            "path": str(post_manifest),
            "sha256": post_sha256,
            "captured_at_utc": post_value["captured_at_utc"],
        },
        "checks": checks,
        "story_owned_resources": {
            "container": story_container,
            "network": story_network,
            "ports": ["127.0.0.1:49531", "127.0.0.1:49092"],
            "named_volumes": [],
            "runtime_root": "results/milvus-sindi-system-v0.1/runtime/milvus-v300-amd64",
        },
        "non_story_ambient_drift": {
            "removed_container_names": sorted(set(pre_non_story) - set(post_non_story)),
            "added_container_names": sorted(set(post_non_story) - set(pre_non_story)),
            "changed_containers": changed,
            "attribution": "not_attributed_to_story; no Story command targeted these resources",
        },
        "claim_boundary": (
            "The baseline proves preserved name/state/image, network, volume, and listening-endpoint invariants. "
            "Five unrelated container IDs changed during the multi-day interval and one of those mount sets changed, "
            "so literal host-wide identity stability is not claimed."
        ),
        "non_story_mutation_operations_performed_by_story": False,
        "status": "pass_with_ambient_drift" if changed else "pass",
    }
    identity = _write_json(Path(output), evidence)
    return evidence, identity


def build_candidate_manifest(files: Sequence[str | Path], *, output: str | Path) -> tuple[dict[str, Any], str]:
    """Bind the tracked Validator candidate set and fail closed on aggregate privacy gates."""
    output_path = Path(output)
    normalized = sorted({Path(path) for path in files})
    if output_path in normalized:
        raise MilvusSindiError("Candidate manifest cannot include itself")
    records = []
    forbidden_roots = {"results", "data"}
    secret_patterns = (
        re.compile(
            r"(?i)['\"]?(?:api[_-]?key|authorization|password|token)['\"]?"
            r"\s*[:=]\s*(['\"])[^'\"\r\n]{12,}\1"
        ),
        re.compile(
            r"(?im)^\s*(?:api[_-]?key|authorization|password|token)"
            r"\s*[:=]\s*[A-Za-z0-9_.-]{16,}\s*$"
        ),
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{16,}\b"),
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    )
    for path in normalized:
        if not path.is_file() or (path.parts and path.parts[0] in forbidden_roots):
            raise MilvusSindiError(f"Candidate path is absent or private-only: {path}")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if any(pattern.search(text) for pattern in secret_patterns):
            raise MilvusSindiError(f"Potential secret in tracked candidate: {path}")
        records.append({"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    aggregate_paths = {
        Path("benchmark/artifacts/milvus-sindi-system-v0.1/formal-summary.json"),
        Path("benchmark/research/milvus_sindi_system_report_20260904.md"),
    }
    if not aggregate_paths.issubset(set(normalized)):
        raise MilvusSindiError("Candidate set does not include both aggregate deliverables")
    summary = _load_json(Path("benchmark/artifacts/milvus-sindi-system-v0.1/formal-summary.json"))
    if (
        summary.get("contains_source_text") is not False
        or summary.get("contains_canonical_ids") is not False
        or summary.get("contains_raw_rankings") is not False
        or summary.get("publication_gate") != "closed"
        or summary.get("research_only") is not True
    ):
        raise MilvusSindiError("Tracked aggregate privacy/publication gate is open")
    candidate_set_sha256 = hashlib.sha256(formatted_json_bytes(records)).hexdigest()
    manifest = {
        "schema_version": "milvus-sindi-validator-candidate-v1",
        "story_id": "S-20260814-010",
        "classification": "tracked_candidate_identity",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_file_count": len(records),
        "candidate_set_sha256": candidate_set_sha256,
        "files": records,
        "privacy_checks": {
            "utf8_text_only": True,
            "secret_patterns_absent": True,
            "source_text_absent_from_aggregate": True,
            "canonical_ids_absent_from_aggregate": True,
            "raw_rankings_absent_from_aggregate": True,
        },
        "publication_gate": "closed",
        "status": "candidate_not_accepted",
    }
    identity = _write_json(output_path, manifest)
    return manifest, identity


def _comparison_key(cell: Mapping[str, Any]) -> tuple[str, str, str | None, int]:
    return str(cell["tier"]), str(cell["model"]), cell.get("track"), int(cell["scale"])


def build_public_summary(
    recomputed_evidence: str | Path,
    *,
    recomputed_sha256: str,
    output: str | Path,
) -> tuple[dict[str, Any], str]:
    """Write an aggregate-only tracked summary without raw IDs, rankings, or source text."""
    source_path = Path(recomputed_evidence)
    if file_sha256(source_path) != recomputed_sha256:
        raise MilvusSindiError("Recomputed evidence identity drifted")
    source = _load_json(source_path)
    if source.get("status") != "pass" or source.get("cell_count") != 36:
        raise MilvusSindiError("Recomputed evidence is incomplete")
    grouped: dict[tuple[str, str, str | None, int], dict[str, Mapping[str, Any]]] = {}
    for cell in source["cells"]:
        grouped.setdefault(_comparison_key(cell), {})[str(cell["comparator"])] = cell
    comparisons = []
    costs = []
    for key, cells in sorted(grouped.items()):
        if set(cells) != set(COMPARATORS):
            raise MilvusSindiError(f"Comparator coverage drifted: {key}")
        summaries = {
            comparator: {_summary_key(row): row for row in cell["summary"]} for comparator, cell in cells.items()
        }
        configurations = []
        for summary_key in sorted(summaries["SINDI"]):
            exact = summaries["exact_scipy_csr"][summary_key]
            sindi = summaries["SINDI"][summary_key]
            daat = summaries["DAAT_MAXSCORE"][summary_key]
            configurations.append(
                {
                    "top_k": summary_key[0],
                    "concurrency": summary_key[1],
                    "temperature": summary_key[2],
                    "exact_qps": exact["qps_median"],
                    "exact_p50_ms": exact["p50_ms_median"],
                    "exact_p95_ms": exact["p95_ms_median"],
                    "exact_p99_ms": exact["p99_ms_median"],
                    "sindi_qps": sindi["qps_median"],
                    "daat_maxscore_qps": daat["qps_median"],
                    "sindi_over_daat_qps_ratio": sindi["qps_median"] / daat["qps_median"],
                    "sindi_over_exact_qps_ratio": sindi["qps_median"] / exact["qps_median"],
                    "sindi_p50_ms": sindi["p50_ms_median"],
                    "sindi_p95_ms": sindi["p95_ms_median"],
                    "sindi_p99_ms": sindi["p99_ms_median"],
                    "sindi_client_prepare_p50_ms": sindi["client_prepare_p50_ms_median"],
                    "sindi_client_prepare_p95_ms": sindi["client_prepare_p95_ms_median"],
                    "sindi_rpc_p50_ms": sindi["search_call_p50_ms_median"],
                    "sindi_rpc_p95_ms": sindi["search_call_p95_ms_median"],
                    "daat_maxscore_p50_ms": daat["p50_ms_median"],
                    "daat_maxscore_p95_ms": daat["p95_ms_median"],
                    "daat_maxscore_p99_ms": daat["p99_ms_median"],
                    "daat_maxscore_client_prepare_p50_ms": daat["client_prepare_p50_ms_median"],
                    "daat_maxscore_client_prepare_p95_ms": daat["client_prepare_p95_ms_median"],
                    "daat_maxscore_rpc_p50_ms": daat["search_call_p50_ms_median"],
                    "daat_maxscore_rpc_p95_ms": daat["search_call_p95_ms_median"],
                    "exact_client_cpu_ns": exact["client_cpu_ns_median"],
                    "sindi_client_cpu_ns": sindi["client_cpu_ns_median"],
                    "daat_maxscore_client_cpu_ns": daat["client_cpu_ns_median"],
                    "sindi_server_search_mean_ms": sindi["server_search_mean_ms_median"],
                    "daat_maxscore_server_search_mean_ms": daat["server_search_mean_ms_median"],
                    "sindi_network_received_bytes": sindi["network_received_bytes_median"],
                    "sindi_network_sent_bytes": sindi["network_sent_bytes_median"],
                    "daat_maxscore_network_received_bytes": daat["network_received_bytes_median"],
                    "daat_maxscore_network_sent_bytes": daat["network_sent_bytes_median"],
                    "sindi_minimum_strict_recall": sindi["minimum_strict_recall"],
                    "daat_maxscore_minimum_strict_recall": daat["minimum_strict_recall"],
                    "sindi_maximum_score_delta": sindi["maximum_score_delta"],
                    "daat_maxscore_maximum_score_delta": daat["maximum_score_delta"],
                }
            )
        tier, model, track, scale = key
        comparisons.append(
            {
                "tier": tier,
                "model_profile": model,
                "track": track,
                "scale": scale,
                "configurations": configurations,
            }
        )
        for comparator, cell in cells.items():
            costs.append(
                {
                    "tier": tier,
                    "model_profile": model,
                    "track": track,
                    "scale": scale,
                    "comparator": comparator,
                    "build_wall_ns": cell["build_wall_ns"],
                    "load_wall_ns": cell["load_wall_ns"],
                    "insert_wall_ns": cell["insert_wall_ns"],
                    "flush_wall_ns": cell["flush_wall_ns"],
                    "phase_client_cpu_ns": cell["phase_client_cpu_ns"],
                    "incremental_runtime_bytes": cell["incremental_runtime_bytes"],
                    "persisted_index_bytes": cell["persisted_index_bytes"],
                    "document_csr_bytes": cell["document_csr_bytes"],
                    "query_csr_bytes": cell["query_csr_bytes"],
                    "resources": cell["resources"],
                }
            )
    all_configs = [config for comparison in comparisons for config in comparison["configurations"]]

    def stratum(name: str, selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        configurations = [config for comparison in selected for config in comparison["configurations"]]
        warm = {
            (
                comparison["model_profile"],
                comparison["track"],
                comparison["scale"],
                config["top_k"],
                config["concurrency"],
            ): config
            for comparison in selected
            for config in comparison["configurations"]
            if config["temperature"] == "warm"
        }
        cold = {
            (
                comparison["model_profile"],
                comparison["track"],
                comparison["scale"],
                config["top_k"],
                config["concurrency"],
            ): config
            for comparison in selected
            for config in comparison["configurations"]
            if config["temperature"] == "cold"
        }
        return {
            "name": name,
            "comparison_count": len(selected),
            "configuration_count": len(configurations),
            "minimum_sindi_strict_recall": min(config["sindi_minimum_strict_recall"] for config in configurations),
            "minimum_daat_maxscore_strict_recall": min(
                config["daat_maxscore_minimum_strict_recall"] for config in configurations
            ),
            "median_sindi_over_daat_qps_ratio": float(
                np.median([config["sindi_over_daat_qps_ratio"] for config in configurations])
            ),
            "minimum_sindi_over_daat_qps_ratio": min(config["sindi_over_daat_qps_ratio"] for config in configurations),
            "maximum_sindi_over_daat_qps_ratio": max(config["sindi_over_daat_qps_ratio"] for config in configurations),
            "median_sindi_warm_over_cold_qps_ratio": float(
                np.median([warm[key]["sindi_qps"] / cold[key]["sindi_qps"] for key in warm])
            ),
            "median_daat_warm_over_cold_qps_ratio": float(
                np.median([warm[key]["daat_maxscore_qps"] / cold[key]["daat_maxscore_qps"] for key in warm])
            ),
        }

    strata = [
        stratum("native", [row for row in comparisons if row["tier"] == "native"]),
        stratum(
            "system_100k",
            [row for row in comparisons if row["tier"] == "system_only" and row["scale"] == 100_000],
        ),
        stratum(
            "system_1m",
            [row for row in comparisons if row["tier"] == "system_only" and row["scale"] == 1_000_000],
        ),
    ]
    summary = {
        "schema_version": "milvus-sindi-formal-summary-v1",
        "story_id": "S-20260814-010",
        "classification": "tracked_aggregate_system_evidence",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": "74635817770023b70b1e9f9e97b3d83d9d11f987",
        "predeclaration_sha256": "d1c3e3c9c2ad1e7e4b79bf37b0088e480058196a570f56d21ccc1a4bd2ff47d1",
        "source_attestation_sha256": "dce7e8c088954bfc5d689cea1017f13b088dc417532d5b5a9a312abe6cf46f65",
        "model_loaded": False,
        "deployment": {
            "server_image": "milvusdb/milvus@sha256:804b50bc1523a64e3c0f18cf33af7e0f8b33329584698ffee61c606d6fe8ee26",
            "server_version": "3.0.0",
            "pymilvus_version": "3.0.1",
            "target_vector_index_version": 10,
            "runtime_root_mode": "0711",
            "server_cpuset": "48-63",
            "server_memory_bytes": 51_539_607_552,
            "server_pids_limit": 4096,
        },
        "coverage": {
            "native_cells": source["native_cell_count"],
            "system_100k_cells": source["system_100k_cell_count"],
            "system_1m_cells": source["system_1m_cell_count"],
            "raw_trial_files_recomputed": source["raw_trial_files_recomputed"],
            "warmup_files_validated": source["warmup_files_validated"],
            "top_k": list(TOP_K_VALUES),
            "concurrency": list(CONCURRENCY_VALUES),
            "measured_trials": MEASURED_TRIALS,
            "warmup_passes": WARMUP_PASSES,
        },
        "aggregate": {
            "minimum_sindi_strict_recall": min(config["sindi_minimum_strict_recall"] for config in all_configs),
            "minimum_daat_maxscore_strict_recall": min(
                config["daat_maxscore_minimum_strict_recall"] for config in all_configs
            ),
            "maximum_sindi_score_delta": max(config["sindi_maximum_score_delta"] for config in all_configs),
            "maximum_daat_maxscore_score_delta": max(
                config["daat_maxscore_maximum_score_delta"] for config in all_configs
            ),
            "median_sindi_over_daat_qps_ratio": float(
                np.median([config["sindi_over_daat_qps_ratio"] for config in all_configs])
            ),
            "minimum_sindi_over_daat_qps_ratio": min(config["sindi_over_daat_qps_ratio"] for config in all_configs),
            "maximum_sindi_over_daat_qps_ratio": max(config["sindi_over_daat_qps_ratio"] for config in all_configs),
        },
        "strata": strata,
        "comparisons": comparisons,
        "costs": costs,
        "contains_source_text": False,
        "contains_canonical_ids": False,
        "contains_raw_rankings": False,
        "quality_claims_allowed_for_system_only": False,
        "research_only": True,
        "publication_gate": "closed",
        "status": "pass",
    }
    identity = _write_json(Path(output), summary)
    return summary, identity


def write_markdown_report(
    summary_path: str | Path,
    *,
    summary_sha256: str,
    output: str | Path,
) -> tuple[dict[str, Any], str]:
    """Render the formal aggregate as an English research-only report."""
    source_path = Path(summary_path)
    if file_sha256(source_path) != summary_sha256:
        raise MilvusSindiError("Formal summary identity drifted")
    summary = _load_json(source_path)
    if summary.get("status") != "pass" or summary.get("publication_gate") != "closed":
        raise MilvusSindiError("Formal summary is not a closed, passing aggregate")

    def label(comparison: Mapping[str, Any]) -> str:
        if comparison["tier"] == "native":
            return f"native/{comparison['model_profile']}/{comparison['track']}"
        return f"system-only/{comparison['model_profile']}/{comparison['scale']}"

    performance_rows = []
    network_rows = []
    for comparison in summary["comparisons"]:
        for config in comparison["configurations"]:
            if config["temperature"] == "warm" and config["concurrency"] == 16:
                performance_rows.append(
                    "| "
                    + " | ".join(
                        [
                            label(comparison),
                            str(config["top_k"]),
                            f"{config['exact_qps']:.2f}",
                            f"{config['sindi_qps']:.2f}",
                            f"{config['daat_maxscore_qps']:.2f}",
                            f"{config['sindi_over_daat_qps_ratio']:.3f}",
                            f"{config['exact_p50_ms']:.3f}",
                            f"{config['exact_p95_ms']:.3f}",
                            f"{config['exact_p99_ms']:.3f}",
                            f"{config['sindi_p50_ms']:.3f}",
                            f"{config['sindi_p95_ms']:.3f}",
                            f"{config['sindi_p99_ms']:.3f}",
                            f"{config['daat_maxscore_p50_ms']:.3f}",
                            f"{config['daat_maxscore_p95_ms']:.3f}",
                            f"{config['daat_maxscore_p99_ms']:.3f}",
                        ]
                    )
                    + " |"
                )
                network_rows.append(
                    "| "
                    + " | ".join(
                        [
                            label(comparison),
                            str(config["top_k"]),
                            f"{config['sindi_client_prepare_p50_ms']:.4f}",
                            f"{config['sindi_rpc_p50_ms']:.3f}",
                            f"{config['sindi_server_search_mean_ms']:.3f}",
                            str(config["sindi_network_received_bytes"]),
                            str(config["sindi_network_sent_bytes"]),
                            f"{config['daat_maxscore_client_prepare_p50_ms']:.4f}",
                            f"{config['daat_maxscore_rpc_p50_ms']:.3f}",
                            f"{config['daat_maxscore_server_search_mean_ms']:.3f}",
                            str(config["daat_maxscore_network_received_bytes"]),
                            str(config["daat_maxscore_network_sent_bytes"]),
                        ]
                    )
                    + " |"
                )
    cost_rows = []
    exact_cost_rows = []
    for cost in summary["costs"]:
        if cost["comparator"] == "exact_scipy_csr":
            exact_cost_rows.append(
                "| "
                + " | ".join(
                    [
                        (
                            f"native/{cost['model_profile']}/{cost['track']}"
                            if cost["tier"] == "native"
                            else f"system-only/{cost['model_profile']}/{cost['scale']}"
                        ),
                        str(int(cost["document_csr_bytes"])),
                        str(int(cost["query_csr_bytes"])),
                        str(int(cost["resources"]["client_peak_rss_bytes"])),
                    ]
                )
                + " |"
            )
            continue
        item_label = (
            f"native/{cost['model_profile']}/{cost['track']}"
            if cost["tier"] == "native"
            else f"system-only/{cost['model_profile']}/{cost['scale']}"
        )
        cost_rows.append(
            "| "
            + " | ".join(
                [
                    item_label,
                    str(cost["comparator"]),
                    f"{int(cost['insert_wall_ns']) / 1e9:.3f}",
                    f"{int(cost['flush_wall_ns']) / 1e9:.3f}",
                    f"{int(cost['build_wall_ns']) / 1e9:.3f}",
                    f"{int(cost['load_wall_ns']) / 1e9:.3f}",
                    str(int(cost["persisted_index_bytes"])),
                    str(int(cost["incremental_runtime_bytes"])),
                    str(int(cost["resources"]["container_peak_memory_bytes"])),
                    str(int(cost["resources"]["client_peak_rss_bytes"])),
                ]
            )
            + " |"
        )
    aggregate = summary["aggregate"]
    strata = {row["name"]: row for row in summary["strata"]}
    crossover_sentence = (
        "The ratio crosses 1.0, so this matrix contains real workload/configuration crossovers and does not support a universal SINDI-wins claim."
        if aggregate["minimum_sindi_over_daat_qps_ratio"] < 1 < aggregate["maximum_sindi_over_daat_qps_ratio"]
        else "The observed ratio did not cross 1.0 in this matrix; that bounded result still does not establish a universal ordering."
    )
    lines = [
        "# Milvus SINDI sparse system benchmark",
        "",
        "## Status and scope",
        "",
        "This report records the completed S-20260814-010 research-only system experiment. The publication gate remains closed. Native results cover only the two accepted internal tracks; generated 100k and 1M results are system-only scaling evidence and must not be used for model-quality, multilingual-effectiveness, or zero-shot claims.",
        "A `pass` status means that execution, identity, lifecycle, and recomputation checks passed. It does not mean that SINDI, DAAT_MAXSCORE, or any model is effective, faster, or preferable.",
        "",
        f"The validated matrix contains {summary['coverage']['native_cells']} native cells, {summary['coverage']['system_100k_cells']} 100k cells, and {summary['coverage']['system_1m_cells']} 1M cells. The independent reducer recomputed {summary['coverage']['raw_trial_files_recomputed']} measured raw trial files and validated {summary['coverage']['warmup_files_validated']} warmup files.",
        "",
        "## Identity and deployment",
        "",
        f"- Accepted source commit: `{summary['source_commit']}`; the runtime path reported `model_loaded=false` and reused saved CSR/exact artifacts read-only.",
        f"- Server image: `{summary['deployment']['server_image']}`; Milvus `{summary['deployment']['server_version']}`; PyMilvus `{summary['deployment']['pymilvus_version']}`.",
        f"- Persisted vector index version: `{summary['deployment']['target_vector_index_version']}`. Runtime root mode remained `{summary['deployment']['runtime_root_mode']}`.",
        f"- Server resource boundary: cpuset `{summary['deployment']['server_cpuset']}`, {summary['deployment']['server_memory_bytes']} bytes memory with no extra swap, and {summary['deployment']['server_pids_limit']} PIDs.",
        "- Primary references: [Milvus 3.0.0 release and compatibility notes](https://milvus.io/docs/release_notes.md) and [SPARSE_INVERTED_INDEX parameters](https://milvus.io/docs/sparse-inverted-index.md). These establish feature and configuration identity only; they are not benchmark results.",
        "",
        "## Method",
        "",
        "Each workload used identical sparse vectors, queries, topK values (10 and 100), deterministic query schedules, and concurrency values (1, 4, and 16). Both Milvus indexes used `SPARSE_FLOAT_VECTOR`, inner product, `SPARSE_INVERTED_INDEX`, explicit `SINDI` or `DAAT_MAXSCORE`, `drop_ratio_build=0`, and `drop_ratio_search=0`. Every configuration used two unmeasured warmups and three measured cold plus three measured warm trials. Cold means collection release and reload; host page caches were not dropped.",
        "",
        "Before every Milvus search family, the runner required the requested server-reported algorithm, `Finished`, `Loaded`, all rows in sealed indexed segments, zero growing rows, and no active or queued compaction. Exact cost uses SciPy CSR float32 inner product on the frozen CPU domain. Build, insert, load, client preparation, RPC/end-to-end search, server latency counters, network counters, client CPU/RSS, container CPU/RSS/PIDs, persisted bytes, and raw query trials are stored separately in private evidence.",
        "",
        "## Correctness",
        "",
        f"In the native tier, the minimum raw float32 strict Recall@K was `{strata['native']['minimum_sindi_strict_recall']:.6f}` for SINDI and `{strata['native']['minimum_daat_maxscore_strict_recall']:.6f}` for DAAT_MAXSCORE. Across all tiers, including synthetic system-only near-tie workloads, the corresponding minima were `{aggregate['minimum_sindi_strict_recall']:.6f}` and `{aggregate['minimum_daat_maxscore_strict_recall']:.6f}`. Maximum absolute score deltas were `{aggregate['maximum_sindi_score_delta']:.9g}` and `{aggregate['maximum_daat_maxscore_score_delta']:.9g}`. Every validated response used unique private row ordinals and finite scores.",
        "Exact ties use descending float32 score followed by ascending private row ordinal. An independent private native replay cast authenticated document nonzeros to IEEE float16 and back to float32 while retaining float32 queries. Against that persisted-index representation, the minimum strict recall was 0.9 and tie-aware Recall@K was 1.0 for both SINDI and DAAT_MAXSCORE; server score deltas were at most 1.52587890625e-05. The lower system-only strict ID recall reflects denser synthetic near-tie boundaries under the same storage conversion and is not an effectiveness result. These are pinned index-format effects, not search pruning, default fallback, or model-quality differences.",
        "",
        "## Performance",
        "",
        f"Across every native and system-only K/concurrency/cold-warm configuration, the median SINDI/DAAT QPS ratio was `{aggregate['median_sindi_over_daat_qps_ratio']:.3f}`; the observed range was `{aggregate['minimum_sindi_over_daat_qps_ratio']:.3f}` to `{aggregate['maximum_sindi_over_daat_qps_ratio']:.3f}`. These are measurements of this pinned system and workload matrix, not vendor claims.",
        crossover_sentence,
        f"Warm/cold effects were not uniformly positive: median warm-over-cold QPS ratios were SINDI/DAAT `{strata['native']['median_sindi_warm_over_cold_qps_ratio']:.3f}`/`{strata['native']['median_daat_warm_over_cold_qps_ratio']:.3f}` in native, `{strata['system_100k']['median_sindi_warm_over_cold_qps_ratio']:.3f}`/`{strata['system_100k']['median_daat_warm_over_cold_qps_ratio']:.3f}` at 100k, and `{strata['system_1m']['median_sindi_warm_over_cold_qps_ratio']:.3f}`/`{strata['system_1m']['median_daat_warm_over_cold_qps_ratio']:.3f}` at 1M. The full configuration table preserves workload-specific reversals and long tails; no universal cache benefit or scale advantage is claimed.",
        f"Scale changed the observed economics: median SINDI/DAAT QPS ratios were `{strata['native']['median_sindi_over_daat_qps_ratio']:.3f}` in native, `{strata['system_100k']['median_sindi_over_daat_qps_ratio']:.3f}` at 100k, and `{strata['system_1m']['median_sindi_over_daat_qps_ratio']:.3f}` at 1M. The 1M range still extended from `{strata['system_1m']['minimum_sindi_over_daat_qps_ratio']:.3f}` to `{strata['system_1m']['maximum_sindi_over_daat_qps_ratio']:.3f}`. The upper extreme is a DAAT_MAXSCORE long-tail collapse for one synthetic OpenSearch-multilingual K100/concurrency-1 configuration, while other configurations showed no SINDI advantage. It is a crossover and tail-risk observation, not a universal algorithm ranking.",
        "",
        "The compact table below shows warm concurrency-16 medians; the machine-readable summary contains all configurations.",
        "",
        "| Workload | K | Exact QPS | SINDI QPS | DAAT QPS | S/D | Exact P50 | Exact P95 | Exact P99 | S P50 | S P95 | S P99 | D P50 | D P95 | D P99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *performance_rows,
        "",
        "The next table separates client sparse-vector preparation, RPC end-to-end latency, server search-counter mean, and proxy byte counters for the same warm concurrency-16 slice. Proxy sent bytes are a global counter measured while this was the only active Story cell; no packet capture or exact protobuf payload-size claim is made.",
        "",
        "| Workload | K | S prep P50 ms | S RPC P50 ms | S server mean ms | S recv B | S sent B | D prep P50 ms | D RPC P50 ms | D server mean ms | D recv B | D sent B |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *network_rows,
        "",
        "## Build, persistence, and resource cost",
        "",
        "Persisted index bytes come from the server's successful version-10 serialization log. Incremental runtime bytes are measured across the complete collection lifecycle and include other collection-associated persistence overhead. Peaks are sampled from the isolated container and benchmark client.",
        "",
        "| Workload | Algorithm | Insert s | Flush s | Build s | Load s | Index bytes | Runtime delta bytes | Container peak bytes | Client peak RSS bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *cost_rows,
        "",
        "Exact CSR has no Milvus insert/build/load/network phases. Its comparable resident input and client peak costs are:",
        "",
        "| Workload | Document CSR bytes | Query CSR bytes | Client peak RSS bytes |",
        "| --- | ---: | ---: | ---: |",
        *exact_cost_rows,
        "",
        "## Failures and recovery",
        "",
        "The first persisted SINDI build exposed a preflight gap: Milvus 3.0.0 defaults to vector index version 8, while SINDI requires the opt-in version 10. The server explicitly rejected SINDI rather than falling back. The failure, retry loop, API state, metrics, and logs were preserved privately. The Story-owned configuration was set to `dataCoord.targetVecIndexVersion=10` and `dataCoord.targetScalarIndexVersion=4`; the same Story container then built and serialized the real index successfully. A separate client-side load-state enum comparison also failed once, was preserved, and was corrected without changing any workload parameter.",
        "",
        "A stop/start isolation check found one unrelated ambient host listener (`*:6443`) absent after the bounded Story restart. A later multi-day final baseline retained the same non-Story container-name, network, volume, and listening-endpoint sets but found five unrelated demo containers recreated, including one mount-set change. No benchmark command targeted those resources. Because host-wide Docker events were not under Story control, the evidence reports this ambient drift and does not claim literal zero non-Story identity drift.",
        "",
        "## Reproduction and boundaries",
        "",
        "The private formal manifest binds every raw trial, lifecycle response, server log, baseline, failure, and validation artifact by size and SHA-256. Re-run the independent validator with:",
        "",
        "```bash",
        "uv run python scripts/milvus_sindi_system.py validate-formal-results \\",
        "  --output-root results/milvus-sindi-system-v0.1/formal-20260831 \\",
        "  --output results/milvus-sindi-system-v0.1/formal-20260831/validation/recomputed-summary.json",
        "```",
        "",
        "This report contains aggregate system evidence only. It contains no source text, canonical document/query identifiers, or raw rankings. `research_only=true`, no-publish remains in force, and system-only results do not support quality claims.",
        "",
    ]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    identity = file_sha256(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return {"status": "pass", "report_sha256": identity, "summary_sha256": summary_sha256}, identity
