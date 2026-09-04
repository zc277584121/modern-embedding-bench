"""Deterministic, privacy-closed packaging for the formal multi-vector matrix."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from mm_embed.benchmark.bright_multi_vector import (
    ACTIVE_PREDECLARATION_SHA256,
    PUBLICATION,
    TRACKS,
    load_json,
    validate_canonical_data,
    validate_gate_summary,
    validate_model_encoding,
    validate_predeclaration,
)
from mm_embed.benchmark.bright_multidomain_v02 import load_materialized
from mm_embed.benchmark.retrieval_v01 import (
    METRIC_NAMES,
    aggregate_metrics,
    paired_bootstrap_deltas,
    tokens,
)
from mm_embed.providers.real_multi_vector import SELECTED_KEYS, file_sha256

PACKAGE_VERSION = "bright-multi-vector-v0.2"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_826
OUTPUT_NAMES = ("summary.json", "pairwise.json", "slices.json", "cases.json", "resources.json")
IMPLEMENTATION_CANDIDATES = (
    "pyproject.toml",
    "uv.lock",
    "benchmark/artifacts/bright-multi-vector-v0.1/inventory.json",
    "benchmark/artifacts/bright-multi-vector-v0.1/inventory.json.sha256",
    "benchmark/artifacts/bright-multi-vector-v0.1/predeclaration.json",
    "benchmark/artifacts/bright-multi-vector-v0.1/predeclaration.json.sha256",
    "benchmark/artifacts/bright-multi-vector-v0.2/chronology.json",
    "benchmark/artifacts/bright-multi-vector-v0.2/chronology.json.sha256",
    "benchmark/artifacts/bright-multi-vector-v0.2/predeclaration.json",
    "benchmark/artifacts/bright-multi-vector-v0.2/predeclaration.json.sha256",
    "benchmark/artifacts/bright-multi-vector-v0.2/gate-summary.json",
    "benchmark/artifacts/bright-multi-vector-v0.2/gate-summary.json.sha256",
    "schemas/bright-real-multi-vector-v01.schema.json",
    "scripts/bright_multi_vector.py",
    "scripts/build_bright_multi_vector_candidate.py",
    "scripts/check_bright_multi_vector_tamper.py",
    "scripts/freeze_bright_multi_vector.py",
    "src/mm_embed/benchmark/bright_multi_vector.py",
    "src/mm_embed/benchmark/bright_multi_vector_package.py",
    "src/mm_embed/providers/real_multi_vector.py",
    "tests/test_bright_multi_vector.py",
)
PACKAGE_CANDIDATE_ROOT = "benchmark/artifacts/bright-multi-vector-results-v0.2"
REPORT_CANDIDATE_PATH = "benchmark/research/bright_multi_vector_v02_20260904.md"


def candidate_paths() -> tuple[str, ...]:
    """Return the exact ordered tracked candidate set, excluding its identity file."""
    package_files = tuple(
        f"{PACKAGE_CANDIDATE_ROOT}/{name}"
        for name in (
            "cases.json",
            "cases.json.sha256",
            "manifest.json",
            "manifest.json.sha256",
            "pairwise.json",
            "pairwise.json.sha256",
            "resources.json",
            "resources.json.sha256",
            "slices.json",
            "slices.json.sha256",
            "summary.json",
            "summary.json.sha256",
        )
    )
    return tuple(
        sorted((*IMPLEMENTATION_CANDIDATES, *package_files, REPORT_CANDIDATE_PATH, f"{REPORT_CANDIDATE_PATH}.sha256"))
    )


FORBIDDEN_KEYS = {
    "query_id",
    "document_id",
    "positive_document_ids",
    "hits",
    "rankings",
    "per_query",
    "source_text",
    "content",
    "text",
    "representation_root",
}
SCHEMA_PATH = Path(__file__).parents[3] / "schemas" / "bright-real-multi-vector-v01.schema.json"


class MultiVectorPackageError(ValueError):
    """Raised when a formal input or tracked output violates the frozen contract."""


@dataclass(frozen=True)
class Cell:
    """One hash-authenticated formal cell and restricted row-level evidence."""

    model: str
    track: str
    path: Path
    identity: str
    value: dict[str, Any]
    per_query: dict[str, dict[str, float]]
    rankings: dict[str, list[dict[str, Any]]]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def formatted_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def validate_schema(value: Mapping[str, Any]) -> None:
    """Validate a tracked evidence object against the shared Story schema."""
    try:
        Draft202012Validator(load_json(SCHEMA_PATH)).validate(value)
    except Exception as exc:
        raise MultiVectorPackageError(f"Multi-vector schema validation failed: {exc}") from exc


def _sidecar(path: Path) -> str:
    try:
        return path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise MultiVectorPackageError(f"Missing identity sidecar: {path}") from exc


def _load_restricted(entry: Mapping[str, Any], path: Path) -> Any:
    if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
        raise MultiVectorPackageError(f"Restricted formal artifact drifted: {path.name}")
    sidecar_entry = entry.get("sidecar")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not isinstance(sidecar_entry, Mapping)
        or sidecar.name != sidecar_entry["path"]
        or sidecar.stat().st_size != sidecar_entry["bytes"]
        or file_sha256(sidecar) != sidecar_entry["sha256"]
        or sidecar.read_text(encoding="ascii").strip() != entry["sha256"]
    ):
        raise MultiVectorPackageError(f"Restricted formal artifact sidecar drifted: {path.name}")
    return load_json(path)


def validate_formal_cell_artifact(
    path: str | Path,
    *,
    model: str,
    track: str,
    encoding: Mapping[str, Any],
    data_root: str | Path,
) -> Cell:
    """Authenticate a formal cell, its sidecar, and restricted row-level files."""
    artifact = Path(path)
    identity = file_sha256(artifact)
    if _sidecar(artifact) != identity:
        raise MultiVectorPackageError(f"Formal cell identity drifted: {model}:{track}")
    value = load_json(artifact)
    if (
        value.get("finalized") is not True
        or value.get("publication") != PUBLICATION
        or value.get("predeclaration_sha256") != ACTIVE_PREDECLARATION_SHA256
        or value["model"]["key"] != model
        or value["track"]["track"] != track
        or value["execution"]["model_loaded"] is not False
        or value["protocol"]["exact"] is not True
        or value["representation"]["encoding_manifest_sha256"] != encoding["manifest_sha256"]
        or value["representation"]["manifest_sha256"] != encoding["representations"][track]["manifest_sha256"]
        or value["representation"]["storage"] != encoding["representations"][track]["storage"]
    ):
        raise MultiVectorPackageError(f"Formal cell contract drifted: {model}:{track}")
    restricted = value["restricted_recompute"]
    per_rows = _load_restricted(restricted["per_query"], artifact.with_name(restricted["per_query"]["path"]))
    ranking_rows = _load_restricted(restricted["rankings"], artifact.with_name(restricted["rankings"]["path"]))
    per_query = {str(row["query_id"]): dict(row["metrics"]) for row in per_rows}
    rankings = {str(row["query_id"]): list(row["hits"]) for row in ranking_rows}
    data = load_materialized(data_root, track)
    expected_ids = {row["id"] for row in data.queries}
    if set(per_query) != expected_ids or set(rankings) != expected_ids:
        raise MultiVectorPackageError(f"Formal query coverage drifted: {model}:{track}")
    if aggregate_metrics(per_query) != value["quality"]["metrics"]:
        raise MultiVectorPackageError(f"Formal aggregate metrics drifted: {model}:{track}")
    expected_files = {
        artifact.name,
        artifact.with_suffix(artifact.suffix + ".sha256").name,
        restricted["per_query"]["path"],
        restricted["per_query"]["sidecar"]["path"],
        restricted["rankings"]["path"],
        restricted["rankings"]["sidecar"]["path"],
    }
    cell_files = {candidate.name for candidate in artifact.parent.glob(f"{artifact.stem}*") if candidate.is_file()}
    if cell_files != expected_files:
        raise MultiVectorPackageError(f"Formal cell file set drifted: {model}:{track}")
    return Cell(model, track, artifact, identity, value, per_query, rankings)


def collect_cells(
    *,
    cell_root: str | Path,
    predeclaration_path: str | Path,
    gate_summary_path: str | Path,
    data_root: str | Path,
) -> tuple[dict[tuple[str, str], Cell], dict[str, Any], dict[str, Any]]:
    """Authenticate exactly the frozen 3x2 matrix without loading a model."""
    predecl = validate_predeclaration(predeclaration_path, ACTIVE_PREDECLARATION_SHA256)
    repo_root = Path(__file__).parents[3]
    inventory_path = repo_root / predecl["bindings"]["inventory"]["path"]
    validate_schema(load_json(inventory_path))
    validate_schema(predecl)
    validate_schema(load_json(Path(predeclaration_path).with_name("chronology.json")))
    gate_summary = validate_gate_summary(gate_summary_path)
    validate_schema(gate_summary)
    validate_canonical_data(predecl, data_root)
    root = Path(cell_root)
    cells: dict[tuple[str, str], Cell] = {}
    representation_inputs: dict[str, Any] = {}
    for model in SELECTED_KEYS:
        encoding = validate_model_encoding(
            root.parent / "representations" / model,
            model_key=model,
            predeclaration_sha256=ACTIVE_PREDECLARATION_SHA256,
            gate_summary_sha256=file_sha256(Path(gate_summary_path)),
            data_root=data_root,
        )
        representation_inputs[model] = {
            "encoding_manifest_sha256": encoding["manifest_sha256"],
            "encoding_manifest_sidecar_sha256": encoding["manifest_sidecar_sha256"],
            "tracks": {
                track: {
                    "representation_manifest_sha256": encoding["representations"][track]["manifest_sha256"],
                    "representation_manifest_sidecar": encoding["representations"][track]["manifest_sidecar"],
                    "storage": encoding["representations"][track]["storage"],
                    "files": encoding["representations"][track]["files"],
                }
                for track in TRACKS
            },
        }
        for track in TRACKS:
            path = root / f"{model}-{track}.json"
            cells[(model, track)] = validate_formal_cell_artifact(
                path,
                model=model,
                track=track,
                encoding=encoding,
                data_root=data_root,
            )
    if len(cells) != 6:
        raise MultiVectorPackageError("The formal package requires exactly six cells")
    return cells, predecl, representation_inputs


def _paired(left: Mapping[str, Mapping[str, float]], right: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    result = paired_bootstrap_deltas(dict(left), dict(right), samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    return {
        metric: {
            **row,
            "verdict": "left_wins" if row["low"] > 0 else "right_wins" if row["high"] < 0 else "tie_or_uncertain",
        }
        for metric, row in result.items()
    }


def build_quality(cells: Mapping[tuple[str, str], Cell]) -> tuple[dict[str, Any], dict[str, Any]]:
    tracks: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for track in TRACKS:
        rows = [
            {
                "model_key": model,
                "metrics": cells[(model, track)].value["quality"]["metrics"],
                "confidence_intervals": cells[(model, track)].value["quality"]["uncertainty"],
                "formal_cell_sha256": cells[(model, track)].identity,
            }
            for model in SELECTED_KEYS
        ]
        tracks[track] = {
            "primary_order": [
                row["model_key"] for row in sorted(rows, key=lambda row: (-row["metrics"]["ndcg@10"], row["model_key"]))
            ],
            "rows": rows,
            "metric_orders": {
                metric: [
                    row["model_key"]
                    for row in sorted(rows, key=lambda row: (-row["metrics"][metric], row["model_key"]))
                ]
                for metric in METRIC_NAMES
            },
        }
        paired[track] = [
            {
                "left": left,
                "right": right,
                "n": len(cells[(left, track)].per_query),
                "metrics": _paired(cells[(left, track)].per_query, cells[(right, track)].per_query),
            }
            for left, right in itertools.combinations(SELECTED_KEYS, 2)
        ]
    summary = {
        "schema_version": "bright-real-multi-vector-summary-v1",
        "package_version": PACKAGE_VERSION,
        "story_id": "S-20260814-012",
        "publication": PUBLICATION,
        "track_aggregation": "independent_no_cross_track_micro_average_or_overall_rank",
        "tracks": tracks,
    }
    pairwise = {
        "schema_version": "bright-real-multi-vector-pairwise-v1",
        "package_version": PACKAGE_VERSION,
        "publication": PUBLICATION,
        "method": {
            "name": "query_aligned_paired_percentile_bootstrap_left_minus_right",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "decisive_rule": "95_percentile_interval_excludes_zero",
        },
        "tracks": paired,
    }
    return summary, pairwise


def _memberships(data: Any, frozen: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    documents = {row["id"]: row["content"] for row in data.corpus}
    output = {}
    for query in data.queries:
        query_tokens = tokens(query["text"].strip())
        unique_query = set(query_tokens)
        positives = sorted(data.qrels[query["id"]])
        positive_tokens = [tokens(documents[document_id]) for document_id in positives]
        overlap = max(
            (len(unique_query & set(row)) / len(unique_query) if unique_query else 0.0) for row in positive_tokens
        )
        median_length = float(np.median([len(row) for row in positive_tokens]))
        density = len(set(positives))
        output[query["id"]] = {
            "query_length": "<=64" if len(query_tokens) <= 64 else "65-128" if len(query_tokens) <= 128 else ">128",
            "positive_lexical_overlap": "0"
            if overlap == 0
            else "(0,0.25]"
            if overlap <= 0.25
            else "(0.25,0.50]"
            if overlap <= 0.50
            else ">0.50",
            "positive_document_length": "<=128"
            if median_length <= 128
            else "129-512"
            if median_length <= 512
            else ">512",
            "positive_qrel_density": "1" if density == 1 else "2-4" if density <= 4 else ">=5",
        }
    allowed = {row["key"]: set(row["bins"]) for row in frozen}
    if any(value not in allowed[key] for row in output.values() for key, value in row.items()):
        raise MultiVectorPackageError("Derived slice membership escaped the frozen bins")
    return output


def build_slices(
    cells: Mapping[tuple[str, str], Cell], data_root: str | Path, predecl: Mapping[str, Any]
) -> dict[str, Any]:
    tracks = {}
    for track in TRACKS:
        membership = _memberships(load_materialized(data_root, track), predecl["slices"])
        dimensions = []
        for frozen in predecl["slices"]:
            bins = []
            for label in frozen["bins"]:
                ids = sorted(query_id for query_id, values in membership.items() if values[frozen["key"]] == label)
                models = []
                for model in SELECTED_KEYS:
                    values = {query_id: cells[(model, track)].per_query[query_id] for query_id in ids}
                    models.append(
                        {"model_key": model, "n": len(ids), "metrics": aggregate_metrics(values) if values else None}
                    )
                pairs = []
                if len(ids) >= 10:
                    for left, right in itertools.combinations(SELECTED_KEYS, 2):
                        left_values = {query_id: cells[(left, track)].per_query[query_id] for query_id in ids}
                        right_values = {query_id: cells[(right, track)].per_query[query_id] for query_id in ids}
                        pairs.append(
                            {"left": left, "right": right, "n": len(ids), "metrics": _paired(left_values, right_values)}
                        )
                bins.append(
                    {
                        "bin": label,
                        "n": len(ids),
                        "membership_sha256": canonical_sha256(ids),
                        "models": models,
                        "pairwise": pairs,
                        "inference": "paired_ci_allowed" if len(ids) >= 10 else "point_estimates_only_no_win_claim",
                    }
                )
            dimensions.append({"key": frozen["key"], "definition": frozen["definition"], "bins": bins})
        tracks[track] = dimensions
    return {
        "schema_version": "bright-real-multi-vector-slices-v1",
        "package_version": PACKAGE_VERSION,
        "publication": PUBLICATION,
        "active_predeclaration_sha256": ACTIVE_PREDECLARATION_SHA256,
        "small_slice_policy": predecl["uncertainty"]["small_slice_policy"],
        "tracks": tracks,
    }


def _best_positive_rank(hits: Sequence[Mapping[str, Any]], positives: set[str]) -> int:
    return min((rank for rank, hit in enumerate(hits, 1) if hit["document_id"] in positives), default=101)


def build_cases(cells: Mapping[tuple[str, str], Cell], data_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    tracked, restricted = [], []
    for track in TRACKS:
        data = load_materialized(data_root, track)
        candidates = []
        for query_id in sorted(data.qrels):
            positives = set(data.qrels[query_id])
            evidence, ranks, recall_hits, ndcgs = {}, [], [], []
            for model in SELECTED_KEYS:
                cell = cells[(model, track)]
                rank = _best_positive_rank(cell.rankings[query_id], positives)
                metric = cell.per_query[query_id]
                ranks.append(rank)
                recall_hits.append(metric["recall@10"] > 0)
                ndcgs.append(metric["ndcg@10"])
                evidence[model] = {"best_positive_rank": rank, "metrics": metric, "formal_cell_sha256": cell.identity}
            candidates.append(
                {
                    "query_id": query_id,
                    "positive_ids": sorted(positives),
                    "models": evidence,
                    "rank_spread": max(ranks) - min(ranks),
                    "recall10_pair_disagreements": sum(a != b for a, b in itertools.combinations(recall_hits, 2)),
                    "best_ndcg10": max(ndcgs),
                    "mean_ndcg10": float(np.mean(ndcgs)),
                    "zero_ndcg10_models": sum(value == 0 for value in ndcgs),
                }
            )
        disagreements = sorted(
            candidates,
            key=lambda row: (
                -row["rank_spread"],
                -row["recall10_pair_disagreements"],
                canonical_sha256(row["query_id"]),
            ),
        )[:6]
        used = {row["query_id"] for row in disagreements}
        failures = sorted(
            (row for row in candidates if row["query_id"] not in used),
            key=lambda row: (
                row["best_ndcg10"],
                row["mean_ndcg10"],
                -row["zero_ndcg10_models"],
                canonical_sha256(row["query_id"]),
            ),
        )[:6]
        for category, rows in (("disagreement", disagreements), ("failure", failures)):
            for ordinal, row in enumerate(rows, 1):
                opaque = f"{track}-{category}-{ordinal:02d}"
                local = {
                    "track": track,
                    "category": category,
                    "query_id": row["query_id"],
                    "positive_document_ids": row["positive_ids"],
                    "selection": {
                        key: row[key]
                        for key in (
                            "rank_spread",
                            "recall10_pair_disagreements",
                            "best_ndcg10",
                            "mean_ndcg10",
                            "zero_ndcg10_models",
                        )
                    },
                    "models": row["models"],
                    "source_text_included": False,
                }
                restricted.append({"opaque_case_id": opaque, **local})
                tracked.append(
                    {
                        "opaque_case_id": opaque,
                        "track": track,
                        "category": category,
                        "selection_rule": "frozen_score_only_with_query_id_hash_tie_break",
                        "mechanism_summary": "Frozen late-interaction checkpoints materially disagree on positive placement."
                        if category == "disagreement"
                        else "All frozen late-interaction checkpoints have low effectiveness for this query.",
                        "local_evidence_sha256": canonical_sha256(local),
                        "restricted_content_included": False,
                    }
                )
    local_mapping = {
        "schema_version": "bright-real-multi-vector-local-cases-v1",
        "classification": "research_only_no_publish",
        "source_text_included": False,
        "cases": restricted,
    }
    return (
        {
            "schema_version": "bright-real-multi-vector-cases-v1",
            "package_version": PACKAGE_VERSION,
            "publication": PUBLICATION,
            "counts": {"total": 24, "per_track_disagreement": 6, "per_track_failure": 6},
            "local_mapping_sha256": canonical_sha256(local_mapping),
            "cases": tracked,
        },
        local_mapping,
    )


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _canonical_document_token_stats(cell: Cell) -> dict[str, float]:
    """Aggregate saved window token counts back to canonical passages."""
    root = Path(cell.value["restricted_recompute"]["representation_root"])
    manifest_path = root / "representation.json"
    if file_sha256(manifest_path) != cell.value["representation"]["manifest_sha256"]:
        raise MultiVectorPackageError("Representation identity drifted during resource aggregation")
    manifest = load_json(manifest_path)
    documents = manifest["representations"]["documents"]
    mapping = manifest["representations"]["files"]["document_indices"]
    for entry in (documents["offsets"], mapping):
        path = root / entry["path"]
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise MultiVectorPackageError("Representation mapping drifted during resource aggregation")
    offsets = np.load(root / documents["offsets"]["path"], allow_pickle=False)
    document_indices = np.load(root / mapping["path"], allow_pickle=False)
    window_counts = np.diff(offsets).astype(np.float64)
    canonical_counts = np.bincount(
        document_indices,
        weights=window_counts,
        minlength=int(cell.value["track"]["documents"]),
    )
    if len(window_counts) != len(document_indices) or np.any(canonical_counts <= 0):
        raise MultiVectorPackageError("Canonical passage token-vector aggregation is incomplete")
    return _percentiles(canonical_counts)


def build_resources(
    cells: Mapping[tuple[str, str], Cell], predecl: Mapping[str, Any], gate_summary: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = {row["key"]: row for row in predecl["models"]}
    gates = {row["key"]: row for row in gate_summary["models"]}
    models = []
    for model in SELECTED_KEYS:
        spec, gate = frozen[model], gates[model]
        tracks = []
        for track in TRACKS:
            value = cells[(model, track)].value
            execution, representation = value["execution"], value["representation"]
            document_encoding = execution["encoding"]["documents"]
            query_encoding = execution["encoding"]["queries"]
            encoding_roles = (document_encoding, query_encoding)
            oom_count = sum(int(row["failure"]["oom_count"]) for row in encoding_roles)
            other_failure_count = sum(int(row["failure"]["other_failure_count"]) for row in encoding_roles)
            retry_count = sum(int(row["retry_count"]) for row in encoding_roles)
            fallback_attempt_count = sum(
                sum(count for batch, count in row["attempt_counts"].items() if int(batch) != row["requested_batch"])
                for row in encoding_roles
            )
            canonical_documents = int(value["track"]["documents"])
            tracks.append(
                {
                    "track": track,
                    "token_vectors": {
                        "queries": representation["query_token_count"],
                        "document_windows": representation["document_token_count"],
                        "canonical_passages": _canonical_document_token_stats(cells[(model, track)]),
                        "query_total": representation["query_token_vectors"],
                        "document_total": representation["document_token_vectors"],
                    },
                    "representation_storage": representation["storage"],
                    "encoding": execution["encoding"],
                    "document_encoding_units": {
                        "canonical_passages": canonical_documents,
                        "model_input_windows": int(document_encoding["items"]),
                        "canonical_passage_amortized_ms": float(document_encoding["wall_ms"]) / canonical_documents,
                        "canonical_passage_throughput_per_second": canonical_documents
                        / (float(document_encoding["wall_ms"]) / 1000.0),
                    },
                    "build_load_seconds": execution["build_load_seconds"],
                    "exact_search_seconds": execution["exact_search_seconds"],
                    "exact_search_latency_ms": execution["exact_search_latency_ms"],
                    "query_throughput_per_second": execution["query_throughput_per_second"],
                    "end_to_end_seconds": execution["end_to_end_seconds"],
                    "process_maxrss_bytes": execution["process_maxrss_bytes"],
                    "search_peak_vram_bytes": execution["peak_vram_bytes"],
                    "batch_fallback": execution["batch_fallback"],
                    "execution_outcome": {
                        "failure_count": oom_count + other_failure_count,
                        "oom_count": oom_count,
                        "other_failure_count": other_failure_count,
                        "retry_count": retry_count,
                        "fallback_attempt_count": fallback_attempt_count,
                        "fallback_triggered": bool(document_encoding["fallback_triggered"])
                        or bool(query_encoding["fallback_triggered"]),
                        "document_attempt_counts": document_encoding["attempt_counts"],
                        "query_attempt_counts": query_encoding["attempt_counts"],
                        "terminal_failure": None,
                        "formal_search_attempt_count": 1,
                        "formal_search_failure_count": 0,
                        "no_failure_observed": oom_count + other_failure_count == 0,
                        "formal_cell_finalized": True,
                    },
                }
            )
        models.append(
            {
                "model_key": model,
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "license": spec["license"],
                "license_evidence": "Pinned primary-source Hub API and model-card evidence in inventory.json",
                "lineage": spec["lineage"],
                "languages": spec["languages"],
                "parameters": spec["parameters"],
                "dimensions": spec["dimensions"],
                "query_length": spec["query_length"],
                "document_length": spec["document_length"],
                "query_route": spec["query_route"],
                "document_route": spec["document_route"],
                "mask_policy": spec["mask_policy"],
                "padding_policy": spec["padding_policy"],
                "snapshot_bytes": gate["snapshot_bytes"],
                "snapshot_identity_sha256": gate["snapshot_identity_sha256"],
                "gate_peak_vram_bytes": gate["peak_vram_bytes"],
                "tracks": tracks,
            }
        )
    return {
        "schema_version": "bright-real-multi-vector-resources-v1",
        "package_version": PACKAGE_VERSION,
        "publication": PUBLICATION,
        "download": {
            "conservative_snapshot_bytes": sum(row["snapshot_bytes"] for row in gates.values()),
            "recorded_incremental_successful_gate_bytes": sum(
                row["incremental_download_bytes"] for row in gates.values()
            ),
            "boundary": "The successful ColBERT gate recorded zero incremental bytes because its snapshot was downloaded by an earlier failed semantic-validation attempt; conservative snapshot bytes are the defensible story total.",
        },
        "models": models,
        "boundaries": [
            "Process ru_maxrss is a per-process observed peak, not an externally sampled machine-lifetime peak.",
            "Exact search uses saved float32 token vectors; no approximate index is used.",
            "Effectiveness applies only to the two fixed English research tracks.",
            "Training overlap is unknown; no verified zero-shot claim is allowed.",
        ],
    }


def render_report(
    summary: Mapping[str, Any],
    pairwise: Mapping[str, Any],
    slices: Mapping[str, Any],
    cases: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> str:
    lines = [
        "# BRIGHT Real Multi-Vector First-Round Comparison",
        "",
        "This package is research-only. Publication, public export, and leaderboard gates remain closed.",
        "",
        "The matrix contains three independently pinned late-interaction checkpoints on economics and psychology. Tracks remain independent; there is no pooled score or cross-paradigm claim.",
        "",
        "## Frozen checkpoint identities",
        "",
        "| Model | Revision | License | Parameters | Dimension | Query / document length |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in resources["models"]:
        lines.append(
            f"| {row['model_key']} | `{row['revision']}` | {row['license']} | {row['parameters']:,} | "
            f"{row['dimensions']} | {row['query_length']} / {row['document_length']} |"
        )
    lines.extend(
        [
            "",
            "Primary-source Hub API, model-card, saved-config, and artifact-metadata URLs or content hashes are frozen in `inventory.json`. All three checkpoints are public and ungated, use allowlisted safetensors, and load with `trust_remote_code=False`.",
        ]
    )
    for track in TRACKS:
        lines.extend(
            [
                "",
                f"## {track.title()} quality",
                "",
                "| Model | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        rows = {row["model_key"]: row for row in summary["tracks"][track]["rows"]}
        for model in summary["tracks"][track]["primary_order"]:
            metric = rows[model]["metrics"]
            lines.append(
                f"| {model} | {metric['ndcg@10']:.6f} | {metric['map@100']:.6f} | {metric['mrr@10']:.6f} | {metric['recall@10']:.6f} | {metric['recall@100']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Paired uncertainty",
            "",
            f"All three pairs per track and five metrics use {BOOTSTRAP_SAMPLES:,} matched-query percentile bootstrap samples with seed {BOOTSTRAP_SEED}. A directional claim requires the 95% interval to exclude zero.",
        ]
    )
    decisive = []
    for track, comparisons in pairwise["tracks"].items():
        for comparison in comparisons:
            for metric, value in comparison["metrics"].items():
                if value["verdict"] != "tie_or_uncertain":
                    winner = comparison["left"] if value["verdict"] == "left_wins" else comparison["right"]
                    loser = comparison["right"] if value["verdict"] == "left_wins" else comparison["left"]
                    decisive.append(
                        f"- {track}: {winner} exceeds {loser} on {metric}; the paired interval excludes zero."
                    )
    lines.extend(decisive or ["No paired interval supports a directional claim."])
    lines.extend(["", "## Predeclared slices and cases", ""])
    for track in TRACKS:
        for dimension in slices["tracks"][track]:
            counts = ", ".join(f"{row['bin']}: n={row['n']}" for row in dimension["bins"])
            lines.append(f"- {track} `{dimension['key']}`: {counts}")
    lines.extend(
        [
            "",
            "Bins with fewer than 10 queries are descriptive only and were not merged. The tracked case set has six disagreement and six shared-failure cases per track, using opaque labels and non-reversible local-evidence hashes only.",
            "",
            "## Protocol, resource, and maintenance guidance",
            "",
            "Exact float32 MaxSim is recomputed from saved token vectors, followed by frozen max-over-window passage aggregation. The 1x4 adapter gate independently matched NumPy MaxSim and exercised boolean padding masks before formal scoring.",
            "",
            "Representation storage is split into token-array bytes, mapping/identity bytes, representation-manifest bytes, complete replayable non-sidecar bytes, and sidecar bytes. The complete figure is the full per-track directory required for authenticated replay, not only values/offsets/masks.",
            "",
            "Encoding latency distributions are chunk-amortized observations: each measured chunk wall time divided by its item count. They are not presented as individually timed item percentiles. Every model/track also records attempt counts, OOM count, retries, fallback use, and terminal failure status.",
            "",
            "Choose by track and target metric, require a paired interval for directional quality claims, then compare checkpoint size, token-vector footprint, encoding throughput, and exact-search latency. The small AnswerAI checkpoint is the resource-oriented option; GTE-ModernColBERT and ColBERTv2 provide complementary architecture/scale points. These are shortlist recommendations, not a universal ranking.",
            "",
            "Training overlap is unknown. The data covers two English tracks only. The package does not redistribute source text or weights, and does not establish multilingual effectiveness, verified zero-shot behavior, production SLA, or complete cross-paradigm superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def assert_tracked_privacy(files: Mapping[str, bytes], data_root: str | Path) -> dict[str, Any]:
    text = b"\n".join(files.values()).decode("utf-8")
    forbidden_fragments = (
        "results/",
        "/data",
        "/home",
        ".cache/",
        "snapshots/",
        "corpus.jsonl",
        "queries.jsonl",
        "qrels.jsonl",
    )
    if any(fragment in text for fragment in forbidden_fragments):
        raise MultiVectorPackageError("Tracked output contains a private or restricted path")

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if FORBIDDEN_KEYS & set(value):
                raise MultiVectorPackageError(
                    f"Tracked output contains forbidden row-level fields: {sorted(FORBIDDEN_KEYS & set(value))}"
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    values: set[str] = set()
    for name, payload in files.items():
        if name.endswith(".json"):
            value = json.loads(payload)
            walk(value)

            def strings(item: Any) -> None:
                if isinstance(item, str):
                    values.add(item)
                elif isinstance(item, dict):
                    for key, child in item.items():
                        values.add(str(key))
                        strings(child)
                elif isinstance(item, list):
                    for child in item:
                        strings(child)

            strings(value)
    restricted_ids, restricted_texts = set(), []
    for track in TRACKS:
        data = load_materialized(data_root, track)
        restricted_ids.update(row["id"] for row in data.corpus)
        restricted_texts.extend(row["text"] for row in data.queries)
        restricted_texts.extend(row["content"] for row in data.corpus)
    if values & restricted_ids or any(len(value) >= 24 and value in text for value in restricted_texts):
        raise MultiVectorPackageError("Tracked output contains a canonical ID or restricted source text")
    return {
        "status": "pass",
        "tracked_files_scanned": len(files),
        "canonical_ids_found": 0,
        "canonical_source_texts_found": 0,
        "private_paths_found": 0,
        "public_gate": "closed",
    }


def compute_outputs(
    *,
    cells: Mapping[tuple[str, str], Cell],
    predecl: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    data_root: str | Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    summary, pairwise = build_quality(cells)
    slices = build_slices(cells, data_root, predecl)
    cases, local_mapping = build_cases(cells, data_root)
    resources = build_resources(cells, predecl, gate_summary)
    values = {
        "summary.json": summary,
        "pairwise.json": pairwise,
        "slices.json": slices,
        "cases.json": cases,
        "resources.json": resources,
    }
    for value in values.values():
        validate_schema(value)
    files = {name: formatted_json_bytes(value) for name, value in values.items()}
    files["report.md"] = render_report(summary, pairwise, slices, cases, resources).encode()
    return files, {"local_mapping": local_mapping, "privacy": assert_tracked_privacy(files, data_root)}


def make_manifest(
    *,
    files: Mapping[str, bytes],
    cells: Mapping[tuple[str, str], Cell],
    predecl_path: Path,
    inventory_path: Path,
    gate_summary_path: Path,
    representation_inputs: Mapping[str, Any],
    local_mapping: Mapping[str, Any],
    privacy: Mapping[str, Any],
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    repo = Path(repo_root)
    manifest = {
        "schema_version": "bright-real-multi-vector-candidate-v1",
        "package_version": PACKAGE_VERSION,
        "story_id": "S-20260814-012",
        "status": "candidate_not_accepted",
        "publication": PUBLICATION,
        "bindings": {
            "predeclaration_sha256": file_sha256(predecl_path),
            "inventory_sha256": file_sha256(inventory_path),
            "gate_summary_sha256": file_sha256(gate_summary_path),
            "formal_cells": {
                f"{model}:{track}": cells[(model, track)].identity for model in SELECTED_KEYS for track in TRACKS
            },
            "representation_inputs": dict(representation_inputs),
            "local_mapping_sha256": canonical_sha256(local_mapping),
        },
        "tracked_files": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in sorted(files.items())
        },
        "implementation_files": {
            name: {"bytes": (repo / name).stat().st_size, "sha256": file_sha256(repo / name)}
            for name in IMPLEMENTATION_CANDIDATES
        },
        "privacy_scan": dict(privacy),
        "content_policy": "Aggregate statistics and opaque case labels only; no canonical IDs, source text, raw rankings, per-query rows, secrets, private paths, data, or weights.",
        "validator": {"independent_acceptance_required": True, "accepted": False},
    }
    validate_schema(manifest)
    return manifest


def write_or_check_package(
    *,
    package_root: str | Path,
    report_path: str | Path,
    local_mapping_path: str | Path,
    files: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    local_mapping: Mapping[str, Any],
    check_only: bool,
) -> str:
    root, report, local = Path(package_root), Path(report_path), Path(local_mapping_path)
    manifest_payload = formatted_json_bytes(manifest)
    identity = hashlib.sha256(manifest_payload).hexdigest()
    expected = {name: payload for name, payload in files.items() if name != "report.md"}
    if check_only:
        if file_sha256(root / "manifest.json") != identity or _sidecar(root / "manifest.json") != identity:
            raise MultiVectorPackageError("Candidate manifest cannot be rebuilt deterministically")
        expected_names = {
            "manifest.json",
            "manifest.json.sha256",
            *(name for name in expected),
            *(f"{name}.sha256" for name in expected),
        }
        if {path.name for path in root.iterdir() if path.is_file()} != expected_names:
            raise MultiVectorPackageError("Candidate package contains an unexpected or missing file")
        for name, payload in expected.items():
            path = root / name
            if path.read_bytes() != payload or _sidecar(path) != hashlib.sha256(payload).hexdigest():
                raise MultiVectorPackageError(f"Tracked package drifted: {name}")
        if (
            report.read_bytes() != files["report.md"]
            or _sidecar(report) != hashlib.sha256(files["report.md"]).hexdigest()
            or load_json(local) != local_mapping
            or _sidecar(local) != file_sha256(local)
        ):
            raise MultiVectorPackageError("Report or restricted case mapping cannot be rebuilt")
        for name, expected_identity in manifest["implementation_files"].items():
            path = Path(name)
            if path.stat().st_size != expected_identity["bytes"] or file_sha256(path) != expected_identity["sha256"]:
                raise MultiVectorPackageError(f"Implementation candidate drifted: {name}")
        return identity
    if root.exists() and any(root.iterdir()):
        raise MultiVectorPackageError("Refusing to overwrite a non-empty candidate package")
    if report.exists() or local.exists():
        raise MultiVectorPackageError("Refusing to overwrite an existing report or restricted mapping")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    local.parent.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        path = root / name
        path.write_bytes(payload)
        path.with_suffix(path.suffix + ".sha256").write_text(
            hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
        )
    report.write_bytes(files["report.md"])
    report.with_suffix(report.suffix + ".sha256").write_text(
        hashlib.sha256(files["report.md"]).hexdigest() + "\n", encoding="ascii"
    )
    local.write_bytes(formatted_json_bytes(local_mapping))
    local.with_suffix(local.suffix + ".sha256").write_text(file_sha256(local) + "\n", encoding="ascii")
    (root / "manifest.json").write_bytes(manifest_payload)
    (root / "manifest.json.sha256").write_text(identity + "\n", encoding="ascii")
    return identity
