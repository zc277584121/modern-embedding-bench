"""Fail-closed aggregation for the research-only BRIGHT learned-sparse matrix."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from mm_embed.benchmark import bright_learned_sparse_batch_a as batch_a
from mm_embed.benchmark import bright_learned_sparse_batch_b as batch_b
from mm_embed.benchmark.bright_multidomain_v02 import load_materialized
from mm_embed.benchmark.retrieval_v01 import (
    METRIC_NAMES,
    aggregate_metrics,
    paired_bootstrap_deltas,
    tokens,
)

PACKAGE_VERSION = "bright-learned-sparse-main-v0.1"
INPUT_SCHEMA = "bright-learned-sparse-main-inputs-v01.schema.json"
SUMMARY_SCHEMA = "bright-learned-sparse-main-summary-v01.schema.json"
PAIRWISE_SCHEMA = "bright-learned-sparse-main-pairwise-v01.schema.json"
SLICES_SCHEMA = "bright-learned-sparse-main-slices-v01.schema.json"
CASES_SCHEMA = "bright-learned-sparse-main-cases-v01.schema.json"
RESOURCES_SCHEMA = "bright-learned-sparse-main-resources-v01.schema.json"
MANIFEST_SCHEMA = "bright-learned-sparse-main-manifest-v01.schema.json"
REPLAY_SCHEMA = "bright-learned-sparse-main-replay-v01.schema.json"

MODELS = (
    "granite-30m-sparse",
    "opensearch-doc-v2-mini",
    "opensearch-doc-v3",
    "bge-m3",
    "splade-tiny",
    "opensearch-doc-v2-distill",
    "opensearch-multilingual",
)
TRACKS = ("economics", "psychology")
BATCH_A_MODELS = MODELS[:4]
BATCH_B_MODELS = MODELS[4:]
BOOTSTRAP_SEED = 20_260_826
BOOTSTRAP_SAMPLES = 10_000
PUBLICATION = {
    "publish": False,
    "leaderboard_publish": False,
    "public_export_allowed": False,
    "gate": "closed",
    "classification": "research_only",
}
PACKAGE_PROTOCOL = {
    "schema_version": "bright-learned-sparse-main-protocol-v1",
    "matrix": {"models": list(MODELS), "tracks": list(TRACKS), "cells": 14},
    "aggregation": "independent_track_macro_average_no_cross_track_micro_average",
    "metrics": list(METRIC_NAMES),
    "pairwise": {
        "pairs_per_track": 21,
        "method": "query_aligned_paired_bootstrap_left_minus_right",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "decisive_rule": "95_percentile_interval_excludes_zero",
        "otherwise": "tie_or_uncertain",
    },
    "slices": "active_predeclaration_v2_exact_definitions_and_bins",
    "cases": {"disagreement_per_track": 6, "failure_per_track": 6},
    "publication": PUBLICATION,
    "models_loaded": False,
}


class PackageError(ValueError):
    """Raised when an input or package violates the fixed aggregation contract."""


@dataclass(frozen=True)
class Cell:
    """One externally authenticated raw cell and its local row-level evidence."""

    model: str
    track: str
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    per_query: dict[str, dict[str, float]]
    rankings: dict[str, list[dict[str, Any]]]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formatted_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _require_sha256(value: str | None, label: str) -> str:
    if value is None or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise PackageError(f"Missing or invalid externally supplied {label}")
    return value


def _load_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"Missing or invalid JSON: {path}") from exc


def _validate_schema(value: Any, schema_name: str) -> None:
    schema_path = Path(__file__).parents[3] / "schemas" / schema_name
    try:
        schema = _load_json(schema_path)
        Draft202012Validator(schema).validate(value)
    except Exception as exc:
        raise PackageError(f"{schema_name} validation failed: {exc}") from exc


def load_input_contract(path: str | Path, expected_sha256: str | None) -> dict[str, Any]:
    """Load the identity-only input contract using an out-of-band expected hash."""
    expected = _require_sha256(expected_sha256, "input contract SHA256")
    if file_sha256(path) != expected:
        raise PackageError("Input contract does not match its externally supplied identity")
    value = _load_json(path)
    _validate_schema(value, INPUT_SCHEMA)
    if value["package_protocol_sha256"] != canonical_sha256(PACKAGE_PROTOCOL):
        raise PackageError("Package protocol identity drifted")
    expected_cells = {f"{model}:{track}" for model in MODELS for track in TRACKS}
    if set(value["raw_manifest_sha256"]) != expected_cells:
        raise PackageError("Input contract does not bind exactly the 14 fixed cells")
    if set(value["batch_b_replay_sha256"]) != {
        f"{model}:{track}" for model in BATCH_B_MODELS for track in TRACKS
    }:
        raise PackageError("Input contract does not bind exactly six Batch-B replays")
    if set(value["snapshot_evidence_sha256"]) != set(BATCH_B_MODELS):
        raise PackageError("Input contract does not bind exactly three Batch-B snapshots")
    return value


def _verify_batch_a_package(path: Path, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise PackageError("Batch-A accepted package manifest identity drifted")
    manifest = _load_json(path)
    for name, identity in manifest["tracked_files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PackageError("Batch-A package contains an unsafe tracked path")
        artifact = Path(__file__).parents[3] / relative
        if artifact.stat().st_size != identity["bytes"] or file_sha256(artifact) != identity["sha256"]:
            raise PackageError(f"Batch-A accepted tracked artifact drifted: {name}")
    return manifest


def _load_cell_rows(root: Path, manifest: dict[str, Any], identity: str) -> Cell:
    model = str(manifest["model"]["key"])
    track = str(manifest["data"]["track"])
    per_query = _load_json(root / "per_query_metrics.json")
    ranking_rows = _load_json(root / "rankings.json")
    rankings = {str(row["query_id"]): list(row["hits"]) for row in ranking_rows}
    if set(per_query) != set(rankings):
        raise PackageError(f"Per-query/ranking query alignment drifted for {model}:{track}")
    return Cell(model, track, root, manifest, identity, per_query, rankings)


def collect_authenticated_cells(
    *,
    contract: Mapping[str, Any],
    repo_root: str | Path,
    data_root: str | Path,
    batch_a_raw_root: str | Path,
    batch_b_raw_root: str | Path,
    batch_b_replay_root: str | Path,
    snapshot_evidence_root: str | Path,
) -> dict[tuple[str, str], Cell]:
    """Validate all immutable inputs and return exactly 14 authenticated cells."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PackageError("Aggregation requires CUDA_VISIBLE_DEVICES='' and never loads a model")
    repo = Path(repo_root)
    predecl_path = repo / "benchmark/artifacts/bright-learned-sparse-batch-b-v0.1/predeclaration.json"
    predecl = batch_b.validate_predeclaration(
        predecl_path, expected_sha256=contract["active_predeclaration_sha256"]
    )
    if contract["active_predeclaration_sha256"] != batch_b.ACTIVE_PREDECLARATION_SHA256:
        raise PackageError("Input contract does not bind the unique active predeclaration")
    if contract["batch_b_protocol_sha256"] != batch_b.formal_protocol()["identity_sha256"]:
        raise PackageError("Batch-B formal protocol identity drifted")
    if contract["canonical_data_sha256"] != predecl["bindings"]["canonical_data"]["manifest_sha256"]:
        raise PackageError("Canonical data contract identity drifted")
    batch_b.validate_canonical_binding(predecl, data_root)
    batch_a_package_path = repo / "benchmark/artifacts/bright-learned-sparse-batch-a-v0.1/manifest.json"
    batch_a_package = _verify_batch_a_package(
        batch_a_package_path, contract["batch_a_package_manifest_sha256"]
    )
    if batch_a_package["protocol"]["identity_sha256"] != contract["batch_a_protocol_sha256"]:
        raise PackageError("Batch-A protocol identity drifted")
    accepted_a = {
        f"{row['model_key']}:{row['track']}": row["manifest_sha256"]
        for row in batch_a_package["raw_results"]
    }
    expected_a = {
        key: value for key, value in contract["raw_manifest_sha256"].items()
        if key.split(":", 1)[0] in BATCH_A_MODELS
    }
    if accepted_a != expected_a:
        raise PackageError("Batch-A accepted package/raw bindings disagree")

    cells: dict[tuple[str, str], Cell] = {}
    for model in BATCH_A_MODELS:
        for track in TRACKS:
            key = f"{model}:{track}"
            root = Path(batch_a_raw_root) / model / track
            manifest = batch_a.validate_raw_result(
                root, data_root, expected_manifest_sha256=contract["raw_manifest_sha256"][key]
            )
            cell = _load_cell_rows(root, manifest, contract["raw_manifest_sha256"][key])
            if (cell.model, cell.track) in cells:
                raise PackageError("Duplicate cell detected")
            cells[(cell.model, cell.track)] = cell

    for model in BATCH_B_MODELS:
        evidence_path = Path(snapshot_evidence_root) / model / "snapshot.json"
        evidence_sha = contract["snapshot_evidence_sha256"][model]
        if file_sha256(evidence_path) != evidence_sha:
            raise PackageError(f"Snapshot evidence identity drifted for {model}")
        snapshot_evidence = _load_json(evidence_path)
        for track in TRACKS:
            key = f"{model}:{track}"
            root = Path(batch_b_raw_root) / model / track
            manifest_sha = contract["raw_manifest_sha256"][key]
            replay_result = batch_b.replay_formal_cell(
                result_root=root,
                data_root=data_root,
                predeclaration=predecl,
                predeclaration_sha256=contract["active_predeclaration_sha256"],
                snapshot_evidence=snapshot_evidence,
                snapshot_evidence_sha256=evidence_sha,
                expected_manifest_sha256=manifest_sha,
                require_cpu_only=True,
            )
            replay_path = Path(batch_b_replay_root) / model / f"{track}.json"
            replay_sha = contract["batch_b_replay_sha256"][key]
            if file_sha256(replay_path) != replay_sha:
                raise PackageError(f"Batch-B replay identity drifted for {key}")
            saved_replay = _load_json(replay_path)
            _validate_schema(saved_replay, batch_b.FORMAL_REPLAY_SCHEMA)
            if saved_replay != replay_result["evidence"]:
                raise PackageError(f"Saved Batch-B replay evidence drifted for {key}")
            manifest = _load_json(root / "manifest.json")
            cell = _load_cell_rows(root, manifest, manifest_sha)
            if (cell.model, cell.track) in cells:
                raise PackageError("Duplicate cell detected")
            cells[(cell.model, cell.track)] = cell

    expected = {(model, track) for model in MODELS for track in TRACKS}
    if set(cells) != expected or len(cells) != 14:
        raise PackageError("Aggregation requires exactly the fixed 7x2 matrix")
    for track in TRACKS:
        query_sets = {frozenset(cells[(model, track)].per_query) for model in MODELS}
        if len(query_sets) != 1:
            raise PackageError(f"Per-query alignment drifted within {track}")
    return cells


def _paired(left: Mapping[str, Mapping[str, float]], right: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    result = paired_bootstrap_deltas(
        dict(left), dict(right), samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    return {
        metric: {
            **values,
            "verdict": (
                "left_wins" if values["low"] > 0 else
                "right_wins" if values["high"] < 0 else
                "tie_or_uncertain"
            ),
        }
        for metric, values in result.items()
    }


def build_leaderboards(cells: Mapping[tuple[str, str], Cell]) -> tuple[dict[str, Any], dict[str, Any]]:
    tracks: dict[str, Any] = {}
    pairwise_tracks: dict[str, Any] = {}
    for track in TRACKS:
        rows = [
            {
                "model_key": model,
                "metrics": cells[(model, track)].manifest["metrics"],
                "raw_manifest_sha256": cells[(model, track)].manifest_sha256,
            }
            for model in MODELS
        ]
        tracks[track] = {
            "primary_order": [
                row["model_key"]
                for row in sorted(rows, key=lambda row: (-row["metrics"]["ndcg@10"], row["model_key"]))
            ],
            "rows": sorted(rows, key=lambda row: row["model_key"]),
            "metric_orders": {
                metric: [
                    row["model_key"]
                    for row in sorted(rows, key=lambda row: (-row["metrics"][metric], row["model_key"]))
                ]
                for metric in METRIC_NAMES
            },
        }
        comparisons = []
        for left, right in itertools.combinations(sorted(MODELS), 2):
            comparisons.append({
                "left": left,
                "right": right,
                "n": len(cells[(left, track)].per_query),
                "metrics": _paired(cells[(left, track)].per_query, cells[(right, track)].per_query),
            })
        if len(comparisons) != 21:
            raise PackageError("Each track requires all 21 model pairs")
        pairwise_tracks[track] = comparisons
    summary = {
        "schema_version": "bright-learned-sparse-main-summary-v1",
        "package_version": PACKAGE_VERSION,
        "publication": PUBLICATION,
        "track_aggregation": "independent_no_cross_track_micro_average_or_overall_rank",
        "tracks": tracks,
    }
    pairwise = {
        "schema_version": "bright-learned-sparse-main-pairwise-v1",
        "package_version": PACKAGE_VERSION,
        "publication": PUBLICATION,
        "method": PACKAGE_PROTOCOL["pairwise"],
        "tracks": pairwise_tracks,
    }
    return summary, pairwise


def _slice_memberships(data: Any, frozen_slices: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    documents = {row["id"]: row["content"] for row in data.corpus}
    output: dict[str, dict[str, str]] = {}
    for query in data.queries:
        query_id = query["id"]
        query_tokens = tokens(query["text"].strip())
        unique_query = set(query_tokens)
        positives = sorted(data.qrels[query_id])
        positive_tokens = [tokens(documents[document_id]) for document_id in positives]
        overlap = max(
            (len(unique_query & set(values)) / len(unique_query) if unique_query else 0.0)
            for values in positive_tokens
        )
        median_length = float(np.median([len(values) for values in positive_tokens]))
        density = len(set(positives))
        output[query_id] = {
            "query_length": "<=64" if len(query_tokens) <= 64 else "65-128" if len(query_tokens) <= 128 else ">128",
            "positive_lexical_overlap": (
                "0" if overlap == 0 else "(0,0.25]" if overlap <= 0.25 else
                "(0.25,0.50]" if overlap <= 0.50 else ">0.50"
            ),
            "positive_document_length": "<=128" if median_length <= 128 else "129-512" if median_length <= 512 else ">512",
            "positive_qrel_density": "1" if density == 1 else "2-4" if density <= 4 else ">=5",
        }
    frozen = {row["key"]: list(row["bins"]) for row in frozen_slices}
    observed = {key: sorted({values[key] for values in output.values()}) for key in frozen}
    for key, bins in frozen.items():
        if set(observed[key]) - set(bins):
            raise PackageError(f"Derived slice bin drifted for {key}")
    return output


def build_slices(
    cells: Mapping[tuple[str, str], Cell], data_root: str | Path, predeclaration: Mapping[str, Any]
) -> dict[str, Any]:
    frozen_slices = predeclaration["slices"]
    tracks: dict[str, Any] = {}
    for track in TRACKS:
        data = load_materialized(data_root, track)
        membership = _slice_memberships(data, frozen_slices)
        slice_rows = []
        for frozen in frozen_slices:
            bins = []
            for bin_name in frozen["bins"]:
                query_ids = sorted(query_id for query_id, values in membership.items() if values[frozen["key"]] == bin_name)
                model_rows = []
                for model in MODELS:
                    subset = {query_id: cells[(model, track)].per_query[query_id] for query_id in query_ids}
                    model_rows.append({
                        "model_key": model,
                        "n": len(query_ids),
                        "metrics": aggregate_metrics(subset) if subset else None,
                    })
                pairs = []
                if len(query_ids) >= 10:
                    for left, right in itertools.combinations(sorted(MODELS), 2):
                        left_values = {query_id: cells[(left, track)].per_query[query_id] for query_id in query_ids}
                        right_values = {query_id: cells[(right, track)].per_query[query_id] for query_id in query_ids}
                        pairs.append({"left": left, "right": right, "n": len(query_ids), "metrics": _paired(left_values, right_values)})
                bins.append({
                    "bin": bin_name,
                    "n": len(query_ids),
                    "membership_sha256": canonical_sha256(query_ids),
                    "models": model_rows,
                    "pairwise": pairs,
                    "inference": "paired_ci_allowed" if len(query_ids) >= 10 else "point_estimates_only_no_win_claim",
                })
            slice_rows.append({
                "key": frozen["key"],
                "definition": frozen["definition"],
                "limitations": frozen["limitations"],
                "sample_rule": frozen["sample_rule"],
                "bins": bins,
            })
        tracks[track] = slice_rows
    return {
        "schema_version": "bright-learned-sparse-main-slices-v1",
        "package_version": PACKAGE_VERSION,
        "publication": PUBLICATION,
        "active_predeclaration_sha256": batch_b.ACTIVE_PREDECLARATION_SHA256,
        "tracks": tracks,
    }


def _best_positive_rank(hits: Sequence[Mapping[str, Any]], positives: set[str]) -> int:
    return min((int(hit["rank"]) for hit in hits if hit["document_id"] in positives), default=101)


def build_cases(
    cells: Mapping[tuple[str, str], Cell], data_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    tracked: list[dict[str, Any]] = []
    local: list[dict[str, Any]] = []
    for track in TRACKS:
        data = load_materialized(data_root, track)
        query_rows = {row["id"]: row for row in data.queries}
        candidates = []
        for query_id in sorted(query_rows):
            positives = set(data.qrels[query_id])
            model_evidence = {}
            positive_ranks = []
            recall_hits = []
            for model in MODELS:
                cell = cells[(model, track)]
                metrics = cell.per_query[query_id]
                rank = _best_positive_rank(cell.rankings[query_id], positives)
                positive_ranks.append(rank)
                recall_hits.append(metrics["recall@10"] > 0)
                model_evidence[model] = {
                    "best_positive_rank": rank,
                    "metrics": metrics,
                    "raw_manifest_sha256": cell.manifest_sha256,
                    "local_result_reference": f"{model}/{track}",
                }
            pair_disagreements = sum(left != right for left, right in itertools.combinations(recall_hits, 2))
            candidates.append({
                "query_id": query_id,
                "positive_ids": sorted(positives),
                "rank_range": max(positive_ranks) - min(positive_ranks),
                "recall10_pair_disagreements": pair_disagreements,
                "mean_ndcg10": float(np.mean([model_evidence[m]["metrics"]["ndcg@10"] for m in MODELS])),
                "zero_ndcg10_models": sum(model_evidence[m]["metrics"]["ndcg@10"] == 0 for m in MODELS),
                "mean_recall100": float(np.mean([model_evidence[m]["metrics"]["recall@100"] for m in MODELS])),
                "models": model_evidence,
            })
        disagreements = sorted(
            candidates,
            key=lambda row: (-row["rank_range"], -row["recall10_pair_disagreements"], row["query_id"]),
        )[:6]
        disagreement_ids = {row["query_id"] for row in disagreements}
        failures = sorted(
            (row for row in candidates if row["query_id"] not in disagreement_ids),
            key=lambda row: (row["mean_ndcg10"], -row["zero_ndcg10_models"], row["mean_recall100"], row["query_id"]),
        )[:6]
        if len(disagreements) != 6 or len(failures) != 6 or disagreement_ids & {row["query_id"] for row in failures}:
            raise PackageError("Case selection count or mutual exclusion contract failed")
        for category, rows in (("disagreement", disagreements), ("failure", failures)):
            for ordinal, row in enumerate(rows, 1):
                opaque_id = f"{track}-{category}-{ordinal:02d}"
                evidence = {
                    "track": track,
                    "category": category,
                    "query_id": row["query_id"],
                    "positive_document_ids": row["positive_ids"],
                    "selection": {
                        "rank_range": row["rank_range"],
                        "recall10_pair_disagreements": row["recall10_pair_disagreements"],
                        "mean_ndcg10": row["mean_ndcg10"],
                        "zero_ndcg10_models": row["zero_ndcg10_models"],
                        "mean_recall100": row["mean_recall100"],
                    },
                    "models": row["models"],
                    "source_text_included": False,
                }
                local.append({"opaque_case_id": opaque_id, **evidence})
                tracked.append({
                    "opaque_case_id": opaque_id,
                    "track": track,
                    "category": category,
                    "selection_rule": (
                        "descending_positive_rank_range_then_recall10_pair_disagreement_then_canonical_id"
                        if category == "disagreement" else
                        "ascending_mean_ndcg10_then_zero_count_desc_then_mean_recall100_then_canonical_id"
                    ),
                    "mechanism_summary": (
                        "The seven frozen sparse mechanisms disagree materially on positive-document placement."
                        if category == "disagreement" else
                        "The seven frozen sparse mechanisms share a low-effectiveness failure region for this query."
                    ),
                    "local_evidence_sha256": canonical_sha256(evidence),
                    "restricted_content_included": False,
                })
    if len(tracked) != 24 or len({row["opaque_case_id"] for row in tracked}) != 24:
        raise PackageError("Tracked case evidence must contain exactly 24 unique opaque cases")
    local_mapping = {
        "schema_version": "bright-learned-sparse-main-local-cases-v1",
        "classification": "research_only_no_publish",
        "source_text_included": False,
        "cases": local,
    }
    cases = {
        "schema_version": "bright-learned-sparse-main-cases-v1",
        "package_version": PACKAGE_VERSION,
        "publication": PUBLICATION,
        "counts": {"total": 24, "per_track_disagreement": 6, "per_track_failure": 6},
        "local_mapping_sha256": canonical_sha256(local_mapping),
        "cases": tracked,
    }
    return cases, local_mapping


def _representation_bytes(cell: Cell, role: str) -> tuple[int, int]:
    representation = cell.manifest["representations"][role]
    rows = int(representation["shape"][0])
    nnz = int(representation["nnz"])
    csr_bytes = int(representation.get("csr_memory_bytes", nnz * 8 + (rows + 1) * 4))
    if "npz_bytes" in representation:
        stored = int(representation["npz_bytes"])
    elif role == "documents":
        stored = sum(
            identity["bytes"] for name, identity in cell.manifest["artifacts"].items()
            if name.startswith("document_chunks/") and name.endswith(".npz")
        )
    else:
        stored = int(cell.manifest["artifacts"]["queries.npz"]["bytes"])
    return csr_bytes, stored


def build_resources(
    cells: Mapping[tuple[str, str], Cell], predeclaration: Mapping[str, Any], batch_a_inventory: Mapping[str, Any]
) -> dict[str, Any]:
    a_inventory = {row["key"]: row for row in batch_a_inventory["models"]}
    b_predecl = {row["key"]: row for row in predeclaration["models"]}
    rows = []
    for model in MODELS:
        model_rows = []
        for track in TRACKS:
            cell = cells[(model, track)]
            manifest = cell.manifest
            if model in BATCH_A_MODELS:
                documents = manifest["audit"]["document_chunks"]
                queries = manifest["audit"]["query"]
                document_rss = int(documents["peak_ram_bytes"])
                query_rss = int(queries["peak_ram_bytes"])
                attempts = documents.get("batch_sizes_attempted", documents.get("batch_size_attempts", documents["batch_sizes_used"]))
                query_attempts = queries.get("batch_size_attempts", [queries["batch_size_used"]])
                fallback = len(set(attempts)) > 1 or len(set(query_attempts)) > 1
                exact_search = float(manifest["audit"]["exact_search_s"])
            else:
                documents = manifest["resources"]["documents"]
                queries = manifest["resources"]["queries"]
                document_rss = int(documents["encoding_observed_peak_rss_bytes"])
                query_rss = int(queries["encoding_observed_peak_rss_bytes"])
                attempts = documents["batch_sizes_attempted"]
                query_attempts = queries["batch_size_attempted"]
                fallback = bool(manifest["resources"]["fallback_oom"]["document_fallback_chunks"])
                exact_search = float(manifest["resources"]["exact_search_s"])
            doc_csr, doc_stored = _representation_bytes(cell, "documents")
            query_csr, query_stored = _representation_bytes(cell, "queries")
            model_rows.append({
                "track": track,
                "query_nnz": manifest["nnz"]["queries"],
                "document_nnz": manifest["nnz"]["documents"],
                "truncation": {"queries": int(queries["truncated_count"]), "documents": int(documents["truncated_count"])},
                "encoding": {
                    "query_latency_ms": float(queries["latency_ms"]),
                    "document_latency_ms": float(documents["latency_ms"]),
                    "query_throughput_items_per_second": len(cell.per_query) / (float(queries["latency_ms"]) / 1000.0),
                    "document_throughput_items_per_second": int(manifest["data"]["documents"]) / (float(documents["latency_ms"]) / 1000.0),
                    "query_observed_rss_bytes": query_rss,
                    "document_observed_rss_bytes": document_rss,
                    "rss_scope": "encoding_observed_not_whole_cell_peak",
                    "query_peak_vram_bytes": int(queries["peak_vram_bytes"]),
                    "document_peak_vram_bytes": int(documents["peak_vram_bytes"]),
                },
                "exact_search_seconds": exact_search,
                "representation": {
                    "query_csr_bytes": query_csr,
                    "document_csr_bytes": doc_csr,
                    "index_bytes": doc_csr,
                    "query_stored_bytes": query_stored,
                    "document_stored_bytes": doc_stored,
                },
                "batch": {
                    "requested": 8,
                    "document_attempted": list(attempts),
                    "document_used": list(documents["batch_sizes_used"]),
                    "query_attempted": list(query_attempts),
                    "query_used": int(queries["batch_size_used"]),
                    "fallback_observed": fallback,
                    "batch1_oom": bool(manifest.get("resources", {}).get("fallback_oom", {}).get("batch1_oom", False)),
                },
            })
        model_contract = cells[(model, TRACKS[0])].manifest["model"]
        frozen = a_inventory[model] if model in BATCH_A_MODELS else b_predecl[model]
        rows.append({
            "model_key": model,
            "repo_id": model_contract["repo_id"],
            "revision": model_contract["revision"],
            "license": model_contract["license"],
            "license_boundary": (
                predeclaration["model_evidence"][model]["license_boundary"]
                if model in BATCH_B_MODELS else model_contract["license"]
            ),
            "mechanism": model_contract["mechanism"],
            "query_route": model_contract["query_route"],
            "document_route": model_contract["document_route"],
            "dimensions": int(model_contract["dimensions"]),
            "snapshot_identity_sha256": model_contract["snapshot_identity_sha256"],
            "snapshot_bytes": int(frozen["estimated_snapshot_bytes"]),
            "snapshot_bytes_kind": "frozen_estimate",
            "parameter_count": frozen.get("parameter_count"),
            "tracks": model_rows,
        })
    return {
        "schema_version": "bright-learned-sparse-main-resources-v1",
        "package_version": PACKAGE_VERSION,
        "publication": PUBLICATION,
        "models": rows,
        "boundaries": [
            "Encoding-observed RSS is not a whole-cell process peak.",
            "Effectiveness is established only on the two fixed English BRIGHT tracks.",
            "No verified zero-shot or multilingual-effectiveness claim is allowed.",
            "The package is research-only and is not publication readiness evidence.",
        ],
    }


def _significant_claims(pairwise: Mapping[str, Any]) -> list[str]:
    claims = []
    for track, rows in pairwise["tracks"].items():
        for row in rows:
            for metric, result in row["metrics"].items():
                if result["verdict"] == "left_wins":
                    claims.append(f"{track}: {row['left']} exceeds {row['right']} on {metric}; paired 95% CI excludes zero.")
                elif result["verdict"] == "right_wins":
                    claims.append(f"{track}: {row['right']} exceeds {row['left']} on {metric}; paired 95% CI excludes zero.")
    return claims


def render_report(
    summary: Mapping[str, Any], pairwise: Mapping[str, Any], slices: Mapping[str, Any],
    cases: Mapping[str, Any], resources: Mapping[str, Any]
) -> str:
    lines = [
        "# BRIGHT Learned-Sparse Main Comparison",
        "",
        "This package is research-only. The public gate is closed. It contains aggregate statistics and opaque case identifiers only.",
        "",
        "Results are reported independently for economics and psychology. There is no cross-track micro-average or overall model ranking.",
    ]
    for track in TRACKS:
        lines.extend(["", f"## {track.title()} leaderboard", "", "| Model | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |", "|---|---:|---:|---:|---:|---:|"])
        rows = {row["model_key"]: row for row in summary["tracks"][track]["rows"]}
        for model in summary["tracks"][track]["primary_order"]:
            metric = rows[model]["metrics"]
            lines.append(f"| {model} | {metric['ndcg@10']:.6f} | {metric['map@100']:.6f} | {metric['mrr@10']:.6f} | {metric['recall@10']:.6f} | {metric['recall@100']:.6f} |")
        lines.extend(["", "Metric-specific orders are stored in `summary.json`; the table order uses nDCG@10 only within this track."])
    claims = _significant_claims(pairwise)
    lines.extend(["", "## Paired uncertainty", ""])
    lines.append(f"All 21 model pairs per track and all five metrics use {BOOTSTRAP_SAMPLES:,} query-aligned samples with seed {BOOTSTRAP_SEED}.")
    lines.append("A win statement appears only when the paired percentile interval excludes zero; every other result is tie/uncertain.")
    if claims:
        lines.extend(["", "Significant comparisons are fully enumerated in `pairwise.json`. Representative package-level findings:"])
        lines.extend(f"- {claim}" for claim in claims[:12])
    lines.extend(["", "## Predeclared slices", ""])
    for track in TRACKS:
        lines.append(f"### {track.title()}")
        lines.append("")
        for row in slices["tracks"][track]:
            counts = ", ".join(f"{item['bin']}: n={item['n']}" for item in row["bins"])
            lines.append(f"- `{row['key']}` — {counts}")
    lines.extend([
        "", "Bins with n=1–9 have point estimates only; n=0 is NA. Bins are never merged, and thresholds are unchanged from the active predeclaration.",
        "", "## Opaque case review", "",
        "Exactly six disagreement and six failure cases are selected per track. Tracked files contain only opaque case IDs, rule labels, mechanism-level summaries, and non-reversible local evidence hashes.",
    ])
    for row in cases["cases"]:
        lines.append(f"- `{row['opaque_case_id']}`: {row['mechanism_summary']}")
    resource_by_model = {row["model_key"]: row for row in resources["models"]}
    lines.extend([
        "", "## Mechanism and resource guidance", "",
        "- Quality priority: start from the per-track metric table and paired intervals; do not infer a universal winner from point estimates alone.",
        "- Lightweight or throughput priority: compare encoding throughput, snapshot scale, CSR/index bytes, and paired quality intervals together. SPLADE-Tiny is the smallest frozen Batch-B mechanism but is local research-only.",
        "- Static-query/document-expansion priority: compare opensearch-doc-v2-mini, opensearch-doc-v3, and opensearch-multilingual against neural-query alternatives. The multilingual model's route is static lookup plus document expansion; these English-track results do not establish multilingual effectiveness.",
        "- Uncertain conclusions: treat any paired interval crossing zero, every slice with n<10, training-overlap uncertainty, and cross-track reversals as inconclusive.",
        "", "Encoding RSS values are encoding-observed measurements, not whole-cell peaks. No model is described as verified zero-shot.",
        "", "## License and publication boundaries", "",
        f"SPLADE-Tiny: {resource_by_model['splade-tiny']['license_boundary']}",
        "The package does not redistribute weights. The independent Validator must recheck the SPLADE-Tiny license boundary and the predeclaration supersession lineage.",
        "Public export, publication readiness, and leaderboard publication remain disabled.",
    ])
    return "\n".join(lines) + "\n"


def assert_tracked_privacy(files: Mapping[str, bytes], data_root: str | Path) -> dict[str, Any]:
    """Reject row-level fields, document IDs, or canonical source text in tracked output."""
    forbidden_keys = {"query_id", "document_id", "positive_document_ids", "hits", "rankings", "per_query", "source_text", "content", "text"}
    forbidden_path_fragments = ("results/", ".cache/", "snapshots/", "corpus.jsonl", "queries.jsonl", "qrels.jsonl")
    text = b"\n".join(files.values()).decode("utf-8")
    if any(fragment in text for fragment in forbidden_path_fragments):
        raise PackageError("Tracked output leaks a raw, cache, snapshot, or canonical-data path")

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if forbidden_keys & set(value):
                raise PackageError(f"Tracked output contains forbidden row-level fields: {sorted(forbidden_keys & set(value))}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for name, payload in files.items():
        if name.endswith(".json"):
            walk(json.loads(payload))
    document_ids = set()
    source_texts = []
    for track in TRACKS:
        data = load_materialized(data_root, track)
        document_ids.update(row["id"] for row in data.corpus)
        source_texts.extend(row["text"] for row in data.queries)
        source_texts.extend(row["content"] for row in data.corpus)
    values = set()

    def strings(value: Any) -> None:
        if isinstance(value, str):
            values.add(value)
        elif isinstance(value, dict):
            for key, child in value.items():
                values.add(str(key))
                strings(child)
        elif isinstance(value, list):
            for child in value:
                strings(child)

    for name, payload in files.items():
        if name.endswith(".json"):
            strings(json.loads(payload))
    if values & document_ids:
        raise PackageError("Tracked output contains a canonical document ID")
    if any(len(source) >= 24 and source in text for source in source_texts):
        raise PackageError("Tracked output contains canonical restricted source text")
    return {
        "status": "pass",
        "tracked_files_scanned": len(files),
        "forbidden_row_fields": sorted(forbidden_keys),
        "canonical_document_ids_found": 0,
        "canonical_source_texts_found": 0,
        "public_gate": "closed",
    }


def compute_outputs(
    *, cells: Mapping[tuple[str, str], Cell], data_root: str | Path,
    predeclaration: Mapping[str, Any], batch_a_inventory: Mapping[str, Any]
) -> tuple[dict[str, bytes], dict[str, Any]]:
    summary, pairwise = build_leaderboards(cells)
    slices = build_slices(cells, data_root, predeclaration)
    cases, local_mapping = build_cases(cells, data_root)
    resources = build_resources(cells, predeclaration, batch_a_inventory)
    values = {
        "summary.json": summary,
        "pairwise.json": pairwise,
        "slices.json": slices,
        "cases.json": cases,
        "resources.json": resources,
    }
    schemas = {
        "summary.json": SUMMARY_SCHEMA, "pairwise.json": PAIRWISE_SCHEMA,
        "slices.json": SLICES_SCHEMA, "cases.json": CASES_SCHEMA, "resources.json": RESOURCES_SCHEMA,
    }
    for name, value in values.items():
        _validate_schema(value, schemas[name])
    report = render_report(summary, pairwise, slices, cases, resources).encode("utf-8")
    files = {name: formatted_json_bytes(value) for name, value in values.items()}
    files["report.md"] = report
    privacy = assert_tracked_privacy(files, data_root)
    return files, {"local_mapping": local_mapping, "privacy": privacy}


def package_manifest(
    *, files: Mapping[str, bytes], contract: Mapping[str, Any], input_contract_sha256: str,
    local_mapping_sha256: str, privacy: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = {
        "schema_version": "bright-learned-sparse-main-manifest-v1",
        "package_version": PACKAGE_VERSION,
        "publication": PUBLICATION,
        "protocol": {**PACKAGE_PROTOCOL, "identity_sha256": canonical_sha256(PACKAGE_PROTOCOL)},
        "bindings": {
            "input_contract_sha256": input_contract_sha256,
            "active_predeclaration_sha256": contract["active_predeclaration_sha256"],
            "canonical_data_sha256": contract["canonical_data_sha256"],
            "batch_a_package_manifest_sha256": contract["batch_a_package_manifest_sha256"],
            "batch_a_protocol_sha256": contract["batch_a_protocol_sha256"],
            "batch_b_protocol_sha256": contract["batch_b_protocol_sha256"],
            "raw_manifest_sha256": contract["raw_manifest_sha256"],
            "batch_b_replay_sha256": contract["batch_b_replay_sha256"],
            "snapshot_evidence_sha256": contract["snapshot_evidence_sha256"],
            "local_mapping_sha256": local_mapping_sha256,
        },
        "tracked_files": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in sorted(files.items())
        },
        "privacy_scan": dict(privacy),
        "content_policy": "Aggregate statistics and opaque case identifiers only; no canonical IDs, text, rankings, per-query rows, raw data, cache content, or model snapshots.",
    }
    _validate_schema(manifest, MANIFEST_SCHEMA)
    return manifest


def write_package(
    *, package_root: str | Path, report_path: str | Path, local_mapping_path: str | Path,
    files: Mapping[str, bytes], manifest: Mapping[str, Any], local_mapping: Mapping[str, Any],
    expected_package_sha256: str | None,
) -> str:
    expected = _require_sha256(expected_package_sha256, "package manifest SHA256")
    manifest_payload = formatted_json_bytes(manifest)
    actual = hashlib.sha256(manifest_payload).hexdigest()
    if actual != expected:
        raise PackageError(f"Computed package identity {actual} does not match external expected identity")
    package = Path(package_root)
    report = Path(report_path)
    sidecar = package / "manifest.sha256"
    observed = package.exists() and any(package.iterdir())
    if observed or report.exists():
        if not (package / "manifest.json").is_file() or not sidecar.is_file() or not report.is_file():
            raise PackageError("Existing package is incomplete and will not be resumed or overwritten")
        if file_sha256(package / "manifest.json") != expected or sidecar.read_text(encoding="ascii").strip() != expected:
            raise PackageError("Existing package identity drifted and will not be self-signed")
        return expected
    package.mkdir(parents=True, exist_ok=False)
    for name, payload in files.items():
        target = report if name == "report.md" else package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (package / "manifest.json").write_bytes(manifest_payload)
    sidecar.write_text(expected + "\n", encoding="ascii")
    local_path = Path(local_mapping_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(formatted_json_bytes(local_mapping))
    return actual


def validate_complete_package(
    *, package_root: str | Path, report_path: str | Path, expected_package_sha256: str | None,
    expected_files: Mapping[str, bytes], expected_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _require_sha256(expected_package_sha256, "package manifest SHA256")
    package = Path(package_root)
    manifest_path = package / "manifest.json"
    sidecar = package / "manifest.sha256"
    if file_sha256(manifest_path) != expected or sidecar.read_text(encoding="ascii").strip() != expected:
        raise PackageError("Package manifest or sidecar does not match the external expected identity")
    manifest = _load_json(manifest_path)
    _validate_schema(manifest, MANIFEST_SCHEMA)
    if manifest != expected_manifest:
        raise PackageError("Package manifest cannot be rebuilt from authenticated raw inputs")
    expected_set = set(expected_files) - {"report.md"}
    actual_set = {
        path.relative_to(package).as_posix() for path in package.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }
    if actual_set != expected_set:
        raise PackageError("Tracked package file set is incomplete or contains unexpected files")
    for name, payload in expected_files.items():
        path = Path(report_path) if name == "report.md" else package / name
        if path.read_bytes() != payload:
            raise PackageError(f"Tracked package artifact drifted: {name}")
    return manifest
