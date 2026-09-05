"""Build the frozen S-009 cross-paradigm research candidate."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from mm_embed.benchmark import bright_learned_sparse_package as sparse_package
from mm_embed.benchmark import bright_multi_vector_package as multi_package
from mm_embed.benchmark.bright_cross_paradigm import (
    ARTIFACT_ROOT,
    METRICS,
    ROOT,
    STORY_ID,
    TRACKS,
    canonical_bytes,
    publication_policy,
    sha256_file,
    verify_freeze,
)
from mm_embed.benchmark.bright_cross_paradigm_run import (
    RESULT_ROOT,
    validate_formal_cell,
)
from mm_embed.benchmark.bright_multidomain_v02 import load_materialized
from mm_embed.benchmark.retrieval_v01 import (
    aggregate_metrics,
    bootstrap_confidence_intervals,
    paired_bootstrap_deltas,
    tokens,
)

PACKAGE_ROOT = ROOT / "benchmark/artifacts/bright-cross-paradigm-results-v0.1"
REPORT_PATH = ROOT / "benchmark/research/bright_cross_paradigm_20260905.md"
LOCAL_CASE_PATH = RESULT_ROOT / "case-mapping.json"
ACCEPTED_VERIFICATION_ROOT = RESULT_ROOT / "accepted-verification"
RESOURCE_PATH = RESULT_ROOT / "resources/resource-summary.json"
DATA_ROOT = ROOT / "data/bright-nontechnical-pilot-v0.2"
INPUT_CONTRACT = ROOT / "benchmark/contracts/bright-learned-sparse-main-v0.1-inputs.json"
INPUT_CONTRACT_SHA256 = "18ec21848358fd19a454ed2f7362e1eb6117d6c508ca6f30cc9980d2642d1dc6"
SPARSE_PACKAGE_SHA256 = "3accbed7b2a9961eedff20e601375a23c50167cfc7d8e77adbf91399aed91f47"
MULTI_PACKAGE_SHA256 = "fa80bbd1f8b711ef76ae27f606319cf9b856f7e9baf49f9643276413e2a822f8"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_826
TOP_K = 100
TFIDF_REPRESENTATION_SCOPE = "complete_document_and_query_representation_plus_idf_for_tfidf"
MINILM_REPRESENTATION_SCOPE = "complete_float32_dense_document_and_query_vectors_only"
FAMILY_ORDER = ("lexical", "dense", "learned_sparse", "multi_vector")
METHODS = (
    "bm25-unicode",
    "tfidf-word-sublinear",
    "bge-m3-dense",
    "all-minilm-l6-v2",
    "granite-30m-sparse",
    "opensearch-doc-v2-mini",
    "opensearch-doc-v3",
    "bge-m3",
    "splade-tiny",
    "opensearch-doc-v2-distill",
    "opensearch-multilingual",
    "colbert-v2",
    "answerai-colbert-small",
    "gte-modern-colbert",
)
SCHEMAS = {
    "summary.json": "bright-cross-paradigm-summary-v01.schema.json",
    "pairwise.json": "bright-cross-paradigm-pairwise-v01.schema.json",
    "slices.json": "bright-cross-paradigm-slices-v01.schema.json",
    "cases.json": "bright-cross-paradigm-cases-v01.schema.json",
    "lineage.json": "bright-cross-paradigm-lineage-v01.schema.json",
    "resources.json": "bright-cross-paradigm-resources-v01.schema.json",
    "descriptive-sensitivity.json": "bright-cross-paradigm-descriptive-sensitivity-v01.schema.json",
    "scenario-guidance.json": "bright-cross-paradigm-guidance-v01.schema.json",
}
CANDIDATE_ORDER = (
    "manifest.json",
    "summary.json",
    "pairwise.json",
    "slices.json",
    "cases.json",
    "lineage.json",
    "resources.json",
    "descriptive-sensitivity.json",
    "scenario-guidance.json",
    "benchmark/research/bright_cross_paradigm_20260905.md",
)
IMPLEMENTATION_CANDIDATES = (
    "schemas/bright-cross-paradigm-cases-v01.schema.json",
    "schemas/bright-cross-paradigm-guidance-v01.schema.json",
    "schemas/bright-cross-paradigm-lineage-v01.schema.json",
    "schemas/bright-cross-paradigm-manifest-v01.schema.json",
    "schemas/bright-cross-paradigm-pairwise-v01.schema.json",
    "schemas/bright-cross-paradigm-resources-v01.schema.json",
    "schemas/bright-cross-paradigm-descriptive-sensitivity-v01.schema.json",
    "schemas/bright-cross-paradigm-slices-v01.schema.json",
    "schemas/bright-cross-paradigm-summary-v01.schema.json",
    "scripts/bright_cross_paradigm.py",
    "scripts/bright_cross_paradigm_package.py",
    "scripts/bright_cross_paradigm_resources.py",
    "scripts/bright_cross_paradigm_run.py",
    "scripts/build_bright_cross_paradigm_candidate.py",
    "scripts/check_bright_cross_paradigm_mutations.py",
    "src/mm_embed/benchmark/bright_cross_paradigm.py",
    "src/mm_embed/benchmark/bright_cross_paradigm_package.py",
    "src/mm_embed/benchmark/bright_cross_paradigm_resources.py",
    "src/mm_embed/benchmark/bright_cross_paradigm_run.py",
    "tests/test_bright_cross_paradigm.py",
    "tests/test_bright_cross_paradigm_package.py",
)


class CrossParadigmPackageError(ValueError):
    """Raised when a frozen input or candidate invariant fails."""


@dataclass(frozen=True)
class Cell:
    """Normalized restricted cell used only during local aggregation."""

    method: str
    family: str
    track: str
    identity: str
    source: str
    quality_origin: str
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]
    rankings: dict[str, list[dict[str, Any]]]
    recomputation: str
    contextual_anchor: dict[str, Any] | None = None


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    identity = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return identity


def _inventory() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = _json(ARTIFACT_ROOT / "inventory.json")
    rows = {row["key"]: row for row in value["methods"]}
    if tuple(rows) != METHODS:
        raise CrossParadigmPackageError("Frozen method inventory order drifted")
    return value, rows


def _hits_from_tuples(rankings: Mapping[str, Sequence[tuple[str, float]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        query_id: [
            {"document_id": document_id, "rank": rank, "score": float(score)}
            for rank, (document_id, score) in enumerate(hits, 1)
        ]
        for query_id, hits in rankings.items()
    }


def _verify_replay_evidence() -> None:
    replay = _json(ACCEPTED_VERIFICATION_ROOT / "learned-sparse-replay.json")
    if (
        replay.get("status") != "pass"
        or replay.get("cpu_only") is not True
        or replay.get("model_loaded") is not False
        or replay.get("package_manifest_sha256") != SPARSE_PACKAGE_SHA256
        or replay.get("input_contract_sha256") != INPUT_CONTRACT_SHA256
    ):
        raise CrossParadigmPackageError("Learned-sparse accepted replay evidence is missing or invalid")


def collect_cells() -> dict[tuple[str, str], Cell]:
    """Authenticate and normalize exactly the frozen 28 quality cells."""
    verify_freeze()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CrossParadigmPackageError("Candidate aggregation requires CUDA_VISIBLE_DEVICES='' ")
    _, inventory = _inventory()
    cells: dict[tuple[str, str], Cell] = {}

    for method in ("bm25-unicode", "bge-m3-dense"):
        for track in TRACKS:
            result = validate_formal_cell(method, track)
            cell_path = RESULT_ROOT / "cells" / f"{method}-{track}" / "cell.json"
            rankings = {row["query_id"]: row["hits"] for row in _json(cell_path.with_name("rankings.json"))}
            per_query = {
                row["query_id"]: row["metrics"] for row in _json(cell_path.with_name("per_query_metrics.json"))
            }
            anchor = result.get("accepted_anchor_comparison")
            if not anchor or anchor.get("accepted_cell_sha256") != inventory[method]["accepted_cell_sha256"][track]:
                raise CrossParadigmPackageError(f"S-005 accepted identity drifted: {method}:{track}")
            cells[(method, track)] = Cell(
                method,
                inventory[method]["family"],
                track,
                sha256_file(cell_path),
                STORY_ID,
                "s005_method_representation_backed_return1_rerun",
                dict(result["quality"]["metrics"]),
                per_query,
                rankings,
                "exact_saved_representation_or_scores_top100_and_metrics_recomputed",
                dict(anchor),
            )

    contract = sparse_package.load_input_contract(INPUT_CONTRACT, INPUT_CONTRACT_SHA256)
    sparse_cells = sparse_package.collect_authenticated_cells(
        contract=contract,
        repo_root=ROOT,
        data_root=DATA_ROOT,
        batch_a_raw_root=ROOT / "results/bright-learned-sparse-batch-a",
        batch_b_raw_root=ROOT / "results/bright-learned-sparse-batch-b",
        batch_b_replay_root=ROOT / "results/bright-learned-sparse-batch-b-replay",
        snapshot_evidence_root=ROOT / "results/learned-sparse-batch-b/active-v2",
    )
    _verify_replay_evidence()
    for (method, track), source in sparse_cells.items():
        if source.manifest_sha256 != inventory[method]["accepted_cell_sha256"][track]:
            raise CrossParadigmPackageError(f"S-008 accepted identity drifted: {method}:{track}")
        cells[(method, track)] = Cell(
            method,
            inventory[method]["family"],
            track,
            source.manifest_sha256,
            "S-008",
            "accepted_read_only",
            dict(source.manifest["metrics"]),
            dict(source.per_query),
            dict(source.rankings),
            "exact_csr_top100_and_metrics_recomputed",
        )

    multi_cells, _, _ = multi_package.collect_cells(
        cell_root=ROOT / "results/bright-multi-vector-v0.2/cells",
        predeclaration_path=ROOT / "benchmark/artifacts/bright-multi-vector-v0.2/predeclaration.json",
        gate_summary_path=ROOT / "benchmark/artifacts/bright-multi-vector-v0.2/gate-summary.json",
        data_root=DATA_ROOT,
    )
    if sha256_file(ROOT / "benchmark/artifacts/bright-multi-vector-results-v0.2/manifest.json") != MULTI_PACKAGE_SHA256:
        raise CrossParadigmPackageError("S-012 accepted package identity drifted")
    for (method, track), source in multi_cells.items():
        if source.identity != inventory[method]["accepted_cell_sha256"][track]:
            raise CrossParadigmPackageError(f"S-012 accepted identity drifted: {method}:{track}")
        replay_path = ACCEPTED_VERIFICATION_ROOT / "multi-vector" / f"{method}-{track}.rankings.json"
        if _json(replay_path) != [{"query_id": query_id, "hits": hits} for query_id, hits in source.rankings.items()]:
            raise CrossParadigmPackageError(f"S-012 exact top-100 replay drifted: {method}:{track}")
        cells[(method, track)] = Cell(
            method,
            inventory[method]["family"],
            track,
            source.identity,
            "S-012",
            "accepted_read_only",
            dict(source.value["quality"]["metrics"]),
            dict(source.per_query),
            dict(source.rankings),
            "exact_maxsim_top100_and_metrics_recomputed",
        )

    for method in ("tfidf-word-sublinear", "all-minilm-l6-v2"):
        for track in TRACKS:
            result = validate_formal_cell(method, track)
            cell_path = RESULT_ROOT / "cells" / f"{method}-{track}" / "cell.json"
            rankings = {row["query_id"]: row["hits"] for row in _json(cell_path.with_name("rankings.json"))}
            per_query = {
                row["query_id"]: row["metrics"] for row in _json(cell_path.with_name("per_query_metrics.json"))
            }
            cells[(method, track)] = Cell(
                method,
                inventory[method]["family"],
                track,
                sha256_file(cell_path),
                STORY_ID,
                "new_frozen_formal_run",
                dict(result["quality"]["metrics"]),
                per_query,
                rankings,
                "exact_saved_representation_top100_and_metrics_recomputed",
            )

    expected = {(method, track) for method in METHODS for track in TRACKS}
    if set(cells) != expected or len(cells) != 28:
        raise CrossParadigmPackageError("Candidate requires exactly the frozen 14x2 matrix")
    for track in TRACKS:
        query_sets = {frozenset(cells[(method, track)].per_query) for method in METHODS}
        if len(query_sets) != 1:
            raise CrossParadigmPackageError(f"Per-query alignment drifted: {track}")
        for method in METHODS:
            cell = cells[(method, track)]
            if aggregate_metrics(cell.per_query) != cell.metrics:
                raise CrossParadigmPackageError(f"Core metric replay drifted: {method}:{track}")
    return cells


def _paired(left: Cell, right: Cell, query_ids: Sequence[str] | None = None) -> dict[str, Any]:
    ids = sorted(left.per_query) if query_ids is None else sorted(query_ids)
    left_values = {query_id: left.per_query[query_id] for query_id in ids}
    right_values = {query_id: right.per_query[query_id] for query_id in ids}
    result = paired_bootstrap_deltas(left_values, right_values, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    return {
        metric: {
            **values,
            "verdict": (
                "left_wins" if values["low"] > 0 else "right_wins" if values["high"] < 0 else "tie_or_uncertain"
            ),
        }
        for metric, values in result.items()
    }


def build_summary(cells: Mapping[tuple[str, str], Cell]) -> dict[str, Any]:
    tracks: dict[str, Any] = {}
    for track in TRACKS:
        rows = []
        for method in METHODS:
            cell = cells[(method, track)]
            rows.append(
                {
                    "method_key": method,
                    "family": cell.family,
                    "quality_origin": cell.quality_origin,
                    "cell_sha256": cell.identity,
                    "metrics": cell.metrics,
                    "confidence_intervals": bootstrap_confidence_intervals(
                        cell.per_query, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
                    ),
                }
            )
        tracks[track] = {
            "queries": len(cells[(METHODS[0], track)].per_query),
            "methods": rows,
            "metric_orders": {
                metric: sorted(METHODS, key=lambda method: (-cells[(method, track)].metrics[metric], method))
                for metric in METRICS
            },
        }
    return {
        "schema_version": "bright-cross-paradigm-summary-v1",
        "story_id": STORY_ID,
        "publication": publication_policy(),
        "matrix": {"methods": 14, "tracks": 2, "cells": 28, "complete": True},
        "track_aggregation": "independent_no_cross_track_micro_average_or_universal_winner",
        "tracks": tracks,
    }


def build_pairwise(cells: Mapping[tuple[str, str], Cell]) -> dict[str, Any]:
    tracks = {}
    for track in TRACKS:
        rows = [
            {
                "left": left,
                "right": right,
                "n": len(cells[(left, track)].per_query),
                "metrics": _paired(cells[(left, track)], cells[(right, track)]),
            }
            for left, right in itertools.combinations(sorted(METHODS), 2)
        ]
        if len(rows) != 91:
            raise CrossParadigmPackageError("Each track requires exactly 91 method pairs")
        tracks[track] = rows
    return {
        "schema_version": "bright-cross-paradigm-pairwise-v1",
        "story_id": STORY_ID,
        "publication": publication_policy(),
        "method": "paired_percentile_bootstrap_over_matched_queries",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "tracks": tracks,
    }


def _slice_memberships(data: Any, frozen: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    documents = {row["id"]: row["content"] for row in data.corpus}
    output = {}
    for query in data.queries:
        query_tokens = tokens(query["text"].strip())
        unique_query = set(query_tokens)
        positive_tokens = [tokens(documents[document_id]) for document_id in sorted(data.qrels[query["id"]])]
        overlap = max(
            (len(unique_query & set(values)) / len(unique_query) if unique_query else 0.0) for values in positive_tokens
        )
        positive_length = float(np.median([len(values) for values in positive_tokens]))
        density = len(positive_tokens)
        output[query["id"]] = {
            "query_length": "<=64" if len(query_tokens) <= 64 else "65-128" if len(query_tokens) <= 128 else ">128",
            "positive_lexical_overlap": (
                "0"
                if overlap == 0
                else "(0,0.25]"
                if overlap <= 0.25
                else "(0.25,0.50]"
                if overlap <= 0.50
                else ">0.50"
            ),
            "positive_passage_length": (
                "<=128" if positive_length <= 128 else "129-512" if positive_length <= 512 else ">512"
            ),
            "qrel_density": "1" if density == 1 else "2-4" if density <= 4 else ">=5",
        }
    allowed = {row["key"]: set(row["bins"]) for row in frozen}
    for values in output.values():
        if set(values) != set(allowed) or any(values[key] not in allowed[key] for key in allowed):
            raise CrossParadigmPackageError("Derived slice membership drifted from the freeze")
    return output


def build_slices(cells: Mapping[tuple[str, str], Cell], predecl: Mapping[str, Any]) -> dict[str, Any]:
    tracks = {}
    for track in TRACKS:
        membership = _slice_memberships(load_materialized(DATA_ROOT, track), predecl["slices"])
        slice_rows = []
        for definition in predecl["slices"]:
            bins = []
            for name in definition["bins"]:
                ids = sorted(query_id for query_id, row in membership.items() if row[definition["key"]] == name)
                methods = [
                    {
                        "method_key": method,
                        "n": len(ids),
                        "metrics": (
                            aggregate_metrics(
                                {query_id: cells[(method, track)].per_query[query_id] for query_id in ids}
                            )
                            if ids
                            else None
                        ),
                    }
                    for method in METHODS
                ]
                pairs = []
                if len(ids) >= 10:
                    pairs = [
                        {
                            "left": left,
                            "right": right,
                            "n": len(ids),
                            "metrics": _paired(cells[(left, track)], cells[(right, track)], ids),
                        }
                        for left, right in itertools.combinations(sorted(METHODS), 2)
                    ]
                    if len(pairs) != 91:
                        raise CrossParadigmPackageError("Eligible slice requires all 91 pairs")
                bins.append(
                    {
                        "bin": name,
                        "n": len(ids),
                        "membership_sha256": _canonical_sha(ids),
                        "methods": methods,
                        "pairwise": pairs,
                        "inference": (
                            "paired_ci_allowed"
                            if len(ids) >= 10
                            else "point_estimates_only_no_win_claim"
                            if ids
                            else "not_applicable"
                        ),
                    }
                )
            slice_rows.append({**definition, "bins": bins})
        tracks[track] = slice_rows
    return {
        "schema_version": "bright-cross-paradigm-slices-v1",
        "story_id": STORY_ID,
        "publication": publication_policy(),
        "small_slice_policy": predecl["statistics"]["small_slice_policy"],
        "tracks": tracks,
    }


def _best_positive_rank(hits: Sequence[Mapping[str, Any]], positives: set[str]) -> int:
    return min(
        (int(hit.get("rank", ordinal)) for ordinal, hit in enumerate(hits, 1) if hit["document_id"] in positives),
        default=TOP_K + 1,
    )


def build_cases(cells: Mapping[tuple[str, str], Cell]) -> tuple[dict[str, Any], dict[str, Any]]:
    tracked = []
    restricted = []
    for track in TRACKS:
        data = load_materialized(DATA_ROOT, track)
        candidates = []
        for query_id in sorted(cells[(METHODS[0], track)].per_query):
            positives = set(data.qrels[query_id])
            ranks = {
                method: _best_positive_rank(cells[(method, track)].rankings[query_id], positives) for method in METHODS
            }
            best_ndcg = max(cells[(method, track)].per_query[query_id]["ndcg@10"] for method in METHODS)
            query_hash = hashlib.sha256(f"{STORY_ID}:{track}:{query_id}".encode()).hexdigest()
            candidates.append(
                {
                    "query_id": query_id,
                    "query_hash": query_hash,
                    "positive_ids": sorted(positives),
                    "ranks": ranks,
                    "rank_spread": max(ranks.values()) - min(ranks.values()),
                    "best_method_ndcg@10": best_ndcg,
                    "per_method": {method: cells[(method, track)].per_query[query_id] for method in METHODS},
                }
            )
        disagreements = sorted(candidates, key=lambda row: (-row["rank_spread"], row["query_hash"]))[:6]
        selected = {row["query_id"] for row in disagreements}
        failures = sorted(
            (row for row in candidates if row["query_id"] not in selected),
            key=lambda row: (row["best_method_ndcg@10"], row["query_hash"]),
        )[:6]
        for category, rows in (("disagreement", disagreements), ("failure", failures)):
            for row in rows:
                case_hash = hashlib.sha256(f"{STORY_ID}:{track}:{category}:{row['query_id']}".encode()).hexdigest()
                evidence = {
                    "case_hash": case_hash,
                    "track": track,
                    "category": category,
                    "query_id": row["query_id"],
                    "positive_document_ids": row["positive_ids"],
                    "rank_spread": row["rank_spread"],
                    "best_method_ndcg@10": row["best_method_ndcg@10"],
                    "best_positive_rank": row["ranks"],
                    "per_method": row["per_method"],
                    "source_text_included": False,
                }
                restricted.append(evidence)
                tracked.append(
                    {
                        "case_hash": case_hash,
                        "track": track,
                        "category": category,
                        "selection_rule": (
                            "largest_best_positive_rank_spread_then_query_hash"
                            if category == "disagreement"
                            else "lowest_best_method_ndcg_at_10_then_query_hash"
                        ),
                        "rank_spread": row["rank_spread"],
                        "best_method_ndcg@10": row["best_method_ndcg@10"],
                        "local_evidence_sha256": _canonical_sha(evidence),
                        "restricted_content_included": False,
                    }
                )
    if len(tracked) != 24 or len({row["case_hash"] for row in tracked}) != 24:
        raise CrossParadigmPackageError("Frozen cases require 24 unique selections")
    local = {
        "schema_version": "bright-cross-paradigm-local-cases-v1",
        "classification": "research_only_no_publish",
        "source_text_included": False,
        "cases": restricted,
    }
    return (
        {
            "schema_version": "bright-cross-paradigm-cases-v1",
            "story_id": STORY_ID,
            "publication": publication_policy(),
            "counts": {"total": 24, "per_track_disagreement": 6, "per_track_failure": 6},
            "local_mapping_sha256": _canonical_sha(local),
            "cases": tracked,
        },
        local,
    )


def build_lineage(cells: Mapping[tuple[str, str], Cell], inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "bright-cross-paradigm-lineage-v1",
        "story_id": STORY_ID,
        "publication": publication_policy(),
        "canonical_data_manifest_sha256": _json(ARTIFACT_ROOT / "predeclaration.json")["bindings"]["canonical_data"][
            "manifest_sha256"
        ],
        "accepted_package_sha256": {
            "S-005": "6b48e45d4f8332a5386729898845c1a62b02b2fd1481b4665d501947a082fd86",
            "S-008": SPARSE_PACKAGE_SHA256,
            "S-012": MULTI_PACKAGE_SHA256,
        },
        "contextual_anchors": [
            {
                "cell_id": f"{method}:{track}",
                **cells[(method, track)].contextual_anchor,
            }
            for method in ("bm25-unicode", "bge-m3-dense")
            for track in TRACKS
            if cells[(method, track)].contextual_anchor is not None
        ],
        "cells": [
            {
                "cell_id": f"{method}:{track}",
                "method_key": method,
                "family": cells[(method, track)].family,
                "track": track,
                "quality_origin": cells[(method, track)].quality_origin,
                "source_story": cells[(method, track)].source,
                "cell_sha256": cells[(method, track)].identity,
                "revision": inventory[method].get("revision"),
                "recomputation": cells[(method, track)].recomputation,
                "quality_comparability": "direct_same_input_candidate_retrieval_unit_top100_metrics",
                "resource_comparability": "unified_rerun_only",
                "system_error": "none_exact_search",
            }
            for method in METHODS
            for track in TRACKS
        ],
        "fairness": {
            "quality": "direct_with_method_intrinsic_representation_and_scoring_differences_disclosed",
            "latency": "direct_only_within_successful_unified_measurements",
            "representation_bytes": "stratified_by_paradigm_no_synthetic_normalization",
            "historical_resources": "contextual_anchor_only",
            "approximation_error": "not_applicable_exact_full_corpus",
        },
    }


def build_resources() -> dict[str, Any]:
    value = _json(RESOURCE_PATH)
    expected = {f"{method}:{track}" for method in METHODS for track in TRACKS}
    rows = value.get("cells", [])
    if {row.get("cell_id") for row in rows} != expected or len(rows) != 28:
        raise CrossParadigmPackageError("Unified resources require exactly 28 method-track cells")
    for row in rows:
        if row.get("status") not in {"pass", "failed_closed"}:
            raise CrossParadigmPackageError("Unified resource cell has an invalid terminal status")
        if row.get("search", {}).get("trials_completed") not in {0, 3}:
            raise CrossParadigmPackageError("Unified resource trial count drifted")
    passed = sum(row["status"] == "pass" for row in rows)
    return {
        "schema_version": "bright-cross-paradigm-resources-v1",
        "story_id": STORY_ID,
        "publication": publication_policy(),
        "protocol": _json(ARTIFACT_ROOT / "predeclaration.json")["resource_protocol"],
        "coverage": {"planned": 28, "passed": passed, "failed_closed": 28 - passed, "complete": True},
        "comparability": {
            "timing": "within_unified_rerun_only",
            "representation_bytes": "stratified_by_paradigm",
            "historical": "contextual_only",
        },
        "cells": rows,
    }


def build_sensitivity(slices: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    observations = []
    for track, slice_rows in slices["tracks"].items():
        for slice_row in slice_rows:
            bins = {row["bin"]: row for row in slice_row["bins"]}
            nonempty = [row for row in slice_row["bins"] if row["n"]]
            for method in METHODS:
                values = []
                for row in nonempty:
                    method_row = next(item for item in row["methods"] if item["method_key"] == method)
                    values.append({"bin": row["bin"], "n": row["n"], "ndcg@10": method_row["metrics"]["ndcg@10"]})
                observations.append(
                    {
                        "track": track,
                        "slice": slice_row["key"],
                        "method_key": method,
                        "bins": values,
                        "range_ndcg@10": max(row["ndcg@10"] for row in values) - min(row["ndcg@10"] for row in values),
                        "all_frozen_bins_present": len(bins) == len(slice_row["bins"]),
                    }
                )
    return {
        "schema_version": "bright-cross-paradigm-descriptive-sensitivity-v1",
        "story_id": STORY_ID,
        "publication": publication_policy(),
        "analysis_type": "descriptive_slice_and_representation_sensitivity_no_protocol_deviation",
        "method_intrinsic_caps": [
            {
                "method_key": method,
                "family": inventory[method]["family"],
                "max_length": inventory[method].get("formal_max_length", inventory[method].get("max_length")),
                "windowed": inventory[method]["family"] == "multi_vector",
            }
            for method in METHODS
        ],
        "observations": observations,
        "limitations": [
            "Slice differences are descriptive and not causal estimates.",
            "No protocol, cap, seed, slice, candidate pool, or stopping condition was changed after scores became visible.",
            "The accepted S-005 MiniLM diagnostic is not relabeled or included as a formal cell.",
        ],
    }


def _comparison_for_recommendation(
    pairs: Sequence[Mapping[str, Any]], recommended: str, alternative: str
) -> dict[str, Any]:
    row = next(item for item in pairs if {item["left"], item["right"]} == {recommended, alternative})
    metric = dict(row["metrics"]["ndcg@10"])
    if row["left"] != recommended:
        metric["mean"] = -metric["mean"]
        metric["low"], metric["high"] = -metric["high"], -metric["low"]
        if metric["verdict"] == "left_wins":
            metric["verdict"] = "alternative_wins"
        elif metric["verdict"] == "right_wins":
            metric["verdict"] = "recommended_wins"
    else:
        metric["verdict"] = {
            "left_wins": "recommended_wins",
            "right_wins": "alternative_wins",
            "tie_or_uncertain": "tie_or_uncertain",
        }[metric["verdict"]]
    return {"alternative": alternative, "metric": "ndcg@10", **metric}


def _resource_tradeoffs(ordered: Sequence[str], track: str, resources: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_cell = {row["cell_id"]: row for row in resources["cells"]}
    output = []
    for method in ordered:
        row = by_cell[f"{method}:{track}"]
        if row["status"] != "pass":
            continue
        output.append(
            {
                "method_key": method,
                "family": row["family"],
                "median_queries_per_second": row["search"]["median_queries_per_second"],
                "document_encoding_seconds": row["document_encoding"]["seconds"],
                "query_encoding_seconds": row["query_encoding"]["seconds"],
                "representation_bytes": row["representation"]["bytes"],
                "representation_stratum": row["representation"]["paradigm"],
            }
        )
        if len(output) == 3:
            break
    return output


def build_guidance(
    summary: Mapping[str, Any],
    pairwise: Mapping[str, Any],
    slices: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    scenarios = []
    for track in TRACKS:
        overall_order = summary["tracks"][track]["metric_orders"]["ndcg@10"]
        scenarios.append(
            {
                "track": track,
                "scenario": "overall_exact_quality",
                "metric": "ndcg@10",
                "n": summary["tracks"][track]["queries"],
                "recommended_method": overall_order[0],
                "ordered_methods": overall_order,
                "paired_evidence": _comparison_for_recommendation(
                    pairwise["tracks"][track], overall_order[0], overall_order[1]
                ),
                "successful_resource_tradeoffs": _resource_tradeoffs(overall_order, track, resources),
                "instruction": "Prefer the point leader only when the paired verdict is decisive; otherwise choose among tied methods using successful within-track resource measurements.",
            }
        )
        for slice_key, bin_name in (
            ("positive_lexical_overlap", "(0,0.25]"),
            ("positive_passage_length", ">512"),
            ("qrel_density", ">=5"),
        ):
            slice_row = next(row for row in slices["tracks"][track] if row["key"] == slice_key)
            bin_row = next(row for row in slice_row["bins"] if row["bin"] == bin_name)
            ordered = sorted(
                (row for row in bin_row["methods"] if row["metrics"] is not None),
                key=lambda row: (-row["metrics"]["ndcg@10"], row["method_key"]),
            )
            if not ordered or bin_row["n"] < 10 or not bin_row["pairwise"]:
                continue
            ordered_methods = [row["method_key"] for row in ordered]
            scenarios.append(
                {
                    "track": track,
                    "scenario": f"{slice_key}:{bin_name}",
                    "metric": "ndcg@10",
                    "n": bin_row["n"],
                    "recommended_method": ordered_methods[0],
                    "ordered_methods": ordered_methods,
                    "paired_evidence": _comparison_for_recommendation(
                        bin_row["pairwise"], ordered_methods[0], ordered_methods[1]
                    ),
                    "successful_resource_tradeoffs": _resource_tradeoffs(ordered_methods, track, resources),
                    "instruction": "Use the paired slice verdict for the top two point estimates, then apply successful unified resource evidence as a separate operational filter.",
                }
            )
    if not scenarios or any(not row["ordered_methods"] for row in scenarios):
        raise CrossParadigmPackageError("Guidance cannot publish an empty recommendation scenario")
    return {
        "schema_version": "bright-cross-paradigm-guidance-v1",
        "story_id": STORY_ID,
        "publication": publication_policy(),
        "universal_winner_claim": False,
        "resource_coverage": resources["coverage"],
        "scenarios": scenarios,
        "selection_notes": [
            "Keep economics and psychology separate.",
            "Use quality, latency, and representation storage as separate decision dimensions.",
            "Do not compare representation bytes across paradigms without the supplied stratification.",
            "Treat failed-closed resource cells as unavailable, not as zero cost.",
            "A tie_or_uncertain verdict means the resource tradeoff may decide; it is not evidence of equivalence.",
        ],
    }


def _report(files: Mapping[str, Any]) -> str:
    summary = files["summary.json"]
    pairwise = files["pairwise.json"]
    resources = files["resources.json"]
    lineage = files["lineage.json"]
    guidance = files["scenario-guidance.json"]
    lines = [
        "# BRIGHT Cross-Paradigm Exact Retrieval Study",
        "",
        "Status: research-only candidate. Publication, leaderboard, and export gates remain closed.",
        "",
        "## Scope and chronology",
        "",
        (
            "This study compares 14 frozen retrieval methods on the economics and psychology BRIGHT pilot tracks. "
            "It contains 20 accepted read-only quality cells and eight S-009 exact full-corpus cells: the four "
            "already-valid new baselines plus four representation/score-backed reruns of the frozen S-005 methods. "
            "The original S-005 cells remain authenticated contextual anchors rather than direct formal cells. The disclosed "
            "pre-freeze synthetic TF-IDF unit-test readiness event is retained in the immutable chronology; it did "
            "not expose formal scores. No inventory, identity, protocol, seed, slice, resource plan, or stopping "
            "condition was changed after new formal scores became visible."
        ),
        "",
        "## Quality protocol",
        "",
        (
            "Every quality cell uses the same canonical 7,500-passage candidate pool per track, canonical passage "
            "retrieval unit, top-100 depth, five core metrics, and deterministic score/ID tie policy. Method-native "
            "tokenization, context limits, sparse/dense/MaxSim scoring, and multi-vector window aggregation are the "
            "intrinsic mechanisms under test. All searches are exact; approximation error is therefore separated "
            "from method quality as not applicable."
        ),
        "",
        "## Aggregate results",
        "",
    ]
    for track in TRACKS:
        lines.extend(
            [
                f"### {track.title()}",
                "",
                "| Method | Family | nDCG@10 | MAP@100 | MRR@10 | R@10 | R@100 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["tracks"][track]["methods"]:
            metric = row["metrics"]
            lines.append(
                f"| {row['method_key']} | {row['family']} | {metric['ndcg@10']:.4f} | "
                f"{metric['map@100']:.4f} | {metric['mrr@10']:.4f} | {metric['recall@10']:.4f} | "
                f"{metric['recall@100']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## S-005 contextual-anchor deltas",
            "",
            "The accepted S-005 identities remain unchanged and provide chronology and contextual source evidence. The formal matrix uses the new score/representation-backed reruns below.",
            "",
            "| Cell | Exact top-100 order queries | Mismatches | Minimum overlap | nDCG@10 delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for anchor in lineage["contextual_anchors"]:
        lines.append(
            f"| {anchor['cell_id']} | {anchor['exact_order_queries']}/{anchor['queries']} | "
            f"{anchor['mismatched_order_queries']} | {anchor['top100_overlap_min']}/100 | "
            f"{anchor['metric_delta_rerun_minus_accepted']['ndcg@10']:+.8f} |"
        )
    lines.append("")
    decisive = {
        track: sum(
            any(metric["verdict"] != "tie_or_uncertain" for metric in row["metrics"].values())
            for row in pairwise["tracks"][track]
        )
        for track in TRACKS
    }
    lines.extend(
        [
            "## Uncertainty, slices, and cases",
            "",
            (
                f"All 91 within-track method pairs were bootstrapped with {BOOTSTRAP_SAMPLES:,} matched-query "
                f"resamples at seed {BOOTSTRAP_SEED}. At least one metric was decisive for "
                f"{decisive['economics']} economics pairs and {decisive['psychology']} psychology pairs. "
                "All frozen slice bins were retained; bins with fewer than ten queries contain point estimates only. "
                "The case artifact contains six largest-rank-spread disagreements and six lowest-best-method failures "
                "per track. Canonical IDs and row-level evidence remain only in the restricted local mapping."
            ),
            "",
            "## Resource interpretation",
            "",
            (
                f"The unified resource matrix has {resources['coverage']['passed']} successful and "
                f"{resources['coverage']['failed_closed']} failed-closed cells out of 28. Timing comparisons are valid "
                "only within successful unified runs. Representation storage is stratified by paradigm and is not "
                "normalized into a synthetic common unit. TF-IDF scope includes document/query representations plus "
                "IDF, while MiniLM scope is dense document/query vectors only. Historical measurements remain "
                "contextual anchors."
            ),
            "",
            "## Scenario selection",
            "",
            (
                "Choose the track and scenario first, inspect the paired nDCG@10 verdict for the point leader versus "
                "the runner-up, and then use only successful unified resource cells as a separate operational filter. "
                "A tie-or-uncertain interval permits a resource-driven choice but does not prove equivalence."
            ),
            "",
            "| Track | Scenario | n | Point leader | Runner-up verdict | Successful resource options |",
            "|---|---|---:|---|---|---|",
            *[
                f"| {row['track']} | {row['scenario']} | {row['n']} | {row['recommended_method']} | "
                f"{row['paired_evidence']['verdict']} vs {row['paired_evidence']['alternative']} | "
                f"{', '.join(item['method_key'] for item in row['successful_resource_tradeoffs'])} |"
                for row in guidance["scenarios"]
            ],
            "",
            "## Descriptive slice and representation sensitivity",
            "",
            (
                "The sensitivity artifact is descriptive: it contrasts frozen slice outcomes and discloses intrinsic "
                "representation caps. It does not vary the frozen protocol and therefore is not protocol-deviation evidence."
            ),
            "",
            "## Limitations",
            "",
            (
                "Unjudged passages are not confirmed negatives. Two tracks do not establish a universal winner. "
                "Intrinsic context limits and training overlap boundaries remain potential explanatory factors. "
                "S-005 accepted cells remain contextual anchors; their replacement formal reruns provide score- or "
                "representation-level replay, and small floating-point order deltas are explicitly quantified. Slice "
                "and representation sensitivity is descriptive, not causal. Failed-closed resource cells remain unavailable "
                "for operational comparison. This candidate is not approved for publication or export."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _validate_schema(name: str, value: Any) -> None:
    schema = _json(ROOT / "schemas" / SCHEMAS[name])
    Draft202012Validator(schema).validate(value)


def validate_machine_contracts(files: Mapping[str, Any]) -> None:
    """Enforce cross-file invariants that JSON Schema cannot express succinctly."""
    expected_cells = {f"{method}:{track}" for method in METHODS for track in TRACKS}
    expected_pairs = set(itertools.combinations(sorted(METHODS), 2))
    expected_metrics = set(METRICS)
    for name, value in files.items():
        if value.get("publication") != publication_policy():
            raise CrossParadigmPackageError(f"Publication policy drifted: {name}")
    summary = files["summary.json"]
    if summary.get("matrix") != {"methods": 14, "tracks": 2, "cells": 28, "complete": True}:
        raise CrossParadigmPackageError("Summary matrix count drifted")
    if set(summary.get("tracks", {})) != set(TRACKS):
        raise CrossParadigmPackageError("Summary tracks drifted")
    for track, query_count in (("economics", 103), ("psychology", 101)):
        section = summary["tracks"][track]
        rows = section.get("methods", [])
        if section.get("queries") != query_count or [row.get("method_key") for row in rows] != list(METHODS):
            raise CrossParadigmPackageError(f"Summary method coverage drifted: {track}")
        for row in rows:
            if (
                set(row.get("metrics", {})) != expected_metrics
                or set(row.get("confidence_intervals", {})) != expected_metrics
            ):
                raise CrossParadigmPackageError(f"Summary metric set drifted: {track}")
        orders = section.get("metric_orders", {})
        if set(orders) != expected_metrics or any(
            len(order) != 14 or set(order) != set(METHODS) for order in orders.values()
        ):
            raise CrossParadigmPackageError(f"Summary metric orders drifted: {track}")
    pairwise = files["pairwise.json"]
    for track, query_count in (("economics", 103), ("psychology", 101)):
        rows = pairwise.get("tracks", {}).get(track, [])
        if len(rows) != 91 or {(row.get("left"), row.get("right")) for row in rows} != expected_pairs:
            raise CrossParadigmPackageError(f"Pairwise coverage drifted: {track}")
        if any(row.get("n") != query_count or set(row.get("metrics", {})) != expected_metrics for row in rows):
            raise CrossParadigmPackageError(f"Pairwise metric contract drifted: {track}")
    lineage = files["lineage.json"]
    if len(lineage.get("cells", [])) != 28 or {row.get("cell_id") for row in lineage["cells"]} != expected_cells:
        raise CrossParadigmPackageError("Lineage cell coverage drifted")
    anchors = lineage.get("contextual_anchors", [])
    if len(anchors) != 4 or {row.get("cell_id") for row in anchors} != {
        f"{method}:{track}" for method in ("bm25-unicode", "bge-m3-dense") for track in TRACKS
    }:
        raise CrossParadigmPackageError("S-005 contextual anchor coverage drifted")
    resources = files["resources.json"]
    rows = resources.get("cells", [])
    if len(rows) != 28 or {row.get("cell_id") for row in rows} != expected_cells:
        raise CrossParadigmPackageError("Resource cell coverage drifted")
    for row in rows:
        status = row.get("status")
        if status == "pass":
            if (
                row.get("failure") is not None
                or not isinstance(row.get("document_encoding"), Mapping)
                or not isinstance(row.get("query_encoding"), Mapping)
                or row.get("representation", {}).get("bytes") is None
                or row.get("search", {}).get("trials_completed") != 3
            ):
                raise CrossParadigmPackageError(f"Passing resource cell is incomplete: {row.get('cell_id')}")
        elif status == "failed_closed":
            failure = row.get("failure")
            if not isinstance(failure, Mapping) or failure.get("stopping_condition_applied") is not True:
                raise CrossParadigmPackageError(f"Failed resource cell lacks terminal evidence: {row.get('cell_id')}")
            if "quality_verification" not in row or row.get("end_to_end_seconds", 0) <= 0:
                raise CrossParadigmPackageError(f"Failed resource stage evidence is incomplete: {row.get('cell_id')}")
        else:
            raise CrossParadigmPackageError(f"Resource cell lacks a terminal status: {row.get('cell_id')}")
        expected_scope = {
            "tfidf-word-sublinear": ("lexical", "lexical", TFIDF_REPRESENTATION_SCOPE),
            "all-minilm-l6-v2": ("dense", "dense", MINILM_REPRESENTATION_SCOPE),
        }.get(row.get("method_key"))
        if expected_scope is not None:
            family, paradigm, scope = expected_scope
            if (
                row.get("family") != family
                or row.get("representation", {}).get("paradigm") != paradigm
                or row.get("representation", {}).get("scope") != scope
            ):
                raise CrossParadigmPackageError(f"Method/family representation scope drifted: {row.get('cell_id')}")
    guidance = files["scenario-guidance.json"]
    if not guidance.get("scenarios") or any(
        not row.get("recommended_method")
        or not row.get("ordered_methods")
        or not row.get("paired_evidence")
        or not row.get("successful_resource_tradeoffs")
        for row in guidance["scenarios"]
    ):
        raise CrossParadigmPackageError("Guidance contains an empty or non-actionable scenario")


def _privacy_scan(files: Mapping[str, Any], report: str) -> dict[str, Any]:
    data_ids: set[str] = set()
    source_texts: set[str] = set()
    for track in TRACKS:
        data = load_materialized(DATA_ROOT, track)
        data_ids.update(row["id"] for row in data.corpus)
        data_ids.update(row["id"] for row in data.queries)
        source_texts.update(row["content"] for row in data.corpus)
        source_texts.update(row["text"] for row in data.queries)

    def string_values(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, Mapping):
            return {text for key, item in value.items() for text in ({str(key)} | string_values(item))}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return {text for item in value for text in string_values(item)}
        return set()

    payloads = {name: canonical_bytes(value).decode("utf-8") for name, value in files.items()}
    candidate_strings = set().union(*(string_values(value) for value in files.values()))
    forbidden_keys = ('"query_id"', '"document_id"', '"rankings"', '"positive_document_ids"')
    for name, payload in payloads.items():
        if any(key in payload for key in forbidden_keys):
            raise CrossParadigmPackageError(f"Tracked candidate contains a restricted row-level key: {name}")
        if "/data" in payload or "/home/" in payload:
            raise CrossParadigmPackageError(f"Tracked candidate contains a private path: {name}")
    distinctive_ids = {identifier for identifier in data_ids if len(identifier) >= 8}
    if candidate_strings & distinctive_ids or any(identifier in report for identifier in distinctive_ids):
        raise CrossParadigmPackageError("Tracked candidate contains a canonical ID")
    distinctive_texts = {text for text in source_texts if len(text) >= 64}
    if candidate_strings & distinctive_texts or any(text in report for text in distinctive_texts):
        raise CrossParadigmPackageError("Tracked candidate contains restricted source text")
    if "/data" in report or "/home/" in report:
        raise CrossParadigmPackageError("Research report contains a private path")
    return {
        "status": "pass",
        "files_scanned": len(payloads),
        "canonical_ids": len(data_ids),
        "short_ambiguous_ids_covered_by_forbidden_key_scan": len(data_ids - distinctive_ids),
        "source_rows": len(source_texts),
        "short_source_fragments_excluded_from_literal_scan": len(source_texts - distinctive_texts),
        "restricted_keys_absent": True,
        "private_paths_absent": True,
        "secret_scan_required_externally": True,
    }


def compute_outputs() -> tuple[dict[str, Any], dict[str, Any], str]:
    cells = collect_cells()
    inventory_value, inventory = _inventory()
    predecl = _json(ARTIFACT_ROOT / "predeclaration.json")
    summary = build_summary(cells)
    pairwise = build_pairwise(cells)
    slices = build_slices(cells, predecl)
    cases, local = build_cases(cells)
    resources = build_resources()
    files = {
        "summary.json": summary,
        "pairwise.json": pairwise,
        "slices.json": slices,
        "cases.json": cases,
        "lineage.json": build_lineage(cells, inventory),
        "resources.json": resources,
        "descriptive-sensitivity.json": build_sensitivity(slices, inventory),
        "scenario-guidance.json": build_guidance(summary, pairwise, slices, resources),
    }
    for name, value in files.items():
        _validate_schema(name, value)
    validate_machine_contracts(files)
    report = _report(files)
    privacy = _privacy_scan(files, report)
    evidence = {
        "local_mapping": local,
        "privacy": privacy,
        "inventory_sha256": sha256_file(ARTIFACT_ROOT / "inventory.json"),
        "predeclaration_sha256": sha256_file(ARTIFACT_ROOT / "predeclaration.json"),
        "inventory_selection_score_blind": inventory_value["selection_score_blind"],
    }
    return files, evidence, report


def _manifest(files: Mapping[str, Any], evidence: Mapping[str, Any], report: str) -> dict[str, Any]:
    tracked = {
        name: {"bytes": len(canonical_bytes(value)), "sha256": _canonical_sha(value)} for name, value in files.items()
    }
    report_bytes = report.encode("utf-8")
    tracked[REPORT_PATH.relative_to(ROOT).as_posix()] = {
        "bytes": len(report_bytes),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    return {
        "schema_version": "bright-cross-paradigm-results-manifest-v1",
        "story_id": STORY_ID,
        "publication": publication_policy(),
        "candidate_order": list(CANDIDATE_ORDER),
        "bindings": {
            "inventory_sha256": evidence["inventory_sha256"],
            "predeclaration_sha256": evidence["predeclaration_sha256"],
            "gate_summary_sha256": sha256_file(ARTIFACT_ROOT / "gate-summary.json"),
            "chronology_sha256": sha256_file(ARTIFACT_ROOT / "chronology.json"),
            "resource_summary_sha256": sha256_file(RESOURCE_PATH),
            "local_mapping_sha256": _canonical_sha(evidence["local_mapping"]),
        },
        "matrix": {"methods": 14, "tracks": 2, "cells": 28, "complete": True},
        "statistics": {"pairs_per_track": 91, "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        "privacy": evidence["privacy"],
        "tracked_files": tracked,
    }


def candidate_paths() -> tuple[str, ...]:
    """Return the exact ordered tracked candidate set, excluding its identity file."""
    freeze = tuple(
        f"benchmark/artifacts/bright-cross-paradigm-v0.1/{name}{suffix}"
        for name in (
            "chronology.json",
            "gate-summary.json",
            "identity-audit.json",
            "inventory.json",
            "metadata-preflight.json",
            "predeclaration.json",
        )
        for suffix in ("", ".sha256")
    )
    results = tuple(
        f"benchmark/artifacts/bright-cross-paradigm-results-v0.1/{name}{suffix}"
        for name in (
            "cases.json",
            "lineage.json",
            "manifest.json",
            "pairwise.json",
            "descriptive-sensitivity.json",
            "resources.json",
            "scenario-guidance.json",
            "slices.json",
            "summary.json",
        )
        for suffix in ("", ".sha256")
    )
    report = (
        "benchmark/research/bright_cross_paradigm_20260905.md",
        "benchmark/research/bright_cross_paradigm_20260905.md.sha256",
    )
    paths = tuple(sorted((*IMPLEMENTATION_CANDIDATES, *freeze, *results, *report)))
    if len(paths) != len(set(paths)):
        raise CrossParadigmPackageError("Candidate paths are not unique")
    return paths


def build_or_check(*, check_only: bool) -> str:
    files, evidence, report = compute_outputs()
    manifest = _manifest(files, evidence, report)
    Draft202012Validator(_json(ROOT / "schemas/bright-cross-paradigm-manifest-v01.schema.json")).validate(manifest)
    if check_only:
        saved_manifest = _json(PACKAGE_ROOT / "manifest.json")
        if saved_manifest != manifest:
            raise CrossParadigmPackageError("Candidate manifest cannot be deterministically rebuilt")
        for name, value in files.items():
            if (PACKAGE_ROOT / name).read_bytes() != canonical_bytes(value):
                raise CrossParadigmPackageError(f"Candidate artifact cannot be rebuilt: {name}")
        if REPORT_PATH.read_text(encoding="utf-8") != report:
            raise CrossParadigmPackageError("Research report cannot be deterministically rebuilt")
        if _json(LOCAL_CASE_PATH) != evidence["local_mapping"]:
            raise CrossParadigmPackageError("Restricted case mapping cannot be deterministically rebuilt")
        return sha256_file(PACKAGE_ROOT / "manifest.json")
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        _write(PACKAGE_ROOT / name, value)
    _write(LOCAL_CASE_PATH, evidence["local_mapping"])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".sha256").write_text(
        sha256_file(REPORT_PATH) + "\n", encoding="ascii"
    )
    return _write(PACKAGE_ROOT / "manifest.json", manifest)


__all__ = [
    "CANDIDATE_ORDER",
    "IMPLEMENTATION_CANDIDATES",
    "METHODS",
    "Cell",
    "CrossParadigmPackageError",
    "build_or_check",
    "candidate_paths",
    "collect_cells",
    "compute_outputs",
    "validate_machine_contracts",
]
