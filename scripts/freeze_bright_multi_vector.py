#!/usr/bin/env python3
"""Freeze the S-20260814-012 inventory and predeclaration before formal scoring."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from mm_embed.providers.real_multi_vector import INVENTORY, SELECTED_KEYS

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = ROOT / "benchmark/artifacts/bright-multi-vector-v0.1"
OUTPUT = ROOT / "benchmark/artifacts/bright-multi-vector-v0.2"
ORIGINAL_PREDECLARATION_SHA256 = "ec9c100e8470e6dd754fe87388dbd5bf9562ab62fdf12d6daac25ba8a92c761f"
ORIGINAL_PREDECLARATION_MTIME = "2026-09-04T16:24:54.083245294Z"
FIRST_GATE_MTIME = "2026-09-04T16:31:28.396810066Z"
VALIDATOR_SESSION = "ccu-codex-modern-embedding-bench-20260904172923-3779564"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def write_once(path: Path, value: object) -> str:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError(f"Frozen artifact already exists: {path}")
    data = canonical_bytes(value)
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def selected_contract() -> list[dict[str, object]]:
    keys = (
        "key",
        "repo_id",
        "revision",
        "license",
        "gated",
        "private",
        "lineage",
        "languages",
        "parameters",
        "model_weight_bytes",
        "estimated_snapshot_bytes",
        "download_cap_bytes",
        "backbone",
        "query_route",
        "document_route",
        "dimensions",
        "query_length",
        "document_length",
        "query_expansion",
        "mask_policy",
        "padding_policy",
        "trust_remote_code",
        "remote_code_risk",
        "call_stack",
        "allowlist",
    )
    return [{key: getattr(INVENTORY[name], key) for key in keys} for name in SELECTED_KEYS]


SLICES = [
    {
        "key": "query_length",
        "bins": ["<=64", "65-128", ">128"],
        "definition": "Lowercase strip canonical query text and count Unicode word tokens with (?u)\\b\\w+\\b.",
    },
    {
        "key": "positive_lexical_overlap",
        "bins": ["0", "(0,0.25]", "(0.25,0.50]", ">0.50"],
        "definition": "For each positive passage, divide unique lowercase Unicode query/document token intersection by unique query tokens and use the maximum across positives.",
    },
    {
        "key": "positive_document_length",
        "bins": ["<=128", "129-512", ">512"],
        "definition": "Count Unicode word tokens for every positive passage and bin a query by the median positive-passage count.",
    },
    {
        "key": "positive_qrel_density",
        "bins": ["1", "2-4", ">=5"],
        "definition": "Count unique canonical positive qrel passage IDs per query after materialization validation.",
    },
]


def predeclaration(inventory_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "bright-real-multi-vector-predeclaration-v1",
        "story_id": "S-20260814-012",
        "created_at": "2026-09-04T16:32:00Z",
        "formal_scores_observed": False,
        "bindings": {
            "inventory": {
                "path": "benchmark/artifacts/bright-multi-vector-v0.1/inventory.json",
                "sha256": inventory_sha256,
            },
            "canonical_data": {
                "dataset_version": "bright-nontechnical-pilot-v0.2",
                "root": "data/bright-nontechnical-pilot-v0.2",
                "manifest_sha256": "8174b0c01a32e977cf0c5d89522356c485196b0aa0571251cfe1595c5579cb1f",
                "source_revision": "3066d29c9651a576c8aba4832d249807b181ecae",
                "tracks": {
                    "economics": {
                        "documents": 7500,
                        "queries": 103,
                        "qrels": 800,
                        "selection_ids_sha256": "13e2bf56644a8d5b8d8701be74ead7b0f147057bbda34e6a8f246b515f4b1398",
                        "corpus_sha256": "697bf4deb34bcac5423a7dfcaf60ce27ea0c961190c0931d189a1620a703e188",
                        "queries_sha256": "34f8dfb6ddb616be14582c20f291fa286f8548fe797af09e37bf444b8fb34665",
                        "qrels_sha256": "e09a3c96a3b112d4233e26baacf5d0f021b6025378fded32478f94e4a2bf3fa3",
                    },
                    "psychology": {
                        "documents": 7500,
                        "queries": 101,
                        "qrels": 692,
                        "selection_ids_sha256": "d56ba2a26cc69163b412c35ebea5c1fa5141ab03fe54215a19abd053d516d3e4",
                        "corpus_sha256": "a7513e2a67d2fb9b80504d12253f041863fba895637c40d6ae7efbcba1700089",
                        "queries_sha256": "ead6a7004ea306ca90e8db58f5c6809d05cb448e34c276834aebf7bbe0c90d9d",
                        "qrels_sha256": "52360d1c95984d8db2b04e018f68e99737063836e889bf3e7f6865b76d26bc44",
                    },
                },
            },
            "accepted_foundations": {
                "exact_maxsim_fixture_commit": "06498af01352cfe734615c6eb768f4ed543ee336",
                "learned_sparse_batch_b_predeclaration_sha256": "2d30ac69b58e7bbd217550418619bab085e2c8b73e1453ee02e0d237827a32ef",
                "learned_sparse_phase_one_manifest_sha256": "075e1029e7ca62e87968d1b513ac40a2e4e32a2f9855315fcdb639d7a7a05909",
                "learned_sparse_phase_one_commit": "082e1adccb2fc389290e44c3db4d420d5428318e",
            },
        },
        "models": selected_contract(),
        "protocol": {
            "tracks": ["economics", "psychology"],
            "candidate_policy": "Full independent 7,500-passage track after query-specific excluded IDs; no learned candidate generator.",
            "retrieval_unit": "Canonical BRIGHT pilot passage; no cross-track pooling.",
            "windowing": {
                "tokenizer": "unicode-word-regex-v1",
                "pattern": "(?u)\\b\\w+\\b",
                "window_words": 128,
                "stride_words": 96,
                "short_passage": "Preserve the exact canonical string when it has at most 128 word tokens.",
                "long_passage": "Use overlapping source substrings; force a final tail-aligned window and remove duplicate spans.",
            },
            "aggregation": "max_passage",
            "aggregation_definition": "Score every window by exact MaxSim and retain the maximum score per canonical passage; ties choose the lower window ordinal.",
            "score": "For each valid query token, take the maximum float32 dot product over valid document-window tokens, then sum in query-token order.",
            "normalization": "Use checkpoint-native per-token normalization loaded by MultiVectorEncoder; no post-hoc score normalization.",
            "mask": "Use the model's final scoring attention mask after query expansion and document punctuation filtering; compact valid rows for storage and retain an explicit boolean-valid mask.",
            "padding": "Padded rows in adapter validation are zero and mask=false; compact formal storage contains only mask=true rows plus offsets.",
            "tie_policy": "Sort by descending float32 score, then canonical passage ID; public artifacts never expose that ID.",
            "top_k": 100,
            "dtype": "float32",
            "exact_backend": "torch_exact_maxsim_with_independent_numpy_gate",
            "representation_storage": "Concatenated float32 token vectors, int64 offsets, boolean valid-token mask, and restricted window-to-passage mapping.",
        },
        "gate": {
            "queries": 1,
            "documents": 4,
            "source": "Project-authored synthetic English facts with no benchmark source text or identifiers.",
            "requirements": [
                "immutable snapshot and license preflight passes",
                "trust_remote_code remains false",
                "query and document routes produce the frozen dimension",
                "scoring masks are boolean, nonempty, and exclude zero adapter padding",
                "all vectors and scores are finite",
                "project exact MaxSim equals independent NumPy MaxSim within 1e-5",
                "the predeclared positive is uniquely rank one",
            ],
            "wall_cap_seconds_per_model": 600,
        },
        "metrics": ["ndcg@10", "map@100", "mrr@10", "recall@10", "recall@100"],
        "uncertainty": {
            "method": "paired percentile bootstrap over matched queries",
            "samples": 10000,
            "seed": 20260826,
            "interval": 0.95,
            "comparisons": "All three model pairs within each track; never pool tracks.",
            "small_slice_policy": "n=0 is NA; n=1-9 has point estimates only; n>=10 has paired intervals; bins are never merged.",
        },
        "slices": SLICES,
        "cases": {
            "per_track": {"disagreement": 6, "failure": 6},
            "selection": "Score-only deterministic selection after all cells: largest rank spread and lowest best-model nDCG@10 contribution, with query-ID hash tie breaks.",
            "tracked_content": "Opaque case IDs, aggregate mechanism summary, and restricted local evidence hashes only.",
        },
        "execution": {
            "document_batch_fallback": [32, 16, 8, 4, 2, 1],
            "query_batch_fallback": [64, 32, 16, 8, 4, 2, 1],
            "document_encode_chunk": 256,
            "query_encode_chunk": 128,
            "search_query_batch": 1,
            "maxsim_chunk_elements": 20000000,
            "warmup": "One synthetic query against up to 256 synthetic/bounded windows before measured track search.",
            "measured_trials": "One full exact top-100 search per canonical query; record per-query latency.",
            "resume": "Finalized representation and cell manifests must validate by external SHA256 before reuse; partial cells are rejected by default.",
        },
        "hardware": {
            "device": "cuda:0",
            "gpu": "NVIDIA GeForce RTX 3080 Ti",
            "gpu_memory_bytes": 12884901888,
            "driver": "565.57.01",
            "cuda_runtime": "12.8",
            "python": "3.12.9",
            "torch": "2.10.0+cu128",
            "transformers": "5.3.0",
            "sentence_transformers": "6.0.1",
            "numpy": "2.4.3",
            "rss_scope": "Process ru_maxrss observed during each fresh per-model execution; not an externally sampled lifetime peak.",
        },
        "resource_caps": {
            "story_download_bytes": 2147483648,
            "story_private_artifact_bytes": 12884901888,
            "model_vram_bytes": 11811160064,
            "formal_cell_wall_seconds": 3600,
            "formal_model_wall_seconds": 7200,
        },
        "stopping_conditions": [
            "Any selected snapshot becomes gated/private, changes license, requires remote code, or exceeds its download cap.",
            "Any gate fails, any representation is non-finite/empty/wrong-dimensional, or exact scorers disagree.",
            "Observed VRAM exceeds 11 GiB after exhausting the frozen batch fallback.",
            "A cell exceeds one hour or a model exceeds two hours; do not silently change semantic lengths, windows, aggregation, candidates, metrics, or dtype.",
            "Any tracked artifact contains benchmark source text, canonical IDs, raw rankings, reversible mappings, secrets, or private filesystem paths.",
        ],
        "publication": {
            "classification": "research_only",
            "gate": "closed",
            "publish": False,
            "leaderboard_publish": False,
            "public_export_allowed": False,
            "weight_redistribution": False,
            "limitations": [
                "English economics and psychology only",
                "training overlap unknown; not verified zero-shot",
                "120-query review unfinished",
                "document-level redistribution rights unresolved",
                "no production SLA or complete cross-paradigm ranking claim",
            ],
        },
    }


def chronology(original: dict[str, object], reissued_at: str) -> dict[str, object]:
    semantic_contract = dict(original)
    semantic_contract.pop("created_at")
    return {
        "schema_version": "bright-real-multi-vector-chronology-v1",
        "story_id": "S-20260814-012",
        "status": "validator-confirmed-pre-score-freeze-with-transparent-chronology-reissue",
        "reissued_at": reissued_at,
        "independent_validator_session": VALIDATOR_SESSION,
        "original_pre_score_artifact": {
            "sha256": ORIGINAL_PREDECLARATION_SHA256,
            "observed_filesystem_mtime": ORIGINAL_PREDECLARATION_MTIME,
            "embedded_created_at": original["created_at"],
            "embedded_created_at_status": "incorrect_future_literal",
            "formal_scores_observed": original["formal_scores_observed"],
            "semantic_contract_sha256": hashlib.sha256(canonical_bytes(semantic_contract)).hexdigest(),
        },
        "first_gate": {
            "observed_filesystem_mtime": FIRST_GATE_MTIME,
            "chronology": "original_pre_score_artifact_precedes_first_gate",
        },
        "correction": {
            "models_changed": False,
            "protocol_changed": False,
            "tracks_changed": False,
            "metrics_changed": False,
            "resource_caps_changed": False,
            "reason": "Correct the impossible embedded timestamp while preserving and explicitly binding the independently observed pre-score contract.",
        },
        "publication": original["publication"],
    }


def active_predeclaration(
    original: dict[str, object], *, chronology_sha256: str, reissued_at: str
) -> dict[str, object]:
    value = dict(original)
    value["schema_version"] = "bright-real-multi-vector-predeclaration-v2"
    value["created_at"] = reissued_at
    value["formal_scores_observed"] = True
    value["chronology"] = {
        "status": "transparent_reissue_of_validator-confirmed-pre-score-contract",
        "issued_after_prior_scores_observed": True,
        "model_or_protocol_reselection": False,
        "chronology_attestation_sha256": chronology_sha256,
        "original_pre_score_predeclaration_sha256": ORIGINAL_PREDECLARATION_SHA256,
        "original_pre_score_frozen_at": ORIGINAL_PREDECLARATION_MTIME,
    }
    return value


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_path = ORIGINAL_ROOT / "predeclaration.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != ORIGINAL_PREDECLARATION_SHA256:
        raise ValueError("Original pre-score predeclaration identity drifted")
    deterministic = json.loads(
        json.dumps(predeclaration(hashlib.sha256((ORIGINAL_ROOT / "inventory.json").read_bytes()).hexdigest()))
    )
    if original != deterministic:
        raise ValueError("Original pre-score predeclaration no longer matches its deterministic contract")
    observed_mtime = datetime.fromtimestamp(original_path.stat().st_mtime_ns / 1_000_000_000, tz=UTC)
    expected_mtime = datetime.fromisoformat(ORIGINAL_PREDECLARATION_MTIME.replace("Z", "+00:00"))
    first_gate = datetime.fromisoformat(FIRST_GATE_MTIME.replace("Z", "+00:00"))
    if observed_mtime != expected_mtime or observed_mtime >= first_gate:
        raise ValueError("Original filesystem chronology no longer proves a pre-score freeze")
    reissued_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    chronology_sha256 = write_once(OUTPUT / "chronology.json", chronology(original, reissued_at))
    predeclaration_sha256 = write_once(
        OUTPUT / "predeclaration.json",
        active_predeclaration(original, chronology_sha256=chronology_sha256, reissued_at=reissued_at),
    )
    print(
        json.dumps(
            {
                "original_pre_score_sha256": ORIGINAL_PREDECLARATION_SHA256,
                "chronology_sha256": chronology_sha256,
                "predeclaration_sha256": predeclaration_sha256,
                "reissued_at": reissued_at,
            }
        )
    )


if __name__ == "__main__":
    main()
