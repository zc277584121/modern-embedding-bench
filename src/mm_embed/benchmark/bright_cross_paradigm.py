"""Pre-result freeze and bounded gates for the BRIGHT cross-paradigm study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = ROOT / "benchmark/artifacts/bright-cross-paradigm-v0.1"
STORY_ID = "S-20260814-009"
TRACKS = ("economics", "psychology")
METRICS = ("ndcg@10", "map@100", "mrr@10", "recall@10", "recall@100")
ACCEPTED_COMMITS = {
    "s005": "10573c1f24975272bd68b61c6bece2ac1a25681e",
    "s008": "74635817770023b70b1e9f9e97b3d83d9d11f987",
    "s011": "082e1adccb2fc389290e44c3db4d420d5428318e",
    "s012": "3c414af21f0713cf0d440bb2e951a45302ff0259",
}
PROTECTED_FILES = {
    "benchmark/artifacts/milvus-sindi-system-v0.1/validator-candidate.json": (
        "45815999cdb1d03ccf00e7aea9b9c1cab24bc833544ee492fccc3f6cd65e8100"
    ),
    "benchmark/artifacts/milvus-sindi-system-v0.1/validator-candidate.json.sha256": (
        "ca81a689018b3537c1b0bdbe32fb885473ba665254f36ecbd3faf69677339a02"
    ),
}
ACCEPTED_ARTIFACTS = {
    "s005_manifest": (
        "benchmark/artifacts/bright-nontechnical-pilot-v0.2/manifest.json",
        "6b48e45d4f8332a5386729898845c1a62b02b2fd1481b4665d501947a082fd86",
    ),
    "s005_summary": (
        "benchmark/artifacts/bright-nontechnical-pilot-v0.2/summary.json",
        "7aa2f3561263d2f8d82026244841304b26b725caff185b4a21a8a868c610c249",
    ),
    "s008_manifest": (
        "benchmark/artifacts/bright-learned-sparse-main-v0.1/manifest.json",
        "3accbed7b2a9961eedff20e601375a23c50167cfc7d8e77adbf91399aed91f47",
    ),
    "s008_summary": (
        "benchmark/artifacts/bright-learned-sparse-main-v0.1/summary.json",
        "80e57e7648047753d21639660d89aa3676c084f5c8107f1a538a4cfbb4598a79",
    ),
    "s011_manifest": (
        "benchmark/artifacts/learned-sparse-phase-one-v0.1/manifest.json",
        "075e1029e7ca62e87968d1b513ac40a2e4e32a2f9855315fcdb639d7a7a05909",
    ),
    "s012_inventory": (
        "benchmark/artifacts/bright-multi-vector-v0.1/inventory.json",
        "0185edb9948f390403d2eb23400c78734448fb130e6a408af0ff55e8cfc3fa2c",
    ),
    "s012_predeclaration": (
        "benchmark/artifacts/bright-multi-vector-v0.2/predeclaration.json",
        "2ef6bbc245d9f196d71adcd8498575b14ecd650c29242fb9d0eb284704c16f05",
    ),
    "s012_manifest": (
        "benchmark/artifacts/bright-multi-vector-results-v0.2/manifest.json",
        "fa80bbd1f8b711ef76ae27f606319cf9b856f7e9baf49f9643276413e2a822f8",
    ),
    "s012_summary": (
        "benchmark/artifacts/bright-multi-vector-results-v0.2/summary.json",
        "947ac644fb66d0911e308ea243bce1c8155a5d6cc3af17e391b083cc875710f6",
    ),
}
DATA_IDENTITY = {
    "dataset_version": "bright-nontechnical-pilot-v0.2",
    "source_revision": "3066d29c9651a576c8aba4832d249807b181ecae",
    "manifest_sha256": "8174b0c01a32e977cf0c5d89522356c485196b0aa0571251cfe1595c5579cb1f",
    "tracks": {
        "economics": {
            "documents": 7500,
            "queries": 103,
            "qrels": 800,
            "corpus_sha256": "697bf4deb34bcac5423a7dfcaf60ce27ea0c961190c0931d189a1620a703e188",
            "queries_sha256": "34f8dfb6ddb616be14582c20f291fa286f8548fe797af09e37bf444b8fb34665",
            "qrels_sha256": "e09a3c96a3b112d4233e26baacf5d0f021b6025378fded32478f94e4a2bf3fa3",
            "selection_ids_sha256": "13e2bf56644a8d5b8d8701be74ead7b0f147057bbda34e6a8f246b515f4b1398",
        },
        "psychology": {
            "documents": 7500,
            "queries": 101,
            "qrels": 692,
            "corpus_sha256": "a7513e2a67d2fb9b80504d12253f041863fba895637c40d6ae7efbcba1700089",
            "queries_sha256": "ead6a7004ea306ca90e8db58f5c6809d05cb448e34c276834aebf7bbe0c90d9d",
            "qrels_sha256": "52360d1c95984d8db2b04e018f68e99737063836e889bf3e7f6865b76d26bc44",
            "selection_ids_sha256": "d56ba2a26cc69163b412c35ebea5c1fa5141ab03fe54215a19abd053d516d3e4",
        },
    },
}
MINILM = {
    "key": "all-minilm-l6-v2",
    "family": "dense",
    "status": "selected_new_formal_run",
    "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    "license": "Apache-2.0",
    "gated": False,
    "private": False,
    "trust_remote_code": False,
    "parameters": 22_713_728,
    "dimensions": 384,
    "max_length": 256,
    "snapshot_bytes": 91_578_367,
    "snapshot_sha256": "04c1a2ae6a1c7299745f8eb7f8db75456a7df9652dd4f48a40be2ee59d7c37c1",
    "query_route": "symmetric SentenceTransformer encode without prompt",
    "document_route": "symmetric SentenceTransformer encode without prompt",
    "representation": "mean-pooled L2-normalized float32 dense vector",
    "similarity": "exact cosine via float32 inner product over normalized vectors",
    "reason": (
        "Independent mature 22.7M MiniLM sentence encoder already cached at the pinned revision; "
        "it complements the 568M BGE-M3 anchor in size, dimension, and context budget."
    ),
}


class CrossParadigmError(RuntimeError):
    """Fail-closed contract error."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, value: object) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Frozen artifact already exists: {path}")
    data = canonical_bytes(value)
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    sidecar.write_text(digest + "\n", encoding="ascii")
    return digest


def verify_sidecar(path: Path) -> str:
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
    actual = sha256_file(path)
    if actual != expected:
        raise CrossParadigmError(f"Sidecar mismatch: {path}")
    return actual


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_foundations() -> dict[str, Any]:
    checks: dict[str, Any] = {"accepted_commits": {}, "accepted_artifacts": {}, "protected_files": {}}
    for name, commit in ACCEPTED_COMMITS.items():
        result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=False)
        if result.returncode:
            raise CrossParadigmError(f"Accepted commit is not an ancestor of HEAD: {commit}")
        checks["accepted_commits"][name] = {"commit": commit, "ancestor_of_head": True}
    for name, (relative, expected) in ACCEPTED_ARTIFACTS.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise CrossParadigmError(f"Accepted artifact drifted: {relative}")
        checks["accepted_artifacts"][name] = {"path": relative, "sha256": actual}
    for relative, expected in PROTECTED_FILES.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise CrossParadigmError(f"Protected artifact drifted: {relative}")
        checks["protected_files"][relative] = actual
    return checks


def snapshot_inventory(snapshot: Path) -> dict[str, Any]:
    files = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    return {
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "aggregate_sha256": aggregate.hexdigest(),
    }


def minilm_snapshot_path() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return (
        Path(HF_HUB_CACHE) / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / str(MINILM["revision"])
    )


def metadata_preflight() -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/revision/{MINILM['revision']}"
    request = urllib.request.Request(url, headers={"User-Agent": "modern-embedding-bench-metadata-preflight"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if int(response.headers.get("Content-Length", "0") or "0") > 2_000_000:
            raise CrossParadigmError("Metadata response exceeds the 2 MB cap")
        remote = json.load(response)
    snapshot = minilm_snapshot_path()
    if not snapshot.is_dir():
        raise CrossParadigmError("Pinned MiniLM snapshot is not already cached; preflight will not download it")
    local = snapshot_inventory(snapshot)
    if local["total_bytes"] != MINILM["snapshot_bytes"] or local["aggregate_sha256"] != MINILM["snapshot_sha256"]:
        raise CrossParadigmError("Pinned MiniLM snapshot identity drifted")
    allowed = {
        "1_Pooling/config.json",
        "README.md",
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
    if {item["path"] for item in local["files"]} != allowed:
        raise CrossParadigmError("Pinned MiniLM local allowlist drifted")
    model_config = read_json(snapshot / "config.json")
    sentence_config = read_json(snapshot / "sentence_bert_config.json")
    pooling_config = read_json(snapshot / "1_Pooling/config.json")
    modules = read_json(snapshot / "modules.json")
    sklearn_metadata = importlib.metadata.metadata("scikit-learn")
    return {
        "schema_version": "bright-cross-paradigm-metadata-preflight-v1",
        "story_id": STORY_ID,
        "observed_at": utc_now(),
        "network_policy": "Metadata API only; no model files were downloaded.",
        "lexical": {
            "implementation": "sklearn.feature_extraction.text.TfidfVectorizer",
            "package": "scikit-learn",
            "version": importlib.metadata.version("scikit-learn"),
            "license_expression": sklearn_metadata.get("License-Expression"),
            "installed_before_freeze": True,
            "external_service": False,
        },
        "dense": {
            "repo_id": remote.get("id"),
            "requested_revision": MINILM["revision"],
            "resolved_revision": remote.get("sha"),
            "private": remote.get("private"),
            "gated": remote.get("gated"),
            "license_tag_present": "license:apache-2.0" in remote.get("tags", []),
            "library_name": remote.get("library_name"),
            "pipeline_tag": remote.get("pipeline_tag"),
            "safetensors_parameters": remote.get("safetensors", {}).get("parameters"),
            "remote_repository_files": sorted(item["rfilename"] for item in remote.get("siblings", [])),
            "remote_python_files_not_loaded": sorted(
                item["rfilename"] for item in remote.get("siblings", []) if item["rfilename"].endswith(".py")
            ),
            "local_snapshot": local,
            "local_only_gate_required": True,
            "allowlist_exact": True,
            "auto_map": model_config.get("auto_map"),
            "architecture": model_config.get("architectures"),
            "hidden_size": model_config.get("hidden_size"),
            "max_sequence_length": sentence_config.get("max_seq_length"),
            "pooling": {
                "dimension": pooling_config.get("word_embedding_dimension"),
                "mean": pooling_config.get("pooling_mode_mean_tokens"),
                "cls": pooling_config.get("pooling_mode_cls_token"),
                "max": pooling_config.get("pooling_mode_max_tokens"),
            },
            "module_types": [item["type"] for item in modules],
            "trust_remote_code": False,
        },
        "decision": "pass",
    }


def _accepted_sparse_methods() -> list[dict[str, Any]]:
    batch_a_inventory = read_json(ROOT / "benchmark/artifacts/bright-learned-sparse-batch-a-v0.1/inventory.json")[
        "models"
    ]
    batch_b = read_json(ROOT / "benchmark/artifacts/bright-learned-sparse-batch-b-v0.1/predeclaration.json")["models"]
    selected = [
        item
        for item in batch_a_inventory
        if item["key"] in {"granite-30m-sparse", "opensearch-doc-v2-mini", "opensearch-doc-v3", "bge-m3"}
    ]
    selected.extend(batch_b)
    raw_hashes = read_json(ROOT / "benchmark/artifacts/bright-learned-sparse-main-v0.1/manifest.json")["bindings"][
        "raw_manifest_sha256"
    ]
    methods = []
    for item in selected:
        methods.append(
            {
                "key": item["key"],
                "family": "learned_sparse",
                "status": "accepted_read_only",
                "repo_id": item["repo_id"],
                "revision": item["revision"],
                "license": item["license"],
                "gated": item.get("gated", False),
                "trust_remote_code": item["trust_remote_code"],
                "query_route": item["query_route"],
                "document_route": item["document_route"],
                "max_length": item["max_length"],
                "dimensions": item["dimensions"],
                "estimated_snapshot_bytes": item["estimated_snapshot_bytes"],
                "accepted_cell_sha256": {track: raw_hashes[f"{item['key']}:{track}"] for track in TRACKS},
            }
        )
    return methods


def _accepted_multi_vector_methods() -> list[dict[str, Any]]:
    inventory = read_json(ROOT / ACCEPTED_ARTIFACTS["s012_inventory"][0])
    selected = set(inventory["selected_keys"])
    cell_hashes = read_json(ROOT / ACCEPTED_ARTIFACTS["s012_manifest"][0])["bindings"]["formal_cells"]
    methods = []
    for item in inventory["models"]:
        if item["key"] not in selected:
            continue
        methods.append(
            {
                "key": item["key"],
                "family": "multi_vector",
                "status": "accepted_read_only",
                "repo_id": item["repo_id"],
                "revision": item["revision"],
                "license": item["license"],
                "gated": item["gated"],
                "trust_remote_code": item["trust_remote_code"],
                "query_route": item["query_route"],
                "document_route": item["document_route"],
                "query_length": item["query_length"],
                "document_length": item["document_length"],
                "dimensions": item["dimensions"],
                "parameters": item["parameters"],
                "estimated_snapshot_bytes": item["estimated_snapshot_bytes"],
                "accepted_cell_sha256": {track: cell_hashes[f"{item['key']}:{track}"] for track in TRACKS},
            }
        )
    return methods


def baseline_inventory() -> dict[str, Any]:
    s005 = read_json(ROOT / ACCEPTED_ARTIFACTS["s005_summary"][0])
    methods: list[dict[str, Any]] = [
        {
            "key": "bm25-unicode",
            "family": "lexical",
            "status": "accepted_read_only",
            "implementation": "project in-process exact BM25",
            "license": "project license",
            "mechanism": "probabilistic term saturation and document-length normalization",
            "tokenizer": "unicode-word-lower-v1",
            "k1": 1.2,
            "b": 0.75,
            "accepted_cell_sha256": {
                track: s005["artifact_identities"][f"bm25-{track}"]["result_sha256"] for track in TRACKS
            },
        },
        {
            "key": "tfidf-word-sublinear",
            "family": "lexical",
            "status": "selected_new_formal_run",
            "implementation": "sklearn.feature_extraction.text.TfidfVectorizer",
            "package": "scikit-learn==1.8.0",
            "license": "BSD-3-Clause",
            "mechanism": "sublinear term frequency times smoothed inverse document frequency with L2 cosine",
            "configuration": {
                "analyzer": "word",
                "lowercase": True,
                "token_pattern": "(?u)\\b\\w+\\b",
                "sublinear_tf": True,
                "use_idf": True,
                "smooth_idf": True,
                "norm": "l2",
                "dtype": "float32",
            },
            "reason": (
                "Mature in-process sparse vector-space retrieval with no service/JVM and a materially different "
                "weighting/normalization mechanism from BM25."
            ),
        },
        {
            "key": "bge-m3-dense",
            "family": "dense",
            "status": "accepted_read_only",
            "repo_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "license": "MIT",
            "gated": False,
            "trust_remote_code": False,
            "parameters": 568_000_000,
            "dimensions": 1024,
            "declared_max_length": 8192,
            "formal_max_length": 1024,
            "query_route": "symmetric SentenceTransformer dense encode without prompt",
            "document_route": "symmetric SentenceTransformer dense encode without prompt",
            "similarity": "exact cosine via float32 inner product over normalized vectors",
            "accepted_cell_sha256": {
                track: s005["artifact_identities"][f"long_dense-{track}"]["result_sha256"] for track in TRACKS
            },
        },
        dict(MINILM),
    ]
    methods.extend(_accepted_sparse_methods())
    methods.extend(_accepted_multi_vector_methods())
    if len(methods) != 14 or len({item["key"] for item in methods}) != 14:
        raise CrossParadigmError("Inventory must contain exactly 14 distinct methods")
    return {
        "schema_version": "bright-cross-paradigm-inventory-v1",
        "story_id": STORY_ID,
        "selection_basis": [
            "mechanism diversity",
            "license clarity",
            "implementation maturity",
            "cache and resource complementarity",
        ],
        "selection_score_blind": False,
        "selection_score_blind_explanation": (
            "Accepted S-005, S-008, S-012, and prior MiniLM diagnostic scores were already visible. "
            "All accepted learned-sparse and multi-vector identities are retained unchanged; new choices are "
            "justified only by the declared non-score criteria."
        ),
        "methods": methods,
        "counts": {"lexical": 2, "dense": 2, "learned_sparse": 7, "multi_vector": 3},
        "exclusions": [
            {
                "candidate": "BM25L/BM25+ parameter variants",
                "family": "lexical",
                "reason": "Too close to the accepted BM25 family to establish the required material mechanism distinction.",
            },
            {
                "candidate": "character n-gram TF-IDF",
                "family": "lexical",
                "reason": "Deferred to avoid adding a third lexical configuration before the two-method minimum is complete.",
            },
            {
                "candidate": "intfloat/multilingual-e5-small",
                "family": "dense",
                "reason": "Cached but adds multilingual/prefix semantics while the shared tracks are English; MiniLM is the cleaner small symmetric complement.",
            },
            {
                "candidate": "Alibaba-NLP/gte-multilingual-base",
                "family": "dense",
                "reason": "Cached but substantially larger and less resource-complementary than MiniLM for the second dense slot.",
            },
            {
                "candidate": "BAAI/bge-small-en-v1.5",
                "family": "dense",
                "reason": "Same BGE lineage as the accepted anchor and not already cached; weaker independence and cache complementarity.",
            },
        ],
        "publication": publication_policy(),
    }


def publication_policy() -> dict[str, Any]:
    return {
        "classification": "research_only",
        "gate": "closed",
        "publish": False,
        "leaderboard_publish": False,
        "public_export_allowed": False,
        "restricted_content_allowed": False,
    }


def identity_audit() -> dict[str, Any]:
    foundations = assert_foundations()
    return {
        "schema_version": "bright-cross-paradigm-identity-audit-v1",
        "story_id": STORY_ID,
        "foundations": foundations,
        "canonical_data": DATA_IDENTITY,
        "shared_quality_identity": {
            "corpus_query_qrels": "equivalent_exact_hashes_all_accepted_families",
            "retrieval_unit": "equivalent_canonical_passage",
            "candidate_pool": "equivalent_full_independent_7500_passages_per_track",
            "excluded_ids": "equivalent_pre_ranking_filter; selected revision contains zero excluded IDs",
            "top_k": 100,
            "tie_policy": "descending score then canonical passage ID",
            "metrics": list(METRICS),
            "track_aggregation": "never micro-average across tracks",
        },
        "accepted_family_findings": [
            {
                "family": "lexical",
                "cells": 2,
                "decision": "reuse_quality",
                "source": "S-005 BM25",
                "reason": "Exact full-corpus rankings and per-query metrics match the canonical data identity.",
            },
            {
                "family": "dense",
                "cells": 2,
                "decision": "reuse_quality",
                "source": "S-005 BGE-M3 long_dense",
                "reason": "Exact full-corpus cosine rankings at the frozen 1024-token cap; per-query evidence is present.",
            },
            {
                "family": "learned_sparse",
                "cells": 14,
                "decision": "reuse_quality",
                "source": "S-008",
                "reason": "Exact scipy CSR search, canonical passage IDs, full candidate pools, top-100, and all five metrics match.",
            },
            {
                "family": "multi_vector",
                "cells": 6,
                "decision": "reuse_quality",
                "source": "S-012",
                "reason": "Exact float32 MaxSim is aggregated back to the same canonical passage retrieval unit and full candidate pool.",
            },
        ],
        "method_intrinsic_differences": [
            "Tokenizer and token cap remain method-specific and must be disclosed per method.",
            "Learned sparse uses sparse dot product; dense uses normalized cosine; multi-vector uses exact MaxSim with max-passage window aggregation.",
            "These representation/ranking functions are the methods under test and do not alter qrels, retrieval unit, or candidate visibility.",
        ],
        "rerun_requirements": [
            {
                "method": "tfidf-word-sublinear",
                "cells": 2,
                "reason": "New lexical method with no accepted formal cells.",
            },
            {
                "method": "all-minilm-l6-v2",
                "cells": 2,
                "reason": "Prior complete diagnostic scores cannot be relabeled; rerun under this frozen formal protocol.",
            },
        ],
        "contextual_anchor_only": [
            {
                "scope": "all historical resource measurements",
                "reason": "Hardware/process scope, warmup, trial count, batch fallback, and timing intervals are not fully equivalent.",
            },
            {
                "scope": "S-005 MiniLM diagnostic",
                "reason": "It predates this formal predeclaration and was explicitly accepted only as truncation-degraded diagnostic evidence.",
            },
        ],
        "quality_reuse_cells": 24,
        "quality_rerun_cells": 4,
        "quality_contextual_only_cells": 0,
        "resource_remeasure_methods": 14,
        "publication": publication_policy(),
    }


def _matrix(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    cells = []
    for method in inventory["methods"]:
        for track in TRACKS:
            accepted = method["status"] == "accepted_read_only"
            cells.append(
                {
                    "cell_id": f"{method['key']}:{track}",
                    "method_key": method["key"],
                    "family": method["family"],
                    "track": track,
                    "quality_action": "certify_read_only_reuse" if accepted else "run_after_gate",
                    "formal_status_at_freeze": "accepted_source" if accepted else "planned_not_run",
                    "source_cell_sha256": method.get("accepted_cell_sha256", {}).get(track),
                    "resource_action": "remeasure_unified_harness",
                }
            )
    if len(cells) != 28:
        raise CrossParadigmError("Formal quality matrix must contain exactly 28 cells")
    return cells


def predeclaration(
    *, inventory_sha256: str, audit_sha256: str, preflight_sha256: str, frozen_at: str
) -> dict[str, Any]:
    inventory = read_json(ARTIFACT_ROOT / "inventory.json")
    return {
        "schema_version": "bright-cross-paradigm-predeclaration-v1",
        "story_id": STORY_ID,
        "frozen_at": frozen_at,
        "formal_new_baseline_scores_observed": False,
        "prior_score_visibility": {
            "s005_anchor_scores": True,
            "s008_learned_sparse_scores": True,
            "s012_multi_vector_scores": True,
            "s005_minilm_diagnostic_scores": True,
            "mitigation": "Retain all accepted sparse/multi-vector identities and select new methods only by the frozen non-score criteria.",
        },
        "bindings": {
            "inventory_sha256": inventory_sha256,
            "identity_audit_sha256": audit_sha256,
            "metadata_preflight_sha256": preflight_sha256,
            "accepted_commits": ACCEPTED_COMMITS,
            "canonical_data": DATA_IDENTITY,
        },
        "matrix": {
            "methods": 14,
            "tracks": list(TRACKS),
            "formal_quality_cells": 28,
            "cells": _matrix(inventory),
        },
        "quality_protocol": {
            "candidate_policy": "Exact full independent 7,500-passage pool per track; no candidate generator or approximate index.",
            "retrieval_unit": "Canonical BRIGHT pilot passage; method-intrinsic windows must aggregate to this unit before ranking.",
            "filtering": "Apply query excluded IDs before ranking; fail if the accepted zero-exclusion identity changes.",
            "top_k": 100,
            "tie_policy": "Descending native float32 score, then canonical passage ID ascending.",
            "metrics": list(METRICS),
            "missing_labels": "Unjudged passages remain unjudged and are never treated as confirmed negatives in claims.",
            "track_reporting": "Report economics and psychology separately; no cross-track micro-average or universal winner.",
            "representation_policy": "Keep each frozen method-native tokenizer, route, cap, normalization, scoring, window, and aggregation; disclose them per cell.",
            "recomputation": "Persist restricted top-100 and per-query metrics sufficient for deterministic metric recomputation.",
        },
        "new_method_protocol": {
            "tfidf-word-sublinear": {
                "configuration": inventory["methods"][1]["configuration"],
                "ranking": "Exact scipy CSR cosine from L2-normalized float32 TF-IDF; no pruning or vocabulary cap.",
                "empty_query": "All-zero score vector; canonical passage ID tie order.",
            },
            "all-minilm-l6-v2": {
                "revision": MINILM["revision"],
                "snapshot_sha256": MINILM["snapshot_sha256"],
                "local_files_only": True,
                "trust_remote_code": False,
                "max_length": 256,
                "batch_fallback": {
                    "documents": [128, 64, 32, 16, 8, 4, 2, 1],
                    "queries": [256, 128, 64, 32, 16, 8, 4, 2, 1],
                },
                "dtype": "float32",
                "normalize_embeddings": True,
                "ranking": "Exact cosine through blocked float32 matrix multiplication; no approximate index.",
            },
        },
        "bounded_gate": {
            "must_follow_freeze": True,
            "queries": 1,
            "documents": 4,
            "source": "Project-authored synthetic English facts without benchmark text or identifiers.",
            "selected_methods": ["tfidf-word-sublinear", "all-minilm-l6-v2"],
            "requirements": [
                "frozen artifact sidecars and protected files verify",
                "exactly one query and four documents",
                "all representations and similarities are finite",
                "the predeclared positive is uniquely rank one",
                "dense output is 384-dimensional float32 and L2-normalized",
                "dense model loads from the pinned local snapshot with trust_remote_code=false",
            ],
            "wall_cap_seconds_per_method": 600,
        },
        "statistics": {
            "paired_uncertainty": "paired percentile bootstrap over matched queries",
            "samples": 10000,
            "seed": 20260826,
            "interval": 0.95,
            "comparisons": "All 91 method pairs within each track; never pool tracks.",
            "decisive_rule": "95% interval excludes zero; otherwise tie/uncertain.",
            "small_slice_policy": "n=0 is NA; n=1-9 point estimate only; n>=10 paired intervals; never merge bins post-score.",
        },
        "slices": [
            {"key": "query_length", "bins": ["<=64", "65-128", ">128"], "unit": "lowercase Unicode word tokens"},
            {
                "key": "positive_lexical_overlap",
                "bins": ["0", "(0,0.25]", "(0.25,0.50]", ">0.50"],
                "unit": "max unique-token overlap across positives",
            },
            {
                "key": "positive_passage_length",
                "bins": ["<=128", "129-512", ">512"],
                "unit": "median positive Unicode word-token count",
            },
            {"key": "qrel_density", "bins": ["1", "2-4", ">=5"], "unit": "unique positive passages per query"},
        ],
        "failure_cases": {
            "per_track": {"disagreement": 6, "failure": 6},
            "selection": "After all cells, choose largest rank spread and lowest best-method nDCG@10 contribution with query-hash tie breaks.",
            "content": "Restricted local evidence only; tracked artifacts use opaque case hashes and aggregate descriptions.",
        },
        "resource_protocol": {
            "scope": "Remeasure all 14 methods on both tracks; historical resource figures remain contextual anchors.",
            "hardware": {
                "device": "cuda:0 for neural methods; same host CPU for lexical methods",
                "gpu_uuid": "GPU-435154a8-568c-35a7-ee5f-da29fbf97a39",
                "gpu": "NVIDIA GeForce RTX 3080 Ti",
                "gpu_memory_bytes": 12_884_901_888,
                "driver": "565.57.01",
                "python": "3.12.9",
                "torch": "2.10.0+cu128",
                "transformers": "5.3.0",
                "sentence_transformers": "6.0.1",
                "numpy": "2.4.3",
                "scipy": "1.17.1",
                "scikit_learn": "1.8.0",
            },
            "process_isolation": "Fresh process per method; one track at a time; no concurrent benchmark workload on cuda:0.",
            "warmup": "Run the frozen synthetic 1-query/4-document gate once before measured work; run one unrecorded query search before each measured search series.",
            "encoding_trials": "One complete document encoding and one complete query encoding per method-track, timed separately after warmup.",
            "search_trials": "Three complete ordered query sweeps per method-track; report each trial and median throughput plus pooled P50/P95/P99 per-query latency.",
            "batch_fallback": "Use each accepted method's frozen fallback; MiniLM uses its new-method fallback; log every OOM/retry and final batch.",
            "measurements": [
                "document_encoding",
                "query_encoding",
                "representation_or_index_bytes",
                "build",
                "load",
                "search",
                "throughput",
                "p50",
                "p95",
                "p99",
                "process_peak_rss",
                "cuda_peak_vram",
                "network_bytes",
                "end_to_end",
            ],
            "network": "Zero during measured runs; all snapshots local-only.",
            "comparability": "Compare timing only within the unified rerun; report representation bytes stratified by paradigm without synthetic normalization.",
        },
        "stopping_conditions": [
            "Stop before any formal run if a frozen file, accepted artifact, canonical input, or protected file hash drifts.",
            "Stop if metadata becomes gated/private, license becomes unclear, remote code is required, or an allowlist changes.",
            "Stop a method if its 1-query/4-document gate fails or outputs empty, non-finite, or wrong-shaped representations.",
            "Stop after exhausting the frozen batch fallback on OOM; never reduce token caps, windows, candidate pools, dtype, or metrics silently.",
            "Stop a formal cell at 3600 seconds and a method at 7200 seconds; preserve failure evidence without filling the matrix.",
            "Stop if cuda:0 is materially occupied by an unrelated process at launch; do not terminate or alter that process.",
            "Stop if new downloads would exceed 2 GiB for this Story phase or private artifacts would exceed 12 GiB.",
            "Stop if tracked/public artifacts contain source text, canonical IDs, rankings, reversible mappings, private paths, or secrets.",
            "Never open publication, leaderboard, or export gates in this Story.",
        ],
        "publication": publication_policy(),
    }


def chronology(
    *,
    preflight_sha256: str,
    inventory_sha256: str,
    audit_sha256: str,
    predeclaration_sha256: str,
    preflight_at: str,
    frozen_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": "bright-cross-paradigm-chronology-v1",
        "story_id": STORY_ID,
        "facts": [
            {"at": preflight_at, "event": "download-free metadata preflight completed", "sha256": preflight_sha256},
            {"at": frozen_at, "event": "14-method inventory frozen", "sha256": inventory_sha256},
            {"at": frozen_at, "event": "accepted identity audit frozen", "sha256": audit_sha256},
            {"at": frozen_at, "event": "28-cell protocol and fairness plan frozen", "sha256": predeclaration_sha256},
        ],
        "attestation": {
            "accepted_formal_scores_visible_before_freeze": True,
            "prior_minilm_diagnostic_scores_visible_before_freeze": True,
            "new_tfidf_formal_scores_visible_before_freeze": False,
            "new_minilm_formal_scores_visible_before_freeze": False,
            "any_bounded_gate_run_before_freeze": True,
            "pre_freeze_gate_deviation": (
                "A synthetic TF-IDF 1-query/4-document implementation unit test ran before durable artifact "
                "freeze. It read no benchmark inputs and generated no formal score. The dense gate did not run, "
                "and neither method nor protocol was changed from the already implemented non-score selection."
            ),
            "any_new_formal_score_generated_before_freeze": False,
            "inventory_selected_by_scores": False,
        },
        "next_allowed_action": "Run only the frozen synthetic 1-query/4-document gates.",
        "publication": publication_policy(),
    }


def freeze() -> dict[str, str]:
    preflight_path = ARTIFACT_ROOT / "metadata-preflight.json"
    preflight_sha256 = verify_sidecar(preflight_path)
    preflight = read_json(preflight_path)
    if preflight["decision"] != "pass":
        raise CrossParadigmError("Metadata preflight did not pass")
    assert_foundations()
    frozen_at = utc_now()
    inventory_sha256 = write_once(ARTIFACT_ROOT / "inventory.json", baseline_inventory())
    audit_sha256 = write_once(ARTIFACT_ROOT / "identity-audit.json", identity_audit())
    predeclaration_sha256 = write_once(
        ARTIFACT_ROOT / "predeclaration.json",
        predeclaration(
            inventory_sha256=inventory_sha256,
            audit_sha256=audit_sha256,
            preflight_sha256=preflight_sha256,
            frozen_at=frozen_at,
        ),
    )
    chronology_sha256 = write_once(
        ARTIFACT_ROOT / "chronology.json",
        chronology(
            preflight_sha256=preflight_sha256,
            inventory_sha256=inventory_sha256,
            audit_sha256=audit_sha256,
            predeclaration_sha256=predeclaration_sha256,
            preflight_at=preflight["observed_at"],
            frozen_at=frozen_at,
        ),
    )
    return {
        "metadata_preflight_sha256": preflight_sha256,
        "inventory_sha256": inventory_sha256,
        "identity_audit_sha256": audit_sha256,
        "predeclaration_sha256": predeclaration_sha256,
        "chronology_sha256": chronology_sha256,
    }


def verify_freeze() -> dict[str, str]:
    hashes = {}
    for name in ("metadata-preflight", "inventory", "identity-audit", "predeclaration", "chronology"):
        hashes[name] = verify_sidecar(ARTIFACT_ROOT / f"{name}.json")
    assert_foundations()
    predeclaration_value = read_json(ARTIFACT_ROOT / "predeclaration.json")
    if predeclaration_value["bindings"]["inventory_sha256"] != hashes["inventory"]:
        raise CrossParadigmError("Predeclaration inventory binding drifted")
    if predeclaration_value["bindings"]["identity_audit_sha256"] != hashes["identity-audit"]:
        raise CrossParadigmError("Predeclaration audit binding drifted")
    if len(predeclaration_value["matrix"]["cells"]) != 28:
        raise CrossParadigmError("Frozen matrix no longer contains 28 cells")
    return hashes


SYNTHETIC_QUERY = "Which planet has a ring system made mostly of ice particles?"
SYNTHETIC_DOCUMENTS = (
    "Saturn has a prominent ring system composed mostly of ice particles with some rock and dust.",
    "Mars has two small moons named Phobos and Deimos.",
    "Venus has a dense atmosphere dominated by carbon dioxide.",
    "Mercury is the closest planet to the Sun and has no natural moons.",
)


def tfidf_gate() -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float32,
    )
    documents = vectorizer.fit_transform(SYNTHETIC_DOCUMENTS)
    query = vectorizer.transform([SYNTHETIC_QUERY])
    scores = np.asarray((query @ documents.T).toarray()[0], dtype=np.float32)
    order = sorted(range(4), key=lambda index: (-float(scores[index]), index))
    passed = bool(np.isfinite(scores).all() and order[0] == 0 and scores[0] > scores[order[1]])
    return {
        "method": "tfidf-word-sublinear",
        "queries": 1,
        "documents": 4,
        "shape": [int(documents.shape[0]), int(documents.shape[1])],
        "query_nnz": int(query.nnz),
        "document_nnz": int(documents.nnz),
        "finite": bool(np.isfinite(scores).all()),
        "positive_rank": order.index(0) + 1,
        "unique_rank_one": bool(scores[0] > scores[order[1]]),
        "ranking_sha256": hashlib.sha256(canonical_bytes(order)).hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "passed": passed,
    }


def minilm_gate() -> dict[str, Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    snapshot = minilm_snapshot_path()
    if snapshot_inventory(snapshot)["aggregate_sha256"] != MINILM["snapshot_sha256"]:
        raise CrossParadigmError("MiniLM snapshot drifted before gate")
    started = time.perf_counter()
    model = SentenceTransformer(
        str(snapshot),
        device="cuda:0",
        trust_remote_code=False,
        model_kwargs={"local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
    )
    if model.max_seq_length != 256:
        raise CrossParadigmError("MiniLM max sequence length drifted")
    embeddings = model.encode(
        [SYNTHETIC_QUERY, *SYNTHETIC_DOCUMENTS],
        batch_size=5,
        convert_to_numpy=True,
        normalize_embeddings=True,
        precision="float32",
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    query, documents = embeddings[:1], embeddings[1:]
    scores = np.asarray(query @ documents.T, dtype=np.float32)[0]
    order = sorted(range(4), key=lambda index: (-float(scores[index]), index))
    norms = np.linalg.norm(embeddings, axis=1)
    finite = bool(np.isfinite(embeddings).all() and np.isfinite(scores).all())
    normalized = bool(np.allclose(norms, 1.0, atol=1e-5))
    passed = bool(
        finite and normalized and embeddings.shape == (5, 384) and order[0] == 0 and scores[0] > scores[order[1]]
    )
    peak_vram = int(torch.cuda.max_memory_allocated(0))
    return {
        "method": "all-minilm-l6-v2",
        "queries": 1,
        "documents": 4,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "finite": finite,
        "l2_normalized": normalized,
        "positive_rank": order.index(0) + 1,
        "unique_rank_one": bool(scores[0] > scores[order[1]]),
        "ranking_sha256": hashlib.sha256(canonical_bytes(order)).hexdigest(),
        "peak_vram_bytes": peak_vram,
        "wall_seconds": time.perf_counter() - started,
        "local_files_only": True,
        "trust_remote_code": False,
        "passed": passed,
    }


def run_gates() -> dict[str, Any]:
    frozen = verify_freeze()
    lexical = tfidf_gate()
    dense = minilm_gate()
    if not lexical["passed"] or not dense["passed"]:
        raise CrossParadigmError("At least one bounded gate failed")
    return {
        "schema_version": "bright-cross-paradigm-gate-summary-v1",
        "story_id": STORY_ID,
        "executed_at": utc_now(),
        "freeze_hashes": frozen,
        "scope": "Synthetic readiness only; no benchmark corpus, query, qrel, ranking, metric, or formal score was read or generated.",
        "gates": [lexical, dense],
        "all_passed": True,
        "publication": publication_policy(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("metadata-preflight", "freeze", "check", "gate"))
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    global ARTIFACT_ROOT
    args = build_parser().parse_args(argv)
    ARTIFACT_ROOT = args.output
    if args.command == "metadata-preflight":
        result: object = {
            "metadata_preflight_sha256": write_once(ARTIFACT_ROOT / "metadata-preflight.json", metadata_preflight())
        }
    elif args.command == "freeze":
        result = freeze()
    elif args.command == "check":
        result = verify_freeze()
    else:
        result = {"gate_summary_sha256": write_once(ARTIFACT_ROOT / "gate-summary.json", run_gates())}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
