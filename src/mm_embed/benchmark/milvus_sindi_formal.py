"""Formal research-only Milvus SINDI system benchmark runner.

The module consumes only authenticated saved CSR artifacts. It never loads an
embedding model and never emits canonical source identifiers into tracked data.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import random
import re
import resource
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
from scipy import sparse

from mm_embed.benchmark.milvus_sindi_fixture import (
    CONTAINER_NAME,
    FIXTURE_ALGORITHMS,
    INDEX_NAME,
    _jsonable,
    _validate_recovery_compaction_metrics,
    score_fixture_results,
)
from mm_embed.benchmark.milvus_sindi_system import (
    ACTIVE_PREDECLARATION_SHA256,
    PYMILVUS_VERSION,
    SERVER_IMAGE,
    SOURCE_COMMIT,
    STORY_ID,
    MilvusSindiError,
    canonical_sha256,
    file_sha256,
    formatted_json_bytes,
    validate_deployment_evidence,
    validate_predeclaration,
    validate_source_attestation,
)

FORMAL_ROOT = Path("results/milvus-sindi-system-v0.1/formal-20260831")
SOURCE_CATALOG = Path("results/milvus-sindi-system-v0.1/source-auth/csr-exact-catalog.json")
SOURCE_CATALOG_SHA256 = "65ab0b99186b35fb82a4dbd969eeb8a1ca49f47b6faf733d8f7a7dc7bc5f2528"
SOURCE_CATALOG_FILE_SHA256 = "bb16544edd6469fcb6a4d8fdc54298b61a5209d9754a6cd4c3a3d45b00437840"
SOURCE_ATTESTATION = Path("benchmark/artifacts/milvus-sindi-system-v0.1/source-attestation.json")
SOURCE_ATTESTATION_SHA256 = "dce7e8c088954bfc5d689cea1017f13b088dc417532d5b5a9a312abe6cf46f65"
PREDECLARATION = Path("benchmark/artifacts/milvus-sindi-system-v0.1/predeclaration.json")
FORMAL_GATE = Path("results/milvus-sindi-system-v0.1/verification/formal-gate-phase3-revalidated.json")
FORMAL_GATE_SHA256 = "b4cba2042e70ccd84dba78ada30dfa768d49b4c86ccb5875a81fda4a9a0b5655"
HARDENED_DEPLOYMENT = Path(
    "results/milvus-sindi-system-v0.1/deployment/phase3-hardened-20260830T233700Z/deployment-evidence.json"
)
HARDENED_DEPLOYMENT_SHA256 = "9bf5af3912a707a66593a0cc2fde768fbf138c085b2260cf8ebea409cb86ee55"
REPRESENTATIVES = ("bge-m3", "granite-30m-sparse", "opensearch-multilingual")
TRACKS = ("economics", "psychology")
MODEL_ROOTS = {
    "bge-m3": Path("results/bright-learned-sparse-batch-a"),
    "granite-30m-sparse": Path("results/bright-learned-sparse-batch-a"),
    "opensearch-multilingual": Path("results/bright-learned-sparse-batch-b"),
}
MODEL_SHORT = {
    "bge-m3": "bge",
    "granite-30m-sparse": "granite",
    "opensearch-multilingual": "osm",
}
TRACK_SHORT = {"economics": "econ", "psychology": "psych"}
ALGORITHM_SHORT = {"SINDI": "sindi", "DAAT_MAXSCORE": "daat"}
COLLECTION_PREFIX = "meb_s010_sindi_v01_"
TOP_K_VALUES = (10, 100)
CONCURRENCY_VALUES = (1, 4, 16)
MEASURED_TRIALS = 3
WARMUP_PASSES = 2
QUERY_SEED = 20260830
CELL_SEED = 20260831
CLIENT_CPUSET = frozenset(range(32, 40))
EXACT_CPUSET = frozenset(range(48, 64))
PROCESS_ALLOWED_CPUSET = frozenset(os.sched_getaffinity(0))
SYSTEM_QUERY_COUNT = 256
GENERATOR_SEED = 20260830
GENERATED_SHARD_ROWS = 10_000
RUNTIME_CONFIG = FORMAL_ROOT.parent / "runtime/milvus-v300-amd64/config/user.yaml"
RUNTIME_CONFIG_SHA256 = "85d1c0f0d44167c535a729c2349a53121d20c8d7ba53f5a04e29c5083d7c96a8"
TARGET_VECTOR_INDEX_VERSION = 10
TARGET_SCALAR_INDEX_VERSION = 4


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(formatted_json_bytes(_jsonable(value)))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _write_gzip_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _jsonable(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csr(path: Path) -> sparse.csr_matrix:
    matrix = sparse.load_npz(path).tocsr().astype(np.float32, copy=False)
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise MilvusSindiError(f"Non-finite saved sparse values: {path}")
    if matrix.has_sorted_indices is False:
        matrix.sort_indices()
    return matrix


def _source_root(model: str, track: str) -> Path:
    return MODEL_ROOTS[model] / model / track


def _verify_record(root: Path, relative_path: str, record: Mapping[str, Any]) -> Path:
    path = root / relative_path
    if not path.is_file() or path.stat().st_size != int(record["bytes"]) or file_sha256(path) != record["sha256"]:
        raise MilvusSindiError(f"Authenticated source artifact drifted: {path}")
    return path


def _load_native_cell(model: str, track: str) -> dict[str, Any]:
    if model not in REPRESENTATIVES or track not in TRACKS:
        raise MilvusSindiError("Native workload is outside the predeclared matrix")
    root = _source_root(model, track)
    catalog = _load_json(SOURCE_CATALOG)
    key = f"{model}:{track}"
    cell = catalog["cells"][key]
    manifest_path = root / "manifest.json"
    if file_sha256(manifest_path) != cell["raw_manifest_sha256"]:
        raise MilvusSindiError(f"Native manifest drifted: {key}")
    manifest = _load_json(manifest_path)
    artifacts = manifest["artifacts"]
    query_path = _verify_record(root, "queries.npz", artifacts["queries.npz"])
    ranking_path = _verify_record(root, "rankings.json", artifacts["rankings.json"])
    document_id_path = _verify_record(root, "document_ids.json", artifacts["document_ids.json"])
    query_id_path = _verify_record(root, "query_ids.json", artifacts["query_ids.json"])
    chunk_names = sorted(name for name in artifacts if name.startswith("document_chunks/") and name.endswith(".npz"))
    chunks = [_load_csr(_verify_record(root, name, artifacts[name])) for name in chunk_names]
    documents = sparse.vstack(chunks, format="csr", dtype=np.float32)
    queries = _load_csr(query_path)
    expected_document_shape = tuple(manifest["representations"]["documents"]["shape"])
    expected_query_shape = tuple(manifest["representations"]["queries"]["shape"])
    if documents.shape != expected_document_shape or queries.shape != expected_query_shape:
        raise MilvusSindiError(f"Native CSR shape drifted: {key}")
    document_ids = _load_json(document_id_path)
    query_ids = _load_json(query_id_path)
    rankings = _load_json(ranking_path)
    if len(document_ids) != documents.shape[0] or len(query_ids) != queries.shape[0]:
        raise MilvusSindiError(f"Native private ID cardinality drifted: {key}")
    accepted = []
    document_ordinal = {value: index for index, value in enumerate(document_ids)}
    for query_ordinal, ranking in enumerate(rankings):
        hits = ranking["hits"]
        accepted.append(
            [
                {
                    "id": document_ordinal[hit["document_id"]],
                    "score": float(np.float32(hit["score"])),
                }
                for hit in hits
            ]
        )
        if len(hits) != 100 or query_ordinal >= queries.shape[0]:
            raise MilvusSindiError(f"Accepted exact top-100 cardinality drifted: {key}")
    recomputed = _exact_topk(documents, queries, 100, tie_ids=document_ids)
    mismatch = 0
    maximum_score_delta = 0.0
    for expected, actual in zip(accepted, recomputed, strict=True):
        if [row["id"] for row in expected] != [row["id"] for row in actual]:
            mismatch += 1
        maximum_score_delta = max(
            maximum_score_delta,
            max(abs(float(a["score"]) - float(b["score"])) for a, b in zip(expected, actual, strict=True)),
        )
    if mismatch or maximum_score_delta > 1e-5:
        raise MilvusSindiError(f"Recomputed CSR does not match accepted exact rankings: {key}, {mismatch}")
    ordinal_exact = _exact_topk(documents, queries, 100)
    return {
        "model": model,
        "track": track,
        "documents": documents,
        "queries": queries,
        "exact": ordinal_exact,
        "manifest_sha256": cell["raw_manifest_sha256"],
        "artifact_bytes": sum(int(artifacts[name]["bytes"]) for name in artifacts),
        "document_csr_bytes": _csr_bytes(documents),
        "query_csr_bytes": _csr_bytes(queries),
        "accepted_exact_mismatch_queries": mismatch,
        "accepted_exact_max_score_delta": maximum_score_delta,
        "model_loaded": False,
    }


def _csr_bytes(matrix: sparse.csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _exact_topk(
    documents: sparse.csr_matrix,
    queries: sparse.csr_matrix,
    top_k: int,
    *,
    tie_ids: Sequence[str] | None = None,
) -> list[list[dict[str, Any]]]:
    product = (queries @ documents.transpose()).tocsr()
    results: list[list[dict[str, Any]]] = []
    for query_ordinal in range(product.shape[0]):
        start, end = product.indptr[query_ordinal : query_ordinal + 2]
        pairs = [
            (int(column), float(np.float32(score)))
            for column, score in zip(product.indices[start:end], product.data[start:end], strict=True)
        ]
        if tie_ids is None:
            pairs.sort(key=lambda item: (-item[1], item[0]))
        else:
            pairs.sort(key=lambda item: (-item[1], tie_ids[item[0]]))
        if len(pairs) < top_k:
            raise MilvusSindiError("Exact sparse result has fewer than requested positive hits")
        results.append([{"id": pk, "score": score} for pk, score in pairs[:top_k]])
    return results


def validate_formal_runtime() -> dict[str, Any]:
    """Recheck all immutable authorization inputs and the hardened live runtime."""
    predeclaration = validate_predeclaration(PREDECLARATION, ACTIVE_PREDECLARATION_SHA256)
    source = validate_source_attestation(SOURCE_ATTESTATION, SOURCE_ATTESTATION_SHA256)
    catalog = _load_json(SOURCE_CATALOG)
    if (
        file_sha256(SOURCE_CATALOG) != SOURCE_CATALOG_FILE_SHA256
        or canonical_sha256(catalog["cells"]) != SOURCE_CATALOG_SHA256
    ):
        raise MilvusSindiError("Private CSR/exact catalog identity drifted")
    if file_sha256(FORMAL_GATE) != FORMAL_GATE_SHA256:
        raise MilvusSindiError("Formal gate identity drifted")
    gate = _load_json(FORMAL_GATE)
    deployment = validate_deployment_evidence(HARDENED_DEPLOYMENT, HARDENED_DEPLOYMENT_SHA256)
    if file_sha256(RUNTIME_CONFIG) != RUNTIME_CONFIG_SHA256:
        raise MilvusSindiError("Formal Milvus index-version configuration drifted")
    runtime_config = RUNTIME_CONFIG.read_text(encoding="utf-8")
    if "targetVecIndexVersion: 10" not in runtime_config or "targetScalarIndexVersion: 4" not in runtime_config:
        raise MilvusSindiError("Formal Milvus target index versions are absent")
    runtime_root = Path(predeclaration["isolation"]["data_root"])
    mode = os.stat(runtime_root, follow_symlinks=False).st_mode & 0o7777
    if (
        source["model_loaded"] is not False
        or gate.get("status") != "ready"
        or gate.get("formal_performance_executed") is not False
        or gate.get("publication_gate") != "closed"
        or mode != 0o711
        or deployment["container_id"] != gate["container_id"]
    ):
        raise MilvusSindiError("Formal runtime authorization or hardening drifted")
    return {
        "status": "pass",
        "model_loaded": False,
        "source_commit": SOURCE_COMMIT,
        "predeclaration_sha256": ACTIVE_PREDECLARATION_SHA256,
        "source_attestation_sha256": SOURCE_ATTESTATION_SHA256,
        "source_catalog_sha256": SOURCE_CATALOG_SHA256,
        "source_catalog_file_sha256": SOURCE_CATALOG_FILE_SHA256,
        "formal_gate_sha256": FORMAL_GATE_SHA256,
        "hardened_deployment_sha256": HARDENED_DEPLOYMENT_SHA256,
        "runtime_root_mode": "0711",
        "runtime_config_sha256": RUNTIME_CONFIG_SHA256,
        "target_vector_index_version": TARGET_VECTOR_INDEX_VERSION,
        "target_scalar_index_version": TARGET_SCALAR_INDEX_VERSION,
        "publication_gate": "closed",
        "server_image": SERVER_IMAGE,
        "pymilvus_version": PYMILVUS_VERSION,
    }


def audit_native_sources(output: str | Path) -> tuple[dict[str, Any], str]:
    """Authenticate all six native representative cells without serializing private IDs."""
    runtime = validate_formal_runtime()
    cells: dict[str, Any] = {}
    for model in REPRESENTATIVES:
        for track in TRACKS:
            loaded = _load_native_cell(model, track)
            key = f"{model}:{track}"
            cells[key] = {
                "manifest_sha256": loaded["manifest_sha256"],
                "document_shape": list(loaded["documents"].shape),
                "query_shape": list(loaded["queries"].shape),
                "document_nnz": int(loaded["documents"].nnz),
                "query_nnz": int(loaded["queries"].nnz),
                "document_csr_bytes": loaded["document_csr_bytes"],
                "query_csr_bytes": loaded["query_csr_bytes"],
                "source_artifact_bytes": loaded["artifact_bytes"],
                "accepted_exact_mismatch_queries": 0,
                "accepted_exact_max_score_delta": loaded["accepted_exact_max_score_delta"],
                "model_loaded": False,
            }
    evidence = {
        "schema_version": "milvus-sindi-formal-source-audit-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "captured_at_utc": _now(),
        "runtime": runtime,
        "cells": cells,
        "contains_source_text": False,
        "contains_canonical_ids": False,
        "contains_raw_rankings": False,
        "model_loaded": False,
        "status": "pass",
    }
    path = Path(output)
    if path.exists():
        raise MilvusSindiError("Formal source audit output already exists")
    _write_json(path, evidence)
    identity = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity


def _row_dict(matrix: sparse.csr_matrix, row: int) -> dict[int, float]:
    start, end = matrix.indptr[row : row + 2]
    return {
        int(index): float(value) for index, value in zip(matrix.indices[start:end], matrix.data[start:end], strict=True)
    }


def _collection_name(*, model: str, algorithm: str, track: str | None = None, scale: int | None = None) -> str:
    if track is not None and scale is None:
        return f"{COLLECTION_PREFIX}n_{MODEL_SHORT[model]}_{TRACK_SHORT[track]}_{ALGORITHM_SHORT[algorithm]}"
    if scale is not None and track is None:
        suffix = "100k" if scale == 100_000 else "1m"
        return f"{COLLECTION_PREFIX}s_{MODEL_SHORT[model]}_{suffix}_{ALGORITHM_SHORT[algorithm]}"
    raise MilvusSindiError("Collection name requires exactly one workload tier")


def _set_affinity(cpus: frozenset[int]) -> list[int]:
    if not cpus.issubset(PROCESS_ALLOWED_CPUSET):
        raise MilvusSindiError(f"Required CPU affinity is unavailable: {sorted(cpus)}")
    os.sched_setaffinity(0, cpus)
    return sorted(os.sched_getaffinity(0))


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise MilvusSindiError("Client RSS was unavailable")


def _docker_stats() -> dict[str, Any]:
    command = (
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{json .}}",
        CONTAINER_NAME,
    )
    result = subprocess.run(command, capture_output=True, check=False, timeout=30)
    if result.returncode != 0:
        raise MilvusSindiError("Story Docker stats capture failed")
    return json.loads(result.stdout)


def _du_runtime_bytes() -> int:
    roots = (
        "/var/lib/milvus/config",
        "/var/lib/milvus/data",
        "/var/lib/milvus/etcd",
        "/var/lib/milvus/rdb_data",
        "/var/lib/milvus/rdb_data_meta_kv",
    )
    command = ("docker", "exec", "-u", "999", CONTAINER_NAME, "du", "-sk", *roots)
    result = subprocess.run(command, capture_output=True, check=False, timeout=120)
    if result.returncode != 0:
        raise MilvusSindiError("Story runtime byte capture failed")
    return sum(int(line.split()[0]) * 1024 for line in result.stdout.splitlines())


class _StatsSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append({"captured_at_utc": _now(), "docker": _docker_stats()})
            except (json.JSONDecodeError, MilvusSindiError, OSError, subprocess.SubprocessError, ValueError) as exc:
                # Evidence retains sampler failures explicitly.
                self.samples.append({"captured_at_utc": _now(), "error": str(exc)})
            self._stop.wait(0.5)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=35)
        if self._thread.is_alive():
            raise MilvusSindiError("Docker stats sampler did not terminate")


def _measure_phase(call: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    rss_before = _rss_bytes()
    cpu_before = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    with _StatsSampler() as sampler:
        result = call()
    wall_end = time.perf_counter_ns()
    return result, {
        "wall_ns": wall_end - wall_start,
        "client_cpu_ns": time.process_time_ns() - cpu_before,
        "client_rss_before_bytes": rss_before,
        "client_rss_after_bytes": _rss_bytes(),
        "client_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "docker_samples": sampler.samples,
    }


def _metrics_text() -> str:
    with urllib.request.urlopen("http://127.0.0.1:49092/metrics", timeout=15) as response:
        if response.status != 200:
            raise MilvusSindiError("Story metrics endpoint did not return HTTP 200")
        return response.read().decode("utf-8", errors="strict")


def _selected_metrics(text: str, collection_name: str) -> list[str]:
    names = (
        "milvus_proxy_collection_sq_latency_",
        "milvus_proxy_receive_bytes_count",
        "milvus_proxy_send_bytes_count",
        "milvus_proxy_req_count",
        "milvus_proxy_received_nq",
        "milvus_datanode_pool_active_threads",
        "milvus_datanode_pool_queue_depth",
        "milvus_datanode_slot",
        "milvus_datacoord_compaction_task_num",
    )
    rows = []
    for line in text.splitlines():
        if line.startswith(names) and (
            f'collection_name="{collection_name}"' in line
            or "collection_name=" not in line
            or "CompactionExecPool" in line
            or "compaction" in line.lower()
        ):
            rows.append(line)
    return rows


def _counter(lines: Sequence[str], prefix: str, required: str | None = None) -> float:
    values = []
    for line in lines:
        if line.startswith(prefix) and (required is None or required in line):
            try:
                values.append(float(line.rsplit(" ", 1)[1]))
            except ValueError as exc:
                raise MilvusSindiError(f"Invalid metric sample: {line}") from exc
    if not values:
        return 0.0
    return sum(values)


def _metric_delta(before: Sequence[str], after: Sequence[str]) -> dict[str, Any]:
    server_sum = _counter(after, "milvus_proxy_collection_sq_latency_sum") - _counter(
        before, "milvus_proxy_collection_sq_latency_sum"
    )
    server_count = _counter(after, "milvus_proxy_collection_sq_latency_count") - _counter(
        before, "milvus_proxy_collection_sq_latency_count"
    )
    received = _counter(after, "milvus_proxy_receive_bytes_count", 'msg_type="search"') - _counter(
        before, "milvus_proxy_receive_bytes_count", 'msg_type="search"'
    )
    sent = _counter(after, "milvus_proxy_send_bytes_count") - _counter(before, "milvus_proxy_send_bytes_count")
    return {
        "server_search_sum_ms": server_sum,
        "server_search_count": int(server_count),
        "server_search_mean_ms": server_sum / server_count if server_count else None,
        "proxy_received_bytes": int(received),
        "proxy_sent_bytes_global_single_active_cell": int(sent),
    }


def _schedule(query_count: int, *, family: str) -> list[int]:
    seed = QUERY_SEED + int.from_bytes(hashlib.sha256(family.encode()).digest()[:4], "big")
    order = list(range(query_count))
    random.Random(seed).shuffle(order)
    return order


def _run_milvus_pass(
    *,
    client: Any,
    collection_name: str,
    queries: sparse.csr_matrix,
    top_k: int,
    concurrency: int,
    family: str,
    output_path: Path,
) -> dict[str, Any]:
    order = _schedule(queries.shape[0], family=family)
    before_lines = _selected_metrics(_metrics_text(), collection_name)
    cpu_before = time.process_time_ns()
    rss_before = _rss_bytes()
    wall_start = time.perf_counter_ns()

    def search_one(position_query: tuple[int, int]) -> dict[str, Any]:
        position, query_ordinal = position_query
        end_prepare_start = time.perf_counter_ns()
        vector = _row_dict(queries, query_ordinal)
        prepare_ns = time.perf_counter_ns() - end_prepare_start
        rpc_start = time.perf_counter_ns()
        response = client.search(
            collection_name,
            data=[vector],
            anns_field="sparse_vector",
            limit=top_k,
            search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0}},
            output_fields=["pk"],
            timeout=120,
        )
        rpc_ns = time.perf_counter_ns() - rpc_start
        hits = response[0]
        return {
            "schedule_position": position,
            "query_ordinal": query_ordinal,
            "prepare_ns": prepare_ns,
            "rpc_end_to_end_ns": rpc_ns,
            "end_to_end_ns": prepare_ns + rpc_ns,
            "ids": [int(hit["id"] if "id" in hit else hit["pk"]) for hit in hits],
            "scores": [float(hit["distance"]) for hit in hits],
        }

    with _StatsSampler() as sampler, concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        rows = list(executor.map(search_one, enumerate(order)))
    wall_ns = time.perf_counter_ns() - wall_start
    after_lines = _selected_metrics(_metrics_text(), collection_name)
    if any(len(row["ids"]) != top_k for row in rows):
        raise MilvusSindiError("Milvus returned fewer hits than requested")
    raw_record = _write_gzip_jsonl(output_path, rows)
    return {
        "family": family,
        "schedule_sha256": hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest(),
        "query_count": len(rows),
        "top_k": top_k,
        "concurrency": concurrency,
        "wall_ns": wall_ns,
        "qps": len(rows) / (wall_ns / 1e9),
        "client_cpu_ns": time.process_time_ns() - cpu_before,
        "client_rss_before_bytes": rss_before,
        "client_rss_after_bytes": _rss_bytes(),
        "client_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "server_metrics_before": before_lines,
        "server_metrics_after": after_lines,
        "server_metric_delta": _metric_delta(before_lines, after_lines),
        "docker_samples": sampler.samples,
        "raw_queries": raw_record,
    }


def _run_exact_pass(
    *,
    documents: sparse.csr_matrix,
    queries: sparse.csr_matrix,
    top_k: int,
    concurrency: int,
    family: str,
    output_path: Path,
) -> dict[str, Any]:
    order = _schedule(queries.shape[0], family=family)
    cpu_before = time.process_time_ns()
    rss_before = _rss_bytes()
    wall_start = time.perf_counter_ns()

    def search_one(position_query: tuple[int, int]) -> dict[str, Any]:
        position, query_ordinal = position_query
        start = time.perf_counter_ns()
        product = (queries.getrow(query_ordinal) @ documents.transpose()).tocsr()
        pairs = [
            (int(column), float(np.float32(score))) for column, score in zip(product.indices, product.data, strict=True)
        ]
        pairs.sort(key=lambda item: (-item[1], item[0]))
        hits = pairs[:top_k]
        latency_ns = time.perf_counter_ns() - start
        return {
            "schedule_position": position,
            "query_ordinal": query_ordinal,
            "prepare_ns": 0,
            "search_ns": latency_ns,
            "end_to_end_ns": latency_ns,
            "ids": [row[0] for row in hits],
            "scores": [row[1] for row in hits],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        rows = list(executor.map(search_one, enumerate(order)))
    wall_ns = time.perf_counter_ns() - wall_start
    if any(len(row["ids"]) != top_k for row in rows):
        raise MilvusSindiError("Exact CSR returned fewer hits than requested")
    return {
        "family": family,
        "schedule_sha256": hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest(),
        "query_count": len(rows),
        "top_k": top_k,
        "concurrency": concurrency,
        "wall_ns": wall_ns,
        "qps": len(rows) / (wall_ns / 1e9),
        "client_cpu_ns": time.process_time_ns() - cpu_before,
        "client_rss_before_bytes": rss_before,
        "client_rss_after_bytes": _rss_bytes(),
        "client_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "raw_queries": _write_gzip_jsonl(output_path, rows),
    }


def _host_preflight(*, generated_root: Path | None = None) -> dict[str, Any]:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.split()[0]) * 1024
    statvfs = os.statvfs(FORMAL_ROOT.parent)
    filesystem_free = statvfs.f_bavail * statvfs.f_frsize
    load1 = os.getloadavg()[0]
    runtime_bytes = _du_runtime_bytes()
    generated_bytes = 0
    if generated_root is not None and generated_root.exists():
        generated_bytes = sum(path.stat().st_size for path in generated_root.rglob("*") if path.is_file())
    checks = {
        "host_available_memory_at_least_192gib": meminfo["MemAvailable"] >= 206_158_430_208,
        "load1_at_most_48": load1 <= 48.0,
        "story_filesystem_free_at_least_200gib": filesystem_free >= 214_748_364_800,
        "runtime_bytes_at_most_72gib": runtime_bytes <= 77_309_411_328,
        "generated_bytes_at_most_36gib": generated_bytes <= 38_654_705_664,
        "client_rss_at_most_16gib": _rss_bytes() <= 17_179_869_184,
        "runtime_root_mode_0711": (os.stat(FORMAL_ROOT.parent / "runtime/milvus-v300-amd64").st_mode & 0o7777) == 0o711,
    }
    evidence = {
        "captured_at_utc": _now(),
        "mem_available_bytes": meminfo["MemAvailable"],
        "load1": load1,
        "filesystem_free_bytes": filesystem_free,
        "runtime_bytes": runtime_bytes,
        "generated_bytes": generated_bytes,
        "client_rss_bytes": _rss_bytes(),
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    if evidence["status"] != "pass":
        raise MilvusSindiError(f"Formal host preflight failed: {evidence}")
    return evidence


def _docker_logs_since(since: str) -> str:
    result = subprocess.run(
        ("docker", "logs", "--since", since, CONTAINER_NAME),
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise MilvusSindiError("Story server log capture failed")
    return (result.stdout + result.stderr).decode("utf-8", errors="replace")


def _wait_index(client: Any, collection_name: str, *, build_started_at: str) -> dict[str, Any]:
    observations = []
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        described = _jsonable(client.describe_index(collection_name, INDEX_NAME, timeout=60))
        observations.append({"captured_at_utc": _now(), "describe_index": described})
        if described.get("state") == "Finished":
            return {"observations": observations, "final": described}
        if described.get("state") == "Failed":
            raise MilvusSindiError(f"Index build failed: {collection_name}")
        server_log = _docker_logs_since(build_started_at)
        if "Unsupported sparse inverted index algorithm" in server_log or "failed to build index" in server_log:
            raise MilvusSindiError(f"Server reported an index-build failure: {collection_name}")
        time.sleep(1)
    raise MilvusSindiError(f"Index build timed out: {collection_name}")


def _validate_loaded_index(client: Any, collection_name: str, algorithm: str, expected_rows: int) -> dict[str, Any]:
    state = {
        "describe_collection": _jsonable(client.describe_collection(collection_name, timeout=60)),
        "describe_index": _jsonable(client.describe_index(collection_name, INDEX_NAME, timeout=60)),
        "load_state": _jsonable(client.get_load_state(collection_name, timeout=60)),
        "stats": _jsonable(client.get_collection_stats(collection_name, timeout=60)),
        "count": _jsonable(client.query(collection_name, filter="", output_fields=["count(*)"], timeout=60)),
        "segments": _jsonable(client._get_connection().get_query_segment_info(collection_name, timeout=60)),
    }
    index = state["describe_index"]
    if (
        index.get("inverted_index_algo") != algorithm
        or index.get("index_type") != "SPARSE_INVERTED_INDEX"
        or index.get("metric_type") != "IP"
        or index.get("drop_ratio_build") != "0"
        or index.get("state") != "Finished"
        or int(index.get("total_rows", -1)) != expected_rows
        or int(index.get("indexed_rows", -1)) != expected_rows
        or int(index.get("pending_index_rows", -1)) != 0
        or state["load_state"] != {"state": "Loaded"}
        or int(state["stats"].get("row_count", -1)) != expected_rows
        or state["count"] != [{"count(*)": expected_rows}]
    ):
        raise MilvusSindiError(f"Loaded index state drifted: {collection_name}")
    segments = state["segments"]
    if not segments or any(
        segment.get("state") != "Sealed" or not segment.get("index_name") or int(segment.get("indexID", 0)) <= 0
        for segment in segments
    ):
        raise MilvusSindiError(f"Growing or unindexed segment observed: {collection_name}")
    sealed_rows = sum(int(segment["num_rows"]) for segment in segments)
    if sealed_rows != expected_rows:
        raise MilvusSindiError(f"Sealed segment rows drifted: {collection_name}")
    metrics = _metrics_text().encode()
    compaction = _validate_recovery_compaction_metrics(metrics)
    return {
        **state,
        "reported_algorithm": index["inverted_index_algo"],
        "sealed_rows": sealed_rows,
        "sealed_segment_count": len(segments),
        "growing_segment_count": 0,
        "all_segments_indexed": True,
        "compaction": compaction,
        "selected_metrics": _selected_metrics(metrics.decode(), collection_name),
    }


def _create_and_load_collection(
    *,
    client: Any,
    collection_name: str,
    algorithm: str,
    documents: sparse.csr_matrix,
    cell_root: Path,
) -> dict[str, Any]:
    from pymilvus import DataType

    if not collection_name.startswith(COLLECTION_PREFIX):
        raise MilvusSindiError("Refusing to manage a collection outside the Story namespace")
    retries = []
    if client.has_collection(collection_name, timeout=60):
        existing = _jsonable(client.describe_collection(collection_name, timeout=60))
        retries.append(
            {
                "captured_at_utc": _now(),
                "action": "drop_exact_incomplete_story_collection",
                "collection": collection_name,
                "existing": existing,
            }
        )
        client.drop_collection(collection_name, timeout=120)
    runtime_before = _du_runtime_bytes()
    lifecycle_started_at = _now()
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description="Formal research-only sparse system workload",
    )
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    create_response, create_resource = _measure_phase(
        lambda: client.create_collection(
            collection_name=collection_name,
            schema=schema,
            consistency_level="Strong",
            timeout=120,
        )
    )
    batch_rows = 1000
    batch_records = []

    def insert_all() -> None:
        for start in range(0, documents.shape[0], batch_rows):
            end = min(start + batch_rows, documents.shape[0])
            conversion_start = time.perf_counter_ns()
            rows = [{"pk": row, "sparse_vector": _row_dict(documents, row)} for row in range(start, end)]
            conversion_ns = time.perf_counter_ns() - conversion_start
            insert_start = time.perf_counter_ns()
            response = _jsonable(client.insert(collection_name, rows, timeout=300))
            insert_ns = time.perf_counter_ns() - insert_start
            if int(response.get("insert_count", -1)) != end - start:
                raise MilvusSindiError(f"Insert count drifted: {collection_name}")
            batch_records.append(
                {
                    "start": start,
                    "end": end,
                    "rows": end - start,
                    "conversion_ns": conversion_ns,
                    "insert_rpc_ns": insert_ns,
                    "response": response,
                }
            )

    _, insert_resource = _measure_phase(insert_all)
    flush_response, flush_resource = _measure_phase(lambda: client.flush(collection_name, timeout=600))
    index_params = client.prepare_index_params()
    requested_index = {
        "field_name": "sparse_vector",
        "index_name": INDEX_NAME,
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
        "params": {"inverted_index_algo": algorithm, "drop_ratio_build": 0},
    }
    index_params.add_index(**requested_index)

    def build_index() -> dict[str, Any]:
        response = _jsonable(client.create_index(collection_name, index_params=index_params, timeout=120, sync=False))
        return {
            "response": response,
            "wait": _wait_index(client, collection_name, build_started_at=lifecycle_started_at),
        }

    index_result, index_resource = _measure_phase(build_index)
    load_response, load_resource = _measure_phase(lambda: client.load_collection(collection_name, timeout=7200))
    state = _validate_loaded_index(client, collection_name, algorithm, documents.shape[0])
    collection_id = int(state["describe_collection"]["collection_id"])
    server_log = _docker_logs_since(lifecycle_started_at)
    relevant_log_lines = [
        line
        for line in server_log.splitlines()
        if (
            f"collectionID={collection_id}" in line
            or f"collection={collection_id}" in line
            or f"collection_id:{collection_id}" in line
            or collection_name in line
        )
        and ("index" in line.lower() or "collection" in line.lower() or "segment" in line.lower())
    ]
    joined_log = "\n".join(relevant_log_lines) + "\n"
    log_path = cell_root / "raw/server-index-build.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(joined_log, encoding="utf-8")
    log_record = {"path": str(log_path), "bytes": log_path.stat().st_size, "sha256": file_sha256(log_path)}
    serialized_sizes = [int(value) for value in re.findall(r"serializedSize=(\d+)", joined_log)]
    memory_sizes = [int(value) for value in re.findall(r"memSize=(\d+)", joined_log)]
    if (
        f"currentIndexVersion={TARGET_VECTOR_INDEX_VERSION}" not in joined_log
        or "Successfully build index" not in joined_log
        or not serialized_sizes
        or "failed to build index" in joined_log
        or "Unsupported sparse inverted index algorithm" in joined_log
    ):
        raise MilvusSindiError(f"Persisted index-version evidence is incomplete: {collection_name}")
    runtime_after = _du_runtime_bytes()
    return {
        "retries": retries,
        "collection": collection_name,
        "requested_algorithm": algorithm,
        "requested_index": requested_index,
        "create_response": _jsonable(create_response),
        "insert_batches": _write_gzip_jsonl(cell_root / "raw/insert-batches.jsonl.gz", batch_records),
        "flush_response": _jsonable(flush_response),
        "index_result": index_result,
        "load_response": _jsonable(load_response),
        "state_before_search": state,
        "persisted_index": {
            "target_vector_index_version": TARGET_VECTOR_INDEX_VERSION,
            "server_log": log_record,
            "serialized_bytes": sum(serialized_sizes),
            "memory_bytes": sum(memory_sizes),
            "segment_index_file_count": len(serialized_sizes),
        },
        "phases": {
            "create": create_resource,
            "insert": insert_resource,
            "flush": flush_resource,
            "index_build": index_resource,
            "load": load_resource,
        },
        "runtime_bytes_before": runtime_before,
        "runtime_bytes_after": runtime_after,
        "incremental_runtime_bytes": runtime_after - runtime_before,
    }


def _trial_configs(workload_key: str) -> list[tuple[int, int]]:
    configs = [(top_k, concurrency) for top_k in TOP_K_VALUES for concurrency in CONCURRENCY_VALUES]
    seed = CELL_SEED + int.from_bytes(hashlib.sha256(workload_key.encode()).digest()[:4], "big")
    random.Random(seed).shuffle(configs)
    return configs


def _run_milvus_trials(
    *,
    client: Any,
    collection_name: str,
    algorithm: str,
    queries: sparse.csr_matrix,
    exact: Sequence[Sequence[Mapping[str, Any]]],
    workload_key: str,
    cell_root: Path,
) -> dict[str, Any]:
    configurations = []
    for top_k, concurrency in _trial_configs(workload_key):
        family_base = f"{workload_key}:k{top_k}:c{concurrency}"
        cold_trials = []
        for trial in range(MEASURED_TRIALS):
            client.release_collection(collection_name, timeout=300)
            load_result, load_resource = _measure_phase(lambda: client.load_collection(collection_name, timeout=7200))
            if _jsonable(client.get_load_state(collection_name, timeout=60)) != {"state": "Loaded"}:
                raise MilvusSindiError("Cold trial collection did not load")
            cold_trials.append(
                {
                    "trial": trial,
                    "load_response": _jsonable(load_result),
                    "load_resource": load_resource,
                    "search": _run_milvus_pass(
                        client=client,
                        collection_name=collection_name,
                        queries=queries,
                        top_k=top_k,
                        concurrency=concurrency,
                        family=f"{family_base}:measured",
                        output_path=cell_root / f"trials/cold-k{top_k}-c{concurrency}-t{trial}.jsonl.gz",
                    ),
                }
            )
        warmups = [
            _run_milvus_pass(
                client=client,
                collection_name=collection_name,
                queries=queries,
                top_k=top_k,
                concurrency=concurrency,
                family=f"{family_base}:measured",
                output_path=cell_root / f"trials/warmup-k{top_k}-c{concurrency}-p{warmup}.jsonl.gz",
            )
            for warmup in range(WARMUP_PASSES)
        ]
        warm_trials = [
            {
                "trial": trial,
                "search": _run_milvus_pass(
                    client=client,
                    collection_name=collection_name,
                    queries=queries,
                    top_k=top_k,
                    concurrency=concurrency,
                    family=f"{family_base}:measured",
                    output_path=cell_root / f"trials/warm-k{top_k}-c{concurrency}-t{trial}.jsonl.gz",
                ),
            }
            for trial in range(MEASURED_TRIALS)
        ]
        configurations.append(
            {
                "top_k": top_k,
                "concurrency": concurrency,
                "cold_trials": cold_trials,
                "warmups": warmups,
                "warm_trials": warm_trials,
            }
        )
    final_state = _validate_loaded_index(
        client,
        collection_name,
        algorithm,
        queries.shape[0] * 0 + int(client.get_collection_stats(collection_name)["row_count"]),
    )
    return {"configurations": configurations, "state_after_search": final_state}


def _run_exact_trials(
    *,
    loader: Callable[[], tuple[sparse.csr_matrix, sparse.csr_matrix]],
    documents: sparse.csr_matrix,
    queries: sparse.csr_matrix,
    workload_key: str,
    cell_root: Path,
) -> dict[str, Any]:
    configurations = []
    for top_k, concurrency in _trial_configs(workload_key):
        family_base = f"{workload_key}:k{top_k}:c{concurrency}"
        cold_trials = []
        for trial in range(MEASURED_TRIALS):
            load_start = time.perf_counter_ns()
            cold_documents, cold_queries = loader()
            load_ns = time.perf_counter_ns() - load_start
            cold_trials.append(
                {
                    "trial": trial,
                    "load_ns": load_ns,
                    "document_csr_bytes": _csr_bytes(cold_documents),
                    "query_csr_bytes": _csr_bytes(cold_queries),
                    "search": _run_exact_pass(
                        documents=cold_documents,
                        queries=cold_queries,
                        top_k=top_k,
                        concurrency=concurrency,
                        family=f"{family_base}:measured",
                        output_path=cell_root / f"trials/cold-k{top_k}-c{concurrency}-t{trial}.jsonl.gz",
                    ),
                }
            )
        warmups = [
            _run_exact_pass(
                documents=documents,
                queries=queries,
                top_k=top_k,
                concurrency=concurrency,
                family=f"{family_base}:measured",
                output_path=cell_root / f"trials/warmup-k{top_k}-c{concurrency}-p{warmup}.jsonl.gz",
            )
            for warmup in range(WARMUP_PASSES)
        ]
        warm_trials = [
            {
                "trial": trial,
                "search": _run_exact_pass(
                    documents=documents,
                    queries=queries,
                    top_k=top_k,
                    concurrency=concurrency,
                    family=f"{family_base}:measured",
                    output_path=cell_root / f"trials/warm-k{top_k}-c{concurrency}-t{trial}.jsonl.gz",
                ),
            }
            for trial in range(MEASURED_TRIALS)
        ]
        configurations.append(
            {
                "top_k": top_k,
                "concurrency": concurrency,
                "cold_trials": cold_trials,
                "warmups": warmups,
                "warm_trials": warm_trials,
            }
        )
    return {"configurations": configurations}


def _read_trial(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(record["path"])
    if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
        raise MilvusSindiError(f"Raw trial identity drifted: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _correctness(
    exact: Sequence[Sequence[Mapping[str, Any]]], rows: Sequence[Mapping[str, Any]], top_k: int
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["query_ordinal"]))
    actual = [
        [{"id": pk, "pk": pk, "distance": score} for pk, score in zip(row["ids"], row["scores"], strict=True)]
        for row in ordered
    ]
    score = score_fixture_results([list(query[:top_k]) for query in exact], actual, tolerance=1e-5)
    ranking_difference_count = sum(
        [hit["id"] for hit in expected[:top_k]] != row["ids"] for expected, row in zip(exact, ordered, strict=True)
    )
    relative = []
    for expected, row in zip(exact, ordered, strict=True):
        expected_scores = {int(hit["id"]): float(hit["score"]) for hit in expected}
        for pk, actual_score in zip(row["ids"], row["scores"], strict=True):
            if int(pk) in expected_scores:
                denominator = max(abs(expected_scores[int(pk)]), 1e-12)
                relative.append(abs(float(actual_score) - expected_scores[int(pk)]) / denominator)
    return {
        **score,
        "maximum_relative_score_delta": max(relative, default=math.inf),
        "ranking_difference_query_count": ranking_difference_count,
    }


def _quantile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _reduce_trials(trials: Mapping[str, Any], exact: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    summaries = []
    for config in trials["configurations"]:
        for temperature, group in (
            ("cold", config["cold_trials"]),
            ("warm", config["warm_trials"]),
        ):
            trial_summaries = []
            for trial in group:
                search = trial["search"]
                rows = _read_trial(search["raw_queries"])
                latencies_ms = [float(row["end_to_end_ns"]) / 1e6 for row in rows]
                trial_summaries.append(
                    {
                        "trial": trial["trial"],
                        "qps": search["qps"],
                        "p50_ms": _quantile(latencies_ms, 50),
                        "p95_ms": _quantile(latencies_ms, 95),
                        "p99_ms": _quantile(latencies_ms, 99),
                        "wall_ns": search["wall_ns"],
                        "client_cpu_ns": search["client_cpu_ns"],
                        "client_max_rss_bytes": search["client_max_rss_bytes"],
                        "server_metric_delta": search.get("server_metric_delta"),
                        "correctness": _correctness(exact, rows, config["top_k"]),
                    }
                )
            summaries.append(
                {
                    "top_k": config["top_k"],
                    "concurrency": config["concurrency"],
                    "temperature": temperature,
                    "measured_trials": trial_summaries,
                    "qps_median": _quantile([row["qps"] for row in trial_summaries], 50),
                    "p50_ms_median": _quantile([row["p50_ms"] for row in trial_summaries], 50),
                    "p95_ms_median": _quantile([row["p95_ms"] for row in trial_summaries], 50),
                    "p99_ms_median": _quantile([row["p99_ms"] for row in trial_summaries], 50),
                    "minimum_strict_recall": min(
                        row["correctness"]["minimum_strict_id_recall_at_k"] for row in trial_summaries
                    ),
                    "minimum_tie_aware_recall": min(
                        row["correctness"]["minimum_tie_aware_recall_at_k"] for row in trial_summaries
                    ),
                    "maximum_score_delta": max(
                        row["correctness"]["maximum_absolute_score_delta"] for row in trial_summaries
                    ),
                }
            )
    return summaries


def _cell_manifest_path(cell_root: Path) -> Path:
    return cell_root / "cell-evidence.json"


def _write_cell(cell_root: Path, evidence: Mapping[str, Any]) -> str:
    path = _cell_manifest_path(cell_root)
    if path.exists():
        raise MilvusSindiError(f"Cell evidence already exists: {path}")
    _write_json(path, evidence)
    identity = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return identity


def _completed_cell(cell_root: Path) -> dict[str, Any] | None:
    path = _cell_manifest_path(cell_root)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() and not sidecar.exists():
        return None
    if not path.is_file() or not sidecar.is_file():
        raise MilvusSindiError(f"Incomplete formal cell checkpoint: {cell_root}")
    expected = sidecar.read_text(encoding="ascii").strip()
    if file_sha256(path) != expected:
        raise MilvusSindiError(f"Formal cell checkpoint identity drifted: {cell_root}")
    value = _load_json(path)
    if value.get("status") != "pass" or value.get("formal_performance_executed") is not True:
        raise MilvusSindiError(f"Formal cell checkpoint did not pass: {cell_root}")
    return value


def prepare_native_run(output_root: str | Path) -> tuple[dict[str, Any], str]:
    """Freeze the replayable native cell order before formal results exist."""
    runtime = validate_formal_runtime()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cell-order.json"
    if path.exists():
        expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
        if file_sha256(path) != expected:
            raise MilvusSindiError("Native cell order identity drifted")
        return _load_json(path), expected
    cells = []
    for model in REPRESENTATIVES:
        for track in TRACKS:
            cells.append({"model": model, "track": track, "comparator": "exact_scipy_csr"})
            for algorithm in FIXTURE_ALGORITHMS:
                cells.append({"model": model, "track": track, "comparator": algorithm})
    random.Random(CELL_SEED).shuffle(cells)
    evidence = {
        "schema_version": "milvus-sindi-formal-cell-order-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "created_at_utc": _now(),
        "cell_order_seed": CELL_SEED,
        "query_order_seed": QUERY_SEED,
        "cells": cells,
        "top_k": list(TOP_K_VALUES),
        "concurrency": list(CONCURRENCY_VALUES),
        "warmup_passes": WARMUP_PASSES,
        "measured_trials": MEASURED_TRIALS,
        "runtime": runtime,
        "publication_gate": "closed",
        "status": "frozen_before_formal_results",
    }
    _write_json(path, evidence)
    identity = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity


def _native_workload_root(root: Path, model: str, track: str) -> Path:
    return root / "native" / MODEL_SHORT[model] / TRACK_SHORT[track]


def run_native_matrix(output_root: str | Path) -> dict[str, Any]:
    """Execute or resume the complete predeclared native formal matrix."""
    runtime = validate_formal_runtime()
    preflight = _host_preflight()
    root = Path(output_root)
    order, order_sha256 = prepare_native_run(root)
    from pymilvus import MilvusClient
    from pymilvus import __version__ as pymilvus_version

    if pymilvus_version != PYMILVUS_VERSION:
        raise MilvusSindiError("PyMilvus version drifted before native matrix")
    completed = []
    client = MilvusClient(uri="http://127.0.0.1:49531", dedicated=True, timeout=60)
    try:
        for ordinal, cell in enumerate(order["cells"]):
            model = cell["model"]
            track = cell["track"]
            comparator = cell["comparator"]
            workload_root = _native_workload_root(root, model, track)
            cell_root = workload_root / comparator.lower()
            existing = _completed_cell(cell_root)
            if existing is not None:
                completed.append({"ordinal": ordinal, "path": str(_cell_manifest_path(cell_root)), "resumed": True})
                continue
            cell_preflight = _host_preflight()
            loaded = _load_native_cell(model, track)
            exact_record_path = workload_root / "exact-ground-truth.jsonl.gz"
            if not exact_record_path.exists():
                exact_record = _write_gzip_jsonl(
                    exact_record_path,
                    [
                        {"query_ordinal": query_ordinal, "hits": hits}
                        for query_ordinal, hits in enumerate(loaded["exact"])
                    ],
                )
                exact_record_path.with_suffix(exact_record_path.suffix + ".sha256").write_text(
                    exact_record["sha256"] + "\n", encoding="ascii"
                )
            else:
                exact_record = {
                    "path": str(exact_record_path),
                    "bytes": exact_record_path.stat().st_size,
                    "sha256": file_sha256(exact_record_path),
                }
            if comparator == "exact_scipy_csr":
                _set_affinity(EXACT_CPUSET)

                def source_loader(
                    model: str = model, track: str = track
                ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
                    cold = _load_native_cell(model, track)
                    return cold["documents"], cold["queries"]

                trials = _run_exact_trials(
                    loader=source_loader,
                    documents=loaded["documents"],
                    queries=loaded["queries"],
                    workload_key=f"native:{model}:{track}",
                    cell_root=cell_root,
                )
                lifecycle = {
                    "backend": "scipy_csr_float32_inner_product",
                    "document_csr_bytes": loaded["document_csr_bytes"],
                    "query_csr_bytes": loaded["query_csr_bytes"],
                    "network_bytes": 0,
                    "server_search_ms": 0,
                    "resource_domain": "client_exact_cpuset_48_63",
                }
            else:
                _set_affinity(CLIENT_CPUSET)
                collection_name = _collection_name(model=model, track=track, algorithm=comparator)
                lifecycle = _create_and_load_collection(
                    client=client,
                    collection_name=collection_name,
                    algorithm=comparator,
                    documents=loaded["documents"],
                    cell_root=cell_root,
                )
                trials = _run_milvus_trials(
                    client=client,
                    collection_name=collection_name,
                    algorithm=comparator,
                    queries=loaded["queries"],
                    exact=loaded["exact"],
                    workload_key=f"native:{model}:{track}",
                    cell_root=cell_root,
                )
            summary = _reduce_trials(trials, loaded["exact"])
            evidence = {
                "schema_version": "milvus-sindi-formal-cell-v1",
                "story_id": STORY_ID,
                "classification": "private_research_only",
                "captured_at_utc": _now(),
                "tier": "native",
                "model": model,
                "track": track,
                "scale": loaded["documents"].shape[0],
                "query_count": loaded["queries"].shape[0],
                "dimensions": loaded["documents"].shape[1],
                "comparator": comparator,
                "source_manifest_sha256": loaded["manifest_sha256"],
                "source_commit": SOURCE_COMMIT,
                "model_loaded": False,
                "exact_ground_truth": exact_record,
                "cell_order_sha256": order_sha256,
                "preflight": cell_preflight,
                "lifecycle": lifecycle,
                "trials": trials,
                "summary": summary,
                "formal_performance_executed": True,
                "publication_gate": "closed",
                "status": "pass",
            }
            identity = _write_cell(cell_root, evidence)
            completed.append(
                {
                    "ordinal": ordinal,
                    "path": str(_cell_manifest_path(cell_root)),
                    "sha256": identity,
                    "resumed": False,
                }
            )
    finally:
        client.close()
        _set_affinity(CLIENT_CPUSET)
    result = {
        "status": "pass",
        "runtime": runtime,
        "preflight": preflight,
        "cell_order_sha256": order_sha256,
        "completed_cells": completed,
        "completed_cell_count": len(completed),
        "expected_cell_count": 18,
        "formal_performance_executed": True,
        "publication_gate": "closed",
    }
    if len(completed) != 18:
        raise MilvusSindiError("Native matrix did not complete all 18 comparator cells")
    path = root / "native-run.json"
    if path.exists():
        path.unlink()
    _write_json(path, result)
    identity = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    result["native_run_sha256"] = identity
    return result


def _affine_permutation(dimensions: int, model: str) -> tuple[int, int]:
    seed = GENERATOR_SEED + int.from_bytes(hashlib.sha256(model.encode()).digest()[:4], "big")
    rng = random.Random(seed)
    multiplier = rng.randrange(1, dimensions, 2)
    while math.gcd(multiplier, dimensions) != 1:
        multiplier = (multiplier + 2) % dimensions or 1
    return multiplier, rng.randrange(dimensions)


def _permute_csr(matrix: sparse.csr_matrix, multiplier: int, offset: int) -> sparse.csr_matrix:
    result = matrix.copy()
    result.indices = ((result.indices.astype(np.int64) * multiplier + offset) % matrix.shape[1]).astype(np.int32)
    result.sort_indices()
    return result


def _row_signature(matrix: sparse.csr_matrix, row: int) -> bytes:
    start, end = matrix.indptr[row : row + 2]
    digest = hashlib.sha256()
    digest.update(matrix.indices[start:end].tobytes())
    digest.update(matrix.data[start:end].tobytes())
    return digest.digest()


def _load_profile_pool(model: str) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str]]:
    cells = [_load_native_cell(model, track) for track in TRACKS]
    documents = sparse.vstack([cell["documents"] for cell in cells], format="csr", dtype=np.float32)
    queries = sparse.vstack([cell["queries"] for cell in cells], format="csr", dtype=np.float32)
    identities = [cell["manifest_sha256"] for cell in cells]
    return documents, queries, identities


def _system_paths(root: Path, model: str, scale: int) -> Path:
    suffix = "100k" if scale == 100_000 else "1m"
    return root / "system-only" / MODEL_SHORT[model] / suffix


def generate_system_workload(output_root: str | Path, *, model: str, scale: int) -> tuple[dict[str, Any], str]:
    """Generate one deterministic system-only sparse workload without dense vectors."""
    if model not in REPRESENTATIVES or scale not in (100_000, 1_000_000):
        raise MilvusSindiError("System-only workload is outside the predeclared matrix")
    validate_formal_runtime()
    root = _system_paths(Path(output_root), model, scale) / "generated"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        expected = manifest_path.with_suffix(".json.sha256").read_text(encoding="ascii").strip()
        if file_sha256(manifest_path) != expected:
            raise MilvusSindiError("Generated workload manifest identity drifted")
        return _load_json(manifest_path), expected
    root.mkdir(parents=True, exist_ok=True)
    preflight = _host_preflight(generated_root=Path(output_root) / "system-only")
    started = time.perf_counter_ns()
    source_documents, source_queries, source_identities = _load_profile_pool(model)
    dimensions = source_documents.shape[1]
    multiplier, offset = _affine_permutation(dimensions, model)
    source_documents = _permute_csr(source_documents, multiplier, offset)
    source_queries = _permute_csr(source_queries, multiplier, offset)
    rng = np.random.default_rng(GENERATOR_SEED)
    document_order = rng.permutation(source_documents.shape[0]).astype(np.int64)
    selected_source = document_order[np.arange(scale, dtype=np.int64) % document_order.size]
    source_signatures = [_row_signature(source_documents, row) for row in range(source_documents.shape[0])]
    totals: dict[bytes, int] = {}
    for source_row in selected_source:
        signature = source_signatures[int(source_row)]
        totals[signature] = totals.get(signature, 0) + 1
    occurrences: dict[bytes, int] = {}
    query_order = rng.permutation(source_queries.shape[0]).astype(np.int64)
    query_rows = []
    for ordinal in range(SYSTEM_QUERY_COUNT):
        source_row = int(query_order[ordinal % query_order.size])
        row = source_queries.getrow(source_row).copy()
        repeat = ordinal // query_order.size
        factor = np.float32(0.999 + 0.002 * (repeat + 1) / (math.ceil(SYSTEM_QUERY_COUNT / query_order.size) + 1))
        row.data *= factor
        query_rows.append(row)
    queries = sparse.vstack(query_rows, format="csr", dtype=np.float32)
    sparse.save_npz(root / "queries.npz", queries, compressed=True)
    query_record = {
        "path": str(root / "queries.npz"),
        "bytes": (root / "queries.npz").stat().st_size,
        "sha256": file_sha256(root / "queries.npz"),
    }
    exact_candidates: list[dict[int, float]] = [{} for _ in range(SYSTEM_QUERY_COUNT)]
    unique_signatures: set[bytes] = set()
    shard_records = []
    total_nnz = 0
    for start in range(0, scale, GENERATED_SHARD_ROWS):
        end = min(start + GENERATED_SHARD_ROWS, scale)
        rows = []
        for ordinal in range(start, end):
            source_row = int(selected_source[ordinal])
            signature = source_signatures[source_row]
            rank = occurrences.get(signature, 0)
            occurrences[signature] = rank + 1
            factor = np.float32(0.999 + 0.002 * (rank + 1) / (totals[signature] + 1))
            row = source_documents.getrow(source_row).copy()
            row.data *= factor
            generated_signature = _row_signature(row, 0)
            if generated_signature in unique_signatures:
                raise MilvusSindiError("Generated sparse document vector was not unique")
            unique_signatures.add(generated_signature)
            rows.append(row)
        shard = sparse.vstack(rows, format="csr", dtype=np.float32)
        total_nnz += shard.nnz
        shard_path = root / "document-shards" / f"{start:07d}.npz"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(shard_path, shard, compressed=True)
        shard_records.append(
            {
                "start": start,
                "end": end,
                "rows": end - start,
                "nnz": int(shard.nnz),
                "path": str(shard_path),
                "bytes": shard_path.stat().st_size,
                "sha256": file_sha256(shard_path),
            }
        )
        product = (queries @ shard.transpose()).tocsr()
        for query_ordinal in range(SYSTEM_QUERY_COUNT):
            row_start, row_end = product.indptr[query_ordinal : query_ordinal + 2]
            candidates = exact_candidates[query_ordinal]
            for local_id, score in zip(
                product.indices[row_start:row_end],
                product.data[row_start:row_end],
                strict=True,
            ):
                candidates[start + int(local_id)] = float(np.float32(score))
            if len(candidates) > 1000:
                best = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:100]
                exact_candidates[query_ordinal] = dict(best)
    exact = []
    for candidates in exact_candidates:
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:100]
        if len(ranked) != 100:
            raise MilvusSindiError("Generated exact ranking has fewer than 100 positive hits")
        exact.append([{"id": pk, "score": score} for pk, score in ranked])
    exact_record = _write_gzip_jsonl(
        root / "exact-top100.jsonl.gz",
        [{"query_ordinal": query_ordinal, "hits": hits} for query_ordinal, hits in enumerate(exact)],
    )
    generated_bytes = sum(record["bytes"] for record in shard_records) + query_record["bytes"]
    generation_ns = time.perf_counter_ns() - started
    manifest = {
        "schema_version": "milvus-sindi-system-only-workload-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only_system_only",
        "model_profile": model,
        "source_manifest_sha256": source_identities,
        "source_commit": SOURCE_COMMIT,
        "scale": scale,
        "query_count": SYSTEM_QUERY_COUNT,
        "dimensions": dimensions,
        "document_nnz": int(total_nnz),
        "query_nnz": int(queries.nnz),
        "mean_document_nnz": total_nnz / scale,
        "mean_query_nnz": queries.nnz / SYSTEM_QUERY_COUNT,
        "generator": {
            "seed": GENERATOR_SEED,
            "method": "deterministic_empirical_csr_row_resampling_global_affine_dimension_permutation_float32_row_scaling",
            "dimension_permutation": {"multiplier": multiplier, "offset": offset},
            "row_scale_interval": [0.999, 1.001],
            "dense_vector_materialization": False,
            "bounded_sparse_score_buffer_only": True,
            "unique_document_vectors": len(unique_signatures) == scale,
        },
        "queries": query_record,
        "document_shards": shard_records,
        "exact_top100": exact_record,
        "generated_input_bytes": generated_bytes,
        "generation_ns": generation_ns,
        "preflight": preflight,
        "model_loaded": False,
        "quality_claims_allowed": False,
        "publication_gate": "closed",
        "status": "pass",
    }
    _write_json(manifest_path, manifest)
    identity = file_sha256(manifest_path)
    manifest_path.with_suffix(".json.sha256").write_text(identity + "\n", encoding="ascii")
    return manifest, identity


def _load_generated(
    output_root: Path, model: str, scale: int
) -> tuple[dict[str, Any], sparse.csr_matrix, sparse.csr_matrix, list[list[dict[str, Any]]]]:
    root = _system_paths(output_root, model, scale) / "generated"
    manifest_path = root / "manifest.json"
    sidecar = manifest_path.with_suffix(".json.sha256")
    if not manifest_path.is_file() or not sidecar.is_file():
        raise MilvusSindiError("Generated workload manifest is absent")
    expected = sidecar.read_text(encoding="ascii").strip()
    if file_sha256(manifest_path) != expected:
        raise MilvusSindiError("Generated workload manifest identity drifted")
    manifest = _load_json(manifest_path)
    shards = []
    for record in manifest["document_shards"]:
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise MilvusSindiError(f"Generated shard identity drifted: {path}")
        shards.append(_load_csr(path))
    documents = sparse.vstack(shards, format="csr", dtype=np.float32)
    query_record = manifest["queries"]
    query_path = Path(query_record["path"])
    if query_path.stat().st_size != query_record["bytes"] or file_sha256(query_path) != query_record["sha256"]:
        raise MilvusSindiError("Generated query identity drifted")
    queries = _load_csr(query_path)
    exact_record = manifest["exact_top100"]
    exact_rows = _read_trial(exact_record)
    exact = [row["hits"] for row in sorted(exact_rows, key=lambda row: row["query_ordinal"])]
    if (
        documents.shape != (scale, manifest["dimensions"])
        or queries.shape != (SYSTEM_QUERY_COUNT, manifest["dimensions"])
        or documents.nnz != manifest["document_nnz"]
        or queries.nnz != manifest["query_nnz"]
        or len(exact) != SYSTEM_QUERY_COUNT
    ):
        raise MilvusSindiError("Generated workload materialization drifted")
    return manifest, documents, queries, exact


def _prepare_system_order(output_root: Path, scale: int) -> tuple[dict[str, Any], str]:
    root = output_root / "system-only" / ("100k" if scale == 100_000 else "1m")
    path = root / "cell-order.json"
    if path.exists():
        expected = path.with_suffix(".json.sha256").read_text(encoding="ascii").strip()
        if file_sha256(path) != expected:
            raise MilvusSindiError("System cell order identity drifted")
        return _load_json(path), expected
    cells = []
    for model in REPRESENTATIVES:
        cells.append({"model": model, "comparator": "exact_scipy_csr"})
        for algorithm in FIXTURE_ALGORITHMS:
            cells.append({"model": model, "comparator": algorithm})
    random.Random(CELL_SEED + scale).shuffle(cells)
    evidence = {
        "schema_version": "milvus-sindi-system-cell-order-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only_system_only",
        "created_at_utc": _now(),
        "scale": scale,
        "cell_order_seed": CELL_SEED + scale,
        "query_order_seed": QUERY_SEED,
        "cells": cells,
        "top_k": list(TOP_K_VALUES),
        "concurrency": list(CONCURRENCY_VALUES),
        "warmup_passes": WARMUP_PASSES,
        "measured_trials": MEASURED_TRIALS,
        "publication_gate": "closed",
        "status": "frozen_before_scale_results",
    }
    _write_json(path, evidence)
    identity = file_sha256(path)
    path.with_suffix(".json.sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity


def run_system_matrix(output_root: str | Path, *, scale: int) -> dict[str, Any]:
    """Generate and execute or resume one complete system-only scale."""
    if scale not in (100_000, 1_000_000):
        raise MilvusSindiError("System scale is not predeclared")
    runtime = validate_formal_runtime()
    root = Path(output_root)
    if scale == 1_000_000:
        cap_path = root / "system-only/one-million-cap.json"
        if not cap_path.is_file():
            raise MilvusSindiError("1M execution requires a saved cap audit")
        cap = _load_json(cap_path)
        if cap.get("decision") != "run_1m" or cap.get("status") != "pass":
            raise MilvusSindiError("Saved cap audit does not authorize 1M")
    preflight = _host_preflight(generated_root=root / "system-only")
    for model in REPRESENTATIVES:
        generate_system_workload(root, model=model, scale=scale)
    order, order_sha256 = _prepare_system_order(root, scale)
    from pymilvus import MilvusClient

    completed = []
    client = MilvusClient(uri="http://127.0.0.1:49531", dedicated=True, timeout=60)
    try:
        for ordinal, cell in enumerate(order["cells"]):
            model = cell["model"]
            comparator = cell["comparator"]
            workload_root = _system_paths(root, model, scale)
            cell_root = workload_root / comparator.lower()
            existing = _completed_cell(cell_root)
            if existing is not None:
                completed.append({"ordinal": ordinal, "path": str(_cell_manifest_path(cell_root)), "resumed": True})
                continue
            cell_preflight = _host_preflight(generated_root=root / "system-only")
            manifest, documents, queries, exact = _load_generated(root, model, scale)
            if comparator == "exact_scipy_csr":
                _set_affinity(EXACT_CPUSET)

                def generated_loader(model: str = model) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
                    _, cold_documents, cold_queries, _ = _load_generated(root, model, scale)
                    return cold_documents, cold_queries

                trials = _run_exact_trials(
                    loader=generated_loader,
                    documents=documents,
                    queries=queries,
                    workload_key=f"system:{model}:{scale}",
                    cell_root=cell_root,
                )
                lifecycle = {
                    "backend": "scipy_csr_float32_inner_product",
                    "document_csr_bytes": _csr_bytes(documents),
                    "query_csr_bytes": _csr_bytes(queries),
                    "generated_input_bytes": manifest["generated_input_bytes"],
                    "network_bytes": 0,
                    "server_search_ms": 0,
                    "resource_domain": "client_exact_cpuset_48_63",
                }
            else:
                _set_affinity(CLIENT_CPUSET)
                collection_name = _collection_name(model=model, scale=scale, algorithm=comparator)
                lifecycle = _create_and_load_collection(
                    client=client,
                    collection_name=collection_name,
                    algorithm=comparator,
                    documents=documents,
                    cell_root=cell_root,
                )
                trials = _run_milvus_trials(
                    client=client,
                    collection_name=collection_name,
                    algorithm=comparator,
                    queries=queries,
                    exact=exact,
                    workload_key=f"system:{model}:{scale}",
                    cell_root=cell_root,
                )
            summary = _reduce_trials(trials, exact)
            evidence = {
                "schema_version": "milvus-sindi-formal-cell-v1",
                "story_id": STORY_ID,
                "classification": "private_research_only_system_only",
                "captured_at_utc": _now(),
                "tier": "system_only",
                "model_profile": model,
                "scale": scale,
                "query_count": SYSTEM_QUERY_COUNT,
                "dimensions": manifest["dimensions"],
                "comparator": comparator,
                "generated_manifest_sha256": file_sha256(_system_paths(root, model, scale) / "generated/manifest.json"),
                "source_commit": SOURCE_COMMIT,
                "model_loaded": False,
                "cell_order_sha256": order_sha256,
                "preflight": cell_preflight,
                "lifecycle": lifecycle,
                "trials": trials,
                "summary": summary,
                "quality_claims_allowed": False,
                "formal_performance_executed": True,
                "publication_gate": "closed",
                "status": "pass",
            }
            identity = _write_cell(cell_root, evidence)
            completed.append(
                {
                    "ordinal": ordinal,
                    "path": str(_cell_manifest_path(cell_root)),
                    "sha256": identity,
                    "resumed": False,
                }
            )
            del documents, queries, exact
    finally:
        client.close()
        _set_affinity(CLIENT_CPUSET)
    result = {
        "status": "pass",
        "tier": "system_only",
        "scale": scale,
        "runtime": runtime,
        "preflight": preflight,
        "cell_order_sha256": order_sha256,
        "completed_cells": completed,
        "completed_cell_count": len(completed),
        "expected_cell_count": 9,
        "quality_claims_allowed": False,
        "formal_performance_executed": True,
        "publication_gate": "closed",
    }
    if len(completed) != 9:
        raise MilvusSindiError("System matrix did not complete all nine comparator cells")
    name = "system-100k-run.json" if scale == 100_000 else "system-1m-run.json"
    path = root / name
    if path.exists():
        path.unlink()
    _write_json(path, result)
    identity = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    result["run_sha256"] = identity
    return result


def audit_one_million_cap(output_root: str | Path) -> tuple[dict[str, Any], str]:
    """Apply the immutable 1M projection formula to completed 100k measurements."""
    root = Path(output_root)
    run_path = root / "system-100k-run.json"
    if not run_path.is_file():
        raise MilvusSindiError("1M cap audit requires a completed 100k run")
    run = _load_json(run_path)
    if run.get("status") != "pass" or run.get("completed_cell_count") != 9:
        raise MilvusSindiError("100k system matrix is incomplete")
    current = _host_preflight(generated_root=root / "system-only")
    projections = []
    total_projected_runtime = current["runtime_bytes"]
    for model in REPRESENTATIVES:
        generated_100k = _load_json(_system_paths(root, model, 100_000) / "generated/manifest.json")
        for algorithm in FIXTURE_ALGORITHMS:
            cell_path = _system_paths(root, model, 100_000) / algorithm.lower() / "cell-evidence.json"
            cell = _load_json(cell_path)
            observed = max(int(cell["lifecycle"]["incremental_runtime_bytes"]), 0)
            projected = math.ceil(1.5 * 10 * observed)
            projections.append(
                {
                    "model_profile": model,
                    "algorithm": algorithm,
                    "observed_100k_incremental_runtime_bytes": observed,
                    "projected_1m_dedicated_bytes": projected,
                    "under_60gib_policy_limit": projected <= 64_424_509_440,
                }
            )
            total_projected_runtime += projected
        projected_input = 10 * int(generated_100k["generated_input_bytes"])
        projections.append(
            {
                "model_profile": model,
                "algorithm": "generated_input",
                "observed_100k_generated_bytes": generated_100k["generated_input_bytes"],
                "projected_1m_generated_bytes": projected_input,
                "under_36gib_global_input_cap_component": projected_input <= 38_654_705_664,
            }
        )
    projected_generated_total = sum(
        row["projected_1m_generated_bytes"] for row in projections if row["algorithm"] == "generated_input"
    )
    checks = {
        "100k_completed": True,
        "each_algorithm_projection_at_most_60gib": all(
            row.get("under_60gib_policy_limit", True) for row in projections
        ),
        "projected_generated_input_at_most_36gib": projected_generated_total <= 38_654_705_664,
        "projected_total_runtime_at_most_72gib": total_projected_runtime <= 77_309_411_328,
        "filesystem_free_after_projection_at_least_200gib": current["filesystem_free_bytes"]
        - projected_generated_total
        - max(total_projected_runtime - current["runtime_bytes"], 0)
        >= 214_748_364_800,
        "host_preflight_passed": current["status"] == "pass",
    }
    decision = "run_1m" if all(checks.values()) else "stop_at_100k"
    evidence = {
        "schema_version": "milvus-sindi-one-million-cap-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only_system_only",
        "captured_at_utc": _now(),
        "projection_formula": "1.5 * 10 * observed 100k dedicated bytes for the same profile and algorithm",
        "projections": projections,
        "projected_generated_total_bytes": projected_generated_total,
        "projected_total_runtime_bytes": total_projected_runtime,
        "current": current,
        "checks": checks,
        "decision": decision,
        "highest_completed_scale": 100_000,
        "extrapolation_allowed": False,
        "publication_gate": "closed",
        "status": "pass" if all(checks.values()) else "stop",
    }
    path = root / "system-only/one-million-cap.json"
    if path.exists():
        raise MilvusSindiError("1M cap audit output already exists")
    _write_json(path, evidence)
    identity = file_sha256(path)
    path.with_suffix(".json.sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity
