"""Isolated deployment and correctness fixture for the Milvus SINDI benchmark."""

from __future__ import annotations

import enum
import json
import math
import os
import re
import stat
import subprocess
import time
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from mm_embed.benchmark.milvus_sindi_system import (
    PYMILVUS_VERSION,
    SERVER_IMAGE,
    SERVER_IMAGE_INDEX_SHA256,
    SOURCE_COMMIT,
    STORY_ID,
    MilvusSindiError,
    file_sha256,
    formatted_json_bytes,
    validate_predeclaration,
)

CONTAINER_NAME = "meb-s010-milvus-v300-amd64"
NETWORK_NAME = "meb-s010-sindi-net-v01"
COLLECTION_PREFIX = "meb_s010_sindi_v01_"
IMAGE_ID = "sha256:b0dfe54d1e107bbb6b345d50b5133f3486b53d4ed3f56ba812bb10b07246186f"
SERVER_CPUSET = "48-63"
SERVER_MEMORY_BYTES = 51_539_607_552
SERVER_PIDS_LIMIT = 4096
FIXTURE_DIMENSIONS = 8
FIXTURE_TOP_K = 5
FIXTURE_ALGORITHMS = ("SINDI", "DAAT_MAXSCORE")
INDEX_NAME = "sparse_ip_idx"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "DESCRIPTOR") and hasattr(value, "ListFields"):
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(value, preserving_proto_field_name=True)
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if type(value).__name__ == "RepeatedCompositeContainer" or hasattr(value, "__iter__"):
        return [_jsonable(item) for item in value]
    return str(value)


def _run_json(command: Sequence[str]) -> Any:
    result = subprocess.run(command, check=False, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise MilvusSindiError(f"Deployment evidence command failed: {command[0]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MilvusSindiError(f"Deployment evidence command returned invalid JSON: {command[0]}") from exc


def _record(root: Path, relative_path: str, value: Any) -> dict[str, Any]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(formatted_json_bytes(_jsonable(value)))
    return {"path": relative_path, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _record_bytes(root: Path, relative_path: str, value: bytes) -> dict[str, Any]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"path": relative_path, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _read_record(root: Path, record: Mapping[str, Any]) -> Any:
    path = root / record["path"]
    if not path.is_file() or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
        raise MilvusSindiError(f"Raw evidence identity drifted: {record['path']}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MilvusSindiError(f"Raw evidence is invalid JSON: {record['path']}") from exc


def _read_bytes_record(root: Path, record: Mapping[str, Any]) -> bytes:
    path = root / record["path"]
    if not path.is_file() or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
        raise MilvusSindiError(f"Raw evidence identity drifted: {record['path']}")
    return path.read_bytes()


def _metric_values(text: str, metric: str, required_labels: Mapping[str, str]) -> list[float]:
    values = []
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{(?P<labels>[^}}]*)\}})?\s+(?P<value>\S+)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group("labels") or ""))
        if all(labels.get(key) == value for key, value in required_labels.items()):
            sample = float(match.group("value"))
            if not math.isfinite(sample):
                raise MilvusSindiError(f"Non-finite server metric: {metric}")
            values.append(sample)
    if not values:
        raise MilvusSindiError(f"Required server metric was absent: {metric}")
    return values


def _validate_compaction_metrics(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="strict")
    summary = {
        "active_threads": sum(
            _metric_values(
                text,
                "milvus_datanode_pool_active_threads",
                {"pool_name": "CompactionExecPool"},
            )
        ),
        "queue_depth": sum(
            _metric_values(
                text,
                "milvus_datanode_pool_queue_depth",
                {"pool_name": "CompactionExecPool"},
            )
        ),
        "used_slots": sum(_metric_values(text, "milvus_datanode_slot", {"type": "compactionUsed"})),
        "pending_tasks": sum(_metric_values(text, "milvus_datacoord_compaction_task_num", {"status": "pending"})),
        "executing_tasks_net": sum(
            _metric_values(text, "milvus_datacoord_compaction_task_num", {"status": "executing"})
        ),
    }
    if any(value != 0 for value in summary.values()):
        raise MilvusSindiError(f"Active or queued compaction observed: {summary}")
    return {"active": False, **summary}


def _validate_recovery_compaction_metrics(raw: bytes) -> dict[str, Any]:
    """Require the post-restart gauges that directly expose active and queued work."""
    text = raw.decode("utf-8", errors="strict")
    active_threads = sum(
        _metric_values(
            text,
            "milvus_datanode_pool_active_threads",
            {"pool_name": "CompactionExecPool"},
        )
    )
    queue_depth = sum(
        _metric_values(
            text,
            "milvus_datanode_pool_queue_depth",
            {"pool_name": "CompactionExecPool"},
        )
    )
    if active_threads != 0 or queue_depth != 0:
        raise MilvusSindiError("Active or queued compaction observed after restart")
    return {
        "active": False,
        "active_threads": active_threads,
        "queue_depth": queue_depth,
        "direct_gauges_required": [
            "milvus_datanode_pool_active_threads",
            "milvus_datanode_pool_queue_depth",
        ],
        "optional_slot_gauge_present": "milvus_datanode_slot" in text,
        "optional_datacoord_task_gauge_present": "milvus_datacoord_compaction_task_num" in text,
    }


def _capture_compaction_metrics(
    root: Path, relative_path: str = "raw/server-metrics-after-fixture.prom"
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:49092/metrics", timeout=15) as response:
            if response.status != 200:
                raise MilvusSindiError("Milvus metrics endpoint did not return HTTP 200")
            raw = response.read()
    except OSError as exc:
        raise MilvusSindiError("Milvus metrics endpoint was unavailable") from exc
    record = _record_bytes(root, relative_path, raw)
    return record, _validate_compaction_metrics(raw)


def capture_deployment_evidence(
    output: str | Path,
    *,
    predeclaration_path: str | Path,
    predeclaration_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Capture and validate only the Story-owned Docker deployment."""
    predeclaration = validate_predeclaration(predeclaration_path, predeclaration_sha256)
    root = Path(output)
    if root.exists():
        raise MilvusSindiError("Deployment evidence output must not already exist")
    root.mkdir(parents=True)
    image_raw = _run_json(("docker", "image", "inspect", SERVER_IMAGE))
    container_raw = _run_json(("docker", "inspect", CONTAINER_NAME))
    network_raw = _run_json(("docker", "network", "inspect", NETWORK_NAME))
    raw = {
        "image_inspect": _record(root, "raw/image-inspect.json", image_raw),
        "container_inspect": _record(root, "raw/container-inspect.json", container_raw),
        "network_inspect": _record(root, "raw/network-inspect.json", network_raw),
    }
    try:
        with urllib.request.urlopen("http://127.0.0.1:49092/healthz", timeout=10) as response:
            health = {"status": response.status, "body": response.read().decode("utf-8", errors="replace")}
    except OSError as exc:
        raise MilvusSindiError("Story-owned Milvus health endpoint was unavailable") from exc
    raw["health"] = _record(root, "raw/health.json", health)

    if len(image_raw) != 1 or len(container_raw) != 1 or len(network_raw) != 1:
        raise MilvusSindiError("Docker inspect did not return one unique Story object")
    image = image_raw[0]
    container = container_raw[0]
    network = network_raw[0]
    repo_digests = set(image["RepoDigests"])
    expected_digests = {
        f"milvusdb/milvus@sha256:{SERVER_IMAGE_INDEX_SHA256}",
        SERVER_IMAGE,
    }
    if (
        image["Id"] != IMAGE_ID
        or image["Architecture"] != "amd64"
        or image["Os"] != "linux"
        or not expected_digests.issubset(repo_digests)
    ):
        raise MilvusSindiError("Pulled image identity or platform drifted")

    isolation = predeclaration["isolation"]
    host = container["HostConfig"]
    if container["Name"] != f"/{CONTAINER_NAME}" or container["Config"]["Image"] != SERVER_IMAGE:
        raise MilvusSindiError("Story container identity drifted")
    if not container["State"]["Running"] or container["State"]["OOMKilled"]:
        raise MilvusSindiError("Story container is not running cleanly")
    if (
        host["Memory"] != SERVER_MEMORY_BYTES
        or host["MemorySwap"] != SERVER_MEMORY_BYTES
        or host["CpusetCpus"] != SERVER_CPUSET
        or host["PidsLimit"] != SERVER_PIDS_LIMIT
        or host["RestartPolicy"]["Name"] != "no"
    ):
        raise MilvusSindiError("Story container resource controls drifted")
    expected_ports = {
        "19530/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(isolation["host_ports"]["grpc"])}],
        "9091/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(isolation["host_ports"]["webui"])}],
    }
    if host["PortBindings"] != expected_ports:
        raise MilvusSindiError("Story container port binding drifted")
    project_root = Path(__file__).parents[3].resolve()
    expected_data_root = (project_root / isolation["data_root"]).resolve()
    mounts = container["Mounts"]
    data_mounts = [row for row in mounts if row["Destination"] == "/var/lib/milvus"]
    if (
        len(data_mounts) != 1
        or data_mounts[0]["Type"] != "bind"
        or Path(data_mounts[0]["Source"]).resolve() != expected_data_root
        or not data_mounts[0]["RW"]
    ):
        raise MilvusSindiError("Story bind data root drifted")
    data_root_stat_result = os.stat(expected_data_root, follow_symlinks=False)
    data_root_stat = {
        "path": isolation["data_root"],
        "mode_octal": f"0{stat.S_IMODE(data_root_stat_result.st_mode):03o}",
        "uid": data_root_stat_result.st_uid,
        "gid": data_root_stat_result.st_gid,
        "inode": data_root_stat_result.st_ino,
        "device": data_root_stat_result.st_dev,
        "is_directory": stat.S_ISDIR(data_root_stat_result.st_mode),
        "world_writable": bool(data_root_stat_result.st_mode & stat.S_IWOTH),
    }
    raw["data_root_stat"] = _record(root, "raw/data-root-stat.json", data_root_stat)
    if data_root_stat["mode_octal"] != "0711" or data_root_stat["world_writable"]:
        raise MilvusSindiError("Story bind root must be mode 0711 without world-write")
    networks = container["NetworkSettings"]["Networks"]
    if set(networks) != {NETWORK_NAME}:
        raise MilvusSindiError("Story container joined an undeclared network")
    attached = network.get("Containers") or {}
    if set(attached) != {container["Id"]}:
        raise MilvusSindiError("Story network contains a non-Story container")
    if health != {"status": 200, "body": "OK"}:
        raise MilvusSindiError("Story health response drifted")

    evidence = {
        "schema_version": "milvus-sindi-deployment-evidence-v2",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "predeclaration_sha256": predeclaration_sha256,
        "server_image": SERVER_IMAGE,
        "image_id": IMAGE_ID,
        "repo_digests": sorted(repo_digests),
        "platform": "linux/amd64",
        "container": CONTAINER_NAME,
        "container_id": container["Id"],
        "network": NETWORK_NAME,
        "network_id": network["Id"],
        "data_root": isolation["data_root"],
        "data_root_stat": data_root_stat,
        "host_ports": isolation["host_ports"],
        "resource_limits": {
            "memory_bytes": host["Memory"],
            "memory_swap_bytes": host["MemorySwap"],
            "cpuset_cpus": host["CpusetCpus"],
            "pids_limit": host["PidsLimit"],
            "swap_allowed": False,
        },
        "raw": raw,
        "status": "pass",
    }
    path = root / "deployment-evidence.json"
    path.write_bytes(formatted_json_bytes(evidence))
    identity = file_sha256(path)
    (root / "deployment-evidence.sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity


def _fixture_input() -> dict[str, Any]:
    documents = [
        {"pk": 1000, "sparse_vector": {0: 1.0, 1: 0.5}},
        {"pk": 1001, "sparse_vector": {0: 1.0, 1: 0.5}},
        {"pk": 1002, "sparse_vector": {0: 0.9, 2: 0.4}},
        {"pk": 1003, "sparse_vector": {1: 1.0, 3: 0.2}},
        {"pk": 1004, "sparse_vector": {2: 1.0, 4: 0.3}},
        {"pk": 1005, "sparse_vector": {3: 1.0, 5: 0.2}},
        {"pk": 1006, "sparse_vector": {4: 1.0, 6: 0.1}},
        {"pk": 1007, "sparse_vector": {5: 1.0, 7: 0.5}},
        {"pk": 1008, "sparse_vector": {0: 0.2, 7: 1.0}},
        {"pk": 1009, "sparse_vector": {1: 0.3, 2: 0.3, 3: 0.3}},
        {"pk": 1010, "sparse_vector": {6: 1.0, 7: 1.0}},
        {"pk": 1011, "sparse_vector": {0: 0.4, 4: 0.4, 5: 0.4}},
    ]
    queries = [
        {"query_ordinal": 0, "sparse_vector": {0: 1.0, 1: 0.5}},
        {"query_ordinal": 1, "sparse_vector": {2: 1.0, 3: 0.5, 4: 0.2}},
        {"query_ordinal": 2, "sparse_vector": {5: 0.5, 6: 0.5, 7: 1.0}},
        {"query_ordinal": 3, "sparse_vector": {0: 0.5, 2: 0.5, 4: 0.5, 6: 0.5}},
    ]
    return {
        "dimensions": FIXTURE_DIMENSIONS,
        "top_k": FIXTURE_TOP_K,
        "documents": documents,
        "queries": queries,
    }


def _to_csr(rows: Sequence[Mapping[int, float]], dimensions: int) -> csr_matrix:
    indptr = [0]
    indices: list[int] = []
    data: list[np.float32] = []
    for row in rows:
        for index, value in sorted(row.items()):
            indices.append(int(index))
            data.append(np.float32(value))
        indptr.append(len(indices))
    return csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
        shape=(len(rows), dimensions),
        dtype=np.float32,
    )


def _exact_results(fixture: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    documents = fixture["documents"]
    queries = fixture["queries"]
    docs = _to_csr([row["sparse_vector"] for row in documents], fixture["dimensions"])
    query_matrix = _to_csr([row["sparse_vector"] for row in queries], fixture["dimensions"])
    product = (query_matrix @ docs.transpose()).tocsr()
    results: list[list[dict[str, Any]]] = []
    for query_index in range(product.shape[0]):
        start, end = product.indptr[query_index : query_index + 2]
        scores = {
            documents[int(column)]["pk"]: float(np.float32(score))
            for column, score in zip(product.indices[start:end], product.data[start:end], strict=True)
        }
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) < fixture["top_k"]:
            raise MilvusSindiError("Fixture query did not have enough positive exact results")
        results.append([{"id": pk, "score": score} for pk, score in ranked])
    return results


def _wait_for_index(client: Any, collection_name: str) -> dict[str, Any]:
    observations = []
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        described = _jsonable(client.describe_index(collection_name, INDEX_NAME, timeout=30))
        state = client._get_connection().get_index_state(collection_name, INDEX_NAME, timeout=30)
        state_value = int(state[0]) if isinstance(state, tuple) and state else -1
        observations.append(
            {
                "describe_index": described,
                "index_state": _jsonable(state),
                "index_state_name": "Finished" if state_value == 3 else f"enum_{state_value}",
            }
        )
        if described.get("state") == "Finished" and state_value == 3:
            return {"observations": observations, "final": observations[-1]}
        if described.get("state") == "Failed" or state_value == 4:
            raise MilvusSindiError(f"Index build failed for {collection_name}")
        time.sleep(0.5)
    raise MilvusSindiError(f"Index build timed out for {collection_name}")


def run_sparse_fixture(
    output: str | Path,
    *,
    predeclaration_path: str | Path,
    predeclaration_sha256: str,
    deployment_evidence_path: str | Path,
    deployment_evidence_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Run the bounded deterministic correctness fixture; never run a performance cell."""
    predeclaration = validate_predeclaration(predeclaration_path, predeclaration_sha256)
    from mm_embed.benchmark.milvus_sindi_system import validate_deployment_evidence

    validate_deployment_evidence(deployment_evidence_path, deployment_evidence_sha256)
    from pymilvus import DataType, MilvusClient
    from pymilvus import __version__ as pymilvus_version

    if pymilvus_version != PYMILVUS_VERSION:
        raise MilvusSindiError("PyMilvus version drifted before fixture")
    root = Path(output)
    if root.exists():
        raise MilvusSindiError("Fixture evidence output must not already exist")
    root.mkdir(parents=True)
    fixture = _fixture_input()
    exact = _exact_results(fixture)
    raw: dict[str, Any] = {
        "input": _record(root, "raw/fixture-input.json", fixture),
        "exact": _record(root, "raw/exact-csr-results.json", exact),
    }
    client = MilvusClient(uri="http://127.0.0.1:49531", dedicated=True, timeout=30)
    try:
        server = client.get_server_version(detail=True, timeout=30)
        raw["server_version"] = _record(root, "raw/server-version.json", server)
        raw["collections_before"] = _record(root, "raw/collections-before.json", client.list_collections(timeout=30))
        algorithms: dict[str, Any] = {}
        for algorithm in FIXTURE_ALGORITHMS:
            suffix = algorithm.lower()
            collection_name = f"{COLLECTION_PREFIX}fixture_{suffix}"
            if client.has_collection(collection_name, timeout=30):
                raise MilvusSindiError(f"Fixture collection already exists: {collection_name}")
            algorithm_raw: dict[str, Any] = {}
            schema = client.create_schema(
                auto_id=False, enable_dynamic_field=False, description="Deterministic sparse fixture"
            )
            schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            create_response = client.create_collection(
                collection_name=collection_name,
                schema=schema,
                consistency_level="Strong",
                timeout=30,
            )
            algorithm_raw["create_collection"] = _record(root, f"raw/{suffix}/create-collection.json", create_response)
            algorithm_raw["describe_collection"] = _record(
                root,
                f"raw/{suffix}/describe-collection.json",
                client.describe_collection(collection_name, timeout=30),
            )
            insert_response = client.insert(collection_name, fixture["documents"], timeout=30)
            algorithm_raw["insert"] = _record(root, f"raw/{suffix}/insert.json", insert_response)
            flush_response = client.flush(collection_name, timeout=60)
            algorithm_raw["flush"] = _record(root, f"raw/{suffix}/flush.json", flush_response)
            algorithm_raw["stats_after_flush"] = _record(
                root,
                f"raw/{suffix}/stats-after-flush.json",
                client.get_collection_stats(collection_name, timeout=30),
            )
            index_params = client.prepare_index_params()
            requested_params = {
                "field_name": "sparse_vector",
                "index_name": INDEX_NAME,
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {"inverted_index_algo": algorithm, "drop_ratio_build": 0},
            }
            index_params.add_index(**requested_params)
            algorithm_raw["index_request"] = _record(root, f"raw/{suffix}/index-request.json", requested_params)
            create_index_response = client.create_index(collection_name, index_params=index_params, timeout=120)
            algorithm_raw["create_index"] = _record(root, f"raw/{suffix}/create-index.json", create_index_response)
            index_wait = _wait_for_index(client, collection_name)
            algorithm_raw["index_wait"] = _record(root, f"raw/{suffix}/index-wait.json", index_wait)
            load_response = client.load_collection(collection_name, timeout=120)
            algorithm_raw["load"] = _record(root, f"raw/{suffix}/load.json", load_response)
            load_state = client.get_load_state(collection_name, timeout=30)
            algorithm_raw["load_state"] = _record(root, f"raw/{suffix}/load-state.json", load_state)
            stats = client.get_collection_stats(collection_name, timeout=30)
            algorithm_raw["stats_loaded"] = _record(root, f"raw/{suffix}/stats-loaded.json", stats)
            count = client.query(collection_name, filter="", output_fields=["count(*)"], timeout=30)
            algorithm_raw["count"] = _record(root, f"raw/{suffix}/count.json", count)
            segments = client._get_connection().get_query_segment_info(collection_name, timeout=30)
            algorithm_raw["segments"] = _record(root, f"raw/{suffix}/segments.json", segments)
            search = client.search(
                collection_name,
                data=[row["sparse_vector"] for row in fixture["queries"]],
                anns_field="sparse_vector",
                limit=FIXTURE_TOP_K,
                search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0}},
                output_fields=["pk"],
                timeout=30,
            )
            algorithm_raw["search"] = _record(root, f"raw/{suffix}/search.json", search)
            algorithms[algorithm] = {
                "collection": collection_name,
                "raw": algorithm_raw,
            }
        raw["collections_after"] = _record(root, "raw/collections-after.json", client.list_collections(timeout=30))
    finally:
        client.close()

    evidence = {
        "schema_version": "milvus-sindi-fixture-evidence-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration_sha256": predeclaration_sha256,
        "deployment_evidence_sha256": deployment_evidence_sha256,
        "server_image": SERVER_IMAGE,
        "pymilvus_version": pymilvus_version,
        "formal_performance_executed": False,
        "native_inputs_used": False,
        "system_scale_inputs_used": False,
        "fixture_rows": len(fixture["documents"]),
        "fixture_queries": len(fixture["queries"]),
        "top_k": FIXTURE_TOP_K,
        "metric_type": "IP",
        "index_type": "SPARSE_INVERTED_INDEX",
        "algorithms": algorithms,
        "common_raw": raw,
        "publication_gate": predeclaration["publication"]["gate"],
        "status": "pending_validation",
    }
    path = root / "fixture-evidence.json"
    path.write_bytes(formatted_json_bytes(evidence))
    identity = file_sha256(path)
    (root / "fixture-evidence.sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity


def score_fixture_results(
    exact: Sequence[Sequence[Mapping[str, Any]]],
    actual: Sequence[Sequence[Mapping[str, Any]]],
    *,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    if len(actual) != len(exact):
        raise MilvusSindiError("Fixture search query count drifted")
    per_query = []
    all_ids: list[int] = []
    all_scores: list[float] = []
    for query_ordinal, (expected_hits, actual_hits) in enumerate(zip(exact, actual, strict=True)):
        expected_top_k = expected_hits[: len(actual_hits)]
        if not actual_hits or len(actual_hits) > len(expected_hits):
            raise MilvusSindiError("Fixture search result count drifted")
        actual_ids = [int(hit["id"] if "id" in hit else hit["pk"]) for hit in actual_hits]
        actual_scores = [float(hit["distance"]) for hit in actual_hits]
        if len(set(actual_ids)) != len(actual_ids) or not all(map(math.isfinite, actual_scores)):
            raise MilvusSindiError("Fixture results must contain unique IDs and finite scores")
        expected_ids = [int(hit["id"]) for hit in expected_top_k]
        expected_scores = {int(hit["id"]): float(hit["score"]) for hit in expected_hits}
        kth = float(expected_top_k[-1]["score"])
        eligible = {int(hit["id"]) for hit in expected_hits if float(hit["score"]) >= kth - tolerance}
        strict_recall = len(set(actual_ids) & set(expected_ids)) / len(expected_ids)
        tie_aware_recall = len(set(actual_ids) & eligible) / len(expected_ids)
        score_deltas = [
            abs(score - expected_scores[pk])
            for pk, score in zip(actual_ids, actual_scores, strict=True)
            if pk in expected_scores
        ]
        per_query.append(
            {
                "query_ordinal": query_ordinal,
                "strict_id_recall_at_k": strict_recall,
                "tie_aware_recall_at_k": tie_aware_recall,
                "max_absolute_score_delta": max(score_deltas, default=math.inf),
                "returned_ids_unique": True,
                "scores_finite": True,
            }
        )
        all_ids.extend(actual_ids)
        all_scores.extend(actual_scores)
    return {
        "per_query": per_query,
        "minimum_strict_id_recall_at_k": min(row["strict_id_recall_at_k"] for row in per_query),
        "minimum_tie_aware_recall_at_k": min(row["tie_aware_recall_at_k"] for row in per_query),
        "maximum_absolute_score_delta": max(row["max_absolute_score_delta"] for row in per_query),
        "all_query_ids_unique": all(
            len({int(hit["id"] if "id" in hit else hit["pk"]) for hit in hits}) == len(hits) for hits in actual
        ),
        "all_scores_finite": all(map(math.isfinite, all_scores)),
    }


def _validate_algorithm_raw(
    *,
    root: Path,
    algorithm: str,
    algorithm_evidence: Mapping[str, Any],
    fixture: Mapping[str, Any],
    exact: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    raw = {name: _read_record(root, record) for name, record in algorithm_evidence["raw"].items()}
    schema = raw["describe_collection"]
    fields = {(field["name"], field["type"], bool(field.get("is_primary", False))) for field in schema["fields"]}
    if (
        schema["collection_name"] != algorithm_evidence["collection"]
        or schema["auto_id"] is not False
        or schema["enable_dynamic_field"] is not False
        or schema["consistency_level_name"] != "Strong"
        or fields != {("pk", "INT64", True), ("sparse_vector", "SPARSE_FLOAT_VECTOR", False)}
    ):
        raise MilvusSindiError(f"Collection schema drifted for {algorithm}")
    requested = raw["index_request"]
    expected_request = {
        "field_name": "sparse_vector",
        "index_name": INDEX_NAME,
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
        "params": {"drop_ratio_build": 0, "inverted_index_algo": algorithm},
    }
    if requested != expected_request:
        raise MilvusSindiError(f"Index request drifted for {algorithm}")
    index = raw["index_wait"]["final"]["describe_index"]
    if (
        index.get("inverted_index_algo") != algorithm
        or index.get("index_type") != "SPARSE_INVERTED_INDEX"
        or index.get("metric_type") != "IP"
        or index.get("field_name") != "sparse_vector"
        or index.get("index_name") != INDEX_NAME
        or index.get("drop_ratio_build") != "0"
        or index.get("state") != "Finished"
        or index.get("total_rows") != len(fixture["documents"])
        or index.get("indexed_rows") != len(fixture["documents"])
        or index.get("pending_index_rows") != 0
        or raw["index_wait"]["final"].get("index_state_name") != "Finished"
    ):
        raise MilvusSindiError(f"Server did not report the exact requested algorithm for {algorithm}")
    if raw["insert"].get("insert_count") != len(fixture["documents"]):
        raise MilvusSindiError(f"Insert response row count drifted for {algorithm}")
    if (
        raw["stats_after_flush"].get("row_count") != len(fixture["documents"])
        or raw["stats_loaded"].get("row_count") != len(fixture["documents"])
        or raw["count"] != [{"count(*)": len(fixture["documents"])}]
        or raw["load_state"] != {"state": "Loaded"}
    ):
        raise MilvusSindiError(f"Flush/load/count state drifted for {algorithm}")
    verification = raw["verification_state"]
    if verification["describe_index"] != index:
        raise MilvusSindiError(f"Post-search index state drifted for {algorithm}")
    if verification["load_state"] != {"state": "Loaded"}:
        raise MilvusSindiError(f"Post-search load state drifted for {algorithm}")
    segments = verification["segments"]
    if not isinstance(segments, list) or not segments:
        raise MilvusSindiError(f"No structured segment evidence for {algorithm}")
    if any(
        segment.get("state") != "Sealed" or not segment.get("index_name") or int(segment.get("indexID", 0)) <= 0
        for segment in segments
    ):
        raise MilvusSindiError(f"Growing or unindexed segment observed for {algorithm}")
    sealed_rows = sum(int(segment["num_rows"]) for segment in segments)
    if sealed_rows != len(fixture["documents"]):
        raise MilvusSindiError(f"Sealed segment rows drifted for {algorithm}")
    search_request = raw["verification_search_request"]
    if search_request != {
        "anns_field": "sparse_vector",
        "limit": FIXTURE_TOP_K,
        "search_params": {"metric_type": "IP", "params": {"drop_ratio_search": 0}},
    }:
        raise MilvusSindiError(f"Search request drifted for {algorithm}")
    score = score_fixture_results(exact, raw["verification_search"])
    if (
        score["minimum_strict_id_recall_at_k"] != 1.0
        or score["minimum_tie_aware_recall_at_k"] != 1.0
        or score["maximum_absolute_score_delta"] > 1e-5
        or not score["all_query_ids_unique"]
        or not score["all_scores_finite"]
    ):
        raise MilvusSindiError(f"Exact CSR correctness failed for {algorithm}")
    return {
        "reported_algorithm": index["inverted_index_algo"],
        "index_state": index["state"],
        "total_rows": index["total_rows"],
        "indexed_rows": index["indexed_rows"],
        "pending_index_rows": index["pending_index_rows"],
        "load_state": verification["load_state"]["state"],
        "segment_count": len(segments),
        "sealed_segment_count": len(segments),
        "growing_segment_count": 0,
        "sealed_rows": sealed_rows,
        "all_segments_indexed": True,
        "drop_ratio_search": 0,
        "correctness": score,
    }


def validate_fixture_evidence(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Recompute fixture correctness from every hash-bound raw API response."""
    evidence_path = Path(path)
    if (
        not evidence_path.is_file()
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or file_sha256(evidence_path) != expected_sha256
    ):
        raise MilvusSindiError("Fixture evidence identity drifted")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    from mm_embed.benchmark.milvus_sindi_system import FIXTURE_EVIDENCE_SCHEMA, _validate_schema

    _validate_schema(evidence, FIXTURE_EVIDENCE_SCHEMA)
    if evidence["status"] != "pass" or evidence["formal_performance_executed"] is not False:
        raise MilvusSindiError("Fixture evidence is not finalized and performance-free")
    root = evidence_path.parent
    common = {name: _read_record(root, record) for name, record in evidence["common_raw"].items()}
    fixture = common["input"]
    exact = common["exact"]
    if exact != _exact_results(fixture):
        raise MilvusSindiError("Saved exact results do not match recomputed float32 CSR IP")
    server = common["server_version"]
    if (
        server.get("version") != "3.0.0"
        or server.get("git_commit") != "f46a032855"
        or server.get("deploy_mode") != "STANDALONE"
    ):
        raise MilvusSindiError("Server version response drifted")
    validation = {
        algorithm: _validate_algorithm_raw(
            root=root,
            algorithm=algorithm,
            algorithm_evidence=evidence["algorithms"][algorithm],
            fixture=fixture,
            exact=exact,
        )
        for algorithm in FIXTURE_ALGORITHMS
    }
    expected_summary = {
        "server_version": server,
        "algorithms": validation,
        "sealed_indexed_segments_only": True,
        "compaction": _validate_compaction_metrics(_read_bytes_record(root, evidence["compaction_metrics_raw"])),
        "tie_policy": {
            "exact_order": "descending_float32_ip_score_then_ascending_private_row_ordinal",
            "milvus_order": "record_returned_order_without_assuming_stable_ties",
            "strict_metric": "exact ordered top-k ID intersection divided by k",
            "tie_aware_metric": "expand exact membership at the kth score within 1e-5",
        },
    }
    if evidence.get("validation") != expected_summary:
        raise MilvusSindiError("Fixture validation summary does not match raw responses")
    return evidence


def finalize_fixture_evidence(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    """Acquire independent post-search state, then seal the fixture manifest."""
    evidence_path = Path(path)
    if file_sha256(evidence_path) != expected_sha256:
        raise MilvusSindiError("Pending fixture evidence identity drifted")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "pending_validation":
        raise MilvusSindiError("Fixture evidence is not pending validation")
    root = evidence_path.parent
    fixture = _read_record(root, evidence["common_raw"]["input"])
    exact = _read_record(root, evidence["common_raw"]["exact"])
    from pymilvus import MilvusClient

    client = MilvusClient(uri="http://127.0.0.1:49531", dedicated=True, timeout=30)
    try:
        summaries = {}
        for algorithm in FIXTURE_ALGORITHMS:
            item = evidence["algorithms"][algorithm]
            collection_name = item["collection"]
            state = {
                "describe_collection": client.describe_collection(collection_name, timeout=30),
                "describe_index": client.describe_index(collection_name, INDEX_NAME, timeout=30),
                "load_state": client.get_load_state(collection_name, timeout=30),
                "stats": client.get_collection_stats(collection_name, timeout=30),
                "count": client.query(collection_name, filter="", output_fields=["count(*)"], timeout=30),
                "segments": client._get_connection().get_query_segment_info(collection_name, timeout=30),
            }
            suffix = algorithm.lower()
            item["raw"]["verification_state"] = _record(root, f"raw/{suffix}/verification-state.json", state)
            search_request = {
                "anns_field": "sparse_vector",
                "limit": FIXTURE_TOP_K,
                "search_params": {"metric_type": "IP", "params": {"drop_ratio_search": 0}},
            }
            item["raw"]["verification_search_request"] = _record(
                root, f"raw/{suffix}/verification-search-request.json", search_request
            )
            search = client.search(
                collection_name,
                data=[row["sparse_vector"] for row in fixture["queries"]],
                output_fields=["pk"],
                timeout=30,
                **search_request,
            )
            item["raw"]["verification_search"] = _record(root, f"raw/{suffix}/verification-search.json", search)
            summaries[algorithm] = _validate_algorithm_raw(
                root=root,
                algorithm=algorithm,
                algorithm_evidence=item,
                fixture=fixture,
                exact=exact,
            )
    finally:
        client.close()
    server = _read_record(root, evidence["common_raw"]["server_version"])
    compaction_record, compaction = _capture_compaction_metrics(root)
    evidence["compaction_metrics_raw"] = compaction_record
    evidence["validation"] = {
        "server_version": server,
        "algorithms": summaries,
        "sealed_indexed_segments_only": True,
        "compaction": compaction,
        "tie_policy": {
            "exact_order": "descending_float32_ip_score_then_ascending_private_row_ordinal",
            "milvus_order": "record_returned_order_without_assuming_stable_ties",
            "strict_metric": "exact ordered top-k ID intersection divided by k",
            "tie_aware_metric": "expand exact membership at the kth score within 1e-5",
        },
    }
    evidence["status"] = "pass"
    evidence_path.write_bytes(formatted_json_bytes(evidence))
    identity = file_sha256(evidence_path)
    evidence_path.with_suffix(".sha256").write_text(identity + "\n", encoding="ascii")
    validate_fixture_evidence(evidence_path, identity)
    return evidence, identity


def _validate_recovery_algorithm(
    *,
    root: Path,
    algorithm: str,
    collection_name: str,
    records: Mapping[str, Any],
    fixture: Mapping[str, Any],
    exact: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    raw = {name: _read_record(root, record) for name, record in records.items()}
    description = raw["describe_collection"]
    fields = {(field["name"], field["type"], bool(field.get("is_primary", False))) for field in description["fields"]}
    if (
        description["collection_name"] != collection_name
        or description["auto_id"] is not False
        or description["enable_dynamic_field"] is not False
        or fields != {("pk", "INT64", True), ("sparse_vector", "SPARSE_FLOAT_VECTOR", False)}
    ):
        raise MilvusSindiError(f"Recovered collection schema drifted for {algorithm}")
    index = raw["describe_index"]
    if (
        index.get("inverted_index_algo") != algorithm
        or index.get("drop_ratio_build") != "0"
        or index.get("metric_type") != "IP"
        or index.get("index_type") != "SPARSE_INVERTED_INDEX"
        or index.get("field_name") != "sparse_vector"
        or index.get("index_name") != INDEX_NAME
        or index.get("state") != "Finished"
        or index.get("total_rows") != len(fixture["documents"])
        or index.get("indexed_rows") != len(fixture["documents"])
        or index.get("pending_index_rows") != 0
    ):
        raise MilvusSindiError(f"Recovered index algorithm or state drifted for {algorithm}")
    if (
        raw["load_state"] != {"state": "Loaded"}
        or raw["stats"].get("row_count") != len(fixture["documents"])
        or raw["count"] != [{"count(*)": len(fixture["documents"])}]
    ):
        raise MilvusSindiError(f"Recovered row or load state drifted for {algorithm}")
    segments = raw["segments"]
    if not isinstance(segments, list) or not segments:
        raise MilvusSindiError(f"Recovered segment evidence is absent for {algorithm}")
    if any(
        segment.get("state") != "Sealed" or not segment.get("index_name") or int(segment.get("indexID", 0)) <= 0
        for segment in segments
    ):
        raise MilvusSindiError(f"Recovered growing or unindexed segment observed for {algorithm}")
    sealed_rows = sum(int(segment["num_rows"]) for segment in segments)
    if sealed_rows != len(fixture["documents"]):
        raise MilvusSindiError(f"Recovered sealed row count drifted for {algorithm}")
    expected_search_request = {
        "anns_field": "sparse_vector",
        "limit": FIXTURE_TOP_K,
        "search_params": {"metric_type": "IP", "params": {"drop_ratio_search": 0}},
    }
    if raw["search_request"] != expected_search_request:
        raise MilvusSindiError(f"Recovered search request drifted for {algorithm}")
    correctness = score_fixture_results(exact, raw["search"])
    if (
        correctness["minimum_strict_id_recall_at_k"] != 1.0
        or correctness["minimum_tie_aware_recall_at_k"] != 1.0
        or correctness["maximum_absolute_score_delta"] > 1e-5
        or not correctness["all_query_ids_unique"]
        or not correctness["all_scores_finite"]
    ):
        raise MilvusSindiError(f"Recovered exact CSR correctness failed for {algorithm}")
    return {
        "reported_algorithm": index["inverted_index_algo"],
        "index_state": index["state"],
        "row_count": raw["stats"]["row_count"],
        "load_state": raw["load_state"]["state"],
        "sealed_segment_count": len(segments),
        "growing_segment_count": 0,
        "sealed_rows": sealed_rows,
        "all_segments_indexed": True,
        "drop_ratio_search": 0,
        "correctness": correctness,
    }


def capture_recovery_evidence(
    output: str | Path,
    *,
    predeclaration_path: str | Path,
    predeclaration_sha256: str,
    fixture_evidence_path: str | Path,
    fixture_evidence_sha256: str,
    hardened_deployment_path: str | Path,
    hardened_deployment_sha256: str,
    before_stat_path: str | Path,
    after_stat_path: str | Path,
    prehardening_baseline_sha256: str,
    stopped_baseline_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Revalidate persisted fixture state after restarting the exact Story container."""
    predeclaration = validate_predeclaration(predeclaration_path, predeclaration_sha256)
    fixture_evidence = validate_fixture_evidence(fixture_evidence_path, fixture_evidence_sha256)
    from mm_embed.benchmark.milvus_sindi_system import validate_deployment_evidence

    deployment = validate_deployment_evidence(hardened_deployment_path, hardened_deployment_sha256)
    if deployment["schema_version"] != "milvus-sindi-deployment-evidence-v2":
        raise MilvusSindiError("Recovery requires hardened deployment evidence v2")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    evidence_path = root / "recovery-evidence.json"
    if evidence_path.exists() or (root / "raw").exists():
        raise MilvusSindiError("Recovery evidence output must not already contain evidence")
    before_stat = json.loads(Path(before_stat_path).read_text(encoding="utf-8"))
    after_stat = json.loads(Path(after_stat_path).read_text(encoding="utf-8"))
    if (
        before_stat.get("mode_octal") != "0777"
        or before_stat.get("world_writable") is not True
        or after_stat.get("mode_octal") != "0711"
        or after_stat.get("world_writable") is not False
        or before_stat.get("inode") != after_stat.get("inode")
        or before_stat.get("device_decimal") != after_stat.get("device_decimal")
    ):
        raise MilvusSindiError("Before/after hardening stat transition is invalid")
    data_root = Path(__file__).parents[3] / predeclaration["isolation"]["data_root"]
    current = os.stat(data_root, follow_symlinks=False)
    current_stat = {
        "path": predeclaration["isolation"]["data_root"],
        "mode_octal": f"0{stat.S_IMODE(current.st_mode):03o}",
        "uid": current.st_uid,
        "gid": current.st_gid,
        "inode": current.st_ino,
        "device": current.st_dev,
        "is_directory": stat.S_ISDIR(current.st_mode),
        "world_writable": bool(current.st_mode & stat.S_IWOTH),
    }
    if (
        current_stat["mode_octal"] != "0711"
        or current_stat["world_writable"]
        or current_stat["inode"] != after_stat["inode"]
        or current_stat["device"] != after_stat["device_decimal"]
    ):
        raise MilvusSindiError("Live bind root no longer matches hardened stat evidence")

    fixture_root = Path(fixture_evidence_path).parent
    fixture = _read_record(fixture_root, fixture_evidence["common_raw"]["input"])
    exact = _read_record(fixture_root, fixture_evidence["common_raw"]["exact"])
    if exact != _exact_results(fixture):
        raise MilvusSindiError("Original fixture exact CSR evidence drifted")
    raw: dict[str, Any] = {
        "before_stat": _record(root, "raw/before-stat.json", before_stat),
        "after_stat": _record(root, "raw/after-stat.json", after_stat),
        "current_stat": _record(root, "raw/current-stat.json", current_stat),
        "fixture_input": _record(root, "raw/fixture-input.json", fixture),
        "exact": _record(root, "raw/exact-csr-results.json", exact),
    }
    from pymilvus import MilvusClient
    from pymilvus import __version__ as pymilvus_version

    if pymilvus_version != PYMILVUS_VERSION:
        raise MilvusSindiError("PyMilvus version drifted before recovery validation")
    client = MilvusClient(uri="http://127.0.0.1:49531", dedicated=True, timeout=30)
    algorithms: dict[str, Any] = {}
    try:
        server = client.get_server_version(detail=True, timeout=30)
        raw["server_version"] = _record(root, "raw/server-version.json", server)
        collections = client.list_collections(timeout=30)
        raw["collections"] = _record(root, "raw/collections.json", collections)
        expected_collections = {
            algorithm: f"{COLLECTION_PREFIX}fixture_{algorithm.lower()}" for algorithm in FIXTURE_ALGORITHMS
        }
        if set(collections) != set(expected_collections.values()):
            raise MilvusSindiError("Recovered collection namespace drifted")
        for algorithm, collection_name in expected_collections.items():
            suffix = algorithm.lower()
            search_request = {
                "anns_field": "sparse_vector",
                "limit": FIXTURE_TOP_K,
                "search_params": {"metric_type": "IP", "params": {"drop_ratio_search": 0}},
            }
            values = {
                "describe_collection": client.describe_collection(collection_name, timeout=30),
                "describe_index": client.describe_index(collection_name, INDEX_NAME, timeout=30),
                "load_state": client.get_load_state(collection_name, timeout=30),
                "stats": client.get_collection_stats(collection_name, timeout=30),
                "count": client.query(collection_name, filter="", output_fields=["count(*)"], timeout=30),
                "segments": client._get_connection().get_query_segment_info(collection_name, timeout=30),
                "search_request": search_request,
                "search": client.search(
                    collection_name,
                    data=[row["sparse_vector"] for row in fixture["queries"]],
                    output_fields=["pk"],
                    timeout=30,
                    **search_request,
                ),
            }
            records = {
                name: _record(root, f"raw/{suffix}/{name.replace('_', '-')}.json", value)
                for name, value in values.items()
            }
            algorithms[algorithm] = {
                "collection": collection_name,
                "raw": records,
                "validation": _validate_recovery_algorithm(
                    root=root,
                    algorithm=algorithm,
                    collection_name=collection_name,
                    records=records,
                    fixture=fixture,
                    exact=exact,
                ),
            }
    finally:
        client.close()
    try:
        with urllib.request.urlopen("http://127.0.0.1:49092/metrics", timeout=15) as response:
            if response.status != 200:
                raise MilvusSindiError("Milvus metrics endpoint did not return HTTP 200")
            metrics_raw = response.read()
    except OSError as exc:
        raise MilvusSindiError("Milvus metrics endpoint was unavailable") from exc
    metrics_record = _record_bytes(root, "raw/server-metrics-after-recovery.prom", metrics_raw)
    compaction = _validate_recovery_compaction_metrics(metrics_raw)
    server_value = _read_record(root, raw["server_version"])
    if (
        server_value.get("version") != "3.0.0"
        or server_value.get("git_commit") != "f46a032855"
        or server_value.get("deploy_mode") != "STANDALONE"
    ):
        raise MilvusSindiError("Recovered server version response drifted")
    evidence = {
        "schema_version": "milvus-sindi-recovery-evidence-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "predeclaration_sha256": predeclaration_sha256,
        "fixture_evidence_sha256": fixture_evidence_sha256,
        "hardened_deployment_sha256": hardened_deployment_sha256,
        "prehardening_baseline_sha256": prehardening_baseline_sha256,
        "stopped_baseline_sha256": stopped_baseline_sha256,
        "container_id": deployment["container_id"],
        "network_id": deployment["network_id"],
        "server_image": SERVER_IMAGE,
        "pymilvus_version": pymilvus_version,
        "formal_performance_executed": False,
        "data_persisted_after_restart": True,
        "runtime_root_mode": "0711",
        "world_writable": False,
        "raw": raw,
        "algorithms": algorithms,
        "compaction_metrics_raw": metrics_record,
        "compaction": compaction,
        "tie_policy": fixture_evidence["validation"]["tie_policy"],
        "publication_gate": predeclaration["publication"]["gate"],
        "status": "pass",
    }
    from mm_embed.benchmark.milvus_sindi_system import RECOVERY_EVIDENCE_SCHEMA, _validate_schema

    _validate_schema(evidence, RECOVERY_EVIDENCE_SCHEMA)
    evidence_path.write_bytes(formatted_json_bytes(evidence))
    identity = file_sha256(evidence_path)
    (root / "recovery-evidence.sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity


def validate_recovery_evidence(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Recompute hardened restart recovery claims from hash-bound raw responses."""
    evidence_path = Path(path)
    if not evidence_path.is_file() or file_sha256(evidence_path) != expected_sha256:
        raise MilvusSindiError("Recovery evidence identity drifted")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    from mm_embed.benchmark.milvus_sindi_system import RECOVERY_EVIDENCE_SCHEMA, _validate_schema

    _validate_schema(evidence, RECOVERY_EVIDENCE_SCHEMA)
    root = evidence_path.parent
    raw = {name: _read_record(root, record) for name, record in evidence["raw"].items()}
    if (
        raw["before_stat"]["mode_octal"] != "0777"
        or raw["before_stat"]["world_writable"] is not True
        or raw["after_stat"]["mode_octal"] != "0711"
        or raw["after_stat"]["world_writable"] is not False
        or raw["current_stat"]["mode_octal"] != "0711"
        or raw["current_stat"]["world_writable"] is not False
        or raw["before_stat"]["inode"] != raw["after_stat"]["inode"]
        or raw["after_stat"]["inode"] != raw["current_stat"]["inode"]
    ):
        raise MilvusSindiError("Recovery stat chain drifted")
    fixture = raw["fixture_input"]
    exact = raw["exact"]
    if exact != _exact_results(fixture):
        raise MilvusSindiError("Recovery exact CSR evidence drifted")
    if (
        raw["server_version"].get("version") != "3.0.0"
        or raw["server_version"].get("git_commit") != "f46a032855"
        or raw["server_version"].get("deploy_mode") != "STANDALONE"
        or set(raw["collections"]) != {evidence["algorithms"][name]["collection"] for name in FIXTURE_ALGORITHMS}
    ):
        raise MilvusSindiError("Recovery server or collection namespace drifted")
    summaries = {
        algorithm: _validate_recovery_algorithm(
            root=root,
            algorithm=algorithm,
            collection_name=evidence["algorithms"][algorithm]["collection"],
            records=evidence["algorithms"][algorithm]["raw"],
            fixture=fixture,
            exact=exact,
        )
        for algorithm in FIXTURE_ALGORITHMS
    }
    if any(evidence["algorithms"][name]["validation"] != summaries[name] for name in summaries):
        raise MilvusSindiError("Recovery validation summary drifted from raw responses")
    compaction = _validate_recovery_compaction_metrics(_read_bytes_record(root, evidence["compaction_metrics_raw"]))
    if evidence["compaction"] != compaction or compaction["active"] is not False:
        raise MilvusSindiError("Recovery compaction summary drifted")
    if (
        evidence["formal_performance_executed"] is not False
        or evidence["data_persisted_after_restart"] is not True
        or evidence["publication_gate"] != "closed"
        or evidence["runtime_root_mode"] != "0711"
        or evidence["world_writable"] is not False
    ):
        raise MilvusSindiError("Recovery safety gates drifted")
    return evidence


def attach_compaction_evidence(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    """Upgrade an already validated fixture with raw no-active-compaction metrics."""
    evidence_path = Path(path)
    if not evidence_path.is_file() or file_sha256(evidence_path) != expected_sha256:
        raise MilvusSindiError("Fixture evidence identity drifted before compaction attachment")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "pass" or "compaction_metrics_raw" in evidence:
        raise MilvusSindiError("Fixture evidence is not eligible for compaction attachment")
    record, summary = _capture_compaction_metrics(evidence_path.parent)
    evidence["compaction_metrics_raw"] = record
    evidence["validation"]["compaction"] = summary
    evidence_path.write_bytes(formatted_json_bytes(evidence))
    identity = file_sha256(evidence_path)
    evidence_path.with_suffix(".sha256").write_text(identity + "\n", encoding="ascii")
    validate_fixture_evidence(evidence_path, identity)
    return evidence, identity
