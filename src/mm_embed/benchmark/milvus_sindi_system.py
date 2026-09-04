"""Fail-closed phase-one contracts for the Milvus SINDI system benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

STORY_ID = "S-20260814-010"
PREDECLARATION_SCHEMA = "milvus-sindi-system-predeclaration-v01.schema.json"
SOURCE_ATTESTATION_SCHEMA = "milvus-sindi-system-source-attestation-v01.schema.json"
DEPLOYMENT_EVIDENCE_SCHEMA = "milvus-sindi-deployment-evidence-v01.schema.json"
HARDENED_DEPLOYMENT_EVIDENCE_SCHEMA = "milvus-sindi-deployment-evidence-v02.schema.json"
FIXTURE_EVIDENCE_SCHEMA = "milvus-sindi-fixture-evidence-v01.schema.json"
RECOVERY_EVIDENCE_SCHEMA = "milvus-sindi-recovery-evidence-v01.schema.json"
ACTIVE_PREDECLARATION_SHA256 = "d1c3e3c9c2ad1e7e4b79bf37b0088e480058196a570f56d21ccc1a4bd2ff47d1"
SOURCE_COMMIT = "74635817770023b70b1e9f9e97b3d83d9d11f987"
SOURCE_TREE = "dc4a28d1805c82de5d4e2397fa163275a671638c"
SOURCE_PROTOCOL_SHA256 = "d6d615a9b753245767f895847a8828e99f1a961a3d8f7247470b9caa4edcbcaf"
SOURCE_MANIFEST_SHA256 = "3accbed7b2a9961eedff20e601375a23c50167cfc7d8e77adbf91399aed91f47"
SOURCE_INPUT_CONTRACT_SHA256 = "18ec21848358fd19a454ed2f7362e1eb6117d6c508ca6f30cc9980d2642d1dc6"
SERVER_IMAGE = "milvusdb/milvus@sha256:804b50bc1523a64e3c0f18cf33af7e0f8b33329584698ffee61c606d6fe8ee26"
SERVER_IMAGE_INDEX_SHA256 = "49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862"
PYMILVUS_VERSION = "3.0.1"
REGISTRY_REPOSITORY = "milvusdb/milvus"
REGISTRY_TAG = "v3.0.0"
SERVER_IMAGE_MANIFEST_SHA256 = SERVER_IMAGE.rsplit(":", maxsplit=1)[1]


class MilvusSindiError(ValueError):
    """Raised when phase-one evidence violates the frozen contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique compact JSON representation used for identities."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def formatted_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str | None, label: str) -> str:
    if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MilvusSindiError(f"Missing or invalid externally supplied {label}")
    return value


def _load_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MilvusSindiError(f"Missing or invalid JSON: {path}") from exc


def _registry_get(url: str, *, accept: str | None = None, token: str | None = None) -> tuple[bytes, dict[str, str]]:
    headers = {"User-Agent": "modern-embedding-bench-milvus-sindi/0.1"}
    if accept is not None:
        headers["Accept"] = accept
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            body = response.read()
            safe_headers = {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower() in {"content-type", "content-length", "docker-content-digest", "etag"}
            }
    except OSError as exc:
        raise MilvusSindiError(f"Registry request failed: {url}") from exc
    return body, safe_headers


def capture_registry_manifests(output: str | Path) -> tuple[dict[str, Any], str]:
    """Capture raw registry manifests without persisting registry credentials."""
    root = Path(output)
    if root.exists():
        raise MilvusSindiError("Registry evidence output must not already exist")
    root.mkdir(parents=True)
    query = urllib.parse.urlencode({"service": "registry.docker.io", "scope": f"repository:{REGISTRY_REPOSITORY}:pull"})
    token_body, _ = _registry_get(f"https://auth.docker.io/token?{query}")
    try:
        token = json.loads(token_body)["token"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MilvusSindiError("Registry bearer token response was invalid") from exc
    if not isinstance(token, str) or not token:
        raise MilvusSindiError("Registry bearer token was empty")

    accept = (
        "application/vnd.oci.image.index.v1+json, "
        "application/vnd.docker.distribution.manifest.list.v2+json, "
        "application/vnd.oci.image.manifest.v1+json, "
        "application/vnd.docker.distribution.manifest.v2+json"
    )
    base = f"https://registry-1.docker.io/v2/{REGISTRY_REPOSITORY}/manifests"
    index_body, index_headers = _registry_get(f"{base}/{REGISTRY_TAG}", accept=accept, token=token)
    index_path = root / "v3.0.0-index.raw.json"
    index_path.write_bytes(index_body)
    index_digest = hashlib.sha256(index_body).hexdigest()
    if index_digest != SERVER_IMAGE_INDEX_SHA256:
        raise MilvusSindiError("Registry tag index digest drifted")
    if index_headers.get("docker-content-digest") != f"sha256:{SERVER_IMAGE_INDEX_SHA256}":
        raise MilvusSindiError("Registry tag response omitted or changed Docker-Content-Digest")
    try:
        index = json.loads(index_body)
        amd64 = [row for row in index["manifests"] if row.get("platform") == {"architecture": "amd64", "os": "linux"}]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MilvusSindiError("Registry tag response was not a valid image index") from exc
    if len(amd64) != 1 or amd64[0].get("digest") != f"sha256:{SERVER_IMAGE_MANIFEST_SHA256}":
        raise MilvusSindiError("Registry index did not uniquely bind the frozen linux/amd64 manifest")

    child_body, child_headers = _registry_get(
        f"{base}/sha256:{SERVER_IMAGE_MANIFEST_SHA256}", accept=accept, token=token
    )
    child_path = root / "linux-amd64-manifest.raw.json"
    child_path.write_bytes(child_body)
    child_digest = hashlib.sha256(child_body).hexdigest()
    if child_digest != SERVER_IMAGE_MANIFEST_SHA256:
        raise MilvusSindiError("Registry linux/amd64 manifest digest drifted")
    if child_headers.get("docker-content-digest") != f"sha256:{SERVER_IMAGE_MANIFEST_SHA256}":
        raise MilvusSindiError("Registry child response omitted or changed Docker-Content-Digest")
    try:
        child = json.loads(child_body)
    except json.JSONDecodeError as exc:
        raise MilvusSindiError("Registry child response was not valid JSON") from exc
    if not isinstance(child.get("config", {}).get("digest"), str):
        raise MilvusSindiError("Registry child manifest did not contain a config digest")

    evidence = {
        "schema_version": "milvus-sindi-registry-manifests-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "credentials_persisted": False,
        "repository": REGISTRY_REPOSITORY,
        "tag": REGISTRY_TAG,
        "platform": {"os": "linux", "architecture": "amd64"},
        "index": {
            "digest": f"sha256:{index_digest}",
            "raw_path": index_path.name,
            "raw_bytes": len(index_body),
            "headers": index_headers,
        },
        "child": {
            "digest": f"sha256:{child_digest}",
            "raw_path": child_path.name,
            "raw_bytes": len(child_body),
            "headers": child_headers,
            "config_digest": child["config"]["digest"],
        },
        "status": "pass",
    }
    evidence_path = root / "manifest.json"
    evidence_path.write_bytes(formatted_json_bytes(evidence))
    identity = file_sha256(evidence_path)
    (root / "manifest.sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity


def _validate_schema(value: Any, schema_name: str) -> None:
    schema_path = Path(__file__).parents[3] / "schemas" / schema_name
    try:
        schema = _load_json(schema_path)
        Draft202012Validator(schema).validate(value)
    except Exception as exc:
        raise MilvusSindiError(f"{schema_name} validation failed: {exc}") from exc


def validate_predeclaration(path: str | Path, expected_sha256: str | None) -> dict[str, Any]:
    """Validate the frozen protocol against an out-of-band identity."""
    expected = _require_sha256(expected_sha256, "predeclaration SHA256")
    if expected != ACTIVE_PREDECLARATION_SHA256 or file_sha256(path) != expected:
        raise MilvusSindiError("Predeclaration does not match the unique active identity")
    value = _load_json(path)
    _validate_schema(value, PREDECLARATION_SCHEMA)
    source = value["source"]
    if source["commit"] != SOURCE_COMMIT or source["tree"] != SOURCE_TREE:
        raise MilvusSindiError("Accepted source commit or tree identity drifted")
    if source["main_protocol_sha256"] != SOURCE_PROTOCOL_SHA256:
        raise MilvusSindiError("Accepted source protocol identity drifted")
    if source["main_manifest_sha256"] != SOURCE_MANIFEST_SHA256:
        raise MilvusSindiError("Accepted source manifest identity drifted")
    deployment = value["deployment"]
    if deployment["server_image"] != SERVER_IMAGE:
        raise MilvusSindiError("Milvus server image digest drifted")
    if deployment["server_image_index_sha256"] != SERVER_IMAGE_INDEX_SHA256:
        raise MilvusSindiError("Milvus image index digest drifted")
    if deployment["pymilvus_version"] != PYMILVUS_VERSION:
        raise MilvusSindiError("PyMilvus version drifted")
    if value["publication"] != {
        "classification": "research_only",
        "gate": "closed",
        "leaderboard_publish": False,
        "public_export_allowed": False,
        "publish": False,
    }:
        raise MilvusSindiError("Publication gate must remain closed")
    if value["matrix"]["search_params"] != {
        "metric_type": "IP",
        "params": {"drop_ratio_search": 0.0},
    }:
        raise MilvusSindiError("Main comparison must set drop_ratio_search=0")
    if value["matrix"]["index_algorithms"] != ["SINDI", "DAAT_MAXSCORE"]:
        raise MilvusSindiError("Main comparison algorithms drifted")
    if value["matrix"]["top_k"] != [10, 100] or value["matrix"]["concurrency"] != [1, 4, 16]:
        raise MilvusSindiError("Required topK or concurrency matrix drifted")
    if value["matrix"]["measured_trials"] < 3:
        raise MilvusSindiError("At least three measured trials are required")
    representatives = value["workloads"]["representatives"]
    if [row["sparsity_band"] for row in representatives] != ["low", "medium", "high"]:
        raise MilvusSindiError("Representative workload sparsity order drifted")
    if any(row["native_tracks"] != ["economics", "psychology"] for row in representatives):
        raise MilvusSindiError("Every representative must run on both native tracks")
    if value["workloads"]["system_only"]["document_scales"] != [100000, 1000000]:
        raise MilvusSindiError("System-only scale matrix drifted")
    return value


def validate_source_attestation(path: str | Path, expected_sha256: str | None) -> dict[str, Any]:
    """Validate the redacted result of a full CPU-only S-008 replay."""
    expected = _require_sha256(expected_sha256, "source attestation SHA256")
    if file_sha256(path) != expected:
        raise MilvusSindiError("Source attestation identity drifted")
    value = _load_json(path)
    _validate_schema(value, SOURCE_ATTESTATION_SCHEMA)
    fixed = {
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "input_contract_sha256": SOURCE_INPUT_CONTRACT_SHA256,
        "main_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "main_manifest_sha256": SOURCE_MANIFEST_SHA256,
    }
    if any(value[key] != expected_value for key, expected_value in fixed.items()):
        raise MilvusSindiError("Source attestation does not bind the accepted S-008 source")
    if value["raw_cells"] != 14 or value["model_loaded"] is not False or value["cpu_only"] is not True:
        raise MilvusSindiError("Source replay must authenticate 14 cells without loading a model")
    if value["publication_gate"] != "closed":
        raise MilvusSindiError("Source publication gate drifted")
    return value


def build_csr_exact_catalog(
    *, input_contract_path: str | Path, batch_a_root: str | Path, batch_b_root: str | Path
) -> dict[str, Any]:
    """Build an identity-only catalog; never serialize IDs, vectors, scores, or rankings."""
    contract = _load_json(input_contract_path)
    expected_models_a = {
        "granite-30m-sparse",
        "opensearch-doc-v2-mini",
        "opensearch-doc-v3",
        "bge-m3",
    }
    cells: dict[str, Any] = {}
    for key, expected_manifest_sha256 in sorted(contract["raw_manifest_sha256"].items()):
        model, track = key.split(":", maxsplit=1)
        root = Path(batch_a_root if model in expected_models_a else batch_b_root) / model / track
        manifest_path = root / "manifest.json"
        if file_sha256(manifest_path) != expected_manifest_sha256:
            raise MilvusSindiError(f"Raw manifest identity drifted: {key}")
        manifest = _load_json(manifest_path)
        artifacts = manifest["artifacts"]
        document_parts = [
            {"path": name, "bytes": row["bytes"], "sha256": row["sha256"]}
            for name, row in sorted(artifacts.items())
            if name == "documents.npz" or (name.startswith("document_chunks/") and name.endswith(".npz"))
        ]
        if not document_parts:
            raise MilvusSindiError(f"No saved document CSR parts found: {key}")
        cells[key] = {
            "raw_manifest_sha256": expected_manifest_sha256,
            "document_csr_parts": document_parts,
            "queries_npz": artifacts["queries.npz"],
            "rankings_json": artifacts["rankings.json"],
            "per_query_metrics_json": artifacts["per_query_metrics.json"],
            "document_item_ids_sha256": manifest["representations"]["documents"]["item_ids_sha256"],
            "query_item_ids_sha256": manifest["representations"]["queries"]["item_ids_sha256"],
            "search": manifest["search"],
        }
    if len(cells) != 14:
        raise MilvusSindiError("CSR/exact catalog must contain exactly 14 accepted cells")
    return {
        "schema_version": "milvus-sindi-source-catalog-v1",
        "classification": "private_research_only",
        "contains_canonical_ids": False,
        "contains_vectors": False,
        "contains_rankings": False,
        "cells": cells,
    }


_BASELINE_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("uname.txt", ("uname", "-a")),
    ("lscpu.json", ("lscpu", "--json")),
    ("cpu_topology.txt", ("lscpu", "-e=CPU,NODE,SOCKET,CORE,ONLINE")),
    ("meminfo.txt", ("cat", "/proc/meminfo")),
    ("loadavg.txt", ("cat", "/proc/loadavg")),
    ("free.txt", ("free", "-b")),
    ("filesystems.txt", ("df", "-B1", "/", "/data1", "/data2")),
    ("listening_ports.txt", ("ss", "-H", "-lntup")),
    ("docker_version.json", ("docker", "version", "--format", "{{json .}}")),
    (
        "docker_containers.tsv",
        (
            "docker",
            "ps",
            "-a",
            "--no-trunc",
            "--size",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.State}}\t{{.Status}}\t{{.Ports}}\t{{.Mounts}}\t{{.Networks}}\t{{.Size}}",
        ),
    ),
    (
        "docker_networks.tsv",
        ("docker", "network", "ls", "--no-trunc", "--format", "{{.ID}}\t{{.Name}}\t{{.Driver}}\t{{.Scope}}"),
    ),
    (
        "docker_volumes.tsv",
        ("docker", "volume", "ls", "--format", "{{.Name}}\t{{.Driver}}\t{{.Scope}}\t{{.Mountpoint}}"),
    ),
    (
        "docker_stats.tsv",
        (
            "docker",
            "stats",
            "--no-stream",
            "--no-trunc",
            "--format",
            "{{.ID}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}",
        ),
    ),
    (
        "processes.tsv",
        ("ps", "-eo", "pid=,ppid=,user=,comm=,rss=,vsz=,pcpu=,pmem="),
    ),
)


def _run(command: Sequence[str], timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(command, check=False, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MilvusSindiError(f"Baseline command could not complete: {command[0]}") from exc


def _write_captured(
    root: Path, name: str, command: Sequence[str], result: subprocess.CompletedProcess[bytes]
) -> dict[str, Any]:
    path = root / name
    path.write_bytes(result.stdout)
    stderr_path = path.with_suffix(path.suffix + ".stderr")
    stderr_path.write_bytes(result.stderr)
    return {
        "argv": list(command),
        "exit_code": result.returncode,
        "stdout": {"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)},
        "stderr": {
            "path": stderr_path.name,
            "bytes": stderr_path.stat().st_size,
            "sha256": file_sha256(stderr_path),
        },
    }


def capture_baseline(output: str | Path) -> tuple[dict[str, Any], str]:
    """Capture private read-only host and Docker state without inspecting environment values."""
    root = Path(output)
    if root.exists():
        raise MilvusSindiError("Baseline output must not already exist")
    root.mkdir(parents=True)
    records: dict[str, Any] = {}
    for name, command in _BASELINE_COMMANDS:
        result = _run(command)
        records[name] = _write_captured(root, name, command, result)

    container_lines = (root / "docker_containers.tsv").read_text(encoding="utf-8", errors="replace").splitlines()
    related_ids = []
    for line in container_lines:
        fields = line.split("\t")
        if len(fields) >= 3 and re.search(
            r"milvus|zilliz|attu|woodpecker|(?:^|[-_])(?:etcd|minio)(?:[-_]|$)",
            "\t".join(fields[1:3]),
            flags=re.IGNORECASE,
        ):
            related_ids.append(fields[0])
    inspect_command = (
        "docker",
        "inspect",
        "--format",
        "{{json .Name}}\t{{json .Id}}\t{{json .Created}}\t{{json .Config.Image}}\t{{json .State}}\t{{json .HostConfig.Memory}}\t{{json .HostConfig.MemorySwap}}\t{{json .HostConfig.NanoCpus}}\t{{json .HostConfig.CpusetCpus}}\t{{json .NetworkSettings.Networks}}\t{{json .Mounts}}",
        *related_ids,
    )
    inspect_result = _run(inspect_command) if related_ids else subprocess.CompletedProcess(inspect_command, 0, b"", b"")
    records["milvus_related_container_inspect.tsv"] = _write_captured(
        root, "milvus_related_container_inspect.tsv", inspect_command, inspect_result
    )

    failures = sorted(name for name, row in records.items() if row["exit_code"] != 0)
    manifest = {
        "schema_version": "milvus-sindi-isolation-baseline-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "read_only": True,
        "environment_values_inspected": False,
        "related_container_count": len(related_ids),
        "records": records,
        "failed_records": failures,
        "status": "pass" if not failures else "incomplete",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(formatted_json_bytes(manifest))
    identity = file_sha256(manifest_path)
    (root / "manifest.sha256").write_text(identity + "\n", encoding="ascii")
    if failures:
        raise MilvusSindiError(f"Baseline capture was incomplete: {failures}")
    return manifest, identity


def validate_baseline(
    path: str | Path, expected_sha256: str | None, predeclaration: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a captured baseline and prove all proposed isolation names are unused."""
    expected = _require_sha256(expected_sha256, "baseline manifest SHA256")
    manifest_path = Path(path)
    if file_sha256(manifest_path) != expected:
        raise MilvusSindiError("Baseline manifest identity drifted")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "pass" or manifest.get("read_only") is not True:
        raise MilvusSindiError("Baseline must be complete and read-only")
    root = manifest_path.parent
    for record in manifest["records"].values():
        for stream in ("stdout", "stderr"):
            item = record[stream]
            artifact = root / item["path"]
            if artifact.stat().st_size != item["bytes"] or file_sha256(artifact) != item["sha256"]:
                raise MilvusSindiError(f"Baseline artifact identity drifted: {item['path']}")
    isolation = predeclaration["isolation"]
    combined_names = "\n".join(
        (root / name).read_text(encoding="utf-8", errors="replace")
        for name in ("docker_containers.tsv", "docker_networks.tsv", "docker_volumes.tsv")
    )
    for name in isolation["exclusive_names"]:
        if name in combined_names:
            raise MilvusSindiError(f"Proposed isolation name already exists: {name}")
    ports = (root / "listening_ports.txt").read_text(encoding="utf-8", errors="replace")
    for port in isolation["host_ports"].values():
        if re.search(rf":{port}(?:\s|$)", ports):
            raise MilvusSindiError(f"Proposed host port is already listening: {port}")
    return manifest


def _validate_record_set(root: Path, records: Mapping[str, Any]) -> dict[str, Any]:
    loaded = {}
    for name, record in records.items():
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise MilvusSindiError(f"Raw evidence identity drifted: {record['path']}")
        loaded[name] = _load_json(path)
    return loaded


def validate_deployment_evidence(path: str | Path, expected_sha256: str | None) -> dict[str, Any]:
    """Re-derive the isolated deployment facts from saved Docker responses."""
    expected = _require_sha256(expected_sha256, "deployment evidence SHA256")
    evidence_path = Path(path)
    if file_sha256(evidence_path) != expected:
        raise MilvusSindiError("Deployment evidence identity drifted")
    evidence = _load_json(evidence_path)
    schema_name = {
        "milvus-sindi-deployment-evidence-v1": DEPLOYMENT_EVIDENCE_SCHEMA,
        "milvus-sindi-deployment-evidence-v2": HARDENED_DEPLOYMENT_EVIDENCE_SCHEMA,
    }.get(evidence.get("schema_version"))
    if schema_name is None:
        raise MilvusSindiError("Unknown deployment evidence schema version")
    _validate_schema(evidence, schema_name)
    raw = _validate_record_set(evidence_path.parent, evidence["raw"])
    image = raw["image_inspect"]
    container = raw["container_inspect"]
    network = raw["network_inspect"]
    if len(image) != 1 or len(container) != 1 or len(network) != 1:
        raise MilvusSindiError("Deployment raw responses were not unique")
    image = image[0]
    container = container[0]
    network = network[0]
    if (
        image["Id"] != evidence["image_id"]
        or sorted(image["RepoDigests"]) != evidence["repo_digests"]
        or image["Architecture"] != "amd64"
        or image["Os"] != "linux"
    ):
        raise MilvusSindiError("Deployment summary does not match raw image inspect")
    if (
        container["Id"] != evidence["container_id"]
        or container["Name"] != f"/{evidence['container']}"
        or not container["State"]["Running"]
        or container["State"]["OOMKilled"]
        or container["HostConfig"]["Memory"] != evidence["resource_limits"]["memory_bytes"]
        or container["HostConfig"]["MemorySwap"] != evidence["resource_limits"]["memory_swap_bytes"]
        or container["HostConfig"]["CpusetCpus"] != evidence["resource_limits"]["cpuset_cpus"]
        or container["HostConfig"]["PidsLimit"] != evidence["resource_limits"]["pids_limit"]
    ):
        raise MilvusSindiError("Deployment summary does not match raw container inspect")
    attached = network.get("Containers") or {}
    if network["Id"] != evidence["network_id"] or set(attached) != {container["Id"]}:
        raise MilvusSindiError("Deployment raw network evidence is not exclusive")
    if raw["health"] != {"body": "OK", "status": 200}:
        raise MilvusSindiError("Deployment raw health response did not pass")
    if evidence["schema_version"] == "milvus-sindi-deployment-evidence-v2":
        data_root_stat = raw["data_root_stat"]
        expected_stat = evidence["data_root_stat"]
        if (
            data_root_stat != expected_stat
            or expected_stat["mode_octal"] != "0711"
            or expected_stat["world_writable"] is not False
            or expected_stat["is_directory"] is not True
        ):
            raise MilvusSindiError("Hardened bind root stat did not prove mode 0711")
    return evidence


def _validated_baseline_root(path: str | Path, expected_sha256: str | None) -> tuple[dict[str, Any], Path]:
    expected = _require_sha256(expected_sha256, "baseline manifest SHA256")
    manifest_path = Path(path)
    if file_sha256(manifest_path) != expected:
        raise MilvusSindiError("Baseline manifest identity drifted")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "pass" or manifest.get("read_only") is not True:
        raise MilvusSindiError("Baseline must be complete and read-only")
    root = manifest_path.parent
    for record in manifest["records"].values():
        for stream in ("stdout", "stderr"):
            item = record[stream]
            artifact = root / item["path"]
            if artifact.stat().st_size != item["bytes"] or file_sha256(artifact) != item["sha256"]:
                raise MilvusSindiError(f"Baseline artifact identity drifted: {item['path']}")
    return manifest, root


def _container_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.split("\t")
        if len(fields) != 9:
            raise MilvusSindiError("Container baseline row shape drifted")
        container_id, name, image, state, _status, ports, mounts, networks, _size = fields
        rows[name] = {
            "id": container_id,
            "image": image,
            "state": state,
            "ports": sorted(filter(None, ports.split(", "))),
            "mounts": sorted(filter(None, mounts.split(","))),
            "networks": sorted(filter(None, networks.split(","))),
        }
    return rows


def _named_rows(path: Path) -> dict[str, list[str]]:
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            raise MilvusSindiError("Docker object baseline row shape drifted")
        rows[fields[1]] = fields
    return rows


def _listening_endpoints(path: Path) -> set[tuple[str, str, str]]:
    endpoints = set()
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.split()
        if len(fields) < 5:
            raise MilvusSindiError("Listening port baseline row shape drifted")
        endpoints.add((fields[0], fields[1], fields[4]))
    return endpoints


def compare_isolation_baselines(
    *,
    prelaunch_manifest_path: str | Path,
    prelaunch_manifest_sha256: str | None,
    postfixture_manifest_path: str | Path,
    postfixture_manifest_sha256: str | None,
    output: str | Path,
) -> tuple[dict[str, Any], str]:
    """Prove that only the predeclared Story container/network/ports appeared."""
    pre_manifest, pre = _validated_baseline_root(prelaunch_manifest_path, prelaunch_manifest_sha256)
    post_manifest, post = _validated_baseline_root(postfixture_manifest_path, postfixture_manifest_sha256)
    output_path = Path(output)
    if output_path.exists():
        raise MilvusSindiError("Isolation regression output must not already exist")

    pre_containers = _container_rows(pre / "docker_containers.tsv")
    post_containers = _container_rows(post / "docker_containers.tsv")
    story_container = "meb-s010-milvus-v300-amd64"
    if story_container in pre_containers or story_container not in post_containers:
        raise MilvusSindiError("Story container baseline transition drifted")
    pre_non_story = {key: value for key, value in pre_containers.items() if key != story_container}
    post_non_story = {key: value for key, value in post_containers.items() if key != story_container}

    pre_networks = _named_rows(pre / "docker_networks.tsv")
    post_networks = _named_rows(post / "docker_networks.tsv")
    story_network = "meb-s010-sindi-net-v01"
    if story_network in pre_networks or story_network not in post_networks:
        raise MilvusSindiError("Story network baseline transition drifted")
    pre_non_story_networks = {key: value for key, value in pre_networks.items() if key != story_network}
    post_non_story_networks = {key: value for key, value in post_networks.items() if key != story_network}

    pre_volumes = (pre / "docker_volumes.tsv").read_text(encoding="utf-8", errors="strict").splitlines()
    post_volumes = (post / "docker_volumes.tsv").read_text(encoding="utf-8", errors="strict").splitlines()
    pre_ports = _listening_endpoints(pre / "listening_ports.txt")
    post_ports = _listening_endpoints(post / "listening_ports.txt")
    story_port_pattern = re.compile(r":(?:49531|49092)$")
    pre_non_story_ports = {row for row in pre_ports if not story_port_pattern.search(row[2])}
    post_non_story_ports = {row for row in post_ports if not story_port_pattern.search(row[2])}
    expected_story_ports = {
        ("tcp", "LISTEN", "127.0.0.1:49531"),
        ("tcp", "LISTEN", "127.0.0.1:49092"),
    }
    actual_story_ports = {row for row in post_ports if story_port_pattern.search(row[2])}

    checks = {
        "non_story_containers_unchanged": pre_non_story == post_non_story,
        "non_story_container_states_unchanged": {key: value["state"] for key, value in pre_non_story.items()}
        == {key: value["state"] for key, value in post_non_story.items()},
        "non_story_networks_unchanged": pre_non_story_networks == post_non_story_networks,
        "docker_volumes_unchanged": sorted(pre_volumes) == sorted(post_volumes),
        "non_story_listening_endpoints_unchanged": pre_non_story_ports == post_non_story_ports,
        "only_expected_story_listening_endpoints_added": actual_story_ports == expected_story_ports,
        "story_container_running": post_containers[story_container]["state"] == "running",
        "story_network_exclusive_name_present": story_network in post_networks,
    }
    if not all(checks.values()):
        raise MilvusSindiError(f"Isolation regression failed: {checks}")
    evidence = {
        "schema_version": "milvus-sindi-isolation-regression-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "prelaunch": {
            "manifest_sha256": prelaunch_manifest_sha256,
            "captured_at_utc": pre_manifest["captured_at_utc"],
        },
        "postfixture": {
            "manifest_sha256": postfixture_manifest_sha256,
            "captured_at_utc": post_manifest["captured_at_utc"],
        },
        "checks": checks,
        "allowed_additions": {
            "containers": [story_container],
            "networks": [story_network],
            "volumes": [],
            "listening_endpoints": [list(row) for row in sorted(expected_story_ports)],
        },
        "status": "pass",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(formatted_json_bytes(evidence))
    identity = file_sha256(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity


def validate_isolation_regression(
    *,
    isolation_path: str | Path,
    isolation_sha256: str | None,
    prelaunch_manifest_path: str | Path,
    prelaunch_manifest_sha256: str | None,
    postfixture_manifest_path: str | Path,
    postfixture_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Validate a hash-bound isolation result and both complete raw baselines."""
    expected = _require_sha256(isolation_sha256, "isolation regression SHA256")
    path = Path(isolation_path)
    if not path.is_file() or file_sha256(path) != expected:
        raise MilvusSindiError("Isolation regression identity drifted")
    evidence = _load_json(path)
    pre, _ = _validated_baseline_root(prelaunch_manifest_path, prelaunch_manifest_sha256)
    post, _ = _validated_baseline_root(postfixture_manifest_path, postfixture_manifest_sha256)
    if (
        evidence.get("status") != "pass"
        or not evidence.get("checks")
        or not all(evidence["checks"].values())
        or evidence.get("prelaunch")
        != {
            "manifest_sha256": prelaunch_manifest_sha256,
            "captured_at_utc": pre["captured_at_utc"],
        }
        or evidence.get("postfixture")
        != {
            "manifest_sha256": postfixture_manifest_sha256,
            "captured_at_utc": post["captured_at_utc"],
        }
    ):
        raise MilvusSindiError("Isolation regression does not bind two passing baselines")
    return evidence


def compare_recovery_baselines(
    *,
    prehardening_manifest_path: str | Path,
    prehardening_manifest_sha256: str | None,
    stopped_manifest_path: str | Path,
    stopped_manifest_sha256: str | None,
    posthardening_manifest_path: str | Path,
    posthardening_manifest_sha256: str | None,
    output: str | Path,
) -> tuple[dict[str, Any], str]:
    """Prove stop/start changed only the Story container state and Story ports."""
    pre_manifest, pre = _validated_baseline_root(prehardening_manifest_path, prehardening_manifest_sha256)
    stopped_manifest, stopped = _validated_baseline_root(stopped_manifest_path, stopped_manifest_sha256)
    post_manifest, post = _validated_baseline_root(posthardening_manifest_path, posthardening_manifest_sha256)
    output_path = Path(output)
    if output_path.exists():
        raise MilvusSindiError("Recovery isolation output must not already exist")
    story_container = "meb-s010-milvus-v300-amd64"
    story_network = "meb-s010-sindi-net-v01"
    container_sets = [_container_rows(root / "docker_containers.tsv") for root in (pre, stopped, post)]
    if any(story_container not in rows for rows in container_sets):
        raise MilvusSindiError("Story container disappeared during recovery")
    story_rows = [rows[story_container] for rows in container_sets]
    story_identity_fields = ("id", "image", "mounts", "networks")
    story_identity_unchanged = all(
        all(row[field] == story_rows[0][field] for field in story_identity_fields) for row in story_rows[1:]
    )
    non_story_containers = [
        {name: row for name, row in rows.items() if name != story_container} for rows in container_sets
    ]
    network_sets = [_named_rows(root / "docker_networks.tsv") for root in (pre, stopped, post)]
    volume_sets = [
        sorted((root / "docker_volumes.tsv").read_text(encoding="utf-8").splitlines()) for root in (pre, stopped, post)
    ]
    port_sets = [_listening_endpoints(root / "listening_ports.txt") for root in (pre, stopped, post)]
    story_port_pattern = re.compile(r":(?:49531|49092)$")
    story_ports = [
        {endpoint for endpoint in endpoints if story_port_pattern.search(endpoint[2])} for endpoints in port_sets
    ]
    non_story_ports = [
        {endpoint for endpoint in endpoints if not story_port_pattern.search(endpoint[2])} for endpoints in port_sets
    ]
    expected_story_ports = {
        ("tcp", "LISTEN", "127.0.0.1:49531"),
        ("tcp", "LISTEN", "127.0.0.1:49092"),
    }
    checks = {
        "same_story_container_identity": story_identity_unchanged,
        "story_state_running_exited_running": [row["state"] for row in story_rows] == ["running", "exited", "running"],
        "story_ports_present_released_restored": story_ports == [expected_story_ports, set(), expected_story_ports],
        "non_story_containers_unchanged": non_story_containers[0] == non_story_containers[1] == non_story_containers[2],
        "docker_networks_unchanged": network_sets[0] == network_sets[1] == network_sets[2],
        "story_network_id_unchanged": all(
            networks.get(story_network) == network_sets[0].get(story_network) for networks in network_sets[1:]
        ),
        "docker_volumes_unchanged": volume_sets[0] == volume_sets[1] == volume_sets[2],
        "non_story_listening_endpoints_unchanged": non_story_ports[0] == non_story_ports[1] == non_story_ports[2],
    }
    if not all(checks.values()):
        raise MilvusSindiError(f"Recovery isolation regression failed: {checks}")
    evidence = {
        "schema_version": "milvus-sindi-recovery-isolation-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "prehardening": {
            "manifest_sha256": prehardening_manifest_sha256,
            "captured_at_utc": pre_manifest["captured_at_utc"],
        },
        "stopped": {
            "manifest_sha256": stopped_manifest_sha256,
            "captured_at_utc": stopped_manifest["captured_at_utc"],
        },
        "posthardening": {
            "manifest_sha256": posthardening_manifest_sha256,
            "captured_at_utc": post_manifest["captured_at_utc"],
        },
        "checks": checks,
        "story_container_id": story_rows[0]["id"],
        "story_container_image": story_rows[0]["image"],
        "status": "pass",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(formatted_json_bytes(evidence))
    identity = file_sha256(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return evidence, identity


def validate_recovery_isolation(
    path: str | Path,
    expected_sha256: str | None,
) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, "recovery isolation SHA256")
    evidence_path = Path(path)
    if not evidence_path.is_file() or file_sha256(evidence_path) != expected:
        raise MilvusSindiError("Recovery isolation identity drifted")
    evidence = _load_json(evidence_path)
    if (
        evidence.get("schema_version") != "milvus-sindi-recovery-isolation-v1"
        or evidence.get("status") != "pass"
        or not evidence.get("checks")
        or not all(evidence["checks"].values())
    ):
        raise MilvusSindiError("Recovery isolation did not pass every check")
    return evidence


def validate_formal_gate(
    *,
    predeclaration_path: str | Path,
    predeclaration_sha256: str | None,
    source_attestation_path: str | Path,
    source_attestation_sha256: str | None,
    baseline_manifest_path: str | Path,
    baseline_manifest_sha256: str | None,
    fixture_evidence_path: str | Path,
    fixture_evidence_sha256: str | None,
    deployment_evidence_path: str | Path | None = None,
    deployment_evidence_sha256: str | None = None,
    postfixture_manifest_path: str | Path | None = None,
    postfixture_manifest_sha256: str | None = None,
    isolation_regression_path: str | Path | None = None,
    isolation_regression_sha256: str | None = None,
    hardened_deployment_evidence_path: str | Path | None = None,
    hardened_deployment_evidence_sha256: str | None = None,
    recovery_evidence_path: str | Path | None = None,
    recovery_evidence_sha256: str | None = None,
    prehardening_manifest_path: str | Path | None = None,
    prehardening_manifest_sha256: str | None = None,
    stopped_manifest_path: str | Path | None = None,
    stopped_manifest_sha256: str | None = None,
    posthardening_manifest_path: str | Path | None = None,
    posthardening_manifest_sha256: str | None = None,
    recovery_isolation_path: str | Path | None = None,
    recovery_isolation_sha256: str | None = None,
) -> dict[str, Any]:
    """Open the formal gate only after all immutable prerequisite evidence exists."""
    predeclaration = validate_predeclaration(predeclaration_path, predeclaration_sha256)
    source = validate_source_attestation(source_attestation_path, source_attestation_sha256)
    baseline = validate_baseline(baseline_manifest_path, baseline_manifest_sha256, predeclaration)
    fixture_expected = _require_sha256(fixture_evidence_sha256, "fixture evidence SHA256")
    from mm_embed.benchmark.milvus_sindi_fixture import validate_fixture_evidence

    fixture = validate_fixture_evidence(fixture_evidence_path, fixture_expected)
    if fixture["server_image"] != SERVER_IMAGE or fixture["pymilvus_version"] != PYMILVUS_VERSION:
        raise MilvusSindiError("Fixture deployment identity drifted")
    if deployment_evidence_path is None:
        raise MilvusSindiError("Deployment evidence path is required")
    deployment_expected = _require_sha256(deployment_evidence_sha256, "deployment evidence SHA256")
    deployment = validate_deployment_evidence(deployment_evidence_path, deployment_expected)
    if fixture["deployment_evidence_sha256"] != deployment_expected:
        raise MilvusSindiError("Fixture does not bind the validated deployment evidence")
    if postfixture_manifest_path is None or isolation_regression_path is None:
        raise MilvusSindiError("Post-fixture baseline and isolation regression are required")
    isolation = validate_isolation_regression(
        isolation_path=isolation_regression_path,
        isolation_sha256=isolation_regression_sha256,
        prelaunch_manifest_path=baseline_manifest_path,
        prelaunch_manifest_sha256=baseline_manifest_sha256,
        postfixture_manifest_path=postfixture_manifest_path,
        postfixture_manifest_sha256=postfixture_manifest_sha256,
    )
    required_hardening_paths = {
        "hardened deployment": hardened_deployment_evidence_path,
        "recovery evidence": recovery_evidence_path,
        "prehardening baseline": prehardening_manifest_path,
        "stopped baseline": stopped_manifest_path,
        "posthardening baseline": posthardening_manifest_path,
        "recovery isolation": recovery_isolation_path,
    }
    missing = [name for name, path in required_hardening_paths.items() if path is None]
    if missing:
        raise MilvusSindiError(f"Hardening gate evidence is required: {', '.join(missing)}")
    hardened_deployment_expected = _require_sha256(
        hardened_deployment_evidence_sha256, "hardened deployment evidence SHA256"
    )
    hardened_deployment = validate_deployment_evidence(hardened_deployment_evidence_path, hardened_deployment_expected)
    from mm_embed.benchmark.milvus_sindi_fixture import validate_recovery_evidence

    recovery_expected = _require_sha256(recovery_evidence_sha256, "recovery evidence SHA256")
    recovery = validate_recovery_evidence(recovery_evidence_path, recovery_expected)
    prehardening, _ = _validated_baseline_root(prehardening_manifest_path, prehardening_manifest_sha256)
    stopped, _ = _validated_baseline_root(stopped_manifest_path, stopped_manifest_sha256)
    posthardening, _ = _validated_baseline_root(posthardening_manifest_path, posthardening_manifest_sha256)
    recovery_isolation = validate_recovery_isolation(recovery_isolation_path, recovery_isolation_sha256)
    expected_recovery_bindings = {
        "fixture_evidence_sha256": fixture_expected,
        "hardened_deployment_sha256": hardened_deployment_expected,
        "prehardening_baseline_sha256": prehardening_manifest_sha256,
        "stopped_baseline_sha256": stopped_manifest_sha256,
    }
    if any(recovery[key] != value for key, value in expected_recovery_bindings.items()):
        raise MilvusSindiError("Recovery evidence does not bind the exact prerequisite identities")
    expected_isolation_bindings = {
        "prehardening": {
            "manifest_sha256": prehardening_manifest_sha256,
            "captured_at_utc": prehardening["captured_at_utc"],
        },
        "stopped": {
            "manifest_sha256": stopped_manifest_sha256,
            "captured_at_utc": stopped["captured_at_utc"],
        },
        "posthardening": {
            "manifest_sha256": posthardening_manifest_sha256,
            "captured_at_utc": posthardening["captured_at_utc"],
        },
    }
    if any(recovery_isolation.get(key) != value for key, value in expected_isolation_bindings.items()):
        raise MilvusSindiError("Recovery isolation does not bind all three raw baselines")
    if (
        hardened_deployment["schema_version"] != "milvus-sindi-deployment-evidence-v2"
        or hardened_deployment["container_id"] != deployment["container_id"]
        or hardened_deployment["network_id"] != deployment["network_id"]
        or recovery["container_id"] != deployment["container_id"]
        or recovery["network_id"] != deployment["network_id"]
        or recovery["runtime_root_mode"] != "0711"
        or recovery["world_writable"] is not False
    ):
        raise MilvusSindiError("Hardened runtime identity or permission chain drifted")
    return {
        "schema_version": "milvus-sindi-formal-gate-v1",
        "story_id": STORY_ID,
        "classification": "private_research_only",
        "status": "ready",
        "formal_performance_executed": False,
        "predeclaration_sha256": predeclaration_sha256,
        "source_attestation_sha256": source_attestation_sha256,
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "fixture_evidence_sha256": fixture_evidence_sha256,
        "deployment_evidence_sha256": deployment_evidence_sha256,
        "postfixture_baseline_manifest_sha256": postfixture_manifest_sha256,
        "isolation_regression_sha256": isolation_regression_sha256,
        "hardened_deployment_evidence_sha256": hardened_deployment_evidence_sha256,
        "recovery_evidence_sha256": recovery_evidence_sha256,
        "prehardening_baseline_manifest_sha256": prehardening_manifest_sha256,
        "stopped_baseline_manifest_sha256": stopped_manifest_sha256,
        "posthardening_baseline_manifest_sha256": posthardening_manifest_sha256,
        "recovery_isolation_sha256": recovery_isolation_sha256,
        "runtime_root_mode": "0711",
        "world_writable": False,
        "container_id": recovery["container_id"],
        "network_id": recovery["network_id"],
        "publication_gate": predeclaration["publication"]["gate"],
        "source_replay_sha256": source["source_replay_sha256"],
        "baseline_captured_at_utc": baseline["captured_at_utc"],
        "deployment_captured_at_utc": deployment["captured_at_utc"],
        "postfixture_captured_at_utc": isolation["postfixture"]["captured_at_utc"],
    }


def write_identity_artifact(path: str | Path, value: Any) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise MilvusSindiError("Identity artifact output must not already exist")
    output.write_bytes(formatted_json_bytes(value))
    identity = file_sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(identity + "\n", encoding="ascii")
    return identity
