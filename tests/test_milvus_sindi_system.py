from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from mm_embed.benchmark import milvus_sindi_fixture as fixture_system
from mm_embed.benchmark import milvus_sindi_results as formal_results
from mm_embed.benchmark import milvus_sindi_system as system

PREDECLARATION = Path("benchmark/artifacts/milvus-sindi-system-v0.1/predeclaration.json")
SOURCE_ATTESTATION = Path("benchmark/artifacts/milvus-sindi-system-v0.1/source-attestation.json")
PREDECLARATION_SHA256 = "d1c3e3c9c2ad1e7e4b79bf37b0088e480058196a570f56d21ccc1a4bd2ff47d1"
SOURCE_ATTESTATION_SHA256 = "dce7e8c088954bfc5d689cea1017f13b088dc417532d5b5a9a312abe6cf46f65"
FORMAL_CELL_SCHEMA = Path("schemas/milvus-sindi-formal-cell-v01.schema.json")
FORMAL_SUMMARY_SCHEMA = Path("schemas/milvus-sindi-formal-summary-v01.schema.json")


def test_predeclaration_is_active_complete_and_closed() -> None:
    value = system.validate_predeclaration(PREDECLARATION, PREDECLARATION_SHA256)
    assert value["deployment"]["server_image"] == system.SERVER_IMAGE
    assert value["deployment"]["pymilvus_version"] == system.PYMILVUS_VERSION
    assert [row["model_key"] for row in value["workloads"]["representatives"]] == [
        "bge-m3",
        "granite-30m-sparse",
        "opensearch-multilingual",
    ]
    assert value["matrix"]["search_params"]["params"]["drop_ratio_search"] == 0.0
    assert value["publication"]["gate"] == "closed"


def test_predeclaration_requires_out_of_band_identity() -> None:
    with pytest.raises(system.MilvusSindiError, match="unique active"):
        system.validate_predeclaration(PREDECLARATION, "0" * 64)
    with pytest.raises(system.MilvusSindiError, match="externally supplied"):
        system.validate_predeclaration(PREDECLARATION, None)


def test_tracked_sidecars_match_their_artifacts() -> None:
    for path in (PREDECLARATION, SOURCE_ATTESTATION):
        assert path.with_suffix(".sha256").read_text(encoding="ascii").strip() == system.file_sha256(path)


def test_source_attestation_proves_cpu_only_replay_without_private_rows() -> None:
    value = system.validate_source_attestation(SOURCE_ATTESTATION, SOURCE_ATTESTATION_SHA256)
    assert value["raw_cells"] == 14
    assert value["document_csr_parts"] == 426
    assert value["model_loaded"] is False
    assert value["source_artifacts_mutated"] is False
    text = SOURCE_ATTESTATION.read_text(encoding="utf-8")
    for forbidden in ("query_id", "document_id", "rankings.json"):
        assert forbidden not in text
    assert value["contains_source_text"] is False
    assert value["contains_canonical_ids"] is False
    assert value["contains_raw_rankings"] is False


def test_csr_exact_catalog_binds_all_saved_parts_without_payloads() -> None:
    value = system.build_csr_exact_catalog(
        input_contract_path="benchmark/contracts/bright-learned-sparse-main-v0.1-inputs.json",
        batch_a_root="results/bright-learned-sparse-batch-a",
        batch_b_root="results/bright-learned-sparse-batch-b",
    )
    assert len(value["cells"]) == 14
    assert sum(len(row["document_csr_parts"]) for row in value["cells"].values()) == 426
    assert system.canonical_sha256(value["cells"]) == system.canonical_sha256(value["cells"])
    assert value["contains_canonical_ids"] is False
    assert value["contains_vectors"] is False
    assert value["contains_rankings"] is False


def test_baseline_capture_is_read_only_identity_bound_and_rejects_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str] | tuple[str, ...], timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
        del timeout
        output = b""
        if list(command[:3]) == ["docker", "ps", "-a"]:
            output = b"abc\texisting-milvus\tmilvusdb/milvus:v2.6.22\trunning\tUp\t0.0.0.0:19530->19530/tcp\told\toldnet\t1GB\n"
        return subprocess.CompletedProcess(command, 0, output, b"")

    monkeypatch.setattr(system, "_run", fake_run)
    root = tmp_path / "baseline"
    manifest, identity = system.capture_baseline(root)
    assert manifest["read_only"] is True
    assert manifest["environment_values_inspected"] is False
    assert manifest["related_container_count"] == 1
    predeclaration = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    validated = system.validate_baseline(root / "manifest.json", identity, predeclaration)
    assert validated["status"] == "pass"
    with pytest.raises(system.MilvusSindiError, match="must not already exist"):
        system.capture_baseline(root)


def test_baseline_rejects_port_or_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proposed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))

    def fake_run(command: list[str] | tuple[str, ...], timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
        del timeout
        if command[0] == "ss":
            return subprocess.CompletedProcess(command, 0, b"tcp LISTEN 0 128 127.0.0.1:49531 0.0.0.0:*\n", b"")
        if list(command[:3]) == ["docker", "ps", "-a"]:
            output = b"abc\tmeb-s010-milvus-v300-amd64\tother\texited\tExited\t\t\t\t0B\n"
            return subprocess.CompletedProcess(command, 0, output, b"")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(system, "_run", fake_run)
    root = tmp_path / "baseline"
    _, identity = system.capture_baseline(root)
    with pytest.raises(system.MilvusSindiError, match="isolation name already exists"):
        system.validate_baseline(root / "manifest.json", identity, proposed)


def test_formal_gate_fails_closed_without_exact_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predeclaration = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_ATTESTATION.read_text(encoding="utf-8"))
    baseline = {"captured_at_utc": "2026-08-30T00:00:00+00:00"}
    monkeypatch.setattr(system, "validate_predeclaration", lambda *_args: copy.deepcopy(predeclaration))
    monkeypatch.setattr(system, "validate_source_attestation", lambda *_args: copy.deepcopy(source))
    monkeypatch.setattr(system, "validate_baseline", lambda *_args: copy.deepcopy(baseline))
    with pytest.raises(system.MilvusSindiError, match="Fixture evidence identity drifted"):
        system.validate_formal_gate(
            predeclaration_path="unused",
            predeclaration_sha256="1" * 64,
            source_attestation_path="unused",
            source_attestation_sha256="2" * 64,
            baseline_manifest_path="unused",
            baseline_manifest_sha256="3" * 64,
            fixture_evidence_path=tmp_path / "missing.json",
            fixture_evidence_sha256="4" * 64,
        )


def test_fixture_exact_csr_scoring_enforces_ties_uniqueness_and_finite_scores() -> None:
    fixture = fixture_system._fixture_input()
    exact = fixture_system._exact_results(fixture)
    actual = [[{"pk": hit["id"], "distance": hit["score"]} for hit in query[: fixture["top_k"]]] for query in exact]
    score = fixture_system.score_fixture_results(exact, actual)
    assert score["minimum_strict_id_recall_at_k"] == 1.0
    assert score["minimum_tie_aware_recall_at_k"] == 1.0
    assert score["maximum_absolute_score_delta"] == 0.0
    assert score["all_query_ids_unique"] is True
    assert score["all_scores_finite"] is True
    assert len(exact[3]) > fixture["top_k"]
    assert exact[3][4]["score"] == exact[3][5]["score"]


def test_isolation_regression_allows_only_exact_story_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = {"name": "pre"}

    def fake_run(command: list[str] | tuple[str, ...], timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
        del timeout
        output = b""
        if list(command[:3]) == ["docker", "ps", "-a"]:
            mounts = "b,a" if stage["name"] == "pre" else "a,b"
            base = f"baseid\texisting\texample:v1\trunning\tUp\t127.0.0.1:12345->80/tcp\t{mounts}\tbase-net\t1MB\n"
            story = (
                "storyid\tmeb-s010-milvus-v300-amd64\tmilvusdb/milvus@sha256:804b50bc1523a64e3c0f18cf33af7e0f8b33329584698ffee61c606d6fe8ee26\t"
                "running\tUp\t127.0.0.1:49531->19530/tcp, 127.0.0.1:49092->9091/tcp\tstory\tmeb-s010-sindi-net-v01\t1MB\n"
            )
            output = (base + (story if stage["name"] == "post" else "")).encode()
        elif list(command[:3]) == ["docker", "network", "ls"]:
            base = "netid\tbase-net\tbridge\tlocal\n"
            story = "storynetid\tmeb-s010-sindi-net-v01\tbridge\tlocal\n"
            output = (base + (story if stage["name"] == "post" else "")).encode()
        elif list(command[:3]) == ["docker", "volume", "ls"]:
            output = b"volume-a\tlocal\tlocal\t/path\n"
        elif command[0] == "ss":
            base = "tcp LISTEN 0 128 127.0.0.1:12345 0.0.0.0:*\n"
            story = "tcp LISTEN 0 128 127.0.0.1:49531 0.0.0.0:*\ntcp LISTEN 0 128 127.0.0.1:49092 0.0.0.0:*\n"
            output = (base + (story if stage["name"] == "post" else "")).encode()
        return subprocess.CompletedProcess(command, 0, output, b"")

    monkeypatch.setattr(system, "_run", fake_run)
    pre_manifest, pre_sha = system.capture_baseline(tmp_path / "pre")
    stage["name"] = "post"
    post_manifest, post_sha = system.capture_baseline(tmp_path / "post")
    evidence, identity = system.compare_isolation_baselines(
        prelaunch_manifest_path=tmp_path / "pre" / "manifest.json",
        prelaunch_manifest_sha256=pre_sha,
        postfixture_manifest_path=tmp_path / "post" / "manifest.json",
        postfixture_manifest_sha256=post_sha,
        output=tmp_path / "isolation.json",
    )
    assert pre_manifest["status"] == post_manifest["status"] == "pass"
    assert evidence["status"] == "pass"
    assert all(evidence["checks"].values())
    assert identity == system.file_sha256(tmp_path / "isolation.json")


def test_compaction_metrics_validation_requires_every_idle_signal() -> None:
    raw = b"""\
milvus_datanode_pool_active_threads{node_id="1",pool_name="CompactionExecPool"} 0
milvus_datanode_pool_queue_depth{node_id="1",pool_name="CompactionExecPool"} 0
milvus_datanode_slot{node_id="1",type="compactionUsed"} 0
milvus_datacoord_compaction_task_num{node_id="-1",status="pending"} 0
milvus_datacoord_compaction_task_num{node_id="0",status="executing"} 3
milvus_datacoord_compaction_task_num{node_id="1",status="executing"} -3
"""
    assert fixture_system._validate_compaction_metrics(raw)["active"] is False
    with pytest.raises(system.MilvusSindiError, match="Active or queued compaction"):
        fixture_system._validate_compaction_metrics(
            raw.replace(b'type="compactionUsed"} 0', b'type="compactionUsed"} 1')
        )


def test_recovery_compaction_requires_direct_active_and_queue_gauges() -> None:
    raw = b"""\
milvus_datanode_pool_active_threads{node_id="2",pool_name="CompactionExecPool"} 0
milvus_datanode_pool_queue_depth{node_id="2",pool_name="CompactionExecPool"} 0
"""
    summary = fixture_system._validate_recovery_compaction_metrics(raw)
    assert summary["active"] is False
    assert summary["optional_slot_gauge_present"] is False
    assert summary["optional_datacoord_task_gauge_present"] is False
    with pytest.raises(system.MilvusSindiError, match="Active or queued compaction"):
        fixture_system._validate_recovery_compaction_metrics(
            raw.replace(b"pool_active_threads{", b"pool_active_threads{").replace(
                b'pool_name="CompactionExecPool"} 0',
                b'pool_name="CompactionExecPool"} 1',
                1,
            )
        )


def test_formal_cell_schema_is_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(json.loads(FORMAL_CELL_SCHEMA.read_text(encoding="utf-8")))
    Draft202012Validator.check_schema(json.loads(FORMAL_SUMMARY_SCHEMA.read_text(encoding="utf-8")))


def test_independent_formal_correctness_reducer_rejects_duplicate_or_nonfinite_hits() -> None:
    exact = [[{"id": 1, "score": 0.5}, {"id": 2, "score": 0.25}]]
    valid = [{"query_ordinal": 0, "ids": [1, 2], "scores": [0.5, 0.25]}]
    result = formal_results._independent_correctness(valid, exact, 2)
    assert result == {
        "minimum_strict_recall": 1.0,
        "all_ids_unique": True,
        "all_scores_finite": True,
        "maximum_absolute_score_delta": 0.0,
    }
    duplicate = [{"query_ordinal": 0, "ids": [1, 1], "scores": [0.5, float("nan")]}]
    result = formal_results._independent_correctness(duplicate, exact, 2)
    assert result["all_ids_unique"] is False
    assert result["all_scores_finite"] is False


def test_recovery_isolation_requires_running_stopped_running_same_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = {"name": "pre"}

    def fake_run(command: list[str] | tuple[str, ...], timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
        del timeout
        output = b""
        if list(command[:3]) == ["docker", "ps", "-a"]:
            mounts = {"pre": "b,a", "stopped": "a,b", "post": "b,a"}[stage["name"]]
            base = f"baseid\texisting\texample:v1\trunning\tUp\t127.0.0.1:12345->80/tcp\t{mounts}\tbase-net\t1MB\n"
            state = "exited" if stage["name"] == "stopped" else "running"
            ports = "" if stage["name"] == "stopped" else "127.0.0.1:49531->19530/tcp, 127.0.0.1:49092->9091/tcp"
            story = (
                "storyid\tmeb-s010-milvus-v300-amd64\tmilvusdb/milvus@sha256:804b50bc1523a64e3c0f18cf33af7e0f8b33329584698ffee61c606d6fe8ee26\t"
                f"{state}\tstate-detail\t{ports}\tstory\tmeb-s010-sindi-net-v01\t1MB\n"
            )
            output = (base + story).encode()
        elif list(command[:3]) == ["docker", "network", "ls"]:
            output = b"netid\tbase-net\tbridge\tlocal\nstorynetid\tmeb-s010-sindi-net-v01\tbridge\tlocal\n"
        elif list(command[:3]) == ["docker", "volume", "ls"]:
            output = b"volume-a\tlocal\tlocal\t/path\n"
        elif command[0] == "ss":
            base = "tcp LISTEN 0 128 127.0.0.1:12345 0.0.0.0:*\n"
            story = (
                ""
                if stage["name"] == "stopped"
                else ("tcp LISTEN 0 128 127.0.0.1:49531 0.0.0.0:*\ntcp LISTEN 0 128 127.0.0.1:49092 0.0.0.0:*\n")
            )
            output = (base + story).encode()
        return subprocess.CompletedProcess(command, 0, output, b"")

    monkeypatch.setattr(system, "_run", fake_run)
    identities = {}
    for name in ("pre", "stopped", "post"):
        stage["name"] = name
        _, identities[name] = system.capture_baseline(tmp_path / name)
    evidence, identity = system.compare_recovery_baselines(
        prehardening_manifest_path=tmp_path / "pre" / "manifest.json",
        prehardening_manifest_sha256=identities["pre"],
        stopped_manifest_path=tmp_path / "stopped" / "manifest.json",
        stopped_manifest_sha256=identities["stopped"],
        posthardening_manifest_path=tmp_path / "post" / "manifest.json",
        posthardening_manifest_sha256=identities["post"],
        output=tmp_path / "recovery-isolation.json",
    )
    assert evidence["status"] == "pass"
    assert all(evidence["checks"].values())
    assert identity == system.file_sha256(tmp_path / "recovery-isolation.json")


def test_formal_run_manifest_validation_resolves_only_missing_inline_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "formal"
    cell_paths = []
    specs = (("native", "native-run.json", 18), ("100k", "system-100k-run.json", 9), ("1m", "system-1m-run.json", 9))
    for group, filename, count in specs:
        paths = []
        for ordinal in range(count):
            prefix = root / "native" if group == "native" else root / "system-only" / group
            path = prefix / str(ordinal) / "cell-evidence.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"status":"pass"}\n', encoding="utf-8")
            identity = system.file_sha256(path)
            path.with_suffix(".json.sha256").write_text(identity + "\n", encoding="ascii")
            paths.append(path)
            cell_paths.append(path)
        completed = []
        for ordinal, path in enumerate(paths):
            record = {"ordinal": ordinal, "path": str(path), "resumed": ordinal == 0}
            if not (group == "native" and ordinal == 0):
                record["sha256"] = system.file_sha256(path)
            completed.append(record)
        run = {
            "status": "pass",
            "completed_cell_count": count,
            "completed_cells": completed,
            "runtime": {"model_loaded": False},
            "publication_gate": "closed",
        }
        run_path = root / filename
        run_path.write_bytes(system.formatted_json_bytes(run))
        run_path.with_suffix(".json.sha256").write_text(system.file_sha256(run_path) + "\n", encoding="ascii")
    monkeypatch.setattr(formal_results, "_cell_paths", lambda _root: cell_paths)
    evidence, _ = formal_results.validate_run_manifests(root, output=tmp_path / "integrity.json")
    assert evidence["total_cell_count"] == 36
    assert sum(run["missing_inline_sha256_count"] for run in evidence["runs"]) == 1
    assert evidence["runs"][0]["normalized_cells"][0]["resolution"] == "content_and_sidecar"

    native = root / "native-run.json"
    tampered = json.loads(native.read_text(encoding="utf-8"))
    tampered["completed_cells"][1]["sha256"] = "0" * 64
    native.write_bytes(system.formatted_json_bytes(tampered))
    native.with_suffix(".json.sha256").write_text(system.file_sha256(native) + "\n", encoding="ascii")
    with pytest.raises(system.MilvusSindiError, match="inline cell identity drifted"):
        formal_results.validate_run_manifests(root, output=tmp_path / "rejected.json")


def test_candidate_manifest_rejects_private_paths_and_open_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    aggregate_dir = Path("benchmark/artifacts/milvus-sindi-system-v0.1")
    report_dir = Path("benchmark/research")
    aggregate_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    summary = aggregate_dir / "formal-summary.json"
    summary.write_text(
        json.dumps(
            {
                "contains_source_text": False,
                "contains_canonical_ids": False,
                "contains_raw_rankings": False,
                "publication_gate": "closed",
                "research_only": True,
            }
        ),
        encoding="utf-8",
    )
    report = report_dir / "milvus_sindi_system_report_20260904.md"
    report.write_text("research-only aggregate\n", encoding="utf-8")
    source = Path("token_parser.py")
    source.write_text('token = json.loads(body)["token"]\n', encoding="utf-8")
    evidence, _ = formal_results.build_candidate_manifest([summary, report, source], output=Path("candidate.json"))
    assert evidence["status"] == "candidate_not_accepted"
    credential_name = "tok" + "en"
    secret_value = "literal-" + "value-that-must-not-pass"
    source.write_text(f'{credential_name} = "{secret_value}"\n', encoding="utf-8")
    with pytest.raises(system.MilvusSindiError, match="Potential secret"):
        formal_results.build_candidate_manifest([summary, report, source], output=Path("secret.json"))
    private = Path("results/private.txt")
    private.parent.mkdir()
    private.write_text("private", encoding="utf-8")
    with pytest.raises(system.MilvusSindiError, match="private-only"):
        formal_results.build_candidate_manifest([summary, report, private], output=Path("rejected.json"))
    value = json.loads(summary.read_text(encoding="utf-8"))
    value["publication_gate"] = "open"
    summary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(system.MilvusSindiError, match="privacy/publication gate"):
        formal_results.build_candidate_manifest([summary, report], output=Path("open.json"))


def test_final_isolation_records_ambient_identity_drift_but_rejects_state_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = {name: tmp_path / name for name in ("pre", "post")}
    for root in roots.values():
        root.mkdir()
        (root / "docker_volumes.tsv").write_text("volume-a\tlocal\tlocal\t/path\n", encoding="utf-8")
    story = {
        "id": "story-id",
        "image": "pinned",
        "state": "running",
        "ports": ["story-ports"],
        "mounts": ["story-root"],
        "networks": ["story-network"],
    }
    base = {
        "id": "base-before",
        "image": "base-image",
        "state": "running",
        "ports": ["base-port"],
        "mounts": ["base-mount"],
        "networks": ["base-network"],
    }
    post_base = {**base, "id": "base-after"}
    rows = {
        "pre": {"meb-s010-milvus-v300-amd64": story, "unrelated": base},
        "post": {"meb-s010-milvus-v300-amd64": story, "unrelated": post_base},
    }

    def fake_validated(path: str | Path, expected: str) -> tuple[dict[str, str], Path]:
        del expected
        key = "pre" if "pre" in str(path) else "post"
        return {"captured_at_utc": f"2026-09-04T00:00:0{int(key == 'post')}+00:00"}, roots[key]

    monkeypatch.setattr(system, "_validated_baseline_root", fake_validated)
    monkeypatch.setattr(system, "_container_rows", lambda path: rows[path.parent.name])
    monkeypatch.setattr(system, "_named_rows", lambda _path: {"base-network": ["id", "base-network"]})
    endpoints = {
        ("tcp", "LISTEN", "127.0.0.1:49531"),
        ("tcp", "LISTEN", "127.0.0.1:49092"),
        ("tcp", "LISTEN", "127.0.0.1:12345"),
    }
    monkeypatch.setattr(system, "_listening_endpoints", lambda _path: endpoints)
    evidence, _ = formal_results.build_final_isolation_evidence(
        pre_manifest="pre/manifest.json",
        pre_sha256="1" * 64,
        post_manifest="post/manifest.json",
        post_sha256="2" * 64,
        output=tmp_path / "isolation.json",
    )
    assert evidence["status"] == "pass_with_ambient_drift"
    assert len(evidence["non_story_ambient_drift"]["changed_containers"]) == 1

    rows["post"]["unrelated"] = {**post_base, "state": "exited"}
    with pytest.raises(system.MilvusSindiError, match="boundary invariant failed"):
        formal_results.build_final_isolation_evidence(
            pre_manifest="pre/manifest.json",
            pre_sha256="1" * 64,
            post_manifest="post/manifest.json",
            post_sha256="2" * 64,
            output=tmp_path / "rejected.json",
        )
