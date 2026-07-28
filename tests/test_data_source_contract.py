from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mm_embed.benchmark.materialization import (
    DATA_SOURCE_CONTRACT_VERSION,
    MaterializationContractError,
    asset_manifest_sha256,
    legacy_unknown_data_source_contract,
    ordered_jsonl_manifest_sha256,
    prepare_data_source_contract,
    validate_data_source_snapshot,
    validate_materialization_authorization,
    validate_materialization_manifest,
)
from mm_embed.benchmark.registry import BenchmarkCatalog, RunManifest, RunTask, TaskSpec, load_catalog
from mm_embed.benchmark.results import make_result_record
from mm_embed.benchmark.runner import BenchmarkRunner
from mm_embed.benchmark.training_overlap import load_relationship_registry
from mm_embed.data import real_data
from mm_embed.hf_publish.export import export_dataset_repo
from mm_embed.providers.base import ModalityType
from mm_embed.tasks.base import EvalResult
from mm_embed.tasks.cross_modal_retrieval import CrossModalRetrievalTask
from mm_embed.tasks.crosslingual_retrieval import CrossLingualRetrievalTask
from mm_embed.tasks.mrl_stress import MRLStressTask
from mm_embed.tasks.needle_in_haystack import NeedleInHaystackTask


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_task() -> TaskSpec:
    return TaskSpec.from_dict(
        {
            "id": "invented_task",
            "display_name": "Invented task",
            "task": "invented_task",
            "description": "Invented materialization contract fixture.",
            "dataset_version": "invented-v1",
            "evaluation_sources": {
                "disclosure": "complete",
                "sources": [
                    {
                        "source_id": "invented:source",
                        "usage": "evaluation",
                        "source_revision": "source-r1",
                        "config": "default",
                        "split": "test",
                        "transformation_id": "invented-transform-v1",
                    }
                ],
                "public_provenance": {
                    "urls": ["https://example.invalid/source"],
                    "evidence_revision": "evidence-r1",
                    "reviewed_at": "2026-07-28T00:00:00Z",
                    "reviewed_by": "reviewer",
                },
                "review": {"state": "approved"},
            },
            "materialization": {"transformation_parameters": {"ordering": "source"}},
        },
        Path("invented-task.yaml"),
    )


def _write_manifest_fixture(tmp_path: Path) -> tuple[TaskSpec, Path, Path, Path, Path]:
    benchmark_root = tmp_path / "benchmark"
    manifest_path = benchmark_root / "data_manifests" / "invented_task" / "invented-v1.json"
    payload = tmp_path / "data" / "invented" / "rows.jsonl"
    asset = tmp_path / "data" / "invented" / "assets" / "asset.bin"
    code = tmp_path / "scripts" / "prepare_invented.py"
    manifest_path.parent.mkdir(parents=True)
    payload.parent.mkdir(parents=True)
    asset.parent.mkdir(parents=True)
    code.parent.mkdir(parents=True)
    payload.write_text(
        json.dumps({"id": "row-1", "text": "alpha"})
        + "\n"
        + json.dumps({"id": "row-2", "text": "beta"})
        + "\n",
        encoding="utf-8",
    )
    asset.write_bytes(b"invented-asset")
    code.write_text("def prepare():\n    return None\n", encoding="utf-8")
    row_count, row_digest, duplicates = ordered_jsonl_manifest_sha256([payload])
    assets = [
        {
            "path": "data/invented/assets/asset.bin",
            "bytes": asset.stat().st_size,
            "sha256": _sha256(asset),
        }
    ]
    manifest = {
        "schema_version": "1",
        "task_id": "invented_task",
        "dataset_version": "invented-v1",
        "manifest_revision": "2026-07-28.1",
        "created_at": "2026-07-28T00:00:00Z",
        "sources": [
            {
                "source_id": "invented:source",
                "source_revision": "source-r1",
                "config": "default",
                "split": "test",
                "role": "examples",
                "provenance_urls": ["https://example.invalid/source"],
                "license_id": "invented-license",
                "authorship": "invented",
                "attribution": "invented-attribution",
                "redistribution": "allowed",
                "privacy_review": "approved",
                "publication_status": "public",
            }
        ],
        "transformation": {
            "transformation_id": "invented-transform-v1",
            "code_commit": "invented-commit",
            "code_path": "scripts/prepare_invented.py",
            "code_sha256": _sha256(code),
            "parameters": {"ordering": "source"},
            "seed": None,
            "determinism": "deterministic",
            "generator": None,
        },
        "materialization": {
            "row_identity_scheme": "canonical-row-sha256",
            "row_count": row_count,
            "ordered_row_manifest_sha256": row_digest,
            "asset_count": len(assets),
            "asset_manifest_sha256": asset_manifest_sha256(assets),
            "files": [
                {
                    "role": "examples",
                    "path": "data/invented/rows.jsonl",
                    "bytes": payload.stat().st_size,
                    "rows": row_count,
                    "sha256": _sha256(payload),
                }
            ],
            "assets": assets,
        },
        "validation": {
            "required_files_complete": True,
            "missing_assets": 0,
            "orphan_assets": 0,
            "exact_duplicate_rows": duplicates,
            "source_binding_complete": True,
        },
        "review": {
            "state": "approved",
            "reviewed_at": "2026-07-28T00:00:00Z",
            "reviewed_by": "reviewer",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _contract_task(), benchmark_root, manifest_path, payload, asset


def _write_public_loader_manifest(
    tmp_path: Path,
    *,
    task_id: str,
    dataset_version: str,
    transformation_id: str,
    payloads: list[tuple[str, str, list[dict]]],
    assets: list[tuple[str, bytes]] | None = None,
) -> tuple[TaskSpec, Path, Path, dict[str, Path], dict[str, Path]]:
    benchmark_root = tmp_path / "benchmark"
    manifest_path = benchmark_root / "data_manifests" / task_id / f"{dataset_version}.json"
    code = tmp_path / "scripts" / f"prepare_{task_id}.py"
    manifest_path.parent.mkdir(parents=True)
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("def prepare():\n    return None\n", encoding="utf-8")

    payload_paths: dict[str, Path] = {}
    payload_entries = []
    ordered_paths = []
    for role, relative, rows in payloads:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        payload_paths[role] = path
        ordered_paths.append(path)
        payload_entries.append(
            {
                "role": role,
                "path": relative,
                "bytes": path.stat().st_size,
                "rows": len(rows),
                "sha256": _sha256(path),
            }
        )

    asset_paths: dict[str, Path] = {}
    asset_entries = []
    for relative, content in assets or []:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        asset_paths[relative] = path
        asset_entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    row_count, row_digest, duplicates = ordered_jsonl_manifest_sha256(ordered_paths)
    source_id = f"invented:{task_id}"
    transformation_parameters = {"fixture": "authorized"}
    task = TaskSpec.from_dict(
        {
            "id": task_id,
            "display_name": f"Invented {task_id}",
            "task": task_id,
            "description": "Invented public-loader authorization fixture.",
            "dataset_version": dataset_version,
            "evaluation_sources": {
                "disclosure": "complete",
                "sources": [
                    {
                        "source_id": source_id,
                        "usage": "evaluation",
                        "source_revision": "source-r1",
                        "config": "default",
                        "split": "test",
                        "transformation_id": transformation_id,
                    }
                ],
                "public_provenance": {
                    "urls": ["https://example.invalid/source"],
                    "evidence_revision": "evidence-r1",
                    "reviewed_at": "2026-07-28T00:00:00Z",
                    "reviewed_by": "reviewer",
                },
                "review": {"state": "approved"},
            },
            "materialization": {"transformation_parameters": transformation_parameters},
        },
        Path("invented-task.yaml"),
    )
    manifest = {
        "schema_version": "1",
        "task_id": task_id,
        "dataset_version": dataset_version,
        "manifest_revision": "2026-07-28.1",
        "created_at": "2026-07-28T00:00:00Z",
        "sources": [
            {
                "source_id": source_id,
                "source_revision": "source-r1",
                "config": "default",
                "split": "test",
                "role": "examples",
                "provenance_urls": ["https://example.invalid/source"],
                "license_id": "invented-license",
                "authorship": "invented",
                "attribution": "invented-attribution",
                "redistribution": "allowed",
                "privacy_review": "approved",
                "publication_status": "public",
            }
        ],
        "transformation": {
            "transformation_id": transformation_id,
            "code_commit": "invented-commit",
            "code_path": str(code.relative_to(tmp_path)),
            "code_sha256": _sha256(code),
            "parameters": transformation_parameters,
            "seed": None,
            "determinism": "deterministic",
            "generator": None,
        },
        "materialization": {
            "row_identity_scheme": "canonical-row-sha256",
            "row_count": row_count,
            "ordered_row_manifest_sha256": row_digest,
            "asset_count": len(asset_entries),
            "asset_manifest_sha256": asset_manifest_sha256(asset_entries),
            "files": payload_entries,
            "assets": asset_entries,
        },
        "validation": {
            "required_files_complete": True,
            "missing_assets": 0,
            "orphan_assets": 0,
            "exact_duplicate_rows": duplicates,
            "source_binding_complete": True,
        },
        "review": {
            "state": "approved",
            "reviewed_at": "2026-07-28T00:00:00Z",
            "reviewed_by": "reviewer",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return task, benchmark_root, manifest_path, payload_paths, asset_paths


def _public_result(*, data_source_contract: dict | None, include_version: bool = True) -> dict:
    record = {
        "schema_version": "2.0",
        "run": {"id": "invented-public-benchmark", "publish": True, "evidence_tier": "benchmark"},
        "timestamps": {"duration_s": 1.0},
        "model": {"id": "invented-model", "display_name": "Invented", "provider": "invented"},
        "provider_result": {"provider": "invented", "model_name": "invented-model"},
        "task": {
            "id": "needle_in_haystack",
            "display_name": "Needle",
            "dataset_version": "needle-v1",
            "primary_metric": "overall_accuracy",
            "kwargs": {
                "data_mode": (
                    data_source_contract.get("data_mode", "real")
                    if isinstance(data_source_contract, dict)
                    else "real"
                )
            },
        },
        "metrics": {"overall_accuracy": 0.75},
        "details": {},
        "error": None,
    }
    if include_version:
        record["data_source_contract_version"] = DATA_SOURCE_CONTRACT_VERSION
    if data_source_contract is not None:
        record["data_source_contract"] = data_source_contract
    return record


def _validated_snapshot() -> dict:
    snapshot = {
        "schema_version": "1",
        "task_id": "needle_in_haystack",
        "dataset_version": "needle-v1",
        "manifest_revision": "invented.1",
        "manifest_sha256": "a" * 64,
        "data_mode": "real",
        "validation_status": "validated",
        "source_ids": ["invented:source"],
        "transformation_id": "invented-transform-v1",
        "row_count": 2,
        "asset_count": 0,
        "reason_codes": [],
    }
    validate_data_source_snapshot(snapshot)
    return snapshot


def test_materialization_manifest_schema_and_valid_binding(tmp_path) -> None:
    schema = json.loads(Path("schemas/materialization-manifest.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "1"

    task, benchmark_root, _manifest_path, _payload, _asset = _write_manifest_fixture(tmp_path)
    snapshot = validate_materialization_manifest(task, benchmark_root=benchmark_root)

    assert snapshot["validation_status"] == "validated"
    assert snapshot["data_mode"] == "real"
    assert snapshot["source_ids"] == ["invented:source"]
    assert snapshot["row_count"] == 2
    assert snapshot["asset_count"] == 1


def test_materialization_validator_accepts_relative_benchmark_root(tmp_path, monkeypatch) -> None:
    task, _benchmark_root, _manifest_path, _payload, _asset = _write_manifest_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    snapshot = validate_materialization_manifest(task, benchmark_root="benchmark")

    assert snapshot["validation_status"] == "validated"


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda manifest: manifest.update({"dataset_version": "stale-v0"}), "dataset_version_mismatch"),
        (lambda manifest: manifest["sources"][0].update({"source_id": "invented:other"}), "source_mismatch"),
        (lambda manifest: manifest["materialization"].update({"row_count": 3}), "row_count_mismatch"),
        (
            lambda manifest: manifest["transformation"].update({"parameters": {"ordering": "shuffled"}}),
            "parameters_mismatch",
        ),
        (
            lambda manifest: manifest["materialization"]["files"][0].update({"sha256": "0" * 64}),
            "payload_hash_mismatch",
        ),
    ],
)
def test_materialization_manifest_rejects_identity_mismatches(tmp_path, mutation, error_code) -> None:
    task, benchmark_root, manifest_path, _payload, _asset = _write_manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MaterializationContractError) as exc_info:
        validate_materialization_manifest(task, benchmark_root=benchmark_root)
    assert exc_info.value.code == error_code


def test_materialization_manifest_rejects_missing_manifest_and_asset(tmp_path) -> None:
    task, benchmark_root, manifest_path, _payload, asset = _write_manifest_fixture(tmp_path)
    asset.unlink()
    with pytest.raises(MaterializationContractError) as exc_info:
        validate_materialization_manifest(task, benchmark_root=benchmark_root)
    assert exc_info.value.code == "missing_asset"

    manifest_path.unlink()
    with pytest.raises(MaterializationContractError) as exc_info:
        validate_materialization_manifest(task, benchmark_root=benchmark_root)
    assert exc_info.value.code == "missing_manifest"


def test_materialization_manifest_rejects_incomplete_generator_metadata(tmp_path) -> None:
    task, benchmark_root, manifest_path, _payload, _asset = _write_manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["role"] = "generated_caption"
    manifest["transformation"]["determinism"] = "generator_logged"
    manifest["transformation"]["generator"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MaterializationContractError, match="generator") as exc_info:
        validate_materialization_manifest(task, benchmark_root=benchmark_root)
    assert exc_info.value.code == "incomplete_generator_metadata"


def test_explicit_fixture_mode_and_missing_real_manifest_are_distinct() -> None:
    catalog = load_catalog()
    task = catalog.require_task("mrl_stress")

    fixture = prepare_data_source_contract(task, {"data_mode": "fixture"}, catalog.root)
    missing_real = prepare_data_source_contract(task, {"data_mode": "real"}, catalog.root)

    assert fixture is not None and fixture.error is None
    assert fixture.snapshot["validation_status"] == "fixture"
    assert missing_real is not None and "manifest is missing" in str(missing_real.error)
    assert missing_real.snapshot["validation_status"] == "invalid"
    assert missing_real.snapshot["reason_codes"] == ["missing_manifest"]


@pytest.mark.parametrize(
    "task",
    [
        MRLStressTask(data_mode="real"),
        CrossLingualRetrievalTask(data_mode="real"),
        NeedleInHaystackTask(data_mode="real"),
        CrossModalRetrievalTask(data_mode="real"),
    ],
)
def test_real_task_direct_paths_require_validated_binding(task) -> None:
    provider = type(
        "Provider",
        (),
        {
            "name": "invented",
            "model": "invented-model",
            "supports_mrl": True,
            "supported_modalities": {ModalityType.TEXT, ModalityType.IMAGE},
        },
    )()
    result = task.run(provider)

    assert result.error is not None
    assert "requires a validated materialization binding" in result.error


def test_mrl_loader_reads_declared_payload_instead_of_hardcoded_path(tmp_path) -> None:
    task, benchmark_root, _manifest_path, _payloads, _assets = _write_public_loader_manifest(
        tmp_path,
        task_id="mrl_stress",
        dataset_version="invented-mrl-v1",
        transformation_id="invented-mrl-transform-v1",
        payloads=[
            (
                "mrl_stsb_pairs",
                "data/authorized_mrl/rows.jsonl",
                [{"text_a": "authorized-a", "text_b": "authorized-b", "score": 4.5}],
            )
        ],
    )
    decoy = tmp_path / "data" / "mrl_stress" / "stsb_test.jsonl"
    decoy.parent.mkdir(parents=True)
    decoy.write_text(
        json.dumps({"text_a": "decoy-a", "text_b": "decoy-b", "score": 0.0}) + "\n",
        encoding="utf-8",
    )
    prepared = prepare_data_source_contract(task, {"data_mode": "real"}, benchmark_root)
    assert prepared is not None and prepared.error is None and prepared.authorization is not None
    authorization = prepared.authorization

    rows = real_data.load_mrl_continuous_data(authorization)

    assert rows == [("authorized-a", "authorized-b", 4.5)]


def test_public_task_rejects_unmapped_payload_role(tmp_path) -> None:
    task, benchmark_root, _manifest_path, _payloads, _assets = _write_public_loader_manifest(
        tmp_path,
        task_id="mrl_stress",
        dataset_version="invented-mrl-v1",
        transformation_id="invented-mrl-transform-v1",
        payloads=[
            (
                "examples",
                "data/authorized_mrl/rows.jsonl",
                [{"text_a": "authorized-a", "text_b": "authorized-b", "score": 4.5}],
            )
        ],
    )

    with pytest.raises(MaterializationContractError) as exc_info:
        validate_materialization_authorization(task, benchmark_root=benchmark_root)
    assert exc_info.value.code == "payload_role_mismatch"


def test_crosslingual_loader_reads_declared_payload_instead_of_hardcoded_path(tmp_path) -> None:
    task, benchmark_root, _manifest_path, _payloads, _assets = _write_public_loader_manifest(
        tmp_path,
        task_id="crosslingual_retrieval",
        dataset_version="invented-crosslingual-v1",
        transformation_id="invented-crosslingual-transform-v1",
        payloads=[
            (
                "crosslingual_pairs",
                "data/authorized_crosslingual/pairs.jsonl",
                [
                    {
                        "zh": "authorized zh text",
                        "en": "authorized text",
                        "difficulty": "easy",
                        "category": "invented",
                        "hard_negatives_en": [],
                        "hard_negatives_zh": [],
                    }
                ],
            )
        ],
    )
    decoy = tmp_path / "data" / "crosslingual" / "parallel_pairs.jsonl"
    decoy.parent.mkdir(parents=True)
    decoy.write_text(
        json.dumps({"zh": "decoy zh text", "en": "decoy"}) + "\n",
        encoding="utf-8",
    )
    authorization = validate_materialization_authorization(task, benchmark_root=benchmark_root)

    rows = real_data.load_crosslingual_data(authorization)

    assert len(rows) == 1
    assert rows[0].zh == "authorized zh text"
    assert rows[0].en == "authorized text"


def test_needle_loader_reads_declared_payloads_instead_of_hardcoded_paths(tmp_path) -> None:
    task, benchmark_root, _manifest_path, _payloads, _assets = _write_public_loader_manifest(
        tmp_path,
        task_id="needle_in_haystack",
        dataset_version="invented-needle-v1",
        transformation_id="invented-needle-transform-v1",
        payloads=[
            (
                "needle_haystacks",
                "data/authorized_needle/haystacks.jsonl",
                [{"length": 1000, "text": "authorized haystack"}],
            ),
            (
                "needle_facts",
                "data/authorized_needle/needles.jsonl",
                [{"needle": "authorized fact", "query": "authorized query"}],
            ),
        ],
    )
    decoy_root = tmp_path / "data" / "needle_haystack"
    decoy_root.mkdir(parents=True)
    (decoy_root / "haystacks.jsonl").write_text(
        json.dumps({"length": 1000, "text": "decoy haystack"}) + "\n",
        encoding="utf-8",
    )
    (decoy_root / "needles.jsonl").write_text(
        json.dumps({"needle": "decoy fact", "query": "decoy query"}) + "\n",
        encoding="utf-8",
    )
    authorization = validate_materialization_authorization(task, benchmark_root=benchmark_root)

    rows = real_data.load_needle_haystack_real_data(
        authorization,
        haystack_lengths=[1000],
        needle_positions=[0.0],
    )

    assert len(rows) == 1
    assert rows[0]["query"] == "authorized query"
    assert "authorized fact" in rows[0]["document"]
    assert "decoy" not in rows[0]["document"]


def test_cross_modal_loader_uses_only_declared_metadata_and_assets(tmp_path) -> None:
    task, benchmark_root, _manifest_path, _payloads, _assets = _write_public_loader_manifest(
        tmp_path,
        task_id="cross_modal_retrieval",
        dataset_version="invented-cross-modal-v1",
        transformation_id="invented-cross-modal-transform-v1",
        payloads=[
            (
                "cross_modal_metadata",
                "data/authorized_cross_modal/metadata.jsonl",
                [{"image_path": "assets/authorized.bin", "caption": "authorized caption"}],
            )
        ],
        assets=[("data/authorized_cross_modal/assets/authorized.bin", b"authorized-image")],
    )
    decoy_metadata = tmp_path / "data" / "cross_modal" / "metadata.jsonl"
    decoy_asset = tmp_path / "data" / "cross_modal" / "images" / "decoy.bin"
    decoy_metadata.parent.mkdir(parents=True)
    decoy_asset.parent.mkdir(parents=True)
    decoy_asset.write_bytes(b"decoy-image")
    decoy_metadata.write_text(
        json.dumps({"image_path": "images/decoy.bin", "caption": "decoy caption"}) + "\n",
        encoding="utf-8",
    )
    authorization = validate_materialization_authorization(task, benchmark_root=benchmark_root)

    rows = real_data.load_cross_modal_real_data(authorization)

    assert len(rows) == 1
    assert rows[0].text == "authorized caption"
    assert rows[0].image_bytes == b"authorized-image"


def test_cross_modal_rejects_metadata_asset_substitution(tmp_path) -> None:
    task, benchmark_root, manifest_path, payloads, _assets = _write_public_loader_manifest(
        tmp_path,
        task_id="cross_modal_retrieval",
        dataset_version="invented-cross-modal-v1",
        transformation_id="invented-cross-modal-transform-v1",
        payloads=[
            (
                "cross_modal_metadata",
                "data/authorized_cross_modal/metadata.jsonl",
                [{"image_path": "assets/authorized.bin", "caption": "authorized caption"}],
            )
        ],
        assets=[("data/authorized_cross_modal/assets/authorized.bin", b"authorized-image")],
    )
    authorization = validate_materialization_authorization(task, benchmark_root=benchmark_root)
    replacement = tmp_path / "data" / "authorized_cross_modal" / "assets" / "substitute.bin"
    replacement.write_bytes(b"substitute-image")
    metadata = payloads["cross_modal_metadata"]
    metadata.write_text(
        json.dumps({"image_path": "assets/substitute.bin", "caption": "authorized caption"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(MaterializationContractError) as exc_info:
        real_data.load_cross_modal_real_data(authorization)
    assert exc_info.value.code == "payload_hash_mismatch"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row_count, row_digest, duplicates = ordered_jsonl_manifest_sha256([metadata])
    manifest["materialization"]["files"][0].update(
        {
            "bytes": metadata.stat().st_size,
            "rows": row_count,
            "sha256": _sha256(metadata),
        }
    )
    manifest["materialization"]["row_count"] = row_count
    manifest["materialization"]["ordered_row_manifest_sha256"] = row_digest
    manifest["validation"]["exact_duplicate_rows"] = duplicates
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MaterializationContractError) as exc_info:
        validate_materialization_authorization(task, benchmark_root=benchmark_root)
    assert exc_info.value.code == "asset_reference_mismatch"


def test_real_loaders_reject_missing_asset_and_requested_needle_length(tmp_path) -> None:
    cross_modal_task, cross_modal_root, _manifest_path, _payloads, assets = _write_public_loader_manifest(
        tmp_path / "cross-modal",
        task_id="cross_modal_retrieval",
        dataset_version="invented-cross-modal-v1",
        transformation_id="invented-cross-modal-transform-v1",
        payloads=[
            (
                "cross_modal_metadata",
                "data/authorized_cross_modal/metadata.jsonl",
                [{"image_path": "assets/authorized.bin", "caption": "authorized caption"}],
            )
        ],
        assets=[("data/authorized_cross_modal/assets/authorized.bin", b"authorized-image")],
    )
    cross_modal_authorization = validate_materialization_authorization(
        cross_modal_task,
        benchmark_root=cross_modal_root,
    )
    assets["data/authorized_cross_modal/assets/authorized.bin"].unlink()
    with pytest.raises(MaterializationContractError) as exc_info:
        real_data.load_cross_modal_real_data(cross_modal_authorization)
    assert exc_info.value.code == "missing_payload"

    needle_task, needle_root, _manifest_path, _payloads, _assets = _write_public_loader_manifest(
        tmp_path / "needle",
        task_id="needle_in_haystack",
        dataset_version="invented-needle-v1",
        transformation_id="invented-needle-transform-v1",
        payloads=[
            (
                "needle_haystacks",
                "data/authorized_needle/haystacks.jsonl",
                [{"length": 1000, "text": "invented haystack"}],
            ),
            (
                "needle_facts",
                "data/authorized_needle/needles.jsonl",
                [{"needle": "invented fact", "query": "invented query"}],
            ),
        ],
    )
    needle_authorization = validate_materialization_authorization(needle_task, benchmark_root=needle_root)
    with pytest.raises(ValueError, match="missing requested haystack lengths"):
        real_data.load_needle_haystack_real_data(
            needle_authorization,
            haystack_lengths=[1000, 4000],
        )


def test_runner_fails_before_provider_initialization_when_manifest_is_missing(tmp_path, monkeypatch) -> None:
    catalog = load_catalog()
    model = catalog.require_model("openai-text-embedding-3-small")
    task = catalog.require_task("mrl_stress")
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    (benchmark_root / "training_overlap_relationships.yaml").write_text(
        "schema_version: '1'\nrevision: invented.1\nsources: []\nrelationships: []\n",
        encoding="utf-8",
    )
    isolated_catalog = BenchmarkCatalog(root=benchmark_root, models={model.id: model}, tasks={task.id: task})
    runner = BenchmarkRunner(catalog=isolated_catalog, output=tmp_path / "results.jsonl")
    monkeypatch.setattr("mm_embed.benchmark.runner.get_provider", lambda *args, **kwargs: pytest.fail("provider init"))

    records = runner.run_manifest(
        RunManifest(
            id="missing-manifest",
            model_ids=[model.id],
            tasks=[RunTask(id=task.id)],
            evidence_tier="benchmark",
        )
    )

    assert records[0]["data_source_contract"]["validation_status"] == "invalid"
    assert "materialization manifest is missing" in records[0]["error"]


def test_make_result_record_snapshots_binding_without_registry_backfill() -> None:
    catalog = load_catalog()
    model = catalog.require_model("openai-text-embedding-3-small")
    task = catalog.require_task("needle_in_haystack")
    snapshot = _validated_snapshot()
    record = make_result_record(
        run=RunManifest(
            id="snapshot-test",
            model_ids=[model.id],
            tasks=[RunTask(id=task.id)],
            evidence_tier="benchmark",
        ),
        model=model,
        task=task,
        run_task=RunTask(id=task.id, kwargs={"data_mode": "real"}),
        result=EvalResult(
            task_name=task.task,
            provider_name=model.provider,
            model_name=model.id,
            metrics={"overall_accuracy": 0.5},
        ),
        started_at="2026-07-28T00:00:00Z",
        finished_at="2026-07-28T00:00:01Z",
        duration_s=1.0,
        catalog=catalog,
        relationship_registry=load_relationship_registry(),
        data_source_contract=snapshot,
    )
    snapshot["source_ids"].append("invented:mutated")

    assert record["data_source_contract_version"] == "1"
    assert record["data_source_contract"]["source_ids"] == ["invented:source"]
    assert record["data_source_contract"]["manifest_sha256"] == "a" * 64


def test_public_export_rejects_missing_or_fixture_post_contract_binding(tmp_path) -> None:
    missing = _public_result(data_source_contract=None)
    results = tmp_path / "missing.jsonl"
    results.write_text(json.dumps(missing) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing data_source_contract"):
        export_dataset_repo(output_dir=tmp_path / "missing-export", results_path=results)

    catalog = load_catalog()
    fixture = prepare_data_source_contract(
        catalog.require_task("needle_in_haystack"),
        {"data_mode": "fixture"},
        catalog.root,
    )
    assert fixture is not None
    results.write_text(json.dumps(_public_result(data_source_contract=fixture.snapshot)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fixture data"):
        export_dataset_repo(output_dir=tmp_path / "fixture-export", results_path=results)

    invalid = prepare_data_source_contract(
        catalog.require_task("needle_in_haystack"),
        {"data_mode": "real"},
        catalog.root,
    )
    assert invalid is not None and invalid.error is not None
    results.write_text(json.dumps(_public_result(data_source_contract=invalid.snapshot)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a valid real-data materialization binding"):
        export_dataset_repo(output_dir=tmp_path / "invalid-export", results_path=results)


def test_public_export_accepts_snapshot_and_preserves_legacy_score(tmp_path) -> None:
    valid = _public_result(data_source_contract=_validated_snapshot())
    legacy = _public_result(data_source_contract=None, include_version=False)
    legacy["run"]["id"] = "legacy:historical"
    legacy["metrics"]["overall_accuracy"] = 0.25
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(valid) + "\n" + json.dumps(legacy) + "\n", encoding="utf-8")

    output = export_dataset_repo(output_dir=tmp_path / "dataset", results_path=results)
    exported = [json.loads(line) for line in (output / "results" / "latest.jsonl").read_text().splitlines()]

    assert exported[0]["data_source_contract"] == _validated_snapshot()
    assert exported[1]["data_source_contract"] == legacy_unknown_data_source_contract(legacy)
    assert exported[1]["metrics"]["overall_accuracy"] == 0.25
