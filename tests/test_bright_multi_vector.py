from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mm_embed.benchmark import bright_multi_vector as benchmark
from mm_embed.benchmark import bright_multi_vector_package as package
from mm_embed.benchmark.retrieval_v01 import METRIC_NAMES, aggregate_metrics
from mm_embed.providers.multi_vector_base import MultiVectorRepresentation, MultiVectorRole, MultiVectorRoute
from mm_embed.providers.real_multi_vector import (
    INVENTORY,
    SELECTED_KEYS,
    SentenceTransformerMultiVectorProvider,
)


def _metric(value: float) -> dict[str, float]:
    return {name: value for name in METRIC_NAMES}


def _cells(query_ids: list[str]) -> dict[tuple[str, str], package.Cell]:
    cells = {}
    for model_index, model in enumerate(SELECTED_KEYS):
        for track in benchmark.TRACKS:
            per_query = {
                query_id: _metric((model_index + query_index + 1) / 100.0)
                for query_index, query_id in enumerate(query_ids)
            }
            rankings = {
                query_id: [
                    {"document_id": f"positive-{query_id}", "score": 1.0},
                    {"document_id": "negative", "score": 0.0},
                ]
                for query_id in query_ids
            }
            value = {
                "quality": {"metrics": aggregate_metrics(per_query), "uncertainty": {}},
                "execution": {},
                "representation": {},
            }
            cells[(model, track)] = package.Cell(
                model, track, Path("unused"), f"{model_index + 1:064x}", value, per_query, rankings
            )
    return cells


def _fake_data() -> SimpleNamespace:
    return SimpleNamespace(
        corpus=({"id": "d0", "content": "first"}, {"id": "d1", "content": "second"}),
        queries=({"id": "q0", "text": "query"},),
        qrels={"q0": {"d0": 1}},
    )


def _write_fake_encoding(root: Path, *, gate_sha: str = "a" * 64) -> dict[str, object]:
    model = "colbert-v2"
    tracks = {}
    for track in benchmark.TRACKS:
        track_root = root / model / track
        track_root.mkdir(parents=True)
        documents = benchmark._save_ragged(
            track_root,
            "documents",
            [np.ones((2, 128), dtype=np.float32), np.full((1, 128), 2, dtype=np.float32)],
        )
        queries = benchmark._save_ragged(track_root, "queries", [np.ones((1, 128), dtype=np.float32)])
        files = {
            "document_indices": benchmark._save_array(
                track_root / "window_document_indices.npy", np.array([0, 1], dtype=np.int32)
            ),
            "window_ordinals": benchmark._save_array(
                track_root / "window_ordinals.npy", np.array([0, 0], dtype=np.int32)
            ),
            "document_ids": benchmark._save_json(track_root / "document_ids.json", ["d0", "d1"]),
            "query_ids": benchmark._save_json(track_root / "query_ids.json", ["q0"]),
        }
        representation = {
            "schema_version": "bright-real-multi-vector-representation-v1",
            "story_id": "S-20260814-012",
            "predeclaration_sha256": benchmark.ACTIVE_PREDECLARATION_SHA256,
            "gate_summary_sha256": gate_sha,
            "model": {
                "key": model,
                "repo_id": INVENTORY[model].repo_id,
                "revision": INVENTORY[model].revision,
                "dimension": 128,
            },
            "data": {"track": track, "documents": 2, "queries": 1, "qrels": 1},
            "windowing": {"windows": 2},
            "representations": {"documents": documents, "queries": queries, "files": files},
            "encoding": {},
            "publication": benchmark.PUBLICATION,
            "finalized": True,
        }
        identity = benchmark.write_json_with_hash(track_root / "representation.json", representation)
        tracks[track] = {"representation_sha256": identity, "path": str(track_root)}
    manifest = {
        "schema_version": "bright-real-multi-vector-model-encoding-v1",
        "story_id": "S-20260814-012",
        "predeclaration_sha256": benchmark.ACTIVE_PREDECLARATION_SHA256,
        "gate_summary_sha256": gate_sha,
        "model_key": model,
        "tracks": tracks,
        "model_wall_seconds": 1.0,
        "publication": benchmark.PUBLICATION,
        "finalized": True,
    }
    benchmark.write_json_with_hash(root / model / "encoding-manifest.json", manifest)
    return benchmark.validate_model_encoding(
        root / model,
        model_key=model,
        predeclaration_sha256=benchmark.ACTIVE_PREDECLARATION_SHA256,
        gate_summary_sha256=gate_sha,
    )


def test_frozen_inventory_uses_three_distinct_immutable_checkpoints() -> None:
    selected = [INVENTORY[key] for key in SELECTED_KEYS]
    assert len(selected) == 3
    assert len({row.repo_id for row in selected}) == 3
    assert len({row.revision for row in selected}) == 3
    assert all(len(row.revision) == 40 and row.status == "selected" for row in selected)
    assert all(row.trust_remote_code is False and row.gated is False and row.private is False for row in selected)
    assert INVENTORY["jina-colbert-v2"].status == "excluded"
    assert INVENTORY["lighton-colbert-v2-wrapper"].status == "excluded"


def test_passage_windowing_is_deterministic_and_tail_aligned() -> None:
    short = "alpha beta gamma"
    assert benchmark.passage_windows(short) == [short]
    long = " ".join(f"word{index}" for index in range(260))
    first = benchmark.passage_windows(long)
    assert first == benchmark.passage_windows(long)
    assert len(first) == 3
    assert first[0].startswith("word0 ")
    assert first[-1].startswith("word132 ") and first[-1].endswith(" word259")


def test_adapter_pack_has_boolean_mask_and_zero_padding() -> None:
    provider = object.__new__(SentenceTransformerMultiVectorProvider)
    provider.spec = SimpleNamespace(dimensions=2)
    provider.representation = MultiVectorRepresentation("fixture", 2)
    provider.query_route = MultiVectorRoute.QUERY
    provider.document_route = MultiVectorRoute.DOCUMENT
    provider.name = "fixture"
    provider.model = "fixture"
    provider.revision = "0" * 40
    result = provider._pack(
        [np.ones((2, 2), dtype=np.float32), np.full((1, 2), 2, dtype=np.float32)],
        item_ids=("a", "b"),
        passage_ids=("a", "b"),
        document_ids=("a", "b"),
        role=MultiVectorRole.DOCUMENT,
        latency_ms=1.0,
        peak_vram_bytes=0,
    )
    assert result.embeddings.mask.dtype == np.bool_
    assert result.embeddings.mask.tolist() == [[True, True], [True, False]]
    assert np.all(result.embeddings.values[1, 1] == 0)


def test_frozen_artifacts_validate_shared_schema() -> None:
    inventory = Path("benchmark/artifacts/bright-multi-vector-v0.1/inventory.json")
    active = Path("benchmark/artifacts/bright-multi-vector-v0.2")
    for path in (inventory, active / "chronology.json", active / "predeclaration.json"):
        package.validate_schema(json.loads(path.read_text(encoding="utf-8")))


def test_pairwise_and_small_slice_contract_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    query_ids = [f"q{index:02d}" for index in range(12)]
    cells = _cells(query_ids)
    corpus = tuple({"id": f"positive-{query_id}", "content": "short positive document"} for query_id in query_ids)
    queries = tuple(
        {"id": query_id, "text": "short query" if index < 9 else " ".join(["long"] * 140)}
        for index, query_id in enumerate(query_ids)
    )
    data = SimpleNamespace(
        corpus=corpus,
        queries=queries,
        qrels={query_id: {f"positive-{query_id}": 1} for query_id in query_ids},
    )
    monkeypatch.setattr(package, "load_materialized", lambda _root, _track: data)
    first_summary, first_pairwise = package.build_quality(cells)
    assert (first_summary, first_pairwise) == package.build_quality(cells)
    assert all(len(first_pairwise["tracks"][track]) == 3 for track in benchmark.TRACKS)
    predecl = json.loads(Path("benchmark/artifacts/bright-multi-vector-v0.2/predeclaration.json").read_text())
    slices = package.build_slices(cells, "unused", predecl)
    for track in benchmark.TRACKS:
        dimension = next(row for row in slices["tracks"][track] if row["key"] == "query_length")
        small = next(row for row in dimension["bins"] if row["bin"] == ">128")
        assert small["n"] == 3 and small["pairwise"] == []


def test_cases_are_bounded_opaque_and_privacy_scan_rejects_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    query_ids = [f"q{index:02d}" for index in range(12)]
    cells = _cells(query_ids)
    data = SimpleNamespace(
        corpus=tuple({"id": f"positive-{query_id}", "content": "restricted document body"} for query_id in query_ids),
        queries=tuple({"id": query_id, "text": "restricted query body"} for query_id in query_ids),
        qrels={query_id: {f"positive-{query_id}": 1} for query_id in query_ids},
    )
    monkeypatch.setattr(package, "load_materialized", lambda _root, _track: data)
    tracked, local = package.build_cases(cells, "unused")
    assert tracked["counts"] == {"total": 24, "per_track_disagreement": 6, "per_track_failure": 6}
    assert all("query_id" not in row for row in tracked["cases"])
    assert all(row["source_text_included"] is False for row in local["cases"])
    with pytest.raises(package.MultiVectorPackageError, match="forbidden row-level"):
        package.assert_tracked_privacy({"bad.json": b'{"query_id":"secret"}\n'}, "unused")


def test_representation_and_encoding_tamper_fail_closed(tmp_path: Path) -> None:
    evidence = _write_fake_encoding(tmp_path)
    assert set(evidence["representations"]) == set(benchmark.TRACKS)
    economics = evidence["representations"]["economics"]
    storage = economics["storage"]
    assert storage["complete_replayable_non_sidecar_bytes"] == (
        storage["token_array_bytes"] + storage["mapping_identity_bytes"] + storage["representation_manifest_bytes"]
    )
    assert storage["complete_replayable_with_sidecars_bytes"] == (
        storage["complete_replayable_non_sidecar_bytes"] + storage["sidecar_bytes"]
    )
    assert len(economics["files"]) == 10
    assert all("sidecar" in entry for entry in economics["files"].values())
    values = tmp_path / "colbert-v2/economics/documents.values.npy"
    original = values.read_bytes()
    tampered = bytearray(original)
    tampered[-1] ^= 1
    values.write_bytes(tampered)
    with pytest.raises(benchmark.BrightMultiVectorError, match="file identity drifted"):
        benchmark.validate_model_encoding(
            tmp_path / "colbert-v2",
            model_key="colbert-v2",
            predeclaration_sha256=benchmark.ACTIVE_PREDECLARATION_SHA256,
            gate_summary_sha256="a" * 64,
        )
    values.write_bytes(original)
    encoding_manifest = tmp_path / "colbert-v2/encoding-manifest.json"
    encoding_manifest.write_bytes(encoding_manifest.read_bytes().replace(b'"finalized": true', b'"finalized":false'))
    with pytest.raises(benchmark.BrightMultiVectorError, match="Encoding manifest sidecar drifted"):
        benchmark.validate_model_encoding(
            tmp_path / "colbert-v2",
            model_key="colbert-v2",
            predeclaration_sha256=benchmark.ACTIVE_PREDECLARATION_SHA256,
            gate_summary_sha256="a" * 64,
        )


def test_representation_component_sidecar_tamper_fails_closed(tmp_path: Path) -> None:
    _write_fake_encoding(tmp_path)
    sidecar = tmp_path / "colbert-v2/psychology/query_ids.json.sha256"
    sidecar.write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(benchmark.BrightMultiVectorError, match="sidecar identity drifted"):
        benchmark.validate_model_encoding(
            tmp_path / "colbert-v2",
            model_key="colbert-v2",
            predeclaration_sha256=benchmark.ACTIVE_PREDECLARATION_SHA256,
            gate_summary_sha256="a" * 64,
        )


def test_candidate_paths_are_ordered_and_exclude_private_or_protected_files() -> None:
    paths = package.candidate_paths()
    assert paths == tuple(sorted(paths))
    assert len(paths) == len(set(paths))
    assert all(not path.startswith(("results/", "cache/", ".cache/")) for path in paths)
    assert all("milvus-sindi-system-v0.1/validator-candidate" not in path for path in paths)


def test_cell_and_package_tamper_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    encoding = _write_fake_encoding(tmp_path / "representations")
    monkeypatch.setattr(package, "load_materialized", lambda _root, _track: _fake_data())
    cell_root = tmp_path / "cells"
    cell_root.mkdir()
    per_query = package.formatted_json_bytes([{"query_id": "q0", "metrics": _metric(1.0)}])
    rankings = package.formatted_json_bytes([{"query_id": "q0", "hits": [{"document_id": "d0", "score": 1.0}]}])
    per_entry = benchmark._save_json(cell_root / "colbert-v2-economics.per-query.json", json.loads(per_query))
    ranking_entry = benchmark._save_json(cell_root / "colbert-v2-economics.rankings.json", json.loads(rankings))
    cell = {
        "predeclaration_sha256": benchmark.ACTIVE_PREDECLARATION_SHA256,
        "model": {"key": "colbert-v2"},
        "track": {"track": "economics"},
        "protocol": {"exact": True},
        "representation": {
            "encoding_manifest_sha256": encoding["manifest_sha256"],
            "manifest_sha256": encoding["representations"]["economics"]["manifest_sha256"],
            "storage": encoding["representations"]["economics"]["storage"],
        },
        "quality": {"metrics": _metric(1.0)},
        "execution": {"model_loaded": False},
        "restricted_recompute": {"per_query": per_entry, "rankings": ranking_entry},
        "publication": benchmark.PUBLICATION,
        "finalized": True,
    }
    cell_path = cell_root / "colbert-v2-economics.json"
    benchmark.write_json_with_hash(cell_path, cell)
    package.validate_formal_cell_artifact(
        cell_path,
        model="colbert-v2",
        track="economics",
        encoding=encoding,
        data_root="unused",
    )
    cell_path.write_bytes(cell_path.read_bytes().replace(b'"finalized": true', b'"finalized":false'))
    with pytest.raises(package.MultiVectorPackageError, match="cell identity drifted"):
        package.validate_formal_cell_artifact(
            cell_path,
            model="colbert-v2",
            track="economics",
            encoding=encoding,
            data_root="unused",
        )

    package_root = tmp_path / "package"
    report = tmp_path / "report.md"
    local = tmp_path / "local.json"
    files = {"summary.json": b"{}\n", "report.md": b"report\n"}
    manifest = {"implementation_files": {}}
    package.write_or_check_package(
        package_root=package_root,
        report_path=report,
        local_mapping_path=local,
        files=files,
        manifest=manifest,
        local_mapping={},
        check_only=False,
    )
    (package_root / "summary.json").write_bytes(b'{"tampered":true}\n')
    with pytest.raises(package.MultiVectorPackageError, match="Tracked package drifted"):
        package.write_or_check_package(
            package_root=package_root,
            report_path=report,
            local_mapping_path=local,
            files=files,
            manifest=manifest,
            local_mapping={},
            check_only=True,
        )


def test_resume_rejects_tampered_finalized_representation_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    representation_root = tmp_path / "representations"
    gate_summary = tmp_path / "gate-summary.json"
    gate_summary.write_bytes(b"gate")
    _write_fake_encoding(representation_root, gate_sha=benchmark.file_sha256(gate_summary))
    values = representation_root / "colbert-v2/economics/documents.values.npy"
    tampered = bytearray(values.read_bytes())
    tampered[-1] ^= 1
    values.write_bytes(tampered)
    monkeypatch.setattr(benchmark, "validate_predeclaration", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(benchmark, "validate_gate_summary", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(benchmark, "validate_canonical_data", lambda *_args, **_kwargs: {"tracks": {}})
    monkeypatch.setattr(benchmark, "load_materialized", lambda _root, _track: _fake_data())
    monkeypatch.setattr(
        benchmark,
        "resolve_snapshot",
        lambda *_args, **_kwargs: pytest.fail("tampered resume must fail before model resolution"),
    )
    with pytest.raises(benchmark.BrightMultiVectorError, match="file identity drifted"):
        benchmark.encode_formal_model(
            "colbert-v2",
            predeclaration_path="unused",
            predeclaration_sha256=benchmark.ACTIVE_PREDECLARATION_SHA256,
            gate_summary_path=gate_summary,
            data_root="unused",
            output_root=representation_root,
        )
