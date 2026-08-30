from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mm_embed.benchmark import bright_learned_sparse_package as package


def _metrics(value: float) -> dict[str, float]:
    return {metric: value for metric in package.METRIC_NAMES}


def _cells(query_ids: list[str]) -> dict[tuple[str, str], package.Cell]:
    cells = {}
    for model_index, model in enumerate(package.MODELS):
        for track in package.TRACKS:
            per_query = {
                query_id: _metrics((model_index + query_index + 1) / 100.0)
                for query_index, query_id in enumerate(query_ids)
            }
            rankings = {
                query_id: [{"rank": 1 + (model_index + query_index) % 12, "document_id": f"positive-{query_id}"}]
                for query_index, query_id in enumerate(query_ids)
            }
            manifest = {
                "model": {"key": model},
                "data": {"track": track},
                "metrics": package.aggregate_metrics(per_query),
            }
            cells[(model, track)] = package.Cell(
                model=model,
                track=track,
                root=Path("unused"),
                manifest=manifest,
                manifest_sha256=f"{model_index + 1:064x}",
                per_query=per_query,
                rankings=rankings,
            )
    return cells


def test_input_contract_requires_external_identity() -> None:
    path = Path("benchmark/contracts/bright-learned-sparse-main-v0.1-inputs.json")
    expected = package.file_sha256(path)
    contract = package.load_input_contract(path, expected)
    assert len(contract["raw_manifest_sha256"]) == 14
    with pytest.raises(package.PackageError, match="externally supplied identity"):
        package.load_input_contract(path, "0" * 64)


def test_main_pairwise_is_track_local_complete_and_deterministic() -> None:
    cells = _cells([f"q{index:02d}" for index in range(12)])
    summary, first = package.build_leaderboards(cells)
    _, second = package.build_leaderboards(cells)
    assert first == second
    assert set(summary["tracks"]) == set(package.TRACKS)
    assert all(len(first["tracks"][track]) == 21 for track in package.TRACKS)
    assert all(
        result["samples"] == 10_000 and result["seed"] == 20_260_826
        for track in package.TRACKS
        for pair in first["tracks"][track]
        for result in pair["metrics"].values()
    )


def test_case_selector_is_exact_mutually_exclusive_and_opaque(monkeypatch: pytest.MonkeyPatch) -> None:
    query_ids = [f"q{index:02d}" for index in range(14)]
    cells = _cells(query_ids)
    data = SimpleNamespace(
        queries=tuple({"id": query_id} for query_id in query_ids),
        qrels={query_id: {f"positive-{query_id}": 1} for query_id in query_ids},
    )
    monkeypatch.setattr(package, "load_materialized", lambda _root, _track: data)
    tracked, local = package.build_cases(cells, "unused")
    assert tracked["counts"] == {"total": 24, "per_track_disagreement": 6, "per_track_failure": 6}
    assert len({row["opaque_case_id"] for row in tracked["cases"]}) == 24
    for track in package.TRACKS:
        disagreement = {
            row["opaque_case_id"] for row in tracked["cases"]
            if row["track"] == track and row["category"] == "disagreement"
        }
        failure = {
            row["opaque_case_id"] for row in tracked["cases"]
            if row["track"] == track and row["category"] == "failure"
        }
        assert len(disagreement) == len(failure) == 6
        assert disagreement.isdisjoint(failure)
    assert all("query_id" not in row for row in tracked["cases"])
    assert all(row["source_text_included"] is False for row in local["cases"])


def test_slice_contract_keeps_small_bins_point_only(monkeypatch: pytest.MonkeyPatch) -> None:
    query_ids = [f"q{index:02d}" for index in range(12)]
    cells = _cells(query_ids)
    corpus = tuple(
        {"id": f"positive-{query_id}", "content": "short positive document"}
        for query_id in query_ids
    )
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
    predecl = json.loads(Path("benchmark/artifacts/bright-learned-sparse-batch-b-v0.1/predeclaration.json").read_text())
    result = package.build_slices(cells, "unused", predecl)
    for track in package.TRACKS:
        query_length = next(row for row in result["tracks"][track] if row["key"] == "query_length")
        small = next(row for row in query_length["bins"] if row["bin"] == ">128")
        assert small["n"] == 3
        assert small["pairwise"] == []
        assert small["inference"] == "point_estimates_only_no_win_claim"


def test_package_write_requires_external_expected_identity_and_rejects_incomplete(tmp_path: Path) -> None:
    files = {"summary.json": b"{}\n", "report.md": b"report\n"}
    manifest = {"tracked_files": {}}
    expected = package.hashlib.sha256(package.formatted_json_bytes(manifest)).hexdigest()
    with pytest.raises(package.PackageError, match="does not match external expected"):
        package.write_package(
            package_root=tmp_path / "package",
            report_path=tmp_path / "report.md",
            local_mapping_path=tmp_path / "local.json",
            files=files,
            manifest=manifest,
            local_mapping={},
            expected_package_sha256="0" * 64,
        )
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "partial.json").write_text("{}")
    with pytest.raises(package.PackageError, match="incomplete"):
        package.write_package(
            package_root=incomplete,
            report_path=tmp_path / "other-report.md",
            local_mapping_path=tmp_path / "other-local.json",
            files=files,
            manifest=manifest,
            local_mapping={},
            expected_package_sha256=expected,
        )


def test_privacy_scan_rejects_row_level_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = SimpleNamespace(corpus=(), queries=())
    monkeypatch.setattr(package, "load_materialized", lambda _root, _track: empty)
    with pytest.raises(package.PackageError, match="forbidden row-level"):
        package.assert_tracked_privacy({"bad.json": b'{"query_id":"secret"}\n'}, "unused")


def test_aggregation_requires_explicit_cpu_only_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(package.PackageError, match="CUDA_VISIBLE_DEVICES"):
        package.collect_authenticated_cells(
            contract={}, repo_root=".", data_root="unused", batch_a_raw_root="unused",
            batch_b_raw_root="unused", batch_b_replay_root="unused", snapshot_evidence_root="unused",
        )
