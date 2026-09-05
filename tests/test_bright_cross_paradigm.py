from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_embed.benchmark import bright_cross_paradigm as cross
from mm_embed.benchmark.bright_cross_paradigm_run import FORMAL_METHODS, validate_formal_cell


def test_inventory_and_matrix_are_complete() -> None:
    inventory = cross.baseline_inventory()
    assert inventory["counts"] == {"lexical": 2, "dense": 2, "learned_sparse": 7, "multi_vector": 3}
    assert len(inventory["methods"]) == 14
    assert len({method["key"] for method in inventory["methods"]}) == 14
    cells = cross._matrix(inventory)
    assert len(cells) == 28
    assert sum(cell["quality_action"] == "certify_read_only_reuse" for cell in cells) == 24
    assert sum(cell["quality_action"] == "run_after_gate" for cell in cells) == 4


def test_tfidf_gate_is_bounded_and_passes() -> None:
    gate = cross.tfidf_gate()
    assert gate["queries"] == 1
    assert gate["documents"] == 4
    assert gate["positive_rank"] == 1
    assert gate["unique_rank_one"] is True
    assert gate["passed"] is True


def test_write_once_and_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    digest = cross.write_once(path, {"b": 2, "a": 1})
    assert cross.verify_sidecar(path) == digest
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    with pytest.raises(FileExistsError):
        cross.write_once(path, {"a": 1})


def test_foundations_and_protected_files_are_unchanged() -> None:
    checks = cross.assert_foundations()
    assert set(checks["accepted_commits"]) == {"s005", "s008", "s011", "s012"}
    assert checks["protected_files"] == cross.PROTECTED_FILES


def test_all_representation_or_score_backed_formal_cells_replay() -> None:
    cells = {
        f"{method}:{track}": validate_formal_cell(method, track) for method in FORMAL_METHODS for track in cross.TRACKS
    }
    assert len(cells) == 8
    for method in ("bm25-unicode", "bge-m3-dense"):
        for track in cross.TRACKS:
            anchor = cells[f"{method}:{track}"]["accepted_anchor_comparison"]
            assert anchor["role"] == "accepted_S005_contextual_anchor_only"
            assert anchor["top100_overlap_min"] == 100
