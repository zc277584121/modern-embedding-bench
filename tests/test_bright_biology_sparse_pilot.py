from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from scipy import sparse


SCRIPT = Path(__file__).parents[1] / "scripts" / "bright_biology_sparse_pilot.py"
SPEC = importlib.util.spec_from_file_location("bright_biology_sparse_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _chunk(path: Path, start: int, count: int, truncated: int, latency_ms: float, peak_vram: int) -> None:
    matrix_path = path / f"{start:06d}.npz"
    sparse.save_npz(matrix_path, sparse.csr_matrix((count, 3)))
    audit = {
        "start": start,
        "count": count,
        "latency_ms": latency_ms,
        "truncated_count": truncated,
        "peak_vram_bytes": peak_vram,
        "sha256": MODULE.sha256(matrix_path),
    }
    (path / f"{start:06d}.json").write_text(json.dumps(audit), encoding="utf-8")


def test_resume_rebuilds_complete_latency_truncation_and_vram(tmp_path: Path) -> None:
    _chunk(tmp_path, 0, 2, 1, 12.5, 100)
    _chunk(tmp_path, 2, 1, 2, 7.5, 250)
    assert MODULE.load_chunk_audits(tmp_path, 3, 2) == {
        "count": 3,
        "latency_ms": 20.0,
        "truncated_count": 3,
        "peak_vram_bytes": 250,
    }


def test_resume_fails_closed_for_incomplete_audit(tmp_path: Path) -> None:
    _chunk(tmp_path, 0, 2, 1, 12.5, 100)
    audit_path = tmp_path / "000000.json"
    audit = json.loads(audit_path.read_text())
    del audit["peak_vram_bytes"]
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="audit is incomplete"):
        MODULE.load_chunk_audits(tmp_path, 2, 2)


def test_resume_detects_154_versus_155_manifest_style_mismatch(tmp_path: Path) -> None:
    _chunk(tmp_path, 0, 1, 154, 1.0, 10)
    _chunk(tmp_path, 1, 1, 1, 1.0, 20)
    aggregate = MODULE.load_chunk_audits(tmp_path, 2, 1)
    assert aggregate["truncated_count"] == 155
    assert aggregate["truncated_count"] != 154
