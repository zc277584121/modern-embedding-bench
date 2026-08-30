from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import mm_embed.providers.learned_sparse_inventory as inventory_module
from mm_embed.benchmark.bright_learned_sparse_batch_b import (
    BatchBError,
    PUBLICATION,
    assert_research_only,
    formal_protocol,
    refuse_incomplete_resume,
    replay_formal_cell,
    run_formal_cell,
    validate_batch_a_readonly,
    validate_evidence_file,
    validate_hub_metadata,
    validate_predeclaration,
    validate_supersession,
    write_evidence,
)
from mm_embed.providers.learned_sparse_inventory import (
    BATCH_B_DOWNLOAD_CAP_BYTES,
    BATCH_B_INVENTORY,
    BATCH_B_SELECTED_KEYS,
    BoundedSnapshotResolver,
    SnapshotPolicyError,
)
from mm_embed.providers.sentence_transformers_sparse_provider import SentenceTransformersSparseProvider, to_csr
from mm_embed.providers.snapshot_identity import file_sha256

REPO_ROOT = Path(__file__).parents[1]
PREDECLARATION = REPO_ROOT / "benchmark/artifacts/bright-learned-sparse-batch-b-v0.1/predeclaration.json"
SUPERSEDED_PREDECLARATION = (
    REPO_ROOT
    / "benchmark/artifacts/bright-learned-sparse-batch-b-v0.1/predeclaration-v1-superseded.json"
)
PREDECLARATION_SHA256 = "2d30ac69b58e7bbd217550418619bab085e2c8b73e1453ee02e0d237827a32ef"
SUPERSEDED_PREDECLARATION_SHA256 = "7b49d361dd69b48e98de93ec9938e7e09b29cb28de8984ef8281dccd66ad5932"


def _frozen() -> dict:
    return validate_predeclaration(PREDECLARATION, expected_sha256=PREDECLARATION_SHA256)


def test_tracked_predeclaration_is_externally_bound_and_batch_a_is_read_only() -> None:
    frozen = _frozen()
    validate_batch_a_readonly(frozen, REPO_ROOT)
    assert frozen["bindings"]["batch_a"]["read_only"] is True
    assert [row["key"] for row in frozen["models"]] == list(BATCH_B_SELECTED_KEYS)
    assert frozen["formal_scores_observed"] is False
    assert frozen["supersedes_sha256"] == SUPERSEDED_PREDECLARATION_SHA256
    assert validate_supersession(frozen, SUPERSEDED_PREDECLARATION)["created_at"] == "2026-08-29T22:55:00Z"
    with pytest.raises(BatchBError, match="unique active revision"):
        validate_predeclaration(PREDECLARATION, expected_sha256="0" * 64)


def test_predeclaration_payload_cannot_be_synchronously_resigned(tmp_path: Path) -> None:
    value = json.loads(PREDECLARATION.read_text())
    value["resources"]["requested_batch_size"] = 4
    attacked = tmp_path / "predeclaration.json"
    attacked.write_text(json.dumps(value) + "\n")
    attacked.with_suffix(".sha256").write_text(file_sha256(attacked) + "\n")
    with pytest.raises(BatchBError, match="externally supplied identity"):
        validate_predeclaration(attacked, expected_sha256=PREDECLARATION_SHA256)


def test_supersession_changes_only_audit_lineage_time_schema_and_identity() -> None:
    active = _frozen()
    superseded = validate_supersession(active, SUPERSEDED_PREDECLARATION)
    excluded = {
        "created_at", "formal_scores_observed", "identity", "schema_version",
        "superseded_readiness_evidence", "supersedes_sha256", "supersession_reason",
    }
    active_business = {key: value for key, value in active.items() if key not in excluded}
    old_business = {key: value for key, value in superseded.items() if key not in excluded}
    assert active_business == old_business
    assert active["schema_version"] == "bright-learned-sparse-batch-b-predeclaration-v2"
    assert active["created_at"] == "2026-08-29T23:02:00Z"
    assert active["supersession_reason"] == "incorrect embedded created_at before formal scoring"


@pytest.mark.parametrize(
    ("key", "revision", "dimension", "query_route", "document_route", "bytes_"),
    [
        ("splade-tiny", "7391972eac4411e33efff5fad27b886ec97895c0", 30522, "neural", "neural", 18_618_173),
        ("opensearch-doc-v2-distill", "269e6638b2c4f648996691f6d751495285d8f330", 30522, "neural", "neural", 268_899_492),
        ("opensearch-multilingual", "1e0f096c2b51c234f1d20725c793e1b5b6d556db", 105879, "static_lookup", "document_expansion", 683_815_451),
    ],
)
def test_batch_b_exact_model_contracts(key, revision, dimension, query_route, document_route, bytes_) -> None:
    spec = BATCH_B_INVENTORY[key]
    assert spec.revision == revision and len(spec.revision) == 40
    assert (spec.dimensions, spec.max_length) == (dimension, 512)
    assert (spec.query_route, spec.document_route) == (query_route, document_route)
    assert spec.estimated_snapshot_bytes == bytes_
    assert not spec.trust_remote_code
    assert any(name.endswith(".safetensors") for name in spec.allowlist)
    assert all(not name.endswith((".py", ".bin", ".pt", ".pth", ".pkl", ".pickle")) for name in spec.allowlist)


def test_metadata_requires_exact_revision_license_gating_and_file_bytes() -> None:
    spec = BATCH_B_INVENTORY["splade-tiny"]
    sizes = [1] * len(spec.allowlist)
    sizes[-1] += spec.estimated_snapshot_bytes - len(sizes)
    siblings = [SimpleNamespace(rfilename=name, size=size) for name, size in zip(spec.allowlist, sizes, strict=True)]
    info = SimpleNamespace(
        sha=spec.revision, private=False, gated=False, card_data={"license": "mit"}, siblings=siblings
    )
    api = SimpleNamespace(model_info=lambda *args, **kwargs: info)
    evidence = validate_hub_metadata(spec.key, _frozen(), api=api, predeclaration_sha256=PREDECLARATION_SHA256)
    assert evidence["declared_bytes"] == spec.estimated_snapshot_bytes
    info.sha = "0" * 40
    with pytest.raises(BatchBError, match="revision drifted"):
        validate_hub_metadata(spec.key, _frozen(), api=api, predeclaration_sha256=PREDECLARATION_SHA256)


def test_batch_plan_caps_are_aggregate_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_snapshot_download(**kwargs):
        spec = next(row for row in BATCH_B_INVENTORY.values() if row.repo_id == kwargs["repo_id"])
        total = int(spec.estimated_snapshot_bytes)
        rows = []
        for index, name in enumerate(spec.allowlist):
            size = total - len(spec.allowlist) + 1 if index == 0 else 1
            rows.append(SimpleNamespace(filename=name, file_size=size, commit_hash=spec.revision, is_cached=False, will_download=True))
        return rows

    monkeypatch.setattr(inventory_module, "snapshot_download", fake_snapshot_download)
    plan = BoundedSnapshotResolver().plan_batch(
        BATCH_B_SELECTED_KEYS,
        inventory=BATCH_B_INVENTORY,
        batch_cap_bytes=BATCH_B_DOWNLOAD_CAP_BYTES,
    )
    assert plan["declared_bytes"] == 971_333_116 < BATCH_B_DOWNLOAD_CAP_BYTES
    with pytest.raises(SnapshotPolicyError, match="hard caps"):
        BoundedSnapshotResolver().plan_batch(
            BATCH_B_SELECTED_KEYS, inventory=BATCH_B_INVENTORY, batch_cap_bytes=plan["declared_bytes"] - 1
        )


@pytest.mark.parametrize("name", ["loader.py", "model.bin", "state.pkl"])
def test_batch_b_allowlist_rejects_behavior_code_and_pickle(name: str) -> None:
    spec = replace(BATCH_B_INVENTORY["splade-tiny"], allowlist=("model.safetensors", name))
    with pytest.raises(SnapshotPolicyError, match="Python code|pickle-compatible"):
        BoundedSnapshotResolver._validate_spec(spec)


def test_snapshot_rejects_auto_map(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "x.Y"}}))
    (tmp_path / "model.safetensors").write_bytes(b"safe")
    spec = replace(
        BATCH_B_INVENTORY["splade-tiny"],
        allowlist=("config.json", "model.safetensors"),
        expected_identity=(),
        download_cap_bytes=100,
    )
    with pytest.raises(SnapshotPolicyError, match="auto_map"):
        BoundedSnapshotResolver()._verify_snapshot(spec, tmp_path)


class _Tokenizer:
    def __call__(self, texts, **kwargs):
        return {"length": [4 for _ in texts]}


class _Encoder:
    dimensions = 30522

    def __init__(self, *_args, device="cpu", **_kwargs):
        self.device = device
        self.tokenizer = _Tokenizer()
        self.max_seq_length = None

    def _encode(self, texts, **kwargs):
        import torch

        indices = torch.tensor([[row for row in range(len(texts))], [row for row in range(len(texts))]])
        return torch.sparse_coo_tensor(indices, torch.ones(len(texts)), (len(texts), self.dimensions))

    encode_query = _encode
    encode_document = _encode


def test_provider_accepts_only_external_snapshot_identity_and_frozen_shape(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    identity = {"config.json": file_sha256(tmp_path / "config.json")}
    spec = replace(BATCH_B_INVENTORY["splade-tiny"], allowlist=("config.json",))
    provider = SentenceTransformersSparseProvider(
        model_key=spec.key,
        model_spec=spec,
        expected_snapshot_identity=identity,
        snapshot_path=str(tmp_path),
        device="cpu",
        batch_size=8,
        encoder_factory=_Encoder,
    )
    result = provider.encode_sparse_documents(["a", "b"], item_ids=["d1", "d2"])
    assert result.embeddings.values.shape == (2, 30522)
    assert provider.query_route.value == "neural" and provider.max_length == 512
    with pytest.raises(ValueError, match="max_length is fixed"):
        SentenceTransformersSparseProvider(
            model_key=spec.key, model_spec=spec, expected_snapshot_identity=identity,
            snapshot_path=str(tmp_path), max_length=256, encoder_factory=_Encoder,
        )


def test_cuda_oom_fallback_is_exact_8_4_2_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    (tmp_path / "config.json").write_text("{}")
    identity = {"config.json": file_sha256(tmp_path / "config.json")}
    spec = replace(BATCH_B_INVENTORY["splade-tiny"], allowlist=("config.json",))

    class OOMEncoder(_Encoder):
        calls = []

        def _encode(self, texts, **kwargs):
            self.calls.append(kwargs["batch_size"])
            if kwargs["batch_size"] > 1:
                raise torch.cuda.OutOfMemoryError("bounded test")
            return super()._encode(texts, **kwargs)

        def encode_query(self, texts, **kwargs):
            return self._encode(texts, **kwargs)

        def encode_document(self, texts, **kwargs):
            return self._encode(texts, **kwargs)

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args, **_kwargs: 100)
    provider = SentenceTransformersSparseProvider(
        model_key=spec.key, model_spec=spec, expected_snapshot_identity=identity,
        snapshot_path=str(tmp_path), device="cuda:0", batch_size=8, encoder_factory=OOMEncoder,
    )
    result = provider.encode_sparse_query("query", item_id="q1")
    assert result.metadata_dict()["batch_size_attempts"] == [8, 4, 2, 1]
    assert result.metadata_dict()["batch_size_used"] == 1


def test_csr_is_canonical_finite_nonnegative_and_dense_is_forbidden() -> None:
    coo = sparse.coo_matrix(([1.0, 2.0], ([0, 0], [1, 1])), shape=(1, 3))
    matrix = to_csr(coo, 1, 3)
    assert matrix.has_canonical_format and matrix.nnz == 1 and matrix.data[0] == 3
    with pytest.raises(TypeError, match="dense materialization"):
        to_csr(np.ones((1, 3), dtype=np.float32), 1, 3)
    with pytest.raises(ValueError, match="non-finite"):
        to_csr(sparse.csr_matrix([[np.nan]]), 1, 1)
    with pytest.raises(ValueError, match="negative"):
        to_csr(sparse.csr_matrix([[-1.0]]), 1, 1)


def test_incomplete_resume_and_public_export_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    (output / "partial.npz").write_bytes(b"partial")
    with pytest.raises(BatchBError, match="not externally authenticated"):
        refuse_incomplete_resume(output)
    assert_research_only({"publication": PUBLICATION})
    with pytest.raises(BatchBError, match="not research-only"):
        assert_research_only({"publication": {**PUBLICATION, "public_export_allowed": True}})


def test_research_evidence_requires_external_identity_and_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    identity = write_evidence(path, {"publication": PUBLICATION, "value": 1})
    assert validate_evidence_file(path, expected_sha256=identity, label="test evidence")["value"] == 1
    path.with_suffix(".json.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(BatchBError, match="sidecar mismatch"):
        validate_evidence_file(path, expected_sha256=identity, label="test evidence")


def test_formal_protocol_is_fixed_and_track_independent() -> None:
    protocol = formal_protocol()
    assert protocol["requested_batch_size"] == 8
    assert protocol["batch_fallback"] == [8, 4, 2, 1]
    assert protocol["document_chunk_size"] == 256
    assert protocol["search"] == {
        "backend": "scipy_csr_exact",
        "similarity": "inner_product",
        "top_k": 100,
        "tie_break": "document_id_ascending",
    }
    assert protocol["uncertainty"] == {
        "method": "query_level_bootstrap", "samples": 10_000, "seed": 20_260_826,
    }
    assert protocol["track_aggregation"] == "independent_macro_average_no_micro_average"


def test_formal_run_rejects_incomplete_resume_before_model_loading(tmp_path: Path) -> None:
    output = tmp_path / "cell"
    output.mkdir()
    (output / "documents.npz").write_bytes(b"partial")
    with pytest.raises(BatchBError, match="cannot resume"):
        run_formal_cell(
            model_key="splade-tiny", track="economics", predeclaration=_frozen(),
            predeclaration_sha256=PREDECLARATION_SHA256, snapshot_evidence={},
            snapshot_evidence_sha256="1" * 64, data_root=tmp_path, output_root=output,
        )


def test_formal_replay_requires_explicit_cpu_only_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(BatchBError, match="CPU-only"):
        replay_formal_cell(
            result_root=tmp_path, data_root=tmp_path, predeclaration=_frozen(),
            predeclaration_sha256=PREDECLARATION_SHA256, snapshot_evidence={},
            snapshot_evidence_sha256="1" * 64, expected_manifest_sha256="2" * 64,
        )
