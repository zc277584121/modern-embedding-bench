from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import mm_embed.benchmark.bright_learned_sparse_batch_a as batch_module
from mm_embed.benchmark.bright_learned_sparse_batch_a import (
    BatchAError,
    encode_document_chunks,
    tracked_aggregate,
    validate_raw_result,
)
from mm_embed.benchmark.registry import load_catalog
from mm_embed.benchmark.retrieval_v01 import TrackData
from mm_embed.providers import learned_sparse_inventory as inventory_module
from mm_embed.providers.learned_sparse_inventory import (
    BoundedSnapshotResolver,
    INVENTORY,
    SnapshotPolicyError,
    inventory_document,
)
from mm_embed.providers.sparse_base import (
    SparseEmbeddingBatch,
    SparseEmbeddingResult,
    SparseEmbeddingRole,
    SparseEncodingRoute,
    SparseRepresentation,
)
from mm_embed.providers.sentence_transformers_sparse_provider import SentenceTransformersSparseProvider, to_csr
from mm_embed.providers.snapshot_identity import file_sha256


def _dry_rows(spec, *, bytes_each: int = 10):
    return [
        SimpleNamespace(
            filename=name, file_size=bytes_each, commit_hash=spec.revision,
            is_cached=False, will_download=True, local_path=f"/cache/{name}",
        )
        for name in spec.allowlist
    ]


def test_resolver_budgets_full_allowlist_and_rejects_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = INVENTORY["granite-30m-sparse"]
    monkeypatch.setattr(inventory_module, "snapshot_download", lambda **kwargs: _dry_rows(spec))
    plan = BoundedSnapshotResolver().plan(spec)
    assert plan["declared_bytes"] == len(spec.allowlist) * 10
    assert plan["download_bytes"] == plan["declared_bytes"]
    assert plan["cache"]["hub_cache_dir"].endswith("/hub")
    with pytest.raises(SnapshotPolicyError, match="40-character"):
        BoundedSnapshotResolver._validate_spec(replace(spec, revision="main"))


def test_inventory_revision_evidence_binds_real_opensearch_v2_distill_pair() -> None:
    spec = INVENTORY["opensearch-doc-v2-distill"]
    assert spec.repo_id == "opensearch-project/opensearch-neural-sparse-encoding-v2-distill"
    assert spec.revision == "269e6638b2c4f648996691f6d751495285d8f330"
    row = next(item for item in inventory_document()["models"] if item["key"] == spec.key)
    assert row["revision_evidence"] == {
        "kind": "huggingface_immutable_tree",
        "url": f"https://huggingface.co/{spec.repo_id}/tree/{spec.revision}",
    }


@pytest.mark.parametrize("name", ["loader.py", "pytorch_model.bin", "state.pt"])
def test_resolver_rejects_code_and_pickle_allowlist(name: str) -> None:
    spec = replace(INVENTORY["granite-30m-sparse"], allowlist=(name, "model.safetensors"))
    with pytest.raises(SnapshotPolicyError, match="Python code|pickle-compatible"):
        BoundedSnapshotResolver._validate_spec(spec)


def test_resolver_rejects_auto_map_after_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "external.Model"}}))
    (tmp_path / "model.safetensors").write_bytes(b"safe")
    spec = replace(
        INVENTORY["granite-30m-sparse"], allowlist=("config.json", "model.safetensors"),
        download_cap_bytes=100, expected_identity=(),
    )
    rows = _dry_rows(spec, bytes_each=4)
    rows[0].file_size = (tmp_path / "config.json").stat().st_size
    monkeypatch.setattr(inventory_module, "snapshot_download", lambda **kwargs: rows if kwargs.get("dry_run") else str(tmp_path))
    plan = BoundedSnapshotResolver().plan(spec)
    plan["declared_bytes"] = sum((tmp_path / name).stat().st_size for name in spec.allowlist)
    with pytest.raises(SnapshotPolicyError, match="auto_map"):
        BoundedSnapshotResolver().resolve(spec, plan=plan)


def test_resolver_rejects_unexpected_snapshot_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"safe")
    (tmp_path / "loader.py").write_text("pass")
    spec = replace(
        INVENTORY["granite-30m-sparse"], allowlist=("config.json", "model.safetensors"),
        download_cap_bytes=100, expected_identity=(),
    )
    plan = {"key": spec.key, "revision": spec.revision, "declared_bytes": 6}
    monkeypatch.setattr(inventory_module, "snapshot_download", lambda **kwargs: str(tmp_path))
    with pytest.raises(SnapshotPolicyError, match="unexpected"):
        BoundedSnapshotResolver().resolve(spec, plan=plan)


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"length": [700 if "long" in text else 4 for text in texts]}


class FakeEncoder:
    def __init__(self, *_args, device="cpu", **_kwargs):
        self.device = device
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.max_seq_length = None

    def _result(self, texts, role, **kwargs):
        import torch

        self.calls.append((role, kwargs))
        indices = torch.tensor([[row for row in range(len(texts))], [row + 1 for row in range(len(texts))]])
        return torch.sparse_coo_tensor(indices, torch.ones(len(texts)), (len(texts), 50265))

    def encode_query(self, texts, **kwargs):
        return self._result(texts, "query", **kwargs)

    def encode_document(self, texts, **kwargs):
        return self._result(texts, "document", **kwargs)


def test_generic_adapter_routes_prunes_and_audits_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.json").write_text("{}")
    identity = (("config.json", file_sha256(tmp_path / "config.json")),)
    spec = replace(INVENTORY["granite-30m-sparse"], expected_identity=identity)
    monkeypatch.setitem(inventory_module.INVENTORY, spec.key, spec)
    import mm_embed.providers.sentence_transformers_sparse_provider as adapter_module
    monkeypatch.setitem(adapter_module.INVENTORY, spec.key, spec)
    provider = SentenceTransformersSparseProvider(
        model_key=spec.key, snapshot_path=str(tmp_path), device="cpu", batch_size=4, encoder_factory=FakeEncoder,
    )
    queries = provider.encode_sparse_queries(["long query"], item_ids=["q1"])
    documents = provider.encode_sparse_documents(["doc one", "doc two"], item_ids=["d1", "d2"])
    assert queries.metadata_dict()["truncated_count"] == 1
    assert queries.metadata_dict()["pruning_max_active_dims"] == 50
    assert documents.metadata_dict()["pruning_max_active_dims"] == 192
    assert queries.embeddings.nnz_total == 1
    assert documents.embeddings.nnz_per_row == (1, 1)
    assert [call[0] for call in provider._encoder.calls] == ["query", "document"]


def test_csr_conversion_rejects_dense_negative_and_non_finite() -> None:
    with pytest.raises(TypeError, match="dense materialization"):
        to_csr(np.ones((1, 2), dtype=np.float32), 1, 2)
    with pytest.raises(ValueError, match="negative"):
        to_csr(sparse.csr_matrix([[-1.0, 0.0]]), 1, 2)
    with pytest.raises(ValueError, match="non-finite"):
        to_csr(sparse.csr_matrix([[np.inf, 0.0]]), 1, 2)


class FakeChunkProvider:
    batch_size = 2

    def encode_sparse_documents(self, texts, *, item_ids):
        matrix = sparse.csr_matrix(np.eye(len(texts), 4, dtype=np.float32))
        return SparseEmbeddingResult(
            SparseEmbeddingBatch(matrix, item_ids, SparseRepresentation("r", "v", 4)), SparseEmbeddingRole.DOCUMENT,
            "model", "provider", "a" * 40, SparseEncodingRoute.NEURAL, SparseEncodingRoute.NEURAL, 2.0,
            device="cpu", peak_vram_bytes=0,
            metadata={"cpu_time_s": 1.0, "truncated_count": 1, "peak_ram_bytes": 100,
                      "max_observed_tokens": 9, "tokenizer_class": "FakeTokenizer", "batch_size_used": 2},
        )


def test_resumable_chunks_bind_content_ids_and_config(tmp_path: Path) -> None:
    documents = tuple({"id": f"d{i}", "content": f"text {i}"} for i in range(3))
    contract = {"chunk_size": 2, "model_revision": "a" * 40, "dimensions": 4}
    first = encode_document_chunks(FakeChunkProvider(), documents, tmp_path, contract)
    second = encode_document_chunks(FakeChunkProvider(), documents, tmp_path, contract)
    assert first["matrix"].shape == second["matrix"].shape == (3, 4)
    audit_path = tmp_path / "000000.json"
    audit = json.loads(audit_path.read_text())
    audit["input_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(BatchAError, match="identity mismatch"):
        encode_document_chunks(FakeChunkProvider(), documents, tmp_path, contract)


def _metrics() -> dict[str, float]:
    return {name: 0.0 for name in ("ndcg@10", "map@100", "mrr@10", "recall@10", "recall@100")}


def _intervals() -> dict[str, dict[str, float | int]]:
    return {
        name: {"mean": 0.0, "low": 0.0, "high": 0.0, "samples": 10_000, "seed": 20_260_826}
        for name in _metrics()
    }


def _nnz(count: int) -> dict[str, float | int]:
    return {"count": count, "total": count, "mean": 1.0, "p50": 1.0, "p95": 1.0, "max": 1, "empty": 0}


def _valid_raw_manifest(model_key: str, track: str) -> dict:
    spec = INVENTORY[model_key]
    queries, qrels = (103, 800) if track == "economics" else (101, 692)
    model = {
        "key": model_key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "license": spec.license,
        "provider": spec.adapter,
        "mechanism": batch_module.MECHANISMS[model_key],
        "snapshot_identity_sha256": batch_module._expected_snapshot_aggregate(model_key),
        "vocabulary_id": spec.vocabulary_id,
        "dimensions": spec.dimensions,
        "query_route": spec.query_route,
        "document_route": spec.document_route,
        "training_overlap_boundary": batch_module.TRAINING_OVERLAP_BOUNDARY[model_key],
    }
    document_audit = {
        "count": 7500,
        "latency_ms": 1.0,
        "cpu_time_s": 1.0,
        "truncated_count": 0,
        "max_observed_tokens": 10,
        "peak_ram_bytes": 1,
        "peak_vram_bytes": 1,
        "tokenizer_classes": ["Tokenizer"],
        "batch_sizes_used": [8],
    }
    query_audit = {
        "count": queries,
        "latency_ms": 1.0,
        "cpu_time_s": 1.0,
        "truncated_count": 0,
        "max_observed_tokens": 10,
        "peak_ram_bytes": 1,
        "peak_vram_bytes": 1,
        "tokenizer_class": "Tokenizer",
        "batch_size_requested": 8,
        "batch_size_used": 8,
        "pruning_max_active_dims": spec.query_pruning,
    }
    return {
        "schema_version": "bright-learned-sparse-result-v1",
        "evidence_tier": "research_benchmark",
        "publication": batch_module.PUBLICATION,
        "model": model,
        "data": {
            "dataset_version": batch_module.DATASET_VERSION,
            "track": track,
            "documents": 7500,
            "queries": queries,
            "qrels": qrels,
            "materialization_manifest_sha256": "a" * 64,
            "source_revision": "b" * 40,
            "selection_ids_sha256": "c" * 64,
            "corpus_sha256": "d" * 64,
            "queries_sha256": "e" * 64,
            "qrels_sha256": "f" * 64,
        },
        "config": {
            "schema_version": "bright-learned-sparse-chunk-v1",
            "dataset_version": batch_module.DATASET_VERSION,
            "model_key": model_key,
            "model_revision": spec.revision,
            "snapshot_identity_sha256": model["snapshot_identity_sha256"],
            "track": track,
            "dimensions": spec.dimensions,
            "max_length": 512,
            "query_route": spec.query_route,
            "document_route": spec.document_route,
            "query_pruning": spec.query_pruning,
            "document_pruning": spec.document_pruning,
            "chunk_size": 256,
            "batch_size": 8,
            "trust_remote_code": False,
        },
        "metrics": _metrics(),
        "confidence_intervals": _intervals(),
        "nnz": {"documents": _nnz(7500), "queries": _nnz(queries)},
        "representations": {
            "documents": {"shape": [7500, spec.dimensions], "nnz": 7500, "item_ids_sha256": "1" * 64, "chunk_count": 30},
            "queries": {"shape": [queries, spec.dimensions], "nnz": queries, "item_ids_sha256": "2" * 64},
        },
        "audit": {"document_chunks": document_audit, "query": query_audit, "exact_search_s": 1.0},
        "search": {"backend": "scipy_csr_exact", "exact": True, "top_k": 100, "tie_break": "document_id_ascending"},
        "artifacts": {f"artifact-{index}": {"bytes": 1, "sha256": str(index) * 64} for index in range(1, 8)},
    }


def test_tracked_aggregate_requires_eight_cells_and_excludes_restricted_fields() -> None:
    rows = [
        _valid_raw_manifest(model, track)
        for model in ("granite-30m-sparse", "opensearch-doc-v2-mini", "opensearch-doc-v3", "bge-m3")
        for track in ("economics", "psychology")
    ]
    aggregate = tracked_aggregate(rows)
    assert len(aggregate["cells"]) == 8
    keys = set()

    def collect(value):
        if isinstance(value, dict):
            keys.update(value)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(aggregate)
    for forbidden in ("query_id", "document_id", "rankings", "per_query_metrics", "text"):
        assert forbidden not in keys


def test_strict_schemas_reject_nested_extra_fields_and_open_publication() -> None:
    inventory = inventory_document()
    inventory["models"][0]["unreviewed"] = True
    with pytest.raises(BatchAError, match="inventory.*validation failed"):
        batch_module._validate_schema(inventory, "learned-sparse-inventory-v01.schema.json")

    raw = _valid_raw_manifest("granite-30m-sparse", "economics")
    raw["audit"]["query"]["unreviewed"] = 1
    with pytest.raises(BatchAError, match="result.*validation failed"):
        batch_module._validate_schema(raw, "bright-learned-sparse-result-v01.schema.json")

    raw = _valid_raw_manifest("granite-30m-sparse", "economics")
    raw["audit"]["query"]["cpu_time_s"] = 0.0
    with pytest.raises(BatchAError, match="result.*validation failed"):
        batch_module._validate_schema(raw, "bright-learned-sparse-result-v01.schema.json")

    resource = {
        "nnz": [1],
        "latency_ms": 1.0,
        "truncated_count": 0,
        "max_observed_tokens": 10,
        "cpu_time_s": 1.0,
        "peak_ram_bytes": 1,
        "peak_vram_bytes": 1,
        "batch_size_used": 8,
        "tokenizer_class": "Tokenizer",
    }
    gate = {
        "schema_version": "learned-sparse-gate-v1",
        "publication": batch_module.PUBLICATION,
        "model": {
            "key": "granite-30m-sparse",
            "repo_id": INVENTORY["granite-30m-sparse"].repo_id,
            "revision": INVENTORY["granite-30m-sparse"].revision,
            "license": "Apache-2.0",
            "mechanism": batch_module.MECHANISMS["granite-30m-sparse"],
            "training_overlap_boundary": batch_module.TRAINING_OVERLAP_BOUNDARY["granite-30m-sparse"],
            "snapshot": {
                "path": "/cache/snapshot",
                "files": {"config.json": "a" * 64},
                "aggregate_identity_sha256": "b" * 64,
                "actual_bytes": 1,
            },
            "query_route": "neural",
            "document_route": "neural",
            "dimensions": 50265,
            "max_length": 512,
            "query_pruning": 50,
            "document_pruning": 192,
        },
        "data": {
            "dataset_version": batch_module.DATASET_VERSION,
            "track": "economics",
            "query_count": 1,
            "document_count": 4,
            "query_ids_sha256": "c" * 64,
            "document_ids_sha256": "d" * 64,
            "materialization_manifest_sha256": "e" * 64,
            "source_revision": "f" * 40,
            "selection_ids_sha256": "1" * 64,
        },
        "query": resource,
        "documents": {**resource, "nnz": [1, 1, 1, 1]},
        "search": {
            "backend": "scipy_csr_exact",
            "exact": True,
            "top_k": 4,
            "tie_break": "document_id_ascending",
            "ranking_fingerprint": "2" * 64,
        },
    }
    batch_module._validate_schema(gate, "learned-sparse-gate-v01.schema.json")
    gate["publication"] = {**batch_module.PUBLICATION, "gate": "open"}
    with pytest.raises(BatchAError, match="gate.*validation failed"):
        batch_module._validate_schema(gate, "learned-sparse-gate-v01.schema.json")

    rows = [
        _valid_raw_manifest(model, track)
        for model in ("granite-30m-sparse", "opensearch-doc-v2-mini", "opensearch-doc-v3", "bge-m3")
        for track in ("economics", "psychology")
    ]
    summary = tracked_aggregate(rows)
    summary["cells"][0]["counts"]["restricted_ids"] = 1
    with pytest.raises(BatchAError, match="summary.*validation failed"):
        batch_module._validate_schema(summary, "bright-learned-sparse-summary-v01.schema.json")

    summary = tracked_aggregate(rows)
    summary["cells"][0]["audit"]["query"]["tokenizer_class"] = "unknown"
    with pytest.raises(BatchAError, match="summary.*validation failed"):
        batch_module._validate_schema(summary, "bright-learned-sparse-summary-v01.schema.json")


def test_failure_and_tracked_manifest_schemas_reject_nested_drift() -> None:
    failure = {
        "schema_version": "bright-learned-sparse-failure-cases-v1",
        "dataset_version": batch_module.DATASET_VERSION,
        "publication": batch_module.PUBLICATION,
        "selection_rule": batch_module.FAILURE_SELECTION_RULE,
        "contains_restricted_ids": True,
        "contains_source_text": False,
        "cases": [
            {
                "model_key": model,
                "track": track,
                "raw_manifest_sha256": "a" * 64,
                "query_id": "restricted-query",
                "metrics": _metrics(),
                "relevant_document_ids": ["restricted-document"],
                "excluded_document_ids": [],
                "hits": [
                    {"rank": rank, "document_id": f"restricted-document-{rank}", "score": 1.0}
                    for rank in range(1, 101)
                ],
            }
            for model in ("granite-30m-sparse", "opensearch-doc-v2-mini", "opensearch-doc-v3", "bge-m3")
            for track in ("economics", "psychology")
        ],
    }
    batch_module._validate_schema(failure, "bright-learned-sparse-failure-cases-v01.schema.json")
    failure["cases"][0]["hits"][0]["text"] = "forbidden"
    with pytest.raises(BatchAError, match="failure-cases.*validation failed"):
        batch_module._validate_schema(failure, "bright-learned-sparse-failure-cases-v01.schema.json")

    tracked = {
        "schema_version": "bright-learned-sparse-artifact-manifest-v1",
        "dataset_version": batch_module.DATASET_VERSION,
        "publication": batch_module.PUBLICATION,
        "protocol": {
            "models": ["granite-30m-sparse", "opensearch-doc-v2-mini", "opensearch-doc-v3", "bge-m3"],
            "tracks": ["economics", "psychology"],
            "dataset_version": batch_module.DATASET_VERSION,
            "canonical_data_root": "data/bright-nontechnical-pilot-v0.2",
            "materialization_manifest_sha256": "a" * 64,
            "source_revision": "b" * 40,
            "device": "cuda:0",
            "batch_size": 8,
            "chunk_size": 256,
            "max_length": 512,
            "search": {
                "backend": "scipy_csr_exact", "exact": True, "top_k": 100,
                "tie_break": "document_id_ascending",
            },
            "trust_remote_code": False,
            "identity_sha256": "c" * 64,
        },
        "tracked_files": {
            "benchmark/artifacts/package/inventory.json": {"bytes": 1, "sha256": "d" * 64},
            "benchmark/artifacts/package/summary.json": {"bytes": 1, "sha256": "e" * 64},
            "benchmark/research/report.md": {"bytes": 1, "sha256": "f" * 64},
        },
        "raw_results": [
            {
                "model_key": model,
                "track": track,
                "manifest_sha256": "1" * 64,
                "sidecar_sha256": "2" * 64,
                "artifact_map_sha256": "3" * 64,
            }
            for model in ("granite-30m-sparse", "opensearch-doc-v2-mini", "opensearch-doc-v3", "bge-m3")
            for track in ("economics", "psychology")
        ],
        "content_policy": (
            "Tracked files contain no source text, query or document identifiers, rankings, or per-query rows."
        ),
    }
    batch_module._validate_schema(tracked, "bright-learned-sparse-artifact-manifest-v01.schema.json")
    tracked["protocol"]["search"]["approximate"] = False
    with pytest.raises(BatchAError, match="artifact-manifest.*validation failed"):
        batch_module._validate_schema(tracked, "bright-learned-sparse-artifact-manifest-v01.schema.json")


def test_catalog_provider_resolves_standard_hub_cache_without_tracked_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_spec = load_catalog().require_model("granite-embedding-30m-sparse")
    assert "snapshot_path" not in catalog_spec.provider_kwargs
    assert catalog_spec.provider_kwargs["allow_download"] is False
    snapshot = tmp_path / "hub" / "snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    frozen = replace(
        INVENTORY["granite-30m-sparse"],
        expected_identity=(("config.json", file_sha256(snapshot / "config.json")),),
    )
    monkeypatch.setitem(inventory_module.INVENTORY, frozen.key, frozen)
    import mm_embed.providers.sentence_transformers_sparse_provider as adapter_module

    monkeypatch.setitem(adapter_module.INVENTORY, frozen.key, frozen)

    def resolve_local(resolver, spec):
        assert resolver.hub_cache_dir == tmp_path / "hub"
        return {
            "snapshot_path": str(snapshot),
            "cache": {"hub_cache_dir": str(resolver.hub_cache_dir)},
        }

    monkeypatch.setattr(BoundedSnapshotResolver, "resolve_local", resolve_local)
    provider = SentenceTransformersSparseProvider(
        **catalog_spec.provider_kwargs,
        cache_root=str(tmp_path),
        device="cpu",
        encoder_factory=FakeEncoder,
    )
    assert provider.hub_cache_dir == str(tmp_path / "hub")


class FakeReplayProvider:
    name = "sentence_transformers_sparse"
    model = INVENTORY["granite-30m-sparse"].repo_id
    revision = INVENTORY["granite-30m-sparse"].revision
    representation = SparseRepresentation("granite-30m-sparse-csr-v1", "roberta-bpe-50265", 50265)
    query_route = SparseEncodingRoute.NEURAL
    document_route = SparseEncodingRoute.NEURAL
    device = "cpu"
    batch_size = 2

    def _encode(self, texts, item_ids, role):
        rows = np.arange(len(texts))
        columns = rows % 2
        matrix = sparse.csr_matrix(
            (np.ones(len(texts), dtype=np.float32), (rows, columns)),
            shape=(len(texts), self.representation.dimensions),
        )
        return SparseEmbeddingResult(
            SparseEmbeddingBatch(matrix, item_ids, self.representation),
            role,
            self.model,
            self.name,
            self.revision,
            self.query_route,
            self.document_route,
            1.0,
            device="cpu",
            peak_vram_bytes=0,
            metadata={
                "cpu_time_s": 0.5,
                "truncated_count": 0,
                "max_observed_tokens": 8,
                "peak_ram_bytes": 100,
                "tokenizer_class": "FakeTokenizer",
                "batch_size_used": 2,
                "pruning_max_active_dims": 50 if role is SparseEmbeddingRole.QUERY else 192,
            },
        )

    def encode_sparse_documents(self, texts, *, item_ids):
        return self._encode(texts, item_ids, SparseEmbeddingRole.DOCUMENT)

    def encode_sparse_queries(self, texts, *, item_ids):
        return self._encode(texts, item_ids, SparseEmbeddingRole.QUERY)


def _mini_replay_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    data = TrackData(
        name="economics",
        corpus=tuple({"id": f"d{index}", "content": f"document {index}"} for index in range(4)),
        queries=(
            {"id": "q0", "text": "query zero", "excluded_ids": ["d3"]},
            {"id": "q1", "text": "query one", "excluded_ids": []},
        ),
        qrels={"q0": {"d0": 1}, "q1": {"d1": 1}},
    )
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "manifest.json").write_text("{}")
    files = {
        name: {"path": name, "bytes": 1, "sha256": character * 64}
        for name, character in (
            ("corpus.jsonl", "a"),
            ("queries.jsonl", "b"),
            ("qrels.jsonl", "c"),
            ("selection.jsonl", "d"),
            ("audit.json", "e"),
            ("sensitive-evidence.jsonl", "f"),
        )
    }
    materialization = {
        "source": {"revision": "1" * 40},
        "tracks": {
            "economics": {
                "documents": 4,
                "queries": 2,
                "positive_qrels": 2,
                "selection_ids_sha256": "2" * 64,
                "files": files,
            }
        },
    }
    monkeypatch.setattr(batch_module, "validate_materialization", lambda root: materialization)
    monkeypatch.setattr(batch_module, "load_materialized", lambda root, track: data)
    monkeypatch.setattr(batch_module, "_validate_schema", lambda value, schema: None)
    monkeypatch.setattr(batch_module, "_resolve_snapshot_path", lambda *args: tmp_path / "snapshot")
    monkeypatch.setattr(
        batch_module,
        "_snapshot_audit",
        lambda *args: {
            "aggregate_identity_sha256": batch_module._expected_snapshot_aggregate("granite-30m-sparse"),
            "files": INVENTORY["granite-30m-sparse"].identity,
            "actual_bytes": 1,
            "path": str(tmp_path / "snapshot"),
        },
    )
    monkeypatch.setattr(batch_module, "_provider", lambda *args: FakeReplayProvider())
    output = tmp_path / "result"
    batch_module.run_track(
        model_key="granite-30m-sparse",
        track="economics",
        data_root=data_root,
        output_root=output,
        device="cpu",
        batch_size=2,
        chunk_size=2,
    )
    validate_raw_result(
        output,
        data_root,
        expected_manifest_sha256=batch_module.sha256_file(output / "manifest.json"),
    )
    return output, data_root


def _refresh_artifact(root: Path, relative: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = root / relative
    manifest["artifacts"][relative] = {
        "bytes": artifact.stat().st_size,
        "sha256": batch_module.sha256_file(artifact),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "manifest.sha256").write_text(batch_module.sha256_file(manifest_path) + "\n")


@pytest.mark.parametrize(
    "tamper",
    ("query_csr", "query_ids", "document_chunk", "rankings", "metrics", "manifest_data"),
)
def test_raw_replay_fails_closed_for_coordinated_tampering(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, data_root = _mini_replay_result(tmp_path, monkeypatch)
    if tamper == "query_csr":
        matrix = sparse.load_npz(root / "queries.npz").tocsr()
        matrix.data[0] += 7.0
        sparse.save_npz(root / "queries.npz", matrix)
        _refresh_artifact(root, "queries.npz")
    elif tamper == "query_ids":
        values = json.loads((root / "query_ids.json").read_text())
        values[0] = "forged-query"
        (root / "query_ids.json").write_text(json.dumps(values) + "\n")
        _refresh_artifact(root, "query_ids.json")
    elif tamper == "document_chunk":
        matrix = sparse.load_npz(root / "document_chunks/000000.npz").tocsr()
        matrix.data[0] += 4.0
        sparse.save_npz(root / "document_chunks/000000.npz", matrix)
        _refresh_artifact(root, "document_chunks/000000.npz")
    elif tamper == "rankings":
        values = json.loads((root / "rankings.json").read_text())
        values[0]["hits"][0]["score"] += 1.0
        (root / "rankings.json").write_text(json.dumps(values) + "\n")
        _refresh_artifact(root, "rankings.json")
    elif tamper == "metrics":
        values = json.loads((root / "per_query_metrics.json").read_text())
        values["q0"]["ndcg@10"] = 0.25
        (root / "per_query_metrics.json").write_text(json.dumps(values) + "\n")
        _refresh_artifact(root, "per_query_metrics.json")
    else:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["data"]["corpus_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(BatchAError):
        validate_raw_result(
            root,
            data_root,
            expected_manifest_sha256=batch_module.sha256_file(root / "manifest.json"),
        )


def test_raw_replay_rejects_coordinated_dormant_csr_attack_without_external_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, data_root = _mini_replay_result(tmp_path, monkeypatch)
    externally_recorded = batch_module.sha256_file(root / "manifest.json")
    matrix_path = root / "document_chunks/000000.npz"
    matrix = sparse.load_npz(matrix_path).tocsr()
    matrix.data[0] += 0.125
    sparse.save_npz(matrix_path, matrix)
    audit_path = root / "document_chunks/000000.json"
    audit = json.loads(audit_path.read_text())
    audit["matrix_sha256"] = batch_module.sha256_file(matrix_path)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    _refresh_artifact(root, "document_chunks/000000.npz")
    _refresh_artifact(root, "document_chunks/000000.json")

    with pytest.raises(BatchAError, match="externally supplied"):
        validate_raw_result(root, data_root)
    with pytest.raises(BatchAError, match="externally supplied identity"):
        validate_raw_result(
            root,
            data_root,
            expected_manifest_sha256=externally_recorded,
        )


def test_finalized_run_resume_requires_and_verifies_external_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, data_root = _mini_replay_result(tmp_path, monkeypatch)
    identity = batch_module.sha256_file(root / "manifest.json")
    with pytest.raises(BatchAError, match="externally supplied"):
        batch_module.run_track(
            model_key="granite-30m-sparse", track="economics", data_root=data_root,
            output_root=root, device="cpu", batch_size=2, chunk_size=2,
        )
    resumed = batch_module.run_track(
        model_key="granite-30m-sparse", track="economics", data_root=data_root,
        output_root=root, device="cpu", batch_size=2, chunk_size=2,
        expected_manifest_sha256=identity,
    )
    assert resumed["model"]["key"] == "granite-30m-sparse"


def test_incomplete_resume_is_explicitly_untrusted_by_default(tmp_path: Path) -> None:
    output = tmp_path / "incomplete"
    chunks = output / "document_chunks"
    chunks.mkdir(parents=True)
    (chunks / "000000.npz").write_bytes(b"partial")
    with pytest.raises(BatchAError, match="not externally authenticated"):
        batch_module.run_track(
            model_key="granite-30m-sparse", track="economics", data_root=tmp_path / "missing",
            output_root=output, device="cpu", batch_size=8, chunk_size=256,
        )
