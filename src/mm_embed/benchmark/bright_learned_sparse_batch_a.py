"""Research-only BRIGHT Batch-A learned-sparse runner and audit contracts."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from huggingface_hub import snapshot_download
from jsonschema import Draft202012Validator
from scipy import sparse

from mm_embed.benchmark.bright_multidomain_v02 import TRACKS, load_materialized, validate_materialization
from mm_embed.benchmark.retrieval_v01 import aggregate_metrics, bootstrap_confidence_intervals, query_metrics
from mm_embed.indexes.sparse_exact import ExactSparseIndex
from mm_embed.providers import get_provider
from mm_embed.providers.learned_sparse_inventory import (
    ANCHOR_KEYS,
    INVENTORY,
    SELECTED_KEYS,
    BoundedSnapshotResolver,
    aggregate_snapshot_identity,
    inventory_document,
    standard_hub_cache_dir,
)
from mm_embed.providers.snapshot_identity import snapshot_identity
from mm_embed.providers.sparse_base import (
    SparseEmbeddingBatch,
    SparseEmbeddingResult,
    SparseEmbeddingRole,
    SparseEncodingRoute,
    SparseRepresentation,
)

DATASET_VERSION = "bright-nontechnical-pilot-v0.2"
TOP_K = 100
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_826
PUBLICATION = {
    "publish": False,
    "leaderboard_publish": False,
    "public_export_allowed": False,
    "gate": "closed",
    "classification": "research_only",
}
TRAINING_OVERLAP_BOUNDARY = {
    "granite-30m-sparse": (
        "Potential source-family overlap with StackExchange-derived BRIGHT data; results must not be described as zero-shot."
    ),
    "opensearch-doc-v2-mini": (
        "Potential source-family overlap with StackExchange-derived BRIGHT data; results must not be described as zero-shot."
    ),
    "opensearch-doc-v3": (
        "Potential source-family overlap with StackExchange-derived BRIGHT data; results must not be described as zero-shot."
    ),
    "bge-m3": (
        "Training overlap with BRIGHT source material is not ruled out; results must not be described as zero-shot."
    ),
}
MECHANISMS = {
    "granite-30m-sparse": "neural SPLADE masked-language-model expansion with role-specific top-k pruning",
    "opensearch-doc-v2-mini": "static sparse query lookup plus neural document expansion",
    "opensearch-doc-v3": "static sparse query lookup plus distilled neural document expansion",
    "bge-m3": "contextual lexical token-weight projection with max pooling",
}
FAILURE_SELECTION_RULE = (
    "Select the minimum nDCG@10 query in each cell; break metric ties by canonical query ID ascending."
)


class BatchAError(ValueError):
    """Raised when a Batch-A artifact or execution violates its fixed contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str | None, *, label: str) -> str:
    if value is None:
        raise BatchAError(f"Missing externally supplied {label}")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise BatchAError(f"Invalid externally supplied {label}")
    return value


def _peek_raw_cell(root: Path) -> tuple[str, str]:
    """Read only routing fields; the caller must still verify the external manifest identity."""
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return str(manifest["model"]["key"]), str(manifest["data"]["track"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BatchAError(f"Cannot identify raw result cell at {root}") from exc


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ids_sha256(values: Iterable[str]) -> str:
    return canonical_sha256(list(values))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _validate_schema(value: Any, schema_name: str) -> None:
    schema_path = Path(__file__).parents[3] / "schemas" / schema_name
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(value)
    except Exception as exc:
        raise BatchAError(f"{schema_name} validation failed: {exc}") from exc


def _nnz_stats(matrix: sparse.csr_matrix) -> dict[str, float | int]:
    values = np.diff(matrix.indptr)
    return {
        "count": int(values.size),
        "total": int(values.sum()),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(values.max()),
        "empty": int(np.sum(values == 0)),
    }


def _provider(model_key: str, snapshot_path: Path, device: str, batch_size: int) -> Any:
    spec = INVENTORY[model_key]
    kwargs = {
        "snapshot_path": str(snapshot_path),
        "device": device,
        "batch_size": batch_size,
        "max_length": spec.max_length,
    }
    if model_key in SELECTED_KEYS:
        kwargs["model_key"] = model_key
    elif model_key == "opensearch-doc-v3":
        kwargs["allow_download"] = False
    elif model_key != "bge-m3":
        raise BatchAError(f"{model_key} is not a Batch-A selected model or anchor")
    return get_provider(spec.adapter, **kwargs)


def _resolve_snapshot_path(
    model_key: str,
    snapshot_path: str | Path | None,
    cache_root: str | Path | None,
) -> Path:
    if snapshot_path is not None:
        return Path(snapshot_path).expanduser().resolve()
    spec = INVENTORY[model_key]
    if model_key in SELECTED_KEYS:
        resolved = BoundedSnapshotResolver(cache_root=cache_root).resolve_local(spec)
        return Path(resolved["snapshot_path"])
    try:
        return Path(
            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                cache_dir=standard_hub_cache_dir(cache_root),
                local_files_only=True,
            )
        ).resolve()
    except Exception as exc:
        raise BatchAError(f"{model_key} is unavailable in the standard local hub cache") from exc


def _snapshot_audit(model_key: str, snapshot_path: Path) -> dict[str, Any]:
    identity = snapshot_identity(snapshot_path)
    spec = INVENTORY[model_key]
    if spec.identity and identity != spec.identity:
        raise BatchAError(f"{model_key} snapshot identity drifted")
    return {
        "path": str(snapshot_path.resolve()),
        "files": identity,
        "aggregate_identity_sha256": aggregate_snapshot_identity(identity),
        "actual_bytes": sum((snapshot_path / name).stat().st_size for name in identity),
    }


def _expected_snapshot_aggregate(model_key: str) -> str:
    spec = INVENTORY[model_key]
    if spec.identity:
        return aggregate_snapshot_identity(spec.identity)
    if model_key == "opensearch-doc-v3":
        from mm_embed.providers.opensearch_neural_sparse_provider import SNAPSHOT_IDENTITY

        return aggregate_snapshot_identity(SNAPSHOT_IDENTITY)
    if model_key == "bge-m3":
        from mm_embed.providers.bge_m3_sparse_provider import SNAPSHOT_IDENTITY

        return aggregate_snapshot_identity(SNAPSHOT_IDENTITY)
    raise BatchAError(f"No frozen snapshot identity for {model_key}")


def _model_contract(model_key: str, provider: Any, snapshot_sha256: str) -> dict[str, Any]:
    spec = INVENTORY[model_key]
    return {
        "key": model_key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "license": spec.license,
        "provider": provider.name,
        "mechanism": MECHANISMS[model_key],
        "snapshot_identity_sha256": snapshot_sha256,
        "vocabulary_id": provider.representation.vocabulary_id,
        "dimensions": provider.representation.dimensions,
        "query_route": provider.query_route.value,
        "document_route": provider.document_route.value,
        "training_overlap_boundary": TRAINING_OVERLAP_BOUNDARY[model_key],
    }


def _result_from_csr(provider: Any, matrix: sparse.csr_matrix, item_ids: list[str], role: SparseEmbeddingRole) -> SparseEmbeddingResult:
    return SparseEmbeddingResult(
        embeddings=SparseEmbeddingBatch(matrix, item_ids, provider.representation),
        role=role,
        model_name=provider.model,
        provider=provider.name,
        model_revision=provider.revision,
        query_route=provider.query_route,
        document_route=provider.document_route,
        latency_ms=0.0,
        device=str(provider.device),
        peak_vram_bytes=0,
        metadata={"reconstructed_from_verified_chunks": True},
    )


def _result_from_manifest(
    model: dict[str, Any],
    matrix: sparse.csr_matrix,
    item_ids: list[str],
    role: SparseEmbeddingRole,
) -> SparseEmbeddingResult:
    representation = SparseRepresentation(
        f"{model['key']}-csr-v1",
        model["vocabulary_id"],
        model["dimensions"],
    )
    return SparseEmbeddingResult(
        embeddings=SparseEmbeddingBatch(matrix, item_ids, representation),
        role=role,
        model_name=model["repo_id"],
        provider=model["provider"],
        model_revision=model["revision"],
        query_route=SparseEncodingRoute(model["query_route"]),
        document_route=SparseEncodingRoute(model["document_route"]),
        latency_ms=0.0,
        metadata={"replayed_without_model": True},
    )


def _validate_csr(matrix: sparse.csr_matrix, *, rows: int, dimensions: int, label: str) -> None:
    if not sparse.isspmatrix_csr(matrix) or matrix.shape != (rows, dimensions):
        raise BatchAError(f"{label} CSR shape mismatch")
    matrix = matrix.copy()
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise BatchAError(f"{label} CSR must contain finite non-negative values")


def _chunk_contract(
    *, model_key: str, track: str, snapshot: dict[str, Any], provider: Any, chunk_size: int
) -> dict[str, Any]:
    spec = INVENTORY[model_key]
    return {
        "schema_version": "bright-learned-sparse-chunk-v1",
        "dataset_version": DATASET_VERSION,
        "model_key": model_key,
        "model_revision": spec.revision,
        "snapshot_identity_sha256": snapshot["aggregate_identity_sha256"],
        "track": track,
        "dimensions": provider.representation.dimensions,
        "max_length": spec.max_length,
        "query_route": provider.query_route.value,
        "document_route": provider.document_route.value,
        "query_pruning": spec.query_pruning,
        "document_pruning": spec.document_pruning,
        "chunk_size": chunk_size,
        "batch_size": int(provider.batch_size),
        "trust_remote_code": False,
    }


def encode_document_chunks(
    provider: Any,
    documents: tuple[dict[str, Any], ...],
    chunk_root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Create or verify resumable CSR chunks bound to content, ids, and configuration."""
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_size = int(contract["chunk_size"])
    contract_sha = canonical_sha256(contract)
    expected_starts = tuple(range(0, len(documents), chunk_size))
    expected_files = {f"{start:06d}{suffix}" for start in expected_starts for suffix in (".npz", ".json")}
    unexpected = {path.name for path in chunk_root.iterdir() if path.is_file()} - expected_files
    if unexpected:
        raise BatchAError(f"Unexpected document chunk files: {sorted(unexpected)}")
    aggregate = {
        "count": 0,
        "latency_ms": 0.0,
        "cpu_time_s": 0.0,
        "truncated_count": 0,
        "max_observed_tokens": 0,
        "peak_ram_bytes": 0,
        "peak_vram_bytes": 0,
        "tokenizer_classes": [],
        "batch_sizes_used": [],
    }
    attempted_batch_sizes: list[int] = []
    matrices: list[sparse.csr_matrix] = []
    item_ids: list[str] = []
    for start in expected_starts:
        rows = documents[start : start + chunk_size]
        matrix_path = chunk_root / f"{start:06d}.npz"
        audit_path = chunk_root / f"{start:06d}.json"
        ids = [row["id"] for row in rows]
        texts = [row["content"] for row in rows]
        input_sha = canonical_sha256([{"id": row["id"], "content": row["content"]} for row in rows])
        if matrix_path.exists() or audit_path.exists():
            if not matrix_path.is_file() or not audit_path.is_file():
                raise BatchAError(f"Incomplete resumable chunk at offset {start}")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            required = {
                "start", "count", "contract_sha256", "input_sha256", "item_ids_sha256", "matrix_sha256",
                "shape", "nnz", "latency_ms", "cpu_time_s", "truncated_count", "peak_ram_bytes",
                "peak_vram_bytes", "tokenizer_class", "batch_size_used",
                "max_observed_tokens",
            }
            optional = {"batch_size_attempts"}
            if set(audit) not in (required, required | optional):
                raise BatchAError(f"Chunk audit fields drifted at offset {start}")
            if (
                audit["start"] != start
                or audit["count"] != len(rows)
                or audit["contract_sha256"] != contract_sha
                or audit["input_sha256"] != input_sha
                or audit["item_ids_sha256"] != ids_sha256(ids)
                or audit["matrix_sha256"] != sha256_file(matrix_path)
            ):
                raise BatchAError(f"Resumable chunk identity mismatch at offset {start}")
            matrix = sparse.load_npz(matrix_path).tocsr()
            if list(matrix.shape) != audit["shape"] or matrix.nnz != audit["nnz"]:
                raise BatchAError(f"Resumable chunk matrix mismatch at offset {start}")
        else:
            result = provider.encode_sparse_documents(texts, item_ids=ids)
            matrix = result.embeddings.values
            sparse.save_npz(matrix_path, matrix, compressed=True)
            metadata = result.metadata_dict()
            audit = {
                "start": start,
                "count": len(rows),
                "contract_sha256": contract_sha,
                "input_sha256": input_sha,
                "item_ids_sha256": ids_sha256(ids),
                "matrix_sha256": sha256_file(matrix_path),
                "shape": list(matrix.shape),
                "nnz": matrix.nnz,
                "latency_ms": result.latency_ms,
                "cpu_time_s": float(metadata["cpu_time_s"]),
                "truncated_count": int(metadata.get("truncated_count", 0)),
                "max_observed_tokens": int(metadata["max_observed_tokens"]),
                "peak_ram_bytes": int(metadata.get("peak_ram_bytes", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)),
                "peak_vram_bytes": int(result.peak_vram_bytes or 0),
                "tokenizer_class": str(metadata["tokenizer_class"]),
                "batch_size_used": int(metadata["batch_size_used"]),
            }
            if "batch_size_attempts" in metadata:
                audit["batch_size_attempts"] = [int(value) for value in metadata["batch_size_attempts"]]
            _write_json(audit_path, audit)
        _validate_csr(
            matrix,
            rows=len(rows),
            dimensions=int(contract["dimensions"]),
            label=f"document chunk {start}",
        )
        matrices.append(matrix)
        item_ids.extend(ids)
        aggregate["count"] += audit["count"]
        aggregate["latency_ms"] += audit["latency_ms"]
        aggregate["cpu_time_s"] += audit["cpu_time_s"]
        aggregate["truncated_count"] += audit["truncated_count"]
        aggregate["max_observed_tokens"] = max(aggregate["max_observed_tokens"], audit["max_observed_tokens"])
        aggregate["peak_ram_bytes"] = max(aggregate["peak_ram_bytes"], audit["peak_ram_bytes"])
        aggregate["peak_vram_bytes"] = max(aggregate["peak_vram_bytes"], audit["peak_vram_bytes"])
        if audit["tokenizer_class"] not in aggregate["tokenizer_classes"]:
            aggregate["tokenizer_classes"].append(audit["tokenizer_class"])
        if audit["batch_size_used"] not in aggregate["batch_sizes_used"]:
            aggregate["batch_sizes_used"].append(audit["batch_size_used"])
        for attempted in audit.get("batch_size_attempts", ()):
            if attempted not in attempted_batch_sizes:
                attempted_batch_sizes.append(attempted)
    if aggregate["count"] != len(documents):
        raise BatchAError("Document chunks do not cover the fixed corpus")
    if attempted_batch_sizes:
        aggregate["batch_sizes_attempted"] = attempted_batch_sizes
    return {"matrix": sparse.vstack(matrices, format="csr"), "item_ids": item_ids, "audit": aggregate}


def _rank(
    documents: SparseEmbeddingResult,
    queries: SparseEmbeddingResult,
    data: Any,
) -> dict[str, list[tuple[str, float]]]:
    index = ExactSparseIndex(documents)
    excluded = {row["id"]: set(row.get("excluded_ids", ())) for row in data.queries}
    candidate_k = min(index.document_count, TOP_K + max((len(values) for values in excluded.values()), default=0))
    result = index.search(queries, k=candidate_k)
    rankings = {}
    for row in result.queries:
        rankings[row.query_id] = [
            (hit.item_id, hit.score) for hit in row.hits if hit.item_id not in excluded[row.query_id]
        ][:TOP_K]
    return rankings


def _raw_ranking_rows(rankings: dict[str, list[tuple[str, float]]]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": query_id,
            "hits": [
                {"rank": rank, "document_id": document_id, "score": score}
                for rank, (document_id, score) in enumerate(hits, 1)
            ],
        }
        for query_id, hits in sorted(rankings.items())
    ]


def run_track(
    *,
    model_key: str,
    track: str,
    snapshot_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    data_root: str | Path,
    output_root: str | Path,
    device: str,
    batch_size: int,
    chunk_size: int,
    expected_manifest_sha256: str | None = None,
    allow_incomplete_resume: bool = False,
) -> dict[str, Any]:
    """Run a cell or validate a finalized cell before reusing it.

    Incomplete chunk resume is an explicitly opted-in recovery cache. Its local
    hashes detect accidental corruption but are not an external adversarial
    identity; only a completed manifest pinned outside the raw directory can
    establish that boundary.
    """
    if model_key not in SELECTED_KEYS + ANCHOR_KEYS or track not in TRACKS:
        raise BatchAError("Unsupported Batch-A model or BRIGHT track")
    output = Path(output_root)
    manifest_exists = (output / "manifest.json").exists()
    sidecar_exists = (output / "manifest.sha256").exists()
    if manifest_exists or sidecar_exists:
        if not manifest_exists or not sidecar_exists:
            raise BatchAError("Finalized output has an incomplete manifest identity pair")
        finalized = validate_raw_result(
            output,
            data_root,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if (finalized["model"]["key"], finalized["data"]["track"]) != (model_key, track):
            raise BatchAError("Finalized output cell does not match the requested run")
        return finalized
    existing_files = output.exists() and any(path.is_file() for path in output.rglob("*"))
    if existing_files and not allow_incomplete_resume:
        raise BatchAError(
            "Incomplete output is not externally authenticated; pass allow_incomplete_resume only for trusted recovery"
        )
    materialization = validate_materialization(data_root)
    data = load_materialized(data_root, track)
    snapshot_path = _resolve_snapshot_path(model_key, snapshot_path, cache_root)
    snapshot = _snapshot_audit(model_key, snapshot_path)
    provider = _provider(model_key, snapshot_path, device, batch_size)
    output.mkdir(parents=True, exist_ok=True)
    contract = _chunk_contract(
        model_key=model_key, track=track, snapshot=snapshot, provider=provider, chunk_size=chunk_size
    )
    encoded_documents = encode_document_chunks(provider, data.corpus, output / "document_chunks", contract)
    documents = _result_from_csr(
        provider, encoded_documents["matrix"], encoded_documents["item_ids"], SparseEmbeddingRole.DOCUMENT
    )
    query_result = provider.encode_sparse_queries(
        [row["text"] for row in data.queries], item_ids=[row["id"] for row in data.queries]
    )
    query_matrix = query_result.embeddings.values
    query_ids = list(query_result.embeddings.item_ids)
    document_ids = list(encoded_documents["item_ids"])
    sparse.save_npz(output / "queries.npz", query_matrix, compressed=True)
    _write_json(output / "query_ids.json", query_ids)
    _write_json(output / "document_ids.json", document_ids)
    search_started = time.perf_counter()
    rankings = _rank(documents, query_result, data)
    search_s = time.perf_counter() - search_started
    per_query = query_metrics(data, rankings)
    query_meta = query_result.metadata_dict()
    raw_rankings = _raw_ranking_rows(rankings)
    _write_json(output / "rankings.json", raw_rankings)
    _write_json(output / "per_query_metrics.json", per_query)
    raw_files = (
        output / "queries.npz",
        output / "query_ids.json",
        output / "document_ids.json",
        output / "rankings.json",
        output / "per_query_metrics.json",
        *sorted((output / "document_chunks").glob("*")),
    )
    query_audit = {
        "count": len(data.queries),
        "latency_ms": query_result.latency_ms,
        "cpu_time_s": float(query_meta["cpu_time_s"]),
        "truncated_count": int(query_meta.get("truncated_count", 0)),
        "max_observed_tokens": int(query_meta["max_observed_tokens"]),
        "peak_ram_bytes": int(query_meta["peak_ram_bytes"]),
        "peak_vram_bytes": int(query_result.peak_vram_bytes or 0),
        "tokenizer_class": str(query_meta["tokenizer_class"]),
        "batch_size_requested": batch_size,
        "batch_size_used": int(query_meta["batch_size_used"]),
        "pruning_max_active_dims": query_meta.get("pruning_max_active_dims"),
    }
    if "batch_size_attempts" in query_meta:
        query_audit["batch_size_attempts"] = [int(value) for value in query_meta["batch_size_attempts"]]
    manifest = {
        "schema_version": "bright-learned-sparse-result-v1",
        "evidence_tier": "research_benchmark",
        "publication": PUBLICATION,
        "model": _model_contract(model_key, provider, snapshot["aggregate_identity_sha256"]),
        "data": {
            "dataset_version": DATASET_VERSION,
            "track": track,
            "documents": len(data.corpus),
            "queries": len(data.queries),
            "qrels": sum(map(len, data.qrels.values())),
            "materialization_manifest_sha256": sha256_file(Path(data_root) / "manifest.json"),
            "source_revision": materialization["source"]["revision"],
            "selection_ids_sha256": materialization["tracks"][track]["selection_ids_sha256"],
            "corpus_sha256": materialization["tracks"][track]["files"]["corpus.jsonl"]["sha256"],
            "queries_sha256": materialization["tracks"][track]["files"]["queries.jsonl"]["sha256"],
            "qrels_sha256": materialization["tracks"][track]["files"]["qrels.jsonl"]["sha256"],
        },
        "config": contract,
        "metrics": aggregate_metrics(per_query),
        "confidence_intervals": bootstrap_confidence_intervals(
            per_query, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        ),
        "nnz": {
            "documents": _nnz_stats(documents.embeddings.values),
            "queries": _nnz_stats(query_result.embeddings.values),
        },
        "representations": {
            "documents": {
                "shape": list(documents.embeddings.values.shape),
                "nnz": documents.embeddings.nnz_total,
                "item_ids_sha256": ids_sha256(document_ids),
                "chunk_count": len(tuple((output / "document_chunks").glob("*.npz"))),
            },
            "queries": {
                "shape": list(query_matrix.shape),
                "nnz": query_result.embeddings.nnz_total,
                "item_ids_sha256": ids_sha256(query_ids),
            },
        },
        "audit": {
            "document_chunks": encoded_documents["audit"],
            "query": query_audit,
            "exact_search_s": search_s,
        },
        "search": {"backend": "scipy_csr_exact", "exact": True, "top_k": TOP_K, "tie_break": "document_id_ascending"},
        "artifacts": {
            path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in raw_files
        },
    }
    _validate_schema(manifest, "bright-learned-sparse-result-v01.schema.json")
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    (output / "manifest.sha256").write_text(sha256_file(manifest_path) + "\n", encoding="ascii")
    return manifest


def run_gate(
    *,
    model_key: str,
    snapshot_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    data_root: str | Path,
    output_path: str | Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Run a real one-query/four-document bounded gate on canonical BRIGHT data."""
    materialization = validate_materialization(data_root)
    data = load_materialized(data_root, "economics")
    query = sorted(data.queries, key=lambda row: row["id"])[0]
    relevant_id = sorted(data.qrels[query["id"]])[0]
    excluded = set(query.get("excluded_ids", ()))
    rows_by_id = {row["id"]: row for row in data.corpus}
    documents = [rows_by_id[relevant_id]]
    documents.extend(
        row for row in sorted(data.corpus, key=lambda row: row["id"])
        if row["id"] != relevant_id and row["id"] not in excluded
    )
    documents = documents[:4]
    snapshot_path = _resolve_snapshot_path(model_key, snapshot_path, cache_root)
    snapshot = _snapshot_audit(model_key, snapshot_path)
    provider = _provider(model_key, snapshot_path, device, batch_size)
    document_result = provider.encode_sparse_documents(
        [row["content"] for row in documents], item_ids=[row["id"] for row in documents]
    )
    query_result = provider.encode_sparse_queries([query["text"]], item_ids=[query["id"]])
    ranking = ExactSparseIndex(document_result).search(query_result, k=4)
    document_meta = document_result.metadata_dict()
    query_meta = query_result.metadata_dict()
    if query_result.query_route.value != INVENTORY[model_key].query_route:
        raise BatchAError("Gate query route does not match inventory")
    if document_result.document_route.value != INVENTORY[model_key].document_route:
        raise BatchAError("Gate document route does not match inventory")
    if query_result.embeddings.nnz_total == 0 or document_result.embeddings.nnz_total == 0:
        raise BatchAError("Gate produced an empty sparse side")
    query_audit = {
        "nnz": list(query_result.embeddings.nnz_per_row),
        "latency_ms": query_result.latency_ms,
        "truncated_count": int(query_meta["truncated_count"]),
        "max_observed_tokens": int(query_meta["max_observed_tokens"]),
        "cpu_time_s": float(query_meta["cpu_time_s"]),
        "peak_ram_bytes": int(query_meta["peak_ram_bytes"]),
        "peak_vram_bytes": int(query_result.peak_vram_bytes or 0),
        "batch_size_used": int(query_meta["batch_size_used"]),
        "tokenizer_class": query_meta["tokenizer_class"],
    }
    document_audit = {
        "nnz": list(document_result.embeddings.nnz_per_row),
        "latency_ms": document_result.latency_ms,
        "truncated_count": int(document_meta["truncated_count"]),
        "max_observed_tokens": int(document_meta["max_observed_tokens"]),
        "cpu_time_s": float(document_meta["cpu_time_s"]),
        "peak_ram_bytes": int(document_meta["peak_ram_bytes"]),
        "peak_vram_bytes": int(document_result.peak_vram_bytes or 0),
        "batch_size_used": int(document_meta["batch_size_used"]),
        "tokenizer_class": document_meta["tokenizer_class"],
    }
    for audit, metadata in ((query_audit, query_meta), (document_audit, document_meta)):
        if "batch_size_requested" in metadata:
            audit["batch_size_requested"] = int(metadata["batch_size_requested"])
        if "batch_size_attempts" in metadata:
            audit["batch_size_attempts"] = [int(value) for value in metadata["batch_size_attempts"]]
    gate = {
        "schema_version": "learned-sparse-gate-v1",
        "publication": PUBLICATION,
        "model": {
            "key": model_key,
            "repo_id": provider.model,
            "revision": provider.revision,
            "license": INVENTORY[model_key].license,
            "mechanism": MECHANISMS[model_key],
            "training_overlap_boundary": TRAINING_OVERLAP_BOUNDARY[model_key],
            "snapshot": snapshot,
            "query_route": provider.query_route.value,
            "document_route": provider.document_route.value,
            "dimensions": provider.representation.dimensions,
            "max_length": INVENTORY[model_key].max_length,
            "query_pruning": INVENTORY[model_key].query_pruning,
            "document_pruning": INVENTORY[model_key].document_pruning,
        },
        "data": {
            "dataset_version": DATASET_VERSION,
            "track": "economics",
            "query_count": 1,
            "document_count": 4,
            "query_ids_sha256": ids_sha256([query["id"]]),
            "document_ids_sha256": ids_sha256([row["id"] for row in documents]),
            "materialization_manifest_sha256": sha256_file(Path(data_root) / "manifest.json"),
            "source_revision": materialization["source"]["revision"],
            "selection_ids_sha256": materialization["tracks"]["economics"]["selection_ids_sha256"],
        },
        "query": query_audit,
        "documents": document_audit,
        "search": {
            "backend": ranking.backend,
            "exact": ranking.exact,
            "top_k": 4,
            "tie_break": "document_id_ascending",
            "ranking_fingerprint": canonical_sha256(
                [[hit.rank, hit.item_id, hit.score] for hit in ranking.queries[0].hits]
            ),
        },
    }
    _validate_schema(gate, "learned-sparse-gate-v01.schema.json")
    _write_json(Path(output_path), gate)
    return gate


def tracked_aggregate(raw_manifests: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build a text-free, restricted-id-free aggregate after all eight cells exist."""
    cells = list(raw_manifests)
    expected = {(model, track) for model in SELECTED_KEYS + ANCHOR_KEYS for track in TRACKS}
    actual = {(row["model"]["key"], row["data"]["track"]) for row in cells}
    if actual != expected or len(cells) != len(expected):
        raise BatchAError("Tracked aggregate requires exactly the eight fixed model/track cells")
    for row in cells:
        _validate_schema(row, "bright-learned-sparse-result-v01.schema.json")
        if row.get("publication") != PUBLICATION or row.get("schema_version") != "bright-learned-sparse-result-v1":
            raise BatchAError("Raw result is not an approved research-only Batch-A cell")
    summary = {
        "schema_version": "bright-learned-sparse-summary-v1",
        "dataset_version": DATASET_VERSION,
        "publication": PUBLICATION,
        "cells": [
            {
                "model_key": row["model"]["key"],
                "revision": row["model"]["revision"],
                "repo_id": row["model"]["repo_id"],
                "license": row["model"]["license"],
                "mechanism": row["model"]["mechanism"],
                "query_route": row["model"]["query_route"],
                "document_route": row["model"]["document_route"],
                "training_overlap_boundary": row["model"]["training_overlap_boundary"],
                "snapshot_identity_sha256": row["model"]["snapshot_identity_sha256"],
                "track": row["data"]["track"],
                "counts": {key: row["data"][key] for key in ("documents", "queries", "qrels")},
                "metrics": row["metrics"],
                "confidence_intervals": row["confidence_intervals"],
                "nnz": row["nnz"],
                "audit": row["audit"],
                "search": row["search"],
            }
            for row in sorted(cells, key=lambda item: (item["model"]["key"], item["data"]["track"]))
        ],
        "content_policy": "Aggregate-only fields; row-level and source-bearing fields are omitted.",
    }
    _validate_schema(summary, "bright-learned-sparse-summary-v01.schema.json")
    return summary


class _ReplayOnlyProvider:
    def encode_sparse_documents(self, texts: Any, *, item_ids: Any) -> Any:
        raise BatchAError("Replay found a missing document chunk and will not call a model")


def _expected_data_identity(data_root: Path, materialization: dict[str, Any], track: str) -> dict[str, Any]:
    track_manifest = materialization["tracks"][track]
    return {
        "dataset_version": DATASET_VERSION,
        "track": track,
        "documents": track_manifest["documents"],
        "queries": track_manifest["queries"],
        "qrels": track_manifest["positive_qrels"],
        "materialization_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "source_revision": materialization["source"]["revision"],
        "selection_ids_sha256": track_manifest["selection_ids_sha256"],
        "corpus_sha256": track_manifest["files"]["corpus.jsonl"]["sha256"],
        "queries_sha256": track_manifest["files"]["queries.jsonl"]["sha256"],
        "qrels_sha256": track_manifest["files"]["qrels.jsonl"]["sha256"],
    }


def validate_raw_result(
    path: str | Path,
    data_root: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay exact retrieval and all metrics from saved CSR and ordered ids without loading a model."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    manifest_identity_path = root / "manifest.sha256"
    try:
        external_identity = _require_sha256(
            expected_manifest_sha256, label="raw manifest SHA256"
        )
        actual_manifest_sha256 = sha256_file(manifest_path)
        if external_identity != actual_manifest_sha256:
            raise BatchAError("Raw manifest does not match the externally supplied identity")
        sidecar_identity = manifest_identity_path.read_text(encoding="ascii").strip()
        if sidecar_identity != actual_manifest_sha256:
            raise BatchAError("Raw manifest sidecar identity mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchAError("Missing or invalid raw manifest") from exc
    _validate_schema(manifest, "bright-learned-sparse-result-v01.schema.json")
    model_key = manifest["model"]["key"]
    if model_key not in SELECTED_KEYS + ANCHOR_KEYS:
        raise BatchAError("Raw result model is outside Batch A")
    spec = INVENTORY[model_key]
    model_expected = {
        "key": model_key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "license": spec.license,
        "provider": spec.adapter,
        "mechanism": MECHANISMS[model_key],
        "snapshot_identity_sha256": _expected_snapshot_aggregate(model_key),
        "vocabulary_id": spec.vocabulary_id,
        "dimensions": spec.dimensions,
        "query_route": spec.query_route,
        "document_route": spec.document_route,
        "training_overlap_boundary": TRAINING_OVERLAP_BOUNDARY[model_key],
    }
    if manifest["model"] != model_expected:
        raise BatchAError("Raw manifest model contract mismatch")
    data_path = Path(data_root)
    materialization = validate_materialization(data_path)
    track = manifest["data"]["track"]
    data = load_materialized(data_path, track)
    if manifest["data"] != _expected_data_identity(data_path, materialization, track):
        raise BatchAError("Raw manifest data identity mismatch")
    actual_files = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item not in {manifest_path, manifest_identity_path}
    }
    if actual_files != set(manifest["artifacts"]):
        raise BatchAError("Raw artifact file set mismatch")
    for name, identity in manifest["artifacts"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise BatchAError(f"Unsafe raw artifact path: {name}")
        artifact = root / name
        if artifact.stat().st_size != identity["bytes"] or sha256_file(artifact) != identity["sha256"]:
            raise BatchAError(f"Raw artifact identity mismatch: {name}")
    query_ids = json.loads((root / "query_ids.json").read_text(encoding="utf-8"))
    document_ids = json.loads((root / "document_ids.json").read_text(encoding="utf-8"))
    expected_query_ids = [row["id"] for row in data.queries]
    expected_document_ids = [row["id"] for row in data.corpus]
    if query_ids != expected_query_ids or document_ids != expected_document_ids:
        raise BatchAError("Saved item id order does not match canonical materialization")
    contract = manifest["config"]
    expected_contract = {
        "schema_version": "bright-learned-sparse-chunk-v1",
        "dataset_version": DATASET_VERSION,
        "model_key": model_key,
        "model_revision": spec.revision,
        "snapshot_identity_sha256": manifest["model"]["snapshot_identity_sha256"],
        "track": track,
        "dimensions": spec.dimensions,
        "max_length": spec.max_length,
        "query_route": spec.query_route,
        "document_route": spec.document_route,
        "query_pruning": spec.query_pruning,
        "document_pruning": spec.document_pruning,
        "chunk_size": contract["chunk_size"],
        "batch_size": contract["batch_size"],
        "trust_remote_code": False,
    }
    if contract != expected_contract:
        raise BatchAError("Raw manifest encoding config mismatch")
    encoded_documents = encode_document_chunks(
        _ReplayOnlyProvider(), data.corpus, root / "document_chunks", contract
    )
    query_matrix = sparse.load_npz(root / "queries.npz").tocsr()
    _validate_csr(
        query_matrix,
        rows=len(query_ids),
        dimensions=spec.dimensions,
        label="query",
    )
    expected_representations = {
        "documents": {
            "shape": list(encoded_documents["matrix"].shape),
            "nnz": int(encoded_documents["matrix"].nnz),
            "item_ids_sha256": ids_sha256(document_ids),
            "chunk_count": len(tuple((root / "document_chunks").glob("*.npz"))),
        },
        "queries": {
            "shape": list(query_matrix.shape),
            "nnz": int(query_matrix.nnz),
            "item_ids_sha256": ids_sha256(query_ids),
        },
    }
    if manifest["representations"] != expected_representations:
        raise BatchAError("Raw representation identity mismatch")
    documents = _result_from_manifest(
        manifest["model"], encoded_documents["matrix"], document_ids, SparseEmbeddingRole.DOCUMENT
    )
    queries = _result_from_manifest(manifest["model"], query_matrix, query_ids, SparseEmbeddingRole.QUERY)
    rankings = _rank(documents, queries, data)
    if json.loads((root / "rankings.json").read_text(encoding="utf-8")) != _raw_ranking_rows(rankings):
        raise BatchAError("Saved rankings do not match exact CSR replay")
    per_query = query_metrics(data, rankings)
    if json.loads((root / "per_query_metrics.json").read_text(encoding="utf-8")) != per_query:
        raise BatchAError("Saved per-query metrics do not match exact replay")
    if manifest["metrics"] != aggregate_metrics(per_query):
        raise BatchAError("Raw manifest aggregate metrics do not match exact replay")
    confidence_intervals = bootstrap_confidence_intervals(
        per_query, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    if manifest["confidence_intervals"] != confidence_intervals:
        raise BatchAError("Raw manifest confidence intervals do not match exact replay")
    expected_nnz = {
        "documents": _nnz_stats(encoded_documents["matrix"]),
        "queries": _nnz_stats(query_matrix),
    }
    if manifest["nnz"] != expected_nnz or manifest["audit"]["document_chunks"] != encoded_documents["audit"]:
        raise BatchAError("Raw manifest CSR or document audit does not match replay")
    return manifest


def write_inventory(path: str | Path | None = None) -> dict[str, Any]:
    document = inventory_document()
    _validate_schema(document, "learned-sparse-inventory-v01.schema.json")
    if path is not None:
        _write_json(Path(path), document)
    return document


def validate_raw_results(
    result_roots: Iterable[str | Path],
    data_root: str | Path,
    expected_manifest_sha256es: Iterable[str] | None,
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    roots = tuple(Path(path) for path in result_roots)
    if expected_manifest_sha256es is None:
        raise BatchAError("Raw result consumption requires externally supplied manifest identities")
    identities = tuple(expected_manifest_sha256es)
    if len(identities) != len(roots):
        raise BatchAError("Each raw result requires one externally supplied manifest identity")
    manifests = [
        validate_raw_result(path, data_root, expected_manifest_sha256=identity)
        for path, identity in zip(roots, identities, strict=True)
    ]
    return roots, manifests


def write_failure_cases(
    result_roots: Iterable[str | Path],
    data_root: str | Path,
    output_path: str | Path,
    expected_manifest_sha256es: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write deterministic restricted-ID failure evidence without source text."""
    roots, manifests = validate_raw_results(
        result_roots, data_root, expected_manifest_sha256es
    )
    expected = {(model, track) for model in SELECTED_KEYS + ANCHOR_KEYS for track in TRACKS}
    actual = {(row["model"]["key"], row["data"]["track"]) for row in manifests}
    if actual != expected or len(manifests) != len(expected):
        raise BatchAError("Failure cases require exactly the eight fixed cells")
    root_by_cell = {
        (manifest["model"]["key"], manifest["data"]["track"]): root
        for root, manifest in zip(roots, manifests, strict=True)
    }
    manifest_by_cell = {
        (manifest["model"]["key"], manifest["data"]["track"]): manifest
        for manifest in manifests
    }
    cases = []
    for model_key in SELECTED_KEYS + ANCHOR_KEYS:
        for track in TRACKS:
            cell = (model_key, track)
            root = root_by_cell[cell]
            manifest = manifest_by_cell[cell]
            per_query = json.loads((root / "per_query_metrics.json").read_text(encoding="utf-8"))
            query_id = min(per_query, key=lambda item: (per_query[item]["ndcg@10"], item))
            rankings = {
                row["query_id"]: row["hits"]
                for row in json.loads((root / "rankings.json").read_text(encoding="utf-8"))
            }
            data = load_materialized(data_root, track)
            query_row = next(row for row in data.queries if row["id"] == query_id)
            cases.append(
                {
                    "model_key": model_key,
                    "track": track,
                    "raw_manifest_sha256": sha256_file(root / "manifest.json"),
                    "query_id": query_id,
                    "metrics": per_query[query_id],
                    "relevant_document_ids": sorted(data.qrels[query_id]),
                    "excluded_document_ids": sorted(query_row.get("excluded_ids", ())),
                    "hits": rankings[query_id],
                }
            )
    artifact = {
        "schema_version": "bright-learned-sparse-failure-cases-v1",
        "dataset_version": DATASET_VERSION,
        "publication": PUBLICATION,
        "selection_rule": FAILURE_SELECTION_RULE,
        "contains_restricted_ids": True,
        "contains_source_text": False,
        "cases": cases,
    }
    _validate_schema(artifact, "bright-learned-sparse-failure-cases-v01.schema.json")
    output = Path(output_path)
    _write_json(output, artifact)
    output.with_suffix(".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    return artifact


def _repo_relative(path: Path) -> str:
    repo_root = Path(__file__).parents[3].resolve()
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise BatchAError(f"Tracked file is outside the repository: {path}") from exc


def build_artifact_manifest(
    *,
    artifact_root: str | Path,
    report_path: str | Path,
    result_roots: Iterable[str | Path],
    data_root: str | Path,
    expected_manifest_sha256es: Iterable[str] | None = None,
    expected_existing_manifest_sha256: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Validate and bind the tracked package to all replayed raw identities."""
    root = Path(artifact_root)
    inventory_path = root / "inventory.json"
    summary_path = root / "summary.json"
    report = Path(report_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_schema(inventory, "learned-sparse-inventory-v01.schema.json")
    _validate_schema(summary, "bright-learned-sparse-summary-v01.schema.json")
    if inventory != inventory_document():
        raise BatchAError("Tracked inventory does not match the executable inventory")

    roots = tuple(Path(path) for path in result_roots)
    existing_manifest_path = root / "manifest.json"
    locked_by_cell: dict[tuple[str, str], str] = {}
    if existing_manifest_path.is_file():
        expected_package_identity = _require_sha256(
            expected_existing_manifest_sha256,
            label="existing package manifest SHA256",
        )
        if sha256_file(existing_manifest_path) != expected_package_identity:
            raise BatchAError("Existing package manifest does not match its external identity")
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        _validate_schema(existing_manifest, "bright-learned-sparse-artifact-manifest-v01.schema.json")
        locked_by_cell = {
            (row["model_key"], row["track"]): row["manifest_sha256"]
            for row in existing_manifest["raw_results"]
        }
    supplied = tuple(expected_manifest_sha256es) if expected_manifest_sha256es is not None else None
    if not locked_by_cell and supplied is None:
        raise BatchAError(
            "Initial finalization requires externally supplied raw manifest identities"
        )
    if supplied is not None and len(supplied) != len(roots):
        raise BatchAError("Each raw result requires one externally supplied manifest identity")
    raw_manifests = []
    for index, path in enumerate(roots):
        cell = _peek_raw_cell(path)
        locked_identity = locked_by_cell.get(cell)
        supplied_identity = supplied[index] if supplied is not None else None
        if locked_by_cell and locked_identity is None:
            raise BatchAError(f"Raw cell {cell} is not present in the finalized package")
        if locked_identity is not None and supplied_identity is not None and locked_identity != supplied_identity:
            raise BatchAError(f"Supplied raw identity conflicts with finalized package for {cell}")
        raw_manifests.append(
            validate_raw_result(
                path,
                data_root,
                expected_manifest_sha256=locked_identity or supplied_identity,
            )
        )
    if summary != tracked_aggregate(raw_manifests):
        raise BatchAError("Tracked summary does not match replayed raw cells")
    raw_by_cell = {
        (row["model"]["key"], row["data"]["track"]): (path, row)
        for path, row in zip(roots, raw_manifests, strict=True)
    }
    expected_cells = [
        (model, track) for model in SELECTED_KEYS + ANCHOR_KEYS for track in TRACKS
    ]
    if set(raw_by_cell) != set(expected_cells) or len(raw_by_cell) != len(expected_cells):
        raise BatchAError("Tracked manifest requires exactly the eight fixed raw cells")

    materialization = validate_materialization(data_root)
    data_path = Path(data_root)
    protocol = {
        "models": list(SELECTED_KEYS + ANCHOR_KEYS),
        "tracks": list(TRACKS),
        "dataset_version": DATASET_VERSION,
        "canonical_data_root": "data/bright-nontechnical-pilot-v0.2",
        "materialization_manifest_sha256": sha256_file(data_path / "manifest.json"),
        "source_revision": materialization["source"]["revision"],
        "device": "cuda:0",
        "batch_size": 8,
        "chunk_size": 256,
        "max_length": 512,
        "search": {
            "backend": "scipy_csr_exact",
            "exact": True,
            "top_k": TOP_K,
            "tie_break": "document_id_ascending",
        },
        "trust_remote_code": False,
    }
    protocol["identity_sha256"] = canonical_sha256(protocol)
    tracked_paths = (inventory_path, summary_path, report)
    manifest = {
        "schema_version": "bright-learned-sparse-artifact-manifest-v1",
        "dataset_version": DATASET_VERSION,
        "publication": PUBLICATION,
        "protocol": protocol,
        "tracked_files": {
            _repo_relative(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in tracked_paths
        },
        "raw_results": [
            {
                "model_key": model,
                "track": track,
                "manifest_sha256": sha256_file(raw_by_cell[(model, track)][0] / "manifest.json"),
                "sidecar_sha256": sha256_file(raw_by_cell[(model, track)][0] / "manifest.sha256"),
                "artifact_map_sha256": canonical_sha256(raw_by_cell[(model, track)][1]["artifacts"]),
            }
            for model, track in expected_cells
        ],
        "content_policy": (
            "Tracked files contain no source text, query or document identifiers, rankings, or per-query rows."
        ),
    }
    _validate_schema(manifest, "bright-learned-sparse-artifact-manifest-v01.schema.json")
    if write:
        _write_json(root / "manifest.json", manifest)
    return manifest


def validate_artifact_manifest(
    *,
    manifest_path: str | Path,
    result_roots: Iterable[str | Path],
    data_root: str | Path,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed on tracked file, protocol, raw identity, or replay drift."""
    path = Path(manifest_path)
    external_identity = _require_sha256(
        expected_manifest_sha256, label="package manifest SHA256"
    )
    if sha256_file(path) != external_identity:
        raise BatchAError("Package manifest does not match its external identity")
    observed = json.loads(path.read_text(encoding="utf-8"))
    _validate_schema(observed, "bright-learned-sparse-artifact-manifest-v01.schema.json")
    repo_root = Path(__file__).parents[3]
    for name, identity in observed["tracked_files"].items():
        tracked = repo_root / name
        if tracked.stat().st_size != identity["bytes"] or sha256_file(tracked) != identity["sha256"]:
            raise BatchAError(f"Tracked artifact identity mismatch: {name}")
    report_names = [name for name in observed["tracked_files"] if name.startswith("benchmark/research/")]
    if len(report_names) != 1:
        raise BatchAError("Tracked manifest must bind exactly one research report")
    rebuilt = build_artifact_manifest(
        artifact_root=path.parent,
        report_path=repo_root / report_names[0],
        result_roots=result_roots,
        data_root=data_root,
        expected_existing_manifest_sha256=external_identity,
        write=False,
    )
    if observed != rebuilt:
        raise BatchAError("Tracked artifact manifest does not match full replay")
    return observed


def resolve_selected(cache_root: str | Path | None = None, *, download: bool) -> dict[str, Any]:
    resolver = BoundedSnapshotResolver(cache_root=cache_root)
    plan = resolver.plan_batch()
    if not download:
        return plan
    resolved = [resolver.resolve(INVENTORY[row["key"]], plan=row) for row in plan["models"]]
    return {**plan, "models": resolved, "actual_bytes": sum(row["actual_bytes"] for row in resolved)}


__all__ = [
    "BatchAError",
    "DATASET_VERSION",
    "encode_document_chunks",
    "resolve_selected",
    "run_gate",
    "run_track",
    "tracked_aggregate",
    "build_artifact_manifest",
    "validate_artifact_manifest",
    "validate_raw_result",
    "validate_raw_results",
    "write_failure_cases",
    "write_inventory",
]
