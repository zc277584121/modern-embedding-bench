"""Pinned real late-interaction model inventory and bounded snapshot loading."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE

from mm_embed.providers.multi_vector_base import (
    MultiVectorBatch,
    MultiVectorRepresentation,
    MultiVectorResult,
    MultiVectorRole,
    MultiVectorRoute,
)

MIB = 1024**2
GIB = 1024**3
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SELECTED_KEYS = ("colbert-v2", "answerai-colbert-small", "gte-modern-colbert")
STORY_DOWNLOAD_CAP_BYTES = 2 * GIB
STORY_ARTIFACT_CAP_BYTES = 12 * GIB
MODEL_VRAM_CAP_BYTES = 11 * GIB


class MultiVectorSnapshotError(ValueError):
    """Raised when immutable model evidence violates the frozen policy."""


@dataclass(frozen=True)
class RealMultiVectorSpec:
    """One selected, deferred, or excluded late-interaction checkpoint."""

    key: str
    status: str
    reason: str
    repo_id: str
    revision: str
    license: str
    gated: bool
    private: bool
    lineage: str
    languages: tuple[str, ...]
    parameters: int | None
    model_weight_bytes: int
    estimated_snapshot_bytes: int
    download_cap_bytes: int | None
    backbone: str
    query_route: str
    document_route: str
    dimensions: int
    query_length: int
    document_length: int
    query_expansion: str
    mask_policy: str
    padding_policy: str
    trust_remote_code: bool
    remote_code_risk: str
    call_stack: tuple[str, ...]
    allowlist: tuple[str, ...]
    primary_sources: tuple[tuple[str, str, str], ...]


COLBERT_FILES = (
    "README.md",
    "artifact.metadata",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
GTE_FILES = (
    "1_Dense/config.json",
    "1_Dense/model.safetensors",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _specs() -> tuple[RealMultiVectorSpec, ...]:
    stack = (
        "sentence-transformers==6.0.1 MultiVectorEncoder",
        "transformers==5.3.0",
        "torch==2.10.0+cu128",
        "project exact MaxSim and max_passage aggregation",
    )
    return (
        RealMultiVectorSpec(
            "colbert-v2",
            "selected",
            "Canonical 110M BERT-base ColBERTv2 baseline with the original 128-dimensional late interaction.",
            "colbert-ir/colbertv2.0",
            "c1e84128e85ef755c096a95bdb06b47793b13acf",
            "MIT",
            False,
            False,
            "Official ColBERTv2 checkpoint trained with distillation on MS MARCO passages.",
            ("English",),
            110_000_000,
            438_349_816,
            439_062_205,
            512 * MIB,
            "BERT-base, 12 layers, hidden size 768",
            "encode_query with [Q]/[unused0] marker and fixed mask-token expansion",
            "encode_document with [D]/[unused1] marker",
            128,
            32,
            180,
            "fixed to 32 scoring positions; expansion masks are scored but not attended",
            "document punctuation is removed from the scoring mask; query expansion positions remain valid",
            "ragged scoring-mask compaction; any adapter padding is zero with an explicit boolean mask",
            False,
            "No Python files or auto_map; only safetensors and tokenizer/config metadata are allowlisted.",
            stack,
            COLBERT_FILES,
            (
                (
                    "hub_api",
                    "https://huggingface.co/api/models/colbert-ir/colbertv2.0/revision/c1e84128e85ef755c096a95bdb06b47793b13acf",
                    "immutable revision, MIT, ungated, file sizes",
                ),
                (
                    "model_card",
                    "https://huggingface.co/colbert-ir/colbertv2.0/raw/c1e84128e85ef755c096a95bdb06b47793b13acf/README.md",
                    "44e68292037d4fac7db91b9b24ae36535ae5807eef2b323adcc83a615b91751d",
                ),
                (
                    "artifact_metadata",
                    "https://huggingface.co/colbert-ir/colbertv2.0/raw/c1e84128e85ef755c096a95bdb06b47793b13acf/artifact.metadata",
                    "0ddc5a54234cff6d13bc9411250a5479d9d96f3ffbace76d8a1884144377e434",
                ),
            ),
        ),
        RealMultiVectorSpec(
            "answerai-colbert-small",
            "selected",
            "A 33M narrow BERT checkpoint using the JaColBERTv2.5 recipe and a smaller 96-dimensional index.",
            "answerdotai/answerai-colbert-small-v1",
            "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e",
            "Apache-2.0",
            False,
            False,
            "Answer.AI English small ColBERT trained with a JaColBERTv2.5-derived recipe.",
            ("English",),
            33_000_000,
            133_610_664,
            134_567_234,
            160 * MIB,
            "narrow BERT, 12 layers, hidden size 384",
            "encode_query with [Q]/[unused0] marker and fixed mask-token expansion",
            "encode_document with [D]/[unused1] marker",
            96,
            32,
            300,
            "fixed to 32 scoring positions; expansion masks are scored but not attended",
            "document punctuation is removed from the scoring mask; query expansion positions remain valid",
            "ragged scoring-mask compaction; any adapter padding is zero with an explicit boolean mask",
            False,
            "No Python files or auto_map; only safetensors and tokenizer/config metadata are allowlisted.",
            stack,
            COLBERT_FILES,
            (
                (
                    "hub_api",
                    "https://huggingface.co/api/models/answerdotai/answerai-colbert-small-v1/revision/934fa8bb4ce2284f4c2baa232d81aca4d076fa5e",
                    "immutable revision, Apache-2.0, ungated, file sizes",
                ),
                (
                    "model_card",
                    "https://huggingface.co/answerdotai/answerai-colbert-small-v1/raw/934fa8bb4ce2284f4c2baa232d81aca4d076fa5e/README.md",
                    "5852bad161ce37c95108ffd1b1d862abdff536f2c9efb66bdc37db688571a7a1",
                ),
                (
                    "artifact_metadata",
                    "https://huggingface.co/answerdotai/answerai-colbert-small-v1/raw/934fa8bb4ce2284f4c2baa232d81aca4d076fa5e/artifact.metadata",
                    "f09c7e00cde4fda0b272b47e07d537d7b698756d0d5ffcf8b82ec17553d1b4e4",
                ),
            ),
        ),
        RealMultiVectorSpec(
            "gte-modern-colbert",
            "selected",
            "A larger ModernBERT checkpoint with knowledge distillation, native ragged queries, and 128-dimensional output.",
            "lightonai/GTE-ModernColBERT-v1",
            "25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639",
            "Apache-2.0",
            False,
            False,
            "PyLate checkpoint based on Alibaba-NLP/gte-modernbert-base and distilled on MS MARCO.",
            ("English",),
            149_000_000,
            596_469_584,
            600_484_363,
            700 * MIB,
            "ModernBERT-base, 22 layers, hidden size 768",
            "encode_query with saved query route and no mask-token expansion",
            "encode_document with saved document route",
            128,
            48,
            300,
            "disabled; queries retain only the saved scoring-mask positions up to 48 tokens",
            "document punctuation is removed from the scoring mask; ordinary attention padding is excluded",
            "ragged scoring-mask compaction; any adapter padding is zero with an explicit boolean mask",
            False,
            "No Python files or auto_map; standard ModernBERT plus saved Sentence Transformers modules.",
            stack,
            GTE_FILES,
            (
                (
                    "hub_api",
                    "https://huggingface.co/api/models/lightonai/GTE-ModernColBERT-v1/revision/25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639",
                    "immutable revision, Apache-2.0, ungated, file sizes",
                ),
                (
                    "model_card",
                    "https://huggingface.co/lightonai/GTE-ModernColBERT-v1/raw/25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639/README.md",
                    "8bac6a7e2d6c2de129f643aeb75e6f19e350d6c180e50362d86c0212a8c07b6c",
                ),
                (
                    "saved_config",
                    "https://huggingface.co/lightonai/GTE-ModernColBERT-v1/raw/25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639/config_sentence_transformers.json",
                    "edbe6fd9b4ef756645baf9c979c3d8fe4307351fd8c1963018d0457deaedfed1",
                ),
            ),
        ),
        RealMultiVectorSpec(
            "mxbai-edge-colbert-17m",
            "deferred",
            "The 17M/48D edge checkpoint is a useful later resource lower bound, but overlaps the selected lightweight role and its 32K card claim exceeds the pinned 7,999-position config.",
            "mixedbread-ai/mxbai-edge-colbert-v0-17m",
            "592c6417c1c6687572043408ed1ae5196bce16b1",
            "Apache-2.0",
            False,
            False,
            "Mixedbread edge ColBERT based on the 17M Ettin ModernBERT family.",
            ("English",),
            17_000_000,
            67_818_744,
            71_500_000,
            None,
            "ModernBERT-style Ettin-17M, 7 layers, hidden size 256",
            "encode_query",
            "encode_document",
            48,
            48,
            512,
            "disabled",
            "saved multi-vector scoring mask",
            "ragged",
            False,
            "No repository Python or auto_map at the pinned revision.",
            stack,
            (),
            (
                (
                    "hub_api",
                    "https://huggingface.co/api/models/mixedbread-ai/mxbai-edge-colbert-v0-17m/revision/592c6417c1c6687572043408ed1ae5196bce16b1",
                    "immutable metadata",
                ),
            ),
        ),
        RealMultiVectorSpec(
            "jina-colbert-v2",
            "excluded",
            "The pinned config delegates through auto_map and the primary usage requires trust_remote_code=True; the default fail-closed policy excludes it.",
            "jinaai/jina-colbert-v2",
            "a9dc5cd7293d4c71dbbba04829923ba4d0e4f6ea",
            "CC-BY-NC-4.0",
            False,
            False,
            "Jina multilingual XLM-RoBERTa-flash late-interaction checkpoint.",
            ("Multilingual",),
            560_000_000,
            1_119_027_888,
            1_120_000_000,
            None,
            "24-layer XLM-RoBERTa-flash, hidden size 1024",
            "custom query marker",
            "custom document marker",
            128,
            32,
            8_192,
            "model-specific",
            "model-specific",
            "ragged",
            True,
            "Pinned auto_map references jinaai/xlm-roberta-flash-implementation and the model card explicitly requests trust_remote_code=True.",
            stack,
            (),
            (
                (
                    "model_card",
                    "https://huggingface.co/jinaai/jina-colbert-v2/raw/a9dc5cd7293d4c71dbbba04829923ba4d0e4f6ea/README.md",
                    "c2f5c279bba7b5ec7a8a9bd591564b3ee90ce4a7cd97f0b66d03779c433f817b",
                ),
            ),
        ),
        RealMultiVectorSpec(
            "lighton-colbert-v2-wrapper",
            "excluded",
            "This is a Sentence Transformers/PyLate conversion of the selected colbert-ir checkpoint, not an independent checkpoint identity.",
            "lightonai/colbertv2.0",
            "0b9b1e380096a3b72b6b4877f68d59effb62e0f2",
            "MIT",
            False,
            False,
            "Converted wrapper whose model card declares colbert-ir/colbertv2.0 as its fine-tuned base.",
            ("English",),
            110_000_000,
            438_344_632,
            439_297_945,
            None,
            "BERT-base ColBERTv2 conversion",
            "encode_query",
            "encode_document",
            128,
            32,
            180,
            "fixed",
            "punctuation skiplist",
            "ragged",
            False,
            "No remote code, but checkpoint-lineage duplication makes it ineligible.",
            stack,
            (),
            (
                (
                    "hub_api",
                    "https://huggingface.co/api/models/lightonai/colbertv2.0/revision/0b9b1e380096a3b72b6b4877f68d59effb62e0f2",
                    "base-model lineage metadata",
                ),
            ),
        ),
    )


INVENTORY = {spec.key: spec for spec in _specs()}


def file_sha256(path: Path) -> str:
    """Hash a file without loading a model-sized blob into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_document() -> dict[str, Any]:
    """Return the deterministic public-safe inventory."""
    rows = []
    for spec in INVENTORY.values():
        row = asdict(spec)
        row["primary_sources"] = [
            {"kind": kind, "url": url, "evidence": evidence} for kind, url, evidence in spec.primary_sources
        ]
        rows.append(row)
    return {
        "schema_version": "bright-real-multi-vector-inventory-v1",
        "selected_keys": list(SELECTED_KEYS),
        "policy": {
            "fixed_revision": "40-character lowercase hexadecimal commit",
            "trust_remote_code": False,
            "weights": "allowlisted safetensors only",
            "story_download_cap_bytes": STORY_DOWNLOAD_CAP_BYTES,
            "story_artifact_cap_bytes": STORY_ARTIFACT_CAP_BYTES,
            "model_vram_cap_bytes": MODEL_VRAM_CAP_BYTES,
            "publication_gate": "closed",
        },
        "models": rows,
    }


def _license(info: Any) -> str | None:
    card = getattr(info, "card_data", None)
    if hasattr(card, "to_dict"):
        card = card.to_dict()
    return card.get("license") if isinstance(card, dict) else None


def validate_hub_metadata(key: str, *, api: Any | None = None) -> dict[str, Any]:
    """Validate immutable primary-source metadata without loading weights."""
    spec = INVENTORY[key]
    if spec.status != "selected":
        raise MultiVectorSnapshotError("Only selected checkpoints may pass the execution preflight")
    if not REVISION_PATTERN.fullmatch(spec.revision):
        raise MultiVectorSnapshotError("Model revision is not an immutable 40-character commit")
    info = (api or HfApi()).model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
    if str(info.sha) != spec.revision or bool(info.private) != spec.private or bool(info.gated) != spec.gated:
        raise MultiVectorSnapshotError("Hub identity, visibility, or gated state drifted")
    observed_license = _license(info)
    expected_license = spec.license.lower().replace("-", "")
    if not observed_license or observed_license.lower().replace("-", "") != expected_license:
        raise MultiVectorSnapshotError(f"Hub license drifted for {key}: {observed_license!r}")
    siblings = {item.rfilename: item for item in info.siblings}
    missing = sorted(set(spec.allowlist) - set(siblings))
    if missing:
        raise MultiVectorSnapshotError(f"Pinned snapshot files are missing for {key}: {missing}")
    python_files = sorted(path for path in siblings if path.endswith(".py"))
    selected_bytes = sum(int(siblings[path].size or 0) for path in spec.allowlist)
    if python_files or selected_bytes > int(spec.download_cap_bytes or 0):
        raise MultiVectorSnapshotError("Remote Python or the download cap violates the frozen policy")
    return {
        "key": key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "license": observed_license,
        "gated": bool(info.gated),
        "private": bool(info.private),
        "allowlisted_bytes": selected_bytes,
        "repository_files": len(siblings),
        "repository_python_files": python_files,
    }


def resolve_snapshot(key: str, *, cache_dir: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Download only the frozen files and return a content-addressed snapshot identity."""
    spec = INVENTORY[key]
    validate_hub_metadata(key)
    cached_before = 0
    for relative in spec.allowlist:
        cached = try_to_load_from_cache(
            spec.repo_id,
            relative,
            revision=spec.revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        if isinstance(cached, str) and Path(cached).is_file():
            cached_before += Path(cached).stat().st_size
    target = Path(
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            allow_patterns=list(spec.allowlist),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    files: dict[str, dict[str, Any]] = {}
    total = 0
    for relative in spec.allowlist:
        path = target / relative
        if not path.is_file():
            raise MultiVectorSnapshotError(f"Allowlisted snapshot file is missing: {relative}")
        size = path.stat().st_size
        files[relative] = {"bytes": size, "sha256": file_sha256(path)}
        total += size
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    if config.get("auto_map"):
        raise MultiVectorSnapshotError("Selected snapshot unexpectedly requires remote code")
    if total > int(spec.download_cap_bytes or 0):
        raise MultiVectorSnapshotError("Resolved snapshot exceeds its frozen cap")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return target, {
        "snapshot_files": files,
        "snapshot_bytes": total,
        "cached_before_bytes": cached_before,
        "incremental_download_bytes": total - cached_before,
        "snapshot_identity_sha256": hashlib.sha256(canonical).hexdigest(),
        "cache_root": str(Path(cache_dir or HF_HUB_CACHE).resolve()),
    }


class SentenceTransformerMultiVectorProvider:
    """Real MultiVectorEncoder adapter with explicit padded mask output."""

    name = "sentence_transformers_multi_vector"
    query_route = MultiVectorRoute.QUERY
    document_route = MultiVectorRoute.DOCUMENT

    def __init__(
        self,
        key: str,
        snapshot_path: str | Path,
        *,
        device: str = "cuda:0",
        batch_size: int = 32,
    ) -> None:
        if key not in SELECTED_KEYS:
            raise MultiVectorSnapshotError("Provider model is outside the frozen selected set")
        from sentence_transformers import MultiVectorEncoder

        self.spec = INVENTORY[key]
        self.model = self.spec.repo_id
        self.revision = self.spec.revision
        self.device = device
        self.batch_size = batch_size
        self.representation = MultiVectorRepresentation(
            f"{self.spec.repo_id}@{self.spec.revision}:native-scoring-mask-fp32",
            self.spec.dimensions,
        )
        self.encoder = MultiVectorEncoder(
            str(Path(snapshot_path).resolve()),
            device=device,
            trust_remote_code=False,
            local_files_only=True,
            similarity_fn_name="maxsim",
        )
        self._validate_loaded_semantics()

    def _validate_loaded_semantics(self) -> None:
        transformer = self.encoder[0]
        expansion = getattr(transformer, "query_expansion", None)
        effective_query_length = transformer.query_length
        if effective_query_length is None and expansion is not None:
            effective_query_length = expansion["length"]
        observed = {
            "query_length": int(effective_query_length),
            "document_length": int(transformer.document_length),
            "dimension": int(self.encoder.get_embedding_dimension()),
        }
        expected = {
            "query_length": self.spec.query_length,
            "document_length": self.spec.document_length,
            "dimension": self.spec.dimensions,
        }
        if observed != expected:
            raise MultiVectorSnapshotError(f"Loaded multi-vector semantics drifted: {observed} != {expected}")
        if getattr(self.encoder, "similarity_fn_name", None) != "maxsim":
            raise MultiVectorSnapshotError("Loaded model is not using exact MaxSim semantics")

    @property
    def semantic_evidence(self) -> dict[str, Any]:
        """Return public-safe loaded route and mask evidence."""
        transformer = self.encoder[0]
        expansion = getattr(transformer, "query_expansion", None)
        effective_query_length = transformer.query_length
        if effective_query_length is None and expansion is not None:
            effective_query_length = expansion["length"]
        masks = [
            {
                "skiplist_words": list(getattr(module, "skiplist_words", [])),
                "skiplist_tasks": list(getattr(module, "skiplist_tasks", [])),
            }
            for module in self.encoder
            if module.__class__.__name__ == "MultiVectorMask"
        ]
        return {
            "query_length": int(effective_query_length),
            "document_length": int(transformer.document_length),
            "query_expansion": expansion,
            "mask_modules": masks,
            "dimension": int(self.encoder.get_embedding_dimension()),
            "similarity": str(self.encoder.similarity_fn_name),
            "prompts": dict(self.encoder.prompts),
            "trust_remote_code": False,
        }

    def encode_compact(
        self,
        texts: Sequence[str],
        *,
        role: MultiVectorRole,
        batch_size: int | None = None,
    ) -> tuple[list[np.ndarray], float, int | None]:
        """Encode to compact float32 arrays after the model's final scoring mask."""
        import torch

        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("Real multi-vector encoding requires non-empty text")
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        method = self.encoder.encode_query if role is MultiVectorRole.QUERY else self.encoder.encode_document
        arrays = method(
            list(texts),
            batch_size=batch_size or self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        normalized: list[np.ndarray] = []
        for array in arrays:
            value = np.asarray(array, dtype=np.float32)
            if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != self.spec.dimensions:
                raise MultiVectorSnapshotError("Encoded token-vector shape violates the frozen contract")
            if not np.all(np.isfinite(value)):
                raise MultiVectorSnapshotError("Encoded token vectors contain non-finite values")
            normalized.append(np.ascontiguousarray(value))
        peak = int(torch.cuda.max_memory_allocated(self.device)) if self.device.startswith("cuda") else None
        if peak is not None and peak > MODEL_VRAM_CAP_BYTES:
            raise MultiVectorSnapshotError("Observed model VRAM exceeded the frozen cap")
        return normalized, elapsed_ms, peak

    def _pack(
        self,
        arrays: Sequence[np.ndarray],
        *,
        item_ids: Sequence[str],
        passage_ids: Sequence[str],
        document_ids: Sequence[str],
        role: MultiVectorRole,
        latency_ms: float,
        peak_vram_bytes: int | None,
    ) -> MultiVectorResult:
        if not (len(arrays) == len(item_ids) == len(passage_ids) == len(document_ids)):
            raise ValueError("Encoded arrays and identities are misaligned")
        width = max(array.shape[0] for array in arrays)
        values = np.zeros((len(arrays), width, self.spec.dimensions), dtype=np.float32)
        mask = np.zeros((len(arrays), width), dtype=np.bool_)
        for index, array in enumerate(arrays):
            values[index, : array.shape[0]] = array
            mask[index, : array.shape[0]] = True
        return MultiVectorResult(
            embeddings=MultiVectorBatch(
                values,
                mask,
                tuple(item_ids),
                tuple(passage_ids),
                tuple(document_ids),
                self.representation,
            ),
            role=role,
            route=self.query_route if role is MultiVectorRole.QUERY else self.document_route,
            provider=self.name,
            model_name=self.model,
            model_revision=self.revision,
            latency_ms=latency_ms,
            peak_vram_bytes=peak_vram_bytes,
        )

    def encode_multi_vector_queries(self, texts: Sequence[str], *, item_ids: Sequence[str]) -> MultiVectorResult:
        arrays, latency, peak = self.encode_compact(texts, role=MultiVectorRole.QUERY)
        return self._pack(
            arrays,
            item_ids=item_ids,
            passage_ids=item_ids,
            document_ids=item_ids,
            role=MultiVectorRole.QUERY,
            latency_ms=latency,
            peak_vram_bytes=peak,
        )

    def encode_multi_vector_query(self, text: str, *, item_id: str) -> MultiVectorResult:
        return self.encode_multi_vector_queries([text], item_ids=[item_id])

    def encode_multi_vector_passages(
        self,
        texts: Sequence[str],
        *,
        passage_ids: Sequence[str],
        document_ids: Sequence[str],
    ) -> MultiVectorResult:
        arrays, latency, peak = self.encode_compact(texts, role=MultiVectorRole.DOCUMENT)
        return self._pack(
            arrays,
            item_ids=passage_ids,
            passage_ids=passage_ids,
            document_ids=document_ids,
            role=MultiVectorRole.DOCUMENT,
            latency_ms=latency,
            peak_vram_bytes=peak,
        )
