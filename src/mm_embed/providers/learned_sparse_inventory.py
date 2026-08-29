"""Pinned learned-sparse inventory and bounded Hugging Face snapshot resolver."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from mm_embed.providers.snapshot_identity import snapshot_identity, verify_snapshot_identity

MIB = 1024**2
GIB = 1024**3
BATCH_DOWNLOAD_CAP_BYTES = 512 * MIB
STORY_DISK_CAP_BYTES = 4 * GIB
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PICKLE_SUFFIXES = frozenset({".bin", ".ckpt", ".pickle", ".pkl", ".pt", ".pth"})


class SnapshotPolicyError(ValueError):
    """Raised when a repository or local snapshot violates the bounded policy."""


@dataclass(frozen=True)
class LearnedSparseSpec:
    """One reviewable learned-sparse candidate or benchmark anchor."""

    key: str
    status: str
    reason: str
    repo_id: str
    revision: str
    license: str
    gated: bool | str
    adapter: str
    query_route: str
    document_route: str
    vocabulary_id: str
    dimensions: int
    max_length: int
    query_pruning: int | None
    document_pruning: int | None
    dependencies: tuple[str, ...]
    download_cap_bytes: int | None
    estimated_snapshot_bytes: int | None
    trust_remote_code: bool
    remote_code_risk: str
    allowlist: tuple[str, ...] = ()
    expected_identity: tuple[tuple[str, str], ...] = ()

    @property
    def identity(self) -> dict[str, str]:
        return dict(self.expected_identity)


GRANITE_ALLOWLIST = (
    "1_SpladePooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "merges.txt",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

OPENSEARCH_MINI_ALLOWLIST = (
    "config.json",
    "config_sentence_transformers.json",
    "document_1_SpladePooling/config.json",
    "idf.json",
    "model.safetensors",
    "modules.json",
    "query_0_SparseStaticEmbedding/config.json",
    "query_0_SparseStaticEmbedding/model.safetensors",
    "query_0_SparseStaticEmbedding/special_tokens_map.json",
    "query_0_SparseStaticEmbedding/tokenizer.json",
    "query_0_SparseStaticEmbedding/tokenizer_config.json",
    "query_0_SparseStaticEmbedding/vocab.txt",
    "query_token_weights.txt",
    "router_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)

GRANITE_30M_SPARSE_IDENTITY = (
    ("1_SpladePooling/config.json", "07580f9ee2553c16b418853be6811a7b0abd7393ee96c487f32b6b0ba1a5e6b4"),
    ("config.json", "6a5eb43349f9708c99940b5c297d87708746190ba723fb2a7160629cf27b3c4c"),
    ("config_sentence_transformers.json", "44e5b5295415281c3f61341f0670bf6752ce1d85880380faeebae3752889d718"),
    ("merges.txt", "fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862"),
    ("model.safetensors", "08632803138e24c4ba36ca59e76a78a8cc1a21f29257c5fd87461a3d912e1a66"),
    ("modules.json", "49d17c5888db1d0f18dde2855770b5f796240df89ddf56f8589380072479e1e3"),
    ("sentence_bert_config.json", "948201d8329907aae938fa62f9ceeed53f5694dacc2b87b9f3b78b37ee986529"),
    ("special_tokens_map.json", "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d"),
    ("tokenizer.json", "33465117406b9007673e8ba283f7f1383d9b5094df947481af60eec94ed7d7bd"),
    ("tokenizer_config.json", "37d6615d67cc6d6ca861957eafd9edee2bcec5c5b3301c4acb629bde1925f2cb"),
    ("vocab.json", "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e"),
)

OPENSEARCH_DOC_V2_MINI_IDENTITY = (
    ("config.json", "5324b88812089731800d6f0e397690f91840ac4c24521360181d23002db5a677"),
    ("config_sentence_transformers.json", "44e5b5295415281c3f61341f0670bf6752ce1d85880380faeebae3752889d718"),
    ("document_1_SpladePooling/config.json", "07580f9ee2553c16b418853be6811a7b0abd7393ee96c487f32b6b0ba1a5e6b4"),
    ("idf.json", "da23a1c0b9252776cc8c6d70fd14723e218f484d489cd9027ac6e4065d5b9edd"),
    ("model.safetensors", "d90a4233e0ca352cf73d6c3adf3c41fac7fee9a49ccd9ee3e806a86c1e05368a"),
    ("modules.json", "2e12f8a5fbc625578d7bbaee4c18b748ccd82f0d4549c3d952638693ce4058cf"),
    ("query_0_SparseStaticEmbedding/config.json", "f1aa3269d4139c461e2d3b7b8f1570f507bc59529cffb004c6a3702da4f5087a"),
    ("query_0_SparseStaticEmbedding/model.safetensors", "711ec64837a7962d2ae106996079782b7ee87860089a0b2348bf7cb840f252d3"),
    ("query_0_SparseStaticEmbedding/special_tokens_map.json", "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a"),
    ("query_0_SparseStaticEmbedding/tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    ("query_0_SparseStaticEmbedding/tokenizer_config.json", "bf9431564f2867411d0c5ec40cfaa1d73275a9847dbf34f23c8961b1f2981cc4"),
    ("query_0_SparseStaticEmbedding/vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    ("query_token_weights.txt", "79c55cf7c5c7e1680d6332225aa58dda9b71445cf02e21b348b97c3ee5c81f07"),
    ("router_config.json", "c373414574b65ea6612a43c16a17d8a5942e91ccde1cd1433e41619af02a8264"),
    ("special_tokens_map.json", "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
    ("tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    ("tokenizer_config.json", "ae83fa6ca0333117ff12606020af925d648667ef70d92ff7f27d781ba0ca4544"),
    ("vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
)


def _specs() -> tuple[LearnedSparseSpec, ...]:
    common = (
        "huggingface-hub>=0.30",
        "sentence-transformers>=5.2",
        "torch>=2.0",
        "transformers>=4.57",
        "scipy>=1.12",
    )
    return (
        LearnedSparseSpec(
            "granite-30m-sparse", "selected", "Small neural/neural SPLADE complement to the static-query anchor.",
            "ibm-granite/granite-embedding-30m-sparse", "ad82b1fd09541c998c8d45045d601c51fdb8a9b7",
            "Apache-2.0", False, "sentence_transformers_sparse", "neural", "neural",
            "roberta-bpe-50265", 50265, 512, 50, 192, common, 80 * MIB, 63_318_503, False,
            "Standard RobertaForMaskedLM plus Sentence Transformers SpladePooling; repository utility Python is excluded.",
            GRANITE_ALLOWLIST, GRANITE_30M_SPARSE_IDENTITY,
        ),
        LearnedSparseSpec(
            "opensearch-doc-v2-mini", "selected", "Static query lookup plus document expansion with a small 30,522-term vocabulary.",
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
            "4af867a426867dfdd744097531046f4289a32fdd", "Apache-2.0", False,
            "sentence_transformers_sparse", "static_lookup", "document_expansion", "bert-wordpiece-30522",
            30522, 512, None, None, common, 128 * MIB, 94_631_975, False,
            "Standard BertForMaskedLM, Router, SparseStaticEmbedding, and SpladePooling; no auto_map.",
            OPENSEARCH_MINI_ALLOWLIST, OPENSEARCH_DOC_V2_MINI_IDENTITY,
        ),
        LearnedSparseSpec(
            "opensearch-doc-v3", "anchor", "Existing formal learned-sparse anchor; retained for paired Batch-A comparisons.",
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
            "babf71f3c48695e2e53a978208e8aba48335e3c0", "Apache-2.0", False,
            "opensearch_neural_sparse", "static_lookup", "document_expansion", "sentencepiece-wordpiece-vocab-30522",
            30522, 512, None, None, common, 300 * MIB, 259 * MIB, False,
            "Existing pinned local adapter and frozen behavior identity.",
        ),
        LearnedSparseSpec(
            "bge-m3", "anchor", "Existing contextual lexical anchor with a much larger multilingual coordinate space.",
            "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", "MIT", False,
            "bge_m3_sparse", "neural", "neural", "bge-m3-xlm-roberta-vocab-250002", 250002, 512,
            None, None, ("torch>=2.0", "transformers>=4.57", "scipy>=1.12"), 3 * GIB, 3 * GIB, False,
            "Existing weights-only custom sparse head and frozen behavior identity.",
        ),
        LearnedSparseSpec(
            "opensearch-doc-v2-distill", "deferred", "Mechanistically redundant after mini; useful only as a capacity control.",
            "opensearch-project/opensearch-neural-sparse-encoding-v2-distill",
            "269e6638b2c4f648996691f6d751495285d8f330", "Apache-2.0", False,
            "sentence_transformers_sparse", "neural", "neural", "bert-wordpiece-30522", 30522, 512,
            None, None, common, 300 * MIB, 268 * MIB, False, "Standard model files; not needed for Batch A.",
        ),
        LearnedSparseSpec(
            "opensearch-multilingual", "deferred", "No fixed multilingual formal track in this Story.",
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
            "1e0f096c2b51c234f1d20725c793e1b5b6d556db", "Apache-2.0", False,
            "sentence_transformers_sparse", "static_lookup", "document_expansion", "multilingual-wordpiece-105879",
            105879, 512, None, None, common, 768 * MIB, 670 * MIB, False, "Standard model files; track mismatch.",
        ),
        LearnedSparseSpec(
            "splade-v3-distilbert", "deferred", "Non-commercial share-alike license needs a separate publication decision.",
            "naver/splade-v3-distilbert", "2db06b86d65e316e2ca9907aa1aa8be6f8c4e739",
            "CC-BY-NC-SA-4.0", False, "sentence_transformers_sparse", "neural", "neural",
            "bert-wordpiece-30522", 30522, 512, None, None, common, None, None, False,
            "Standard architecture, but license is outside the Apache/MIT Batch-A publication posture.",
        ),
        LearnedSparseSpec(
            "splade-v3", "excluded", "Gated access and non-commercial license are outside this bounded batch.",
            "naver/splade-v3", "fdfeceb91d7b9de7985b38addd3ba9f53a59a355", "CC-BY-NC-SA-4.0", "auto",
            "sentence_transformers_sparse", "neural", "neural", "bert-wordpiece-30522", 30522, 512,
            None, None, common, None, None, False, "Gated repository.",
        ),
        LearnedSparseSpec(
            "splade-pp-cocondenser-ensemble", "excluded", "Non-commercial license and redundant SPLADE mechanism.",
            "naver/splade-cocondenser-ensembledistil", "49cf4c7b0db5b870a401ddf5e2669993ef3699c7",
            "CC-BY-NC-SA-4.0", False, "sentence_transformers_sparse", "neural", "neural",
            "bert-wordpiece-30522", 30522, 512, None, None, common, None, None, False,
            "Standard architecture; licensing excludes it from Batch A.",
        ),
        LearnedSparseSpec(
            "splade-v2-distil",
            "excluded",
            "Legacy SPLADE v2 is mechanistically redundant and its non-commercial license is outside Batch A.",
            "naver/splade_v2_distil",
            "b5e5ebfb992c31a99ba3be148fb9d841f5c6d6a1",
            "CC-BY-NC-SA-4.0",
            False,
            "transformers_sparse_legacy",
            "neural",
            "neural",
            "bert-wordpiece-30522",
            30522,
            512,
            None,
            None,
            ("torch>=2.0", "transformers>=4.57", "scipy>=1.12"),
            None,
            None,
            False,
            "Pinned standard BERT masked-language-model files; legacy integration and license add no Batch-A value.",
        ),
        LearnedSparseSpec(
            "splade-mini", "deferred", "Hub metadata says MIT, but repository-level license evidence remains incomplete.",
            "rasyosef/splade-mini", "98cb9db1eb2af3399c0bee30a6edca3fe9c0c057", "MIT (Hub metadata only)", False,
            "sentence_transformers_sparse", "neural", "neural", "bert-wordpiece-30522", 30522, 512,
            None, None, common, 64 * MIB, 45_800_000, False, "Standard architecture; license evidence is not sufficient.",
        ),
        LearnedSparseSpec(
            "splade-tiny", "deferred", "Hub metadata says MIT, but repository-level license evidence remains incomplete.",
            "rasyosef/splade-tiny", "7391972eac4411e33efff5fad27b886ec97895c0", "MIT (Hub metadata only)", False,
            "sentence_transformers_sparse", "neural", "neural", "bert-wordpiece-30522", 30522, 512,
            None, None, common, 32 * MIB, 18_600_000, False, "Standard architecture; license evidence is not sufficient.",
        ),
        LearnedSparseSpec(
            "gte-multilingual-sparse", "excluded", "Pinned config delegates to external code and requires trust_remote_code=True.",
            "Alibaba-NLP/gte-multilingual-base", "9bbca17d9273fd0d03d5725c7a4b0f6b45142062", "Apache-2.0", False,
            "unsupported_remote_code", "neural", "neural", "xlm-roberta-vocab-250048", 250048, 8192,
            None, None, ("transformers", "torch"), None, 599 * MIB, True,
            "Non-empty auto_map points to Alibaba-NLP/new-impl; forbidden by Batch-A policy.",
        ),
    )


INVENTORY = {spec.key: spec for spec in _specs()}
SELECTED_KEYS = ("granite-30m-sparse", "opensearch-doc-v2-mini")
ANCHOR_KEYS = ("opensearch-doc-v3", "bge-m3")


def inventory_document() -> dict[str, Any]:
    """Return a deterministic, executable inventory document."""
    rows = []
    for spec in INVENTORY.values():
        row = json.loads(json.dumps(asdict(spec)))
        row["expected_identity"] = dict(spec.expected_identity)
        row["revision_evidence"] = {
            "kind": "huggingface_immutable_tree",
            "url": f"https://huggingface.co/{spec.repo_id}/tree/{spec.revision}",
        }
        rows.append(row)
    return {
        "schema_version": "learned-sparse-inventory-v1",
        "policy": {
            "batch_download_cap_bytes": BATCH_DOWNLOAD_CAP_BYTES,
            "story_disk_cap_bytes": STORY_DISK_CAP_BYTES,
            "trust_remote_code": False,
            "revision": "40-character lowercase hexadecimal commit",
            "weights": "safetensors only for newly resolved snapshots",
        },
        "models": rows,
    }


def aggregate_snapshot_identity(identity: dict[str, str]) -> str:
    """Hash the canonical path-to-content-hash map."""
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contains_auto_map(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("auto_map"):
            return True
        return any(_contains_auto_map(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_auto_map(item) for item in value)
    return False


def standard_hub_cache_dir(cache_root: str | Path | None = None) -> Path:
    """Resolve the standard hub cache, optionally from an explicit Hugging Face home."""
    if cache_root is None:
        return Path(HF_HUB_CACHE).expanduser().resolve()
    return Path(cache_root).expanduser().resolve() / "hub"


class BoundedSnapshotResolver:
    """Dry-run, download, and verify exact allowlisted snapshots."""

    def __init__(self, *, cache_root: str | Path | None = None) -> None:
        if cache_root is None:
            self.cache_root = None
            self.hub_cache_dir = standard_hub_cache_dir()
            self.cache_source = "HF_HUB_CACHE"
        else:
            self.cache_root = Path(cache_root).expanduser().resolve()
            self.hub_cache_dir = standard_hub_cache_dir(self.cache_root)
            self.cache_source = "explicit_cache_root"

    def cache_audit(self) -> dict[str, str | None]:
        """Return unambiguous Hugging Face home and hub cache paths."""
        return {
            "source": self.cache_source,
            "cache_root": str(self.cache_root) if self.cache_root else None,
            "hub_cache_dir": str(self.hub_cache_dir),
        }

    @staticmethod
    def _validate_spec(spec: LearnedSparseSpec) -> None:
        if not REVISION_PATTERN.fullmatch(spec.revision):
            raise SnapshotPolicyError(f"{spec.key} revision must be a pinned 40-character commit")
        if spec.trust_remote_code:
            raise SnapshotPolicyError(f"{spec.key} requires forbidden remote code")
        if not spec.allowlist or spec.download_cap_bytes is None:
            raise SnapshotPolicyError(f"{spec.key} has no bounded snapshot contract")
        if len(set(spec.allowlist)) != len(spec.allowlist):
            raise SnapshotPolicyError(f"{spec.key} allowlist contains duplicates")
        for name in spec.allowlist:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise SnapshotPolicyError(f"{spec.key} has an unsafe allowlist path: {name}")
            suffix = path.suffix.lower()
            if suffix == ".py":
                raise SnapshotPolicyError(f"{spec.key} allowlist contains Python code: {name}")
            if suffix in PICKLE_SUFFIXES:
                raise SnapshotPolicyError(f"{spec.key} allowlist contains pickle-compatible weights: {name}")
        if not any(name.endswith(".safetensors") for name in spec.allowlist):
            raise SnapshotPolicyError(f"{spec.key} has no allowlisted safetensors weights")

    def plan(self, spec: LearnedSparseSpec) -> dict[str, Any]:
        """Resolve the exact remote file set and compute full and incremental bytes."""
        self._validate_spec(spec)
        rows = snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            cache_dir=self.hub_cache_dir,
            allow_patterns=list(spec.allowlist),
            dry_run=True,
        )
        actual_names = {row.filename for row in rows}
        if actual_names != set(spec.allowlist):
            missing = sorted(set(spec.allowlist) - actual_names)
            unexpected = sorted(actual_names - set(spec.allowlist))
            raise SnapshotPolicyError(f"{spec.key} dry-run file mismatch; missing={missing}, unexpected={unexpected}")
        commits = {row.commit_hash for row in rows}
        if commits != {spec.revision}:
            raise SnapshotPolicyError(f"{spec.key} dry-run resolved a different commit: {sorted(commits)}")
        declared_bytes = sum(int(row.file_size) for row in rows)
        download_bytes = sum(int(row.file_size) for row in rows if row.will_download)
        if declared_bytes > spec.download_cap_bytes:
            raise SnapshotPolicyError(
                f"{spec.key} allowlisted bytes {declared_bytes} exceed cap {spec.download_cap_bytes}"
            )
        return {
            "key": spec.key,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "files": [
                {
                    "path": row.filename,
                    "bytes": int(row.file_size),
                    "cached": bool(row.is_cached),
                    "will_download": bool(row.will_download),
                }
                for row in sorted(rows, key=lambda item: item.filename)
            ],
            "declared_bytes": declared_bytes,
            "download_bytes": download_bytes,
            "cap_bytes": spec.download_cap_bytes,
            "cache": self.cache_audit(),
        }

    def plan_batch(self, keys: tuple[str, ...] = SELECTED_KEYS) -> dict[str, Any]:
        plans = [self.plan(INVENTORY[key]) for key in keys]
        declared = sum(row["declared_bytes"] for row in plans)
        download = sum(row["download_bytes"] for row in plans)
        if declared > BATCH_DOWNLOAD_CAP_BYTES or declared > STORY_DISK_CAP_BYTES:
            raise SnapshotPolicyError("Batch-A allowlisted bytes exceed the approved hard caps")
        return {
            "models": plans,
            "declared_bytes": declared,
            "download_bytes": download,
            "batch_cap_bytes": BATCH_DOWNLOAD_CAP_BYTES,
            "story_disk_cap_bytes": STORY_DISK_CAP_BYTES,
            "cache": self.cache_audit(),
        }

    def _verify_snapshot(
        self,
        spec: LearnedSparseSpec,
        path: Path,
        *,
        declared_bytes: int | None = None,
    ) -> dict[str, Any]:
        identity = snapshot_identity(path)
        if set(identity) != set(spec.allowlist):
            missing = sorted(set(spec.allowlist) - set(identity))
            unexpected = sorted(set(identity) - set(spec.allowlist))
            raise SnapshotPolicyError(f"{spec.key} snapshot file mismatch; missing={missing}, unexpected={unexpected}")
        for name in identity:
            if PurePosixPath(name).suffix.lower() in PICKLE_SUFFIXES | {".py"}:
                raise SnapshotPolicyError(f"{spec.key} snapshot contains a forbidden file: {name}")
        for config_path in sorted(path.rglob("*.json")):
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SnapshotPolicyError(f"{spec.key} contains invalid JSON: {config_path.name}") from exc
            if _contains_auto_map(config):
                raise SnapshotPolicyError(f"{spec.key} snapshot contains forbidden auto_map: {config_path.name}")
        if spec.identity:
            verify_snapshot_identity(path, spec.identity, label=spec.key)
        actual_bytes = sum((path / name).stat().st_size for name in spec.allowlist)
        if actual_bytes > int(spec.download_cap_bytes or 0):
            raise SnapshotPolicyError(f"{spec.key} downloaded bytes exceed the bounded cap")
        if declared_bytes is not None and actual_bytes != declared_bytes:
            raise SnapshotPolicyError(f"{spec.key} downloaded bytes do not match the bounded plan")
        return {
            "snapshot_path": str(path.resolve()),
            "actual_bytes": actual_bytes,
            "identity": identity,
            "aggregate_identity_sha256": aggregate_snapshot_identity(identity),
            "cache": self.cache_audit(),
        }

    def resolve_local(self, spec: LearnedSparseSpec) -> dict[str, Any]:
        """Resolve and verify a pinned snapshot from the standard local hub cache only."""
        self._validate_spec(spec)
        try:
            path = Path(
                snapshot_download(
                    repo_id=spec.repo_id,
                    revision=spec.revision,
                    cache_dir=self.hub_cache_dir,
                    allow_patterns=list(spec.allowlist),
                    local_files_only=True,
                )
            )
        except Exception as exc:
            raise SnapshotPolicyError(
                f"{spec.key} is not complete in local hub cache {self.hub_cache_dir}"
            ) from exc
        return {
            "key": spec.key,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            **self._verify_snapshot(spec, path),
        }

    def resolve(self, spec: LearnedSparseSpec, *, plan: dict[str, Any] | None = None) -> dict[str, Any]:
        """Download exactly the planned files and return a reproducible identity."""
        plan = plan or self.plan(spec)
        if plan["key"] != spec.key or plan["revision"] != spec.revision:
            raise SnapshotPolicyError("Snapshot download plan does not match the requested model")
        path = Path(
            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                cache_dir=self.hub_cache_dir,
                allow_patterns=list(spec.allowlist),
            )
        )
        return {
            **plan,
            **self._verify_snapshot(spec, path, declared_bytes=plan["declared_bytes"]),
        }


__all__ = [
    "ANCHOR_KEYS",
    "BATCH_DOWNLOAD_CAP_BYTES",
    "BoundedSnapshotResolver",
    "INVENTORY",
    "LearnedSparseSpec",
    "SELECTED_KEYS",
    "STORY_DISK_CAP_BYTES",
    "SnapshotPolicyError",
    "aggregate_snapshot_identity",
    "inventory_document",
    "standard_hub_cache_dir",
]
