"""Bounded, no-publish source materialization for code-edit retrieval smokes."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import resource
import shutil
import signal
import tarfile
import tempfile
import threading
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse


STAGE_A_POLICY_VERSION = "code-edit-stage-a-v0"
STAGE_B_POLICY_VERSION = "code-edit-stage-b-v0"
NORMALIZATION_VERSION = "utf8-bom-strip-lf-v0"
CHUNKER_VERSION = "code-edit-ast-fallback-smoke-v0"
QREL_GENERATOR_VERSION = "patch-preimage-and-insertion-anchor-smoke-v0"
ARTIFACT_SUMMARY_VERSION = "code-edit-source-materialization-summary-v0"

ALLOWED_GITHUB_HOSTS = {
    "api.github.com",
    "codeload.github.com",
    "github.com",
    "patch-diff.githubusercontent.com",
}
ALLOWED_SUFFIXES = (
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
)
EXCLUDED_SEGMENTS = {
    ".git",
    "vendor",
    "vendors",
    "third_party",
    "third-party",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".tox",
    ".nox",
    ".pytest_cache",
    "__pycache__",
}
EXCLUDED_BASENAMES = {
    "package-lock.json",
    "npm-shrinkwrap.json",
    "pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
    "pnpm-lock.yaml",
    "cargo.lock",
    "gemfile.lock",
    "composer.lock",
}
LFS_PREFIXES = (
    b"version https://git-lfs.github.com/spec/v1\n",
    b"version https://git-lfs.github.com/spec/v1\r\n",
)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FetchBytes = Callable[[str, int], bytes]


class SourceMaterializationError(RuntimeError):
    """A safe contract failure with a stable status and error code."""

    def __init__(self, code: str, message: str, *, status: str = "BLOCKED") -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class _RunDeadline:
    """Convert the configured deadline and SIGTERM into cleanup-safe failures."""

    def __init__(self, wall_seconds: int) -> None:
        self.wall_seconds = wall_seconds
        self.deadline = time.monotonic() + wall_seconds
        self._started = 0.0
        self._previous_alarm: Any = None
        self._previous_term: Any = None
        self._previous_timer = (0.0, 0.0)
        self._active = False

    def start(self) -> None:
        if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
            raise SourceMaterializationError(
                "deadline_unavailable",
                "A main-thread POSIX signal deadline is required for bounded materialization",
                status="FAILED",
            )
        self._started = time.monotonic()
        self.deadline = self._started + self.wall_seconds
        self._previous_alarm = signal.getsignal(signal.SIGALRM)
        self._previous_term = signal.getsignal(signal.SIGTERM)
        self._previous_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.setitimer(signal.ITIMER_REAL, float(self.wall_seconds))
        self._active = True

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        if signum == signal.SIGALRM:
            raise SourceMaterializationError(
                "wall_clock_cap",
                f"Configured {self.wall_seconds}s wall-clock deadline expired",
            )
        raise SourceMaterializationError(
            "terminated",
            "Received SIGTERM; materialization stopped before cleanup",
        )

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise SourceMaterializationError(
                "wall_clock_cap",
                f"Configured {self.wall_seconds}s wall-clock deadline expired",
            )

    def begin_cleanup(self) -> None:
        if not self._active:
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def restore(self) -> None:
        if not self._active:
            return
        elapsed = time.monotonic() - self._started
        signal.signal(signal.SIGALRM, self._previous_alarm)
        signal.signal(signal.SIGTERM, self._previous_term)
        previous_remaining, previous_interval = self._previous_timer
        if previous_remaining > 0.0:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.0, previous_remaining - elapsed),
                previous_interval,
            )
        self._active = False


@dataclass(frozen=True)
class ResourceCaps:
    """Hard limits applied before or during source materialization."""

    archive_bytes: int
    extracted_regular_file_bytes: int
    stage_a_candidate_bytes: int
    eligible_normalized_text_bytes: int
    tracked_files: int
    chunks: int
    target_rss_bytes: int
    wall_seconds: int


@dataclass(frozen=True)
class ChunkingPolicy:
    """Model-independent deterministic line windows."""

    definition_window_lines: int
    definition_overlap_lines: int
    fallback_window_lines: int
    fallback_overlap_lines: int


@dataclass(frozen=True)
class SourceContract:
    """Pinned identities and limits for one repository/query smoke."""

    contract_version: str
    repository_id: str
    issue_number: int
    pr_number: int
    base_commit: str
    tree_sha: str
    pr_head_sha: str
    expected_changed_paths: tuple[str, ...]
    license_path: str
    license_blob_sha: str
    license_sha256: str
    query_text_sha256: str
    source_sha256: Mapping[str, str]
    caps: ResourceCaps
    chunking: ChunkingPolicy
    publish: bool
    evidence_tier: str
    query_review: Mapping[str, str]

    @property
    def api_root(self) -> str:
        return f"https://api.github.com/repos/{self.repository_id}"

    @property
    def source_urls(self) -> dict[str, str]:
        return {
            "issue_api": f"{self.api_root}/issues/{self.issue_number}",
            "pr_api": f"{self.api_root}/pulls/{self.pr_number}",
            "pr_files_api": f"{self.api_root}/pulls/{self.pr_number}/files",
            "tree_api": f"{self.api_root}/git/trees/{self.tree_sha}?recursive=1",
            "patch": f"https://github.com/{self.repository_id}/pull/{self.pr_number}.patch",
            "archive": f"https://github.com/{self.repository_id}/archive/{self.base_commit}.tar.gz",
        }


@dataclass(frozen=True)
class TreeEntry:
    """One bounded recursive-tree entry used for archive verification."""

    path: str
    mode: str
    entry_type: str
    sha: str
    size: int


@dataclass(frozen=True)
class FileAudit:
    """Stage A/Stage B audit metadata without retained source text."""

    path: str
    git_mode: str
    blob_sha: str
    raw_bytes: int
    stage_a_result: str
    stage_a_reason: str
    stage_b_result: str
    stage_b_reason: str
    bom_present: bool
    decoded_utf8_bytes: int | None
    normalized_utf8_bytes: int | None
    normalized_sha256: str | None


@dataclass(frozen=True)
class MaterializedTextFile:
    """One accepted file retained only inside the temporary run."""

    audit: FileAudit
    normalized_text: str = field(repr=False)


@dataclass(frozen=True)
class SourceChunk:
    """One deterministic retrieval candidate retained only for the run."""

    chunk_id: str
    repository_id: str
    base_commit: str
    path: str
    blob_sha: str
    candidate_family: str
    symbol: str | None
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    text_sha256: str
    ordinal_in_file: int
    text: str = field(repr=False)


@dataclass(frozen=True)
class PatchTarget:
    """One exact preimage line group or addition-only insertion anchor."""

    target_id: str
    changed_path: str
    patch_change_type: str
    preimage_line_start: int | None
    preimage_line_end: int | None
    insertion_anchor_after_line: int | None
    target_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class SourceQrel:
    """One direct grade-2 qrel derived from a pinned preimage target."""

    qrel_id: str
    chunk_id: str
    target_id: str
    target_unit_ids: tuple[str, ...]
    relevance: int
    label_family: str
    changed_path: str
    patch_change_type: str
    overlap_lines: int
    mapping_status: str


@dataclass(frozen=True)
class MaterializationArtifacts:
    """Validated in-memory artifacts; source text never enters the summary."""

    source_hashes: Mapping[str, str]
    source_bytes: Mapping[str, int]
    file_audits: tuple[FileAudit, ...]
    files: tuple[MaterializedTextFile, ...]
    chunks: tuple[SourceChunk, ...]
    patch_targets: tuple[PatchTarget, ...]
    qrels: tuple[SourceQrel, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _git_blob_sha(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ascii_lower(value: str) -> str:
    return value.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))


def _safe_repository_path(value: str) -> tuple[str, ...]:
    if not value or value.startswith("/") or "\0" in value or "\\" in value:
        raise SourceMaterializationError("unsafe_repository_path", f"Unsafe repository path: {value!r}")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise SourceMaterializationError("unsafe_repository_path", f"Unsafe repository path: {value!r}")
    return parts


def _expect_sha(value: str, pattern: re.Pattern[str], field_name: str) -> None:
    if not pattern.fullmatch(value):
        raise SourceMaterializationError("invalid_contract", f"Invalid {field_name}: {value!r}", status="FAILED")


def _validate_contract(contract: SourceContract) -> None:
    if contract.publish or contract.evidence_tier != "smoke":
        raise SourceMaterializationError(
            "publication_contract",
            "Source materialization must remain publish=false and evidence_tier=smoke",
            status="FAILED",
        )
    if "/" not in contract.repository_id or contract.repository_id.count("/") != 1:
        raise SourceMaterializationError("invalid_contract", "repository_id must be owner/repository", status="FAILED")
    _expect_sha(contract.base_commit, SHA1_RE, "base_commit")
    _expect_sha(contract.tree_sha, SHA1_RE, "tree_sha")
    _expect_sha(contract.pr_head_sha, SHA1_RE, "pr_head_sha")
    _expect_sha(contract.license_blob_sha, SHA1_RE, "license_blob_sha")
    _expect_sha(contract.license_sha256, SHA256_RE, "license_sha256")
    _expect_sha(contract.query_text_sha256, SHA256_RE, "query_text_sha256")
    _safe_repository_path(contract.license_path)
    if not contract.expected_changed_paths:
        raise SourceMaterializationError("invalid_contract", "Expected changed paths cannot be empty", status="FAILED")
    for path in contract.expected_changed_paths:
        _safe_repository_path(path)
    if tuple(sorted(set(contract.expected_changed_paths))) != contract.expected_changed_paths:
        raise SourceMaterializationError(
            "invalid_contract",
            "Expected changed paths must be unique and sorted",
            status="FAILED",
        )
    known_source_keys = set(contract.source_urls)
    pinned_source_keys = set(contract.source_sha256)
    if not {"archive", "patch"}.issubset(pinned_source_keys) or not pinned_source_keys.issubset(known_source_keys):
        raise SourceMaterializationError(
            "invalid_contract",
            "source_sha256 must pin patch/archive and may only contain known source names",
            status="FAILED",
        )
    for key, value in contract.source_sha256.items():
        _expect_sha(value, SHA256_RE, f"source_sha256.{key}")
    required_reviews = {
        "answer_leak_review",
        "privacy_review",
        "prompt_injection_review",
        "secret_review",
    }
    if set(contract.query_review) != required_reviews or any(
        contract.query_review[key] != "pass" for key in required_reviews
    ):
        raise SourceMaterializationError(
            "query_review_contract",
            "All pinned query review gates must be present and pass",
            status="FAILED",
        )
    caps = contract.caps
    if any(value <= 0 for value in asdict(caps).values()):
        raise SourceMaterializationError("invalid_contract", "All resource caps must be positive", status="FAILED")
    chunking = contract.chunking
    if (
        chunking.definition_window_lines <= 0
        or chunking.fallback_window_lines <= 0
        or not 0 <= chunking.definition_overlap_lines < chunking.definition_window_lines
        or not 0 <= chunking.fallback_overlap_lines < chunking.fallback_window_lines
    ):
        raise SourceMaterializationError("invalid_contract", "Invalid chunking windows", status="FAILED")


def load_source_contract(path: str | Path) -> SourceContract:
    """Load and validate a small JSON source contract."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    contract = SourceContract(
        contract_version=str(payload["contract_version"]),
        repository_id=str(payload["repository_id"]),
        issue_number=int(payload["issue_number"]),
        pr_number=int(payload["pr_number"]),
        base_commit=str(payload["base_commit"]),
        tree_sha=str(payload["tree_sha"]),
        pr_head_sha=str(payload["pr_head_sha"]),
        expected_changed_paths=tuple(str(item) for item in payload["expected_changed_paths"]),
        license_path=str(payload["license_path"]),
        license_blob_sha=str(payload["license_blob_sha"]),
        license_sha256=str(payload["license_sha256"]),
        query_text_sha256=str(payload["query_text_sha256"]),
        source_sha256={str(key): str(value) for key, value in payload["source_sha256"].items()},
        caps=ResourceCaps(**{key: int(value) for key, value in payload["caps"].items()}),
        chunking=ChunkingPolicy(**{key: int(value) for key, value in payload["chunking"].items()}),
        publish=bool(payload["publish"]),
        evidence_tier=str(payload["evidence_tier"]),
        query_review={str(key): str(value) for key, value in payload["query_review"].items()},
    )
    _validate_contract(contract)
    return contract


def _default_fetch_bytes(url: str, max_bytes: int) -> bytes:
    import httpx

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_GITHUB_HOSTS:
        raise SourceMaterializationError("network_host", f"Disallowed source URL host: {url}")
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0),
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "modern-embedding-bench-source-materializer/0",
            },
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                for hop in [*response.history, response]:
                    hop_host = hop.url.host
                    if hop.url.scheme != "https" or hop_host not in ALLOWED_GITHUB_HOSTS:
                        raise SourceMaterializationError(
                            "network_redirect_host",
                            f"Disallowed GitHub redirect host: {hop.url}",
                        )
                content_length = response.headers.get("content-length")
                if content_length is not None and int(content_length) > max_bytes:
                    raise SourceMaterializationError(
                        "download_cap",
                        f"Source response exceeds the {max_bytes}-byte cap before download",
                    )
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise SourceMaterializationError(
                            "download_cap",
                            f"Source response exceeds the {max_bytes}-byte cap",
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
    except SourceMaterializationError:
        raise
    except Exception as exc:
        raise SourceMaterializationError("network_access", f"GitHub source request failed: {exc}") from exc


def _fetch_sources(contract: SourceContract, fetch_bytes: FetchBytes) -> dict[str, bytes]:
    sources: dict[str, bytes] = {}
    metadata_cap = min(contract.caps.archive_bytes, 5_000_000)
    for name, url in contract.source_urls.items():
        cap = contract.caps.archive_bytes if name == "archive" else metadata_cap
        content = fetch_bytes(url, cap)
        digest = _sha256_bytes(content)
        expected_digest = contract.source_sha256.get(name)
        if expected_digest is not None and digest != expected_digest:
            raise SourceMaterializationError(
                "source_pin_drift",
                f"Pinned {name} SHA-256 drifted: expected {expected_digest}, observed {digest}",
            )
        sources[name] = content
    return sources


def _decode_json(source_name: str, content: bytes) -> Any:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceMaterializationError("invalid_github_metadata", f"Invalid {source_name} JSON") from exc


def _validate_metadata(contract: SourceContract, sources: Mapping[str, bytes]) -> tuple[tuple[TreeEntry, ...], str]:
    issue = _decode_json("issue_api", sources["issue_api"])
    pr = _decode_json("pr_api", sources["pr_api"])
    pr_files = _decode_json("pr_files_api", sources["pr_files_api"])
    tree = _decode_json("tree_api", sources["tree_api"])
    if not isinstance(issue, dict) or int(issue.get("number", -1)) != contract.issue_number:
        raise SourceMaterializationError("issue_pin_drift", "Issue number no longer matches the source contract")
    expected_repo_api = f"https://api.github.com/repos/{contract.repository_id}"
    if issue.get("repository_url") != expected_repo_api:
        raise SourceMaterializationError("issue_pin_drift", "Issue repository identity drifted")
    title = issue.get("title")
    body = issue.get("body")
    if not isinstance(title, str) or body is not None and not isinstance(body, str):
        raise SourceMaterializationError("issue_contract", "Issue title/body is not usable UTF-8 text")
    query_text = f"{title}\n\n{body or ''}"
    if _sha256_text(query_text) != contract.query_text_sha256:
        raise SourceMaterializationError("query_pin_drift", "Pinned issue query text SHA-256 drifted")
    lowered_query = _ascii_lower(query_text)
    if any(_ascii_lower(path) in lowered_query for path in contract.expected_changed_paths):
        raise SourceMaterializationError("answer_leak", "Issue query contains an expected changed path")

    if not isinstance(pr, dict) or int(pr.get("number", -1)) != contract.pr_number:
        raise SourceMaterializationError("pr_pin_drift", "PR number no longer matches the source contract")
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    if base.get("sha") != contract.base_commit or head.get("sha") != contract.pr_head_sha:
        raise SourceMaterializationError("pr_pin_drift", "PR base or head SHA drifted")
    if pr.get("state") != "closed" or not pr.get("merged_at"):
        raise SourceMaterializationError("pr_state", "Pinned PR is not closed and merged")
    if not isinstance(pr_files, list):
        raise SourceMaterializationError("pr_files_contract", "PR files response is not a list")
    observed_changed_paths = tuple(sorted(str(item.get("filename")) for item in pr_files if isinstance(item, dict)))
    if observed_changed_paths != contract.expected_changed_paths:
        raise SourceMaterializationError(
            "changed_path_drift",
            f"Expected changed paths {contract.expected_changed_paths}, observed {observed_changed_paths}",
        )

    if not isinstance(tree, dict) or tree.get("sha") != contract.tree_sha or tree.get("truncated") is not False:
        raise SourceMaterializationError("tree_pin_drift", "Pinned recursive tree is missing, truncated, or drifted")
    raw_entries = tree.get("tree")
    if not isinstance(raw_entries, list):
        raise SourceMaterializationError("tree_contract", "Recursive tree entries are missing")
    entries: list[TreeEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise SourceMaterializationError("tree_contract", "Recursive tree contains a non-object entry")
        path = str(raw.get("path", ""))
        _safe_repository_path(path)
        entry_type = str(raw.get("type", ""))
        mode = str(raw.get("mode", ""))
        sha = str(raw.get("sha", ""))
        size = int(raw.get("size", 0) or 0)
        if entry_type == "blob":
            _expect_sha(sha, SHA1_RE, f"tree blob {path}")
            if size < 0:
                raise SourceMaterializationError("tree_contract", f"Negative blob size for {path}")
        entries.append(TreeEntry(path=path, mode=mode, entry_type=entry_type, sha=sha, size=size))
    blob_entries = tuple(entry for entry in entries if entry.entry_type == "blob")
    if len(blob_entries) > contract.caps.tracked_files:
        raise SourceMaterializationError(
            "tracked_file_cap",
            f"Tracked blob count {len(blob_entries)} exceeds cap {contract.caps.tracked_files}",
        )
    unsafe_modes = [entry.path for entry in blob_entries if entry.mode not in {"100644", "100755"}]
    submodules = [entry.path for entry in entries if entry.entry_type == "commit" or entry.mode == "160000"]
    if unsafe_modes or submodules:
        raise SourceMaterializationError(
            "unsupported_tree_entry",
            f"Tree contains links, unsupported blob modes, or submodules: {sorted(unsafe_modes + submodules)}",
        )
    return tuple(sorted(blob_entries, key=lambda item: item.path)), query_text


def _validate_archive_member_name(name: str) -> tuple[str, ...]:
    if not name or name.startswith("/") or "\0" in name or "\\" in name:
        raise SourceMaterializationError("unsafe_archive_entry", f"Unsafe archive entry: {name!r}")
    pure = PurePosixPath(name)
    parts = pure.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise SourceMaterializationError("unsafe_archive_entry", f"Unsafe archive entry: {name!r}")
    return parts


def _extract_verified_archive(
    contract: SourceContract,
    archive: bytes,
    tree_entries: Sequence[TreeEntry],
    destination: Path,
) -> dict[str, bytes]:
    if len(archive) > contract.caps.archive_bytes:
        raise SourceMaterializationError(
            "archive_cap",
            f"Archive bytes {len(archive)} exceed cap {contract.caps.archive_bytes}",
        )
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            members = bundle.getmembers()
            roots: set[str] = set()
            regular: list[tuple[tarfile.TarInfo, str]] = []
            seen_paths: set[str] = set()
            declared_bytes = 0
            for member in members:
                parts = _validate_archive_member_name(member.name.rstrip("/"))
                roots.add(parts[0])
                if member.isdir():
                    continue
                if not member.isreg():
                    raise SourceMaterializationError(
                        "unsafe_archive_link",
                        f"Archive entry is not a regular file or directory: {member.name}",
                    )
                if len(parts) < 2:
                    raise SourceMaterializationError("unsafe_archive_entry", "Archive file lacks one root directory")
                relative = "/".join(parts[1:])
                _safe_repository_path(relative)
                if relative in seen_paths:
                    raise SourceMaterializationError("duplicate_archive_entry", f"Duplicate archive path: {relative}")
                seen_paths.add(relative)
                declared_bytes += member.size
                if declared_bytes > contract.caps.extracted_regular_file_bytes:
                    raise SourceMaterializationError(
                        "extracted_byte_cap",
                        f"Archive regular bytes exceed cap {contract.caps.extracted_regular_file_bytes}",
                    )
                regular.append((member, relative))
            if len(roots) != 1:
                raise SourceMaterializationError("archive_root", "Archive must contain exactly one root directory")

            extracted: dict[str, bytes] = {}
            for member, relative in regular:
                handle = bundle.extractfile(member)
                if handle is None:
                    raise SourceMaterializationError("archive_read", f"Cannot read archive member: {member.name}")
                content = handle.read(member.size + 1)
                if len(content) != member.size:
                    raise SourceMaterializationError("archive_read", f"Archive size mismatch for {relative}")
                output = destination.joinpath(*relative.split("/"))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(content)
                extracted[relative] = content
    except SourceMaterializationError:
        raise
    except (tarfile.TarError, OSError) as exc:
        raise SourceMaterializationError("archive_format", f"Invalid GitHub source archive: {exc}") from exc

    expected = {entry.path: entry for entry in tree_entries}
    if set(extracted) != set(expected):
        missing = sorted(set(expected) - set(extracted))
        extra = sorted(set(extracted) - set(expected))
        raise SourceMaterializationError(
            "incomplete_archive",
            f"Archive/tree file set mismatch; missing={missing}, extra={extra}",
        )
    for path, entry in expected.items():
        content = extracted[path]
        if len(content) != entry.size or _git_blob_sha(content) != entry.sha:
            raise SourceMaterializationError("blob_content_mismatch", f"Blob identity mismatch for {path}")
    license_entry = expected.get(contract.license_path)
    if license_entry is None or license_entry.sha != contract.license_blob_sha:
        raise SourceMaterializationError("license_pin_drift", "Pinned license blob identity is missing or drifted")
    if _sha256_bytes(extracted[contract.license_path]) != contract.license_sha256:
        raise SourceMaterializationError("license_pin_drift", "Pinned license content SHA-256 drifted")
    return extracted


def stage_a_reason(entry: TreeEntry) -> tuple[str, str]:
    """Apply the exact metadata eligibility policy from the accepted minispec."""
    if entry.entry_type != "blob" or entry.mode not in {"100644", "100755"}:
        return "rejected", "excluded_mode"
    parts = _safe_repository_path(entry.path)
    lowered = tuple(_ascii_lower(part) for part in parts)
    if any(part in EXCLUDED_SEGMENTS for part in lowered):
        return "rejected", "excluded_segment"
    basename = lowered[-1]
    if basename in EXCLUDED_BASENAMES:
        return "rejected", "excluded_basename"
    if not basename.endswith(ALLOWED_SUFFIXES):
        return "rejected", "excluded_suffix"
    return "accepted", "accepted"


def _stage_b_audit(entry: TreeEntry, content: bytes, max_candidate_bytes: int) -> MaterializedTextFile | FileAudit:
    if len(content) != entry.size or _git_blob_sha(content) != entry.sha:
        raise SourceMaterializationError("blob_content_mismatch", f"Blob identity mismatch for {entry.path}")
    base = {
        "path": entry.path,
        "git_mode": entry.mode,
        "blob_sha": entry.sha,
        "raw_bytes": len(content),
        "stage_a_result": "accepted",
        "stage_a_reason": "accepted",
    }
    if len(content) > max_candidate_bytes:
        return FileAudit(
            **base,
            stage_b_result="rejected",
            stage_b_reason="oversize",
            bom_present=False,
            decoded_utf8_bytes=None,
            normalized_utf8_bytes=None,
            normalized_sha256=None,
        )
    if content.startswith(LFS_PREFIXES):
        return FileAudit(
            **base,
            stage_b_result="rejected",
            stage_b_reason="lfs_pointer",
            bom_present=False,
            decoded_utf8_bytes=None,
            normalized_utf8_bytes=None,
            normalized_sha256=None,
        )
    if b"\0" in content:
        return FileAudit(
            **base,
            stage_b_result="rejected",
            stage_b_reason="binary_nul",
            bom_present=False,
            decoded_utf8_bytes=None,
            normalized_utf8_bytes=None,
            normalized_sha256=None,
        )
    bom_present = content.startswith(b"\xef\xbb\xbf")
    decodable = content[3:] if bom_present else content
    try:
        text = decodable.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return FileAudit(
            **base,
            stage_b_result="rejected",
            stage_b_reason="invalid_utf8",
            bom_present=bom_present,
            decoded_utf8_bytes=None,
            normalized_utf8_bytes=None,
            normalized_sha256=None,
        )
    control_count = sum(
        1
        for byte in decodable
        if 0x01 <= byte <= 0x08 or byte == 0x0B or 0x0E <= byte <= 0x1F or byte == 0x7F
    )
    if 100 * control_count > max(1, len(decodable)):
        return FileAudit(
            **base,
            stage_b_result="rejected",
            stage_b_reason="control_heavy",
            bom_present=bom_present,
            decoded_utf8_bytes=len(decodable),
            normalized_utf8_bytes=None,
            normalized_sha256=None,
        )
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_bytes = normalized.encode("utf-8")
    audit = FileAudit(
        **base,
        stage_b_result="accepted",
        stage_b_reason="accepted",
        bom_present=bom_present,
        decoded_utf8_bytes=len(decodable),
        normalized_utf8_bytes=len(normalized_bytes),
        normalized_sha256=_sha256_bytes(normalized_bytes),
    )
    return MaterializedTextFile(audit=audit, normalized_text=normalized)


def audit_eligible_files(
    contract: SourceContract,
    tree_entries: Sequence[TreeEntry],
    extracted: Mapping[str, bytes],
) -> tuple[tuple[FileAudit, ...], tuple[MaterializedTextFile, ...]]:
    """Apply Stage A and Stage B in order and enforce corpus byte limits."""
    audits: list[FileAudit] = []
    files: list[MaterializedTextFile] = []
    eligible_bytes = 0
    for entry in sorted(tree_entries, key=lambda item: item.path):
        stage_a_result, reason = stage_a_reason(entry)
        if stage_a_result != "accepted":
            audits.append(
                FileAudit(
                    path=entry.path,
                    git_mode=entry.mode,
                    blob_sha=entry.sha,
                    raw_bytes=entry.size,
                    stage_a_result=stage_a_result,
                    stage_a_reason=reason,
                    stage_b_result="not_applicable",
                    stage_b_reason="not_stage_a_candidate",
                    bom_present=False,
                    decoded_utf8_bytes=None,
                    normalized_utf8_bytes=None,
                    normalized_sha256=None,
                )
            )
            continue
        result = _stage_b_audit(entry, extracted[entry.path], contract.caps.stage_a_candidate_bytes)
        if isinstance(result, MaterializedTextFile):
            eligible_bytes += int(result.audit.normalized_utf8_bytes or 0)
            if eligible_bytes > contract.caps.eligible_normalized_text_bytes:
                raise SourceMaterializationError(
                    "eligible_text_cap",
                    f"Eligible normalized bytes exceed cap {contract.caps.eligible_normalized_text_bytes}",
                )
            files.append(result)
            audits.append(result.audit)
        else:
            audits.append(result)
    return tuple(audits), tuple(files)


def _line_char_range(text: str, line_start: int, line_end: int) -> tuple[int, int, str]:
    lines = text.splitlines(keepends=True)
    if not lines and line_start == line_end == 1:
        return 0, 0, ""
    if line_start < 1 or line_end < line_start or line_end > len(lines):
        raise SourceMaterializationError(
            "chunk_span",
            f"Invalid line range {line_start}-{line_end}",
            status="FAILED",
        )
    char_start = sum(len(line) for line in lines[: line_start - 1])
    char_end = sum(len(line) for line in lines[:line_end])
    return char_start, char_end, text[char_start:char_end]


def _window_ranges(line_start: int, line_end: int, window: int, overlap: int) -> list[tuple[int, int]]:
    if line_end - line_start + 1 <= window:
        return [(line_start, line_end)]
    step = window - overlap
    ranges: list[tuple[int, int]] = []
    start = line_start
    while start <= line_end:
        end = min(line_end, start + window - 1)
        ranges.append((start, end))
        if end == line_end:
            break
        start += step
    return ranges


def _python_ranges(
    text: str,
    policy: ChunkingPolicy,
) -> list[tuple[str, str | None, int, int]] | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    definition_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    definitions = sorted(
        (node for node in ast.walk(tree) if isinstance(node, definition_types)),
        key=lambda node: (node.lineno, int(node.end_lineno or node.lineno), node.name),
    )
    ranges: list[tuple[str, str | None, int, int]] = []
    if definitions:
        first_line = min(node.lineno for node in definitions)
        if any(line.strip() for line in text.splitlines()[: first_line - 1]):
            for index, (start, end) in enumerate(
                _window_ranges(1, first_line - 1, policy.definition_window_lines, policy.definition_overlap_lines)
            ):
                family = "module_preamble" if index == 0 and end == first_line - 1 else "module_preamble_window"
                ranges.append((family, None, start, end))
    for node in definitions:
        end_line = int(node.end_lineno or node.lineno)
        if isinstance(node, ast.ClassDef):
            family = "ast_class"
        elif isinstance(parents.get(node), ast.ClassDef):
            family = "ast_method"
        else:
            family = "ast_function"
        windows = _window_ranges(node.lineno, end_line, policy.definition_window_lines, policy.definition_overlap_lines)
        for index, (start, end) in enumerate(windows):
            range_family = family if len(windows) == 1 else f"{family}_window"
            symbol = node.name if len(windows) == 1 else f"{node.name}#{index:04d}"
            ranges.append((range_family, symbol, start, end))
    return sorted(ranges, key=lambda item: (item[2], item[3], item[0], item[1] or ""))


def _fallback_ranges(text: str, policy: ChunkingPolicy) -> list[tuple[str, str | None, int, int]]:
    line_count = len(text.splitlines())
    if line_count == 0:
        return [("empty_file", None, 1, 1)]
    return [
        ("line_fallback", None, start, end)
        for start, end in _window_ranges(
            1,
            line_count,
            policy.fallback_window_lines,
            policy.fallback_overlap_lines,
        )
    ]


def build_chunks(contract: SourceContract, files: Sequence[MaterializedTextFile]) -> tuple[SourceChunk, ...]:
    """Build deterministic AST definitions/methods with fixed-line fallback."""
    chunks: list[SourceChunk] = []
    for file in sorted(files, key=lambda item: item.audit.path):
        lowered = _ascii_lower(file.audit.path)
        ranges = _python_ranges(file.normalized_text, contract.chunking) if lowered.endswith((".py", ".pyi")) else None
        if ranges is None or not ranges:
            ranges = _fallback_ranges(file.normalized_text, contract.chunking)
        for ordinal, (family, symbol, line_start, line_end) in enumerate(ranges):
            char_start, char_end, text = _line_char_range(file.normalized_text, line_start, line_end)
            symbol_part = symbol or "none"
            chunk_id = (
                f"{contract.repository_id}@{contract.base_commit[:12]}:{file.audit.path}:"
                f"{family}:{symbol_part}:{ordinal:04d}"
            )
            chunks.append(
                SourceChunk(
                    chunk_id=chunk_id,
                    repository_id=contract.repository_id,
                    base_commit=contract.base_commit,
                    path=file.audit.path,
                    blob_sha=file.audit.blob_sha,
                    candidate_family=family,
                    symbol=symbol,
                    line_start=line_start,
                    line_end=line_end,
                    char_start=char_start,
                    char_end=char_end,
                    text_sha256=_sha256_text(text),
                    ordinal_in_file=ordinal,
                    text=text,
                )
            )
            if len(chunks) > contract.caps.chunks:
                raise SourceMaterializationError(
                    "chunk_cap",
                    f"Chunk count exceeds cap {contract.caps.chunks}",
                )
    return tuple(chunks)


def _parse_diff_path(line: str, prefix: str) -> str:
    value = line[len(prefix) :].split("\t", 1)[0]
    if value == "/dev/null" or value.startswith('"'):
        raise SourceMaterializationError("unsupported_patch", f"Unsupported patch path: {value}")
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    _safe_repository_path(value)
    return value


def parse_patch_targets(
    contract: SourceContract,
    patch: bytes,
    files: Sequence[MaterializedTextFile],
) -> tuple[PatchTarget, ...]:
    """Validate unified diff preimages exactly and derive direct target units."""
    try:
        patch_text = patch.decode("utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise SourceMaterializationError("patch_encoding", "Patch is not strict UTF-8") from exc
    file_by_path = {file.audit.path: file for file in files}
    lines = patch_text.splitlines()
    targets: list[PatchTarget] = []
    observed_paths: set[str] = set()
    current_path: str | None = None
    index = 0
    target_ordinal = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            if "\"" in line:
                raise SourceMaterializationError("unsupported_patch", "Quoted diff paths are not supported")
            parts = line.split()
            if len(parts) != 4:
                raise SourceMaterializationError("unsupported_patch", f"Invalid diff header: {line}")
            old_path = _parse_diff_path(parts[2], "")
            new_path = _parse_diff_path(parts[3], "")
            if old_path != new_path:
                raise SourceMaterializationError("unsupported_patch", "Renames are not supported by the v0 mapper")
            current_path = old_path
            observed_paths.add(current_path)
            index += 1
            continue
        if line.startswith("--- "):
            old_path = _parse_diff_path(line, "--- ")
            if current_path is not None and old_path != current_path:
                raise SourceMaterializationError("patch_path_mismatch", "Patch old path disagrees with diff header")
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise SourceMaterializationError("unsupported_patch", "Patch new path header is missing")
            new_path = _parse_diff_path(lines[index], "+++ ")
            if old_path != new_path:
                raise SourceMaterializationError("unsupported_patch", "New/deleted/renamed files are not supported")
            current_path = old_path
            observed_paths.add(current_path)
            index += 1
            continue
        match = HUNK_RE.match(line)
        if match:
            if current_path is None or current_path not in file_by_path:
                raise SourceMaterializationError(
                    "patch_missing_preimage",
                    f"Patch path is not an eligible preimage file: {current_path}",
                )
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_count = int(match.group(4) or "1")
            source_lines = file_by_path[current_path].normalized_text.splitlines()
            old_line = old_start
            new_consumed = 0
            deletion_lines: list[int] = []
            addition_anchors: list[int] = []
            in_addition_block = False
            index += 1
            while index < len(lines) and not lines[index].startswith(("diff --git ", "--- ", "@@ ")):
                hunk_line = lines[index]
                if hunk_line == "\\ No newline at end of file":
                    index += 1
                    continue
                if old_line - old_start == old_count and new_consumed == new_count:
                    break
                if not hunk_line or hunk_line[0] not in {" ", "+", "-"}:
                    raise SourceMaterializationError("unsupported_patch", f"Invalid hunk line: {hunk_line!r}")
                marker = hunk_line[0]
                content = hunk_line[1:]
                if marker in {" ", "-"}:
                    if old_line < 1 or old_line > len(source_lines) or source_lines[old_line - 1] != content:
                        raise SourceMaterializationError(
                            "patch_preimage_mismatch",
                            f"Patch preimage does not match {current_path}:{old_line}",
                        )
                    if marker == "-":
                        deletion_lines.append(old_line)
                    old_line += 1
                    if marker == " ":
                        new_consumed += 1
                    in_addition_block = False
                else:
                    new_consumed += 1
                    if not in_addition_block:
                        addition_anchors.append(old_line - 1)
                        in_addition_block = True
                index += 1
            if old_line - old_start != old_count or new_consumed != new_count:
                raise SourceMaterializationError(
                    "patch_preimage_incomplete",
                    "Hunk old/new line counts do not match the header",
                )
            if deletion_lines:
                groups: list[list[int]] = []
                for line_number in deletion_lines:
                    if not groups or line_number != groups[-1][-1] + 1:
                        groups.append([line_number])
                    else:
                        groups[-1].append(line_number)
                for group in groups:
                    identity = f"{current_path}:{group[0]}:{group[-1]}"
                    target_id = f"target_{target_ordinal:04d}_{_sha256_text(identity)[:12]}"
                    targets.append(
                        PatchTarget(
                            target_id=target_id,
                            changed_path=current_path,
                            patch_change_type="modified_or_deleted_preimage",
                            preimage_line_start=group[0],
                            preimage_line_end=group[-1],
                            insertion_anchor_after_line=None,
                            target_unit_ids=tuple(f"{target_id}:line:{number}" for number in group),
                        )
                    )
                    target_ordinal += 1
            else:
                if not addition_anchors:
                    raise SourceMaterializationError("patch_target_incomplete", "Hunk contains no edit target")
                for anchor in sorted(set(addition_anchors)):
                    if anchor < 1 or anchor > len(source_lines):
                        raise SourceMaterializationError(
                            "unsupported_patch",
                            "Insertion before the first preimage line is not supported by the v0 mapper",
                        )
                    target_id = f"target_{target_ordinal:04d}_{_sha256_text(f'{current_path}:anchor:{anchor}')[:12]}"
                    targets.append(
                        PatchTarget(
                            target_id=target_id,
                            changed_path=current_path,
                            patch_change_type="addition",
                            preimage_line_start=None,
                            preimage_line_end=None,
                            insertion_anchor_after_line=anchor,
                            target_unit_ids=(f"{target_id}:anchor:{anchor}",),
                        )
                    )
                    target_ordinal += 1
            continue
        index += 1
    if tuple(sorted(observed_paths)) != contract.expected_changed_paths:
        raise SourceMaterializationError(
            "patch_changed_path_drift",
            f"Patch paths {sorted(observed_paths)} do not match {contract.expected_changed_paths}",
        )
    if not targets:
        raise SourceMaterializationError("patch_target_incomplete", "Patch produced no preimage targets")
    return tuple(targets)


def _select_mapping_chunk(target: PatchTarget, line_number: int, chunks: Sequence[SourceChunk]) -> SourceChunk:
    candidates = [
        chunk
        for chunk in chunks
        if chunk.path == target.changed_path and chunk.line_start <= line_number <= chunk.line_end
    ]
    if not candidates:
        raise SourceMaterializationError(
            "patch_mapping_incomplete",
            f"No chunk covers {target.changed_path}:{line_number}",
        )
    candidates.sort(
        key=lambda chunk: (
            chunk.line_end - chunk.line_start,
            abs(line_number - chunk.line_start) + abs(chunk.line_end - line_number),
            chunk.line_start,
            chunk.chunk_id,
        )
    )
    best = candidates[0]
    duplicate_span = [
        chunk
        for chunk in candidates
        if chunk.chunk_id != best.chunk_id
        and (chunk.line_start, chunk.line_end, chunk.candidate_family)
        == (best.line_start, best.line_end, best.candidate_family)
    ]
    if duplicate_span:
        raise SourceMaterializationError(
            "patch_mapping_ambiguous",
            f"Ambiguous duplicate chunk spans cover {target.changed_path}:{line_number}",
        )
    return best


def map_targets_to_qrels(
    targets: Sequence[PatchTarget],
    chunks: Sequence[SourceChunk],
) -> tuple[SourceQrel, ...]:
    """Map every target unit into one deterministic most-specific chunk."""
    qrels: list[SourceQrel] = []
    for target in targets:
        units_by_chunk: dict[str, list[str]] = defaultdict(list)
        chunk_by_id: dict[str, SourceChunk] = {}
        if target.preimage_line_start is not None and target.preimage_line_end is not None:
            unit_lines = zip(
                target.target_unit_ids,
                range(target.preimage_line_start, target.preimage_line_end + 1),
                strict=True,
            )
            label_family = "modified_or_deleted_preimage_line"
        elif target.insertion_anchor_after_line is not None:
            unit_lines = [(target.target_unit_ids[0], target.insertion_anchor_after_line)]
            label_family = "insert_anchor_containing_chunk"
        else:
            raise SourceMaterializationError("patch_mapping_incomplete", f"Target {target.target_id} is empty")
        for unit_id, line_number in unit_lines:
            chunk = _select_mapping_chunk(target, line_number, chunks)
            chunk_by_id[chunk.chunk_id] = chunk
            units_by_chunk[chunk.chunk_id].append(unit_id)
        for chunk_id in sorted(units_by_chunk):
            chunk = chunk_by_id[chunk_id]
            unit_ids = tuple(units_by_chunk[chunk_id])
            overlap_lines = len(unit_ids) if target.preimage_line_start is not None else 0
            identity = f"{target.target_id}\0{chunk_id}\0{'|'.join(unit_ids)}"
            qrels.append(
                SourceQrel(
                    qrel_id=f"qrel_{_sha256_text(identity)[:16]}",
                    chunk_id=chunk_id,
                    target_id=target.target_id,
                    target_unit_ids=unit_ids,
                    relevance=2,
                    label_family=label_family,
                    changed_path=target.changed_path,
                    patch_change_type=target.patch_change_type,
                    overlap_lines=overlap_lines,
                    mapping_status="exact",
                )
            )
    return tuple(qrels)


def validate_materialization_artifacts(contract: SourceContract, artifacts: MaterializationArtifacts) -> None:
    """Reject incomplete corpus, chunk, qrel, or identity artifacts."""
    accepted_paths = {file.audit.path for file in artifacts.files}
    chunk_paths = {chunk.path for chunk in artifacts.chunks}
    if accepted_paths != chunk_paths:
        raise SourceMaterializationError(
            "incomplete_corpus",
            "Every Stage B accepted file must have at least one chunk and no other file may appear",
            status="FAILED",
        )
    if len({chunk.chunk_id for chunk in artifacts.chunks}) != len(artifacts.chunks):
        raise SourceMaterializationError("duplicate_chunk", "Chunk identities are not unique", status="FAILED")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in artifacts.chunks}
    targets_by_id = {target.target_id: target for target in artifacts.patch_targets}
    if len(targets_by_id) != len(artifacts.patch_targets):
        raise SourceMaterializationError("duplicate_target", "Patch target identities are not unique", status="FAILED")
    expected_units = {unit for target in artifacts.patch_targets for unit in target.target_unit_ids}
    mapped_units: set[str] = set()
    for qrel in artifacts.qrels:
        target = targets_by_id.get(qrel.target_id)
        if target is None or qrel.chunk_id not in chunks_by_id:
            raise SourceMaterializationError(
                "incomplete_qrels",
                "Qrel references an unknown target or chunk",
                status="FAILED",
            )
        if qrel.changed_path != target.changed_path or chunks_by_id[qrel.chunk_id].path != target.changed_path:
            raise SourceMaterializationError("incomplete_qrels", "Qrel path identity is inconsistent", status="FAILED")
        if qrel.relevance != 2 or qrel.mapping_status != "exact":
            raise SourceMaterializationError(
                "incomplete_qrels",
                "Only exact grade-2 qrels are allowed",
                status="FAILED",
            )
        mapped_units.update(qrel.target_unit_ids)
    if mapped_units != expected_units:
        raise SourceMaterializationError(
            "incomplete_qrels",
            "Qrels do not cover every patch target unit",
            status="FAILED",
        )
    target_paths = {target.changed_path for target in artifacts.patch_targets}
    if target_paths != set(contract.expected_changed_paths):
        raise SourceMaterializationError(
            "incomplete_qrels",
            "Not every expected changed path has a target",
            status="FAILED",
        )
    if len(artifacts.chunks) > contract.caps.chunks:
        raise SourceMaterializationError("chunk_cap", "Chunk cap was exceeded", status="FAILED")


def _manifest_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(f"{_canonical_json(record)}\n" for record in records)
    return _sha256_text(payload)


def build_artifact_summary(contract: SourceContract, artifacts: MaterializationArtifacts) -> dict[str, Any]:
    """Build a deterministic small summary that excludes source and chunk text."""
    file_records = [asdict(audit) for audit in artifacts.file_audits]
    corpus_records = [
        {
            "path": file.audit.path,
            "git_mode": file.audit.git_mode,
            "blob_sha": file.audit.blob_sha,
            "normalized_utf8_bytes": file.audit.normalized_utf8_bytes,
            "normalized_sha256": file.audit.normalized_sha256,
        }
        for file in artifacts.files
    ]
    chunk_records = [
        {
            key: value
            for key, value in asdict(chunk).items()
            if key != "text"
        }
        for chunk in artifacts.chunks
    ]
    qrel_records = [asdict(qrel) for qrel in artifacts.qrels]
    stage_a_counts = Counter(f"{audit.stage_a_result}:{audit.stage_a_reason}" for audit in artifacts.file_audits)
    stage_b_counts = Counter(f"{audit.stage_b_result}:{audit.stage_b_reason}" for audit in artifacts.file_audits)
    chunk_family_counts = Counter(chunk.candidate_family for chunk in artifacts.chunks)
    target_units = {unit for target in artifacts.patch_targets for unit in target.target_unit_ids}
    mapped_units = {unit for qrel in artifacts.qrels for unit in qrel.target_unit_ids}
    summary = {
        "summary_version": ARTIFACT_SUMMARY_VERSION,
        "contract_version": contract.contract_version,
        "repository_id": contract.repository_id,
        "issue_number": contract.issue_number,
        "pr_number": contract.pr_number,
        "base_commit": contract.base_commit,
        "tree_sha": contract.tree_sha,
        "pr_head_sha": contract.pr_head_sha,
        "expected_changed_paths": list(contract.expected_changed_paths),
        "publish": contract.publish,
        "evidence_tier": contract.evidence_tier,
        "policy": {
            "stage_a": STAGE_A_POLICY_VERSION,
            "stage_b": STAGE_B_POLICY_VERSION,
            "normalization": NORMALIZATION_VERSION,
            "chunker": CHUNKER_VERSION,
            "qrel_generator": QREL_GENERATOR_VERSION,
        },
        "caps": asdict(contract.caps),
        "source_sha256": dict(sorted(artifacts.source_hashes.items())),
        "archive_download_bytes": artifacts.source_bytes["archive"],
        "tracked_blob_count": len(artifacts.file_audits),
        "tracked_blob_bytes": sum(audit.raw_bytes for audit in artifacts.file_audits),
        "extracted_regular_file_bytes": sum(audit.raw_bytes for audit in artifacts.file_audits),
        "largest_stage_a_candidate_bytes": max(
            (audit.raw_bytes for audit in artifacts.file_audits if audit.stage_a_result == "accepted"),
            default=0,
        ),
        "eligible_text_file_count": len(artifacts.files),
        "eligible_normalized_text_bytes": sum(int(file.audit.normalized_utf8_bytes or 0) for file in artifacts.files),
        "chunk_count": len(artifacts.chunks),
        "patch_target_count": len(artifacts.patch_targets),
        "qrel_count": len(artifacts.qrels),
        "candidate_coverage": len(mapped_units) / len(target_units) if target_units else 0.0,
        "stage_a_counts": dict(sorted(stage_a_counts.items())),
        "stage_b_counts": dict(sorted(stage_b_counts.items())),
        "chunk_family_counts": dict(sorted(chunk_family_counts.items())),
        "file_audit_manifest_sha256": _manifest_sha256(file_records),
        "corpus_manifest_sha256": _manifest_sha256(corpus_records),
        "chunk_manifest_sha256": _manifest_sha256(chunk_records),
        "qrel_manifest_sha256": _manifest_sha256(qrel_records),
    }
    summary["artifact_summary_sha256"] = _sha256_text(_canonical_json(summary))
    return summary


def serialize_artifact_summary(summary: Mapping[str, Any]) -> str:
    """Serialize a source-free artifact summary canonically."""
    return _canonical_json(summary)


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _materialize(
    contract: SourceContract,
    workdir: Path,
    fetch_bytes: FetchBytes,
    deadline: _RunDeadline,
) -> dict[str, Any]:
    sources = _fetch_sources(contract, fetch_bytes)
    deadline.check()
    tree_entries, _query_text = _validate_metadata(contract, sources)
    deadline.check()
    extracted = _extract_verified_archive(
        contract,
        sources["archive"],
        tree_entries,
        workdir / "snapshot",
    )
    deadline.check()
    audits, files = audit_eligible_files(contract, tree_entries, extracted)
    deadline.check()
    chunks = build_chunks(contract, files)
    deadline.check()
    targets = parse_patch_targets(contract, sources["patch"], files)
    deadline.check()
    qrels = map_targets_to_qrels(targets, chunks)
    artifacts = MaterializationArtifacts(
        source_hashes={name: _sha256_bytes(content) for name, content in sources.items()},
        source_bytes={name: len(content) for name, content in sources.items()},
        file_audits=audits,
        files=files,
        chunks=chunks,
        patch_targets=targets,
        qrels=qrels,
    )
    validate_materialization_artifacts(contract, artifacts)
    return build_artifact_summary(contract, artifacts)


def run_source_materialization(
    contract: SourceContract,
    *,
    fetch_bytes: FetchBytes | None = None,
    temp_root: str | Path | None = None,
    temp_prefix: str = "meb-code-edit-source-",
) -> dict[str, Any]:
    """Run one bounded source contract and always remove its dedicated path."""
    _validate_contract(contract)
    started = time.monotonic()
    workdir = Path(tempfile.mkdtemp(prefix=temp_prefix, dir=temp_root))
    cleanup_path = str(workdir)
    status = "FAILED"
    error: dict[str, str] | None = None
    artifacts: dict[str, Any] | None = None
    deadline = _RunDeadline(contract.caps.wall_seconds)
    try:
        deadline.start()
        artifacts = _materialize(contract, workdir, fetch_bytes or _default_fetch_bytes, deadline)
        deadline.check()
        peak_rss = _peak_rss_bytes()
        if peak_rss > contract.caps.target_rss_bytes:
            raise SourceMaterializationError(
                "rss_cap",
                f"Peak RSS {peak_rss} exceeds target {contract.caps.target_rss_bytes}",
            )
        status = "PASS"
    except SourceMaterializationError as exc:
        status = exc.status
        error = {"code": exc.code, "message": str(exc)}
    except Exception as exc:
        status = "FAILED"
        error = {"code": "unexpected_error", "message": f"Unexpected materialization failure: {exc}"}
    finally:
        deadline.begin_cleanup()
        try:
            shutil.rmtree(workdir, ignore_errors=False)
        except OSError as exc:
            status = "FAILED"
            error = {"code": "cleanup_failed", "message": f"Could not remove dedicated path: {exc}"}
        finally:
            deadline.restore()
    cleanup_absent = not workdir.exists()
    if not cleanup_absent:
        status = "FAILED"
        error = {"code": "cleanup_failed", "message": f"Dedicated path still exists: {cleanup_path}"}
    result = {
        "status": status,
        "artifacts": artifacts,
        "error": error,
        "runtime": {
            "wall_seconds": round(time.monotonic() - started, 6),
            "peak_rss_bytes": _peak_rss_bytes(),
            "cleanup_path": cleanup_path,
            "cleanup_absent": cleanup_absent,
        },
    }
    return result
