"""Deterministic invented fixture for issue-to-edit chunk localization."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable


DATASET_VERSION = "code-edit-chunk-localization-fixture-v0"
SOURCE_KIND = "self_created_issue_patch_repository_fixture"
LICENSE_STATUS = "self_created_fixture_not_for_publication"
SPLIT = "fixture_only"
REPOSITORY_ID = "cedar-harbor/relay-control"
BASE_COMMIT = "4f2b7a1c6d8e9031a5b4c7d2e6f80913a4b5c6d7"
CHUNKER_VERSION = "code-edit-ast-fallback-fixture-v0"
QREL_GENERATOR_VERSION = "patch-preimage-and-insertion-anchor-fixture-v0"
NORMALIZATION = "utf8-lf-v1"
PUBLIC_SCORE_ELIGIBLE = False
EXPECTED_SERIALIZATION_SHA256 = "32cfe40ad170e26405b82e3ed80e431516fd0e2bd2938ab0f5244058d7a37233"


@dataclass(frozen=True)
class RepositorySnapshot:
    """Identity and eligibility summary for the invented repository."""

    repository_id: str
    base_commit: str
    tree_sha: str
    repository_license_spdx: str
    eligible_text_file_count: int
    eligible_normalized_text_bytes: int
    source_kind: str
    source_revision: str
    source_audit_status: str
    public_redistribution: bool
    public_score_eligible: bool
    snapshot_sha256: str


@dataclass(frozen=True)
class RepositoryFile:
    """One eligible normalized file from the invented base snapshot."""

    file_id: str
    repository_id: str
    base_commit: str
    path: str
    path_family: str
    language: str
    git_mode: str
    blob_sha: str
    normalized_text: str
    normalized_bytes: int
    text_sha256: str
    source_kind: str
    source_revision: str
    license_status: str
    public_score_eligible: bool


@dataclass(frozen=True)
class EditQuery:
    """An invented issue used as a retrieval query."""

    query_id: str
    repository_id: str
    title: str
    body: str
    text: str
    issue_type: str
    edit_type: str
    language: str
    source_family: str
    source_id: str
    query_text_sha256: str
    split: str
    answer_leak_review: str
    privacy_review: str
    public_score_eligible: bool


@dataclass(frozen=True)
class CodeChunk:
    """A stable definition-level or fixed-line retrieval candidate."""

    chunk_id: str
    repository_id: str
    base_commit: str
    file_id: str
    path: str
    path_family: str
    blob_sha: str
    language: str
    candidate_family: str
    symbol: str | None
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    text: str
    text_sha256: str
    chunker_version: str
    ordinal_in_file: int
    source_kind: str
    source_revision: str
    license_status: str
    public_score_eligible: bool


@dataclass(frozen=True)
class PatchTarget:
    """One patch preimage range or insertion anchor to map into chunks."""

    target_id: str
    query_id: str
    repository_id: str
    base_commit: str
    changed_path: str
    patch_change_type: str
    preimage_line_start: int | None
    preimage_line_end: int | None
    insertion_anchor_after_line: int | None
    preimage_text_sha256: str | None
    insertion_anchor_text_sha256: str | None
    patch_raw_sha256: str
    mapping_status: str
    source_kind: str
    source_revision: str
    public_score_eligible: bool


@dataclass(frozen=True)
class EditQrel:
    """A direct grade-2 patch-to-chunk relevance judgment."""

    qrel_id: str
    query_id: str
    chunk_id: str
    target_id: str
    target_unit_ids: tuple[str, ...]
    relevance: int
    label_family: str
    changed_path: str
    patch_change_type: str
    overlap_lines: int
    mapping_status: str
    patch_raw_sha256: str
    source_kind: str
    source_revision: str
    public_score_eligible: bool


@dataclass(frozen=True)
class HardNegative:
    """An audited query-specific hard-negative link."""

    hard_negative_id: str
    query_id: str
    chunk_id: str
    source_chunk_id: str
    negative_family: str
    reason: str
    false_negative_review: str
    review_metadata: str
    source_kind: str
    source_revision: str
    public_score_eligible: bool


@dataclass(frozen=True)
class ProvenanceRecord:
    """Local provenance for an invented issue, patch, and snapshot tuple."""

    record_id: str
    query_id: str
    issue_source_id: str
    issue_content_sha256: str
    patch_source_id: str
    patch_content_sha256: str
    repository_source_id: str
    repository_snapshot_sha256: str
    normalization: str
    chunker_version: str
    qrel_generator_version: str
    review_status: str
    source_kind: str
    source_revision: str
    license_status: str
    public_score_eligible: bool


@dataclass(frozen=True)
class CodeEditChunkLocalizationFixture:
    """Complete fixture contract for full-corpus edit-chunk retrieval."""

    dataset_version: str
    split: str
    fixture_only: bool
    license_status: str
    leaderboard_publish: bool
    network: str
    provider_api_calls: int
    model_downloads: int
    repository: RepositorySnapshot
    files: tuple[RepositoryFile, ...]
    queries: tuple[EditQuery, ...]
    chunks: tuple[CodeChunk, ...]
    patch_targets: tuple[PatchTarget, ...]
    qrels: tuple[EditQrel, ...]
    hard_negatives: tuple[HardNegative, ...]
    provenance: tuple[ProvenanceRecord, ...]
    serialization_sha256: str


@dataclass(frozen=True)
class _FileSpec:
    path: str
    path_family: str
    language: str
    text: str


@dataclass(frozen=True)
class _TargetSpec:
    target_id: str
    query_id: str
    changed_path: str
    patch_change_type: str
    preimage_lines: tuple[str, ...] = ()
    insertion_anchor_line: str | None = None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_blob_sha(text: str) -> str:
    raw = text.encode("utf-8")
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized if normalized.endswith("\n") else f"{normalized}\n"


def _fixture_file_specs() -> tuple[_FileSpec, ...]:
    docs_lines = [
        "# Harbor Control Operations",
        "",
        "Queue retries use a short first delay and increase only after repeated saturation.",
        "The queue report summarizes retry counts but does not choose retry timing.",
        "",
        "Cleanup removes bundles after the active retention period.",
        "Legacy grace behavior is described only for migration history.",
        "",
        "Alert audit markers identify escalated notices after recipient validation.",
        "The marker description does not define the configuration edit location.",
        "",
        "Operators review retry, cleanup, and alert events in separate dashboards.",
    ]
    return (
        _FileSpec(
            path="src/harbor/queue.py",
            path_family="implementation",
            language="python",
            text='''"""Queue retry helpers for Harbor Control."""

DEFAULT_RETRY_SECONDS = 30


def normalize_queue_name(name: str) -> str:
    """Return the canonical queue label."""
    return name.strip().lower()


def choose_retry_delay(attempt: int) -> int:
    """Return retry delay seconds for a transient saturation."""
    base_delay = 30
    return base_delay * max(1, attempt)


def schedule_retry(queue_name: str, attempt: int) -> tuple[str, int]:
    """Build a retry schedule entry."""
    return normalize_queue_name(queue_name), choose_retry_delay(attempt)
''',
        ),
        _FileSpec(
            path="src/harbor/cleanup.py",
            path_family="implementation",
            language="python",
            text='''"""Bundle cleanup policy for Harbor Control."""

LEGACY_GRACE_DAYS = 7


def should_keep_bundle(age_days: int, retention_days: int) -> bool:
    """Return whether a bundle remains inside the supported retention window."""
    if age_days <= retention_days:
        return True
    if age_days <= retention_days + LEGACY_GRACE_DAYS:
        return True
    return False


def remove_stale_bundles(bundle_ages: list[int], retention_days: int) -> list[int]:
    """Return only bundle ages that are still retained."""
    return [age for age in bundle_ages if should_keep_bundle(age, retention_days)]
''',
        ),
        _FileSpec(
            path="src/harbor/alerts.py",
            path_family="implementation",
            language="python",
            text='''"""Escalated alert construction helpers."""


def validate_recipient(recipient: str) -> str:
    """Reject an empty escalation recipient."""
    if not recipient.strip():
        raise ValueError("recipient is required")
    return recipient.strip()


def build_escalated_notice(recipient: str, message: str) -> dict[str, str]:
    """Build the current escalated notice payload."""
    validated = validate_recipient(recipient)
    return {"recipient": validated, "message": message}
''',
        ),
        _FileSpec(
            path="src/harbor/reporting.py",
            path_family="implementation",
            language="python",
            text='''"""Operational summaries that intentionally do not edit policy."""


def retry_summary(attempts: list[int]) -> str:
    """Describe observed retry attempts without selecting their delays."""
    return f"retry-count={len(attempts)}"


def audit_marker_summary(markers: list[str]) -> str:
    """Describe alert markers without constructing alert payloads."""
    return f"marker-count={len(markers)}"
''',
        ),
        _FileSpec(
            path="src/harbor/storage.py",
            path_family="implementation",
            language="python",
            text='''"""Storage records used by cleanup operations."""


class BundleStore:
    """Keep invented bundle ages for a local cleanup run."""

    def __init__(self, ages: list[int]) -> None:
        self.ages = ages

    def snapshot(self) -> list[int]:
        return list(self.ages)
''',
        ),
        _FileSpec(
            path="tests/test_queue.py",
            path_family="test",
            language="python",
            text='''"""Invented queue behavior tests."""

from harbor.queue import choose_retry_delay, normalize_queue_name


def test_first_retry_uses_default_delay() -> None:
    assert choose_retry_delay(1) == 30


def test_queue_names_are_normalized() -> None:
    assert normalize_queue_name(" Primary ") == "primary"
''',
        ),
        _FileSpec(
            path="config/alerts.yaml",
            path_family="configuration",
            language="yaml",
            text='''alerts:
  escalation:
    enabled: true
    recipient_policy: validated
    severity: high
    audit_channel: review
  routine:
    enabled: true
    severity: normal
''',
        ),
        _FileSpec(
            path="docs/operations.md",
            path_family="documentation",
            language="markdown",
            text="\n".join(docs_lines),
        ),
    )


def _build_files() -> tuple[RepositoryFile, ...]:
    files = []
    for spec in _fixture_file_specs():
        text = _normalize_text(spec.text)
        files.append(
            RepositoryFile(
                file_id=f"{REPOSITORY_ID}@{BASE_COMMIT[:12]}:{spec.path}",
                repository_id=REPOSITORY_ID,
                base_commit=BASE_COMMIT,
                path=spec.path,
                path_family=spec.path_family,
                language=spec.language,
                git_mode="100644",
                blob_sha=_git_blob_sha(text),
                normalized_text=text,
                normalized_bytes=len(text.encode("utf-8")),
                text_sha256=_sha256_text(text),
                source_kind=SOURCE_KIND,
                source_revision=DATASET_VERSION,
                license_status=LICENSE_STATUS,
                public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
            )
        )
    return tuple(files)


def _tree_sha(files: Iterable[RepositoryFile]) -> str:
    payload = "\n".join(f"{item.git_mode} {item.path}\0{item.blob_sha}" for item in sorted(files, key=lambda x: x.path))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _snapshot_sha(files: Iterable[RepositoryFile], tree_sha: str) -> str:
    payload = {
        "repository_id": REPOSITORY_ID,
        "base_commit": BASE_COMMIT,
        "tree_sha": tree_sha,
        "files": [(item.path, item.blob_sha, item.text_sha256) for item in sorted(files, key=lambda x: x.path)],
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _line_char_range(text: str, line_start: int, line_end: int) -> tuple[int, int, str]:
    lines = text.splitlines(keepends=True)
    if line_start < 1 or line_end < line_start or line_end > len(lines):
        raise ValueError(f"Invalid line range {line_start}-{line_end}")
    char_start = sum(len(line) for line in lines[: line_start - 1])
    char_end = sum(len(line) for line in lines[:line_end])
    return char_start, char_end, text[char_start:char_end]


def _python_chunk_ranges(file: RepositoryFile) -> tuple[tuple[str, str | None, int, int], ...]:
    tree = ast.parse(file.normalized_text, filename=file.path)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not definitions:
        return ()
    ranges: list[tuple[str, str | None, int, int]] = []
    first_line = min(node.lineno for node in definitions)
    if any(line.strip() for line in file.normalized_text.splitlines()[: first_line - 1]):
        ranges.append(("module_preamble", None, 1, first_line - 1))
    for node in definitions:
        family = "ast_class" if isinstance(node, ast.ClassDef) else "ast_function"
        ranges.append((family, node.name, node.lineno, int(node.end_lineno or node.lineno)))
    return tuple(ranges)


def _fallback_chunk_ranges(file: RepositoryFile, window_lines: int = 4) -> tuple[tuple[str, str | None, int, int], ...]:
    line_count = len(file.normalized_text.splitlines())
    return tuple(
        ("line_fallback", None, line_start, min(line_count, line_start + window_lines - 1))
        for line_start in range(1, line_count + 1, window_lines)
    )


def _build_chunks(files: tuple[RepositoryFile, ...]) -> tuple[CodeChunk, ...]:
    chunks: list[CodeChunk] = []
    for file in files:
        ranges = _python_chunk_ranges(file) if file.language == "python" else _fallback_chunk_ranges(file)
        if not ranges:
            ranges = _fallback_chunk_ranges(file)
        for ordinal, (family, symbol, line_start, line_end) in enumerate(ranges):
            char_start, char_end, text = _line_char_range(file.normalized_text, line_start, line_end)
            symbol_part = symbol or "none"
            chunk_id = (
                f"{REPOSITORY_ID}@{BASE_COMMIT[:12]}:{file.path}:"
                f"{family}:{symbol_part}:{ordinal:04d}"
            )
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    repository_id=REPOSITORY_ID,
                    base_commit=BASE_COMMIT,
                    file_id=file.file_id,
                    path=file.path,
                    path_family=file.path_family,
                    blob_sha=file.blob_sha,
                    language=file.language,
                    candidate_family=family,
                    symbol=symbol,
                    line_start=line_start,
                    line_end=line_end,
                    char_start=char_start,
                    char_end=char_end,
                    text=text,
                    text_sha256=_sha256_text(text),
                    chunker_version=CHUNKER_VERSION,
                    ordinal_in_file=ordinal,
                    source_kind=SOURCE_KIND,
                    source_revision=DATASET_VERSION,
                    license_status=LICENSE_STATUS,
                    public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
                )
            )
    return tuple(chunks)


def _build_queries() -> tuple[EditQuery, ...]:
    specs = (
        (
            "q_retry_delay_replacement",
            "Retry saturated queues sooner",
            "The first saturated queue retry waits too long. Shorten the first delay and update its focused test "
            "while preserving later backoff.",
            "bug",
            "replacement",
            "issue_fixture_001",
        ),
        (
            "q_legacy_grace_deletion",
            "Remove obsolete cleanup grace behavior",
            "Bundle cleanup still retains expired entries through an obsolete grace branch. Remove that legacy "
            "branch without changing active retention.",
            "refactor",
            "deletion",
            "issue_fixture_002",
        ),
        (
            "q_alert_audit_insertion",
            "Record audit markers for escalated notices",
            "Escalated notices need an audit marker after the reviewed channel setting so downstream review can "
            "distinguish them from routine alerts.",
            "feature",
            "insertion_only",
            "issue_fixture_003",
        ),
    )
    queries = []
    for query_id, title, body, issue_type, edit_type, source_id in specs:
        text = f"{title}\n\n{body}"
        queries.append(
            EditQuery(
                query_id=query_id,
                repository_id=REPOSITORY_ID,
                title=title,
                body=body,
                text=text,
                issue_type=issue_type,
                edit_type=edit_type,
                language="en",
                source_family="local_invented_issue_patch_pair",
                source_id=source_id,
                query_text_sha256=_sha256_text(text),
                split=SPLIT,
                answer_leak_review="pass",
                privacy_review="pass",
                public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
            )
        )
    return tuple(queries)


def _find_contiguous_lines(text: str, expected: tuple[str, ...]) -> tuple[int, int]:
    lines = text.splitlines()
    for index in range(len(lines) - len(expected) + 1):
        if tuple(lines[index : index + len(expected)]) == expected:
            return index + 1, index + len(expected)
    raise ValueError(f"Could not find target lines: {expected!r}")


def _target_specs() -> tuple[_TargetSpec, ...]:
    return (
        _TargetSpec(
            target_id="target_retry_delay_impl",
            query_id="q_retry_delay_replacement",
            changed_path="src/harbor/queue.py",
            patch_change_type="replacement",
            preimage_lines=("    base_delay = 30", "    return base_delay * max(1, attempt)"),
        ),
        _TargetSpec(
            target_id="target_retry_delay_test",
            query_id="q_retry_delay_replacement",
            changed_path="tests/test_queue.py",
            patch_change_type="replacement",
            preimage_lines=("    assert choose_retry_delay(1) == 30",),
        ),
        _TargetSpec(
            target_id="target_cleanup_legacy_branch",
            query_id="q_legacy_grace_deletion",
            changed_path="src/harbor/cleanup.py",
            patch_change_type="deletion",
            preimage_lines=(
                "    if age_days <= retention_days + LEGACY_GRACE_DAYS:",
                "        return True",
            ),
        ),
        _TargetSpec(
            target_id="target_alert_audit_anchor",
            query_id="q_alert_audit_insertion",
            changed_path="config/alerts.yaml",
            patch_change_type="insertion_only",
            insertion_anchor_line="    audit_channel: review",
        ),
    )


def _patch_sha(query_id: str, specs: Iterable[_TargetSpec]) -> str:
    payload = [asdict(spec) for spec in specs if spec.query_id == query_id]
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _build_patch_targets(files: tuple[RepositoryFile, ...]) -> tuple[PatchTarget, ...]:
    file_by_path = {file.path: file for file in files}
    specs = _target_specs()
    targets: list[PatchTarget] = []
    for spec in specs:
        file = file_by_path[spec.changed_path]
        patch_sha = _patch_sha(spec.query_id, specs)
        if spec.preimage_lines:
            line_start, line_end = _find_contiguous_lines(file.normalized_text, spec.preimage_lines)
            preimage_text = "\n".join(spec.preimage_lines)
            anchor_line = None
            anchor_sha = None
            preimage_sha = _sha256_text(preimage_text)
        else:
            line_start = None
            line_end = None
            anchor_start, anchor_end = _find_contiguous_lines(
                file.normalized_text,
                (str(spec.insertion_anchor_line),),
            )
            if anchor_start != anchor_end:
                raise ValueError("Insertion anchor must resolve to exactly one line")
            anchor_line = anchor_start
            anchor_sha = _sha256_text(str(spec.insertion_anchor_line))
            preimage_sha = None
        targets.append(
            PatchTarget(
                target_id=spec.target_id,
                query_id=spec.query_id,
                repository_id=REPOSITORY_ID,
                base_commit=BASE_COMMIT,
                changed_path=spec.changed_path,
                patch_change_type=spec.patch_change_type,
                preimage_line_start=line_start,
                preimage_line_end=line_end,
                insertion_anchor_after_line=anchor_line,
                preimage_text_sha256=preimage_sha,
                insertion_anchor_text_sha256=anchor_sha,
                patch_raw_sha256=patch_sha,
                mapping_status="exact",
                source_kind=SOURCE_KIND,
                source_revision=DATASET_VERSION,
                public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
            )
        )
    return tuple(targets)


def target_unit_ids(target: PatchTarget) -> tuple[str, ...]:
    """Return atomic modified/deleted lines or one insertion-anchor unit."""
    if target.preimage_line_start is not None and target.preimage_line_end is not None:
        return tuple(
            f"{target.target_id}:line:{line_number}"
            for line_number in range(target.preimage_line_start, target.preimage_line_end + 1)
        )
    if target.insertion_anchor_after_line is not None:
        return (f"{target.target_id}:anchor:{target.insertion_anchor_after_line}",)
    raise ValueError(f"Target {target.target_id} has no measurable edit units")


def generate_qrels(
    patch_targets: tuple[PatchTarget, ...],
    chunks: tuple[CodeChunk, ...],
) -> tuple[EditQrel, ...]:
    """Map patch preimage lines and insertion anchors into the fixed corpus."""
    chunks_by_path: dict[str, list[CodeChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_path[chunk.path].append(chunk)

    qrels: list[EditQrel] = []
    for target in patch_targets:
        path_chunks = chunks_by_path[target.changed_path]
        if target.preimage_line_start is not None and target.preimage_line_end is not None:
            mapped = [
                chunk
                for chunk in path_chunks
                if chunk.line_start <= target.preimage_line_end and chunk.line_end >= target.preimage_line_start
            ]
            label_family = (
                "deleted_preimage_line" if target.patch_change_type == "deletion" else "modified_preimage_line"
            )
        else:
            anchor = int(target.insertion_anchor_after_line or 0)
            mapped = [chunk for chunk in path_chunks if chunk.line_start <= anchor <= chunk.line_end]
            label_family = "insert_anchor_containing_chunk"
        if len(mapped) != 1:
            raise ValueError(f"Target {target.target_id} must map to exactly one fixed chunk")
        chunk = mapped[0]
        overlap_lines = 0
        if target.preimage_line_start is not None and target.preimage_line_end is not None:
            overlap_lines = max(
                0,
                min(chunk.line_end, target.preimage_line_end) - max(chunk.line_start, target.preimage_line_start) + 1,
            )
        qrel_identity = f"{target.query_id}\0{chunk.chunk_id}\0{target.target_id}"
        qrels.append(
            EditQrel(
                qrel_id=f"qrel_{_sha256_text(qrel_identity)[:16]}",
                query_id=target.query_id,
                chunk_id=chunk.chunk_id,
                target_id=target.target_id,
                target_unit_ids=target_unit_ids(target),
                relevance=2,
                label_family=label_family,
                changed_path=target.changed_path,
                patch_change_type=target.patch_change_type,
                overlap_lines=overlap_lines,
                mapping_status=target.mapping_status,
                patch_raw_sha256=target.patch_raw_sha256,
                source_kind=SOURCE_KIND,
                source_revision=DATASET_VERSION,
                public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
            )
        )
    return tuple(qrels)


def _chunk_by_path_symbol(
    chunks: tuple[CodeChunk, ...],
    path: str,
    symbol: str | None,
    ordinal: int | None = None,
) -> CodeChunk:
    matches = [
        chunk
        for chunk in chunks
        if chunk.path == path
        and chunk.symbol == symbol
        and (ordinal is None or chunk.ordinal_in_file == ordinal)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one chunk for {path}/{symbol}/{ordinal}")
    return matches[0]


def _build_hard_negatives(
    chunks: tuple[CodeChunk, ...],
    qrels: tuple[EditQrel, ...],
) -> tuple[HardNegative, ...]:
    positive_by_query: dict[str, list[str]] = defaultdict(list)
    for qrel in qrels:
        positive_by_query[qrel.query_id].append(qrel.chunk_id)

    specs = (
        (
            "q_retry_delay_replacement",
            "src/harbor/queue.py",
            "normalize_queue_name",
            None,
            "same_file_neighbor",
            "Nearby queue helper is unedited and does not select retry timing.",
        ),
        (
            "q_retry_delay_replacement",
            "src/harbor/reporting.py",
            "retry_summary",
            None,
            "same_symbol_family",
            "Shares retry vocabulary but only summarizes observed attempts.",
        ),
        (
            "q_retry_delay_replacement",
            "docs/operations.md",
            None,
            0,
            "documentation_code_collision",
            "Describes retry behavior but is not a patch-aligned code target.",
        ),
        (
            "q_legacy_grace_deletion",
            "src/harbor/cleanup.py",
            "remove_stale_bundles",
            None,
            "same_file_neighbor",
            "Calls cleanup policy but does not contain the obsolete grace branch.",
        ),
        (
            "q_legacy_grace_deletion",
            "src/harbor/storage.py",
            "BundleStore",
            None,
            "same_subsystem_path",
            "Stores bundle ages without deciding retention or grace behavior.",
        ),
        (
            "q_legacy_grace_deletion",
            "docs/operations.md",
            None,
            1,
            "documentation_code_collision",
            "Mentions cleanup history but is not the deleted implementation branch.",
        ),
        (
            "q_alert_audit_insertion",
            "src/harbor/alerts.py",
            "build_escalated_notice",
            None,
            "error_message_collision",
            "Builds a related notice but does not contain the configuration insertion anchor.",
        ),
        (
            "q_alert_audit_insertion",
            "src/harbor/reporting.py",
            "audit_marker_summary",
            None,
            "same_symbol_family",
            "Shares audit-marker wording but only reports marker counts.",
        ),
        (
            "q_alert_audit_insertion",
            "docs/operations.md",
            None,
            2,
            "documentation_code_collision",
            "Describes marker intent but is not the exact configuration anchor.",
        ),
    )
    negatives: list[HardNegative] = []
    for query_id, path, symbol, ordinal, family, reason in specs:
        chunk = _chunk_by_path_symbol(chunks, path, symbol, ordinal)
        source_chunk_id = sorted(positive_by_query[query_id])[0]
        identity = f"{query_id}\0{chunk.chunk_id}\0{family}"
        negatives.append(
            HardNegative(
                hard_negative_id=f"hard_{_sha256_text(identity)[:16]}",
                query_id=query_id,
                chunk_id=chunk.chunk_id,
                source_chunk_id=source_chunk_id,
                negative_family=family,
                reason=reason,
                false_negative_review="pass",
                review_metadata="invented_fixture_manual_contract_review_v0",
                source_kind=SOURCE_KIND,
                source_revision=DATASET_VERSION,
                public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
            )
        )
    return tuple(negatives)


def _build_provenance(
    queries: tuple[EditQuery, ...],
    targets: tuple[PatchTarget, ...],
    repository: RepositorySnapshot,
) -> tuple[ProvenanceRecord, ...]:
    targets_by_query: dict[str, list[PatchTarget]] = defaultdict(list)
    for target in targets:
        targets_by_query[target.query_id].append(target)
    records = []
    for query in queries:
        patch_sha = {target.patch_raw_sha256 for target in targets_by_query[query.query_id]}
        if len(patch_sha) != 1:
            raise ValueError(f"Query {query.query_id} must have one stable patch identity")
        records.append(
            ProvenanceRecord(
                record_id=f"prov_{query.query_id}_v0",
                query_id=query.query_id,
                issue_source_id=query.source_id,
                issue_content_sha256=query.query_text_sha256,
                patch_source_id=f"patch_{query.query_id}_v0",
                patch_content_sha256=next(iter(patch_sha)),
                repository_source_id="repository_fixture_cedar_harbor_v0",
                repository_snapshot_sha256=repository.snapshot_sha256,
                normalization=NORMALIZATION,
                chunker_version=CHUNKER_VERSION,
                qrel_generator_version=QREL_GENERATOR_VERSION,
                review_status="invented_content_review_pass",
                source_kind=SOURCE_KIND,
                source_revision=DATASET_VERSION,
                license_status=LICENSE_STATUS,
                public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
            )
        )
    return tuple(records)


def fixture_to_dict(fixture: CodeEditChunkLocalizationFixture) -> dict[str, Any]:
    """Return a JSON-compatible representation with deterministic field order."""
    return asdict(fixture)


def serialize_fixture(fixture: CodeEditChunkLocalizationFixture) -> str:
    """Serialize every fixture identity and label byte-for-byte deterministically."""
    return json.dumps(fixture_to_dict(fixture), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fixture_digest(fixture: CodeEditChunkLocalizationFixture) -> str:
    payload = serialize_fixture(replace(fixture, serialization_sha256=""))
    return _sha256_text(payload)


def fixture_counts(fixture: CodeEditChunkLocalizationFixture) -> dict[str, int]:
    """Return stable repository and label counts for audit details."""
    return {
        "n_files": len(fixture.files),
        "n_queries": len(fixture.queries),
        "n_chunks": len(fixture.chunks),
        "n_patch_targets": len(fixture.patch_targets),
        "n_qrels": len(fixture.qrels),
        "n_hard_negatives": len(fixture.hard_negatives),
        "n_provenance_records": len(fixture.provenance),
    }


def fixture_slice_counts(fixture: CodeEditChunkLocalizationFixture) -> dict[str, dict[str, int]]:
    """Return auditable query, corpus, mapping, and hard-negative slice counts."""
    return {
        "edit_type": dict(sorted(Counter(query.edit_type for query in fixture.queries).items())),
        "issue_type": dict(sorted(Counter(query.issue_type for query in fixture.queries).items())),
        "file_path_family": dict(sorted(Counter(file.path_family for file in fixture.files).items())),
        "candidate_family": dict(sorted(Counter(chunk.candidate_family for chunk in fixture.chunks).items())),
        "qrel_mapping_status": dict(sorted(Counter(qrel.mapping_status for qrel in fixture.qrels).items())),
        "hard_negative_family": dict(
            sorted(Counter(negative.negative_family for negative in fixture.hard_negatives).items())
        ),
    }


def serialize_chunk_document(chunk: CodeChunk) -> str:
    """Serialize one fixed chunk for the flat document embedding interface."""
    symbol = chunk.symbol or "none"
    return (
        f"Repository: {chunk.repository_id}\nPath: {chunk.path}\n"
        f"Candidate family: {chunk.candidate_family}\nSymbol: {symbol}\nContent:\n{chunk.text}"
    )


def _build_fixture() -> CodeEditChunkLocalizationFixture:
    files = _build_files()
    tree_sha = _tree_sha(files)
    repository = RepositorySnapshot(
        repository_id=REPOSITORY_ID,
        base_commit=BASE_COMMIT,
        tree_sha=tree_sha,
        repository_license_spdx="LicenseRef-Self-Created-Fixture",
        eligible_text_file_count=len(files),
        eligible_normalized_text_bytes=sum(file.normalized_bytes for file in files),
        source_kind=SOURCE_KIND,
        source_revision=DATASET_VERSION,
        source_audit_status="self_created_content_reviewed",
        public_redistribution=False,
        public_score_eligible=PUBLIC_SCORE_ELIGIBLE,
        snapshot_sha256=_snapshot_sha(files, tree_sha),
    )
    queries = _build_queries()
    chunks = _build_chunks(files)
    targets = _build_patch_targets(files)
    qrels = generate_qrels(targets, chunks)
    hard_negatives = _build_hard_negatives(chunks, qrels)
    provenance = _build_provenance(queries, targets, repository)
    fixture = CodeEditChunkLocalizationFixture(
        dataset_version=DATASET_VERSION,
        split=SPLIT,
        fixture_only=True,
        license_status=LICENSE_STATUS,
        leaderboard_publish=False,
        network="forbidden",
        provider_api_calls=0,
        model_downloads=0,
        repository=repository,
        files=files,
        queries=queries,
        chunks=chunks,
        patch_targets=targets,
        qrels=qrels,
        hard_negatives=hard_negatives,
        provenance=provenance,
        serialization_sha256="",
    )
    return replace(fixture, serialization_sha256=_fixture_digest(fixture))


def validate_code_edit_chunk_localization_fixture(fixture: CodeEditChunkLocalizationFixture) -> None:
    """Reject identity, mapping, provenance, hard-negative, and publication drift."""
    if (
        fixture.dataset_version != DATASET_VERSION
        or fixture.split != SPLIT
        or not fixture.fixture_only
        or fixture.license_status != LICENSE_STATUS
        or fixture.leaderboard_publish
        or fixture.network != "forbidden"
        or fixture.provider_api_calls != 0
        or fixture.model_downloads != 0
    ):
        raise ValueError("Fixture-only provenance or no-publish contract changed")
    if fixture_counts(fixture) != {
        "n_files": 8,
        "n_queries": 3,
        "n_chunks": 24,
        "n_patch_targets": 4,
        "n_qrels": 4,
        "n_hard_negatives": 9,
        "n_provenance_records": 3,
    }:
        raise ValueError("Unexpected code-edit fixture shape")

    if fixture.repository.repository_id != REPOSITORY_ID or fixture.repository.base_commit != BASE_COMMIT:
        raise ValueError("Repository identity changed")
    if fixture.repository.public_redistribution or fixture.repository.public_score_eligible:
        raise ValueError("Fixture repository cannot be public-score eligible")

    files_by_path = {file.path: file for file in fixture.files}
    if len(files_by_path) != len(fixture.files):
        raise ValueError("Duplicate repository file path")
    if fixture.repository.eligible_text_file_count != len(fixture.files):
        raise ValueError("Eligible file count does not cover the complete fixture repository")
    if fixture.repository.eligible_normalized_text_bytes != sum(file.normalized_bytes for file in fixture.files):
        raise ValueError("Eligible byte count does not cover the complete fixture repository")
    if fixture.repository.tree_sha != _tree_sha(fixture.files):
        raise ValueError("Repository tree identity changed")

    file_ids: set[str] = set()
    for file in fixture.files:
        if file.file_id in file_ids:
            raise ValueError("Duplicate file id")
        file_ids.add(file.file_id)
        if (
            file.repository_id != REPOSITORY_ID
            or file.base_commit != BASE_COMMIT
            or file.source_kind != SOURCE_KIND
            or file.source_revision != DATASET_VERSION
            or file.license_status != LICENSE_STATUS
            or file.public_score_eligible
        ):
            raise ValueError(f"Invalid file provenance for {file.path}")
        if file.normalized_text != _normalize_text(file.normalized_text):
            raise ValueError(f"Invalid newline normalization for {file.path}")
        if file.normalized_bytes != len(file.normalized_text.encode("utf-8")):
            raise ValueError(f"Invalid byte count for {file.path}")
        if (
            file.blob_sha != _git_blob_sha(file.normalized_text)
            or file.text_sha256 != _sha256_text(file.normalized_text)
        ):
            raise ValueError(f"File hash mismatch for {file.path}")
    if fixture.repository.snapshot_sha256 != _snapshot_sha(fixture.files, fixture.repository.tree_sha):
        raise ValueError("Repository snapshot hash changed")

    chunks_by_path: dict[str, list[CodeChunk]] = defaultdict(list)
    chunk_by_id: dict[str, CodeChunk] = {}
    for chunk in fixture.chunks:
        if chunk.chunk_id in chunk_by_id:
            raise ValueError("Duplicate chunk id")
        chunk_by_id[chunk.chunk_id] = chunk
        chunks_by_path[chunk.path].append(chunk)
        file = files_by_path.get(chunk.path)
        if file is None or chunk.file_id != file.file_id or chunk.blob_sha != file.blob_sha:
            raise ValueError(f"Chunk file identity mismatch for {chunk.chunk_id}")
        expected_id = (
            f"{REPOSITORY_ID}@{BASE_COMMIT[:12]}:{chunk.path}:"
            f"{chunk.candidate_family}:{chunk.symbol or 'none'}:{chunk.ordinal_in_file:04d}"
        )
        if chunk.chunk_id != expected_id:
            raise ValueError(f"Noncanonical chunk id: {chunk.chunk_id}")
        char_start, char_end, text = _line_char_range(file.normalized_text, chunk.line_start, chunk.line_end)
        if (chunk.char_start, chunk.char_end, chunk.text) != (char_start, char_end, text):
            raise ValueError(f"Chunk span does not reconstruct for {chunk.chunk_id}")
        if chunk.text_sha256 != _sha256_text(chunk.text):
            raise ValueError(f"Chunk text hash mismatch for {chunk.chunk_id}")
        if (
            chunk.repository_id != REPOSITORY_ID
            or chunk.base_commit != BASE_COMMIT
            or chunk.chunker_version != CHUNKER_VERSION
            or chunk.source_kind != SOURCE_KIND
            or chunk.source_revision != DATASET_VERSION
            or chunk.license_status != LICENSE_STATUS
            or chunk.public_score_eligible
        ):
            raise ValueError(f"Chunk provenance changed for {chunk.chunk_id}")
    if set(chunks_by_path) != set(files_by_path) or any(not values for values in chunks_by_path.values()):
        raise ValueError("Full-corpus candidate coverage is missing a repository file")
    for path, path_chunks in chunks_by_path.items():
        if [chunk.ordinal_in_file for chunk in path_chunks] != list(range(len(path_chunks))):
            raise ValueError(f"Chunk order changed for {path}")

    queries_by_id = {query.query_id: query for query in fixture.queries}
    if len(queries_by_id) != len(fixture.queries):
        raise ValueError("Duplicate query id")
    if Counter(query.edit_type for query in fixture.queries) != Counter(
        {"replacement": 1, "deletion": 1, "insertion_only": 1}
    ):
        raise ValueError("Fixture must retain replacement, deletion, and insertion-only cases")
    for query in fixture.queries:
        if (
            query.repository_id != REPOSITORY_ID
            or query.text != f"{query.title}\n\n{query.body}"
            or query.query_text_sha256 != _sha256_text(query.text)
            or query.split != SPLIT
            or query.language != "en"
            or query.source_family != "local_invented_issue_patch_pair"
            or query.answer_leak_review != "pass"
            or query.privacy_review != "pass"
            or query.public_score_eligible
        ):
            raise ValueError(f"Invalid query identity or provenance for {query.query_id}")

    targets_by_id = {target.target_id: target for target in fixture.patch_targets}
    if len(targets_by_id) != len(fixture.patch_targets):
        raise ValueError("Duplicate patch target id")
    for target in fixture.patch_targets:
        if target.query_id not in queries_by_id or target.changed_path not in files_by_path:
            raise ValueError(f"Unknown patch target reference for {target.target_id}")
        file = files_by_path[target.changed_path]
        if target.preimage_line_start is not None and target.preimage_line_end is not None:
            _, _, text = _line_char_range(file.normalized_text, target.preimage_line_start, target.preimage_line_end)
            if target.preimage_text_sha256 != _sha256_text(text.rstrip("\n")):
                raise ValueError(f"Patch preimage hash mismatch for {target.target_id}")
            if target.insertion_anchor_after_line is not None or target.insertion_anchor_text_sha256 is not None:
                raise ValueError(f"Preimage target carries an insertion anchor for {target.target_id}")
        elif target.insertion_anchor_after_line is not None:
            _, _, text = _line_char_range(
                file.normalized_text,
                target.insertion_anchor_after_line,
                target.insertion_anchor_after_line,
            )
            if target.insertion_anchor_text_sha256 != _sha256_text(text.rstrip("\n")):
                raise ValueError(f"Insertion anchor hash mismatch for {target.target_id}")
            if target.preimage_text_sha256 is not None:
                raise ValueError(f"Insertion target carries preimage text for {target.target_id}")
        else:
            raise ValueError(f"Unmappable patch target {target.target_id}")
        if (
            target.repository_id != REPOSITORY_ID
            or target.base_commit != BASE_COMMIT
            or target.mapping_status != "exact"
            or target.source_kind != SOURCE_KIND
            or target.source_revision != DATASET_VERSION
            or target.public_score_eligible
        ):
            raise ValueError(f"Patch target provenance changed for {target.target_id}")

    expected_qrels = generate_qrels(fixture.patch_targets, fixture.chunks)
    if fixture.qrels != expected_qrels:
        raise ValueError("Qrel mapping does not reproduce from patch targets and fixed chunks")
    qrel_ids: set[str] = set()
    positive_keys: set[tuple[str, str]] = set()
    mapped_units: dict[str, set[str]] = defaultdict(set)
    for qrel in fixture.qrels:
        if qrel.qrel_id in qrel_ids or (qrel.query_id, qrel.chunk_id) in positive_keys:
            raise ValueError("Duplicate qrel identity")
        qrel_ids.add(qrel.qrel_id)
        positive_keys.add((qrel.query_id, qrel.chunk_id))
        mapped_units[qrel.query_id].update(qrel.target_unit_ids)
        target = targets_by_id.get(qrel.target_id)
        if (
            target is None
            or qrel.query_id != target.query_id
            or qrel.changed_path != target.changed_path
            or qrel.patch_raw_sha256 != target.patch_raw_sha256
            or qrel.relevance != 2
            or qrel.mapping_status != "exact"
            or qrel.public_score_eligible
        ):
            raise ValueError(f"Invalid direct qrel {qrel.qrel_id}")
    target_units_by_query: dict[str, set[str]] = defaultdict(set)
    for target in fixture.patch_targets:
        target_units_by_query[target.query_id].update(target_unit_ids(target))
    if mapped_units != target_units_by_query:
        raise ValueError("Candidate coverage does not map every edit target unit")

    hard_ids: set[str] = set()
    hard_by_query: dict[str, list[HardNegative]] = defaultdict(list)
    for negative in fixture.hard_negatives:
        if negative.hard_negative_id in hard_ids:
            raise ValueError("Duplicate hard-negative id")
        hard_ids.add(negative.hard_negative_id)
        hard_by_query[negative.query_id].append(negative)
        if negative.query_id not in queries_by_id or negative.chunk_id not in chunk_by_id:
            raise ValueError("Unknown hard-negative reference")
        if (negative.query_id, negative.chunk_id) in positive_keys:
            raise ValueError("Hard negative overlaps a positive qrel")
        positive_source_ids = {
            qrel.chunk_id for qrel in fixture.qrels if qrel.query_id == negative.query_id
        }
        if negative.source_chunk_id not in positive_source_ids:
            raise ValueError("Hard-negative source chunk is not a positive qrel")
        if (
            negative.false_negative_review != "pass"
            or not negative.reason
            or not negative.review_metadata
            or negative.source_kind != SOURCE_KIND
            or negative.source_revision != DATASET_VERSION
            or negative.public_score_eligible
        ):
            raise ValueError("Hard-negative audit metadata is incomplete")
    if any(len(hard_by_query[query.query_id]) != 3 for query in fixture.queries):
        raise ValueError("Every query must retain three audited hard negatives")

    provenance_by_query = {record.query_id: record for record in fixture.provenance}
    if set(provenance_by_query) != set(queries_by_id):
        raise ValueError("Every query must retain one provenance record")
    for query_id, record in provenance_by_query.items():
        query = queries_by_id[query_id]
        patch_shas = {target.patch_raw_sha256 for target in fixture.patch_targets if target.query_id == query_id}
        if (
            record.issue_source_id != query.source_id
            or record.issue_content_sha256 != query.query_text_sha256
            or patch_shas != {record.patch_content_sha256}
            or record.repository_snapshot_sha256 != fixture.repository.snapshot_sha256
            or record.normalization != NORMALIZATION
            or record.chunker_version != CHUNKER_VERSION
            or record.qrel_generator_version != QREL_GENERATOR_VERSION
            or record.review_status != "invented_content_review_pass"
            or record.source_kind != SOURCE_KIND
            or record.source_revision != DATASET_VERSION
            or record.license_status != LICENSE_STATUS
            or record.public_score_eligible
        ):
            raise ValueError(f"Invalid provenance for {query_id}")

    expected_digest = _fixture_digest(fixture)
    if fixture.serialization_sha256 != expected_digest:
        raise ValueError("Fixture serialization hash is unstable")
    if EXPECTED_SERIALIZATION_SHA256 and expected_digest != EXPECTED_SERIALIZATION_SHA256:
        raise ValueError("Fixture identity changed without a dataset-version update")
    serialized = serialize_fixture(fixture)
    if json.dumps(json.loads(serialized), ensure_ascii=False, sort_keys=True, separators=(",", ":")) != serialized:
        raise ValueError("Fixture serialization is not canonical JSON")


def load_code_edit_chunk_localization_fixture() -> CodeEditChunkLocalizationFixture:
    """Build and validate the deterministic, zero-network, no-publish fixture."""
    fixture = _build_fixture()
    validate_code_edit_chunk_localization_fixture(fixture)
    return fixture
