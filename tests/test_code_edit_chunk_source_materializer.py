from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import socket
import tarfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from mm_embed.data.code_edit_chunk_source_materializer import (
    ChunkingPolicy,
    FileAudit,
    MaterializationArtifacts,
    MaterializedTextFile,
    PatchTarget,
    ResourceCaps,
    SourceChunk,
    SourceContract,
    SourceMaterializationError,
    SourceQrel,
    TreeEntry,
    audit_eligible_files,
    load_source_contract,
    map_targets_to_qrels,
    run_source_materialization,
    serialize_artifact_summary,
    validate_materialization_artifacts,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _archive_bytes(
    files: dict[str, bytes],
    *,
    unsafe_name: str | None = None,
    symlink_name: str | None = None,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        root = tarfile.TarInfo("fixture-root/")
        root.type = tarfile.DIRTYPE
        bundle.addfile(root)
        for path, content in sorted(files.items()):
            info = tarfile.TarInfo(f"fixture-root/{path}")
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
        if unsafe_name is not None:
            content = b"unsafe\n"
            info = tarfile.TarInfo(unsafe_name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
        if symlink_name is not None:
            info = tarfile.TarInfo(f"fixture-root/{symlink_name}")
            info.type = tarfile.SYMTYPE
            info.linkname = "src/app.py"
            bundle.addfile(info)
    return output.getvalue()


def _build_case(
    *,
    archive_files: dict[str, bytes] | None = None,
    unsafe_name: str | None = None,
    symlink_name: str | None = None,
) -> tuple[SourceContract, dict[str, bytes]]:
    repository_id = "example/source-fixture"
    base_commit = "1" * 40
    tree_sha = "2" * 40
    pr_head_sha = "3" * 40
    files = archive_files or {
        "LICENSE": b"Self-created fixture license\n",
        "docs/guide.md": b"# Guide\r\n\r\nUse the local fixture.\r\n",
        "metadata/bad.json": b"{\xff}\n",
        "src/app.py": b"def choose_delay(attempt: int) -> int:\n    return 30 * max(1, attempt)\n",
    }
    tree_entries = [
        {
            "path": path,
            "mode": "100644",
            "type": "blob",
            "sha": _git_blob_sha(content),
            "size": len(content),
        }
        for path, content in sorted(files.items())
    ]
    issue_title = "Shorten the first retry delay"
    issue_body = "The first retry waits too long while later backoff should remain unchanged."
    issue = _json_bytes(
        {
            "number": 7,
            "repository_url": f"https://api.github.com/repos/{repository_id}",
            "title": issue_title,
            "body": issue_body,
        }
    )
    pr = _json_bytes(
        {
            "number": 8,
            "state": "closed",
            "merged_at": "2026-07-22T00:00:00Z",
            "base": {"sha": base_commit},
            "head": {"sha": pr_head_sha},
        }
    )
    pr_files = _json_bytes([{"filename": "src/app.py", "status": "modified"}])
    tree = _json_bytes({"sha": tree_sha, "truncated": False, "tree": tree_entries})
    patch = (
        "From fixture@example.invalid Wed Jul 22 00:00:00 2026\n"
        "Subject: [PATCH] Retry sooner\n"
        "---\n"
        " src/app.py | 1 +\n"
        " 1 file changed, 1 insertion(+)\n"
        "\n"
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def choose_delay(attempt: int) -> int:\n"
        "+    first_delay = 10\n"
        "     return 30 * max(1, attempt)\n"
        "-- \n"
        "2.39.0\n"
    ).encode()
    archive = _archive_bytes(files, unsafe_name=unsafe_name, symlink_name=symlink_name)
    sources = {
        "issue_api": issue,
        "pr_api": pr,
        "pr_files_api": pr_files,
        "tree_api": tree,
        "patch": patch,
        "archive": archive,
    }
    contract = SourceContract(
        contract_version="self-created-source-contract-v0",
        repository_id=repository_id,
        issue_number=7,
        pr_number=8,
        base_commit=base_commit,
        tree_sha=tree_sha,
        pr_head_sha=pr_head_sha,
        expected_changed_paths=("src/app.py",),
        license_path="LICENSE",
        license_blob_sha=_git_blob_sha(files["LICENSE"]),
        license_sha256=_sha256(files["LICENSE"]),
        query_text_sha256=_sha256(f"{issue_title}\n\n{issue_body}".encode()),
        source_sha256={key: _sha256(sources[key]) for key in ("archive", "patch")},
        caps=ResourceCaps(
            archive_bytes=1_000_000,
            extracted_regular_file_bytes=1_000_000,
            stage_a_candidate_bytes=100_000,
            eligible_normalized_text_bytes=100_000,
            tracked_files=20,
            chunks=100,
            target_rss_bytes=2_000_000_000,
            wall_seconds=30,
        ),
        chunking=ChunkingPolicy(
            definition_window_lines=120,
            definition_overlap_lines=20,
            fallback_window_lines=80,
            fallback_overlap_lines=10,
        ),
        publish=False,
        evidence_tier="smoke",
        query_review={
            "answer_leak_review": "pass",
            "privacy_review": "pass",
            "prompt_injection_review": "pass",
            "secret_review": "pass",
        },
    )
    return contract, sources


def _fetcher(contract: SourceContract, sources: dict[str, bytes]):
    by_url = {contract.source_urls[name]: content for name, content in sources.items()}

    def fetch(url: str, max_bytes: int) -> bytes:
        content = by_url[url]
        if len(content) > max_bytes:
            raise SourceMaterializationError("download_cap", "Self-created response exceeds cap")
        return content

    return fetch


def test_zero_network_run_is_deterministic_source_free_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, sources = _build_case()

    def forbidden_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("Zero-network fixture attempted a socket call")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    first = run_source_materialization(contract, fetch_bytes=_fetcher(contract, sources), temp_root=tmp_path)
    second = run_source_materialization(contract, fetch_bytes=_fetcher(contract, sources), temp_root=tmp_path)

    assert first["status"] == second["status"] == "PASS"
    assert first["artifacts"] == second["artifacts"]
    assert first["artifacts"]["tracked_blob_count"] == 4
    assert first["artifacts"]["archive_download_bytes"] == len(sources["archive"])
    assert first["artifacts"]["extracted_regular_file_bytes"] == sum(
        map(
            len,
            (
                b"Self-created fixture license\n",
                b"# Guide\r\n\r\nUse the local fixture.\r\n",
                b"{\xff}\n",
                b"def choose_delay(attempt: int) -> int:\n    return 30 * max(1, attempt)\n",
            ),
        )
    )
    assert first["artifacts"]["eligible_text_file_count"] == 2
    assert first["artifacts"]["stage_b_counts"]["rejected:invalid_utf8"] == 1
    assert first["artifacts"]["candidate_coverage"] == 1.0
    serialized = serialize_artifact_summary(first["artifacts"])
    assert "choose_delay" not in serialized
    assert "local fixture" not in serialized
    assert first["artifacts"]["artifact_summary_sha256"] == second["artifacts"]["artifact_summary_sha256"]
    for result in (first, second):
        assert result["runtime"]["cleanup_absent"] is True
        assert not Path(result["runtime"]["cleanup_path"]).exists()
    assert list(tmp_path.iterdir()) == []


def test_pinned_requests_contract_is_bounded_and_no_publish() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_source_contract(
        root / "benchmark/research/code_edit_chunk_requests_source_contract_20260722.json"
    )

    assert contract.repository_id == "psf/requests"
    assert contract.expected_changed_paths == ("requests/sessions.py", "test_requests.py")
    assert contract.publish is False
    assert contract.evidence_tier == "smoke"
    assert set(contract.source_sha256) == {"archive", "patch"}
    assert contract.caps.archive_bytes == 10_000_000
    assert contract.caps.chunks == 1_000


def test_source_pin_drift_is_blocked_and_cleans_on_failure(tmp_path: Path) -> None:
    contract, sources = _build_case()
    sources["patch"] += b" "

    result = run_source_materialization(contract, fetch_bytes=_fetcher(contract, sources), temp_root=tmp_path)

    assert result["status"] == "BLOCKED"
    assert result["error"]["code"] == "source_pin_drift"
    assert result["runtime"]["cleanup_absent"] is True
    assert list(tmp_path.iterdir()) == []


def test_configured_deadline_interrupts_blocking_work_and_cleans(tmp_path: Path) -> None:
    contract, _sources = _build_case()
    contract = replace(contract, caps=replace(contract.caps, wall_seconds=1))
    calls: list[str] = []
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def blocking_fetch(url: str, _max_bytes: int) -> bytes:
        calls.append(url)
        time.sleep(5)
        raise AssertionError("Configured deadline did not interrupt blocking work")

    started = time.monotonic()
    result = run_source_materialization(contract, fetch_bytes=blocking_fetch, temp_root=tmp_path)

    assert result["status"] == "BLOCKED"
    assert result["error"]["code"] == "wall_clock_cap"
    assert time.monotonic() - started < 2.0
    assert len(calls) == 1
    assert result["runtime"]["cleanup_absent"] is True
    assert list(tmp_path.iterdir()) == []
    assert signal.getsignal(signal.SIGALRM) == previous_alarm_handler
    assert signal.getitimer(signal.ITIMER_REAL) == previous_timer


def test_sigterm_stops_work_through_python_cleanup(tmp_path: Path) -> None:
    contract, _sources = _build_case()
    fetch_started = threading.Event()
    previous_term_handler = signal.getsignal(signal.SIGTERM)

    def blocking_fetch(_url: str, _max_bytes: int) -> bytes:
        fetch_started.set()
        time.sleep(5)
        raise AssertionError("SIGTERM did not interrupt blocking work")

    def terminate() -> None:
        assert fetch_started.wait(timeout=1.0)
        os.kill(os.getpid(), signal.SIGTERM)

    terminator = threading.Thread(target=terminate, daemon=True)
    terminator.start()
    started = time.monotonic()
    result = run_source_materialization(contract, fetch_bytes=blocking_fetch, temp_root=tmp_path)
    terminator.join(timeout=1.0)

    assert result["status"] == "BLOCKED"
    assert result["error"]["code"] == "terminated"
    assert time.monotonic() - started < 2.0
    assert result["runtime"]["cleanup_absent"] is True
    assert list(tmp_path.iterdir()) == []
    assert signal.getsignal(signal.SIGTERM) == previous_term_handler


@pytest.mark.parametrize(
    ("cap_name", "cap_value", "error_code"),
    [
        ("archive_bytes", 10, "download_cap"),
        ("extracted_regular_file_bytes", 10, "extracted_byte_cap"),
        ("tracked_files", 2, "tracked_file_cap"),
        ("eligible_normalized_text_bytes", 10, "eligible_text_cap"),
        ("chunks", 1, "chunk_cap"),
    ],
)
def test_byte_file_and_chunk_caps_are_hard(
    tmp_path: Path,
    cap_name: str,
    cap_value: int,
    error_code: str,
) -> None:
    contract, sources = _build_case()
    contract = replace(contract, caps=replace(contract.caps, **{cap_name: cap_value}))

    result = run_source_materialization(contract, fetch_bytes=_fetcher(contract, sources), temp_root=tmp_path)

    assert result["status"] == "BLOCKED"
    assert result["error"]["code"] == error_code
    assert result["runtime"]["cleanup_absent"] is True


@pytest.mark.parametrize(
    ("unsafe_name", "symlink_name", "error_code"),
    [
        ("fixture-root/../escape.py", None, "unsafe_archive_entry"),
        (None, "linked.py", "unsafe_archive_link"),
    ],
)
def test_unsafe_archive_entries_and_links_are_rejected(
    tmp_path: Path,
    unsafe_name: str | None,
    symlink_name: str | None,
    error_code: str,
) -> None:
    contract, sources = _build_case(unsafe_name=unsafe_name, symlink_name=symlink_name)

    result = run_source_materialization(contract, fetch_bytes=_fetcher(contract, sources), temp_root=tmp_path)

    assert result["status"] == "BLOCKED"
    assert result["error"]["code"] == error_code
    assert result["runtime"]["cleanup_absent"] is True


def test_blob_content_mismatch_is_rejected(tmp_path: Path) -> None:
    contract, sources = _build_case()
    mutated_files = {
        "LICENSE": b"Self-created fixture license\n",
        "docs/guide.md": b"# Guide\r\n\r\nUse the local fixture.\r\n",
        "metadata/bad.json": b"{\xff}\n",
        "src/app.py": b"def choose_delay(attempt: int) -> int:\n    return 31 * max(1, attempt)\n",
    }
    sources["archive"] = _archive_bytes(mutated_files)
    contract = replace(
        contract,
        source_sha256={**contract.source_sha256, "archive": _sha256(sources["archive"])},
    )

    result = run_source_materialization(contract, fetch_bytes=_fetcher(contract, sources), temp_root=tmp_path)

    assert result["status"] == "BLOCKED"
    assert result["error"]["code"] == "blob_content_mismatch"


def test_stage_b_rejections_cover_all_ordered_content_gates() -> None:
    contract, _ = _build_case()
    contents = {
        "oversize.py": b"x" * 5,
        "pointer.py": b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\n",
        "nul.py": b"x\0y",
        "invalid.py": b"\xff",
        "control.py": b"abc\x01",
    }
    entries = tuple(
        TreeEntry(path=path, mode="100644", entry_type="blob", sha=_git_blob_sha(content), size=len(content))
        for path, content in sorted(contents.items())
    )
    contract = replace(contract, caps=replace(contract.caps, stage_a_candidate_bytes=4))

    audits, files = audit_eligible_files(contract, entries, contents)

    assert files == ()
    assert {audit.path: audit.stage_b_reason for audit in audits} == {
        "control.py": "control_heavy",
        "invalid.py": "invalid_utf8",
        "nul.py": "binary_nul",
        "oversize.py": "oversize",
        "pointer.py": "oversize",
    }

    contract = replace(contract, caps=replace(contract.caps, stage_a_candidate_bytes=1_000))
    audits, _ = audit_eligible_files(contract, entries, contents)
    assert {audit.path: audit.stage_b_reason for audit in audits}["pointer.py"] == "lfs_pointer"


def _mapping_fixture() -> tuple[PatchTarget, SourceChunk]:
    target = PatchTarget(
        target_id="target_0000",
        changed_path="src/app.py",
        patch_change_type="addition",
        preimage_line_start=None,
        preimage_line_end=None,
        insertion_anchor_after_line=2,
        target_unit_ids=("target_0000:anchor:2",),
    )
    chunk = SourceChunk(
        chunk_id="chunk-a",
        repository_id="example/source-fixture",
        base_commit="1" * 40,
        path="src/app.py",
        blob_sha="2" * 40,
        candidate_family="ast_function",
        symbol="choose_delay",
        line_start=1,
        line_end=3,
        char_start=0,
        char_end=10,
        text_sha256="3" * 64,
        ordinal_in_file=0,
        text="fixture",
    )
    return target, chunk


def test_patch_mapping_ambiguity_and_incompleteness_are_rejected() -> None:
    target, chunk = _mapping_fixture()
    with pytest.raises(SourceMaterializationError, match="No chunk covers") as incomplete:
        map_targets_to_qrels((target,), ())
    assert incomplete.value.code == "patch_mapping_incomplete"

    duplicate = replace(chunk, chunk_id="chunk-b")
    with pytest.raises(SourceMaterializationError, match="Ambiguous duplicate chunk spans") as ambiguous:
        map_targets_to_qrels((target,), (chunk, duplicate))
    assert ambiguous.value.code == "patch_mapping_ambiguous"


def test_incomplete_corpus_and_qrels_are_rejected() -> None:
    contract, _ = _build_case()
    target, chunk = _mapping_fixture()
    audit = FileAudit(
        path="src/app.py",
        git_mode="100644",
        blob_sha=chunk.blob_sha,
        raw_bytes=7,
        stage_a_result="accepted",
        stage_a_reason="accepted",
        stage_b_result="accepted",
        stage_b_reason="accepted",
        bom_present=False,
        decoded_utf8_bytes=7,
        normalized_utf8_bytes=7,
        normalized_sha256="4" * 64,
    )
    file = MaterializedTextFile(audit=audit, normalized_text="fixture")
    source_hashes = {key: value for key, value in contract.source_sha256.items()}

    incomplete_corpus = MaterializationArtifacts(
        source_hashes=source_hashes,
        source_bytes={"archive": 1},
        file_audits=(audit,),
        files=(file,),
        chunks=(),
        patch_targets=(target,),
        qrels=(),
    )
    with pytest.raises(SourceMaterializationError) as corpus_error:
        validate_materialization_artifacts(contract, incomplete_corpus)
    assert corpus_error.value.code == "incomplete_corpus"

    incomplete_qrels = replace(incomplete_corpus, chunks=(chunk,))
    with pytest.raises(SourceMaterializationError) as qrel_error:
        validate_materialization_artifacts(contract, incomplete_qrels)
    assert qrel_error.value.code == "incomplete_qrels"

    complete_qrel = SourceQrel(
        qrel_id="qrel-a",
        chunk_id=chunk.chunk_id,
        target_id=target.target_id,
        target_unit_ids=target.target_unit_ids,
        relevance=2,
        label_family="insert_anchor_containing_chunk",
        changed_path=target.changed_path,
        patch_change_type=target.patch_change_type,
        overlap_lines=0,
        mapping_status="exact",
    )
    validate_materialization_artifacts(contract, replace(incomplete_qrels, qrels=(complete_qrel,)))
