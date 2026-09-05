#!/usr/bin/env python3
"""Build or verify the exact ordered S-009 validator candidate identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from mm_embed.benchmark.bright_cross_paradigm import publication_policy, sha256_file
from mm_embed.benchmark.bright_cross_paradigm_package import candidate_paths

OUTPUT = Path("benchmark/artifacts/bright-cross-paradigm-results-v0.1/validator-candidate.json")
PROTECTED = {
    "benchmark/artifacts/milvus-sindi-system-v0.1/validator-candidate.json",
    "benchmark/artifacts/milvus-sindi-system-v0.1/validator-candidate.json.sha256",
}


def formatted(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True).stdout.strip()


def dependency(repo: Path, path: str) -> dict[str, object]:
    head_blob = git(repo, "rev-parse", f"HEAD:{path}")
    working_blob = git(repo, "hash-object", path)
    if head_blob != working_blob:
        raise ValueError(f"Unchanged dependency has working-tree drift: {path}")
    target = repo / path
    return {
        "path": path,
        "role": "unchanged_tree_dependency_excluded_from_commit_delta",
        "git_blob_oid": head_blob,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def alternate_index(repo: Path, paths: tuple[str, ...]) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="s009-alternate-index-") as temporary:
        index = Path(temporary) / "index"
        environment = {**os.environ, "GIT_INDEX_FILE": str(index)}
        git(repo, "read-tree", "HEAD", env=environment)
        git(repo, "add", "--", *paths, env=environment)
        staged = tuple(
            line
            for line in git(
                repo,
                "diff",
                "--cached",
                "--name-only",
                "--diff-filter=ACMRTUXB",
                "HEAD",
                env=environment,
            ).splitlines()
            if line
        )
        if staged != paths:
            raise ValueError("Alternate-index commit delta differs from the declared candidate order")
        git(repo, "diff", "--cached", "--check", env=environment)
        tree = git(repo, "write-tree", env=environment)
        remote_style = tuple(
            line
            for line in git(repo, "diff-tree", "-r", "--name-only", "--no-commit-id", "HEAD", tree).splitlines()
            if line
        )
        if remote_style != paths:
            raise ValueError("git diff-tree cannot reproduce the declared commit delta")
    return {
        "base_commit": git(repo, "rev-parse", "HEAD"),
        "alternate_index_tree_oid": tree,
        "alternate_index_paths": list(staged),
        "diff_tree_paths": list(remote_style),
        "alternate_index_diff_check": "pass",
    }


def build(repo: Path) -> dict[str, object]:
    paths = candidate_paths()
    if tuple(sorted(paths)) != paths or len(paths) != len(set(paths)):
        raise ValueError("Candidate paths must be unique and lexicographically ordered")
    if any(path.startswith(("results/", "cache/", ".cache/")) or path in PROTECTED for path in paths):
        raise ValueError("Candidate set contains a restricted or protected path")
    records = [
        {"path": path, "bytes": (repo / path).stat().st_size, "sha256": sha256_file(repo / path)} for path in paths
    ]
    index_evidence = alternate_index(repo, paths)
    return {
        "schema_version": "bright-cross-paradigm-validator-candidate-v1",
        "story_id": "S-20260814-009",
        "status": "candidate_not_accepted",
        "classification": "tracked_candidate_identity",
        "candidate_file_count": len(records),
        "candidate_set_sha256": hashlib.sha256(formatted(records)).hexdigest(),
        "candidate_path_order_sha256": hashlib.sha256(formatted(list(paths))).hexdigest(),
        "files": records,
        "commit_delta": index_evidence,
        "unchanged_dependencies": [dependency(repo, "pyproject.toml"), dependency(repo, "uv.lock")],
        "validator_identity_files_are_review_metadata_not_commit_delta": [
            OUTPUT.as_posix(),
            OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").as_posix(),
        ],
        "excluded_prefixes": ["results/", "cache/", ".cache/"],
        "protected_files_excluded": sorted(PROTECTED),
        "publication": publication_policy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build(Path("."))
    payload = formatted(value)
    identity = hashlib.sha256(payload).hexdigest()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.check:
        if args.output.read_bytes() != payload or sidecar.read_text(encoding="ascii").strip() != identity:
            raise ValueError("Validator candidate identity cannot be rebuilt exactly")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        sidecar.write_text(identity + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "candidate_file_count": len(value["files"]),
                "candidate_set_sha256": value["candidate_set_sha256"],
                "manifest_sha256": identity,
                "status": "pass" if args.check else "built",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
