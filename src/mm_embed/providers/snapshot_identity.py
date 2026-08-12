"""Fail-closed identity checks for pinned local Hugging Face snapshots."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path


# These repository presentation files are not read by Transformers or Sentence
# Transformers. Everything else in a snapshot is treated as behavior-affecting
# and therefore participates in the fail-closed identity.
_NON_BEHAVIOR_ROOT_FILES = frozenset({".gitattributes", "README.md"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_identity(path: Path) -> dict[str, str]:
    """Return a path-stable identity for every behavior file in a snapshot."""
    if not path.is_dir():
        raise ValueError(f"snapshot_path must be a directory: {path}")
    files: dict[str, str] = {}
    for item in path.rglob("*"):
        relative = item.relative_to(path).as_posix()
        parts = item.relative_to(path).parts
        if relative in _NON_BEHAVIOR_ROOT_FILES or (parts and parts[0] == ".cache"):
            continue
        if item.is_file():
            files[relative] = file_sha256(item)
    return dict(sorted(files.items()))


def verify_snapshot_identity(path: Path, expected: Mapping[str, str], *, label: str) -> None:
    """Verify immutable content identity, independent of cache/copy layout."""
    if not path.is_dir():
        raise ValueError(f"{label} snapshot_path must be a directory: {path}")
    actual = snapshot_identity(path)
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"{label} snapshot is missing identity files: {', '.join(missing)}")
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        raise ValueError(f"{label} snapshot has unexpected identity files: {', '.join(unexpected)}")
    mismatched = sorted(name for name, digest in expected.items() if actual[name] != digest)
    if mismatched:
        raise ValueError(f"{label} snapshot identity mismatch for revision-pinned files: {', '.join(mismatched)}")
