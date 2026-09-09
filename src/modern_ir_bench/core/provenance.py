"""Immutable source provenance for benchmark runs."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository_url(remote: str) -> str:
    if remote.startswith("git@github.com:"):
        remote = f"https://github.com/{remote.removeprefix('git@github.com:')}"
    return remote.removesuffix(".git")


@dataclass(frozen=True)
class RunProvenance:
    """Immutable pointer from a result to its executable source."""

    source_commit: str
    source_module: str
    source_path: str
    source_line: int
    repository_url: str
    created_at: str

    @classmethod
    def capture(
        cls,
        *,
        source_module: str,
        source_path: str,
        source_line: int,
    ) -> RunProvenance:
        return cls(
            source_commit=_git("rev-parse", "HEAD"),
            source_module=source_module,
            source_path=source_path,
            source_line=source_line,
            repository_url=_repository_url(_git("remote", "get-url", "origin")),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def source_url(self) -> str:
        return f"{self.repository_url}/blob/{self.source_commit}/{self.source_path}#L{self.source_line}"
