"""Solution identity shared by benchmark participants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Solution:
    """A model, algorithm, pipeline, or agent evaluated by a task."""

    id: str
    title: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id or any(character.isspace() for character in self.id):
            raise ValueError("Solution id must be a non-empty string without whitespace")
        if not self.title:
            raise ValueError("Solution title must be non-empty")
