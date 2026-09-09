"""Execution controls that are independent from task semantics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Runtime:
    """Controls batching while leaving dataset loading to Hugging Face."""

    query_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")

    def batch_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        for row in rows:
            batch.append(dict(row))
            if len(batch) == self.query_batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
