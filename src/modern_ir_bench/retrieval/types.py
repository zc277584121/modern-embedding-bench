"""Small runtime values shared by ranked retrieval implementations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

ResourceInput = TypeVar("ResourceInput")


@dataclass(frozen=True)
class RetrievalResource(Generic[ResourceInput]):
    """One runtime resource, independent of the source Dataset schema."""

    id: str
    value: ResourceInput


class MappedResourceSource(Generic[ResourceInput]):
    """A lazy view that preserves the iteration behavior of its Dataset rows."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        id_of: Callable[[Mapping[str, Any]], str],
        value_of: Callable[[Mapping[str, Any]], ResourceInput],
    ) -> None:
        self._rows = rows
        self._id_of = id_of
        self._value_of = value_of

    def __iter__(self) -> Iterator[RetrievalResource[ResourceInput]]:
        for row in self._rows:
            yield RetrievalResource(
                id=str(self._id_of(row)),
                value=self._value_of(row),
            )


@dataclass(frozen=True)
class SearchHit:
    """One ranked retrieval hit."""

    id: str
    score: float


class SearchSession(Protocol):
    """Prepared retrieval state with explicit search and cleanup phases."""

    def search_batch(
        self,
        queries: Sequence[Any],
        *,
        top_k: int,
    ) -> list[list[SearchHit]]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...
