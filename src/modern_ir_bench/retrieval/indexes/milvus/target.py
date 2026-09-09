"""Explicit deployment targets for the shared Milvus client API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class MilvusTarget(Protocol):
    """Connection and compatibility policy for one Milvus deployment family."""

    @property
    def deployment(self) -> str: ...

    @property
    def default_index_type(self) -> str: ...

    def client_arguments(self) -> Mapping[str, str]: ...

    def validate_index_type(self, index_type: str) -> None: ...


@dataclass(frozen=True)
class MilvusLite:
    """Embedded Milvus target with deliberate FLAT-only validation."""

    path: str | Path

    @property
    def deployment(self) -> str:
        return "milvus-lite"

    @property
    def default_index_type(self) -> str:
        return "FLAT"

    def client_arguments(self) -> Mapping[str, str]:
        return {"uri": str(Path(self.path).expanduser().resolve())}

    def validate_index_type(self, index_type: str) -> None:
        if index_type != "FLAT":
            raise ValueError("Milvus Lite only supports FLAT; refusing a silently ignored index type")


@dataclass(frozen=True)
class MilvusServer:
    """Self-hosted Milvus Standalone or Distributed target."""

    uri: str
    token: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("Milvus Server uri must be non-empty")

    @property
    def deployment(self) -> str:
        return "milvus-server"

    @property
    def default_index_type(self) -> str:
        return "FLAT"

    def client_arguments(self) -> Mapping[str, str]:
        arguments = {"uri": self.uri}
        if self.token:
            arguments["token"] = self.token
        return arguments

    def validate_index_type(self, index_type: str) -> None:
        if not index_type:
            raise ValueError("Milvus Server index_type must be non-empty")


@dataclass(frozen=True)
class ZillizCloud:
    """Managed Zilliz Cloud target with AUTOINDEX-only validation."""

    uri: str
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("Zilliz Cloud uri must be non-empty")
        if not self.token:
            raise ValueError("Zilliz Cloud token must be non-empty")

    @property
    def deployment(self) -> str:
        return "zilliz-cloud"

    @property
    def default_index_type(self) -> str:
        return "AUTOINDEX"

    def client_arguments(self) -> Mapping[str, str]:
        return {"uri": self.uri, "token": self.token}

    def validate_index_type(self, index_type: str) -> None:
        if index_type != "AUTOINDEX":
            raise ValueError("Zilliz Cloud uses AUTOINDEX for vector fields")
