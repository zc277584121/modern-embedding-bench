"""Narrow embedding protocols shared by retrieval solutions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

DenseVectors = npt.NDArray[np.floating[Any]]


class DenseEmbedding(Protocol):
    """Batch-native dense embedding with distinct document and query paths."""

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, inputs: Sequence[Any]) -> DenseVectors: ...

    def encode_queries(self, inputs: Sequence[Any]) -> DenseVectors: ...
