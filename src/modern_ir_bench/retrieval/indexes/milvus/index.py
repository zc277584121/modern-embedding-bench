"""Isolated dense index sessions backed by the MilvusClient API."""

from __future__ import annotations

import importlib.metadata
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modern_ir_bench.embeddings.protocols import DenseVectors
from modern_ir_bench.retrieval.indexes._validation import (
    as_dense_matrix,
    normalize_dense_metric,
)
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.retrieval.types import SearchHit

_COLLECTION_PART = re.compile(r"[^a-zA-Z0-9_]")


def _collection_name(prefix: str) -> str:
    normalized = _COLLECTION_PART.sub("_", prefix).strip("_")
    if not normalized:
        raise ValueError("collection_prefix must contain a letter or number")
    return f"{normalized[:96]}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class MilvusDenseIndex:
    """Create an isolated Milvus collection for one dense evaluation."""

    target: MilvusTarget
    metric: str = "COSINE"
    index_type: str | None = None
    index_params: Mapping[str, Any] = field(default_factory=dict)
    search_params: Mapping[str, Any] = field(default_factory=dict)
    collection_prefix: str = "modern_ir_bench"

    def __post_init__(self) -> None:
        metric = normalize_dense_metric(self.metric)
        index_type = (self.index_type or self.target.default_index_type).upper()
        self.target.validate_index_type(index_type)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "index_type", index_type)

    def open(self, *, dimension: int) -> MilvusDenseSession:
        if dimension < 1:
            raise ValueError("dimension must be positive")

        from pymilvus import DataType, MilvusClient

        arguments = dict(self.target.client_arguments())
        if self.target.deployment == "milvus-lite":
            Path(arguments["uri"]).parent.mkdir(parents=True, exist_ok=True)
        client = MilvusClient(**arguments)
        collection_name = _collection_name(self.collection_prefix)
        try:
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=2048,
                is_primary=True,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=dimension,
            )
            indexes = MilvusClient.prepare_index_params()
            indexes.add_index(
                field_name="vector",
                index_name="vector_index",
                index_type=self.index_type,
                metric_type=self.metric,
                params=dict(self.index_params),
            )
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
            )
        except Exception:
            client.close()
            raise

        return MilvusDenseSession(
            client=client,
            collection_name=collection_name,
            dimension=dimension,
            metric=self.metric,
            index_type=self.index_type,
            search_params=dict(self.search_params),
            deployment=self.target.deployment,
        )


class MilvusDenseSession:
    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        dimension: int,
        metric: str,
        index_type: str,
        search_params: Mapping[str, Any],
        deployment: str,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.search_params = dict(search_params)
        self.deployment = deployment
        self._ids: set[str] = set()
        self._sealed = False
        self._closed = False
        self._actual_index: Mapping[str, Any] | None = None
        self._server_version: str | None = None
        self._server_count: int | None = None

    def add(self, ids: Sequence[str], vectors: DenseVectors) -> None:
        if self._closed:
            raise RuntimeError("Cannot add vectors to a closed Milvus index")
        if self._sealed:
            raise RuntimeError("Cannot add vectors after the index has been sealed")
        matrix = as_dense_matrix(vectors, dimension=self.dimension)
        normalized_ids = [str(item_id) for item_id in ids]
        if len(normalized_ids) != matrix.shape[0]:
            raise ValueError("ids and vectors must contain the same number of rows")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("ids must be unique within a batch")
        duplicates = self._ids.intersection(normalized_ids)
        if duplicates:
            raise ValueError(f"Duplicate ids across batches: {sorted(duplicates)}")
        self.client.insert(
            collection_name=self.collection_name,
            data=[
                {"id": item_id, "vector": vector.tolist()}
                for item_id, vector in zip(normalized_ids, matrix, strict=True)
            ],
        )
        self._ids.update(normalized_ids)

    def seal(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot seal a closed Milvus index")
        if self._sealed:
            return
        if not self._ids:
            raise ValueError("Cannot seal an empty dense index")
        self.client.load_collection(collection_name=self.collection_name)
        self._actual_index = self.client.describe_index(
            collection_name=self.collection_name,
            index_name="vector_index",
        )
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        self._server_count = int(stats["row_count"])
        if self._server_count != len(self._ids):
            raise RuntimeError(
                f"Milvus reports {self._server_count} rows after inserting {len(self._ids)}"
            )
        self._server_version = self.client.get_server_version()
        self._sealed = True

    def search(
        self,
        query_vectors: DenseVectors,
        *,
        top_k: int,
    ) -> list[list[SearchHit]]:
        if not self._sealed or self._closed:
            raise RuntimeError("The Milvus index must be sealed before search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        queries = as_dense_matrix(query_vectors, dimension=self.dimension)
        results = self.client.search(
            collection_name=self.collection_name,
            data=queries.tolist(),
            anns_field="vector",
            limit=min(top_k, len(self._ids)),
            search_params=dict(self.search_params),
            consistency_level="Strong",
        )
        output: list[list[SearchHit]] = []
        for query_hits in results:
            hits = []
            for hit in query_hits:
                distance = float(hit["distance"])
                score = -distance if self.metric == "L2" else distance
                hits.append(SearchHit(id=str(hit["id"]), score=score))
            output.append(hits)
        return output

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "milvus",
            "deployment": self.deployment,
            "metric": self.metric,
            "requested_index_type": self.index_type,
            "actual_index": dict(self._actual_index) if self._actual_index else None,
            "dimension": self.dimension,
            "count": len(self._ids),
            "server_count": self._server_count,
            "pymilvus_version": importlib.metadata.version("pymilvus"),
            "server_version": self._server_version,
            "consistency_level": "Strong",
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.client.drop_collection(collection_name=self.collection_name)
        finally:
            self.client.close()
            self._closed = True
