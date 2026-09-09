from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from datasets import Dataset

from modern_ir_bench.retrieval import DenseRetrievalSolution, MappedResourceSource
from modern_ir_bench.retrieval.indexes import NumpyFlatIndex


class LookupEmbedding:
    dimension = 2

    def __init__(self) -> None:
        self.vectors = {
            "north": [1.0, 0.0],
            "east": [0.0, 1.0],
            "north east": [1.0, 1.0],
        }

    def _encode(self, inputs: Sequence[str]) -> np.ndarray:
        return np.asarray([self.vectors[value] for value in inputs], dtype=np.float32)

    def encode_documents(self, inputs: Sequence[str]) -> np.ndarray:
        return self._encode(inputs)

    def encode_queries(self, inputs: Sequence[str]) -> np.ndarray:
        return self._encode(inputs)


def test_dense_solution_streams_resources_into_a_replaceable_index() -> None:
    dataset = Dataset.from_dict(
        {
            "record_key": ["n", "e", "ne"],
            "payload": ["north", "east", "north east"],
        }
    )
    resources = MappedResourceSource(
        dataset,
        id_of=lambda row: row["record_key"],
        value_of=lambda row: row["payload"],
    )
    solution = DenseRetrievalSolution(
        id="lookup-flat",
        title="Lookup Flat",
        embedding=LookupEmbedding(),
        index=NumpyFlatIndex(metric="COSINE"),
        document_batch_size=2,
    )

    session = solution.prepare(resources)
    try:
        results = session.search_batch(["north", "east"], top_k=2)
        assert [[hit.id for hit in hits] for hits in results] == [
            ["n", "ne"],
            ["e", "ne"],
        ]
    finally:
        session.close()


def test_resource_mapping_is_lazy_and_reiterable() -> None:
    dataset = Dataset.from_dict({"key": ["a", "b"], "text": ["north", "east"]})
    resources = MappedResourceSource(
        dataset,
        id_of=lambda row: row["key"],
        value_of=lambda row: row["text"],
    )

    assert [resource.id for resource in resources] == ["a", "b"]
    assert [resource.value for resource in resources] == ["north", "east"]
