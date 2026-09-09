from __future__ import annotations

import numpy as np
import pytest

from modern_ir_bench.retrieval.indexes import NumpyFlatIndex, VerifiedDenseIndex
from modern_ir_bench.retrieval.indexes.milvus import (
    MilvusDenseIndex,
    MilvusLite,
    ZillizCloud,
)


def test_numpy_flat_is_an_exact_cosine_reference() -> None:
    session = NumpyFlatIndex(metric="cosine").open(dimension=2)
    session.add(
        ["north", "east", "south"],
        np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32),
    )
    session.seal()

    results = session.search(np.asarray([[1, 0]], dtype=np.float32), top_k=3)

    assert [hit.id for hit in results[0]] == ["north", "east", "south"]
    assert session.metadata["exact"] is True


def test_milvus_lite_is_verified_against_an_independent_flat_oracle(tmp_path) -> None:
    index = VerifiedDenseIndex(
        primary=MilvusDenseIndex(target=MilvusLite(tmp_path / "milvus.db")),
        oracle=NumpyFlatIndex(metric="COSINE"),
    )
    session = index.open(dimension=2)
    try:
        session.add(
            ["north", "east", "south"],
            np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32),
        )
        session.seal()
        results = session.search(
            np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            top_k=2,
        )

        assert [[hit.id for hit in hits] for hits in results] == [
            ["north", "east"],
            ["east", "north"],
        ]
        assert session.mean_index_recall == 1.0
        assert session.metadata["primary"]["deployment"] == "milvus-lite"
        assert session.metadata["primary"]["server_count"] == 3
        assert session.metadata["primary"]["actual_index"]["index_type"] == "FLAT"
    finally:
        session.close()


def test_targets_reject_index_settings_the_deployment_will_not_honor(tmp_path) -> None:
    with pytest.raises(ValueError, match="Milvus Lite only supports FLAT"):
        MilvusDenseIndex(
            target=MilvusLite(tmp_path / "milvus.db"),
            index_type="HNSW",
        )

    with pytest.raises(ValueError, match="Zilliz Cloud uses AUTOINDEX"):
        MilvusDenseIndex(
            target=ZillizCloud(uri="https://example.invalid", token="secret"),
            index_type="FLAT",
        )


def test_verified_index_rejects_invalid_recall_threshold() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        VerifiedDenseIndex(
            primary=NumpyFlatIndex(),
            oracle=NumpyFlatIndex(),
            minimum_index_recall=1.1,
        )
