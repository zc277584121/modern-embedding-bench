from __future__ import annotations

import hashlib
import re

import numpy as np

from mm_embed.benchmark.registry import load_catalog
from mm_embed.data.agent_procedural_tool_memory import (
    load_agent_procedural_tool_memory_fixture,
    serialize_tool_document,
)
from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType
from mm_embed.tasks.registry import get_task


class DeterministicTextProvider(EmbeddingProvider):
    name = "deterministic-local"
    model = "hashed-token-overlap"
    supported_modalities = {ModalityType.TEXT}

    def __init__(self) -> None:
        super().__init__()
        self.task_type_calls: list[str | None] = []

    def embed_with_cache(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        return self.embed(inputs, dimensions=dimensions, task_type=task_type)

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        self.task_type_calls.append(task_type)
        dim = dimensions or 256
        embeddings = np.vstack([self._embed_text(str(item.content), dim) for item in inputs])
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=dim,
            model_name=self.model,
            provider=self.name,
            latency_ms=0.0,
        )

    @staticmethod
    def _embed_text(text: str, dim: int) -> np.ndarray:
        vector = np.zeros(dim, dtype=float)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % dim
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        return vector


def test_agent_procedural_tool_memory_fixture_shape() -> None:
    fixture = load_agent_procedural_tool_memory_fixture()

    assert len(fixture.queries) == 12
    assert len(fixture.documents) == 36
    assert sum(len(query.hard_negative_doc_ids) for query in fixture.queries) == 36
    assert fixture.source_datasets == ("local_invented_tool_memory_fixture",)
    assert fixture.license_audit_status == "local_invented_sanitized_no_external_sources"

    doc_ids = {document.doc_id for document in fixture.documents}
    for query in fixture.queries:
        assert query.positive_doc_id in doc_ids
        assert len(query.hard_negative_doc_ids) == 3
        assert query.positive_doc_id not in query.hard_negative_doc_ids
        assert set(query.hard_negative_doc_ids).issubset(doc_ids)

    serialized = "\n".join(serialize_tool_document(document) for document in fixture.documents)
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert "@" not in serialized


def test_agent_procedural_tool_memory_catalog_and_registry() -> None:
    catalog = load_catalog()

    assert "agent_procedural_tool_memory" in catalog.tasks
    spec = catalog.tasks["agent_procedural_tool_memory"]
    assert spec.primary_metric == "hard_mrr"
    assert spec.required_modalities == ["text"]
    assert {"agentic", "memory", "tool-retrieval", "hard-negative", "text"}.issubset(spec.tags)

    task = get_task("agent_procedural_tool_memory")
    assert task.name == "agent_procedural_tool_memory"
    assert task.required_modalities == {ModalityType.TEXT}


def test_agent_procedural_tool_memory_runs_with_deterministic_provider() -> None:
    task = get_task("agent_procedural_tool_memory")
    provider = DeterministicTextProvider()

    result = task.run(provider)

    assert result.passed, result.error
    assert provider.task_type_calls == ["retrieval_query", "retrieval_document"]

    required_metrics = {
        "recall@1",
        "recall@5",
        "mrr",
        "ndcg@10",
        "hard_recall@1",
        "hard_recall@5",
        "hard_mrr",
        "hard_ndcg@10",
    }
    assert required_metrics.issubset(result.metrics)
    assert all(0.0 <= result.metrics[key] <= 1.0 for key in required_metrics)

    assert result.details["n_queries"] == 12
    assert result.details["n_documents"] == 36
    assert result.details["n_qrels"] == 12
    assert result.details["n_hard_negatives"] == 36
    assert result.details["source_datasets"] == ["local_invented_tool_memory_fixture"]
    assert result.details["license_audit_status"] == "local_invented_sanitized_no_external_sources"
    assert result.details["slices"] == {
        "category_collision": 3,
        "near_name": 3,
        "parameter_constraints": 3,
        "single_tool": 3,
    }
    assert result.details["hard_pool_size"] == 4


def test_agent_procedural_tool_memory_respects_max_queries() -> None:
    task = get_task("agent_procedural_tool_memory", max_queries=5)
    provider = DeterministicTextProvider()

    result = task.run(provider)

    assert result.passed, result.error
    assert result.details["n_queries"] == 5
    assert result.details["n_qrels"] == 5
    assert result.details["n_hard_negatives"] == 15
