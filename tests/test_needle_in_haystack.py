from __future__ import annotations

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType
from mm_embed.tasks.needle_in_haystack import NeedleInHaystackTask


class EvidenceTextProvider(EmbeddingProvider):
    name = "evidence-test"
    model = "unit-vector-test"
    supported_modalities = {ModalityType.TEXT}
    max_text_length = 8192

    def __init__(
        self,
        *,
        truncate_documents: bool = False,
        incomplete_optional_call: int | None = None,
    ) -> None:
        super().__init__()
        self.calls: list[tuple[str | None, int]] = []
        self.cached_calls: list[tuple[str | None, int]] = []
        self.truncate_documents = truncate_documents
        self.incomplete_optional_call = incomplete_optional_call

    def embed_with_cache(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        self.cached_calls.append((task_type, len(inputs)))
        result = self.embed(inputs, dimensions=dimensions, task_type=task_type)
        result.metadata["cache_hit"] = True
        return result

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        self.calls.append((task_type, len(inputs)))
        call_number = len(self.calls)
        response_count = len(inputs)
        if self.truncate_documents and task_type == "retrieval_document":
            response_count -= 1
        optional_fields_exposed = call_number != self.incomplete_optional_call
        embeddings = np.zeros((response_count, 4), dtype=float)
        embeddings[:, 0] = 1.0
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=4,
            model_name=self.model,
            provider=self.name,
            latency_ms=12.5,
            token_usage=response_count * 2 if optional_fields_exposed else None,
            cost_usd=response_count / 1000 if optional_fields_exposed else None,
        )


def test_needle_task_records_fresh_embedding_call_evidence() -> None:
    provider = EvidenceTextProvider()
    task = NeedleInHaystackTask(
        haystack_lengths=[1000],
        needle_positions=[0.0],
        use_mock=True,
        use_cache=False,
    )

    result = task.run(provider)

    assert result.error is None
    assert provider.calls == [
        ("retrieval_query", 5),
        ("retrieval_document", 5),
        ("retrieval_document", 1),
    ]
    assert provider.cached_calls == []
    assert result.details["fresh_provider_calls"] is True
    assert result.details["input_cardinality"] == {
        "queries": 5,
        "documents_with_needle": 5,
        "documents_without_needle": 1,
        "total": 11,
    }
    assert result.details["response_cardinality"]["total"] == 11
    assert result.details["embedding_dimensions"] == [4]
    assert result.details["provider_latency_ms"] == 37.5
    assert result.details["token_usage"] == 22
    assert result.details["cost_usd"] == 0.011
    assert result.details["all_embeddings_finite"] is True
    assert result.details["all_embeddings_unit_norm"] is True
    assert result.details["norm_range"] == {"min": 1.0, "max": 1.0}


def test_needle_task_stops_on_response_cardinality_mismatch() -> None:
    provider = EvidenceTextProvider(truncate_documents=True)
    task = NeedleInHaystackTask(
        haystack_lengths=[1000],
        needle_positions=[0.0],
        use_mock=True,
        use_cache=False,
    )

    result = task.run(provider)

    assert result.error == "documents_with_needle response cardinality mismatch: expected 5, got 4"
    assert provider.calls == [
        ("retrieval_query", 5),
        ("retrieval_document", 5),
    ]


def test_needle_task_optional_totals_require_every_call_value() -> None:
    provider = EvidenceTextProvider(incomplete_optional_call=2)
    task = NeedleInHaystackTask(
        haystack_lengths=[1000],
        needle_positions=[0.0],
        use_mock=True,
        use_cache=False,
    )

    result = task.run(provider)

    assert result.error is None
    assert result.details["embedding_calls"][0]["token_usage"] == 10
    assert result.details["embedding_calls"][1]["token_usage"] is None
    assert result.details["embedding_calls"][2]["token_usage"] == 2
    assert result.details["token_usage"] is None
    assert result.details["cost_usd"] is None


def test_needle_task_default_path_preserves_cache_use() -> None:
    provider = EvidenceTextProvider()
    task = NeedleInHaystackTask(
        haystack_lengths=[1000],
        needle_positions=[0.0],
        use_mock=True,
    )

    result = task.run(provider)

    expected_calls = [
        ("retrieval_query", 5),
        ("retrieval_document", 5),
        ("retrieval_document", 1),
    ]
    assert result.error is None
    assert provider.cached_calls == expected_calls
    assert provider.calls == expected_calls
    assert result.details["cache_enabled"] is True
    assert result.details["fresh_provider_calls"] is False
    assert all(call["cache_hit"] is True for call in result.details["embedding_calls"])
