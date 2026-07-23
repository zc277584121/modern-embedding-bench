from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from google import genai
from google.genai import types

from mm_embed import cache
from mm_embed.providers.base import EmbeddingInput, ModalityType
from mm_embed.providers.cohere_provider import CohereProvider
from mm_embed.providers.dashscope_provider import DashScopeProvider
from mm_embed.providers.gemini_provider import GeminiProvider
from mm_embed.providers.jina_provider import JinaProvider
from mm_embed.providers.sentence_transformers_provider import SentenceTransformersProvider
from mm_embed.providers.voyage_provider import VoyageProvider


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict]] = []

    def _record(self, method: str, texts: list[str], kwargs: dict) -> np.ndarray:
        self.calls.append((method, texts, kwargs))
        return np.array([[3.0, 4.0, 0.0, 12.0] for _ in texts])

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        return self._record("encode", texts, kwargs)

    def encode_query(self, texts: list[str], **kwargs) -> np.ndarray:
        return self._record("encode_query", texts, kwargs)

    def encode_document(self, texts: list[str], **kwargs) -> np.ndarray:
        return self._record("encode_document", texts, kwargs)


def make_sentence_transformers_provider(model) -> SentenceTransformersProvider:
    provider = SentenceTransformersProvider(model="fake-model", device="cpu")
    provider._st_model = model
    provider._native_dim = 4
    provider.default_dimensions = 4
    return provider


def test_sentence_transformers_retrieval_query_uses_encode_query() -> None:
    model = FakeSentenceTransformer()
    provider = make_sentence_transformers_provider(model)

    provider.embed_text(["query"], task_type="retrieval_query")

    method, texts, kwargs = model.calls[0]
    assert method == "encode_query"
    assert texts == ["query"]
    assert kwargs == {
        "show_progress_bar": False,
        "batch_size": 64,
        "normalize_embeddings": True,
    }


def test_sentence_transformers_retrieval_document_uses_encode_document() -> None:
    model = FakeSentenceTransformer()
    provider = make_sentence_transformers_provider(model)

    provider.embed_text(["document"], task_type="retrieval_document")

    assert model.calls[0][0] == "encode_document"


def test_sentence_transformers_unspecified_task_uses_generic_encode() -> None:
    model = FakeSentenceTransformer()
    provider = make_sentence_transformers_provider(model)

    provider.embed_text(["text"])

    assert model.calls[0][0] == "encode"


def test_sentence_transformers_missing_specialized_method_falls_back_to_encode() -> None:
    class GenericModel:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], dict]] = []

        def encode(self, texts: list[str], **kwargs) -> np.ndarray:
            self.calls.append((texts, kwargs))
            return np.array([[1.0, 0.0, 0.0, 0.0] for _ in texts])

    model = GenericModel()
    provider = make_sentence_transformers_provider(model)

    provider.embed_text(["query"], task_type="retrieval_query")

    assert len(model.calls) == 1
    assert model.calls[0][0] == ["query"]


def test_sentence_transformers_does_not_mask_specialized_method_type_error() -> None:
    class InvalidQueryModel:
        def encode(self, texts: list[str], **kwargs) -> np.ndarray:
            raise AssertionError("Generic encode must not be used as an error fallback")

        def encode_query(self, texts: list[str], **kwargs) -> np.ndarray:
            raise TypeError("query encoding failed")

    provider = make_sentence_transformers_provider(InvalidQueryModel())

    with pytest.raises(TypeError, match="query encoding failed"):
        provider.embed_text(["query"], task_type="retrieval_query")


def test_sentence_transformers_retrieval_truncation_preserves_normalization_and_metadata() -> None:
    model = FakeSentenceTransformer()
    provider = make_sentence_transformers_provider(model)

    result = provider.embed_text(["query"], task_type="retrieval_query", dimensions=2)

    np.testing.assert_allclose(result.embeddings, [[0.6, 0.8]])
    assert result.dimensions == 2
    assert result.model_name == "fake-model"
    assert result.provider == "sentence_transformers"
    assert result.latency_ms >= 0.0


def test_voyage_text_models_use_text_embedding_endpoint(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeVoyageClient:
        def __init__(self, api_key: str | None = None):
            self.api_key = api_key

        def embed(self, **kwargs):
            calls.append(("embed", kwargs))
            return SimpleNamespace(embeddings=[[0.0] * 1024 for _ in kwargs["texts"]], total_tokens=7)

        def multimodal_embed(self, **kwargs):
            calls.append(("multimodal_embed", kwargs))
            return SimpleNamespace(embeddings=[[0.0] * 1024 for _ in kwargs["inputs"]], total_tokens=7)

    monkeypatch.setitem(sys.modules, "voyageai", SimpleNamespace(Client=FakeVoyageClient))

    provider = VoyageProvider(api_key="test", model="voyage-4")
    result = provider.embed([EmbeddingInput(ModalityType.TEXT, "hello")], task_type="retrieval_query")

    assert result.embeddings.shape == (1, 1024)
    assert calls[0][0] == "embed"
    assert calls[0][1]["input_type"] == "query"


def test_cohere_v4_uses_client_v2(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClientV2:
        def __init__(self, api_key: str | None = None):
            self.api_key = api_key

        def embed(self, **kwargs):
            calls.append(kwargs)
            count = len(kwargs.get("texts") or kwargs.get("images") or [])
            return SimpleNamespace(
                embeddings=SimpleNamespace(float_=[[0.0] * 1536 for _ in range(count)]),
                meta=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=11)),
            )

    class FakeClient:
        def __init__(self, api_key: str | None = None):
            self.api_key = api_key

    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=FakeClientV2, Client=FakeClient))

    provider = CohereProvider(api_key="test", model="embed-v4.0")
    result = provider.embed([EmbeddingInput(ModalityType.TEXT, "hello")], task_type="retrieval_query")

    assert result.embeddings.shape == (1, 1536)
    assert calls[0]["input_type"] == "search_query"
    assert "output_dimension" not in calls[0]


def test_jina_v5_defaults_do_not_force_v4_dimension_or_task() -> None:
    captured: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"index": 0, "embedding": [0.0] * 1024}], "usage": {"total_tokens": 3}}

    class FakeHttpx:
        @staticmethod
        def post(url: str, json: dict, headers: dict, timeout: float):
            captured.append(json)
            return FakeResponse()

    provider = JinaProvider(api_key="test", model="jina-embeddings-v5-text-small")
    response, _ = provider._send_batch([{"text": "hello"}], None, "retrieval_query", FakeHttpx)

    assert len(response["data"][0]["embedding"]) == 1024
    assert provider.default_dimensions == 1024
    assert "dimensions" not in captured[0]
    assert "task" not in captured[0]


def test_gemini_default_model_is_formal_embedding_2() -> None:
    provider = GeminiProvider(api_key="test")

    assert provider.model == "gemini-embedding-2"


def test_gemini_embedding_2_wraps_flat_inputs_and_preserves_order(monkeypatch) -> None:
    calls: list[dict] = []
    vector_by_text = {
        "alpha": [1.0, 0.0],
        "beta": [0.0, 1.0],
        "gamma": [1.0, 1.0],
    }

    class FakeModels:
        @staticmethod
        def embed_content(**kwargs):
            calls.append(kwargs)
            texts = [content.parts[0].text for content in kwargs["contents"]]
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=vector_by_text[text]) for text in texts],
            )

    monkeypatch.setattr(genai, "Client", lambda api_key=None: SimpleNamespace(models=FakeModels()))
    monkeypatch.setattr("mm_embed.providers.gemini_provider.time.sleep", lambda _: None)

    provider = GeminiProvider(api_key="test", model="gemini-embedding-2")
    provider.default_batch_size = 2
    result = provider.embed(
        [
            EmbeddingInput(ModalityType.TEXT, "alpha"),
            EmbeddingInput(ModalityType.TEXT, "beta"),
            EmbeddingInput(ModalityType.TEXT, "gamma"),
        ],
        dimensions=2,
    )

    assert [[content.parts[0].text for content in call["contents"]] for call in calls] == [
        ["alpha", "beta"],
        ["gamma"],
    ]
    assert all(isinstance(content, types.Content) for call in calls for content in call["contents"])
    assert all("task_type" not in call["config"] for call in calls)
    np.testing.assert_allclose(result.embeddings, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


@pytest.mark.parametrize(
    ("task_type", "embedding_input", "expected_text"),
    [
        (
            "retrieval_query",
            EmbeddingInput(ModalityType.TEXT, "where is the document?"),
            "task: search result | query: where is the document?",
        ),
        (
            "retrieval_document",
            EmbeddingInput(ModalityType.TEXT, "document body", metadata={"title": "Example"}),
            "title: Example | text: document body",
        ),
        (
            "retrieval_document",
            EmbeddingInput(ModalityType.TEXT, "untitled body"),
            "title: none | text: untitled body",
        ),
    ],
)
def test_gemini_embedding_2_uses_official_retrieval_text_format(
    monkeypatch,
    task_type: str,
    embedding_input: EmbeddingInput,
    expected_text: str,
) -> None:
    calls: list[dict] = []

    class FakeModels:
        @staticmethod
        def embed_content(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])])

    monkeypatch.setattr(genai, "Client", lambda api_key=None: SimpleNamespace(models=FakeModels()))

    provider = GeminiProvider(api_key="test", model="gemini-embedding-2")
    provider.embed([embedding_input], dimensions=2, task_type=task_type)

    assert calls[0]["contents"][0].parts[0].text == expected_text
    assert calls[0]["config"] == {"output_dimensionality": 2}


def test_gemini_legacy_model_preserves_list_and_task_type_routing(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeModels:
        @staticmethod
        def embed_content(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[float(index), 0.0]) for index, _ in enumerate(kwargs["contents"])],
            )

    monkeypatch.setattr(genai, "Client", lambda api_key=None: SimpleNamespace(models=FakeModels()))

    provider = GeminiProvider(api_key="test", model="gemini-embedding-001")
    result = provider.embed(
        [EmbeddingInput(ModalityType.TEXT, "alpha"), EmbeddingInput(ModalityType.TEXT, "beta")],
        dimensions=2,
        task_type="retrieval_query",
    )

    assert calls[0]["contents"] == ["alpha", "beta"]
    assert calls[0]["config"] == {
        "output_dimensionality": 2,
        "task_type": "RETRIEVAL_QUERY",
    }
    np.testing.assert_allclose(result.embeddings, [[0.0, 0.0], [1.0, 0.0]])


def test_gemini_rejects_malformed_embedding_cardinality(monkeypatch) -> None:
    class FakeModels:
        @staticmethod
        def embed_content(**kwargs):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])])

    monkeypatch.setattr(genai, "Client", lambda api_key=None: SimpleNamespace(models=FakeModels()))

    provider = GeminiProvider(api_key="test", model="gemini-embedding-2")

    with pytest.raises(ValueError, match="returned 1 embeddings for 2 logical inputs"):
        provider.embed(
            [EmbeddingInput(ModalityType.TEXT, "alpha"), EmbeddingInput(ModalityType.TEXT, "beta")],
            dimensions=2,
        )


def test_gemini_cache_uses_effective_text_and_ignores_malformed_rows(monkeypatch) -> None:
    captured_cache_inputs: list[str] = []
    api_calls: list[dict] = []

    def fake_make_cache_key(**kwargs):
        captured_cache_inputs.extend(kwargs["inputs_content"])
        return "test-cache-key"

    class FakeModels:
        @staticmethod
        def embed_content(**kwargs):
            api_calls.append(kwargs)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[float(index), 1.0]) for index, _ in enumerate(kwargs["contents"])],
            )

    monkeypatch.setattr(cache, "make_cache_key", fake_make_cache_key)
    monkeypatch.setattr(cache, "cache_get", lambda *args: np.array([[99.0, 99.0]]))
    monkeypatch.setattr(cache, "cache_put", lambda *args: None)
    monkeypatch.setattr(genai, "Client", lambda api_key=None: SimpleNamespace(models=FakeModels()))

    provider = GeminiProvider(api_key="test", model="gemini-embedding-2")
    result = provider.embed_with_cache(
        [
            EmbeddingInput(ModalityType.TEXT, "query one"),
            EmbeddingInput(ModalityType.TEXT, "query two"),
        ],
        dimensions=2,
        task_type="retrieval_query",
    )

    assert captured_cache_inputs == [
        "gemini-embedding-2-flat-content-v1\0task: search result | query: query one",
        "gemini-embedding-2-flat-content-v1\0task: search result | query: query two",
    ]
    assert len(api_calls) == 1
    np.testing.assert_allclose(result.embeddings, [[0.0, 1.0], [1.0, 1.0]])


def test_gemini_cache_ignores_one_dimensional_array_when_length_matches(monkeypatch) -> None:
    api_calls: list[dict] = []

    class FakeModels:
        @staticmethod
        def embed_content(**kwargs):
            api_calls.append(kwargs)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[float(index), 1.0]) for index, _ in enumerate(kwargs["contents"])],
            )

    monkeypatch.setattr(cache, "cache_get", lambda *args: np.array([99.0, 99.0]))
    monkeypatch.setattr(cache, "cache_put", lambda *args: None)
    monkeypatch.setattr(genai, "Client", lambda api_key=None: SimpleNamespace(models=FakeModels()))

    provider = GeminiProvider(api_key="test", model="gemini-embedding-2")
    result = provider.embed_with_cache(
        [
            EmbeddingInput(ModalityType.TEXT, "alpha"),
            EmbeddingInput(ModalityType.TEXT, "beta"),
        ],
        dimensions=2,
    )

    assert len(api_calls) == 1
    assert result.metadata.get("cache_hit") is not True
    np.testing.assert_allclose(result.embeddings, [[0.0, 1.0], [1.0, 1.0]])


def test_dashscope_qwen3_vl_uses_multimodal_endpoint(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeTextEmbedding:
        @staticmethod
        def call(**kwargs):
            raise AssertionError(f"Unexpected text embedding call: {kwargs}")

    class FakeMultiModalEmbedding:
        @staticmethod
        def call(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output={"embeddings": [{"embedding": [0.0] * 1024}]},
                usage={"total_tokens": 3},
            )

    monkeypatch.setitem(
        sys.modules,
        "dashscope",
        SimpleNamespace(
            api_key=None,
            TextEmbedding=FakeTextEmbedding,
            MultiModalEmbedding=FakeMultiModalEmbedding,
        ),
    )

    provider = DashScopeProvider(api_key="test", model="qwen3-vl-embedding")
    result = provider.embed([EmbeddingInput(ModalityType.TEXT, "hello")])

    assert result.embeddings.shape == (1, 1024)
    assert calls[0]["model"] == "qwen3-vl-embedding"
    assert calls[0]["input"] == [{"text": "hello"}]
