from __future__ import annotations

import sys
from types import SimpleNamespace

from mm_embed.providers.base import EmbeddingInput, ModalityType
from mm_embed.providers.cohere_provider import CohereProvider
from mm_embed.providers.dashscope_provider import DashScopeProvider
from mm_embed.providers.gemini_provider import GeminiProvider
from mm_embed.providers.jina_provider import JinaProvider
from mm_embed.providers.voyage_provider import VoyageProvider


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
