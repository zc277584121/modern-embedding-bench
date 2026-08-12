"""Provider registry — lazy-load providers to avoid import errors when SDKs are missing."""

from __future__ import annotations

from typing import Any

# Registry: name -> (module_path, class_name)
PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "dashscope": ("mm_embed.providers.dashscope_provider", "DashScopeProvider"),
    "volcengine": ("mm_embed.providers.volcengine_provider", "VolcEngineProvider"),
    "gemini": ("mm_embed.providers.gemini_provider", "GeminiProvider"),
    "voyage": ("mm_embed.providers.voyage_provider", "VoyageProvider"),
    "cohere": ("mm_embed.providers.cohere_provider", "CohereProvider"),
    "openai": ("mm_embed.providers.openai_provider", "OpenAIProvider"),
    "geevec_api": ("mm_embed.providers.geevec_provider", "GeeVecAPIProvider"),
    "geevec_lite": ("mm_embed.providers.geevec_provider", "GeeVecLiteProvider"),
    "jina": ("mm_embed.providers.jina_provider", "JinaProvider"),
    "ark": ("mm_embed.providers.ark_provider", "ArkProvider"),
    "transformers": ("mm_embed.providers.transformers_provider", "TransformersProvider"),
    "ollama": ("mm_embed.providers.ollama_provider", "OllamaProvider"),
    "sentence_transformers": ("mm_embed.providers.sentence_transformers_provider", "SentenceTransformersProvider"),
    "deterministic_sparse_fixture": (
        "mm_embed.providers.deterministic_sparse_provider",
        "DeterministicSparseFixtureProvider",
    ),
    "deterministic_multi_vector_fixture": (
        "mm_embed.providers.deterministic_multi_vector_provider",
        "DeterministicMultiVectorFixtureProvider",
    ),
    "opensearch_neural_sparse": (
        "mm_embed.providers.opensearch_neural_sparse_provider",
        "OpenSearchNeuralSparseProvider",
    ),
}


def get_provider(name: str, **kwargs: Any) -> Any:
    """Instantiate a provider by name.

    Args:
        name: Provider name (e.g. "dashscope", "gemini")
        **kwargs: Passed to provider constructor (api_key, model, etc.)

    Returns:
        An initialized EmbeddingProvider instance.

    Raises:
        KeyError: If provider name is not registered.
        ImportError: If the provider's SDK is not installed.
    """
    if name not in PROVIDER_REGISTRY:
        available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise KeyError(f"Unknown provider '{name}'. Available: {available}")

    module_path, class_name = PROVIDER_REGISTRY[name]

    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
