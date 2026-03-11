"""Provider registry — lazy-load providers to avoid import errors when SDKs are missing."""

from __future__ import annotations

from typing import Any

from mm_embed.providers.base import EmbeddingProvider


# Registry: name -> (module_path, class_name)
PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "dashscope": ("mm_embed.providers.dashscope_provider", "DashScopeProvider"),
    "volcengine": ("mm_embed.providers.volcengine_provider", "VolcEngineProvider"),
    "gemini": ("mm_embed.providers.gemini_provider", "GeminiProvider"),
    "voyage": ("mm_embed.providers.voyage_provider", "VoyageProvider"),
    "cohere": ("mm_embed.providers.cohere_provider", "CohereProvider"),
    "openai": ("mm_embed.providers.openai_provider", "OpenAIProvider"),
    "jina": ("mm_embed.providers.jina_provider", "JinaProvider"),
}


def get_provider(name: str, **kwargs: Any) -> EmbeddingProvider:
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
