"""Voyage AI provider — Voyage Multimodal 3.5 / Voyage 3 Large."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType


class VoyageProvider(EmbeddingProvider):
    """Voyage AI embedding models.

    Models:
        - voyage-multimodal-3.5: Multimodal (text+image+video+screenshot/PDF)
        - voyage-3-large: Text-only, RTEB retrieval #1
        - voyage-3: Text-only, balanced
        - voyage-3-lite: Text-only, fast and cheap

    Pricing: $0.06/M tokens (multimodal), $0.06/M tokens (3-large)
    Access: Requires VPN from China mainland. Free tier: $3 credit.
    """

    name = "voyage"
    supported_modalities = {ModalityType.TEXT, ModalityType.IMAGE, ModalityType.VIDEO, ModalityType.DOCUMENT}
    max_text_length = 32768
    default_dimensions = 1024
    supports_mrl = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-multimodal-3.5",
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key or os.environ.get("VOYAGE_API_KEY"), **kwargs)
        self.model = model

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        import voyageai

        client = voyageai.Client(api_key=self.api_key)
        dim = dimensions or self.default_dimensions

        # Voyage multimodal accepts a list of content items
        # Each item can be: str (text), or [[image_base64, None, "png"]] (image),
        # or a mixed list for interleaved text+image
        voyage_inputs = []
        for inp in inputs:
            voyage_inputs.append(self._build_input(inp))

        input_type = None
        if task_type == "retrieval_query":
            input_type = "query"
        elif task_type == "retrieval_document":
            input_type = "document"

        embed_kwargs: dict[str, Any] = {
            "inputs": voyage_inputs,
            "model": self.model,
        }
        if input_type:
            embed_kwargs["input_type"] = input_type
        if dim != self.default_dimensions:
            embed_kwargs["output_dimension"] = dim

        response, latency = self._timed_call(client.multimodal_embed, **embed_kwargs)

        embeddings = np.array(response.embeddings)
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=embeddings.shape[1],
            model_name=self.model,
            provider=self.name,
            latency_ms=latency,
            token_usage=response.total_tokens if hasattr(response, "total_tokens") else None,
        )

    def _build_input(self, inp: EmbeddingInput) -> Any:
        """Build a single input for Voyage multimodal API."""
        if inp.modality == ModalityType.TEXT:
            return [inp.content]

        if inp.modality == ModalityType.IMAGE:
            from PIL import Image as PILImage

            if isinstance(inp.content, (str, Path)):
                img = PILImage.open(inp.content)
            else:
                import io
                img = PILImage.open(io.BytesIO(inp.content))
            return [[img]]

        if inp.modality == ModalityType.DOCUMENT:
            # Voyage treats screenshots/PDFs as images
            from PIL import Image as PILImage

            if isinstance(inp.content, (str, Path)):
                img = PILImage.open(inp.content)
            else:
                import io
                img = PILImage.open(io.BytesIO(inp.content))
            return [[img]]

        raise ValueError(f"Unsupported modality for Voyage: {inp.modality}")
