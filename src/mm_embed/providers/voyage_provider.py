"""Voyage AI provider — Voyage Multimodal 3.5 / Voyage 3 Large."""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType

logger = logging.getLogger(__name__)


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

        # Build all inputs
        voyage_inputs = []
        for inp in inputs:
            voyage_inputs.append(self._build_input(inp))

        input_type = None
        if task_type == "retrieval_query":
            input_type = "query"
        elif task_type == "retrieval_document":
            input_type = "document"

        # Batch to avoid rate limits (free tier: 3 RPM)
        # Voyage SDK accepts up to ~128 text inputs, but for images keep batches small
        has_images = any(inp.modality != ModalityType.TEXT for inp in inputs)
        batch_size = 1 if has_images else 50

        all_embeddings = []
        total_latency = 0.0
        total_tokens = 0
        n_batches = (len(voyage_inputs) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(voyage_inputs))
            batch = voyage_inputs[start:end]

            if n_batches > 1 and (batch_idx + 1) % 5 == 0:
                logger.info("Voyage batch %d/%d...", batch_idx + 1, n_batches)

            embed_kwargs: dict[str, Any] = {
                "inputs": batch,
                "model": self.model,
            }
            if input_type:
                embed_kwargs["input_type"] = input_type
            if dim != self.default_dimensions:
                embed_kwargs["output_dimension"] = dim

            def _call(kw=embed_kwargs):
                return client.multimodal_embed(**kw)

            response, latency = self._call_with_retry(_call)
            total_latency += latency

            all_embeddings.extend(response.embeddings)
            if hasattr(response, "total_tokens") and response.total_tokens:
                total_tokens += response.total_tokens

            # Rate limit: sleep between batches (25s for free tier 3 RPM)
            if batch_idx < n_batches - 1:
                sleep_time = 25.0 if has_images else 1.0
                if has_images:
                    logger.info("Sleeping %.0fs for rate limit...", sleep_time)
                time.sleep(sleep_time)

        embeddings = np.array(all_embeddings)
        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=embeddings.shape[1],
            model_name=self.model,
            provider=self.name,
            latency_ms=total_latency,
            token_usage=total_tokens if total_tokens > 0 else None,
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
            return [img]

        if inp.modality == ModalityType.DOCUMENT:
            # Voyage treats screenshots/PDFs as images
            from PIL import Image as PILImage

            if isinstance(inp.content, (str, Path)):
                img = PILImage.open(inp.content)
            else:
                import io
                img = PILImage.open(io.BytesIO(inp.content))
            return [img]

        raise ValueError(f"Unsupported modality for Voyage: {inp.modality}")
