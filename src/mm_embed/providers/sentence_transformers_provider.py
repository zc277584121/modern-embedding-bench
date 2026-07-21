"""SentenceTransformers provider — local GPU embedding with HuggingFace models."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingProvider, EmbeddingResult, ModalityType

logger = logging.getLogger(__name__)

# Model names (or substrings) that indicate a CLIP-style vision-language model
_CLIP_PATTERNS = ("clip", "siglip", "openclip", "ViT-B", "ViT-L", "ViT-H", "ViT-G")


class SentenceTransformersProvider(EmbeddingProvider):
    """Local GPU embedding using sentence-transformers library.

    Supports any HuggingFace model compatible with sentence-transformers.
    Runs at full precision (no quantization) on GPU.

    For CLIP models (e.g. clip-ViT-B-32, clip-ViT-L-14), IMAGE modality
    is automatically enabled. CLIP models accept mixed lists of strings
    and PIL Images in a single encode() call.

    Example models:
        - BAAI/bge-m3: 1024d, multilingual, strong on MTEB
        - clip-ViT-B-32: 512d, classic CLIP baseline
        - clip-ViT-L-14: 768d, larger CLIP
    """

    name = "sentence_transformers"
    max_text_length = 8192

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model
        self.model = model  # for compatibility with getattr(provider, "model")
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._st_model = None
        self._native_dim: int | None = None
        self._is_clip: bool = any(p.lower() in model.lower() for p in _CLIP_PATTERNS)

        # Set modalities and MRL support based on model type
        if self._is_clip:
            self.supported_modalities = {ModalityType.TEXT, ModalityType.IMAGE}
        else:
            self.supported_modalities = {ModalityType.TEXT}
        # All models support MRL evaluation via truncation on the eval side
        self.supports_mrl = True

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._st_model is not None:
            return

        # Patch for models with custom code that lack all_tied_weights_keys
        # (needed for transformers >= 5.x compatibility with older custom models)
        from transformers import modeling_utils
        _orig_mark = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized

        def _patched_mark(self_model, loading_info):
            if not hasattr(self_model, "all_tied_weights_keys"):
                self_model.all_tied_weights_keys = {}
            return _orig_mark(self_model, loading_info)

        modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark

        from sentence_transformers import SentenceTransformer

        logger.info("Loading model %s on %s...", self.model_name_or_path, self.device)
        self._st_model = SentenceTransformer(
            self.model_name_or_path,
            device=self.device,
            trust_remote_code=True,
        )
        # Detect native dimensions
        test_emb = self._st_model.encode(["test"], show_progress_bar=False)
        self._native_dim = test_emb.shape[1]
        self.default_dimensions = self._native_dim
        logger.info(
            "Model loaded: %s, native dim=%d, is_clip=%s",
            self.model_name_or_path, self._native_dim, self._is_clip,
        )

    @staticmethod
    def _load_pil_image(content: str | bytes | Path):
        """Convert image content to a PIL Image."""
        from PIL import Image

        if isinstance(content, (str, Path)):
            path = Path(content)
            if path.exists():
                return Image.open(path).convert("RGB")
        if isinstance(content, bytes):
            return Image.open(io.BytesIO(content)).convert("RGB")
        raise ValueError(f"Cannot load image from {type(content)}")

    def embed(
        self,
        inputs: list,
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        self._load_model()

        # For CLIP models: build a mixed list of strings and PIL Images
        if self._is_clip:
            encode_inputs = []
            for inp in inputs:
                if inp.modality == ModalityType.TEXT:
                    encode_inputs.append(inp.content)
                elif inp.modality == ModalityType.IMAGE:
                    encode_inputs.append(self._load_pil_image(inp.content))
                else:
                    raise ValueError(f"Unsupported modality: {inp.modality}")
            return self._encode_mixed(encode_inputs, dimensions)

        # For text-only models
        texts = []
        for inp in inputs:
            if inp.modality != ModalityType.TEXT:
                raise ValueError(f"Model {self.model_name_or_path} only supports text, got {inp.modality}")
            texts.append(inp.content)
        return self._embed_texts(texts, dimensions, task_type)

    def embed_text(self, texts: list[str], **kwargs: Any) -> EmbeddingResult:
        self._load_model()
        return self._embed_texts(texts, kwargs.get("dimensions"), kwargs.get("task_type"))

    def _encode_mixed(
        self,
        inputs: list,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        """Encode a mixed list of strings and PIL Images (for CLIP models)."""
        import time

        encode_kwargs: dict[str, Any] = {
            "show_progress_bar": len(inputs) > 50,
            "batch_size": 32,
            "normalize_embeddings": True,
        }

        start = time.perf_counter()
        embeddings = self._st_model.encode(inputs, **encode_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        # MRL: truncate if requested (CLIP models also support this)
        dim = dimensions or self._native_dim
        if dim and dim < embeddings.shape[1]:
            embeddings = embeddings[:, :dim]
            # Re-normalize after truncation
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=embeddings.shape[1],
            model_name=self.model_name_or_path,
            provider=self.name,
            latency_ms=latency_ms,
        )

    def _embed_texts(
        self,
        texts: list[str],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        import time

        # Build encode kwargs
        encode_kwargs: dict[str, Any] = {
            "show_progress_bar": len(texts) > 100,
            "batch_size": 64,
            "normalize_embeddings": True,
        }

        encode_method = self._st_model.encode
        if task_type == "retrieval_query":
            specialized_method = getattr(self._st_model, "encode_query", None)
            if callable(specialized_method):
                encode_method = specialized_method
        elif task_type == "retrieval_document":
            specialized_method = getattr(self._st_model, "encode_document", None)
            if callable(specialized_method):
                encode_method = specialized_method

        start = time.perf_counter()
        embeddings = encode_method(texts, **encode_kwargs)

        latency_ms = (time.perf_counter() - start) * 1000

        # MRL: truncate if requested
        dim = dimensions or self._native_dim
        if dim and dim < embeddings.shape[1]:
            embeddings = embeddings[:, :dim]
            # Re-normalize after truncation
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=embeddings.shape[1],
            model_name=self.model_name_or_path,
            provider=self.name,
            latency_ms=latency_ms,
        )
