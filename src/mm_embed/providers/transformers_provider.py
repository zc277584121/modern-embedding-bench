"""Transformers provider — native HuggingFace models that need custom loading.

Supports:
- Qwen3-VL-Embedding-2B: multimodal (text + image), uses Qwen3VLEmbedder
- SigLIP 2: vision-language model for cross-modal embeddings
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from mm_embed.providers.base import EmbeddingInput, EmbeddingProvider, EmbeddingResult, ModalityType

logger = logging.getLogger(__name__)

# Model type detection patterns
_QWEN_VL_PATTERNS = ("qwen3-vl-embedding", "qwen3_vl_embedding")
_SIGLIP_PATTERNS = ("siglip",)


class TransformersProvider(EmbeddingProvider):
    """Native HuggingFace transformers models with custom loading logic.

    Supports models that don't work with sentence-transformers:
    - Qwen3-VL-Embedding: multimodal, uses special embedder class
    - SigLIP 2: CLIP-style vision-language model
    """

    name = "transformers"
    max_text_length = 8192

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str | None = None,
        torch_dtype: str = "auto",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model
        self.model = model
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch_dtype = torch_dtype
        self._loaded_model = None
        self._model_type: str | None = None

        # Detect model type
        model_lower = model.lower()
        if any(p in model_lower for p in _QWEN_VL_PATTERNS):
            self._model_type = "qwen_vl"
            self.supported_modalities = {ModalityType.TEXT, ModalityType.IMAGE}
            self.default_dimensions = 2048
            self.supports_mrl = True
        elif any(p in model_lower for p in _SIGLIP_PATTERNS):
            self._model_type = "siglip"
            self.supported_modalities = {ModalityType.TEXT, ModalityType.IMAGE}
            self.default_dimensions = 1152
            self.supports_mrl = False
        else:
            raise ValueError(f"Unknown model type for TransformersProvider: {model}")

    def _load_model(self):
        if self._loaded_model is not None:
            return

        if self._model_type == "qwen_vl":
            self._load_qwen_vl()
        elif self._model_type == "siglip":
            self._load_siglip()

    def _load_qwen_vl(self):
        """Load Qwen3-VL-Embedding using transformers directly."""
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoProcessor

        model_path = Path(self.model_name_or_path)
        if not model_path.exists():
            from huggingface_hub import snapshot_download
            model_path = Path(snapshot_download(self.model_name_or_path))

        dtype_map = {
            "auto": torch.bfloat16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

        logger.info("Loading Qwen3-VL-Embedding from %s on %s (dtype=%s)...",
                     model_path, self.device, torch_dtype)

        model = AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        model.eval()

        processor = AutoProcessor.from_pretrained(str(model_path), padding_side="right")

        self._loaded_model = {
            "model": model,
            "processor": processor,
            "model_path": model_path,
        }
        logger.info("Qwen3-VL-Embedding loaded, dim=%d", self.default_dimensions)

    def _load_siglip(self):
        """Load SigLIP 2 model for cross-modal embeddings."""
        import torch
        from transformers import AutoModel, AutoProcessor

        logger.info("Loading SigLIP 2 from %s on %s...", self.model_name_or_path, self.device)

        dtype_map = {
            "auto": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype, torch.float32)

        model = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        model.eval()

        processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        self._loaded_model = {"model": model, "processor": processor}
        logger.info("SigLIP 2 loaded, dim=%d", self.default_dimensions)

    def embed(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        self._load_model()

        if self._model_type == "qwen_vl":
            return self._embed_qwen_vl(inputs, dimensions, task_type)
        elif self._model_type == "siglip":
            return self._embed_siglip(inputs, dimensions)
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    def _embed_qwen_vl(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> EmbeddingResult:
        """Embed using Qwen3-VL-Embedding with native transformers."""
        import torch
        import torch.nn.functional as F
        from qwen_vl_utils.vision_process import process_vision_info

        model = self._loaded_model["model"]
        processor = self._loaded_model["processor"]

        # Build conversations in the Qwen3-VL format
        instruction = "Represent the user's input."
        if task_type == "retrieval_query":
            instruction = "Find relevant documents for this query."
        elif task_type == "retrieval_document":
            instruction = "Represent this document for retrieval."

        conversations_list = []
        for inp in inputs:
            content = []
            if inp.modality == ModalityType.TEXT:
                content.append({"type": "text", "text": inp.content})
            elif inp.modality == ModalityType.IMAGE:
                pil_img = self._load_pil_image(inp.content)
                content.append({"type": "image", "image": pil_img, "min_pixels": 4 * 32 * 32, "max_pixels": 1800 * 32 * 32})
            else:
                raise ValueError(f"Unsupported modality for Qwen3-VL: {inp.modality}")

            conv = [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content},
            ]
            conversations_list.append(conv)

        # Process in batches
        batch_size = 4
        all_embeddings = []
        total_latency = 0.0
        n_batches = (len(conversations_list) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(conversations_list))
            batch_convs = conversations_list[start_idx:end_idx]

            if n_batches > 1 and (batch_idx + 1) % 5 == 0:
                logger.info("Qwen3-VL batch %d/%d...", batch_idx + 1, n_batches)

            start_t = time.perf_counter()

            # Apply chat template
            texts = processor.apply_chat_template(
                batch_convs, add_generation_prompt=True, tokenize=False
            )

            # Process vision info
            try:
                images, video_inputs, video_kwargs = process_vision_info(
                    batch_convs, image_patch_size=16,
                    return_video_metadata=True, return_video_kwargs=True
                )
            except Exception:
                images = None
                video_inputs = None
                video_kwargs = {"do_sample_frames": False}

            # Empty lists cause tensor shape mismatches — normalize to None
            if not images:
                images = None
            if not video_inputs:
                video_inputs = None

            if video_inputs is not None:
                videos, video_metadata = zip(*video_inputs)
                videos = list(videos)
                video_metadata = list(video_metadata)
            else:
                videos, video_metadata = None, None

            proc_inputs = processor(
                text=texts, images=images, videos=videos, video_metadata=video_metadata,
                truncation=True, max_length=8192, padding=True, do_resize=False,
                return_tensors="pt", **video_kwargs
            )
            proc_inputs = {k: v.to(self.device) for k, v in proc_inputs.items()}

            with torch.no_grad():
                outputs = model(**proc_inputs)
                hidden = outputs.last_hidden_state
                attn_mask = proc_inputs["attention_mask"]
                # Last-token pooling
                flipped = attn_mask.flip(dims=[1])
                last_pos = flipped.argmax(dim=1)
                col = attn_mask.shape[1] - last_pos - 1
                row = torch.arange(hidden.shape[0], device=hidden.device)
                embs = hidden[row, col]
                embs = F.normalize(embs, p=2, dim=-1)

            latency = (time.perf_counter() - start_t) * 1000
            total_latency += latency
            all_embeddings.append(embs.cpu().float().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)

        # MRL: truncate if requested
        dim = dimensions or self.default_dimensions
        if dim < embeddings.shape[1]:
            embeddings = embeddings[:, :dim]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=embeddings.shape[1],
            model_name=self.model_name_or_path,
            provider=self.name,
            latency_ms=total_latency,
        )

    def _embed_siglip(
        self,
        inputs: list[EmbeddingInput],
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        """Embed using SigLIP 2 — separate text and image encoding paths."""
        import torch
        from PIL import Image

        model = self._loaded_model["model"]
        processor = self._loaded_model["processor"]

        # Separate text and image inputs, preserving original order
        text_indices = []
        image_indices = []
        texts = []
        images = []
        for i, inp in enumerate(inputs):
            if inp.modality == ModalityType.TEXT:
                text_indices.append(i)
                texts.append(inp.content)
            elif inp.modality == ModalityType.IMAGE:
                image_indices.append(i)
                images.append(self._load_pil_image(inp.content))
            else:
                raise ValueError(f"Unsupported modality for SigLIP: {inp.modality}")

        all_embeddings = np.zeros((len(inputs), self.default_dimensions), dtype=np.float32)
        total_latency = 0.0

        # Encode texts
        if texts:
            start_t = time.perf_counter()
            batch_size = 32
            text_embs_list = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                text_inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items() if k in ("input_ids", "attention_mask")}
                with torch.no_grad():
                    text_outputs = model.get_text_features(**text_inputs)
                    # Handle both tensor and ModelOutput return types
                    if hasattr(text_outputs, "pooler_output"):
                        text_outputs = text_outputs.pooler_output
                text_embs_list.append(text_outputs.cpu().numpy())
            text_embs = np.concatenate(text_embs_list, axis=0)
            # Normalize
            norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
            text_embs = text_embs / np.maximum(norms, 1e-12)
            for idx, emb in zip(text_indices, text_embs):
                all_embeddings[idx] = emb
            total_latency += (time.perf_counter() - start_t) * 1000

        # Encode images
        if images:
            start_t = time.perf_counter()
            batch_size = 16
            image_embs_list = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                image_inputs = processor(images=batch, return_tensors="pt", padding=True)
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items() if k == "pixel_values"}
                with torch.no_grad():
                    image_outputs = model.get_image_features(**image_inputs)
                    if hasattr(image_outputs, "pooler_output"):
                        image_outputs = image_outputs.pooler_output
                image_embs_list.append(image_outputs.cpu().numpy())
            image_embs = np.concatenate(image_embs_list, axis=0)
            # Normalize
            norms = np.linalg.norm(image_embs, axis=1, keepdims=True)
            image_embs = image_embs / np.maximum(norms, 1e-12)
            for idx, emb in zip(image_indices, image_embs):
                all_embeddings[idx] = emb
            total_latency += (time.perf_counter() - start_t) * 1000

        return EmbeddingResult(
            embeddings=all_embeddings,
            dimensions=self.default_dimensions,
            model_name=self.model_name_or_path,
            provider=self.name,
            latency_ms=total_latency,
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
