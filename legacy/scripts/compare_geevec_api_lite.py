"""Compare GeeVec API embeddings with a local Lite model.

This script is intentionally small and diagnostic-focused. It checks whether the
API model behaves like the local Lite model by comparing direct vector alignment
and within-model similarity structure on the same text set.

Usage:
    uv run --extra local python scripts/compare_geevec_api_lite.py \
        --local-model Geeknow/GeeVec-Embeddings-Lite \
        --api-model geevec-embeddings-general-1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from scipy.stats import spearmanr


DEFAULT_BASE_URL = "https://www.geevec.com"
DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast fox leaped over a sleepy dog.",
    "深度学习模型的可解释性一直是学术界关注的热点问题。",
    "Model interpretability remains a major topic in deep learning research.",
    "Write a Python function that computes cosine similarity between two vectors.",
    "def cosine_similarity(a, b): return dot(a, b) / (norm(a) * norm(b))",
    "A database index can speed up retrieval by reducing the search space.",
    "Vector databases store embeddings and support approximate nearest neighbor search.",
    "The contract is valid only after both parties sign the final version.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "Explain why chain-of-thought style reasoning can help solve multi-step math problems.",
    "A long-context embedding model should preserve a small clue hidden near the end of a document.",
]


@dataclass
class CompareReport:
    api_model: str
    local_model: str
    api_dim: int
    local_dim: int
    text_count: int
    direct_diag_cosine_mean: float | None
    direct_diag_cosine_min: float | None
    direct_self_match_rate: float | None
    similarity_spearman: float
    similarity_pearson: float
    api_hash_rounded_6: str
    local_hash_rounded_6: str
    interpretation: str


def _unit_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols]


def _rounded_hash(embeddings: np.ndarray) -> str:
    rounded = np.round(embeddings.astype(np.float64), 6)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def _post_embeddings(
    client: httpx.Client,
    url: str,
    api_key: str,
    model: str,
    texts: list[str],
    dimensions: int | None,
    max_retries: int,
) -> list[list[float]]:
    payload: dict[str, Any] = {
        "input": texts,
        "model": model,
        "encoding_format": "float",
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
            rows = sorted(body["data"], key=lambda item: item["index"])
            return [row["embedding"] for row in rows]
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"GeeVec API embedding request failed after retries: {last_error}")


def embed_api(
    base_url: str,
    api_key: str,
    model: str,
    texts: list[str],
    dimensions: int | None,
    batch_size: int,
    max_retries: int,
    timeout: float,
) -> np.ndarray:
    url = f"{base_url.rstrip('/')}/openapi/v1/embeddings"
    vectors: list[list[float]] = []
    with httpx.Client(timeout=timeout) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(
                _post_embeddings(
                    client=client,
                    url=url,
                    api_key=api_key,
                    model=model,
                    texts=batch,
                    dimensions=dimensions,
                    max_retries=max_retries,
                )
            )
            if start + batch_size < len(texts):
                time.sleep(1.0)
    return np.array(vectors, dtype=np.float32)


def embed_local(
    model_name: str,
    texts: list[str],
    device: str | None,
    batch_size: int,
    domain: str | None,
) -> np.ndarray:
    from transformers import modeling_utils
    from sentence_transformers import SentenceTransformer

    # Some custom embedding models still expose the Transformers 4.x-style
    # `_tied_weights_keys` list. Transformers 5.x expects a mapping.
    _orig_expand = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys
    _orig_mark = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized

    def _patched_expand(self_model, all_submodels: bool = False):
        tied_mapping = getattr(self_model, "_tied_weights_keys", None)
        if isinstance(tied_mapping, list):
            return {}
        return _orig_expand(self_model, all_submodels=all_submodels)

    def _patched_mark(self_model, loading_info):
        if not hasattr(self_model, "all_tied_weights_keys"):
            self_model.all_tied_weights_keys = {}
        return _orig_mark(self_model, loading_info)

    modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = _patched_expand
    modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device:
        kwargs["device"] = device
    try:
        model = SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load local Lite model '{model_name}'. "
            "Pass the exact Hugging Face model id or a local snapshot path via --local-model."
        ) from exc
    encode_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "normalize_embeddings": False,
        "show_progress_bar": False,
    }
    if domain and domain != "general":
        encode_kwargs["domain"] = domain
    embeddings = model.encode(texts, **encode_kwargs)
    return np.array(embeddings, dtype=np.float32)


def compare_embeddings(api_emb: np.ndarray, local_emb: np.ndarray, api_model: str, local_model: str) -> CompareReport:
    api_norm = _unit_normalize(api_emb)
    local_norm = _unit_normalize(local_emb)

    direct_diag_mean: float | None = None
    direct_diag_min: float | None = None
    direct_self_match_rate: float | None = None
    if api_norm.shape[1] == local_norm.shape[1]:
        cross = api_norm @ local_norm.T
        diag = np.diag(cross)
        direct_diag_mean = float(np.mean(diag))
        direct_diag_min = float(np.min(diag))
        direct_self_match_rate = float(np.mean(np.argmax(cross, axis=1) == np.arange(cross.shape[0])))

    api_sim = api_norm @ api_norm.T
    local_sim = local_norm @ local_norm.T
    api_vals = _upper_triangle_values(api_sim)
    local_vals = _upper_triangle_values(local_sim)
    similarity_spearman = float(spearmanr(api_vals, local_vals).statistic)
    similarity_pearson = float(np.corrcoef(api_vals, local_vals)[0, 1])

    interpretation = interpret(
        direct_diag_mean=direct_diag_mean,
        direct_self_match_rate=direct_self_match_rate,
        similarity_spearman=similarity_spearman,
    )

    return CompareReport(
        api_model=api_model,
        local_model=local_model,
        api_dim=int(api_emb.shape[1]),
        local_dim=int(local_emb.shape[1]),
        text_count=int(api_emb.shape[0]),
        direct_diag_cosine_mean=direct_diag_mean,
        direct_diag_cosine_min=direct_diag_min,
        direct_self_match_rate=direct_self_match_rate,
        similarity_spearman=similarity_spearman,
        similarity_pearson=similarity_pearson,
        api_hash_rounded_6=_rounded_hash(api_emb),
        local_hash_rounded_6=_rounded_hash(local_emb),
        interpretation=interpretation,
    )


def interpret(
    direct_diag_mean: float | None,
    direct_self_match_rate: float | None,
    similarity_spearman: float,
) -> str:
    if direct_diag_mean is not None and direct_diag_mean > 0.999 and direct_self_match_rate == 1.0:
        return "The API and Lite outputs are effectively identical on this sample."
    if direct_diag_mean is not None and direct_diag_mean > 0.98 and direct_self_match_rate and direct_self_match_rate >= 0.95:
        return "The API and Lite outputs are extremely close; this is strong evidence they share the same embedding space."
    if similarity_spearman > 0.95:
        return "The API and Lite similarity rankings are very similar, but the raw vectors are not proven identical."
    if similarity_spearman > 0.80:
        return "The API and Lite models behave similarly on this small sample, but they are not identical by this test."
    return "The API and Lite models differ clearly on this sample."


def load_texts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_TEXTS
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            texts.append(line)
    if len(texts) < 4:
        raise ValueError("Need at least four non-empty texts for a meaningful similarity comparison.")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("GEE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-model", default="geevec-embeddings-general-1.0")
    parser.add_argument("--local-model", default=os.environ.get("GEE_LITE_MODEL", "Geeknow/GeeVec-Embeddings-Lite"))
    parser.add_argument("--local-domain", choices=["general", "coding", "reasoning"])
    parser.add_argument("--texts-file", type=Path)
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--api-batch-size", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--device", default=os.environ.get("CUDA_DEVICE"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/geevec_api_vs_lite_compare.json"))
    args = parser.parse_args()

    api_key = os.environ.get("GEE_API_KEY")
    if not api_key:
        raise SystemExit("GEE_API_KEY is not set")

    texts = load_texts(args.texts_file)
    print(f"Embedding {len(texts)} texts with API model: {args.api_model}", flush=True)
    api_emb = embed_api(
        base_url=args.base_url,
        api_key=api_key,
        model=args.api_model,
        texts=texts,
        dimensions=args.dimensions,
        batch_size=args.api_batch_size,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    print(f"Embedding {len(texts)} texts with local model: {args.local_model}", flush=True)
    local_emb = embed_local(
        model_name=args.local_model,
        texts=texts,
        device=args.device,
        batch_size=args.local_batch_size,
        domain=args.local_domain,
    )

    report = compare_embeddings(api_emb, local_emb, args.api_model, args.local_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
