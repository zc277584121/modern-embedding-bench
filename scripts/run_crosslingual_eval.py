"""Run cross-lingual retrieval (Chinese <-> English) on ALL text-capable models.

Usage:
    uv run python scripts/run_crosslingual_eval.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crosslingual_eval")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"eval_crosslingual_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-2B")

# All text-capable models
MODELS: list[dict[str, Any]] = [
    # Local GPU models
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "BAAI/bge-m3", "device": DEVICE},
        "label": "BGE-M3 (GPU FP)",
    },
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "jinaai/jina-embeddings-v3", "device": DEVICE},
        "label": "Jina v3 (Local)",
    },
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "clip-ViT-B-32", "device": DEVICE},
        "label": "CLIP ViT-B-32",
    },
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "clip-ViT-L-14", "device": DEVICE},
        "label": "CLIP ViT-L-14",
    },
    {
        "provider": "transformers",
        "kwargs": {"model": QWEN_VL_MODEL, "device": DEVICE},
        "label": "Qwen3-VL-2B",
    },
    # Ollama models
    {
        "provider": "ollama",
        "kwargs": {"model": "mxbai-embed-large"},
        "label": "MxbAI Large (Ollama)",
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "nomic-embed-text"},
        "label": "Nomic Embed (Ollama)",
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "bge-m3"},
        "label": "BGE-M3 (Ollama)",
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "snowflake-arctic-embed:335m"},
        "label": "Snowflake 335M (Ollama)",
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "dengcao/Qwen3-Embedding-8B:Q5_K_M"},
        "label": "Qwen3-8B Q5 (Ollama)",
    },
    # API models
    {
        "provider": "dashscope",
        "kwargs": {"model": "text-embedding-v3"},
        "label": "Alibaba text-emb-v3",
    },
    {
        "provider": "openai",
        "kwargs": {"model": "text-embedding-3-large"},
        "label": "OpenAI 3-large",
    },
    {
        "provider": "jina",
        "kwargs": {"model": "jina-embeddings-v4"},
        "label": "Jina v4 (API)",
    },
    {
        "provider": "voyage",
        "kwargs": {"model": "voyage-multimodal-3.5"},
        "label": "Voyage MM-3.5",
    },
    {
        "provider": "cohere",
        "kwargs": {"model": "embed-v4.0"},
        "label": "Cohere v4",
    },
    {
        "provider": "ark",
        "kwargs": {"model": "doubao-embedding-text-240715"},
        "label": "Doubao (ARK)",
    },
    {
        "provider": "gemini",
        "kwargs": {"model": "gemini-embedding-2-preview"},
        "label": "Gemini Embedding",
    },
]


def free_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _serialize(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def save_results(results: list[dict]) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_serialize)


def main() -> None:
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    all_results: list[dict] = []
    total = len(MODELS)

    logger.info("Cross-lingual evaluation: %d models, 166 ZH-EN pairs + hard negatives", total)
    logger.info("Results: %s", RESULTS_FILE)

    prev_provider = None

    for i, model_def in enumerate(MODELS, 1):
        prov = model_def["provider"]
        kwargs = model_def["kwargs"]
        label = model_def["label"]

        # Free GPU when switching away from GPU models
        if prev_provider in ("sentence_transformers", "transformers") and prov != prev_provider:
            free_gpu()
        prev_provider = prov

        logger.info("=" * 60)
        logger.info("[%d/%d] %s (%s/%s)", i, total, label, prov, kwargs.get("model", ""))
        logger.info("=" * 60)

        try:
            provider = get_provider(prov, **kwargs)
            test_result = provider.embed_text(["test"])
            logger.info("[OK] %s dim=%d", label, test_result.dimensions)
        except Exception as e:
            logger.error("[FAIL] %s: %s", label, e)
            all_results.append({
                "provider": prov,
                "model": kwargs.get("model", prov),
                "task": "crosslingual_retrieval",
                "error": f"connectivity: {e}",
                "metrics": {},
            })
            save_results(all_results)
            continue

        task = get_task("crosslingual_retrieval")
        start = time.time()
        try:
            result = task.run(provider)
        except Exception as e:
            elapsed = time.time() - start
            logger.error("%s failed after %.1fs: %s", label, elapsed, e)
            all_results.append({
                "provider": prov,
                "model": kwargs.get("model", prov),
                "task": "crosslingual_retrieval",
                "error": str(e),
                "metrics": {},
                "elapsed_s": round(elapsed, 1),
            })
            save_results(all_results)
            continue

        elapsed = time.time() - start
        m = result.metrics
        parts = []
        if "avg_recall@1" in m:
            parts.append(f"R@1={m['avg_recall@1']:.3f}")
        if "hard_avg_recall@1" in m:
            parts.append(f"hard_R@1={m['hard_avg_recall@1']:.3f}")
        if "language_gap" in m:
            parts.append(f"gap={m['language_gap']:.2f}")

        logger.info("%s done in %.1fs (%s)", label, elapsed,
                     ", ".join(parts) if parts else (result.error or "ok"))

        entry = json.loads(json.dumps({
            "provider": prov,
            "model": result.model_name,
            "task": "crosslingual_retrieval",
            "metrics": result.metrics,
            "details": result.details or {},
            "error": result.error,
            "elapsed_s": round(elapsed, 1),
        }, default=_serialize))
        all_results.append(entry)
        save_results(all_results)

    # Print ranking
    print_ranking(all_results)


def print_ranking(results: list[dict]) -> None:
    print("\n" + "=" * 130)
    print("  CROSS-LINGUAL RETRIEVAL RANKING (Chinese <-> English)")
    print("=" * 130)
    print(f"{'Model':<40} {'avg_R@1':>9} {'hard_R@1':>10} {'zh→en R@1':>10} {'en→zh R@1':>10} "
          f"{'zh→en_hard':>11} {'en→zh_hard':>11} {'gap':>6} {'Time':>6}")
    print("-" * 130)

    # Sort by hard_avg_recall@1 or avg_recall@1
    ok_results = [r for r in results if not r.get("error")]
    err_results = [r for r in results if r.get("error")]

    ok_results.sort(key=lambda r: r.get("metrics", {}).get("hard_avg_recall@1",
                                        r.get("metrics", {}).get("avg_recall@1", 0)),
                    reverse=True)

    for r in ok_results:
        model = r.get("model", r["provider"])
        display = model if len(model) <= 38 else "..." + model[-35:]
        m = r.get("metrics", {})

        avg_r1 = m.get("avg_recall@1", 0)
        hard_avg = m.get("hard_avg_recall@1", 0)
        zh2en = m.get("zh2en_recall@1", 0)
        en2zh = m.get("en2zh_recall@1", 0)
        zh2en_h = m.get("zh2en_hard_recall@1", 0)
        en2zh_h = m.get("en2zh_hard_recall@1", 0)
        gap = m.get("language_gap", 0)
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"

        print(f"{display:<40} {avg_r1:>9.3f} {hard_avg:>10.3f} {zh2en:>10.3f} {en2zh:>10.3f} "
              f"{zh2en_h:>11.3f} {en2zh_h:>11.3f} {gap:>6.2f} {elapsed:>6}")

    for r in err_results:
        model = r.get("model", r["provider"])
        display = model if len(model) <= 38 else "..." + model[-35:]
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"
        print(f"{display:<40} {'ERROR':>9} {'':>10} {'':>10} {'':>10} "
              f"{'':>11} {'':>11} {'':>6} {elapsed:>6}")
        print(f"{'':40} Error: {r.get('error', '')[:80]}")

    print("=" * 130)
    ok = sum(1 for r in results if not r.get("error"))
    err = sum(1 for r in results if r.get("error"))
    print(f"Total: {len(results)} models ({ok} OK, {err} errors)")
    print(f"Results: {RESULTS_FILE}")

    # Per-difficulty breakdown
    print(f"\n{'Model':<40} {'easy_zh→en':>11} {'med_zh→en':>11} {'hard_zh→en':>11} "
          f"{'easy_en→zh':>11} {'med_en→zh':>11} {'hard_en→zh':>11}")
    print("-" * 130)
    for r in ok_results:
        model = r.get("model", r["provider"])
        display = model if len(model) <= 38 else "..." + model[-35:]
        m = r.get("metrics", {})
        vals = []
        for direction in ("zh2en", "en2zh"):
            for diff in ("easy", "medium", "hard"):
                key = f"{direction}_recall@1_{diff}"
                v = m.get(key, 0)
                vals.append(f"{v:>11.3f}")
        print(f"{display:<40} {''.join(vals)}")
    print("=" * 130)


if __name__ == "__main__":
    main()
