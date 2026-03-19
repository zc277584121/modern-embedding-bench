"""Run ALL models on ALL compatible tasks with reduced default samples.

This script uses the new default sample sizes (Part 2) and embedding cache (Part 1)
to efficiently evaluate all 19 models across 3 tasks.

Usage:
    uv run python scripts/run_all_eval.py
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
logger = logging.getLogger("all_eval")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"eval_all_hard_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-2B")
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL_PATH", "google/siglip2-so400m-patch14-384")

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
# Each entry: (provider, provider_kwargs, tasks_to_run, label)
#   tasks_to_run: subset of {"mrl_stress", "cross_modal_retrieval", "needle_in_haystack"}

MODELS: list[dict[str, Any]] = [
    # ── Local GPU models (fast, run first) ────────────────────────
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "BAAI/bge-m3", "device": DEVICE},
        "tasks": ["mrl_stress", "needle_in_haystack"],
        "label": "BGE-M3 (GPU FP)",
    },
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "jinaai/jina-embeddings-v3", "device": DEVICE},
        "tasks": ["mrl_stress", "needle_in_haystack"],
        "label": "Jina v3 (Local)",
    },
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "clip-ViT-B-32", "device": DEVICE},
        "tasks": ["mrl_stress", "cross_modal_retrieval"],
        "label": "CLIP ViT-B-32",
    },
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "clip-ViT-L-14", "device": DEVICE},
        "tasks": ["mrl_stress", "cross_modal_retrieval"],
        "label": "CLIP ViT-L-14",
    },
    {
        "provider": "transformers",
        "kwargs": {"model": QWEN_VL_MODEL, "device": DEVICE},
        "tasks": ["mrl_stress", "cross_modal_retrieval", "needle_in_haystack"],
        "label": "Qwen3-VL-2B",
    },
    {
        "provider": "transformers",
        "kwargs": {"model": SIGLIP_MODEL, "device": DEVICE},
        "tasks": ["cross_modal_retrieval"],
        "label": "SigLIP2-400M",
    },
    # ── Ollama models (local CPU, medium speed) ───────────────────
    {
        "provider": "ollama",
        "kwargs": {"model": "mxbai-embed-large"},
        "tasks": ["mrl_stress", "needle_in_haystack"],
        "label": "MxbAI Large (Ollama)",
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "nomic-embed-text"},
        "tasks": ["mrl_stress", "needle_in_haystack"],
        "label": "Nomic Embed (Ollama)",
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "bge-m3"},
        "tasks": ["mrl_stress", "needle_in_haystack"],
        "label": "BGE-M3 (Ollama)",
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "snowflake-arctic-embed:335m"},
        "tasks": ["mrl_stress", "needle_in_haystack"],
        "label": "Snowflake 335M (Ollama)",
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "dengcao/Qwen3-Embedding-8B:Q5_K_M"},
        "tasks": ["mrl_stress", "cross_modal_retrieval", "needle_in_haystack"],
        "label": "Qwen3-8B Q5 (Ollama)",
    },
    # ── API models (slower, rate-limited) ─────────────────────────
    {
        "provider": "dashscope",
        "kwargs": {"model": "text-embedding-v3"},
        "tasks": ["mrl_stress", "needle_in_haystack"],
        "label": "Alibaba text-emb-v3",
    },
    {
        "provider": "dashscope",
        "kwargs": {"model": "multimodal-embedding-v1"},
        "tasks": ["cross_modal_retrieval"],
        "label": "Alibaba MM-v1",
    },
    {
        "provider": "openai",
        "kwargs": {"model": "text-embedding-3-large"},
        "tasks": ["mrl_stress", "needle_in_haystack"],
        "label": "OpenAI 3-large",
    },
    {
        "provider": "jina",
        "kwargs": {"model": "jina-embeddings-v4"},
        "tasks": ["mrl_stress", "needle_in_haystack"],
        "label": "Jina v4 (API)",
    },
    {
        "provider": "voyage",
        "kwargs": {"model": "voyage-multimodal-3.5"},
        "tasks": ["mrl_stress", "cross_modal_retrieval", "needle_in_haystack"],
        "label": "Voyage MM-3.5",
    },
    {
        "provider": "cohere",
        "kwargs": {"model": "embed-v4.0"},
        "tasks": ["needle_in_haystack"],  # no MRL support
        "label": "Cohere v4",
    },
    {
        "provider": "ark",
        "kwargs": {"model": "doubao-embedding-text-240715"},
        "tasks": ["needle_in_haystack"],  # no MRL support
        "label": "Doubao (ARK)",
    },
    {
        "provider": "gemini",
        "kwargs": {"model": "gemini-embedding-2-preview"},
        "tasks": ["mrl_stress", "cross_modal_retrieval", "needle_in_haystack"],
        "label": "Gemini Embedding",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def free_gpu():
    """Free GPU memory between models."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_provider(provider_name: str, **kwargs) -> bool:
    """Quick connectivity test."""
    from mm_embed.providers import get_provider

    try:
        p = get_provider(provider_name, **kwargs)
        result = p.embed_text(["test"])
        logger.info("[OK] %s/%s dim=%d", provider_name, kwargs.get("model", ""), result.dimensions)
        return True
    except Exception as e:
        logger.warning("[FAIL] %s/%s: %s", provider_name, kwargs.get("model", ""), e)
        return False


def run_single(provider_name: str, provider_kwargs: dict, task_name: str,
               provider_instance=None) -> dict:
    """Run a single model+task evaluation. Reuses provider_instance if given."""
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    model_label = provider_kwargs.get("model", provider_name)

    try:
        provider = provider_instance or get_provider(provider_name, **provider_kwargs)
        task = get_task(task_name)
    except Exception as e:
        logger.error("Init failed for %s/%s: %s", provider_name, task_name, e)
        return {
            "provider": provider_name,
            "model": model_label,
            "task": task_name,
            "error": f"init: {e}",
            "metrics": {},
        }

    start = time.time()
    try:
        result = task.run(provider)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("%s / %s failed after %.1fs: %s", model_label, task_name, elapsed, e)
        return {
            "provider": provider_name,
            "model": model_label,
            "task": task_name,
            "error": str(e),
            "metrics": {},
            "elapsed_s": round(elapsed, 1),
        }

    elapsed = time.time() - start

    # Summarize key metrics for logging
    m = result.metrics
    parts = []
    if "spearman_dim_1024" in m or "spearman_dim_2048" in m:
        for k, v in m.items():
            if k.startswith("spearman_dim_"):
                parts.append(f"{k.split('_')[-1]}d={v:.3f}")
                break
    if "overall_accuracy" in m:
        parts.append(f"acc={m['overall_accuracy']:.3f}")
    if "avg_recall@1" in m:
        parts.append(f"R@1={m['avg_recall@1']:.3f}")

    summary = ", ".join(parts) if parts else ("ERROR: " + result.error if result.error else "ok")
    logger.info("%s / %s done in %.1fs (%s)", model_label, task_name, elapsed, summary)

    def _serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    return json.loads(json.dumps({
        "provider": provider_name,
        "model": result.model_name,
        "task": task_name,
        "metrics": result.metrics,
        "details": result.details or {},
        "error": result.error,
        "elapsed_s": round(elapsed, 1),
    }, default=_serialize))


def save_results(all_results: list[dict]) -> None:
    """Save results incrementally."""
    def _ser(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=_ser)


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 110)
    print("  FULL EVALUATION SUMMARY")
    print("=" * 110)
    print(f"{'Model':<40} {'Task':<25} {'Status':<8} {'Time':>6}  Key Metrics")
    print("-" * 110)

    for r in results:
        model = r.get("model", r["provider"])
        if len(model) > 38:
            model = "..." + model[-35:]
        task = r.get("task", "?")
        status = "ERROR" if r.get("error") else "OK"
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"

        m = r.get("metrics", {})
        parts = []
        for k, v in m.items():
            if k.startswith("spearman_dim_"):
                parts.append(f"ρ={v:.3f}")
                break
        if "overall_accuracy" in m:
            parts.append(f"acc={m['overall_accuracy']:.3f}")
        if "avg_recall@1" in m:
            parts.append(f"R@1={m['avg_recall@1']:.3f}")
        if "modality_gap" in m:
            parts.append(f"gap={m['modality_gap']:.2f}")

        key_str = ", ".join(parts[:4]) if parts else (r.get("error", "")[:50] if r.get("error") else "—")
        print(f"{model:<40} {task:<25} {status:<8} {elapsed:>6}  {key_str}")

    print("=" * 110)
    ok = sum(1 for r in results if not r.get("error"))
    err = sum(1 for r in results if r.get("error"))
    print(f"Total: {len(results)} runs ({ok} OK, {err} errors)")
    print(f"Results: {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_results: list[dict] = []
    total = sum(len(m["tasks"]) for m in MODELS)
    done = 0

    logger.info("Starting full evaluation: %d models, %d total runs", len(MODELS), total)
    logger.info("Results will be saved to: %s", RESULTS_FILE)
    logger.info("HARD MODE: MRL=150 (mid-range bias), CrossModal=50, Needle=4x5x10=200 (context-filtered)")

    prev_provider = None

    for model_def in MODELS:
        prov = model_def["provider"]
        kwargs = model_def["kwargs"]
        label = model_def["label"]
        tasks = model_def["tasks"]

        # Free GPU when switching away from GPU models
        if prev_provider in ("sentence_transformers", "transformers") and prov != prev_provider:
            free_gpu()
        prev_provider = prov

        logger.info("=" * 60)
        logger.info(">>> %s (%s/%s) <<<", label, prov, kwargs.get("model", ""))
        logger.info("=" * 60)

        # Create provider instance once and reuse for all tasks
        from mm_embed.providers import get_provider
        try:
            provider_inst = get_provider(prov, **kwargs)
            # Quick connectivity test
            test_result = provider_inst.embed_text(["test"])
            logger.info("[OK] %s/%s dim=%d", prov, kwargs.get("model", ""), test_result.dimensions)
        except Exception as e:
            logger.warning("[FAIL] %s/%s: %s", prov, kwargs.get("model", ""), e)
            for task in tasks:
                done += 1
                logger.warning("Skipping %s / %s (connectivity failed)", label, task)
                all_results.append({
                    "provider": prov,
                    "model": kwargs.get("model", prov),
                    "task": task,
                    "error": f"connectivity: {e}",
                    "metrics": {},
                })
            save_results(all_results)
            continue

        for task in tasks:
            done += 1
            logger.info("[%d/%d] Running %s / %s...", done, total, label, task)

            result = run_single(prov, kwargs, task, provider_instance=provider_inst)
            all_results.append(result)
            save_results(all_results)

    # Final
    save_results(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()
