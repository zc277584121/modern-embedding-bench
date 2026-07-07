"""Run evaluations for local HuggingFace models.

Models:
  1. Jina v3 (sentence-transformers)
  2. Qwen3-VL-Embedding-2B (transformers, multimodal)
  3. SigLIP 2 (transformers, cross-modal only)
  4. NV-Embed-v2 (sentence-transformers, 7B — may OOM)

Usage:
    uv run python scripts/run_local_eval.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("local_eval")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "eval_local_new_20260312.json"

DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-2B")
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL_PATH", "google/siglip2-so400m-patch14-384")


def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def test_connectivity(provider_name: str, **provider_kwargs) -> bool:
    """Test a provider with a simple embed call."""
    from mm_embed.providers import get_provider

    logger.info("[%s] Testing %s (%s)...", _ts(), provider_name, provider_kwargs.get("model", ""))
    try:
        provider = get_provider(provider_name, **provider_kwargs)
        result = provider.embed_text(["Hello world"])
        dim = result.dimensions
        logger.info("[%s] %s OK! dim=%d, latency=%.0fms", _ts(), provider_kwargs.get("model", provider_name), dim, result.latency_ms)
        return True
    except Exception as e:
        logger.error("[%s] %s FAILED: %s", _ts(), provider_kwargs.get("model", provider_name), e)
        return False


def run_eval(
    provider_name: str,
    task_name: str,
    provider_kwargs: dict | None = None,
    **task_kwargs,
) -> dict:
    """Run a single provider+task evaluation."""
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    model_label = (provider_kwargs or {}).get("model", provider_name)
    logger.info("[%s] Starting %s / %s...", _ts(), model_label, task_name)

    try:
        provider = get_provider(provider_name, **(provider_kwargs or {}))
        task = get_task(task_name, **task_kwargs)
    except Exception as e:
        logger.error("[%s] Init failed: %s", _ts(), e)
        return {
            "provider": provider_name,
            "model": model_label,
            "task": task_name,
            "error": f"init failed: {e}",
            "metrics": {},
        }

    start = time.time()
    try:
        result = task.run(provider)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] %s / %s failed after %.1fs: %s", _ts(), model_label, task_name, elapsed, e)
        return {
            "provider": provider_name,
            "model": model_label,
            "task": task_name,
            "error": str(e),
            "metrics": {},
            "elapsed_s": round(elapsed, 1),
        }

    elapsed = time.time() - start

    # Log key metrics
    key_metrics = []
    m = result.metrics
    if "spearman_full" in m:
        key_metrics.append(f"ρ={m['spearman_full']:.3f}")
    if "overall_accuracy" in m:
        key_metrics.append(f"acc={m['overall_accuracy']:.3f}")
    if "avg_recall@1" in m:
        key_metrics.append(f"R@1={m['avg_recall@1']:.3f}")
    if "t2i_recall@1" in m:
        key_metrics.append(f"t2i={m['t2i_recall@1']:.3f}")
    if "min_viable_dim" in m:
        key_metrics.append(f"min_dim={int(m['min_viable_dim'])}")

    metrics_str = ", ".join(key_metrics) if key_metrics else "done"
    logger.info("[%s] %s / %s done in %.1fs (%s)", _ts(), model_label, task_name, elapsed, metrics_str)

    return {
        "provider": provider_name,
        "model": result.model_name,
        "task": task_name,
        "metrics": result.metrics,
        "details": result.details if result.details else {},
        "error": result.error,
        "elapsed_s": round(elapsed, 1),
    }


def save_results(all_results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=_serialize)

    logger.info("Results saved to %s (%d entries)", RESULTS_FILE, len(all_results))


def free_gpu_memory():
    """Free GPU memory between models."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("GPU memory freed")


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("  LOCAL MODEL EVALUATION SUMMARY")
    print("=" * 100)
    print(f"{'Model':<40} {'Task':<25} {'Status':<8} {'Time':>8}  Key Metrics")
    print("-" * 100)

    for r in results:
        model = r.get("model", r["provider"])
        if len(model) > 38:
            model = "..." + model[-35:]
        status = "ERROR" if r.get("error") else "OK"
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"

        metrics = r.get("metrics", {})
        parts = []
        if "spearman_full" in metrics:
            parts.append(f"ρ={metrics['spearman_full']:.3f}")
        if "overall_accuracy" in metrics:
            parts.append(f"acc={metrics['overall_accuracy']:.3f}")
        if "avg_recall@1" in metrics:
            parts.append(f"R@1={metrics['avg_recall@1']:.3f}")
        if "t2i_recall@1" in metrics:
            parts.append(f"t2i={metrics['t2i_recall@1']:.3f}")
        if "i2t_recall@1" in metrics:
            parts.append(f"i2t={metrics['i2t_recall@1']:.3f}")
        if "modality_gap" in metrics:
            parts.append(f"gap={metrics['modality_gap']:.3f}")
        if "min_viable_dim" in metrics:
            parts.append(f"min_dim={int(metrics['min_viable_dim'])}")
        if "degradation_rate" in metrics:
            parts.append(f"degrad={metrics['degradation_rate']:.3f}")

        key_str = ", ".join(parts[:5]) if parts else (r.get("error", "")[:40] if r.get("error") else "—")
        print(f"{model:<40} {status:<8} {elapsed:>8}  {key_str}")

    print("=" * 100)


def main() -> None:
    all_results: list[dict] = []

    # =========================================================================
    # 1. Jina v3 local (sentence-transformers)
    # =========================================================================
    logger.info("=" * 60)
    logger.info(">>> JINA v3 LOCAL <<<")
    logger.info("=" * 60)

    jina_kwargs = {"model": "jinaai/jina-embeddings-v3", "device": DEVICE}

    if test_connectivity("sentence_transformers", **jina_kwargs):
        # MRL stress test
        r = run_eval("sentence_transformers", "mrl_stress",
                      provider_kwargs=jina_kwargs,
                      use_mock=False, max_samples=200)
        all_results.append(r)
        save_results(all_results)

        # Needle-in-haystack (smaller config to avoid OOM)
        r = run_eval("sentence_transformers", "needle_in_haystack",
                      provider_kwargs=jina_kwargs,
                      use_mock=False,
                      haystack_lengths=[1000, 4000],
                      needle_positions=[0.0, 0.5, 1.0])
        all_results.append(r)
        save_results(all_results)

    free_gpu_memory()

    # =========================================================================
    # 2. Qwen3-VL-Embedding-2B (transformers, multimodal)
    # =========================================================================
    logger.info("=" * 60)
    logger.info(">>> QWEN3-VL-EMBEDDING-2B <<<")
    logger.info("=" * 60)

    qwen_path = QWEN_VL_MODEL
    qwen_kwargs = {"model": qwen_path, "device": DEVICE}

    if test_connectivity("transformers", **qwen_kwargs):
        # MRL stress test
        r = run_eval("transformers", "mrl_stress",
                      provider_kwargs=qwen_kwargs,
                      use_mock=False, max_samples=200)
        all_results.append(r)
        save_results(all_results)

        # Cross-modal retrieval
        r = run_eval("transformers", "cross_modal_retrieval",
                      provider_kwargs=qwen_kwargs,
                      use_mock=False, max_samples=20)
        all_results.append(r)
        save_results(all_results)

        # Needle-in-haystack (shorter lengths to save time)
        r = run_eval("transformers", "needle_in_haystack",
                      provider_kwargs=qwen_kwargs,
                      use_mock=False,
                      haystack_lengths=[1000, 4000],
                      needle_positions=[0.0, 0.5, 1.0])
        all_results.append(r)
        save_results(all_results)

    free_gpu_memory()

    # =========================================================================
    # 3. SigLIP 2 (cross-modal only)
    # =========================================================================
    logger.info("=" * 60)
    logger.info(">>> SIGLIP 2 <<<")
    logger.info("=" * 60)

    siglip_path = SIGLIP_MODEL
    siglip_kwargs = {"model": siglip_path, "device": DEVICE}

    if test_connectivity("transformers", **siglip_kwargs):
        # Cross-modal only
        r = run_eval("transformers", "cross_modal_retrieval",
                      provider_kwargs=siglip_kwargs,
                      use_mock=False, max_samples=50)
        all_results.append(r)
        save_results(all_results)

    free_gpu_memory()

    # =========================================================================
    # 4. NV-Embed-v2 (try sentence-transformers, may OOM at 7B)
    # =========================================================================
    logger.info("=" * 60)
    logger.info(">>> NV-EMBED-V2 (7B — may OOM) <<<")
    logger.info("=" * 60)

    nv_kwargs = {"model": "nvidia/NV-Embed-v2", "device": DEVICE}

    if test_connectivity("sentence_transformers", **nv_kwargs):
        # MRL stress test with small samples
        r = run_eval("sentence_transformers", "mrl_stress",
                      provider_kwargs=nv_kwargs,
                      use_mock=False, max_samples=50)
        all_results.append(r)
        save_results(all_results)

        # Needle with small config
        r = run_eval("sentence_transformers", "needle_in_haystack",
                      provider_kwargs=nv_kwargs,
                      use_mock=False,
                      haystack_lengths=[1000, 4000],
                      needle_positions=[0.0, 0.5, 1.0])
        all_results.append(r)
        save_results(all_results)
    else:
        logger.warning("NV-Embed-v2 failed connectivity test (likely OOM). Skipping.")

    free_gpu_memory()

    # =========================================================================
    # Final summary
    # =========================================================================
    save_results(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()
