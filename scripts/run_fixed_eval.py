"""Run evaluations with fixed metrics on all providers.

Fixes applied:
- MRL: Spearman correlation with continuous STS-B scores (all 1379 pairs)
- Cross-Modal: Full 200-image pool + hard negatives
- Needle: unchanged (methodology is sound, just models differ)

Usage:
    uv run python scripts/run_fixed_eval.py
"""

from __future__ import annotations

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
logger = logging.getLogger("fixed_eval")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_eval(provider_name: str, task_name: str, provider_kwargs: dict | None = None, **task_kwargs) -> dict:
    """Run a single evaluation."""
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    pk = provider_kwargs or {}
    model_display = pk.get("model", "default")
    logger.info("=" * 60)
    logger.info("Running: %s(%s) / %s", provider_name, model_display, task_name)
    logger.info("=" * 60)

    try:
        provider = get_provider(provider_name, **pk)
        task = get_task(task_name, **task_kwargs)
    except Exception as e:
        logger.error("Init failed: %s", e)
        return {"provider": provider_name, "model": model_display, "task": task_name, "error": f"init: {e}", "metrics": {}}

    start = time.time()
    try:
        result = task.run(provider)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Failed after %.1fs: %s", elapsed, e)
        return {"provider": provider_name, "model": model_display, "task": task_name, "error": str(e), "metrics": {}, "elapsed_s": round(elapsed, 1)}

    elapsed = time.time() - start
    logger.info("Completed in %.1fs", elapsed)

    if result.error:
        logger.warning("Error: %s", result.error)

    for k, v in sorted(result.metrics.items()):
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)

    return {
        "provider": provider_name,
        "model": result.model_name,
        "task": task_name,
        "metrics": result.metrics,
        "details": result.details if result.details else {},
        "error": result.error,
        "elapsed_s": round(elapsed, 1),
    }


def print_summary(results: list[dict]) -> None:
    """Print summary table."""
    print("\n" + "=" * 100)
    print("  EVALUATION SUMMARY (FIXED METRICS)")
    print("=" * 100)
    print(f"{'Provider':<15} {'Model':<35} {'Task':<22} {'Status':<6} {'Time':>5}  Key Metrics")
    print("-" * 100)

    for r in results:
        status = "ERR" if r.get("error") else "OK"
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"
        metrics = r.get("metrics", {})

        parts = []
        # MRL metrics (Spearman)
        for k, v in sorted(metrics.items()):
            if k.startswith("spearman_dim_"):
                dim = k.replace("spearman_dim_", "")
                parts.append(f"r@{dim}={v:.3f}")
        if "min_viable_dim" in metrics:
            parts.append(f"min={int(metrics['min_viable_dim'])}")
        # Cross-modal
        if "avg_recall@1" in metrics:
            parts.append(f"R@1={metrics['avg_recall@1']:.3f}")
        if "i2t_hard_recall@1" in metrics:
            parts.append(f"hard_R@1={metrics['i2t_hard_recall@1']:.3f}")
        if "modality_gap" in metrics:
            parts.append(f"gap={metrics['modality_gap']:.2f}")
        # Needle
        if "overall_accuracy" in metrics:
            parts.append(f"acc={metrics['overall_accuracy']:.3f}")
        if "degradation_rate" in metrics:
            parts.append(f"deg={metrics['degradation_rate']:.3f}")

        key_str = ", ".join(parts[:6]) if parts else (r.get("error", "")[:45] if r.get("error") else "-")

        model_short = r["model"][:33]
        print(f"{r['provider']:<15} {model_short:<35} {r['task']:<22} {status:<6} {elapsed:>5}  {key_str}")

    print("=" * 100)


def main() -> None:
    all_results: list[dict] = []

    needle_kwargs = dict(
        use_mock=False,
        haystack_lengths=[1000, 4000, 8000],
        needle_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    # =========================================================================
    # 1. DashScope (API)
    # =========================================================================
    logger.info(">>> DASHSCOPE <<<")

    r = run_eval("dashscope", "mrl_stress", use_mock=False)
    all_results.append(r)

    r = run_eval(
        "dashscope", "cross_modal_retrieval",
        provider_kwargs={"model": "multimodal-embedding-v1"},
        use_mock=False,
    )
    all_results.append(r)

    r = run_eval("dashscope", "needle_in_haystack", **needle_kwargs)
    all_results.append(r)

    # =========================================================================
    # 2. OpenAI (API)
    # =========================================================================
    logger.info(">>> OPENAI <<<")

    r = run_eval("openai", "mrl_stress", use_mock=False)
    all_results.append(r)

    r = run_eval("openai", "needle_in_haystack", **needle_kwargs)
    all_results.append(r)

    # =========================================================================
    # 3. Ollama local models
    # =========================================================================
    ollama_text_models = [
        "nomic-embed-text",
        "mxbai-embed-large",
        "bge-m3",
        "snowflake-arctic-embed:335m",
    ]

    for model in ollama_text_models:
        logger.info(">>> OLLAMA: %s <<<", model)

        r = run_eval("ollama", "mrl_stress", provider_kwargs={"model": model}, use_mock=False)
        all_results.append(r)

        r = run_eval("ollama", "needle_in_haystack", provider_kwargs={"model": model}, **needle_kwargs)
        all_results.append(r)

    # Qwen3 multimodal via Ollama
    logger.info(">>> OLLAMA: Qwen3-Embedding-8B <<<")
    qwen3_model = "dengcao/Qwen3-Embedding-8B:Q5_K_M"

    r = run_eval("ollama", "mrl_stress", provider_kwargs={"model": qwen3_model}, use_mock=False)
    all_results.append(r)

    r = run_eval("ollama", "needle_in_haystack", provider_kwargs={"model": qwen3_model}, **needle_kwargs)
    all_results.append(r)

    r = run_eval("ollama", "cross_modal_retrieval", provider_kwargs={"model": qwen3_model}, use_mock=False)
    all_results.append(r)

    # =========================================================================
    # 4. SentenceTransformers (full precision GPU)
    # =========================================================================
    st_models = [
        ("BAAI/bge-m3", os.environ.get("CUDA_DEVICE", "cuda:0")),
    ]

    for model, device in st_models:
        logger.info(">>> SENTENCE_TRANSFORMERS: %s on %s <<<", model, device)

        r = run_eval("sentence_transformers", "mrl_stress",
                      provider_kwargs={"model": model, "device": device}, use_mock=False)
        all_results.append(r)

        r = run_eval("sentence_transformers", "needle_in_haystack",
                      provider_kwargs={"model": model, "device": device}, **needle_kwargs)
        all_results.append(r)

    # =========================================================================
    # Save results
    # =========================================================================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"eval_fixed_{date_str}.json"

    def _serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=_serialize)

    logger.info("Results saved to %s", out_path)
    print_summary(all_results)


if __name__ == "__main__":
    main()
