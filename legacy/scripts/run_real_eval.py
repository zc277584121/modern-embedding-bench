"""Run real evaluations across all available providers and tasks.

Usage:
    uv run python scripts/run_real_eval.py
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_runner")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_eval(provider_name: str, task_name: str, provider_kwargs: dict | None = None, **task_kwargs) -> dict:
    """Run a single provider+task evaluation and return serializable result."""
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    logger.info("=" * 60)
    logger.info("Running: %s / %s (kwargs=%s)", provider_name, task_name, task_kwargs)
    logger.info("=" * 60)

    try:
        provider = get_provider(provider_name, **(provider_kwargs or {}))
        task = get_task(task_name, **task_kwargs)
    except Exception as e:
        logger.error("Failed to instantiate: %s", e)
        return {
            "provider": provider_name,
            "task": task_name,
            "error": f"init failed: {e}",
            "metrics": {},
        }

    start = time.time()
    try:
        result = task.run(provider)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Task failed after %.1fs: %s", elapsed, e)
        return {
            "provider": provider_name,
            "task": task_name,
            "error": str(e),
            "metrics": {},
            "elapsed_s": round(elapsed, 1),
        }

    elapsed = time.time() - start
    logger.info("Completed in %.1fs", elapsed)

    if result.error:
        logger.warning("Task returned error: %s", result.error)

    # Log key metrics
    for k, v in sorted(result.metrics.items()):
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

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
    """Print a summary table."""
    print("\n" + "=" * 80)
    print("  EVALUATION SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Provider':<15} {'Task':<25} {'Status':<10} {'Time':>8}  Key Metrics")
    print("-" * 80)

    for r in results:
        status = "ERROR" if r.get("error") else "OK"
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"

        # Pick key metrics to show
        metrics = r.get("metrics", {})
        key_parts = []
        if "overall_accuracy" in metrics:
            key_parts.append(f"acc={metrics['overall_accuracy']:.3f}")
        if "avg_recall@1" in metrics:
            key_parts.append(f"R@1={metrics['avg_recall@1']:.3f}")
        if "t2i_recall@1" in metrics:
            key_parts.append(f"t2i={metrics['t2i_recall@1']:.3f}")
        if "i2t_recall@1" in metrics:
            key_parts.append(f"i2t={metrics['i2t_recall@1']:.3f}")
        if "modality_gap" in metrics:
            key_parts.append(f"gap={metrics['modality_gap']:.3f}")
        for k, v in sorted(metrics.items()):
            if k.startswith("auc_dim_"):
                dim = k.replace("auc_dim_", "")
                key_parts.append(f"auc@{dim}={v:.3f}")
        if "min_viable_dim" in metrics:
            key_parts.append(f"min_dim={int(metrics['min_viable_dim'])}")
        if "degradation_rate" in metrics:
            key_parts.append(f"degrad={metrics['degradation_rate']:.3f}")

        key_str = ", ".join(key_parts[:5]) if key_parts else (r.get("error", "")[:40] if r.get("error") else "no metrics")

        print(f"{r['provider']:<15} {r['task']:<25} {status:<10} {elapsed:>8}  {key_str}")

    print("=" * 80)


def main() -> None:
    all_results: list[dict] = []

    # =========================================================================
    # DashScope (Qwen3-VL-Embedding) — supports all 3 tasks
    # =========================================================================
    logger.info(">>> DASHSCOPE EVALUATIONS <<<")

    # MRL Stress — text-embedding-v3 (text-only, supports MRL)
    r = run_eval("dashscope", "mrl_stress", use_mock=False, max_samples=200)
    all_results.append(r)

    # Cross-Modal Retrieval — multimodal-embedding-v1 (supports image+text)
    r = run_eval(
        "dashscope", "cross_modal_retrieval",
        provider_kwargs={"model": "multimodal-embedding-v1"},
        use_mock=False, max_samples=50,
    )
    all_results.append(r)

    # Needle-in-Haystack — text-embedding-v3, limit to shorter lengths
    r = run_eval(
        "dashscope", "needle_in_haystack",
        use_mock=False,
        haystack_lengths=[1000, 4000, 8000],
        needle_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    all_results.append(r)

    # =========================================================================
    # OpenAI (text-embedding-3-large) — text-only tasks
    # =========================================================================
    logger.info(">>> OPENAI EVALUATIONS <<<")

    # MRL Stress
    r = run_eval("openai", "mrl_stress", use_mock=False, max_samples=200)
    all_results.append(r)

    # Needle-in-Haystack
    r = run_eval(
        "openai", "needle_in_haystack",
        use_mock=False,
        haystack_lengths=[1000, 4000, 8000],
        needle_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    all_results.append(r)

    # =========================================================================
    # Gemini — try with small samples (rate limited)
    # =========================================================================
    logger.info(">>> GEMINI EVALUATIONS (small samples) <<<")

    r = run_eval("gemini", "mrl_stress", use_mock=False, max_samples=30)
    all_results.append(r)

    # If MRL succeeded, try cross-modal with very small sample
    if not r.get("error"):
        r = run_eval("gemini", "cross_modal_retrieval", use_mock=False, max_samples=10)
        all_results.append(r)
    else:
        logger.warning("Skipping Gemini cross-modal due to MRL failure")

    # =========================================================================
    # Save results
    # =========================================================================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"eval_run_{date_str}.json"

    # Convert any non-serializable values
    def _serialize(obj):
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):  # numpy array
            return obj.tolist()
        return str(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=_serialize)

    logger.info("Results saved to %s", out_path)

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
