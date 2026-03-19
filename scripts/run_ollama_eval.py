"""Run evaluations on local Ollama embedding models.

Usage:
    uv run python scripts/run_ollama_eval.py
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
logger = logging.getLogger("ollama_eval")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_eval(model: str, task_name: str, **task_kwargs) -> dict:
    """Run a single ollama model + task evaluation."""
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    logger.info("=" * 60)
    logger.info("Running: ollama/%s / %s", model, task_name)
    logger.info("=" * 60)

    try:
        provider = get_provider("ollama", model=model)
        task = get_task(task_name, **task_kwargs)
    except Exception as e:
        logger.error("Init failed: %s", e)
        return {"provider": "ollama", "model": model, "task": task_name, "error": f"init: {e}", "metrics": {}}

    start = time.time()
    try:
        result = task.run(provider)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Failed after %.1fs: %s", elapsed, e)
        return {"provider": "ollama", "model": model, "task": task_name, "error": str(e), "metrics": {}, "elapsed_s": round(elapsed, 1)}

    elapsed = time.time() - start
    logger.info("Completed in %.1fs", elapsed)

    if result.error:
        logger.warning("Error: %s", result.error)

    for k, v in sorted(result.metrics.items()):
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)

    return {
        "provider": "ollama",
        "model": result.model_name,
        "task": task_name,
        "metrics": result.metrics,
        "details": result.details if result.details else {},
        "error": result.error,
        "elapsed_s": round(elapsed, 1),
    }


def print_summary(results: list[dict]) -> None:
    """Print summary table."""
    print("\n" + "=" * 90)
    print("  OLLAMA EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Model':<40} {'Task':<25} {'Status':<8} {'Time':>6}  Key Metrics")
    print("-" * 90)

    for r in results:
        status = "ERROR" if r.get("error") else "OK"
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"
        metrics = r.get("metrics", {})

        parts = []
        if "overall_accuracy" in metrics:
            parts.append(f"acc={metrics['overall_accuracy']:.3f}")
        if "avg_recall@1" in metrics:
            parts.append(f"R@1={metrics['avg_recall@1']:.3f}")
        if "modality_gap" in metrics:
            parts.append(f"gap={metrics['modality_gap']:.3f}")
        for k, v in sorted(metrics.items()):
            if k.startswith("auc_dim_"):
                dim = k.replace("auc_dim_", "")
                parts.append(f"auc@{dim}={v:.3f}")
        if "min_viable_dim" in metrics:
            parts.append(f"min_dim={int(metrics['min_viable_dim'])}")
        if "degradation_rate" in metrics:
            parts.append(f"degrad={metrics['degradation_rate']:.3f}")

        key_str = ", ".join(parts[:5]) if parts else (r.get("error", "")[:50] if r.get("error") else "no metrics")
        model_short = r["model"][:38]
        print(f"{model_short:<40} {r['task']:<25} {status:<8} {elapsed:>6}  {key_str}")

    print("=" * 90)


def main() -> None:
    all_results: list[dict] = []

    models = [
        ("nomic-embed-text", False),
        ("mxbai-embed-large", False),
        ("dengcao/Qwen3-Embedding-8B:Q5_K_M", True),  # supports images
    ]

    for model, supports_images in models:
        logger.info(">>> MODEL: %s <<<", model)

        # MRL Stress
        r = run_eval(model, "mrl_stress", use_mock=False, max_samples=200)
        all_results.append(r)

        # Needle-in-Haystack
        r = run_eval(
            model, "needle_in_haystack",
            use_mock=False,
            haystack_lengths=[1000, 4000, 8000],
            needle_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        all_results.append(r)

        # Cross-Modal (only for multimodal models)
        if supports_images:
            r = run_eval(model, "cross_modal_retrieval", use_mock=False, max_samples=50)
            all_results.append(r)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"ollama_eval_{date_str}.json"

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
