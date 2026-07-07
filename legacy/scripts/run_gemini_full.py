"""Run all missing Gemini embedding evaluations.

Priority order (most valuable first, least API calls first):
1. Cross-Lingual Retrieval (~660 items)
2. Cross-Modal Retrieval (~1000 items)
3. Needle-in-Haystack (~500+ items)

Saves results incrementally so partial results survive quota exhaustion.

Usage:
    uv run python scripts/run_gemini_full.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gemini_full")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "eval_gemini_full_20260312.json"

PROVIDER_NAME = "gemini"
MODEL = "gemini-embedding-2-preview"

# Tasks in priority order
TASKS = [
    {
        "name": "crosslingual_retrieval",
        "label": "Cross-Lingual Retrieval (166 pairs + HN)",
        "est_items": "~660",
        "kwargs": {},
    },
    {
        "name": "cross_modal_retrieval",
        "label": "Cross-Modal Retrieval (200 pairs + HN)",
        "est_items": "~1000",
        "kwargs": {"max_samples": None},
    },
    {
        "name": "needle_in_haystack",
        "label": "Needle-in-Haystack (hard, 4K-32K)",
        "est_items": "~500+",
        "kwargs": {},
    },
]


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

    logger.info("=" * 60)
    logger.info("Gemini Full Evaluation: %d tasks", len(TASKS))
    logger.info("Model: %s", MODEL)
    logger.info("Results: %s", RESULTS_FILE)
    logger.info("=" * 60)

    # Create provider once and reuse
    try:
        provider = get_provider(PROVIDER_NAME, model=MODEL)
        test_result = provider.embed_text(["connectivity test"])
        logger.info("[OK] Gemini connected, dim=%d", test_result.dimensions)
    except Exception as e:
        logger.error("[FAIL] Cannot connect to Gemini: %s", e)
        return

    for i, task_def in enumerate(TASKS, 1):
        task_name = task_def["name"]
        label = task_def["label"]
        est = task_def["est_items"]
        kwargs = task_def["kwargs"]

        logger.info("")
        logger.info("=" * 60)
        logger.info("[%d/%d] %s (est. %s API items)", i, len(TASKS), label, est)
        logger.info("=" * 60)

        # Small delay between tasks to avoid burst
        if i > 1:
            logger.info("Sleeping 3s between tasks...")
            time.sleep(3)

        start = time.time()
        try:
            task = get_task(task_name, **kwargs)
            result = task.run(provider)
        except Exception as e:
            elapsed = time.time() - start
            logger.error("%s failed after %.1fs: %s", label, elapsed, e)
            all_results.append({
                "provider": PROVIDER_NAME,
                "model": MODEL,
                "task": task_name,
                "error": str(e),
                "metrics": {},
                "elapsed_s": round(elapsed, 1),
            })
            save_results(all_results)
            logger.info("Saved partial results (%d entries). Continuing to next task...",
                        len(all_results))
            continue

        elapsed = time.time() - start

        # Log key metrics
        m = result.metrics
        parts = []
        if "avg_recall@1" in m:
            parts.append(f"R@1={m['avg_recall@1']:.3f}")
        if "hard_avg_recall@1" in m:
            parts.append(f"hard_R@1={m['hard_avg_recall@1']:.3f}")
        if "overall_accuracy" in m:
            parts.append(f"acc={m['overall_accuracy']:.3f}")

        status = ", ".join(parts) if parts else (f"ERROR: {result.error}" if result.error else "ok")
        logger.info("%s done in %.1fs (%s)", label, elapsed, status)

        entry = json.loads(json.dumps({
            "provider": PROVIDER_NAME,
            "model": result.model_name,
            "task": task_name,
            "metrics": result.metrics,
            "details": result.details or {},
            "error": result.error,
            "elapsed_s": round(elapsed, 1),
        }, default=_serialize))
        all_results.append(entry)
        save_results(all_results)
        logger.info("Saved (%d entries so far)", len(all_results))

    # Final summary
    print_summary(all_results)


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("  GEMINI FULL EVALUATION SUMMARY")
    print("=" * 100)
    print(f"{'Task':<35} {'Status':<8} {'Time':>6}  Key Metrics")
    print("-" * 100)

    for r in results:
        task = r.get("task", "?")
        err = r.get("error")
        status = "ERROR" if err else "OK"
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"
        m = r.get("metrics", {})

        parts = []
        for key in ("avg_recall@1", "hard_avg_recall@1", "overall_accuracy",
                     "zh2en_recall@1", "en2zh_recall@1", "language_gap",
                     "t2i_recall@1", "i2t_hard_recall@1", "modality_gap",
                     "degradation_rate"):
            if key in m:
                parts.append(f"{key}={m[key]:.3f}")

        metrics_str = ", ".join(parts[:6]) if parts else (err[:70] if err else "—")
        print(f"{task:<35} {status:<8} {elapsed:>6}  {metrics_str}")

    print("=" * 100)
    ok = sum(1 for r in results if not r.get("error"))
    err = sum(1 for r in results if r.get("error"))
    print(f"Total: {len(results)} tasks ({ok} OK, {err} errors)")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
