"""Run Gemini Embedding 2 Preview evaluation on all 3 tasks.

Runs MRL Stress -> Needle-in-Haystack -> Cross-Modal Retrieval in order,
saving results incrementally after each task.

Usage:
    uv run python scripts/run_gemini_eval.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gemini_eval")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fixed_eval import run_eval

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_PATH = RESULTS_DIR / "eval_gemini_20260312.json"

# Wall-clock tracking
T0 = time.time()


def elapsed() -> str:
    """Return elapsed time as [MM:SS]."""
    s = int(time.time() - T0)
    return f"[{s // 60:02d}:{s % 60:02d}]"


def save_results(results: list[dict]) -> None:
    """Save results incrementally."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_serialize)
    logger.info("Results saved to %s (%d entries)", OUT_PATH, len(results))


def main() -> None:
    all_results: list[dict] = []
    model = "gemini-embedding-2-preview"
    pk = {"model": model}

    # =========================================================================
    # 1. MRL Stress
    #    - 2552 unique sentences, batch_size=100 -> 26 API calls
    #    - 10s sleep between batches -> ~260s + API time
    # =========================================================================
    logger.info("%s Starting MRL Stress (estimated ~5 min, 26 API calls)", elapsed())
    r = run_eval("gemini", "mrl_stress", provider_kwargs=pk, use_mock=False)
    all_results.append(r)
    save_results(all_results)

    if r.get("error"):
        logger.warning("MRL Stress failed: %s", r["error"])
    else:
        m = r.get("metrics", {})
        best_dim = max(
            ((k, v) for k, v in m.items() if k.startswith("spearman_dim_")),
            key=lambda x: x[1], default=("?", 0),
        )
        logger.info("%s MRL Stress complete. Best: %s=%.4f", elapsed(), best_dim[0], best_dim[1])

    # =========================================================================
    # 2. Needle-in-Haystack
    #    - 10 queries + 150 docs_with + 15 docs_without = ~8 API calls (batched)
    # =========================================================================
    logger.info("%s Starting Needle-in-Haystack (estimated ~3 min, ~8 API calls)", elapsed())
    r = run_eval(
        "gemini", "needle_in_haystack",
        provider_kwargs=pk,
        use_mock=False,
        haystack_lengths=[1000, 4000, 8000],
        needle_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    all_results.append(r)
    save_results(all_results)

    if r.get("error"):
        logger.warning("Needle failed: %s", r["error"])
    else:
        acc = r["metrics"].get("overall_accuracy", 0)
        logger.info("%s Needle complete. Accuracy=%.3f", elapsed(), acc)

    # =========================================================================
    # 3. Cross-Modal Retrieval
    #    - 200 text (2 batches) + 200 images (one by one) + 600 hard neg texts (6 batches)
    #    - ~208 API calls with image sleeps
    # =========================================================================
    logger.info("%s Starting Cross-Modal Retrieval (estimated ~15 min, ~208 API calls)", elapsed())
    r = run_eval(
        "gemini", "cross_modal_retrieval",
        provider_kwargs=pk,
        use_mock=False,
    )
    all_results.append(r)
    save_results(all_results)

    if r.get("error"):
        logger.warning("Cross-Modal failed: %s", r["error"])
    else:
        m = r.get("metrics", {})
        logger.info(
            "%s Cross-Modal complete. avg_R@1=%.3f, hard_R@1=%.3f, gap=%.2f",
            elapsed(),
            m.get("avg_recall@1", 0),
            m.get("i2t_hard_recall@1", 0),
            m.get("modality_gap", 0),
        )

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("%s All done!", elapsed())

    print(f"\n{'=' * 90}")
    print("  GEMINI EVALUATION SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Task':<25} {'Status':<6} {'Time':>6}  Key Metrics")
    print(f"{'-' * 90}")
    for r in all_results:
        status = "ERR" if r.get("error") else "OK"
        t = f"{r.get('elapsed_s', 0):.0f}s"
        m = r.get("metrics", {})
        parts = []
        for k, v in sorted(m.items()):
            if k.startswith("spearman_dim_"):
                dim = k.replace("spearman_dim_", "")
                parts.append(f"r@{dim}={v:.3f}")
        if "overall_accuracy" in m:
            parts.append(f"acc={m['overall_accuracy']:.3f}")
        if "avg_recall@1" in m:
            parts.append(f"R@1={m['avg_recall@1']:.3f}")
        if "i2t_hard_recall@1" in m:
            parts.append(f"hard_R@1={m['i2t_hard_recall@1']:.3f}")
        key_str = ", ".join(parts[:6]) if parts else (r.get("error", "")[:50] if r.get("error") else "-")
        print(f"{r['task']:<25} {status:<6} {t:>6}  {key_str}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
