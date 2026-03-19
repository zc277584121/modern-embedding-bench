"""Run remaining evaluations for Jina CLIP v2 and DashScope multimodal-embedding-v1.

Tasks:
1. Jina CLIP v2: crosslingual, needle (2K/4K/8K)
   - MRL: skipped (supports_mrl=False for CLIP models)
2. DashScope multimodal-embedding-v1: test text support, then crosslingual, needle, MRL

Results saved incrementally to results/eval_supplementary_20260314.json.

Usage:
    uv run python scripts/run_eval_supplementary.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_supplementary")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "eval_supplementary_20260314.json"


def _serialize(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def save_results(results: list[dict]) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_serialize)
    logger.info("Saved %d entries to %s", len(results), RESULTS_FILE)


def run_task(provider, task_name: str, task_kwargs: dict, label: str, all_results: list[dict]) -> None:
    """Run a single task and append result."""
    from mm_embed.tasks import get_task

    logger.info("")
    logger.info("=" * 60)
    logger.info("%s", label)
    logger.info("=" * 60)

    start = time.time()
    try:
        task = get_task(task_name, **task_kwargs)
        result = task.run(provider)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("%s FAILED after %.1fs: %s", label, elapsed, e)
        all_results.append({
            "provider": provider.name,
            "model": getattr(provider, "model", provider.name),
            "task": task_name,
            "error": str(e),
            "metrics": {},
            "elapsed_s": round(elapsed, 1),
        })
        save_results(all_results)
        return

    elapsed = time.time() - start
    m = result.metrics
    parts = []
    for key in ("avg_recall@1", "hard_avg_recall@1", "overall_accuracy",
                 "zh2en_recall@1", "en2zh_recall@1", "language_gap",
                 "t2i_recall@1", "i2t_hard_recall@1", "modality_gap",
                 "degradation_rate"):
        if key in m:
            parts.append(f"{key}={m[key]:.3f}")
    status = ", ".join(parts) if parts else ("ok" if not result.error else f"ERROR: {result.error}")
    logger.info("%s done in %.1fs (%s)", label, elapsed, status)

    entry = json.loads(json.dumps({
        "provider": provider.name,
        "model": result.model_name,
        "task": task_name,
        "metrics": result.metrics,
        "details": result.details or {},
        "error": result.error,
        "elapsed_s": round(elapsed, 1),
    }, default=_serialize))
    all_results.append(entry)
    save_results(all_results)


def main() -> None:
    from mm_embed.providers import get_provider

    all_results: list[dict] = []

    # ============================================================
    # Part 1: Jina CLIP v2
    # ============================================================
    logger.info("=" * 60)
    logger.info("PART 1: Jina CLIP v2")
    logger.info("=" * 60)

    try:
        jina = get_provider("jina", model="jina-clip-v2")
        test = jina.embed_text(["connectivity test"])
        logger.info("[OK] Jina CLIP v2 connected, dim=%d, supports_mrl=%s",
                     test.dimensions, jina.supports_mrl)
    except Exception as e:
        logger.error("[FAIL] Jina CLIP v2: %s", e)
        jina = None

    if jina:
        # 1a. Crosslingual retrieval
        run_task(jina, "crosslingual_retrieval", {},
                 "Jina CLIP v2 — Crosslingual Retrieval", all_results)
        time.sleep(2)

        # 1b. MRL — skip if not supported
        if jina.supports_mrl:
            run_task(jina, "mrl_stress", {},
                     "Jina CLIP v2 — MRL Stress Test", all_results)
            time.sleep(2)
        else:
            logger.info("Jina CLIP v2: MRL not supported (supports_mrl=False), skipping.")
            all_results.append({
                "provider": "jina",
                "model": "jina-clip-v2",
                "task": "mrl_stress",
                "error": "MRL not supported by jina-clip-v2 (CLIP models do not support MRL)",
                "metrics": {},
                "elapsed_s": 0,
            })
            save_results(all_results)

        # 1c. Needle-in-haystack with reduced lengths (8K context limit)
        run_task(jina, "needle_in_haystack",
                 {"haystack_lengths": [2000, 4000, 8000]},
                 "Jina CLIP v2 — Needle-in-Haystack (2K/4K/8K)", all_results)

    # ============================================================
    # Part 2: DashScope multimodal-embedding-v1
    # ============================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PART 2: DashScope multimodal-embedding-v1")
    logger.info("=" * 60)

    try:
        ds = get_provider("dashscope", model="multimodal-embedding-v1")
        test = ds.embed_text(["connectivity test for text embedding"])
        logger.info("[OK] DashScope multimodal-embedding-v1 text works! dim=%d", test.dimensions)
    except Exception as e:
        logger.error("[FAIL] DashScope multimodal-embedding-v1 text: %s", e)
        all_results.append({
            "provider": "dashscope",
            "model": "multimodal-embedding-v1",
            "task": "text_support_check",
            "error": f"Text embedding not supported: {e}",
            "metrics": {},
            "elapsed_s": 0,
        })
        save_results(all_results)
        ds = None

    if ds:
        # 2a. Crosslingual retrieval
        run_task(ds, "crosslingual_retrieval", {},
                 "DashScope multimodal-embedding-v1 — Crosslingual Retrieval", all_results)
        time.sleep(2)

        # 2b. Needle-in-haystack (standard lengths)
        run_task(ds, "needle_in_haystack", {},
                 "DashScope multimodal-embedding-v1 — Needle-in-Haystack", all_results)
        time.sleep(2)

        # 2c. MRL stress test
        # Note: multimodal-embedding-v1 does NOT support dimension param,
        # but MRL task truncates locally so it may still work
        run_task(ds, "mrl_stress", {},
                 "DashScope multimodal-embedding-v1 — MRL Stress Test", all_results)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 100)
    print("  EVALUATION SUMMARY")
    print("=" * 100)
    print(f"{'Provider':<12} {'Model':<30} {'Task':<28} {'Status':<7} {'Time':>6}  Key Metrics")
    print("-" * 100)

    for r in all_results:
        prov = r.get("provider", "?")
        model = r.get("model", "?")
        task = r.get("task", "?")
        err = r.get("error")
        status = "ERROR" if err else "OK"
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"
        m = r.get("metrics", {})
        parts = []
        for key in ("avg_recall@1", "hard_avg_recall@1", "overall_accuracy",
                     "zh2en_recall@1", "en2zh_recall@1", "degradation_rate"):
            if key in m:
                parts.append(f"{key}={m[key]:.3f}")
        metrics_str = ", ".join(parts[:4]) if parts else (err[:60] if err else "—")
        print(f"{prov:<12} {model:<30} {task:<28} {status:<7} {elapsed:>6}  {metrics_str}")

    print("=" * 100)
    ok = sum(1 for r in all_results if not r.get("error"))
    err = sum(1 for r in all_results if r.get("error"))
    print(f"Total: {len(all_results)} entries ({ok} OK, {err} errors/skipped)")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
