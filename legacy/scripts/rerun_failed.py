"""Re-run only the failed tasks and append results to an existing results file."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rerun_failed")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "results" / "eval_all_20260312_0956.json"
DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-2B")

# Tasks to re-run: (provider, kwargs, task_name, label)
RERUN_TASKS = [
    # Fix 1: Gemini model name was wrong (404) — now corrected to gemini-embedding-2-preview
    ("gemini", {"model": "gemini-embedding-2-preview"}, "mrl_stress", "Gemini Embedding"),
    ("gemini", {"model": "gemini-embedding-2-preview"}, "cross_modal_retrieval", "Gemini Embedding"),
    ("gemini", {"model": "gemini-embedding-2-preview"}, "needle_in_haystack", "Gemini Embedding"),
    # Fix 2: Qwen3-VL-2B needle tensor mismatch — fixed empty image list handling
    ("transformers", {"model": QWEN_VL_MODEL, "device": DEVICE},
     "needle_in_haystack", "Qwen3-VL-2B"),
]


def main() -> None:
    # Import here so module-level errors are caught
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_all_eval import run_single, save_results, free_gpu

    # Load existing results
    with open(RESULTS_FILE, encoding="utf-8") as f:
        all_results: list[dict] = json.load(f)
    logger.info("Loaded %d existing results from %s", len(all_results), RESULTS_FILE)

    # Remove the old failed entries that we're re-running
    rerun_keys = {(t[0], t[1].get("model", t[0]), t[2]) for t in RERUN_TASKS}
    kept = []
    removed = 0
    for r in all_results:
        key = (r["provider"], r["model"], r["task"])
        if key in rerun_keys:
            removed += 1
            logger.info("Removing old result: %s / %s / %s (error=%s)",
                        r["provider"], r["model"], r["task"], r.get("error", "none"))
        else:
            kept.append(r)
    all_results = kept
    logger.info("Removed %d old failed entries, %d remaining", removed, len(all_results))

    # Run each failed task
    from mm_embed.providers import get_provider

    prev_provider = None
    for prov_name, kwargs, task_name, label in RERUN_TASKS:
        # Free GPU when switching away from GPU models
        if prev_provider in ("sentence_transformers", "transformers") and prov_name != prev_provider:
            free_gpu()
        prev_provider = prov_name

        logger.info("=" * 60)
        logger.info(">>> Re-running: %s / %s <<<", label, task_name)
        logger.info("=" * 60)

        try:
            provider_inst = get_provider(prov_name, **kwargs)
            test_result = provider_inst.embed_text(["test"])
            logger.info("[OK] %s dim=%d", label, test_result.dimensions)
        except Exception as e:
            logger.error("[FAIL] %s: %s", label, e)
            all_results.append({
                "provider": prov_name,
                "model": kwargs.get("model", prov_name),
                "task": task_name,
                "error": f"connectivity: {e}",
                "metrics": {},
            })
            continue

        result = run_single(prov_name, kwargs, task_name, provider_instance=provider_inst)
        all_results.append(result)

        # Save incrementally
        def _ser(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return str(obj)

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=_ser)
        logger.info("Saved (now %d results)", len(all_results))

    # Final summary of re-run results
    logger.info("=" * 60)
    logger.info("RE-RUN COMPLETE")
    logger.info("=" * 60)
    for prov_name, kwargs, task_name, label in RERUN_TASKS:
        model = kwargs.get("model", prov_name)
        matching = [r for r in all_results if r["provider"] == prov_name and r["model"] == model and r["task"] == task_name]
        if matching:
            r = matching[-1]
            status = "ERROR: " + r["error"] if r.get("error") else "OK"
            logger.info("  %s / %s: %s", label, task_name, status)


if __name__ == "__main__":
    main()
