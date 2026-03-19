"""Run cross-modal retrieval ONLY for applicable models with hard settings.

Uses all 200 pairs (no subsampling) + hard negative metrics.

Usage:
    uv run python scripts/run_crossmodal_hard.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crossmodal_hard")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"eval_crossmodal_hard_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-2B")
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL_PATH", "google/siglip2-so400m-patch14-384")

# Only models that support cross-modal (text + image)
MODELS = [
    # Local GPU models
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
    {
        "provider": "transformers",
        "kwargs": {"model": SIGLIP_MODEL, "device": DEVICE},
        "label": "SigLIP2-400M",
    },
    # Ollama
    {
        "provider": "ollama",
        "kwargs": {"model": "dengcao/Qwen3-Embedding-8B:Q5_K_M"},
        "label": "Qwen3-8B Q5 (Ollama)",
    },
    # API models
    {
        "provider": "dashscope",
        "kwargs": {"model": "multimodal-embedding-v1"},
        "label": "Alibaba MM-v1",
    },
    {
        "provider": "voyage",
        "kwargs": {"model": "voyage-multimodal-3.5"},
        "label": "Voyage MM-3.5",
    },
    {
        "provider": "gemini",
        "kwargs": {"model": "gemini-embedding-2-preview"},
        "label": "Gemini Embedding",
    },
]

# Previous easy results for comparison (from eval_all_hard with max_samples=50)
PREV_RESULTS = {
    "clip-ViT-B-32": {"avg_recall@1": 1.0, "i2t_hard_recall@1": None},
    "clip-ViT-L-14": {"avg_recall@1": 0.99, "i2t_hard_recall@1": None},
    QWEN_VL_MODEL: {"avg_recall@1": 1.0, "i2t_hard_recall@1": None},
    SIGLIP_MODEL: {"avg_recall@1": 1.0, "i2t_hard_recall@1": None},
    "dengcao/Qwen3-Embedding-8B:Q5_K_M": {"avg_recall@1": 0.02, "i2t_hard_recall@1": None},
    "multimodal-embedding-v1": {"avg_recall@1": 1.0, "i2t_hard_recall@1": None},
    "voyage-multimodal-3.5": {"avg_recall@1": 1.0, "i2t_hard_recall@1": None},
    "gemini-embedding-2-preview": {"avg_recall@1": 1.0, "i2t_hard_recall@1": None},
}


def free_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    all_results: list[dict] = []
    total = len(MODELS)

    logger.info("Cross-modal HARD evaluation: %d models, all 200 pairs + hard negatives", total)
    logger.info("Results: %s", RESULTS_FILE)

    # Load previous results for comparison
    prev_file = ROOT / "results" / "eval_all_hard_20260312_1053.json"
    if prev_file.exists():
        with open(prev_file, encoding="utf-8") as f:
            prev_data = json.load(f)
        for r in prev_data:
            if r.get("task") == "cross_modal_retrieval" and not r.get("error"):
                model = r["model"]
                PREV_RESULTS[model] = {
                    "avg_recall@1": r["metrics"].get("avg_recall@1"),
                    "i2t_hard_recall@1": r["metrics"].get("i2t_hard_recall@1"),
                    "hard_avg_recall@1": r["metrics"].get("hard_avg_recall@1"),
                }
        logger.info("Loaded previous results for comparison")

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
                "task": "cross_modal_retrieval",
                "error": f"connectivity: {e}",
                "metrics": {},
            })
            _save(all_results)
            continue

        task = get_task("cross_modal_retrieval")
        start = time.time()
        try:
            result = task.run(provider)
        except Exception as e:
            elapsed = time.time() - start
            logger.error("%s failed after %.1fs: %s", label, elapsed, e)
            all_results.append({
                "provider": prov,
                "model": kwargs.get("model", prov),
                "task": "cross_modal_retrieval",
                "error": str(e),
                "metrics": {},
                "elapsed_s": round(elapsed, 1),
            })
            _save(all_results)
            continue

        elapsed = time.time() - start
        m = result.metrics
        summary_parts = []
        if "avg_recall@1" in m:
            summary_parts.append(f"R@1={m['avg_recall@1']:.3f}")
        if "hard_avg_recall@1" in m:
            summary_parts.append(f"hard_R@1={m['hard_avg_recall@1']:.3f}")
        if "modality_gap" in m:
            summary_parts.append(f"gap={m['modality_gap']:.2f}")

        logger.info("%s done in %.1fs (%s)", label, elapsed, ", ".join(summary_parts))

        def _serialize(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return str(obj)

        entry = json.loads(json.dumps({
            "provider": prov,
            "model": result.model_name,
            "task": "cross_modal_retrieval",
            "metrics": result.metrics,
            "details": result.details or {},
            "error": result.error,
            "elapsed_s": round(elapsed, 1),
        }, default=_serialize))
        all_results.append(entry)
        _save(all_results)

    # Print comparison
    print_comparison(all_results)


def _save(results: list[dict]) -> None:
    def _ser(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_ser)


def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 120)
    print("  CROSS-MODAL HARD EVALUATION: Previous (50 pairs) vs New (200 pairs + hard negatives)")
    print("=" * 120)
    print(f"{'Model':<35} {'prev R@1':>10} {'new R@1':>10} {'Δ':>8}  {'hard_avg_R@1':>14} {'i2t_hard_R@1':>14} {'gap':>6} {'n':>4} {'Time':>6}")
    print("-" * 120)

    for r in results:
        model = r.get("model", r["provider"])
        display = model if len(model) <= 33 else "..." + model[-30:]
        m = r.get("metrics", {})

        if r.get("error"):
            print(f"{display:<35} {'—':>10} {'ERROR':>10} {'':>8}  {'':>14} {'':>14} {'':>6} {'':>4} {r.get('elapsed_s', 0):>5.0f}s")
            print(f"{'':35} Error: {r['error'][:80]}")
            continue

        new_r1 = m.get("avg_recall@1", 0)
        hard_avg = m.get("hard_avg_recall@1", 0)
        i2t_hard = m.get("i2t_hard_recall@1", 0)
        gap_val = m.get("modality_gap", 0)
        n_pairs = int(m.get("n_hard_negatives", 0)) if "n_hard_negatives" in m else "—"

        # Find previous
        prev = PREV_RESULTS.get(model, {})
        prev_r1 = prev.get("avg_recall@1")
        prev_str = f"{prev_r1:.3f}" if prev_r1 is not None else "—"
        delta = f"{new_r1 - prev_r1:+.3f}" if prev_r1 is not None else "—"

        elapsed = f"{r.get('elapsed_s', 0):.0f}s"
        print(f"{display:<35} {prev_str:>10} {new_r1:>10.3f} {delta:>8}  {hard_avg:>14.3f} {i2t_hard:>14.3f} {gap_val:>6.2f} {n_pairs!s:>4} {elapsed:>6}")

    print("=" * 120)
    ok = sum(1 for r in results if not r.get("error"))
    err = sum(1 for r in results if r.get("error"))
    print(f"Total: {len(results)} models ({ok} OK, {err} errors)")
    print(f"Results: {RESULTS_FILE}")
    print()
    print("Key: prev R@1 = avg_recall@1 with 50 pairs, new R@1 = avg_recall@1 with 200 pairs")
    print("     hard_avg_R@1 = (t2i_R@1 + i2t_hard_R@1) / 2  [PRIMARY discrimination metric]")
    print("     i2t_hard_R@1 = image->text recall with 200 original + 600 hard negative captions")


if __name__ == "__main__":
    main()
