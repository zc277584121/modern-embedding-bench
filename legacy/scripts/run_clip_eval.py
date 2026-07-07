"""Run CLIP model evaluations: cross-modal retrieval + MRL stress.

Usage:
    uv run python scripts/run_clip_eval.py
"""

from __future__ import annotations

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
logger = logging.getLogger("clip_eval")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fixed_eval import run_eval, print_summary

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")


def main() -> None:
    all_results: list[dict] = []

    clip_models = [
        "clip-ViT-B-32",   # 512d, classic baseline
        "clip-ViT-L-14",   # 768d, larger
    ]

    for model in clip_models:
        logger.info(">>> CLIP: %s <<<", model)

        # Cross-modal retrieval (text <-> image, 200 COCO pairs + hard negatives)
        r = run_eval(
            "sentence_transformers", "cross_modal_retrieval",
            provider_kwargs={"model": model, "device": DEVICE},
            use_mock=False,
        )
        all_results.append(r)

        # MRL stress (text only, full dim — CLIP has fixed dims so no reduction)
        r = run_eval(
            "sentence_transformers", "mrl_stress",
            provider_kwargs={"model": model, "device": DEVICE},
            use_mock=False,
        )
        all_results.append(r)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_clip_20260312.json"

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
