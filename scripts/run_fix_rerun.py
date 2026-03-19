"""Re-run the two evaluations that failed in the fixed eval run.

1. DashScope multimodal-embedding-v1 cross_modal_retrieval (was failing due to 'dimension' param)
2. SentenceTransformers BAAI/bge-m3 needle_in_haystack (was failing due to wrong prompt_name)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fixed_eval import run_eval, print_summary

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    results = []

    # 1. DashScope cross-modal (fix: no 'dimension' param for multimodal-embedding-v1)
    r = run_eval(
        "dashscope", "cross_modal_retrieval",
        provider_kwargs={"model": "multimodal-embedding-v1"},
        use_mock=False,
    )
    results.append(r)

    # 2. SentenceTransformers needle (fix: removed bad prompt_name)
    r = run_eval(
        "sentence_transformers", "needle_in_haystack",
        provider_kwargs={"model": "BAAI/bge-m3", "device": os.environ.get("CUDA_DEVICE", "cuda:0")},
        use_mock=False,
        haystack_lengths=[1000, 4000, 8000],
        needle_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    results.append(r)

    print_summary(results)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_rerun_fixes.json"

    def _serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_serialize)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
