"""Re-run ALL evaluations after Bug #1-11 fixes.

Bugs fixed:
  #1-3:  API response ordering (OpenAI, ARK, Jina)
  #4-6:  task_type asymmetry removed (crosslingual, cross-modal)
  #7:    Ollama MRL re-normalization after truncation
  #8:    OpenAI hardcoded 3072 dimension check
  #9:    SentenceTransformers CLIP _encode_mixed missing MRL truncation
  #10:   Cache atomic write
  #11:   Needle str.replace replaces all occurrences

Since almost every model-task combo is affected, we re-run everything.
Results saved incrementally to results/eval_rerun_bugfix_20260315.json.

Usage:
    uv run python scripts/run_rerun_all.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rerun_all")

DEVICE = os.environ.get("CUDA_DEVICE", "cuda:0")
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL_PATH", "Qwen/Qwen3-VL-Embedding-2B")
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL_PATH", "google/siglip2-so400m-patch14-384")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "eval_rerun_bugfix_20260315.json"


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


def run_one(
    provider_name: str,
    provider_kwargs: dict[str, Any],
    task_name: str,
    task_kwargs: dict[str, Any],
    label: str,
    provider_cache: dict[str, Any],
    all_results: list[dict],
) -> None:
    """Run a single provider-task combination."""
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    logger.info("")
    logger.info("=" * 70)
    logger.info("%s", label)
    logger.info("=" * 70)

    # Reuse cached provider instance
    cache_key = f"{provider_name}|{json.dumps(provider_kwargs, sort_keys=True)}"
    if cache_key not in provider_cache:
        try:
            provider_cache[cache_key] = get_provider(provider_name, **provider_kwargs)
            # Quick connectivity check
            if hasattr(provider_cache[cache_key], "embed_text"):
                test = provider_cache[cache_key].embed_text(["connectivity test"])
                logger.info("[OK] %s connected, dim=%d", label.split(" — ")[0], test.dimensions)
        except Exception as e:
            logger.error("[FAIL] Cannot init %s: %s", label, e)
            all_results.append({
                "provider": provider_name,
                "model": provider_kwargs.get("model", provider_name),
                "task": task_name,
                "error": f"provider init failed: {e}",
                "metrics": {},
                "elapsed_s": 0,
            })
            save_results(all_results)
            return

    provider = provider_cache[cache_key]

    start = time.time()
    try:
        task = get_task(task_name, **task_kwargs)
        result = task.run(provider)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("%s FAILED after %.1fs: %s", label, elapsed, e)
        all_results.append({
            "provider": provider_name,
            "model": provider_kwargs.get("model", provider_name),
            "task": task_name,
            "error": str(e),
            "metrics": {},
            "elapsed_s": round(elapsed, 1),
        })
        save_results(all_results)
        return

    elapsed = time.time() - start

    # Log key metrics
    m = result.metrics
    parts = []
    for key in ("avg_recall@1", "hard_avg_recall@1", "overall_accuracy",
                "zh2en_recall@1", "en2zh_recall@1", "language_gap",
                "t2i_recall@1", "i2t_hard_recall@1", "modality_gap",
                "degradation_rate", "min_viable_dim"):
        if key in m:
            parts.append(f"{key}={m[key]:.3f}")

    status = ", ".join(parts) if parts else ("ok" if not result.error else f"ERROR: {result.error}")
    logger.info("%s done in %.1fs (%s)", label, elapsed, status)

    entry = json.loads(json.dumps({
        "provider": provider_name,
        "model": result.model_name,
        "task": task_name,
        "metrics": result.metrics,
        "details": result.details or {},
        "error": result.error,
        "elapsed_s": round(elapsed, 1),
    }, default=_serialize))
    all_results.append(entry)
    save_results(all_results)


# ============================================================================
# Model-task definitions
# ============================================================================

# Needle configs (matching original eval scripts)
NEEDLE_STANDARD = {"haystack_lengths": [1000, 4000, 8000], "needle_positions": [0.0, 0.25, 0.5, 0.75, 1.0]}
NEEDLE_DEFAULT_HARD = {}  # uses default 4K-32K
NEEDLE_SHORT = {"haystack_lengths": [1000, 4000], "needle_positions": [0.0, 0.5, 1.0]}
NEEDLE_JINA_CLIP = {"haystack_lengths": [2000, 4000, 8000]}

# Group A: Local GPU models (fastest, no rate limits)
LOCAL_GPU_MODELS = [
    # BGE-M3 (sentence_transformers)
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "BAAI/bge-m3", "device": DEVICE},
        "label": "BGE-M3 (GPU)",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    # Jina v3 local (sentence_transformers)
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "jinaai/jina-embeddings-v3", "device": DEVICE},
        "label": "Jina v3 (GPU)",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    # CLIP ViT-B-32 (sentence_transformers) — 77 token limit, no needle
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "clip-ViT-B-32", "device": DEVICE},
        "label": "CLIP ViT-B-32",
        "tasks": [
            ("crosslingual_retrieval", {}),
            ("cross_modal_retrieval", {}),
        ],
    },
    # CLIP ViT-L-14 (sentence_transformers) — 77 token limit, no needle
    {
        "provider": "sentence_transformers",
        "kwargs": {"model": "clip-ViT-L-14", "device": DEVICE},
        "label": "CLIP ViT-L-14",
        "tasks": [
            ("crosslingual_retrieval", {}),
            ("cross_modal_retrieval", {}),
        ],
    },
    # Qwen3-VL-2B (transformers)
    {
        "provider": "transformers",
        "kwargs": {"model": QWEN_VL_MODEL, "device": DEVICE},
        "label": "Qwen3-VL-2B",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("cross_modal_retrieval", {}),
            ("needle_in_haystack", NEEDLE_SHORT),
        ],
    },
    # SigLIP2 (transformers) — cross-modal only
    {
        "provider": "transformers",
        "kwargs": {"model": SIGLIP_MODEL, "device": DEVICE},
        "label": "SigLIP2-400M",
        "tasks": [
            ("cross_modal_retrieval", {}),
        ],
    },
]

# Group B: Ollama models (local, but sequential)
OLLAMA_MODELS = [
    {
        "provider": "ollama",
        "kwargs": {"model": "nomic-embed-text"},
        "label": "Ollama nomic-embed-text",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "mxbai-embed-large"},
        "label": "Ollama mxbai-embed-large",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "snowflake-arctic-embed:335m"},
        "label": "Ollama snowflake-arctic-embed:335m",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "bge-m3"},
        "label": "Ollama bge-m3",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    {
        "provider": "ollama",
        "kwargs": {"model": "dengcao/Qwen3-Embedding-8B:Q5_K_M"},
        "label": "Ollama Qwen3-8B",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("cross_modal_retrieval", {"max_samples": 50}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
]

# Group C: API models (rate limited)
API_MODELS = [
    # OpenAI text-embedding-3-large
    {
        "provider": "openai",
        "kwargs": {"model": "text-embedding-3-large"},
        "label": "OpenAI text-embedding-3-large",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    # DashScope text-embedding-v3
    {
        "provider": "dashscope",
        "kwargs": {"model": "text-embedding-v3"},
        "label": "DashScope text-embedding-v3",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    # DashScope multimodal-embedding-v1 (no needle — 10K char limit)
    {
        "provider": "dashscope",
        "kwargs": {"model": "multimodal-embedding-v1"},
        "label": "DashScope multimodal-embedding-v1",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("cross_modal_retrieval", {}),
        ],
    },
    # Gemini
    {
        "provider": "gemini",
        "kwargs": {"model": "gemini-embedding-2-preview"},
        "label": "Gemini embedding-2-preview",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("cross_modal_retrieval", {}),
            ("needle_in_haystack", NEEDLE_DEFAULT_HARD),
        ],
    },
    # Jina v4
    {
        "provider": "jina",
        "kwargs": {"model": "jina-embeddings-v4"},
        "label": "Jina v4",
        "tasks": [
            ("mrl_stress", {}),
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    # Jina CLIP v2
    {
        "provider": "jina",
        "kwargs": {"model": "jina-clip-v2"},
        "label": "Jina CLIP v2",
        "tasks": [
            ("crosslingual_retrieval", {}),
            ("cross_modal_retrieval", {}),
            ("needle_in_haystack", NEEDLE_JINA_CLIP),
        ],
    },
    # Voyage multimodal-3.5 (3 RPM for images — very slow)
    {
        "provider": "voyage",
        "kwargs": {},
        "label": "Voyage multimodal-3.5",
        "tasks": [
            ("mrl_stress", {"max_samples": 50}),
            ("crosslingual_retrieval", {}),
            ("cross_modal_retrieval", {"max_samples": 10}),
            ("needle_in_haystack", NEEDLE_SHORT),
        ],
    },
    # Cohere embed-v4.0 (no MRL)
    {
        "provider": "cohere",
        "kwargs": {},
        "label": "Cohere embed-v4.0",
        "tasks": [
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_STANDARD),
        ],
    },
    # ARK doubao (no MRL)
    {
        "provider": "ark",
        "kwargs": {},
        "label": "ARK doubao",
        "tasks": [
            ("crosslingual_retrieval", {}),
            ("needle_in_haystack", NEEDLE_SHORT),
        ],
    },
]


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 110)
    print("  BUGFIX RE-RUN EVALUATION SUMMARY")
    print("=" * 110)
    print(f"{'Provider':<14} {'Model':<35} {'Task':<26} {'Status':<7} {'Time':>6}  Key Metrics")
    print("-" * 110)

    for r in results:
        prov = r.get("provider", "?")
        model = r.get("model", "?")
        if len(model) > 33:
            model = model[:30] + "..."
        task = r.get("task", "?")
        err = r.get("error")
        status = "ERROR" if err else "OK"
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"
        m = r.get("metrics", {})
        parts = []
        for key in ("avg_recall@1", "hard_avg_recall@1", "overall_accuracy",
                     "zh2en_recall@1", "en2zh_recall@1",
                     "t2i_recall@1", "i2t_hard_recall@1",
                     "degradation_rate", "min_viable_dim"):
            if key in m:
                parts.append(f"{key}={m[key]:.3f}")
        metrics_str = ", ".join(parts[:4]) if parts else (err[:50] if err else "—")
        print(f"{prov:<14} {model:<35} {task:<26} {status:<7} {elapsed:>6}  {metrics_str}")

    print("=" * 110)
    ok = sum(1 for r in results if not r.get("error"))
    err = sum(1 for r in results if r.get("error"))
    print(f"Total: {len(results)} entries ({ok} OK, {err} errors)")
    print(f"Results: {RESULTS_FILE}")


def main() -> None:
    all_results: list[dict] = []
    provider_cache: dict[str, Any] = {}
    total_combos = sum(len(m["tasks"]) for m in LOCAL_GPU_MODELS + OLLAMA_MODELS + API_MODELS)

    logger.info("=" * 70)
    logger.info("BUGFIX RE-RUN: %d model-task combinations", total_combos)
    logger.info("Results: %s", RESULTS_FILE)
    logger.info("=" * 70)

    combo_idx = 0

    for group_name, models in [
        ("LOCAL GPU", LOCAL_GPU_MODELS),
        ("OLLAMA", OLLAMA_MODELS),
        ("API", API_MODELS),
    ]:
        logger.info("")
        logger.info("#" * 70)
        logger.info("# GROUP: %s (%d models)", group_name, len(models))
        logger.info("#" * 70)

        for model_def in models:
            prov = model_def["provider"]
            kwargs = model_def["kwargs"]
            model_label = model_def["label"]

            for task_name, task_kwargs in model_def["tasks"]:
                combo_idx += 1
                label = f"[{combo_idx}/{total_combos}] {model_label} — {task_name}"
                run_one(prov, kwargs, task_name, task_kwargs, label, provider_cache, all_results)

                # Small delay between API tasks to avoid burst rate limits
                if group_name == "API":
                    time.sleep(2)

    print_summary(all_results)


if __name__ == "__main__":
    main()
