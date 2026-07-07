"""Run evaluations for new API embedding providers: Cohere, Jina, Voyage, ARK.

Usage:
    uv run python scripts/run_api_eval.py
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
logger = logging.getLogger("api_eval")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "eval_api_new_20260312.json"


def _ts() -> str:
    """Return a short timestamp for progress logs."""
    return datetime.now().strftime("%H:%M")


def test_connectivity(provider_name: str, **provider_kwargs) -> bool:
    """Test a provider with a simple 1-sentence embed call."""
    from mm_embed.providers import get_provider

    logger.info("[%s] Testing connectivity for %s...", _ts(), provider_name)
    try:
        provider = get_provider(provider_name, **provider_kwargs)
        result = provider.embed_text(["Hello world"])
        dim = result.dimensions
        logger.info("[%s] %s OK! dim=%d, latency=%.0fms", _ts(), provider_name, dim, result.latency_ms)
        return True
    except Exception as e:
        logger.error("[%s] %s FAILED: %s", _ts(), provider_name, e)
        return False


def run_eval(
    provider_name: str,
    task_name: str,
    provider_kwargs: dict | None = None,
    **task_kwargs,
) -> dict:
    """Run a single provider+task evaluation."""
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    logger.info("[%s] Starting %s / %s...", _ts(), provider_name, task_name)

    try:
        provider = get_provider(provider_name, **(provider_kwargs or {}))
        task = get_task(task_name, **task_kwargs)
    except Exception as e:
        logger.error("[%s] Init failed: %s", _ts(), e)
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
        logger.error("[%s] %s / %s failed after %.1fs: %s", _ts(), provider_name, task_name, elapsed, e)
        return {
            "provider": provider_name,
            "task": task_name,
            "error": str(e),
            "metrics": {},
            "elapsed_s": round(elapsed, 1),
        }

    elapsed = time.time() - start

    # Log key metrics inline
    key_metrics = []
    m = result.metrics
    if "spearman_full" in m:
        key_metrics.append(f"ρ={m['spearman_full']:.3f}")
    if "overall_accuracy" in m:
        key_metrics.append(f"acc={m['overall_accuracy']:.3f}")
    if "avg_recall@1" in m:
        key_metrics.append(f"R@1={m['avg_recall@1']:.3f}")
    if "t2i_recall@1" in m:
        key_metrics.append(f"t2i={m['t2i_recall@1']:.3f}")
    if "min_viable_dim" in m:
        key_metrics.append(f"min_dim={int(m['min_viable_dim'])}")

    metrics_str = ", ".join(key_metrics) if key_metrics else "done"
    logger.info("[%s] %s / %s done in %.1fs (%s)", _ts(), provider_name, task_name, elapsed, metrics_str)

    return {
        "provider": provider_name,
        "model": result.model_name,
        "task": task_name,
        "metrics": result.metrics,
        "details": result.details if result.details else {},
        "error": result.error,
        "elapsed_s": round(elapsed, 1),
    }


def save_results(all_results: list[dict]) -> None:
    """Save results incrementally to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=_serialize)

    logger.info("Results saved to %s (%d entries)", RESULTS_FILE, len(all_results))


def print_summary(results: list[dict]) -> None:
    """Print a summary table."""
    print("\n" + "=" * 90)
    print("  API PROVIDER EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Provider':<12} {'Task':<25} {'Status':<8} {'Time':>8}  Key Metrics")
    print("-" * 90)

    for r in results:
        status = "ERROR" if r.get("error") else "OK"
        elapsed = f"{r.get('elapsed_s', 0):.0f}s"

        metrics = r.get("metrics", {})
        parts = []
        if "spearman_full" in metrics:
            parts.append(f"ρ={metrics['spearman_full']:.3f}")
        if "overall_accuracy" in metrics:
            parts.append(f"acc={metrics['overall_accuracy']:.3f}")
        if "avg_recall@1" in metrics:
            parts.append(f"R@1={metrics['avg_recall@1']:.3f}")
        if "t2i_recall@1" in metrics:
            parts.append(f"t2i={metrics['t2i_recall@1']:.3f}")
        if "i2t_recall@1" in metrics:
            parts.append(f"i2t={metrics['i2t_recall@1']:.3f}")
        if "modality_gap" in metrics:
            parts.append(f"gap={metrics['modality_gap']:.3f}")
        if "min_viable_dim" in metrics:
            parts.append(f"min_dim={int(metrics['min_viable_dim'])}")
        if "degradation_rate" in metrics:
            parts.append(f"degrad={metrics['degradation_rate']:.3f}")

        key_str = ", ".join(parts[:5]) if parts else (r.get("error", "")[:40] if r.get("error") else "—")
        print(f"{r['provider']:<12} {r['task']:<25} {status:<8} {elapsed:>8}  {key_str}")

    print("=" * 90)


def main() -> None:
    all_results: list[dict] = []

    # =========================================================================
    # Phase 2: Test connectivity for each provider
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Connectivity Tests")
    logger.info("=" * 60)

    providers_ok: dict[str, bool] = {}
    for name in ["cohere", "jina", "voyage", "ark"]:
        providers_ok[name] = test_connectivity(name)

    working = [k for k, v in providers_ok.items() if v]
    failed = [k for k, v in providers_ok.items() if not v]
    logger.info("Working providers: %s", working)
    if failed:
        logger.warning("Failed providers (will skip): %s", failed)

    # =========================================================================
    # Phase 3: Run evaluations (fastest first)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Evaluations")
    logger.info("=" * 60)

    # --- Cohere: Needle only (text-only, doesn't support MRL) ---
    if providers_ok.get("cohere"):
        logger.info(">>> COHERE EVALUATIONS <<<")

        r = run_eval(
            "cohere", "needle_in_haystack",
            use_mock=False,
            haystack_lengths=[1000, 4000, 8000],
            needle_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        all_results.append(r)
        save_results(all_results)

    # --- Jina: MRL + Needle (text-only) ---
    if providers_ok.get("jina"):
        logger.info(">>> JINA EVALUATIONS <<<")

        r = run_eval("jina", "mrl_stress", use_mock=False, max_samples=200)
        all_results.append(r)
        save_results(all_results)

        r = run_eval(
            "jina", "needle_in_haystack",
            use_mock=False,
            haystack_lengths=[1000, 4000, 8000],
            needle_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        all_results.append(r)
        save_results(all_results)

    # --- Voyage: MRL + Needle + Cross-Modal (multimodal, 3 RPM → slow) ---
    if providers_ok.get("voyage"):
        logger.info(">>> VOYAGE EVALUATIONS (3 RPM — this will be slow) <<<")

        r = run_eval("voyage", "mrl_stress", use_mock=False, max_samples=50)
        all_results.append(r)
        save_results(all_results)

        r = run_eval(
            "voyage", "needle_in_haystack",
            use_mock=False,
            haystack_lengths=[1000, 4000],
            needle_positions=[0.0, 0.5, 1.0],
        )
        all_results.append(r)
        save_results(all_results)

        r = run_eval("voyage", "cross_modal_retrieval", use_mock=False, max_samples=10)
        all_results.append(r)
        save_results(all_results)

    # --- ARK: Needle with small config (skip MRL as it doesn't support dimension reduction) ---
    if providers_ok.get("ark"):
        logger.info(">>> ARK EVALUATIONS (limited tokens — small needle test) <<<")

        r = run_eval(
            "ark", "needle_in_haystack",
            use_mock=False,
            haystack_lengths=[1000, 4000],
            needle_positions=[0.0, 0.5, 1.0],
        )
        all_results.append(r)
        save_results(all_results)

    # =========================================================================
    # Final summary
    # =========================================================================
    save_results(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()
