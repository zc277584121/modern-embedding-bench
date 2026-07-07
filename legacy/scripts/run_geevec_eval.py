"""Run bounded GeeVec evaluation combinations.

Examples:
    uv run --extra local python scripts/run_geevec_eval.py --preset smoke --dry-run-count 20
    uv run --extra local python scripts/run_geevec_eval.py --preset standard --provider geevec_lite --domain coding
    uv run python scripts/run_geevec_eval.py --provider geevec_api --task needle_in_haystack --allow-api-calls
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("geevec_eval")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geevec"
DEFAULT_PROVIDERS = ["geevec_lite"]
DEFAULT_DOMAINS = ["general"]
DEFAULT_TASKS = ["mrl_stress", "needle_in_haystack", "crosslingual_retrieval"]
API_PROVIDER = "geevec_api"
PRESETS = ["smoke", "standard", "full"]
SUPPORTED_TEXT_TASKS = [
    "mrl_stress",
    "needle_in_haystack",
    "crosslingual_retrieval",
    "chinese_multimodal",
    "autonomous_driving",
]
TASKS_WITH_TEXT_ONLY_FALLBACK = {"chinese_multimodal", "autonomous_driving"}

PRESET_TASK_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "smoke": {
        "mrl_stress": {"use_mock": False, "max_samples": 12},
        "needle_in_haystack": {
            "use_mock": False,
            "haystack_lengths": [1000],
            "needle_positions": [0.0, 0.5, 1.0],
        },
        "crosslingual_retrieval": {"use_mock": False},
        "chinese_multimodal": {},
        "autonomous_driving": {},
    },
    "standard": {
        "mrl_stress": {"use_mock": False, "max_samples": 150},
        "needle_in_haystack": {
            "use_mock": False,
            "haystack_lengths": [1000, 4000, 8000],
            "needle_positions": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
        "crosslingual_retrieval": {"use_mock": False},
        "chinese_multimodal": {},
        "autonomous_driving": {},
    },
    "full": {
        "mrl_stress": {"use_mock": False, "max_samples": None},
        "needle_in_haystack": {
            "use_mock": False,
            "haystack_lengths": [1000, 4000, 8000, 16000, 32000],
            "needle_positions": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
        "crosslingual_retrieval": {"use_mock": False},
        "chinese_multimodal": {},
        "autonomous_driving": {},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="smoke")
    parser.add_argument("--provider", nargs="+", choices=["geevec_api", "geevec_lite"], default=DEFAULT_PROVIDERS)
    parser.add_argument("--domain", nargs="+", choices=["general", "coding", "reasoning"], default=DEFAULT_DOMAINS)
    parser.add_argument("--task", nargs="+", choices=SUPPORTED_TEXT_TASKS, default=DEFAULT_TASKS)
    parser.add_argument(
        "--mrl-max-samples",
        type=int,
        help="Override MRL max_samples. Use preset full for all samples.",
    )
    parser.add_argument("--max-samples", type=int, help="Alias for --mrl-max-samples.")
    parser.add_argument("--needle-haystack-lengths", type=int, nargs="+", help="Override needle haystack lengths.")
    parser.add_argument("--needle-positions", type=float, nargs="+", help="Override needle positions, e.g. 0 0.5 1.")
    parser.add_argument(
        "--dry-run-count",
        type=int,
        nargs="?",
        const=20,
        help="Print planned combinations and input estimates without running them.",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print the full plan and input estimates without running.",
    )
    parser.add_argument("--allow-api-calls", action="store_true", help="Required before geevec_api sends requests.")
    parser.add_argument("--api-batch-size", type=int, default=2, help="Small GeeVec API batch size for bounded runs.")
    parser.add_argument("--lite-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Output JSON path. Defaults under results/geevec/.")
    return parser.parse_args()


def build_combinations(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {"provider": provider, "domain": domain, "task": task}
        for provider in args.provider
        for domain in args.domain
        for task in args.task
    ]


def provider_kwargs(provider_name: str, domain: str, args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"domain": domain}
    if provider_name == API_PROVIDER:
        kwargs["batch_size"] = args.api_batch_size
    else:
        kwargs["batch_size"] = args.lite_batch_size
    return kwargs


def task_kwargs(task_name: str, args: argparse.Namespace) -> dict[str, Any]:
    kwargs = dict(PRESET_TASK_DEFAULTS[args.preset].get(task_name, {"use_mock": False}))
    if task_name == "mrl_stress":
        mrl_max_samples = args.mrl_max_samples if args.mrl_max_samples is not None else args.max_samples
        if mrl_max_samples is not None:
            kwargs["max_samples"] = mrl_max_samples
    if task_name == "needle_in_haystack":
        if args.needle_haystack_lengths is not None:
            kwargs["haystack_lengths"] = args.needle_haystack_lengths
        if args.needle_positions is not None:
            kwargs["needle_positions"] = args.needle_positions
    return kwargs


def estimate_inputs(task_name: str, kwargs: dict[str, Any]) -> tuple[int, str]:
    if task_name == "mrl_stress":
        return estimate_mrl_inputs(kwargs)
    if task_name == "needle_in_haystack":
        return estimate_needle_inputs(kwargs)
    if task_name == "crosslingual_retrieval":
        return estimate_crosslingual_inputs()
    return 0, "unknown task"


def estimate_mrl_inputs(kwargs: dict[str, Any]) -> tuple[int, str]:
    max_samples = kwargs.get("max_samples")
    try:
        from mm_embed.data.real_data import load_mrl_continuous_data

        data = load_mrl_continuous_data()
        if max_samples is not None:
            data = data[:max_samples]
        unique_texts = {text for a, b, _ in data for text in (a, b)}
        return len(unique_texts), f"unique STS texts from {len(data)} pairs"
    except Exception:
        if max_samples is None:
            return 2758, "fallback: 2 * full STS-B test pairs"
        return int(max_samples) * 2, "fallback: 2 * selected MRL pairs"


def estimate_needle_inputs(kwargs: dict[str, Any]) -> tuple[int, str]:
    lengths = kwargs.get("haystack_lengths")
    positions = kwargs.get("needle_positions")
    try:
        from mm_embed.data.real_data import load_needle_haystack_real_data

        cases = load_needle_haystack_real_data(haystack_lengths=lengths, needle_positions=positions)
        queries = {case["query"] for case in cases}
        docs_with = {case["document"] for case in cases}
        docs_without = {case["document"].replace(case["needle"], "", 1) for case in cases}
        total = len(queries) + len(docs_with) + len(docs_without)
        return total, f"{len(queries)} queries + {len(docs_with)} docs_with + {len(docs_without)} docs_without"
    except Exception:
        n_lengths = len(lengths or [])
        n_positions = len(positions or [])
        approx_needles = 10
        cases = n_lengths * n_positions * approx_needles
        return approx_needles + cases * 2, "fallback: approx unique queries + with/without docs"


def estimate_crosslingual_inputs() -> tuple[int, str]:
    try:
        from mm_embed.data.real_data import load_crosslingual_data

        data = load_crosslingual_data()
        hard_neg_en = sum(len(item.hard_negatives_en) for item in data)
        hard_neg_zh = sum(len(item.hard_negatives_zh) for item in data)
        total = len(data) * 2 + hard_neg_en + hard_neg_zh
        return total, f"{len(data)} zh + {len(data)} en + {hard_neg_en} hard_neg_en + {hard_neg_zh} hard_neg_zh"
    except Exception:
        return 0, "crosslingual data unavailable"


def print_plan(args: argparse.Namespace, combinations: list[dict[str, str]]) -> None:
    estimates = {task: estimate_inputs(task, task_kwargs(task, args)) for task in args.task}
    total_api_inputs = 0

    print(f"Preset: {args.preset}")
    print("Task settings:")
    for task in args.task:
        estimate, note = estimates[task]
        print(f"  {task}: {task_kwargs(task, args)}; estimated inputs={estimate} ({note})")

    print("\nPlanned combinations:")
    limit = len(combinations) if args.estimate else args.dry_run_count
    for combo in combinations[:limit]:
        estimate, note = estimates[combo["task"]]
        is_api = combo["provider"] == API_PROVIDER
        if is_api:
            total_api_inputs += estimate
        suffix = f"estimated_inputs={estimate}"
        if is_api and not args.allow_api_calls:
            suffix += "; API skipped unless --allow-api-calls is set"
        print(f"  {combo['provider']}/{combo['domain']}/{combo['task']} - {suffix} ({note})")

    if not args.estimate and len(combinations) > limit:
        print(f"  ... {len(combinations) - limit} more")

    print(f"\nApprox GeeVec API embedding input texts for printed API combos: {total_api_inputs}")


def result_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"geevec_eval_{stamp}.json"


def serialize_result(result: Any, elapsed_s: float, provider: str, domain: str, task: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "domain": domain,
        "task": task,
        "model": result.model_name,
        "metrics": result.metrics,
        "details": result.details,
        "error": result.error,
        "elapsed_s": round(elapsed_s, 2),
    }


def run_one(combo: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    from mm_embed.providers import get_provider
    from mm_embed.tasks import get_task

    provider_name = combo["provider"]
    domain = combo["domain"]
    task_name = combo["task"]

    if provider_name == API_PROVIDER and not args.allow_api_calls:
        return {
            **combo,
            "metrics": {},
            "details": {},
            "error": "Skipped: pass --allow-api-calls to run geevec_api.",
        }

    started = time.time()
    try:
        provider = get_provider(provider_name, **provider_kwargs(provider_name, domain, args))
        task = get_task(task_name, **task_kwargs(task_name, args))
    except Exception as exc:
        return {**combo, "metrics": {}, "details": {}, "error": f"init failed: {exc}"}

    if task_name not in TASKS_WITH_TEXT_ONLY_FALLBACK and not task.check_compatibility(provider):
        return {
            **combo,
            "model": getattr(provider, "model", provider_name),
            "metrics": {},
            "details": {},
            "error": f"Skipped: {provider_name} is not compatible with {task_name}.",
            "elapsed_s": round(time.time() - started, 2),
        }

    logger.info("Running %s / %s / %s", provider_name, domain, task_name)
    result = task.run(provider)
    return serialize_result(result, time.time() - started, provider_name, domain, task_name)


def save(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    logger.info("Saved %d result entries to %s", len(results), path)


def main() -> None:
    args = parse_args()
    combinations = build_combinations(args)

    if args.dry_run_count is not None or args.estimate:
        print_plan(args, combinations)
        return

    output = result_path(args)
    results: list[dict[str, Any]] = []
    for combo in combinations:
        entry = run_one(combo, args)
        results.append(entry)
        save(results, output)

    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
