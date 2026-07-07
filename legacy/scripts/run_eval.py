#!/usr/bin/env python3
"""Quick evaluation script — run specific provider + task combos.

Usage:
    python scripts/run_eval.py --provider dashscope --task mrl_stress
    python scripts/run_eval.py --provider dashscope gemini --task cross_modal_retrieval needle_in_haystack
    python scripts/run_eval.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mm_embed.providers import PROVIDER_REGISTRY, get_provider
from mm_embed.tasks import TASK_REGISTRY, get_task


def main():
    parser = argparse.ArgumentParser(description="Run embedding evaluations")
    parser.add_argument("--provider", "-p", nargs="+", default=[], help="Provider names")
    parser.add_argument("--task", "-t", nargs="+", default=[], help="Task names")
    parser.add_argument("--all", action="store_true", help="Run all providers × all tasks")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.all:
        provider_names = list(PROVIDER_REGISTRY.keys())
        task_names = list(TASK_REGISTRY.keys())
    else:
        provider_names = args.provider
        task_names = args.task

    if not provider_names or not task_names:
        print("Specify --provider and --task, or use --all")
        print(f"\nProviders: {', '.join(PROVIDER_REGISTRY.keys())}")
        print(f"Tasks: {', '.join(TASK_REGISTRY.keys())}")
        sys.exit(1)

    results = []

    for p_name in provider_names:
        print(f"\n{'='*60}")
        print(f"Provider: {p_name}")
        print(f"{'='*60}")

        try:
            provider = get_provider(p_name)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        for t_name in task_names:
            print(f"\n  Task: {t_name}")
            try:
                task = get_task(t_name)
                result = task.run(provider)

                if result.error:
                    print(f"    ERROR: {result.error}")
                else:
                    for k, v in sorted(result.metrics.items()):
                        print(f"    {k}: {v:.4f}")

                results.append({
                    "provider": p_name,
                    "task": t_name,
                    "metrics": result.metrics,
                    "error": result.error,
                })
            except Exception as e:
                print(f"    EXCEPTION: {e}")
                results.append({
                    "provider": p_name,
                    "task": t_name,
                    "metrics": {},
                    "error": str(e),
                })

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
