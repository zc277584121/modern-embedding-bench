"""Run a benchmark v2 manifest.

Example:
    uv run python scripts/run_benchmark.py --manifest benchmark/runs/openai-smoke.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mm_embed.benchmark.runner import BenchmarkRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark-v2.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Remove the output file before running.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    runner, manifest = BenchmarkRunner.from_manifest_path(
        run_manifest=args.manifest,
        benchmark_root=args.root,
        output=args.output,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    records = runner.run_manifest(manifest)
    print(f"Wrote {len(records)} result record(s) to {args.output}")


if __name__ == "__main__":
    main()
