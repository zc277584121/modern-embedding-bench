"""Build a CSV leaderboard from benchmark v2 JSONL results."""

from __future__ import annotations

import argparse
from pathlib import Path

from mm_embed.benchmark.leaderboard import build_from_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/leaderboard.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_from_file(args.results, args.output, benchmark_root=args.root)
    print(f"Wrote {len(rows)} leaderboard row(s) to {args.output}")


if __name__ == "__main__":
    main()
