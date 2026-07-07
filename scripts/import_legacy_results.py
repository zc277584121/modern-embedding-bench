"""Import legacy results/*.json files into benchmark v2 JSONL format."""

from __future__ import annotations

import argparse
from pathlib import Path

from mm_embed.benchmark.results import import_legacy_result_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/legacy-import.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    for path in args.inputs:
        total += import_legacy_result_file(path, args.output)
    print(f"Imported {total} legacy record(s) into {args.output}")


if __name__ == "__main__":
    main()
