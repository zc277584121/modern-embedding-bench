"""Export Hugging Face Dataset repo files for benchmark publishing."""

from __future__ import annotations

import argparse
from pathlib import Path

from mm_embed.hf_publish import export_dataset_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/huggingface/dataset"))
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--leaderboard", type=Path, default=None)
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_dataset_repo(
        output_dir=args.output_dir,
        benchmark_root=args.benchmark_root,
        results_path=args.results,
        leaderboard_path=args.leaderboard,
        include_data=args.include_data,
        include_images=args.include_images,
        clean=not args.no_clean,
    )
    print(f"Exported Hugging Face Dataset folder to {output}")


if __name__ == "__main__":
    main()
