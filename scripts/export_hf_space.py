"""Export Hugging Face Gradio Space files for benchmark publishing."""

from __future__ import annotations

import argparse
from pathlib import Path

from mm_embed.hf_publish import export_space_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/huggingface/space"))
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--leaderboard", type=Path, default=None)
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_space_repo(
        output_dir=args.output_dir,
        dataset_repo_id=args.dataset_repo_id,
        bundled_leaderboard=args.leaderboard,
        clean=not args.no_clean,
    )
    print(f"Exported Hugging Face Space folder to {output}")


if __name__ == "__main__":
    main()
