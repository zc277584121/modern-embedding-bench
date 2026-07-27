"""Export validated retrieval-answer system fixture results locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from mm_embed.system_evaluation import export_retrieval_answer_utility_fixture
from mm_embed.system_evaluation.export import DEFAULT_SYSTEM_EXPORT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SYSTEM_EXPORT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_retrieval_answer_utility_fixture(args.output_dir)
    print(f"Exported fixture-only system results to {output}")


if __name__ == "__main__":
    main()
