#!/usr/bin/env python3
"""Build or verify the learned-sparse phase-one synthesis package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_embed.benchmark.learned_sparse_phase_one import check_outputs, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = write_outputs(args.repo_root) if args.command == "build" else check_outputs(args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
