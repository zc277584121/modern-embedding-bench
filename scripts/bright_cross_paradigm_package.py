#!/usr/bin/env python3
"""Build or check the frozen S-009 cross-paradigm research candidate."""

from __future__ import annotations

import argparse
import json

from mm_embed.benchmark.bright_cross_paradigm_package import build_or_check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    args = parser.parse_args()
    identity = build_or_check(check_only=args.command == "check")
    print(
        json.dumps(
            {"status": "pass" if args.command == "check" else "candidate_built", "manifest_sha256": identity},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
