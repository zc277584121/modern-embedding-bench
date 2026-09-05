#!/usr/bin/env python3
"""Run or collect frozen S-009 unified resource measurements."""

from __future__ import annotations

import argparse
import json
import sys

from mm_embed.benchmark.bright_cross_paradigm_package import METHODS
from mm_embed.benchmark.bright_cross_paradigm_resources import (
    collect,
    correct_minilm_scope_return2,
    install_timeout,
    write_cell,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-cell")
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--track", choices=("economics", "psychology"), required=True)
    run.add_argument("--replace-failed-return1", action="store_true")
    commands.add_parser("collect")
    commands.add_parser("correct-minilm-scope-return2")
    args = parser.parse_args()
    if args.command == "collect":
        print(json.dumps({"status": "pass", "resource_summary_sha256": collect()}, sort_keys=True))
        return
    if args.command == "correct-minilm-scope-return2":
        print(json.dumps({"status": "pass", **correct_minilm_scope_return2()}, sort_keys=True))
        return
    install_timeout()
    identity, status = write_cell(
        args.method,
        args.track,
        replace_failed_return1=args.replace_failed_return1,
    )
    if status == "failed_closed":
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "cell": f"{args.method}:{args.track}",
                    "identity": identity,
                },
                sort_keys=True,
            )
        )
        sys.exit(1)
    print(json.dumps({"status": "pass", "cell": f"{args.method}:{args.track}", "identity": identity}, sort_keys=True))


if __name__ == "__main__":
    main()
