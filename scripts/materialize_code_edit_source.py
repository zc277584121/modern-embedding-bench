#!/usr/bin/env python3
"""Run one bounded no-publish code-edit source materialization contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_embed.data.code_edit_chunk_source_materializer import (
    load_source_contract,
    run_source_materialization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Pinned JSON source contract")
    parser.add_argument("--temp-root", type=Path, default=None, help="Optional parent for the dedicated temp path")
    args = parser.parse_args()

    contract = load_source_contract(args.config)
    result = run_source_materialization(contract, temp_root=args.temp_root)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
