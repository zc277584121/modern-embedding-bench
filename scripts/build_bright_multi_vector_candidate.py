#!/usr/bin/env python3
"""Build or verify the exact ordered S-20260814-012 validator candidate identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mm_embed.benchmark.bright_multi_vector_package import PUBLICATION, candidate_paths
from mm_embed.providers.real_multi_vector import file_sha256

OUTPUT = Path("benchmark/artifacts/bright-multi-vector-v0.2/validator-candidate.json")
PROTECTED = {
    "benchmark/artifacts/milvus-sindi-system-v0.1/validator-candidate.json",
    "benchmark/artifacts/milvus-sindi-system-v0.1/validator-candidate.json.sha256",
}


def formatted(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def build(repo: Path) -> dict[str, object]:
    paths = candidate_paths()
    if tuple(sorted(paths)) != paths or len(paths) != len(set(paths)):
        raise ValueError("Candidate paths must be unique and lexicographically ordered")
    if any(path.startswith(("results/", "cache/", ".cache/")) or path in PROTECTED for path in paths):
        raise ValueError("Candidate set contains a restricted or protected path")
    records = [
        {"path": path, "bytes": (repo / path).stat().st_size, "sha256": file_sha256(repo / path)} for path in paths
    ]
    candidate_set_sha256 = hashlib.sha256(formatted(records)).hexdigest()
    return {
        "schema_version": "bright-real-multi-vector-validator-candidate-v1",
        "story_id": "S-20260814-012",
        "status": "candidate_not_accepted",
        "classification": "tracked_candidate_identity",
        "candidate_file_count": len(records),
        "candidate_set_sha256": candidate_set_sha256,
        "files": records,
        "excluded_prefixes": ["results/", "cache/", ".cache/"],
        "protected_files_excluded": sorted(PROTECTED),
        "publication": PUBLICATION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build(Path("."))
    payload = formatted(value)
    identity = hashlib.sha256(payload).hexdigest()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.check:
        if args.output.read_bytes() != payload or sidecar.read_text(encoding="ascii").strip() != identity:
            raise ValueError("Validator candidate identity cannot be rebuilt exactly")
    else:
        if args.output.exists() or sidecar.exists():
            raise ValueError("Refusing to overwrite an existing validator candidate identity")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        sidecar.write_text(identity + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "candidate_file_count": len(value["files"]),
                "candidate_set_sha256": value["candidate_set_sha256"],
                "manifest_sha256": identity,
                "status": "pass" if args.check else "built",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
