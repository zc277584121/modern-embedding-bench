# Milvus SINDI system Validator guide

This guide applies only to S-20260814-010. The candidate is research-only, the publication gate is
closed, and `pass` means that execution and evidence checks passed. It does not mean that SINDI,
DAAT_MAXSCORE, or any model is better.

## Immutable completed runs

Do not run any formal benchmark command. The completed private run manifests are:

- `results/milvus-sindi-system-v0.1/formal-20260831/native-run.json`: 18 native cells;
- `results/milvus-sindi-system-v0.1/formal-20260831/system-100k-run.json`: 9 system-only cells;
- `results/milvus-sindi-system-v0.1/formal-20260831/system-1m-run.json`: 9 system-only cells.

Their SHA-256 identities are, respectively,
`a6982b48d6be8651f8060a8667a1a0c4aec7487b893032a464b198f77b179ab2`,
`29418222dddc6fec61c0609338f142a864fe700d8ce3284bdbec3f3bf94f617a`, and
`87c152bcf45780ec2d97e751684efef083e029edf9b54f0b1dee6d5f8fa9e9ac`.
The native run has one resumed entry without an inline cell hash. The independent run-manifest
validator resolves that one identity from the immutable cell content and its matching sidecar; it
does not rewrite the run or cell.

## Independent validation

Run these commands from the repository root:

```bash
uv run python scripts/milvus_sindi_system.py validate-formal-run-manifests \
  --output-root results/milvus-sindi-system-v0.1/formal-20260831 \
  --output results/milvus-sindi-system-v0.1/formal-20260831/validation/run-manifest-integrity.json

uv run python scripts/milvus_sindi_system.py validate-formal-results \
  --output-root results/milvus-sindi-system-v0.1/formal-20260831 \
  --output results/milvus-sindi-system-v0.1/formal-20260831/validation/recomputed-summary.json
```

The second command independently reads and hashes every cell sidecar, generated-workload manifest,
exact reference, lifecycle artifact, server index log, and raw trial. It recomputes 1,296 measured
trial files and validates 432 warmup files. It also requires identical comparator query schedules,
explicit algorithms, index state `Finished`, load state `Loaded`, all rows in indexed sealed
segments, zero growing rows, and zero active or queued compaction.

Validate the aggregate schema with:

```bash
uv run python - <<'PY'
import json
from jsonschema import Draft202012Validator, FormatChecker

schema = json.load(open("schemas/milvus-sindi-formal-summary-v01.schema.json", encoding="utf-8"))
value = json.load(open("benchmark/artifacts/milvus-sindi-system-v0.1/formal-summary.json", encoding="utf-8"))
Draft202012Validator.check_schema(schema)
Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
PY
```

## Identity and interpretation boundaries

The accepted source commit is `74635817770023b70b1e9f9e97b3d83d9d11f987`. The entry path
reports `model_loaded=false` and reuses saved sparse CSR and exact top-100 artifacts read-only. No
model encoding or model download is part of validation.

The server is Milvus 3.0.0 from
`milvusdb/milvus@sha256:804b50bc1523a64e3c0f18cf33af7e0f8b33329584698ffee61c606d6fe8ee26`,
with PyMilvus 3.0.1 and vector index version 10. The first real SINDI build on the default vector
index version 8 failed closed; the server did not fall back. Version 10 persisted document weights
as float16. In the native private replay, both algorithms have minimum strict recall 0.9 against
that representation and tie-aware Recall@K 1.0. System-only strict recall can be lower because its
synthetic near-tie boundary is denser; it is not model-quality evidence.

The aggregate preserves native and system-only tiers separately. It includes QPS, P50/P95/P99,
client preparation, RPC end-to-end, server search-counter means, proxy network byte counters,
client CPU/RSS, container CPU/RSS/PIDs, insert/flush/build/load time, persisted index bytes, runtime
bytes, and exact CSR bytes. Proxy sent bytes are a global counter captured while the Story cell was
the only active benchmark cell; no packet-level or exact protobuf-size claim is made.

## Isolation and deferred cleanup

The Story-owned container, network, loopback ports, runtime root, and 26-collection allowlist remain
preserved. The runtime root must remain mode 0711. Final cleanup is not authorized by this guide.
The exact private cleanup inventory is
`results/milvus-sindi-system-v0.1/formal-20260831/validation/final-cleanup-inventory.json`.

The final multi-day isolation comparison preserves the non-Story container-name, state, image,
network, volume, and listening-endpoint sets. Five unrelated demo container IDs changed and one of
those mount sets changed during the interval. No Story command targeted them, but host-wide Docker
event attribution is outside this evidence, so literal global object-identity stability is not
claimed.

## Quality gates

```bash
uv run python -m pytest tests/test_milvus_sindi_system.py -q
uv run python -m pytest
uv run python -m compileall -q src scripts tests
uv run ruff check \
  scripts/milvus_sindi_system.py \
  src/mm_embed/benchmark/milvus_sindi_fixture.py \
  src/mm_embed/benchmark/milvus_sindi_formal.py \
  src/mm_embed/benchmark/milvus_sindi_results.py \
  src/mm_embed/benchmark/milvus_sindi_system.py \
  tests/test_milvus_sindi_system.py
uv run ruff format --check \
  scripts/milvus_sindi_system.py \
  src/mm_embed/benchmark/milvus_sindi_fixture.py \
  src/mm_embed/benchmark/milvus_sindi_formal.py \
  src/mm_embed/benchmark/milvus_sindi_results.py \
  src/mm_embed/benchmark/milvus_sindi_system.py \
  tests/test_milvus_sindi_system.py
uv lock --check
git diff --check
```

The Story-scoped Ruff commands pass. Repository-wide `ruff check .` and
`ruff format --check .` were also attempted and remain red because of unrelated pre-existing files;
the format check reports 98 files outside this Story scope that would be reformatted. This candidate
does not modify that unrelated lint debt.

Tracked aggregate artifacts contain no source text, canonical document/query identifiers, or raw
rankings. The generated system-only workloads cannot support quality, multilingual-effectiveness,
verified zero-shot, or public benchmark claims.
