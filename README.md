# Modern Embedding Bench

A maintainable benchmark framework for evaluating embedding models on practical
retrieval scenarios that are under-covered by broad public leaderboards.

The project is being refactored from a one-off evaluation repo into a
data-backed benchmark that can publish clean artifacts to Hugging Face:

- reviewable model registry
- reviewable task registry
- manifest-based runs
- JSONL result records
- generated leaderboard tables
- legacy result import

## Current Focus

The benchmark focuses on scenario gaps that matter for RAG, multimodal search,
and agent systems:

- MRL / dimension compression robustness
- Chinese-English cross-lingual retrieval with hard negatives
- long-document needle retrieval for embedding models
- text-image retrieval with hard negative captions
- domain retrieval tasks that can later grow into agent memory, tool-doc, and
  code-aware retrieval tracks

## Repository Layout

```text
benchmark/
  models/                 # Model specs in YAML
  tasks/                  # Task specs, metrics, dataset versions
  runs/                   # Run manifests
schemas/                  # JSON schemas for model/task/run/result artifacts
src/mm_embed/
  benchmark/              # v2 registry, runner, result, leaderboard utilities
  providers/              # Provider adapters
  tasks/                  # Evaluation task implementations
  data/                   # Dataset loaders
scripts/
  run_benchmark.py        # Manifest runner
  build_leaderboard.py    # JSONL -> CSV leaderboard
  import_legacy_results.py
  export_hf_dataset.py    # Build a Hugging Face Dataset repo folder
  export_hf_space.py      # Build a Hugging Face Gradio Space folder
  upload_hf.py            # Upload prepared folders to Hugging Face Hub
  prepare_*.py            # Dataset preparation scripts
legacy/
  scripts/                # Old one-off runners and report generators
```

Generated `results/` and `reports/` directories are ignored by git.

## Install

```bash
uv sync
```

Install optional extras only when needed:

```bash
uv sync --extra openai
uv sync --extra local
uv sync --extra data
```

## Inspect The Registry

```bash
uv run modern-embed-bench benchmark models
uv run modern-embed-bench benchmark tasks
```

Model and task definitions live in `benchmark/models/*.yaml` and
`benchmark/tasks/*.yaml`. Adding a new model should usually start as a YAML
change before any new provider code is written.

## Run A Smoke Benchmark

```bash
uv run modern-embed-bench benchmark run \
  --manifest benchmark/runs/openai-smoke.yaml \
  --output results/openai-smoke.jsonl \
  --overwrite

uv run modern-embed-bench benchmark leaderboard \
  --results results/openai-smoke.jsonl \
  --output results/openai-smoke-leaderboard.csv
```

The same commands are available as scripts:

```bash
uv run python scripts/run_benchmark.py --manifest benchmark/runs/openai-smoke.yaml --overwrite
uv run python scripts/build_leaderboard.py --results results/benchmark-v2.jsonl
```

## Reproduce The Pinned Code-Source Contract Smoke

The accepted `psf/requests` issue-to-edit source contract can be materialized
locally without model scoring or publication. The command verifies pinned
GitHub metadata and payload hashes, applies the Stage A/Stage B eligibility
policy, builds deterministic chunks and patch-derived qrels under explicit
caps, prints a source-free evidence summary, and removes its dedicated
temporary path on PASS, FAILED, or BLOCKED:

```bash
uv run --no-sync python scripts/materialize_code_edit_source.py \
  --config benchmark/research/code_edit_chunk_requests_source_contract_20260722.json
```

The configured `wall_seconds` deadline is enforced inside the Python process,
including while a source request is blocked. `SIGTERM` is converted into a
controlled stop, the deadline is disabled while the dedicated path is removed,
and the prior process signal handlers are restored afterward. An external
`timeout` wrapper is therefore not required for the 30-minute bound.

This path uses one repository and one issue, caps the archive at 10 MB,
extracted regular files at 25 MB, each Stage A candidate at 2 MB, eligible
normalized text at 5 MB, tracked files at 500, chunks at 1,000, target RSS at
256 MiB, and wall time at 30 minutes. It does not call provider APIs, download
models or datasets, retain third-party source, register a public task or score,
or perform any Hugging Face operation.

## Result Shape

Each evaluation writes one JSONL record per model-task pair. Records include:

- schema version
- run id, metadata, publication intent, and normalized evidence tier
- git sha
- model spec id and provider kwargs without secrets
- task spec id and task kwargs
- metrics and details
- error, if the model-task run failed

Legacy JSON result files can be converted:

```bash
uv run python scripts/import_legacy_results.py legacy/results/eval_rerun_bugfix_20260315.json \
  --output results/legacy-import.jsonl
```

New result records store `run.publish` and `run.evidence_tier` (`legacy`,
`smoke`, `benchmark`, `fixture`, or `unknown`). Explicit `publish: false`
records are kept out of public result and leaderboard exports. Historical v2
records without `run.publish` remain public by default, and older evidence
metadata continues to use compatibility classification.

## Hugging Face Publishing

Export a Dataset repo folder:

```bash
uv run python scripts/export_hf_dataset.py \
  --results results/openai-smoke.jsonl \
  --leaderboard results/openai-smoke-leaderboard.csv \
  --output-dir dist/huggingface/dataset
```

Export a Gradio Space folder:

```bash
uv run python scripts/export_hf_space.py \
  --dataset-repo-id <namespace>/modern-embedding-bench \
  --leaderboard results/openai-smoke-leaderboard.csv \
  --output-dir dist/huggingface/space
```

Upload with a token from `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or
`HUGGINGFACE_TOKEN`:

```bash
uv run python scripts/upload_hf.py \
  --folder dist/huggingface/dataset \
  --repo-type dataset \
  --repo-id <namespace>/modern-embedding-bench

uv run python scripts/upload_hf.py \
  --folder dist/huggingface/space \
  --repo-type space \
  --repo-id <namespace>/modern-embedding-bench-leaderboard \
  --space-dataset-repo-id <namespace>/modern-embedding-bench
```

Use `--private` during dry runs if you want to avoid publishing public artifacts.

## Compatibility CLI

The historical `mm-bench` command still exists for compatibility. New work
should prefer `modern-embed-bench benchmark run --manifest ...`.
