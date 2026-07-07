# Benchmark v2

This directory is the data-backed control plane for the benchmark.

- `models/*.yaml` lists reviewable model specs. Adding a model should usually be a YAML change first.
- `tasks/*.yaml` defines task metadata, default kwargs, dataset versions, and primary leaderboard metrics.
- `runs/*.yaml` defines concrete evaluation manifests.

The Python runner reuses the provider and task implementations under `src/mm_embed/`.

```bash
uv run modern-embed-bench benchmark models
uv run modern-embed-bench benchmark tasks
uv run modern-embed-bench benchmark run --manifest benchmark/runs/openai-smoke.yaml --output results/openai-smoke.jsonl
uv run modern-embed-bench benchmark leaderboard --results results/openai-smoke.jsonl --output results/openai-smoke-leaderboard.csv
```

Generated results remain outside the tracked source tree by default. Use
`scripts/import_legacy_results.py` to convert older `results/*.json` artifacts
into the v2 JSONL shape when comparing against historical runs.
