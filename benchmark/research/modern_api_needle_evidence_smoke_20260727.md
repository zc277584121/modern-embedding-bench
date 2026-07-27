# Modern API Needle Evidence Smoke - 2026-07-27

Status: **PASS**

Unique session: `meb-modern-embedding-leaderboard-1-1785146402-2-modern-api-needle-7f3c91a4d2b6`

Selected item: `leaderboard/modern-api-needle-evidence-smoke`

This run produced fresh, explicit `smoke` evidence for the current OpenAI and Voyage text
models. It used only `needle_in_haystack`, did not publish anything, and did not overwrite the
existing Hugging Face Dataset or Space snapshots.

## Startup Readiness

- Initial `git status --short`: clean.
- `OPENAI_API_KEY`: present.
- `VOYAGE_API_KEY`: present.
- Official current model-name checks passed for `text-embedding-3-small` and `voyage-4-lite`.
- The project environment did not have the optional OpenAI SDK linked. The exact locked
  `openai==2.26.0` and `jiter==0.13.0` wheels were already present in the uv cache, so the run used
  `uv run --no-sync` with those cached wheels on `PYTHONPATH`. No dependency, lockfile, or
  environment update was performed. The command below intentionally replaces the machine-local
  cache directories with redacted placeholders. It is not portable as written: another machine
  must resolve equivalent local roots for the same locked wheel versions before reproducing it.

## Exact Manifest

Tracked as `benchmark/runs/modern-api-needle-evidence-smoke.yaml`:

```yaml
id: modern-api-needle-evidence-smoke
description: Fresh needle-in-haystack smoke evidence for current OpenAI and Voyage text models.
evidence_tier: smoke
models:
  - openai-text-embedding-3-small
  - voyage-4-lite
tasks:
  - id: needle_in_haystack
    kwargs:
      haystack_lengths: [1000, 4000, 8000]
      needle_positions: [0.0, 0.5, 1.0]
      use_cache: false
metadata:
  tier: smoke
  scope: modern-api-needle-evidence
  expected_case_count_per_model: 90
  expected_provider_calls_per_model: 3
  expected_input_count_per_model: 109
  expected_cost: below-usd-1
```

The local dataset supplied 10 needles. Each model therefore evaluated
`10 needles x 3 lengths x 3 positions = 90` cases. Deduplication produced 10 query inputs,
90 documents with needles, and 9 documents without needles: 109 inputs per model.

## Provider Evidence

Git SHA recorded in both result rows: `55186a2d85de86f494847947d69e2f1d4c90ba34`.

| Evidence | OpenAI | Voyage |
|---|---:|---:|
| Registry model | `openai-text-embedding-3-small` | `voyage-4-lite` |
| Provider model | `text-embedding-3-small` | `voyage-4-lite` |
| Cases / skipped | 90 / 0 | 90 / 0 |
| Inputs / responses | 109 / 109 | 109 / 109 |
| Fresh provider calls | 3, cache disabled | 3, cache disabled |
| Dimensions | 1536 | 1024 |
| All finite | yes | yes |
| All unit norm within `1e-3` | yes | yes |
| Observed norm range | 0.9994981753 to 1.0004825746 | 0.9999998900 to 1.0000000897 |
| Provider latency | 7357.979 ms | 4667.493 ms |
| End-to-end duration | 8.046 s | 10.332 s |
| SDK token usage | 89,430 | 90,505 |
| Conservative billed cost at USD 0.02/M tokens | USD 0.00178860 | USD 0.00181010 |
| Error | none | none |

The actual combined token count was 179,935, for a conservative billed total of
USD 0.00359870. Independently, treating every UTF-8 input byte as one token gives a much looser
two-provider ceiling of 893,466 tokens and USD 0.01786932. Both are below the USD 1 limit. Voyage's
official pricing page also states that the first 200 million `voyage-4-lite` tokens are free per
account; the table above intentionally retains the conservative non-free calculation.

Per-call token usage and provider latency:

| Model | Call | Inputs | Responses | Tokens | Latency |
|---|---|---:|---:|---:|---:|
| OpenAI | queries | 10 | 10 | 129 | 3675.206 ms |
| OpenAI | documents with needle | 90 | 90 | 81,504 | 2143.330 ms |
| OpenAI | documents without needle | 9 | 9 | 7,797 | 1539.443 ms |
| Voyage | queries | 10 | 10 | 127 | 1325.951 ms |
| Voyage | documents with needle | 90 | 90 | 82,527 | 2651.650 ms |
| Voyage | documents without needle | 9 | 9 | 7,851 | 689.892 ms |

No retry, access, quota, billing, cardinality, finite-value, or provider-contract error occurred.

## Task Metrics

Both models recorded:

- `overall_accuracy`: 1.0
- `accuracy_len_1000`: 1.0
- `accuracy_len_4000`: 1.0
- `accuracy_len_8000`: 1.0
- `accuracy_pos_0pct`: 1.0
- `accuracy_pos_50pct`: 1.0
- `accuracy_pos_100pct`: 1.0
- `degradation_rate`: 0.0

## Local Hugging Face Dry-Run

Dedicated ignored artifacts:

- `results/modern-api-needle-evidence-smoke-20260727.jsonl`
- `results/modern-api-needle-combined-20260727.jsonl`
- `results/modern-api-needle-combined-leaderboard-20260727.csv`
- `dist/huggingface-modern-api-needle-smoke-20260727/dataset/`
- `dist/huggingface-modern-api-needle-smoke-20260727/space/`

The combined result is the unchanged 328-row legacy history followed by the two new smoke rows.
The dedicated dry-run produced:

| Check | Result |
|---|---:|
| Combined result rows | 330 |
| Public Dataset result rows | 294 |
| Public successful result rows | 248 |
| Public failed result rows | 46 |
| Dataset leaderboard rows | 241 |
| Space bundled leaderboard rows | 241 |
| New smoke leaderboard rows | 2 |
| Leaderboard task/model pairs | 62 |
| Duplicate task/model repeats retained | 179 |
| Latest task/model rows | 62 |
| Evidence tiers | 239 legacy, 2 smoke |

Both new rows have `evidence_tier=smoke`, duplicate count 1, run rank 1, and
`is_latest_for_task_model=true`. Dataset and Space leaderboard rows match exactly. The Space enables
latest-only mode when latest markers are available.

Validation also confirmed:

- Existing `dist/huggingface/dataset` and `dist/huggingface/space` snapshot hash was unchanged:
  `f3f58ac0fbf6a67bdd314b3fb9e315fec3aca9a557cabcd4dd5ba192226db3bd`.
- Failed rows remain in `results/latest.jsonl`; `results/latest-successful.jsonl` contains only the
  248 successful rows.
- Five unpublished/non-leaderboard tasks and two private model specs were excluded.
- No GeeVec/private marker appears in public results or leaderboard rows.
- No API key value, `/data2/`, `/home/`, or repository absolute path appears in the dedicated
  Dataset/Space export.

## Versions

- Python: 3.13.2
- uv: 0.11.26
- modern-embedding-bench: 0.1.0
- openai: 2.26.0
- jiter: 0.13.0
- voyageai: 0.3.7
- numpy: 2.4.3
- PyYAML: 6.0.3

## Primary Sources

- OpenAI embeddings guide and model table:
  https://developers.openai.com/api/docs/guides/embeddings
- Voyage embedding model table:
  https://docs.voyageai.com/docs/embeddings
- Voyage pricing:
  https://docs.voyageai.com/docs/pricing

## Commands

```bash
git status --short
for var_name in OPENAI_API_KEY VOYAGE_API_KEY; do if [ -n "${!var_name:-}" ]; then printf '%s present\n' "$var_name"; else printf '%s missing\n' "$var_name"; fi; done
PYTHONPATH=<UV_CACHE_OPENAI_2_26_0_ROOT>:<UV_CACHE_JITER_0_13_0_ROOT> uv run --no-sync python scripts/run_benchmark.py --manifest benchmark/runs/modern-api-needle-evidence-smoke.yaml --output results/modern-api-needle-evidence-smoke-20260727.jsonl --overwrite
jq -c '.' results/legacy-full-history.jsonl results/modern-api-needle-evidence-smoke-20260727.jsonl > results/modern-api-needle-combined-20260727.jsonl
uv run --no-sync python scripts/build_leaderboard.py --results results/modern-api-needle-combined-20260727.jsonl --output results/modern-api-needle-combined-leaderboard-20260727.csv
uv run --no-sync python scripts/export_hf_dataset.py --results results/modern-api-needle-combined-20260727.jsonl --leaderboard results/modern-api-needle-combined-leaderboard-20260727.csv --output-dir dist/huggingface-modern-api-needle-smoke-20260727/dataset
uv run --no-sync python scripts/export_hf_space.py --leaderboard dist/huggingface-modern-api-needle-smoke-20260727/dataset/leaderboards/latest.csv --output-dir dist/huggingface-modern-api-needle-smoke-20260727/space
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest tests/test_needle_in_haystack.py tests/test_benchmark_v2.py tests/test_provider_api_compat.py -q
git diff --check
```

Test result: `37 passed in 2.17s`.
