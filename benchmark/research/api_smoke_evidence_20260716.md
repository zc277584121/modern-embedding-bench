# API Smoke Evidence Closeout - 2026-07-16

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_1-1784204551-2_execute`

This note reconciles ignored local API smoke outputs for provider status triage. It is not public
leaderboard evidence.

## Evidence Sources

- `results/api-coverage-smoke-20260716-layer1.jsonl` - ignored local smoke output from commit `51792db`.
- `results/api-coverage-smoke-20260716-layer1-dashscope.jsonl` - ignored local DashScope remainder output from commit `51792db`.
- `results/openai-smoke.jsonl` - ignored local OpenAI plumbing smoke output from commit `e53f6f0`.
- `results/jina-api-smoke-20260716-closeout.jsonl` - ignored local Jina-only closeout output from commit `efdbe42`.
- `.perpetuum/modern-embedding-leaderboard/context.md` - local operational snapshot that previously recorded Jina v5/v4 success.

## Environment Gate

Only variable names and present/missing status were printed:

- `OPENAI_API_KEY`: present
- `VOYAGE_API_KEY`: present
- `GEMINI_API_KEY`: present
- `JINA_API_KEY`: present
- `DASHSCOPE_API_KEY`: present
- `COHERE_API_KEY`: present
- `ARK_API_KEY`: present

No secret values were printed.

## Local Result Summary

| Source | Model | Run id | Task | Status | Duration | Token usage | Error |
|---|---|---:|---|---|---:|---:|---|
| `api-coverage-smoke-20260716-layer1.jsonl` | `openai-text-embedding-3-small` | `api-coverage-smoke` | `mrl_stress` | usable | 3.132s | n/a | n/a |
| `api-coverage-smoke-20260716-layer1.jsonl` | `voyage-4-lite` | `api-coverage-smoke` | `mrl_stress` | usable | 4.481s | n/a | n/a |
| `api-coverage-smoke-20260716-layer1.jsonl` | `gemini-embedding-2` | `api-coverage-smoke` | `mrl_stress` | usable | 2.355s | n/a | n/a |
| `api-coverage-smoke-20260716-layer1.jsonl` | `jina-embeddings-v5-text-nano` | `api-coverage-smoke` | `mrl_stress` | access_recheck | 120.193s | n/a | `[Errno 101] Network is unreachable` |
| `api-coverage-smoke-20260716-layer1-dashscope.jsonl` | `dashscope-text-embedding-v4` | `api-coverage-smoke-dashscope-remainder` | `mrl_stress` | usable | 2.575s | n/a | n/a |
| `api-coverage-smoke-20260716-layer1-dashscope.jsonl` | `dashscope-qwen3-vl-embedding` | `api-coverage-smoke-dashscope-remainder` | `mrl_stress` | usable | 6.603s | n/a | n/a |
| `openai-smoke.jsonl` | `openai-text-embedding-3-large` | `openai-smoke` | `mrl_stress` | usable | 0.954s | n/a | n/a |
| `openai-smoke.jsonl` | `openai-text-embedding-3-large` | `openai-smoke` | `needle_in_haystack` | usable | 0.004s | n/a | n/a |
| `jina-api-smoke-20260716-closeout.jsonl` | `jina-embeddings-v5-text-nano` | `jina-api-smoke-closeout` | `mrl_stress` | access_recheck | 121.230s | n/a | `[Errno 101] Network is unreachable` |
| `jina-api-smoke-20260716-closeout.jsonl` | `jina-embeddings-v4` | `jina-api-smoke-closeout` | `mrl_stress` | usable | 2.447s | n/a | n/a |

## Provider Status

- `openai-text-embedding-3-small`: usable for local tiny API smoke evidence.
- `openai-text-embedding-3-large`: usable for local plumbing smoke evidence.
- `voyage-4-lite`: usable for local tiny API smoke evidence.
- `gemini-embedding-2`: usable for local tiny API smoke evidence.
- `dashscope-text-embedding-v4`: usable for local tiny API smoke evidence.
- `dashscope-qwen3-vl-embedding`: usable for local tiny API smoke evidence on text input through the multimodal endpoint.
- `jina-embeddings-v4`: usable for local tiny API smoke evidence.
- `jina-embeddings-v5-text-nano`: access_recheck. The closeout rerun reproduced the earlier transport failure with `[Errno 101] Network is unreachable`, while Jina v4 succeeded in the same Jina-only command. Treat this as model/path-specific or transient network/access state until rechecked from another network path.
- `cohere-embed-v4` and `cohere-embed-multilingual-v3`: blocked/access_recheck from existing registry/context due recent 403 access errors; not rerun here.
- `ark-doubao-embedding-large-text-250515` and `volcengine-doubao-embedding-vision-251215`: not evaluated here; the current smoke manifests exclude them from this MRL closeout path.

## Unsuitable For Public Leaderboard Evidence

- All files above are ignored local smoke outputs, not committed benchmark artifacts.
- The closeout Jina run used only two STS pairs to minimize API scope; it is an access check, not quality evidence.
- `jina-embeddings-v5-text-nano` still has no successful local closeout row in the inspected evidence.
- The Jina v4 closeout row contains `NaN` retention values because the sample is intentionally tiny; it should not be promoted to public scoring.
- These rows should remain provider-status evidence until a public evidence tier and reproducible run path intentionally include them.

## Commands Run

```bash
git status --short
sed -n '1,260p' .perpetuum/modern-embedding-leaderboard/plan.md
sed -n '1,260p' .perpetuum/modern-embedding-leaderboard/context.md
sed -n '1,560p' benchmark/models/core.yaml
sed -n '1,220p' benchmark/runs/api-coverage-smoke.yaml
sed -n '1,220p' benchmark/runs/api-modern-smoke.yaml
sed -n '1,260p' src/mm_embed/providers/jina_provider.py
sed -n '1,220p' src/mm_embed/providers/openai_provider.py
sed -n '1,240p' src/mm_embed/providers/voyage_provider.py
sed -n '1,260p' src/mm_embed/providers/gemini_provider.py
sed -n '1,300p' src/mm_embed/providers/dashscope_provider.py
sed -n '1,260p' src/mm_embed/tasks/mrl_stress.py
sed -n '1,320p' src/mm_embed/benchmark/runner.py
sed -n '1,260p' tests/test_provider_api_compat.py
sed -n '1,700p' tests/test_benchmark_v2.py
for name in OPENAI_API_KEY VOYAGE_API_KEY GEMINI_API_KEY JINA_API_KEY DASHSCOPE_API_KEY COHERE_API_KEY ARK_API_KEY; do if [ -n "${!name+x}" ] && [ -n "${!name}" ]; then printf '%s=present\n' "$name"; else printf '%s=missing\n' "$name"; fi; done
wc -l results/api-coverage-smoke-20260716-layer1.jsonl results/api-coverage-smoke-20260716-layer1-dashscope.jsonl results/openai-smoke.jsonl
head -n 2 results/api-coverage-smoke-20260716-layer1.jsonl
head -n 2 results/api-coverage-smoke-20260716-layer1-dashscope.jsonl
head -n 2 results/openai-smoke.jsonl
jq -s -r '.[] | [.model.id, .provider_result.provider, .provider_result.model_name, .task.id, (.run.id // ""), (.timestamps.duration_s|tostring), (if .error then "ERROR" else "PASS" end), (.provider_result.token_usage // .provider_result.total_tokens // .metrics.token_usage // ""), (.error // "")] | @tsv' results/api-coverage-smoke-20260716-layer1.jsonl results/api-coverage-smoke-20260716-layer1-dashscope.jsonl results/openai-smoke.jsonl
git check-ignore -v results/api-coverage-smoke-20260716-layer1.jsonl results/api-coverage-smoke-20260716-layer1-dashscope.jsonl results/openai-smoke.jsonl results/jina-api-smoke-20260716-closeout.jsonl || true
timeout 360 uv run python scripts/run_benchmark.py --manifest results/jina-api-smoke-20260716-closeout.yaml --output results/jina-api-smoke-20260716-closeout.jsonl --overwrite
jq -s -r '.[] | [.model.id, .provider_result.provider, .provider_result.model_name, .task.id, (.run.id // ""), (.timestamps.duration_s|tostring), (if .error then "ERROR" else "PASS" end), (.provider_result.token_usage // .provider_result.total_tokens // .metrics.token_usage // ""), (.error // "")] | @tsv' results/jina-api-smoke-20260716-closeout.jsonl
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest tests/test_benchmark_v2.py tests/test_provider_api_compat.py -q
git diff --check
git status --short --ignored=matching benchmark/research/api_smoke_evidence_20260716.md results/jina-api-smoke-20260716-closeout.yaml results/jina-api-smoke-20260716-closeout.jsonl
git ls-files --others --exclude-standard benchmark/research/api_smoke_evidence_20260716.md
rg -n --hidden "1784204551|dispatch_1-1784204551-2_execute|--replace|delegate" .cc-use .perpetuum/modern-embedding-leaderboard -S
```

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest tests/test_benchmark_v2.py tests/test_provider_api_compat.py -q` - 14 passed.
- `git diff --check` - passed.

## Source Links Used

Source links were inspected from the local registry, not freshly fetched:

- https://platform.openai.com/docs/guides/embeddings
- https://docs.voyageai.com/docs/embeddings
- https://ai.google.dev/gemini-api/docs/embeddings
- https://jina.ai/models/jina-embeddings-v5-text-nano/
- https://jina.ai/models/jina-embeddings-v4/
- https://www.alibabacloud.com/help/en/model-studio/embedding
- https://docs.cohere.com/docs/cohere-embed
- https://www.volcengine.com/docs/82379/1330310
- https://www.volcengine.com/docs/82379/1409291

## Recommendation

Do not change provider code or registry status in this closeout. The evidence supports accepting
OpenAI, Voyage, Gemini, DashScope, and Jina v4 as locally usable smoke paths, while keeping Jina v5
nano in access recheck until a successful rerun occurs from a known-good network path. Keep these
rows out of public leaderboard scoring unless a future run creates reproducible, committed evidence
with a suitable public evidence tier.
