# Voyage 4 Nano Pinned Local Compatibility Smoke - 2026-07-21

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_5-1784640763-2_execute.md`

Selected item: `models/voyage-4-nano-local-smoke`

Unique Layer-1 session:
`meb-modern-embedding-leaderboard-5-1784640763-2-voyage4nano-6f2d91c8`

Status: **BLOCKED** before source review, snapshot download, or model loading.
The host resolver could not resolve `huggingface.co` for the anonymous pinned
download. No alternate IP, floating revision, provider API, or substitute model
was used.

## Scope And Conclusion

This run attempted only the bounded compatibility smoke for
`voyageai/voyage-4-nano` at the exact revision
`67fabc9bef010dabc5f6024aa1b1b6b93410426f`. It did not add a model registry
entry, run a benchmark matrix, create a leaderboard score, upload an artifact,
or call the Voyage API.

The result is **BLOCKED**, not a model compatibility failure. The first pinned
Hugging Face metadata request failed with `httpx.ConnectError: [Errno -2] Name
or service not known`. Because the custom source could not be fetched and
reviewed, the security gate correctly stopped the run before downloading model
weights or loading remote code. No inference evidence was produced.

## Readiness And Locked Environment

- Initial `git status --short`: clean.
- `/data2` available bytes: `308684763136`.
- GPU 0: `NVIDIA GeForce RTX 3080 Ti`, `12288 MiB` total, `11933 MiB` free.
- Dedicated cache before the run: absent.
- `VOYAGE_API_KEY`: present, but not read or used.
- `HF_TOKEN`: present, but explicitly unset for the download process.
- `HUGGINGFACE_HUB_TOKEN`: missing.
- `HUGGINGFACE_TOKEN`: missing.
- No secret value was printed or persisted.

The locked local environment was synchronized with:

```bash
uv sync --extra local --locked
```

Observed versions:

```text
sentence-transformers=5.2.3
transformers=5.3.0
torch=2.10.0
huggingface-hub=1.6.0
torch_cuda_available=True
torch_device_0=NVIDIA GeForce RTX 3080 Ti
```

The dependency files were unchanged across synchronization:

```text
bd836ed6be7771f511fb583b6fbffd6c8284286b90933945440d1bcf1ea07df5  pyproject.toml
922d707935f6675d78f49ad2312cfa3de990a87d74d7dae60444b31f101c79de  uv.lock
```

## Pinned Source-Review Attempt

The source-review stage used anonymous Hugging Face access, disabled implicit
tokens, and requested only `modeling_qwen3_bidirectional.py` and `config.json`:

```bash
env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HUGGINGFACE_TOKEN \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  MODEL_ID='voyageai/voyage-4-nano' \
  MODEL_REVISION='67fabc9bef010dabc5f6024aa1b1b6b93410426f' \
  MEB_SMOKE_CACHE='/data2/hf-cache/modern-embedding-bench/local-openweight-20260721/voyage-4-nano-67fabc9' \
  HF_HOME='/data2/hf-cache/modern-embedding-bench/local-openweight-20260721/voyage-4-nano-67fabc9/hf-home' \
  HF_HUB_CACHE='/data2/hf-cache/modern-embedding-bench/local-openweight-20260721/voyage-4-nano-67fabc9/hf-home/hub' \
  uv run --extra local --locked python - <<'PY'
from __future__ import annotations

import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
    cache_dir=os.environ["HF_HUB_CACHE"],
    local_dir=os.path.join(os.environ["MEB_SMOKE_CACHE"], "review"),
    allow_patterns=["modeling_qwen3_bidirectional.py", "config.json"],
    token=False,
)
print(f"review_path={path}")
PY
```

Observed failure:

```text
httpx.ConnectError: [Errno -2] Name or service not known
huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying
to locate the files on the Hub and the pinned snapshot was not present locally.
```

A subsequent `getent hosts huggingface.co` produced no output. The failure
occurred during the initial repository metadata lookup, before either requested
file was written.

Source-review result: **not performed because the pinned files were
unavailable**. Consequently there is no custom-source SHA-256, import summary,
or runtime side-effect assessment from this run. The previously accepted
research description was not treated as a substitute for reviewing the bytes
that would actually execute.

## Download And Cache Gates

- Expected selected-snapshot ceiling: `850000000` bytes.
- Hard dedicated-cache stop gate: `1000000000` bytes.
- Selected snapshot bytes downloaded: `0`.
- Dedicated-cache disk usage after the failed request: `0` bytes; path absent.
- Downloaded file inventory: empty.
- Unexpected artifacts: none.
- Model weights downloaded: no.

Because the request failed before creating the dedicated path, both byte gates
were trivially respected. No shared Hugging Face cache was inspected, modified,
or deleted.

## Inference Evidence

Inference did not run because source review is a mandatory pre-load gate.
Therefore device placement beyond CUDA readiness, model dtype, GPU peak memory,
load latency, inference latency, query/document shapes, finiteness, post-MRL
norms, specialized method availability, prompt configuration, similarity
scores, and Mars-document ranking are all **not available**.

The intended bounded workload remained one provider-internal `"test"` encode,
one retrieval query, and three retrieval documents at 512 dimensions on
`cuda:0`; none of those inputs reached the model.

## Cleanup

The exact dedicated-prefix guard was applied:

```bash
MEB_SMOKE_CACHE='/data2/hf-cache/modern-embedding-bench/local-openweight-20260721/voyage-4-nano-67fabc9'
case "$MEB_SMOKE_CACHE" in
  /data2/hf-cache/modern-embedding-bench/local-openweight-20260721/*)
    rm -rf -- "$MEB_SMOKE_CACHE"
    ;;
  *)
    echo 'Refusing to remove a non-dedicated cache path' >&2
    exit 2
    ;;
esac
```

Observed cleanup result: `cleanup=success`; the dedicated path was verified
absent. No shared cache was deleted.

## Validation

The focused no-network provider compatibility suite passed under the locked
project interpreter:

```bash
uv run --extra local --locked python -m pytest -q tests/test_provider_api_compat.py
```

Result: `11 passed in 0.08s`.

The shorter `uv run ... pytest` form was not used as evidence because this
shell resolved that executable from an unrelated legacy Conda environment;
the explicit `python -m pytest` invocation above used the locked project
interpreter. Final repository checks were `git diff --check` and
`git status --short`.

## Limitations And Bounded Follow-Up

This note is operational compatibility evidence only and contains no model
quality evidence. A bounded retry is appropriate only after normal DNS access
to `huggingface.co` works from the host. The retry must use the same exact
revision, anonymous token-disabled download, allow-list, cache prefix, source
review, byte gates, five-input provider workload, and cleanup guard. It must not
use a hard-coded alternate IP, the Voyage API, a floating branch, or a different
model as a workaround for this item.
