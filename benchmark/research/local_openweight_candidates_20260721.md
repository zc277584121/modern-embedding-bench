# Local Open-Weight Embedding Candidates - 2026-07-21

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_4-1784637310-2_execute.md`

Selected item: `models/local-openweight-small`

Status: research complete; no model was loaded or downloaded. Recommend one
bounded compatibility smoke for `voyageai/voyage-4-nano` before adding any
registry entry or treating any result as leaderboard evidence.

## Decision

Ranking for the next local compatibility smoke:

1. **`voyageai/voyage-4-nano` - recommended first.** It has the smallest core
   weight file by a clear margin (692,919,112 bytes), explicit query and
   document prompts, native `encode_query()` / `encode_document()` examples,
   four documented MRL dimensions, and documented output quantization. Its
   custom bidirectional Qwen implementation requires `trust_remote_code=True`,
   so the smoke must pin and inspect the exact revision before execution.
2. **`Qwen/Qwen3-Embedding-0.6B` - second.** It is the lower-risk fallback:
   Apache-2.0, no repository custom Python, mature Sentence Transformers
   support, and a 1.19 GB core weight that easily fits this host. It is second
   because it is roughly 1.7x the Voyage weight and does not document native
   output quantization for this base checkpoint; it is the immediate fallback
   if the Voyage remote-code or Transformers compatibility check fails.
3. **`codefuse-ai/ML-Embed-0.6B` - third.** Its 3D Matryoshka features are
   promising and compatibility mode should fit the host, but it is newer, adds
   two optional pickle-based factorization artifacts, and has an unresolved
   discrepancy between the manual query wrapper and the shorter prompt stored
   in `config_sentence_transformers.json`. It should follow only after the
   simpler routing contract is proven with one of the first two candidates.

All three normal compatibility paths are comfortably below the requested
approximately 5 GB download preference. The ranking is about bounded execution
risk and repo-contract fit, not a claim about retrieval quality.

## Current Repo And Host Contract

- `src/mm_embed/providers/sentence_transformers_provider.py` routes
  `retrieval_query` through `encode_query()` and `retrieval_document` through
  `encode_document()` when those methods are callable; otherwise it uses
  `encode()`. It normalizes embeddings and truncates plus re-normalizes when a
  smaller dimension is requested.
- The provider currently constructs `SentenceTransformer(...,
  trust_remote_code=True)` for every local model and does not expose revision,
  dtype, attention implementation, or cache arguments. A pinned future smoke
  should therefore download an allow-listed snapshot first and pass its local
  path to the provider.
- `tests/test_provider_api_compat.py` already supplies no-network regressions
  for query/document method selection, generic fallback, error propagation,
  and normalized truncation. The proposed smoke is for real dependency/model
  compatibility, not a replacement for those tests.
- `pyproject.toml` declares `local = ["sentence-transformers>=3.0",
  "torch>=2.0"]` and a base dependency of `transformers>=4.57.0`.
- Read-only `uv run --no-sync` inspection found the current environment has
  `torch==2.10.0`, `transformers==5.3.0`, and `huggingface-hub==1.6.0`, while
  `sentence-transformers` is missing. `uv.lock` resolves the local extra to
  `sentence-transformers==5.2.3`. `accelerate` and `flash-attn` are also
  missing.
- The host currently reports four NVIDIA GeForce RTX 3080 Ti GPUs, each with
  12,288 MiB total VRAM and compute capability 8.6. `/data2` reports about
  288 GB free. The smoke should use only `cuda:0`; multi-GPU loading would add
  complexity without helping these sub-1B models.
- Existing local entries remain in `benchmark/models/core.yaml`; this research
  does not propose a YAML change until a separate dispatch reviews smoke
  evidence.

## Evidence Method And Limits

The Hugging Face host could not resolve from the project shell, so current
official pages, raw text files, Xet/LFS pointers, and commit/tree pages were
retrieved through AnySearch. An initial retrieval inherited an existing
`ANYSEARCH_API_KEY` environment variable before that was detected; no secret
value was printed or persisted. Every fact cited in this note was then
re-fetched with `ANYSEARCH_API_KEY` explicitly unset. No Hugging Face token or
embedding-provider key was used. Anonymous access reached each pinned
README/config and weight pointer, which is direct evidence that all three
repositories are public and ungated at the time of research.

The Hugging Face JSON metadata endpoint could not be consumed through that
path, so exact repository-wide `usedStorage` values remain unresolved. Exact
core-weight bytes come from the repositories' raw Xet/LFS pointer files;
repository totals below distinguish those weights from visible tokenizer,
configuration, and optional artifacts rather than inventing a total.

## Candidate 1: `voyageai/voyage-4-nano`

### Pinned identity, access, license, and size

- Current `main` revision observed on 2026-07-21:
  [`67fabc9bef010dabc5f6024aa1b1b6b93410426f`](https://huggingface.co/voyageai/voyage-4-nano/commit/67fabc9bef010dabc5f6024aa1b1b6b93410426f).
  It was obtained from both the official
  [commit history](https://huggingface.co/voyageai/voyage-4-nano/commits/main)
  and [tree page](https://huggingface.co/voyageai/voyage-4-nano/tree/main), not
  inferred from a floating `main` download.
- License: Apache-2.0 in the pinned
  [model card](https://huggingface.co/voyageai/voyage-4-nano/blob/67fabc9bef010dabc5f6024aa1b1b6b93410426f/README.md)
  and repository `LICENSE.txt`. Access: public and ungated; pinned raw files
  were readable without Hugging Face authentication.
- Source-reported parameters: 180M non-embedding plus 160M embedding parameters
  (340M total); the Hugging Face UI rounds this to 0.3B BF16 parameters.
- Core `model.safetensors`: exactly **692,919,112 bytes** (about 661 MiB),
  SHA-256 `3dae0c63c81dcab79ac213af331940a1bf2b8a53ec8646be878552890291ad30`,
  from the pinned [raw pointer](https://huggingface.co/voyageai/voyage-4-nano/raw/67fabc9bef010dabc5f6024aa1b1b6b93410426f/model.safetensors).
- The tree additionally reports `tokenizer.json` at 7.03 MB, `merges.txt` at
  1.67 MB, `vocab.json` at 2.78 MB, plus small configs, custom Python, license,
  and docs. Exact total repository bytes are unresolved. The allow-listed smoke
  should stay below an **0.85 GB expected download ceiling**; stop and inspect
  if the dedicated cache exceeds 1.0 GB.

### Encoding contract and dependencies

- Default dimension 2048; documented MRL dimensions are 2048, 1024, 512, and
  256.
- Documented output precisions are `float32`, `int8`, `uint8`, `binary`, and
  `ubinary`, enabled by quantization-aware training. This is output-vector
  quantization, not a promise that the BF16 model weights are quantized.
- The card reports a 32,000-token context. `sentence_bert_config.json` sets
  `max_seq_length` to 32,768, while the base config declares
  `max_position_embeddings=40,960`. Treat **32,000/32,768 as the supported
  application ceiling** and do not infer that 40,960 has been validated.
- Pooling is attention-mask mean pooling, includes prompt tokens, then
  normalization. The custom model projects hidden size 1024 to output size
  2048 before pooling.
- Query prompt: `Represent the query for retrieving supporting documents: `.
  Document prompt: `Represent the document for retrieval: `.
  The pinned Sentence Transformers config exposes them under `query` and
  `document`, matching the repo provider's specialized routing.
- `trust_remote_code=True` is required because `config.json` maps `AutoModel`
  to `modeling_qwen3_bidirectional.Qwen3BidirectionalModel`. The pinned
  [custom source](https://huggingface.co/voyageai/voyage-4-nano/blob/67fabc9bef010dabc5f6024aa1b1b6b93410426f/modeling_qwen3_bidirectional.py)
  changes Qwen3 attention to bidirectional and adds a 2048-dimensional linear
  projection. The file should be re-reviewed if the revision changes.
- The snapshot records Sentence Transformers 5.0.0 and Transformers 4.51.3 in
  its generation metadata, but its current custom Python imports Transformers
  masking and kwargs APIs also present in the repo's locked Transformers 5.3.0.
  This cross-version surface is the main reason to run a compatibility smoke.
- The model card recommends Flash Attention 2 for speed but explicitly allows
  eager or SDPA. Because `flash-attn` is absent, the first smoke should accept
  the installed SDPA/eager path and must not install Flash Attention.

### Host fit

The 661 MiB BF16 weight is a comfortable single-GPU fit. Even a conservative
FP32 materialization is roughly 1.3 GiB of raw parameters before runtime
overhead, leaving ample room in 12 GB VRAM for five very short inputs. Use
`cuda:0`; do not test the 32k context in this compatibility smoke.

## Candidate 2: `Qwen/Qwen3-Embedding-0.6B`

### Pinned identity, access, license, and size

- Current `main` revision observed on 2026-07-21:
  [`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/commit/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3),
  obtained from the official
  [commit history](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/commits/main)
  and [tree page](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/main).
- License: Apache-2.0 in the pinned
  [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md)
  and the official [release post](https://qwenlm.github.io/blog/qwen3-embedding/).
  Access: public and ungated; pinned raw files were anonymously readable.
- Source-reported scale: 0.6B parameters, 28 layers, BF16 checkpoint.
- Core `model.safetensors`: exactly **1,191,586,416 bytes** (about 1.11 GiB),
  SHA-256 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`,
  from the pinned [raw pointer](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/model.safetensors).
- The tree additionally reports `tokenizer.json` at 11.4 MB, `merges.txt` at
  1.67 MB, `vocab.json` at 2.78 MB, and small configs. Exact repository total
  is unresolved. Expected allow-listed smoke download ceiling: **1.5 GB**.

### Encoding contract and dependencies

- Native dimension 1024; the card documents user-defined MRL output dimensions
  from 32 through 1024. The first smoke should use 512 dimensions to exercise
  the provider's truncation and re-normalization path.
- The base repository does not document native weight or output quantization.
  Community quantized derivatives are not evidence for this pinned base smoke.
- Supported context is 32k; `config.json` declares 32,768 positions. Keep the
  compatibility inputs short.
- Pooling is the last token/EOS representation with normalization. Query inputs
  receive the configured instruction; retrieval documents receive no prefix.
- Stored query prompt:
  `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`.
  Stored document prompt is empty. The provider's `encode_query()` and
  `encode_document()` routes therefore align with the model's intended
  asymmetry.
- The card requires `transformers>=4.51.0` and
  `sentence-transformers>=2.7.0`; the locked future environment
  (`transformers==5.3.0`, `sentence-transformers==5.2.3`) satisfies those lower
  bounds. No repository custom Python or `trust_remote_code` is required by
  the model itself, although the current provider still passes the flag.

### Host fit

The 1.11 GiB BF16 weight is an easy single-3080-Ti fit. A conservative FP32
materialization is around 2.4 GB before runtime overhead, still suitable for a
five-input smoke on 12 GB VRAM. This is the safest fallback if the recommended
Voyage custom-code path fails before inference.

## Candidate 3: `codefuse-ai/ML-Embed-0.6B`

### Pinned identity, access, license, and size

- Current `main` revision observed on 2026-07-21:
  [`fb82458ca9732a15a9526df89df84d5718efe89f`](https://huggingface.co/codefuse-ai/ML-Embed-0.6B/commit/fb82458ca9732a15a9526df89df84d5718efe89f),
  obtained from the official
  [commit history](https://huggingface.co/codefuse-ai/ML-Embed-0.6B/commits/main)
  and [tree page](https://huggingface.co/codefuse-ai/ML-Embed-0.6B/tree/main).
- License: Apache-2.0 in the pinned
  [model card](https://huggingface.co/codefuse-ai/ML-Embed-0.6B/blob/fb82458ca9732a15a9526df89df84d5718efe89f/README.md).
  Access: public and ungated; pinned raw files were anonymously readable.
- Source-reported scale: Qwen3-0.6B base, 28 layers, BF16 checkpoint.
- Core `model.safetensors`: exactly **1,192,133,232 bytes** (about 1.11 GiB),
  SHA-256 `ca85b81bd1a00aeaad1d11801c56b2faf6d51d19e4df6f0633d4818fb94d177e`,
  from the pinned [raw pointer](https://huggingface.co/codefuse-ai/ML-Embed-0.6B/raw/fb82458ca9732a15a9526df89df84d5718efe89f/model.safetensors).
- Optional MEL factorization artifacts are `U.pth` at exactly 155,583,486
  bytes and `V.pth` at 1,049,598 bytes. Together with the core weight, those
  three weight artifacts total **1,348,766,316 bytes**, before tokenizer and
  config files. Both `.pth` files contain pickle imports. They are unnecessary
  for compatibility mode and must be excluded from the first smoke.
- Exact repository total is unresolved. The allow-listed compatibility smoke,
  excluding `U.pth` and `V.pth`, should remain below **1.5 GB**; a future MEL
  smoke would have a separate approximately 1.7 GB ceiling and security review.

### Encoding contract and dependencies

- Native dimension 1024 with last-token/EOS pooling and normalization.
- MRL supports prefix truncation; the card reports training down to a smallest
  Matryoshka dimension of 8 and shows `truncate_dim=512`. Normalize after
  truncation. MLL can reduce the 28-layer depth and MEL can factorize the token
  embedding matrix, but neither should be enabled in the first compatibility
  smoke.
- No native quantization behavior is documented. MRL, layer truncation, and
  factorization are compression mechanisms but must not be mislabeled as
  quantization.
- `config.json` declares `max_position_embeddings=40,960`; the card does not
  make an equally explicit supported-context claim. Treat **40,960 as a config
  value, not a validated application limit**, and keep the smoke short.
- Manual Transformers usage wraps retrieval queries as
  `Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: `
  and leaves documents unprefixed. However, the pinned Sentence Transformers
  config stores only `Given a question, retrieve passages that can help answer
  the question.` under `query` and an empty `document` value. The exact
  automatic serialized query string is therefore unresolved until a pinned
  Sentence Transformers smoke observes behavior. This discrepancy is the main
  reason for third place.
- The default release is described as standard Qwen3 compatibility mode and
  does not require repository custom Python. Its config records Transformers
  4.51.0. The card warns that Transformers 5 layer-truncation experiments also
  require truncating `layer_types`; this is irrelevant to the first full-depth
  smoke but is a future MLL compatibility risk.

### Host fit

Full compatibility mode is an easy single-GPU fit, similar to Qwen3-Embedding.
Do not load `U.pth` or `V.pth`, edit layer counts, or test factorized embeddings
in the first smoke. Those are separate experimental surfaces with different
correctness and security checks.

## Exact Future Smoke Procedure

This procedure is proposed for a later, explicitly authorized dispatch. It was
not run during this research task.

### 1. Install the already-locked local extra

```bash
uv sync --extra local --locked
```

This should install the lock-resolved `sentence-transformers==5.2.3` without
changing `pyproject.toml` or `uv.lock`. Stop if `uv` proposes a lock update.
Do not add Flash Attention for this smoke.

### 2. Select exactly one pinned candidate

Recommended first smoke:

```bash
export MODEL_ID='voyageai/voyage-4-nano'
export MODEL_REVISION='67fabc9bef010dabc5f6024aa1b1b6b93410426f'
export MODEL_SLUG='voyage-4-nano-67fabc9'
export EXPECTED_DOWNLOAD_BYTES=850000000
```

Second-place fallback:

```bash
export MODEL_ID='Qwen/Qwen3-Embedding-0.6B'
export MODEL_REVISION='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
export MODEL_SLUG='qwen3-embedding-0.6b-97b0c61'
export EXPECTED_DOWNLOAD_BYTES=1500000000
```

Third-place candidate:

```bash
export MODEL_ID='codefuse-ai/ML-Embed-0.6B'
export MODEL_REVISION='fb82458ca9732a15a9526df89df84d5718efe89f'
export MODEL_SLUG='ml-embed-0.6b-fb82458'
export EXPECTED_DOWNLOAD_BYTES=1500000000
```

Then set a dedicated cache that cannot collide with an existing shared cache:

```bash
export MEB_SMOKE_CACHE="/data2/hf-cache/modern-embedding-bench/local-openweight-20260721/${MODEL_SLUG}"
export HF_HOME="${MEB_SMOKE_CACHE}/hf-home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export CUDA_VISIBLE_DEVICES=0
```

For the Voyage candidate only, inspect the pinned custom source again before
running Python:

```bash
uv run --extra local --locked python - <<'PY'
from __future__ import annotations

import os

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
    cache_dir=os.environ["HF_HUB_CACHE"],
    local_dir=os.path.join(os.environ["MEB_SMOKE_CACHE"], "review"),
    allow_patterns=["modeling_qwen3_bidirectional.py", "config.json"],
)
PY
sed -n '1,260p' "${MEB_SMOKE_CACHE}/review/modeling_qwen3_bidirectional.py"
```

Stop before model loading if the source differs from the reviewed revision or
introduces imports or side effects that have not been assessed. Do not fall
back to floating `main`.

### 3. Run the common five-input provider smoke

The provider performs one internal `"test"` encode during lazy loading, then
the script sends one retrieval query and three documents: five short inputs in
total. It uses `cuda:0`; the repositories declare BF16 weights, but the current
provider does not expose a dtype knob, so the script records the actual dtype.
BF16 is expected. FP32 is a warning if VRAM remains safe, not a leaderboard or
quality failure.

```bash
uv run --extra local --locked python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from mm_embed.providers.sentence_transformers_provider import SentenceTransformersProvider


model_id = os.environ["MODEL_ID"]
revision = os.environ["MODEL_REVISION"]
hub_cache = Path(os.environ["HF_HUB_CACHE"])
expected_download_bytes = int(os.environ["EXPECTED_DOWNLOAD_BYTES"])

allow_patterns = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
    "modeling_*.py",
    "1_Pooling/*",
    "2_Normalize/*",
]

snapshot_path = snapshot_download(
    repo_id=model_id,
    revision=revision,
    cache_dir=hub_cache,
    allow_patterns=allow_patterns,
)

snapshot_bytes = sum(
    path.stat().st_size
    for path in Path(snapshot_path).rglob("*")
    if path.is_file()
)
if snapshot_bytes > expected_download_bytes:
    raise RuntimeError(
        f"Download ceiling exceeded: {snapshot_bytes} > {expected_download_bytes}"
    )

provider = SentenceTransformersProvider(model=snapshot_path, device="cuda:0")

query = "Which planet is known as the Red Planet?"
documents = [
    "Mars is called the Red Planet because iron minerals in its soil oxidize.",
    "Venus has a dense carbon-dioxide atmosphere and sulfuric-acid clouds.",
    "Jupiter is the largest planet and has a long-lived Great Red Spot.",
]

query_result = provider.embed_text(
    [query],
    task_type="retrieval_query",
    dimensions=512,
)
document_result = provider.embed_text(
    documents,
    task_type="retrieval_document",
    dimensions=512,
)

assert provider.device == "cuda:0"
assert callable(getattr(provider._st_model, "encode_query", None))
assert callable(getattr(provider._st_model, "encode_document", None))
assert query_result.embeddings.shape == (1, 512)
assert document_result.embeddings.shape == (3, 512)
assert np.isfinite(query_result.embeddings).all()
assert np.isfinite(document_result.embeddings).all()
np.testing.assert_allclose(
    np.linalg.norm(query_result.embeddings, axis=1),
    np.ones(1),
    atol=1e-3,
)
np.testing.assert_allclose(
    np.linalg.norm(document_result.embeddings, axis=1),
    np.ones(3),
    atol=1e-3,
)

scores = query_result.embeddings @ document_result.embeddings.T
if int(np.argmax(scores[0])) != 0:
    raise AssertionError(f"Semantic sanity check failed: scores={scores.tolist()}")

first_float_parameter = next(
    parameter for parameter in provider._st_model.parameters() if parameter.is_floating_point()
)
assert first_float_parameter.device.type == "cuda"
if first_float_parameter.dtype != torch.bfloat16:
    print(f"WARN dtype={first_float_parameter.dtype}; repository metadata declares BF16")

print(
    {
        "status": "PASS",
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": snapshot_path,
        "snapshot_bytes": snapshot_bytes,
        "dtype": str(first_float_parameter.dtype),
        "query_shape": query_result.embeddings.shape,
        "document_shape": document_result.embeddings.shape,
        "scores": scores.tolist(),
    }
)
PY
```

Success means all of the following:

- the exact pinned snapshot stays inside its candidate-specific download
  ceiling;
- one model loads on `cuda:0` without installing extra dependencies or using a
  provider API;
- both specialized query/document calls are available and complete;
- outputs have shapes `(1, 512)` and `(3, 512)`, contain finite values, and are
  normalized after MRL truncation;
- the obvious Mars document ranks first in this four-user-input sanity set;
- actual device, dtype, revision, snapshot path, byte count, and scores are
  recorded.

Failure means load/import/prompt routing, CUDA OOM, non-finite output, wrong
shape, missing specialized methods, normalization failure, semantic sanity
failure, or download-ceiling violation. For Voyage, any unexpected custom-code
diff is a stop-before-execution failure. For CodeFuse, any attempt to fetch
`U.pth` or `V.pth` is a scope failure for this compatibility-mode smoke.

### 4. Cleanup only the dedicated cache

After evidence has been captured and only if the path matches this smoke's
dedicated prefix:

```bash
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

Do not delete `~/.cache/huggingface`, another user's cache, or any pre-existing
shared cache. No cache was deleted during this research dispatch.

## Why The Smoke Is Not Leaderboard Evidence

The proposed run uses one hand-authored query and three hand-authored documents,
one GPU, one dtype/load path, one embedding dimension, no repeated timings, no
registered benchmark dataset, no contamination audit, no baselines, and no
task-level metrics. Its only purpose is to prove that the pinned model,
dependencies, provider loading, query/document routing, MRL truncation, and
normalization work together on this host. A passing smoke may justify a later
registry proposal and benchmark dispatch; it must not be published as a model
quality score.

## Primary Sources

- Voyage model card and pinned files:
  <https://huggingface.co/voyageai/voyage-4-nano>,
  <https://huggingface.co/voyageai/voyage-4-nano/tree/main>,
  <https://huggingface.co/voyageai/voyage-4-nano/commits/main>,
  <https://huggingface.co/voyageai/voyage-4-nano/blob/67fabc9bef010dabc5f6024aa1b1b6b93410426f/config.json>,
  <https://huggingface.co/voyageai/voyage-4-nano/blob/67fabc9bef010dabc5f6024aa1b1b6b93410426f/config_sentence_transformers.json>,
  <https://huggingface.co/voyageai/voyage-4-nano/blob/67fabc9bef010dabc5f6024aa1b1b6b93410426f/1_Pooling/config.json>,
  <https://huggingface.co/voyageai/voyage-4-nano/raw/67fabc9bef010dabc5f6024aa1b1b6b93410426f/model.safetensors>,
  and <https://blog.voyageai.com/2026/01/15/voyage-4/>.
- Qwen model card and pinned files:
  <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>,
  <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/main>,
  <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/commits/main>,
  <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config.json>,
  <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config_sentence_transformers.json>,
  <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/1_Pooling/config.json>,
  <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/model.safetensors>,
  <https://qwenlm.github.io/blog/qwen3-embedding/>, and
  <https://arxiv.org/abs/2506.05176>.
- CodeFuse model card and pinned files:
  <https://huggingface.co/codefuse-ai/ML-Embed-0.6B>,
  <https://huggingface.co/codefuse-ai/ML-Embed-0.6B/tree/main>,
  <https://huggingface.co/codefuse-ai/ML-Embed-0.6B/commits/main>,
  <https://huggingface.co/codefuse-ai/ML-Embed-0.6B/blob/fb82458ca9732a15a9526df89df84d5718efe89f/config.json>,
  <https://huggingface.co/codefuse-ai/ML-Embed-0.6B/blob/fb82458ca9732a15a9526df89df84d5718efe89f/config_sentence_transformers.json>,
  <https://huggingface.co/codefuse-ai/ML-Embed-0.6B/blob/fb82458ca9732a15a9526df89df84d5718efe89f/1_Pooling/config.json>,
  <https://huggingface.co/codefuse-ai/ML-Embed-0.6B/raw/fb82458ca9732a15a9526df89df84d5718efe89f/model.safetensors>,
  <https://huggingface.co/codefuse-ai/ML-Embed-0.6B/raw/fb82458ca9732a15a9526df89df84d5718efe89f/U.pth>,
  <https://huggingface.co/codefuse-ai/ML-Embed-0.6B/raw/fb82458ca9732a15a9526df89df84d5718efe89f/V.pth>,
  and <https://arxiv.org/abs/2605.15081>.
