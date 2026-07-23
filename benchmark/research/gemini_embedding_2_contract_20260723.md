# Gemini Embedding 2 Flat-Input Contract Evidence (2026-07-23)

## Scope and Readiness

- Perpetuum session: `meb-modern-embedding-leaderboard-1-1784800802-2-gemini2-3bd8a2c09d06`
- Selected item: `providers/gemini-embedding-2-contract`
- Baseline commit: `a81420eeba40792512d4996a2b441b9093857db2`
- Initial `git status --short`: clean
- Only the selected Gemini Embedding 2 provider-contract item was handled.

This note records a current primary-source audit, a bounded live attempt, and deterministic no-network
evidence. The live API was blocked by a Google location restriction before it returned any embedding.
Accordingly, this note does not claim live verification of the repaired cardinality path.

## Primary-Source Conclusions

### Google Gemini embeddings documentation

Source: <https://ai.google.dev/gemini-api/docs/embeddings> (fetched 2026-07-23 UTC)

- A plain list of strings is not a batch of independent logical inputs for `gemini-embedding-2`.
  Google states that Gemini Embedding 2 produces one aggregated embedding for multiple direct inputs.
- To obtain one embedding per logical input, each input must be wrapped in its own `Content` object,
  or sent through the asynchronous Batch API.
- This differs from `gemini-embedding-001`, which supports one individual embedding per string in a
  list and retains the legacy `task_type` API contract.
- Gemini Embedding 2 retrieval instructions belong in the text, not in `EmbedContentConfig.task_type`:
  - query: `task: search result | query: {content}`
  - document: `title: {title} | text: {content}`
  - document without a title: `title: none | text: {content}`
- The same page specifies text prefixes for classification, clustering, and sentence similarity.
  The provider preserves its existing public task names and maps them to those official prefixes.

The relevant page sections are "Generating embeddings", "Specify task type to improve performance",
"Embedding aggregation", and "Batch embeddings".

### MTEB fix

Commit: `b243378e5b16f511c7e529fcbb4ffe608db59d48`

- Commit URL:
  <https://github.com/embeddings-benchmark/mteb/commit/b243378e5b16f511c7e529fcbb4ffe608db59d48>
- Implementation at that commit:
  <https://raw.githubusercontent.com/embeddings-benchmark/mteb/b243378e5b16f511c7e529fcbb4ffe608db59d48/mteb/models/model_implementations/google_gemini.py>
- Commit title: `fix: use text task prefixes for gemini-embedding-2; improve doc prefix (#4851)`
- The fix removes `taskType` from Gemini Embedding 2 configuration, applies the same query prefixes
  documented by Google, and always formats retrieval documents as `title: ... | text: ...`, using
  `title: none` when no title exists.

### Pricing boundary

Source: <https://ai.google.dev/gemini-api/docs/pricing#gemini-embedding-2>
(fetched 2026-07-23 UTC)

- Standard paid-tier text input: USD 0.20 per 1 million tokens.
- The bounded live plan allowed at most seven logical text inputs and no publication.
- The live attempt stopped after the first 25-character ASCII input failed. Google exposed no usage
  metadata. Even if the rejected request were conservatively counted as 25 text tokens, the upper
  bound would be USD 0.000005. No successful billable response is claimed.

## Installed Environment

- `uv 0.11.26`
- `Python 3.13.2`
- installed `google-genai==1.66.0`
- `uv.lock` pins `google-genai` to `1.66.0`; no dependency or lockfile change was made.
- `GEMINI_API_KEY`: present at the gate; its value was not printed or otherwise inspected.

## Bounded Live Contract Attempt

The planned smoke contained seven logical items, below the eight-item ceiling:

1. one plain string;
2. two strings in the provider's pre-fix `contents=[str, str]` shape;
3. one formatted retrieval query;
4. one formatted retrieval document;
5. the same query and document wrapped as two separate `Content` objects for an order comparison.

The command stopped during item 1, as required by the access gate. No later call was attempted.

Sanitized outcome:

- attempted logical inputs: 1
- successful logical inputs: 0
- request structure: `model=gemini-embedding-2`, `contents=<single string>`,
  `output_dimensionality=768`
- response embedding count/order: unavailable; no response embedding was returned
- dimensions and norms: unavailable
- response usage fields: unavailable
- command wall time observed by the runner: approximately 2.1 seconds
- exact blocker:
  `400 FAILED_PRECONDITION: User location is not supported for the API use.`
- action: stopped immediately; no retry, alternate endpoint, proxy, or account workaround was used

Exact bounded command:

```bash
uv run --no-sync python - <<'PY'
from __future__ import annotations

import json
import time

import numpy as np
from google import genai
from google.genai import types

MODEL = "gemini-embedding-2"
DIMENSIONS = 768
client = genai.Client()
config = types.EmbedContentConfig(output_dimensionality=DIMENSIONS)
records: list[dict[str, object]] = []


def invoke(label: str, structure: str, logical_items: int, contents: object):
    started = time.perf_counter()
    response = client.models.embed_content(model=MODEL, contents=contents, config=config)
    latency_ms = (time.perf_counter() - started) * 1000
    vectors = [np.asarray(item.values, dtype=float) for item in response.embeddings or []]
    metadata = response.metadata
    records.append(
        {
            "label": label,
            "request_structure": structure,
            "logical_input_count": logical_items,
            "response_embedding_count": len(vectors),
            "dimensions": [int(vector.shape[0]) for vector in vectors],
            "norms": [round(float(np.linalg.norm(vector)), 6) for vector in vectors],
            "latency_ms": round(latency_ms, 3),
            "usage": {
                "billable_character_count": getattr(metadata, "billable_character_count", None),
            },
        }
    )
    return vectors


raw_single = invoke(
    "flat_single",
    "contents=<single string>",
    1,
    "flat contract probe alpha",
)
raw_multi = invoke(
    "flat_two_current_shape",
    "contents=[<string 1>, <string 2>]",
    2,
    ["flat contract probe beta", "flat contract probe gamma"],
)
query_text = "task: search result | query: contract probe query"
document_text = "title: none | text: contract probe document"
query_single = invoke(
    "retrieval_query_single",
    "contents=<formatted query string>",
    1,
    query_text,
)
document_single = invoke(
    "retrieval_document_single",
    "contents=<formatted document string>",
    1,
    document_text,
)
wrapped_pair = invoke(
    "wrapped_query_document_pair",
    "contents=[Content(query), Content(document)]",
    2,
    [
        types.Content(parts=[types.Part.from_text(text=query_text)]),
        types.Content(parts=[types.Part.from_text(text=document_text)]),
    ],
)

order = {
    "available": len(query_single) == 1 and len(document_single) == 1 and len(wrapped_pair) == 2,
}
if order["available"]:
    similarities = np.array(
        [
            [
                float(np.dot(wrapped_pair[i], reference) / (np.linalg.norm(wrapped_pair[i]) * np.linalg.norm(reference)))
                for reference in (query_single[0], document_single[0])
            ]
            for i in range(2)
        ]
    )
    order.update(
        {
            "pair_0_matches": "query" if similarities[0, 0] > similarities[0, 1] else "document",
            "pair_1_matches": "document" if similarities[1, 1] > similarities[1, 0] else "query",
            "diagonal_cosine": [round(float(similarities[0, 0]), 9), round(float(similarities[1, 1]), 9)],
            "off_diagonal_cosine": [round(float(similarities[0, 1]), 9), round(float(similarities[1, 0]), 9)],
        }
    )

print(json.dumps({"model": MODEL, "configured_dimensions": DIMENSIONS, "logical_input_total": 7, "calls": records, "order_check": order}, indent=2))
PY
```

The exception occurred on the first `invoke` call, so none of the vector-reporting code emitted data.

## Code and Test Conclusions

The repository fix is intentionally narrow:

- `gemini-embedding-2` text batches now wrap each logical text input in a separate `Content` object.
- Gemini Embedding 2 applies official text instructions and omits legacy `task_type` configuration.
- `gemini-embedding-001` and other legacy Gemini model names retain the prior list-of-strings request
  and `task_type` routing.
- Every Gemini response is checked for exact input/embedding cardinality before rows are associated
  with input indices.
- Gemini Embedding 2 text cache keys include a request-contract version and the effective formatted
  text, preventing old task-type semantics from colliding with the repaired path.
- Cache hits must be at least two-dimensional and have a first-axis row count equal to the logical
  input count. One-dimensional or wrong-cardinality arrays are ignored and regenerated.
  Therefore `MRLStressTask`, which calls `provider.embed_text()` once for all unique sentences, cannot
  silently consume a missing or aggregated Gemini row through either a live response or cache hit.

Deterministic doubles cover:

- N-input/N-vector cardinality across multiple provider batches;
- stable response-to-input ordering;
- `Content` wrapping for Gemini Embedding 2;
- official retrieval-query and titled/untitled retrieval-document formats;
- absence of legacy `task_type` for Gemini Embedding 2;
- preservation of legacy list-of-strings plus `RETRIEVAL_QUERY` routing;
- rejection of malformed live-response cardinality;
- rejection/regeneration of malformed cached cardinality;
- rejection/regeneration of a one-dimensional cache array whose length equals the input count;
- cache-key isolation by effective text.

## Treatment of the 2026-07-16 Ignored Smoke Row

`results/api-coverage-smoke-20260716-layer1.jsonl` is ignored local evidence. Its Gemini row reports
12 unique sentences and successful MRL metrics, but it does not record the sanitized request shape,
response embedding count/order, SDK usage metadata, or whether a cache contributed. The provider at
that commit also lacked an explicit cardinality check and used the now-documented aggregation-prone
plain-list request shape.

The July 16 Gemini status is therefore provisional provider-smoke evidence only. It is not current
contract proof and must not be promoted to public leaderboard evidence. A future rerun from a supported
location should use the repaired provider, record exact cardinality/order, and remain a separate evidence
tier from quality scoring.

## Validation Commands

Focused no-network validation performed before the full suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/test_provider_api_compat.py -q
uv run --no-sync python -m py_compile \
  src/mm_embed/providers/base.py \
  src/mm_embed/providers/gemini_provider.py \
  tests/test_provider_api_compat.py
git diff --check
```

Focused result: `19 passed in 1.15s`.

Final validation:

```bash
set -euo pipefail
awk 'length($0) > 120 { print FILENAME ":" FNR ":" length($0) ":" $0; failed=1 } END { exit failed }' \
  src/mm_embed/providers/base.py \
  src/mm_embed/providers/gemini_provider.py \
  tests/test_provider_api_compat.py
uv run --no-sync python -m py_compile \
  src/mm_embed/providers/base.py \
  src/mm_embed/providers/gemini_provider.py \
  tests/test_provider_api_compat.py
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest -q
```

- 120-column check: passed for all changed Python files.
- Python compilation: passed.
- `git diff --check`: passed.
- Full repository suite: `134 passed in 19.00s`.

## Boundaries Observed

- No secret value or request header was printed.
- No benchmark matrix was run.
- No model or dataset was downloaded.
- No Hugging Face endpoint, model, dataset, alternate IP, or upload was used.
- No result, score, note, or artifact was published remotely.
- No dependency was installed, updated, or changed.
- No commit or push was made.
