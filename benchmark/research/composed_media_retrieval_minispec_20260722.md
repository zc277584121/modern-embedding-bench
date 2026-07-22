# Composed-Media Retrieval Minispec - 2026-07-22

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_8-1784733404-2_execute.md`

Selected item: `tasks/composed-media-retrieval-scout`

Decision: **DIFFERENTIATE.**

Adopt a narrow provider-neutral composed-media input contract, but keep it as a
separate capability and task family. It is required for queries such as
text+reference-image, text+reference-video, or source-audio+difference-text that
must become exactly one logical embedding. It is **not** required before adding
ordinary video-to-video, audio-to-audio, audio-to-image, image-to-audio, or other
independently encoded media retrieval tasks.

This document is a decision record and minispec only. It does not implement a
contract, fixture, task, provider adapter, registry entry, manifest, result, or
Hugging Face artifact.

Labels used below:

- **VERIFIED**: checked in the current repository or a primary source on
  2026-07-22.
- **DECISION**: the selected v0 behavior.
- **GATE**: a condition that later implementation or publication must satisfy.
- **NON-GOAL**: deliberately excluded from v0.

## 1. Decision in One Screen

The repository currently conflates two different ideas under broad modality
labels:

1. an item happens to be audio, image, or video; and
2. one logical item is composed from several ordered heterogeneous parts.

Those are not the same capability.

| Shape | Example | Needs composed-item v0? | Reason |
|---|---|---:|---|
| One logical query with ordered heterogeneous parts | text+image -> video | Yes | All parts must produce one query vector and one identity. |
| One logical query with ordered heterogeneous parts | audio+text -> audio | Yes | Flattening creates two query vectors and changes the task. |
| Independently encoded cross-modal items | audio -> image | No | Query and corpus items are individually encoded. |
| Same-modality retrieval | video -> video, audio -> audio | No | The flat item contract is sufficient for each item. |
| Provider-native fused embedding | Gemini/Qwen/DashScope/Voyage/Volcengine mixed input | Yes | The provider receives one multi-part item and returns one vector. |
| Benchmark-side sum/mean of independent embeddings | MTEB Fusion wrapper | Separate system track | It is a system fusion policy, not provider-native embedding evidence. |
| Pairwise reranker over query and document | Qwen3-VL-Reranker | No embedding claim | It returns a relevance score, not an independently indexable embedding. |

**DECISION.** Define a separate `ComposedMediaEmbeddingProvider`-style
capability. Never overload `EmbeddingProvider.embed()` with an ambiguous nested
list and never infer composed support from `supported_modalities` alone.

**DECISION.** Ordinary media retrieval work may proceed independently. It may
still need a general retrieval evaluator with explicit corpus ids, multi-positive
qrels, and deterministic ranking, but it does not need composed-item routing.

## 2. Repository Baseline and Exact Gap

### 2.1 Flat provider input

**VERIFIED.** `src/mm_embed/providers/base.py` defines `EmbeddingInput` as one
modality plus one content payload. `EmbeddingProvider.embed()` accepts a list of
those flat items and returns an array whose first axis is the item count.

This is suitable for:

- one text query;
- one image corpus item;
- one audio query;
- one video corpus item; or
- a batch of independent items.

It cannot state that two adjacent entries are parts of one logical item. A list
containing text, image, text, image could mean four independent embeddings, two
composed items, or one four-part item.

### 2.2 Cache identity

**VERIFIED.** `src/mm_embed/cache.py` hashes the model, dimensions, task type,
and a linear sequence of modality/content hashes. It does not encode logical
item boundaries, part ids, media metadata, provider routing mode, preprocessing,
or whether a result is provider-native fusion versus benchmark-side fusion.

The current key is therefore safe only for the current flat batch contract. It
must not be reused for composed v0.

### 2.3 Current cross-modal evaluator

**VERIFIED.** `src/mm_embed/tasks/cross_modal_retrieval.py` separately embeds
all text and all images, then assigns diagonal one-to-one ground truth. It does
not preserve explicit corpus ids, cannot represent a text+reference-media query,
and cannot express general multi-positive qrels.

This evaluator remains useful as an ordinary independently encoded text-image
task, but it is not a base class for composed-media retrieval.

### 2.4 Grouped chunks are a different capability

**VERIFIED.** `src/mm_embed/providers/grouped_chunks.py` preserves ordered chunk
groups under one parent document and returns one embedding per supplied chunk.
Its purpose is contextual chunk embedding without crossing document boundaries.

That contract should be reused as a design pattern, not as the composed-media
surface:

- grouped chunks: one parent group -> many canonical chunk embeddings;
- composed media: one ordered logical item -> exactly one embedding.

Reusing `GroupedChunkEmbeddingProvider` would invert the output cardinality and
would encourage accidental flattening.

### 2.5 Current provider adapters

**VERIFIED.** The current adapters all expose only flat item routing:

- Gemini builds one API content from one `EmbeddingInput` at a time;
- DashScope wraps one text, image, or video in a one-element content list;
- Volcengine builds one text or one media element per repository input;
- Voyage builds one one-part multimodal input per repository input; and
- no adapter reports a composed-media capability or echoes a logical-item
  fingerprint.

The model registry already lists broad modality support, but a modality list is
not evidence that the repository can send several parts as one item.

## 3. MTEB 2.18.6 Task Evidence

### 3.1 Release pin

**VERIFIED.** The annotated Git tag `2.18.6` peels to commit
`fa36ee7a6274bba288a925259c74e3382b53532d`. The tag source defines explicit
categories such as `it2v`, `vt2v`, and `at2a`; `TaskMetadata.get_modalities()`
parses the query-side characters in order.

Release inclusion is evidence that the shapes are real benchmark categories. It
is not automatic approval to import their media, licenses, metrics, or task
implementation unchanged.

### 3.2 Task comparison

The size values below are rough current repository-page totals discovered
through anonymous read-only extraction. The host itself could not resolve
`huggingface.co`, so they are not claimed as exact byte totals for every pinned
revision. Durations and row counts come from the exact MTEB task commits and
their checked-in descriptive statistics.

| Task and exact task commit | Logical shape | Queries / corpus / positives | Primary metric | Rough media fit | License/provenance note |
|---|---|---|---|---|---|
| MomentSeeker TI2V/TV2V, `9082aa7d6e02047975701466dfb12d2d36898214` | image+text -> video; video+text -> video | 400 / 3,092 / 502 qrels; 1-9 positives per query | `map_at_5` | Derived page currently reports about 5.09 GB. Corpus is about 92,075 s (~25.6 h) of mostly 30 s, 360p chunks. TV2V reference clips total about 1,496 s. | Dataset revision `e87fc71839cd14dfd09aa83b378cd51074318b3c`; beta; CC-BY-NC-SA-4.0. Upstream page reports 93.8 GB and warns it does not own raw-video copyrights. |
| CLD AT2A, `3e4ec5254f5a017329aa567c43616c93af291c93` | audio+text -> audio | 2,000 / 1,045 / 2,000 qrels; one positive per query | `ndcg_at_10` | Derived page currently reports about 6.03 GB. Corpus totals about 23,416 s (~6.5 h); query source-audio rows total about 44,953 s, with repeated source clips. | Dataset revision `4db74f3a92fb2e5efad0d35ca5373807b9628c47`. MTEB metadata says MIT, but the derived HF card is empty; ADIFF and Clotho asset terms need a publication audit. |
| SpeechCOCO A2I/I2A, `e8f0b218192c39dbc92c421cbb6ad697f6560972` | audio -> image; image -> audio | 1,000 / 2,048 / 1,000 qrels; one positive per query | `ndcg_at_10` | Derived A2I page currently reports about 449 MB. Query speech totals about 3,543 s (~59 min). | Revisions `217c6660258de6e60002f748abdf11be623c8e0e` and `afb9e08254be4e9c2e5af432912291dde7528b68`; CC-BY-4.0. Upstream SpeechCOCO is built over MSCOCO and synthetic TTS. |
| VSC2022, `80f6e3b89539a433589ac4a685d3a9f86e4f0f10` | video -> video copy retrieval | 1,926 / 3,000 / 2,448 qrels; 1-3 positives per query | `map_at_10` | Derived page currently reports about 6.58 GB. Query videos total about 89,761 s (~24.9 h); corpus totals about 95,803 s (~26.6 h). | Dataset revision `ba6224b845a2181d3de4199aa4eeb8b5b874a1a3`; MTEB metadata says CC-BY-4.0; source clips are from YFCC100M. This is copy detection, not semantic retrieval. |
| VimSketch A2A, `9d1529723c36c6f60a9764df0a15d436a81736ad` | vocal imitation audio -> reference audio | 2,168 / 542 / 2,168 qrels; one positive per query | `ndcg_at_10` | Derived page currently reports about 1.7 GB. Query audio totals about 11,980 s (~3.3 h); corpus totals about 2,543 s (~42 min). | Dataset revision `466e0ea0ed8f50bad9c240f3bfc8426c08430aa2`; CC-BY-4.0. Zenodo v1.0 is 4,488,796,632 bytes and combines Vocal Imitation Set and VocalSketch sources. |
| SoundingEarth A2I/I2A, `81958bba2d7a60518e9269c5f01c09c8c0042ec6` | audio -> aerial image; image -> audio | 1,000 / 2,048 / 1,000 qrels; one positive per query | `ndcg_at_10` | Derived A2I page currently reports about 9.94 GB. Query recordings total about 109,140 s (~30.3 h), capped at 120 s. | Revisions `e97546c9ce77b71da1c8a97b6bb6c034f4378bee` and `137f91ccf6a3ce892697bd6022560b22ac3342f8`; CC-BY-4.0. Upstream Zenodo v1.0 is 24,359,464,960 bytes and derives field recordings from Radio Aporee plus Google Earth imagery. |

### 3.3 What the comparison proves

**VERIFIED.** MomentSeeker and CLD change the identity of the query itself: the
text modifies or constrains reference media. Encoding the parts independently
and treating both vectors as queries would not evaluate the published task.

**VERIFIED.** SpeechCOCO, VSC2022, VimSketch, and SoundingEarth do not require
that composition surface. Each query and corpus item can be independently
encoded. They may require audio/video loading, explicit ids, multi-positive
qrels, or copy-detection diagnostics, but not a multi-part query object.

**DECISION.** Do not make ordinary audio/video task implementation wait for the
composed provider adapters. Keep the schema families distinct so ordinary tasks
cannot silently inherit fusion behavior.

## 4. Real Mixed-Input Behavior

### 4.1 Provider and model matrix

| Source | Mixed-input evidence | Correct v0 label | Current repository status |
|---|---|---|---|
| Qwen3-VL-Embedding source at `393e2978d27852b0d0230d6994f37f9c15bed73c` | One input dictionary may contain text, image, video, or mixtures; the embedding model maps that single mixed object to one vector. The 2B variant is 2B parameters, 32K context, up to 2,048 dimensions, and documents up to 64 video frames. | `provider_native_fusion` for embedding. | No local adapter and no model download. Registry modality labels alone are insufficient. |
| Qwen3-VL-Reranker in the same source | A mixed query/document pair is jointly processed into a relevance score. | `reranker_system_only` | Must never be presented as an embedding-provider score. |
| MTEB Fusion Embedding wrapper at `796793983bc6f142e70ffe176cc224f0c22a16d5` | It separately embeds each present modality and sums vectors elementwise. | `benchmark_system_fusion` | Useful comparator, not provider-native fusion evidence. |
| Gemini Embedding 2 official docs | Multiple parts in one content produce one aggregated embedding; multiple `Content` objects produce separate embeddings. Text, image, audio, video, and PDF share one space. | `provider_native_fusion` | Current adapter creates one-part contents and cannot preserve a composed item. |
| DashScope Qwen3-VL-Embedding official docs | `contents` may hold text, images, and video. `enable_fusion=true` returns one fused embedding; default `false` returns independent embeddings. | `provider_native_fusion` only with explicit fusion route evidence | Current adapter never sets `enable_fusion` and sends one primitive item at a time. |
| Voyage Multimodal 3.5 official docs | Each top-level input is an ordered sequence of interleaved text, images, and videos and yields one vector. | `provider_native_fusion` | Current adapter builds a one-part top-level input for each flat repository item. |
| Volcengine/Doubao official docs | `client.multimodal_embeddings.create(..., input=[...])` accepts mixed text, image, and video parts in one request object; official use cases include image+text retrieval. | `provider_native_fusion` when all parts are in one `input` list | Current adapter uses a different flat `embeddings.create` path and one repository item at a time. |

### 4.2 Embedding-valid versus system-only

**DECISION.** A score is `provider_valid_embedding` only when all of the
following are true:

1. the provider/model exposes an embedding endpoint or embedding model;
2. all ordered parts of one logical item are sent together through the
   provider's documented fused-input route;
3. the provider returns exactly one vector for that logical item;
4. no benchmark-side sum, mean, concatenation, projection, or reranking score is
   substituted; and
5. request and response evidence identify the routing mode without secrets.

The following labels are separate and cannot populate the same primary model
row:

- `provider_valid_embedding`;
- `benchmark_system_fusion`;
- `reranker_system_only`; and
- `contract_fixture_only`.

## 5. Narrow v0 Scope

### 5.1 In scope

**DECISION.** v0 supports:

- one logical query or corpus item;
- one or more ordered parts;
- heterogeneous text, image, audio, video, or document parts;
- exactly one embedding per logical item;
- independently encoded corpus items or composed corpus items;
- explicit multi-positive graded qrels;
- provider-native and separately labeled benchmark-system fusion tracks; and
- a self-authored no-network fixture for contracts and metrics.

### 5.2 Non-goals

**NON-GOAL.** v0 does not include:

- provider-specific prompt templates as benchmark truth;
- automatic media captioning or transcription;
- generative query rewriting;
- late-interaction multi-vector retrieval;
- pairwise reranking as an embedding score;
- provider-side search/index products;
- temporal localization inside a returned video;
- media downloading from the six audited datasets;
- public quality claims from the self-authored fixture; or
- automatic fallback from fused input to separate part embeddings.

## 6. Data Schemas

All ids are non-empty UTF-8 strings. Timestamps are audit metadata and are
excluded from content fingerprints. Absolute local paths are never part of a
portable identity.

### 6.1 Ordered part

```json
{
  "schema_version": "composed-media-part-v0",
  "part_id": "q_ti_0001:p00",
  "part_index": 0,
  "modality": "text",
  "payload": {
    "kind": "inline_text",
    "text": "Find the clip immediately after the pictured state."
  },
  "mime_type": "text/plain; charset=utf-8",
  "content_sha256": "...",
  "byte_length": 54,
  "normalization": "unicode-nfc-lf-v1",
  "media": null,
  "provenance_id": "prov_fixture_v0"
}
```

For tracked media, `payload` uses a repository-relative `path` plus an immutable
content hash. `media` may contain only applicable integer fields:

```json
{
  "duration_ms": 2000,
  "sample_rate_hz": 8000,
  "frame_count": 16,
  "width_px": 64,
  "height_px": 64,
  "representation": "ordered_png_frames"
}
```

**DECISION.** `part_index` must be contiguous from zero and must match array
position. Part order is included in every item fingerprint even when a system's
fusion operation is mathematically commutative.

### 6.2 Logical item

```json
{
  "schema_version": "composed-media-item-v0",
  "item_id": "q_ti_0001",
  "role": "query",
  "parts": [
    {"part_id": "q_ti_0001:p00", "part_index": 0},
    {"part_id": "q_ti_0001:p01", "part_index": 1}
  ],
  "instruction": "Retrieve the target media item.",
  "item_sha256": "...",
  "provenance_id": "prov_fixture_v0"
}
```

The serialized artifact stores full part records, not only the abbreviated
references shown above. The example emphasizes that order and grouping belong
to the item.

### 6.3 Query

```json
{
  "query_id": "q_ti_0001",
  "item_id": "q_ti_0001",
  "split": "fixture_only",
  "shape": "text_image_to_video",
  "family": "temporal_successor",
  "target_modalities": ["video"],
  "provider_valid_required": true,
  "item_sha256": "..."
}
```

### 6.4 Corpus item

```json
{
  "corpus_id": "video_0004",
  "item_id": "video_0004",
  "split": "fixture_only",
  "shape": "video",
  "item_sha256": "..."
}
```

An ordinary one-part audio or video item is valid in this schema, but ordinary
tasks do not need to use this composed task implementation.

### 6.5 Qrel

```json
{
  "query_id": "q_ti_0001",
  "corpus_id": "video_0004",
  "relevance": 2,
  "judgment": "exact_target",
  "provenance_id": "prov_fixture_v0"
}
```

Relevance is an integer greater than or equal to zero. `relevance > 0` is
positive for binary metrics; the integer grade is used by nDCG. Duplicate
query/corpus qrels, qrels for unknown ids, and queries with no positive are
invalid.

### 6.6 Provenance

```json
{
  "provenance_id": "prov_fixture_v0",
  "source_kind": "self_created_fixture",
  "source_uri": null,
  "source_revision": "composed-media-retrieval-fixture-v0",
  "license_id": "not_for_publication",
  "derivation": "deterministic_project_owned_generator",
  "transform_version": "composed-media-fixture-generator-v0",
  "network_required": false
}
```

For external data, provenance must include the exact source URI, immutable
revision, upstream license statement, transformations, and any unresolved asset
rights. A task-file license label alone is not sufficient publication evidence.

## 7. Stable Identity and Serialization

### 7.1 Canonical bytes

**DECISION.** Fingerprints use canonical UTF-8 JSON with:

- Unicode NFC and LF normalization for text;
- sorted object keys;
- array order preserved;
- separators `,` and `:` with no insignificant whitespace;
- integers for byte counts, durations, frame counts, dimensions, and sample
  rates;
- no NaN, infinity, platform paths, or implicit defaults; and
- an explicit schema-version domain separator.

Text `content_sha256` hashes the normalized UTF-8 bytes. Media
`content_sha256` hashes the exact tracked bytes. A frame-sequence video hashes
the canonical manifest containing ordered frame hashes, fps numerator and
denominator, duration, dimensions, and color-space label.

### 7.2 Fingerprint layers

**DECISION.** Keep four identities separate:

1. `part_sha256`: modality, normalized payload identity, media metadata, and
   provenance id;
2. `item_sha256`: schema, role, instruction, and the ordered sequence of full
   part identities;
3. `request_sha256`: ordered item fingerprints plus provider/model revision,
   dimensions, task route, preprocessing, and fusion strategy; and
4. `result_sha256`: request fingerprint plus ordered item ids, embeddings,
   dimensions, and route evidence.

The composed cache key is `request_sha256`. The flat `make_cache_key()` is not
used.

### 7.3 Identity invariants

Reordering parts, changing a media byte, changing a MIME type, changing a
duration/frame manifest, changing an instruction, changing provider routing, or
moving a part across logical-item boundaries must change the relevant
fingerprint.

Renaming an absolute checkout path or changing an audit timestamp must not.

## 8. Provider Capability and Routing

### 8.1 Separate capability

**DECISION.** A later implementation should expose a separate protocol shaped
like:

```python
class ComposedMediaEmbeddingProvider(Protocol):
    def embed_composed_media(
        self,
        request: ComposedMediaEmbeddingRequest,
    ) -> ComposedMediaEmbeddingResult:
        ...
```

Do not change `EmbeddingProvider.embed()` in v0. Ordinary providers and tasks
remain source compatible.

### 8.2 Capability declaration

A provider capability record must state:

- `composition_mode`: `provider_native_fusion`,
  `benchmark_system_fusion`, or `unsupported`;
- supported part modalities;
- allowed query and corpus shapes;
- maximum parts and per-modality limits;
- whether multiple parts of the same modality are supported;
- provider model id and immutable revision when available;
- endpoint/method name;
- required routing flags such as DashScope `enable_fusion=true`;
- preprocessing/version identity; and
- whether file paths, bytes, URLs, uploaded files, or frame sequences are
  accepted.

### 8.3 Cardinality rule

**DECISION.** For `N` logical items, the provider must return exactly `N`
embeddings in the same item order. A response with one embedding per part is a
hard failure, not an alternative valid result.

Every returned row must echo:

- `item_id`;
- `item_sha256`;
- `request_sha256`;
- provider/model/revision;
- `composition_mode`;
- dimensions; and
- sanitized route evidence.

### 8.4 No silent fallback

If a provider cannot natively fuse the requested parts, the provider-valid
track stops with `unsupported_composition`. It must not:

- call the flat provider once per part;
- average or sum vectors;
- concatenate vectors;
- drop an unsupported part;
- convert media to generated text; or
- use a reranker score.

A separately configured system-fusion track may do those operations only when
its complete algorithm and labels are explicit and reproducible.

## 9. Scoring Contract

### 9.1 Retrieval matrix

Each logical query produces one vector and each corpus item produces one vector.
The default similarity is cosine similarity after rejecting zero-norm and
non-finite vectors. The result stores the unrounded similarity used for ranking.

### 9.2 Exact tie behavior

**DECISION.** Exact score ties are broken by normalized `corpus_id` ascending in
UTF-8 byte order. Ranking key:

```text
(-similarity_score, corpus_id_utf8_bytes)
```

No epsilon is used to manufacture ties, and scores are not rounded before
ranking. This is stricter than relying on library or accelerator ordering. MTEB
2.18.6 currently combines `torch.topk`, heap ordering, and `pytrec_eval`; the
task source does not make an exact cross-backend tie policy part of its public
task metadata.

### 9.3 Metrics

**DECISION.** v0 primary and secondary metrics are:

- primary: `composed_ndcg@10` using integer qrel grades;
- secondary: `composed_map@5`;
- secondary: `composed_recall@1`, `@5`, and `@10`;
- secondary: `composed_mrr@10`, using the first positive;
- secondary: `composed_hit_rate@10`; and
- diagnostic: positive-set coverage at each k.

MAP, recall, MRR, and hit rate treat every `relevance > 0` qrel as positive.
nDCG retains the grades.

### 9.4 Paired diagnostics

When the same provider supports both a native fused route and a declared
independent-part baseline, report paired deltas only when every query/corpus id
and preprocessing identity is pairable:

- `native_fusion_gain_ndcg@10`;
- `native_fusion_gain_map@5`; and
- `native_fusion_gain_recall@1`.

These are diagnostics, not substitutes for the provider-valid primary score.
Unpairable, partially failed, or differently preprocessed runs are rejected.

### 9.5 Required slices

Report at least:

- query shape: text+image, text+video, audio+text;
- part count;
- single-positive versus multi-positive;
- target modality;
- reference-media duration bucket;
- hard-negative family;
- provider-native versus system fusion;
- order-sensitive fixture family; and
- media-reuse versus distinct-media query.

The fixture reports contract behavior only and is always labeled
`not_for_publication`.

## 10. Hard Negatives and Quality Gates

### 10.1 Hard-negative families

The smallest useful fixture must contain reviewed negatives from these
families:

1. **Text-match/media-mismatch**: the description matches but the reference
   media points to a different target.
2. **Media-neighbor/text-contradiction**: the reference media is visually or
   acoustically close, but the text specifies the opposite transformation.
3. **Temporal neighbor**: the candidate is immediately before rather than after
   the reference event, or vice versa.
4. **Difference-direction inversion**: for audio+text, the candidate applies the
   described change in the wrong source/target direction.
5. **Part-order swap**: the same part set appears in a different order and has a
   different logical identity.
6. **Cross-family collision**: a candidate shares color, shape, pitch, rhythm,
   or motion tokens but belongs to the wrong authored family.
7. **Duplicate/near-duplicate leakage**: a candidate reuses forbidden bytes or
   frames from the positive beyond an explicitly allowed query reference.

Select negatives before evaluating any real model. Do not mine them from a
candidate provider's scores for the primary fixture labels.

### 10.2 Leakage gates

Reject the fixture or later external task if:

- a query contains a unique identifier that trivially names the positive;
- a positive and negative have unintended byte-identical media;
- train/dev/test share derived assets without a declared grouping rule;
- a reference crop/clip directly includes the answer-only frame or audio event
  when the task intends a temporal or difference relation;
- qrels can be reconstructed from corpus ordering;
- hard negatives become valid positives under full authored context; or
- external source license/provenance is unresolved for public publication.

## 11. Deterministic Rejection Gates

A later validator and tests must reject all of the following:

- flattened parts returned as separate embeddings;
- reordered parts;
- non-contiguous or duplicate `part_index` values;
- duplicate part or item ids;
- a part attached to the wrong logical item;
- changed media bytes, MIME type, duration, frame order, or content hash;
- a result count different from the logical-item count;
- reordered result item ids;
- missing request/item fingerprints;
- unsupported provider fusion silently routed through flat embedding;
- system-fusion output labeled provider-valid;
- reranker scores labeled embeddings;
- NaN, infinity, zero-norm, or inconsistent-dimension vectors;
- missing multi-positive qrels;
- qrels for unknown query/corpus ids;
- nondeterministic canonical serialization;
- ties whose order changes between repeated evaluations; and
- a network attempt while running the fixture-only task.

The test suite must explicitly mutate one field at a time so each rejection is
attributable.

## 12. Smallest Self-Authored Fixture

### 12.1 Purpose

**DECISION.** The first fixture validates schemas, identities, routing
cardinality, metrics, slices, and corruption gates. It does not compare model
quality and does not call a provider.

### 12.2 Asset design

Use only deterministic project-owned assets:

- 6 short video manifests, each 16 ordered 64x64 PNG frames at 8 fps for 2 s;
- 6 short mono PCM WAV files, each 2 s at 8 kHz;
- 6 still PNG reference images reused from authored video frames; and
- text descriptions written specifically for the fixture.

The video contract fixture uses ordered PNG frames rather than requiring a
codec. A later provider smoke may materialize a provider-supported MP4 with a
pinned toolchain and record both manifest and container hashes, but that is not
part of the first implementation item.

The complete tracked fixture should target less than 10 MiB. The exact total is
an implementation acceptance value, not a fabricated value in this minispec.

### 12.3 Query and qrel shape

Create 12 composed queries:

- 4 text+image -> video queries;
- 4 text+video -> video queries; and
- 4 source-audio+text -> audio queries.

Use the 6 videos and 6 audio clips as full candidate pools for their respective
query families. Produce 16 positive qrels: eight queries with one grade-2
positive and four queries with two positives, one grade 2 and one grade 1.

Attach exactly three pre-reviewed hard negatives per query, covering all seven
families above. Full-corpus metrics remain authoritative; the hard pool is a
diagnostic slice.

### 12.4 Authoring procedure

1. Generate media and immutable hashes from a pinned deterministic generator.
2. Define positive relations before writing query text.
3. Write queries without target ids or unique answer-only tokens.
4. Assign qrels and hard negatives before any model score is visible.
5. Blind-review query+reference parts against the full candidate pool.
6. Verify that each multi-positive grade is defensible.
7. Run serialization twice in separate temporary directories and compare all
   canonical bytes and hashes.
8. Run the evaluator with a deterministic test double and with sockets blocked.

### 12.5 Publication status

The fixture must set:

- `fixture_only: true`;
- `publish: false`;
- `leaderboard_publish: false`;
- `license_status: not_for_publication`;
- `network: forbidden`; and
- `provider_api_calls: 0`.

## 13. Cost, Access, and Machine Fit

### 13.1 API providers

Current official pricing/access evidence is provider-specific and can change.
Every live follow-up must refresh it before a call.

- **Gemini Embedding 2.** Official standard pricing currently lists text at
  $0.20/1M tokens, images at $0.00012 each, audio at $0.00016/s, and video at
  $0.00079/frame; the page also lists a free tier. The official limits include
  up to 180 s audio, 120 s video, and at most 32 sampled video frames. A tiny
  fixture smoke would be inexpensive, but it still requires an explicit
  provider-adapter item and a preflight.
- **DashScope Qwen3-VL-Embedding.** Official China (Beijing) docs list text at
  $0.10/1M input tokens and image/video at $0.258/1M input tokens, with a maximum
  of 20 content elements, 5 images, and 1 video. Fusion must be explicitly
  enabled. Region-specific access and API keys apply.
- **Voyage Multimodal 3.5.** Official docs currently describe interleaved mixed
  inputs, 32K tokens, up to 20 MB per image/video, 200M free text tokens and
  150B free pixels per account, followed by $0.12/1M text tokens and $0.60/B
  pixels. The repository adapter does not yet expose mixed parts.
- **Volcengine/Doubao.** Current official docs confirm mixed text/image/video
  request lists and the `multimodal_embeddings.create` path. The checked page
  did not provide a durable price table suitable for this record. Refresh
  endpoint availability, deployment id, dimensions, and price before any call;
  do not infer price from another model or older page.

Any billing, quota, account, or region restriction stops the provider item and
is reported. The existing approximately USD 30 investigation ceiling remains
in force.

### 13.2 Local/open models

The current host has four RTX 3080 Ti GPUs with 12,288 MiB each and about 287 GB
free on `/data2` as of `2026-07-22T15:38:52Z`.

- Qwen3-VL-Embedding-2B is Apache-2.0 and documents 2B parameters. BF16 weights
  alone are roughly 4 GB by parameter arithmetic, so a bounded single-GPU smoke
  is plausible, but video activations, dependencies, and exact artifact bytes
  remain unverified. No download is authorized by this item.
- The 8B variant is roughly 16 GB for BF16 weights alone and is not a clean
  one-card 12 GB first path without quantization or sharding.
- MTEB's Fusion Embedding 2 metadata reports about 10,681 MB model memory,
  `trust_remote_code`, a frozen base plus audio tower, and CC-BY-NC-4.0. The
  small remaining margin on a 12,288 MiB card, remote-code dependency chain,
  and non-commercial license make it a poor first smoke.

### 13.3 Dataset fit

The six external task families range from roughly 449 MB to around 10 GB in the
current derived pages, while the upstream MomentSeeker release is 93.8 GB and
upstream SoundingEarth is 24,359,464,960 bytes. Even when disk is available,
licenses, media decoding, provider cost, and runtime make them inappropriate for
the first contract implementation.

## 14. Current Hugging Face Connectivity

**VERIFIED.** A token-disabled normal-hostname check on 2026-07-22 failed DNS
resolution for `huggingface.co`; HTTPS therefore returned no status. No alternate
or hard-coded IP was used.

Anonymous AnySearch was run with `ANYSEARCH_API_KEY` unset and an explicit empty
CLI key. It could read current public page metadata, which is why rough current
page sizes are recorded above. Those proxy-visible values do not remove the
normal-hostname gate for downloads, exact pinned-revision inspection, or any
Hugging Face operation.

**GATE.** Before a later model/media download or publish step, a fresh
token-disabled check must confirm normal `huggingface.co` DNS and HTTPS. If it
still fails, leave the item blocked rather than using an alternate IP or host.

## 15. Git and Hugging Face Product Path

### 15.1 Bounded Git path

The first implementation should remain small and reviewable:

- provider-neutral schema/protocol and validators;
- a deterministic self-authored fixture generator or checked-in fixture under
  the size cap;
- a fixture-only evaluator;
- focused corruption/metric tests; and
- no provider adapter in the same item.

Generated temporary media must be cleaned up. Only the canonical bounded
fixture, source definitions, and deterministic metadata belong in Git.

### 15.2 Eventual real task path

After the contract fixture passes, evaluate one machine-fit, license-clear
external or newly authored pilot. Do not import all six MTEB task families at
once. A real task must pin source revision, file hashes, media transforms,
decoder/tool versions, qrels, and split leakage rules.

### 15.3 Hugging Face Dataset and Space

The self-authored contract fixture is excluded from public Dataset/Space output.
An eventual audited task may publish:

- query/item/part/qrel/provenance tables;
- immutable media asset references or bounded project-owned media;
- provider/model revision and route evidence;
- `score_validity` and `composition_mode` columns;
- primary and slice metrics; and
- latest-run markers consistent with the current leaderboard export policy.

The Space must visibly separate provider-valid embeddings, system fusion, and
reranker-only results. It must not rank them as interchangeable rows.

## 16. One Restrained Follow-up Item

Recommended next item:

`tasks/composed-media-contract-fixture`

Acceptance criteria:

1. add a separate composed-media request/result protocol without changing the
   flat `EmbeddingProvider.embed()` signature;
2. implement canonical part/item/request/result fingerprints and validators;
3. add the 12-query self-authored zero-network fixture with the exact qrel and
   hard-negative shape specified here;
4. implement deterministic full-corpus metrics, tie behavior, slices, and
   provider-valid/system-only labels using a test double;
5. reject flattening, reordering, identity loss, invalid qrels, non-finite
   vectors, mislabeled fusion, unpairable deltas, and network attempts;
6. register it only as `fixture_only`, `publish: false`, and
   `leaderboard_publish: false`; and
7. do not add or modify Gemini, DashScope, Voyage, Volcengine, Qwen, or Fusion
   adapters; do not call a provider or download a model/media dataset.

Provider adapters should be separate later items, one provider at a time, after
the contract is accepted.

## 17. Source and Revision Ledger

### MTEB release and evaluator

- Tag `2.18.6` ->
  `fa36ee7a6274bba288a925259c74e3382b53532d`:
  <https://github.com/embeddings-benchmark/mteb/tree/2.18.6>
- Task metadata and modality-category parsing:
  <https://github.com/embeddings-benchmark/mteb/blob/fa36ee7a6274bba288a925259c74e3382b53532d/mteb/abstasks/task_metadata.py>
- Retrieval wrapper and ranking path:
  <https://github.com/embeddings-benchmark/mteb/blob/fa36ee7a6274bba288a925259c74e3382b53532d/mteb/models/search_wrappers.py>
- Retrieval metrics and `pytrec_eval` path:
  <https://github.com/embeddings-benchmark/mteb/blob/fa36ee7a6274bba288a925259c74e3382b53532d/mteb/_evaluators/retrieval_metrics.py>

### Exact MTEB task commits

- MomentSeeker, `9082aa7d6e02047975701466dfb12d2d36898214`:
  <https://github.com/embeddings-benchmark/mteb/blob/9082aa7d6e02047975701466dfb12d2d36898214/mteb/tasks/retrieval/eng/moment_seeker.py>
- CLD, `3e4ec5254f5a017329aa567c43616c93af291c93`:
  <https://github.com/embeddings-benchmark/mteb/blob/3e4ec5254f5a017329aa567c43616c93af291c93/mteb/tasks/retrieval/eng/cld_at2a_retrieval.py>
- SpeechCOCO, `e8f0b218192c39dbc92c421cbb6ad697f6560972`:
  <https://github.com/embeddings-benchmark/mteb/blob/e8f0b218192c39dbc92c421cbb6ad697f6560972/mteb/tasks/retrieval/eng/speech_coco.py>
- VSC2022, `80f6e3b89539a433589ac4a685d3a9f86e4f0f10`:
  <https://github.com/embeddings-benchmark/mteb/blob/80f6e3b89539a433589ac4a685d3a9f86e4f0f10/mteb/tasks/retrieval/zxx/vsc2022_retrieval.py>
- VimSketch, `9d1529723c36c6f60a9764df0a15d436a81736ad`:
  <https://github.com/embeddings-benchmark/mteb/blob/9d1529723c36c6f60a9764df0a15d436a81736ad/mteb/tasks/retrieval/zxx/vim_sketch_retrieval.py>
- SoundingEarth, `81958bba2d7a60518e9269c5f01c09c8c0042ec6`:
  <https://github.com/embeddings-benchmark/mteb/blob/81958bba2d7a60518e9269c5f01c09c8c0042ec6/mteb/tasks/retrieval/eng/sounding_earth.py>

### Model and provider behavior

- Qwen3-VL-Embedding source, `393e2978d27852b0d0230d6994f37f9c15bed73c`:
  <https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/README.md>
- MTEB Fusion Embedding wrapper, `796793983bc6f142e70ffe176cc224f0c22a16d5`:
  <https://github.com/embeddings-benchmark/mteb/blob/796793983bc6f142e70ffe176cc224f0c22a16d5/mteb/models/model_implementations/fusion_embedding_models.py>
- Gemini embeddings and aggregation:
  <https://ai.google.dev/gemini-api/docs/embeddings>
- Gemini pricing:
  <https://ai.google.dev/gemini-api/docs/pricing>
- DashScope multimodal embedding API:
  <https://www.alibabacloud.com/help/en/model-studio/multimodal-embedding-api-reference>
- Voyage multimodal embeddings:
  <https://docs.voyageai.com/docs/multimodal-embeddings>
- Voyage pricing:
  <https://docs.voyageai.com/docs/pricing>
- Volcengine/Doubao embedding request contract:
  <https://www.volcengine.com/docs/82379/1409291>

### Upstream dataset provenance

- MomentSeeker upstream repository and license statement:
  <https://github.com/yhy-2000/MomentSeeker>
- ADIFF paper:
  <https://arxiv.org/abs/2502.04476>
- SpeechCOCO Zenodo record, CC-BY-4.0:
  <https://zenodo.org/records/4282267>
- VSC2022 paper:
  <https://arxiv.org/abs/2306.09489>
- VimSketch Zenodo v1.0, CC-BY-4.0:
  <https://zenodo.org/records/2596911>
- SoundingEarth Zenodo v1.0, CC-BY-4.0:
  <https://zenodo.org/records/5600379>

## 18. Final Decision Record

**DIFFERENTIATE is accepted by this minispec.**

The repository needs a provider-neutral composed-media v0 because current flat
input, cache, and task contracts cannot represent one ordered heterogeneous
logical item or prove that it produced exactly one provider-native embedding.
Primary sources show that this is a real embedding capability for current Qwen,
Gemini, DashScope, Voyage, and Volcengine paths, while MTEB's Fusion wrapper
demonstrates a materially different benchmark-side fusion strategy.

The contract must remain separate from ordinary media retrieval. SpeechCOCO,
VSC2022, VimSketch, SoundingEarth, and other independently encoded audio/video
tasks can move forward with explicit ids, qrels, and deterministic metrics
without waiting for composed-provider adapters. MomentSeeker- and CLD-shaped
tasks must wait for the composed contract and must never be approximated by
silently flattening their parts.
