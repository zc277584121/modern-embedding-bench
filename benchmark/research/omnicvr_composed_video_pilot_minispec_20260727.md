# OmniCVR Composed-Video Pilot Minispec - 2026-07-27

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_7-1785159379-2_execute.md`

Selected item: `tasks/omnicvr-composed-video-pilot-minispec`

Session: `meb-modern-embedding-leaderboard-7-1785159379-2-omnicvr-minispec-bbc6c7d7c7bc`

Repository baseline: `1031450fb6c56f95756e557b464058be9c82f2ba`

Decision: **PAUSE.**

OmniCVR is a relevant composed-media direction, but this repository should not
authorize even a bounded real-data pilot yet. The exact pinned Hugging Face
revision could not be resolved through the normal `huggingface.co` hostname,
the official OpenReview paper page and PDF were blocked by browser verification,
the open MTEB adaptation has no descriptive-statistics artifact, and its
global-corpus conversion is explicitly not comparable to the paper's
per-query-gallery protocol. In addition, the current MTEB query shape and the
most immediate model wrapper path do not prove faithful handling of OmniCVR's
acoustic and integrated modification slices.

`PAUSE` is one decision. It is not a rejection of the task and it is not
permission to run a smoke. The next justified work is a separate metadata-only
source contract after normal Hugging Face and official paper access are both
available.

Labels used below:

- **VERIFIED**: checked in this repository or a current primary source on
  2026-07-27.
- **UPSTREAM CLAIM**: stated by an upstream author or maintainer but not
  independently reproduced in this work.
- **DERIVED**: arithmetic or a direct consequence of verified code and stated
  upstream counts.
- **UNKNOWN**: unavailable from the allowed normal primary-source paths.
- **PROPOSAL**: the repository contract to consider only after the pause gates
  clear.
- **GATE**: a condition that must pass before implementation, execution, or
  publication.

## 1. Decision in One Screen

| Question | Finding | Consequence |
| --- | --- | --- |
| Is the upstream task current? | **VERIFIED.** MTEB issue `#4994`, PR `#5036`, and companion model PR `#5035` are open. | Relevant research lead, not accepted benchmark evidence. |
| Is the exact task implementation pinned? | **VERIFIED.** PR `#5036` head is `c0622ee67c974ecad5aca48a57c29e07a3946886`. | Code semantics can be audited immutably. |
| Is the dataset pinned? | **VERIFIED in code only.** The task names `Jun-Yang/OmniCVR@81f254d1e5993dfec408fa111990150c32c3e50f`. | The intended revision is known, but its contents are not verified. |
| Can the pinned dataset be verified normally? | **VERIFIED failure.** Normal `huggingface.co` DNS failed with curl exit `6` and HTTP `000`. | Dispatch requires `PAUSE`; no alternate host, IP, mirror, token, or cache may be used. |
| Can the paper be checked directly? | **VERIFIED failure.** OpenReview API/PDF returned `403`, and a normal Chrome page required an interactive verification challenge. | Paper metadata, tables, exact metrics, source provenance, and rights statements remain partly unknown. |
| Are MTEB descriptive stats present? | **VERIFIED no.** PR `#5036` changes only the task module and its import; the PR says stats are still being generated. | Exact counts, bytes, duration, codec, duplicates, and slice distributions are not independently checked. |
| Is the MTEB score paper-comparable? | **VERIFIED no.** The task description says the approximately 16,316-video global corpus is not directly comparable to each paper query's 2,000-video gallery. | Never place the scores in one comparable series. |
| Is audio semantics preserved? | **UNKNOWN / currently unproven.** The dataset is described as visual, acoustic, and integrated, but the task exposes only `video` plus `text`; the companion video wrapper samples frames and does not consume an audio feature. | Acoustic and integrated slices need an explicit audio-stream contract before a faithful pilot. |
| Does the host fit the raw pool? | **DERIVED yes for storage, unknown for runtime.** About 290 GB is free, but exact cache expansion, decode cost, duration, and model runtime are unknown. | Capacity is not the blocker and must not be mistaken for authorization. |

The current evidence supports a metadata-first follow-up, not a media fetch,
model run, provider call, task patch, or public result.

## 2. Current Official MTEB State

### 2.1 Release and issue

**VERIFIED.** The latest MTEB release is `2.18.7`, published on
2026-07-26. OmniCVR is not in that release; it is proposed by open issue
`#4994` and open PR `#5036`.

**VERIFIED.** Issue `#4994`, `Add dataset: OmniCVR`, was opened on
2026-07-23, remains open with label `new dataset`, and links the Hugging Face
dataset and OpenReview forum.

The issue makes these **UPSTREAM CLAIMS**:

- 5,000 evaluation queries;
- one source video plus one natural-language modification instruction per
  query;
- one target video to retrieve;
- a separate 2,000-video candidate gallery for every query, containing the
  target; and
- visual, audio, and integrated multimodal modifications.

These counts and categories could not be checked against the pinned annotation
file because the normal Hugging Face hostname did not resolve.

### 2.2 PR `#5036`

**VERIFIED.** Current PR state:

| Field | Value |
| --- | --- |
| PR | `embeddings-benchmark/mteb#5036` |
| Title | `Add OmniCVR composed video retrieval task` |
| State | open, non-draft |
| Head | `c0622ee67c974ecad5aca48a57c29e07a3946886` |
| Base at inspection | `f38c9692061e46664e3aa8e50fcf2a3628f1a55d` |
| Mergeability | `mergeable=true`, `mergeable_state=unstable` |
| Changed files | 2 |
| GitHub check runs | 0 reported |
| Combined commit status | pending with 0 reported statuses |

The only changed files are:

1. `mteb/tasks/retrieval/eng/__init__.py`, adding the import/export; and
2. `mteb/tasks/retrieval/eng/omni_cvr.py`, adding 120 lines of task code.

No descriptive-statistics JSON is in the PR. The PR body says it is being
generated locally because the pool is approximately 27 GB and requires
`torchcodec`.

### 2.3 Exact task metadata

**VERIFIED at immutable head.** `OmniCVRVT2VRetrieval` declares:

| Field | Value |
| --- | --- |
| Dataset path | `Jun-Yang/OmniCVR` |
| Dataset revision | `81f254d1e5993dfec408fa111990150c32c3e50f` |
| Annotation file | `omnicvr.jsonl` at the same revision |
| Task type | `Any2AnyRetrieval` |
| Category | `vt2v` |
| Modalities | `video`, `text` |
| Split exposed to MTEB | `test` |
| Main score | `ndcg_at_10` |
| Task license field | `cc-by-4.0` |
| Annotation creator | `human-annotated` |
| Sample creation | `found` |
| Beta flag | `true` |

The task's own description states that the paper uses a per-query 2,000-video
gallery, while MTEB takes the union of candidate ids, deduplicates that union,
and evaluates all queries against one shared corpus of approximately 16,316
videos. It also states that absolute scores are not directly comparable to the
paper because the corpus is about eight times larger.

### 2.4 Companion PR `#5035`

**VERIFIED.** The immediate companion model proposal is open PR `#5035` at
head `16acbe08896fc16714669655e00cdcf4614d1632`. GitHub reports
`mergeable=false` and `mergeable_state=dirty`.

The PR adds LanguageBind video, audio, image, and omni wrappers plus an optional
dependency. Its video path:

- requires FFmpeg according to the wrapper documentation;
- uses `VideoCollator` and decoded frame tensors;
- defaults to 8 sampled frames;
- embeds text and video separately and sums the vectors when both fields are
  present; and
- reports a pinned video checkpoint revision
  `13f52c20ce666a7d017bcd00522039f4ab034a66`, about 427.6 million
  parameters, and model memory metadata of 1,631 MB.

Under this repository's accepted labels, that sum is
`benchmark_system_fusion`, not `provider_valid_embedding`. The wrapper does not
turn an MP4 audio track into an explicit audio feature. The Omni wrapper can
sum an `audio` feature when one exists, but PR `#5036` exposes no separate audio
column.

## 3. Official Paper and Dataset Access State

### 3.1 OpenReview

The official reference is:

<https://openreview.net/forum?id=KxxR7emO5K>

**VERIFIED access failure.** On 2026-07-27:

- `api2.openreview.net/notes?forum=KxxR7emO5K` returned `403`;
- `api.openreview.net/notes?forum=KxxR7emO5K` returned `403`;
- the official PDF and attachment endpoints returned `403`; and
- the normal page in the user's Chrome session redirected to an interactive
  `Verifying your browser` challenge.

No challenge was solved, no login was attempted, and no alternate paper host
was used.

The immutable MTEB task code carries the following **UPSTREAM CLAIMS** from its
bibliography block: the title is *OmniCVR: A Benchmark for Omni-Composed Video
Retrieval with Vision, Audio, and Text*, it is attributed to Junyang Ji and
coauthors, and it is listed for ICLR 2026. Those claims are not a substitute
for checking the paper body, appendix, datasheet, or supplementary material.

The paper's exact retrieval metrics, reported scores, candidate construction,
source-dataset ledger, participant/privacy statements, duplicate policy, and
media redistribution terms remain **UNKNOWN** here.

### 3.2 Pinned Hugging Face revision

The intended immutable source is:

<https://huggingface.co/datasets/Jun-Yang/OmniCVR/tree/81f254d1e5993dfec408fa111990150c32c3e50f>

**VERIFIED access failure.** Token-free requests to the revision page,
annotation file, and Dataset API revision endpoint all failed before HTTPS:

```text
curl: (6) Could not resolve host: huggingface.co
http_code=000
```

No alternate Hugging Face host, hard-coded IP, mirror, proxy, token-bearing
request, or cached private artifact was used.

Therefore all of the following are **UNKNOWN**:

- whether revision `81f254d...` currently resolves anonymously;
- the Dataset card text and its license scope;
- exact file tree, LFS objects, shard layout, and total bytes;
- exact `omnicvr.jsonl` bytes, hash, row count, and schema;
- exact video-key count and whether every referenced id exists once;
- whether the reported approximately 27 GB is decimal GB, GiB, compressed
  transfer, repository size, decoded cache size, or another measurement;
- duration, frame rate, resolution, container, codec, and audio-stream
  distributions;
- visual/audio/integrated slice counts and row labels;
- source-video provenance and per-asset rights;
- face, voice, location, bystander, minor, and sensitive-content prevalence;
  and
- exact duplicate and near-duplicate media status.

This inaccessible pinned revision is independently sufficient for `PAUSE`
under the dispatch contract.

## 4. Exact MTEB Adaptation Semantics

### 4.1 Query construction

**VERIFIED in task code.** The loader:

1. loads `omnicvr.jsonl` as a JSON dataset;
2. loads the pinned `Jun-Yang/OmniCVR` train split;
3. maps each video `__key__` to `<key>.mp4`;
4. reads `source_id`, `target_id`, `instruction`, and `candidates` from every
   annotation row;
5. selects each source video by `source_id`;
6. assigns query ids as decimal row positions `0..N-1`; and
7. emits query fields `id`, `video`, and `text`.

The query id is deterministic only while row order and revision remain fixed.
It is not a content-derived semantic id. A later repository source contract
should preserve the upstream row id if present and derive a stable hash from
the pinned annotation identity, source id, and exact instruction.

### 4.2 Corpus construction

**VERIFIED in task code.** The corpus id list is:

```text
sorted(union(all annotation candidate ids))
```

The code then selects those video rows, renames `mp4` to `video`, and exposes
`id` plus `video`.

Important limits of this deduplication:

- it deduplicates candidate **id strings**, not content hashes;
- distinct ids with byte-identical or near-identical video remain separate;
- duplicate `__key__` rows are silently collapsed by a Python dictionary to
  the last index;
- source videos are not automatically corpus items unless they also occur in
  a candidate list; and
- no content-level duplicate or leakage report is produced.

The approximately 16,316 corpus count is an **UPSTREAM CLAIM** in the task
description, not a checked statistic in this work.

### 4.3 Qrels

**VERIFIED in task code.** Each query has exactly one proposed qrel:

```text
query row id -> target_id with relevance 1
```

The code does not explicitly validate that:

- every target id is in that row's 2,000 candidates;
- every target id exists in the video key map;
- a byte-identical duplicate target under another id is also labeled positive;
- repeated source/instruction/target triples are absent; or
- another semantically valid candidate becomes a false negative in the global
  union.

Those properties must be checked against the pinned annotation and media
metadata before a pilot.

### 4.4 Cardinality and bytes

| Quantity | Status | Value |
| --- | --- | ---: |
| Paper evaluation queries | **UPSTREAM CLAIM** | 5,000 |
| Paper candidates per query | **UPSTREAM CLAIM** | 2,000 |
| MTEB query rows | **UPSTREAM CLAIM** | 5,000 |
| MTEB global corpus | **UPSTREAM CLAIM** | approximately 16,316 |
| Positive qrels | **DERIVED if 5,000 rows hold** | 5,000 |
| Positives per query | **VERIFIED code shape** | 1 |
| Pool size | **UPSTREAM CLAIM** | approximately 27 GB |
| Annotation bytes | **UNKNOWN** | unknown |
| Dataset repository/LFS bytes | **UNKNOWN** | unknown |
| Decoded/cache bytes | **UNKNOWN** | unknown |
| Per-video bytes and durations | **UNKNOWN** | unknown |

Using the claimed decimal values only:

- the global corpus is `16,316 / 2,000 = 8.158x` the paper gallery size;
- `27,000,000,000 / 16,316` is about 1.655 MB per claimed corpus video on
  average, but this is not a download estimate because the numerator's scope is
  unknown;
- a 5,000 by 16,316 float32 similarity matrix is about 326.3 MB;
- 16,316 corpus embeddings are about 66.8 MB at 1,024 dimensions, 167.1 MB at
  2,560 dimensions, or 200.5 MB at 3,072 dimensions; and
- 5,000 query embeddings are about 20.5 MB, 51.2 MB, or 61.4 MB at the same
  dimensions.

The embedding and score matrices fit host RAM. Video decoding and model
inference, not vector storage, dominate the unknown resource envelope.

### 4.5 Metric and slices

**VERIFIED.** The task's primary metric is `ndcg_at_10`.

With one relevance-1 target, binary nDCG@10 is a discounted function of that
one target's rank through rank 10. It is not automatically the same claim as a
paper recall metric, median rank, or any metric computed within a different
candidate set.

**VERIFIED.** PR `#5036` does not preserve a visual/audio/integrated category
field in the query table. It emits only `id`, `video`, and `text`. No
modification-family slices, duration slices, codec slices, source-dataset
slices, or duplicate slices are defined by the task code. The annotation may
contain more columns, but that is **UNKNOWN** until the pinned file is readable.

## 5. Gallery-to-Global-Corpus Comparability

The MTEB conversion is a new evaluation protocol, not a larger implementation
of the same protocol.

| Property | Paper protocol, upstream claim | MTEB PR `#5036` |
| --- | --- | --- |
| Candidate set | Query-specific gallery | One shared corpus |
| Candidate count | 2,000 per query | approximately 16,316 for every query |
| Positive labels | Target in each gallery | One target id per query |
| Negatives | Curated/sampled for that query | Union of every query's candidates |
| Extra valid candidates | Governed by paper gallery construction | Potentially introduced and unlabeled |
| Primary score | **UNKNOWN here** because paper body was blocked | `ndcg_at_10` |

Consequences:

1. **DERIVED.** Candidate cardinality changes by about 8.158x.
2. **DERIVED.** Every query receives negatives that were not selected for its
   original gallery.
3. **UNKNOWN.** Some union-only candidates may satisfy a query but remain
   relevance 0 because only `target_id` is labeled.
4. **DERIVED.** Even identical embeddings can receive a different target rank
   after the candidate set changes.
5. **VERIFIED upstream warning.** The task description explicitly says the
   absolute scores are not directly comparable.

No primary evidence inspected here provides a calibration, rank correlation,
or transformation between paper-gallery scores and MTEB-global scores.
Therefore:

- never report a global-corpus result as a reproduction of a paper score;
- never mix the two in one leaderboard series or delta;
- version them as separate task semantics if both are eventually supported;
  and
- require full candidate and qrel identity in every public result.

## 6. Modality Fidelity: Visual, Acoustic, and Integrated Changes

This is a second independent pause gate.

**UPSTREAM CLAIM.** OmniCVR includes visual, acoustic, and integrated
modifications.

**VERIFIED.** The MTEB task exposes the source as a `Video` feature and the
instruction as text. It does not expose a separate source-audio feature,
target-audio feature, modification type, or audio transform identity.

**VERIFIED.** The companion LanguageBind video wrapper decodes video frames,
samples 8 frames by default, and uses the visual tower. Its text+video result is
a sum of separately produced vectors. It consumes audio only when the dataset
has an explicit `audio` feature, which this task does not.

**VERIFIED from current official model documentation.** Other plausible video
embedding paths also need careful classification:

- Gemini Embedding 2 can aggregate multiple parts into one embedding, but its
  documented video path samples at most 32 frames and does not process the
  audio track embedded in a video file.
- DashScope `qwen3-vl-embedding` can create one fused vector from text, images,
  and video when `enable_fusion=true`, but the checked documentation did not
  establish source-video audio handling.
- Qwen3-VL-Embedding accepts mixed text/image/video input, but its documented
  input modalities do not include audio.
- Qwen3-VL-Reranker can score a mixed query/document pair, but that score is a
  reranker system result, not an independently indexable embedding.

A faithful acoustic or integrated query may need explicit, pinned extraction
of the source and target audio streams and a logical item such as:

```text
query  = ordered(source_visual_stream, source_audio_stream, instruction)
corpus = ordered(target_visual_stream, target_audio_stream)
```

That shape is compatible with this repository's provider-neutral composed-item
idea, but it is not the shape proven by PR `#5036`. Extraction would introduce
new decoder, stream-selection, duration, synchronization, normalization, and
rights identities. It must not be silently added during a benchmark run.

Until the paper and pinned data establish exactly how acoustic and integrated
queries are intended to be encoded and scored, a visual-only route would be an
unlabeled task mutation.

## 7. Comparison with Accepted Repository Contracts

### 7.1 Accepted composed-media fixture

**VERIFIED.** `composed_media_retrieval_minispec_20260722.md` and the accepted
implementation establish:

- one ordered heterogeneous logical item produces exactly one embedding;
- provider-native fusion, benchmark-side fusion, reranking, and fixture-only
  evidence are separate labels;
- no silent flattening, sum, mean, dropped part, caption substitution, or
  reranker substitution is allowed;
- exact part, item, request, result, media, route, and preprocessing identity is
  hashed;
- full-corpus cosine ranking uses deterministic UTF-8 corpus-id tie breaks;
- primary `composed_ndcg@10` is accompanied by MAP, recall, MRR, hit-rate,
  positive-set, hard-negative, and slice diagnostics; and
- the current self-authored fixture is always no-publish.

The accepted fixture contains 12 queries, 12 corpus items, 16 graded positive
qrels, 36 pre-reviewed hard negatives, and seven hard-negative families. It
tracks 1,411,262 bytes and contains no external media or network dependency.

OmniCVR is directionally compatible with the `text_video_to_video` family, but
it does not yet satisfy the accepted real-data requirements:

- no verified source bytes or content hashes;
- no verified stable row/media ids;
- no verified part/stream transform identity;
- no explicit audio part for acoustic/integrated cases;
- no content-level duplicate policy;
- no deterministic repository-owned hard-negative audit;
- no rights/provenance ledger; and
- no score-comparability boundary beyond the warning text.

### 7.2 Song Describer T2A

**VERIFIED.** `song_describer_real_audio_pilot_minispec_20260727.md` reached
`GO, T2A only` because the paper direction, 1,106 captions, 706 tracks, one
positive per query, Zenodo release, exact archive manifest, and 706-row
per-track license ledger were available from primary sources. Even that task is
paused before materialization while the pinned MTEB Dataset revision cannot be
checked normally.

OmniCVR is less ready than Song Describer:

| Property | Song Describer T2A | OmniCVR current state |
| --- | --- | --- |
| Query shape | Flat text | Composed source media plus instruction |
| Paper access | Verified arXiv version | OpenReview blocked here |
| Origin archive | Verified Zenodo record and files | Unknown |
| Per-asset rights | Verified heterogeneous 706-row ledger | Unknown |
| Exact MTEB stats | Present in release | Missing from open PR |
| Pinned HF verification | Blocked | Blocked |
| Media semantics | Audio route specified, transform still gated | Visual/audio/integrated stream semantics unproven |
| Protocol comparability | Paper-native T2A selected | Paper gallery changed to global corpus |

The Song Describer decision therefore does not justify relaxing any OmniCVR
gate.

## 8. Provenance, Rights, Privacy, and Publication

The MTEB task's single `cc-by-4.0` field is not sufficient evidence that every
source or target video may be downloaded, transformed, embedded by a provider,
redistributed, or shown in a public Space.

The following must be distinguished:

1. annotation and instruction text copyright;
2. Dataset card and packaging license;
3. underlying source-video copyright and platform terms;
4. music, speech, and other audio rights inside the videos;
5. performer, creator, uploader, and attribution obligations;
6. consent and privacy for identifiable people, faces, voices, locations, and
   bystanders;
7. rights to create decoded frames, audio extracts, thumbnails, and clips; and
8. rights to send media to a third-party provider and to publish derived
   embeddings or examples.

All are **UNKNOWN** from the allowed accessible evidence.

### 8.1 Required rights ledger

Before any media fetch, every selected source, target, and negative needs at
least:

```json
{
  "video_id": "<stable upstream id>",
  "origin_uri": "<primary source>",
  "origin_revision": "<immutable revision>",
  "source_video_license": "<exact license>",
  "audio_rights": "<exact or separately unknown>",
  "attribution": "<required credit>",
  "redistribution_allowed": false,
  "transform_allowed": false,
  "provider_upload_allowed": false,
  "contains_person_or_face": "unknown",
  "contains_voice": "unknown",
  "privacy_review": "pending"
}
```

An `unknown` in any field relevant to the proposed operation blocks that
operation. This is a conservative repository policy, not legal advice.

### 8.2 Public eligibility

No OmniCVR public score, Dataset, Space example, thumbnail, clip, audio preview,
or media mirror is eligible until:

- the paper and pinned Dataset revision are directly verified;
- the complete source and per-asset rights chain is recorded;
- face/person/voice/privacy review is complete;
- duplicate and false-negative gates pass;
- exact task semantics and candidate set are versioned;
- media transforms and decoder/tool versions are reproducible;
- the model route faithfully handles the required modality slice;
- the full task, not a smoke subset, is evaluated; and
- paper-gallery and MTEB-global results are displayed as separate tasks.

## 9. Codec, Transform, and Dependency Contract

**UPSTREAM CLAIM.** PR `#5036` says the approximately 27 GB pool requires
`torchcodec` for descriptive statistics.

**VERIFIED.** PR `#5035` adds an optional LanguageBind dependency path, a
compatibility shim for current `torchvision`, `VideoCollator`, and an FFmpeg
requirement. None of those changes are accepted in MTEB main at the inspected
time.

Exact OmniCVR media properties remain **UNKNOWN**, so no canonical transform
can be selected yet. A future source contract must record:

- container and stream inventory;
- video codec/profile, width, height, frame rate, time base, duration, and
  rotation;
- audio codec, sample rate, channel count, duration, and synchronization;
- decoder library and exact version;
- source byte hash and byte length;
- decoded visual/audio transform ids and hashes;
- frame sampling or temporal chunking policy;
- behavior for variable frame rate, missing audio, corrupt media, and multiple
  streams; and
- whether the target representation includes both visual and audio streams.

No route may silently:

- strip audio from an acoustic/integrated item;
- use only a visually convenient slice;
- take the first frames without a declared policy;
- resample, normalize, crop, transcode, or caption media without a new
  transform identity;
- trust remote code; or
- substitute provider-specific preprocessing for benchmark truth.

## 10. Machine Fit and Small-First Credibility

### 10.1 Current host

**VERIFIED on 2026-07-27.** The host has:

- 290 GB free on `/data2`;
- 503 GiB RAM total and 470 GiB available at inspection; and
- four NVIDIA GeForce RTX 3080 Ti GPUs, each with 12,288 MiB total, 4 MiB used,
  and 0% utilization.

### 10.2 Full claimed pool

The claimed approximately 27 GB transfer is only about 9.3% of current free
disk. Storage therefore appears feasible in isolation. A realistic materialized
workflow may temporarily retain:

- source shards or archives;
- Hugging Face cache objects;
- extracted containers;
- decoded frame/audio caches;
- model weights;
- embeddings; and
- logs and manifests.

Without the file tree and codec/duration distribution, a conservative working
set could be multiple times the claimed pool. Even 54-100 GB would fit current
disk, but that is a planning bound, not a verified requirement and not
permission to download.

RAM is ample for embeddings and score matrices. GPU fit is route-dependent:

- Qwen3-VL-Embedding-2B has approximately 2 billion parameters; BF16 parameter
  bytes alone are roughly 4 GB, making a single 12 GB card plausible before
  activations and runtime overhead.
- The 8B model needs roughly 16 GB for BF16 parameter bytes alone and is not a
  clean single-card first path without quantization or sharding.
- The proposed LanguageBind video wrapper reports 1,631 MB model memory and 8
  sampled frames, but its full dependency, decoder, activation, and throughput
  envelope is not verified here.

Four GPUs do not make the task automatically small. Video decode, frame
sampling, model batching, shared cache pressure, and serial data loading may
dominate. No authoritative per-video runtime exists in the inspected sources,
so full-pool wall time remains **UNKNOWN**.

### 10.3 Why an eight-query exact score is not small

For the MTEB-global protocol, even one query is ranked against the entire
approximately 16,316-video corpus. An eight-query exact smoke would still need
the full corpus and nearly the full reported media pool.

For the paper protocol, eight query-specific galleries contain 16,000 candidate
occurrences. Their unique union could approach the full MTEB corpus. Exact
gallery evaluation is therefore also not credibly tiny without verified shard
locality and overlap statistics.

A reduced candidate pool can validate schemas, decoding, routing, and failure
behavior, but its score is neither a paper-gallery score nor an MTEB-global
score. That distinction must be explicit.

## 11. Plausible Model Tracks

These are candidates for later source-contract work, not runnable evidence.

### 11.1 Provider-native composed embedding

| Candidate | Primary-source evidence | Correct label | Blocking facts |
| --- | --- | --- | --- |
| Gemini Embedding 2 | Official docs say multiple parts can produce one aggregated embedding in a shared text/image/video/audio/document space. | `provider_valid_embedding` only if all required parts are sent together. | Video audio tracks are not processed; max video duration is 120 seconds and at most 32 frames are sampled. Separate audio extraction, privacy, billing, and exact route need review. |
| DashScope `qwen3-vl-embedding` | Official docs say `enable_fusion=true` creates one fused vector from mixed text/image/video content. | `provider_valid_embedding` only with explicit fusion route evidence. | Checked docs do not establish audio-track handling; current repository adapter uses flat one-item routing and does not enable fusion. |
| Qwen3-VL-Embedding-2B local | Official repository at `393e2978d27852b0d0230d6994f37f9c15bed73c` accepts mixed text/image/video objects and emits one vector; code license is Apache-2.0. | `provider_valid_embedding` for a reviewed local mixed-input route. | No audio modality, no accepted repository adapter, no model/source bytes, and no normal HF access. |

No inspected provider-native candidate currently proves faithful visual plus
acoustic plus instruction encoding from the pinned OmniCVR media.

### 11.2 Benchmark-side fusion

The LanguageBind wrapper in PR `#5035` is the most immediate local comparator.
It independently embeds present modalities and sums their vectors. Under the
accepted repository contract it must be labeled:

```text
composition_mode = benchmark_system_fusion
track_label      = benchmark_system_fusion
```

It must not populate a provider-native row. The open PR is dirty, its Omni
wrapper composes three separately loaded checkpoints, and its use of an audio
branch still requires an explicit extracted `audio` feature that PR `#5036`
does not provide.

### 11.3 Reranking

The official Qwen3-VL repository documents 2B and 8B rerankers that accept a
mixed query/document pair and return pointwise relevance scores. A later
two-stage experiment could use an embedding model for recall and rerank a small
top-k set.

Correct label:

```text
score_validity = reranker_system_only
```

It is not an embedding score, cannot replace full-corpus embedding recall, and
must never be compared as the same row as provider-native or benchmark-fusion
embeddings.

## 12. Metadata-First Resume Contract

This is the only justified next step after the current pause gates clear.

### 12.1 Resume conditions

All must be true:

1. token-free DNS and HTTPS for the normal `huggingface.co` hostname succeed;
2. `Jun-Yang/OmniCVR@81f254d1e5993dfec408fa111990150c32c3e50f` is
   anonymously readable;
3. the OpenReview forum and paper PDF are accessible without login or an
   unsolved interactive challenge; and
4. PR `#5036` head and state are refreshed before relying on its code.

### 12.2 Inspection scope

The metadata-only item should record:

- exact paper version, paper metrics, paper score tables, gallery construction,
  dataset sources, license, privacy, and supplementary statements;
- exact Dataset card, file tree, LFS/shard bytes, file hashes, and schema;
- SHA-256 and byte length for `omnicvr.jsonl`;
- exact row, source-id, target-id, candidate-id, and video-key counts;
- exactly 2,000 candidates per query and target membership, or every exception;
- exact union-corpus cardinality;
- stable query/corpus/qrel id proposal;
- visual/audio/integrated slice counts and definitions;
- source/target/candidate duplicate ids and available content-hash evidence;
- source-video provenance, license scope, attribution, redistribution, transform,
  provider-upload, and privacy fields;
- exact media bytes by shard and whether a selected row can be fetched without
  pulling a large unrelated shard; and
- codec/duration/audio-stream distributions from existing metadata only.

Hard caps for that future metadata item:

- network: 20 MiB;
- temporary disk: 50 MiB;
- wall time: 30 minutes;
- media bytes: 0;
- model bytes: 0;
- provider/API cost: USD 0; and
- cleanup on pass, failure, timeout, or interruption.

Any unexplained count, source, rights, schema, or revision mismatch returns
`BLOCKED`; it must not be repaired heuristically.

## 13. Conditional At-Most-Eight-Query No-Publish Smoke

No smoke is authorized by this note. The shape below is only the smallest one
worth considering after the metadata source contract passes and after one
modality-faithful model route is separately approved.

### 13.1 Selection

- at most eight queries;
- cover visual, acoustic, and integrated slices when those labels are verified;
- select within each slice by ascending SHA-256 of the stable query id;
- include each selected target;
- include exactly three pre-reviewed negatives per query from that query's
  original paper gallery;
- use visual-near, acoustic-near, instruction-inversion, or integrated-conflict
  families as applicable;
- select negatives before any evaluated model score is visible; and
- reject rather than replace a selected row when rights, identity, codec,
  duration, audio, duplicate, or privacy metadata is incomplete.

The bounded pool would contain at most eight source videos and at most 32
candidate occurrences before content deduplication. It is a route/contract
smoke, not a retrieval-quality sample.

### 13.2 Data shape

Each query must preserve:

- stable query id and annotation hash;
- exact source container id and source byte hash;
- exact instruction bytes;
- visual/audio/integrated slice;
- ordered visual, audio, and text parts actually consumed by the route;
- transform and decoder identities; and
- one exact target qrel.

Each corpus item must preserve its target/negative source id, byte hash, rights,
stream identities, transforms, and review label. Content-identical corpus items
cannot be left as false negatives.

### 13.3 Metrics and determinism

Report only:

- target rank;
- bounded-pool `ndcg@10`;
- `recall@1`, `MRR@10`, and hit rate;
- hard-negative outrank rate;
- per-slice counts and metrics; and
- route/cardinality/cleanup pass or failure.

Rank by exact descending score and then `corpus_id` ascending by UTF-8 bytes.
No score is rounded before ranking. Repeat serialization and ranking to confirm
identical ids, bytes, and ties.

Every result must carry:

```text
publish = false
leaderboard_publish = false
evidence_tier = smoke
score_validity = smoke_only
candidate_protocol = bounded_contract_pool
```

It must not be called a paper-gallery or MTEB-global score.

### 13.4 Resource caps

The eventual smoke should stop before transfer if its verified selected media
cannot fit all caps:

- at most 8 source videos and 32 candidate occurrences;
- at most 256 MiB media transfer;
- at most 2 GiB temporary disk including decode outputs;
- at most 16 GiB process RSS;
- at most one 12 GB GPU and 11 GiB measured VRAM;
- at most 45 minutes total wall time;
- one approved model/provider route, no fallback;
- provider cost at most USD 0.20 if a separately approved API route is used;
- no automatic billable retry;
- no publication or media retention; and
- cleanup on pass, failure, timeout, or interruption.

These caps are deliberately much smaller than the full claimed pool. Failure
to obtain selected objects without downloading a large shard returns
`BLOCKED`.

## 14. Failure Gates and Eventual Product Path

### 14.1 Failure gates

A later source contract, smoke, or full task fails if any of the following is
true:

- normal official source access is unavailable;
- paper version or pinned Dataset revision is unresolved;
- exact query/corpus/qrel counts differ without explanation;
- any target is absent from its declared gallery or global corpus;
- duplicate ids, missing keys, or content duplicates create false negatives;
- visual/audio/integrated labels are missing or silently dropped;
- a model route ignores required source or target audio;
- source, rights, attribution, redistribution, transform, provider-upload, or
  privacy fields are incomplete;
- decoder or transform output is nondeterministic;
- media is silently cropped, sampled, stripped, captioned, or rewritten;
- provider-native, benchmark-fusion, and reranker labels are mixed;
- resource, cost, privacy, or cleanup caps are exceeded;
- a bounded smoke is represented as public quality evidence; or
- a global-corpus score is represented as comparable to a paper-gallery score.

### 14.2 Git

An eventual implementation may track only bounded reviewable artifacts:

- source and revision ledger;
- query/corpus/qrel reference tables;
- per-asset rights and privacy review tables;
- duplicate/leakage reports;
- transform/decoder manifests and hashes;
- task/provider route definitions; and
- full benchmark-eligible result records.

Git must not contain the approximately 27 GB pool, model weights, decoded
frames/audio, provider caches, or smoke results.

### 14.3 Hugging Face Dataset

An eventual Dataset product may publish metadata and references only after
rights review:

- stable query/corpus ids and qrels;
- source revision and immutable media references;
- instruction text only if its license permits;
- per-asset attribution, license, privacy, and redistribution fields;
- transform and model/provider provenance;
- candidate-protocol identity; and
- evidence-tier and score-validity labels.

It must not mirror source video/audio bytes, derived clips, frames, thumbnails,
or waveforms without a separate explicit rights decision.

### 14.4 Hugging Face Space

The Space may expose OmniCVR only after at least one full, public-eligible run.
It must show paper-gallery and MTEB-global task versions separately, display the
candidate count and protocol, preserve provider-native/system-fusion/reranker
labels, disclose contamination and rights state, and exclude all smoke rows.

No media playback or preview is part of the v0 product path.

## 15. Primary Source and Revision Ledger

### MTEB

- Latest release `2.18.7`:
  <https://github.com/embeddings-benchmark/mteb/releases/tag/2.18.7>
- OmniCVR issue `#4994`:
  <https://github.com/embeddings-benchmark/mteb/issues/4994>
- OmniCVR PR `#5036`:
  <https://github.com/embeddings-benchmark/mteb/pull/5036>
- Exact PR head:
  <https://github.com/embeddings-benchmark/mteb/commit/c0622ee67c974ecad5aca48a57c29e07a3946886>
- Exact task source:
  <https://github.com/embeddings-benchmark/mteb/blob/c0622ee67c974ecad5aca48a57c29e07a3946886/mteb/tasks/retrieval/eng/omni_cvr.py>
- Companion LanguageBind PR `#5035`:
  <https://github.com/embeddings-benchmark/mteb/pull/5035>
- Exact companion head and wrapper:
  <https://github.com/embeddings-benchmark/mteb/blob/16acbe08896fc16714669655e00cdcf4614d1632/mteb/models/model_implementations/language_bind_models.py>

### OmniCVR

- Official OpenReview forum, blocked by interactive verification here:
  <https://openreview.net/forum?id=KxxR7emO5K>
- Pinned Dataset revision, blocked by normal-hostname DNS here:
  <https://huggingface.co/datasets/Jun-Yang/OmniCVR/tree/81f254d1e5993dfec408fa111990150c32c3e50f>
- Pinned annotation path, not downloaded:
  <https://huggingface.co/datasets/Jun-Yang/OmniCVR/resolve/81f254d1e5993dfec408fa111990150c32c3e50f/omnicvr.jsonl>

### Model and provider evidence

- Qwen3-VL-Embedding and Reranker source at
  `393e2978d27852b0d0230d6994f37f9c15bed73c`:
  <https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/README.md>
- Qwen source code license at the same revision:
  <https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/LICENSE>
- LanguageBind source at `7070c53375661cdb235801176b564b45f96f0648`:
  <https://github.com/PKU-YuanGroup/LanguageBind/tree/7070c53375661cdb235801176b564b45f96f0648>
- Gemini Embedding 2 multimodal input and aggregation:
  <https://ai.google.dev/gemini-api/docs/embeddings>
- DashScope multimodal embedding and fusion:
  <https://www.alibabacloud.com/help/en/model-studio/embedding>

### Repository contracts

- `benchmark/research/composed_media_retrieval_minispec_20260722.md`
- `benchmark/research/song_describer_real_audio_pilot_minispec_20260727.md`
- `src/mm_embed/providers/composed_media.py`
- `src/mm_embed/tasks/composed_media_retrieval.py`
- `src/mm_embed/data/composed_media_retrieval.py`
- `tests/test_composed_media_retrieval.py`

## 16. Source-Access and Harness Notes

- Normal Hugging Face DNS failed; no workaround was used.
- OpenReview required an interactive browser verification challenge; it was not
  solved and the temporary browser tab was closed.
- One official documentation HTML response unexpectedly contained
  credential-like public frontend strings. That source channel was stopped
  immediately. The values were not saved, reused, or repeated. A fixed-marker
  scan found no such string in `.perpetuum/modern-embedding-leaderboard`,
  `benchmark/research`, or `.env`, including recently modified artifacts.
- AnySearch CLI remained paused and was not invoked.

## 17. Final Decision Record

**PAUSE is accepted by this minispec.**

OmniCVR addresses a real gap: source video plus an instruction retrieving a
target video is precisely the kind of logical composed query the repository's
accepted contract was designed to represent. The current MTEB proposal is also
useful evidence that the direction is active.

It is not ready for a real-data pilot. The pinned Dataset cannot be verified
through the required normal path, the paper body and supplementary material
were inaccessible, descriptive statistics are absent, the candidate protocol
was changed from 2,000 per-query videos to an approximately 16,316-video global
corpus, source and media rights are unknown, content-level duplicates are not
audited, and acoustic/integrated modality fidelity is not proven by the task or
the companion wrapper.

The pause resumes only with a metadata-first source contract. A later bounded
smoke may validate materialization and routing after all source, rights,
privacy, codec, stream, duplicate, and modality gates pass, but it will remain
no-publish and non-comparable to either full protocol. No task, provider,
registry, benchmark, result, Dataset, or Space work is authorized by this
decision.
