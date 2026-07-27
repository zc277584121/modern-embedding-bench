# Song Describer Real-Audio Pilot Minispec - 2026-07-27

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_5-1785155204-2_execute.md`

Selected item: `tasks/song-describer-real-audio-pilot-minispec`

Session: `meb-modern-embedding-leaderboard-5-1785155204-2-song-describer-6f3a9c1d72e4`

Repository baseline: `6fc29097a1e3c8b0471b3da87e8b39114129ec03`

Decision: **GO, with T2A as the only v0 direction.**

The implementation and any real-audio run remain gated. The smallest justified
next step is a metadata-only source contract after normal Hugging Face DNS/HTTPS
works again. This note does not implement a task, provider, registry entry,
manifest, fixture, run, result, or publishing change, and it does not alter the
accepted composed-media contract or fixture.

Labels used below:

- **VERIFIED**: checked in the repository or a current primary source on
  2026-07-27.
- **UPSTREAM CLAIM**: stated by an upstream author or maintainer but not
  independently reproduced here.
- **PROPOSAL**: the repository behavior selected by this minispec.
- **GATE**: a condition that must pass before implementation, execution, or
  publication.
- **NON-GOAL**: deliberately excluded from v0.

## 1. Decision in One Screen

Song Describer is a strong fit for this repository's first real audio-text
retrieval pilot because it is small enough to audit, uses human-written music
captions, has an archival release with an explicit per-track license ledger,
and is now an MTEB release task. It also probes a currently under-covered gap:
semantic text-to-real-music retrieval rather than text-only, image-only, or a
self-authored contract fixture.

The selected direction is **text-to-audio retrieval (T2A)**:

- 1,106 unique text-caption queries;
- 706 audio corpus items;
- exactly one positive track per caption query in the MTEB port;
- MTEB primary metric `recall_at_5`;
- the direction evaluated in the Song Describer paper; and
- a natural product shape for music search.

Audio-to-text retrieval (A2T) is not part of v0. It is a later inversion in the
MTEB contribution, not a task reported in the paper. It also has 1-5 positive
captions per audio query, so `recall@5`, hit rate, MRR, and positive-set
coverage answer materially different questions.

The decision is `GO` rather than `PAUSE` because the upstream task, archival
release, task direction, counts, duration, and rights structure are sufficiently
clear to select a design. Execution is still blocked by precise source and
rights gates; `GO` does not authorize bypassing them.

## 2. Why This Is Not the Composed-Media Fixture

**VERIFIED.** The accepted composed-media fixture validates one logical item
made from several ordered heterogeneous parts. Its provider-neutral protocol
requires one embedding per composed item, preserves part order and grouping,
and separately labels provider-native fusion and benchmark-side fusion.

Song Describer T2A is different:

| Property | Composed-media fixture | Song Describer T2A proposal |
| --- | --- | --- |
| Query | One ordered multi-part logical item | One caption string |
| Corpus item | Authored fixture media item | One real music recording |
| Required capability | `ComposedMediaEmbeddingProvider` | Existing flat one-item embedding shape |
| Data role | Contract and metric fixture | Real external evaluation data |
| Network | Forbidden | Required only for an approved source materialization |
| Publication | Always no-publish | Eligible only after full source, rights, and score gates |
| Quality claim | None | Possible only from the complete audited task |

**VERIFIED.** `EmbeddingInput` already represents one `AUDIO` item, while
`EmbeddingProvider.embed()` preserves one output row per input item. No part
grouping is needed for a caption query or an audio track. Song Describer should
therefore use an ordinary explicit-id retrieval task and must not be routed
through the composed-media protocol merely because audio is present.

**PROPOSAL.** Reuse only the accepted fixture's audit lessons:

- immutable media identity;
- explicit query, corpus, and qrel ids;
- deterministic ranking ties;
- route and transform provenance;
- score-validity labels; and
- hard separation between smoke evidence and public benchmark evidence.

## 3. Current Upstream Evidence

### 3.1 MTEB release and merge

**VERIFIED.** Annotated tag `2.18.7` points to tag object
`f4605b99feb4a4e9e68e04159bfad00bf41f690e`, which peels to commit
`794f50399472059f4b518a5ed47c274459b704f1`. The tag was created on
2026-07-26.

**VERIFIED.** MTEB PR `#4988`, titled `task: add Song Describer text-music
retrieval (t2a, a2t)`, merged on 2026-07-23. The PR head was
`6707b785484f350ad65b875a4e6f8b2d0643768d`; the merge commit was
`d206daee291975d4c17de4bf1fc6403cf8c246c5`.

The exact MTEB 2.18.7 task source declares:

| Task | Dataset path | Pinned dataset revision | Main score | MTEB license field |
| --- | --- | --- | --- | --- |
| `SongDescriberT2ARetrieval` | `dukesun99/SongDescriber-T2A` | `c58ad9b08343e56ce412d4da41accd15bffdbd6d` | `recall_at_5` | `cc-by-sa-4.0` |
| `SongDescriberA2TRetrieval` | `dukesun99/SongDescriber-A2T` | `205f679a640b5fecf9ca9ee74bb715b339d228a5` | `recall_at_5` | `cc-by-sa-4.0` |

The single MTEB license field is not sufficient evidence for the audio bytes.
The original release has a dataset-level license and a heterogeneous per-track
audio license ledger; Section 8 audits that distinction.

### 3.2 Descriptive statistics

**VERIFIED.** The release-pinned MTEB statistics report:

| Direction | Queries | Corpus | Positive qrels | Positive cardinality | Exact text uniqueness |
| --- | ---: | ---: | ---: | --- | ---: |
| T2A | 1,106 captions | 706 audio tracks | 1,106 | exactly 1 per query | 1,106 / 1,106 |
| A2T | 706 audio tracks | 1,106 captions | 1,106 | 1-5 per query, mean 1.5666 | 1,106 / 1,106 |

Both directions refer to the same 706 unique audio recordings. MTEB reports:

- total duration: `83,433.33318494898` seconds, about 23.18 hours;
- minimum duration: `31.077936507936506` seconds;
- mean duration: `118.1775257577181` seconds;
- maximum duration: `119.99453514739228` seconds;
- 682 files at 44.1 kHz and 24 files at 48 kHz.

These are descriptive statistics generated by MTEB, not verified file hashes.
The exact Hugging Face files, rows, byte counts, schemas, and per-row licenses
are unresolved while the normal hostname is unavailable.

### 3.3 Paper and archival release

**VERIFIED.** arXiv `2311.10057v3`, published 2023-11-16 and updated
2023-11-22, is titled *The Song Describer Dataset: a Corpus of Audio Captions
for Music-and-Language Evaluation*. It was accepted to the NeurIPS 2023
Machine Learning for Audio workshop.

The paper describes:

- 1,106 human-written captions for 706 MTG-Jamendo recordings;
- up to five captions per recording, written by different participants;
- 25% of recordings with more than one caption;
- audio segments up to two minutes, with 95% at two minutes;
- 320 kbps, 44.1 kHz MP3 release audio;
- an evaluation-only intent and no recommended train/dev/test split;
- a cleaned valid subset of 746 captions for 547 recordings in the paper's
  datasheet, while the immutable official repository README reports 546
  recordings for the same 746-caption subset; this one-track discrepancy is
  unresolved and does not affect the 1,106-caption full set used by MTEB; and
- a text-to-music retrieval experiment reporting R@1, R@5, R@10, and median
  rank, but no A2T retrieval experiment.

The paper's Table 5 reports CLAP T2A R@1/5/10 of `4.42 / 17.02 / 26.01`
and TTMR `2.95 / 10.19 / 17.43` on Song Describer. These paper values are
reference evidence, not scores produced by this repository.

**VERIFIED.** Zenodo record `10072001`, DOI
`10.5281/zenodo.10072001`, is version `1.0.0`, publication date
2023-11-16, record revision 8, and marked open access. Its current file
manifest includes:

| File | Bytes | Recorded checksum |
| --- | ---: | --- |
| `audio.zip` | 3,318,019,617 | `md5:2126b8facfe9468cf806c6154e09bbe5` |
| `song_describer.csv` | 186,208 | `md5:e90e9459c22bfbe69f5462dc1434d573` |
| `audio_metadata.tsv` | 97,667 | `md5:fa5d126659b08680719d189387d61e39` |
| `audio_licenses.txt` | 162,171 | `md5:cda13a080c03e14b8319430e1486ae6e` |
| `song_describer_14_04_23.mtg-jamendo.tsv` | 108,663 | `md5:3532f2df8b4c21a7ea85d9121eae244a` |

No file from the audio archive was downloaded for this note.

### 3.4 Current Hugging Face gate

**VERIFIED.** A token-disabled request on 2026-07-27 failed before HTTPS:

```text
curl: (6) Could not resolve host: huggingface.co
```

No alternate IP, hostname, mirror, proxy, cached private artifact, account, or
region was used. Therefore this note does not claim the exact MTEB-derived
dataset schema, file tree, file sizes, file hashes, card content, or audio-byte
identity.

## 4. T2A Versus A2T Claim Strength

### 4.1 T2A

T2A asks each caption query to retrieve its one associated track from 706
candidates.

Its claim is strongest because:

1. it is the direction explicitly evaluated in the Song Describer paper;
2. the paper describes music search from language as a core use case;
3. every query has exactly one positive in the MTEB port;
4. the primary metric matches MTEB and the paper's recall family; and
5. the user-facing task is easy to explain: type a music description and
   retrieve the described recording.

For one-positive qrels:

- `recall@k` is 1 if the positive is in the top k, otherwise 0;
- `hit_rate@k` is identical to `recall@k`;
- `positive_set_coverage@k` is also identical to `recall@k`;
- `MRR@k` rewards the reciprocal rank of that one positive; and
- binary `nDCG@k` rewards its log-discounted position.

The equalities should be tested, not used to publish several apparently
independent quality claims.

### 4.2 A2T

A2T asks each audio query to retrieve all captions associated with the track.
It has 706 queries, 1,106 caption candidates, and 1-5 positive captions per
query.

**VERIFIED.** In the MTEB PR discussion, the contributor explicitly stated
that A2T was an inversion not present in the paper. The same discussion gave
an illustrative `laion/larger_clap_general` result where A2T `recall@5` was
15.9 while `hit_rate@5` was 21.2. The task was changed to `recall_at_5`
because MTEB maintainers preferred alignment with the paper's metric family.

For multi-positive A2T qrels:

- `recall@5` is the fraction of all positive captions retrieved in the top 5;
- `hit_rate@5` asks only whether at least one positive caption was retrieved;
- `MRR@k` uses only the first positive rank;
- `nDCG@k` incorporates positions of multiple positives; and
- `positive_set_coverage@5` is the transparent name for the same set-coverage
  quantity as binary recall@5.

A2T is a valid later diagnostic, but it is a weaker first public claim because
the direction is not paper-native and its result depends more heavily on how
multiple captions per track are weighted.

### 4.3 Direction decision

**PROPOSAL.** Implement only T2A in v0. Do not register A2T in the same first
item, do not average the two directions, and do not call a bidirectional mean
the Song Describer primary score.

## 5. Proposed Source and Data Contract

### 5.1 Three distinct revisions

The task must retain three independently auditable revisions:

1. `task_semantics_revision`:
   MTEB tag commit `794f50399472059f4b518a5ed47c274459b704f1` and the exact
   `song_describer.py` source;
2. `dataset_port_revision`:
   `dukesun99/SongDescriber-T2A@c58ad9b08343e56ce412d4da41accd15bffdbd6d`;
3. `origin_release_revision`:
   Zenodo record `10072001`, version `1.0.0`, record revision 8, including the
   file manifest and per-track license ledger.

These revisions must not be collapsed into one generic `dataset_version`.

**GATE.** Once normal Hugging Face access returns, every MTEB port row must
crosswalk to the origin release by stable track identity and exact caption
text. Missing, added, rewritten, or unlicensed rows block implementation until
explained.

### 5.2 Query contract

Each T2A query contains:

```json
{
  "query_id": "sdd-v1:t2a:q:<caption_sha256>",
  "caption_text": "<exact normalized source caption>",
  "source_track_id": "<verified origin track id>",
  "caption_sha256": "<sha256>",
  "split": "test",
  "source_revision": "c58ad9b08343e56ce412d4da41accd15bffdbd6d"
}
```

**PROPOSAL.** Caption normalization is Unicode NFC plus LF newline
normalization only. Do not lowercase, trim internal whitespace, remove
punctuation, rewrite spelling, or concatenate metadata.

`caption_sha256` is the SHA-256 of canonical UTF-8 bytes over:

```text
song-describer-caption-v1 NUL source_track_id NUL normalized_caption_text
```

This remains stable even if a source row order changes. Full hashes are stored;
short display ids are optional and collision-checked.

### 5.3 Corpus and media-identity contract

Each corpus row contains:

```json
{
  "corpus_id": "sdd-v1:track:<source_track_id>",
  "source_track_id": "<verified origin track id>",
  "source_relative_path": "<exact origin-release path>",
  "source_audio_sha256": "<exact MP3 byte hash>",
  "source_byte_length": 0,
  "duration_samples": 0,
  "sample_rate_hz": 0,
  "channels": 0,
  "source_codec": "mp3",
  "license_id": "<per-track license id>",
  "license_uri": "<per-track license URI>",
  "attribution": "<title, creator, source URL>",
  "origin_release": "10.5281/zenodo.10072001"
}
```

The source media identity is the exact origin MP3 bytes. Decoded or resampled
representations have separate hashes and never replace `source_audio_sha256`.
Absolute machine paths are forbidden in portable artifacts.

### 5.4 Qrel contract

Each v0 qrel is:

```json
{
  "query_id": "sdd-v1:t2a:q:<caption_sha256>",
  "corpus_id": "sdd-v1:track:<source_track_id>",
  "relevance": 1,
  "judgment": "caption_track_pair",
  "source_revision": "c58ad9b08343e56ce412d4da41accd15bffdbd6d"
}
```

There must be exactly one positive qrel per T2A query and no qrel for an
unknown query or corpus id. Duplicate qrels are invalid.

### 5.5 Split contract

Song Describer remains a single evaluation-only `test` split. No subset may be
named `train` or used for tuning prompts, dimensions, pooling, cropping,
normalization, or model selection.

The paper says the recordings were sampled from the MTG-Jamendo split-0 test
set. Any model known to have trained on Song Describer, those exact recordings,
or a derived caption set is contamination-flagged. Undisclosed pretraining is
recorded as `unknown`, never as `clean`.

## 6. Decoder and Transform Contract

### 6.1 Canonical local representation

**PROPOSAL.** Preserve the exact MP3 as the source artifact and create a
temporary canonical provider input only when a route requires it:

- decode the first audio stream only;
- no silence trimming;
- no loudness normalization;
- no temporal crop;
- mono;
- 48,000 Hz;
- signed 16-bit little-endian PCM WAV;
- metadata stripped from the derived container; and
- exact decoder name, version, command, input hash, output hash, sample count,
  and wall time recorded.

The intended transform identity is
`sdd-audio-pcm-s16le-mono-48k-full-v0`. A future implementation must pin and
test the exact decoder command before materialization. If a model requires a
different crop, chunk, resample, or pooling policy, that route receives a
different transform id and is not directly interchangeable with the canonical
public track.

The maximum observed duration is below Gemini Embedding 2's current official
180-second audio limit. That does not prove any other model accepts the full
recording.

### 6.2 Long-audio behavior

**GATE.** No route may silently use only the first or a random 10-second crop.
For local CLAP, the exact long-audio behavior of the pinned processor and model
must be specified and repeatable. If the model cannot produce one deterministic
full-track embedding under a reviewed policy, it is unavailable for the public
v0 matrix rather than silently approximated.

### 6.3 Derived-byte publication

Derived WAV bytes are temporary and no-publish. Even though Creative Commons
deeds state that merely changing format does not itself create a derivative,
the release contains older licenses and many NoDerivatives tracks, while
resampling, channel mixing, cropping, or normalization can raise additional
questions. The benchmark should publish transform metadata and hashes, not the
transformed audio.

## 7. Duplication, Negatives, and Leakage Audit

### 7.1 Captions and track weighting

**VERIFIED.** MTEB reports 1,106 unique text strings, but the paper says 25% of
tracks have multiple captions and up to five captions may describe one track.
Thus exact caption duplication is not currently visible in MTEB statistics,
while intended track-level duplication is substantial.

The MTEB-compatible caption-macro primary score gives more weight to tracks
with more captions. v0 must therefore also report a track-macro diagnostic:
average each track's caption-query metrics first, then average across the 706
tracks.

Metadata inspection must audit:

- exact duplicate normalized captions;
- the same caption associated with different track ids;
- captions differing only by normalization;
- caption count per track;
- shared artist, album, or release metadata that could act as shortcuts; and
- presence of source ids, titles, or artist names in captions.

### 7.2 Audio duplication

Before any public score, audit:

- exact duplicate MP3 hashes across different track ids;
- identical decoded PCM hashes;
- suspiciously near-identical recordings or alternate encodes;
- same artist/album concentration; and
- references whose license row or attribution does not match the audio path.

If different corpus ids have identical audio, either all identical items must
be positive for affected queries or the duplicate ids must be resolved before
scoring. A one-positive qrel is invalid when another byte-identical corpus item
is treated as a negative.

### 7.3 Negative-pool strength

The corpus is a random, play-count-weighted subset of MTG-Jamendo split-0 test,
not a curated hard-negative set. All non-positive tracks are implicit negatives,
but they may be musically easy or weakly related.

**PROPOSAL.** Do not market the task as a hard-negative benchmark. Report
metadata-based diagnostics for artist, album, genre/tag overlap when those
fields are rights-cleared and source-pinned. Do not mine negatives from the
evaluated model's scores for the primary task.

### 7.4 Training contamination

For every model, record one of:

- `no_known_direct_exposure` with reviewed evidence;
- `known_direct_exposure`;
- `possible_source_overlap`; or
- `unknown`.

Known Song Describer training or fine-tuning makes the result diagnostic-only.
The paper already warns that in-domain training distributions can inflate
retrieval performance. Model age alone is not evidence of no overlap with the
underlying music recordings.

### 7.5 Privacy and content risk

The paper's datasheet says participation was anonymous and no personal or
personally identifiable information was collected. Captions were intended for
public release. The full set nevertheless includes annotations that were
invalid or borderline under the quality review, and recordings may contain
lyrics some listeners consider offensive.

v0 must not publish annotator nicknames, contributor identifiers, IP-related
metadata, or collection-platform logs. Caption text, track attribution, and
license data are the only human-origin fields required for retrieval.

## 8. License and Redistribution Audit

### 8.1 Dataset-level and per-track rights are different

**VERIFIED.** Zenodo and the official Song Describer repository state that the
dataset/annotations are released under CC BY-SA 4.0. The paper states that the
audio is redistributed with the respective Creative Commons licenses and that
the release includes a list of individual audio licenses.

**VERIFIED.** The fixed Zenodo `audio_licenses.txt` has 706 track entries. A
read-only count of its stated license families gives:

| License family | Tracks |
| --- | ---: |
| CC BY | 37 |
| CC BY-SA | 150 |
| CC BY-ND | 14 |
| CC BY-NC | 15 |
| CC BY-NC-SA | 287 |
| CC BY-NC-ND | 195 |
| License Art Libre | 8 |
| Total | 706 |

Therefore 497 tracks carry a NonCommercial condition and 209 carry a
NoDerivatives condition; these sets overlap. A blanket `cc-by-sa-4.0` task
label does not describe the audio rights.

### 8.2 Obligations

Creative Commons deeds require appropriate credit, a license link, and a
change indication. ShareAlike licenses require adaptations to be distributed
under the same or a compatible license. NonCommercial licenses prohibit
commercial use. NoDerivatives licenses permit sharing unchanged material but
prohibit distribution of modified material. Older license versions may have
slightly different attribution details, including title requirements.

Every local audio row must therefore retain:

- track title;
- artist/creator;
- Jamendo track URL;
- exact license name, version, jurisdiction, and URL;
- origin archive path and hashes; and
- any transform/change indication.

### 8.3 MTG-Jamendo current terms

**VERIFIED.** The current MTG-Jamendo repository at commit
`cafd8e20c265ed84f1e61f1c875327971f43a62f` says its metadata is
CC BY-NC-SA 4.0, audio files use individual Creative Commons licenses, and the
MTG-Jamendo Dataset is made available solely for non-commercial research and
academic use unless Jamendo authorizes commercial use.

The Song Describer archival release is the immediate origin proposed here, but
the current MTG terms reinforce that the repository must not treat the audio as
unrestricted public product material.

### 8.4 Repository publication policy

**PROPOSAL.** v0 may use locally obtained audio bytes only for a bounded,
no-publish evaluation after per-track rights mapping passes. It must not copy
audio bytes into Git or the repository's Hugging Face Dataset/Space.

The public product may contain:

- captions under their stated dataset license;
- stable track ids and origin references;
- per-track attribution and license rows;
- exact source and transform hashes;
- qrels and task metadata;
- provider/model/transform provenance; and
- score-validity labels.

The public product must not contain audio bytes, derived WAV files, waveform
previews, or Space playback in v0. A later proposal to redistribute audio needs
a separate rights review and multi-license product design.

This is a conservative repository policy, not legal advice.

## 9. Metrics and Determinism

### 9.1 Ranking

Use cosine similarity after rejecting non-finite, inconsistent-dimension, and
zero-norm vectors. Do not round scores before ranking.

Exact score ties are broken by `corpus_id` ascending in UTF-8 byte order:

```text
(-similarity_score, corpus_id_utf8_bytes)
```

MTEB 2.18.7 computes recall, nDCG, MAP, precision, and success/hit rate through
`pytrec_eval`; its public task metadata does not define this repository's exact
cross-backend tie policy. The proposed tie rule is repository-owned and must be
tested with repeated evaluations.

### 9.2 Primary and diagnostic metrics

**PROPOSAL.** T2A v0 reports:

- primary: `recall@5`, caption-macro, for MTEB/paper-family compatibility;
- secondary: `recall@1` and `recall@10`;
- secondary: `MRR@10`;
- secondary: binary `nDCG@10`;
- diagnostic: `hit_rate@5`, which must equal `recall@5` here;
- diagnostic: `positive_set_coverage@5`, which must equal `recall@5` here;
- diagnostic: median rank; and
- diagnostic: `track_macro_recall@5` and corresponding track-macro R@1/R@10.

Required slices are:

- one caption versus multiple captions for the target track;
- caption-validity status, only if the pinned source exposes and verifies it;
- audio-duration bucket;
- source sample rate;
- per-track license family;
- artist/album overlap when rights-cleared metadata is present;
- exact/near-duplicate audit status; and
- model contamination status.

Slices with fewer than 20 queries are diagnostic-only and must show their
counts. No global average across T2A and A2T is defined.

## 10. Minimum Future Model and Provider Matrix

Registry modality labels are planning hints, not proof of a working audio
route.

| Route | Why it belongs | Current evidence | Current status |
| --- | --- | --- | --- |
| `gemini-embedding-2` through `GeminiProvider` | Modern unified audio-text API; official docs support MP3/WAV up to 180 seconds and one shared embedding space | Registry declares audio; adapter builds one WAV part; official docs and current price page are reachable | **Candidate, not runnable evidence.** No call was made. Exact MP3/WAV route, location, billing, retry cap, privacy, and full-track behavior require a separate provider item. |
| `laion/larger_clap_general@ada0c23a36c4e8582805bb38fec3905903f18b41` | Paper-comparable open-weight reference; MTEB PR says it nearly reproduces the paper | MTEB 2.18.7 registers the revision, 512 dimensions, about 194M parameters, and a 48 kHz audio path | **Unavailable here.** No repository provider, model bytes, or normal HF access. Long-audio preprocessing remains a gate. |
| `laion/clap-htsat-fused@cca9e288ab447cee67d9ada1f85ddb46500f1401` | Smaller standard MTEB audio baseline for parity diagnostics | MTEB registers 512 dimensions, about 154M parameters, and the same audio/text wrapper | **Optional later diagnostic.** It does not replace the paper model and is blocked by the same provider/HF gates. |

Current repository routes that are honestly unsuitable for this v0 include:

- OpenAI, ARK, GeeVec, and text embedding models: text-only;
- Voyage Multimodal 3.5: repository route covers text/image/video/document, not
  audio;
- DashScope Qwen3-VL-Embedding: text/image/video, not audio;
- Volcengine Doubao vision embedding: text/image/video, not audio;
- Jina and Cohere multimodal routes: text/image/document, not audio; and
- current sentence-transformers/transformers registry paths: no registered
  audio-text CLAP adapter in this repository.

The minimum quality matrix for a future full benchmark is Gemini Embedding 2
plus `laion/larger_clap_general`. If either route cannot meet the same full-track
and provenance contract, the task may run as a provider-specific pilot but is
not yet a cross-model leaderboard.

## 11. Resource-Bounded Inspection and Smoke

### 11.1 Metadata-first inspection

This must happen before any audio fetch.

After normal Hugging Face DNS/HTTPS returns, inspect only the pinned T2A
dataset revision and the small Zenodo metadata/license files. Acceptance:

1. the HF revision resolves through `huggingface.co` without an alternate host;
2. card, file tree, schemas, file sizes, and immutable file identities are
   recorded;
3. counts are exactly 1,106 queries, 706 corpus tracks, and 1,106 qrels;
4. every query has exactly one positive;
5. all caption strings and track ids crosswalk to Zenodo v1.0.0;
6. all 706 tracks have a per-track license and attribution row;
7. exact caption/audio duplicates and multi-caption track counts are reported;
8. no audio bytes, model, provider, benchmark, or score are used; and
9. any unexplained mismatch produces `BLOCKED`, not an inferred schema repair.

Resource caps:

- network: 10 MiB;
- disk: 25 MiB in a dedicated temporary directory;
- wall time: 20 minutes;
- provider/API cost: USD 0;
- audio/model bytes: 0; and
- cleanup: remove the temporary directory on pass, failure, or interruption.

### 11.2 At-most-eight-track no-publish smoke

This is defined here but must not run until metadata inspection passes, normal
HF connectivity remains healthy, per-selected-track rights are mapped, and one
provider route has an approved cost/privacy preflight.

Selection:

- at most eight tracks;
- up to four one-caption tracks and up to four multi-caption tracks;
- select within each stratum by ascending SHA-256 of canonical `corpus_id`;
- include all captions for selected tracks;
- reject any selected track with missing rights, duplicate-byte ambiguity, or
  duration above the provider limit; and
- never replace a rejected row with a hand-picked easier example without
  recording and versioning the rule.

Hard caps:

- source audio: at most 8 files and 960 aggregate seconds;
- transferred bytes: 64 MiB; if the source packaging requires a larger shard or
  full 3.3 GB archive, stop as `BLOCKED`;
- disk: 256 MiB including source and decoded temporary files;
- wall time: 30 minutes total, 10 minutes decode, 10 minutes provider work;
- decoder: one pinned build and the canonical full-track transform only;
- provider: one approved route, no fallback and no automatic billable retry;
- submitted audio: at most 960 seconds;
- provider cost: USD 0.20 hard ceiling, based on refreshed current pricing;
- model/API downloads: none beyond a separately approved, already-resolved
  provider route;
- result labels: `publish: false`, `leaderboard_publish: false`,
  `evidence_tier: smoke`, and `score_validity: smoke_only`; and
- cleanup: delete audio and decoded temporary files on pass, failure, timeout,
  or interruption.

At current official Gemini Embedding 2 pricing of USD 0.00016 per audio second,
960 seconds would cost at most about USD 0.1536 before text input. The USD 0.20
cap leaves only a small margin; pricing must be refreshed before the call. Any
billing, quota, credential, location, or privacy restriction stops the item.

The smoke validates source materialization, decoding, provider cardinality,
metrics, and cleanup only. Its score is never public quality evidence.

## 12. Failure Conditions and Public Eligibility

The task or run fails if any of the following occurs:

- source revision, file hash, or origin crosswalk is missing;
- query/corpus/qrel counts differ from the pinned contract;
- a T2A query has zero or multiple positives;
- unknown or duplicate ids/qrels exist;
- an exact duplicate audio item is left as a false negative;
- caption normalization or source bytes change without a new revision;
- decoder output is nondeterministic or duration/sample counts drift;
- a provider silently crops, chunks, drops, captions, or rewrites audio;
- embedding rows are missing, reordered, non-finite, zero-norm, or dimensionally
  inconsistent;
- tie order changes across repeated evaluation;
- any selected track lacks a complete attribution/license row;
- provider cost, wall, byte, disk, or audio-duration caps are exceeded;
- cleanup leaves source or decoded audio behind; or
- a smoke/partial run is presented as benchmark evidence.

A result is public-score eligible only when:

1. it uses all 1,106 queries and all 706 corpus items at the pinned revisions;
2. every source and transformed media identity passes;
3. every qrel, duplicate, license, and attribution gate passes;
4. the provider/model revision and full-track preprocessing are reproducible;
5. primary and required diagnostics are complete with deterministic ties;
6. the result is labeled `evidence_tier: benchmark` and not `smoke`;
7. no known direct Song Describer training/fine-tuning exposure exists; and
8. the public Dataset/Space export includes provenance, rights, transform, and
   score-validity fields while excluding audio bytes.

An `unknown` broad-pretraining overlap may be disclosed without claiming the
model is contamination-free. Known direct exposure makes the row
diagnostic-only.

## 13. Eventual Git and Hugging Face Product Path

### 13.1 Git-tracked artifacts

An eventual implementation may track:

- task metadata and the three-revision source ledger;
- query, corpus-reference, qrel, provenance, and per-track license tables;
- exact source/transform hashes and decoder identity;
- validation reports with counts and duplicate findings;
- provider/model route manifests; and
- benchmark result records.

Git must not track the 3.3 GB archive, extracted MP3 files, decoded WAV files,
provider caches, or smoke results.

### 13.2 Hugging Face Dataset

The Dataset product may publish:

- caption queries;
- stable track references, not audio bytes;
- qrels;
- source and transform revision tables;
- per-track attribution and license URIs;
- model/provider/score provenance; and
- public-eligibility and contamination labels.

It must present the release as multi-license. A single repository-level
`cc-by-sa-4.0` label must not imply that all audio is CC BY-SA 4.0. The Dataset
card must explain that the audio remains at the origin release under individual
licenses and is not redistributed by this benchmark.

### 13.3 Hugging Face Space

The Space may show a T2A leaderboard only after full benchmark-eligible rows
exist. It must display:

- primary `recall@5`;
- evidence tier and score validity;
- model/provider revision;
- transform id;
- contamination status;
- task/source revision; and
- an explicit note that no audio is hosted or played by the Space.

Smoke rows, partial subsets, known-contaminated rows, and rights-blocked runs
must not enter the public ranking.

## 14. One Restrained Next Item

Recommended next item, only after normal Hugging Face connectivity resumes:

`tasks/song-describer-t2a-metadata-source-contract`

Acceptance criteria:

1. inspect `dukesun99/SongDescriber-T2A` only at revision
   `c58ad9b08343e56ce412d4da41accd15bffdbd6d` through the normal HF hostname;
2. record the exact card, schema, file tree, file sizes, immutable identities,
   and stated license fields;
3. reconcile exactly 1,106 caption queries, 706 tracks, and 1,106 one-positive
   qrels with Zenodo `10072001` v1.0.0;
4. produce stable proposed ids plus a complete per-track attribution/license
   crosswalk;
5. report exact caption/track duplication and unresolved media identities;
6. use no audio bytes, model, provider API, benchmark run, or score;
7. stay within 10 MiB network, 25 MiB disk, 20 minutes, and USD 0; and
8. stop as `BLOCKED` on any unexplained row, schema, revision, license, or
   connectivity mismatch.

The precise resume condition is: token-disabled DNS and HTTPS for
`https://huggingface.co/` succeed through the normal hostname, and the pinned
T2A revision is anonymously readable. Until then, do not create a task,
provider patch, registry entry, run manifest, fixture, or result.

## 15. Primary Source Ledger

### MTEB 2.18.7 and Song Describer task

- MTEB tag commit `794f50399472059f4b518a5ed47c274459b704f1`:
  <https://github.com/embeddings-benchmark/mteb/tree/794f50399472059f4b518a5ed47c274459b704f1>
- PR `#4988` and merge discussion:
  <https://github.com/embeddings-benchmark/mteb/pull/4988>
- Exact task source:
  <https://github.com/embeddings-benchmark/mteb/blob/794f50399472059f4b518a5ed47c274459b704f1/mteb/tasks/retrieval/zxx/song_describer.py>
- T2A descriptive statistics:
  <https://github.com/embeddings-benchmark/mteb/blob/794f50399472059f4b518a5ed47c274459b704f1/mteb/descriptive_stats/Image/Any2AnyRetrieval/SongDescriberT2ARetrieval.json>
- A2T descriptive statistics:
  <https://github.com/embeddings-benchmark/mteb/blob/794f50399472059f4b518a5ed47c274459b704f1/mteb/descriptive_stats/Image/Any2AnyRetrieval/SongDescriberA2TRetrieval.json>
- Retrieval metric implementation:
  <https://github.com/embeddings-benchmark/mteb/blob/794f50399472059f4b518a5ed47c274459b704f1/mteb/_evaluators/retrieval_metrics.py>
- MTEB CLAP model metadata and revisions:
  <https://github.com/embeddings-benchmark/mteb/blob/794f50399472059f4b518a5ed47c274459b704f1/mteb/models/model_implementations/clap_models.py>

### Song Describer paper and release

- arXiv v3 metadata and paper:
  <https://arxiv.org/abs/2311.10057v3>
- Zenodo v1.0.0 record:
  <https://zenodo.org/records/10072001>
- Official dataset repository at commit
  `ed7e754c0c0739b6cd84828fc044588ee26d1e20`:
  <https://github.com/mulab-mir/song-describer-dataset/tree/ed7e754c0c0739b6cd84828fc044588ee26d1e20>
- Official datasheet:
  <https://github.com/mulab-mir/song-describer-dataset/blob/ed7e754c0c0739b6cd84828fc044588ee26d1e20/docs/datasheet.md>
- Fixed Zenodo per-track license ledger metadata:
  <https://zenodo.org/api/records/10072001>

### MTG-Jamendo and licenses

- Current MTG-Jamendo repository at commit
  `cafd8e20c265ed84f1e61f1c875327971f43a62f`:
  <https://github.com/MTG/mtg-jamendo-dataset/blob/cafd8e20c265ed84f1e61f1c875327971f43a62f/README.md>
- CC BY-SA 4.0 deed:
  <https://creativecommons.org/licenses/by-sa/4.0/>
- CC BY-NC-SA 3.0 deed:
  <https://creativecommons.org/licenses/by-nc-sa/3.0/>
- CC BY-NC-ND 3.0 deed:
  <https://creativecommons.org/licenses/by-nc-nd/3.0/>
- CC BY-ND 3.0 deed:
  <https://creativecommons.org/licenses/by-nd/3.0/>

### Provider evidence

- Gemini Embedding 2 multimodal embeddings and audio limits:
  <https://ai.google.dev/gemini-api/docs/embeddings>
- Gemini Embedding 2 pricing:
  <https://ai.google.dev/gemini-api/docs/pricing>
- LAION CLAP official repository:
  <https://github.com/LAION-AI/CLAP>

## 16. Final Decision Record

**GO is accepted, with T2A only.**

Song Describer is modern and under-covered for this repository because it
provides real, human-captioned, two-minute music retrieval rather than another
text task or a self-authored media fixture. Its scale is bounded enough for an
auditable pilot, and T2A has the strongest external claim: it is the paper's
retrieval direction, has one positive per query, and maps cleanly to music
search.

The accepted composed-media contract remains unchanged because Song Describer
items are independently encoded. The real blockers are source accessibility,
exact MTEB-port schema and byte identity, heterogeneous per-track rights,
duplicate/leakage review, deterministic full-track preprocessing, and a
genuinely working audio provider route.

The first follow-up is therefore metadata-only. No real-audio smoke begins
until the normal Hugging Face hostname works, the pinned rows crosswalk to the
Zenodo release, every selected track has an audited license/attribution record,
and an approved provider route fits the explicit byte, disk, time, privacy, and
cost caps.
