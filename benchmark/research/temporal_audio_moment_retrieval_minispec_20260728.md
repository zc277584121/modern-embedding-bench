# Temporal Audio Moment Retrieval Minispec - 2026-07-28

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_9-1785252357-2_execute.md`

Selected item: `tasks/temporal-audio-moment-retrieval-minispec`

Decision: **GO for a distinct temporal audio segment retrieval family. PAUSE any CASTELLA-backed data implementation.**

The repository should distinguish caption-to-audio-segment retrieval from both
whole-recording text-to-audio retrieval and end-to-end interval regression.
The useful embedding benchmark is collection-level ranking: a text caption is
embedded once, independently generated audio segments from every eligible
recording in the split are embedded once, and the caption must rank the relevant
temporal segment or segments above segments from the same and other recordings.

The family decision is `GO` because this preserves temporal retrieval as an
embedding problem and covers a real gap left by the merged MTEB CASTELLA port.
CASTELLA data use is `PAUSE` because the accessible primary sources do not
explain the MTEB port's `640 -> 566` recording reduction, the official sources
disagree on one caption count, the initial window construction and overlap
denominator are not fully specified, the source interval endpoint convention is
not documented, the underlying audio rights are not licensed by the annotation
repository, and the pinned Hugging Face payload was intentionally not accessed.

This is a research and contract minispec only. It does not add a task, provider,
registry entry, run, result, score, dataset, media file, or publication path.

Labels used below:

- **VERIFIED**: checked in the current repository or a primary source on
  2026-07-28.
- **DECISION**: repository-owned v0 behavior.
- **GATE**: a condition a later item must satisfy.
- **UNKNOWN**: not established by the inspected primary sources and not inferred.
- **NON-GOAL**: deliberately outside this benchmark family.

## 1. Decision in One Screen

| Evaluation shape | Unit ranked | Temporal signal | This family? |
| --- | --- | --- | ---: |
| Caption -> complete recording | One 1-5 minute recording | Lost after the correct recording is found | No; ordinary text-to-audio retrieval |
| Caption -> fixed candidate segment across a recording collection | One immutable time-bounded audio segment | Preserved through segment identity and boundary qrels | **Yes** |
| Caption + known recording -> regressed interval | A model-produced continuous interval | Preserved, but depends on a localization head and post-processing | No; end-to-end localization/system evaluation |
| Audio -> class or cluster | Whole audio item and label/cluster | No caption-to-segment ranking | No |
| Text+reference-media -> media | One heterogeneous composed query | Depends on fused-input semantics | No; composed-media retrieval |

**DECISION.** Name the family `temporal_audio_segment_retrieval`. A task in this
family has a flat text query, a flat audio-segment corpus, stable recording and
segment identities, explicit temporal qrels, multi-positive ranking metrics,
and deterministic score ties.

**DECISION.** The primary benchmark is collection-level. The query payload does
not reveal its source recording, and candidates include segments from every
eligible recording in the evaluation split. Restricting candidates to the known
source recording would change the task into localization-only evaluation.

**DECISION.** The family does not require the repository's composed-media input
capability. Each caption and each audio segment is independently encoded through
the existing one-item `EmbeddingInput` shape. The new work is in source identity,
segment materialization, qrels, and evaluation, not heterogeneous query fusion.

**GATE.** The first implementation may only be a self-authored metadata-and-score
contract fixture. It must remain `fixture_only`, `publish: false`, use no audio
bytes, and make no model-quality claim.

## 2. Primary-Source Findings

### 2.1 CASTELLA paper v2

**VERIFIED.** arXiv `2511.15131v2`, updated 2026-01-29, defines audio moment
retrieval as predicting one or more `(start, end)` moments and confidence scores
from a text query and one long input recording.

The paper reports:

| Split | Recordings | Local captions | Timestamps |
| --- | ---: | ---: | ---: |
| Train | 1,009 | 2,182 | 6,160 |
| Validation | 213 | 352 | 973 |
| Test | 640 | 1,347 | 4,175 |
| Total | 1,862 | 3,881 | 11,308 |

Other verified facts relevant to a retrieval contract are:

- recordings are one to five minutes and total more than 120 hours;
- the source is a YouTube subset used by AudioCaps;
- audio shorter than one minute was filtered and longer source videos were
  trimmed to the first five minutes;
- each recording has about 2.1 local captions on average;
- each local caption has about 2.9 temporal boundaries on average;
- boundaries were collected at one-second resolution;
- annotators were instructed to mark ranges that could include some surrounding
  non-relevant audio;
- every corresponding occurrence was intended to be annotated, so one query may
  have multiple moments;
- selected moments may overlap;
- a second annotator reviewed the annotations, and the authors removed captions
  relying on visual-only information or inaudible content;
- English captions were translated from Japanese with experienced translators,
  with machine translations available as aids; and
- 242 recordings without local captions were removed for the paper's baseline
  feature-extraction experiment.

The paper does not give a per-split breakdown of those 242 removals. This note
does not use that statement to explain the MTEB port's 74-recording test-set
difference.

**VERIFIED.** The paper evaluates end-to-end localization with Recall1 and mAP
at temporal IoU thresholds. Recall1 considers the most confident predicted
moment; mAP considers multiple predictions and multiple ground-truth moments.
The reported primary-style result is Recall1@0.7. These are localization metrics,
not ordinary corpus-ranking metrics.

### 2.2 Official CASTELLA project and repositories

**VERIFIED.** The official project page links the annotation repository, audio
downloader repository, paper, and extracted-feature releases.

The annotation repository was inspected at commit
`34a60e1eafe4b3a25d0ee10945ddbd5e0bea1c87`, whose commit message says it fixes
timestamp bugs. The repository:

- contains English and Japanese annotations, global captions, local captions,
  and timestamp arrays;
- does not contain raw audio;
- applies CC BY 4.0 to the repository material; and
- currently states `640` test recordings, `1,360` test local captions, and
  `4,175` test timestamps in its README.

The `1,360` README caption count conflicts with the paper v2 and merged MTEB
count of `1,347`. The inspected sources do not explain the difference.

**VERIFIED.** The audio repository was inspected at commit
`a085be2a401d76a7c3acf1bb8d9b026010a005c3`. It contains downloader and feature
extraction scripts, says downloads can fail when the upstream source changes,
and directs users to contact an organizer if audio cannot be downloaded. GitHub
reports no repository license for this audio repository.

**GATE.** CC BY 4.0 on the annotation repository is not evidence that the same
license covers the underlying YouTube audio. The license text also does not
grant third-party privacy, publicity, trademark, or similar personality rights.

### 2.3 DCASE 2026 Task 6

**VERIFIED.** The official DCASE 2026 Task 6 page defines Audio Moment Retrieval
from Long Audio as returning timestamps within a known long recording. It uses
CASTELLA and Clotho-Moment for development, renaming CASTELLA train/validation/
test to development-training/development-validation/development-testing.

The official task page states:

- the separate challenge evaluation set contains 100 recordings;
- challenge evaluation raw audio is not distributed;
- CASTELLA raw audio is to be downloaded by participants or obtained by
  contacting the organizers;
- organizers provide or link extracted MS-CLAP features;
- participants must not annotate the challenge evaluation data or use visual
  information from the original videos;
- the output identity includes query id, recording id, and predicted intervals;
- multiple predictions may be submitted in confidence order;
- Recall1@0.7 is primary; and
- participants are required to predict only one moment even when multiple
  ground-truth moments exist, while multiple predictions affect mAP.

**VERIFIED.** The official baseline repository was inspected at commit
`45ef471ee47ea75a2141d75bd9cfdb8c45dfc101`. Its code uses standard temporal
intersection and union over half-open-style numeric spans, locks each ground
truth to at most one prediction for detection AP, and reports results before
NMS in the documented baseline runs. The baseline code is MIT licensed. That
software license does not license CASTELLA annotations or audio.

**DECISION.** DCASE metrics remain a separate system track. A repository segment
ranking may report strict IoU diagnostics, but it must not present them as a
reproduction of DCASE Recall1@IoU because the candidate space, model interface,
and known-recording assumption differ.

### 2.4 MTEB 2.18.7 and PR #4984

**VERIFIED.** MTEB release `2.18.7` is commit
`794f50399472059f4b518a5ed47c274459b704f1`, published 2026-07-26. It contains
merged PR `#4984` at commit
`885f7404133f5a033de9ae671a1ad1cf686a39e7`.

The PR history shows two materially different tasks:

| Revision | Corpus and qrels | Verified statistics |
| --- | --- | --- |
| Initial commit `f07e4222f4ad8489c0c7aa08ade8d4e7c8ee57ce`; stats commit `1c808c3e0a244c7d2695378782be230e15c6bcd2` | Caption retrieves a window up to 10 seconds; relevance described as at least 50 percent temporal overlap | 12,046 documents, 1,347 queries, 1,154 unique texts, 6,530 qrels, 1-30 positives/query, 4.8478 mean positives/query |
| Switch commit `4698bd1d4b26eb1f1bea0c5d79996b307729ec5a`; merged at `885f7404133f5a033de9ae671a1ad1cf686a39e7` | Caption retrieves one complete recording; its source recording is the only relevant document | 566 documents, 1,347 queries, 1,154 unique texts, 1,347 qrels, exactly one positive/query |

The initial task points to dataset revision
`6b2a9905a1d0e787e84f635f4f9408941a384883`. The merged task points to revision
`083b2816174890f60f36fa9f145cfe79c2f94d0a` and declares
`license="not specified"`.

**VERIFIED.** The review discussion explains the change. A reviewer questioned
why the statistics showed short audio when the paper reports 60-300 second
recordings. The contributor said the recordings had been chunked into windows
up to 10 seconds to adapt localization to retrieval, called the initial choice
fairly arbitrary, and was testing other window sizes and overlaps. The reviewer
asked for the full version without chunking. The contributor then switched to
566 complete recordings, one gold recording per caption, to match the paper's
audio duration and simplify the task.

**VERIFIED.** The merged PR body is explicit that the port is a different task
type from the paper: the paper regresses intervals and uses Recall1@IoU, while
the MTEB task ranks complete recordings and has no paper-score reproduction.

### 2.5 Count reconciliation without repair

The currently verified numbers are not one coherent published data contract:

| Source | Recordings | Query/caption rows | Unique caption texts | Positive target |
| --- | ---: | ---: | ---: | --- |
| Paper v2 test split | 640 | 1,347 local captions | Not reported | 4,175 temporal boundaries |
| Annotation repository README at `34a60e1...` | 640 | 1,360 local captions | Not reported | 4,175 timestamps |
| Initial MTEB port | 566 source recordings stated in metadata; 12,046 windows | 1,347 | 1,154 | 6,530 positive windows |
| Merged MTEB port | 566 whole recordings | 1,347 | 1,154 | one source recording/query |

The MTEB statistics contain 193 query occurrences beyond the unique-text count,
so exact caption text is not a query identity. Repeated text may refer to the
same or different recordings and moments; the payload was not inspected.

**UNKNOWN.** The inspected sources do not establish:

- which 566 of the paper's 640 test recordings are present in MTEB;
- whether the reduction is caused by missing local captions, unavailable media,
  preprocessing, payload construction, or another rule;
- why the annotation README reports 1,360 test captions while the paper and
  MTEB report 1,347;
- the initial port's exact window hop, tail policy, source transform, or overlap
  denominator;
- how query ids relate to source annotation ids and repeated caption text;
- whether the MTEB 566 recordings are byte-identical to an official release;
  or
- whether excluded records, rewritten captions, or normalized boundaries exist.

This note does not subtract counts and assign a cause. Those are source-contract
questions for a separate payload audit.

## 3. Family Boundary

### 3.1 What the family measures

For an evaluation split with recordings `R`, an annotation-independent proposal
protocol produces a frozen segment corpus `S(R)`. A query contains caption text
only. The model ranks all segments in `S(R)`, including segments from recordings
that never contained the query annotation.

This tests two embedding capabilities together:

1. find the correct recording among a collection; and
2. find the relevant temporal region inside that recording.

The evaluator separately reports recording-hit diagnostics so a model that finds
the right recording but wrong moment is distinguishable from a model that misses
the recording entirely.

### 3.2 Explicit non-goals

**NON-GOAL.** This family is not:

- whole-recording text-to-audio retrieval such as Song Describer T2A or merged
  MTEB CASTELLA;
- audio classification, event labeling, clustering, or sound event detection;
- a composed-input task, because the query is one text item;
- a reranking task over jointly encoded query-segment pairs;
- a DETR or interval-regression benchmark;
- DCASE Recall1@0.7 reproduction;
- automatic captioning or transcription;
- temporal proposal learning;
- provider-side search/index evaluation;
- a license to redistribute CASTELLA audio; or
- evidence of model quality from a self-authored fixture.

## 4. Stable Identity Contract

All portable identities use canonical UTF-8 fields and full SHA-256 values.
Absolute local paths, row numbers, corpus order, and mutable download URLs are
not identities.

### 4.1 Query identity

```json
{
  "schema_version": "temporal-audio-query-v0",
  "query_id": "taq:<sha256>",
  "language": "eng-Latn",
  "caption_sha256": "<sha256>",
  "source_annotation_id": "<verified stable source id>",
  "source_revision": "<immutable revision>",
  "normalized_caption": "<source caption after declared normalization>"
}
```

`query_id` hashes the schema version, source revision, stable source annotation
id, language, and normalized caption. It does not expose `recording_id`, a time
range, filename, row number, or corpus position. Only the caption is embedded.

Identical normalized caption text in different source annotations remains
distinct when its source annotation or moment set differs. An additional
`caption_text_group_sha256` supports text-macro diagnostics without changing
query identity.

### 4.2 Recording identity

```json
{
  "schema_version": "temporal-audio-recording-v0",
  "recording_id": "tar:<sha256>",
  "source_collection_id": "<canonical collection>",
  "source_recording_id": "<verified source id>",
  "source_revision": "<immutable revision>",
  "source_locator": "<audited non-secret origin locator>",
  "source_audio_sha256": "<exact source bytes or explicit unknown>",
  "duration_ms": 0,
  "split": "test"
}
```

The source id, source revision, byte identity, duration, transform chain,
rights row, and split must be reviewable. A downloader result without a stable
origin id and byte hash is not sufficient for publication evidence.

### 4.3 Segment identity

```json
{
  "schema_version": "temporal-audio-segment-v0",
  "segment_id": "tas:<sha256>",
  "recording_id": "tar:<sha256>",
  "proposal_revision": "fixed-grid-v0",
  "transform_revision": "audio-transform-v0",
  "start_ms": 0,
  "end_ms": 10000,
  "source_audio_sha256": "<sha256>",
  "segment_audio_sha256": "<sha256 or fixture-only null>"
}
```

`segment_id` hashes recording identity, proposal revision, transform revision,
and exact boundaries. Two rows with the same recording and boundaries under the
same revisions are a duplicate and invalidate the task. Different temporal
windows that decode to identical silence bytes remain different temporal
segments but share a reported byte-duplicate group.

### 4.4 Moment and qrel identity

```json
{
  "schema_version": "temporal-audio-qrel-v0",
  "query_id": "taq:<sha256>",
  "segment_id": "tas:<sha256>",
  "moment_id": "tam:<sha256>",
  "moment_start_ms": 0,
  "moment_end_ms": 0,
  "intersection_ms": 0,
  "temporal_iou": 0.0,
  "moment_coverage": 0.0,
  "segment_coverage": 0.0,
  "overlap_coefficient": 0.0,
  "relevance": 1,
  "source_revision": "<immutable revision>",
  "qrel_revision": "temporal-overlap-v0"
}
```

`moment_id` hashes query identity, source boundary identity or boundary ordinal,
canonical start/end times, and source revision. Overlapping source moments are
not silently merged. A later source audit may define an explicit duplicate-
annotation grouping rule, but it must be versioned and reviewed.

## 5. Temporal Semantics and Qrels

### 5.1 Canonical interval convention

**DECISION.** Repository intervals are integer-millisecond, half-open ranges
`[start_ms, end_ms)`. Start must be non-negative, end must be greater than start,
and end must not exceed the verified recording duration. Invalid or out-of-range
source intervals are rejected; they are not silently clipped.

**UNKNOWN.** CASTELLA documents one-second annotation resolution but the sources
inspected here do not state whether stored integer end timestamps are inclusive
or exclusive. A real-source audit must establish the conversion. The repository
must not assume that source `[10, 20]` means canonical `[10000, 20000)`.

### 5.2 Proposal protocol

**DECISION.** Segments are generated by a deterministic, annotation-independent
proposal protocol frozen before test qrels are materialized. The protocol records:

- window duration or ordered multi-scale durations;
- hop for every duration;
- start anchor;
- tail handling and minimum tail duration;
- resampling, channel, decoding, padding, and clipping behavior;
- recording-duration source;
- proposal ordering; and
- an immutable proposal revision.

The initial MTEB 10-second construction is evidence that segment retrieval is
possible, not an accepted protocol. Its contributor called the window choice
fairly arbitrary, and the inspected metadata does not expose the hop or tail
rule. A later CASTELLA audit may evaluate candidate protocols on train and
validation coverage only, freeze one protocol, and then materialize test qrels.
It must never use evaluated model scores or test labels to choose windows.

**GATE.** Every query must have at least one positive candidate after the frozen
proposal protocol is applied. If not, protocol construction fails. It may not
insert annotation-aligned windows only for failed queries because that would
leak the answer into the corpus.

### 5.3 Exact overlap functions

For segment `S=[s0,s1)` and moment `M=[m0,m1)`, use integer arithmetic:

```text
intersection = max(0, min(s1, m1) - max(s0, m0))
segment_len = s1 - s0
moment_len = m1 - m0
union = segment_len + moment_len - intersection
temporal_iou = intersection / union
moment_coverage = intersection / moment_len
segment_coverage = intersection / segment_len
overlap_coefficient = intersection / min(segment_len, moment_len)
```

Threshold decisions are made by integer cross-multiplication before decimal
diagnostics are serialized:

- IoU >= 0.7 iff `10 * intersection >= 7 * union`;
- IoU >= 0.5 iff `2 * intersection >= union`; and
- overlap coefficient >= 0.5 iff
  `2 * intersection >= min(segment_len, moment_len)`.

### 5.4 Graded qrels

**DECISION.** A segment receives one qrel per source moment it matches:

| Relevance | Exact condition |
| ---: | --- |
| 3 | temporal IoU >= 0.7 |
| 2 | temporal IoU >= 0.5 and < 0.7 |
| 1 | overlap coefficient >= 0.5 and temporal IoU < 0.5 |
| 0 | no stored qrel |

The overlap-coefficient fallback prevents a fixed proposal grid from declaring
all short moments or all moments much longer than a window unanswerable. IoU is
retained as the stricter boundary-fit grade and diagnostic. The exact initial
MTEB phrase "at least 50 percent overlap" is not treated as evidence for this
definition because its denominator was not published in the inspected sources.

All positive windows are stored, not only the best window. A segment matching
two source moments receives two qrel rows with different `moment_id` values.

## 6. Ranking, Duplicate Credit, and Metrics

### 6.1 Ranking and ties

Use cosine similarity after rejecting non-finite, zero-norm, inconsistent-
dimension, missing, or reordered embeddings. Do not round scores before ranking.

Exact ties use:

```text
(-similarity_score, segment_id_utf8_bytes)
```

This follows the repository's accepted composed-media determinism pattern while
keeping the segment task independent of the composed-input capability.

### 6.2 No temporal NMS in the primary track

**DECISION.** The embedding benchmark ranks the full frozen segment corpus and
does not apply temporal NMS. NMS changes results through a system policy that is
not part of independent embedding similarity. An optional NMS or interval-
merging experiment must be labeled `system_temporal_postprocessing`, use the
same raw scores, and remain outside the primary leaderboard score.

Exact duplicate segment rows are invalid. Adjacent or overlapping windows are
valid distinct candidates. They must not inflate credit merely by retrieving
several windows around the same moment.

### 6.3 Primary moment-aware nDCG

**DECISION.** Primary metric: `moment_ndcg@10`.

For each ranked segment in order:

1. collect its qrels for as-yet-unmatched `moment_id` values;
2. choose the qrel with highest relevance, then highest unrounded temporal IoU,
   then `moment_id` ascending by UTF-8 bytes;
3. assign that gain to the rank and mark that moment matched; and
4. give zero gain if no unmatched positive moment remains for the segment.

Gain is `2^relevance - 1`. IDCG is computed from the best achievable relevance
for each source moment in the frozen corpus, sorted descending and truncated to
10. A retrieved segment can cover at most one source moment for metric credit.
This prevents a broad or duplicated neighboring window from satisfying several
moments at once.

### 6.4 Required secondary metrics

Report query-macro:

- `moment_recall@1`, `@5`, and `@10`: unique matched moments divided by all
  source moments for the query;
- `any_moment_hit_rate@1`, `@5`, and `@10`;
- `first_moment_mrr@10`;
- `window_ndcg@10` over ordinary graded segment qrels;
- `window_map@10` over all positive segment ids;
- `window_positive_set_coverage@10`;
- `strict_iou_hit@1_0.5` and `strict_iou_hit@1_0.7`;
- `source_recording_hit@1`, `@5`, and `@10`; and
- median first-positive segment rank.

Also report caption-text-group macro and recording macro where the source
identity audit supports them. Query macro remains primary because repeated
caption text and multiple captions per recording otherwise alter weighting.

Strict IoU diagnostics are not DCASE score reproductions. They rank a frozen
cross-recording corpus instead of regressing an interval inside a supplied
recording.

## 7. Leakage, Ambiguity, Negatives, and Slices

### 7.1 Recording-level isolation

All segments inherit their recording split. Train, validation, and test must be
isolated by a source group containing, when available:

- canonical source recording or video id;
- exact source audio hash;
- decoded canonical audio hash;
- perceptual near-duplicate group;
- original uploader/channel or collection identity when rights permit; and
- derivation lineage.

No source group may appear in more than one split. If an official split violates
the rule, the task is blocked or the complete source group is assigned through
a reviewed, versioned policy; individual windows are never moved independently.

Model training overlap is separately recorded as reviewed clean, known overlap,
possible source overlap, or unknown. AudioCaps/AudioSet lineage makes model-age
or model-name heuristics insufficient.

### 7.2 Caption duplicates and answer leakage

Required audits include:

- exact duplicate normalized captions;
- captions equal after Unicode, whitespace, or punctuation normalization;
- the same caption text with different recording or moment sets;
- multiple captions for one moment or recording;
- source ids, filenames, timestamps, titles, channels, or unique proper names in
  query text;
- qrels reconstructible from query ids, corpus ids, row order, or filenames;
- global captions accidentally used as local queries; and
- translated captions with source-language or visual-only leakage.

Query ids and filenames must not reveal their positive recording or boundaries.
Corpus order must be independent of qrel order.

### 7.3 Annotation ambiguity

CASTELLA intentionally permits contextual inference and somewhat broad temporal
boundaries. Overlapping moments and multiple occurrences are therefore normal,
not automatically annotation errors.

The task must preserve every verified source boundary and report ambiguity
diagnostics. It may define reviewed duplicate-annotation groups, but may not
silently merge overlapping boundaries, select only the easiest moment, or
replace a source boundary with a model-preferred interval.

Queries whose canonical endpoint semantics, language mapping, or source moment
identity cannot be established are excluded only through an explicit reason
code and count. An unexplained exclusion is a source-contract failure.

### 7.4 Hard-negative families

Hard negatives are declared from source structure before model evaluation:

1. **Adjacent boundary:** same recording, immediately before or after a
   positive window.
2. **Partial-overlap below threshold:** same recording and acoustically similar,
   but just outside the positive qrel rule.
3. **Wrong moment, same recording:** another salient annotated event in the
   source recording.
4. **Repeated event, wrong occurrence:** same broad sound but the wrong temporal
   occurrence for a context-sensitive caption.
5. **Same scene, cross recording:** similar AudioCaps/AudioSet context in a
   different recording.
6. **Duplicate caption, wrong source:** identical normalized text with a
   different recording or moment set.
7. **Boundary dilution:** a broad segment contains the event but too much
   unrelated content to satisfy the qrel.
8. **Silence/background:** neighboring low-information audio that may inherit
   recording-level similarity.
9. **Byte or perceptual duplicate:** a different corpus id with duplicate audio
   whose relevance equivalence has not been resolved.

The primary task must not mine negatives from the evaluated model's ranking.
Model-mined sets may be a later diagnostic revision only.

### 7.5 Required slices

Report counts and primary/secondary metrics for:

- source moment duration: `[1,5)`, `[5,10)`, `[10,30)`, `[30,60)`, and `60+`
  seconds, adjusted only if source audit proves different minimum semantics;
- recording duration;
- one versus multiple source moments per query;
- one versus multiple local captions per recording;
- unique versus duplicate normalized caption text;
- English versus other reviewed language tracks;
- best achievable proposal IoU: `>=0.7`, `[0.5,0.7)`, and `<0.5`;
- moment alignment versus proposal-grid boundary;
- number of positive windows and overlapping positive windows;
- hard-negative family;
- correct-recording/wrong-moment errors versus wrong-recording errors;
- exact/near-duplicate audio status;
- source availability and rights status; and
- model/source training-overlap status.

Slices with fewer than 20 queries are diagnostic-only and display their query,
recording, moment, and qrel counts.

### 7.6 Toy-data risk

The contract fixture defined below proves only identity, qrel, tie, and metric
behavior. A real benchmark score requires the complete declared evaluation
split after source and rights gates pass. A hand-selected mini subset, a corpus
containing only source recordings, or a fixture with synthetic score matrices
must never appear as public quality evidence.

## 8. Provenance, Privacy, and Rights

### 8.1 Source identity

The currently verified lineage is CASTELLA annotation -> AudioCaps-selected
YouTube source -> one-to-five-minute audio after filtering/trimming. That is not
yet a reproducible byte contract.

A real source audit must record:

- stable source video/recording id and official annotation id;
- exact annotation revision;
- downloader revision and command;
- source retrieval timestamp and status;
- original and transformed byte hashes;
- media duration before and after transformation;
- decoder, resampling, channel, and clipping versions;
- removed/unavailable source reason codes; and
- a complete crosswalk from source recording, query, moment, and segment ids.

Source media disappearing or changing must not silently produce a new task under
the same revision.

### 8.2 Annotation ownership

**VERIFIED.** The official annotation repository applies CC BY 4.0 to its
repository material. The paper attributes construction to its authors and
acknowledges organizational review and translation work.

**UNKNOWN.** The inspected sources do not separately document per-field
ownership, translator assignments, crowd-worker terms, or the authority chain
for every caption and temporal boundary. A metadata/source-contract audit must
record the repository license, attribution text, source authors, modifications,
and any field-level exceptions before publication.

### 8.3 Audio redistribution

**VERIFIED.** The annotation repository contains no raw audio. The audio tool
repository has no declared license. DCASE says CASTELLA and challenge evaluation
raw audio are not distributed by the task page; participants download CASTELLA
or contact organizers, and the challenge evaluation audio requires contact.
MTEB declares the port's license as `not specified`.

**DECISION.** No CASTELLA audio, derived WAV, waveform preview, or audio-bearing
Hugging Face artifact is authorized by this minispec. Local materialization,
evaluation, redistribution, and publication are separate permissions. A later
rights review must determine whether locally obtained audio can be processed at
all and whether any derived bytes may leave the evaluation machine.

### 8.4 Privacy and sensitive content

The source is real-world YouTube audio and may contain voices, names, locations,
incidental conversations, or other identifying content. The paper describes
annotation quality review but does not provide a privacy, consent, takedown,
biometric, minors, or sensitive-content policy for benchmark publication.

**GATE.** A real task requires a privacy and content-risk review covering source
terms, personally identifying speech, sensitive-event captions, access control,
retention, deletion/takedown handling, logs, and whether public query text or
media-derived artifacts are acceptable. Unknown is not treated as low risk.

## 9. What Remains Unknown Without Hugging Face Payload Access

No Hugging Face page, file tree, row, dataset script, audio, token, mirror,
alternate host, cache, or private artifact was accessed for this note.

The following remain unknown for revisions
`6b2a9905a1d0e787e84f635f4f9408941a384883` and
`083b2816174890f60f36fa9f145cfe79c2f94d0a`:

- card contents, file tree, schemas, sizes, hashes, and builder behavior;
- exact 566-recording membership and exclusion reasons;
- source recording ids and their crosswalk to official CASTELLA;
- query ids, caption normalization, and duplicate-text grouping;
- interval storage and endpoint convention;
- window duration, hop, tail, padding, and transform rules;
- the denominator used by the initial `>=50 percent overlap` qrels;
- whether every official timestamp was preserved;
- whether windows or whole recordings were recompressed or resampled;
- exact qrel rows and multi-moment mapping;
- dataset and per-media license statements;
- annotation modifications relative to official commit `34a60e1...`;
- missing/private media handling; and
- whether the payload contains fields inappropriate for public export.

These are not implementation details to guess. They are blocking source-contract
facts.

## 10. Smallest Machine-Fit Follow-up Smoke

Recommended next implementation item:

`tasks/temporal-audio-segment-contract-fixture`

It is a self-authored, no-network, no-audio, no-provider, no-publish fixture that
validates only identities, proposal generation, qrels, duplicate treatment,
metrics, slices, and deterministic ties.

### 10.1 Fixture contents

- six invented recording metadata rows, two per train/validation/test split;
- invented 60-second durations and no media payload;
- a frozen fixture protocol of 10-second windows with 5-second hops, start at
  zero, and a final shorter tail only when at least one second remains;
- eight test queries using newly written generic captions;
- at least one single moment, one repeated moment, one query with two moments,
  one pair of identical caption texts with different source annotations, one
  boundary-straddling moment, and one segment that overlaps two moments;
- explicit same-recording adjacent, wrong-moment, cross-recording, and
  duplicate-caption hard negatives;
- one valid score matrix, one exact-tie score matrix, and one duplicate-heavy
  score matrix; and
- expected exact metrics and slice counts checked without embeddings.

The fixture's 10/5-second grid is chosen only to exercise overlapping-window
logic. It is not approval of that protocol for CASTELLA.

### 10.2 Required rejection mutations

Tests must reject:

- duplicate query, recording, segment, moment, or qrel identity;
- query ids exposing recording or timestamp fields;
- segment boundaries outside the recording;
- zero-length or non-integer canonical intervals;
- qrels for unknown ids;
- a proposal window added only because it matches an annotation;
- a query with no positive proposal;
- inconsistent overlap numbers or relevance grade;
- a segment credited twice for the same moment;
- a source recording group spanning splits;
- score rounding before ranking;
- nondeterministic tie order;
- NMS applied to the primary metric;
- any network, media, provider, model, benchmark-run, result-publication, or
  Hugging Face operation; and
- `fixture_only: false`, `publish: true`, or leaderboard publication.

### 10.3 Fixture acceptance

Acceptance requires exact repeated results on two evaluations, balanced
serialization, full mutation coverage, zero network access, zero audio bytes,
and explicit output labels:

```text
fixture_only: true
publish: false
leaderboard_publish: false
evidence_tier: fixture
score_validity: contract_only
```

The fixture may not be described as easy, hard, representative, CASTELLA-like
in quality, or evidence that any model localizes audio.

## 11. Follow-up Gates

### 11.1 What a later implementation item may do now

Only the self-authored contract fixture may proceed from this decision. It may:

- add repository-owned dataclasses and validators for the fixture;
- implement deterministic proposal and qrel math over metadata;
- implement score-matrix evaluation and exact tie behavior;
- add fixture-only task metadata, tests, and no-publish controls; and
- reuse general ranking patterns from the composed-media fixture without using
  composed inputs.

It may not add CASTELLA ids, captions, timestamps, audio, a real task registry
entry, provider work, a model run, a benchmark score, or a public result.

### 11.2 Separate CASTELLA metadata/source-contract audit

A distinct audit is required before any real-data implementation. It must:

1. inspect the official annotation source at a pinned revision and the MTEB
   dataset only through its normal public hostname and exact pinned revision;
2. record card, file tree, schemas, sizes, hashes, stated licenses, and builder
   behavior without audio or model downloads;
3. reconcile `640`, `566`, `1,347`, `1,360`, `1,154`, and all qrel counts without
   inferred repairs;
4. establish source interval endpoint semantics;
5. crosswalk every eligible recording, query, and moment to official ids;
6. explain every exclusion and modification;
7. reconstruct the initial window/qrel protocol exactly or mark it unknown;
8. audit caption duplicates, multiple moments, overlapping boundaries, and
   unavailable media;
9. document annotation ownership, audio access terms, privacy, and redistribution
   constraints; and
10. stop as `BLOCKED` on any unexplained identity, count, boundary, or rights
    mismatch.

That audit does not authorize audio download or scoring.

### 11.3 Later real task criteria

Only after the source-contract and rights/privacy audits pass may a separate
item propose a local no-publish real task. It must additionally prove:

- a frozen annotation-independent segment proposal selected without test-score
  tuning;
- at least one positive proposal for every retained query;
- complete recording-level split isolation and duplicate grouping;
- reproducible source and transformed byte identities;
- a bounded materialization path with cleanup and no raw-media publication;
- at least two genuinely compatible audio-text embedding routes before any
  cross-model leaderboard claim;
- explicit training-overlap status for every model;
- complete moment-aware metric and slice output; and
- no comparison of segment-ranking scores to DCASE localization scores.

### 11.4 Permanent rejection conditions

Reject CASTELLA as a source for this repository's temporal segment family if any
of the following cannot be resolved through primary evidence:

- stable recording, query, moment, and source revision identities;
- unambiguous conversion of source boundaries to canonical intervals;
- a full explanation of the 640/566 and 1,360/1,347 differences;
- recording-level split isolation;
- an annotation-independent proposal protocol with positives for all retained
  queries;
- reproducible access to the eligible source audio under acceptable terms;
- sufficient rights for the intended local evaluation and metadata publication;
- an acceptable privacy, takedown, and sensitive-content policy; or
- a task that ranks temporal segments rather than silently reverting to complete
  recordings.

Also reject the direction if model-compatible segment materialization requires
answer-informed cropping, if source churn changes the task under one revision,
or if only a single provider-specific system can evaluate it. Another dataset
could satisfy the abstract family later; failure of CASTELLA source gates does
not justify weakening the family into whole-recording retrieval.

## 12. Source and Revision Ledger

### CASTELLA paper and project

- arXiv API metadata:
  <https://export.arxiv.org/api/query?id_list=2511.15131>
- arXiv v2 paper:
  <https://arxiv.org/abs/2511.15131v2>
- arXiv v2 HTML used for structured extraction:
  <https://arxiv.org/html/2511.15131v2>
- official project page:
  <https://h-munakata.github.io/CASTELLA-demo/>
- official annotation repository at
  `34a60e1eafe4b3a25d0ee10945ddbd5e0bea1c87`:
  <https://github.com/line/CASTELLA/tree/34a60e1eafe4b3a25d0ee10945ddbd5e0bea1c87>
- official audio-tool repository at
  `a085be2a401d76a7c3acf1bb8d9b026010a005c3`:
  <https://github.com/h-munakata/CASTELLA-audio/tree/a085be2a401d76a7c3acf1bb8d9b026010a005c3>

### DCASE 2026 Task 6

- official task page, retrieved 2026-07-28:
  <https://dcase.community/challenge2026/task-audio-moment-retrieval-from-long-audio>
- official results page, retrieved 2026-07-28:
  <https://dcase.community/challenge2026/task-audio-moment-retrieval-from-long-audio-results>
- linked baseline/evaluator at
  `45ef471ee47ea75a2141d75bd9cfdb8c45dfc101`:
  <https://github.com/awkrail/dcase2026_task6_baseline/tree/45ef471ee47ea75a2141d75bd9cfdb8c45dfc101>

### MTEB 2.18.7 and PR history

- release `2.18.7`, tag commit
  `794f50399472059f4b518a5ed47c274459b704f1`:
  <https://github.com/embeddings-benchmark/mteb/releases/tag/2.18.7>
- PR `#4984`:
  <https://github.com/embeddings-benchmark/mteb/pull/4984>
- initial window task commit
  `f07e4222f4ad8489c0c7aa08ade8d4e7c8ee57ce`:
  <https://github.com/embeddings-benchmark/mteb/commit/f07e4222f4ad8489c0c7aa08ade8d4e7c8ee57ce>
- initial descriptive-statistics commit
  `1c808c3e0a244c7d2695378782be230e15c6bcd2`:
  <https://github.com/embeddings-benchmark/mteb/commit/1c808c3e0a244c7d2695378782be230e15c6bcd2>
- switch-to-whole-recording commit
  `4698bd1d4b26eb1f1bea0c5d79996b307729ec5a`:
  <https://github.com/embeddings-benchmark/mteb/commit/4698bd1d4b26eb1f1bea0c5d79996b307729ec5a>
- merged commit and exact task source
  `885f7404133f5a033de9ae671a1ad1cf686a39e7`:
  <https://github.com/embeddings-benchmark/mteb/blob/885f7404133f5a033de9ae671a1ad1cf686a39e7/mteb/tasks/retrieval/eng/castella_amr.py>
- merged descriptive statistics:
  <https://github.com/embeddings-benchmark/mteb/blob/885f7404133f5a033de9ae671a1ad1cf686a39e7/mteb/descriptive_stats/Image/Any2AnyRetrieval/CASTELLAAMRRetrieval.json>

## 13. Resource and Operation Record

Research used `304,282` bytes of successful compressed response bodies. The
temporary research directory peaked below 2 MiB and is removed after validation.
The 15 MiB network and 25 MiB temporary-disk budgets were not approached.

There were zero provider/model API calls, zero embedding calls, zero benchmark
runs, zero GPU operations, zero model or dataset downloads, zero audio/media
downloads, zero Hugging Face operations, and zero publication, commit, or push
operations.

## 14. Final Decision Record

**GO:** add `temporal_audio_segment_retrieval` as a distinct conceptual benchmark
family, beginning only with the metadata-and-score contract fixture.

The family is justified because neither merged MTEB CASTELLA nor ordinary
text-to-audio retrieval measures where the described moment occurs. It remains
an embedding benchmark by ranking a frozen collection of independently encoded
segments, while moment-aware qrels and metrics prevent overlapping windows from
inflating credit.

**PAUSE:** do not implement a CASTELLA-backed task, source loader, provider,
registry entry, run, or score. The next real-data action is a separate metadata,
source-identity, boundary-semantics, rights, and privacy audit. If that audit
cannot resolve the stated permanent rejection conditions, CASTELLA must not be
used for this repository's temporal segment family.
