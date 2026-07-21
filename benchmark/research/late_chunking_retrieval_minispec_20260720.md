# Late Chunking Retrieval Minispec - 2026-07-21 Refresh

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_1-1784628002-2_execute.md`

Selected item: `tasks/late-chunking-retrieval-minispec`

Decision: **PROCEED with a provider-neutral, fixture-first benchmark direction.**
The first implementation should validate grouped chunk identity, qrels, metrics,
and paired comparisons without calling a provider or loading a model. Public
scores require a larger audited dataset and at least one real generic local
late-pooling path.

## 1. Residue Audit and Disposition

The target path existed before this dispatch as an untracked file. Its header
pointed to the older dispatch
`.perpetuum/modern-embedding-leaderboard/state/dispatch_1-1784541601-2_execute.md`.
The durable loop record says that older execute phase was rejected because the
Layer-1 session could not be proven ready, fresh, and exclusively owned; the
artifact was therefore never accepted.

**Disposition: replaced in full.** The previous text was treated only as an
untrusted lead list. This document was rebuilt against the current repository
and sources refreshed on 2026-07-21. No old implementation, pricing, machine,
or provider claim is accepted merely because it appeared in the residue.

Labels used below:

- **VERIFIED**: checked in the current repository or a primary source on
  2026-07-21.
- **PROPOSAL**: benchmark design recommended by this minispec.
- **GATE**: an assumption or decision that must be validated before a real run
  or public release.

## 2. Current Evidence

### Late chunking

**VERIFIED.** arXiv:2409.04701 is currently at v3, revised 2025-07-07. It
defines late chunking as:

1. choose chunk boundaries in the original document;
2. tokenize and encode the whole document, or the largest feasible macro
   window, into contextual token representations;
3. map original chunk boundaries to token spans; and
4. mean-pool token representations within each span to produce one embedding
   per chunk.

The paper states that the generic method can be applied to long-context
mean-pooling embedding models without additional training. Its long-document
extension uses overlapping macro windows when a document exceeds the model
context. The paper also reports an important limitation: on synthetic
Needle-8192 and Passkey-8192 cases, the surrounding text is unrelated to the
answer, so late chunking is not useful and naive chunking can be competitive or
better. A benchmark for contextual chunk embeddings therefore must contain
real cross-chunk dependencies and an irrelevant-context control, not only a
hidden-fact needle.

### Voyage contextualized chunk embeddings

**VERIFIED.** Voyage currently documents `voyage-context-4` for contextualized
chunk embeddings. The documented Python method is
`voyageai.Client.contextualized_embed()` and accepts either pre-chunked ordered
document groups (`List[List[str]]`) or full documents with backend auto
chunking. Current documented limits and outputs include:

- 32,000 tokens per chunk context window and 120,000 total context tokens;
- dimensions 1024 by default, with 256, 512, and 2048 also supported;
- at most 1,000 inputs, 120K total tokens, and 16K chunks per request;
- embeddings ordered to match chunk order; and
- returned chunk text through `chunk_texts` in the Python SDK or `text` in the
  REST response.

This is useful evidence for the grouped-input contract, but it must remain an
optional provider row rather than define the task.

### Runnable local/open path

**VERIFIED.** The official `jina-ai/late-chunking` repository contains Python,
tests, an example notebook, chunked-pooling code, and retrieval evaluation
scripts under Apache-2.0. Its reference path uses a long-context Jina embedding
model and applies pooling after token-level encoding. The current
`jinaai/jina-embeddings-v2-small-en` model card declares Apache-2.0, 8,192-token
input support, about 33M parameters, mean pooling, and a Transformers loading
path with `trust_remote_code=True`.

**GATE.** No model or code was downloaded or executed in this dispatch. The
upstream late-chunking project pins an older Python/Transformers/Torch stack
than this repository, so a later local adapter must pin a model revision,
review remote code, and prove compatibility in an isolated bounded smoke. The
existence of a runnable upstream path is verified; compatibility with this
repository is not.

## 3. Capability and Failure Mode

### Why this is not `needle_in_haystack`

**VERIFIED.** The current `NeedleInHaystackTask`:

- creates a long document containing an inserted fact;
- creates a matched document with that fact removed;
- independently embeds the query, whole document with the fact, and whole
  document without the fact;
- marks a hit when `similarity(query, with_fact) > similarity(query,
  without_fact)`; and
- slices accuracy by document length and insertion position.

It does not rank retrievable chunks from a corpus, preserve parent-document
groups, represent qrels or hard negatives, or test whether a locally ambiguous
chunk becomes retrievable when embedded in parent context.

**PROPOSAL.** `late_chunking_retrieval` measures a different modern RAG
failure: the answer-bearing retrieval unit is too locally ambiguous or
incomplete to rank correctly after independent embedding, even though the
parent document supplies enough scope, antecedent, definition, or section
context to disambiguate it. This is the failure introduced when a RAG index
must retrieve small chunks but the embedding representation discards
cross-chunk dependencies.

The benchmark must keep an `irrelevant_context` control slice because late
conditioning should not receive automatic credit when document context adds no
useful information.

## 4. Provider-Neutral Task Contract

Task id: `late_chunking_retrieval`

**PROPOSAL.** Treat segmentation and embedding conditioning as two independent
axes:

- **Segmentation policy** decides the exact retrieval units: fixed,
  fixed-with-overlap, structure-aware, semantic/model-aware, or provider auto
  chunking.
- **Conditioning policy** decides how each unit is represented: independent
  embedding, late pooling over contextual token states, or a grouped
  contextualized embedding API/model.

The benchmark owns queries, normalized parent documents, evidence spans,
required-context spans, split membership, and qrels-generation rules. Every
strategy must return auditable chunk text and identity.

### Primary fixed-unit track

The primary comparison freezes exact chunk text and ids across conditioning
policies. Each document is supplied as an ordered group of non-overlapping
chunks. Valid systems produce one vector per supplied chunk while preventing
context from crossing `document_id` boundaries.

This track supports the cleanest paired comparison:

```text
same query + same parent + same chunk ids + same model
independent pooling vs contextual/late pooling
```

### Variable-unit track

Adaptive, semantic, structure-aware, and provider auto chunkers change the
retrieval units. They must return exact chunk text, canonical parent offsets,
order, and chunker version. Their chunk-level metrics are not directly paired
with the fixed-unit track; compare them using evidence-span and parent-level
metrics plus index-cost diagnostics.

### Out-of-scope track

Generating new contextual prose and prepending it to each chunk changes indexed
content and adds a generative model, cost, and privacy surface. It may be a
later `context_text_augmentation` track, but it must not be mixed into the
embedding-only primary score.

## 5. Self-Created Dataset Design

### No-publish fixture

**PROPOSAL.** Author a deterministic local fixture with invented organizations,
systems, policies, dates, and identifiers:

- 8 parent documents arranged as 4 deliberately confusable pairs;
- 16 queries, 2 per document and 4 per failure family;
- 768 canonical whitespace-token units per document after Unicode NFC and LF
  normalization;
- one minimal evidence span and at least one required-context span per query;
- one misleading neighboring chunk inside the gold document; and
- at least one cross-document hard negative with nearly identical local text.

The fixture is for schema and metric tests only. It has split
`fixture_only`, is excluded from public leaderboards, and does not support model
quality claims.

### Failure families

1. **Split antecedent/reference**: an earlier section names a component or
   entity; the evidence chunk uses only an alias, pronoun, or generic noun.
2. **Ambiguous local policy**: paired documents contain the same local policy
   sentence, but an earlier scope statement assigns it to different products,
   regions, or time periods.
3. **Definition/example separation**: an earlier definition or exception is
   required to interpret a later answer-bearing example.
4. **Section-title dependence with adversarial neighbor**: the gold body is
   meaningful only under its title, while an adjacent section contains stronger
   query terms but applies to the wrong operation.

Each family also gets an `irrelevant_context` control query whose evidence
chunk is locally self-contained. A contextual strategy should not be rewarded
for merely seeing more tokens.

### Canonical layouts

The fixture materializes three layouts before any evaluated model is run:

- `fixed_192_v1`: 192 units, no overlap, 4 chunks per document, 32 chunks.
- `fixed_192_overlap_48_v1`: 192 units with stride 144, 5 chunks per document,
  40 chunks.
- `structure_adaptive_fixture_v1`: 4 variable-length authored sections per
  document, 32 chunks. This validates variable-boundary contracts but does not
  claim to measure a real adaptive model.

Canonical character offsets are authoritative. Whitespace-token indices are
only the deterministic fixture chunking rule. Real local/provider token spans
are derived metadata tied to a pinned tokenizer revision.

### Authoring and review procedure

1. Write parent documents and scope statements before queries.
2. Mark minimal-evidence and required-context character spans on normalized
   text.
3. Materialize all benchmark-owned layouts and hashes.
4. Write queries without unique lexical tokens that appear only in the gold
   local chunk.
5. Select hard negatives before inspecting any evaluated-model scores.
6. Blind-review each item twice:
   - the local gold chunk must be ambiguous against at least one negative;
   - the full gold parent must make the answer uniquely defensible.
7. Reject any item for which a hard negative becomes a valid answer after full
   parent-document review.

For a later score-bearing pilot, expand to at least 24 documents and 48 queries,
split by document template/family rather than query paraphrase. The fixture and
pilot must remain separately versioned.

## 6. Schemas and Reproducible Identity

All ids are strings. JSONL is the recommended artifact format.

### Query

```json
{
  "query_id": "q_policy_001",
  "split": "fixture_only",
  "text": "Which retry rule applies to the archive import worker?",
  "family": "ambiguous_local_policy",
  "gold_document_id": "doc_archive_ops_01",
  "evidence_span_ids": ["span_archive_ops_01_e2"],
  "required_context_span_ids": ["span_archive_ops_01_c1"],
  "context_required": true
}
```

### Parent document and spans

```json
{
  "document_id": "doc_archive_ops_01",
  "split": "fixture_only",
  "title": "Archive Import Operations",
  "text": "...",
  "source_type": "self_created_fixture",
  "source_revision": "late-chunking-retrieval-fixture-v0",
  "normalization": "unicode-nfc-lf-v1",
  "license_status": "not_for_publication",
  "sha256": "..."
}
```

```json
{
  "span_id": "span_archive_ops_01_e2",
  "document_id": "doc_archive_ops_01",
  "role": "minimal_evidence",
  "char_start": 1510,
  "char_end": 1658,
  "text_sha256": "..."
}
```

Offsets are zero-based, half-open Unicode code-point ranges into normalized
parent text. Chunk text must exactly equal the recorded parent substring.

### Chunk grouping

```json
{
  "layout_id": "fixed_192_v1",
  "document_id": "doc_archive_ops_01",
  "chunk_id": "fixed_192_v1:doc_archive_ops_01:0002",
  "chunk_index": 2,
  "char_start": 1432,
  "char_end": 2087,
  "text": "...",
  "chunker_version": "fixture-whitespace-v1",
  "text_sha256": "..."
}
```

`(layout_id, document_id, chunk_index)` is unique. Groups are ordered by
`chunk_index`. A provider result may add returned text, tokenizer spans, and
provider ids, but may not overwrite canonical fields.

### Qrels

```json
{
  "query_id": "q_policy_001",
  "layout_id": "fixed_192_v1",
  "chunk_id": "fixed_192_v1:doc_archive_ops_01:0002",
  "relevance": 2,
  "evidence_span_id": "span_archive_ops_01_e2"
}
```

Qrels are generated separately for every layout. A chunk receives relevance 2
when it fully contains a minimal-evidence span. Relevance 1 is allowed only for
a manually reviewed partial chunk that independently answers the query. If
overlap produces two complete evidence chunks, both are positives.

### Hard negatives

```json
{
  "query_id": "q_policy_001",
  "layout_id": "fixed_192_v1",
  "chunk_id": "fixed_192_v1:doc_backup_ops_02:0002",
  "negative_family": "cross_document_scope_collision",
  "reason": "same local retry wording, wrong worker scope",
  "false_negative_review": "pass"
}
```

Every query has three curated hard negatives per layout:

- one cross-document local-text collision;
- one misleading neighbor from the gold parent; and
- one same-family scope or definition collision.

### Returned result identity

Every evaluated row persists:

- canonical `document_id`, `chunk_id`, `chunk_index`, text, offsets, and hash;
- strategy/model/provider and revision;
- returned chunk text and provider chunk id when the provider performs
  chunking; and
- an exact mapping status: `exact`, `normalized_exact`, `span_mapped`, or
  `unmapped`.

`unmapped` provider chunks cannot receive chunk-level scores. They may be
reported in a separate exploratory system-chunker result but cannot enter the
primary leaderboard.

## 7. Fair Comparison Matrix

| Strategy | Segmentation | Conditioning | Fair comparison |
|---|---|---|---|
| `fixed_independent` | Benchmark `fixed_192_v1` | Each chunk embedded independently | Primary naive/ordinary embedding anchor |
| `fixed_overlap_independent` | Benchmark `fixed_192_overlap_48_v1` | Each chunk embedded independently | Tests overlap; report index inflation |
| `structure_adaptive_independent` | Benchmark variable sections or a pinned adaptive chunker | Each returned chunk embedded independently | Boundary-policy comparison; use remapped qrels |
| `fixed_late_same_model` | Same ids/text as `fixed_independent` | Encode parent, then pool supplied chunk spans with the same local model | Cleanest causal context comparison |
| `fixed_contextual_model` | Same ids/text as `fixed_independent` | Generic grouped contextual model/API | Provider-neutral capability row |
| `fixed_voyage_context` | Same ordered fixed chunks | Optional `voyage-context-4` pre-chunked groups | Model and conditioning both change; not causal |
| `adaptive_contextual` | Pinned adaptive/model-aware boundaries | Generic late/contextual conditioning | Combined system result; compare evidence/parent metrics |
| `whole_parent_independent` | One item per parent | Ordinary embedding | Parent-retrieval reference, not chunk retrieval |

Rules:

- Queries and parent-document split are identical across all rows.
- The primary context delta requires identical chunk ids and text.
- Overlap and adaptive rows report chunk count, indexed units, and duplicated
  indexed units.
- Voyage backend auto chunking belongs to the variable-unit track. Pre-chunked
  Voyage groups may enter the primary fixed-unit track.
- Whole-parent retrieval cannot claim evidence-chunk quality.

## 8. Metrics, Deltas, and Slices

### Full-corpus metrics

- `chunk_recall@1`, `chunk_recall@5`, `chunk_recall@10`
- `chunk_mrr`
- `chunk_ndcg@10`
- `evidence_span_recall@1`, `@5`, `@10`
- `parent_recall@1`, `parent_recall@5` after deduplicating by first parent hit

An evidence-span hit requires a retrieved canonical chunk to fully contain a
gold minimal-evidence span unless the exact chunk has an explicit reviewed
graded qrel.

### Hard-pool metrics

- `hard_recall@1`, `hard_recall@5`
- `hard_mrr`
- `hard_ndcg@10`
- `neighbor_confusion_rate`: fraction of queries for which the adversarial
  neighboring chunk outranks every positive

### Paired and robustness diagnostics

- `context_gain_<metric>`: contextual minus independent score per query on
  identical fixed chunk ids; required for `chunk_mrr`, `chunk_ndcg@10`, and
  `hard_mrr`.
- `overlap_gain_<metric>`: fixed-overlap independent minus fixed independent.
- `adaptive_gain_<metric>`: adaptive independent minus fixed independent,
  limited to evidence-span and parent metrics when units differ.
- `boundary_failure_rate@k`: gold parent appears in top-k but no complete gold
  evidence chunk does.
- `irrelevant_context_delta`: contextual minus independent performance on the
  control slice; large positive or negative movement must be inspected rather
  than assumed beneficial.

Outside the tiny fixture, report mean paired deltas with a deterministic
document-level bootstrap confidence interval.

### Required slices

- failure family;
- context-required versus irrelevant-context control;
- distance in chunks between required context and evidence;
- evidence position and parent length bucket;
- lexical shortcut present/absent;
- same-parent neighbor versus cross-document hardest negative;
- segmentation layout and boundary-overlap status; and
- exact versus normalized/span-mapped returned chunks.

## 9. Leakage, Toy-Risk, License, Privacy, and Rejection Gates

Reject an item, run, or release when any of the following holds:

- document groups, order, offsets, text hashes, or ids are missing or
  nondeterministic;
- a provider receives context from another `document_id`;
- a hard negative is a valid answer after full-document review;
- a context-required query is uniquely answerable from the local chunk alone;
- the gold parent does not make the answer uniquely defensible;
- overlap or adaptive chunking creates unrecorded positives;
- queries, templates, or normalized paragraphs leak across score-bearing
  splits;
- hard negatives or gold labels are changed after evaluated-model scores are
  inspected;
- real credentials, auth headers, private URLs, customer text, emails, phone
  numbers, or other personal data are present;
- hosted contextual embedding would transmit text without an explicit privacy
  and provider-terms review; or
- dataset/model/code license and pinned revision are absent from provenance.

**Toy-risk gate.** The self-created fixture is intentionally small and may
contain stylistic artifacts. It is valid only for schema, metric, and adapter
tests. A score-bearing pilot must pass lexical/BM25 shortcut checks, template
deduplication, blinded human review, and a minimum-corpus difficulty check. If
ordinary baselines saturate, rewrite the data before any publication.

**License gate.** The fixture remains `not_for_publication` until its authorship
and dataset license are explicitly approved. The repository currently has no
root license file, so this minispec does not infer permission to publish the
fixture. External papers, docs, model cards, or code are evidence sources, not
text to copy into the dataset.

## 10. Smallest No-Publish Smoke Manifest

```yaml
id: late-chunking-retrieval-local-smoke
publish: false
dataset_version: late-chunking-retrieval-fixture-v0
fixture:
  documents: 8
  queries: 16
  failure_families: 4
  required_context_queries: 12
  irrelevant_context_controls: 4
layouts:
  fixed_192_v1:
    chunks: 32
    positive_qrels: 16
    hard_negative_links: 48
  fixed_192_overlap_48_v1:
    chunks: 40
    positive_qrels: 24
    hard_negative_links: 48
  structure_adaptive_fixture_v1:
    chunks: 32
    positive_qrels: 16
    hard_negative_links: 48
strategies:
  - deterministic_independent_stub
  - deterministic_contextual_stub
network: forbidden
provider_api_calls: 0
model_downloads: 0
leaderboard_publish: false
```

The fixture authoring must place exactly eight evidence spans in overlap zones,
producing two positives for those queries in the overlap layout and one for the
other eight queries. This makes the qrel count deterministic and tests that
overlap-created positives are not mislabeled as negatives.

### Status criteria

- **PASS**: schemas validate; parent substrings and hashes reconstruct exactly;
  grouping/order changes are detected; qrels and hard negatives reproduce;
  full/hard metrics and all required deltas match deterministic expected
  values; returned text/ids survive result serialization; no external model or
  network is imported by tests.
- **FAILED**: grouping can be flattened without detection; chunk identity or
  text is lost; metrics/qrels are nondeterministic; a reviewed hard negative is
  relevant; overlap positives are missed; or context deltas cannot be paired
  on identical ids.
- **BLOCKED**: validating the local contract unexpectedly requires a provider
  key/call, model/data download, incompatible dependency change, or unresolved
  authorship/license decision.
- **ABANDON**: the task cannot separate segmentation from conditioning, cannot
  distinguish contextual retrieval from the existing whole-document needle
  task, or can only be implemented through one vendor's opaque auto chunker.

## 11. Fit With the Current Repository

**VERIFIED.** `EmbeddingProvider.embed()` currently maps a flat
`list[EmbeddingInput]` to a flat `EmbeddingResult`. Its cache key includes
input contents and modalities but not ordered document-group boundaries.
`VoyageProvider` currently calls normal `Client.embed()` for text or
`multimodal_embed()`; it does not expose `contextualized_embed()`. The local
SentenceTransformers path independently calls `encode()` on a flat text list.
The Jina adapter also has a flat input/result contract and does not preserve the
document-group/chunk identity required by this task.

**PROPOSAL.** Do not silently change `EmbeddingProvider.embed()` semantics.
Add an explicit capability, for example:

```text
embed_grouped_chunks(groups, task_type, dimensions) -> GroupedChunkEmbeddingResult
```

The request must contain ordered `document_id` groups and canonical chunk ids.
The result must retain group boundaries, chunk order, canonical ids, returned
text, mapping status, embeddings, token usage, latency, and cost when available.
Cache keys must include ordered group boundaries and strategy/chunker version.

Likely bounded implementation artifacts:

- `src/mm_embed/data/late_chunking_retrieval.py`
- `src/mm_embed/tasks/late_chunking_retrieval.py`
- `tests/fixtures/late_chunking_retrieval/`
- `tests/test_late_chunking_retrieval.py`
- a grouped-provider protocol/result type and deterministic test double
- later task registry and `benchmark/tasks/core.yaml` entries
- later local late-pooling and optional Voyage adapters

Candidate catalog entry after the smoke is accepted:

```yaml
- id: late_chunking_retrieval
  display_name: Context-aware chunk retrieval
  task: late_chunking_retrieval
  description: Retrieval of locally ambiguous evidence chunks using parent-document context.
  default_kwargs:
    dataset_version: late-chunking-retrieval-fixture-v0
  required_modalities: [text]
  primary_metric: chunk_ndcg@10
  metric_direction: higher
  dataset_version: late-chunking-retrieval-fixture-v0
  tags: [rag, long-context, chunking, contextual, hard-negative, text]
```

Product path:

1. keep this minispec and deterministic fixture/task tests in Git;
2. add an explicitly non-publishable local smoke manifest;
3. produce normal JSONL results only after a real model/provider run is
   separately authorized;
4. expand and audit a score-bearing dataset with provenance and license;
5. export the approved task/data/results through the existing Hugging Face
   Dataset path; and
6. expose only score-bearing, non-fixture runs in the Space leaderboard.

The existing result `details` object can carry slices and counts, but the
leaderboard currently selects one primary metric per task. The first public
task entry should therefore use `chunk_ndcg@10`, while paired deltas and slices
remain required supporting evidence.

## 12. Bounded Follow-Up Work

1. **Implement the deterministic no-publish smoke only**: authored fixture,
   validators, qrels/hard negatives, metrics, grouped test double, and tests.
2. **Add the explicit grouped-chunk provider contract** and a generic local
   late-pooling adapter behind unit tests; do not download a model in that
   implementation dispatch unless separately authorized.
3. **After privacy/cost/model gates are approved**, run one bounded real local
   same-model independent-versus-late comparison; treat a tiny
   `voyage-context-4` pre-chunked call as optional corroboration, not a
   prerequisite.

## Primary Sources Refreshed

- Late Chunking v3 metadata:
  https://arxiv.org/abs/2409.04701
- Late Chunking v3 full text:
  https://arxiv.org/html/2409.04701v3
- Official Jina late-chunking implementation:
  https://github.com/jina-ai/late-chunking
- Official implementation license:
  https://raw.githubusercontent.com/jina-ai/late-chunking/main/LICENSE
- Jina v2 small model card:
  https://huggingface.co/jinaai/jina-embeddings-v2-small-en
- Jina v2 small raw model-card metadata:
  https://huggingface.co/jinaai/jina-embeddings-v2-small-en/raw/main/README.md
- Voyage contextualized chunk documentation:
  https://docs.voyageai.com/docs/contextualized-chunk-embeddings
- Voyage contextualized embeddings API reference:
  https://docs.voyageai.com/reference/contextualized-embeddings-api

## Final Minispec Judgment

**PASS as a research/design artifact.** The direction is distinct from the
existing whole-document needle task, provider-neutral, auditable without a
network, and connected to the repository's tracked task/result/Hugging Face
paths. It does not establish any model score or provider acceptance. Those
remain future, explicitly gated implementation and execution steps.
