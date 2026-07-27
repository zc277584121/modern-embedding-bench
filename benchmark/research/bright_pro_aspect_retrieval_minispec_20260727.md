# BRIGHT-Pro Aspect-Aware Retrieval Minispec - 2026-07-27

## 0. Decision

**Conclusion: PAUSE.**

BRIGHT-Pro defines a coherent embedding-auditable retrieval family that is
materially different from both ordinary binary retrieval and retrieval-answer
system evaluation. Its static claim is narrow and useful: given one fixed query,
rank one fixed corpus so that high-value, complementary reasoning aspects are
covered early. Weighted A-Recall and weighted alpha-nDCG can be computed from a
fixed query-document ranking without a generator or an LLM judge.

The family is not ready for implementation or publication in this repository.
Three gates remain unresolved under this dispatch's source and network limits:

1. arXiv `2605.04018v1` says aspect importance is annotated on a Likert scale
   from 1 to 5, while `yale-nlp/Bright-Pro` commit
   `5df9e9baf5a0525a2b962d73f213c9dee63c5f3e` documents raw weights as
   `{1, 2, 3}` in `bright_pro_data.py`. The fixed source set does not reconcile
   that schema conflict.
2. The paper reports rounded per-domain average aspect counts, but the exact
   aspect-row count and the complete row-level field schema cannot be verified
   without reading the dataset payload. Hugging Face access was explicitly
   forbidden in this item, so the unknown was not worked around.
3. The repository MIT license, the MTEB task metadata value `license="mit"`,
   and Stack Exchange's CC BY-SA terms do not establish redistribution rights
   for the externally collected passage text, BRIGHT-Pro annotations, or a
   repackaged public Dataset.

The smallest defensible next step is therefore a metadata-first, no-publish,
zero-model contract smoke after an authorized, pinned metadata bundle and a
row-level rights manifest are available. This note specifies that smoke but does
not run it.

## 1. Scope and Evidence Labels

This artifact covers only the static aspect-aware retrieval surface. It does not
implement a task, download a dataset or model, run an embedding provider, copy
paper scores as local evidence, judge generated answers, or publish any result.

Evidence labels used below:

- **VERIFIED**: directly observed in a pinned primary source or current local
  source file.
- **DECLARED**: stated by an upstream author or metadata record but not
  independently checked against the underlying payload.
- **DERIVED**: arithmetic or contract reasoning from verified inputs.
- **UNKNOWN**: not reconcilable from the permitted sources.
- **PROPOSED**: a future local contract, not current behavior.

Local repository baseline:

- repository commit: `017f80eb75e9b670c36203584eab25cd93f9c38e`;
- branch: `main`;
- startup Git-tracked state: clean;
- selected item: `tasks/bright-pro-aspect-retrieval-minispec` only.

## 2. Primary Sources, Revisions, and Roles

### 2.1 BRIGHT-Pro repository

**VERIFIED.** The audited repository revision is:

- repository: <https://github.com/yale-nlp/Bright-Pro>;
- commit: [`5df9e9baf5a0525a2b962d73f213c9dee63c5f3e`](https://github.com/yale-nlp/Bright-Pro/commit/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e);
- commit API: <https://api.github.com/repos/yale-nlp/Bright-Pro/commits/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e>;
- recursive tree API: <https://api.github.com/repos/yale-nlp/Bright-Pro/git/trees/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e?recursive=1>.

Pinned raw files and their source roles:

| Source | Role |
| --- | --- |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/README.md> | Declares the seven domains, static versus agentic split, fixed 175-query agentic subset, data configs, and metric entry points. |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/bright_pro_data.py> | Defines the `examples`, `documents`, and `aspects` configs, seven domain names, document-to-aspect construction, and weight normalization. |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/run.py> | Shows query, corpus, instruction, binary qrel, full-score, and top-200 output behavior for static retrieval. |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/evaluation/weighted_aspect_recall.py> | Executable interpretation of weighted A-Recall and current edge behavior. |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/evaluation/alpha-ndcg-evaluation.py> | Executable interpretation of weighted alpha-DCG, greedy IDCG, and current aggregation behavior. |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/metrics.py> | Shows the ordinary binary qrel diagnostics computed with `pytrec_eval`. |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/agentic_sample_ids.json> | Immutable fixed 175-query id set at the repository revision. |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/agentic_retrieval/scripts_evaluation/sample_agentic_qids.py> | Defines seed 42 and 25 sampled query ids per domain. |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/agentic_eval_outputs/README.md> | Distinguishes agentic traces and their per-line schema from static rankings. |
| <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/LICENSE> | MIT license text for the repository; not accepted as a blanket data-rights grant. |

The commit adds large raw agentic-evaluation outputs. Those payloads are
irrelevant to the static contract and were not downloaded.

### 2.2 Paper

**VERIFIED.** The paper source is arXiv `2605.04018v1`, published 2026-05-05:

- metadata: <https://export.arxiv.org/api/query?id_list=2605.04018>;
- abstract: <https://arxiv.org/abs/2605.04018v1>;
- PDF: <https://arxiv.org/pdf/2605.04018v1>.

The paper is the primary source for the benchmark construction narrative,
Table 1 cardinalities, human annotation procedure, metric equations, and the
separation between static retrieval and LLM-based agentic answer evaluation.
No reported model score was copied into this repository as local evidence.

### 2.3 MTEB PR 4929

**VERIFIED at 2026-07-27 UTC.** The official GitHub API reports:

- PR: <https://github.com/embeddings-benchmark/mteb/pull/4929>;
- API: <https://api.github.com/repos/embeddings-benchmark/mteb/pulls/4929>;
- files API: <https://api.github.com/repos/embeddings-benchmark/mteb/pulls/4929/files?per_page=100>;
- state: `open`;
- draft: `false`;
- merged: `false`;
- mergeability at observation: `clean`;
- head repository/ref: `yilunzhao/mteb:add-bright-pro-retrieval`;
- head SHA: `a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64`, matching the dispatch expectation;
- base SHA at observation: `6e72309defe615d46b38ee6a671d9149f06e53e6`.

Pinned task source:

- <https://raw.githubusercontent.com/embeddings-benchmark/mteb/a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64/mteb/tasks/retrieval/eng/bright_pro_retrieval.py>.

The PR declares source dataset revision
`dbdc22babbef310210e267b99249e7cec86d5edf`. That value is **DECLARED**, not
independently verified here because Hugging Face access was forbidden. The MTEB
task file instead pins seven converted per-domain repositories:

| Domain | MTEB dataset path | Revision |
| --- | --- | --- |
| biology | `mteb/BrightProBiologyRetrieval` | `8d356ed8a3b65123b5ec78793fbb7e80010345db` |
| earth_science | `mteb/BrightProEarthScienceRetrieval` | `66a901dddee938a175f87d3dededeae506c13af8` |
| economics | `mteb/BrightProEconomicsRetrieval` | `e7ce2bbbfc5fab8c0048dadc83abf2b5dd98bc05` |
| psychology | `mteb/BrightProPsychologyRetrieval` | `18b7681648b7064e0a0f46843f3ab677d7e6a2c4` |
| robotics | `mteb/BrightProRoboticsRetrieval` | `25648862e812eef5fbeb51de859ce3127dbc055e` |
| stackoverflow | `mteb/BrightProStackoverflowRetrieval` | `cfadbcdc4d1ee17e91751218ae0ebfa001c8feea` |
| sustainable_living | `mteb/BrightProSustainableLivingRetrieval` | `3f10f4998714e03b9fda8064360a768b0a3fbb4f` |

Pinned descriptive-statistics files at the PR head were used to reconcile the
query, corpus, and binary-qrel counts. They are listed in the network ledger in
Section 13.

### 2.4 Stack Exchange rights sources

These current, unversioned official pages were used only to establish the
known Stack Exchange boundary:

- <https://stackoverflow.com/help/licensing>;
- <https://stackoverflow.com/help/referencing>;
- <https://stackoverflow.com/legal/terms-of-service/public>.

They were fetched into the dedicated temporary directory and inspected through
tightly scoped text extraction. Complete HTML bodies were not streamed into the
session output.

## 3. Dataset and Evaluation Contract Observed Upstream

### 3.1 Seven static domains and reconciled cardinalities

**VERIFIED and DERIVED.** Paper Table 1 and the seven MTEB descriptive-statistics
files agree on every per-domain query and corpus count. Their sums independently
reconcile the paper's `739` queries and `526,319` documents. Therefore the
paper's approximate `526k` claim can be sharpened to `526,319` for this source
cutoff.

| Domain | Queries | Documents | Binary positive qrels | Avg. positives/query | Avg. aspects/query from paper | Unique document texts in MTEB stats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| biology | 103 | 59,513 | 804 | 7.8058 | 3.94 | 51,584 |
| earth_science | 115 | 123,575 | 856 | 7.4435 | 3.83 | 119,953 |
| economics | 99 | 52,240 | 773 | 7.8081 | 3.71 | 42,611 |
| psychology | 100 | 54,741 | 707 | 7.0700 | 3.84 | 45,660 |
| robotics | 101 | 63,920 | 623 | 6.1683 | 3.71 | 42,382 |
| stackoverflow | 115 | 109,188 | 529 | 4.6000 | 3.32 | 68,375 |
| sustainable_living | 106 | 63,142 | 980 | 9.2453 | 3.86 | 52,487 |
| **Total / paper overall** | **739** | **526,319** | **5,272** | **7.13** | **3.74** | **423,052 within-domain sum** |

The `423,052` value is a sum of seven within-domain `unique_texts` counters. It
is not a globally deduplicated corpus count. The difference of `103,267`
between document rows and within-domain unique-text rows proves substantial
exact-text duplication inside domain corpora, but says nothing about additional
cross-domain duplication.

The exact total number of aspect rows is **UNKNOWN**. Paper averages are rounded
to two decimals, so multiplying them by query counts cannot recover an exact
cardinality. A rough implied total is about 2.76k aspects, but that estimate must
not be used as a manifest count.

### 3.2 Query schema

**VERIFIED from executable consumers.** The static code requires at least:

```json
{
  "id": "<query id coercible to string>",
  "query": "<query text>",
  "gold_ids": ["<document id>", "..."]
}
```

Cardinality and invariants observed or required by the source:

- exactly 739 rows across the seven full static domains;
- per-domain counts are in Section 3.1;
- `gold_ids` becomes the binary qrel set;
- `excluded_ids` and `gold_ids_long` are documented as removed from BRIGHT-Pro;
- the upstream scripts do not visibly reject duplicate query ids or duplicate
  values inside `gold_ids`; a future auditable contract must reject them.

### 3.3 Corpus schema

**VERIFIED from `retrieval/run.py`.** The static runner consumes:

```json
{
  "id": "<document id>",
  "content": "<passage text>"
}
```

The seven per-domain corpus cardinalities total 526,319 rows. The paper explains
that original BRIGHT positives were re-audited and sometimes merged, while new
positive passages were collected from external web pages and segmented to align
with one reasoning aspect. The corpus also contains non-gold candidates.

The code forms one `doc_ids` list and one `documents` list but does not visibly
enforce unique ids or unique content. MTEB statistics confirm that duplicate
content exists. A future task must preserve logical document ids while recording
content hashes and duplicate groups.

### 3.4 Aspect and aspect-weight schema

**PARTIALLY VERIFIED.** Executable code accesses these fields:

```json
{
  "id": "<aspect id with a query-derived stem>",
  "weight": "<numeric raw importance>",
  "supporting_docs": ["<gold document id>", "..."]
}
```

The paper additionally states that every aspect has a one- to two-sentence
rationale/description. The exact field name for that text is **UNKNOWN** because
the permitted repository code does not read it and the dataset payload was not
accessed.

`build_doc_to_aspect_id` derives `{doc_id: aspect_id}` by iterating
`supporting_docs`. If the same document appears under two aspects, the later
aspect silently overwrites the former. `build_aspect_weights` derives a query
stem by removing an `-aN` or `-aspect-N` suffix from the aspect id, sums raw
weights by stem, and normalizes each aspect to the query-level sum.

The decisive schema conflict is:

- paper v1: raw aspect importance is Likert 1 to 5, normalized to weights whose
  per-query sum is 1;
- pinned loader comments: raw Likert values are `{1, 2, 3}`.

The metric equations only require nonnegative normalized weights, so the metric
family is conceptually sound. The source data contract is not auditable until
the raw range and row schema are reconciled.

### 3.5 Qrel schema

The original static source has a richer relation:

```json
{
  "query_id": "<query id>",
  "doc_id": "<gold document id>",
  "binary_relevance": 1,
  "aspect_id": "<exactly one aspect id>",
  "aspect_weight": "<normalized query-local weight>"
}
```

The first three fields are **VERIFIED** by the static runner and MTEB port. The
last two are **DERIVED** from the aspect table's `supporting_docs` relation and
are exactly the evidence lost by binary conversion.

There are 5,272 binary positive qrels across 739 queries. MTEB descriptive stats
report at least one relevant document per query; per-domain maxima range from 13
to 20 in the converted port. The upstream repository test comment that biology
can have 27 golds conflicts with the MTEB descriptive-statistics maximum of 16
at the audited PR head. That test comment is not used as cardinality evidence.

### 3.6 Instruction schema

The upstream static runner loads per-retriever, per-domain JSON:

```json
{
  "instructions": {
    "query": "<query instruction template>",
    "document": "<document instruction template>"
  },
  "instructions_long": {
    "query": "<long-context query instruction template>",
    "document": "<long-context document instruction template>"
  }
}
```

For the pinned `rtriever-4b/biology.json`, the query templates are:

- `Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery:`;
- long form substitutes `documents` for `passages`;
- document templates are empty.

BM25 has empty instruction maps. The MTEB port standardizes one query prompt per
domain, for example:

`Given a biology post, retrieve relevant passages that help answer the post`

No document prompt is declared in the MTEB task metadata. Instruction choice is
therefore a measured-model input and must be recorded with every future run; it
must not be treated as harmless metadata.

### 3.7 Full static subset

The full static surface is the union of all seven domain splits listed in
Section 3.1:

```text
biology, earth_science, economics, psychology, robotics,
stackoverflow, sustainable_living
```

Each query receives one ranking over its corresponding full domain corpus. The
paper's primary static metric is alpha-nDCG@25 with `alpha=0.5`; weighted
A-Recall@25, binary nDCG, and binary Recall are diagnostics. The repository
saves only the top 200 scores per query by default, which is sufficient for its
declared cutoffs but is not a complete-corpus result artifact.

### 3.8 Fixed 175-query agentic subset

This is a separate system-evaluation sampling surface, not a static benchmark
replacement. The pinned `agentic_sample_ids.json` schema is:

```json
{
  "seed": 42,
  "n_per_task": 25,
  "tasks": {
    "biology": ["<25 query ids>"],
    "earth_science": ["<25 query ids>"],
    "economics": ["<25 query ids>"],
    "psychology": ["<25 query ids>"],
    "robotics": ["<25 query ids>"],
    "stackoverflow": ["<25 query ids>"],
    "sustainable_living": ["<25 query ids>"]
  }
}
```

**VERIFIED:** seven domains times 25 ids equals exactly 175. The generator script
uses `random.Random(42)`, samples without replacement within each domain, then
sorts sampled ids by `(len(id), id)`. The paper says the same 175-query subset is
used for fixed- and adaptive-round agentic evaluation.

This subset must never silently represent the 739-query static benchmark. Any
result produced on it needs `subset=agentic-fixed-175`, `seed=42`, the exact id
file hash, and a non-paper-comparable label if the protocol or candidate pool
differs.

## 4. Metric Contract and Required Invariants

### 4.1 Validated query-local inputs

For each query `q`, define:

- `R`: a ranking of unique document ids with deterministic scores;
- `G`: the unique gold-document set;
- `A`: the unique aspect set;
- `aspect(doc)`: exactly one aspect for every `doc in G`;
- `raw_weight(a)`: a finite nonnegative value;
- `weight(a) = raw_weight(a) / sum(raw_weight(x) for x in A)`;
- `k`: a positive integer cutoff;
- `alpha`: a finite value in `[0, 1]`, fixed to `0.5` for the BRIGHT-Pro v1
  paper-compatible contract.

Validation must occur before scoring:

1. query ids, document ids, aspect ids, qrel pairs, and ranked document ids are
   unique;
2. every ranked id exists in the declared candidate pool;
3. every gold id exists in the candidate pool;
4. every gold id maps to exactly one aspect;
5. every aspect belongs to exactly one query and has at least one gold document;
6. all weights are finite and nonnegative, and the query-local total is greater
   than zero;
7. every expected query has one complete ranking or an explicit failure record;
8. missing query scores are not silently skipped.

### 4.2 Weighted A-Recall pseudocode

```text
function weighted_aspect_recall(ranking, gold_docs, doc_to_aspect,
                                normalized_weights, k):
    require k > 0
    require ranking has unique document ids
    aspects = unique(doc_to_aspect[d] for d in gold_docs)
    require aspects is non-empty
    require every aspect has a declared finite nonnegative weight
    require sum(normalized_weights[a] for a in aspects) == 1 within tolerance

    covered = empty set
    for doc_id in ranking[0:min(k, len(ranking))]:
        if doc_id in gold_docs:
            covered.add(doc_to_aspect[doc_id])

    return sum(normalized_weights[a] for a in covered)
```

Invariants:

- one or many retrieved documents from the same aspect receive the same single
  aspect credit;
- retrieving all positive-weight aspects returns 1;
- a zero-weight aspect is valid but contributes zero when covered;
- an all-zero query weight vector is invalid, not a score of zero;
- a zero-aspect query is invalid, not silently included with score zero;
- duplicate aspect rows, duplicate document ids, duplicate ranking entries, or
  multi-aspect mappings for one gold document are hard failures;
- `k` larger than the ranking length uses the complete ranking without padding;
- the metric is independent of score magnitudes after deterministic ranking.

The pinned upstream script returns zero for a query with no discovered aspects
and uses a set for covered aspects, but it does not enforce the complete input
invariants above. The stricter behavior is required for an auditable product.

### 4.3 Weighted alpha-nDCG pseudocode

```text
function alpha_dcg(ranking, gold_docs, doc_to_aspect,
                   normalized_weights, alpha, k):
    seen_count = map defaulting to 0
    dcg = 0
    for rank, doc_id in enumerate(ranking[0:min(k, len(ranking))], start=1):
        if doc_id not in gold_docs:
            continue
        aspect_id = doc_to_aspect[doc_id]
        novelty = (1 - alpha) ** seen_count[aspect_id]
        gain = normalized_weights[aspect_id] * novelty
        dcg += gain / log2(rank + 1)
        seen_count[aspect_id] += 1
    return dcg

function alpha_idcg(gold_docs, doc_to_aspect,
                    normalized_weights, alpha, k):
    remaining[a] = number of unique gold documents mapped to aspect a
    seen_count[a] = 0
    idcg = 0
    for rank in 1..min(k, len(gold_docs)):
        candidates = aspects a where remaining[a] > 0
        if candidates is empty:
            break
        choose a minimizing the deterministic key:
            (-normalized_weights[a] * (1 - alpha) ** seen_count[a],
             utf8_bytes(a))
        marginal = normalized_weights[a] * (1 - alpha) ** seen_count[a]
        idcg += marginal / log2(rank + 1)
        remaining[a] -= 1
        seen_count[a] += 1
    return idcg

function alpha_ndcg(...):
    dcg = alpha_dcg(...)
    idcg = alpha_idcg(...)
    require idcg > 0
    return dcg / idcg
```

Invariants:

- only unique gold documents contribute gain;
- repeated documents in a submitted ranking are invalid rather than receiving
  repeated novelty-discounted gain;
- repeated retrieval from the same aspect is allowed but discounted;
- non-gold documents receive zero gain and still consume rank positions;
- zero-weight aspects receive zero gain and never win an IDCG tie against a
  positive-weight aspect;
- all-zero weights or zero aspects make IDCG zero and invalidate the query;
- tied retrieval scores are ordered by descending score, then UTF-8 byte order
  of `doc_id`;
- tied ideal marginal gains are ordered by UTF-8 byte order of `aspect_id`;
- cutoff behavior is `min(k, candidate_count)` with no padding;
- normalization uses the same `alpha`, weights, gold set, and cutoff as DCG.

The pinned upstream score parser sorts a score mapping by score only and inherits
insertion order for ties. The pinned IDCG loop inherits aspect-map insertion
order for equal marginal gain. Those are reproducible only when upstream JSON
ordering is frozen; the proposed contract replaces them with explicit ids.

### 4.4 Aggregation

The required future outputs are:

1. one metric value and validation status per query;
2. `domain_query_macro`: arithmetic mean over the fixed, fully validated query
   set for each domain;
3. `overall_domain_macro`: unweighted arithmetic mean of the seven domain
   macros for a full seven-domain run;
4. `overall_query_macro`: arithmetic mean over all 739 queries, reported as a
   separate diagnostic because domain sizes differ;
5. attempted, valid, failed, and missing-query counts.

No query may disappear because its score entry, gold list, aspect map, or IDCG
is missing. The current upstream aspect scripts skip missing score entries; the
alpha script also excludes zero-IDCG queries. That behavior is unsuitable for a
public comparable result unless the denominator and failure counts are explicit.

The current `--task all` helpers average per-file/domain averages equally. A
future implementation must name that operation `overall_domain_macro` rather
than a generic `overall`.

## 5. What the Standard MTEB Port Preserves and Loses

MTEB PR 4929 adds seven ordinary `AbsTaskRetrieval` tasks and a `BRIGHT-Pro`
benchmark grouping.

### Preserved

- seven domain identities and full per-domain query/corpus counts;
- query text and corpus text in converted MTEB datasets;
- binary positive document ids derived from `gold_ids`;
- one query instruction per domain;
- fixed dataset revisions for each converted task;
- ordinary static retrieval semantics and standard nDCG@10 as `main_score`;
- expert-annotation metadata and the upstream reference URL;
- a benchmark grouping that allows users to resolve all seven tasks.

### Lost or not evaluated

- aspect ids and descriptions;
- raw aspect importance values and normalized weights;
- document-to-aspect assignments;
- the distinction between covering one aspect repeatedly and covering several
  complementary aspects;
- weighted A-Recall and weighted alpha-nDCG;
- paper-primary alpha-nDCG@25 with `alpha=0.5`;
- aspect-schema validation and weight normalization provenance;
- any direct link between one binary qrel and the aspect it supports.

The MTEB conversion is a useful binary baseline, not an aspect-aware port. Its
`ndcg_at_10` score must not be relabeled as BRIGHT-Pro's aspect-aware primary
metric, and an aspect-aware score must not overwrite the ordinary binary task
row.

## 6. Distinction from Current Repository Surfaces

| Surface | Evaluated subject and contract | Why BRIGHT-Pro aspect retrieval is distinct |
| --- | --- | --- |
| `src/mm_embed/tasks/needle_in_haystack.py` | Compares similarity of a query to a long document with versus without one inserted fact; primary metric is pairwise accuracy across length/position cells. | No ranked full corpus, multi-positive qrels, aspects, aspect weights, or diversity normalization. |
| `src/mm_embed/tasks/code_edit_chunk_localization.py` | Ranks every chunk in a frozen invented repository, requires unique ids and complete-corpus scoring, applies deterministic score/repository/path/line/chunk ties, and reports graded patch-localization metrics. | Useful precedent for deterministic ranking and no-publish fixtures, but its labels are patch-aligned edit targets rather than complementary reasoning aspects. |
| `benchmark/research/code_context_retrieval_minispec_20260722.md` | Defines metadata-first source reconstruction, full-corpus code localization, provenance hashes, licensing gates, and private fixture/smoke/benchmark evidence tiers. | Supplies the right provenance and product discipline, but not BRIGHT-Pro's aspect-weighted metric claim. |
| `benchmark/research/retrieval_answer_utility_minispec_20260727.md` | Separates retrieval-answer systems from embedding rankings; specifies typed answers, citations, brackets, cost, latency, and trace completeness. | BRIGHT-Pro static retrieval stops at a ranking. Generated answer completeness, overall quality, agent rounds, and LLM judging belong to the separate system family. |
| `src/mm_embed/system_evaluation/retrieval_answer_utility.py` and `schemas/system_result.schema.json` | Fixture-only `evaluation.level=system`, `mode=answer_utility`, `subject.kind=retrieval_answer_system`, deterministic typed-answer judging, and `publish=false`. | Aspect-aware static results should remain embedding-level ranking records and must never be written into this system schema. |

The correct family boundary is therefore:

```text
ordinary binary retrieval:      query -> ranked docs -> binary nDCG/Recall
aspect-aware static retrieval:  query -> ranked docs -> weighted coverage/diversity
retrieval-answer system:        query -> retrieval + context + generation -> answer/citation utility
```

An LLM-as-judge answer score is never an embedding metric. Conversely, a static
alpha-nDCG score makes no claim about answer correctness or agent efficiency.

## 7. Risk Register

### 7.1 Duplicate and identity risk

- MTEB statistics show 526,319 document rows but only 423,052 within-domain
  unique-text counts when summed across domains.
- Same-text rows may have different logical ids, sources, aspects, or qrels.
- The upstream aspect map silently overwrites a document assigned to multiple
  aspects.
- The current score-map parser has no explicit tie id after score.
- Top-200 truncation is sufficient for declared cutoffs but cannot prove a
  complete-corpus ranking contract.

Required mitigation: preserve logical ids, compute content hashes, record exact
duplicate and near-duplicate groups, reject duplicate ids and ambiguous
multi-aspect mappings, and make ranking ties explicit.

### 7.2 Leakage and contamination risk

- Queries originate from public Stack Exchange posts and may have been present
  in model pretraining.
- Accepted or high-quality community answers were provided to annotators as a
  starting point and may reveal solution language.
- Positive passages are externally sourced web text and may be discoverable by
  models trained on the open web.
- The benchmark and paper are public, so later models may train on query ids,
  aspect descriptions, weights, positives, or reference answers.
- Exact or near-duplicate passages can cross queries or domains.

Required mitigation: publish benchmark dates, source URLs and hashes where
lawful, duplicate groups, query/answer visibility flags, and model-card training
disclosures. Do not claim contamination-free evaluation for undisclosed models.

### 7.3 Annotation-quality risk

The paper reports field-specific expert annotation, a second same-field review
for every example, and a 50-query independent weight-rescoring sample with
weighted Cohen's kappa 0.742. This is meaningful evidence, but it does not prove:

- exact row-level consistency across all 739 queries;
- absence of overlapping or missing aspects;
- stable doc-to-aspect mapping;
- correct raw weight range in the released payload;
- that every external passage is current, credible, and licensed for reuse.

The 1-to-5 versus 1-to-3 source conflict is a schema-quality blocker, not a
cosmetic documentation issue.

### 7.4 Instruction risk

Retriever-specific instructions differ. Some retrievers use query and document
templates, BM25 uses none, and MTEB supplies one standardized query prompt. A
score can move because of instruction routing rather than embedding weights.

Every future record must include exact query/document templates, their SHA-256,
task type routing, tokenizer/context limits, and whether the model used its
native prompt or the benchmark prompt.

### 7.5 Answer-generation and LLM-derived-label risk

The static aspect annotations and supporting-document judgments are described as
human expert work. However:

- annotators were allowed to use conventional or AI-assisted web search to find
  candidate positives;
- original BRIGHT negative discovery included LLM-generated keywords;
- the paper's reference answers were generated with GPT-5 from human aspects and
  positive passages, then used in agentic answer evaluation;
- agentic completeness and overall quality are assigned by GPT-5;
- the separate RTriever-Synth training corpus is LLM-generated.

Reference answers, agent traces, judge outputs, and synthetic training labels
must stay out of the static embedding primary metric. If retained as diagnostic
metadata, they require model, prompt, revision, cost, and human-review provenance.

## 8. Rights and Republishing Boundary

### 8.1 What is known

- The Bright-Pro GitHub repository presents an MIT license. This clearly covers
  the repository's licensed code and documentation to the extent the licensor
  owns them.
- Stack Overflow's official licensing page says publicly accessible user
  contributions are versioned by contribution date: CC BY-SA 2.5 before
  2011-04-08 UTC, CC BY-SA 3.0 from that date until 2018-05-02 UTC, and CC BY-SA
  4.0 on or after 2018-05-02 UTC. The applicable license for each revision is
  available in the post timeline.
- Stack Overflow's official referencing guidance requires a link to the
  original page, quotation of only the relevant portion, and the original
  author's name when copying or closely rephrasing content.
- The current public-network terms distinguish Stack Overflow content from
  individual subscriber content and state that the Creative Commons data dump
  is licensed under CC BY-SA.

### 8.2 What those facts do not prove

- They do not prove that the Bright-Pro query rows preserve the revision-level
  author, URL, timestamp, and applicable CC BY-SA version needed for compliant
  republication.
- They do not license externally collected positive passage text merely because
  the query came from Stack Exchange.
- They do not establish that copied external pages, merged passages, or manually
  edited segments may be redistributed in a new Dataset.
- They do not establish ownership or public redistribution terms for aspect
  descriptions, weights, document assignments, reference answers, or judge
  annotations.
- They do not make the MTEB metadata value `license="mit"` authoritative for
  underlying text.

### 8.3 Required row-level rights manifest

Before public data or annotations are exported, every retained row needs:

```json
{
  "record_id": "<stable id>",
  "record_kind": "query|document|aspect|qrel|reference_answer",
  "source_url": "<original URL or null>",
  "source_revision": "<immutable revision or capture hash>",
  "source_author": "<required attribution identity or null>",
  "source_timestamp": "<revision timestamp or null>",
  "source_license": "<SPDX/CC identifier or unknown>",
  "source_license_evidence_url": "<evidence URL>",
  "annotation_owner": "<rights holder or unknown>",
  "annotation_license": "<license or unknown>",
  "redistribution_status": "allowed|metadata_only|unknown|forbidden",
  "attribution_text": "<required text or null>"
}
```

Any `unknown` external-text or annotation right forces metadata-only,
`publish=false` handling. The upstream repository license and MTEB metadata are
never substitutes for this manifest.

## 9. Proposed Future Metadata-First Smoke

This is a contract smoke, not a benchmark run and not paper-score comparable.

### 9.1 Preconditions

- an authorized local bundle pins the exact query, document, aspect, qrel, and
  rights metadata needed for at most five queries;
- no Hugging Face, mirror, proxy, alternate host, or cached private artifact is
  used by the smoke;
- source revision, schema, raw weight range, and row counts are reconciled;
- all selected text is either locally authorized for this private test or
  represented by deterministic invented placeholders linked only to metadata;
- manifest sets `publish=false`, `leaderboard_publish=false`, and
  `evidence_tier=smoke`.

If any precondition is absent, the smoke returns `PAUSE` before materialization.

### 9.2 Deterministic domain and query selection

Use salt `bright-pro-aspect-smoke-v0`.

1. Compute `sha256(salt + "\0" + domain)` for all seven canonical domain names.
2. Select the five domains with the lexicographically smallest lowercase hash.
3. Within each selected domain, keep queries that have unique ids, at least two
   unique gold documents, at least two unique aspects, exactly one aspect per
   gold document, finite nonnegative weights with a positive sum, and complete
   rights/provenance metadata.
4. Select the eligible query with the smallest
   `sha256(salt + "\0" + domain + "\0" + query_id)`.
5. Record the full eligible-query count and selected hash so selection cannot be
   hand-tuned to model behavior.

The result is exactly five queries if all five selected domains have an eligible
row; otherwise the smoke pauses. It does not substitute the fixed agentic-175
sample and does not estimate the 739-query static result.

### 9.3 Bounded candidate pool

For each selected query:

1. include every unique gold document;
2. select non-gold documents only from the same domain and pinned corpus;
3. order non-golds by
   `sha256(salt + "\0" + domain + "\0" + query_id + "\0" + doc_id)`;
4. take the first non-golds needed to reach at most 100 total candidates;
5. fail if the gold set itself exceeds 100 or any id/content/aspect ambiguity is
   present;
6. freeze candidate ids and metadata hashes in the smoke manifest.

This bounded pool intentionally changes the retrieval problem. All outputs must
say `candidate_pool=max-100-metadata-smoke` and
`paper_score_comparable=false`.

### 9.4 Deterministic rankings and metric checks

No model, provider, tokenizer, embedding, reranker, generator, or judge runs.
Use pure score fixtures:

- `all_tied`: every candidate score is zero; expected order is UTF-8 `doc_id`;
- `oracle_diverse`: first uncovered positive-weight aspect receives the next
  highest score, with aspect-id and doc-id ties explicit;
- `aspect_collapsed`: documents from the lexicographically first aspect receive
  the highest scores, demonstrating that binary recall can improve while
  weighted coverage/diversity remains limited;
- `reverse_oracle`: gold documents appear after deterministic non-golds;
- invalid variants inject a duplicate doc, duplicate aspect, missing weight,
  multi-aspect gold mapping, zero-aspect query, and all-zero weights.

The smoke asserts exact per-query rankings, weighted A-Recall, alpha-DCG, IDCG,
alpha-nDCG, binary Recall, binary nDCG, domain/query aggregate denominators, and
failure reason codes. Repeat generation must be byte-for-byte stable after
normalizing timestamps.

### 9.5 Failure and cleanup conditions

Fail or pause before scoring when:

- any pinned revision differs;
- the 1-to-5 versus 1-to-3 raw-weight conflict remains unresolved;
- exact aspect or qrel cardinality cannot be reproduced;
- a selected query, document, aspect, or rights record is missing;
- ids are duplicated, a gold document maps to zero or multiple aspects, or an
  aspect has no supporting gold;
- score, weight, alpha, or normalization contains a non-finite value;
- the candidate cap, disk cap, RAM cap, or runtime cap would be exceeded;
- publication flags are not false;
- cleanup cannot remove all staged third-party payloads.

On every outcome, remove the dedicated temporary directory and retain only the
small manifest, expected fixture results, and validation report if Layer 2
accepts a later implementation.

### 9.6 Future smoke resource envelope

| Resource | Exact value or hard cap |
| --- | ---: |
| External network | exactly 0 bytes; requires a pre-authorized local bundle |
| Temporary regular-file bytes | at most 10 MiB |
| Retained Git artifact | at most 250 KiB, metadata and deterministic results only |
| Peak incremental RAM | at most 128 MiB |
| GPU | 0 devices, 0 GPU-seconds |
| Runtime | at most 120 seconds |
| Provider/model cost | USD 0.00 |
| Queries | exactly 5 or PAUSE |
| Candidates | at most 100 per query, at most 500 query-candidate pairs |

## 10. Git, Task Registry, Dataset, and Space Product Path

### 10.1 Current product boundary

The current repository has two task control planes:

- `src/mm_embed/tasks/registry.py` maps implementation names to Python task
  classes;
- `benchmark/tasks/core.yaml` declares reviewable v2 task ids, primary metrics,
  dataset versions, publication flags, and tags.

`src/mm_embed/hf_publish/export.py` currently exports public registry rows to:

```text
models.jsonl
tasks.jsonl
runs/*.yaml
results/latest.jsonl
results/latest-successful.jsonl
leaderboards/latest.csv
benchmark_data/**              # only when include_data is requested
```

The current Space reads `leaderboards/latest.csv`, `models.jsonl`, and
`tasks.jsonl`. The exporter includes only tasks for which both `publish` and
`leaderboard_publish` are true. This is adequate for a future private smoke if
both flags remain false, but the current flat `primary_metric`/`score` CSV does
not preserve aspect rows or support separate binary and aspect-aware score
families under one task id.

### 10.2 Smallest future Git path

After the pause gates are resolved, implement in separate reviewable steps:

1. `src/mm_embed/data/bright_pro_aspect_retrieval.py`
   - metadata loader, revision and rights validator, duplicate audit, aspect
     normalization, and no-network smoke materializer;
2. `src/mm_embed/tasks/bright_pro_aspect_retrieval.py`
   - fixed query/document embedding routing, deterministic ranking, weighted
     metrics, binary diagnostics, and explicit denominator/failure reporting;
3. `tests/test_bright_pro_aspect_retrieval.py`
   - invented fixtures and the at-most-five-query authorized metadata smoke;
4. `src/mm_embed/tasks/registry.py`
   - add only after the fixture contract passes;
5. `benchmark/tasks/core.yaml`
   - start with `publish: false`, `leaderboard_publish: false`, and a distinct
     id such as `bright_pro_aspect_retrieval_smoke`;
6. a separate later public task id `bright_pro_aspect_retrieval` only after
   rights, schema, full-corpus, and 739-query gates pass.

Do not reuse an MTEB binary task id or primary metric. Proposed primary metric:
`aspect_alpha_ndcg@25`; required secondary metric:
`weighted_aspect_recall@25`; binary `ndcg@10` and `recall@k` remain diagnostics.

### 10.3 Metadata-first Dataset path

The current generic export paths should remain backward-compatible. A future
aspect-aware artifact should add task-scoped metadata without copying text by
default:

```text
tasks/bright-pro-aspect-aware/task_manifest.json
tasks/bright-pro-aspect-aware/queries.metadata.jsonl
tasks/bright-pro-aspect-aware/corpus.metadata.jsonl
tasks/bright-pro-aspect-aware/aspects.metadata.jsonl
tasks/bright-pro-aspect-aware/qrels.metadata.jsonl
tasks/bright-pro-aspect-aware/licenses.jsonl
tasks/bright-pro-aspect-aware/provenance.jsonl
results/embedding/bright-pro-aspect-aware/latest.jsonl
leaderboards/embedding/bright-pro-aspect-aware/latest.csv
```

`corpus.metadata.jsonl` may contain document ids, source URLs, capture hashes,
content hashes, duplicate-group ids, domain, and reconstruction instructions.
It must not contain third-party passage text until row-level redistribution is
authorized. Aspect descriptions and weights also remain metadata-only or
withheld when annotation rights are unknown.

Every result must pin query, corpus, aspect, qrel, instruction, normalization,
and rights-manifest hashes. The Dataset card must explain that binary and
aspect-aware metrics answer different questions and are not one score family.

### 10.4 Space path

Add a task-specific `Aspect-aware embedding retrieval` view only after public
gates pass. It should show:

- alpha-nDCG and weighted A-Recall side by side;
- binary nDCG/Recall as diagnostics;
- domain/query macro choice and attempted/failed denominators;
- query, corpus, aspect, qrel, instruction, and metric revisions;
- duplicate and rights coverage indicators;
- evidence tier and paper-comparability status.

It must not:

- insert alpha-nDCG into an ordinary binary task row;
- average binary nDCG and aspect-aware metrics;
- treat the five-query smoke or fixed agentic-175 subset as the full benchmark;
- display LLM-judged answer quality as an embedding score;
- mix retrieval-answer systems into the same sortable subject table.

If retrieval-answer utility is later published, it retains the separate system
schema and Space surface defined by the existing system-evaluation contract.

## 11. GO, PAUSE, and NO-GO Gates

### GO

Change the current conclusion to `GO` for a private implementation only when:

- the exact dataset revision and every per-domain file are pinned;
- complete row schemas and exact query/document/aspect/qrel counts reconcile;
- the raw weight scale conflict is resolved by the dataset owner or payload;
- every gold document maps to exactly one unique aspect;
- instructions and normalization are frozen;
- row-level source and annotation rights support the intended local handling;
- deterministic tie, cutoff, normalization, and aggregation tests pass;
- the first implementation remains no-publish and zero-model.

Public `GO` additionally requires row-level redistribution and attribution
approval, full 739-query/full-corpus reproducibility, leakage and duplicate
reporting, and a Dataset/Space path that cannot mix score families.

### PAUSE

Pause when any schema, revision, aspect mapping, weight normalization, rights,
or full-corpus fact cannot be reconciled within the authorized source cap. That
is the present state.

### NO-GO

Choose `NO-GO` if stakeholders require any of the following:

- collapse aspect weights into binary qrels;
- use LLM answer judging as the embedding primary metric;
- present the 175-query agentic subset or five-query smoke as the 739-query
  static benchmark;
- redistribute external passages or annotations under a blanket MIT claim;
- combine binary, aspect-aware, answer-quality, latency, or cost metrics into
  one global score;
- omit deterministic ranking, denominator, revision, or rights provenance.

## 12. Resource Estimates

### 12.1 This research step

| Resource | Observed or conservative value |
| --- | ---: |
| Accounted network bytes | 5,907,075 bytes (5.63 MiB), including saved response headers |
| Network cap | 12 MiB; passed |
| Peak dedicated temporary directory | 6,044,045 bytes by `du -sb` before cleanup |
| Temporary disk cap | 25 MiB; passed |
| Largest payload | arXiv PDF, 5,146,529 body bytes |
| Retained third-party payload | 0 bytes after cleanup |
| Retained artifact | one Markdown file, conservatively below 64 KiB; exact final size is reported to Layer 2 |
| Peak incremental RAM | conservatively below 256 MiB; no corpus/model materialization |
| GPU | 0 devices, 0 GPU-seconds |
| Provider/model/Hugging Face cost | USD 0.00 |
| Provider/model/Hugging Face operations | 0 |
| Runtime budget | 30 minutes; no benchmark or long-running computation used |

The RAM value is a conservative process envelope, not a profiler measurement.
The PDF-to-text extraction and small JSON/HTML parsing dominate transient use.

## 13. Network and Temporary-Disk Ledger

Every network response was written under
`/tmp/bright-pro-minispec.0sl7p5`, measured as body plus saved response headers,
and deleted after validation.

| Response | Body bytes | Header bytes | Accounted bytes | Exact URL |
| --- | ---: | ---: | ---: | --- |
| `bright_commit` | 34,391 | 1,303 | 35,694 | <https://api.github.com/repos/yale-nlp/Bright-Pro/commits/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e> |
| `bright_tree` | 72,758 | 1,308 | 74,066 | <https://api.github.com/repos/yale-nlp/Bright-Pro/git/trees/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e?recursive=1> |
| `bright_repo` | 6,297 | 1,298 | 7,595 | <https://api.github.com/repos/yale-nlp/Bright-Pro> |
| `mteb_pr` | 21,398 | 1,298 | 22,696 | <https://api.github.com/repos/embeddings-benchmark/mteb/pulls/4929> |
| `mteb_pr_files` | 33,075 | 1,304 | 34,379 | <https://api.github.com/repos/embeddings-benchmark/mteb/pulls/4929/files?per_page=100> |
| `arxiv_abs` | 2,975 | 550 | 3,525 | <https://export.arxiv.org/api/query?id_list=2605.04018> |
| `arxiv_pdf` | 5,146,529 | 746 | 5,147,275 | <https://arxiv.org/pdf/2605.04018v1> |
| `bright_readme` | 5,607 | 899 | 6,506 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/README.md> |
| `bright_license` | 1,065 | 899 | 1,964 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/LICENSE> |
| `bright_retrieval_readme` | 2,834 | 899 | 3,733 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/README.md> |
| `bright_weighted_recall` | 11,503 | 900 | 12,403 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/evaluation/weighted_aspect_recall.py> |
| `bright_alpha_ndcg` | 13,580 | 899 | 14,479 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/evaluation/alpha-ndcg-evaluation.py> |
| `bright_metrics` | 2,694 | 899 | 3,593 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/metrics.py> |
| `bright_run` | 11,449 | 900 | 12,349 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/run.py> |
| `bright_test_metrics` | 6,672 | 899 | 7,571 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/test_metrics.py> |
| `bright_agentic_readme` | 2,910 | 899 | 3,809 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/agentic_eval_outputs/README.md> |
| `bright_data_loader` | 3,640 | 899 | 4,539 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/bright_pro_data.py> |
| `bright_agentic_ids` | 2,339 | 899 | 3,238 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/agentic_sample_ids.json> |
| `bright_sample_script` | 1,433 | 899 | 2,332 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/agentic_retrieval/scripts_evaluation/sample_agentic_qids.py> |
| `bright_config_biology` | 313 | 898 | 1,211 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/configs/rtriever-4b/biology.json> |
| `bright_config_bm25` | 51 | 897 | 948 | <https://raw.githubusercontent.com/yale-nlp/Bright-Pro/5df9e9baf5a0525a2b962d73f213c9dee63c5f3e/retrieval/configs/bm25/biology.json> |
| `mteb_task` | 9,456 | 899 | 10,355 | <https://raw.githubusercontent.com/embeddings-benchmark/mteb/a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64/mteb/tasks/retrieval/eng/bright_pro_retrieval.py> |
| `mteb_stats_Biology` | 1,240 | 899 | 2,139 | <https://raw.githubusercontent.com/embeddings-benchmark/mteb/a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64/mteb/descriptive_stats/Retrieval/BrightProBiologyRetrieval.json> |
| `mteb_stats_EarthScience` | 1,246 | 899 | 2,145 | <https://raw.githubusercontent.com/embeddings-benchmark/mteb/a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64/mteb/descriptive_stats/Retrieval/BrightProEarthScienceRetrieval.json> |
| `mteb_stats_Economics` | 1,239 | 899 | 2,138 | <https://raw.githubusercontent.com/embeddings-benchmark/mteb/a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64/mteb/descriptive_stats/Retrieval/BrightProEconomicsRetrieval.json> |
| `mteb_stats_Psychology` | 1,218 | 899 | 2,117 | <https://raw.githubusercontent.com/embeddings-benchmark/mteb/a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64/mteb/descriptive_stats/Retrieval/BrightProPsychologyRetrieval.json> |
| `mteb_stats_Robotics` | 1,247 | 899 | 2,146 | <https://raw.githubusercontent.com/embeddings-benchmark/mteb/a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64/mteb/descriptive_stats/Retrieval/BrightProRoboticsRetrieval.json> |
| `mteb_stats_Stackoverflow` | 1,236 | 899 | 2,135 | <https://raw.githubusercontent.com/embeddings-benchmark/mteb/a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64/mteb/descriptive_stats/Retrieval/BrightProStackoverflowRetrieval.json> |
| `mteb_stats_SustainableLiving` | 1,242 | 899 | 2,141 | <https://raw.githubusercontent.com/embeddings-benchmark/mteb/a91eef0eb7e02dc75e7e6c6b0b38ae562ecd8a64/mteb/descriptive_stats/Retrieval/BrightProSustainableLivingRetrieval.json> |
| `stack_licensing` | 145,714 | 1,457 | 147,171 | <https://stackoverflow.com/help/licensing> |
| `stack_referencing` | 145,179 | 1,457 | 146,636 | <https://stackoverflow.com/help/referencing> |
| `stack_terms` | 182,590 | 1,457 | 184,047 | <https://stackoverflow.com/legal/terms-of-service/public> |
| **Total** |  |  | **5,907,075** | Under the 12 MiB cap |

Temporary-disk ledger before cleanup:

- network bodies and headers: 5,907,075 bytes;
- extracted arXiv text and other local derivatives brought total regular-file
  bytes to 6,039,949;
- `du -sb` for the dedicated directory: 6,044,045 bytes;
- retained third-party files after cleanup: 0.

## 14. Final Decision

**PAUSE.**

The static aspect-aware task family is technically differentiated and worth
preserving. Its paper-level metric equations, seven-domain query/corpus/qrel
counts, and separation from binary and system evaluation are clear. The current
source set does not provide a sufficiently reconciled data-and-rights contract
for implementation or publication: raw weight range conflicts, exact aspect
schema/cardinality is unavailable under the no-Hugging-Face boundary, and
third-party passage plus annotation redistribution rights remain unproven.

Do not implement, benchmark, publish, or add a public task until those gates are
resolved. The future five-query metadata smoke may proceed only as a separately
authorized no-network, no-model, no-publish contract item.
