# Code Context Retrieval Decision Minispec - 2026-07-22

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_4-1784719569-2_execute.md`

Selected item: `tasks/code-context-retrieval-scout`

Unique session: `meb-modern-embedding-leaderboard-4-1784719569-2-codecontext-5e8c6a91d2f4`

Decision: **DIFFERENTIATE**.

Recommended v0 surface: **edit-chunk/region localization** over a frozen,
query-visible full-repository corpus.

Do not add another file-localization task that merely republishes MTEB's
`SWEbenchCodeRetrieval`, and do not make broader supporting-context retrieval
the first public surface. The repository should use patch-aligned edit regions
as the auditable static target, while treating trajectory-derived, LLM-judged,
and patch-conditioned human context as separate diagnostic evidence until its
provenance, false-negative, and redistribution gates are stronger.

This decision is positive for a bounded task family, but **all real-data
ingestion and all public scoring remain deferred**. The next implementation
should be an invented zero-network fixture contract. The only recommended
real-source execution after that is a one-issue, one-repository, no-publish
smoke once the metadata gates below pass.

## 1. Evidence Labels and Scope

This minispec uses four labels:

- **VERIFIED**: observed in this repository or a current primary source during
  the 2026-07-22 audit.
- **SOURCE CLAIM**: stated by a paper, card, or project, but not independently
  reproducible from the released metadata available in this audit.
- **PROPOSAL**: the recommended repository design.
- **GATE**: a requirement that must pass before a real-data run, public score,
  or redistribution step.

The intended claim is narrow:

> Given a natural-language software issue and the exact pre-change repository
> state, rank deterministic code chunks so that chunks aligned to known patch
> edit spans or insertion anchors appear early.

This is an embedding-retrieval claim. It excludes:

- interactive repository browsing;
- coding-agent trajectories and tool-use efficiency;
- patch generation or test execution as a leaderboard metric;
- LLM-authored queries, labels, or relevance judgments as the v0 gold source;
- family-aware or graph-aware post-processing hidden inside an embedding row;
- shortlist reranking presented as full retrieval; and
- a changed-files-only corpus presented as repository search.

Static cross-encoder reranking may be evaluated later as a separately named
system track over the same fixed corpus and qrels. It must not be merged with
the embedding-only score.

## 2. Current Repository Baseline

### 2.1 Product and runner shape

**VERIFIED.** This repository is a manifest-driven embedding benchmark with
reviewable task/model registries, JSONL results, evidence tiers, and generated
Hugging Face Dataset/Space outputs. New work should continue to use:

- lazy task registration under `src/mm_embed/tasks/registry.py`;
- task metadata under `benchmark/tasks/core.yaml`;
- explicit query/document routing through `retrieval_query` and
  `retrieval_document` where supported;
- deterministic no-network fixtures before real external data;
- `publish: false` for fixtures and contract smokes; and
- explicit `run.publish` and `run.evidence_tier` in future run records.

**GATE.** The repository still has no root `LICENSE`, `COPYING`, or `NOTICE`
file. Even self-authored fixture publication needs a repository-license
decision, and third-party issue/code redistribution needs a stricter per-source
license chain.

### 2.2 Why this is not agent memory

`agent_procedural_tool_memory` evaluates a task request against sanitized tool
cards. It currently has one positive document per query and curated procedural
hard negatives. It does not contain:

- a repository snapshot;
- issue-to-code alignment;
- path, blob, commit, or line-span identity;
- patch-derived edit evidence; or
- millions of dense in-repository distractors.

The proposed code task therefore is not a renamed memory slice. Tool cards are
stable knowledge objects; code chunks are versioned artifacts whose relevance
depends on an issue and a specific repository state.

### 2.3 Why this is not skill routing

`agent_skill_compatible_set_retrieval` and
`agent_skill_same_capability_risk` evaluate multi-positive compatible sets and
query-specific risky siblings over an invented skill library. Their labels are
set and family relations, not edit spans. They intentionally avoid execution
claims and use a tiny no-publish fixture.

The proposed code task has no compatible-set or risky-sibling ontology. It
uses repository-local chunks, binary or explicitly graded edit overlap, and
path/commit provenance. Similar implementations or tests are hard negatives,
not globally risky siblings.

### 2.4 Why this is not late chunking

`late_chunking_retrieval` evaluates whether independent and contextualized
chunk embeddings preserve evidence from a self-created parent document. It
varies segmentation layouts while keeping query, parent, evidence span, and
chunk identity deterministic.

The proposed code task instead fixes a code chunker and asks which chunks in a
real, frozen repository should be edited for an issue. Contextualized or late
chunk embeddings could later be one model strategy on this task, but the task
does not measure contextualization gain. The qrels source, failure modes,
license risks, and leakage surface are different.

## 3. Primary-Source Audit

### 3.1 MTEB `SWEbenchCodeRetrieval`

#### Current task and release state

**VERIFIED.** MTEB `main` was audited at commit
`80f6e3b89539a433589ac4a685d3a9f86e4f0f10` (2026-07-21). The latest
GitHub release observed during this audit was `2.18.5` (2026-07-19).

The current task source declares:

- task name: `SWEbenchCodeRetrieval`;
- source family: SWE-bench Verified;
- 500 real issues from 12 Python repositories;
- query: issue description;
- corpus: Python source files;
- qrels: source files changed by the gold patch;
- dataset: `embedding-benchmark/SWEbenchCodeRetrieval`;
- pinned dataset revision:
  `440b0e732b8d02c16df2c95352ab6770abe997da`;
- main score: `ndcg_at_10`; and
- task-level license claim: `mit`.

The task was introduced by MTEB commit
`dcc5965c3ebf663f58c8f51fdb957a9d00bd3326` on 2026-05-12.
The corresponding PR distinguishes this task from prior SWE-bench reranking:
the full corpus is searched rather than a prefiltered shortlist.

#### Dataset shape and size

**VERIFIED through the current consumer contract and descriptive statistics.**
The current standard MTEB retrieval loader expects:

```json
{"config": "corpus", "columns": ["_id or id", "title optional", "text"]}
{"config": "queries", "columns": ["_id or id", "text"]}
{"config": "default or qrels", "columns": ["query-id", "corpus-id", "score"]}
```

The task's current descriptive-statistics file reports:

- 58,058 corpus documents;
- 500 queries;
- 621 qrels;
- 56,786 unique corpus texts;
- 1,307,815,716 corpus characters;
- 849,866 query characters;
- 1 to 21 relevant documents per query, mean 1.242; and
- maximum corpus document length 1,508,731 characters.

The addition PR says the 58,058 documents were content-hash deduplicated from
roughly 700K raw files and uploaded in standard MTEB format. The generation
script and row-level source manifest are not present in the MTEB repository.

**GATE.** Normal `huggingface.co` DNS resolution failed from this host during
this audit. The pinned dataset card and file list therefore could not be
refreshed directly without using a forbidden alternate IP. The pinned MTEB
task, loader, descriptive statistics, PR, and results are available, but the
card text, compressed repository size, row examples, and card-specific license
statement remain unverified in this run.

#### Qrels and embedding validity

**VERIFIED.** The qrels are patch-file labels: a file is relevant when the
gold patch changes it. This is a valid static embedding task because the query,
corpus, qrels, and metric are fixed before model scoring.

However, file labels are coarse:

- a large file receives one positive label even if only one small function is
  edited;
- every changed file is treated as relevant even when a patch contains
  mechanical or incidental changes;
- supporting files that enable a valid alternate solution are false negatives;
  and
- file retrieval cannot distinguish finding the right module from finding the
  right region.

The standard MTEB loader exposes one global corpus and no query-specific
snapshot filter. Given that the PR describes content-hash deduplication over a
much larger collected raw-file pool, this creates an unresolved risk that a
query is scored against stale or future versions of same-repository files. The
dataset generation code is required to confirm exactly how document ids encode
repository, path, and commit.

#### Reference-result evidence

**VERIFIED.** The official results repository contains a merged reference
result commit `dc3f7e890385c960973d53de6e831bebe7c979f9` and currently
tracks four reference rows at the pinned dataset revision. Examples:

| Reference row | nDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| BM25S baseline | 0.07764 | 0.15883 | 0.055659 |
| multilingual-e5-small | 0.04537 | 0.08450 | 0.036547 |
| static-similarity-mrl-multilingual-v1 | 0.02545 | 0.05600 | 0.016603 |
| potion-multilingual-128M | 0.02428 | 0.05000 | 0.018363 |

This is enough to establish that the task is runnable and nontrivial, but not
enough to resolve the provenance and cross-version questions above.

#### License conclusion

**GATE.** `license="mit"` in MTEB task metadata is a dataset-level claim, not
proof that Python files copied from 12 independently licensed repositories can
all be redistributed under MIT. A future import must preserve each repository
license at the exact base commit, its notice/attribution requirements, and the
relationship between the dataset wrapper license and upstream file licenses.

### 3.2 CORE-Bench

#### Current version and release state

**VERIFIED.** The current arXiv record is `2606.11864v2`, revised
2026-07-13. It describes three levels:

1. code understanding;
2. issue-to-edit localization; and
3. broader context retrieval.

The dispatch names CORE-Bench v1, but the public evaluation repository has no
`v1` tag or GitHub release. This audit therefore pins the current paper v2 and
the initial public evaluation commit rather than inventing a version alias.

The official evaluation repository was audited at commit
`5d00cdcaade77796fa6e5ac4852404002029c00a` (2026-07-13). It has no
Git tag or GitHub release. Its README points to
`zhangfw123/CORE-Bench` on Hugging Face, but the evaluation code uses
local dataset paths with floating `revision="main"` metadata rather than a
pinned dataset commit.

The repository API reports no declared code license, and the Level-2/Level-3
MTEB task metadata sets `license=None`.

#### Level 2: issue-to-edit localization

**VERIFIED from paper v2 and current evaluator source.** The construction is:

1. collect PR/issue metadata from SWE-bench-family sources;
2. filter answer-leaking, underspecified, or non-actionable queries with
   Qwen3.5-397B-A17B;
3. check out the repository immediately before the PR;
4. chunk source and documentation with AST and LangChain splitters, retaining
   paths and line spans; and
5. align modified files and edited line ranges back to pre-change chunks.

The released Level-2 shape is:

- 632 repositories;
- 5,061 queries;
- 9,377,120 chunks;
- 52,712 qrels;
- 10.42 relevant chunks per query on average;
- 1,558.5 average query characters; and
- 1,004.9 average corpus characters.

The source families are SWE-bench Pro, Verified, Live, SWE-bench++,
SWE-bench+, Multi-SWE-bench, and SWE-bench Multilingual. The evaluator
loads one `corpus.jsonl`, `queries.jsonl`, and `qrels.jsonl` per repository,
with this minimum shape:

```json
{"_id": "chunk-id", "text": "pre-change code or documentation chunk"}
{"_id": "query-id", "text": "issue or PR request", "filtered_corpus_id": "snapshot-filter-id"}
{"query_id": "query-id", "corpus_id": "chunk-id", "score": 1}
```

The paper reports query-visible scoring over repository/commit-filtered chunks
after merging and deduplicating temporal snapshots. This is more faithful than
a global cross-version corpus, but the current released loader only stores
`filtered_corpus_id`; the filtering behavior lives in the modified MTEB path
and must be reproduced exactly before results are comparable.

CORE-Bench reports `NDCG@10` and `Recall@100`. The former emphasizes early
edit localization; the latter emphasizes coverage of broader context.

**Embedding conclusion:** Level 2 is the strongest direct design precedent for
this repository's v0. Patch-aligned pre-change chunks are static, auditable,
and compatible with dense, sparse, and reranking evaluation.

#### Level 3: broader context retrieval

**VERIFIED from paper v2.** Level 3 starts with Level-2 edit qrels, then adds
supporting chunks from an automated agent-and-judge pipeline:

- Mini-SWE-Agent reruns tasks with Qwen3.5-397B-A17B;
- `cat`, `grep`, `head`, and `sed` browse actions are aligned to chunks;
- two Qwen votes and one Claude Sonnet 4.6 vote judge relevance;
- allowlist reruns compare Level-2 and Level-3 context utility; and
- feedback iterations add more judged trajectory context.

The released Level-3 shape is:

- 97 repositories;
- 2,580 queries;
- 2,609,581 chunks;
- 106,479 qrels;
- 41.27 relevant chunks per query on average;
- 1,074.4 average query characters; and
- 1,154.8 average corpus characters.

**Embedding conclusion:** the final retrieval evaluation is static, but its
gold construction is not embedding-neutral. It inherits explorer behavior,
model-specific blind spots, LLM relevance votes, and allowlist-agent
validation. It is useful as a separate evidence tier, not as the first public
embedding gold standard.

#### License and release conclusion

The paper states that CORE-Bench uses public benchmark data and open-source
repositories and follows upstream licenses. That is a **SOURCE CLAIM**, not a
row-level redistribution ledger. The current code repository has no license,
the task metadata has no license, the Hugging Face dataset is not pinned by
the evaluator, and normal HF DNS prevented a direct card/file audit.

**GATE.** Do not download, ingest, or republish CORE-Bench data until the
dataset revision, file list, data license, per-row source benchmark, base
commit, repository license, chunk text hash, and qrel score semantics are
audited.

### 3.3 SWE-Explore

#### Current version and release state

**VERIFIED.** The current paper is `2606.07297v1` (2026-06-05). The official
repository was audited at commit
`3c12dc5a551937038afcbdb6eb6bbf19f3ddd8c1` (2026-06-08). It has no tag
or GitHub release. The code repository is MIT licensed.

The README explicitly says the dataset has data-specific terms on Hugging
Face; the code license is not presented as the dataset license. Normal HF DNS
prevented direct inspection of the dataset card and current dataset commit.

#### Task and released record contract

SWE-Explore asks an explorer to return a ranked list of regions:

```text
(issue, repository snapshot) -> [(path_1, start_1, end_1), (path_2, start_2, end_2)]
```

The released benchmark contains:

- 848 issues;
- 203 open-source repositories;
- 10 programming languages;
- 4.3 core files per instance on average;
- 4.7 core regions per instance on average;
- 1,578 core lines per instance on average;
- 2.9 successful source trajectories per instance on average;
- 759 non-test files per repository snapshot on average; and
- about 179.6K non-test source lines per snapshot on average.

Each record includes issue and repository metadata, repository-relative line
regions, trajectory provenance, core context, model-specific optional context,
modified-core files, and main files. Evaluation requires the exact base-commit
repository snapshot; the repository helper fetches GitHub archives rather than
shipping all code inside the benchmark record.

#### Core versus optional labels

**VERIFIED.** An instance is retained only when at least two independent
agent trajectories successfully solve it. Read operations are converted into
file/line regions.

- Raw core: file-wise line intersection across successful trajectories.
- Raw optional: the successful-read union outside the intersection.
- Refined core: selected optional regions are promoted by an LLM-assisted
  refinement step, then every promoted region is manually audited.
- Main scoring target: refined core only.
- Optional context: diagnostics and context-efficiency computation.

This is a strong behavioral contract, but it is not a model-independent notion
of all valid code context. The paper explicitly notes survival bias toward
agent-solvable issues and that valid alternate solutions may use different
evidence.

#### Metrics and release-code gate

The paper and README define:

- line precision, recall, and F1;
- file and region hit rates;
- weighted core coverage;
- context efficiency and noise;
- first useful hit; and
- line-budget `Recall@B` and `nDCG@B`, with `B` in `{100, 300, 500}` and
  500 as the primary paper budget.

The fixed line budget is valuable because a whole-file result can consume the
budget without proportional gain.

**GATE.** The current evaluator and paper are not obviously identical. At the
audited commit, `eval.py` gives overlap in `main_files` a 1.5 gain multiplier
inside line-budget nDCG and uses a gain-density ideal order. Paper v1 describes
gain as newly covered core lines and a greedy marginal-gain ideal order.
Before adapting or comparing scores, the project must identify the canonical
metric contract and pin code plus data revisions together.

#### Embedding conclusion

SWE-Explore can evaluate a static retriever if one adds a deterministic
repository chunker and converts each chunk to a ranked region. That adaptation
would be new: the official benchmark intentionally compares classical
retrievers, arbitrary region-producing agents, and specialized localizers in
one output format. Static embedding evaluation cannot claim exploration,
trajectory efficiency, or downstream repair quality.

SWE-Explore is therefore design evidence for line-budget metrics and region
identity, not the recommended source for v0 qrels.

### 3.4 ContextBench

#### Current version and release state

**VERIFIED.** The current paper is `2602.05892v3` (2026-02-11). The official
repository was audited at commit
`1436c28a8eb95496da4ea69ad458b9f8a8eb7d61` (2026-06-12). It has no tag
or GitHub release and declares Apache-2.0 at the repository root.

The dispatch names ContextBench v1, but the official repository exposes no
`v1` tag or GitHub release. Current facts below are pinned to paper v3 and the
audited Git commit.

Unlike the other two recent releases, the GitHub repository directly tracks
the data artifacts. At the audited commit:

- `data/full.parquet` is 26,102,607 bytes;
- `data/contextbench_verified.parquet` is 9,558,250 bytes; and
- the released full table has 1,136 tasks, with a 500-instance verified
  subset.

The parser expects fields including:

```json
{
  "instance_id": "source_instance_id",
  "original_inst_id": "original_source_instance_id",
  "repo": "owner/name",
  "repo_url": "https://github.com/owner/name",
  "base_commit": "40-hex-sha",
  "gold_context": "serialized file/start_line/end_line records",
  "patch": "source_patch_text",
  "test_patch": "source_test_patch_text",
  "source": "source benchmark",
  "language": "source_language"
}
```

#### Human gold-context contract

ContextBench pools SWE-bench Verified, Multi-SWE-bench, SWE-PolyBench PB500,
and SWE-bench Pro, then performs:

1. exact and embedding-based task deduplication;
2. selection using agent solvability, edit scope, and edit dispersion;
3. patch-driven dependency tracing by expert developers;
4. sufficiency verification by asking a strong LLM for five patches using
   only the annotated context and running the official tests; and
5. cross-annotator compactness review and consensus.

The release contains:

- 1,136 tasks from 66 repositories and eight languages;
- 4,548 gold files;
- 23,116 gold blocks; and
- 522,115 human-verified gold-context lines.

Gold context is evaluated at file, symbol, span, line, and edit-location
granularity. Process metrics include final recall/precision/F1, trajectory
AUC-coverage, redundancy, and evidence drop between explored and finally used
context.

#### Why it is not directly adaptable as v0

A deterministic query-to-chunk ranking could be scored against ContextBench
gold spans, but that would omit the central official contract:

- intermediate context snapshots;
- order of repository exploration;
- AUC-coverage and repeated-read redundancy;
- evidence retained for patching; and
- the distinction between explored and utilized context.

The gold is also conditioned on the reference patch and traced dependency
paths. Alternate valid patches can require different context. The paper studies
gold robustness on only 82 multi-patch cases, so a public static embedding task
would need a broader alternate-solution audit and an explicit false-negative
policy.

**License GATE.** Apache-2.0 at the benchmark repository root does not erase
the licenses of issue text, patches, and copied code context from upstream
repositories. Before redistribution, retain per-row source benchmark,
repository/base commit, license file hash, attribution, and data-specific
terms.

ContextBench is valuable as later human-context diagnostic evidence. It is not
the recommended first embedding leaderboard surface.

### 3.5 Smaller releases intentionally excluded

No smaller public code-retrieval dataset is added to this decision matrix.
The four required families already cover the relevant design space, and adding
another release without a directly audited artifact revision, repository
snapshot map, row provenance, and data-specific license would broaden the
survey without improving the decision.

## 4. Decision Matrix

| Family | Task object | Query | Candidate unit | Qrels source | Main metrics | Scale | Machine fit | License/provenance | Leakage and false-negative risk | Static embedding validity |
|---|---|---|---|---|---|---:|---|---|---|---|
| MTEB `SWEbenchCodeRetrieval` | Full retrieval | SWE-bench Verified issue | Whole Python file in a global deduplicated corpus | Gold patch changed files | nDCG@10 plus standard MTEB retrieval metrics | 500 queries; 58,058 files | Feasible but not tiny | Pinned dataset; task says MIT; row/source ledger unverified | Cross-version corpus, patch-file coarseness, alternate solutions | **High** |
| CORE Level 2 | Repository/commit-filtered full retrieval | Filtered issue or PR request | AST/LangChain code/doc chunk | Patch file plus edited-line overlap | NDCG@10, Recall@100 | 5,061 queries; 9.38M chunks | Full set is heavy | Dataset is floating in evaluator; code/task license absent | Patch bias, generated filtering, temporal duplicates, alternate edits | **High** |
| CORE Level 3 | Repository/commit-filtered full retrieval | Same family as Level 2 | Code/doc chunk | Level-2 edits plus trajectory extraction and LLM votes | NDCG@10, Recall@100 | 2,580 queries; 2.61M chunks | Heavy but smaller than L2 | Same release gates as L2 | Agent/LLM label bias, survival bias, supporting-context false negatives | **Medium** |
| SWE-Explore | Ranked region production | Issue plus snapshot | Arbitrary `(path, line range)` emitted by retriever or agent | Successful-trajectory intersection, LLM promotion, manual audit | Line recall/F1, nDCG@500 lines, efficiency, first hit | 848 issues; 203 repos | Metadata small; snapshots and agent runs heavy | Code MIT; dataset terms and revision unverified | Agent-solvable survival bias, trajectory/model bias, metric-code drift | **Medium after a new chunk adapter** |
| ContextBench | Process-oriented trajectory evaluation | Issue plus snapshot | Files, symbols, spans, and cumulative observed context | Patch-driven expert annotation plus LLM/test verification | Recall/precision/F1, AUC-coverage, redundancy, evidence drop | 1,136 tasks; 66 repos | Data table small; repo snapshots/process runs heavy | Repo Apache-2.0; upstream row rights still mixed | Patch-conditioned gold, alternate solutions, process/static mismatch | **Low for direct adoption; medium for a new diagnostic** |
| Proposed v0 | Repository/commit-filtered full retrieval | Original issue text at a pinned revision | Deterministic pre-change edit chunk/region | Exact patch-hunk overlap or insertion-anchor mapping | edit-chunk nDCG@10 primary; coverage diagnostics | One-repo smoke first; audited multi-repo later | Bounded by explicit caps | Metadata-only until per-row rights pass | Known patch bias, surfaced through slices and evidence tier | **High** |

## 5. Decision Rationale

### Why not ADOPT file localization

MTEB already has a current, pinned, runnable full-retrieval file task with
reference results. Recreating it here would add maintenance and license risk
without a distinct scientific claim.

File-level diagnostics should still be emitted for interpretability, but they
must be secondary metrics derived from the edit-chunk ranking.

### Why DIFFERENTIATE with edit chunks

Edit-chunk/region localization is the smallest surface that:

- measures a harder gap than whole-file retrieval;
- remains static and provider-neutral;
- uses exact pre-change code rather than patch text as a candidate;
- has deterministic qrels from modified line overlap;
- supports genuine full-repository distractors;
- can compare dense, sparse, late/contextual, and static reranking strategies;
- does not require agents, patch generation, or LLM annotations; and
- creates a useful bridge to line-budget context evaluation later.

### Why broader context is deferred

Broader context is important, but the strongest current labels are either:

- successful-agent reads plus LLM votes (CORE Level 3);
- successful-trajectory intersections plus LLM promotion and manual audit
  (SWE-Explore); or
- patch-conditioned expert dependency traces verified through LLM-authored
  patches (ContextBench).

These sources can support a later `diagnostic` or `human_context` label tier.
They should not define the first embedding primary because they confound
retrieval quality with label-generator behavior and leave many alternate valid
contexts unlabeled.

## 6. Proposed V0 Contract

Provisional task id: `code_edit_chunk_localization`

Display name: `Issue-to-edit chunk localization`

Required modality: `text`

Primary metric: `edit_chunk_ndcg@10`

Metric direction: higher

Initial publication state: `publish: false`

### 6.1 Query schema

```json
{
  "query_id": "requests-1920",
  "repository_id": "psf/requests",
  "source_family": "github_issue_pr_pair",
  "source_dataset": null,
  "source_dataset_revision": null,
  "issue_url": "https://github.com/psf/requests/issues/1920",
  "issue_number": 1920,
  "issue_title": "Removing a default header of a session",
  "issue_body": "normalized_issue_body_from_source",
  "issue_updated_at": "2021-09-09T00:10:02Z",
  "query_text_sha256": "required_at_materialization",
  "base_commit": "3c88e520da24ae6f736929a750876e7654accc3d",
  "change_pr_url": "https://github.com/psf/requests/pull/1921",
  "split": "smoke_only",
  "language": "en",
  "answer_leak_review": "pass",
  "privacy_review": "pass",
  "public_score_eligible": false
}
```

The public issue title/body is the query. PR title/body, patch, changed-file
names, commit message, and post-change code are forbidden query inputs.

### 6.2 Repository snapshot schema

```json
{
  "repository_id": "psf/requests",
  "base_commit": "3c88e520da24ae6f736929a750876e7654accc3d",
  "tree_sha": "cf441cdcaa76806a078893bffa79922351a92b0a",
  "archive_url": "https://github.com/psf/requests/archive/3c88e520da24ae6f736929a750876e7654accc3d.tar.gz",
  "tracked_blob_count": 121,
  "tracked_blob_bytes": 1435436,
  "suffix_candidate_file_count": 99,
  "suffix_candidate_blob_bytes": 865106,
  "eligible_text_file_count": null,
  "eligible_normalized_text_bytes": null,
  "repository_license_spdx": "Apache-2.0",
  "license_file_path": "LICENSE",
  "license_file_sha256": "89478f1915fbb6b6585a685d071bf006ba5649d2615fab787f66e9693c622ae4",
  "source_audit_status": "metadata_verified_content_not_downloaded",
  "public_redistribution": false
}
```

The tracked-blob counts are **VERIFIED metadata** from the recursive GitHub
tree API response at the base commit. The `99` files and `865106` bytes are
more limited: they were derived from that response by selecting every entry
whose `type` was `blob` and whose path matched this case-insensitive regex:

```regex
\.(py|pyi|md|rst|txt|toml|cfg|ini|yaml|yml|json)$
```

The exact metadata calculation was:

```jq
[
  .tree[]
  | select(.type == "blob")
  | select(.path | test("\\.(py|pyi|md|rst|txt|toml|cfg|ini|yaml|yml|json)$"; "i"))
]
| {
    suffix_candidate_file_count: length,
    suffix_candidate_blob_bytes: (map(.size) | add // 0)
  }
```

The byte total is the sum of the API's Git blob `size` fields. This calculation
did not inspect Git modes, apply the path and basename exclusions below, or
download and decode blob content. The values are therefore suffix-candidate
metadata, not verified eligible-text counts. Final eligible counts remain
`null` until the pinned tree is materialized and every candidate passes the
following policy.

#### 6.2.1 Reproducible eligible-corpus policy

Eligibility is the conjunction of Stage A metadata selection and Stage B
content validation. Implementations must record the outcome and rejection
reason at each stage; they must not add repository-specific exclusions.

**Stage A: metadata candidate selection**

1. Read the pinned recursive Git tree and consider only regular blobs whose
   mode is exactly `100644` or `100755`. Exclude symlink blobs (`120000`),
   submodules (`160000`), trees, and any other mode.
2. Treat the Git tree API `path` string as a repository-relative POSIX path and
   retain the exact string for identity. Reject an empty path, a leading `/`, a
   NUL or backslash, or any empty, `.` or `..` path segment. Do not apply
   Unicode normalization or locale-sensitive case folding. For comparisons
   below, ASCII lowercase means mapping only `A`-`Z` to `a`-`z`.
3. Exclude a path if any ASCII-lowercased segment is exactly one of:
   `.git`, `vendor`, `vendors`, `third_party`, `third-party`, `node_modules`,
   `.venv`, `venv`, `dist`, `build`, `.tox`, `.nox`, `.pytest_cache`, or
   `__pycache__`.
4. Exclude a path if its ASCII-lowercased basename is exactly one of:
   `package-lock.json`, `npm-shrinkwrap.json`, `pipfile.lock`, `poetry.lock`,
   `uv.lock`, `yarn.lock`, `pnpm-lock.yaml`, `cargo.lock`, `gemfile.lock`, or
   `composer.lock`.
5. After those exclusions, include a path only when its ASCII-lowercased
   basename ends in exactly one of these suffixes: `.py`, `.pyi`, `.md`,
   `.rst`, `.txt`, `.toml`, `.cfg`, `.ini`, `.yaml`, `.yml`, or `.json`.
   The suffix includes the final dot; a name such as `file.py.txt` is included
   by its final `.txt` suffix.
6. Exclude every other suffix and every extensionless basename. In particular,
   names such as `LICENSE`, `NOTICE`, `COPYING`, `README`, `Makefile`, and
   `MANIFEST.in` are not candidates unless a future dataset version explicitly
   adds them. Minified JavaScript/CSS, source maps, bytecode, archives, images,
   audio, video, and compiled objects are excluded because none of their
   suffixes is allowlisted; there is no additional heuristic file-type rule.

Stage A does not infer whether an allowlisted file is actually text, generated,
or safe to redistribute. A generated `.py` or `.json` file outside an excluded
path remains a candidate. Conversely, an extensionless UTF-8 file remains
excluded. Because the recorded `99`/`865106` calculation applied only the blob
type and suffix regex, the Stage A count must be recomputed rather than assumed
to equal those values.

**Stage B: content validation after materialization**

For each Stage A candidate, read the exact blob identified by the pinned tree
and apply these checks in order:

1. Verify its Git blob SHA and API size, then reject it as `oversize` when the
   raw blob is larger than `2,000,000` bytes.
2. Reject it as `lfs_pointer` when the raw bytes begin with the ASCII line
   `version https://git-lfs.github.com/spec/v1` followed by LF or CRLF.
3. Reject it as `binary_nul` when any raw byte is `0x00`.
4. Remove at most one leading UTF-8 BOM byte sequence (`EF BB BF`) for decoding,
   then decode the complete remainder with strict UTF-8. Reject any decoding
   error as `invalid_utf8`; never replace or ignore invalid bytes.
5. Let `n` be the post-BOM byte length and let `c` be the count of bytes in
   `01`-`08`, `0B`, `0E`-`1F`, or `7F` hexadecimal. TAB (`09`), LF (`0A`),
   form feed (`0C`), and CR (`0D`) are allowed. Reject as `control_heavy` when
   `100 * c > max(1, n)`, which is a strict greater-than-one-percent rule.
6. For accepted text, normalize CRLF to LF and then remaining CR to LF. Do not
   perform Unicode normalization, whitespace trimming, tab expansion, or final
   newline insertion.

Record, at minimum, the original path, Git mode, blob SHA, raw byte count,
Stage A result/reason, Stage B result/reason, BOM presence, decoded UTF-8 byte
count before newline normalization, normalized UTF-8 byte count, and SHA-256 of
the normalized UTF-8 bytes. A file is an eligible corpus file only when both
stages accept it. `eligible_text_file_count` and
`eligible_normalized_text_bytes` are sums over those accepted files and must
remain unknown until this inspection is complete.

### 6.3 Corpus chunk schema

```json
{
  "chunk_id": "psf_requests@3c88e520:requests/sessions.py:ast:function:merge_setting:0001",
  "repository_id": "psf/requests",
  "base_commit": "3c88e520da24ae6f736929a750876e7654accc3d",
  "path": "requests/sessions.py",
  "blob_sha": "required_at_materialization",
  "language": "python",
  "candidate_family": "ast_function",
  "symbol": "merge_setting",
  "line_start": null,
  "line_end": null,
  "char_start": null,
  "char_end": null,
  "text": "exact_normalized_pre_change_text",
  "text_sha256": "required_at_materialization",
  "chunker_version": "code-edit-ast-fallback-v0",
  "ordinal_in_file": null,
  "repository_license_spdx": "Apache-2.0",
  "redistribution_status": "local_materialization_only"
}
```

Chunking policy:

1. Parse supported code files into definition-level functions, methods,
   classes, and module preambles.
2. Preserve path, symbol, and exact 1-indexed closed line spans.
3. Split definitions above the model-independent maximum into deterministic
   overlapping windows.
4. Use deterministic fixed-line fallback for unsupported syntax, prose,
   configuration, and parse failures.
5. Include every eligible tracked text file in the base commit, not only files
   mentioned by the issue or patch.
6. Apply Section 6.2.1 exactly. Do not make ad hoc generated-file, vendored,
   encoding, or repository-specific exclusions.

The chunker is part of the dataset version. Model tokenizers must not change
candidate boundaries.

### 6.4 Qrels schema

```json
{
  "query_id": "requests-1920",
  "chunk_id": "psf_requests@3c88e520:requests/sessions.py:ast:function:merge_setting:0001",
  "relevance": 2,
  "label_family": "insert_anchor_containing_chunk",
  "patch_url": "https://github.com/psf/requests/pull/1921.patch",
  "patch_base_commit": "3c88e520da24ae6f736929a750876e7654accc3d",
  "changed_path": "requests/sessions.py",
  "patch_change_type": "addition",
  "preimage_line_start": null,
  "preimage_line_end": null,
  "insertion_anchor_after_line": 61,
  "overlap_lines": 0,
  "mapping_status": "exact",
  "patch_raw_sha256": "aa2328cd30a6815cdaf612ccc4dcb7a2626870368948bfe4c96225b4216abfaf",
  "alternate_solution_review": "unknown",
  "public_score_eligible": false
}
```

Recommended relevance grades:

- `2`: chunk directly intersects modified/deleted preimage lines, or contains
  the exact insertion anchor for an addition-only hunk;
- `1`: optional diagnostic only when a deterministic structure-preserving
  expansion is declared in advance, such as the containing definition for a
  line-fallback edit; and
- `0`: unlabeled, not proven irrelevant.

The primary v0 should use direct grade-2 qrels. Grade-1 expansion must not be
introduced merely to make retrieval easier.

### 6.5 Provenance schema

```json
{
  "record_id": "prov_requests_1920_v0",
  "issue_source": {
    "url": "https://github.com/psf/requests/issues/1920",
    "retrieved_at": "required_at_materialization",
    "sha256": "required_at_materialization"
  },
  "patch_source": {
    "url": "https://github.com/psf/requests/pull/1921.patch",
    "retrieved_at": "required_at_materialization",
    "sha256": "aa2328cd30a6815cdaf612ccc4dcb7a2626870368948bfe4c96225b4216abfaf"
  },
  "repository_source": {
    "url": "https://github.com/psf/requests",
    "base_commit": "3c88e520da24ae6f736929a750876e7654accc3d",
    "tree_sha": "cf441cdcaa76806a078893bffa79922351a92b0a"
  },
  "license_source": {
    "path": "LICENSE",
    "spdx": "Apache-2.0",
    "sha256": "89478f1915fbb6b6585a685d071bf006ba5649d2615fab787f66e9693c622ae4"
  },
  "normalization": "unicode-nfc-lf-v1",
  "chunker_version": "code-edit-ast-fallback-v0",
  "qrel_generator_version": "patch-preimage-and-insertion-anchor-v0",
  "dedup_version": "blob-and-text-sha256-v0",
  "privacy_scan_version": "secret-pii-path-v0",
  "review_status": "metadata_only"
}
```

## 7. Hard-Negative Families

Hard negatives are diagnostics, not extra gold. Each must retain the reason,
source chunk id, and false-negative review status.

1. `same_file_neighbor`
   - a nearby definition in the same changed file that does not overlap an
     edit hunk;
   - high false-negative risk; manual review required.
2. `same_symbol_family`
   - similarly named methods, overloads, adapters, or implementations in
     different modules.
3. `test_implementation_collision`
   - test and implementation chunks sharing the issue vocabulary but serving
     different roles.
4. `error_message_collision`
   - chunks containing the same exception, log message, configuration key, or
     API name without implementing the change.
5. `same_subsystem_path`
   - chunks from the same package/directory that are structurally close but
     not edited.
6. `documentation_code_collision`
   - docs, changelog, or examples that describe the behavior but are not
     patch-aligned edit targets.
7. `cross_version_same_path`
   - for leakage analysis only; a same-path blob from another commit must never
     enter the query-visible v0 corpus.
8. `patch_file_non_edit_chunk`
   - unedited chunks in a changed file; report separately because many may be
     valid supporting context rather than true negatives.

Do not hard-label a candidate when:

- it is an alternate implementation path;
- the patch contains a broad refactor or generated update;
- it is necessary supporting context under a plausible solution;
- its only negative evidence is absence from the reference patch; or
- the path or line mapping is ambiguous.

## 8. Metrics and Diagnostic Slices

### 8.1 Primary and required metrics

Primary:

- `edit_chunk_ndcg@10`: graded nDCG over directly patch-aligned chunks.

Required secondary metrics:

- `edit_chunk_recall@1`;
- `edit_chunk_recall@5`;
- `edit_chunk_recall@10`;
- `edit_chunk_recall@100`;
- `edit_chunk_mrr`;
- `edit_target_recall@100_lines`;
- `edit_target_recall@300_lines`;
- `edit_target_recall@500_lines`;
- `first_edit_hit_rank`;
- `file_recall@1`, `file_recall@5`, and `file_recall@10` after collapsing the
  chunk ranking by first occurrence of each path;
- `hard_mrr` and `hard_ndcg@10` for audited hard pools; and
- `candidate_coverage`: fraction of patch edit targets, counting
  modified/deleted preimage lines and addition-only insertion anchors, that map
  to at least one corpus chunk.

Line-budget metrics operate on the fixed ranked chunks and count newly covered
patch targets. Modified/deleted preimage lines count individually; each
addition-only insertion anchor counts as one target assigned to its exact
containing chunk. These metrics borrow the useful budget concept from
SWE-Explore but do not claim interactive exploration.

Tie breaking must be deterministic by:

1. descending similarity;
2. repository id;
3. path;
4. line start; and
5. chunk id.

### 8.2 Required slices

- source benchmark and repository;
- programming language;
- issue type: bug, feature, refactor, documentation, question, other;
- issue length and presence of stack trace/error message;
- repository eligible bytes, files, and chunk count;
- one versus multiple changed files;
- one versus multiple directly patch-aligned chunks;
- patch change type: insertion-only, deletion, replacement, rename/copy;
- edit dispersion across files/directories;
- implementation-only, test-only, and implementation-plus-test patches;
- AST definition versus fallback chunk;
- short, medium, and long chunk;
- query contains exact symbol/path versus no exact identifier;
- answer-leak review result;
- same-file-neighbor density;
- near-duplicate blob density;
- base-commit date and public-benchmark age;
- alternate-solution risk: low, medium, high, unknown;
- qrel mapping: exact, expanded, ambiguous, dropped; and
- hard-negative family.

No single aggregate score should hide repository, language, or mapping-quality
breakdowns.

## 9. Provenance, License, Leakage, and Label Audit

### 9.1 Issue and code provenance

For every query retain:

- source benchmark and exact dataset revision, if applicable;
- original issue URL and immutable API response hash;
- PR URL and patch hash;
- repository URL;
- exact base commit and tree SHA;
- path plus blob SHA for every corpus file;
- normalization and chunker version; and
- the qrel generator version.

If the source benchmark base commit differs from the upstream PR base SHA,
record both and explain the transformation. Do not silently substitute the
current default branch.

### 9.2 Dataset license versus upstream file licenses

A dataset card license covers the benchmark packaging only to the extent its
authors have those rights. It does not automatically relicense:

- source code;
- issue and PR text;
- patches;
- comments or logs embedded in issues;
- trajectory content; or
- generated LLM annotations.

Each retained repository needs:

- SPDX license at the base commit;
- license/notice file hashes;
- path-level exceptions or vendored subtrees;
- attribution requirements;
- whether redistribution of source text is allowed; and
- whether the HF product stores code text or only metadata/hashes.

Prefer metadata-only public artifacts that reconstruct code from upstream at
the user's request. Do not upload third-party source text until a per-source
redistribution review passes.

### 9.3 Deduplication and query-visible snapshots

Deduplicate storage without changing evaluation visibility:

- reuse embeddings for identical `blob_sha + chunker_version + chunk span`;
- retain separate logical chunk ids for each repository/base-commit/path;
- filter candidates per query to exactly the declared base commit;
- never expose post-change blobs or future versions; and
- report near-duplicate same-path chunks across splits as contamination risk.

Content-hash deduplication must not create a global corpus in which a query can
retrieve an invisible future snapshot.

### 9.4 Train/eval contamination

Required policies:

- no training rows from the same issue/PR as evaluation;
- no patch-derived training row whose repository, file, and time window overlap
  an evaluation query unless the benchmark explicitly labels this risk;
- prefer repository-disjoint or time-forward splits for any future training
  release;
- record whether model cards disclose SWE-bench, GitHub issue, or code-repair
  training;
- keep public benchmark age/date slices; and
- never claim contamination-free performance for models with undisclosed
  training corpora.

### 9.5 Patch-derived label bias

Patch qrels establish what the historical patch edited, not every valid edit
or all useful context. Required mitigations:

- label the task `historical_edit_localization`, not necessary-code retrieval;
- report multiple-patch or alternate-solution evidence where available;
- review large mechanical patches and generated files;
- separate directly patch-aligned chunks from supporting-context diagnostics;
- treat unlabeled chunks as unknown rather than proven irrelevant; and
- include a `patch_file_non_edit_chunk` diagnostic rather than forcing those
  chunks into the negative class.

### 9.6 Cross-version leakage

The corpus must be built only from the base commit. Reject an instance when:

- the patch does not apply cleanly to the declared base;
- rename/copy detection cannot map preimage paths exactly;
- generated or vendored files dominate the patch;
- submodule revisions are unavailable;
- line endings or normalization break patch mapping; or
- the source benchmark uses a reconstructed state without a reproducible tree.

### 9.7 Privacy and generated-label risks

Before any issue or code text enters a tracked or HF artifact, scan for:

- API keys, bearer tokens, credentials, private URLs, and auth headers;
- email addresses, phone numbers, and user-identifying log data;
- crash dumps, proprietary payloads, or pasted customer data;
- malicious prompt text in issues, comments, code, or docs; and
- LLM-generated labels whose prompt/model/version and human review are absent.

The v0 direct-edit task has no LLM-authored gold labels. LLM query filtering,
rewriting, supporting-context judging, and line refinement must be separate
metadata fields and evidence tiers if introduced later.

## 10. Scale and Machine-Fit Estimates

### 10.1 Assumptions

These are planning estimates, not measured benchmark runs.

- Code is mostly ASCII/UTF-8, so published character counts approximate
  decoded bytes before JSON/Parquet overhead.
- Token range: 3 to 5 characters per embedding token.
- Runtime envelope: 10K to 40K input tokens/second aggregate for a bounded
  small/medium local encoder on the available four 12GB GPUs. Large generative
  embedders, remote APIs, rerankers, and preprocessing are excluded.
- Embedding storage is raw float32 without index overhead.
- Runtime estimates do not estimate provider spend.
- Any later paid-provider plan expected to exceed about USD 30, or blocked by
  quota, billing, account, or credential restrictions, is a human escalation
  gate. This minispec authorizes no provider call.

### 10.2 MTEB file localization

| Quantity | Estimate |
|---|---:|
| Corpus documents | 58,058 |
| Decoded corpus text | 1.308B chars, about 1.22 GiB |
| Input tokens | about 262M to 436M |
| Compressed dataset disk | unknown; planning range 0.4 to 1.0 GiB |
| Python object/RAM envelope | about 2 to 5 GiB before embeddings/index |
| 768D / 1024D / 2048D float32 embeddings | 0.166 / 0.221 / 0.443 GiB |
| Embedding-only runtime | about 1.8 to 12.1 hours |

This is machine-fit for a scheduled local run, but not for an unreviewed smoke.

### 10.3 CORE-Bench Level 2

| Quantity | Estimate |
|---|---:|
| Corpus chunks | 9,377,120 |
| Decoded corpus text | about 9.42B chars, 8.78 GiB |
| Input tokens | about 1.88B to 3.14B |
| Compressed data disk | unknown; planning range 2 to 6 GiB plus snapshots/metadata |
| RAM envelope | about 15 to 35 GiB before embeddings/index |
| 768D / 1024D / 2048D float32 embeddings | 26.83 / 35.77 / 71.54 GiB |
| Embedding-only runtime | about 13 to 87 hours |

The full Level-2 release is not a safe first run on this machine. Repository-
local streaming and blob-level embedding reuse would be mandatory.

### 10.4 CORE-Bench Level 3

| Quantity | Estimate |
|---|---:|
| Corpus chunks | 2,609,581 |
| Decoded corpus text | about 3.01B chars, 2.81 GiB |
| Input tokens | about 603M to 1.00B |
| Compressed data disk | unknown; planning range 0.8 to 2.5 GiB plus snapshots/metadata |
| RAM envelope | about 5 to 12 GiB before embeddings/index |
| 768D / 1024D / 2048D float32 embeddings | 7.47 / 9.96 / 19.91 GiB |
| Embedding-only runtime | about 4.2 to 28 hours |

It is smaller than Level 2 but its qrels have the higher label-provenance risk.

### 10.5 SWE-Explore

The release does not ship one canonical static chunk corpus. Using the paper's
averages:

- 848 snapshots x 179.6K non-test lines = about 152.3M source lines;
- at 40 to 100 characters/line, about 5.7 to 14.2 GiB decoded source text;
- about 1.22B to 5.08B tokens;
- a 100-line chunk with 20-line overlap would produce roughly 1.9M chunks;
- 768D / 1024D / 2048D float32 embeddings for 1.9M chunks would be roughly
  5.4 / 7.3 / 14.5 GiB; and
- the embedding-only runtime envelope is roughly 8.5 to 141 hours.

Repository archives, repeated base commits, build files, and non-source text
could materially change this. The HF benchmark-record size is unverified; the
repository snapshots are fetched separately.

### 10.6 ContextBench

**VERIFIED lower bounds:**

- tracked `full.parquet`: 26.1 MB;
- 522,115 gold lines, roughly 20 to 50 MB at 40 to 100 chars/line; and
- 1,136 task-specific repository snapshots referenced by metadata.

The official sources do not publish aggregate query-visible repository bytes,
tokens, or a canonical static chunk count. A full static adaptation is
therefore **BLOCKED from decision-grade estimation**. A deliberately broad
planning envelope would be 5 to 100 GiB of decoded snapshots and 1M to 10M
chunks, but it is too uncertain to schedule. A metadata-only pass must sum
unique base-commit trees and blobs before any run.

### 10.7 Proposed one-repository smoke

For `psf/requests` at
`3c88e520da24ae6f736929a750876e7654accc3d`:

| Quantity | Status, estimate, or cap |
|---|---:|
| Tracked blobs | 121 verified |
| Tracked blob bytes | 1,435,436 verified |
| Suffix-candidate blobs | 99 metadata-derived; not content-verified |
| Suffix-candidate blob bytes | 865,106 metadata-derived; not decoded text |
| Final eligible text files/bytes | unknown pending Stage A recomputation and Stage B inspection |
| Expected input tokens | planning-only 173K to 288K if all suffix candidates survive |
| Hard chunk cap | 1,000 |
| 1024D float32 embedding cap | about 3.9 MiB |
| Corpus plus index RAM cap | 256 MiB |
| End-to-end wall-clock cap | 30 minutes |

The suffix-candidate totals fit the planning envelope, but machine fit is not
confirmed until the archive, extracted tree, Stage A result, and Stage B result
pass the future byte caps.

## 11. One-Repository Tiny-Smoke Design

This dispatch performs metadata planning only. It does not authorize the
smoke run.

### 11.1 Fixed subject

- repository: `psf/requests`;
- issue query: https://github.com/psf/requests/issues/1920;
- change PR: https://github.com/psf/requests/pull/1921;
- base commit: `3c88e520da24ae6f736929a750876e7654accc3d`;
- reference changed files: `requests/sessions.py` and `test_requests.py`;
- base-commit repository license: Apache-2.0; and
- publication: forbidden.

The issue describes a session header set to `None` being sent as the literal
string `None`. The issue does not name the two changed files, making it suitable
for a localization contract smoke.

### 11.2 Run gates

Items 1-6 and 9-10 must pass before any archive download. Item 8 and the Stage
A/Stage B checks in Section 6.2.1 must pass after materialization and before
chunking. Item 7 must pass before qrels are accepted. No count obtained before
content inspection may be relabeled as an eligible-text count.

1. normal `github.com` and `api.github.com` DNS/HTTPS work without alternate
   IPs;
2. issue #1920 and PR #1921 remain publicly accessible;
3. PR base SHA is exactly the pinned commit;
4. the pinned tree metadata remains untruncated;
5. the Apache-2.0 license file exists at that commit and its hash is recorded;
6. the issue title/body pass answer-leak, privacy, secret, and prompt-injection
   review;
7. the patch maps to the pinned preimage without fuzz or post-change text;
8. no submodule, LFS pointer, or missing blob prevents a complete tree;
9. a dedicated temporary directory and cleanup trap are configured; and
10. the future run manifest explicitly sets `publish: false` and
    `evidence_tier: smoke`.

### 11.3 Hard resource caps

- one repository;
- one issue/query;
- archive download: at most 10,000,000 bytes;
- extracted regular-file bytes: at most 25,000,000 bytes;
- each Stage A candidate raw blob: at most 2,000,000 bytes;
- eligible normalized UTF-8 corpus text: at most 5,000,000 bytes;
- tracked files: at most 500;
- chunks: at most 1,000;
- no model or dataset download;
- no provider API call;
- no more than 256 MiB corpus/index RAM target;
- no more than 30 minutes end-to-end; and
- no retained archive, checkout, embedding cache, or generated result after
  evidence capture.

Stop before extraction or embedding if any cap is exceeded.

### 11.4 Corpus and qrels

The corpus must contain every file accepted by both stages of Section 6.2.1 at
the base commit, chunked by the deterministic v0 policy. This includes
unchanged source, tests, documentation, configuration, setup, and metadata
files only when their paths match the allowlist and their contents pass Stage
B. Extensionless files and explicitly excluded path/name families are not
eligible.

The qrels are derived only after corpus construction by mapping modified or
deleted preimage lines and addition-only insertion anchors from the PR patch to
fixed chunks. The two changed files are not used to prefilter the corpus.

This is not shortlist reranking because:

- no BM25/dense/agent/file-name prefilter chooses candidate files;
- no patch path is passed to the retriever;
- all eligible files and all their chunks remain candidates; and
- ranking is performed against the full one-repository chunk set.

This is not designed as a changed-files-only toy. At the metadata-only level,
the two changed `.py` paths are among the 99 suffix candidates, leaving 97
suffix candidates outside those paths. That `97` is not a verified eligible
unchanged-file count: the final number may be lower after Stage A path/mode
checks and Stage B content inspection.

### 11.5 Execution modes

Contract verification must use a deterministic local score matrix or test
double first. A later separately selected model smoke may use an already
installed local encoder only if it requires no model download and records
query/document routing. Model quality thresholds are not part of contract
acceptance.

No agent, LLM annotation, patch generation, test execution, or repository
exploration is needed.

### 11.6 Cleanup

Use a dedicated path such as:

```text
/tmp/meb-code-edit-smoke-required-run-id/
```

On PASS, FAILED, or BLOCKED:

- close file handles and worker processes;
- remove the archive, extracted snapshot, chunk files, embeddings, and index;
- verify the dedicated path is absent;
- retain only a small Git-tracked evidence note in a later dispatch if the
  supervising layer accepts it; and
- do not retain third-party code text in the repository or HF artifacts.

### 11.7 Outcome criteria

PASS:

- every precondition passes;
- the full eligible base-commit corpus is represented;
- Stage A and Stage B results are recorded for every suffix candidate, and the
  final eligible file/byte counts are recomputed rather than copied from the
  `99`/`865106` metadata filter;
- all chunk ids, path/line spans, blob hashes, and text hashes are stable;
- both changed files map to one or more exact grade-2 chunks;
- `candidate_coverage` is 1.0 for modified/deleted preimage lines and
  addition-only insertion anchors;
- repeated generation produces identical corpus/qrels hashes;
- deterministic metric tests reproduce known exact values;
- no hard negative overlaps a qrel;
- resource caps and cleanup pass; and
- the run remains `publish: false`, `evidence_tier: smoke`.

FAILED:

- deterministic corpus, qrels, ranking, metric, serialization, or cleanup
  behavior is incorrect;
- chunk identity changes across repeated runs;
- patch preimage mapping is wrong despite valid source metadata; or
- the implementation silently prefilters to changed or top-ranked files.

BLOCKED:

- normal network access fails;
- license, issue, patch, base commit, or tree metadata cannot be verified;
- a cap is exceeded before safe completion;
- patch mapping is ambiguous due rename, fuzz, or normalization;
- privacy/secret review fails; or
- the only available encoder path requires a download, dependency change, or
  provider API.

ABANDON:

- a genuine full-repository corpus cannot be preserved;
- the only workable qrels depend on post-change code or an LLM judge;
- license terms forbid the required local processing;
- the task can only be made easy by passing patch paths or changed files to the
  retriever; or
- the smoke is proposed as public model-quality evidence.

## 12. Publication and Hugging Face Product Policy

### 12.1 Evidence tiers

- `fixture`: self-created deterministic schema/metric tests only;
- `smoke`: private, no-publish one-repository contract or compatibility run;
- `benchmark`: allowed only after multi-repository provenance, license,
  leakage, alternate-solution, and reproducibility gates pass;
- `unknown`: must never be promoted to a public leaderboard automatically; and
- `legacy`: not applicable to new code-localization runs.

Trajectory-derived, LLM-judged, and human-context labels require an additional
label-provenance field even if the run evidence tier is `benchmark`.

### 12.2 Dataset repository path

The preferred public HF Dataset artifact is metadata-first:

```text
README.md
dataset_manifest.json
repositories.jsonl
queries.jsonl
chunks_metadata.jsonl
qrels.jsonl
licenses.jsonl
provenance.jsonl
leaderboards/latest.csv
results/*.jsonl
```

Do not upload third-party code text by default. `chunks_metadata.jsonl` may
contain ids, paths, spans, blob hashes, text hashes, language, and chunker
version while requiring local reconstruction from pinned upstream commits.

If later legal review authorizes text redistribution for a subset, publish it
as a separately versioned config with explicit per-row license fields. Do not
apply one blanket dataset license to mixed-license code.

### 12.3 Space path

The Space should show:

- task version and source-revision coverage;
- embedding-only versus reranker/system track;
- evidence tier and publication state;
- primary nDCG@10 plus edit-target/file/hard-negative diagnostics;
- repository/language/mapping-quality slices;
- latest versus historical run filtering; and
- warnings for public-benchmark age, unknown training contamination, and
  incomplete alternate-solution review.

Do not publish one global score across code localization, agent memory, skill
routing, and late-chunking tasks.

## 13. Minimal Future Repository Path

No implementation is authorized now. If a later dispatch accepts the positive
direction, the smallest patch should be:

1. `src/mm_embed/data/code_edit_chunk_localization.py`
   - deterministic metadata loader, text materializer, chunker, patch mapper,
     qrel generator, provenance validator, and cleanup-safe smoke helper;
2. `src/mm_embed/tasks/code_edit_chunk_localization.py`
   - flat query/document embedding, full-corpus ranking, metrics, slices, and
     explicit rejection of non-audited data;
3. `tests/test_code_edit_chunk_localization.py`
   - invented zero-network fixture plus exact score-matrix tests;
4. one `publish: false` task entry in `benchmark/tasks/core.yaml` only after
   the fixture contract passes; and
5. one local `publish: false`, `evidence_tier: smoke` manifest only in a
   separate dispatch after the one-repository metadata gate is accepted.

Do not combine the real-data materializer, public task registration, model
run, and HF publication in one change.

## 14. Explicit Next Action

Create a separate `tasks/code-edit-chunk-fixture-contract` item that implements
only an invented, zero-network fixture and exact evaluator tests for the schema,
chunk identity, qrel mapping, full versus hard metrics, line-budget coverage,
and no-publish filtering.

Only after that contract is accepted should another dispatch consider the
`psf/requests` metadata-gated smoke. That smoke must still stop before download
if the normal-network, license, issue/PR, base-commit, or byte gates fail.

## 15. What Would Change the Decision

Change from **DIFFERENTIATE** to **ADOPT file localization** only if:

- MTEB's current dataset provenance and query-visible snapshot semantics are
  fully audited;
- maintaining a separate chunk task would add no reliable information beyond
  file ranks; and
- the product goal changes from modern retrieval gaps to MTEB compatibility.

Change from **DIFFERENTIATE** to **DEFER the whole family** if:

- no one-repository smoke can pass source/license/privacy gates;
- exact patch-to-preimage mapping is not reproducible;
- repository snapshot materialization cannot stay within bounded resources;
- the repository cannot adopt a license suitable for self-authored fixtures;
  or
- all useful candidate datasets require redistributing code without adequate
  rights.

Change the recommended v0 surface from edit chunks to broader context only if:

- an official release provides pinned repository snapshots, row-level source
  and license provenance, and auditable non-LLM or independently validated
  context qrels;
- alternate valid contexts and false negatives are measured at meaningful
  scale;
- the static candidate corpus and metric implementation are pinned; and
- embedding-only results can be separated from trajectory/process results.

Change to **ABANDON** if the only defensible evaluation requires an interactive
agent, patch generator, hidden LLM judge, or changed-file shortlist while being
marketed as static embedding retrieval.

## 16. Primary Sources and Pinned Revisions

### MTEB `SWEbenchCodeRetrieval`

- Current MTEB commit:
  https://github.com/embeddings-benchmark/mteb/commit/80f6e3b89539a433589ac4a685d3a9f86e4f0f10
- Current MTEB release:
  https://github.com/embeddings-benchmark/mteb/releases/tag/2.18.5
- Current task source:
  https://github.com/embeddings-benchmark/mteb/blob/80f6e3b89539a433589ac4a685d3a9f86e4f0f10/mteb/tasks/retrieval/code/swebench_code_retrieval.py
- Current descriptive statistics:
  https://github.com/embeddings-benchmark/mteb/blob/80f6e3b89539a433589ac4a685d3a9f86e4f0f10/mteb/descriptive_stats/Retrieval/SWEbenchCodeRetrieval.json
- Current retrieval loader contract:
  https://github.com/embeddings-benchmark/mteb/blob/80f6e3b89539a433589ac4a685d3a9f86e4f0f10/mteb/abstasks/retrieval_dataset_loaders.py
- Addition PR:
  https://github.com/embeddings-benchmark/mteb/pull/4365
- Addition commit:
  https://github.com/embeddings-benchmark/mteb/commit/dcc5965c3ebf663f58c8f51fdb957a9d00bd3326
- Pinned dataset tree:
  https://huggingface.co/datasets/embedding-benchmark/SWEbenchCodeRetrieval/tree/440b0e732b8d02c16df2c95352ab6770abe997da
- Pinned dataset commit:
  https://huggingface.co/datasets/embedding-benchmark/SWEbenchCodeRetrieval/commit/440b0e732b8d02c16df2c95352ab6770abe997da
- Merged official reference results:
  https://github.com/embeddings-benchmark/results/commit/dc3f7e890385c960973d53de6e831bebe7c979f9
- Reference-results PR:
  https://github.com/embeddings-benchmark/results/pull/557
- BM25S result:
  https://github.com/embeddings-benchmark/results/blob/8aeff94babd584536a3ffc72c4d0d36b1cc0a8c8/results/mteb__baseline-bm25s/0_1_10/SWEbenchCodeRetrieval.json

### CORE-Bench

- Paper v2:
  https://arxiv.org/abs/2606.11864v2
- HTML paper v2:
  https://arxiv.org/html/2606.11864v2
- Official evaluation commit:
  https://github.com/zhangfw123/CORE-Bench-Eval/commit/5d00cdcaade77796fa6e5ac4852404002029c00a
- Pinned README:
  https://github.com/zhangfw123/CORE-Bench-Eval/blob/5d00cdcaade77796fa6e5ac4852404002029c00a/README.md
- Pinned Level-2 loader:
  https://github.com/zhangfw123/CORE-Bench-Eval/blob/5d00cdcaade77796fa6e5ac4852404002029c00a/src/tasks/core_bench_level2_tasks.py
- Pinned Level-3 loader:
  https://github.com/zhangfw123/CORE-Bench-Eval/blob/5d00cdcaade77796fa6e5ac4852404002029c00a/src/tasks/core_bench_level3_tasks.py
- Official dataset:
  https://huggingface.co/datasets/zhangfw123/CORE-Bench

### SWE-Explore

- Paper v1:
  https://arxiv.org/abs/2606.07297v1
- HTML paper v1:
  https://arxiv.org/html/2606.07297v1
- Official code commit:
  https://github.com/Qiushao-E/SWE-Explore-Bench/commit/3c12dc5a551937038afcbdb6eb6bbf19f3ddd8c1
- Pinned README and record schema:
  https://github.com/Qiushao-E/SWE-Explore-Bench/blob/3c12dc5a551937038afcbdb6eb6bbf19f3ddd8c1/README.md
- Pinned evaluator:
  https://github.com/Qiushao-E/SWE-Explore-Bench/blob/3c12dc5a551937038afcbdb6eb6bbf19f3ddd8c1/eval.py
- Pinned benchmark builder:
  https://github.com/Qiushao-E/SWE-Explore-Bench/blob/3c12dc5a551937038afcbdb6eb6bbf19f3ddd8c1/bench_build.py
- Pinned code license:
  https://github.com/Qiushao-E/SWE-Explore-Bench/blob/3c12dc5a551937038afcbdb6eb6bbf19f3ddd8c1/LICENSE
- Official dataset:
  https://huggingface.co/datasets/SWE-Explore-Bench/SWE-Explore-Bench

### ContextBench

- Paper v3:
  https://arxiv.org/abs/2602.05892v3
- HTML paper v3:
  https://arxiv.org/html/2602.05892v3
- Official project page:
  https://contextbench.github.io/
- Official repository commit:
  https://github.com/EuniAI/ContextBench/commit/1436c28a8eb95496da4ea69ad458b9f8a8eb7d61
- Pinned README:
  https://github.com/EuniAI/ContextBench/blob/1436c28a8eb95496da4ea69ad458b9f8a8eb7d61/README.md
- Pinned gold parser and dataset schema:
  https://github.com/EuniAI/ContextBench/blob/1436c28a8eb95496da4ea69ad458b9f8a8eb7d61/contextbench/parsers/gold.py
- Pinned metrics implementation:
  https://github.com/EuniAI/ContextBench/blob/1436c28a8eb95496da4ea69ad458b9f8a8eb7d61/contextbench/metrics/compute.py
- Pinned license:
  https://github.com/EuniAI/ContextBench/blob/1436c28a8eb95496da4ea69ad458b9f8a8eb7d61/LICENSE
- Official dataset:
  https://huggingface.co/datasets/Contextbench/ContextBench

### Proposed tiny smoke metadata

- Issue query:
  https://github.com/psf/requests/issues/1920
- Reference PR:
  https://github.com/psf/requests/pull/1921
- Exact pre-change repository state:
  https://github.com/psf/requests/tree/3c88e520da24ae6f736929a750876e7654accc3d
- Exact Git tree metadata:
  https://api.github.com/repos/psf/requests/git/trees/cf441cdcaa76806a078893bffa79922351a92b0a?recursive=1
- Issue metadata API:
  https://api.github.com/repos/psf/requests/issues/1920
- Raw reference patch:
  https://github.com/psf/requests/pull/1921.patch
- Base-commit license:
  https://github.com/psf/requests/blob/3c88e520da24ae6f736929a750876e7654accc3d/LICENSE

## 17. Final Decision

Proceed with **DIFFERENTIATE** at the design level and choose only
**edit-chunk/region localization** for v0.

Use MTEB file localization as a baseline, CORE Level 2 as the closest design
precedent, SWE-Explore as line-budget/region-metric evidence, and ContextBench
as later human-context/process evidence. Do not ingest any of their data in the
same step.

The immediate next artifact should be an invented zero-network fixture
contract. The first real-source action, if separately selected, is the pinned
`psf/requests` one-issue smoke with a genuine full-repository candidate set,
strict byte/cleanup gates, and `publish: false`. Public data and scores remain
blocked on repository licensing, row provenance, snapshot visibility,
deduplication, contamination, alternate-solution, privacy, and HF data-card
audits.
