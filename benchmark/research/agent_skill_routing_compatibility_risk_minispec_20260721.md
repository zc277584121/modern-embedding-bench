# Agent Skill Routing Compatibility and Risk Minispec - 2026-07-21

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_1-1784643557-2_execute.md`

Selected item: `tasks/agent-skill-routing-compatibility-risk-scout`

Status: **PASS for the repository design decision and an invented no-publish
smoke; BLOCKED for ingestion or redistribution of R3-Skill or
SkillResolve-Bench rows.**

Decision: **keep `agent_procedural_tool_memory` unchanged and split the two new
axes into separate task ids**:

- `agent_skill_compatible_set_retrieval`
- `agent_skill_same_capability_risk`

The two tasks should share one local fixture/data module, but they must not
share one leaderboard primary metric. Query-conditioned compatible-set
retrieval and same-capability risky-sibling exposure have different labels,
failure semantics, metric directions, and system-layer dependencies. Folding
them into the existing one-positive task would silently change its contract;
folding them into one new task would hide a recall-versus-exposure tradeoff
behind one cross-slice scalar. Any registry primary used inside the risk task
must remain a task-local gate with mandatory decomposed helpful-retrieval and
HSR outputs; it is not sufficient evidence on its own.

Labels used below:

- **CONFIRMED**: observed in this repository or a primary source during the
  2026-07-21 audit.
- **SOURCE CLAIM**: stated by a paper or project, but not independently
  reproducible from a discoverable released artifact.
- **PROPOSAL**: repository design recommended by this minispec.
- **GATE**: condition that must pass before real data, public scores, or broader
  system claims are allowed.

## 1. Current Repository Baseline

### `agent_procedural_tool_memory` is ordinary relevance retrieval

**CONFIRMED.** The current task is an invented, sanitized, text-only fixture:

- 12 queries;
- 36 tool documents;
- one positive document per query;
- three curated hard negatives per query;
- query and document routing through `retrieval_query` and
  `retrieval_document`;
- full-corpus `recall@1`, `recall@5`, `mrr`, and `ndcg@10`;
- hard-pool versions of the same metrics; and
- per-slice hard MRR for `single_tool`, `parameter_constraints`, `near_name`,
  and `category_collision`.

Its `ToolQuery` schema has `positive_doc_id`, not a positive set, compatible
set, rejected combination, capability family, or helpful/risky sibling pair.
The shared metric helpers also assume one correct candidate index per query.
The implementation therefore measures:

> Given a query, rank one independently relevant procedural tool card above
> ordinary distractors.

That is a useful baseline and should remain stable. A same-family hard negative
inside this fixture is still just a negative document; the task does not state
that it is query-specific execution risk, nor that a set of retrieved tools is
jointly compatible.

## 2. Three Evaluation Surfaces

| Surface | Retrieval object | Ground truth | Main failure | Correct repository treatment |
|---|---|---|---|---|
| Ordinary procedural relevance | One query against independent tool/skill documents | Binary qrels, currently one positive | Correct document is ranked too low | Keep `agent_procedural_tool_memory` |
| Query-conditioned compatible-set retrieval | One query against a skill library, with two or more jointly wanted skills and query-specific rejected combinations | Multi-positive qrels plus compatible and rejected sets | Individually plausible skills do not form the wanted top set | New `agent_skill_compatible_set_retrieval` task |
| Same-capability risk exposure | One query against a library containing a helpful skill and a query-specific risky sibling in the same family | Helpful id, risky id, capability family, evidence-backed risk relation | Router finds the family but exposes the wrong representative | New `agent_skill_same_capability_risk` task |

The axes are complementary, not interchangeable:

- R3-Skill asks whether the retrieved members form the query-wanted skill set.
- SkillResolve-Bench asks whether a close sibling from the correct capability
  family is the wrong representative for this query.
- Neither is equivalent to adding harder ordinary negatives to the current
  one-positive fixture.

## 3. Primary-Source Audit

### 3.1 R3-Skill

#### Paper state

**CONFIRMED.** The current arXiv paper is `2606.03565v4`, revised
2026-07-08, under CC BY 4.0 for the paper text. It reports:

- 10,246 skills after deduplicating 95,212 raw entries;
- 8,196 train-pool and 2,050 test-pool skills;
- 41,592 accepted queries;
- 32,828 LLM-rejected annotations;
- 30,287 training queries and 5,696 test queries after dropping 5,609 rows
  whose target skills straddled the disjoint pools;
- four directions: `en2en`, `en2zh`, `zh2en`, and `zh2zh`;
- accepted sets of size one, two, or three;
- near-neighbor construction with same-sub-cluster sampling for two-skill
  candidates and same-super-domain sampling for three-skill candidates; and
- eight reported rejection-reason classes plus `unknown`.

The paper defines `Set-Compat` only on queries with at least two positives. A
query scores one when its complete gold set appears in the first `|GT|`
positions. The paper reports `Hit@1`, `NDCG@5/10/15`, `R@5/10/15`,
`Comp@5/10/15`, and `Set-Compat`.

The paper's main mechanism claim is explicitly system-stage dependent:

- the R3 embedding bi-encoder ranks each query-skill pair independently;
- adding SKIP partners to its tested embedding objective made the reported
  `Set-Compat` worse, from 0.2812 to 0.2158;
- the cross-encoder reranker uses SKIP partners as graded listwise supervision;
  and
- the reported full two-stage result reaches `Set-Compat = 0.3188`.

This is evidence that compatible-set outcomes are worth measuring for an
embedding model, but not evidence that a bi-encoder alone performs
query-conditioned set reasoning.

#### Official repository and released artifacts

**CONFIRMED.** `Tencent/R3-Skill` was audited at commit
`57d9285969cc293832c71ecbe42a75ac416bef93`, committed 2026-07-08. The
repository has no tags or GitHub releases. Its tree contains:

- `data/test.jsonl`: 4,358,833 bytes, documented as 5,696 test queries;
- `data/skills_test.jsonl`: 17,786,819 bytes, documented as 2,050 test-pool
  skills;
- `reproduce.py`, `infer.py`, a toy skill file, and documentation; and
- no released train-query file, train skill pool, SKIP annotation file,
  rejection taxonomy file, or source ledger.

The README says released skill ids are opaque (`skill-00000`, ...), and the
reproduction code consumes only:

```json
{"query": "...", "skill_ids": ["skill-00000", "skill-00001"]}
{"id": "skill-00000", "text": "name | description | body"}
```

**Conclusion:** the official repository releases a usable test benchmark and
evaluation script, but not the full 41,592-query/32,828-SKIP training and
compatibility-supervision corpus described by the paper. The current public
files are insufficient for auditing or reusing the rejected-combination signal.

#### Metric contract mismatch

**CONFIRMED.** The v4 paper appendix says `R@K` is an all-ground-truth
completeness indicator:

```text
1[GT is a subset of top-K]
```

At the pinned repository commit, `reproduce.py` instead computes both
`recall@K` and `comp@K` as the fractional number of retrieved positives divided
by the number of positives. This makes the two columns identical whenever
`K >= |GT|`. A real R3-derived implementation must not guess which convention
is authoritative. The proposed fixture below uses explicit, non-overloaded
names: fractional `recall@K`, binary `complete_set@K`, and `set_compat`.

#### Redistribution and provenance

The R3 README lists four source families:

1. `ThakiCloud/SKILLRET`;
2. `pipizhao/SkillRouter-Eval-Core`;
3. `zhoukc42/AiSkill`; and
4. `jnMetaCode/agency-agents-zh`.

The R3 `LICENSE` asserts Apache-2.0 for Tencent's released code, training data,
parameters, and weights, while also saying third-party components retain their
own terms and that referenced open skills require independent license review.
The attribution section names SkillRet and `agency-agents-zh`, but does not
name `SkillRouter-Eval-Core` or `AiSkill`, even though the README lists both as
data sources.

Current source observations:

| Source | Pinned/current state audited | License/provenance observation |
|---|---|---|
| SkillRet | HF dataset v1.0 at `7cae7cfbad2b0e1ebc9170892f568993aae543b0`; GitHub code at `0b9564a5021751661f93333adbf996fc51b577d2` | Rows expose `license`, `repo`, `source_url`, and `raw_url`, but no immutable upstream commit field. The card says benchmark metadata/queries/taxonomy are Apache-2.0 and source skills retain MIT/Apache-2.0 terms. |
| SkillRouter Eval Core | HF dataset at `20a03920e7f08d76b67af367350d25ef4468198e` | The card attributes queries/gold skills to `benchflow-ai/skillsbench` and the pool to `majiayu000/claude-skill-registry`, but exposes no dataset license in the audited card. Repository-level MIT on SkillRouter code is not a row-level data license. |
| AiSkill | `main` resolved to `266f2f57f89e0f719f48827ea4e51bb1243132a8` | GitCode repository metadata has no declared license; a root `LICENSE` and `LICENSE.md` returned 404. The README only shows `license: Apache-2.0` as an optional example frontmatter field. |
| agency-agents-zh | GitHub `main` at `cf086f76a225f7b287db0953940f76df6df8d1e3` | Repository declares MIT, but R3 does not pin which upstream revision supplied each released opaque row. |

**GATE.** A paper license and the top-level R3 Apache notice are not sufficient
proof that every aggregated skill body can be republished by this repository.
The opaque released ids, missing row-to-source manifest, omitted source
attributions, mutable source URLs, and unresolved AiSkill/SkillRouter data
licenses block redistribution. R3 test rows may be used only after a separate
legal/provenance decision; this minispec does not authorize download, import,
or publication.

### 3.2 SkillResolve-Bench 1.0

#### Paper state and schema claims

**CONFIRMED.** The only current paper version is arXiv `2606.10388v1`,
published 2026-06-09 under the arXiv perpetual non-exclusive paper license. It
defines each instance as:

```text
(query, candidate_pool, helpful_skill, risky_sibling, family_relation)
```

The reported benchmark contains:

- 661 helpful/risky sibling pairs;
- 630 SRA-Bench-derived pairs and 31 SkillsBench-derived pairs;
- admission evidence split into 495 answer-backed, 135 executable-test-backed,
  and 31 oracle/verifier-backed pairs;
- risk types: 390 resource-pointer, 232 precondition, 34 procedure, 3 API, and
  2 example risks;
- 6,660 public SkillRet candidates as unlabeled library pressure;
- a fixed 7,982-candidate pool;
- 7,321 family groups: 661 size-two sibling groups plus 6,660 SkillRet
  singletons;
- a query-disjoint release split of 446/68/147; and
- grouped five-fold held-out evaluation over all 661 pairs.

The paper says the release records source identifiers, admission class, risk
type, split, text hashes, family relations, JSON schemas, baseline ranks,
per-pair metric rows, and a release checker at
`scripts/run_release_checks.sh`.

#### Artifact availability

**SOURCE CLAIM, NOT CONFIRMED RELEASE.** The arXiv HTML contains no project,
GitHub, Hugging Face, DOI, Zenodo, or download URL for SkillResolve-Bench. Exact
web searches for the benchmark name, claimed checker path, and author/project
combination found only the paper and unrelated repositories. A current GitHub
repository search returned no matching official repository. Exact Hugging Face
discovery did not surface an official dataset or model artifact; direct
`huggingface.co` API checks from this host were additionally unavailable due to
the already-known host DNS failure.

The paper describes a strong release interface, but the audit could not locate
that interface. Therefore:

- code availability: **not verifiably released**;
- dataset availability: **not verifiably released**;
- family relation and release checker: **not verifiably released**; and
- dataset license: **not stated or discoverable**.

The arXiv paper license covers the paper submission, not the claimed benchmark
bundle. No SkillResolve row may be copied or reconstructed from prose examples
for public use.

#### Metric and claim boundary

SkillResolve-Bench reports:

- helpful `Recall@K`;
- helpful `NDCG@K`; and
- harmful sibling rate `HSR@K`, the fraction of queries whose risky sibling is
  present in the final top K.

The paper explicitly limits HSR to **pre-execution exposure**. It is not an
execution-success, maliciousness, permission, or end-to-end safety metric. The
reported `HSR@3 = 0` is produced by capability resolution plus one-per-family
representative selection. Removing representative selection raises HSR@3 to
0.236 under the same scorer. That zero-exposure claim is therefore a
family-aware system result, not an embedding-only property.

### 3.3 SkillRet as the ordinary multi-positive upstream

**CONFIRMED.** SkillRet's current paper is arXiv `2605.05726v1`, published
2026-05-07. The HF dataset card identifies release version `v1.0`; the audited
main revision is `7cae7cfbad2b0e1ebc9170892f568993aae543b0`, committed
2026-05-08. The official GitHub code revision is
`0b9564a5021751661f93333adbf996fc51b577d2`, also 2026-05-08.

SkillRet provides:

- 17,810 skill records, with 10,123 train skills and 6,660 test skills;
- 63,259 train queries and 4,997 test queries;
- 127,190 train qrels and 8,347 test qrels;
- binary multi-positive qrels;
- English queries and skill bodies;
- a six-major/eighteen-subcategory taxonomy; and
- recommended NDCG, Recall, MAP, MRR, and Completeness metrics.

Its skills were crawled from `claude-plugins.dev`, which indexed public GitHub
skills. The paper says it filtered for declared MIT or Apache-2.0 licenses and
deduplicated normalized content. The released row schema preserves a declared
license plus repository and raw/source URLs.

**GATE.** This is better provenance than the opaque R3 release, but still not a
complete redistribution proof. The schema does not expose an immutable source
commit or the upstream license-file hash used to admit each row. A future
import must resolve and pin every retained source row, retain attribution, and
record what happens when a source URL or license has changed since collection.

## 4. Schema and Metric Comparison

| Dimension | Current procedural task | R3-Skill | SkillResolve-Bench | Proposed repository contract |
|---|---|---|---|---|
| Query | One procedural request | Bilingual request, one to three gold skills | Query paired with helpful/risky sibling | Common query record with explicit `slice` and language |
| Corpus | Invented tool cards | Skill `id` + `name | description | body` text | Helpful/risky skills plus public pressure pool | Invented skill cards with explicit family ids |
| Ordinary qrels | One positive | Multi-positive `skill_ids` | One helpful positive | Separate binary qrels file supporting multiple positives |
| Compatible set | None | Gold skill set; SKIP combinations described in paper | Not the target | Query-specific `compatible_sets` and `rejected_sets` |
| Risk sibling | None | Rejected combinations, not one helpful/risky family pair | One query-specific risky sibling | Separate `risk_pairs` relation |
| Family relation | None | Clusters/super-domains for construction | Released/derived capability families | Explicit `family_id`, used only by risk task metadata |
| Split | Built-in fixture | Skill-pool-disjoint train/test | Query-disjoint 446/68/147 plus held-out folds | `fixture_only`, query-disjoint by construction |
| Hard negatives | Three curated docs/query | Near-neighbor sets; mined rank 20-50 negative for embedding training | Confusable top-50 library alternatives, up to five for training | Audited query-specific hard-negative rows |
| Pair metrics | Single-positive rank | Hit/NDCG/Recall | Helpful rank and helpful-vs-risk competition | MRR/NDCG/Recall plus helpful-over-risk win rate |
| Set metrics | None | Complete-set and Set-Compat | None | Fractional Recall, `complete_set@K`, `set_compat`, rejected-set exposure |
| Exposure metric | None | No published HSR | HSR@K | Raw-ranking HSR@K and `safe_helpful@K` |
| Execution metric | None | Explicitly out of paper's offline scope | Verifier evidence supports labels; HSR remains pre-execution | Out of embedding benchmark scope |

## 5. Provider-Neutral Embedding Boundary

### Valid for an embedding-only task

The following can be computed after independently embedding queries and skill
documents, ranking by the repository's normal similarity path, and applying a
deterministic evaluator:

- ordinary multi-positive `recall@K`, `mrr`, and `ndcg@K`;
- `complete_set@K`: all gold skills appear somewhere in top K;
- `set_compat`: all gold skills appear within top `|GT|`;
- rejected-set exposure: a query-specific rejected combination is fully
  present in top K;
- helpful `recall@K`, `mrr`, and `ndcg@K`;
- raw-ranking `hsr@K`;
- helpful-over-risk score win rate and score margin; and
- `safe_helpful@K`: helpful is in top K and risky is not.

These are provider-neutral ranking diagnostics. They do not claim that the
embedding model performs joint set reasoning or knows an execution contract.

### Requires reranking or another system layer

- Learning from R3 SKIP/rejected combinations as query-conditioned graded
  supervision requires a query-skill joint scorer or another set-aware layer.
- Selecting one representative per capability family requires family-aware
  post-processing. Its output must be a separate system track, not silently
  applied to an embedding provider row.
- Constructing reliable public capability families requires an ontology,
  metadata resolver, or clustering protocol with its own audit.
- Claiming reduced execution failure requires executing skills against public
  verifiers or another downstream agent evaluation.
- Claiming safety certification, malicious-skill detection, permission safety,
  or end-to-end task success is out of scope.

## 6. Task-ID Decision

### Do not extend `agent_procedural_tool_memory`

Extending the current task would break stable assumptions in its data classes,
metric utilities, hard-pool construction, tests, and published task identity.
Its one-positive, one-index ground truth is not merely a small version of a
compatible-set or risk-pair task.

### Do not create one task with two slices

A single task would need one `primary_metric` and one metric direction. The
compatibility slice rewards high `set_compat`; the risk slice must jointly
reward helpful retrieval and penalize HSR. Choosing only HSR rewards an empty
retriever, while choosing only recall hides risky exposure. A composite scalar
that weights or averages the two slices would obscure the exact tradeoff this
scout is intended to expose. This is distinct from `safe_helpful@K` below: that
metric is a non-weighted, per-query Boolean conjunction within the risk task,
and it is valid only when its helpful-retrieval and HSR components are always
reported separately.

### Create two task ids with shared data

**PROPOSAL.** Use:

1. `agent_skill_compatible_set_retrieval`
   - required modality: text;
   - primary metric: `set_compat`;
   - guardrails: fractional `recall@K`, `complete_set@K`, `ndcg@K`, and
     rejected-set exposure;
   - no family-aware selector.
2. `agent_skill_same_capability_risk`
   - required modality: text;
   - fixture registry primary: `safe_helpful@3`, defined only as the
     conjunctive gate `helpful in top 3 AND risky absent from top 3`;
   - mandatory separate metrics: helpful `recall@1/3`, helpful `ndcg@3`,
     `hsr@1/3`, and helpful-over-risk win rate;
   - no execution or safety claim.

The risk task's public leaderboard presentation, if ever enabled, must display
helpful retrieval and HSR side by side even if the registry requires one
primary metric. `safe_helpful@3` must never be used as a standalone ranking or
claim: two systems can have the same conjunctive rate with different helpful
recall and risky-sibling exposure. A risk result that omits the decomposed
helpful and HSR metrics is invalid.

## 7. Invented Deterministic No-Publish Smoke

### Fixture size and content

**PROPOSAL.** Author 12 invented skill documents in four capability families,
three siblings per family:

| Family | Skill ids |
|---|---|
| `schema_change` | `schema_contract_diff`, `schema_sample_mapper`, `schema_doc_outline` |
| `log_diagnostics` | `log_redaction_review`, `error_signature_cluster`, `log_volume_rollup` |
| `archive_lookup` | `archive_day_lookup`, `archive_range_list`, `page_freshness_score` |
| `dependency_audit` | `dependency_license_matrix`, `package_age_report`, `dependency_update_window` |

Use four invented queries:

1. Compatibility: compare two schema versions, then validate a sample record.
   Gold set: `schema_contract_diff` + `schema_sample_mapper`. Rejected set:
   `schema_contract_diff` + `schema_doc_outline`.
2. Compatibility: redact credential-like log fields, then cluster the remaining
   error signatures. Gold set: `log_redaction_review` +
   `error_signature_cluster`. Rejected set: `log_volume_rollup` +
   `error_signature_cluster`.
3. Same-family risk: check whether one public page had a snapshot on one exact
   date. Helpful: `archive_day_lookup`. Risky sibling:
   `archive_range_list`. The sibling is useful for other queries but violates
   this query's exact-date contract.
4. Same-family risk: audit package names, versions, licenses, and policy
   compatibility. Helpful: `dependency_license_matrix`. Risky sibling:
   `package_age_report`. The sibling is not intrinsically unsafe; it is the
   wrong same-family representative for this query.

This fixture is deliberately too small for model-quality claims. It exists to
test schemas, deterministic metrics, provider routing, and failure reporting.

### Skill schema

```json
{
  "skill_id": "archive_day_lookup",
  "family_id": "archive_lookup",
  "name": "Archive Day Lookup",
  "description": "Checks for an archived snapshot on one requested calendar date.",
  "body": "Inputs: page label, target date, exact-or-prior mode. Returns one date-scoped result.",
  "source_kind": "local_invented_fixture",
  "source_id": "skill_archive_001",
  "source_revision": "agent-skill-routing-fixture-v0",
  "license_status": "local_invented_sanitized_no_external_sources",
  "public_score_eligible": false
}
```

### Query schema

```json
{
  "query_id": "q_risk_archive_exact_day",
  "text": "Check whether the public page had an archived snapshot on 2031-04-12; do not return a range listing.",
  "split": "fixture_only",
  "slice": "same_capability_risk",
  "language": "en",
  "source_kind": "local_invented_fixture",
  "public_score_eligible": false
}
```

### Ordinary qrels schema

One row per query-skill positive:

```json
{"query_id": "q_compat_schema_validate", "skill_id": "schema_contract_diff", "relevance": 1}
{"query_id": "q_compat_schema_validate", "skill_id": "schema_sample_mapper", "relevance": 1}
```

### Compatible and rejected set schema

```json
{
  "query_id": "q_compat_schema_validate",
  "set_id": "set_schema_gold",
  "skill_ids": ["schema_contract_diff", "schema_sample_mapper"],
  "label": "compatible",
  "reason_code": "ordered_complementary_operations"
}
```

```json
{
  "query_id": "q_compat_schema_validate",
  "set_id": "set_schema_rejected_outline",
  "skill_ids": ["schema_contract_diff", "schema_doc_outline"],
  "label": "rejected",
  "reason_code": "wrong_second_operation"
}
```

The rejected label is query-specific. It must not be interpreted as saying the
individual skill is globally irrelevant or unsafe.

### Risk-pair schema

```json
{
  "query_id": "q_risk_archive_exact_day",
  "family_id": "archive_lookup",
  "helpful_skill_id": "archive_day_lookup",
  "risky_skill_id": "archive_range_list",
  "risk_type": "output_scope_mismatch",
  "admission_basis": "invented_contract_review",
  "evidence_id": "evidence_archive_exact_day_v0"
}
```

### Hard-negative schema

```json
{
  "query_id": "q_risk_archive_exact_day",
  "skill_id": "page_freshness_score",
  "negative_role": "same_family_non_pair",
  "reason": "Uses the same page vocabulary but measures recency of change, not snapshot existence."
}
```

Every hard negative must be reviewed against the full query. It cannot be an
alias of a positive, a valid alternate procedure, or a sibling whose risk label
depends on hidden execution not represented in the fixture contract.

### Metric definitions

Use explicit definitions and deterministic tie breaking by `skill_id`:

- `recall@K`: average fraction of positive qrels retrieved in top K.
- `complete_set@K`: fraction of compatibility queries for which all positive
  skills are in top K.
- `set_compat`: fraction of multi-positive queries for which all positives are
  in the first `|GT|` ranks.
- `ndcg@K`: standard binary multi-positive NDCG.
- `mrr`: reciprocal rank of the first positive; secondary only for the
  compatibility task because it ignores set completeness.
- `rejected_set_exposure@K`: fraction of query/rejected-set rows whose complete
  rejected set appears in top K. This is a repository diagnostic, not an R3
  paper metric.
- `helpful_recall@K`: fraction of risk queries with the helpful skill in top K.
- `hsr@K`: fraction of risk queries with the risky sibling in top K.
- `helpful_over_risky_win_rate`: fraction whose helpful score is strictly
  greater than the risky score; exact ties count as 0.5 only in a separately
  named tie-aware diagnostic.
- `safe_helpful@K`: fraction with helpful in top K and risky absent from top K.
  This is a conjunctive gate, not a weighted composite or a replacement for
  its components. It must always be emitted and interpreted alongside
  `helpful_recall@K` and `hsr@K`.

Tests should use a hand-authored score matrix with known exact outcomes rather
than asserting that a token-overlap provider must achieve a quality threshold.
A deterministic local provider may still validate the normal query/document
routing and result shape without turning its score into evidence.

### Leakage, toy, privacy, and license checks

Required fixture checks:

- all query, skill, set, family, evidence, and source ids are unique and
  resolvable;
- every compatibility query has at least two positives and at least one
  rejected set;
- every risk query has exactly one helpful and one risky sibling in the same
  family;
- helpful and risky ids differ, and neither appears as its own hard negative;
- no query contains an exact normalized skill name;
- no external URL, repository path, username, email address, auth token,
  endpoint credential, or real private identifier appears;
- all dates, organizations, files, packages, and policy names are invented;
- fixture source and license status are the fixed local values above;
- `public_score_eligible` is false on every record;
- normalized duplicate skill text and duplicate query text are rejected;
- hard-negative and rejected-set review records are present; and
- serialization and metric outputs are stable across repeated runs.

## 8. Acceptance and Stop Criteria

### PASS

The future implementation may pass when:

- both task ids load through the registry and YAML catalog;
- one shared invented fixture validates with no network;
- a deterministic provider receives `retrieval_query` and
  `retrieval_document` routing;
- the hand-authored score-matrix tests reproduce exact fractional recall,
  complete-set, Set-Compat, rejected-set exposure, helpful retrieval, HSR,
  pair-win, and safe-helpful metrics;
- every risk result emits `safe_helpful@K`, `helpful_recall@K`, and `hsr@K`
  together, and no acceptance or comparison uses the conjunctive gate alone;
- result details expose query/document/qrel/set/pair/hard-negative counts,
  slices, license status, and `public_score_eligible=false`; and
- no fixture score is published or described as model quality, compatibility
  intelligence, execution safety, or end-to-end agent success.

### BLOCKED

Real-data work remains blocked when any of these is true:

- SkillResolve has no discoverable official release URL, data license, hashes,
  schemas, or checker;
- R3 rejected-combination annotations remain unreleased;
- R3 rows cannot be mapped from opaque ids to pinned source rows and licenses;
- an upstream dataset offers only a repository-level license or attribution
  note without row-level redistribution authority;
- the R3 Recall/Completeness convention remains ambiguous for the intended
  import;
- a source requires credentials, payment, a large download before schema
  review, or unclear cross-project publishing authority; or
- family-aware post-processing would be silently mixed into an embedding-only
  provider result.

### ABANDON

Abandon a proposed real-data path if its only feasible form requires:

- republishing unclear-license skill bodies;
- recreating unavailable benchmark rows from paper examples;
- treating an arXiv license as a dataset license;
- running a cross-encoder, reader LLM, agent executor, or verifier while
  labeling the result embedding-only;
- collapsing helpful retrieval and HSR into a scalar that hides either side;
  or
- publishing the four-query toy fixture as leaderboard evidence.

## 9. Smallest Repository Integration Path

No implementation is authorized by this dispatch. The smallest later patch is:

1. Add one shared invented data module, for example
   `src/mm_embed/data/agent_skill_routing_fixture.py`, containing the 12 skills,
   four queries, qrels, compatible/rejected sets, risk pairs, hard negatives,
   deterministic serialization, and validators.
2. Add two thin `EvalTask` implementations and two lazy registry/YAML entries:
   one for compatible-set metrics and one for helpful/risky exposure metrics.
   Keep family-aware representative selection out of both provider tasks.
3. Add one focused test module covering fixture validation, catalog resolution,
   provider routing, and exact metrics from a hand-authored score matrix.

The existing `agent_procedural_tool_memory` files do not need to change except
possibly to reuse a future general multi-positive metric helper after that
helper has independent regression coverage.

## 10. Restrained Follow-Ups

1. **Implement only the invented shared fixture and the two task ids.** Do not
   download or ingest R3, SkillRet, SkillRouter, SRA-Bench, SkillsBench, or
   SkillResolve rows in the same change.
2. **Open a provenance/release gate note before any real-data pilot.** Require
   an official SkillResolve artifact URL and license; for R3 require the SKIP
   files plus a row-level source manifest with immutable revisions and license
   evidence.
3. **If a real pilot becomes legally auditable, keep it private and
   non-publishing first.** Validate schema, metric conventions, duplicate and
   leakage checks, and family/rejected-set semantics before any model run or
   leaderboard export.

## 11. Primary Sources and Pinned Revisions

### R3-Skill

- Paper v4: https://arxiv.org/abs/2606.03565v4
- HTML v4: https://arxiv.org/html/2606.03565v4
- Official repository: https://github.com/Tencent/R3-Skill
- Audited commit: https://github.com/Tencent/R3-Skill/commit/57d9285969cc293832c71ecbe42a75ac416bef93
- Pinned README: https://raw.githubusercontent.com/Tencent/R3-Skill/57d9285969cc293832c71ecbe42a75ac416bef93/README.md
- Pinned license/attribution: https://raw.githubusercontent.com/Tencent/R3-Skill/57d9285969cc293832c71ecbe42a75ac416bef93/LICENSE
- Pinned reproduction script: https://raw.githubusercontent.com/Tencent/R3-Skill/57d9285969cc293832c71ecbe42a75ac416bef93/reproduce.py

### SkillResolve-Bench

- Paper v1: https://arxiv.org/abs/2606.10388v1
- HTML v1: https://arxiv.org/html/2606.10388v1

No official code/data URL was present in the paper or confirmed during the
2026-07-21 artifact search.

### SkillRet and R3 upstreams

- SkillRet paper v1: https://arxiv.org/abs/2605.05726v1
- SkillRet HF dataset v1.0, audited revision:
  https://huggingface.co/datasets/ThakiCloud/SKILLRET/tree/7cae7cfbad2b0e1ebc9170892f568993aae543b0
- SkillRet HF commit:
  https://huggingface.co/datasets/ThakiCloud/SKILLRET/commit/7cae7cfbad2b0e1ebc9170892f568993aae543b0
- SkillRet official code commit:
  https://github.com/ThakiCloud/SKILLRET/commit/0b9564a5021751661f93333adbf996fc51b577d2
- SkillRouter Eval Core audited revision:
  https://huggingface.co/datasets/pipizhao/SkillRouter-Eval-Core/tree/20a03920e7f08d76b67af367350d25ef4468198e
- SkillRouter Eval Core commit:
  https://huggingface.co/datasets/pipizhao/SkillRouter-Eval-Core/commit/20a03920e7f08d76b67af367350d25ef4468198e
- AiSkill pinned branch revision:
  https://gitcode.com/zhoukc42/AiSkill/tree/266f2f57f89e0f719f48827ea4e51bb1243132a8
- agency-agents-zh audited commit:
  https://github.com/jnMetaCode/agency-agents-zh/commit/cf086f76a225f7b287db0953940f76df6df8d1e3

## 12. Final Decision

Preserve `agent_procedural_tool_memory` as the repository's ordinary
query-to-procedural-document relevance task. Add, in a later implementation
dispatch, two separate provider-neutral diagnostic task ids backed by one
invented fixture: one for compatible-set retrieval and one for same-capability
risky-sibling exposure.

The fixture path is ready to proceed without network, models, execution, or
publishing. Real R3 ingestion is blocked by unreleased SKIP annotations and an
insufficient row-level redistribution chain. Real SkillResolve ingestion is
blocked because the claimed release bundle and its license could not be found.
