# Retrieval Answer Utility Fit Minispec - 2026-07-27

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_2-1785148745-2_execute.md`

Selected item: `tasks/retrieval-answer-utility-fit-minispec`

Unique session: `meb-modern-embedding-leaderboard-2-1785148745-2-retrieval-answer-utility-eb1003e89b04`

Decision: **GO, as a separately labeled system-level benchmark family.**

Do not add answer utility to the embedding leaderboard, do not reinterpret
answer correctness as an embedding score, and do not extend the existing
embedding task runner until a separate result contract and product surface are
accepted.

The repository has a real use for a downstream answer-utility family because
its agent memory, skill-routing, and code-localization tasks currently stop at
ranking. A fixed answer task can test whether retrieved evidence is converted
into a correct, cited answer under a controlled system envelope. That is useful,
but it is a property of the full retrieval-answer system: corpus, index,
retriever, optional rewrite/rerank logic, context assembly, generator, prompt,
and answer judge. It is not a property of the embedding model alone.

The GO applies to a bounded, deterministic v0 contract and a future invented
no-network fixture. Public data ingestion, model execution, evaluator
integration, and Hugging Face publication remain **BLOCKED** until the schema,
license, privacy, leakage, trace, cost, and latency gates in this note pass.

## 1. Evidence Labels and Decision Boundary

This note uses four labels:

- **VERIFIED**: observed in this repository or in an official GitHub source on
  2026-07-27.
- **UPSTREAM PROPOSAL**: proposed by MTEB issue 4868 or PR 4869 but not merged
  into MTEB `main`.
- **REPOSITORY PROPOSAL**: recommended design for this repository.
- **GATE**: required before implementation, execution, or publication.

The proposed family measures this statement:

> Given a fixed query, a fixed corpus, a pinned system manifest, and a typed
> answer contract, does the system return the correct answer with complete and
> valid citations, under auditable cost and latency accounting?

It does not measure only:

- embedding similarity or ranking quality;
- generator knowledge or reasoning in isolation;
- agent task completion, tool execution, or patch correctness;
- open-web research quality;
- subjective prose quality; or
- an unpinned composition of changing hosted services.

## 2. Upstream MTEB State at the Audit Cutoff

### 2.1 Issue 4868

**VERIFIED.** `Retrieval Systems [Agentic MTEB]` is open. It was created on
2026-06-30 and had no issue comments at the audit cutoff. The issue proposes two
modes over related data:

- existing ranking mode for nDCG, recall, and MAP; and
- answer mode for correctness, cost, and latency.

Its intended `AnswerResult` includes an answer, cited document ids, usage, and a
trace. It proposes closed-book and oracle-context baselines, later RAG and
query-rewrite paradigms, and a future external CLI-agent track. Its first named
dataset candidate is BrowseComp-Plus, but the scope remains explicitly
unfinished.

The issue is useful design evidence, not an adopted standard. No released task,
dataset revision, score schema, or leaderboard policy is pinned by the issue.

### 2.2 PR 4869 status

**VERIFIED.** PR 4869 is the first implementation attempt for the proposed
`mteb.agentic` subsystem.

| Field | Audited value |
| --- | --- |
| PR state | open, not draft, not merged |
| Created | 2026-06-30T15:36:54Z |
| Last updated | 2026-07-27T05:12:12Z |
| Label | `stale` |
| Stale event | GitHub Actions comment on 2026-07-27T05:12:12Z |
| Head commit | `9b66032d0a0e6e3a36d6ad500419e9572c27fec8` |
| Base commit observed by API | `8b8f169ba09b1f76a4b1f22235ad438f60f272d6` |
| Mergeability snapshot | `mergeable=true`, `mergeable_state=clean`, `rebaseable=true` |
| Size | 9 added files, 536 additions, 0 deletions, 2 commits |
| Formal reviews | one `COMMENTED` review; no approval |
| Review summary | "Currently a bit hard to judge without full pipeline" |

The head checks visible through the official Checks API were successful for
lint, typecheck, build, leaderboard, and the reported Linux/Windows test jobs;
the deploy job was skipped. Successful checks do not resolve the product and
measurement questions below. GitHub's combined commit-status endpoint reported
no classic status contexts and returned a pending aggregate state.

The two PR commits are:

- `cef55cf3911dadc401db2752d0fbde4b7d5116a2` - initial answer-mode core; and
- `9b66032d0a0e6e3a36d6ad500419e9572c27fec8` - LLM-judge reasoning-output
  handling.

### 2.3 Contract actually present at the pinned PR head

**VERIFIED.** The pinned head contains:

- `AnswerSystem.answer(question, corpus) -> AnswerResult`;
- a `CorpusHandle` with only `get(doc_id)`;
- `AnswerResult(answer, cited_doc_ids, usage)`;
- usage fields for prompt tokens, completion tokens, LLM call count, optional
  USD cost, and optional latency;
- normalized exact-match and binary LLM judges;
- an evaluator that loops over questions and records answer, correctness,
  citations, latency, cost, and LLM-call count;
- a retrieval-data adapter that adds reference answers and gold document ids;
- closed-book and oracle-context systems; and
- a one-query deterministic unit-test fixture.

The issue says `AnswerResult` carries a trace, but the pinned PR implementation
has no trace field. The issue also mentions an `Embedder`, while the pinned
interface exports no embedder protocol and the corpus handle exposes no search
method.

### 2.4 Unresolved review tensions and contract gaps

#### Exact-id versus LLM judging

**VERIFIED.** A maintainer review comment says an LLM judge may be unnecessary
because ids can be compared and warns that generating and checking with LLMs is
too much LLM checking. The PR nevertheless retains `LLMJudge`.

**REPOSITORY PROPOSAL.** V0 must use typed deterministic answers: exact ids,
canonical alias sets, sets of ids, booleans, or numbers with a declared
tolerance. No LLM judge is allowed in the public v0 primary metric. A later
free-form track may add an LLM judge only as a separately named diagnostic with
pinned judge model, prompt, decoding settings, repetitions, disagreement rate,
judge cost, and judge latency. It must never overwrite deterministic scores.

#### Reuse of existing retrieval indexes

**VERIFIED.** A review comment proposes reusing existing indexes. The pinned
`CorpusHandle` cannot search an index; it can only fetch a document by id.

**REPOSITORY PROPOSAL.** Index reuse is allowed and desirable only when an
immutable `index_id` binds corpus revision, serialization, chunker, embedding
model revision, embedding dimensions, similarity function, normalization,
filtering, and index-build parameters. An index may be reused across system
runs when all hashes match. A prebuilt index must not hide an embedding change,
future-corpus leakage, query-specific filtering, or unreported indexing cost.

#### Citation and trace completeness

**VERIFIED.** The PR records `cited_doc_ids`, but it does not validate that ids
exist, score citation precision or recall, require citations, preserve the
retrieved ranking, or expose a trace. The evaluator's per-question record omits
token counts and does not include the proposed trace.

**REPOSITORY PROPOSAL.** Every retrieval and oracle answer must return ordered
citations and a seekable trace. Public eligibility requires:

- every cited id exists in the pinned corpus;
- retrieved ids, scores, and ranks are recorded up to the context cutoff;
- the context documents actually passed to the generator are recorded by id
  and order;
- answer, citation, usage, and trace fields are present for every attempted
  query, including failures and timeouts; and
- completeness counters equal the attempted-query count.

#### Missing-cost aggregation

**VERIFIED.** The pinned aggregate sums and averages only non-null costs. A run
with one known cost and many missing costs can therefore display a plausible
mean and total without revealing that most costs are absent. Judge-model usage
is also outside the system result and aggregate.

**REPOSITORY PROPOSAL.** Null cost is unknown, not zero. Report known-cost total,
cost coverage, missing-cost count, and separately scoped index-build, online
system, and judge costs. A public `total_cost_usd` or `mean_cost_usd` is valid
only when cost coverage is 100%; otherwise use explicitly named partial-cost
fields and mark cost comparison ineligible.

#### Latency semantics

**VERIFIED.** The evaluator measures wall time around `system.answer` only when
the system has not supplied its own latency. This mixes self-reported and
harness-measured values and does not define inclusion of retrieval, queueing,
retries, judge time, index construction, cache state, concurrency, or timeout
handling.

**REPOSITORY PROPOSAL.** The harness owns online latency. Measure monotonic wall
time from query release to receipt of the final answer and citations, including
online query embedding, retrieval, reranking, context construction, generation,
and system retries. Exclude dataset loading, index construction, and answer
judging, but report them separately. Pin concurrency, warm/cold policy, cache
policy, timeout, retry policy, machine class, and region. Report median, p95,
mean, timeout rate, and attempted-query denominator.

#### Future multimodal scope

**VERIFIED.** A review comment says multimodal support will be needed. The
pinned `Message` and corpus contracts are string mappings and do not define
media identity, bytes, URLs, transforms, or modality-specific citations.

**REPOSITORY PROPOSAL.** V0 is text-only. A future multimodal version requires
content-addressed local assets, explicit modality and transform metadata,
region/time-span citations, and license/privacy review per asset. Do not make
the v0 answer contract vaguely multimodal or accept remote URLs as evidence.

### 2.5 Upstream conclusion

The upstream work validates the need for a separate answer surface, but its
current PR is stale, unapproved, not integrated into the normal MTEB evaluation
path, and incomplete on the exact measurement questions that matter here. This
repository should borrow the separation between ranking and answer modes, not
copy the current implementation contract.

## 3. Current Repository Baseline

The repository state was audited at local commit
`df709b6118c182157a150e4ece878fda64773081`.

### 3.1 Audited repository files

The fit decision was checked against every required local surface:

| File | Relevant verified constraint |
| --- | --- |
| `README.md` | Defines the project as an embedding benchmark, documents one model-task JSONL record, and exposes one Dataset/Space publication path. |
| `benchmark/README.md` | Defines the v2 registry/manifest control plane and task-specific leaderboard generation. |
| `benchmark/tasks/core.yaml` | Registers embedding retrieval tasks with one primary metric; the skill-routing and code-localization fixtures are explicitly non-public. |
| `schemas/result.schema.json` | Requires model/provider-shaped v2 records but has no evaluation-level or subject-kind discriminator. |
| `src/mm_embed/benchmark/results.py` | Materializes one embedding model and one provider result per task record. |
| `src/mm_embed/benchmark/leaderboard.py` | Produces a flat, descending-score table; excludes errors, unpublished rows, and rows without a primary metric; and has no benchmark-level discriminator. |
| `src/mm_embed/hf_publish/export.py` | Publishes the flat embedding result/leaderboard artifacts, builds a single task-specific Space view, and generates a Dataset card warning against cross-task global ranking. |
| `src/mm_embed/tasks/agent_procedural_tool_memory.py` | Evaluates query-to-tool-card ranking with full-corpus and curated hard-pool metrics, without answering or execution. |
| `src/mm_embed/tasks/agent_skill_routing.py` | Separates compatible-set retrieval from risky-sibling exposure and keeps both as deterministic fixture diagnostics. |
| `src/mm_embed/tasks/code_edit_chunk_localization.py` | Requires complete-corpus ranking, deterministic ties, snapshot identity, and patch-aligned localization metrics without patch generation. |
| `benchmark/research/agent_memory_minispec_20260717.md` | Establishes the repository precedent that agent memory should begin as embedding-auditable document retrieval, not a full agent-success task. |
| `benchmark/research/agent_skill_routing_compatibility_risk_minispec_20260721.md` | Establishes separate task identities when claims and primary metrics differ, plus invented no-publish fixtures before real-data ingestion. |
| `benchmark/research/code_context_retrieval_minispec_20260722.md` | Establishes static localization as the embedding surface and defers generation/agent behavior, with strong provenance and HF separation gates. |

Taken together, these files show a model/provider-shaped result path and a flat
task leaderboard. The recommendation to keep answer utility separate is a
design conclusion from those constraints, not an existing repository policy.

### 3.2 Product boundary

**VERIFIED.** The repository describes itself as a maintainable embedding
benchmark for practical retrieval gaps. Its public path is:

1. reviewable model and task YAML registries;
2. one v2 JSONL record per model-task pair;
3. a flat task-specific leaderboard CSV; and
4. a Hugging Face Dataset plus a Gradio Space that read those records and rows.

`README.md` documents one result record per model-task pair, but it does not
state a prohibition on global ranking. The generated Dataset card template in
`src/mm_embed/hf_publish/export.py` states that scores are task-specific and
should not be compared across tasks as one global ranking. The current result
builder identifies the evaluated subject through `model` and `provider_result`;
it does not define a separate system subject.

### 3.3 Existing agent-facing embedding tasks

| Current task | Fixed input and target | What it measures | Why answer utility is different |
| --- | --- | --- | --- |
| `agent_procedural_tool_memory` | task instruction -> one tool card plus curated hard negatives | full-corpus and hard-pool ranking | does not execute or answer; one positive document id is the gold |
| `agent_skill_compatible_set_retrieval` | instruction -> multi-positive compatible skill set | recall, complete-set retrieval, nDCG, rejected-set exposure | does not select or run skills and makes no downstream success claim |
| `agent_skill_same_capability_risk` | instruction -> helpful skill versus risky sibling | helpful retrieval and harmful-sibling exposure | risk is inferred from ranking, not system behavior |
| `code_edit_chunk_localization` | issue -> chunks in a frozen invented repository | edit-chunk ranking, line-budget coverage, file recall, hard-negative diagnostics | does not generate a patch, explanation, or cited answer |

All four tasks embed query and document texts separately, compute a fixed score
matrix, and return deterministic ranking metrics. The skill-routing and code
localization tasks are fixture-only and non-publishable. The code task also
enforces complete-corpus ranking and records snapshot hashes, deterministic
tie-breaking, and explicit public-score eligibility.

Answer utility should use those practices but must not change their claims.
For example:

- tool-memory answer utility may ask for a typed tool id and cite the tool card;
- skill-routing answer utility may return a compatible skill-id set, but must
  not claim execution success;
- code-localization answer utility may return a ranked or bounded set of chunk
  ids with citations, but must not claim patch correctness; and
- a later RAG answer task may use retrieved evidence to produce a factual
  answer, but its correctness belongs to the full system.

### 3.4 V2 result schema mismatch

**VERIFIED.** `schemas/result.schema.json` requires `run`, `model`, `task`,
`provider_result`, `metrics`, `details`, and `error`, while allowing additional
properties. `make_result_record` always writes embedding-model identity and one
provider result. The schema has no required discriminator for:

- embedding versus system evaluation;
- ranking versus answer mode;
- the generator, retriever, reranker, judge, prompt, or index;
- bracket identity;
- answer/citation/trace completeness; or
- cost and latency coverage.

Because additional properties are allowed, a system result could technically
be inserted today. That would be dangerous: downstream readers could treat a
generator or composite system as a model row and present its answer accuracy as
an ordinary task score.

### 3.5 Flat leaderboard mismatch

**VERIFIED.** `build_leaderboard` emits one flat row per successful public
model-task result with `primary_metric`, `score`, and `duration_s`. It filters on
run/task publication flags but has no benchmark-level discriminator. The Space
sorts every selected task by descending `score` and labels the product Modern
Embedding Bench.

This is suitable for one task-specific embedding metric. It cannot faithfully
represent answer quality, citation quality, cost, latency, timeouts, and bracket
position without encouraging a false single-score interpretation.

### 3.6 Hugging Face boundary

**VERIFIED.** The Dataset export writes public model specs, public task specs,
run manifests, `results/latest.jsonl`, and `leaderboards/latest.csv`. The Space
loads that flat CSV and provides task, provider, evidence-tier, and text
filters. It has no separate subject kind or benchmark family.

The export card currently declares `license: mit`, while this repository has no
tracked root `LICENSE`, `COPYING`, or `NOTICE` file at the audited commit. That
existing mismatch is a publication gate for any new self-authored fixture and
must not be used as authority to redistribute third-party corpora or answers.

## 4. Fit Decision

### 4.1 Why GO

There is a coherent missing layer between fixed retrieval rankings and real
agent/RAG outcomes. The same corpus and queries can support two valid but
different questions:

1. did an embedding rank relevant evidence early; and
2. did a pinned retrieval-answer system convert available evidence into a
   correct, cited answer at an acceptable resource cost?

The second question is useful for model and system selection, regression
testing, and diagnosing whether retrieval gains survive context assembly and
generation. The repository already has the provenance, fixture, evidence-tier,
and per-task product patterns needed to define it carefully.

### 4.2 Why it must be a separate family

Answer utility changes the evaluated subject and the causal claim. Its score is
affected by at least:

- query rewriting;
- corpus filtering and index revision;
- retrieval and reranking;
- context-window selection and ordering;
- prompt/template revision;
- generator and decoding parameters;
- tool use or retries;
- answer parsing;
- citation behavior; and
- judge behavior.

Calling that score an embedding score would reward or penalize an embedding for
uncontrolled system components. Therefore the family should be named
`retrieval_answer_utility`, carry `evaluation_level=system`, and live on a
separate system leaderboard surface.

### 4.3 Bounded v0 claim

The first claim should be narrow:

> For typed, closed-corpus questions with deterministic gold answers and gold
> evidence ids, compare pinned retrieval-answer systems under identical
> generator, prompt, context, judge, resource, and execution controls.

V0 excludes open-web browsing, subjective free-form judging, executable tools,
code generation, multimodal evidence, adaptive sandbox agents, private user
memory, and unpinned hosted indexes.

## 5. Proposed V0 Data and System Contract

### 5.1 Immutable task bundle

One task revision contains:

- `queries.jsonl`;
- `corpus.jsonl`;
- `answers.jsonl`;
- `qrels.jsonl`;
- `task_manifest.json`; and
- a content hash for every file plus the complete bundle.

The task manifest pins dataset id and revision, split, answer schema version,
serialization version, license decision, privacy review, leakage review, and
all source revisions.

### 5.2 Query record

```json
{
  "query_id": "q_policy_001",
  "text": "Which retention policy applies to archived audit logs?",
  "answer_type": "entity_id",
  "required_citation_count_min": 1,
  "split": "fixture_only",
  "source_kind": "local_invented_fixture",
  "source_revision": "retrieval-answer-utility-fixture-v0"
}
```

Query text must not contain the canonical answer id, document id, hidden gold
metadata, or a templated phrase that deterministically reveals the answer.

### 5.3 Corpus record

```json
{
  "doc_id": "doc_retention_archive",
  "title": "Archived audit log retention",
  "text": "Archived audit logs use policy RETENTION-7Y.",
  "media_type": "text/plain",
  "source_kind": "local_invented_fixture",
  "source_revision": "retrieval-answer-utility-fixture-v0",
  "content_sha256": "<64 lowercase hex characters>",
  "license_status": "local_invented_pending_repository_license"
}
```

Document ids are stable within a task revision. Corpus order is canonical and
must not encode relevance.

### 5.4 Answer and evidence ground truth

```json
{
  "query_id": "q_policy_001",
  "answer_type": "entity_id",
  "canonical_answer": "RETENTION-7Y",
  "accepted_answers": ["RETENTION-7Y"],
  "normalizer": "casefold_trim_v0",
  "numeric_tolerance": null
}
```

```json
{
  "query_id": "q_policy_001",
  "doc_id": "doc_retention_archive",
  "relevance": 1,
  "support_kind": "direct",
  "required_for_complete_support": true
}
```

Allowed v0 answer types are:

- `entity_id`;
- `entity_id_set`, with order-insensitive exact set equality;
- `boolean`;
- `integer`; and
- `number`, with an explicit absolute and/or relative tolerance.

Free-form prose is not a public v0 answer type. Each query must have at least
one gold evidence document. Multi-document questions must declare whether all
required documents are necessary or whether any one of several alternatives is
sufficient.

### 5.5 Pinned system manifest

Every evaluated subject is a system manifest, not a single model alias:

```json
{
  "system_id": "dense-rag-fixed-generator-v0",
  "system_revision": "<immutable revision or manifest hash>",
  "bracket": "retrieval",
  "retriever": {
    "kind": "dense",
    "model_id": "<model id>",
    "model_revision": "<immutable revision>",
    "index_id": "<content-addressed index id>",
    "top_k": 3
  },
  "reranker": null,
  "generator": {
    "model_id": "<model id>",
    "model_revision": "<immutable revision>",
    "prompt_sha256": "<64 lowercase hex characters>",
    "temperature": 0,
    "max_output_tokens": 64
  },
  "context": {
    "max_documents": 3,
    "max_input_tokens": 2048,
    "ordering": "retrieval_rank"
  },
  "execution": {
    "concurrency": 1,
    "timeout_s": 60,
    "max_retries": 0,
    "cache_policy": "disabled"
  }
}
```

All component revisions and prompt bytes must be recoverable from the manifest
or a content-addressed artifact. Secrets must never be stored.

### 5.6 Answer-system output

```json
{
  "query_id": "q_policy_001",
  "answer": "RETENTION-7Y",
  "answer_type": "entity_id",
  "cited_doc_ids": ["doc_retention_archive"],
  "retrieved": [
    {"doc_id": "doc_retention_archive", "rank": 1, "score": 0.83}
  ],
  "context_doc_ids": ["doc_retention_archive"],
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 6,
    "llm_calls": 1,
    "online_cost_usd": 0.0004,
    "cost_complete": true
  },
  "trace": [
    {"event": "retrieve", "component": "retriever", "start_ms": 0, "end_ms": 4},
    {"event": "generate", "component": "generator", "start_ms": 5, "end_ms": 120}
  ],
  "status": "ok",
  "error": null
}
```

`status` is one of `ok`, `invalid_answer`, `timeout`, `error`, or `refused`.
Every attempted query produces a record. Missing answers, malformed answers,
timeouts, and errors score zero for answer correctness and remain in latency,
cost-coverage, and failure denominators as defined below.

### 5.7 Deterministic judge

The v0 judge must:

1. parse the declared answer type;
2. apply only the pinned normalizer or numeric tolerance;
3. compare against the accepted deterministic gold values;
4. validate that every citation exists;
5. score citation precision and required-evidence recall against qrels; and
6. emit a reason code without calling a model or network service.

Recommended reason codes include `exact_match`, `accepted_alias`,
`set_mismatch`, `numeric_out_of_tolerance`, `malformed_answer`,
`missing_required_citation`, `unknown_citation`, `timeout`, and `system_error`.

## 6. Closed-Book, Oracle, and Retrieval Brackets

The three brackets use the same query set, answer schema, generator revision,
prompt family, decoding settings, token limits, and deterministic judge.

### Closed-book bracket

The generator receives the query and no corpus content. It measures parametric
knowledge and answer-format behavior. It is an anchor, not a guaranteed floor:
a generator may know the answer or an oracle prompt may fail.

### Oracle-context bracket

The generator receives exactly the declared gold evidence documents in a
canonical order and within the same context budget. It measures whether the
answer can be produced from available gold evidence. It is an anchor, not an
assumed mathematical ceiling.

### Retrieval bracket

The system obtains context through its declared retrieval path. It must expose
the full retrieved prefix used for scoring and the exact context ids passed to
the generator. Query rewriting, reranking, hybrid search, and adaptive
retrieval create distinct system manifests and cannot be attributed to the
embedding alone.

### Required bracket diagnostics

Report:

- retrieval answer accuracy minus closed-book answer accuracy;
- oracle answer accuracy minus retrieval answer accuracy;
- retrieval citation recall versus pure ranking recall at the same cutoff;
- answerable-with-oracle count;
- retrieval failures among oracle-answerable queries; and
- generation failures where required evidence was present in context.

These are decomposed diagnostics. Do not turn them into one composite score.

## 7. Cost and Latency Accounting

### 7.1 Cost scopes

Track independently:

- `index_build_cost_usd` and build resource time;
- `online_system_cost_usd` per query;
- `judge_cost_usd`, which is zero for deterministic v0 judging; and
- optional storage cost under an explicit time horizon.

For every scope report attempted count, known count, missing count, and
coverage ratio. Do not publish a complete-looking total when coverage is below
1.0. Precomputed embedding or index cost may be amortized only with a declared
query volume, and raw plus amortized values must both remain available.

### 7.2 Latency scopes

Report separately:

- index build duration;
- online end-to-end latency;
- retrieval-only latency;
- generation latency;
- deterministic judge latency; and
- timeout rate.

The primary system latency is harness-measured online end-to-end wall time.
Use the same machine class, locality, concurrency, warmup count, cache policy,
timeout, and retry policy for comparisons. Report median and p95 in addition to
mean; never replace failed or timed-out queries with missing rows.

## 8. Explicit Confound Controls

A comparison may isolate embedding effects only when all of the following are
fixed:

- task and corpus revision;
- document serialization and chunking;
- index implementation and search parameters;
- query text and any instruction prefix;
- reranker state, usually absent for the embedding-isolation slice;
- generator model and immutable revision;
- prompt bytes and answer schema;
- context token/document budget and ordering;
- decoding parameters;
- deterministic judge revision;
- hardware, region, concurrency, timeout, retry, and cache policy; and
- software environment and git revision.

If query rewriting, reranking, generator, prompt, tool policy, or context
budget changes, the row is a general system comparison. It must not be used to
claim an embedding improvement.

## 9. Metrics and Leaderboard Placement

### 9.1 Metrics that may remain on the embedding leaderboard

Only metrics computed directly from a fixed query-document score matrix or
fixed ranking produced by the embedding evaluation path may remain:

- recall at fixed cutoffs;
- MRR;
- nDCG;
- hard-pool ranking metrics;
- compatible-set and risk-exposure metrics derived only from rankings;
- line-budget or target coverage derived only from the fixed ranking; and
- embedding/query/document latency already scoped to the provider operation.

If an answer experiment also wants an embedding ranking row, it must emit a
separate embedding result record under the original embedding task contract.
The generator output must not affect that row.

### 9.2 Metrics that require the separate system-level view

The system view should report at minimum:

- `answer_accuracy` as the v0 primary metric;
- `citation_precision`;
- `required_citation_recall`;
- `citation_f1`;
- `valid_answer_rate`;
- `trace_complete_rate`;
- `cost_complete_rate`;
- known online cost per attempted query;
- median and p95 online latency;
- timeout and error rates;
- closed-book and oracle bracket values; and
- the decomposed bracket gaps from section 6.

No weighted quality-cost-latency composite is recommended. The Space should
show quality, citation, cost, latency, and failure columns together and allow
the user to filter by bracket and system component revisions.

### 9.3 Prohibited presentation

Do not:

- put `answer_accuracy` in `leaderboards/latest.csv` beside embedding scores;
- call a generator or composite system an embedding model;
- average answer utility with nDCG, MRR, or other task scores;
- sort systems by a hidden quality/cost composite;
- treat oracle-context output as an embedding result; or
- present incomplete cost or trace records as comparable public rows.

## 10. Required Result and Hugging Face Schema Separation

### 10.1 Result discriminator

The current v2 schema is not sufficient by convention alone. A future change
should either add a separately validated system-result schema or introduce a
new schema version with required discriminators. The preferred shape is:

```json
{
  "schema_version": "3.0",
  "evaluation": {
    "family": "retrieval_answer_utility",
    "level": "system",
    "mode": "answer_utility",
    "leaderboard_surface": "system"
  },
  "subject": {
    "kind": "retrieval_answer_system",
    "id": "dense-rag-fixed-generator-v0",
    "manifest_sha256": "<64 lowercase hex characters>",
    "components": {}
  },
  "task": {},
  "metrics": {},
  "resource_usage": {},
  "details": {},
  "error": null
}
```

Embedding records keep `evaluation.level=embedding`,
`evaluation.mode=ranking`, and `subject.kind=embedding_model` if migrated. The
exporter must reject a system record from the embedding leaderboard even when
its task or run publication flags are true.

The current `model` and `provider_result` objects cannot identify a composite
system without overloading their meaning. System records need a subject
manifest and component list. A generator model may still appear as a component,
not as the evaluated embedding model.

### 10.2 Dataset repository paths

Keep the current embedding artifacts for backward compatibility, but add
explicit system paths only after implementation:

```text
results/embedding/latest.jsonl
results/system/latest.jsonl
leaderboards/embedding/latest.csv
leaderboards/system/latest.csv
systems/manifests.jsonl
tasks/answer_utility.jsonl
```

If `results/latest.jsonl` and `leaderboards/latest.csv` remain as aliases, they
must continue to point only to embedding artifacts. The Dataset card must state
that system rows are not embedding scores.

### 10.3 Space boundary

The Space should use separate top-level views or tabs:

- `Embedding ranking`; and
- `Retrieval answer systems`.

The system view needs filters for bracket, retriever, embedding revision,
generator revision, corpus/index revision, evidence tier, and cost completeness.
It should not share a single sortable `score` column with embedding tasks.

## 11. Dataset, License, Privacy, and Leakage Gates

### 11.1 Source and revision gate

Every non-invented query, document, answer, and qrel must preserve:

- official source URL;
- immutable repository commit or dataset revision;
- source split and row id;
- retrieval corpus revision;
- answer/qrel construction method; and
- content hash after normalization.

Do not ingest BrowseComp-Plus or another proposed source merely because it is
named in the MTEB issue. Audit its card, files, exact revision, row schema,
license, privacy terms, answer provenance, evidence ids, and redistribution
rights in a separate item first.

### 11.2 License gate

Distinguish benchmark-package license from source-text licenses. A wrapper
dataset license does not relicense copied documents, queries, answers, or media.
Publication requires a row-level or source-bundle ledger and a resolved license
for this repository. The current hard-coded MIT Dataset card is not sufficient
while the repository has no tracked root license.

### 11.3 Privacy and secret gate

Exclude or redact:

- private conversations and user memory;
- credentials, tokens, auth headers, private URLs, and internal hostnames;
- personal email addresses, phone numbers, and direct identifiers not required
  by the task;
- medical, financial, employment, or other sensitive personal records; and
- documents whose cited evidence would reproduce prohibited private content.

Record the scanner revision, findings count, adjudication, and exclusions.

### 11.4 Leakage gate

Check and record:

- exact and near-duplicate queries across train/dev/test;
- answers leaked in query text, titles, ids, filenames, metadata, or ordering;
- future-corpus or future-index data relative to the query cutoff;
- generator pretraining contamination risk when known;
- prompt examples that contain evaluation answers;
- aliases shared across splits; and
- cached responses from prior benchmark runs.

System comparison is invalid when one row uses a different corpus snapshot or
when the index contains documents not visible to the query revision.

## 12. Tiny Self-Authored No-Network Smoke Design

The first future artifact should validate contracts, not model quality.

### 12.1 Fixture size

- 6 invented queries;
- 12 invented short text documents;
- 4 single-evidence answers;
- 2 two-evidence answers;
- typed canonical answers and accepted alias lists;
- 1 to 2 required citation ids per query;
- deterministic corpus order and hashes;
- no third-party text, names, URLs, or identifiers; and
- `publish: false`, `evidence_tier: fixture`.

Suggested invented domains are retention-policy ids, build-artifact ids,
feature-flag ids, and schema-version ids. Every value should be fictional and
obviously non-operational.

### 12.2 Deterministic smoke systems

Use pure local fixtures with no model or provider:

1. `closed_book_constant`: returns `UNKNOWN` and no citations;
2. `oracle_structured_lookup`: receives gold documents and deterministically
   parses their typed answer fields;
3. `token_overlap_retrieval`: ranks all 12 documents by a fixed token-overlap
   function with doc-id tie-breaking, passes the top two documents to the same
   parser, and cites the documents it used; and
4. intentionally malformed and missing-cost variants for negative tests.

The two-evidence cases should require combining two declared answer parts so
that citation completeness can fail independently of answer parsing.

### 12.3 Exact smoke assertions

The future tests should assert:

- closed-book answer accuracy is 0 and missing citations are counted;
- oracle answer accuracy and required-citation recall are 1;
- retrieval results match a checked-in expected ranking and metric table;
- unknown citation ids are rejected;
- missing one required citation lowers citation recall deterministically;
- malformed typed answers score zero with the expected reason code;
- null cost produces partial-cost fields and blocks complete-cost comparison;
- every attempted query has a trace or explicit failure record;
- repeated runs are byte-for-byte stable after timestamp normalization; and
- the embedding leaderboard exporter excludes all system records.

No benchmark, embedding call, chat call, model inference, download, or Hugging
Face operation is needed for this smoke.

## 13. PASS, BLOCKED, and ABANDON Criteria

### PASS

The family is ready for a later local fixture implementation when:

- the system-versus-embedding decision is accepted;
- typed answers, qrels, citations, brackets, and deterministic judge behavior
  are unambiguous;
- result records have required level/mode/subject discriminators;
- system records cannot enter the embedding leaderboard;
- cost and latency scopes plus completeness denominators are explicit;
- trace completeness is machine-checkable;
- the invented fixture passes without network, models, downloads, or APIs; and
- fixture records remain unpublished and labeled as fixture evidence.

A real public benchmark may pass only after all source, license, privacy,
leakage, reproducibility, and product-separation gates also pass.

### BLOCKED

Implementation or publication is blocked when:

- the only available answers require subjective or LLM-only primary judging;
- source answers or evidence ids are incomplete or cannot be pinned;
- the corpus/index revision cannot be reproduced;
- citations, retrieved context, trace, or attempted-query records are missing;
- cost coverage is incomplete but a complete cost metric is requested;
- latency mixes self-reported and harness-measured values;
- generator, prompt, context, or execution confounds are not pinned;
- third-party license or privacy rights are unclear;
- the repository license and HF card remain inconsistent for publication; or
- the current flat exporter cannot guarantee that system rows stay off the
  embedding leaderboard.

### ABANDON

Abandon this family for this repository if:

- stakeholders require one score that mixes answer quality with embedding
  ranking, cost, or latency;
- generative-system quality must be marketed as an embedding score;
- a defensible task necessarily depends on live open-web state, private user
  data, executable tools, or unpinned hosted services;
- deterministic answer and evidence gold cannot be built for any useful slice;
  or
- maintaining a distinct system schema and product view is out of scope.

## 14. Smallest Later Implementation Step

Create a separate future item named
`tasks/retrieval-answer-utility-fixture-contract`.

That item should implement only:

1. the six-query, twelve-document invented fixture;
2. typed-answer, qrel, citation, usage, and trace validators;
3. the deterministic judge and the three local smoke systems;
4. exact metric and completeness tests; and
5. a regression test proving system records are rejected from the existing
   embedding leaderboard path.

It should not add a public task, run a provider, change an embedding metric,
ingest BrowseComp-Plus, publish to Hugging Face, or add a system leaderboard UI
in the same change. Schema integration and product UI should be later items
after the fixture contract is accepted.

## 15. Primary Sources and Pinned Revisions

Audit date: 2026-07-27 UTC.

### MTEB proposal and PR state

- Issue 4868:
  https://github.com/embeddings-benchmark/mteb/issues/4868
- Official issue API snapshot endpoint:
  https://api.github.com/repos/embeddings-benchmark/mteb/issues/4868
- PR 4869:
  https://github.com/embeddings-benchmark/mteb/pull/4869
- Official PR API snapshot endpoint:
  https://api.github.com/repos/embeddings-benchmark/mteb/pulls/4869
- Pinned PR head:
  https://github.com/embeddings-benchmark/mteb/commit/9b66032d0a0e6e3a36d6ad500419e9572c27fec8
- Initial PR commit:
  https://github.com/embeddings-benchmark/mteb/commit/cef55cf3911dadc401db2752d0fbde4b7d5116a2
- Head tree in the contributor fork:
  https://github.com/AdnanElAssadi56/mteb/tree/9b66032d0a0e6e3a36d6ad500419e9572c27fec8/mteb/agentic
- Changed-files API:
  https://api.github.com/repos/embeddings-benchmark/mteb/pulls/4869/files?per_page=100
- Commits API:
  https://api.github.com/repos/embeddings-benchmark/mteb/pulls/4869/commits?per_page=100
- Reviews API:
  https://api.github.com/repos/embeddings-benchmark/mteb/pulls/4869/reviews?per_page=100
- Review-comments API:
  https://api.github.com/repos/embeddings-benchmark/mteb/pulls/4869/comments?per_page=100
- Conversation-comments API:
  https://api.github.com/repos/embeddings-benchmark/mteb/issues/4869/comments?per_page=100
- Check runs at the pinned head:
  https://api.github.com/repos/embeddings-benchmark/mteb/commits/9b66032d0a0e6e3a36d6ad500419e9572c27fec8/check-runs

### Review and stale-state permalinks

- Review summary:
  https://github.com/embeddings-benchmark/mteb/pull/4869#pullrequestreview-4680816091
- Exact-id versus LLM-judge comment:
  https://github.com/embeddings-benchmark/mteb/pull/4869#discussion_r3567161329
- Tool/trace-surface comment:
  https://github.com/embeddings-benchmark/mteb/pull/4869#discussion_r3567179580
- Existing-index reuse comment:
  https://github.com/embeddings-benchmark/mteb/pull/4869#discussion_r3567185151
- Future multimodal comment:
  https://github.com/embeddings-benchmark/mteb/pull/4869#discussion_r3567187181
- Automatic stale comment:
  https://github.com/embeddings-benchmark/mteb/pull/4869#issuecomment-5087504028

### Pinned upstream implementation files

- Interface:
  https://raw.githubusercontent.com/AdnanElAssadi56/mteb/9b66032d0a0e6e3a36d6ad500419e9572c27fec8/mteb/agentic/interface.py
- Evaluator:
  https://raw.githubusercontent.com/AdnanElAssadi56/mteb/9b66032d0a0e6e3a36d6ad500419e9572c27fec8/mteb/agentic/evaluator.py
- Metrics and judges:
  https://raw.githubusercontent.com/AdnanElAssadi56/mteb/9b66032d0a0e6e3a36d6ad500419e9572c27fec8/mteb/agentic/metrics.py
- Retrieval-data adapter:
  https://raw.githubusercontent.com/AdnanElAssadi56/mteb/9b66032d0a0e6e3a36d6ad500419e9572c27fec8/mteb/agentic/data.py
- Corpus implementation:
  https://raw.githubusercontent.com/AdnanElAssadi56/mteb/9b66032d0a0e6e3a36d6ad500419e9572c27fec8/mteb/agentic/corpus.py
- Closed-book and oracle baselines:
  https://raw.githubusercontent.com/AdnanElAssadi56/mteb/9b66032d0a0e6e3a36d6ad500419e9572c27fec8/mteb/agentic/systems/baselines.py
- Tests:
  https://raw.githubusercontent.com/AdnanElAssadi56/mteb/9b66032d0a0e6e3a36d6ad500419e9572c27fec8/tests/test_agentic.py

## 16. Final Decision

Proceed with `retrieval_answer_utility` as a future, separately labeled
system-level family. Its first implementation should be a deterministic local
contract fixture, not a model benchmark. Preserve the current embedding tasks
and leaderboard as ranking-only surfaces. A retrieval-answer result may explain
whether ranked evidence becomes a correct cited answer, but it must never be
presented as the embedding model's score.
