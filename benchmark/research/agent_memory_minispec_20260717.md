# Agent Memory Retrieval Minispec - 2026-07-17

Dispatch: `.perpetuum/modern-embedding-leaderboard/state/dispatch_1-1784271895-2_execute`

Status: PASS for a small v0 spec. Do not import LMEB wholesale. Start with a
procedural tool-doc memory retrieval slice that is embedding-auditable,
license-checkable, and useful for agent systems without becoming a full agent
success benchmark.

## Why This Gap Matters

The repo positions itself around practical retrieval gaps for RAG, multimodal
search, and agent systems, and explicitly names future agent memory, tool-doc,
and code-aware retrieval tracks in `README.md`. The current task registry has
MRL stress, cross-lingual retrieval, long-document needle retrieval,
text-image retrieval, autonomous-driving retrieval, and Chinese multimodal
retrieval, but no agent-memory or tool-memory family in `benchmark/tasks/core.yaml`.

Current primary sources make this gap concrete:

- LMEB v4 was last revised on 2026-07-13. Its arXiv abstract frames memory
  embeddings as underexplored by traditional text embedding benchmarks and
  reports 22 datasets / 193 zero-shot retrieval tasks across episodic,
  dialogue, semantic, and procedural memory.
- The LMEB repository says LMEB is officially supported by MTEB and can be run
  through the standard MTEB evaluation framework.
- LMEB's correlation analysis argues that LMEB and traditional MTEB retrieval
  measure different capabilities, with especially weak transfer for
  episodic/dialogue memory and only partial transfer for procedural memory.
- LongMemEval-V2, LoCoMo, and Mem0 memory benchmarks show active interest in
  long-term agent memory, but their main evaluation loops involve memory
  systems, histories, readers, latency, or answer scoring. That is valuable, but
  it is larger than this repo's embedding-task surface.

## Recommended V0 Slice

Recommended slice: `agent_procedural_tool_memory`.

The benchmark should model a memory store of tool affordances and procedures.
Each query is a user or agent task that requires recalling the correct tool
documentation. Each corpus document is a stable tool card: tool name,
description, parameter names, parameter descriptions, parameter types, and safe
non-secret defaults. The evaluator embeds queries and documents, ranks the
documents for each query, and scores retrieval against qrels.

This stays embedding-auditable because:

- there is no tool execution;
- there is no planner, reader LLM, or judge LLM;
- the gold label is a document id, not an answer string;
- failure can be inspected by reading the query, retrieved tool docs, and qrels;
- the same `EvalTask` pattern can run against any text embedding provider.

This should be preferred over an episodic-dialogue v0 for the first slice.
LoCoMo-style dialogue memory is relevant, and LoCoMo includes QA evidence ids,
but it pulls the task toward long conversation handling, temporal reasoning, and
answer prediction. Tool-doc procedural memory is smaller, less privacy-sensitive,
and better aligned with the repo's existing retrieval metrics and hard-negative
patterns.

## Proposed Benchmark Shape

Task id: `agent_procedural_tool_memory`

Display name: `Agent procedural tool memory`

Required modality: `text`

Primary metric: `hard_mrr` or `hard_recall@1`. Prefer `hard_mrr` if queries may
have more than one acceptable tool; prefer `hard_recall@1` for the smallest
single-label smoke.

### Query Set

Start with task instructions that describe the user's goal and constraints,
for example "retrieve a tool that checks whether a URL was archived on a
specific date" or "find the tool that forecasts bacterial population given
initial population, growth rate, elapsed time, and doubling time".

Allowed real-source candidates:

- ToolRet evaluation queries from the official ToolRet repository or its
  Hugging Face datasets, after a row-level source/license audit.
- LMEB procedural-memory source ids only after confirming each upstream
  source's license and provenance. Do not vendor full LMEB as the v0.

Smoke-source candidate:

- A tiny hand-authored fixture with invented tool names and descriptions. This
  is suitable for loader and metric tests, not public leaderboard claims.

### Corpus Documents

Serialize each tool document deterministically:

```text
Tool: <name>
Description: <description>
Parameters:
- <parameter_name> (<type>): <parameter_description>. Default: <default_or_none>.
Source: <source_dataset>/<source_split>/<source_id>
```

Do not include API keys, auth headers, bearer tokens, emails, user ids, private
URLs, or live endpoint credentials. Defaults must be sanitized if they look like
secrets or real personal data.

### Qrels / Ground Truth

Use one qrels row per query-tool pair:

```json
{"query_id": "q001", "doc_id": "tool_weather_history", "relevance": 1, "slice": "single_tool_param_constraints"}
```

Support binary qrels first. Add graded relevance only if upstream labels provide
clear multiple positives.

### Hard Negatives

Hard negatives should be query-specific and auditable:

- sibling tools in the same API family or category;
- tools with overlapping parameter names but wrong semantics;
- lexical/BM25 nearest neighbors that are not labeled positive;
- dense nearest neighbors from a fixed baseline only for offline curation, not
  during provider evaluation;
- near-duplicate tool names with different parameter constraints.

Every hard negative should keep its source id. Reject hard negatives if the
source creates likely false negatives or if two tools are aliases for the same
operation.

### Metrics

Compute the usual query-document similarity matrix and report:

- `recall@1`, `recall@5`, `mrr`, `ndcg@10` over the full corpus;
- `hard_recall@1`, `hard_recall@5`, `hard_mrr`, `hard_ndcg@10` over each query's
  positive plus hard-negative pool;
- slice metrics such as `hard_mrr_single_tool`, `hard_mrr_multi_tool`,
  `hard_mrr_param_constraints`, and `hard_mrr_near_name`.

Store details:

- `n_queries`
- `n_documents`
- `n_qrels`
- `n_hard_negatives`
- `source_datasets`
- `license_audit_status`
- `slices`

### Slices

Minimum useful slices:

- `single_tool`: one correct tool;
- `multi_tool`: more than one relevant tool, if labels support it;
- `parameter_constraints`: query includes typed constraints that must match tool
  parameters;
- `near_name`: wrong tools have similar names;
- `category_collision`: wrong tools solve nearby tasks in the same category;
- `instruction_vs_plain`: optional comparison of query-only vs query plus task
  instruction, matching the LMEB/ToolRet interest in instruction use.

### Leakage, Privacy, And License Checks

Required checks before a public v0:

- Keep the source manifest with exact upstream URL, commit or dataset revision,
  split, license, and terms notes.
- Do not mix ToolRet training data into leaderboard evaluation unless the
  evaluation card clearly labels the task as contaminated-risk.
- De-duplicate normalized tool documents and near-duplicate descriptions across
  train/dev/test candidates.
- Scan tool docs for secret-like strings: API keys, bearer tokens, private
  email addresses, phone numbers, auth headers, and private URLs.
- Exclude private conversation logs and unclear-license agent memory traces.
- If using ToolRet or LMEB-derived rows, verify upstream source licenses per
  subset. The ToolRet code repository is Apache-2.0, but the dataset is built
  from multiple existing datasets, so the v0 should not assume every collected
  row is publishable without the source manifest.
- Preserve source ids in any exported benchmark artifact.

## Smallest Safe Smoke

The first implementation should be a local fixture smoke, not a public score:

- `12` queries;
- `36` tool documents;
- `1` positive qrel per query;
- `3` curated hard negatives per query;
- no provider API calls required by tests;
- no model downloads;
- no Hugging Face upload;
- no benchmark run unless a later dispatch explicitly selects it.

Expected future files:

- `src/mm_embed/tasks/agent_procedural_tool_memory.py`
- `src/mm_embed/data/agent_procedural_tool_memory.py`
- `tests/fixtures/agent_procedural_tool_memory_smoke.jsonl`
- `benchmark/tasks/core.yaml` entry for `agent_procedural_tool_memory`
- `tests/test_agent_procedural_tool_memory.py` or additions to
  `tests/test_benchmark_v2.py`

Smoke acceptance criteria:

- PASS: catalog loads; registry resolves the task; mock or deterministic local
  embeddings can run without network; metrics include `hard_mrr` and
  `hard_recall@1`; details expose counts and source/license status.
- FAILED: loader or qrels shape is ambiguous, metrics cannot be reproduced, or
  hard negatives are too toy-like to catch obvious retrieval errors.
- BLOCKED: candidate real source lacks usable license/provenance metadata,
  contains secrets/private data, or requires large downloads before even
  validating schema.
- ABANDON: the only feasible path requires importing full LMEB, copying private
  memory logs, running a reader/judge LLM, or executing tools rather than
  evaluating embedding retrieval.

## Fit With This Repo

The implementation should follow existing `EvalTask` conventions:

- Create an `EvalTask` subclass with `name = "agent_procedural_tool_memory"`,
  `required_modalities = {ModalityType.TEXT}`, and `run(provider, **kwargs)`.
- Load data through a small data loader, with `use_mock` or `fixture_path` for
  local tests and `source_manifest_path` for later real data.
- Embed all query texts with `task_type="retrieval_query"` and all tool docs with
  `task_type="retrieval_document"` where provider adapters support it.
- Use existing `cosine_similarity_matrix`, `recall_at_k`, `mrr`, and
  `ndcg_at_k` utilities.
- Return `EvalResult` with metrics and details, matching
  `crosslingual_retrieval` and `needle_in_haystack`.
- Add the lazy registry entry in `src/mm_embed/tasks/registry.py`.
- Add the task YAML entry in `benchmark/tasks/core.yaml`, with tags such as
  `[agentic, memory, tool-retrieval, hard-negative, text]`.

Candidate YAML shape:

```yaml
- id: agent_procedural_tool_memory
  display_name: Agent procedural tool memory
  task: agent_procedural_tool_memory
  description: Query-to-tool-document retrieval for procedural memory in agent systems.
  default_kwargs:
    use_mock: false
    max_queries: 200
    hard_mode: true
  required_modalities: [text]
  primary_metric: hard_mrr
  metric_direction: higher
  dataset_version: agent-procedural-tool-memory-v0
  tags: [agentic, memory, tool-retrieval, hard-negative, text]
```

## Follow-Up Implementation Item

Add `tasks/agent-procedural-tool-memory-smoke`: implement the `EvalTask`,
registry/YAML entry, and a 12-query local fixture smoke with deterministic
loader tests. Keep real ToolRet/LMEB data ingestion as a separate follow-up
after license and provenance audit.

## Primary Source Links Used

- LMEB repository: https://github.com/KaLM-Embedding/LMEB
- LMEB arXiv v4: https://arxiv.org/abs/2603.12572
- LMEB HTML paper: https://arxiv.org/html/2603.12572v4
- ToolRet repository: https://github.com/mangopy/tool-retrieval-benchmark
- ToolRet paper: https://arxiv.org/abs/2503.01763
- ToolRet queries dataset: https://huggingface.co/datasets/mangopy/ToolRet-Queries
- ToolRet tools dataset: https://huggingface.co/datasets/mangopy/ToolRet-Tools
- LongMemEval-V2 repository: https://github.com/xiaowu0162/LongMemEval-V2
- LoCoMo repository: https://github.com/snap-research/locomo
- Mem0 memory benchmarks repository: https://github.com/mem0ai/memory-benchmarks

## Decision

Proceed with the procedural tool-doc memory slice as a v0 task draft. It is
small enough for this repo's benchmark harness, directly tied to modern
agent-memory retrieval, and safer than dialogue-log memory for the first
tracked implementation. Keep the public data path gated on license/provenance
checks, and use a hand-authored local fixture for the first code smoke.
