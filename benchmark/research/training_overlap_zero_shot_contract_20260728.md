# Training-overlap and zero-shot contract minispec

Date: 2026-07-28

Local baseline: c5e8e4f99ea6716eccf9c3327c0b96b4221402f7

Conclusion: **PASS**

## Decision

The repository can add an auditable training-overlap and strict zero-shot
contract before exposing future real-data task scores. The contract must fail
closed: missing model metadata, unresolved model lineage, missing task source
identity, ambiguous aliases, stale relationship records, or incomplete review
must produce **unknown**, never a zero-shot claim.

Exact canonical-ID matching is necessary but not sufficient for v0. It catches
direct declarations, but it misses explicit aliases, subsets, translations,
adapted datasets, and inherited base-model training. V0 therefore needs:

1. exact matching over reviewable canonical source IDs;
2. a separate, versioned, human-reviewed source relationship table;
3. transitive positive-overlap inheritance through declared model lineage;
4. no heuristic matching from model names, task names, URLs, or free text; and
5. a strict separation between data overlap and related-task exposure.

The resulting status is interpretation evidence. It must not change a score,
adjust a rank, average across tasks, or create a global model ranking.

## Scope and non-goals

This note specifies registry, result, export, migration, review, and validation
behavior. It does not add product code, edit registries, backfill existing
models, infer training data, run a benchmark, or publish anything.

The contract answers a bounded question: what reviewed evidence exists that a
particular model revision or its declared ancestors saw a particular task's
evaluation source, an adapted form of it, or only a related task?

It does not claim to prove that undisclosed training data does not exist. A
reviewed no-overlap result is always scoped to the declared model revision,
lineage, source registry revision, relationship-table revision, and evidence
review recorded in the assessment.

## Current local behavior: facts

The current repository has no training-overlap contract:

- benchmark/models/core.yaml declares model identity, provider, capabilities,
  source, and notes, but no structured training sources or model lineage.
- benchmark/tasks/core.yaml declares a local task ID and dataset_version, but
  no canonical evaluation-source identity, split identity, or adaptation
  relationship.
- schemas/model.schema.json and schemas/task.schema.json permit extra keys, but
  ModelSpec.from_dict and TaskSpec.from_dict only retain their current declared
  dataclass fields. Adding an unparsed YAML key would therefore not make it
  available to results or exports.
- make_result_record snapshots the current model and task fields, but no
  overlap assessment or relationship-registry revision.
- imported legacy rows use model and task placeholders and dataset_version
  "legacy"; they have no safe basis for a historical zero-shot claim.
- build_leaderboard sorts within task by descending score. It has no overlap
  input and no global ranking.
- the Hugging Face Dataset exporter writes public model/task records and
  appends fixed provenance and operational columns to leaderboard rows. The
  Space uses whitelisted catalog rows, while the Dataset model registry export
  currently serializes the full ModelSpec dataclass. Any future review-only
  field therefore needs an explicit public projection rather than accidental
  asdict exposure.
- existing public language already says scores are task-specific and warns
  against global cross-task comparison. The new contract should preserve that
  boundary.

Relevant local files:

- benchmark/models/core.yaml
- benchmark/tasks/core.yaml
- schemas/model.schema.json
- schemas/task.schema.json
- schemas/result.schema.json
- src/mm_embed/benchmark/registry.py
- src/mm_embed/benchmark/results.py
- src/mm_embed/benchmark/leaderboard.py
- src/mm_embed/hf_publish/export.py
- tests/test_benchmark_v2.py
- README.md

## MTEB behavior: primary-source facts

The upstream snapshot reviewed here is MTEB main commit
f38c9692061e46664e3aa8e50fcf2a3628f1a55d, committed 2026-07-26. The latest
release at review time was 2.18.7, published 2026-07-26 and pointing through
annotated tag f4605b99feb4a4e9e68e04159bfad00bf41f690e to commit
794f50399472059f4b518a5ed47c274459b704f1.

At that main revision:

- ModelMeta.training_datasets is set[str] or None, and ModelMeta.adapted_from
  is a single optional model name.
- MTEB documents training_datasets as MTEB task names used to identify
  contamination and zero-shot generalization.
- ModelMeta.is_zero_shot_on returns None when training data is unspecified,
  true when the reviewed training set has no intersection with the selected
  task names, and false on an intersection. An explicit empty set is therefore
  materially different from None.
- ModelMeta.get_training_datasets first copies the model's declared set, then
  tries to inherit the base model's expanded set through adapted_from, then
  expands every dataset through MTEB's similar-task graph.
- Failure to resolve adapted_from is caught and logged at debug level. The
  child declaration remains usable. Consequently, an explicit empty child set
  plus an unresolved base can look empty even though the inherited training
  data is unknown. This behavior is too permissive for a public reviewed
  zero-shot claim.
- The similar-task graph is generated from task metadata adapted_from and
  superseded_by fields. The collector walks both directions recursively. It is
  therefore a symmetric transitive related-task closure, not a narrowly typed
  proof that two sources contain identical examples.
- The current leaderboard cache helpers preserve the None distinction:
  unknown training data gives unknown zero-shot status. The public UI defaults
  to "Allow All"; "Only Zero-shot" and "Remove Unknown" both exclude unknown,
  while only "Only Zero-shot" excludes known overlap.
- The current documentation asks contributors to annotate training_datasets
  with MTEB task names. That is useful upstream convention, but it does not
  provide this repository with pinned source revisions, alias provenance,
  negative-claim scope, or a safe rule for unresolved lineage.

Two open PRs illustrate why the local contract needs more structure:

- PR 5037, head b2fc16b8ea09c6afb4b1ae7b8f1cdd7ee498ff82,
  proposes training_datasets as an explicit empty set for
  minetta/nemotron-3-embed-8b-legal, cites a separate contamination audit, and
  uses adapted_from to inherit the base model's declarations.
- PR 5039, head 3e84dd627f60db5eaad5dc79c903d2c6396951eb,
  declares all BRIGHT MTEB task IDs for a model whose listed training source
  indirectly reconstructs positives from the BRIGHT dataset, while another
  model declares an empty set based on author confirmation and inherits a
  Qwen3 base through adapted_from.

These PRs are valuable current evidence, but they are open, model-specific,
and expressed in MTEB task-name semantics. They do not settle local canonical
source identity, lineage failure, adaptation direction, stale review, or
public export behavior.

## Contract vocabulary

The product must not collapse the following concepts:

### Canonical evaluation source

A reviewable identity for the actual source used by an evaluation task. It is
not the local task ID and not a display name. It includes an authority,
repository or publication identity, pinned revision when available, config or
subset, split, and transformation contract.

### Exact data overlap

The training claim and evaluation task resolve to the same canonical source or
to sources joined only by reviewed identity-equivalent relationships such as
alias_of or same_examples_as.

### Adapted data overlap

The training and evaluation sources are connected by a reviewed relationship
that establishes material sample lineage, such as subset_of, translated_from,
sampled_from, or reformatted_from. It is overlap evidence even when bytes or
surface language differ.

### Same-task or similar-task exposure

The training claim identifies the same task family or a separately reviewed
related task, but the relationship does not establish shared evaluation
examples. This is interpretation evidence, not proof of data contamination.
It must not be promoted to exact or adapted overlap.

### Declared no overlap

A reviewed conclusion reached only when the model's relevant training
disclosure is complete or a source-specific negative assertion covers the
evaluation source, every declared ancestor is resolved and reviewed, the task
source is resolved, and the relationship table is current.

An empty list by itself is not declared no overlap.

### Unknown

Insufficient evidence. Missing metadata, partial disclosure, unresolved
lineage, ambiguous mappings, stale review, or a source identity mismatch all
produce unknown. Unknown is not zero-shot.

## Public status model

Use separate axes so that related-task evidence is never mislabeled as data
overlap:

~~~text
data_overlap_status:
  exact
  adapted
  declared_none
  unknown

task_training_status:
  same_task
  similar_task
  declared_none
  unknown

zero_shot_status:
  no
  reviewed_yes
  unknown
~~~

Derivation:

1. data_overlap_status exact or adapted makes zero_shot_status no.
2. task_training_status same_task or similar_task also makes the strict
   zero_shot_status no, while retaining the more precise axis values.
3. zero_shot_status reviewed_yes requires both axes to be declared_none and
   requires complete, current review across the full resolved model lineage.
4. Every other combination yields unknown.

The public phrase "Reviewed zero-shot" means only the strict combined status
above. It must not be shortened to an unqualified boolean in exported data or
UI copy.

## Canonical IDs and relationship table

Exact string equality is the only implicit match. IDs are opaque,
case-sensitive, registry-owned identifiers. URLs and display names are
provenance, not identity.

Recommended source ID examples:

~~~yaml
id: hf:xlangai/bright
locator:
  authority: huggingface
  repo_id: xlangai/BRIGHT
  revision: immutable-commit-sha
  config: default

id: local:needle-in-haystack-v1
locator:
  authority: local
  contract: benchmark/tasks/core.yaml
  dataset_version: needle-v1
~~~

The proposed separately reviewed file is
benchmark/training_overlap_relationships.yaml, validated by
schemas/training_overlap_relationships.schema.json.

~~~yaml
schema_version: "1"
revision: "2026-07-28.1"
sources:
  - id: hf:xlangai/bright
    locator:
      authority: huggingface
      repo_id: xlangai/BRIGHT
      revision: immutable-commit-sha
    public_provenance:
      urls:
        - https://huggingface.co/datasets/xlangai/BRIGHT
      reviewed_at: "2026-07-28"
      reviewed_by: maintainer-handle
    review:
      state: approved
      private_notes: null

relationships:
  - id: rel-bright-derived-example
    subject: local:derived-source
    predicate: sampled_from
    object: hf:xlangai/bright
    effect:
      data_overlap: adapted
      task_training: same_task
      transitive: true
    applies_to:
      subject_revision: immutable-commit-sha
      object_revision: immutable-commit-sha
    public_provenance:
      urls:
        - https://example.invalid/pinned-primary-source
      reviewed_at: "2026-07-28"
      reviewed_by: maintainer-handle
    review:
      state: approved
      private_notes: null
~~~

Allowed v0 predicates and behavior:

| Predicate | Direction retained | Data effect | Task effect | Transitive |
| --- | --- | --- | --- | --- |
| alias_of | yes | exact | none | equivalence closure |
| same_examples_as | yes | exact | same_task | equivalence closure |
| subset_of | yes | adapted | same_task | only across approved material-overlap edges |
| sampled_from | yes | adapted | same_task | only across approved material-overlap edges |
| translated_from | yes | adapted | same_task | only across approved material-overlap edges |
| reformatted_from | yes | adapted | same_task | only across approved material-overlap edges |
| same_task_as | yes | none | same_task | false |
| similar_task_to | yes | none | similar_task | false |

The original direction remains auditable even when overlap detection treats a
known non-empty sample-lineage edge as evidence that the two endpoints share
material. Similar-task edges are never made transitive and never affect
data_overlap_status.

V0 must not import MTEB's complete similar-task graph wholesale. Each local
edge needs an explicit semantic type, pinned provenance, review state, and
revision applicability.

## Proposed model registry schema

Add a structured, optional training_data object to ModelSpec. Missing legacy
objects normalize to disclosure unknown. New public model additions should
state the unknown explicitly until reviewed.

~~~yaml
training_data:
  disclosure: unknown
  source_claims: []
  negative_claims: []
  adapted_from: []
  lineage_disclosure: unknown
  public_provenance:
    urls: []
    evidence_revision: null
    reviewed_at: null
    reviewed_by: null
  review:
    state: pending
    private_notes: null
~~~

Reviewed positive example:

~~~yaml
training_data:
  disclosure: partial
  source_claims:
    - source_id: hf:xlangai/bright
      relation: trained_on
      scope: material_samples
  negative_claims: []
  adapted_from:
    - Qwen/Qwen3-Embedding-4B
  lineage_disclosure: complete
  public_provenance:
    urls:
      - https://example.invalid/pinned-model-card-or-paper
    evidence_revision: model-revision-or-document-revision
    reviewed_at: "2026-07-28"
    reviewed_by: maintainer-handle
  review:
    state: approved
    private_notes: null
~~~

Rules:

- disclosure is unknown, partial, or complete.
- Positive claims may establish overlap even when disclosure is partial.
- Absence from a partial list never establishes no overlap.
- A complete empty source_claims list is valid only with approved, pinned
  public provenance and complete lineage disclosure.
- negative_claims are source-specific reviewed assertions. A wildcard negative
  assertion is forbidden in v0.
- adapted_from contains exact model registry IDs and may contain more than one
  parent for merged models. Every parent must resolve before a no-overlap or
  reviewed-zero-shot conclusion.
- Positive overlap propagates from all ancestors. Negative conclusions require
  every ancestor to be reviewed; a child's negative assertion does not erase a
  positive ancestor.
- A lineage cycle is invalid.
- Model-card or repository names are never parsed to infer lineage or data.

## Proposed task registry schema

Add evaluation_sources to TaskSpec. Every future public real-data task must
declare at least one approved evaluation source before a new result can be
published under the contract.

~~~yaml
evaluation_sources:
  disclosure: complete
  sources:
    - source_id: hf:xlangai/bright
      usage: evaluation
      config: default
      split: test
      transformation_id: local-bright-task-contract-v1
  public_provenance:
    urls:
      - https://example.invalid/pinned-task-source
    evidence_revision: immutable-revision
    reviewed_at: "2026-07-28"
    reviewed_by: maintainer-handle
  review:
    state: approved
    private_notes: null
~~~

Rules:

- task.id remains the local runnable task contract.
- dataset_version remains the local transformation/version label.
- source_id identifies the upstream or local sample universe.
- config, split, and transformation_id identify the evaluated slice.
- A fixture-only or unpublished task may remain explicitly unknown, but it
  cannot be promoted to a public real-data task without an approved source.
- Changing dataset_version, source revision, split, or transformation ID
  invalidates assessments created against the prior tuple.

## Proposed result schema

Every new public real-data result must snapshot an assessment object. Unknown
is a valid assessment; a missing object is not valid for a post-contract
public row.

~~~json
{
  "training_overlap": {
    "schema_version": "1",
    "relationship_registry_revision": "2026-07-28.1",
    "relationship_registry_sha256": "hex-digest",
    "model_revision": "declared-model-revision",
    "model_training_evidence_revision": "public-evidence-revision",
    "task_dataset_version": "task-dataset-version",
    "task_source_evidence_revision": "public-evidence-revision",
    "data_overlap_status": "unknown",
    "task_training_status": "unknown",
    "zero_shot_status": "unknown",
    "matched_model_ids": [],
    "matched_training_source_ids": [],
    "matched_evaluation_source_ids": [],
    "relationship_ids": [],
    "reason_codes": [
      "model_training_disclosure_unknown"
    ],
    "assessed_at": "2026-07-28T00:00:00Z"
  }
}
~~~

Only registry-owned IDs, public evidence revisions, relationship IDs, reason
codes, and statuses belong in the public result snapshot. Raw reviewer text,
private notes, environment values, local filesystem paths, full model-card
copies, and unreviewed free text do not.

Recommended reason codes include:

- exact_source_match
- approved_alias_match
- approved_adapted_source_match
- same_task_exposure
- similar_task_exposure
- complete_reviewed_non_overlap
- model_training_disclosure_unknown
- task_source_unknown
- unresolved_model_lineage
- ambiguous_source_mapping
- stale_relationship
- stale_model_evidence
- stale_task_evidence
- conflicting_claims
- legacy_missing_contract

## Assessment algorithm

For one model revision and one task result:

1. Load the task's approved evaluation source IDs and exact slice identity.
2. Load the model's positive claims, negative claims, disclosure status, and
   all declared adapted_from ancestors.
3. Resolve the full model lineage. Detect missing parents and cycles.
4. Expand only approved source relations from the pinned relationship-table
   revision:
   - alias and same-example equivalence for exact overlap;
   - approved material sample-lineage edges for adapted overlap;
   - same-task and similar-task edges only for task exposure.
5. Evaluate every model-lineage positive claim against every task source.
6. Choose the strongest positive data status: exact before adapted. Record all
   supporting IDs and relationship paths.
7. Independently compute same-task or similar-task exposure.
8. If no positive match exists, emit declared_none only when disclosure,
   lineage, negative-claim scope, task source, review state, and revision
   freshness are all sufficient. Otherwise emit unknown.
9. Derive zero_shot_status using the strict rules above.
10. Snapshot the assessment and relationship-table digest into the result.

Positive evidence dominates uncertainty for the matched source: known overlap
must still be shown even if another lineage branch is unresolved. Uncertainty
dominates negative claims: unresolved lineage prevents declared_none and
reviewed_yes.

No assessment may be recomputed implicitly during public export from today's
registry for an old result. The result snapshot is historical evidence. A
deliberate migration may create a new enriched artifact with its own
assessment revision, but must not silently rewrite the original record.

## Validation invariants

Registry loading or public export must fail closed on:

1. duplicate source, relationship, model, or task IDs;
2. a referenced source, relationship endpoint, model parent, or task source
   that does not resolve;
3. alias equivalence conflicts or an alias that resolves to multiple canonical
   sources;
4. model-lineage cycles;
5. contradictory positive and negative claims for the same reviewed source
   scope;
6. complete empty declarations without approved pinned provenance;
7. approved relationships without public provenance, reviewer, review date,
   applicability revisions, or effect semantics;
8. relationship applicability revisions that do not match the current model
   or task declaration;
9. a post-contract public real-data result without training_overlap;
10. an unsupported status or reason code;
11. any public projection containing review.private_notes or another
    non-whitelisted review field.

Additional deterministic rules:

- Canonical IDs are compared exactly. Case folding, punctuation stripping,
  basename matching, fuzzy matching, and URL guessing are forbidden.
- Similar-task relationships are never transitive in v0.
- Alias equivalence may be transitive, but an equivalence component must have
  one canonical representative.
- Adaptation traversal is allowed only through edges explicitly marked as
  material overlap and transitive.
- Every assessment carries the relationship registry revision and digest.
- Review freshness is revision-based first. A calendar review_due_at may be
  added, but time alone must not substitute for a pinned source revision.

## Public Dataset, Space, and UI contract

Append, rather than insert or reorder, leaderboard CSV fields:

~~~text
data_overlap_status
task_training_status
zero_shot_status
overlap_reason_codes
overlap_relationship_registry_revision
~~~

The original task-specific score and ordering remain unchanged. Existing rank
or latest-row behavior remains unchanged. No score penalty, boost, replacement
metric, cross-task normalization, or global rank is allowed.

Dataset and Space exports must use explicit public projections:

- Model catalog may show disclosure state and public evidence links, but never
  private notes, raw provider configuration, or reviewer-only text.
- Task catalog may show canonical public source IDs, source revisions, and
  public evidence links.
- Result and leaderboard rows show the frozen statuses and short reason codes.
- The Space may filter by the three status axes, but filtering only changes
  visibility, not rank or score.
- The filter label must be "Reviewed zero-shot only", not "Zero-shot only".
- Unknown rows remain visible by default.

Required warning copy:

> Training-overlap status is interpretation evidence for this model revision,
> task source, and reviewed relationship-table revision. Unknown means
> unreported, incomplete, unresolved, or stale; it does not mean zero-shot.
> Status does not change the task score or ranking.

Status labels:

- Known exact training overlap
- Known adapted training overlap
- Same-task training exposure
- Similar-task training exposure
- Reviewed no declared overlap
- Unknown - no zero-shot claim

## Migration behavior

### Existing registry entries

Missing model training_data and task evaluation_sources load as explicit
unknown for compatibility. They do not receive zero-shot claims.

A later metadata-curation item may add reviewed declarations. That work must
cite model revisions and task source revisions and must not infer from names or
cards without review.

### Existing v2 results

Pre-contract records with no training_overlap remain valid historical score
records. Export represents them as:

~~~text
data_overlap_status = unknown
task_training_status = unknown
zero_shot_status = unknown
reason_codes = legacy_missing_contract
~~~

The exporter must not use the current registry to backfill those rows
silently.

### Imported legacy rows

Imported legacy rows remain unknown even if their display model name resembles
a current registry ID. Only an explicit, reviewed migration that binds the
historical model revision and task source may create an enriched copy.

### Future public rows

After the contract version is enabled, every new public real-data row must
carry the assessment object. Unknown is allowed and must be warned; omission is
not allowed.

## Required failure and false-claim tests

The later product patch must include at least these tests:

1. **unknown is not zero-shot**: missing training_data yields all unknown
   public statuses and is excluded from "Reviewed zero-shot only".
2. **explicit empty is reviewed, not magical**: empty claims without complete
   disclosure and approved provenance fail validation; a complete reviewed
   empty declaration with complete lineage may yield declared_none.
3. **ambiguous alias**: one alias connected to two canonical representatives
   fails registry validation and cannot publish an assessment.
4. **undeclared base lineage**: a child with empty claims and an unresolved
   adapted_from parent yields unknown, never reviewed_yes.
5. **positive ancestor propagation**: a base-model exact or adapted overlap
   makes the descendant non-zero-shot.
6. **multi-parent lineage**: one overlapping parent dominates; one unresolved
   parent prevents a negative conclusion.
7. **transitive material overlap**: approved A sampled_from B and B
   translated_from C detects adapted overlap between A and C.
8. **similar-task non-transitivity**: A similar_task_to B and B
   similar_task_to C does not infer A similar_task_to C.
9. **stale mapping**: a changed task dataset_version, source revision, model
   evidence revision, or applicability revision yields stale_relationship and
   unknown.
10. **conflicting claims**: trained_on and not_trained_on for the same
    equivalence component fail validation.
11. **private-note leakage**: a unique sentinel in review.private_notes is
    absent from Dataset models.jsonl, tasks.jsonl, results JSONL, leaderboard
    CSV, generated Space source, bundled catalog files, README, and export
    manifest.
12. **legacy rows**: missing contract fields export as unknown with
    legacy_missing_contract, while score and task-specific ordering remain
    byte-for-byte equivalent for the original leaderboard columns.
13. **score-order preservation**: changing overlap statuses does not change
    row score, within-task sort, latest marker, duplicate count, or task/model
    run rank.
14. **no heuristic matching**: differently cased IDs, similar basenames, model
    names containing dataset strings, and free-text notes do not match.
15. **post-contract publication gate**: a new public real-data row with no
    assessment is rejected; the same row with an explicit unknown assessment
    is allowed with warning.
16. **relationship digest**: changing an approved relationship changes the
    digest and invalidates an assessment that claims the older revision.

## Smallest bounded follow-up implementation

The smallest coherent later patch is:

1. add schemas/training_overlap_relationships.schema.json and an initially
   small benchmark/training_overlap_relationships.yaml;
2. extend schemas/model.schema.json, schemas/task.schema.json, and
   schemas/result.schema.json;
3. extend ModelSpec and TaskSpec parsing with structured declarations that
   default to unknown;
4. add one pure assessment module under src/mm_embed/benchmark that performs
   exact, relationship, and lineage evaluation without network access;
5. snapshot the assessment in make_result_record;
6. append the five public leaderboard fields and add safe Dataset/Space
   projections plus warning/filter copy;
7. add focused tests in tests/test_benchmark_v2.py for the sixteen cases above,
   using invented local identities only.

The first patch does not need to curate every real model. Existing declarations
may remain explicit unknown. It does need one invented exact case, one adapted
case, one related-task case, one reviewed-empty case, and one unresolved-lineage
case in test fixtures so the public behavior is executable and reviewable.

No provider, task runner, benchmark dataset, model download, Hugging Face
operation, dependency change, or score migration is required for this slice.

## Unresolved assumptions

These assumptions do not block the contract, but each real metadata curation
must resolve them:

- Whether a model vendor's disclosure is complete enough for a negative claim.
- Whether a particular derivative retains material evaluation examples or is
  only task-similar.
- Whether a model's declared base lineage is complete for merged, distilled,
  routed, or multi-stage systems.
- Whether a source revision is immutable and whether a task transformation
  changes the evaluated sample universe.
- Whether related-task training should be shown as strict non-zero-shot or as a
  separate qualified category in another product. This minispec chooses the
  conservative strict definition: reviewed_yes requires no same/similar-task
  exposure.

The safe default for every unresolved assumption is unknown.

## Primary sources

All upstream facts were refreshed anonymously from official MTEB GitHub
sources on 2026-07-28.

- MTEB repository and reviewed main commit:
  https://github.com/embeddings-benchmark/mteb
  https://github.com/embeddings-benchmark/mteb/commit/f38c9692061e46664e3aa8e50fcf2a3628f1a55d
- ModelMeta fields, zero-shot logic, lineage inheritance, and similar-task
  expansion:
  https://github.com/embeddings-benchmark/mteb/blob/f38c9692061e46664e3aa8e50fcf2a3628f1a55d/mteb/models/model_meta.py
- Task metadata dataset identity and adapted_from:
  https://github.com/embeddings-benchmark/mteb/blob/f38c9692061e46664e3aa8e50fcf2a3628f1a55d/mteb/abstasks/task_metadata.py
- Construction of the adapted/superseded similar-task graph:
  https://github.com/embeddings-benchmark/mteb/blob/f38c9692061e46664e3aa8e50fcf2a3628f1a55d/mteb/get_tasks.py
- Cached leaderboard zero-shot behavior:
  https://github.com/embeddings-benchmark/mteb/blob/f38c9692061e46664e3aa8e50fcf2a3628f1a55d/mteb/benchmarks/_create_table.py
- Leaderboard filter behavior and UI choices:
  https://github.com/embeddings-benchmark/mteb/blob/f38c9692061e46664e3aa8e50fcf2a3628f1a55d/mteb/leaderboard/app.py
- Official contamination annotation documentation:
  https://github.com/embeddings-benchmark/mteb/blob/f38c9692061e46664e3aa8e50fcf2a3628f1a55d/docs/get_started/usage/leaderboard.md
- Release 2.18.7:
  https://github.com/embeddings-benchmark/mteb/releases/tag/2.18.7
  https://github.com/embeddings-benchmark/mteb/commit/794f50399472059f4b518a5ed47c274459b704f1
- Open PR 5037 and pinned head:
  https://github.com/embeddings-benchmark/mteb/pull/5037
  https://github.com/embeddings-benchmark/mteb/commit/b2fc16b8ea09c6afb4b1ae7b8f1cdd7ee498ff82
- Open PR 5039 and pinned head:
  https://github.com/embeddings-benchmark/mteb/pull/5039
  https://github.com/embeddings-benchmark/mteb/commit/3e84dd627f60db5eaad5dc79c903d2c6396951eb

## Resource and safety ledger

- Network use was restricted to anonymous, read-only official GitHub API,
  raw-source, commit, release, and PR endpoints for the MTEB repository.
- AnySearch was not called.
- Response-body accounting was 2,068,324 bytes. The ledger uses curl-reported
  compressed body sizes where captured and decoded temporary-file sizes
  otherwise, so it is a conservative accounting figure rather than
  packet-level transfer measurement. It is below the 15 MiB cap.
- The largest dedicated temporary directory observed was 1,405,541 bytes,
  below the 25 MiB cap.
- Every dedicated temporary directory was deleted on command exit. Retained
  third-party payload: 0 bytes.
- No credentials were supplied, printed, stored, or reused.
- No provider API, model inference, benchmark, model or dataset download,
  dependency operation, Hugging Face operation, public upload, or public score
  was performed.
