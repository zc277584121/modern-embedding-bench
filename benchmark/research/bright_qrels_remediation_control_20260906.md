# BRIGHT Qrels Remediation Control Freeze v0.1

## Status and scope

This document freezes the control plane for Story `S-20260906-001` before any
remediation disposition is produced or any model score, ranking advantage, or
method outcome is used for selection. The control ID is
`bright-qrels-remediation-control-v0.1`. It applies only to the accepted,
restricted 120-query economics/psychology audit and existing local evidence.

The future proposal ID is `bright-qrels-remediation-proposal-v0.1`. Every
proposal is restricted, versioned, inactive, no-publish, and default-deny. It
must live under `results/bright-qrels-remediation-v0.1/restricted/`; that path is
gitignored. This control does not create a proposal or authorize activation.

## Validation-contract repair history

Repair round 2 updated only the structural validation contract at
`2026-09-06T20:30:00Z`. The proposal schema identity changed from
`de3f42104e4325b86ff628ae5928a36114d0271f6ce19bce7aae20e5d9e8b389` to
`cdf27a0e6d7dabe93730925d295dd69ff2381b1074076ac2a35546c1f752666b`
to require explicit supporting-span start and end offsets. The control artifact
identity consequently changed from
`d05de5d23f5cdff6a5dd12a38f6e856fecb634239641ef0f689e39b62e3dd9c2` to
`ca007c6810649385ba384e4b7830871a45f932ed402b13fae257db192affbee1`.
The remediation taxonomy, dispositions, evidence levels, stop thresholds, and
three-state precedence remain byte-identical under rules identity
`4dacff181fff81caa87aa7ee40c7c165fe05e227436d53f97410bb6837bbb52c`.

The frozen restricted root is interpreted lexically relative to the repository.
The root, every parent component, every restricted artifact, every sidecar, and
the manifest must be non-symlink filesystem objects. Validation also enforces
no-follow access and realpath containment after the lexical and `lstat` checks.

## Semantic boundary

The retrieval task and the audit rubric answer different questions. Topical or
task relevance, factual support for an answer, and direct answer-bearing
evidence are not interchangeable. In particular:

- `supports_query=false` does not imply `relevance=false`;
- partial support does not mechanically imply an invalid positive;
- an ambiguous, uncertain, or unjudged item is never a negative;
- a retrieved candidate is not relevant merely because a method ranked it;
- an official positive is not retained merely because a method benefited from
  it;
- a task/rubric mismatch is a track-contract problem, not an item-level qrels
  defect to be silently repaired.

The only permitted evidence is already-local corpus text, official qrels,
frozen audit inputs, accepted annotations and adjudications, existing
rankings/manifests used only to locate frozen candidates, and their accepted
identities. Model scores, rank positions, and method wins may never determine a
disposition or supporting-evidence grade.

## Remediation taxonomy

Every query, gold item, and frozen candidate must receive at least one taxonomy
label in a future complete proposal:

- `task_semantics_mismatch`: the retrieval task and the support-oriented rubric
  cannot be reconciled from the frozen task contract;
- `gold_support_defect`: an official positive lacks the evidence required by the
  resolved task semantics;
- `gold_ambiguity`: the official positive or its relationship to the query is
  materially ambiguous;
- `credible_missing_positive`: an existing corpus document has direct evidence
  for positive inclusion but is absent from official qrels;
- `candidate_pool_coverage_gap`: the frozen candidate pool cannot establish
  adequate corpus coverage;
- `insufficient_existing_evidence`: existing local evidence cannot support a
  reliable membership decision;
- `no_qrels_defect`: existing evidence supports no qrels change. This label is
  mutually exclusive with every defect label.

## Dispositions

The exhaustive disposition enumeration is:

- query: `retain_query`, `stop_query_task_semantics_mismatch`,
  `stop_query_insufficient_existing_evidence`, or `not_applicable`;
- official gold: `retain_official_positive`,
  `quarantine_official_positive`, `downgrade_to_uncertain`,
  `abstain_uncertain`, or `not_applicable`;
- frozen candidate: `propose_add_positive`, `no_change_unjudged`,
  `downgrade_to_uncertain`, `abstain_uncertain`, or `not_applicable`.

Quarantine or exclusion means omission from the inactive bounded proposal. It
does not create a relevance-zero judgment. Proposed qrels rows may contain only
positive integer grades and only correspond to `retain_official_positive` or
`propose_add_positive` dispositions.

## Evidence levels and provenance

Evidence levels are ordered for eligibility, not for model comparison:

1. `direct_supporting_span`: an exact existing span directly supports the
   resolved task semantics;
2. `document_level_support`: the existing document supports the semantics but a
   stable exact span is not sufficient by itself;
3. `contextual_or_topical_only`: evidence is related but not answer-bearing;
4. `conflicting`: accepted evidence sources materially disagree;
5. `insufficient`: evidence is absent or cannot resolve the decision.

Only `direct_supporting_span` can justify `propose_add_positive`, and at least
one restricted supporting span is mandatory. Each disposition requires the
judgment origin, agent role, exact session, round, timestamp, rubric identity,
input artifact identity, input-row identity, and accepted source-annotation
identity. Missing or mismatched provenance is a hard failure.

An ambiguous or uncertain source judgment must remain
`downgrade_to_uncertain` or `abstain_uncertain`. An unjudged candidate must
remain `no_change_unjudged` unless direct existing evidence supports a positive
addition. No rule permits negative synthesis.

## Stop thresholds

All thresholds are zero-tolerance because this proposal changes benchmark
membership:

- one missing/invalid provenance record stops validation;
- one unknown identity, duplicate identity, or conflicting disposition stops
  validation;
- one source or baseline identity mismatch stops validation;
- one default-active, publishable, exportable, leaderboard-enabled, or
  registry-integrated setting stops validation;
- one negative grade or automatic conversion of unjudged, ambiguous, or
  uncertain evidence to negative stops validation;
- one unresolved membership-changing item stops its query;
- one stopped or incomplete query prevents a track from being classified as
  remediable with existing data.

Full production must still cover exactly 120 queries, 1,235 gold items, and
2,804 frozen candidate items. Early stopping is a conclusion rule, not
permission to omit required dispositions.

## Three-state derivation

The conclusion is derived without model performance, using this fixed
precedence for each query, then each track, then the overall proposal:

1. `task_semantics_mismatch` if any covered unit has that taxonomy or the query
   uses `stop_query_task_semantics_mismatch`;
2. otherwise `insufficient_existing_evidence` if any required disposition is
   missing, uncertain, abstained, evidence-conflicting/insufficient, affected by
   a candidate-pool gap, or the query uses
   `stop_query_insufficient_existing_evidence`;
3. otherwise `remediable_with_existing_data`.

A track inherits the highest-precedence state of any query. The overall result
inherits the highest-precedence state of either track. No caller may override
the derived state.

## Publication and activation controls

The proposal must declare `status: inactive`, `default_activation: false`,
`registry_integration: false`, `runner_integration: false`,
`public_export_allowed: false`, `leaderboard_allowed: false`, and
`publish_allowed: false`. Validation is deny-by-default: malformed or
incomplete inputs return no usable proposal and no conclusion. Official qrels,
accepted audit evidence, accepted raw/results, benchmark identities, and
publication configuration are immutable inputs.

## Deterministic identity convention

Directory identities use SHA-256 over the byte stream produced by sorting all
regular relative paths lexicographically and emitting one line per file in the
form `<file_sha256><two spaces>./<relative_path>\n`. The control artifact records
only counts, byte identities, rules, and closed-gate metadata; it contains no
query text, document text, canonical ID, ranking, supporting span, proposed
qrels row, reviewer identity, or absolute private path.
