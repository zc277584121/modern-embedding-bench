# BRIGHT Label-Quality Audit Completion Contract

This document describes the deterministic completion checks for Story
`S-20260905-001`. It supplements, but does not modify, the rubric frozen before
primary review. The review plan, audit pack, rubric, primary annotations,
Validator core and supplement annotations, Validator seal, and adjudication
history are immutable evidence.

## Evidence graph

Completion accepts every artifact as an explicit input. It validates the fixed
review-plan and audit-pack hashes and then reconstructs all derived inputs:

- the complete 120-query primary input from the audit pack;
- the 24-query double-review selection from the review plan, including each
  query-length stratum and source-plan-row hash;
- the Validator core input from that exact selection;
- the Validator supplement input from every primary query containing a low
  confidence, uncertain, or abstained gold, candidate, or query judgment;
- the frozen manifest, Validator access manifest, and core seal against the
  files they bind.

The completed predeclaration binds the current hashes of every schema and every
restricted evidence artifact. Any source-content, selection, annotation, seal,
schema, or predeclaration drift makes validation fail closed.

## Adjudication validation

Each adjudication row is validated with the strict adjudication schema. The
validator deterministically reconstructs the trigger set from disagreements
and primary low-confidence or unresolved judgments. A decision must preserve
both complete prior judgments and their provenance, identify the exact trigger
and conflicting fields, bind the source and annotation rows and artifacts, bind
the Validator core seal, and include complete adjudication provenance.

Unknown, missing, duplicate, untriggered, or omitted decisions are rejected.
Validated adjudications are applied only to an in-memory copy of the primary
rows for recomputation; no primary, Validator, or adjudication history is
overwritten.

## Deterministic commands

The `scripts/bright_label_audit.py` entry point provides three completion
commands:

- `bind-completed-predeclaration` validates the full evidence graph before
  writing the safe completed predeclaration and its SHA256 sidecar.
- `validate-completion` validates the same graph and reports only safe coverage,
  adjudication, unresolved, and outcome counts.
- `recompute` validates the graph, applies the validated decisions in memory,
  and writes the safe aggregate and its SHA256 sidecar.

All three commands require explicit paths for the review plan, audit pack,
rubric, primary input and annotations, Validator core input and annotations,
Validator supplement input and annotations, selection, frozen manifest, access
manifest, core seal, and adjudication decisions.

## Fail-closed output

Unresolved accounting traverses every gold, candidate, and query judgment in
the primary, Validator core, Validator supplement, and adjudication histories.
The effective-final count is reported separately because it is derived from
those histories. Any unresolved history, excluded kappa pair, undefined kappa,
identity failure, or threshold failure keeps the overall outcome fail closed.

The aggregate contains only irreversible counts and artifact-level hashes. It
contains no query or document text, canonical identifiers, ranks, private
paths, free-text notes, reviewer/model/session identities, secrets, or
per-row hashes. Publication, leaderboard, and export remain closed, and the
official qrels remain unchanged.
