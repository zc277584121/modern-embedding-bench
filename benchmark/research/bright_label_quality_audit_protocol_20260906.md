# BRIGHT Dual-Track Agent Label-Quality Audit Protocol

## Status and scope

This protocol is frozen before any primary judgment for Story
`S-20260905-001`. It governs an agent label audit of the accepted 120-query
economics/psychology pack. It does not create a human gold standard, change
official qrels, run retrieval, select examples from model scores, or open any
publication, leaderboard, or export gate.

The only admissible source inputs are the accepted review plan with SHA256
`93848a826c12acf9cb6fed50d71dd9490be7baebfcea1122dd4fffb4881fdc4e`
and the accepted audit pack with SHA256
`9549525cbd35e0a3c520ad16b5961b3f5e8658ff7cb4b5061e9edd39bc96c956`.
They contain 120 queries, 1,235 gold items, and 2,804 candidate items. Source
files are immutable inputs.

## Judgment unit and evidence boundary

The reviewer must read the full query and the complete supplied passage for
each item. Relevance is judged against the information need expressed by the
query, not lexical overlap, writing quality, source reputation, or presumed
retriever behavior. The audit may use only the supplied text and metadata.
External facts may help interpret ordinary terminology but may not replace
missing passage evidence.

Primary and validator inputs omit baseline method names, scores, ranks,
existing empty review fields, and the other reviewer's labels. Candidate order
is independently salted. A gold item remains visibly a gold-review unit because
gold support is a separate task; a candidate receives an independent relevance
judgment even if the same passage is also a gold item.

## Gold support and ambiguity

`support` has four values:

- `supports`: the passage contains enough direct evidence to satisfy the query's
  central information need or a clearly valid answer component.
- `partially_supports`: the passage provides material evidence but omits a
  necessary condition, scope, relation, or answer component.
- `does_not_support`: the passage is off-target, merely mentions the subject, or
  cannot answer a material part of the information need.
- `uncertain`: the supplied evidence does not permit a reliable choice among the
  preceding labels.

`ambiguity` has three values:

- `unambiguous`: the support relationship is clear under a reasonable reading.
- `ambiguous`: multiple plausible readings, missing context, contradictory
  content, or boundary-dependent scope materially changes the judgment.
- `uncertain`: the reviewer cannot reliably determine whether ambiguity is
  material.

The unsupported-or-ambiguous numerator includes every gold item whose support
is not `supports` or whose ambiguity is not `unambiguous`. This conservative
definition includes partial, uncertain, and abstained gold judgments.

## Candidate relevance, missing positives, and hard negatives

`relevance` has three values:

- `relevant`: the passage independently satisfies the query's central need or a
  clearly necessary answer component.
- `not_relevant`: it does not materially help answer the query.
- `uncertain`: the supplied evidence does not support a reliable binary choice.

`suspected_missing_positive` is `credible` only when the candidate is relevant
and the source audit pack identifies it as unjudged. It is `not_credible` for an
unjudged but non-relevant candidate, `not_applicable` for an existing positive,
and `uncertain` when label status or relevance cannot be resolved. Tooling must
cross-check the source label status when recomputing the final audit.

`likely_hard_negative` is `yes` only for a non-relevant candidate that is
topically plausible enough to challenge a reasonable retriever, such as a near
miss, wrong condition, wrong population, wrong causal direction, or incomplete
answer. Obvious noise is `no`. A relevant passage cannot be a hard negative.

Query-level `credible_missing_positive` and `likely_hard_negative_present`
must agree with their item-level ID lists. Unreviewed candidates outside the
pack remain unjudged and must never be converted into negatives.

## Query-level quality risks

`qrels_completeness` is `complete_enough` only when the supplied gold set covers
the major answer interpretations visible in the reviewed evidence. It is
`incomplete` when at least one credible missing positive or a missing major
answer facet is established. It is `uncertain` when the bounded pack cannot
support either conclusion.

`answerability` is `answerable`, `no_answer`, or `uncertain`. `no_answer` means
the query cannot be answered reliably from the supplied gold and candidate
evidence; it does not claim that no answer exists in the full corpus.

`label_incompleteness_risk` and `sensitive_information_risk` are `none`, `low`,
`moderate`, `high`, or `uncertain`. Risk summarizes consequence and evidence,
not reviewer confidence. Sensitive types are limited to personal/public contact,
network identifier, credential or secret, health, financial, legal, minor,
reidentification, or other. Notes must not copy credentials, secrets, personal
contacts, or other sensitive strings.

## Confidence, uncertainty, and abstention

Confidence is `high`, `medium`, or `low`:

- `high`: the text directly determines the label with no material competing
  interpretation.
- `medium`: the label is more likely than alternatives but requires limited
  inference or domain interpretation.
- `low`: evidence is weak, truncated, contradictory, or materially ambiguous.

`decision_status` is `decided`, `uncertain`, or `abstain`. A decided judgment
cannot contain an `uncertain` label. An uncertain or abstained judgment must use
the relevant `uncertain` label values and low confidence. `abstain` is reserved
for unreadable, unsafe-to-handle, structurally corrupted, or fundamentally
insufficient input and requires a concise rationale. Low-confidence, uncertain,
and abstained units are included in the Validator supplement; none may be
silently coerced to a binary value.

## Provenance

Every gold, candidate, and query judgment embeds complete provenance:
`judgment_origin` (`agent_judged` or `llm_assisted`), agent role, provider,
model, model revision when available, exact session, review round, UTC judgment
time, rubric SHA256, input artifact SHA256, and input-row SHA256. The strict
schema rejects missing provenance or an identity that does not match the
reviewed input. No judgment may be described as `human_judged`.

## Frozen double review and blind isolation

The double-review set is selected before primary review, without model scores:
four queries from each `(track, query_length)` cell, for 24 total. Within a cell,
selection uses ascending SHA256 of the fixed salt, track, query-length stratum,
and canonical query identity. Only one aggregate selection SHA256 may enter the
tracked predeclaration; the selection map remains restricted.

The freeze command creates a source-only 120-query primary input and a separate
source-only 24-query Validator core input. The Validator receives only its core
input and, later, a source-only supplement containing every query with any
primary low-confidence, uncertain, or abstained unit. Primary annotation files
are outside the Validator input allowlist. Validator annotations are validated
and byte-hashed before comparison. Only after that seal may recomputation read
both reviews. This is procedural and artifact-level isolation; it is not claimed
to be an operating-system access-control boundary.

Cohen's kappa is computed only over paired binary candidate relevance labels in
the frozen 24-query core. Uncertain or abstained pairs are counted separately
and keep the agreement gate fail-closed. Supplementary review supports
resolution and adjudication but cannot change the frozen core selection or be
used to tune kappa.

## Gates and deterministic conclusion

The audit fails closed when any of these predeclared conditions holds:

- unsupported-or-ambiguous gold fraction exceeds 0.05;
- queries with a credible missing positive exceed 0.10;
- binary candidate relevance Cohen's kappa is below 0.70;
- required annotations are incomplete, identities drift, kappa is undefined,
  or unresolved/abstained judgments remain.

The first two comparisons use all 1,235 gold items and all 120 query summaries.
The agreement gate remains pending until independent Validator labels exist.
A failed audit is a valid result and does not authorize qrel changes.

## Output layout

Tracked candidate files contain only code, schemas, this protocol, a safe
predeclaration, tests, and eventually an irreversible aggregate:

```text
benchmark/artifacts/bright-label-quality-audit-v0.1/
  predeclaration.json
  predeclaration.json.sha256
  aggregate.json                 # created only after completed reviews
  aggregate.json.sha256          # created only after completed reviews
```

All row-level materials remain gitignored and restricted:

```text
results/bright-label-quality-audit-v0.1/restricted/
  frozen-input-manifest.json
  control/double-review-selection.jsonl
  primary/input.jsonl
  primary/annotations.jsonl
  validator-blind/access-manifest.json
  validator-blind/core-input.jsonl
  validator-blind/core-annotations.jsonl
  validator-blind/core-seal.json
  validator-blind/supplement-input.jsonl
  validator-blind/supplement-annotations.jsonl
  adjudication/decisions.jsonl
```

The safe aggregate must contain no query or document text, canonical ID,
ranking, private filesystem path, secret, reviewer/session/model identity,
free-text note, or per-query/per-item hash that could serve as a reversible map.
