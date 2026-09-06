# BRIGHT Qrels Remediation Candidate v0.1

## Status

This is a research-only, restricted, inactive, default-deny candidate for Story
`S-20260906-001`, repair round `2`. It does not modify official qrels, accepted evidence,
registries, runners, exporters, leaderboards, or publication configuration.

The restricted proposal is bound by SHA-256 `d4afe0a4848df0a745d9f06fa01e8679fbb8443dee35158db90bc013b6daf8a1`.
It contains complete row-level evidence and provenance but remains gitignored
and no-publish. This tracked report contains no source text, canonical identity,
ranking, supporting span, proposed qrels row, reviewer identity, private path,
or reversible mapping.

## Control contract repair history

Repair round 2 changed only the supporting-span structural contract. At
`2026-09-06T20:30:00Z`, the proposal schema identity
changed from `de3f42104e4325b86ff628ae5928a36114d0271f6ce19bce7aae20e5d9e8b389`
to `cdf27a0e6d7dabe93730925d295dd69ff2381b1074076ac2a35546c1f752666b` by adding
mandatory `start_offset` and `end_offset`. The control identity consequently
changed from `d05de5d23f5cdff6a5dd12a38f6e856fecb634239641ef0f689e39b62e3dd9c2` to
`ca007c6810649385ba384e4b7830871a45f932ed402b13fae257db192affbee1`. Business rules
remain byte-identical under rules identity
`4dacff181fff81caa87aa7ee40c7c165fe05e227436d53f97410bb6837bbb52c`.

## Coverage and disposition

- Queries: 120 / 120
- Gold items: 1235 / 1,235
- Frozen candidates: 2804 / 2,804
- Total dispositions: 4159
- Original positive rows in reviewed scope: 1235
- Proposed positive rows: 0
- Quarantined or uncertain positive rows: 1235
- Added positives: 0
- Synthesized negatives: 0

Accepted item-level evidence is preserved independently from the contract-level
semantic stop:

- Gold support: 201 supports,
  290 partially supports, and
  744 does not support.
- Economics gold support: 131 /
  144 /
  390.
- Psychology gold support: 70 /
  146 /
  354.
- Gold ambiguity: 10 ambiguous and
  1225 unambiguous.
- Credible missing-positive candidates: economics 53
  across 21 queries; psychology 44
  across 21 queries.
- Unresolved accepted query judgments: 1
  total, economics 1,
  psychology 0.
- Accepted adjudications represented in the evidence ledger: 241.

Quarantine means omission from this inactive reviewed-scope candidate. It is not
a relevance-zero judgment and does not alter official qrels.

## Three-state verdict

- Economics: `task_semantics_mismatch`; 60 reviewed queries stopped.
- Psychology: `task_semantics_mismatch`; 60 reviewed queries stopped.
- Overall: `task_semantics_mismatch`.

The source task defines positives as author-curated linked passages associated
with naturally occurring questions. The accepted audit assessed whether a
passage directly supports the central information need or answer. Frozen local
materials do not specify how those standards map. Therefore support failure is
not converted into non-relevance, and accepted candidate relevance is not
converted into a positive qrels row.

No reviewed query is eligible for a bounded repaired track until the task
semantics are resolved. The candidate cannot support corrected relevance gold,
a full-track repair, model-quality recomparison, or a public leaderboard.

## Score-independent sensitivity boundary

- Under the current contracts, the result remains `task_semantics_mismatch`.
- If linked-source relevance is explicitly adopted, existing support judgments
  are insufficient to validate or negate those labels.
- If direct answer support is explicitly adopted, independent item review and
  direct-span evidence are still required for membership changes.

No model score, rank position, method outcome, or ranking file was consumed by
candidate generation.

## Next minimum action

Freeze one benchmark task definition: linked-source/citation relevance, direct
answer-support relevance, or explicitly separate tracks. Then independently
review the same frozen 60-query sample per track under that definition before
activating any qrels. This is a product/research semantics decision because it
changes benchmark identity and supported claims.

The result is not eligible as an input gate for `S-20260814-007`: the frozen
overall state is not `insufficient_existing_evidence`, and that Story was not
modified or promoted.
