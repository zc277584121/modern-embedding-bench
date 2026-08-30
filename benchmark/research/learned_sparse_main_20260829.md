# BRIGHT Learned-Sparse Main Comparison

This package is research-only. The public gate is closed. It contains aggregate statistics and opaque case identifiers only.

Results are reported independently for economics and psychology. There is no cross-track micro-average or overall model ranking.

## Economics leaderboard

| Model | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| opensearch-doc-v2-distill | 0.281309 | 0.231000 | 0.290804 | 0.309436 | 0.688349 |
| bge-m3 | 0.280627 | 0.215270 | 0.325778 | 0.318493 | 0.654319 |
| granite-30m-sparse | 0.259533 | 0.207093 | 0.293296 | 0.295205 | 0.676714 |
| splade-tiny | 0.252678 | 0.193169 | 0.299391 | 0.268240 | 0.524069 |
| opensearch-multilingual | 0.249693 | 0.194916 | 0.294371 | 0.287213 | 0.593892 |
| opensearch-doc-v2-mini | 0.248416 | 0.196430 | 0.287201 | 0.270160 | 0.542179 |
| opensearch-doc-v3 | 0.244365 | 0.195677 | 0.282039 | 0.263687 | 0.580278 |

Metric-specific orders are stored in `summary.json`; the table order uses nDCG@10 only within this track.

## Psychology leaderboard

| Model | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| bge-m3 | 0.322664 | 0.277383 | 0.391686 | 0.335317 | 0.610826 |
| granite-30m-sparse | 0.296319 | 0.256079 | 0.327691 | 0.326074 | 0.625040 |
| opensearch-multilingual | 0.288993 | 0.241773 | 0.358306 | 0.304699 | 0.613330 |
| opensearch-doc-v2-distill | 0.287598 | 0.260424 | 0.333031 | 0.305143 | 0.644500 |
| splade-tiny | 0.256993 | 0.208106 | 0.300122 | 0.293792 | 0.554238 |
| opensearch-doc-v2-mini | 0.250751 | 0.201311 | 0.304495 | 0.275569 | 0.569914 |
| opensearch-doc-v3 | 0.238630 | 0.196372 | 0.282783 | 0.270181 | 0.579766 |

Metric-specific orders are stored in `summary.json`; the table order uses nDCG@10 only within this track.

## Paired uncertainty

All 21 model pairs per track and all five metrics use 10,000 query-aligned samples with seed 20260826.
A win statement appears only when the paired percentile interval excludes zero; every other result is tie/uncertain.

Significant comparisons are fully enumerated in `pairwise.json`. Representative package-level findings:
- economics: bge-m3 exceeds opensearch-doc-v2-mini on recall@100; paired 95% CI excludes zero.
- economics: bge-m3 exceeds opensearch-doc-v3 on recall@100; paired 95% CI excludes zero.
- economics: bge-m3 exceeds opensearch-multilingual on recall@100; paired 95% CI excludes zero.
- economics: bge-m3 exceeds splade-tiny on recall@100; paired 95% CI excludes zero.
- economics: granite-30m-sparse exceeds opensearch-doc-v2-mini on recall@100; paired 95% CI excludes zero.
- economics: granite-30m-sparse exceeds opensearch-doc-v3 on recall@100; paired 95% CI excludes zero.
- economics: granite-30m-sparse exceeds opensearch-multilingual on recall@100; paired 95% CI excludes zero.
- economics: granite-30m-sparse exceeds splade-tiny on recall@100; paired 95% CI excludes zero.
- economics: opensearch-doc-v2-distill exceeds opensearch-doc-v2-mini on map@100; paired 95% CI excludes zero.
- economics: opensearch-doc-v2-distill exceeds opensearch-doc-v2-mini on recall@100; paired 95% CI excludes zero.
- economics: opensearch-doc-v2-distill exceeds opensearch-doc-v3 on ndcg@10; paired 95% CI excludes zero.
- economics: opensearch-doc-v2-distill exceeds opensearch-doc-v3 on map@100; paired 95% CI excludes zero.

## Predeclared slices

### Economics

- `query_length` — <=64: n=15, 65-128: n=46, >128: n=42
- `positive_lexical_overlap` — 0: n=0, (0,0.25]: n=31, (0.25,0.50]: n=64, >0.50: n=8
- `positive_document_length` — <=128: n=18, 129-512: n=69, >512: n=16
- `positive_qrel_density` — 1: n=35, 2-4: n=30, >=5: n=38
### Psychology

- `query_length` — <=64: n=17, 65-128: n=48, >128: n=36
- `positive_lexical_overlap` — 0: n=0, (0,0.25]: n=30, (0.25,0.50]: n=60, >0.50: n=11
- `positive_document_length` — <=128: n=16, 129-512: n=66, >512: n=19
- `positive_qrel_density` — 1: n=35, 2-4: n=32, >=5: n=34

Bins with n=1–9 have point estimates only; n=0 is NA. Bins are never merged, and thresholds are unchanged from the active predeclaration.

## Opaque case review

Exactly six disagreement and six failure cases are selected per track. Tracked files contain only opaque case IDs, rule labels, mechanism-level summaries, and non-reversible local evidence hashes.
- `economics-disagreement-01`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `economics-disagreement-02`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `economics-disagreement-03`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `economics-disagreement-04`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `economics-disagreement-05`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `economics-disagreement-06`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `economics-failure-01`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `economics-failure-02`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `economics-failure-03`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `economics-failure-04`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `economics-failure-05`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `economics-failure-06`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `psychology-disagreement-01`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `psychology-disagreement-02`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `psychology-disagreement-03`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `psychology-disagreement-04`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `psychology-disagreement-05`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `psychology-disagreement-06`: The seven frozen sparse mechanisms disagree materially on positive-document placement.
- `psychology-failure-01`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `psychology-failure-02`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `psychology-failure-03`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `psychology-failure-04`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `psychology-failure-05`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.
- `psychology-failure-06`: The seven frozen sparse mechanisms share a low-effectiveness failure region for this query.

## Mechanism and resource guidance

- Quality priority: start from the per-track metric table and paired intervals; do not infer a universal winner from point estimates alone.
- Lightweight or throughput priority: compare encoding throughput, snapshot scale, CSR/index bytes, and paired quality intervals together. SPLADE-Tiny is the smallest frozen Batch-B mechanism but is local research-only.
- Static-query/document-expansion priority: compare opensearch-doc-v2-mini, opensearch-doc-v3, and opensearch-multilingual against neural-query alternatives. The multilingual model's route is static lookup plus document expansion; these English-track results do not establish multilingual effectiveness.
- Uncertain conclusions: treat any paired interval crossing zero, every slice with n<10, training-overlap uncertainty, and cross-track reversals as inconclusive.

Encoding RSS values are encoding-observed measurements, not whole-cell peaks. No model is described as verified zero-shot.

## License and publication boundaries

SPLADE-Tiny: MIT is frozen only from the pinned model card and Hub API. The repository has no standalone LICENSE file. Use is local research-only, weight redistribution is forbidden, and independent Validator license recheck is mandatory.
The package does not redistribute weights. The independent Validator must recheck the SPLADE-Tiny license boundary and the predeclaration supersession lineage.
Public export, publication readiness, and leaderboard publication remain disabled.
