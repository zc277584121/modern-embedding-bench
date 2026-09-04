# Learned Sparse Retrieval: Phase-One Synthesis

## Decision summary

Phase one establishes a reproducible same-paradigm learned-sparse comparison on two English research tracks, plus a separate sparse-system experiment. It does not establish a universal model winner, a cross-paradigm ranking, multilingual or verified zero-shot effectiveness, a production SLA, or a universal SINDI advantage. The publication gate remains closed.

The most defensible maintenance rule is conditional selection: choose the track and metric first, require paired uncertainty for quality claims, then price the model route, encoding cost, sparsity, exact-search footprint, and deployment system separately. [C-QUALITY, C-UNCERTAINTY, C-RESOURCES, C-SYSTEM]

## Evidence and publication boundary

This report is deterministically derived from the accepted commits for S-20260814-005, S-20260814-006, S-20260814-008, and S-20260814-010. The evidence map binds every table and material recommendation to a tracked SHA-256 identity and accepted commit; private evidence appears only as non-path identities. No model is loaded and no Milvus connection is made by the derivation command.

The underlying BRIGHT passages, queries, qrels, canonical identifiers, raw rankings, per-query rows, and review mappings remain restricted. Repository-level licensing does not establish redistribution rights for every linked upstream document. All outputs are `research_only`, with public export and leaderboard publication disabled. [C-LICENSE]

## Data contract and governance

The source is `xlangai/BRIGHT` at revision `3066d29c9651a576c8aba4832d249807b181ecae`, loaded without remote code. Economics and psychology retain independent candidate pools. Every query and positive passage is retained; remaining passages are selected by a score-blind, salted exact-content-hash policy. `gold_ids` supply grade-1 qrels, excluded IDs are filters rather than negative labels, and unjudged passages remain unjudged. [C-DATA]

| Track | Queries | Passages | Positive qrels | Qrel density | Single-positive queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| economics | 103 | 7,500 | 800 | 0.00103560 | 35 |
| psychology | 101 | 7,500 | 692 | 0.00091353 | 35 |

The combined package therefore contains 204 queries, 15,000 passages, and 1,492 positive qrels. A 120-query stratified manual-review plan was frozen before model execution, but this phase does not represent that review as completed. Training overlap is unknown, and no verified zero-shot claim is allowed. [C-DATA, C-MULTILINGUAL-GAP]

## Model portfolio and routing

The seven-model matrix spans neural query/document expansion, contextual lexical projection, and static query lookup paired with neural document expansion. Revisions are fixed; route differences matter operationally because static-query models move more computation to document ingestion while neural-query models retain learned query encoding. [C-ROUTING]

| Model | Query route | Document route | Dimensions | Snapshot MiB | Economics doc/query nnz | Psychology doc/query nnz |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| bge-m3 | neural | neural | 250,002 | 3072.0 | 51.5/64.3 | 47.9/59.7 |
| granite-30m-sparse | neural | neural | 50,265 | 60.4 | 163.3/50.0 | 161.4/50.0 |
| opensearch-doc-v2-distill | neural | neural | 30,522 | 256.4 | 166.9/187.5 | 169.1/199.0 |
| opensearch-doc-v2-mini | static_lookup | document_expansion | 30,522 | 90.2 | 205.4/87.8 | 203.7/85.6 |
| opensearch-doc-v3 | static_lookup | document_expansion | 30,522 | 259.0 | 182.7/87.8 | 182.3/85.6 |
| opensearch-multilingual | static_lookup | document_expansion | 105,879 | 652.1 | 248.3/91.8 | 272.4/90.5 |
| splade-tiny | neural | neural | 30,522 | 17.8 | 123.1/171.1 | 120.9/164.2 |

SPLADE-Tiny is the smallest frozen snapshot (about 17.8 MiB), but its MIT statement is frozen only from model-card/API evidence and the repository lacks a standalone license file. It remains local research-only and is not a redistribution recommendation. [C-LICENSE]

## Same-paradigm quality evidence

All learned-sparse rows use the same per-track corpus, queries, qrels, candidate policy, maximum length 512, exact SciPy CSR inner product, top-100 depth, and five metrics. Point estimates are useful for shortlisting, but a point leader is not automatically a statistically supported winner. [C-QUALITY, C-UNCERTAINTY]

### Economics

| Model | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| opensearch-doc-v2-distill | 0.281309 | 0.231000 | 0.290804 | 0.309436 | 0.688349 |
| bge-m3 | 0.280627 | 0.215270 | 0.325778 | 0.318493 | 0.654319 |
| granite-30m-sparse | 0.259533 | 0.207093 | 0.293296 | 0.295205 | 0.676714 |
| splade-tiny | 0.252678 | 0.193169 | 0.299391 | 0.268240 | 0.524069 |
| opensearch-multilingual | 0.249693 | 0.194916 | 0.294371 | 0.287213 | 0.593892 |
| opensearch-doc-v2-mini | 0.248416 | 0.196430 | 0.287201 | 0.270160 | 0.542179 |
| opensearch-doc-v3 | 0.244365 | 0.195677 | 0.282039 | 0.263687 | 0.580278 |

The package evaluates 21 pairs × 5 metrics = 105 query-aligned comparisons, each with 10,000 bootstrap samples. 18 intervals exclude zero; the remaining 87 are explicitly uncertain. The nDCG@10 point order shown above is track-specific, not an overall ranking. [C-UNCERTAINTY]

### Psychology

| Model | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bge-m3 | 0.322664 | 0.277383 | 0.391686 | 0.335317 | 0.610826 |
| granite-30m-sparse | 0.296319 | 0.256079 | 0.327691 | 0.326074 | 0.625040 |
| opensearch-multilingual | 0.288993 | 0.241773 | 0.358306 | 0.304699 | 0.613330 |
| opensearch-doc-v2-distill | 0.287598 | 0.260424 | 0.333031 | 0.305143 | 0.644500 |
| splade-tiny | 0.256993 | 0.208106 | 0.300122 | 0.293792 | 0.554238 |
| opensearch-doc-v2-mini | 0.250751 | 0.201311 | 0.304495 | 0.275569 | 0.569914 |
| opensearch-doc-v3 | 0.238630 | 0.196372 | 0.282783 | 0.270181 | 0.579766 |

The package evaluates 21 pairs × 5 metrics = 105 query-aligned comparisons, each with 10,000 bootstrap samples. 31 intervals exclude zero; the remaining 74 are explicitly uncertain. The nDCG@10 point order shown above is track-specific, not an overall ranking. [C-UNCERTAINTY]

## Limited BM25 and dense anchors

S-005 BM25 and BGE-M3 long-dense numbers are retained only as contextual anchors. The source revision, canonical passage/query retrieval units without re-segmentation, independent per-track candidate pools, qrels, exclusion filtering, top-100 depth, and five metrics are identity-linked to the learned-sparse package. Encoder processing batch sizes differ but do not redefine retrieval units. This supports bounded same-pilot context, but not the complete multi-baseline cross-paradigm comparison reserved for S-20260814-009. Dense uses a predeclared 1,024-token cap, whereas learned-sparse uses 512 tokens. [C-ANCHORS]

| Track | Method | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| economics | BM25 | 0.25277 | 0.18900 | 0.29591 | 0.27750 | 0.50482 |
| economics | BGE-M3 long dense | 0.25435 | 0.20298 | 0.26922 | 0.30250 | 0.64916 |
| psychology | BM25 | 0.15962 | 0.13395 | 0.19751 | 0.17996 | 0.41635 |
| psychology | BGE-M3 long dense | 0.29706 | 0.24705 | 0.33280 | 0.36312 | 0.67853 |

Economics does not establish a top-10 dense-versus-BM25 difference because the paired nDCG@10 interval crosses zero. Psychology has five dense-minus-BM25 intervals excluding zero. These statements remain anchor-specific and do not rank paradigms generally. [C-ANCHORS]

## Encoding, sparsity, and exact retrieval

| Model | Track | Doc items/s | Query items/s | Doc CSR MiB | Exact search s | Doc peak VRAM MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bge-m3 | economics | 167.5 | 158.9 | 2.97 | 0.832 | 1190.7 |
| bge-m3 | psychology | 171.2 | 173.3 | 2.77 | 0.768 | 1190.3 |
| granite-30m-sparse | economics | 513.4 | 439.4 | 9.37 | 0.869 | 857.2 |
| granite-30m-sparse | psychology | 527.8 | 518.4 | 9.26 | 0.855 | 857.2 |
| opensearch-doc-v2-distill | economics | 378.5 | 329.6 | 9.58 | 0.844 | 1223.5 |
| opensearch-doc-v2-distill | psychology | 393.5 | 374.4 | 9.70 | 0.824 | 1223.5 |
| opensearch-doc-v2-mini | economics | 482.3 | 926.9 | 11.78 | 0.923 | 1053.8 |
| opensearch-doc-v2-mini | psychology | 479.3 | 1076.9 | 11.68 | 0.916 | 1053.8 |
| opensearch-doc-v3 | economics | 332.8 | 889.7 | 10.48 | 0.928 | 1224.3 |
| opensearch-doc-v3 | psychology | 339.2 | 814.3 | 10.46 | 0.886 | 1223.9 |
| opensearch-multilingual | economics | 188.8 | 883.3 | 14.24 | 0.961 | 3971.1 |
| opensearch-multilingual | psychology | 203.2 | 917.5 | 15.61 | 0.935 | 3971.1 |
| splade-tiny | economics | 696.9 | 604.6 | 7.07 | 0.905 | 983.6 |
| splade-tiny | psychology | 675.5 | 657.2 | 6.95 | 0.870 | 983.6 |

SPLADE-Tiny has the highest measured document throughput in both tracks, while the static-query routes provide high query throughput. BGE-M3 sparse has the smallest document CSR footprint among the seven. These are bounded encoding/exact measurements, not end-to-end production throughput. Every RSS value is encoding-observed rather than a whole-cell peak. [C-RESOURCES]

## Predeclared slices and opaque cases

The package freezes query length, positive lexical overlap, positive-document length, and positive-qrel-density slices before Batch-B scores. It reports 13 bins per track. Empty bins are NA, and any bin with n<10 is descriptive only; no bin is merged after results. [C-SLICES]

| Track | Slice | Bin sizes |
| --- | --- | --- |
| economics | query_length | <=64: n=15, 65-128: n=46, >128: n=42 |
| economics | positive_lexical_overlap | 0: n=0, (0,0.25]: n=31, (0.25,0.50]: n=64, >0.50: n=8 |
| economics | positive_document_length | <=128: n=18, 129-512: n=69, >512: n=16 |
| economics | positive_qrel_density | 1: n=35, 2-4: n=30, >=5: n=38 |
| psychology | query_length | <=64: n=17, 65-128: n=48, >128: n=36 |
| psychology | positive_lexical_overlap | 0: n=0, (0,0.25]: n=30, (0.25,0.50]: n=60, >0.50: n=11 |
| psychology | positive_document_length | <=128: n=16, 129-512: n=66, >512: n=19 |
| psychology | positive_qrel_density | 1: n=35, 2-4: n=32, >=5: n=34 |

The tracked case set contains six disagreements and six shared failures per track, 24 cases total. Only opaque case labels, mechanism-level summaries, and non-reversible hashes are exposed; the canonical mapping and source text remain private. These cases diagnose where mechanisms disagree or jointly fail, but do not create new semantic slice labels. [C-CASES]

## Sparse-system evidence

The system layer is separate from model quality. It uses saved CSR inputs with `model_loaded=false` and compares exact SciPy CSR, Milvus SINDI, and Milvus DAAT_MAXSCORE. The server is Milvus 3.0.0, PyMilvus 3.0.1, vector index version 10, fixed CPU/memory/PID limits, explicit sparse IP algorithms, `drop_ratio_build=0`, and `drop_ratio_search=0`. Searches require serialized indexes, sealed segments, zero growing rows, and no active or queued compaction. [C-EXACT, C-SYSTEM]

| Tier | Cells | Models/tracks | Purpose | Quality eligible |
| --- | ---: | --- | --- | --- |
| Native | 18 | 3 model profiles × 2 tracks × 3 comparators | Correctness and bounded system behavior | No; model quality remains in the original quality package |
| System-only 100k | 9 | 3 profiles × 3 comparators | Scaling and cost | No |
| System-only 1M | 9 | 3 profiles × 3 comparators | Scaling and cost | No |

The validator recomputed 1,296 measured trials and validated 432 warmups. Native raw strict Recall@K reaches a minimum of 0.9; after matching the persisted float16-document/float32-query representation, tie-aware Recall@K is 1.0 and the maximum score delta is 0.00001525878906. The system-only strict minimum of 0.2 occurs at synthetic near-tie boundaries and is not model-quality evidence. [C-SYSTEM, C-SYSTEM-QUALITY]

| System result | Value | Interpretation |
| --- | ---: | --- |
| Median SINDI/DAAT QPS ratio | 1.0039 | Near parity in the pooled tested configurations |
| Observed ratio range | 0.9676–8.2148 | Crosses 1.0; workload/configuration crossover |
| SINDI-larger serialized-index pairs | 12/12 | SINDI used more serialized index bytes in every pair |
| Serialized-index ratio range | 1.099935–1.232598 | Does not include all runtime or operational cost |

The maximum QPS ratio comes from a synthetic 1M OpenSearch-multilingual K100/concurrency-1 DAAT tail collapse; other configurations show parity or DAAT advantages. Warm-over-cold medians exceed 1.0 by stratum, but individual reversals remain. Cold runs reload collections without dropping host page caches. There is no universal algorithm, cache, scale, or production claim. [C-SYSTEM, C-UNIVERSAL-SINDI, C-PRODUCTION]

## Scenario-based maintenance guide

### Quality-first selection

Shortlist by track and target metric, then require a paired interval that excludes zero before claiming an advantage. Economics and psychology point orders differ, so do not collapse them. [C-QUALITY, C-UNCERTAINTY]

Limits: No universal model winner; Training overlap is unknown; Only two English tracks.

### Lightweight or encoding-throughput priority

Use snapshot size, document/query throughput, CSR bytes, and paired quality together. SPLADE-Tiny is the smallest and fastest-document option measured here, but its standalone license evidence is incomplete and its quality trade-offs are track/metric specific. [C-RESOURCES, C-LICENSE]

Limits: Encoding-observed RSS is not whole-cell peak; Local research-only use for SPLADE-Tiny.

### Static-query lookup with document expansion

Compare the three OpenSearch static-query routes when inexpensive query-side execution matters, but budget document-side expansion and index density separately. The English measurements do not establish multilingual behavior. [C-ROUTING, C-RESOURCES]

Limits: No production SLA; No multilingual effectiveness evidence.

### Multilingual candidate selection

Keep OpenSearch multilingual as a candidate because its architecture is multilingual; treat every effectiveness number here as English-only evidence and schedule a multilingual track before deployment claims. [C-MULTILINGUAL-GAP]

Limits: No multilingual track; No verified zero-shot evidence.

### Exact retrieval and quality evaluation

Keep SciPy CSR exact inner product as the quality ground truth and simplest 7,500-document maintenance path. Its system cost is a CPU implementation anchor, not a universal production baseline. [C-EXACT, C-SYSTEM]

Limits: Small native corpus; No production availability or network layer.

### Milvus deployment choice

Benchmark SINDI and DAAT_MAXSCORE on the intended workload rather than selecting from the algorithm name. The accepted matrix crosses over by workload/configuration, and SINDI serialized indexes were larger in every paired cell. [C-SYSTEM, C-INDEX-BYTES]

Limits: Pinned Milvus 3.0.0 build only; System-only scaling is not quality evidence; No production SLA.

## Claim boundary matrix

| Claim ID | Support | Statement | Scope or reason |
| --- | --- | --- | --- |
| C-DATA | supported | The fixed research pilot contains 204 queries, 15,000 passages, and 1,492 grade-1 qrels across independent economics and psychology tracks. | Fixed v0.2 pilot only |
| C-QUALITY | supported | Seven learned-sparse models have per-track results for five quality metrics on the shared pilot. | Same-paradigm, two-track comparison |
| C-UNCERTAINTY | supported | Pairwise statements require query-aligned 10,000-sample intervals that exclude zero; intervals crossing zero remain uncertain. | Per track and metric only |
| C-SLICES | supported | Four predeclared slice dimensions and 13 bins per track are available; bins with n<10 are descriptive only. | No post-hoc semantic categories |
| C-CASES | supported | The tracked package contains 24 opaque disagreement/failure cases without canonical identifiers or source text. | Mechanism-level triage only |
| C-ROUTING | supported | The portfolio covers neural/neural and static-query/document-expansion routes with distinct sparsity and encoding profiles. | Measured at the frozen revisions and protocol |
| C-RESOURCES | supported | Snapshot, nnz, encoding throughput, CSR bytes, exact-search time, VRAM, and encoding-observed RSS can inform bounded maintenance choices. | RSS is not a whole-cell peak; no production SLA |
| C-EXACT | supported | SciPy CSR exact sparse inner product is the quality ground truth for the fixed pilot. | Small native corpus and CPU implementation |
| C-SYSTEM | supported | On the pinned Milvus 3.0.0 matrix, SINDI and DAAT_MAXSCORE exhibit workload/configuration-dependent QPS crossovers. | Pinned build, hardware, and matrix only |
| C-INDEX-BYTES | supported | SINDI serialized indexes are larger in all 12 paired accepted system cells. | Serialized index bytes, not total cost |
| C-LICENSE | supported | SPLADE-Tiny remains local research-only because standalone license evidence is incomplete; the dataset publication gate is closed. | No weight or dataset redistribution authorization |
| C-ANCHORS | limited_anchor | BM25 and BGE-M3 long-dense results provide contextual anchors on the same raw pilot, candidates, qrels, and metrics. | Not the complete S-009 cross-paradigm comparison; dense uses a different token cap |
| C-MULTILINGUAL-GAP | unsupported | The measured results establish multilingual effectiveness or verified zero-shot behavior. | Both tracks are English and training overlap is unknown |
| C-UNIVERSAL-MODEL | unsupported | One model is the universal learned-sparse winner. | Track/metric orders differ and many intervals cross zero |
| C-UNIVERSAL-SINDI | unsupported | SINDI is universally faster or smaller than DAAT_MAXSCORE. | QPS ratios cross one and every measured SINDI index is larger |
| C-SYSTEM-QUALITY | unsupported | The 100k/1M generated workloads demonstrate model-quality gains. | System-only near-tie workloads are quality-ineligible |
| C-PRODUCTION | unsupported | The phase-one results establish a production SLA or deployment recommendation valid across systems. | No production environment or SLA study |
| C-CROSS-TRACK | unsupported | A cross-track micro-average or total ordering is valid. | Independent candidate pools and explicit no-aggregation policy |

## Remaining evidence gaps

- No complete cross-paradigm comparison with multiple representative dense, BM25, and multi-vector baselines.
- No multilingual effectiveness or verified zero-shot evaluation.
- No production SLA, failure-domain, cost-of-operations, or current-version Milvus validation.
- No document-level upstream redistribution-rights clearance; publication gate remains closed.
- The fixed pilot covers 204 queries and 15,000 passages in two English tracks, below the longer-term 3-5-domain target.
- The planned 120-query manual audit has not been represented as completed evidence.

These gaps are inputs to the next milestone route review. They do not authorize S-007 data generation, S-009 cross-paradigm experiments, S-012 multi-vector work, publication, or any new benchmark/index run.

## Deterministic verification

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/learned_sparse_phase_one_report.py check
```

The command authenticates accepted commit ancestry and tracked file hashes, validates source and output schemas, recomputes every derived table and claim object in memory, and compares exact bytes. It neither loads a model nor connects to Milvus.
