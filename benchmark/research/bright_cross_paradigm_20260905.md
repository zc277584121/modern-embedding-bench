# BRIGHT Cross-Paradigm Exact Retrieval Study

Status: research-only candidate. Publication, leaderboard, and export gates remain closed.

## Scope and chronology

This study compares 14 frozen retrieval methods on the economics and psychology BRIGHT pilot tracks. It contains 20 accepted read-only quality cells and eight S-009 exact full-corpus cells: the four already-valid new baselines plus four representation/score-backed reruns of the frozen S-005 methods. The original S-005 cells remain authenticated contextual anchors rather than direct formal cells. The disclosed pre-freeze synthetic TF-IDF unit-test readiness event is retained in the immutable chronology; it did not expose formal scores. No inventory, identity, protocol, seed, slice, resource plan, or stopping condition was changed after new formal scores became visible.

## Quality protocol

Every quality cell uses the same canonical 7,500-passage candidate pool per track, canonical passage retrieval unit, top-100 depth, five core metrics, and deterministic score/ID tie policy. Method-native tokenization, context limits, sparse/dense/MaxSim scoring, and multi-vector window aggregation are the intrinsic mechanisms under test. All searches are exact; approximation error is therefore separated from method quality as not applicable.

## Aggregate results

### Economics

| Method | Family | nDCG@10 | MAP@100 | MRR@10 | R@10 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| bm25-unicode | lexical | 0.2528 | 0.1890 | 0.2959 | 0.2775 | 0.5048 |
| tfidf-word-sublinear | lexical | 0.1589 | 0.1245 | 0.1713 | 0.1910 | 0.4647 |
| bge-m3-dense | dense | 0.2544 | 0.2030 | 0.2692 | 0.3025 | 0.6492 |
| all-minilm-l6-v2 | dense | 0.2632 | 0.2118 | 0.2827 | 0.3024 | 0.6387 |
| granite-30m-sparse | learned_sparse | 0.2595 | 0.2071 | 0.2933 | 0.2952 | 0.6767 |
| opensearch-doc-v2-mini | learned_sparse | 0.2484 | 0.1964 | 0.2872 | 0.2702 | 0.5422 |
| opensearch-doc-v3 | learned_sparse | 0.2444 | 0.1957 | 0.2820 | 0.2637 | 0.5803 |
| bge-m3 | learned_sparse | 0.2806 | 0.2153 | 0.3258 | 0.3185 | 0.6543 |
| splade-tiny | learned_sparse | 0.2527 | 0.1932 | 0.2994 | 0.2682 | 0.5241 |
| opensearch-doc-v2-distill | learned_sparse | 0.2813 | 0.2310 | 0.2908 | 0.3094 | 0.6883 |
| opensearch-multilingual | learned_sparse | 0.2497 | 0.1949 | 0.2944 | 0.2872 | 0.5939 |
| colbert-v2 | multi_vector | 0.2511 | 0.1967 | 0.2942 | 0.2670 | 0.5229 |
| answerai-colbert-small | multi_vector | 0.3027 | 0.2564 | 0.3597 | 0.3003 | 0.6249 |
| gte-modern-colbert | multi_vector | 0.3007 | 0.2469 | 0.3466 | 0.3220 | 0.6863 |

### Psychology

| Method | Family | nDCG@10 | MAP@100 | MRR@10 | R@10 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| bm25-unicode | lexical | 0.1596 | 0.1340 | 0.1975 | 0.1800 | 0.4164 |
| tfidf-word-sublinear | lexical | 0.0825 | 0.0693 | 0.0925 | 0.1188 | 0.4093 |
| bge-m3-dense | dense | 0.2971 | 0.2470 | 0.3328 | 0.3631 | 0.6785 |
| all-minilm-l6-v2 | dense | 0.2781 | 0.2344 | 0.3237 | 0.3189 | 0.6634 |
| granite-30m-sparse | learned_sparse | 0.2963 | 0.2561 | 0.3277 | 0.3261 | 0.6250 |
| opensearch-doc-v2-mini | learned_sparse | 0.2508 | 0.2013 | 0.3045 | 0.2756 | 0.5699 |
| opensearch-doc-v3 | learned_sparse | 0.2386 | 0.1964 | 0.2828 | 0.2702 | 0.5798 |
| bge-m3 | learned_sparse | 0.3227 | 0.2774 | 0.3917 | 0.3353 | 0.6108 |
| splade-tiny | learned_sparse | 0.2570 | 0.2081 | 0.3001 | 0.2938 | 0.5542 |
| opensearch-doc-v2-distill | learned_sparse | 0.2876 | 0.2604 | 0.3330 | 0.3051 | 0.6445 |
| opensearch-multilingual | learned_sparse | 0.2890 | 0.2418 | 0.3583 | 0.3047 | 0.6133 |
| colbert-v2 | multi_vector | 0.3215 | 0.2615 | 0.3553 | 0.3853 | 0.6867 |
| answerai-colbert-small | multi_vector | 0.3544 | 0.3074 | 0.3823 | 0.4091 | 0.7740 |
| gte-modern-colbert | multi_vector | 0.3964 | 0.3316 | 0.4287 | 0.4600 | 0.7658 |

## S-005 contextual-anchor deltas

The accepted S-005 identities remain unchanged and provide chronology and contextual source evidence. The formal matrix uses the new score/representation-backed reruns below.

| Cell | Exact top-100 order queries | Mismatches | Minimum overlap | nDCG@10 delta |
|---|---:|---:|---:|---:|
| bm25-unicode:economics | 103/103 | 0 | 100/100 | +0.00000000 |
| bm25-unicode:psychology | 101/101 | 0 | 100/100 | +0.00000000 |
| bge-m3-dense:economics | 102/103 | 1 | 100/100 | +0.00000000 |
| bge-m3-dense:psychology | 99/101 | 2 | 100/100 | +0.00000000 |

## Uncertainty, slices, and cases

All 91 within-track method pairs were bootstrapped with 10,000 matched-query resamples at seed 20260826. At least one metric was decisive for 63 economics pairs and 70 psychology pairs. All frozen slice bins were retained; bins with fewer than ten queries contain point estimates only. The case artifact contains six largest-rank-spread disagreements and six lowest-best-method failures per track. Canonical IDs and row-level evidence remain only in the restricted local mapping.

## Resource interpretation

The unified resource matrix has 22 successful and 6 failed-closed cells out of 28. Timing comparisons are valid only within successful unified runs. Representation storage is stratified by paradigm and is not normalized into a synthetic common unit. TF-IDF scope includes document/query representations plus IDF, while MiniLM scope is dense document/query vectors only. Historical measurements remain contextual anchors.

## Scenario selection

Choose the track and scenario first, inspect the paired nDCG@10 verdict for the point leader versus the runner-up, and then use only successful unified resource cells as a separate operational filter. A tie-or-uncertain interval permits a resource-driven choice but does not prove equivalence.

| Track | Scenario | n | Point leader | Runner-up verdict | Successful resource options |
|---|---|---:|---|---|---|
| economics | overall_exact_quality | 103 | answerai-colbert-small | tie_or_uncertain vs gte-modern-colbert | answerai-colbert-small, gte-modern-colbert, opensearch-doc-v2-distill |
| economics | positive_lexical_overlap:(0,0.25] | 31 | all-minilm-l6-v2 | tie_or_uncertain vs bge-m3-dense | all-minilm-l6-v2, bge-m3-dense, answerai-colbert-small |
| economics | positive_passage_length:>512 | 16 | answerai-colbert-small | tie_or_uncertain vs gte-modern-colbert | answerai-colbert-small, gte-modern-colbert, splade-tiny |
| economics | qrel_density:>=5 | 38 | gte-modern-colbert | tie_or_uncertain vs bge-m3 | gte-modern-colbert, bge-m3, opensearch-doc-v2-distill |
| psychology | overall_exact_quality | 101 | gte-modern-colbert | tie_or_uncertain vs answerai-colbert-small | gte-modern-colbert, answerai-colbert-small, bge-m3 |
| psychology | positive_lexical_overlap:(0,0.25] | 30 | gte-modern-colbert | tie_or_uncertain vs all-minilm-l6-v2 | gte-modern-colbert, all-minilm-l6-v2, bge-m3-dense |
| psychology | positive_passage_length:>512 | 19 | gte-modern-colbert | tie_or_uncertain vs answerai-colbert-small | gte-modern-colbert, answerai-colbert-small, colbert-v2 |
| psychology | qrel_density:>=5 | 34 | gte-modern-colbert | tie_or_uncertain vs answerai-colbert-small | gte-modern-colbert, answerai-colbert-small, all-minilm-l6-v2 |

## Descriptive slice and representation sensitivity

The sensitivity artifact is descriptive: it contrasts frozen slice outcomes and discloses intrinsic representation caps. It does not vary the frozen protocol and therefore is not protocol-deviation evidence.

## Limitations

Unjudged passages are not confirmed negatives. Two tracks do not establish a universal winner. Intrinsic context limits and training overlap boundaries remain potential explanatory factors. S-005 accepted cells remain contextual anchors; their replacement formal reruns provide score- or representation-level replay, and small floating-point order deltas are explicitly quantified. Slice and representation sensitivity is descriptive, not causal. Failed-closed resource cells remain unavailable for operational comparison. This candidate is not approved for publication or export.
