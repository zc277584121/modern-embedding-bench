# BRIGHT Real Multi-Vector First-Round Comparison

This package is research-only. Publication, public export, and leaderboard gates remain closed.

The matrix contains three independently pinned late-interaction checkpoints on economics and psychology. Tracks remain independent; there is no pooled score or cross-paradigm claim.

## Frozen checkpoint identities

| Model | Revision | License | Parameters | Dimension | Query / document length |
|---|---|---|---:|---:|---:|
| colbert-v2 | `c1e84128e85ef755c096a95bdb06b47793b13acf` | MIT | 110,000,000 | 128 | 32 / 180 |
| answerai-colbert-small | `934fa8bb4ce2284f4c2baa232d81aca4d076fa5e` | Apache-2.0 | 33,000,000 | 96 | 32 / 300 |
| gte-modern-colbert | `25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639` | Apache-2.0 | 149,000,000 | 128 | 48 / 300 |

Primary-source Hub API, model-card, saved-config, and artifact-metadata URLs or content hashes are frozen in `inventory.json`. All three checkpoints are public and ungated, use allowlisted safetensors, and load with `trust_remote_code=False`.

## Economics quality

| Model | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| answerai-colbert-small | 0.302677 | 0.256392 | 0.359709 | 0.300336 | 0.624894 |
| gte-modern-colbert | 0.300705 | 0.246856 | 0.346571 | 0.321985 | 0.686286 |
| colbert-v2 | 0.251082 | 0.196709 | 0.294175 | 0.266999 | 0.522922 |

## Psychology quality

| Model | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| gte-modern-colbert | 0.396402 | 0.331597 | 0.428725 | 0.460017 | 0.765836 |
| answerai-colbert-small | 0.354407 | 0.307367 | 0.382261 | 0.409078 | 0.773960 |
| colbert-v2 | 0.321466 | 0.261455 | 0.355265 | 0.385280 | 0.686697 |

## Paired uncertainty

All three pairs per track and five metrics use 10,000 matched-query percentile bootstrap samples with seed 20260826. A directional claim requires the 95% interval to exclude zero.
- economics: answerai-colbert-small exceeds colbert-v2 on ndcg@10; the paired interval excludes zero.
- economics: answerai-colbert-small exceeds colbert-v2 on map@100; the paired interval excludes zero.
- economics: answerai-colbert-small exceeds colbert-v2 on mrr@10; the paired interval excludes zero.
- economics: answerai-colbert-small exceeds colbert-v2 on recall@10; the paired interval excludes zero.
- economics: answerai-colbert-small exceeds colbert-v2 on recall@100; the paired interval excludes zero.
- economics: gte-modern-colbert exceeds colbert-v2 on ndcg@10; the paired interval excludes zero.
- economics: gte-modern-colbert exceeds colbert-v2 on map@100; the paired interval excludes zero.
- economics: gte-modern-colbert exceeds colbert-v2 on mrr@10; the paired interval excludes zero.
- economics: gte-modern-colbert exceeds colbert-v2 on recall@10; the paired interval excludes zero.
- economics: gte-modern-colbert exceeds colbert-v2 on recall@100; the paired interval excludes zero.
- economics: gte-modern-colbert exceeds answerai-colbert-small on recall@100; the paired interval excludes zero.
- psychology: answerai-colbert-small exceeds colbert-v2 on map@100; the paired interval excludes zero.
- psychology: answerai-colbert-small exceeds colbert-v2 on recall@100; the paired interval excludes zero.
- psychology: gte-modern-colbert exceeds colbert-v2 on ndcg@10; the paired interval excludes zero.
- psychology: gte-modern-colbert exceeds colbert-v2 on map@100; the paired interval excludes zero.
- psychology: gte-modern-colbert exceeds colbert-v2 on mrr@10; the paired interval excludes zero.
- psychology: gte-modern-colbert exceeds colbert-v2 on recall@10; the paired interval excludes zero.
- psychology: gte-modern-colbert exceeds colbert-v2 on recall@100; the paired interval excludes zero.

## Predeclared slices and cases

- economics `query_length`: <=64: n=15, 65-128: n=46, >128: n=42
- economics `positive_lexical_overlap`: 0: n=0, (0,0.25]: n=31, (0.25,0.50]: n=64, >0.50: n=8
- economics `positive_document_length`: <=128: n=18, 129-512: n=69, >512: n=16
- economics `positive_qrel_density`: 1: n=35, 2-4: n=30, >=5: n=38
- psychology `query_length`: <=64: n=17, 65-128: n=48, >128: n=36
- psychology `positive_lexical_overlap`: 0: n=0, (0,0.25]: n=30, (0.25,0.50]: n=60, >0.50: n=11
- psychology `positive_document_length`: <=128: n=16, 129-512: n=66, >512: n=19
- psychology `positive_qrel_density`: 1: n=35, 2-4: n=32, >=5: n=34

Bins with fewer than 10 queries are descriptive only and were not merged. The tracked case set has six disagreement and six shared-failure cases per track, using opaque labels and non-reversible local-evidence hashes only.

## Protocol, resource, and maintenance guidance

Exact float32 MaxSim is recomputed from saved token vectors, followed by frozen max-over-window passage aggregation. The 1x4 adapter gate independently matched NumPy MaxSim and exercised boolean padding masks before formal scoring.

Representation storage is split into token-array bytes, mapping/identity bytes, representation-manifest bytes, complete replayable non-sidecar bytes, and sidecar bytes. The complete figure is the full per-track directory required for authenticated replay, not only values/offsets/masks.

Encoding latency distributions are chunk-amortized observations: each measured chunk wall time divided by its item count. They are not presented as individually timed item percentiles. Every model/track also records attempt counts, OOM count, retries, fallback use, and terminal failure status.

Choose by track and target metric, require a paired interval for directional quality claims, then compare checkpoint size, token-vector footprint, encoding throughput, and exact-search latency. The small AnswerAI checkpoint is the resource-oriented option; GTE-ModernColBERT and ColBERTv2 provide complementary architecture/scale points. These are shortlist recommendations, not a universal ranking.

Training overlap is unknown. The data covers two English tracks only. The package does not redistribute source text or weights, and does not establish multilingual effectiveness, verified zero-shot behavior, production SLA, or complete cross-paradigm superiority.
