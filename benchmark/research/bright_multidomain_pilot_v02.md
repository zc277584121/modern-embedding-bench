# BRIGHT Economics and Psychology Research Pilot v0.2

## Acceptance status

The research-only pilot is runnable and complete at the requested pilot scale:
204 queries, 15,000 independently selected passages, and 1,492 grade-1 positive
qrels across economics and psychology. Each track retains every source query and
positive passage, then fills its own 7,500-passage pool by a fixed salted order
over exact-content SHA256 groups. No baseline or model score participates in
selection.

The formal comparison is fixed-tokenizer pure-Python BM25 versus the dense
representation of `BAAI/bge-m3` at revision
`5617a9f61b028005a4858fdac845db406aefb181`. The previously run
`sentence-transformers/all-MiniLM-L6-v2` revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` remains diagnostic evidence only
because its 256-token limit truncates too many gold passages. Results are
reported per track. The unweighted macro is descriptive and is not a merged
leaderboard.

These slice results must not be compared directly with full-corpus biology or
official BRIGHT results. Candidate pools, corpus scale, and protocol differ.

## Fixed data and label contract

The source is `xlangai/BRIGHT` revision
`3066d29c9651a576c8aba4832d249807b181ecae`, loaded without remote code. The
fixed source file identities and the canonical corpus, query, qrel, selection,
audit, and review-plan identities are recorded in the materialization manifest.
The current manifest SHA256 is
`8174b0c01a32e977cf0c5d89522356c485196b0aa0571251cfe1595c5579cb1f`.

| Track | Queries | Passages | Grade-1 qrels | Unique gold passages | Qrel density |
| --- | ---: | ---: | ---: | ---: | ---: |
| economics | 103 | 7,500 | 800 | 800 | 0.00103560 |
| psychology | 101 | 7,500 | 692 | 688 | 0.00091353 |

The qrel median is three positives per query in both tracks. Each track has 35
single-positive queries and 11 queries with more than 20 positives. This wide
density range motivates query-level reporting and the fixed stratified review.

`gold_ids` are the only passage relevance labels and become grade-1 qrels.
`gold_ids_long` is provenance only. `excluded_ids` is a per-query candidate
filter and never a negative label. At the fixed source revision, no selected
query has a remaining excluded passage. Unjudged candidates remain unjudged;
the runner and audit pack never convert them to negatives or modify official
qrels.

## Long-context cap decision

Before producing any BGE-M3 score, the fixed tokenizer was used to count input
lengths including special tokens at caps 512, 1,024, 2,048, 4,096, and 8,192.
The rule was fixed in advance: select the minimum cap at which the unique-gold
passage truncation rate is at most 10% in every track.

| Track | Role | 512 | 1,024 | 2,048 | 4,096 | 8,192 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| economics | queries | 2/103 | 0/103 | 0/103 | 0/103 | 0/103 |
| economics | candidates | 711/7,500 | 148/7,500 | 23/7,500 | 11/7,500 | 1/7,500 |
| economics | unique gold | 304/800 | 37/800 | 3/800 | 1/800 | 0/800 |
| psychology | queries | 0/101 | 0/101 | 0/101 | 0/101 | 0/101 |
| psychology | candidates | 425/7,500 | 160/7,500 | 56/7,500 | 27/7,500 | 10/7,500 |
| psychology | unique gold | 188/688 | 63/688 | 10/688 | 4/688 | 1/688 |

Cap 512 fails. Cap 1,024 is the minimum passing cap: 4.625% of economics gold
passages and 9.157% of psychology gold passages are truncated. The decision,
all cap counts, `score_used=false`, and the fixed snapshot identity are embedded
in the schema-validated materialization contract. The snapshot contains 13
files and 2,295,435,813 bytes; its aggregate SHA256 is
`9bfc0c6488e958ec7be6223d95c331c9692c4ae47212a9730cb38ccc190e6cec`.

## Retrieval results

All methods use identical raw query and passage text, candidate filters, qrels,
top-100 depth, metrics, and deterministic 10,000-sample query bootstrap seed.
Dense methods use normalized embeddings and blockwise exact cosine search.

| Track | Method | nDCG@10 | MAP@100 | MRR@10 | Recall@10 | Recall@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| economics | BM25 | 0.25277 | 0.18900 | 0.29591 | 0.27750 | 0.50482 |
| economics | BGE-M3 long dense | 0.25435 | 0.20298 | 0.26922 | 0.30250 | 0.64916 |
| psychology | BM25 | 0.15962 | 0.13395 | 0.19751 | 0.17996 | 0.41635 |
| psychology | BGE-M3 long dense | 0.29706 | 0.24705 | 0.33280 | 0.36312 | 0.67853 |

For economics, the paired BGE-M3-minus-BM25 nDCG@10 delta is 0.00158 with a
95% bootstrap interval of [-0.04575, 0.05151], so the pilot does not establish
a top-10 quality difference. Recall@100 improves by 0.14433 with interval
[0.06366, 0.22655]. For psychology, all five paired intervals exclude zero;
nDCG@10 improves by 0.13744 [0.07766, 0.20019] and Recall@100 by 0.26217
[0.18059, 0.34307]. Full per-query metrics and confidence intervals remain in
the restricted local result directory.

The unweighted descriptive macro is BM25 nDCG@10 0.20620 and Recall@100
0.46059 versus BGE-M3 nDCG@10 0.27571 and Recall@100 0.66384. It must not be
used to hide the different economics and psychology behavior.

MiniLM produces economics nDCG@10 0.26320 and psychology nDCG@10 0.27815, but
it truncates 551/800 economics and 359/688 psychology unique gold passages, as
well as 12/103 and 10/101 queries. These scores are diagnostic only and support
no winner claim.

## Resources and exact artifacts

BM25 uses the fixed `unicode-word-lower-v1` tokenizer. Its wall times are 11.73
seconds for economics and 9.39 seconds for psychology, with 3.12 MB and 3.18 MB
indexes and peak RSS below 121 MB.

BGE-M3 uses batch size 4 on one RTX 3080 Ti. Economics takes 85.14 seconds wall
time at 97.49 passages/second; psychology takes 74.54 seconds at 112.52
passages/second. Peak allocated VRAM is 2,486,395,904 bytes for both tracks.
Peak process RSS is 3.66 GB and 4.38 GB. Each passage index is 30,720,000 bytes;
combined passage-plus-query embeddings are 31,141,888 and 31,133,696 bytes.
The maximum score block is 131,072 elements rather than a retained full score
matrix.

Restricted local artifacts include complete top-100 rankings, per-query metric
rows, result contracts, run manifests, the 120-query audit pack, and full
failure cases. Their identities are recorded in
`benchmark/artifacts/bright-nontechnical-pilot-v0.2/summary.json`; the tracked
summary contains only aggregate statistics and hashes.

Stored results are accepted only after binding the materialization manifest,
corpus, queries, qrels, selection file, method, and track identities. The loader
then validates exact query/rank coverage and candidate membership, replays every
per-query and aggregate metric from rankings and qrels, and recomputes the fixed
10,000-sample bootstrap intervals and failure counts. Re-signing a shortened or
otherwise mutated ranking artifact therefore does not make it valid.

## Governance and audit plan

Exact-content duplication within each selected pool is zero by construction.
The audit still finds 42 economics and 35 psychology normalized-content
duplicate groups, which are retained and disclosed because normalized removal
was not part of the fixed selection rule. There are 25 exact content values
shared across tracks, but the candidate pools and metrics remain separate.

The source represents naturally occurring StackExchange questions paired with
linked web passages and author-curated positives. BRIGHT was public from July
2024, while document dates are unknown or mixed. Training overlap for either
dense model is unknown; no verified zero-shot claim is allowed.

The repository and dataset card declare CC-BY-4.0, but this does not establish
redistribution rights for every copied upstream web passage. The entire pilot
therefore remains `research_only`, with public export closed pending upstream
document-level rights review.

Before creating an output directory, the benchmark data exporter streams the
complete copy tree through no-follow metadata checks and fixed limits: depth 32,
100,000 file-or-directory entries, 20 GiB of aggregate regular-file bytes, and
20 GiB of bytes eligible for copying and content scanning. It never materializes
an unbounded path list before applying those limits. Symlinks and non-regular
entries are rejected. Every planned source file is opened with `O_NOFOLLOW` and
bound to its device, inode, mode, size, timestamps, and SHA256. The exact bytes
are copied to private same-filesystem staging, checked again against that binding,
content-validated from staging, rebound by digest, and published with one atomic
directory rename. Failure removes staging and leaves no output directory.

The bounded content checks reject canonical v0.2 corpus, query, qrel, selection,
audit, and sensitive-evidence files even after renaming, plus research-only
results, rankings, per-query metrics, review packs, and failure cases. Restricted
rows remain rejected inside JSON arrays or object wrappers. Each line is limited
to 1 MiB, each scanned file to 128 MiB, complete JSON documents and safe aggregate
contracts to 4 MiB, and JSON nesting to depth 32. Overlong content, byte-order
marks, non-UTF-8 input, control characters, or any exhausted bound fail closed.
Only the tracked aggregate `manifest.json` and `summary.json` contract is
exportable.

The sensitive-shape scan stores no raw match or context in tracked artifacts.
Economics has 164 email/IP-shaped evidence rows, including 120 rows requiring
manual review; psychology has 87 and 38 respectively. Query text has no email
or IP-shaped match. The only safe examples are placeholders such as
`***@<classified-domain>` and `x.x.x.x`; original values never enter committed
evidence. No US-SSN-shaped value was found.

The 120-query plan was fixed before model execution: 60 queries per track,
stratified by query length, qrel density, and `gold_ids_long` multiplicity. Its
SHA256 is
`93848a826c12acf9cb6fed50d71dd9490be7baebfcea1122dd4fffb4881fdc4e`.
The post-model pack attaches gold passages and the de-duplicated union of BM25,
BGE-M3, and diagnostic MiniLM top-10 candidates. Reviewers must not treat an
unjudged candidate as a negative. Twenty-four queries are scheduled for double
review with Cohen's kappa; the target agreement is at least 0.70.

For failure triage, economics has 41 queries hit by both formal methods at
top-10, 10 BM25-only hits, 14 BGE-M3-only hits, and 38 misses by both.
Psychology has 27, 8, 28, and 38 respectively. The restricted failure artifact
contains two deterministic representative cases from every non-empty
track/category cell, while the tracked summary exposes only case hashes,
aggregate features, and metrics.

The review stops publication or expansion if unsupported/ambiguous gold exceeds
5%, credible missing positives occur in more than 10% of reviewed queries, or
agreement falls below 0.70. The current package is a review plan and candidate
pack, not a claim that these manual judgments have already been completed.

## Reproduction

All restricted output paths below are gitignored.

```bash
uv sync --extra data --extra local
uv run python scripts/bright_multidomain_pilot_v02.py materialize \
  --source-root data/bright-nontechnical-pilot-v0.1/source \
  --output data/bright-nontechnical-pilot-v0.2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python scripts/bright_multidomain_pilot_v02.py long-dense-preflight \
  --data data/bright-nontechnical-pilot-v0.2
uv run python scripts/bright_multidomain_pilot_v02.py run \
  --data data/bright-nontechnical-pilot-v0.2 \
  --output results/bright-nontechnical-pilot-v0.2 --method bm25
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python scripts/bright_multidomain_pilot_v02.py run \
  --data data/bright-nontechnical-pilot-v0.2 \
  --output results/bright-nontechnical-pilot-v0.2 --method dense
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python scripts/bright_multidomain_pilot_v02.py run \
  --data data/bright-nontechnical-pilot-v0.2 \
  --output results/bright-nontechnical-pilot-v0.2 --method long_dense
uv run python scripts/bright_multidomain_pilot_v02.py review-pack \
  --data data/bright-nontechnical-pilot-v0.2 \
  --results results/bright-nontechnical-pilot-v0.2 \
  --output results/bright-nontechnical-pilot-v0.2/audit
uv run python scripts/bright_multidomain_pilot_v02.py summarize \
  --data data/bright-nontechnical-pilot-v0.2 \
  --results results/bright-nontechnical-pilot-v0.2 \
  --audit results/bright-nontechnical-pilot-v0.2/audit \
  --output benchmark/artifacts/bright-nontechnical-pilot-v0.2
```
