# Learned Sparse Batch A on BRIGHT Nontechnical Pilot v0.2

## Status and publication boundary

This is a research-only, no-publish benchmark package. It does not authorize leaderboard publication, redistribution of source text, or export of query IDs, document IDs, rankings, or per-query rows. The canonical materialization remains `data/bright-nontechnical-pilot-v0.2`, whose upstream web-content rights are not established for public redistribution. All four model rows also retain an explicit training-overlap boundary: the BRIGHT queries originate from StackExchange, source-family overlap is plausible for Granite and OpenSearch training, and overlap is not ruled out for BGE-M3. None of these measurements is described as zero-shot.

The protocol was fixed before scoring: models were run in the order Granite 30M Sparse, OpenSearch doc-v2-mini, OpenSearch doc-v3, and BGE-M3; each model ran economics and then psychology. Every cell used one `cuda:0` process, batch size 8, document chunk size 256, maximum length 512, exact SciPy CSR sparse dot product, top 100, and ascending document-ID tie breaking. Data, qrels, exclusion filtering, maximum length, and pruning were not changed in response to results.

## Executable inventory

The tracked inventory contains 13 pinned candidates. The selected pair was chosen for mechanism complementarity and bounded local execution, while doc-v3 and BGE-M3 were retained as formal anchors.

| Key | Status | Revision | License | Routes (query/document) | Vocabulary / dimensions | Max / pruning (q,d) | Dependencies | Snapshot estimate or reason |
|---|---|---|---|---|---:|---:|---|---|
| `granite-30m-sparse` | selected | `ad82b1fd09541c998c8d45045d601c51fdb8a9b7` | Apache-2.0 | neural / neural | RoBERTa BPE / 50,265 | 512 / 50,192 | sentence-transformers, transformers, torch, scipy | 63,318,503 B; compact SPLADE complement |
| `opensearch-doc-v2-mini` | selected | `4af867a426867dfdd744097531046f4289a32fdd` | Apache-2.0 | static lookup / document expansion | BERT WordPiece / 30,522 | 512 / none,none | sentence-transformers, transformers, torch, scipy | 94,631,975 B; static-query complement |
| `opensearch-doc-v3` | anchor | `babf71f3c48695e2e53a978208e8aba48335e3c0` | Apache-2.0 | static lookup / document expansion | WordPiece / 30,522 | 512 / none,none | sentence-transformers, transformers, torch, scipy | about 259 MiB; existing formal anchor |
| `bge-m3` | anchor | `5617a9f61b028005a4858fdac845db406aefb181` | MIT | neural / neural | XLM-R / 250,002 | 512 / none,none | transformers, torch, scipy | about 2.14 GiB actual; existing contextual lexical anchor |
| `opensearch-doc-v2-distill` | deferred | `269e6638b2c4f648996691f6d751495285d8f330` | Apache-2.0 | neural / neural | BERT WordPiece / 30,522 | 512 / none,none | sentence-transformers stack | `opensearch-project/opensearch-neural-sparse-encoding-v2-distill`; about 268 MiB; capacity control redundant with mini |
| `opensearch-multilingual` | deferred | `1e0f096c2b51c234f1d20725c793e1b5b6d556db` | Apache-2.0 | static lookup / document expansion | multilingual WordPiece / 105,879 | 512 / none,none | sentence-transformers stack | about 670 MiB; no fixed multilingual formal track |
| `splade-v3-distilbert` | deferred | `2db06b86d65e316e2ca9907aa1aa8be6f8c4e739` | CC-BY-NC-SA-4.0 | neural / neural | BERT WordPiece / 30,522 | 512 / none,none | sentence-transformers stack | separate non-commercial publication decision required |
| `splade-v3` | excluded | `fdfeceb91d7b9de7985b38addd3ba9f53a59a355` | CC-BY-NC-SA-4.0 | neural / neural | BERT WordPiece / 30,522 | 512 / none,none | sentence-transformers stack | gated and non-commercial |
| `splade-pp-cocondenser-ensemble` | excluded | `49cf4c7b0db5b870a401ddf5e2669993ef3699c7` | CC-BY-NC-SA-4.0 | neural / neural | BERT WordPiece / 30,522 | 512 / none,none | sentence-transformers stack | non-commercial and mechanistically redundant |
| `splade-v2-distil` | excluded | `b5e5ebfb992c31a99ba3be148fb9d841f5c6d6a1` | CC-BY-NC-SA-4.0 | neural / neural | BERT WordPiece / 30,522 | 512 / none,none | transformers, torch, scipy | legacy integration and non-commercial license add no Batch-A value |
| `splade-mini` | deferred | `98cb9db1eb2af3399c0bee30a6edca3fe9c0c057` | MIT in Hub metadata only | neural / neural | BERT WordPiece / 30,522 | 512 / none,none | sentence-transformers stack | about 45.8 MB; repository-level license evidence incomplete |
| `splade-tiny` | deferred | `7391972eac4411e33efff5fad27b886ec97895c0` | MIT in Hub metadata only | neural / neural | BERT WordPiece / 30,522 | 512 / none,none | sentence-transformers stack | about 18.6 MB; repository-level license evidence incomplete |
| `gte-multilingual-sparse` | excluded | `9bbca17d9273fd0d03d5725c7a4b0f6b45142062` | Apache-2.0 | neural / neural | XLM-R / 250,048 | 8,192 / none,none | transformers, torch | pinned config requires external code and `trust_remote_code=True` |

All revisions are 40-character commits. Every inventory row carries a deterministic immutable Hugging Face tree URL derived from its exact repository and revision. The OpenSearch v2-distill pair was independently resolved through the Hugging Face model API as a public, ungated Sentence Transformers repository at the pinned commit. New snapshot resolution rejects Python files, `auto_map`, unexpected files, and pickle-compatible weight fallback. `trust_remote_code` is false. The selected-model caps were 80 MiB and 128 MiB, with a 512 MiB batch download cap and 4 GiB Story disk cap.

## Download and bounded gates

The selected allowlists totaled 157,950,478 bytes. After migration to the standard Hugging Face hub cache, a dry run reported zero download bytes. The frozen aggregate snapshot identities are `392f7930e073c6025d3eefd1caa3eedc01789b3dfcb7d7bc734179ccb511afaa` for Granite and `46c91e3325d59555e2d848890835c276f406d96951c510ba25376e8c08c61458` for OpenSearch mini.

| Model | Snapshot bytes | Gate query/document nnz | Query/document encoding | Truncated q/d | Encoding-observed peak RSS | Peak VRAM | Gate artifact SHA256 |
|---|---:|---|---:|---:|---:|---:|---|
| Granite 30M Sparse | 63,318,503 | 50 / 192,192,192,192 | 8.25 / 377.35 ms | 0 / 4 | 1.67 GB | 0.485 GB | `939376b3a5fd27f4aee8160cdbe2818dd5c16ae9a02b433db4bb7b72e78f5831` |
| OpenSearch doc-v2-mini | 94,631,975 | 104 / 344,313,254,290 | 20.68 / 256.58 ms | 0 / 4 | 1.36 GB | 0.603 GB | `19eaba255e7e73c137d2e8e687c232a02bfec64d537e4cddbfd800aa93f36c9f` |

Both gates used one query and four documents from the bound materialization, verified query/document routing, finite non-negative CSR output, truncation accounting, snapshot identity, resource fields, and deterministic exact ranking.

## Formal quality results

Intervals are deterministic 10,000-sample bootstrap intervals with seed 20260826. These are within-protocol diagnostics, not a general model leaderboard.

| Model | Track | nDCG@10 (95% CI) | MAP@100 | MRR@10 | Recall@10 | Recall@100 |
|---|---|---:|---:|---:|---:|---:|
| Granite 30M Sparse | economics | 0.2595 [0.1989, 0.3239] | 0.2071 | 0.2933 | 0.2952 | 0.6767 |
| Granite 30M Sparse | psychology | 0.2963 [0.2267, 0.3687] | 0.2561 | 0.3277 | 0.3261 | 0.6250 |
| OpenSearch doc-v2-mini | economics | 0.2484 [0.1830, 0.3168] | 0.1964 | 0.2872 | 0.2702 | 0.5422 |
| OpenSearch doc-v2-mini | psychology | 0.2508 [0.1889, 0.3159] | 0.2013 | 0.3045 | 0.2756 | 0.5699 |
| OpenSearch doc-v3 | economics | 0.2444 [0.1820, 0.3091] | 0.1957 | 0.2820 | 0.2637 | 0.5803 |
| OpenSearch doc-v3 | psychology | 0.2386 [0.1794, 0.3024] | 0.1964 | 0.2828 | 0.2702 | 0.5798 |
| BGE-M3 sparse | economics | 0.2806 [0.2184, 0.3457] | 0.2153 | 0.3258 | 0.3185 | 0.6543 |
| BGE-M3 sparse | psychology | 0.3227 [0.2517, 0.3976] | 0.2774 | 0.3917 | 0.3353 | 0.6108 |

The intervals overlap substantially. BGE-M3 sparse has the highest point nDCG@10 in both tracks, while Granite has the highest Recall@100 among learned-sparse rows in both tracks. The small OpenSearch mini is close to or above doc-v3 on several point metrics, but the experiment does not establish superiority.

## Sparsity and resource measurements

Encoding columns are model-call wall time from the raw audit. Cell wall includes snapshot verification, loading, encoding, exact search, metrics, bootstrap, and artifact writes. The RSS field is the process high-water mark observed when each encoding call completed; it is not a separately sampled or identity-bound whole-cell maximum. VRAM is the maximum allocator value observed during encoding. Values are reported as decimal GB. Every formal cell used batch size 8; there was no fallback.

| Model | Track | Mean nnz d/q | Encode d/q (s) | Search (s) | Truncated d/q | Encoding RSS / VRAM (GB) | Cell wall | CSR NPZ / cell bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Granite 30M Sparse | economics | 163.30 / 50.00 | 14.607 / 0.234 | 0.869 | 748 / 2 | 1.89 / 0.899 | 35.93 s | 4,127,932 / 6,092,294 |
| Granite 30M Sparse | psychology | 161.35 / 50.00 | 14.209 / 0.195 | 0.855 | 415 / 0 | 2.00 / 0.899 | 36.85 s | 4,162,464 / 6,032,169 |
| OpenSearch doc-v2-mini | economics | 205.36 / 87.80 | 15.550 / 0.111 | 0.923 | 647 / 0 | 1.66 / 1.105 | 36.11 s | 8,445,534 / 10,409,345 |
| OpenSearch doc-v2-mini | psychology | 203.71 / 85.58 | 15.646 / 0.094 | 0.916 | 387 / 0 | 1.79 / 1.105 | 36.91 s | 8,473,948 / 10,345,183 |
| OpenSearch doc-v3 | economics | 182.70 / 87.80 | 22.535 / 0.116 | 0.928 | 647 / 0 | 1.77 / 1.284 | 41.66 s | 7,354,054 / 9,319,621 |
| OpenSearch doc-v3 | psychology | 182.31 / 85.58 | 22.111 / 0.124 | 0.886 | 387 / 0 | 1.83 / 1.283 | 41.48 s | 7,440,155 / 9,313,447 |
| BGE-M3 sparse | economics | 51.46 / 64.27 | 44.775 / 0.648 | 0.832 | 711 / 2 | 3.99 / 1.249 | 71.70 s | 1,680,112 / 3,640,202 |
| BGE-M3 sparse | psychology | 47.87 / 59.72 | 43.797 / 0.583 | 0.768 | 425 / 0 | 3.96 / 1.248 | 67.55 s | 1,590,463 / 3,460,398 |

Total Batch-A raw output, including replay timing and restricted failure evidence, was 58,782,391 bytes. The largest observed GPU allocation was 1,283,808,768 bytes, far below the 10.5 GiB gate. Maximum observed token counts are tokenizer-specific and therefore not directly comparable: document maxima ranged from 8,406 to 81,252 before fixed 512-token truncation. Query truncation occurred only in the economics Granite and BGE-M3 cells, two queries each.

The OpenSearch v3 and BGE-M3 anchor adapters now enforce the same deterministic 8→4→2→1 CUDA OOM fallback and 10.5 GiB runtime allocation cap as the selected-model adapter. Their provider metadata records requested, attempted, and used batch sizes. The eight frozen formal raw cells predate the added attempted-batch field and were deliberately not rewritten: they record requested and used batch size 8, and no OOM or fallback occurred. New cells and gates include the full attempt sequence.

## Protocol-local baseline diagnostics

The existing BM25 and BGE-M3 long-dense rows use the same materialization, qrels, exclusion policy, exact top-100 evaluation, and bootstrap seed. They represent different retrieval mechanisms and sequence-length settings, so this table is diagnostic and must not be read as a cross-paradigm total ranking.

| Track | Row | nDCG@10 | Recall@100 |
|---|---|---:|---:|
| economics | BM25 | 0.2528 | 0.5048 |
| economics | BGE-M3 long-dense, max 1,024 | 0.2544 | 0.6492 |
| economics | Granite sparse | 0.2595 | 0.6767 |
| economics | BGE-M3 sparse | 0.2806 | 0.6543 |
| psychology | BM25 | 0.1596 | 0.4164 |
| psychology | BGE-M3 long-dense, max 1,024 | 0.2971 | 0.6785 |
| psychology | Granite sparse | 0.2963 | 0.6250 |
| psychology | BGE-M3 sparse | 0.3227 | 0.6108 |

The learned sparse rows materially improve over BM25 on psychology point metrics. Economics is more mixed: BM25, long-dense, and sparse nDCG@10 values are close, while the strongest sparse Recall@100 point estimate is higher. Long-dense retains the highest psychology Recall@100. These observations are protocol-local and not significance claims.

## Failure evidence and interpretation

The failure-case rule was declared before inspecting formal results: select the minimum nDCG@10 query in each cell, with metric ties broken by canonical query ID ascending. The resulting research-only artifact contains eight cases, restricted IDs, per-query metrics, relevant and excluded IDs, and top-100 ranking rows, but no source text. It remains under `results/bright-learned-sparse-batch-a` and is not tracked.

All eight selected cases have zero nDCG@10. Across complete cells, zero-nDCG@10 counts were 48/50 for Granite economics/psychology, 58/53 for OpenSearch mini, 57/54 for OpenSearch v3, and 44/43 for BGE-M3. Zero-Recall@100 counts were respectively 16/22, 29/22, 25/23, and 15/20. A zero score can indicate lexical mismatch, truncation, or failure to rank a labeled positive, but it does not prove that retrieved unjudged documents are irrelevant. BRIGHT supplies positive passage labels rather than exhaustive negative judgments.

Granite's fixed top-k pruning saturates every query at 50 active coordinates and most documents near 192, limiting expansion size. The OpenSearch static-query rows have identical query nnz across mini and v3 because they share the static route; their document-expansion capacity differs. BGE-M3 produces much sparser document rows on average and the strongest point nDCG, but its lower psychology Recall@100 than long-dense illustrates that contextual lexical matching and dense semantic matching fail differently. Heavy fixed-length truncation is a material boundary for every model.

One execution issue was found before completing the matrix: the existing OpenSearch v3 provider returned zero or unknown placeholders for CPU time, RAM, tokenizer class, and maximum observed tokens. The common audit contract was changed to require measured fields, OpenSearch v3 and BGE-M3 were instrumented, focused tests passed, and the affected OpenSearch v3 economics cell was deleted and rerun from source chunks. Its metrics and CSR statistics were unchanged after the resource-only correction.

Raw replay requires an externally supplied manifest SHA256. A raw-local sidecar is checked but is not accepted as its own trust root. Package validation obtains all eight expected hashes from the tracked artifact manifest before opening CSR artifacts. A finalized output can be reused by `run` only after the same external identity and full model-free replay succeed. Incomplete chunks have a narrower boundary: local chunk hashes protect against accidental interruption or corruption, not a coordinated writer. They are rejected by default and can be resumed only with the explicit trusted-recovery flag; a completed result still requires an externally recorded manifest hash and independent package validation.

## Reproduction

Run one cell and immediately replay it without exposing a GPU:

```bash
uv run --no-sync python scripts/bright_learned_sparse_batch_a.py run \
  --model granite-30m-sparse \
  --track economics \
  --data-root data/bright-nontechnical-pilot-v0.2 \
  --output results/bright-learned-sparse-batch-a/granite-30m-sparse/economics \
  --device cuda:0 --batch-size 8 --chunk-size 256

CUDA_VISIBLE_DEVICES='' uv run --no-sync python \
  scripts/bright_learned_sparse_batch_a.py validate \
  --result results/bright-learned-sparse-batch-a/granite-30m-sparse/economics \
  --expected-manifest-sha256 e8a8f9ac2aa59e2e819b64dbc3cc189b04b880330658aedf5d9ad5b218daeca1 \
  --data-root data/bright-nontechnical-pilot-v0.2
```

Use the same command in the fixed model and track order. Standalone `aggregate` and `failure-cases` require one repeated `--expected-manifest-sha256` for each repeated `--result`. Initial `finalize` requires the same external identities; later finalization locks to the existing tracked package and rejects raw identity drift. `validate-package` reads the locked identities from its manifest. These commands reconstruct exact CSR retrieval from saved query CSR, ordered IDs, and document chunks; they do not load a model or require a GPU.

The tracked package is useful for comparing these four pinned learned-sparse configurations on these two fixed BRIGHT tracks. It does not support claims about public leaderboard standing, zero-shot generalization, multilingual quality, production latency, exhaustive relevance, or untested sequence lengths and pruning settings.
