# Learned Sparse Retrieval Family and Rendered-Page Dataset Minispec - 2026-07-21

Dispatch:
.perpetuum/modern-embedding-leaderboard/state/dispatch_3-1784648033-2_execute.md

Selected item:
tasks/learned-sparse-retrieval-dataset-scout

Unique session:
meb-modern-embedding-leaderboard-3-1784648033-2-sparse-scout-8c7e41b29d6f

Status: research complete. No benchmark, provider API, model inference, model
or dataset download, external index deployment, package change, Hugging Face
upload, commit, or push was performed.

## Decision

Proceed with a provider-neutral learned-sparse benchmark family, but split the
next work into a low-risk text compatibility smoke and a separate rendered-page
dataset fixture.

The recommended order is:

1. Use
   opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill
   for the first one-query/four-document no-publish smoke. It is the smallest
   released candidate, uses standard Sentence Transformers sparse modules,
   needs no custom remote code, and fits the current host and environment.
2. Keep naver/splade-v3-distilbert as the classic vocabulary-space SPLADE
   control for a later text matrix. Its non-commercial license means it must
   remain an internal, no-publish control unless redistribution and use terms
   are explicitly accepted.
3. Add V-SPLADE Efficient, then Quality, only after a dedicated dependency and
   custom-source review. Both checkpoints fit a 12 GB GPU, but the current repo
   has sentence-transformers 5.2.3 while the Hub route requires 5.6.0 or later.
4. Treat naver/splade-code-06B as a separate code-retrieval track. It is a real
   released checkpoint, not paper-only, but it uses custom code, a
   151,936-dimensional output, a non-commercial license, and potentially large
   token-by-vocabulary logits.
5. Keep SPLARE 2B/7B as paper and venue evidence only. No official runnable
   checkpoint, exact revision, download size, model license, or loading recipe
   was located.
6. Defer naver/splade-code-8B on this host. The Hub repository contains a LoRA
   adapter, but its custom loader also downloads the Qwen3-8B base whose five
   BF16 shards total 16,381,516,776 bytes before the adapter and runtime state.

The rendered-page dataset should not be a repackaged public leaderboard. The
proposal below creates self-authored fictional operations binders, renders the
same pages into canonical-text, OCR-text, and page-image routes, and freezes
qrels and hard negatives before model scores are inspected. This makes lexical,
semantic, visual-layout, and hybrid behavior auditable on the same underlying
evidence.

## Evidence Labels

- CONFIRMED means checked in the current repository or a current primary source
  on 2026-07-21.
- SOURCE-REPORTED means a paper or model card reports the value, but it was not
  reproduced locally.
- UNRESOLVED means the source did not establish the fact or conflicting source
  conventions remain.
- PROPOSAL means this document recommends the design; it was not implemented or
  run in this dispatch.
- GATE means a required condition before a real run or publication.

## Evidence Method and Local Constraints

CONFIRMED local state:

- Four NVIDIA GeForce RTX 3080 Ti GPUs are present. Each reports 12,288 MiB
  total VRAM, approximately 4 MiB used, compute capability 8.6, and zero
  utilization during this research.
- /data2 has approximately 288 GB free and is 92% used.
- The current environment contains numpy 2.4.3, scipy 1.17.1, torch 2.10.0,
  torchvision 0.25.0, transformers 5.3.0, sentence-transformers 5.2.3, and
  huggingface-hub 1.6.0.
- Pyserini and faiss-cpu are not installed.
- The installed Sentence Transformers exposes SparseEncoder plus
  encode_query() and encode_document().

The first anonymous AnySearch batch reported exhausted free quota and returned
an auto-registration response. No credential from that response was used,
saved, or persisted. All durable claims in this note were re-collected from
direct primary sources.

Normal local resolution of huggingface.co remained unavailable. For read-only
metadata and small text/config files, this research refreshed the current
public A record through Google DNS and used a process-local curl resolution
override while preserving the official huggingface.co TLS hostname. It queried
Hub API metadata, model cards, JSON configs, and Python source only. It did not
request model weight bodies or dataset contents.

Hub usedStorage and current snapshot download bytes are not always the same
quantity. Where possible, this note reports both the current LFS/Xet file size
and usedStorage rather than treating either one as a universal download total.

## Feasibility Ranking

### Rank 1 - OpenSearch document v3 distill

Disposition: recommended first smoke and first inference-free text row.

Release maturity: high.

Machine fit: high; CPU-capable and trivial for a 12 GB GPU.

Confirmed facts:

- Model id:
  opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill
- Current revision:
  babf71f3c48695e2e53a978208e8aba48335e3c0
- Public and ungated.
- Apache-2.0 model-card license.
- 66,985,530 float32 parameters from Hub safetensors metadata.
- Core model.safetensors:
  267,954,768 bytes.
- Static query lookup model:
  122,168 bytes.
- Hub usedStorage:
  268,076,936 bytes.
- Output dimensionality:
  30,522 vocabulary dimensions.
- Query route:
  tokenizer plus static learned/IDF lookup; no query-model forward.
- Document route:
  DistilBERT masked-LM logits, max pooling, and the v3 nested-log activation.
- Sentence Transformers config was generated with sentence-transformers 5.0.0,
  transformers 4.50.3, and torch 2.6.0.
- Prompts are explicitly empty for both query and document routes.
- No repository custom Python and no trust_remote_code requirement were found.
- Model-card average across the listed BEIR subset:
  nDCG@10 0.517 and FLOPS 1.8.
- The associated paper evaluates OpenSearch inverted indexing and reports a
  related l0-mask-plus-activation setting at approximately 275 active document
  dimensions and FLOPS 2.13. That paper value is family evidence, not a
  reproduced exact density measurement for this pinned Hub snapshot.

Index path:

- Tiny smoke: SparseEncoder similarity or scipy CSR dot product.
- Fixture/pilot: scipy CSR plus deterministic exact sparse dot product.
- Larger scale: Lucene/OpenSearch neural sparse, Pyserini impact indexing, PISA,
  or another explicitly pinned inverted index.

Unresolved or gated:

- Exact query and document nonzero distributions on the proposed rendered-page
  dataset.
- Compatibility of this exact revision with the current
  sentence-transformers 5.2.3 must be proven by the bounded smoke.
- Apache-2.0 covers the released model path, but the many listed training
  datasets have separate source terms. Do not redistribute training data or
  infer dataset rights from the model license.

Why it belongs:

It isolates a production-relevant inference-free query regime with a small
download, standard code path, vocabulary interpretability, and low machine
risk. It is the cleanest bridge from this repo's current dense-only contract to
a real sparse result.

### Rank 2 - SPLADE v3 DistilBERT

Disposition: classic vocabulary-space sparse control after the first smoke.

Release maturity: high.

Machine fit: high.

Confirmed facts:

- Model id:
  naver/splade-v3-distilbert
- Current revision:
  2db06b86d65e316e2ca9907aa1aa8be6f8c4e739
- Public and ungated.
- License:
  CC BY-NC-SA 4.0.
- Core pytorch_model.bin:
  267,984,305 bytes.
- Hub usedStorage:
  803,893,985 bytes. This is not treated as the current allow-listed snapshot
  size because the only LFS weight in the current revision is approximately
  268 MB.
- DistilBERT, six layers, 30,522 output dimensions, maximum sequence length
  512, and dot-product similarity.
- Standard Sentence Transformers MLMTransformer plus SpladePooling path.
- No model repository custom Python and no trust_remote_code requirement.
- No query or document text prompt is specified.
- Source-reported MS MARCO MRR@10 is 38.7 and BEIR-13 average nDCG@10 is 50.0.

Index path:

- Same CSR and inverted-index routes as the OpenSearch row.
- Unlike the inference-free OpenSearch row, both query and document normally
  require neural encoding.

Unresolved or gated:

- Typical nnz and FLOPS for this exact pinned checkpoint on the proposed data.
- The non-commercial ShareAlike license is unsuitable as a default public
  leaderboard dependency without explicit policy review.
- A future allow-listed download should use a roughly 300 MB ceiling based on
  the pinned current weight, not the larger historical usedStorage number.

Why it belongs:

It is a mature, conventional SPLADE control with standard code and the same
30,522-dimensional vocabulary space as the small OpenSearch candidate. It
separates the effect of inference-free query lookup from a normal neural query
encoder.

### Rank 3 - V-SPLADE Efficient

Disposition: first visual learned-sparse candidate after an isolated dependency
and source-review item.

Release maturity: high, but integration maturity in this repo is low.

Machine fit: high for short-page inference; dependency fit is currently
blocked.

Confirmed facts:

- Model id:
  naver/v-splade-efficient
- Current revision:
  ab0c2260c6d78bcb7d05076a9407a71f55d57eb1
- Official code:
  naver/v-splade at commit
  3f9773785943bd6e62a9e12c92fa3e2a68c3f477
- Public, ungated, Apache-2.0 model and code paths.
- Core model.safetensors:
  660,070,138 bytes.
- Static query lookup model:
  201,552 bytes.
- Hub usedStorage:
  661,963,663 bytes.
- Hub safetensors metadata counts 330,010,817 BF16 parameters.
- The current model card and paper describe a 0.25B or 250M visual encoder.
  The 250M and 330M figures use different counting conventions; this note does
  not collapse them into one exact parameter claim.
- Output dimensionality:
  50,368 vocabulary dimensions.
- Query route:
  inference-free Li-LSR/static lookup.
- Document route:
  page image or text through a ModernVBERT visual/text backbone and SPLADE max
  pooling.
- Hub model card requires sentence-transformers with image support at 5.6.0 or
  later, transformers 5.3.0 or later, and trust_remote_code=True.
- Current repository environment has sentence-transformers 5.2.3, so the Hub
  path is not currently dependency-compatible.
- The pinned Hub config was generated with sentence-transformers 5.6.0 and
  transformers 5.13.0. Meeting the documented lower bound does not prove exact
  compatibility with that newer generated stack.
- The reference code repository instead pins transformers 4.57.6, torch 2.8.0,
  and flash-attn 2.8.3. Its training/reference environment must not be mixed
  silently with the newer Hub SparseEncoder route.
- The Hub custom source imports torch and Transformers ModernVBERT classes and
  provides the custom model and static query module. It still requires a pinned
  revision review before execution.
- The model card's sample page produces 552 active dimensions.
- The paper reports approximately 300 active passage dimensions for its
  efficient trained setting and a related FLOPS value around 1.10.
- The official repository table reports ViDoRe v2 nDCG@5 0.4658 and average
  FLOPS 0.98 for Efficient. These are source-reported benchmark metrics, not
  results for this proposed dataset.

Index path:

- Official encoding code writes scipy CSR sparse_docs.npz plus aligned
  doc_ids.json.
- The paper uses PISA at production scale and reports full and two-stage sparse
  search.
- Tiny fixture and pilot do not need PISA; CSR exact search is sufficient.

Unresolved or gated:

- Upgrade or isolate sentence-transformers 5.6.0 or later in a dedicated
  follow-up without destabilizing the current dense provider.
- Review both pinned custom Python files before model loading.
- Prove the page preprocessing route, returned sparse type, exact nnz, VRAM,
  and latency on a 12 GB GPU.
- Review the source licenses of any external training/evaluation dataset before
  using its rows. The model card notes that the ColPali training collection has
  multiple source-specific licenses.

Why it belongs:

It is the only investigated released row that directly indexes rendered page
images into interpretable vocabulary-space sparse vectors while keeping the
query path inference-free.

### Rank 4 - V-SPLADE Quality

Disposition: visual quality/efficiency comparison after Efficient works.

Release maturity: high.

Machine fit: high; same dependency and source-review gates as Efficient.

Confirmed facts:

- Model id:
  naver/v-splade-quality
- Current revision:
  99bdc93f42460e595b2fb1e78b96edd44e898441
- Public, ungated, Apache-2.0.
- Same core weight size, static query weight size, Hub usedStorage, parameter
  metadata, output dimensionality, and custom loading family as Efficient.
- Official repository reports ViDoRe v2 nDCG@5 0.4990 and average FLOPS 1.51.
- The paper describes Quality as the higher-recall, higher-cost operating point.

Unresolved or gated:

- Typical nnz on the proposed pages.
- Whether Quality's extra postings and latency are justified after Efficient,
  OCR BM25, text sparse, dense, and hybrid controls are measured.
- All V-SPLADE dependency, remote-code, provenance, and preprocessing gates.

Why it belongs:

It provides a within-family quality-versus-index-cost comparison without
changing the query and output representation contract.

### Rank 5 - SPLADE-Code 0.6B

Disposition: separate code-retrieval follow-up; do not mix into the first
rendered-page run.

Release maturity: medium-high.

Machine fit: probable for a one-query, short-sequence, batch-size-one smoke;
not yet proven.

Confirmed facts:

- Model id:
  naver/splade-code-06B
- Current revision:
  e53a0b8bd312d83955598a392dc826b3fc4028f7
- Public and ungated.
- License:
  CC BY-NC-SA 4.0.
- Hub safetensors parameter count:
  596,049,920 BF16 parameters.
- Core model.safetensors:
  1,192,135,096 bytes.
- tokenizer.json:
  11,422,650 bytes.
- Hub usedStorage:
  1,203,557,746 bytes.
- Output dimensionality:
  151,936 Qwen vocabulary dimensions.
- The model-card example reports 1,231 active dimensions and sparsity ratio
  0.9918979 for its SQL input, or approximately 0.81 percent nonzero.
- The checkpoint has custom splade.py and utils.py and requires
  trust_remote_code=True.
- The custom model changes Qwen3 to bidirectional attention with
  is_causal=False, emits full-vocabulary logits, uses ReLU plus max pooling,
  requests BF16, and falls back from Flash Attention 2 to SDPA when
  flash-attn is unavailable.
- config_sentence_transformers.json contains no prompts.
- The custom encode() function's prompt_type selects optional query/document
  top-k pruning. It does not prepend a task instruction.
- Default custom encode settings are max_length 1024 and batch_size 8; those
  defaults are too aggressive for the first 12 GB compatibility smoke because
  the full 151,936-wide logits dominate temporary memory.

Index path:

- scipy CSR or a standard impact/inverted index after converting nonzero
  vocabulary weights.
- A code benchmark needs code-specific token and query provenance; it should not
  borrow rendered-page qrels.

Unresolved or gated:

- Exact model-card revision compatibility with transformers 5.3.0 and
  sentence-transformers 5.2.3.
- Peak VRAM and latency for batch size 1, maximum length 128.
- License acceptance for internal and eventual public use.
- Exact source dataset provenance and contamination against any future code
  task.
- Whether explicit query/document top-k pruning is needed for serving cost.

Why it belongs:

It is evidence that learned sparse retrieval is broader than visual documents
and classic BERT vocabulary models. It should become a code-specific family
row only after the smaller text contract works.

### Rank 6 - uniCOIL historical control

Disposition: optional historical control; defer by default.

Release maturity: medium.

Machine fit: high for the encoder, but dependency and provenance fit are low.

Confirmed facts:

- Model id:
  castorini/unicoil-msmarco-passage
- Current revision:
  a9379ff729899cf1255960e604496c1a638346ce
- Core pytorch_model.bin:
  438,008,073 bytes.
- Hub usedStorage:
  875,970,453 bytes.
- BERT-base vocabulary size:
  30,522.
- Pyserini contains an Apache-2.0 UniCoilEncoder implementation, impact
  indexing, and reproducible MS MARCO recipes.
- The current repo does not have Pyserini installed.

Unresolved or gated:

- The Hub model has no model card and no declared model license in current
  metadata. Pyserini's code license does not establish the weight license.
- The model is from 2021 and is lower priority than a current SPLADE control.
- Full Pyserini/Lucene installation is unnecessary for the first fixture.

Why it is deferred:

It is useful historical context for learned term weighting, but it adds an old
custom encoder path and unresolved weight provenance without increasing the
coverage of the first matrix.

### Rank 7 - SPLARE 2B and 7B

Disposition: paper/venue evidence only.

Release maturity: paper-level.

Machine fit: unresolved for 2B and poor for single-GPU 7B.

Confirmed facts:

- Paper:
  arXiv 2603.13277, Learning Retrieval Models with Sparse Autoencoders.
- arXiv metadata identifies the work as ICLR 2026.
- The paper introduces latent sparse retrieval using frozen SAE encoders and a
  LoRA-tuned LLM backbone.
- The main SPLARE model uses Llama Scope at layer 26, a 131k-wide SAE, and is
  described as 7B parameters including SAE parameters.
- The lighter SPLARE-2B is trained at layer 6.
- Default reported inference pruning is top 40 features for queries and top
  400 for documents.
- The paper reports approximately 5 ms per query for Seismic retrieval over
  8.8M MS MARCO documents at top-k 40/400, excluding model inference.
- The paper reports strong multilingual results and notes weaker relative
  behavior on code tasks.

Artifact search result:

- No official SPLARE model appeared in current Hugging Face model search.
- No official GitHub implementation or checkpoint link appeared in the paper.
- No official model revision, download bytes, source tree, dependency lock,
  trust_remote_code requirement, prompt contract, or model license was found.

Unresolved or gated:

- Every runnable-artifact fact.
- 2B host fit cannot be evaluated without the exact backbone, SAE files,
  checkpoint precision, and loader.
- A nominal 7B BF16 model already exceeds one 12 GB GPU before useful runtime
  state; multi-GPU/offload would be a separate resource decision.

Why it is deferred:

The latent, multilingual representation is strategically important, but a
paper claim cannot enter a reproducible benchmark registry as a model.

### Rank 8 - SPLADE-Code 8B

Disposition: released source evidence, deferred on this host.

Release maturity: medium.

Machine fit: poor for one 12 GB GPU.

Confirmed facts:

- Model id:
  naver/splade-code-8B
- Current revision:
  fe9fb2fc9fd930187ede95085cd189c7dc5d55a4
- Public, ungated, CC BY-NC-SA 4.0.
- The repository stores a 349,244,256-byte LoRA adapter and reports
  360,666,906 bytes usedStorage.
- Custom source explicitly loads Qwen/Qwen3-8B and applies the adapter.
- The current Qwen3-8B base revision reports 8,190,735,360 BF16 parameters.
- Its five current BF16 weight shards total 16,381,516,776 bytes.
- Base shards plus the SPLADE adapter total 16,730,761,032 bytes before
  tokenizer files, cache duplication, activations, and full-vocabulary logits.
- Output dimensionality remains 151,936.
- The model-card example reports 1,120 active dimensions for the sample query.
- The loader requires custom code plus PEFT.

Unresolved or gated:

- Multi-GPU loading, quantization, adapter/base license composition, and actual
  sparse inference memory.

Why it is deferred:

The 0.6B release already covers the code model family. The 8B route adds
multi-GPU/offload complexity and a download above the selected item's safe
scope without providing a new task contract.

## Baseline and Model Matrix for the Dataset Family

The first score-bearing matrix should be staged. Not every row belongs in the
fixture or first pilot.

### Required lexical controls

1. Canonical-text BM25
   - Input: self-authored canonical page text.
   - Purpose: upper lexical control without OCR errors.
   - Query model cost: zero.
   - Report tokenizer, stemming, stopword, k1, b, postings, and index bytes.

2. OCR-text BM25
   - Input: pinned OCR output from each rendered page variant.
   - Purpose: real operational baseline and explicit OCR-cost route.
   - Report OCR engine/revision, preprocessing time, CER/WER, and failures.

### Required learned-sparse controls

3. OpenSearch document v3 distill
   - First inference-free learned-sparse row.
   - Same text route as OCR/canonical BM25.
   - Query lookup has no neural forward.

4. SPLADE v3 DistilBERT
   - Neural-query vocabulary-space control.
   - Internal/no-publish until license policy is approved.

### Required dense control

5. Existing repo dense text provider
   - Start with a current registered local text model such as bge-m3-local or a
     separately accepted small open-weight dense checkpoint.
   - Use the same canonical/OCR text and same qrels.
   - Do not claim that the existing sentence_transformers provider exposes
     BGE-M3's sparse mode; it currently returns dense numpy arrays only.

### Required hybrid control

6. Pre-registered rank fusion
   - Prefer reciprocal-rank fusion for the first pilot because it avoids score
     calibration across BM25, sparse dot products, and cosine similarity.
   - Register k and component weights before test scores.
   - Report component and fusion latency separately.

### Visual-document rows

7. V-SPLADE Efficient
   - Direct page-image learned sparse row.
   - Add only after its dependency/source smoke passes.

8. V-SPLADE Quality
   - Within-family quality/cost comparison.

9. Optional direct visual dense control
   - A pinned, licensed, machine-fit visual page retriever.
   - It is not required for the first text/sparse contract smoke.

### Separate tracks

10. SPLADE-Code 0.6B versus code BM25 and a dense code embedding model
    - Separate code corpus and qrels.
    - Do not merge code metrics into rendered-page scores.

11. SPLARE 2B/7B
    - Add only after official runnable artifacts and licenses are pinned.

### Task-design evidence only

LIMIT+ in informagi/Complex-Set-Compositional-IR at commit
0a4105a328474d4a4c58b8e4fc613ec05c59fc22 contains generation and evaluation
code plus data for set-compositional retrieval, but GitHub reports no declared
license and the root repository has no license file. It is useful evidence for
constructing partial-match and multi-constraint hard negatives. No row, script,
or dataset content should be copied or ingested without a license and
provenance decision.

## Missing Repository Abstractions

CONFIRMED current dense contract:

- EmbeddingResult.embeddings is annotated and used as numpy.ndarray.
- EmbeddingProvider.embed() maps a flat list of EmbeddingInput to one dense
  array result.
- embed_with_cache() stores and loads .npy through numpy.save and numpy.load.
- Cache-hit dimensionality assumes dense ndarray shape and ndim.
- SentenceTransformersProvider loads SentenceTransformer, not SparseEncoder,
  always passes trust_remote_code=True, performs a dense test encode at load,
  normalizes outputs, and applies dense MRL truncation.
- Retrieval tasks call dense cosine or dot-product utilities that accept
  numpy.ndarray.
- Result JSON conversion understands numpy arrays but not scipy sparse matrices,
  torch sparse tensors, posting dictionaries, or sparse index evidence.
- ModelSpec has one dimensions field but no representation kind, query route,
  nnz policy, sparse activation, or index backend contract.

The existing grouped-chunk protocol solves ordered document grouping for late
chunking but still stores each chunk embedding as a dense numpy array. It does
not solve sparse representation or sparse indexing.

PROPOSAL: do not overload EmbeddingResult with an object array or silently
dense-materialize vocabulary vectors. Add explicit sparse types in a later
implementation item.

Minimum missing types:

1. SparseEmbeddingBatch
   - scipy.csr_matrix values with shape n_items by dimensions.
   - aligned item ids.
   - dimensions, nnz total, nnz per row, dtype, and representation vocabulary.
   - optional decoded top terms for audit only.

2. SparseEmbeddingResult
   - embeddings: SparseEmbeddingBatch.
   - model/provider/revision.
   - query_route:
     neural, static_lookup, tokenizer_idf, or none.
   - document_route.
   - latency, token usage, device, peak VRAM, and metadata.

3. SparseEmbeddingProvider protocol
   - encode_sparse_query().
   - encode_sparse_documents().
   - explicit no-neural-query capability.
   - no assumption that MRL truncation is meaningful.

4. SparseIndex protocol
   - build(corpus batch, ids).
   - search(query batch, k).
   - deterministic exact CSR implementation first.
   - future backend adapters for Lucene/OpenSearch, Pyserini/PISA, or Milvus.

5. SparseIndexResult
   - ranked ids and raw scores.
   - backend and exact/approximate mode.
   - build time, query time, index bytes, postings, pruning policy, and version.

6. Sparse cache format
   - .npz CSR plus a JSON manifest containing ids, dimensions, dtype, model
     revision, route, and fingerprint.
   - Cache keys must include representation kind, query/document route,
     tokenizer revision, sparse activation, top-k/pruning, and index version.

7. Result JSON contract
   - Never serialize full 30k to 150k vectors into JSONL.
   - Persist metrics and sparse diagnostics plus references and checksums for
     local CSR/index artifacts.

The first sparse implementation should use scipy CSR because scipy is already
installed and a tiny fixture does not justify Pyserini, OpenSearch, Milvus, or
another service. External index integration should follow only after exact
local scores and serialized identities are stable.

## Original Rendered-Page Dataset Family

Proposed task id:
rendered_page_sparse_retrieval

Proposed fixture version:
rendered-page-sparse-fixture-v0

Proposed pilot version:
rendered-page-sparse-pilot-v0

Public status:
publish: false until all pilot gates pass.

### Intended real scenario

The scenario is retrieval over operational binders: policy pages, maintenance
cards, forms, release notes, incident procedures, and reference sheets that
users search by intent rather than by copying the page wording.

Real systems often have:

- a mixture of born-digital PDFs and scans;
- tables, forms, callout boxes, two-column layouts, stamps, footers, and icons;
- exact identifiers and version numbers that semantic models may blur;
- paraphrased user questions that BM25 may miss;
- OCR corruption that changes identifiers, punctuation, or reading order; and
- a serving choice among OCR BM25, learned sparse, dense, direct page-image,
  and hybrid retrieval.

Mainstream text embedding leaderboards usually evaluate normalized text rather
than the same evidence through canonical, OCR, and page-image routes. Visual
document leaderboards are valuable but do not by themselves provide this
self-authored paired route, pre-registered OCR corruption, exact identifier
slices, and complete quality-plus-index-cost contract. This dataset is designed
to measure that missing operational choice, not replace MTEB, BEIR, or ViDoRe.

### Corpus and page construction

All content is self-authored and fictional.

Fixture:

- 6 fictional organizations.
- 4-page binder per organization.
- 24 canonical pages.
- 30 queries.
- At least 3 curated hard negatives per query.
- Two rendered variants per page:
  clean digital and deterministic degraded scan.
- 48 page-image assets total.
- Split:
  fixture_only.
- License status:
  not_for_publication until authorship and repository licensing are approved.

Pilot:

- 30 fictional binders.
- 4 pages per binder.
- 120 canonical pages.
- 180 queries.
- 18 train/development binders, 6 validation binders, and 6 held-out test
  binders.
- Splits occur by binder/template family, never by query paraphrase.
- At least 4 curated hard-negative links per query.

Page source should be a deterministic structured representation such as
self-authored SVG/HTML plus a canonical JSON scene graph. A pinned renderer
produces page PNG and PDF artifacts. Every render records:

- renderer name and exact version;
- bundled font file, license, and SHA-256;
- source template id and revision;
- deterministic random seed;
- page size and DPI;
- image and PDF SHA-256;
- canonical text and text-span coordinates; and
- region ids for tables, headings, body text, stamps, diagrams, and form cells.

No page may contain copied customer material, crawled PDF text, real credentials,
real contact details, or a real organization's internal policy.

### Render and OCR conditions

Each canonical page receives controlled variants:

1. clean_digital
   - Native resolution, straight page, high contrast.

2. mild_scan
   - Small skew, blur, JPEG artifacts, shadow, and limited speckle.

3. layout_stress
   - Two columns, table headers, callout boxes, repeated footers, or a stamp
     overlapping non-evidence text.

The fixture needs clean_digital and one degraded route. The pilot may add the
third route.

OCR is a recorded preprocessing system, not hidden dataset generation. Each OCR
row records engine, revision, language pack, preprocessing, runtime, raw text,
normalized text, token boxes when available, CER, WER, and reading-order
errors. Canonical source text remains the ground truth and must not be silently
substituted for OCR text in an OCR baseline.

### Query authoring

Queries are written from intent records before the final page wording and hard
negative selection are exposed to the query author.

Required query families:

1. semantic_paraphrase
   - The query uses no distinctive phrase from the evidence.
   - Tests BM25 lexical gaps and learned expansion/dense behavior.

2. exact_identifier
   - Version, form id, error code, date, or component name is essential.
   - Tests dense blurring and OCR character corruption.

3. mixed_semantic_lexical
   - Requires both a paraphrased action and one exact entity or identifier.
   - Expected to motivate hybrid retrieval.

4. layout_grounded
   - The answer depends on a table row/column, form label/value pair, heading,
     or visually scoped callout.
   - Tests OCR reading order and direct visual page encoding.

5. scope_or_version
   - Several pages share wording but differ by product, region, date, or
     revision.
   - Prevents keyword-only success.

6. multi_constraint
   - The correct page satisfies two constraints while negatives satisfy only
     one.
   - Inspired by set-compositional task design, but entirely self-authored.

Every query records whether lexical overlap is intentionally low, whether an
exact identifier is required, whether layout is required, and which retrieval
families the hard negatives are meant to stress. These fields are diagnostic
labels, not guarantees about a model result.

### Qrels and ground truth

Ground truth is page-level with evidence-region support.

Each query has:

- one or more relevance-2 pages containing the complete answer;
- optional relevance-1 pages containing necessary but incomplete support;
- one or more evidence region ids;
- canonical character spans for textual evidence;
- page-image bounding boxes for visual evidence; and
- an adjudication note explaining why each positive is sufficient.

Qrels are frozen before any evaluated-model score is inspected.

For a rendered variant, the page id remains stable and the variant id changes.
Text-route systems rank page ids from canonical or OCR text. Visual systems rank
the corresponding page-image variant. Metrics collapse variants only according
to the registered run protocol; a system cannot receive duplicate credit for
retrieving multiple render variants of the same canonical page.

### Hard negatives

Every query has at least one negative from each applicable family:

1. lexical_collision
   - Shares exact query words but applies to the wrong action or object.

2. semantic_neighbor
   - Describes the same broad intent but lacks the required identifier or
     condition.

3. identifier_collision
   - Differs by one digit, letter, revision, or date.

4. layout_collision
   - Contains the same labels and values in the wrong table row, column, or
     form section.

5. ocr_collision
   - OCR makes a negative token look closer to the query or corrupts the gold
     identifier.

6. partial_constraint
   - Satisfies one of two required constraints.

Hard negatives are selected using canonical evidence and human review, not by
mining the evaluated model's scores. After selection, baseline scores may
diagnose difficulty but may not change test qrels or negatives.

### Dataset schemas

Recommended JSONL records follow.

Page:

    {
      "page_id": "page_forge_001_03",
      "document_id": "binder_forge_001",
      "split": "fixture_only",
      "page_number": 3,
      "title": "Badge Return Procedure",
      "source_type": "self_authored_fictional",
      "template_id": "procedure_card_v1",
      "source_revision": "rendered-page-sparse-fixture-v0",
      "canonical_text": "...",
      "canonical_text_sha256": "...",
      "regions": [
        {
          "region_id": "region_forge_001_03_step2",
          "role": "evidence",
          "char_start": 418,
          "char_end": 512,
          "bbox_xywh": [96, 544, 1160, 180]
        }
      ],
      "license_status": "not_for_publication"
    }

Rendered artifact:

    {
      "page_id": "page_forge_001_03",
      "variant_id": "mild_scan_v1",
      "image_path": "pages/page_forge_001_03__mild_scan_v1.png",
      "image_sha256": "...",
      "pdf_sha256": "...",
      "renderer": "pinned-renderer-name",
      "renderer_version": "...",
      "font_manifest_sha256": "...",
      "seed": 30103,
      "width": 1275,
      "height": 1650,
      "dpi": 150
    }

OCR:

    {
      "page_id": "page_forge_001_03",
      "variant_id": "mild_scan_v1",
      "ocr_system": "pinned-ocr-name",
      "ocr_revision": "...",
      "raw_text": "...",
      "normalized_text": "...",
      "cer": 0.073,
      "wer": 0.118,
      "reading_order_error_count": 1,
      "latency_ms": 184.2
    }

Query:

    {
      "query_id": "q_forge_badge_001",
      "split": "fixture_only",
      "text": "Where should a departed contractor leave an expired building pass?",
      "family": "mixed_semantic_lexical",
      "requires_exact_identifier": false,
      "requires_layout": false,
      "low_lexical_overlap": true,
      "evidence_region_ids": ["region_forge_001_03_step2"]
    }

Qrel:

    {
      "query_id": "q_forge_badge_001",
      "page_id": "page_forge_001_03",
      "relevance": 2,
      "evidence_region_id": "region_forge_001_03_step2"
    }

Hard negative:

    {
      "query_id": "q_forge_badge_001",
      "page_id": "page_harbor_004_02",
      "negative_family": "semantic_neighbor",
      "reason": "Describes badge renewal, not offboarding return.",
      "false_negative_review": "pass"
    }

### Quality metrics

Primary metric:

- page_ndcg@10

Required full-corpus metrics:

- page_recall@1, @5, and @10;
- page_mrr;
- page_ndcg@5 and @10;
- evidence_region_recall@1, @5, and @10;
- exact_identifier_recall@1; and
- hard_negative_pair_win_rate.

Required paired diagnostics:

- ocr_delta_page_ndcg@10:
  OCR-text route minus canonical-text route for the same model.
- render_noise_delta_page_ndcg@10:
  degraded page-image route minus clean page-image route.
- sparse_minus_bm25_page_ndcg@10.
- dense_minus_sparse_page_ndcg@10.
- hybrid_gain_page_ndcg@10:
  hybrid minus the best registered component.
- query_inference_free_delta:
  compare quality and latency between static/tokenizer query lookup and neural
  query encoding where model families make that comparison meaningful.

Do not interpret deltas as causal when both model and input modality change.
V-SPLADE versus OCR BM25 is a system comparison; same-model or same-input
paired rows should be labeled separately.

### Cost and serving metrics

Every run records:

- document preprocessing ms/page;
- OCR or caption extraction ms/page and external cost if applicable;
- document encoding ms/page and pages/second;
- query encoding ms/query on CPU and GPU where relevant;
- query_model_forward_count;
- peak CPU RSS and peak GPU MiB;
- output dimensions;
- mean, median, p95, and max nnz for queries and documents;
- sparse vector bytes before indexing;
- index build wall time and peak memory;
- index bytes total and bytes/page;
- number of postings and postings/page;
- exact or approximate search mode;
- p50, p95, and p99 search ms/query;
- dense vector bytes/page for dense controls; and
- fusion overhead plus component latency for hybrid rows.

V-SPLADE's OCR-free document route must still report page rendering and image
decode/preprocessing cost. OCR BM25 must report OCR cost rather than treating
OCR text as free. Inference-free sparse rows must report that query-model
forward count is zero, not merely low latency.

### Required slices

- query family;
- canonical versus OCR versus page-image route;
- clean versus mild scan versus layout stress;
- OCR CER and WER bucket;
- single-column, two-column, table, form, callout, and diagram layout;
- exact identifier present/absent;
- lexical overlap bucket;
- evidence text length;
- page text-density bucket;
- gold/negative identifier edit distance;
- same-binder versus cross-binder negative;
- visual-layout-required versus text-sufficient;
- query inference-free versus neural query route; and
- model/index pruning policy.

### Leakage, privacy, license, and toy-dataset gates

Reject an item, split, run, or publication when:

- source authorship or asset/font license is missing;
- a real customer, employee, credential, URL, phone number, email, or private
  identifier appears;
- templates, normalized paragraphs, or binder families cross score-bearing
  splits;
- queries are paraphrases of each other across splits;
- query authors copy distinctive page phrases outside the exact-identifier
  slice;
- qrels or hard negatives change after evaluated-model scores are inspected;
- a hard negative is a valid answer under full-page review;
- OCR text is silently substituted with canonical text;
- a direct visual model sees a different canonical page than text controls;
- rendered variants cannot be reconstructed from pinned source and seed;
- model, code, dataset, font, or OCR license is absent from provenance;
- a no-publish fixture score appears in Dataset or Space public outputs; or
- the task can be solved from template, footer, file name, page number, or
  organization-name leakage rather than evidence.

Toy-risk checks:

- BM25 lexical shortcut audit.
- Character and token overlap audit.
- Template classifier audit.
- Duplicate and near-duplicate page/query hashes.
- Blind positive/negative review.
- Score-blind hard-negative review.
- Identifier perturbation validation.
- Random and trivial heuristic baselines.

The fixture is valid only for schema, adapter, index, metric, and serialization
tests. It cannot support model quality claims.

### PASS, BLOCKED, and ABANDON criteria

Fixture PASS:

- all 24 canonical pages and 48 render assets reproduce exact hashes;
- all evidence spans and bounding boxes map to the rendered page;
- all 30 queries have at least one relevance-2 page and three reviewed hard
  negatives;
- OCR variants reproduce registered CER/WER and reading-order diagnostics;
- canonical, OCR, and visual routes preserve the same page identity;
- sparse CSR round trips without densification or id loss;
- exact ranking and all metrics match deterministic test expectations;
- task and run are publish: false and excluded from Dataset/Space outputs; and
- no network, provider API, model download, or external service is needed for
  the deterministic fixture tests.

Pilot PASS:

- 120 pages and 180 queries pass provenance, privacy, license, split, qrel, and
  hard-negative review;
- at least two independent reviewers agree or all disagreements are adjudicated;
- no trivial heuristic saturates the task;
- no required query family has fewer than 20 held-out queries;
- at least BM25, one learned-sparse, one dense, and registered hybrid rows
  complete with cost evidence;
- route and noise deltas are reproducible across two clean runs; and
- public release is separately approved after inspecting the complete artifact
  manifest.

BLOCKED:

- renderer/font/OCR licensing is unresolved;
- deterministic rendering or OCR cannot be reproduced;
- the sparse adapter loses dimensions, ids, nnz, or index metadata;
- a model needs an unreviewed custom source or incompatible package change;
- a required model or dataset is gated, missing, or exceeds the approved
  download/runtime budget; or
- public release terms for the fixture/pilot are not approved.

ABANDON:

- after two score-blind dataset revisions, canonical BM25 or a template
  heuristic still saturates all diagnostic slices;
- the task cannot preserve identical page identity across canonical, OCR, and
  visual routes;
- the only workable implementation is tied to one vendor's opaque index or
  page chunker;
- qrels cannot be grounded in explicit spans/regions; or
- the proposed data adds no diagnostic behavior beyond republishing a public
  visual-document leaderboard.

## Smallest Safe No-Publish Sparse Smoke

Candidate:
opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill

Pinned revision:
babf71f3c48695e2e53a978208e8aba48335e3c0

Purpose:

- prove current package/model compatibility;
- inspect the actual sparse output type;
- prove query/document routing and 30,522 dimensions;
- measure one-query/four-document nnz and latency;
- prove direct sparse scoring without an external index; and
- produce no benchmark or leaderboard claim.

Self-authored smoke inputs:

Query:

    Where should a departed contractor leave an expired building pass?

Documents:

1. Offboarding procedure: return expired building badges to the security
   reception desk within one business day.
2. Badge renewal requests are approved by the facilities manager before the
   current pass expires.
3. Departing contractors must return loaned laptops to the equipment cabinet.
4. Visitor parking passes are collected at the west-gate kiosk.

Future execution gates:

- Use the current locked environment first; do not update packages.
- Use a dedicated cache unique to this smoke.
- Allow-list only model/config/tokenizer/static-query files.
- Expected Hub usedStorage:
  268,076,936 bytes.
- Stop if the dedicated snapshot exceeds 300,000,000 bytes.
- Load from the local pinned snapshot.
- Do not set trust_remote_code.
- Do not install Pyserini, OpenSearch, Flash Attention, or another index.
- Use one query, four short documents, batch size 1 where configurable, and
  cuda:0 or CPU.
- Record revision, file bytes, package versions, device, dtype, output class,
  shape, nnz per row, decoded top terms, latency, scores, and peak VRAM/RSS.
- Clean only the dedicated cache after evidence capture.

Smoke PASS:

- exact revision and byte cap are respected;
- no custom code is loaded;
- query shape is 1 by 30,522 and document shape is 4 by 30,522;
- all values are finite and every row has at least one nonzero;
- sparse dot-product similarity completes without converting the
  SparseEncoder output to a dense 30,522-wide numpy array;
- document 1 ranks first;
- query route is confirmed as static/tokenizer lookup with no neural forward;
  and
- no public result artifact is produced.

Smoke BLOCKED:

- current sentence-transformers 5.2.3 cannot load the pinned router/config;
- output cannot be represented without changing the repo's dense contract;
- the model unexpectedly requests custom remote code;
- snapshot bytes exceed the cap;
- network/DNS prevents pinned metadata or snapshot access; or
- loading requires an unapproved package change.

Smoke ABANDON:

- the pinned release is no longer public or its license changes incompatibly;
- a standard SparseEncoder path is not actually available; or
- the result can only be obtained through a running OpenSearch service.

A passing smoke establishes compatibility only. One hand-authored query and
four documents cannot establish retrieval quality, sparse-versus-dense
superiority, robustness, index performance, or publication readiness.

## Git and Hugging Face Product Path

No implementation or publication is part of this dispatch.

Recommended later Git-tracked artifacts:

- src/mm_embed/providers/sparse_base.py
- src/mm_embed/providers/sparse_sentence_transformers_provider.py
- src/mm_embed/indexes/sparse_exact.py
- src/mm_embed/data/rendered_page_sparse_retrieval.py
- src/mm_embed/tasks/rendered_page_sparse_retrieval.py
- tests/fixtures/rendered_page_sparse_retrieval/
- tests/test_sparse_provider_contract.py
- tests/test_rendered_page_sparse_retrieval.py
- benchmark/tasks/rendered_page_sparse.yaml
- benchmark/runs/rendered-page-sparse-fixture.yaml

The first task and run entries must use publish: false.

Proposed eventual Dataset schema:

- datasets/rendered_page_sparse_retrieval/pages.jsonl
- datasets/rendered_page_sparse_retrieval/rendered_artifacts.jsonl
- datasets/rendered_page_sparse_retrieval/ocr.jsonl
- datasets/rendered_page_sparse_retrieval/queries.jsonl
- datasets/rendered_page_sparse_retrieval/qrels.jsonl
- datasets/rendered_page_sparse_retrieval/hard_negatives.jsonl
- datasets/rendered_page_sparse_retrieval/provenance.jsonl
- datasets/rendered_page_sparse_retrieval/artifact_manifest.jsonl

Before publication, every asset record needs a public license status and stable
content hash. Large page assets should not be added to the normal Dataset
export merely because a JSON schema exists.

Current repository behavior already filters publish: false fixture tasks and
runs from public Dataset, leaderboard, and Space outputs. Preserve that gate.
The Space should show no learned-sparse score until a score-bearing pilot is
approved. A future task card may describe the research status without exposing
fixture scores.

## Concrete Follow-Up Items

1. Run the pinned OpenSearch one-query/four-document sparse compatibility smoke
   in a dedicated cache and record PASS or a concrete compatibility blocker.
2. Implement the provider-neutral sparse CSR/result/index contract with a
   deterministic no-network test double; do not add a real model registry row
   in the same item.
3. Author the 24-page, 30-query rendered-page fixture and its provenance,
   qrels, hard negatives, route identities, and no-publish tests before any
   visual model download.

## Primary Sources

### V-SPLADE

- Paper:
  <https://arxiv.org/abs/2605.30917>
- Official code:
  <https://github.com/naver/v-splade>
- Pinned official code commit:
  <https://github.com/naver/v-splade/commit/3f9773785943bd6e62a9e12c92fa3e2a68c3f477>
- Efficient model:
  <https://huggingface.co/naver/v-splade-efficient>
- Pinned Efficient model card:
  <https://huggingface.co/naver/v-splade-efficient/blob/ab0c2260c6d78bcb7d05076a9407a71f55d57eb1/README.md>
- Pinned Efficient config:
  <https://huggingface.co/naver/v-splade-efficient/blob/ab0c2260c6d78bcb7d05076a9407a71f55d57eb1/config.json>
- Pinned Efficient router:
  <https://huggingface.co/naver/v-splade-efficient/blob/ab0c2260c6d78bcb7d05076a9407a71f55d57eb1/router_config.json>
- Quality model:
  <https://huggingface.co/naver/v-splade-quality>
- Pinned Quality revision:
  <https://huggingface.co/naver/v-splade-quality/commit/99bdc93f42460e595b2fb1e78b96edd44e898441>

### OpenSearch sparse

- Pinned model card:
  <https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill/blob/babf71f3c48695e2e53a978208e8aba48335e3c0/README.md>
- Pinned model config:
  <https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill/blob/babf71f3c48695e2e53a978208e8aba48335e3c0/config.json>
- Paper:
  <https://arxiv.org/abs/2504.14839>
- OpenSearch neural sparse query documentation:
  <https://opensearch.org/docs/latest/query-dsl/specialized/neural-sparse/>
- Model tuning source linked by the paper/model card:
  <https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample/tree/l0_enhance>

### SPLADE and uniCOIL

- SPLADE v3 DistilBERT pinned model card:
  <https://huggingface.co/naver/splade-v3-distilbert/blob/2db06b86d65e316e2ca9907aa1aa8be6f8c4e739/README.md>
- SPLADE v3 paper:
  <https://arxiv.org/abs/2403.06789>
- Official SPLADE code:
  <https://github.com/naver/splade>
- Pinned SPLADE code commit:
  <https://github.com/naver/splade/commit/8dcd33a054d790e74aceda25b128c1b188c5d9c1>
- uniCOIL model:
  <https://huggingface.co/castorini/unicoil-msmarco-passage>
- Pyserini uniCOIL reproduction:
  <https://github.com/castorini/pyserini/blob/master/docs/experiments-unicoil.md>
- Pyserini uniCOIL encoder:
  <https://github.com/castorini/pyserini/blob/master/pyserini/encode/_unicoil.py>

### SPLADE-Code

- Paper:
  <https://arxiv.org/abs/2603.22008>
- Pinned 0.6B model card:
  <https://huggingface.co/naver/splade-code-06B/blob/e53a0b8bd312d83955598a392dc826b3fc4028f7/README.md>
- Pinned 0.6B config:
  <https://huggingface.co/naver/splade-code-06B/blob/e53a0b8bd312d83955598a392dc826b3fc4028f7/config.json>
- Pinned 0.6B custom source:
  <https://huggingface.co/naver/splade-code-06B/blob/e53a0b8bd312d83955598a392dc826b3fc4028f7/splade.py>
- Pinned 8B model card:
  <https://huggingface.co/naver/splade-code-8B/blob/fe9fb2fc9fd930187ede95085cd189c7dc5d55a4/README.md>
- Pinned 8B custom source:
  <https://huggingface.co/naver/splade-code-8B/blob/fe9fb2fc9fd930187ede95085cd189c7dc5d55a4/splade.py>
- Qwen3-8B base:
  <https://huggingface.co/Qwen/Qwen3-8B>

### SPLARE

- Paper and release-status evidence:
  <https://arxiv.org/abs/2603.13277>
- WSDM Cup report using SPLARE:
  <https://arxiv.org/abs/2602.20986>

### LIMIT+ task-design evidence

- Reproducibility paper:
  <https://arxiv.org/abs/2605.03824>
- Current code/data repository:
  <https://github.com/informagi/Complex-Set-Compositional-IR>
- Pinned repository commit:
  <https://github.com/informagi/Complex-Set-Compositional-IR/commit/0a4105a328474d4a4c58b8e4fc613ec05c59fc22>

## Final Judgment

PASS as a research and dataset-design artifact.

The family is no longer anchored to one V-SPLADE checkpoint. It has a
low-risk released first smoke, a classic SPLADE control, explicit visual and
code follow-ups, paper-only handling for SPLARE, a provider-neutral sparse
contract gap, an original auditable rendered-page dataset, quality and serving
metrics, and no-publish product gates.

This artifact does not establish any model score or approve any external data
for ingestion. The next safe action is the pinned OpenSearch compatibility
smoke, followed by a deterministic sparse contract, followed by the authored
rendered-page fixture.
