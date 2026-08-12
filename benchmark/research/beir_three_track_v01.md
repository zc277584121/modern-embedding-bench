# BEIR three-track shared benchmark v0.1

This benchmark keeps SciFact, NFCorpus, and FiQA as three independent retrieval tracks. It does not merge corpora, queries, or qrels, and only permits an unweighted macro average of already-computed track metrics.

## Fixed inputs and text protocol

The only accepted inputs are the UKP-hosted BEIR archives listed in `TRACKS` in `src/mm_embed/benchmark/beir_v01.py`. Both the official MD5 and a locally verified SHA256 and byte size are fixed. The loader reads required files directly from each validated ZIP, rejects path traversal and missing files, and validates IDs, qrel references, grades, and exact sizes.

Document input is model-independent: strip title and text; join non-empty title and text with exactly one newline. Queries are stripped without prompts. Tracks use the BEIR test split. No query selection or qrel modification may depend on baseline scores.

## Provenance and publication gate

BEIR identifies the archives as public downloads and attributes the underlying datasets to their original projects: AllenAI SciFact, the Heidelberg NLP Group NFCorpus, and FiQA 2018. BEIR's software license does not replace dataset-specific terms. The upstream projects and source documents have different or incomplete redistribution statements, so local materialization and evaluation are allowed here while redistribution remains fail-closed. Do not upload the materialized corpus, queries, qrels, or audit pack until a dataset-specific license and attribution review explicitly opens the manifest publish gate.

Primary references:

- https://github.com/beir-cellar/beir#available-datasets
- https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/
- https://github.com/allenai/scifact
- https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/
- https://sites.google.com/view/fiqa/

## Audit and baseline policy

The materializer records sizes, qrel density and grades, missing references, empty text, exact duplicates, a deterministic near-duplicate heuristic, length distributions, exact query/document text overlap, and leakage/label risks. Sparse BEIR judgments mean unjudged documents are not confirmed negatives.

The 150-query review pack allocates 50 queries per track and samples round-robin from qrel-density by query-length-tertile strata in sorted query-ID order. BM25 scores only populate review candidates after query selection. A high-ranked unjudged result is a suspected missing-label candidate only; reviewers must not rewrite qrels in place.

BM25 uses fixed Unicode word tokenization, lowercase, `k1=1.2`, and `b=0.75`. Dense uses `sentence-transformers/all-MiniLM-L6-v2` at revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, `trust_remote_code=False`, normalized embeddings, and blockwise exact dot products. Dense results are not claimed to be verified zero-shot because training overlap with these datasets is unknown.

## Commands

```bash
uv run python scripts/beir_benchmark_v01.py materialize --cache-root /data1/cache/huggingface/datasets/BeIR --output data/beir-three-track-v0.1
uv run python scripts/beir_benchmark_v01.py run --data data/beir-three-track-v0.1 --method bm25 --output results/beir-three-track-v0.1
uv run --extra local python scripts/beir_benchmark_v01.py run --data data/beir-three-track-v0.1 --method dense --output results/beir-three-track-v0.1
uv run python scripts/beir_benchmark_v01.py audit-pack --data data/beir-three-track-v0.1 --bm25-results results/beir-three-track-v0.1 --output results/beir-three-track-v0.1/audit-pack-150.jsonl
```
