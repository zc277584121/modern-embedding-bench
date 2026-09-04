# Milvus SINDI system benchmark phase-one predeclaration

This document freezes the system experiment before any Milvus performance cell is run. The machine-readable contract is `benchmark/artifacts/milvus-sindi-system-v0.1/predeclaration.json`; its SHA256 sidecar is the only active identity.

## Accepted source

The input is the accepted S-008 package at commit `74635817770023b70b1e9f9e97b3d83d9d11f987` and tree `dc4a28d1805c82de5d4e2397fa163275a671638c`. A CPU-only full replay authenticated all 14 raw cells, their saved CSR parts, saved exact rankings, the main protocol, and the tracked package. The replay explicitly reported `model_loaded=false`. No model encoding or download is part of this Story.

The tracked source attestation contains identities and counts only. Detailed operational evidence, canonical IDs, vectors, source text, and rankings remain in gitignored research artifacts.

## Workloads fixed before results

The representative profiles are BGE-M3 for low document nnz, Granite Sparse 30M for medium/pruned nnz, and OpenSearch multilingual for high document nnz. This selection spans three mechanisms and uses only saved resource statistics. Each representative runs on both native economics and psychology tracks.

The system-only layer uses the same three profiles at 100,000 and 1,000,000 documents. Its deterministic generator is seeded with `20260830`, never materializes dense vectors, and must preserve source dimension, nnz, weight, and overlap distributions. Generated documents must be unique. System-only results cannot enter model-quality conclusions.

## Frozen comparison

The comparison is exact SciPy CSR inner product versus explicit Milvus `SINDI` and `DAAT_MAXSCORE`. Both Milvus indexes use `SPARSE_INVERTED_INDEX`, `IP`, `drop_ratio_build=0`, no quantization, and no mmap. Every search sets `drop_ratio_search=0`.

Every workload uses topK 10 and 100, concurrency 1, 4, and 16, two unmeasured warmup passes, and three measured trials. Query order and cell order use fixed seeds. Milvus algorithms receive the same vectors, queries, filters, schedules, container image, CPU allocation, and memory limit. Only one Milvus cell may be active at a time.

Strict ID Recall@K and tie-aware Recall@K are reported separately. Exact ordering uses descending float32 inner-product score and then private row ordinal. A non-1.0 result must remain visible and receive a reproducible explanation.

## Fixed server and schema

The server is Milvus `v3.0.0` on linux/amd64, pinned to `milvusdb/milvus@sha256:804b50bc1523a64e3c0f18cf33af7e0f8b33329584698ffee61c606d6fe8ee26`; the multi-architecture index digest observed from the registry is `sha256:49371c30af46b1013e4d3e0b980e691d81376d69cdbe1b372725baf1d7255862`. PyMilvus is fixed at `3.0.1`.

The collection has an explicit INT64 primary key and one non-null `SPARSE_FLOAT_VECTOR`; auto IDs and dynamic fields are disabled. Searches cannot start until row counts, flush, finished index state, loaded collection state, sealed-only segments, zero growing segments, and the explicit server-reported algorithm are saved.

Official evidence was retrieved on 2026-08-30 from the Milvus v3.0.0 release at commit `f46a0328558be155d11266a1a2b90602ccc9b366`, sparse index documentation, sparse vector documentation, standalone deployment documentation, and Docker registry manifest. These sources establish availability and configuration, not performance.

## Isolation and safety

All names use the `meb-s010` / `meb_s010` prefix. The proposed host ports are bound to loopback only. Existing Milvus, Zilliz, etcd, MinIO, Woodpecker, networks, volumes, ports, and data are observation-only and must never be stopped, restarted, upgraded, executed into, written, or cleaned by this Story.

The server hard limit is 48 GiB RAM with no swap and 16 fixed physical CPU cores. The complete Story-owned disk cap is 120 GiB. Formal work aborts below 192 GiB host-available memory, below 200 GiB free space on the Story filesystem, or above load1 48. The 1M tier is skipped, with saved evidence and no extrapolation, if the conservative 100k-based projection exceeds the declared runtime cap.

The publication classification remains `research_only`, and every publication/export/leaderboard gate remains closed.
