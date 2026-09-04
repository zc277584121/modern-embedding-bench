# Milvus SINDI sparse system benchmark

## Status and scope

This report records the completed S-20260814-010 research-only system experiment. The publication gate remains closed. Native results cover only the two accepted internal tracks; generated 100k and 1M results are system-only scaling evidence and must not be used for model-quality, multilingual-effectiveness, or zero-shot claims.
A `pass` status means that execution, identity, lifecycle, and recomputation checks passed. It does not mean that SINDI, DAAT_MAXSCORE, or any model is effective, faster, or preferable.

The validated matrix contains 18 native cells, 9 100k cells, and 9 1M cells. The independent reducer recomputed 1296 measured raw trial files and validated 432 warmup files.

## Identity and deployment

- Accepted source commit: `74635817770023b70b1e9f9e97b3d83d9d11f987`; the runtime path reported `model_loaded=false` and reused saved CSR/exact artifacts read-only.
- Server image: `milvusdb/milvus@sha256:804b50bc1523a64e3c0f18cf33af7e0f8b33329584698ffee61c606d6fe8ee26`; Milvus `3.0.0`; PyMilvus `3.0.1`.
- Persisted vector index version: `10`. Runtime root mode remained `0711`.
- Server resource boundary: cpuset `48-63`, 51539607552 bytes memory with no extra swap, and 4096 PIDs.
- Primary references: [Milvus 3.0.0 release and compatibility notes](https://milvus.io/docs/release_notes.md) and [SPARSE_INVERTED_INDEX parameters](https://milvus.io/docs/sparse-inverted-index.md). These establish feature and configuration identity only; they are not benchmark results.

## Method

Each workload used identical sparse vectors, queries, topK values (10 and 100), deterministic query schedules, and concurrency values (1, 4, and 16). Both Milvus indexes used `SPARSE_FLOAT_VECTOR`, inner product, `SPARSE_INVERTED_INDEX`, explicit `SINDI` or `DAAT_MAXSCORE`, `drop_ratio_build=0`, and `drop_ratio_search=0`. Every configuration used two unmeasured warmups and three measured cold plus three measured warm trials. Cold means collection release and reload; host page caches were not dropped.

Before every Milvus search family, the runner required the requested server-reported algorithm, `Finished`, `Loaded`, all rows in sealed indexed segments, zero growing rows, and no active or queued compaction. Exact cost uses SciPy CSR float32 inner product on the frozen CPU domain. Build, insert, load, client preparation, RPC/end-to-end search, server latency counters, network counters, client CPU/RSS, container CPU/RSS/PIDs, persisted bytes, and raw query trials are stored separately in private evidence.

## Correctness

In the native tier, the minimum raw float32 strict Recall@K was `0.900000` for SINDI and `0.900000` for DAAT_MAXSCORE. Across all tiers, including synthetic system-only near-tie workloads, the corresponding minima were `0.200000` and `0.200000`. Maximum absolute score deltas were `0.00790405273` and `0.00790023804`. Every validated response used unique private row ordinals and finite scores.
Exact ties use descending float32 score followed by ascending private row ordinal. An independent private native replay cast authenticated document nonzeros to IEEE float16 and back to float32 while retaining float32 queries. Against that persisted-index representation, the minimum strict recall was 0.9 and tie-aware Recall@K was 1.0 for both SINDI and DAAT_MAXSCORE; server score deltas were at most 1.52587890625e-05. The lower system-only strict ID recall reflects denser synthetic near-tie boundaries under the same storage conversion and is not an effectiveness result. These are pinned index-format effects, not search pruning, default fallback, or model-quality differences.

## Performance

Across every native and system-only K/concurrency/cold-warm configuration, the median SINDI/DAAT QPS ratio was `1.004`; the observed range was `0.968` to `8.215`. These are measurements of this pinned system and workload matrix, not vendor claims.
The ratio crosses 1.0, so this matrix contains real workload/configuration crossovers and does not support a universal SINDI-wins claim.
Warm/cold effects were not uniformly positive: median warm-over-cold QPS ratios were SINDI/DAAT `1.090`/`1.078` in native, `1.092`/`1.087` at 100k, and `1.092`/`1.041` at 1M. The full configuration table preserves workload-specific reversals and long tails; no universal cache benefit or scale advantage is claimed.
Scale changed the observed economics: median SINDI/DAAT QPS ratios were `1.001` in native, `1.008` at 100k, and `2.232` at 1M. The 1M range still extended from `0.981` to `8.215`. The upper extreme is a DAAT_MAXSCORE long-tail collapse for one synthetic OpenSearch-multilingual K100/concurrency-1 configuration, while other configurations showed no SINDI advantage. It is a crossover and tail-risk observation, not a universal algorithm ranking.

The compact table below shows warm concurrency-16 medians; the machine-readable summary contains all configurations.

| Workload | K | Exact QPS | SINDI QPS | DAAT QPS | S/D | Exact P50 | Exact P95 | Exact P99 | S P50 | S P95 | S P99 | D P50 | D P95 | D P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native/bge-m3/economics | 10 | 57.71 | 58.13 | 57.54 | 1.010 | 172.251 | 501.280 | 653.094 | 22.104 | 35.213 | 43.859 | 20.263 | 33.878 | 38.285 |
| native/bge-m3/economics | 100 | 59.28 | 60.65 | 59.01 | 1.028 | 193.933 | 431.655 | 554.399 | 28.230 | 48.966 | 70.114 | 28.305 | 58.018 | 65.932 |
| native/bge-m3/psychology | 10 | 57.45 | 54.90 | 56.04 | 0.980 | 219.800 | 551.729 | 659.600 | 21.632 | 37.556 | 46.353 | 20.564 | 36.235 | 47.643 |
| native/bge-m3/psychology | 100 | 53.20 | 58.95 | 58.89 | 1.001 | 226.066 | 467.910 | 587.339 | 25.908 | 45.810 | 62.017 | 27.317 | 48.180 | 60.248 |
| native/granite-30m-sparse/economics | 10 | 40.37 | 59.31 | 57.26 | 1.036 | 340.080 | 630.950 | 800.739 | 21.661 | 39.503 | 53.448 | 21.361 | 37.374 | 44.255 |
| native/granite-30m-sparse/economics | 100 | 40.11 | 59.94 | 59.99 | 0.999 | 354.810 | 645.313 | 727.286 | 27.440 | 53.998 | 69.764 | 23.966 | 41.863 | 52.783 |
| native/granite-30m-sparse/psychology | 10 | 41.33 | 57.05 | 56.03 | 1.018 | 367.608 | 635.465 | 728.107 | 18.441 | 34.570 | 44.221 | 18.805 | 37.934 | 43.825 |
| native/granite-30m-sparse/psychology | 100 | 39.89 | 58.95 | 58.60 | 1.006 | 350.248 | 671.376 | 823.178 | 25.767 | 55.817 | 63.943 | 25.350 | 44.837 | 58.477 |
| native/opensearch-multilingual/economics | 10 | 33.89 | 56.59 | 58.04 | 0.975 | 402.839 | 801.768 | 993.383 | 22.876 | 42.322 | 57.111 | 24.770 | 44.680 | 51.359 |
| native/opensearch-multilingual/economics | 100 | 36.80 | 59.77 | 60.17 | 0.993 | 362.576 | 761.955 | 880.107 | 29.855 | 59.953 | 80.213 | 30.189 | 53.655 | 62.890 |
| native/opensearch-multilingual/psychology | 10 | 31.89 | 56.88 | 57.14 | 0.996 | 458.004 | 783.054 | 932.702 | 24.420 | 45.648 | 50.840 | 21.634 | 33.767 | 38.991 |
| native/opensearch-multilingual/psychology | 100 | 31.71 | 60.07 | 59.28 | 1.013 | 427.566 | 794.318 | 940.594 | 29.510 | 59.083 | 69.733 | 29.491 | 56.832 | 60.813 |
| system-only/bge-m3/100000 | 10 | 4.44 | 148.19 | 146.13 | 1.014 | 3317.894 | 5866.282 | 6750.621 | 29.083 | 58.425 | 77.108 | 31.596 | 54.306 | 71.139 |
| system-only/bge-m3/100000 | 100 | 4.24 | 163.64 | 161.35 | 1.014 | 3380.549 | 5944.545 | 7392.776 | 37.444 | 76.888 | 90.241 | 36.017 | 61.398 | 76.676 |
| system-only/bge-m3/1000000 | 10 | 0.36 | 146.57 | 146.05 | 1.004 | 42772.159 | 56706.753 | 62673.749 | 28.705 | 51.799 | 65.697 | 42.901 | 89.593 | 136.527 |
| system-only/bge-m3/1000000 | 100 | 0.36 | 159.46 | 159.43 | 1.000 | 42342.461 | 58016.314 | 63313.301 | 30.163 | 55.840 | 65.838 | 53.016 | 108.285 | 146.798 |
| system-only/granite-30m-sparse/100000 | 10 | 3.42 | 147.81 | 147.13 | 1.005 | 4444.691 | 6702.227 | 8126.382 | 29.128 | 52.931 | 66.958 | 30.202 | 48.359 | 58.988 |
| system-only/granite-30m-sparse/100000 | 100 | 3.50 | 160.22 | 161.21 | 0.994 | 4296.664 | 6673.954 | 7750.814 | 34.583 | 63.393 | 82.210 | 36.240 | 63.023 | 77.521 |
| system-only/granite-30m-sparse/1000000 | 10 | 0.30 | 147.81 | 150.59 | 0.981 | 52078.217 | 65606.154 | 71327.415 | 31.431 | 58.817 | 70.165 | 87.952 | 106.899 | 116.767 |
| system-only/granite-30m-sparse/1000000 | 100 | 0.28 | 161.83 | 160.05 | 1.011 | 54837.975 | 68331.979 | 75148.780 | 40.013 | 67.735 | 84.820 | 83.605 | 117.516 | 134.884 |
| system-only/opensearch-multilingual/100000 | 10 | 2.77 | 143.11 | 147.90 | 0.968 | 5366.247 | 8159.576 | 9222.220 | 30.125 | 54.722 | 69.830 | 35.657 | 58.914 | 71.418 |
| system-only/opensearch-multilingual/100000 | 100 | 3.10 | 158.18 | 162.33 | 0.974 | 4902.687 | 7265.864 | 8775.549 | 38.773 | 80.252 | 100.108 | 40.318 | 69.309 | 87.826 |
| system-only/opensearch-multilingual/1000000 | 10 | 0.26 | 150.81 | 67.81 | 2.224 | 60890.520 | 73458.448 | 76546.004 | 37.833 | 60.465 | 72.820 | 147.571 | 287.025 | 370.259 |
| system-only/opensearch-multilingual/1000000 | 100 | 0.26 | 162.98 | 71.06 | 2.294 | 60295.722 | 73012.540 | 79752.726 | 46.656 | 80.720 | 98.251 | 162.011 | 302.560 | 387.741 |

The next table separates client sparse-vector preparation, RPC end-to-end latency, server search-counter mean, and proxy byte counters for the same warm concurrency-16 slice. Proxy sent bytes are a global counter measured while this was the only active Story cell; no packet capture or exact protobuf payload-size claim is made.

| Workload | K | S prep P50 ms | S RPC P50 ms | S server mean ms | S recv B | S sent B | D prep P50 ms | D RPC P50 ms | D server mean ms | D recv B | D sent B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native/bge-m3/economics | 10 | 0.0548 | 22.076 | 3.272 | 76856 | 16771 | 0.0505 | 20.175 | 4.650 | 76753 | 16668 |
| native/bge-m3/economics | 100 | 0.0560 | 28.135 | 3.728 | 76959 | 91738 | 0.0580 | 28.275 | 4.641 | 76856 | 91635 |
| native/bge-m3/psychology | 10 | 0.0531 | 21.579 | 3.277 | 71786 | 16540 | 0.0482 | 20.529 | 4.564 | 71685 | 16439 |
| native/bge-m3/psychology | 100 | 0.0490 | 25.867 | 3.396 | 71887 | 89923 | 0.0526 | 27.264 | 4.931 | 71786 | 89822 |
| native/granite-30m-sparse/economics | 10 | 0.0453 | 21.625 | 3.126 | 65508 | 17187 | 0.0448 | 21.326 | 5.340 | 65405 | 17084 |
| native/granite-30m-sparse/economics | 100 | 0.0505 | 27.397 | 3.641 | 65611 | 92162 | 0.0403 | 23.900 | 5.573 | 65508 | 92059 |
| native/granite-30m-sparse/psychology | 10 | 0.0404 | 18.400 | 2.436 | 64337 | 16940 | 0.0392 | 18.767 | 4.743 | 64236 | 16839 |
| native/granite-30m-sparse/psychology | 100 | 0.0484 | 25.730 | 3.208 | 64438 | 90389 | 0.0412 | 25.310 | 5.901 | 64337 | 90288 |
| native/opensearch-multilingual/economics | 10 | 0.0659 | 22.704 | 3.544 | 99568 | 16765 | 0.0726 | 24.670 | 7.680 | 99465 | 16662 |
| native/opensearch-multilingual/economics | 100 | 0.0706 | 29.785 | 3.835 | 99671 | 91732 | 0.0684 | 30.114 | 7.748 | 99568 | 91629 |
| native/opensearch-multilingual/psychology | 10 | 0.0680 | 24.387 | 3.713 | 96693 | 16540 | 0.0614 | 21.589 | 6.990 | 96592 | 16439 |
| native/opensearch-multilingual/psychology | 100 | 0.0738 | 29.416 | 3.723 | 96794 | 89903 | 0.0714 | 29.361 | 8.010 | 96693 | 89802 |
| system-only/bge-m3/100000 | 10 | 0.0774 | 28.988 | 4.035 | 186517 | 46810 | 0.0819 | 31.536 | 10.582 | 186261 | 46554 |
| system-only/bge-m3/100000 | 100 | 0.0852 | 37.319 | 4.559 | 186773 | 271728 | 0.0716 | 35.913 | 11.871 | 186517 | 271472 |
| system-only/bge-m3/1000000 | 10 | 0.0767 | 28.610 | 9.004 | 186005 | 46848 | 0.0879 | 42.777 | 41.887 | 185749 | 46592 |
| system-only/bge-m3/1000000 | 100 | 0.0554 | 30.058 | 7.797 | 186261 | 278746 | 0.0864 | 52.942 | 49.961 | 186005 | 278490 |
| system-only/granite-30m-sparse/100000 | 10 | 0.0834 | 29.042 | 4.957 | 162816 | 47832 | 0.0846 | 30.130 | 12.652 | 162560 | 47576 |
| system-only/granite-30m-sparse/100000 | 100 | 0.0831 | 34.513 | 5.031 | 163072 | 272708 | 0.0837 | 36.154 | 14.797 | 162816 | 272452 |
| system-only/granite-30m-sparse/1000000 | 10 | 0.0860 | 31.360 | 8.484 | 162304 | 47872 | 0.0811 | 87.884 | 78.309 | 162048 | 47616 |
| system-only/granite-30m-sparse/1000000 | 100 | 0.0852 | 39.927 | 12.035 | 162560 | 279774 | 0.0797 | 83.498 | 81.180 | 162304 | 279518 |
| system-only/opensearch-multilingual/100000 | 10 | 0.0865 | 30.032 | 5.105 | 245688 | 46794 | 0.1014 | 35.561 | 16.891 | 245432 | 46538 |
| system-only/opensearch-multilingual/100000 | 100 | 0.0994 | 38.706 | 5.746 | 245944 | 271780 | 0.0932 | 40.220 | 18.320 | 245688 | 271524 |
| system-only/opensearch-multilingual/1000000 | 10 | 0.1035 | 37.709 | 15.871 | 245176 | 46848 | 0.0951 | 147.475 | 156.250 | 244920 | 46592 |
| system-only/opensearch-multilingual/1000000 | 100 | 0.1074 | 46.554 | 16.578 | 245432 | 278730 | 0.0944 | 161.932 | 173.027 | 245176 | 278474 |

## Build, persistence, and resource cost

Persisted index bytes come from the server's successful version-10 serialization log. Incremental runtime bytes are measured across the complete collection lifecycle and include other collection-associated persistence overhead. Peaks are sampled from the isolated container and benchmark client.

| Workload | Algorithm | Insert s | Flush s | Build s | Load s | Index bytes | Runtime delta bytes | Container peak bytes | Client peak RSS bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native/bge-m3/economics | SINDI | 2.011 | 2.011 | 2.012 | 2.068 | 1917681 | 5574656 | 440716493 | 528756736 |
| native/bge-m3/economics | DAAT_MAXSCORE | 2.014 | 2.016 | 2.015 | 2.059 | 1708213 | 5398528 | 341626061 | 285257728 |
| native/bge-m3/psychology | SINDI | 2.012 | 2.011 | 2.013 | 2.011 | 1823865 | 5308416 | 274202624 | 180629504 |
| native/bge-m3/psychology | DAAT_MAXSCORE | 2.012 | 2.015 | 2.012 | 2.270 | 1658157 | 5115904 | 319396250 | 285257728 |
| native/granite-30m-sparse/economics | SINDI | 4.078 | 2.015 | 2.014 | 2.268 | 5329721 | 13926400 | 560254157 | 528756736 |
| native/granite-30m-sparse/economics | DAAT_MAXSCORE | 4.064 | 2.011 | 2.012 | 2.266 | 4323973 | 72245248 | 487902413 | 528756736 |
| native/granite-30m-sparse/psychology | SINDI | 2.510 | 1.538 | 2.010 | 2.065 | 5296261 | 13910016 | 557423002 | 528756736 |
| native/granite-30m-sparse/psychology | DAAT_MAXSCORE | 4.057 | 2.010 | 2.013 | 2.265 | 4354021 | 12963840 | 367421030 | 285257728 |
| native/opensearch-multilingual/economics | SINDI | 4.076 | 2.012 | 2.013 | 2.271 | 8308389 | 31158272 | 457703424 | 306839552 |
| native/opensearch-multilingual/economics | DAAT_MAXSCORE | 4.070 | 2.013 | 2.015 | 2.016 | 6853209 | 29712384 | 546937242 | 528756736 |
| native/opensearch-multilingual/psychology | SINDI | 4.073 | 2.011 | 2.011 | 2.276 | 9083841 | 34467840 | 517052826 | 502046720 |
| native/opensearch-multilingual/psychology | DAAT_MAXSCORE | 4.070 | 2.005 | 2.013 | 2.273 | 7550333 | 32935936 | 620756992 | 528756736 |
| system-only/bge-m3/100000 | SINDI | 14.384 | 2.017 | 4.045 | 2.064 | 20366662 | 106827776 | 1446330237 | 4841324544 |
| system-only/bge-m3/100000 | DAAT_MAXSCORE | 15.322 | 2.014 | 2.149 | 1.881 | 17820794 | 74420224 | 1034839654 | 436056064 |
| system-only/bge-m3/1000000 | SINDI | 130.271 | 2.013 | 4.145 | 1.917 | 201245669 | 1067364352 | 11242076897 | 4387909632 |
| system-only/bge-m3/1000000 | DAAT_MAXSCORE | 133.501 | 2.012 | 8.132 | 2.066 | 175257511 | 1063235584 | 4153233375 | 990769152 |
| system-only/granite-30m-sparse/100000 | SINDI | 28.845 | 1.786 | 2.052 | 2.060 | 65959339 | 399265792 | 1333587345 | 436056064 |
| system-only/granite-30m-sparse/100000 | DAAT_MAXSCORE | 30.762 | 2.011 | 4.054 | 2.016 | 54687298 | 299524096 | 1592359125 | 4841324544 |
| system-only/granite-30m-sparse/1000000 | SINDI | 286.532 | 2.011 | 13.199 | 2.059 | 655568724 | 3008102400 | 12852689633 | 4387909632 |
| system-only/granite-30m-sparse/1000000 | DAAT_MAXSCORE | 285.770 | 1.828 | 16.235 | 2.016 | 542475442 | 2860752896 | 15880641577 | 4387909632 |
| system-only/opensearch-multilingual/100000 | SINDI | 47.264 | 2.016 | 4.055 | 2.064 | 106256344 | 519471104 | 2080911655 | 4841324544 |
| system-only/opensearch-multilingual/100000 | DAAT_MAXSCORE | 45.125 | 2.012 | 6.285 | 2.061 | 88916925 | 625852416 | 1542967001 | 4841324544 |
| system-only/opensearch-multilingual/1000000 | SINDI | 453.840 | 2.013 | 18.341 | 2.067 | 1060257164 | 6321860608 | 16761109873 | 41199599616 |
| system-only/opensearch-multilingual/1000000 | DAAT_MAXSCORE | 490.272 | 2.016 | 30.488 | 2.261 | 886548215 | 5986476032 | 12670153523 | 4387909632 |

Exact CSR has no Milvus insert/build/load/network phases. Its comparable resident input and client peak costs are:

| Workload | Document CSR bytes | Query CSR bytes | Client peak RSS bytes |
| --- | ---: | ---: | ---: |
| native/bge-m3/economics | 3117620 | 53376 | 272015360 |
| native/bge-m3/psychology | 2902492 | 48664 | 266522624 |
| native/granite-30m-sparse/economics | 9828196 | 41616 | 528756736 |
| native/granite-30m-sparse/psychology | 9711180 | 40808 | 528756736 |
| native/opensearch-multilingual/economics | 14928972 | 76088 | 502046720 |
| native/opensearch-multilingual/psychology | 16372628 | 73568 | 528756736 |
| system-only/bge-m3/100000 | 40089172 | 128156 | 2790105088 |
| system-only/bge-m3/1000000 | 401295412 | 128156 | 41199599616 |
| system-only/granite-30m-sparse/100000 | 130194228 | 103428 | 2790105088 |
| system-only/granite-30m-sparse/1000000 | 1302556308 | 103428 | 41199599616 |
| system-only/opensearch-multilingual/100000 | 208724724 | 187324 | 4841324544 |
| system-only/opensearch-multilingual/1000000 | 2086820244 | 187324 | 41199599616 |

## Failures and recovery

The first persisted SINDI build exposed a preflight gap: Milvus 3.0.0 defaults to vector index version 8, while SINDI requires the opt-in version 10. The server explicitly rejected SINDI rather than falling back. The failure, retry loop, API state, metrics, and logs were preserved privately. The Story-owned configuration was set to `dataCoord.targetVecIndexVersion=10` and `dataCoord.targetScalarIndexVersion=4`; the same Story container then built and serialized the real index successfully. A separate client-side load-state enum comparison also failed once, was preserved, and was corrected without changing any workload parameter.

A stop/start isolation check found one unrelated ambient host listener (`*:6443`) absent after the bounded Story restart. A later multi-day final baseline retained the same non-Story container-name, network, volume, and listening-endpoint sets but found five unrelated demo containers recreated, including one mount-set change. No benchmark command targeted those resources. Because host-wide Docker events were not under Story control, the evidence reports this ambient drift and does not claim literal zero non-Story identity drift.

## Reproduction and boundaries

The private formal manifest binds every raw trial, lifecycle response, server log, baseline, failure, and validation artifact by size and SHA-256. Re-run the independent validator with:

```bash
uv run python scripts/milvus_sindi_system.py validate-formal-results \
  --output-root results/milvus-sindi-system-v0.1/formal-20260831 \
  --output results/milvus-sindi-system-v0.1/formal-20260831/validation/recomputed-summary.json
```

This report contains aggregate system evidence only. It contains no source text, canonical document/query identifiers, or raw rankings. `research_only=true`, no-publish remains in force, and system-only results do not support quality claims.
