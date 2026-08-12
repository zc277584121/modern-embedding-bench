# OpenSearch neural sparse smoke attempt

- Run manifest: `benchmark/runs/opensearch-neural-sparse-smoke.yaml`
- Model revision: `babf71f3c48695e2e53a978208e8aba48335e3c0`
- Scope: one query and four fixed documents; exact SciPy CSR only; publish=false; leaderboard_publish=false.
- Snapshot: fixed local revision snapshot under the project `.cache/huggingface` path; runtime uses `allow_download=false`.
- Follow-up: two no-publish artifacts were regenerated from that local snapshot. The verified query route is `static_lookup`, document route is `document_expansion`, and the observed query nnz total is 12. This remains bounded smoke evidence only; no quality or leaderboard conclusion is drawn.
