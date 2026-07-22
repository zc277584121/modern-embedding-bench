# Pinned Requests Source-Contract Smoke - 2026-07-22

## Outcome

**PASS** for the private, metadata-gated source-contract smoke.

- Session: `meb-modern-embedding-leaderboard-6-1784728508-2-requests-37265838cb26`
- Retrieved at: `2026-07-22T14:12:16.043953+00:00`
- Repository: `psf/requests`
- Issue: `1920`
- PR: `1921`
- Publication: `publish: false`
- Evidence tier: `evidence_tier: smoke`

The final bounded run verified the pinned public metadata, reconstructed every
regular blob in the base tree, recomputed Stage A and Stage B eligibility,
built the complete eligible corpus without changed-path or gold-path
prefiltering, mapped the patch to pre-change chunks without fuzz, reproduced
all artifact hashes, passed deterministic score-matrix and corruption checks,
stayed below every hard cap, and removed the dedicated temporary path.

## Pinned identity and provenance

| Field | Verified value |
|---|---|
| Base commit | `3c88e520da24ae6f736929a750876e7654accc3d` |
| Tree SHA | `cf441cdcaa76806a078893bffa79922351a92b0a` |
| PR head SHA | `d2f647cee45fd05cc1977cc3faf4b095b5047b29` |
| Expected and observed changed paths | `requests/sessions.py`, `test_requests.py` |
| Repository license | `Apache-2.0` |
| LICENSE Git blob | `8c5e758401020484b74d00c0aa75debfc7b65155` |
| LICENSE SHA-256 | `89478f1915fbb6b6585a685d071bf006ba5649d2615fab787f66e9693c622ae4` |
| Patch SHA-256 | `aa2328cd30a6815cdaf612ccc4dcb7a2626870368948bfe4c96225b4216abfaf` |

Normal DNS resolution and HTTPS were used for `github.com` and
`api.github.com`; no alternate or hard-coded IP was used. The issue and PR
HTML pages and APIs returned HTTP 200. The patch and archive followed normal
GitHub redirects to `patch-diff.githubusercontent.com` and
`codeload.github.com`, respectively.

The issue title/body was used only as untrusted query data. Manual and
deterministic review found no expected changed path or implementation symbol,
private or secret material, prompt-injection phrase, or privacy pattern. The
query review therefore passed for answer leakage, secrets, privacy, and prompt
injection. The PR was closed and merged, its base SHA matched the pinned base,
and the recursive tree was untruncated.

### Source payload hashes

| Source payload | SHA-256 |
|---|---|
| Issue API response | `0458f487e97a088e378d31a9e5a5968bcc87fe23f181d47eefba58220acafc31` |
| Issue title/body query text | `7477e0b63d5762a786126ed6441fe849d391b3adcaa56ae56b1229a796b61ee0` |
| PR API response | `037630886d0c9721e719ba50a05940968f81c603eff7d9921d54e7aca8768f67` |
| PR files API response | `8b4e60a1ac11f1aa7f81a270dcf701b8a07a1416ef4085dca4e23956c3d05d58` |
| Recursive tree API response | `9c0278add0948893e7424ed7e6244f9fe1dde4f2919b405ef7bc8982c81e54f7` |
| Base-commit archive | `0f6375a24b3322b74234f1a6c12d7759c304e9c4c4739ad059d30072deda8269` |

The issue query-text hash serializes the UTF-8 issue title, two LF bytes, and
the UTF-8 issue body, in that order: `title + "\n\n" + body`. No final newline
is appended before hashing.

## Materialization and eligibility evidence

The dedicated path was
`/tmp/meb-modern-embedding-leaderboard-6-1784728508-2-requests-37265838cb26`.
Cleanup was registered before the path and archive were materialized.

| Quantity | Observed | Cap | Status |
|---|---:|---:|---|
| Recursive tree entries | 140 | - | PASS |
| Tracked blobs | 121 | 500 files | PASS |
| Tracked blob bytes | 1,435,436 | - | PASS |
| Archive download bytes | 614,047 | 10,000,000 | PASS |
| Extracted regular-file bytes | 1,435,436 | 25,000,000 | PASS |
| Largest Stage A candidate blob | 82,594 | 2,000,000 | PASS |
| Eligible normalized UTF-8 bytes | 865,106 | 5,000,000 | PASS |
| Chunks | 974 | 1,000 | PASS |
| Peak process RSS | 66,154,496 bytes | 268,435,456 bytes | PASS |
| Final bounded-run wall clock | 11.018 seconds | 1,800 seconds | PASS |

All 121 archive regular files matched the pinned tree API sizes and Git blob
SHAs. There were no submodules, tree symlink blobs, archive symlinks, archive
hardlinks, or Git LFS pointer blobs.

### Stage A

The exact path, mode, excluded-segment, excluded-basename, and suffix policy in
Section 6.2.1 was applied to the pinned tree.

| Stage A result/reason | Files |
|---|---:|
| `accepted` | 99 |
| `excluded_suffix` | 22 |
| Total tracked blobs | 121 |

The older suffix-only metadata calculation also produced 99 files and 865,106
blob bytes. Those values were not treated as eligible counts: Stage A was
recomputed from modes and paths, then Stage B independently inspected all 99
accepted candidates. The equality of the final values is an observed result,
not a relabeling of the older metadata figures.

### Stage B

Git blob SHA and API size verification, oversize rejection, LFS-pointer
rejection, NUL rejection, strict UTF-8 decoding after at most one leading BOM,
the strict greater-than-one-percent control-byte rule, and CRLF/CR-to-LF
normalization were applied in the required order.

| Stage B result/reason | Files |
|---|---:|
| `accepted` | 99 |
| `oversize` | 0 |
| `lfs_pointer` | 0 |
| `binary_nul` | 0 |
| `invalid_utf8` | 0 |
| `control_heavy` | 0 |

Final recomputed eligible corpus: **99 files and 865,106 normalized UTF-8
bytes**. No accepted file had a UTF-8 BOM.

## Chunk, patch, and qrel evidence

- Normalization: `utf8-bom-strip-lf-v0`.
- Chunker: `code-edit-ast-fallback-smoke-v0`.
- Python definition window: 120 lines with 20-line overlap.
- Fixed-line fallback window: 80 lines with 10-line overlap.
- Qrel generator: `patch-preimage-and-insertion-anchor-smoke-v0`.
- Corpus manifest SHA-256:
  `dd702f755cb66bb0091b9de0655bbdcdf62840ab605a9b7c85c77bcfe3186062`.
- Chunk manifest SHA-256:
  `bbf99d3700b75341c4144bdde68533f01fba67e2491ce98057515fb6a46bd0b6`.
- Qrel manifest SHA-256:
  `c215183d86e8f166f4c66e962a307d5e53c1951084e33cd18611ed3ec3966d68`.

Chunk-family counts were 99 `ast_class`, 54 `ast_class_window`, 114
`ast_function`, 499 `ast_method`, 2 `ast_method_window`, 2 `empty_file`, 148
`line_fallback`, 52 `module_preamble`, and 4 `module_preamble_window`.

The pinned patch produced two addition-only pre-change insertion-anchor
targets and two exact grade-2 qrels. Both expected changed paths mapped to the
complete pre-change candidate set. `candidate_coverage` was `1.0`.

- Full-corpus candidate count: 974.
- Ranked candidate count: 974 for the one query.
- `changed_path_prefilter: false`.
- `gold_path_prefilter: false`.
- Patch paths or post-change source were not included in the query.
- Patch context and anchors matched the pinned preimage exactly; no fuzz was
  used.
- No real-source hard negatives were asserted; count was 0 and qrel overlap
  count was 0. A synthetic overlap corruption probe was rejected.

## Deterministic score and corruption checks

The local score matrix was a contract test double only: exact qrel chunks were
assigned 1.0 and every other full-corpus chunk 0.0. It does not measure model
quality.

- Score-matrix SHA-256:
  `1ee34ebb22faf163dd4a66629195ef92fe0673ff0b4a663271be48ab58a2180b`.
- Repeated ranking: identical.
- Exact tie order: score descending, repository, path, line start, chunk id.
- Test-double `nDCG@10`: 1.0.
- Test-double MRR: 1.0.
- Test-double recall@100: 1.0.
- Repeated corpus, chunk, and qrel hashes: identical.
- Corpus-hash mutation: detected.
- Missing-file corpus probe: rejected.
- Missing-chunk qrel probe: rejected.
- Truncated score matrix: rejected.
- Non-finite score: rejected.
- Hard-negative/qrel overlap probe: rejected.

The existing zero-network contract regression file also passed: `10 passed in
0.40s`.

## Commands and execution audit

Key exact commands were:

```bash
git status --short
git branch --show-current
uv --version
uv run python --version
test ! -e /tmp/meb-modern-embedding-leaderboard-6-1784728508-2-requests-37265838cb26
getent ahosts github.com
getent ahosts api.github.com
timeout --signal=TERM 1800s uv run python .perpetuum/modern-embedding-leaderboard/state/requests_smoke_runner_6_1784728508_2.py
sha256sum .perpetuum/modern-embedding-leaderboard/state/requests_smoke_runner_6_1784728508_2.py
wc -l .perpetuum/modern-embedding-leaderboard/state/requests_smoke_runner_6_1784728508_2.py
uv run pytest -q tests/test_code_edit_chunk_localization.py
uv run python -m pytest -q tests/test_code_edit_chunk_localization.py
git diff --check
```

The transient bounded runner was 1,123 lines with SHA-256
`49110ef86d28418d49fa08c5fdb30b8c03f972068524ab4b6cd302b3761834ce`.
It was removed after evidence capture and is not an intended repository
artifact. Because the transient runner was not retained, the recorded inputs,
hashes, parameters, commands, checks, and outcomes make this evidence
auditable, but this note is not a one-command exact rerun artifact. A
one-command exact rerun requires a separately selected and accepted tracked
materializer.

There were three non-source-gate corrections before the authoritative run:

1. preliminary metadata helpers corrected a guessed LICENSE blob identifier,
   an API `Accept` header reused for HTML, and base64 whitespace handling;
   none of those attempts downloaded the repository archive;
2. the first full runner invocation reached the corruption probes but used an
   ineffective incomplete-corpus mutation that removed only one chunk from a
   multi-chunk file; it reported FAILED and cleaned the dedicated path; and
3. the probe was corrected to remove every chunk for one eligible file, after
   which the authoritative run passed all gates and cleaned the path again.

`uv run pytest` resolved an external Anaconda Python 3.8 pytest/plugin pair and
failed during plugin startup before collecting tests. No dependency was
installed or changed. `uv run python -m pytest` forced the project uv Python
and passed all 10 focused tests.

## Primary-source URLs

- https://github.com/psf/requests
- https://github.com/psf/requests/issues/1920
- https://api.github.com/repos/psf/requests/issues/1920
- https://github.com/psf/requests/pull/1921
- https://api.github.com/repos/psf/requests/pulls/1921
- https://api.github.com/repos/psf/requests/pulls/1921/files
- https://github.com/psf/requests/pull/1921.patch
- https://github.com/psf/requests/tree/3c88e520da24ae6f736929a750876e7654accc3d
- https://api.github.com/repos/psf/requests/git/trees/cf441cdcaa76806a078893bffa79922351a92b0a?recursive=1
- https://github.com/psf/requests/blob/3c88e520da24ae6f736929a750876e7654accc3d/LICENSE
- https://api.github.com/repos/psf/requests/git/blobs/8c5e758401020484b74d00c0aa75debfc7b65155
- https://github.com/psf/requests/archive/3c88e520da24ae6f736929a750876e7654accc3d.tar.gz

## Cleanup, publication, and limitations

The dedicated temporary path was absent after the preliminary failed full run
and after the authoritative PASS. No archive, extracted repository, source
text, generated corpus, qrels file, embedding, cache, index, or large log was
retained.

No provider API was called, no model or dataset was downloaded, no model was
run, no patch was generated, and no Hugging Face Dataset or Space operation was
performed. Provider cost was USD 0. No commit or push was performed.

This is one historical repository and one issue. It is private smoke evidence,
not a public benchmark or a model-quality result. Alternate valid edit
locations and real-source hard negatives were not exhaustively reviewed. The
recorded smoke chunker parameters are deterministic but are not yet exposed as
a reusable product materializer API. Public data or scores remain outside this
item.
