# Public task source and reproducibility contract

Date: 2026-07-28

Scope: the four public tasks declared in `benchmark/tasks/core.yaml`:
`mrl_stress`, `crosslingual_retrieval`, `needle_in_haystack`, and
`cross_modal_retrieval`.

Layer-1 session:
`meb-modern-embedding-leaderboard-6-1785244097-2-public-task-source-contract-a7c4e91b6d2f`

Startup readiness: `READY`. The dispatch file was read before investigation,
the worktree started clean on `main`, and `HEAD`, `origin/main`, and the
dispatch baseline all resolved to
`6823843f1504e289c653e15d0245ca42e096b2f9`.

This is a research artifact only. It does not approve a payload merely because
ignored files exist, and it does not change the registry, loaders, tasks,
result schema, exporter, tests, prepared data, or Perpetuum state.

## Executive decision

| Task | Decision | Short reason |
| --- | --- | --- |
| `mrl_stress` | **BLOCKED** | The upstream repository is named in code, but the preparation call omits a revision and config, the ignored payload has no source manifest, and official Hugging Face metadata could not be resolved in this session. The local 1,379 rows therefore cannot be bound to an immutable upstream revision or reviewed license record. |
| `crosslingual_retrieval` | **Declaration-ready, identity scope only** | The complete sample universe is tracked as repository-authored literals, the transformation is deterministic, and the 166-row ignored payload exactly matches the tracked literals. The declaration must state that no separate dataset license or privacy review is recorded; it must not imply permission for standalone payload redistribution. A fail-closed loader/result patch is still required before accepting new public `benchmark` evidence. |
| `needle_in_haystack` | **BLOCKED** | The ten self-authored needles are reproducible, but the Wikipedia haystacks are not. The preparation path records titles and lengths but not page IDs, revisions, retrieval timestamps, contributor-history attribution, or the exact successful source set. Current page revisions cannot be retroactively assigned to the local haystacks. |
| `cross_modal_retrieval` | **BLOCKED** | The 200 local COCO identities and image bytes are auditable, but the generator alias is mutable and the metadata omits model revision, prompt version, request identity, timestamp, seed, and settings. Per-image rights are heterogeneous, including non-commercial and no-derivatives licenses, and no privacy review is recorded. |

`Declaration-ready, identity scope only` means there is enough evidence to
state the canonical local sample identity without inventing upstream lineage.
It does not mean the current product is safe to publish a new score. All four
tasks still need the common fail-closed implementation described below.

## Existing contract and publication path

The public registry currently declares all four tasks with `use_mock: false`
and public/default publication behavior, but no task contains an
`evaluation_sources` object (`benchmark/tasks/core.yaml:2-54`). Missing source
declarations normalize to `disclosure: unknown` through
`src/mm_embed/benchmark/registry.py:153-198,273-310`. The relationship registry
is versioned but empty (`benchmark/training_overlap_relationships.yaml:1-4`).

The existing research contract already requires at least one approved source
for future public real-data tasks and defines the registry tuple as canonical
`source_id`, revision, config, split, and transformation ID
(`benchmark/research/training_overlap_zero_shot_contract_20260728.md:381-416`).
It also preserves pre-contract results as legacy/unknown without silent
recomputation (`benchmark/research/training_overlap_zero_shot_contract_20260728.md:584-614`).

New result records are built with a training-overlap assessment in
`src/mm_embed/benchmark/results.py:107-151`. Public task source metadata is
projected into exported task catalogs by
`src/mm_embed/benchmark/training_overlap.py:509-527` and
`src/mm_embed/hf_publish/export.py:217-236,1352-1409`. The current exporter has
a post-contract gate for a missing training-overlap assessment, exercised by
`tests/test_benchmark_v2.py:2111-2128`, but there is no equivalent materialized
data/source-manifest gate.

Tests confirm that the default catalog exposes source disclosure as unknown
and the relationship table as empty (`tests/test_benchmark_v2.py:45-56`). They
also preserve historical rows as `legacy_missing_contract` without changing
scores (`tests/test_benchmark_v2.py:79-111,2004-2029`). No current test proves
that these four public tasks used real data, rejects their fixture fallback,
binds payload hashes to registry declarations, or detects missing cross-modal
assets. The dedicated needle tests instantiate `use_mock=True` and validate
embedding-call/cardinality evidence, not real-source identity or missing-data
behavior (`tests/test_needle_in_haystack.py:63-155`).

The fallback fixtures themselves are repository-authored: eight MRL pairs,
ten synthetic solid-color cross-modal objects, and generated needle documents
are defined in `src/mm_embed/data/mock.py:65-88,91-118,121-189`. They are valid
development fixtures but have no source identity compatible with the public
real-data task declarations.

The tracked `core-text-standard` run is explicitly `evidence_tier: benchmark`
and selects `mrl_stress`, `crosslingual_retrieval`, and `needle_in_haystack`
(`benchmark/runs/core-text-standard.yaml:1-16`). Consequently, a loader-level
fallback can become nominal `benchmark` evidence unless a new gate is added.

## Common acceptance rule

A new public row with `run.evidence_tier == "benchmark"` must be rejected
unless all of the following are true:

1. The task has an approved, complete `evaluation_sources` declaration.
2. The run used `data_mode: real`; explicit fixtures are allowed only for an
   unpublished fixture/smoke run.
3. A versioned materialization manifest exists and matches the task ID,
   `dataset_version`, source IDs and revisions, transformation ID, parameters,
   payload hashes, row/asset identities, and exact counts.
4. Every required payload and asset exists and matches its size and SHA256.
5. The loader returned exactly the manifest-declared rows/assets. It did not
   skip unreadable or missing objects.
6. Any generator-backed field is bound to a recorded provider, immutable model
   revision where available, prompt hash, parameters, seed or explicit
   non-determinism declaration, request/time evidence, and output hashes.
7. Rights, attribution, redistribution, privacy, and publication fields are
   reviewed for the actual source revision and transformation. Unknown may be
   recorded, but an approved declaration must not silently imply clearance.
8. The result snapshots the manifest schema version and SHA256. The exporter
   validates the snapshot instead of consulting today's registry to backfill
   old rows.

Historical rows may remain visible as legacy score evidence with source status
`unknown` and reason `legacy_missing_source_contract`. They do not need to be
recomputed solely to remain historical, but they must not be relabeled from
current registry or current local files. A reviewed migration requires the
historical model revision, exact task payload manifest, and run binding.

## Task analysis: `mrl_stress`

### Actual source and transformation

`scripts/prepare_mrl_data.py:26-36` calls
`load_dataset("mteb/stsbenchmark-sts", split="test")`. It does not pass a
revision or config. The tracked transformation rounds scores to two decimal
places, writes all test pairs, creates a first-occurrence unique sentence
corpus, and creates retrieval pairs for scores at least 4.0
(`scripts/prepare_mrl_data.py:39-70,113-139`). The evaluated task does not use
the derived corpus or retrieval-pair files; it loads all continuous STS rows
from `stsb_test.jsonl` (`src/mm_embed/data/real_data.py:53-72`).

The registry selects at most 150 rows, defaults to hard mode, and evaluates
dimension 128 as the primary metric (`benchmark/tasks/core.yaml:2-15`). The
task uses deterministic `random.Random(42)` sampling, preferentially selects
mid-range scores, takes about 20% easy examples, then shuffles the selected
rows (`src/mm_embed/tasks/mrl_stress.py:47-58,84-115`). Thus the effective
evaluation slice is not simply "STS-B test"; it is the ordered prepared test
payload plus the local hard-sampling transform and run parameters.

Self-authored parts are the preparation transform, threshold, local sampler,
and metric implementation. The sentences and human similarity scores are
externally sourced. No generated content is used by the real path.

### Local ignored evidence

| File | Rows | Bytes | SHA256 |
| --- | ---: | ---: | --- |
| `data/mrl_stress/stsb_test.jsonl` | 1,379 | 207,485 | `a5f284b06dc1ba3ea1e5ca3eff88b84dbf7086f8c346d234b42d762e4579c4a7` |
| `data/mrl_stress/corpus.jsonl` | 2,552 | 203,372 | `c7a89e57d903425fe448f2310c814722458c58d622a89c6c9c2aec034600bc93` |
| `data/mrl_stress/retrieval_pairs.jsonl` | 338 | 56,236 | `b5a88af8c4233f86737bcbb6cb9cffe488aec2e550808c1686332ac6a67044bb` |

The ordered hash of per-row canonical hashes for `stsb_test.jsonl` is
`1ba8ef95510cd54e502464b6646590829399598cf8437d438857c758dc0dda43`.
Scores span 0.0 through 5.0. There is one exact duplicate STS row. The corpus
has 2,552 unique IDs and 2,552 unique text hashes. The 338 retrieval rows are
unique, but only 336 distinct corpus IDs are referenced.

These hashes identify the local bytes; they do not prove which Hugging Face
revision produced them. The preparation script SHA256 is
`dec1428ee068333a798933e1dcd16366faf66a78c50dcd4c98a1eddf12d67478`
and its Git blob is `66d75c5ed44d8fe360b0d2a4350637bdd8ef1f55` at the reviewed baseline.

### Pinning and rights status

The only source identity supported by code is
`hf:mteb/stsbenchmark-sts`. `split: test` is supported. A config and immutable
source revision are not currently supported by the preparation record and
must not be invented.

Normal official metadata requests to
`https://huggingface.co/api/datasets/mteb/stsbenchmark-sts`,
`https://datasets-server.huggingface.co/size?dataset=mteb%2Fstsbenchmark-sts`,
and `https://datasets-server.huggingface.co/splits?dataset=mteb%2Fstsbenchmark-sts`
all failed with `curl: (6) Could not resolve host`. Per dispatch, no mirror,
proxy, hard-coded IP, alternate host, credential, or cached dataset material
was used. Therefore this review cannot verify the repository commit, config,
dataset card, authorship, license, attribution requirements, redistribution
conditions, or publication status from the official source.

The source texts are public benchmark material and may contain ordinary
person references, but there is no task-specific privacy review. Local
possession and the task label are not rights evidence.

### Risks and decision

- **Silent fallback:** if `stsb_test.jsonl` is absent, the public task silently
  switches to eight self-authored mock pairs and converts their binary labels
  to 0/5 scores (`src/mm_embed/tasks/mrl_stress.py:72-83`).
- **Stale or cross-machine data:** ignored bytes have no preparation time,
  source revision, library version, config, or manifest. A new machine can
  resolve a different upstream revision.
- **Selection identity:** the result does not record the 150 selected source
  row IDs or their ordered hash. A dataset reorder changes the seed-42 slice.
- **Duplicate and leakage:** one exact input row is duplicated. STS-B is a
  widely published benchmark; no training-overlap conclusion is possible
  until the evaluation source is declared and model training evidence is
  reviewed.

Decision: **BLOCKED**. Do not add an approved registry declaration or accept a
new public benchmark row until official metadata is reachable, the source
revision/config/license review is complete, and a pinned materialization is
created or the current bytes are independently bound to that revision.

Candidate IDs, only after those conditions are met:

- source ID: `hf:mteb/stsbenchmark-sts`
- transformation ID: `meb-mrl-stsb-continuous-hard-sample-v1`

## Task analysis: `crosslingual_retrieval`

### Actual source and transformation

The preparation script declares manually curated and programmatically
constructed Chinese-English pairs and contains the complete sample literals
(`scripts/prepare_crosslingual_data.py:1-15,26-40`). It performs no network
access, randomization, model generation, translation API call, or external
file read. It writes the list in source order to
`parallel_pairs.jsonl` (`scripts/prepare_crosslingual_data.py:542-571`).

The file was introduced in commit
`e53f6f038742490ea840cc9c2e0a77a469be655d` on 2026-03-19 by Cheney Zhang.
At the reviewed baseline its Git blob is
`b31c13f17e1e432ccd9dfd4a9d5bdb4c5b812acd` and its SHA256 is
`58e65916fb7b5fca10a946c883a1401fa6bd9a988e4c44f79d25022d1ab8d17f`.
This is evidence of repository authorship and publication, not a legal
attestation that every sentence is original or cleared.

The real loader requires `parallel_pairs.jsonl` and preserves source order
(`src/mm_embed/data/real_data.py:145-183`). The task evaluates the full pool
and its hard negatives (`src/mm_embed/tasks/crosslingual_retrieval.py:52-71,
82-179`). Although the constructor accepts `use_mock`, the run method never
uses it; missing data produces an error through the outer exception handler
rather than a fixture fallback (`src/mm_embed/tasks/crosslingual_retrieval.py:49-56,199-206`).

### Local ignored evidence

`data/crosslingual/parallel_pairs.jsonl` contains 166 rows, 47,365 bytes, and
has SHA256
`c670044d04631178a754402013c6f621ea48946037f3bf927ab95177ce0ad1ce`.
Its ordered row-manifest SHA256 is
`34408e5c6f3fc89be70f951b1ac2f377f4304858aed8d7ef5de1f2f950a1ffd9`.

A read-only AST reconstruction of the tracked literals found 50 easy-list
rows, 31 medium-list rows, 45 hard-list rows, and 40 additional rows. The
resulting 166 expected objects match the ignored JSONL exactly, in order,
without importing or executing the preparation module. The payload contains:

- 56 easy, 62 medium, and 48 hard rows;
- 166 unique bilingual pair hashes and no exact duplicate rows;
- 152 English hard negatives, all hash-unique;
- 152 Chinese hard negatives, all hash-unique.

The script header's claim of "200+" pairs is stale and must not be used as a
count assertion. No stable row ID is stored; current identity is positional
plus a canonical row hash.

### Rights, privacy, and publication status

The source literals are tracked in the public project repository at
`https://github.com/zc277584121/modern-embedding-bench/blob/6823843f1504e289c653e15d0245ca42e096b2f9/scripts/prepare_crosslingual_data.py`.
No `LICENSE*`, `COPYING*`, or `NOTICE*` file exists in the reviewed tree, and
the preparation file has no dataset-specific license, attribution statement,
source bibliography, or privacy review. Default copyright therefore remains
the conservative interpretation; the note does not grant redistribution
rights. The rows are already publicly visible as source code, but that is not
equivalent to permission to package and redistribute a standalone dataset.

The literals appear to be generic authored examples and no external corpus is
declared. That supports a self-authored source classification, but not a claim
that the rows are free of third-party phrasing, personal data, or cultural
sensitivity. A maintainer should record an authorship/originality and privacy
review before marking the rights review complete.

### Risks and decision

- **Public exposure/leakage:** all evaluation strings have been public in the
  repository since at least 2026-03-19. Later-trained or continuously trained
  models may have seen exact rows. Training-overlap status must remain unknown
  unless model evidence resolves that exposure.
- **Row identity:** the file has no row IDs, source version field, or manifest.
- **Stale documentation:** the advertised count is inconsistent with the
  tracked and materialized count.
- **Inert mode flag:** `use_mock` is accepted but ignored, which is misleading
  even though it does not silently fall back.
- **Rights:** public source availability is not a standalone redistribution
  license.

Decision: **Declaration-ready, identity scope only**. The evidence supports a
complete local-source identity declaration, with rights fields explicitly
stating `license: not_declared`, `redistribution: not_reviewed`, and
`privacy_review: not_recorded`. It does not support a broader claim. New
public `benchmark` evidence still waits for the common fail-closed patch and a
reviewed manifest.

Supported IDs:

- source ID:
  `repo:zc277584121/modern-embedding-bench#crosslingual-pairs`
- source revision: `6823843f1504e289c653e15d0245ca42e096b2f9`
- config: `authored-zh-en-hard-negatives`
- split: `evaluation`
- transformation ID: `meb-crosslingual-literal-jsonl-v1`

## Task analysis: `needle_in_haystack`

### Actual sources and transformation

The task combines two different source classes:

1. Ten externally sourced English Wikipedia article extracts, fetched by
   title through the MediaWiki `extracts` API
   (`scripts/prepare_needle_data.py:21-33,93-118`).
2. Ten self-authored fictional needle/query/category records stored as tracked
   literals (`scripts/prepare_needle_data.py:35-87`).

The script fetches each article independently, catches and continues after an
individual failure, sleeps between requests, concatenates whatever succeeded
in configured order, and truncates the shared concatenation to five target
lengths (`scripts/prepare_needle_data.py:121-138,164-201`). It stores only
article titles and text lengths, not page IDs or revisions
(`scripts/prepare_needle_data.py:257-268`). The loader inserts each needle at
configured positions at run time (`src/mm_embed/data/real_data.py:191-255`).

The local needle literals are self-authored and deterministic. The haystacks
are external and transformed. The final evaluation documents are generated
locally by inserting self-authored needles into externally sourced text; no
model generator is involved.

### Local ignored evidence

| File | Rows | Bytes | SHA256 |
| --- | ---: | ---: | --- |
| `data/needle_haystack/articles_meta.jsonl` | 10 | 474 | `621f76283c4bc22814a6b605ecaf677cb495a32cc941c7d51db4d2307e5cdc6e` |
| `data/needle_haystack/haystacks.jsonl` | 5 | 61,391 | `c203e11a261e8dafdac81cc8256eef9459aac6883d9887954c48270b106a2d8a` |
| `data/needle_haystack/needles.jsonl` | 10 | 2,947 | `7ad20aca88e22d903079fb0a0e46a080ee6b7a608808021017909a8bf526dc59` |

The ten local titles match the tracked configured title set. The haystack rows
declare 1,000, 4,000, 8,000, 16,000, and 32,000 characters; actual lengths are
997, 3,995, 7,997, 15,999, and 31,995. Every shorter haystack is an exact
prefix of the next longer haystack. The ordered `(declared length, text hash)`
manifest SHA256 is
`0a1fe0c3743c99b714356dd58a6cd41900fcdaf7116e5160e06c9884537128f1`.

The ten needle rows exactly match the tracked self-authored literals, with ten
unique needle hashes and ten unique query hashes. Their ordered manifest
SHA256 is
`f8484da82d56581760d934113457ea19054f76f6fc049c3697fff9f1322f8cd2`.
The preparation script SHA256 is
`f1958f9cc362fada01db3d6c4baaa0eca3d2e3ffeb11b2e0feeb05ab6f53a422`
and its reviewed Git blob is `e28c296d0d40a1302acf63bd1d3d53a8f02b4be0`.

### Upstream revision and rights evidence

The official MediaWiki API currently resolves the configured titles to these
page identities and current revisions:

| Page ID | Canonical title | Current revision at review |
| ---: | --- | ---: |
| 7,955 | DNA | 1,366,210,585 |
| 13,692 | History of the Internet | 1,363,175,897 |
| 15,043 | International Space Station | 1,366,309,861 |
| 24,544 | Photosynthesis | 1,366,154,845 |
| 24,944 | Plate tectonics | 1,363,980,827 |
| 25,220 | Quantum computing | 1,366,460,372 |
| 25,532 | Renaissance | 1,365,516,125 |
| 29,664 | Supply and demand | 1,362,821,418 |
| 5,042,951 | Climate change | 1,365,481,997 |
| 42,193,218 | Human digestive system | 1,363,439,273 |

These are current diagnostic identities only. They cannot be assigned to the
local haystacks because the preparation output contains no revision IDs or
retrieval timestamps and the article contents were not downloaded or compared
in this review.

`https://en.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json`
reported English Wikipedia text rights as Creative Commons Attribution-Share
Alike 4.0 and linked to the official license deed. Authorship is collaborative
and revision-specific; compliant attribution requires preserving the source
page and history relationship. The local preparation output omits that
history binding and does not record how share-alike applies to the concatenated
and truncated material. Publication is public on Wikipedia, but privacy review
is not evidenced merely by publication and the source pages may discuss living
people.

### Risks and decision

- **Unpinned source:** titles are not revisions. Current revisions are not
  evidence of the revisions used to create the local bytes.
- **Partial-source drift:** a temporary failure for any page silently changes
  the source set and every haystack while the script still writes output.
- **Duplicate/nested evidence:** all five lengths are nested prefixes of one
  concatenation, so length buckets are not independent samples.
- **Silent fixture fallback:** if either real file is missing, the public task
  silently generates mock haystacks and mock needles
  (`src/mm_embed/tasks/needle_in_haystack.py:58-75`).
- **Silent length omission:** the loader skips any requested length absent from
  the ignored file (`src/mm_embed/data/real_data.py:226-230`).
- **Training leakage:** Wikipedia text is widely used in pretraining. The
  source relationship must be declared before interpreting any zero-shot
  claim. The self-authored needles are public in the repository and may also
  become trainable after publication.
- **Attribution and privacy:** revision histories, license attribution, and a
  privacy review are absent from the materialization record.

Decision: **BLOCKED**. Existing rows may remain legacy/unknown. New benchmark
evidence requires a fresh pinned materialization or independently verifiable
revision binding; current revision metadata is not enough.

Candidate IDs for a future regenerated manifest:

- article source IDs: `wikimedia:enwiki:page/<pageid>` with each exact revision
- self-authored source ID:
  `repo:zc277584121/modern-embedding-bench#needle-facts`
- transformation ID: `meb-enwiki-concat-truncate-needle-insert-v1`

The article source IDs must not be added for the current payload because its
revision tuple is unknown.

## Task analysis: `cross_modal_retrieval`

### Actual sources and transformation

The preparation script downloads the COCO 2017 captions annotation archive,
extracts `captions_val2017.json`, takes the first caption encountered for each
image, shuffles image IDs with `random.Random(42)`, and selects the first
requested IDs (`scripts/prepare_cross_modal_data.py:39-96`). Despite the name
and an unused supercategory constant, this is seeded random selection, not
category-stratified diversity sampling.

For each selected COCO image, the script downloads the JPEG and stores local
row ID, COCO image ID, path, and original COCO caption
(`scripts/prepare_cross_modal_data.py:99-130`). It then calls the mutable
`gpt-4o-mini` alias to generate a fresh caption with image detail `low`,
`max_tokens=200`, and temperature 0.3
(`scripts/prepare_cross_modal_data.py:133-178`). It makes a second call to the
same alias for three hard negatives with `max_tokens=500`, temperature 0.7,
and JSON-object response format
(`scripts/prepare_cross_modal_data.py:181-235`). No API seed is set.

The final metadata keeps the externally sourced original caption, the
generated caption, generated hard negatives, and inferred category, but does
not record the generator model/revision, request ID, system fingerprint,
timestamp, prompt hash, parameters, seed, or generation attempt
(`scripts/prepare_cross_modal_data.py:306-389`).

The real loader uses only the generated caption, image bytes, inferred
category, and generated hard negatives. It silently skips a metadata row if
the referenced image is missing (`src/mm_embed/data/real_data.py:110-137`).
The task can then seed-42 sample the loaded list if it exceeds `max_samples`
(`src/mm_embed/tasks/cross_modal_retrieval.py:58-70`).

External parts are COCO annotations, COCO/Flickr images, and original COCO
captions. Generated parts are the evaluation captions and hard negatives.
Self-authored parts are the selection, prompt, category heuristic, JSONL
transform, loader, and metric.

### Local ignored evidence

- `data/cross_modal/metadata.jsonl`: 200 rows, 298,796 bytes, SHA256
  `908c99851add8f8da59d7e2607fdaad0375d68034199a8833df43beec4aebde8`.
- `data/cross_modal/.cache/captions_val2017.json`: 3,872,473 bytes, SHA256
  `afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9`.
- 200 referenced JPEGs, all present, no orphan files, no duplicate image
  hashes, total 32,933,215 bytes.
- Ordered image asset manifest SHA256:
  `ca0490a84b6b577b15a16576894d6de8b0e24227f79b7c1ce2d9617db8f38f1f`.
- Source mapping manifest SHA256 over local row ID, COCO image ID, image path,
  first annotation ID, and image license ID:
  `41947e8dbf427a6514fbd489e3b85e09bdebeba0f2b2cf8224b35f0a9968e2f2`.
- Row IDs are exactly 0 through 199; there are 200 unique COCO IDs, 200 unique
  image paths, 200 unique generated-caption hashes, and 200 unique original
  caption hashes.
- All 200 stored original captions match the first annotation for the same
  image in the cached official annotation file.
- There are 600 generated hard negatives and all 600 hashes are unique.
- No metadata row records generator model, generator settings, or source
  revision.

The cached annotation metadata identifies COCO Consortium, version 1.0, year
2017, and creation date 2017-09-01. The reviewed script SHA256 is
`61f72431d1f7c445d3806f21787849ec38b86f68eeaa080728d436f4354249e3`
and its Git blob is `f070ead46bcbd61c2ad9404c328f8533c748b3a5`.

### Source revision, rights, and privacy

The official annotation archive endpoint
`http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
returned HTTP 200 headers with `Last-Modified: Tue, 10 Jul 2018 17:58:17 GMT`,
ETag `f4bbac642086de4f52a3fdda2de5fa2c`, and content length 252,907,541 bytes.
No archive body was downloaded. HTTPS failed certificate hostname validation;
the HTTP endpoint is the URL used by the tracked script.

The official COCO site source at revision
`5e1c4da72464b1c6f068df0c02c91e3000ea62c4` states that annotations and the
website are COCO Consortium material under CC BY 4.0, while image copyright is
not owned by COCO and image use must comply with Flickr terms. This is
consistent with the cached annotation license catalog, which binds each image
to a per-image license ID.

Among the 200 selected images, local source mapping yields:

| COCO license ID | License family recorded by COCO | Selected images |
| ---: | --- | ---: |
| 1 | CC BY-NC-SA 2.0 | 65 |
| 2 | CC BY-NC 2.0 | 27 |
| 3 | CC BY-NC-ND 2.0 | 51 |
| 4 | CC BY 2.0 | 29 |
| 5 | CC BY-SA 2.0 | 15 |
| 6 | CC BY-ND 2.0 | 13 |

The preparation output discards each image's license ID, Flickr URL, COCO URL,
and creator attribution even though these fields are available in the source
annotation file. Non-commercial and no-derivatives restrictions require a
specific legal review for the benchmark's use, any generated caption
relationship, screenshots, and redistribution. The local images may contain
identifiable people; there is no consent, sensitive-content, biometric, or
privacy review in the payload.

The generator calls were historical and do not record which API agreement or
terms version governed the output. A read-only request to
`https://openai.com/policies/services-agreement/` returned HTTP 403 in this
session. No alternate source was used, and no ownership or redistribution
conclusion is inferred from the API call or local possession of outputs.

### Risks and decision

- **Generator non-reproducibility:** the mutable model alias, no seed, two
  nonzero temperatures, and missing request/model metadata prevent exact
  regeneration or audit.
- **Rights fragmentation:** every selected image uses a license in IDs 1-6;
  many are non-commercial or no-derivatives. Annotation rights do not clear
  image rights.
- **Silent fixture fallback:** missing metadata switches the public task to ten
  synthetic solid-color fixtures (`src/mm_embed/tasks/cross_modal_retrieval.py:58-65`).
- **Silent asset loss:** missing JPEGs are skipped instead of failing, so
  cross-machine row count can shrink without invalidating the run.
- **Stale source:** the local extracted annotation hash is exact, but there is
  no recorded archive digest or preparation event binding it to the official
  endpoint response.
- **Selection semantics:** the source calls the sample diverse but implements
  only seed-42 shuffle; a change in annotation order changes the selected IDs.
- **Leakage:** COCO images and captions are common multimodal pretraining
  material. The original COCO captions remain in metadata even though the
  loader uses generated captions. Training overlap must consider images,
  original annotations, generated captions, and transformation lineage
  separately.

Decision: **BLOCKED**. Existing historical rows remain legacy/unknown. New
benchmark evidence requires per-image rights/attribution records, a reviewed
privacy decision, an immutable COCO source binding, and a generator manifest
or a regenerated deterministic/fully logged caption set.

Candidate IDs, not yet approved:

- source ID: `coco:2017-validation`
- source revision for the local extracted annotation object:
  `sha256:afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9`
- generator source ID: an immutable OpenAI model revision, not the bare
  `gpt-4o-mini` alias
- transformation ID: `meb-coco-seed42-generated-caption-hardneg-v1`

## Minimal versioned materialization schema

The registry declaration identifies the reviewed source universe. A separate
materialization manifest must bind that declaration to the exact ignored bytes
used by a run. A small v1 schema can be tracked under
`benchmark/data_manifests/<task_id>/<dataset_version>.json` without storing
third-party text or media:

~~~json
{
  "schema_version": "1",
  "task_id": "crosslingual_retrieval",
  "dataset_version": "crosslingual-v1",
  "manifest_revision": "2026-07-28.1",
  "created_at": "RFC3339 timestamp",
  "sources": [
    {
      "source_id": "registry-owned opaque ID",
      "source_revision": "immutable revision or content digest",
      "config": "explicit config or null",
      "split": "explicit split or null",
      "role": "examples|haystack|needle|image|annotation|generated_caption",
      "provenance_urls": ["primary source URL"],
      "license_id": "reviewed license identifier or unknown",
      "authorship": "reviewed status",
      "attribution": "required attribution reference or unknown",
      "redistribution": "allowed|restricted|not_reviewed",
      "privacy_review": "approved|restricted|not_recorded",
      "publication_status": "public|private|unknown"
    }
  ],
  "transformation": {
    "transformation_id": "versioned transform ID",
    "code_commit": "git commit",
    "code_path": "tracked preparation path",
    "code_sha256": "hex digest",
    "parameters": {},
    "seed": 42,
    "determinism": "deterministic|generator_logged|non_reproducible",
    "generator": null
  },
  "materialization": {
    "row_identity_scheme": "stable source ID or canonical-row-sha256",
    "row_count": 0,
    "ordered_row_manifest_sha256": "hex digest",
    "asset_count": 0,
    "asset_manifest_sha256": "hex digest or null",
    "files": [
      {
        "path": "relative ignored payload path",
        "bytes": 0,
        "rows": 0,
        "sha256": "hex digest"
      }
    ]
  },
  "validation": {
    "required_files_complete": true,
    "missing_assets": 0,
    "orphan_assets": 0,
    "exact_duplicate_rows": 0,
    "source_binding_complete": true
  },
  "review": {
    "state": "pending|approved|rejected",
    "reviewed_at": null,
    "reviewed_by": null
  }
}
~~~

For generator-backed material, `transformation.generator` is required and must
contain provider, immutable model revision if exposed, model alias, prompt
SHA256, parameters, requested seed, request/response identifiers or a reviewed
reason they are unavailable, generation timestamps, and ordered output hashes.
Raw prompts and output text need not be public if their tracked code or hash is
sufficient and publication policy forbids copying them.

The manifest SHA256 itself must be stored in each new result as, at minimum:

~~~json
{
  "data_source_contract": {
    "schema_version": "1",
    "task_id": "crosslingual_retrieval",
    "dataset_version": "crosslingual-v1",
    "manifest_revision": "2026-07-28.1",
    "manifest_sha256": "hex digest",
    "data_mode": "real",
    "source_ids": ["registry-owned opaque ID"],
    "transformation_id": "versioned transform ID",
    "row_count": 166,
    "asset_count": 0
  }
}
~~~

## Smallest follow-up implementation plan

1. Add the materialization schema and a read-only manifest validator. Validate
   task ID, dataset version, registry source tuple, transformation, payload
   file hashes/counts, and asset completeness.
2. Remove implicit real-to-fixture fallback from `mrl_stress`,
   `needle_in_haystack`, and `cross_modal_retrieval`. If `use_mock` is false,
   missing or invalid real data must return a distinct source-contract error.
   Make `crosslingual_retrieval.use_mock` either explicit and fixture-only or
   remove the inert option.
3. Make loaders exact: no skipped cross-modal images, no skipped needle
   lengths, no partial preparation success, and no unmanifested row reorder.
4. Snapshot the validated manifest binding in `make_result_record`. Mark
   explicit fixtures as `data_mode: fixture` and forbid them for public
   `benchmark` evidence.
5. Gate export: preserve historical rows as legacy/unknown, but reject every
   post-contract public benchmark row missing a valid source snapshot or using
   fixture/partial data.
6. Add focused tests for missing payload, missing manifest, hash mismatch,
   stale dataset version, registry/manifest mismatch, explicit mock in a
   public benchmark run, skipped asset/length, incomplete generator metadata,
   and legacy rows remaining unchanged.
7. Curate manifests in this order: crosslingual first; MRL after official HF
   metadata and a pinned materialization; needle after revision-pinned
   regeneration and attribution review; cross-modal after rights/privacy and
   generator provenance are resolved.

This plan deliberately does not require recomputing historical scores. It
changes the acceptance rule for new evidence and permits an explicit reviewed
migration only when the original run can be bound to an exact manifest.

## Commands and primary-source checks

Read-only local checks included:

- `git status --short --branch`, `git rev-parse HEAD`, upstream resolution, and
  `git rev-parse origin/main` at start;
- `rg`, `sed`, `nl`, `git ls-tree`, `git log --follow`, `sha256sum`,
  `git check-ignore`, and `find` over the required tracked paths and ignored
  data paths;
- `uv run --no-sync python -` using only the standard library for JSONL
  structure/count/hash checks, duplicate checks, COCO ID/license joins, asset
  completeness, and AST-only literal reconstruction. It did not import or run
  a preparation module;
- no benchmark command, provider/model call, inference, test suite, data
  preparation script, upload/export path, dependency installation/update,
  commit, or push.

Primary-source URLs checked:

- `https://huggingface.co/api/datasets/mteb/stsbenchmark-sts` — DNS resolution
  failure;
- `https://datasets-server.huggingface.co/size?dataset=mteb%2Fstsbenchmark-sts`
  — DNS resolution failure;
- `https://datasets-server.huggingface.co/splits?dataset=mteb%2Fstsbenchmark-sts`
  — DNS resolution failure;
- `https://en.wikipedia.org/w/api.php` — rights metadata and page identity/
  current revision metadata only; no extracts were requested;
- `https://cocodataset.org/#termsofuse` and immutable site source
  `https://github.com/cocodataset/cocodataset.github.io/blob/5e1c4da72464b1c6f068df0c02c91e3000ea62c4/dataset/termsofuse.htm`;
- `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` —
  headers only; archive body not downloaded;
- `https://openai.com/policies/services-agreement/` — HTTP 403;
- `https://github.com/zc277584121/modern-embedding-bench/commit/6823843f1504e289c653e15d0245ca42e096b2f9`
  — reviewed project baseline metadata.

Successful response bodies were small structured metadata or the 2.4 KiB
COCO terms source; no dataset payload, model, media, archive, Wikipedia
extract, full article body, credential, or paid API response was fetched.
