# BRIGHT Non-Technical Three-Track Pilot Audit

## Decision

`biology`, `economics`, and `psychology` form the pilot shortlist. The fixed
source is `xlangai/BRIGHT` at Hugging Face revision
`3066d29c9651a576c8aba4832d249807b181ecae`. The materializer does not use a
dataset loading script and never enables remote code (`trust_remote_code=False`).

The repository and dataset card declare CC-BY-4.0. This is not sufficient
evidence that the BRIGHT authors owned sublicensable public-redistribution
rights for every copied blog, news, article, report, or other upstream web
page. Canonical data is therefore `research_only`; public corpus export is
fail-closed pending a document-level rights review. Queries, labels, hashes,
statistics, and the materializer may be reviewed separately, but this audit
does not authorize their publication.

## Primary sources and fixed evidence

| Evidence | Fixed URL or revision | SHA256 / identity | Finding |
| --- | --- | --- | --- |
| Official HF repository API | `https://huggingface.co/api/datasets/xlangai/BRIGHT/revision/3066d29c9651a576c8aba4832d249807b181ecae` | repository SHA equals the requested revision | Public, non-gated repository; ordinary Parquet files |
| Official HF dataset card | `https://huggingface.co/datasets/xlangai/BRIGHT/raw/3066d29c9651a576c8aba4832d249807b181ecae/README.md` | `e5fc567b168b0072f2813d5441e44dd8f2fff642f3a2dd2d9c3dd31ae203d062` | Declares CC-BY-4.0 and exact schemas/row counts |
| Official GitHub repository | `https://github.com/xlang-ai/BRIGHT`, main observed at `d99e8391d967d4c2b3a74732530d2309e2fc92b6` | README `0c161234b3746adc14ed8c3b0ea28f7dedd2252523c286c5376b6ec398dc301d` | Describes 12 domains and documents as blogs, news, articles, reports, etc. |
| Official GitHub license | `https://raw.githubusercontent.com/xlang-ai/BRIGHT/main/LICENSE` | `7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661` | CC-BY-4.0 text; does not prove chain of title for copied pages |
| Paper | `https://arxiv.org/abs/2407.12883v4` | arXiv identifier and version | First submitted 2024-07-16; updated 2025-03-26 |

The exact six Parquet source hashes, byte sizes, row counts, schemas, footer
metadata, and bounded sample hashes are recorded in
`benchmark/artifacts/bright-nontechnical-pilot-v0.1/preflight.json`.

## Track audit

| Track | Documents | Queries | Source bytes | Label form | Pilot decision |
| --- | ---: | ---: | ---: | --- | --- |
| biology | 57,359 | 103 | 11,246,700 | passage `gold_ids`; long-document IDs separate | Keep; completely materialized |
| economics | 50,220 | 103 | 11,189,139 | same schema and rules | Keep; real bounded preflight passed |
| psychology | 52,835 | 101 | 11,614,422 | same schema and rules | Keep; real bounded preflight passed |

All three source files use the same document schema (`id`, `content`) and
example schema (`query`, `reasoning`, `id`, `excluded_ids`, `gold_ids_long`,
`gold_ids`, `gold_answer`). The same materializer validates them. Economics and
psychology preflight reads the real Parquet footer, exact row counts, exact
schema, and eight real rows from both files; it is not a static inventory.

For biology, `gold_ids` produce positive qrels with grade 1. `excluded_ids`
are evaluation-time candidate filters and never negative qrels. At this pinned
revision every non-technical example represents no excluded passages as the
sole literal sentinel `N/A`; the materializer normalizes exactly `["N/A"]` to
an empty list and rejects mixed or dangling values. `gold_ids_long` are retained
as metadata and are never checked against or mixed with passage IDs.

The complete biology audit reports 372 positive qrels, no empty gold set, no
dangling gold/excluded passage ID, no gold/excluded overlap, and only grade 1.
There are 4,613 exact duplicate-content groups. IDs remain unique; duplicates
are retained because removing them would alter the official candidate space.

## Contamination and evaluation scope

BRIGHT was public from July 2024 and its repository/data continued to be used
by public evaluation implementations. Training overlap for later embedding
models is unknown. Results on these tracks must be labeled
`unknown_not_zero_shot_verified`; release date alone is not evidence of no
contamination. Samples must never be selected using a tested model score.

The runner mapping is direct: canonical corpus `id/content`, query `id/text`,
and positive TSV qrels. A runner must remove per-query `excluded_ids` from the
candidate set before ranking and metrics. The pilot intentionally runs no
model and adds no public leaderboard task.

## Other candidates considered

These candidates are not part of this three-track pilot:

- **FreshStack — retain for a later technical track, not this pilot.** The
  official project and NeurIPS 2025 paper describe five recent technical
  domains built from Stack Overflow and GitHub documentation. It is valuable
  for freshness, but it is technical-only and uses nugget coverage rather than
  this pilot's simple positive-qrel contract. Primary sources:
  `https://fresh-stack.github.io`, `https://github.com/fresh-stack/freshstack`,
  and `https://openreview.net/forum?id=54TTgXlS2U`.
- **Legal RAG Bench — retain for separate rights and legal-domain review.** Its
  official paper targets retrieval over legal corpora, which is complementary,
  but legal sources require a document-level rights and provenance audit and
  are outside the no-default-legal-crawl boundary. Primary paper:
  `https://aclanthology.org/2025.acl-long.168/`.
- **TREC-ToT 2025 — reject for this bounded pilot.** The official track has
  6,407,814 Wikipedia pages and the official Zenodo record is about 9.2 GB.
  Although below the absolute 20 GB ceiling, it is far larger than needed for
  this atomic materialization and is a one-relevant-item identification task.
  Primary sources: `https://trec-tot.github.io/guidelines.html`,
  `https://trec.nist.gov/data/tot.html`, and
  `https://zenodo.org/records/15356599`.
- **NQ-UTD — reject pending source/rights repair.** The official Cocktail paper
  describes recent-event queries and mixed original/LLM-generated documents,
  but the linked HF repository card has missing/empty metadata and the data
  viewer cannot establish a clean schema or redistribution contract. It cannot
  pass this task's fixed-source fail-closed gate. Primary sources:
  `https://aclanthology.org/2024.findings-acl.421/` and
  `https://huggingface.co/datasets/IR-Cocktail/nq-utd`.

## Reproduction

```bash
uv sync --extra data
uv run python scripts/bright_materialize_v01.py download \
  --cache-root data/bright-nontechnical-pilot-v0.1/source
uv run python scripts/bright_materialize_v01.py preflight \
  --cache-root data/bright-nontechnical-pilot-v0.1/source \
  --tracks biology economics psychology --sample-rows 8 \
  --output benchmark/artifacts/bright-nontechnical-pilot-v0.1/preflight.json
uv run python scripts/bright_materialize_v01.py materialize \
  --cache-root data/bright-nontechnical-pilot-v0.1/source \
  --track biology --output data/bright-nontechnical-pilot-v0.1/biology
```
