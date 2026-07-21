"""Export benchmark artifacts into Hugging Face-ready folders."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from mm_embed.benchmark.leaderboard import build_leaderboard, primary_metric_value
from mm_embed.benchmark.registry import BenchmarkCatalog, ModelSpec, load_catalog, load_run_manifest
from mm_embed.benchmark.results import json_safe, load_jsonl


DEFAULT_EXPORT_ROOT = Path("dist/huggingface")
PUBLIC_EXCLUDED_PROVIDERS = {"geevec_api", "geevec_lite"}
PUBLIC_EXCLUDED_MARKERS = ("geevec",)
LEADERBOARD_BASE_FIELDNAMES = [
    "task_id",
    "task",
    "model_id",
    "model",
    "provider",
    "primary_metric",
    "score",
    "run_id",
    "duration_s",
]
LEADERBOARD_PROVENANCE_FIELDNAMES = [
    "evidence_tier",
    "evidence_source",
    "task_model_duplicate_count",
    "task_model_run_rank",
    "is_latest_for_task_model",
]
LEADERBOARD_FIELDNAMES = LEADERBOARD_BASE_FIELDNAMES + LEADERBOARD_PROVENANCE_FIELDNAMES

TASK_NOTES = {
    "mrl_stress": {
        "label": "MRL compression stress",
        "summary": "Semantic stability when embeddings are truncated to smaller dimensions.",
        "metric": "Spearman correlation at the configured low dimension.",
    },
    "crosslingual_retrieval": {
        "label": "Chinese-English retrieval",
        "summary": "Bidirectional technical retrieval with hard negatives across Chinese and English.",
        "metric": "Hard-negative average recall@1.",
    },
    "needle_in_haystack": {
        "label": "Long-document needle retrieval",
        "summary": "Retrieving facts inserted at different positions in long documents.",
        "metric": "Overall accuracy across length and position buckets.",
    },
    "cross_modal_retrieval": {
        "label": "Text-image retrieval",
        "summary": "COCO-style text-image matching with hard negative captions.",
        "metric": "Hard-negative average recall@1.",
    },
}


def _reset_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(json_safe(data), f, sort_keys=False, allow_unicode=True)


def _is_public_model(model: ModelSpec) -> bool:
    return model.publish and model.provider not in PUBLIC_EXCLUDED_PROVIDERS


def _contains_excluded_marker(*values: Any) -> bool:
    for value in values:
        text = str(value or "").lower()
        if any(marker in text for marker in PUBLIC_EXCLUDED_MARKERS):
            return True
    return False


def _public_records(
    records: list[dict[str, Any]],
    private_model_ids: set[str],
    private_task_ids: set[str],
) -> list[dict[str, Any]]:
    public = []
    for record in records:
        model = record.get("model") or {}
        provider_result = record.get("provider_result") or {}
        task = record.get("task") or {}
        providers = {str(model.get("provider") or ""), str(provider_result.get("provider") or "")}
        model_ids = {str(model.get("id") or ""), str(provider_result.get("model_name") or "")}
        if private_model_ids.intersection(model_ids):
            continue
        if PUBLIC_EXCLUDED_PROVIDERS.intersection(providers):
            continue
        if str(task.get("id") or "") in private_task_ids or task.get("publish") is False:
            continue
        if _contains_excluded_marker(
            model.get("id"),
            model.get("display_name"),
            model.get("provider"),
            provider_result.get("provider"),
            provider_result.get("model_name"),
        ):
            continue
        public.append(record)
    return public


def _read_leaderboard_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or LEADERBOARD_FIELDNAMES)


def _write_leaderboard_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_fields = fieldnames or LEADERBOARD_FIELDNAMES
    extras = sorted({key for row in rows for key in row if key not in ordered_fields})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields + extras)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _leaderboard_fieldnames_with_provenance(fieldnames: list[str] | None) -> list[str]:
    fields = list(fieldnames or LEADERBOARD_BASE_FIELDNAMES)
    for field in LEADERBOARD_PROVENANCE_FIELDNAMES:
        if field not in fields:
            fields.append(field)
    return fields


def _public_leaderboard_rows(
    rows: list[dict[str, Any]],
    private_model_ids: set[str],
    private_task_ids: set[str],
) -> list[dict[str, Any]]:
    public = []
    for row in rows:
        provider = str(row.get("provider") or "")
        model_id = str(row.get("model_id") or "")
        model = str(row.get("model") or "")
        task_id = str(row.get("task_id") or "")
        if provider in PUBLIC_EXCLUDED_PROVIDERS:
            continue
        if model_id in private_model_ids:
            continue
        if task_id in private_task_ids:
            continue
        if _contains_excluded_marker(provider, model_id, model):
            continue
        public.append(row)
    return public


def _leaderboard_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("task_id") or ""),
        str(row.get("model_id") or row.get("model") or ""),
        str(row.get("run_id") or ""),
    )


def _leaderboard_group_key(row: dict[str, Any]) -> tuple[str, str]:
    task_id, model_id, _ = _leaderboard_row_key(row)
    return task_id, model_id


def _record_leaderboard_key(record: dict[str, Any]) -> tuple[str, str, str]:
    task = record.get("task") or {}
    model = record.get("model") or {}
    run = record.get("run") or {}
    return (
        str(task.get("id") or ""),
        str(model.get("id") or model.get("display_name") or ""),
        str(run.get("id") or ""),
    )


def _record_evidence_tier(record: dict[str, Any]) -> str:
    run = record.get("run") or {}
    model = record.get("model") or {}
    task = record.get("task") or {}
    metadata = run.get("metadata") or {}
    tags = [*(model.get("tags") or []), *(task.get("tags") or [])]
    text = " ".join(
        str(value or "").lower()
        for value in (
            run.get("id"),
            run.get("description"),
            model.get("access"),
            metadata.get("legacy_source"),
            *tags,
        )
    )
    if metadata.get("legacy_source") or "legacy" in text:
        return "legacy"
    if "smoke" in text:
        return "smoke"
    return "benchmark"


def _row_evidence_tier(row: dict[str, Any]) -> str:
    run_id = str(row.get("run_id") or "").lower()
    if run_id.startswith("legacy:") or "legacy" in run_id:
        return "legacy"
    if "smoke" in run_id:
        return "smoke"
    return "unknown"


def _record_evidence_source(record: dict[str, Any]) -> str:
    run = record.get("run") or {}
    metadata = run.get("metadata") or {}
    for key in ("legacy_source", "source", "results_path"):
        if metadata.get(key):
            return str(metadata[key])
    if run.get("git_sha"):
        return str(run["git_sha"])
    return str(run.get("id") or "")


def _leaderboard_provenance_by_key(
    records: list[dict[str, Any]],
    catalog: BenchmarkCatalog,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    provenance: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.get("error") or primary_metric_value(record, catalog) is None:
            continue
        provenance[_record_leaderboard_key(record)].append(
            {
                "_source_index": index,
                "evidence_tier": _record_evidence_tier(record),
                "evidence_source": _record_evidence_source(record),
            }
        )
    return provenance


def _enrich_leaderboard_rows(
    rows: list[dict[str, Any]],
    *,
    result_records: list[dict[str, Any]],
    catalog: BenchmarkCatalog,
) -> list[dict[str, Any]]:
    enriched = [dict(row) for row in rows]
    provenance = _leaderboard_provenance_by_key(result_records, catalog) if result_records else {}

    for index, row in enumerate(enriched):
        matches = provenance.get(_leaderboard_row_key(row)) or []
        match = matches.pop(0) if matches else {}
        row["_source_index"] = match.get("_source_index", index)
        row["evidence_tier"] = match.get("evidence_tier") or row.get("evidence_tier") or _row_evidence_tier(row)
        row["evidence_source"] = match.get("evidence_source") or row.get("evidence_source") or ""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[_leaderboard_group_key(row)].append(row)

    for group_rows in groups.values():
        group_rows.sort(key=lambda row: int(row.get("_source_index") or 0))
        group_size = len(group_rows)
        for rank, row in enumerate(group_rows, start=1):
            row["task_model_duplicate_count"] = group_size
            row["task_model_run_rank"] = rank
            row["is_latest_for_task_model"] = str(rank == group_size).lower()
            row.pop("_source_index", None)

    return enriched


def _result_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    failed = sum(1 for record in records if record.get("error"))
    total = len(records)
    return {
        "total": total,
        "successful": total - failed,
        "failed": failed,
    }


def _leaderboard_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = Counter(
        _leaderboard_group_key(row)
        for row in rows
        if row.get("task_id") and (row.get("model_id") or row.get("model"))
    )
    evidence_tiers = Counter(str(row.get("evidence_tier") or "unknown") for row in rows)
    return {
        "rows": len(rows),
        "tasks": len({row.get("task_id") for row in rows if row.get("task_id")}),
        "providers": len({row.get("provider") for row in rows if row.get("provider")}),
        "task_model_pairs": len(groups),
        "duplicate_task_model_repeats": sum(count - 1 for count in groups.values() if count > 1),
        "latest_task_model_rows": sum(1 for row in rows if str(row.get("is_latest_for_task_model")).lower() == "true"),
        "evidence_tiers": dict(sorted(evidence_tiers.items())),
    }


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _write_dataset_card(
    path: Path,
    *,
    catalog: BenchmarkCatalog,
    public_models: list,
    excluded_private_models: int,
    result_stats: dict[str, int],
    leaderboard_stats: dict[str, Any],
    include_data: bool,
) -> None:
    task_rows = []
    for task in sorted(catalog.tasks.values(), key=lambda item: item.id):
        note = TASK_NOTES.get(task.id, {})
        task_rows.append(
            "| {id} | {name} | {metric} | {summary} |".format(
                id=task.id,
                name=note.get("label") or task.display_name,
                metric=task.primary_metric or "",
                summary=note.get("summary") or task.description,
            )
        )

    result_note = (
        f"{result_stats['total']} records, {result_stats['successful']} successful, "
        f"{result_stats['failed']} failed"
        if result_stats["total"]
        else "No result file bundled yet."
    )
    data_note = "Bundled JSONL benchmark inputs are included." if include_data else "Benchmark inputs are not bundled."
    evidence_note = _format_counts(leaderboard_stats["evidence_tiers"])
    path.write_text(
        f"""---
license: mit
tags:
  - benchmark
  - embeddings
  - retrieval
  - multimodal
  - rag
pretty_name: Modern Embedding Bench
---

# Modern Embedding Bench

Modern Embedding Bench evaluates embedding models on practical retrieval tasks
that show up in current AI systems but are often under-covered by broad
leaderboards. The focus is on agent memory, tool and document retrieval,
long-context RAG, cross-lingual technical retrieval, coding-oriented retrieval,
and multimodal search rather than a single aggregate score.

The companion leaderboard Space is available at:
https://huggingface.co/spaces/zc277584121/modern-embedding-bench-leaderboard

The source code is available at:
https://github.com/zc277584121/modern-embedding-bench

## Contents

- `models.jsonl`: registered model specs
- `tasks.jsonl`: registered task specs and primary metrics
- `runs/`: run manifests used to produce results
- `results/latest.jsonl`: canonical v2 result records
- `leaderboards/latest.csv`: flat leaderboard table derived from result records
- `benchmark_data/`: optional benchmark input data exported from the local repo

## Current Public Export

- Registry model specs: {len(catalog.models)}
- Public model specs exported: {len(public_models)}
- Excluded private or preview model specs: {excluded_private_models}
- Task specs: {len(catalog.tasks)}
- Result records: {result_note}
- Leaderboard rows: {leaderboard_stats["rows"]}
- Tasks with leaderboard rows: {leaderboard_stats["tasks"]}
- Providers with leaderboard rows: {leaderboard_stats["providers"]}
- Unique task/model leaderboard pairs: {leaderboard_stats["task_model_pairs"]}
- Duplicate task/model repeats kept for inspection: {leaderboard_stats["duplicate_task_model_repeats"]}
- Latest task/model marker rows: {leaderboard_stats["latest_task_model_rows"]}
- Evidence tiers: {evidence_note}
- Data: {data_note}

## Tasks

| ID | Name | Primary metric | What it probes |
| --- | --- | --- | --- |
{chr(10).join(task_rows)}

## Result Format

Each line in `results/latest.jsonl` is one model-task run. Important fields:

- `run`: run id, description, metadata, and git sha when available
- `model`: model id, display name, provider, modalities, dimensions, and tags
- `task`: task id, dataset version, primary metric, and task kwargs
- `metrics`: task-specific metric dictionary
- `details`: diagnostic details for deeper analysis
- `error`: error text for failed runs, otherwise `null`

## Leaderboard Provenance

`leaderboards/latest.csv` keeps every public row, including historical duplicate
runs for the same `task_id` and `model_id`. The first columns remain compatible
with older CSV readers, and provenance columns are appended:

- `evidence_tier`: `legacy`, `smoke`, `benchmark`, or `unknown`
- `evidence_source`: legacy source file, git sha, or run id when available
- `task_model_duplicate_count`: rows kept for the same task/model pair
- `task_model_run_rank`: 1-based order for that task/model pair
- `is_latest_for_task_model`: `true` for the latest exported row in that pair

Latest markers are computed from the order of `results/latest.jsonl` when result
records are available, otherwise from CSV row order. Use
`is_latest_for_task_model=true` to inspect one current row per task/model pair
without losing the full historical trail.

## Usage

Install and inspect the registry:

```bash
uv sync
uv run modern-embed-bench benchmark models
uv run modern-embed-bench benchmark tasks
```

Run a small OpenAI smoke benchmark:

```bash
uv run modern-embed-bench benchmark run \\
  --manifest benchmark/runs/openai-smoke.yaml \\
  --output results/openai-smoke.jsonl \\
  --overwrite

uv run modern-embed-bench benchmark leaderboard \\
  --results results/openai-smoke.jsonl \\
  --output results/openai-smoke-leaderboard.csv
```

## Notes and Limitations

- Rows imported from legacy runs are published for continuity and should be read
  as historical baseline evidence, not as a fully normalized one-shot run.
- Scores are task-specific. Avoid comparing scores across tasks as if they were
  one global ranking.
- Some preview or private-in-progress model results are intentionally excluded
  from the public export until they are ready for publication.
- Image binaries are not bundled by default; `cross_modal` metadata is included
  separately from the source image files.
""",
        encoding="utf-8",
    )


def _write_export_manifest(path: Path, *, kind: str, files: list[str], metadata: dict[str, Any]) -> None:
    _write_yaml(
        path,
        {
            "kind": kind,
            "files": sorted(files),
            "metadata": metadata,
        },
    )


def export_dataset_repo(
    *,
    output_dir: str | Path = DEFAULT_EXPORT_ROOT / "dataset",
    benchmark_root: str | Path | None = None,
    results_path: str | Path | None = None,
    leaderboard_path: str | Path | None = None,
    include_data: bool = False,
    include_images: bool = False,
    clean: bool = True,
) -> Path:
    """Create a Hugging Face Dataset repo folder."""
    output = Path(output_dir)
    _reset_dir(output, clean=clean)
    catalog = load_catalog(benchmark_root)

    public_models = [model for model in catalog.models.values() if _is_public_model(model)]
    private_model_ids = {model.id for model in catalog.models.values() if not _is_public_model(model)}
    public_tasks = [task for task in catalog.tasks.values() if task.publish]
    private_task_ids = {task.id for task in catalog.tasks.values() if not task.publish}
    public_records: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []

    _write_jsonl(output / "models.jsonl", [asdict(model) for model in public_models])
    _write_jsonl(output / "tasks.jsonl", [asdict(task) for task in public_tasks])

    run_dir = catalog.root / "runs"
    if run_dir.exists():
        for run_file in sorted(run_dir.glob("*.yaml")):
            run_manifest = load_run_manifest(run_file)
            if run_manifest.publish and all(run_task.id not in private_task_ids for run_task in run_manifest.tasks):
                _copy_if_exists(run_file, output / "runs" / run_file.name)

    if results_path:
        result_src = Path(results_path)
        public_records = _public_records(load_jsonl(result_src), private_model_ids, private_task_ids)
        _write_jsonl(output / "results" / "latest.jsonl", public_records)
        _write_jsonl(output / "results" / "latest-successful.jsonl", [r for r in public_records if not r.get("error")])

    if leaderboard_path:
        rows, fieldnames = _read_leaderboard_csv(Path(leaderboard_path))
        leaderboard_rows = _public_leaderboard_rows(rows, private_model_ids, private_task_ids)
        leaderboard_rows = _enrich_leaderboard_rows(leaderboard_rows, result_records=public_records, catalog=catalog)
        _write_leaderboard_csv_rows(
            output / "leaderboards" / "latest.csv",
            leaderboard_rows,
            _leaderboard_fieldnames_with_provenance(fieldnames),
        )
    elif results_path:
        leaderboard_rows = build_leaderboard(public_records, catalog)
        leaderboard_rows = _enrich_leaderboard_rows(leaderboard_rows, result_records=public_records, catalog=catalog)
        _write_leaderboard_csv_rows(
            output / "leaderboards" / "latest.csv",
            leaderboard_rows,
            LEADERBOARD_FIELDNAMES,
        )

    if include_data:
        _copy_benchmark_data(Path("data"), output / "benchmark_data", include_images=include_images)
    else:
        (output / "benchmark_data").mkdir(parents=True, exist_ok=True)
        (output / "benchmark_data" / "README.md").write_text(
            "Benchmark input data was not exported. Re-run with --include-data to bundle JSONL data.\n",
            encoding="utf-8",
        )

    _write_dataset_card(
        output / "README.md",
        catalog=catalog,
        public_models=public_models,
        excluded_private_models=len(private_model_ids),
        result_stats=_result_stats(public_records),
        leaderboard_stats=_leaderboard_stats(leaderboard_rows),
        include_data=include_data,
    )

    files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]
    _write_export_manifest(
        output / "export_manifest.yaml",
        kind="hf_dataset",
        files=files,
        metadata={
            "models": len(public_models),
            "registry_models": len(catalog.models),
            "tasks": len(catalog.tasks),
            "excluded_private_models": len(private_model_ids),
            "include_data": include_data,
            "include_images": include_images,
            "leaderboard": _leaderboard_stats(leaderboard_rows),
        },
    )
    return output


def _copy_benchmark_data(data_root: Path, output: Path, *, include_images: bool) -> None:
    if not data_root.exists():
        output.mkdir(parents=True, exist_ok=True)
        (output / "README.md").write_text("Local data directory was not found.\n", encoding="utf-8")
        return

    for path in sorted(data_root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(data_root)
        if _should_skip_data_file(rel, include_images=include_images):
            continue
        _copy_if_exists(path, output / rel)

    note = [
        "# Benchmark Data Export",
        "",
        "This folder mirrors selected local benchmark input data.",
        "",
        "Embedding caches, tool caches, numpy arrays, and temporary files are intentionally skipped.",
    ]
    if not include_images:
        note.append("Image files were skipped. Re-run with `--include-images` to bundle image assets.")
    (output / "README.md").write_text("\n".join(note) + "\n", encoding="utf-8")


def _should_skip_data_file(rel: Path, *, include_images: bool) -> bool:
    parts = set(rel.parts)
    suffix = rel.suffix.lower()
    if "embedding_cache" in parts or ".cache" in parts:
        return True
    if suffix in {".npy", ".npz", ".tmp", ".pyc"}:
        return True
    if any(part == "__pycache__" for part in rel.parts):
        return True
    if suffix in {".jpg", ".jpeg", ".png", ".webp"} and not include_images:
        return True
    return False


def export_space_repo(
    *,
    output_dir: str | Path = DEFAULT_EXPORT_ROOT / "space",
    dataset_repo_id: str | None = None,
    bundled_leaderboard: str | Path | None = None,
    clean: bool = True,
) -> Path:
    """Create a Hugging Face Gradio Space folder."""
    output = Path(output_dir)
    _reset_dir(output, clean=clean)

    (output / "README.md").write_text(_space_readme(dataset_repo_id), encoding="utf-8")
    (output / "requirements.txt").write_text("gradio>=5.0\npandas>=2.0\nhuggingface_hub>=0.30\n", encoding="utf-8")
    (output / "app.py").write_text(_space_app_source(dataset_repo_id), encoding="utf-8")

    if bundled_leaderboard:
        rows, fieldnames = _read_leaderboard_csv(Path(bundled_leaderboard))
        catalog = load_catalog()
        private_task_ids = {task.id for task in catalog.tasks.values() if not task.publish}
        public_rows = _public_leaderboard_rows(rows, set(), private_task_ids)
        _write_leaderboard_csv_rows(output / "leaderboard.csv", public_rows, fieldnames)
    else:
        _write_empty_leaderboard(output / "leaderboard.csv")

    files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]
    _write_export_manifest(
        output / "export_manifest.yaml",
        kind="hf_space",
        files=files,
        metadata={"dataset_repo_id": dataset_repo_id},
    )
    return output


def _write_empty_leaderboard(path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LEADERBOARD_FIELDNAMES)


def _space_readme(dataset_repo_id: str | None) -> str:
    dataset_line = f"DATASET_REPO_ID: {dataset_repo_id}" if dataset_repo_id else "DATASET_REPO_ID: optional"
    return f"""---
title: Modern Embedding Bench
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Modern Embedding Bench Leaderboard

This Gradio Space renders task-specific leaderboard views from a benchmark Dataset repo.

{dataset_line}
"""


def _space_app_source(dataset_repo_id: str | None) -> str:
    default_repo = dataset_repo_id or ""
    source = '''from __future__ import annotations

import csv
import os
from pathlib import Path

import gradio as gr
import pandas as pd

DEFAULT_DATASET_REPO_ID = __DATASET_REPO_ID__
LEADERBOARD_FILE = "leaderboards/latest.csv"
TASK_DETAILS = __TASK_DETAILS__


def load_rows():
    dataset_repo_id = os.environ.get("DATASET_REPO_ID") or DEFAULT_DATASET_REPO_ID
    local_path = Path("leaderboard.csv")
    if dataset_repo_id:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=dataset_repo_id,
                repo_type="dataset",
                filename=LEADERBOARD_FILE,
            )
            local_path = Path(downloaded)
        except Exception as exc:
            print("Could not load leaderboard from {}: {}. Falling back to bundled data.".format(dataset_repo_id, exc))
    if not local_path.exists():
        return []
    with open(local_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truthy(value):
    return str(value or "").lower() in {"1", "true", "yes", "y"}


def task_model_key(row):
    return (row.get("task_id") or "", row.get("model_id") or row.get("model") or "")


def evidence_summary():
    tiers = {}
    for row in ROWS:
        tier = row.get("evidence_tier") or "unknown"
        tiers[tier] = tiers.get(tier, 0) + 1
    if not tiers:
        return "none"
    return ", ".join("{}={}".format(key, tiers[key]) for key in sorted(tiers))


ROWS = load_rows()
TASKS = sorted({row.get("task_id", "") for row in ROWS if row.get("task_id")})
PROVIDERS = ["All"] + sorted({row.get("provider", "") for row in ROWS if row.get("provider")})
PROVENANCE_COLUMNS = [
    "evidence_tier",
    "evidence_source",
    "task_model_duplicate_count",
    "task_model_run_rank",
    "is_latest_for_task_model",
]


def summary_markdown():
    tasks = len({row.get("task_id") for row in ROWS if row.get("task_id")})
    providers = len({row.get("provider") for row in ROWS if row.get("provider")})
    models = len({row.get("model_id") or row.get("model") for row in ROWS if row.get("model_id") or row.get("model")})
    group_counts = {}
    for row in ROWS:
        if row.get("task_id"):
            key = task_model_key(row)
            group_counts[key] = group_counts.get(key, 0) + 1
    task_model_pairs = len(group_counts)
    duplicate_repeats = sum(max(count - 1, 0) for count in group_counts.values())
    latest_rows = sum(1 for row in ROWS if truthy(row.get("is_latest_for_task_model")))
    if not latest_rows and ROWS:
        latest_rows = task_model_pairs
    return (
        "Rows: **{}** | Tasks: **{}** | Providers: **{}** | Models: **{}**  \\n"
        "Task/model pairs: **{}** | Duplicate repeats: **{}** | Latest markers: **{}** | Evidence: **{}**"
    ).format(len(ROWS), tasks, providers, models, task_model_pairs, duplicate_repeats, latest_rows, evidence_summary())


def task_markdown(task_id):
    details = TASK_DETAILS.get(task_id or "", {})
    if not details:
        return "Scores are task-specific and should not be averaged into a global ranking."
    return "**{}**  \\n{}  \\nPrimary signal: `{}`".format(
        details.get("label", task_id),
        details.get("summary", ""),
        details.get("metric", ""),
    )


def render_table(task_id, provider, query, latest_only, top_n):
    filtered = [row.copy() for row in ROWS if row.get("task_id") == task_id]
    if provider != "All":
        filtered = [row for row in filtered if row.get("provider") == provider]
    if latest_only and any("is_latest_for_task_model" in row for row in filtered):
        filtered = [row for row in filtered if truthy(row.get("is_latest_for_task_model"))]
    query = (query or "").strip().lower()
    if query:
        filtered = [
            row
            for row in filtered
            if query in " ".join(
                str(row.get(key) or "").lower()
                for key in (
                    "model",
                    "model_id",
                    "provider",
                    "run_id",
                    "primary_metric",
                    "evidence_tier",
                    "evidence_source",
                )
            )
        ]
    filtered.sort(
        key=lambda row: as_float(row.get("score")) if as_float(row.get("score")) is not None else float("-inf"),
        reverse=True,
    )
    filtered = filtered[: int(top_n or 50)]
    for index, row in enumerate(filtered, start=1):
        row["rank"] = index
        score = as_float(row.get("score"))
        row["score"] = round(score, 6) if score is not None else row.get("score")
    columns = ["rank", "model", "provider", "score", "primary_metric", "run_id", "duration_s"]
    columns.extend(column for column in PROVENANCE_COLUMNS if any(row.get(column) for row in filtered))
    return pd.DataFrame(filtered, columns=columns)


def render(task_id, provider, query, latest_only, top_n):
    if not task_id:
        return "No leaderboard rows are available.", pd.DataFrame()
    return task_markdown(task_id), render_table(task_id, provider, query, latest_only, top_n)


def main():
    default_task = TASKS[0] if TASKS else None
    with gr.Blocks(title="Modern Embedding Bench") as demo:
        gr.Markdown("# Modern Embedding Bench")
        gr.Markdown(summary_markdown())
        task_note = gr.Markdown(task_markdown(default_task) if default_task else "No leaderboard rows are available.")

        with gr.Row():
            task = gr.Dropdown(TASKS, value=default_task, label="Task")
            provider = gr.Dropdown(PROVIDERS, value="All", label="Provider")
            top_n = gr.Slider(5, 100, value=30, step=5, label="Rows")
        latest_only = gr.Checkbox(value=False, label="Latest row per task/model only")
        query = gr.Textbox(label="Search", placeholder="Filter by model, provider, run, or metric")
        table = gr.Dataframe(
            value=render_table(default_task, "All", "", False, 30) if default_task else pd.DataFrame(),
            label="Leaderboard",
            interactive=False,
        )
        controls = [task, provider, query, latest_only, top_n]
        task.change(render, inputs=controls, outputs=[task_note, table])
        provider.change(render, inputs=controls, outputs=[task_note, table])
        query.change(render, inputs=controls, outputs=[task_note, table])
        latest_only.change(render, inputs=controls, outputs=[task_note, table])
        top_n.change(render, inputs=controls, outputs=[task_note, table])
    demo.launch()


if __name__ == "__main__":
    main()
'''
    return source.replace("__DATASET_REPO_ID__", repr(default_repo)).replace("__TASK_DETAILS__", repr(TASK_NOTES))
