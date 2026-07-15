"""Export benchmark artifacts into Hugging Face-ready folders."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from mm_embed.benchmark.leaderboard import build_leaderboard, write_leaderboard_csv
from mm_embed.benchmark.registry import BenchmarkCatalog, ModelSpec, load_catalog
from mm_embed.benchmark.results import json_safe, load_jsonl


DEFAULT_EXPORT_ROOT = Path("dist/huggingface")
PUBLIC_EXCLUDED_PROVIDERS = {"geevec_api", "geevec_lite"}
PUBLIC_EXCLUDED_MARKERS = ("geevec",)
LEADERBOARD_FIELDNAMES = [
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


def _public_records(records: list[dict[str, Any]], private_model_ids: set[str]) -> list[dict[str, Any]]:
    public = []
    for record in records:
        model = record.get("model") or {}
        provider_result = record.get("provider_result") or {}
        providers = {str(model.get("provider") or ""), str(provider_result.get("provider") or "")}
        model_ids = {str(model.get("id") or ""), str(provider_result.get("model_name") or "")}
        if private_model_ids.intersection(model_ids):
            continue
        if PUBLIC_EXCLUDED_PROVIDERS.intersection(providers):
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


def _public_leaderboard_rows(rows: list[dict[str, Any]], private_model_ids: set[str]) -> list[dict[str, Any]]:
    public = []
    for row in rows:
        provider = str(row.get("provider") or "")
        model_id = str(row.get("model_id") or "")
        model = str(row.get("model") or "")
        if provider in PUBLIC_EXCLUDED_PROVIDERS:
            continue
        if model_id in private_model_ids:
            continue
        if _contains_excluded_marker(provider, model_id, model):
            continue
        public.append(row)
    return public


def _result_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    failed = sum(1 for record in records if record.get("error"))
    total = len(records)
    return {
        "total": total,
        "successful": total - failed,
        "failed": failed,
    }


def _leaderboard_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "rows": len(rows),
        "tasks": len({row.get("task_id") for row in rows if row.get("task_id")}),
        "providers": len({row.get("provider") for row in rows if row.get("provider")}),
    }


def _write_dataset_card(
    path: Path,
    *,
    catalog: BenchmarkCatalog,
    public_models: list,
    result_stats: dict[str, int],
    leaderboard_stats: dict[str, int],
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

- Public model specs: {len(public_models)}
- Task specs: {len(catalog.tasks)}
- Result records: {result_note}
- Leaderboard rows: {leaderboard_stats["rows"]}
- Tasks with leaderboard rows: {leaderboard_stats["tasks"]}
- Providers with leaderboard rows: {leaderboard_stats["providers"]}
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
    public_records: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []

    _write_jsonl(output / "models.jsonl", [asdict(model) for model in public_models])
    _write_jsonl(output / "tasks.jsonl", [asdict(task) for task in catalog.tasks.values()])

    run_dir = catalog.root / "runs"
    if run_dir.exists():
        for run_file in sorted(run_dir.glob("*.yaml")):
            _copy_if_exists(run_file, output / "runs" / run_file.name)

    if results_path:
        result_src = Path(results_path)
        public_records = _public_records(load_jsonl(result_src), private_model_ids)
        _write_jsonl(output / "results" / "latest.jsonl", public_records)
        _write_jsonl(output / "results" / "latest-successful.jsonl", [r for r in public_records if not r.get("error")])

    if leaderboard_path:
        rows, fieldnames = _read_leaderboard_csv(Path(leaderboard_path))
        leaderboard_rows = _public_leaderboard_rows(rows, private_model_ids)
        _write_leaderboard_csv_rows(output / "leaderboards" / "latest.csv", leaderboard_rows, fieldnames)
    elif results_path:
        leaderboard_rows = build_leaderboard(public_records, catalog)
        write_leaderboard_csv(leaderboard_rows, output / "leaderboards" / "latest.csv")

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
            "tasks": len(catalog.tasks),
            "excluded_private_models": len(private_model_ids),
            "include_data": include_data,
            "include_images": include_images,
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
        _write_leaderboard_csv_rows(output / "leaderboard.csv", _public_leaderboard_rows(rows, set()), fieldnames)
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


ROWS = load_rows()
TASKS = sorted({row.get("task_id", "") for row in ROWS if row.get("task_id")})
PROVIDERS = ["All"] + sorted({row.get("provider", "") for row in ROWS if row.get("provider")})


def summary_markdown():
    tasks = len({row.get("task_id") for row in ROWS if row.get("task_id")})
    providers = len({row.get("provider") for row in ROWS if row.get("provider")})
    models = len({row.get("model_id") or row.get("model") for row in ROWS if row.get("model_id") or row.get("model")})
    return "Rows: **{}** | Tasks: **{}** | Providers: **{}** | Models: **{}**".format(len(ROWS), tasks, providers, models)


def task_markdown(task_id):
    details = TASK_DETAILS.get(task_id or "", {})
    if not details:
        return "Scores are task-specific and should not be averaged into a global ranking."
    return "**{}**  \\n{}  \\nPrimary signal: `{}`".format(
        details.get("label", task_id),
        details.get("summary", ""),
        details.get("metric", ""),
    )


def render_table(task_id, provider, query, top_n):
    filtered = [row.copy() for row in ROWS if row.get("task_id") == task_id]
    if provider != "All":
        filtered = [row for row in filtered if row.get("provider") == provider]
    query = (query or "").strip().lower()
    if query:
        filtered = [
            row
            for row in filtered
            if query in " ".join(
                str(row.get(key) or "").lower()
                for key in ("model", "model_id", "provider", "run_id", "primary_metric")
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
    return pd.DataFrame(filtered, columns=columns)


def render(task_id, provider, query, top_n):
    if not task_id:
        return "No leaderboard rows are available.", pd.DataFrame()
    return task_markdown(task_id), render_table(task_id, provider, query, top_n)


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
        query = gr.Textbox(label="Search", placeholder="Filter by model, provider, run, or metric")
        table = gr.Dataframe(
            value=render_table(default_task, "All", "", 30) if default_task else pd.DataFrame(),
            label="Leaderboard",
            interactive=False,
        )
        controls = [task, provider, query, top_n]
        task.change(render, inputs=controls, outputs=[task_note, table])
        provider.change(render, inputs=controls, outputs=[task_note, table])
        query.change(render, inputs=controls, outputs=[task_note, table])
        top_n.change(render, inputs=controls, outputs=[task_note, table])
    demo.launch()


if __name__ == "__main__":
    main()
'''
    return source.replace("__DATASET_REPO_ID__", repr(default_repo)).replace("__TASK_DETAILS__", repr(TASK_NOTES))
