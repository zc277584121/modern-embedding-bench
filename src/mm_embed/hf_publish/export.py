"""Export benchmark artifacts into Hugging Face-ready folders."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from mm_embed.benchmark.leaderboard import build_from_file
from mm_embed.benchmark.registry import BenchmarkCatalog, load_catalog
from mm_embed.benchmark.results import json_safe, load_jsonl


DEFAULT_EXPORT_ROOT = Path("dist/huggingface")


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


def _write_dataset_card(path: Path, catalog: BenchmarkCatalog, results_path: Path | None) -> None:
    result_note = f"`results/latest.jsonl` imported from `{results_path}`" if results_path else "No result file bundled yet."
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

This dataset repository contains benchmark metadata and published result
artifacts for `modern-embedding-bench`.

## Contents

- `models.jsonl`: registered model specs
- `tasks.jsonl`: registered task specs and primary metrics
- `runs/`: run manifests used to produce results
- `results/latest.jsonl`: canonical v2 result records
- `leaderboards/latest.csv`: flat leaderboard table derived from result records
- `benchmark_data/`: optional benchmark input data exported from the local repo

## Scope

This benchmark focuses on retrieval scenarios that broad public leaderboards do
not fully cover: MRL compression, cross-lingual retrieval, long-document needle
retrieval, text-image hard negatives, and agent-oriented retrieval tracks.

## Current Registry

- Models: {len(catalog.models)}
- Tasks: {len(catalog.tasks)}
- Results: {result_note}

## Usage

The companion Space can read `leaderboards/latest.csv` and render task-specific
views. The source runner and schemas live in the GitHub repository.
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

    _write_dataset_card(output / "README.md", catalog, Path(results_path) if results_path else None)
    _write_jsonl(output / "models.jsonl", [asdict(model) for model in catalog.models.values()])
    _write_jsonl(output / "tasks.jsonl", [asdict(task) for task in catalog.tasks.values()])

    run_dir = catalog.root / "runs"
    if run_dir.exists():
        for run_file in sorted(run_dir.glob("*.yaml")):
            _copy_if_exists(run_file, output / "runs" / run_file.name)

    if results_path:
        result_src = Path(results_path)
        _copy_if_exists(result_src, output / "results" / "latest.jsonl")
        _write_jsonl(output / "results" / "latest-successful.jsonl", [r for r in load_jsonl(result_src) if not r.get("error")])

    if leaderboard_path:
        _copy_if_exists(Path(leaderboard_path), output / "leaderboards" / "latest.csv")
    elif results_path:
        build_from_file(results_path, output / "leaderboards" / "latest.csv", benchmark_root=benchmark_root)

    if include_data:
        _copy_benchmark_data(Path("data"), output / "benchmark_data", include_images=include_images)
    else:
        (output / "benchmark_data").mkdir(parents=True, exist_ok=True)
        (output / "benchmark_data" / "README.md").write_text(
            "Benchmark input data was not exported. Re-run with --include-data to bundle JSONL data.\n",
            encoding="utf-8",
        )

    files = [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()]
    _write_export_manifest(
        output / "export_manifest.yaml",
        kind="hf_dataset",
        files=files,
        metadata={
            "models": len(catalog.models),
            "tasks": len(catalog.tasks),
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
        _copy_if_exists(Path(bundled_leaderboard), output / "leaderboard.csv")
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
        writer.writerow(["task_id", "task", "model_id", "model", "provider", "primary_metric", "score", "run_id", "duration_s"])


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
    return f'''from __future__ import annotations

import csv
import os
from pathlib import Path

import gradio as gr
import pandas as pd

DEFAULT_DATASET_REPO_ID = {default_repo!r}
LEADERBOARD_FILE = "leaderboards/latest.csv"


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
            print(f"Could not load leaderboard from {{dataset_repo_id}}: {{exc}}. Falling back to bundled data.")
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
TASKS = sorted({{row.get("task_id", "") for row in ROWS if row.get("task_id")}})
PROVIDERS = ["All"] + sorted({{row.get("provider", "") for row in ROWS if row.get("provider")}})


def render_table(task_id, provider):
    filtered = [row.copy() for row in ROWS if row.get("task_id") == task_id]
    if provider != "All":
        filtered = [row for row in filtered if row.get("provider") == provider]
    filtered.sort(
        key=lambda row: as_float(row.get("score")) if as_float(row.get("score")) is not None else float("-inf"),
        reverse=True,
    )
    for index, row in enumerate(filtered, start=1):
        row["rank"] = index
        score = as_float(row.get("score"))
        row["score"] = round(score, 6) if score is not None else row.get("score")
    columns = ["rank", "model", "provider", "score", "primary_metric", "run_id", "duration_s"]
    return pd.DataFrame(filtered, columns=columns)


def main():
    default_task = TASKS[0] if TASKS else None
    with gr.Blocks(title="Modern Embedding Bench") as demo:
        gr.Markdown("# Modern Embedding Bench")
        gr.Markdown("Task-specific embedding benchmark views. Avoid treating these as a single global score.")

        with gr.Row():
            task = gr.Dropdown(TASKS, value=default_task, label="Task")
            provider = gr.Dropdown(PROVIDERS, value="All", label="Provider")
        table = gr.Dataframe(value=render_table(default_task, "All") if default_task else pd.DataFrame(), label="Leaderboard")
        task.change(render_table, inputs=[task, provider], outputs=table)
        provider.change(render_table, inputs=[task, provider], outputs=table)
    demo.launch()


if __name__ == "__main__":
    main()
'''
