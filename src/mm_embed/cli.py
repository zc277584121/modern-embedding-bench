"""CLI entry point for mm-embedding-bench."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from mm_embed.providers import get_provider
from mm_embed.tasks import get_task

console = Console()


@click.group()
@click.version_option()
def main():
    """Multimodal Embedding Model Benchmark & Evaluation Framework."""
    pass


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option("--provider", "-p", multiple=True, help="Provider(s) to evaluate")
@click.option("--task", "-t", multiple=True, help="Task(s) to run")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output JSON file")
def run(config: str | None, provider: tuple[str, ...], task: tuple[str, ...], output: str | None):
    """Run evaluation tasks against embedding providers."""
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        providers_cfg = cfg.get("providers", [])
        tasks_cfg = cfg.get("tasks", [])
    else:
        providers_cfg = [{"name": p} for p in provider]
        tasks_cfg = [{"name": t} for t in task]

    if not providers_cfg:
        console.print("[red]No providers specified. Use --provider or --config.[/red]")
        sys.exit(1)
    if not tasks_cfg:
        console.print("[red]No tasks specified. Use --task or --config.[/red]")
        sys.exit(1)

    all_results = []

    for p_cfg in providers_cfg:
        p_name = p_cfg if isinstance(p_cfg, str) else p_cfg["name"]
        p_kwargs = p_cfg if isinstance(p_cfg, dict) else {}
        p_kwargs.pop("name", None)

        console.print(f"\n[bold blue]Provider: {p_name}[/bold blue]")

        try:
            prov = get_provider(p_name, **p_kwargs)
        except (KeyError, ImportError) as e:
            console.print(f"  [red]Failed to load provider: {e}[/red]")
            continue

        # Health check
        console.print(f"  Checking connectivity...", end=" ")
        if prov.health_check():
            console.print("[green]OK[/green]")
        else:
            console.print("[yellow]FAILED (continuing anyway)[/yellow]")

        for t_cfg in tasks_cfg:
            t_name = t_cfg if isinstance(t_cfg, str) else t_cfg["name"]
            t_kwargs = t_cfg if isinstance(t_cfg, dict) else {}
            t_kwargs.pop("name", None)

            console.print(f"\n  [bold]Task: {t_name}[/bold]")

            try:
                task_obj = get_task(t_name, **t_kwargs)
            except KeyError as e:
                console.print(f"    [red]{e}[/red]")
                continue

            result = task_obj.run(prov)
            all_results.append(result)

            if result.error:
                console.print(f"    [red]ERROR: {result.error}[/red]")
            else:
                for k, v in result.metrics.items():
                    console.print(f"    {k}: {v:.4f}")

    # Summary table
    if all_results:
        console.print("\n")
        _print_summary_table(all_results)

    # Save results
    if output and all_results:
        output_data = []
        for r in all_results:
            output_data.append({
                "task": r.task_name,
                "provider": r.provider_name,
                "model": r.model_name,
                "metrics": r.metrics,
                "error": r.error,
                "details": _serialize_details(r.details),
            })
        Path(output).write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
        console.print(f"\n[green]Results saved to {output}[/green]")


@main.command()
def list_providers():
    """List all available providers."""
    from mm_embed.providers import PROVIDER_REGISTRY

    table = Table(title="Available Providers")
    table.add_column("Name", style="cyan")
    table.add_column("Module", style="dim")
    table.add_column("Class", style="green")

    for name, (module, cls) in sorted(PROVIDER_REGISTRY.items()):
        table.add_row(name, module, cls)

    console.print(table)


@main.command()
def list_tasks():
    """List all available evaluation tasks."""
    from mm_embed.tasks import TASK_REGISTRY

    table = Table(title="Available Tasks")
    table.add_column("Name", style="cyan")
    table.add_column("Module", style="dim")
    table.add_column("Class", style="green")

    for name, (module, cls) in sorted(TASK_REGISTRY.items()):
        table.add_row(name, module, cls)

    console.print(table)


@main.command()
@click.argument("provider_name")
def check(provider_name: str):
    """Check if a provider is properly configured and accessible."""
    try:
        prov = get_provider(provider_name)
        console.print(f"Provider: [cyan]{prov.name}[/cyan]")
        console.print(f"Model: [cyan]{getattr(prov, 'model', 'N/A')}[/cyan]")
        console.print(f"Modalities: {', '.join(m.value for m in prov.supported_modalities)}")
        console.print(f"Max text length: {prov.max_text_length}")
        console.print(f"MRL support: {'Yes' if prov.supports_mrl else 'No'}")
        console.print(f"API key: {'Set' if prov.api_key else '[red]NOT SET[/red]'}")

        if prov.api_key:
            console.print("\nRunning health check...", end=" ")
            if prov.health_check():
                console.print("[green]PASSED[/green]")
            else:
                console.print("[red]FAILED[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.group()
def benchmark():
    """Benchmark v2 registry, manifest, and leaderboard commands."""
    pass


@benchmark.command("models")
@click.option("--root", type=click.Path(exists=True), default=None, help="Benchmark registry root")
def benchmark_models(root: str | None):
    """List registered benchmark model specs."""
    from mm_embed.benchmark.registry import load_catalog

    catalog = load_catalog(root)
    table = Table(title="Benchmark Models")
    table.add_column("ID", style="cyan")
    table.add_column("Provider", style="blue")
    table.add_column("Display Name", style="green")
    table.add_column("Access")
    table.add_column("Status")
    table.add_column("Tags", style="dim")

    for model in sorted(catalog.models.values(), key=lambda item: (item.priority, item.id)):
        table.add_row(
            model.id,
            model.provider,
            model.display_name,
            model.access,
            model.status,
            ", ".join(model.tags),
        )
    console.print(table)


@benchmark.command("tasks")
@click.option("--root", type=click.Path(exists=True), default=None, help="Benchmark registry root")
def benchmark_tasks(root: str | None):
    """List registered benchmark task specs."""
    from mm_embed.benchmark.registry import load_catalog

    catalog = load_catalog(root)
    table = Table(title="Benchmark Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Task", style="blue")
    table.add_column("Primary Metric", style="green")
    table.add_column("Dataset")
    table.add_column("Tags", style="dim")

    for task in sorted(catalog.tasks.values(), key=lambda item: item.id):
        table.add_row(
            task.id,
            task.task,
            task.primary_metric or "",
            task.dataset_version,
            ", ".join(task.tags),
        )
    console.print(table)


@benchmark.command("run")
@click.option("--manifest", "-m", "manifest_path", type=click.Path(exists=True), required=True)
@click.option("--root", type=click.Path(exists=True), default=None, help="Benchmark registry root")
@click.option("--output", "-o", type=click.Path(), default="results/benchmark-v2.jsonl")
@click.option("--limit", type=int, default=None, help="Stop after N model-task combinations")
@click.option("--overwrite", is_flag=True, help="Remove the output file before running.")
def benchmark_run(manifest_path: str, root: str | None, output: str, limit: int | None, overwrite: bool):
    """Run a v2 benchmark manifest and append JSONL results."""
    from mm_embed.benchmark.runner import BenchmarkRunner

    runner, manifest = BenchmarkRunner.from_manifest_path(
        run_manifest=manifest_path,
        benchmark_root=root,
        output=output,
        limit=limit,
        overwrite=overwrite,
    )
    records = runner.run_manifest(manifest)
    console.print(f"[green]Wrote {len(records)} result record(s) to {output}[/green]")


@benchmark.command("leaderboard")
@click.option("--results", "-r", "results_path", type=click.Path(exists=True), required=True)
@click.option("--root", type=click.Path(exists=True), default=None, help="Benchmark registry root")
@click.option("--output", "-o", type=click.Path(), default="results/leaderboard.csv")
def benchmark_leaderboard(results_path: str, root: str | None, output: str):
    """Build a flat CSV leaderboard from v2 result JSONL."""
    from mm_embed.benchmark.leaderboard import build_from_file

    rows = build_from_file(results_path, output, benchmark_root=root)
    console.print(f"[green]Wrote {len(rows)} leaderboard row(s) to {output}[/green]")


def _print_summary_table(results):
    """Print a summary table of all results."""
    table = Table(title="Evaluation Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Provider", style="blue")
    table.add_column("Status", style="bold")
    table.add_column("Key Metrics")

    for r in results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        if r.passed:
            # Show top 3 metrics
            top_metrics = list(r.metrics.items())[:3]
            metrics_str = " | ".join(f"{k}={v:.3f}" for k, v in top_metrics)
        else:
            metrics_str = r.error or "Unknown error"

        table.add_row(r.task_name, f"{r.provider_name}/{r.model_name}", status, metrics_str)

    console.print(table)


def _serialize_details(details: dict[str, Any]) -> dict[str, Any]:
    """Make details JSON-serializable."""
    import numpy as np

    result = {}
    for k, v in details.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, dict):
            result[k] = _serialize_details(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        else:
            result[k] = v
    return result


if __name__ == "__main__":
    main()
