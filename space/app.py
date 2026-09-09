from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

import gradio as gr

DATA_PATH = Path(__file__).parent / "data" / "results.json"
PAYLOAD = json.loads(DATA_PATH.read_text(encoding="utf-8"))
TASKS = {task["id"]: task for task in PAYLOAD["tasks"]}
SOLUTIONS = {solution["id"]: solution for solution in PAYLOAD["solutions"]}
RESULTS = PAYLOAD["results"]


CSS = """
.gradio-container { max-width: 1240px !important; }
.hero {
  padding: 32px;
  border-radius: 24px;
  background: linear-gradient(130deg, #111827 0%, #312e81 52%, #155e75 100%);
  color: white;
  margin-bottom: 18px;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
}
.eyebrow { color: #a5f3fc; font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
.hero h1 { font-size: 42px; line-height: 1.05; margin: 10px 0 10px; }
.hero p { color: #dbeafe; max-width: 760px; font-size: 16px; margin: 0; }
.notice {
  border: 1px solid #fde68a;
  background: #fffbeb;
  color: #92400e;
  border-radius: 14px;
  padding: 12px 16px;
  margin: 12px 0 18px;
}
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 12px 0 20px; }
.stat { border: 1px solid #e2e8f0; border-radius: 16px; padding: 16px; background: white; }
.stat strong { display: block; font-size: 25px; color: #0f172a; }
.stat span { color: #64748b; font-size: 13px; }
.context { border-left: 4px solid #6366f1; padding: 8px 14px; margin: 8px 0 14px; }
.context h3 { margin: 0 0 4px; }
.context p { margin: 0; color: #64748b; }
.table-wrap { overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 16px; background: white; }
.bench-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.bench-table th { background: #f8fafc; color: #475569; text-align: left; padding: 12px 14px; border-bottom: 1px solid #e2e8f0; white-space: nowrap; }
.bench-table td { padding: 13px 14px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
.bench-table tr:last-child td { border-bottom: 0; }
.bench-table tr:hover td { background: #fafafa; }
.rank { color: #64748b; font-variant-numeric: tabular-nums; }
.score { color: #0f172a; font-weight: 700; font-variant-numeric: tabular-nums; }
.primary { color: #4338ca; }
.muted { color: #94a3b8; }
.pill { display: inline-block; border-radius: 999px; padding: 3px 8px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
.pill-demo { background: #fef3c7; color: #92400e; }
.source { color: #4f46e5 !important; font-weight: 600; text-decoration: none !important; }
.source:hover { text-decoration: underline !important; }
.metric-note { color: #64748b; font-size: 13px; margin-top: 10px; }
.footer-note { color: #64748b; font-size: 13px; text-align: center; padding: 24px 0 8px; }
@media (max-width: 760px) {
  .hero { padding: 24px; }
  .hero h1 { font-size: 32px; }
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""


def esc(value: object) -> str:
    return html.escape(str(value))


def score(value: float) -> str:
    return f"{value:.3f}"


def source_link(record: dict[str, object]) -> str:
    commit = str(record["source_commit"])
    return f'<a class="source" href="{esc(record["source_url"])}" target="_blank">{esc(commit[:7])} ↗</a>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{esc(header)}</th>" for header in headers)
    row_html = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        '<div class="table-wrap"><table class="bench-table">'
        f"<thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody>"
        "</table></div>"
    )


def task_dataset_options() -> list[tuple[str, str]]:
    options = []
    seen = set()
    for record in RESULTS:
        key = (record["task_id"], record["dataset_id"], record["dataset_version"])
        if key in seen:
            continue
        seen.add(key)
        task = TASKS[str(record["task_id"])]
        label = f"{task['title']} · {record['dataset_id']} ({record['dataset_version']})"
        value = "|".join(map(str, key))
        options.append((label, value))
    return options


def render_task(selection: str) -> tuple[str, str]:
    task_id, dataset_id, dataset_version = selection.split("|", maxsplit=2)
    task = TASKS[task_id]
    selected = [
        record
        for record in RESULTS
        if record["task_id"] == task_id
        and record["dataset_id"] == dataset_id
        and record["dataset_version"] == dataset_version
    ]
    metric_order = []
    for record in selected:
        if record["metric_id"] not in metric_order:
            metric_order.append(record["metric_id"])
    primary_metric = task["primary_metric"]
    metric_order.sort(key=lambda metric_id: metric_id != primary_metric)

    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for record in selected:
        grouped[str(record["solution_id"])][str(record["metric_id"])] = record
    ordered_solutions = sorted(
        grouped,
        key=lambda solution_id: -float(grouped[solution_id][primary_metric]["value"]),
    )
    rows = []
    for rank, solution_id in enumerate(ordered_solutions, start=1):
        solution = SOLUTIONS[solution_id]
        records = grouped[solution_id]
        metric_cells = []
        for metric_id in metric_order:
            record = records.get(metric_id)
            if record is None:
                metric_cells.append('<span class="muted">—</span>')
                continue
            css_class = "score primary" if metric_id == primary_metric else "score"
            metric_cells.append(f'<span class="{css_class}">{score(float(record["value"]))}</span>')
        first_record = next(iter(records.values()))
        rows.append(
            [
                f'<span class="rank">#{rank}</span>',
                f'<strong>{esc(solution["title"])}</strong><br><span class="muted">{esc(solution["description"])}</span>',
                *metric_cells,
                '<span class="pill pill-demo">Demo</span>',
                source_link(first_record),
            ]
        )
    labels = {str(record["metric_id"]): str(record["metric_label"]) for record in selected}
    summary = (
        '<div class="context">'
        f"<h3>{esc(task['title'])}</h3>"
        f"<p>{esc(task['description'])} Dataset: {esc(dataset_id)} · {esc(dataset_version)}</p>"
        "</div>"
    )
    headers = ["Rank", "Solution", *[labels[item] for item in metric_order], "Status", "Source"]
    return summary, table(headers, rows)


def primary_records() -> list[dict[str, object]]:
    return [record for record in RESULTS if record["primary"]]


def render_solution(solution_id: str) -> tuple[str, str]:
    solution = SOLUTIONS[solution_id]
    records = [record for record in primary_records() if record["solution_id"] == solution_id]
    rows = []
    for record in sorted(records, key=lambda item: str(item["task_id"])):
        task_id = str(record["task_id"])
        peers = [
            item
            for item in primary_records()
            if item["task_id"] == task_id
            and item["dataset_id"] == record["dataset_id"]
            and item["dataset_version"] == record["dataset_version"]
        ]
        ordered = sorted(peers, key=lambda item: -float(item["value"]))
        rank = next(index for index, item in enumerate(ordered, start=1) if item["solution_id"] == solution_id)
        rows.append(
            [
                f"<strong>{esc(TASKS[task_id]['title'])}</strong>",
                esc(record["metric_label"]),
                f'<span class="score primary">{score(float(record["value"]))}</span>',
                f'<span class="rank">#{rank} / {len(peers)}</span>',
                f'{esc(record["dataset_id"])}<br><span class="muted">{esc(record["dataset_version"])}</span>',
                source_link(record),
            ]
        )
    summary = f'<div class="context"><h3>{esc(solution["title"])}</h3><p>{esc(solution["description"])}</p></div>'
    return summary, table(["Task", "Primary metric", "Score", "Rank", "Dataset", "Source"], rows)


def render_matrix() -> str:
    task_ids = sorted(TASKS)
    primary = primary_records()
    rows = []
    for solution_id, solution in SOLUTIONS.items():
        cells = [f"<strong>{esc(solution['title'])}</strong>"]
        for task_id in task_ids:
            record = next(
                (item for item in primary if item["solution_id"] == solution_id and item["task_id"] == task_id),
                None,
            )
            if record is None:
                cells.append('<span class="muted">Not evaluated</span>')
                continue
            peers = [item for item in primary if item["task_id"] == task_id]
            ordered = sorted(peers, key=lambda item: -float(item["value"]))
            rank = next(index for index, item in enumerate(ordered, start=1) if item is record)
            cells.append(
                f'<span class="score">{score(float(record["value"]))}</span> '
                f'<span class="rank">#{rank}</span><br>'
                f'<span class="muted">{esc(record["metric_label"])}</span>'
            )
        rows.append(cells)
    headers = ["Solution", *[str(TASKS[task_id]["title"]) for task_id in task_ids]]
    return table(headers, rows) + (
        '<div class="metric-note">Each task keeps its own primary metric. '
        "Scores are not averaged into an overall score.</div>"
    )


benchmark = PAYLOAD["benchmark"]
task_options = task_dataset_options()
initial_task = task_options[0][1]
initial_solution = next(iter(SOLUTIONS))
task_summary, task_table = render_task(initial_task)
solution_summary, solution_table = render_solution(initial_solution)


with gr.Blocks(
    title="Modern IR Bench",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.HTML(
        '<section class="hero">'
        '<div class="eyebrow">Code-native benchmark</div>'
        "<h1>Modern IR Bench</h1>"
        "<p>Compare models, retrieval algorithms, RAG pipelines, and agent systems "
        "across modern information retrieval tasks.</p>"
        "</section>"
    )
    gr.HTML(f'<div class="notice"><strong>Framework preview.</strong> {esc(benchmark["notice"])}</div>')
    gr.HTML(
        '<div class="stats">'
        f'<div class="stat"><strong>{len(TASKS)}</strong><span>Modern IR tasks</span></div>'
        f'<div class="stat"><strong>{len(SOLUTIONS)}</strong><span>Solutions</span></div>'
        f'<div class="stat"><strong>{len(RESULTS)}</strong><span>Measurements</span></div>'
        f'<div class="stat"><strong>{esc(benchmark["release"])}</strong><span>Active release</span></div>'
        "</div>"
    )

    with gr.Tabs():
        with gr.Tab("Task leaderboards"):
            task_selector = gr.Dropdown(
                choices=task_options,
                value=initial_task,
                label="Task and dataset release",
            )
            task_context = gr.HTML(task_summary)
            task_results = gr.HTML(task_table)
            task_selector.change(render_task, task_selector, [task_context, task_results])

        with gr.Tab("Solution profiles"):
            solution_selector = gr.Dropdown(
                choices=[(solution["title"], solution_id) for solution_id, solution in SOLUTIONS.items()],
                value=initial_solution,
                label="Solution",
            )
            solution_context = gr.HTML(solution_summary)
            solution_results = gr.HTML(solution_table)
            solution_selector.change(
                render_solution,
                solution_selector,
                [solution_context, solution_results],
            )

        with gr.Tab("Cross-task matrix"):
            gr.HTML(
                '<div class="context"><h3>Solution × Task</h3>'
                "<p>A compact view of each solution’s primary score and rank per task.</p></div>"
            )
            gr.HTML(render_matrix())

        with gr.Tab("Method"):
            gr.Markdown(
                """
### One result, one source

Every measurement points to an immutable Git commit and the exact executable Python module that produced it. There is no YAML registry and no public submission queue.

```bash
git checkout <source_commit>
uv sync
uv run python -m benchmarks.mock_showcase
```

### Read-only publication

New Tasks, Datasets, Solutions, and experiments enter through reviewed GitHub pull requests. The official runner produces the result artifacts consumed by this Space.

### Result dimensions

The underlying result model is `Solution × Task × Dataset × Metric × Version`. This release intentionally has no overall score because metrics from unrelated tasks are not directly interchangeable.
"""
            )

    gr.HTML('<div class="footer-note">Modern IR Bench · Flexible tasks, inspectable code, immutable results.</div>')


if __name__ == "__main__":
    demo.launch()
