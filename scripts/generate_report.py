"""Generate evaluation report with Markdown tables and PNG charts.

Usage:
    uv run python scripts/generate_report.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILES = [
    ROOT / "results" / "eval_complete_20260312.json",
    ROOT / "results" / "eval_clip_20260312.json",
    ROOT / "results" / "eval_api_new_20260312.json",
    ROOT / "results" / "eval_local_new_20260312.json",
]
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
REPORT_MD = REPORT_DIR / "evaluation_report.md"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data — merge all result files
# ---------------------------------------------------------------------------
results: list[dict] = []
for rf in RESULTS_FILES:
    if rf.exists():
        results.extend(json.loads(rf.read_text()))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Model display names & metadata
# Color families:
#   Alibaba/通义: blues          (#1a4e8a dark, #4a90d9 light)
#   OpenAI: green                (#2ca02c)
#   New APIs (Jina/Voyage/Cohere/Doubao): gold/yellow (#daa520, #e6a817, #c9951a, #a07810)
#   Ollama local (warm): reds/oranges (#d62728, #e6550d, #ff7f0e, #8c564b, #e377c2)
#   Local GPU full-precision: purples (#7b2d8e, #9b59b6, #b07cc6)
#   CLIP: cyan/teal              (#17becf, #0e8a7d)
#   SigLIP2: olive green         (#6b8e23)
MODEL_META: dict[str, dict] = {
    # --- Alibaba/通义 API models (blue family) ---
    "text-embedding-v3":               {"short": "Alibaba text-emb-v3",     "type": "API",   "provider": "dashscope",  "color": "#1a4e8a", "marker": "o"},
    "multimodal-embedding-v1":         {"short": "Alibaba MM-v1",           "type": "API",   "provider": "dashscope",  "color": "#4a90d9", "marker": "s"},
    # --- OpenAI (green) ---
    "text-embedding-3-large":          {"short": "OpenAI 3-large",     "type": "API",   "provider": "openai",     "color": "#2ca02c", "marker": "D"},
    # --- New API models (gold/yellow family) ---
    "embed-v4.0":                      {"short": "Cohere v4",          "type": "API",   "provider": "cohere",     "color": "#daa520", "marker": "H"},
    "jina-embeddings-v4":              {"short": "Jina v4 (API)",      "type": "API",   "provider": "jina",       "color": "#e6a817", "marker": "8"},
    "voyage-multimodal-3.5":           {"short": "Voyage MM-3.5",      "type": "API",   "provider": "voyage",     "color": "#c9951a", "marker": "d"},
    "doubao-embedding-text-240715":    {"short": "Doubao (ARK)",       "type": "API",   "provider": "ark",        "color": "#a07810", "marker": "*"},
    # --- Ollama local models (red/warm family) ---
    "mxbai-embed-large":               {"short": "MxbAI Large",        "type": "Local",  "provider": "ollama",     "color": "#d62728", "marker": "v"},
    "nomic-embed-text":                {"short": "Nomic Embed",        "type": "Local",  "provider": "ollama",     "color": "#e6550d", "marker": "^"},
    "bge-m3":                          {"short": "BGE-M3 (Ollama)",    "type": "Local",  "provider": "ollama",     "color": "#ff7f0e", "marker": "<"},
    "snowflake-arctic-embed:335m":     {"short": "Snowflake 335M",     "type": "Local",  "provider": "ollama",     "color": "#8c564b", "marker": ">"},
    "dengcao/Qwen3-Embedding-8B:Q5_K_M": {"short": "Qwen3-8B (Q5)",  "type": "Local",  "provider": "ollama",     "color": "#e377c2", "marker": "p"},
    # --- Local GPU full-precision models (purple family) ---
    "BAAI/bge-m3":                     {"short": "BGE-M3 (GPU FP)",    "type": "Local",  "provider": "sentence_transformers", "color": "#7b2d8e", "marker": "h"},
    "jinaai/jina-embeddings-v3":       {"short": "Jina v3 (Local)",    "type": "Local",  "provider": "sentence_transformers", "color": "#9b59b6", "marker": "1"},
    "Qwen/Qwen3-VL-Embedding-2B":
                                       {"short": "Qwen3-VL-2B",       "type": "Local",  "provider": "transformers", "color": "#b07cc6", "marker": "2"},
    # --- CLIP models (cyan/teal family) ---
    "clip-ViT-B-32":                   {"short": "CLIP ViT-B-32",      "type": "Local",  "provider": "sentence_transformers", "color": "#17becf", "marker": "X"},
    "clip-ViT-L-14":                   {"short": "CLIP ViT-L-14",      "type": "Local",  "provider": "sentence_transformers", "color": "#0e8a7d", "marker": "P"},
    # --- SigLIP2 (olive green) ---
    "google/siglip2-so400m-patch14-384":
                                       {"short": "SigLIP2-400M",       "type": "Local",  "provider": "transformers", "color": "#6b8e23", "marker": "3"},
}

def short_name(model: str) -> str:
    return MODEL_META.get(model, {}).get("short", model)

def model_color(model: str) -> str:
    return MODEL_META.get(model, {}).get("color", "#333333")

def model_marker(model: str) -> str:
    return MODEL_META.get(model, {}).get("marker", "o")

def model_type(model: str) -> str:
    return MODEL_META.get(model, {}).get("type", "?")

def by_task(task_name: str) -> list[dict]:
    return [r for r in results if r["task"] == task_name and not r.get("error")]


# ---------------------------------------------------------------------------
# Chart style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
})


# ===================================================================
# CHART 1: MRL Line Chart — Spearman vs Dimension
# ===================================================================
def chart_mrl_line():
    fig, ax = plt.subplots(figsize=(14, 7))
    mrl = by_task("mrl_stress")

    for r in mrl:
        m = r["metrics"]
        dims = []
        spear = []
        for k, v in sorted(m.items()):
            if k.startswith("spearman_dim_"):
                d = int(k.replace("spearman_dim_", ""))
                dims.append(d)
                spear.append(v)
        # Sort by dimension
        pairs = sorted(zip(dims, spear))
        dims, spear = zip(*pairs)

        label = short_name(r["model"])
        ax.plot(dims, spear, marker=model_marker(r["model"]),
                color=model_color(r["model"]), label=label,
                linewidth=2, markersize=7)

    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048, 4096])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("MRL Dimension Reduction: Spearman Correlation by Embedding Dimension")
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)
    ax.set_ylim(0.55, 0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mrl_line_chart.png")
    plt.close(fig)
    print("  -> mrl_line_chart.png")


# ===================================================================
# CHART 2: MRL Bar Chart — Models ranked by full-dim Spearman
# ===================================================================
def chart_mrl_bar():
    fig, ax = plt.subplots(figsize=(12, 6))
    mrl = by_task("mrl_stress")

    entries = []
    for r in mrl:
        d = r.get("details", {})
        full_dim = d.get("full_dim", "?")
        full_spear = d.get("full_spearman", 0)
        entries.append((short_name(r["model"]), full_spear, full_dim, r["model"]))

    entries.sort(key=lambda x: x[1], reverse=True)
    names = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    colors = [model_color(e[3]) for e in entries]

    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ (full dimension)")
    ax.set_title("MRL Stress: Models Ranked by Full-Dimension Spearman ρ")
    ax.set_xlim(0.55, 0.95)

    for i, (bar, e) in enumerate(zip(bars, entries)):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{e[1]:.3f} (d={e[2]})", va="center", fontsize=8)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mrl_bar_chart.png")
    plt.close(fig)
    print("  -> mrl_bar_chart.png")


# ===================================================================
# CHART 3: Cross-Modal Grouped Bar
# ===================================================================
def chart_cross_modal():
    cm = by_task("cross_modal_retrieval")
    n_models = len(cm)

    metrics_keys = ["t2i_recall@1", "t2i_recall@5", "i2t_recall@1", "i2t_recall@5",
                    "i2t_hard_recall@1", "modality_gap"]
    metric_labels = ["T→I R@1", "T→I R@5", "I→T R@1", "I→T R@5", "I→T Hard R@1", "Gap"]

    fig, ax = plt.subplots(figsize=(14, 6))
    n_metrics = len(metrics_keys)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_models, 1)

    for i, r in enumerate(cm):
        vals = []
        for mk in metrics_keys:
            v = r["metrics"].get(mk, 0)
            # Normalize gap to 0-1 scale for visibility (divide by 2)
            if mk == "modality_gap":
                v = v / 2.0
            vals.append(v)
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=short_name(r["model"]),
                      color=model_color(r["model"]), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (Gap scaled ÷2)")
    ax.set_title("Cross-Modal Retrieval: Metric Comparison")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_modal_bar.png")
    plt.close(fig)
    print("  -> cross_modal_bar.png")


# ===================================================================
# CHART 4: Needle-in-Haystack Heatmap Grid
# ===================================================================
def chart_needle_heatmap():
    needle = by_task("needle_in_haystack")
    n = len(needle)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    # Use the union of all lengths and positions across all models
    all_lengths: set[str] = set()
    all_positions: set[str] = set()
    for r in needle:
        heatmap_data = r.get("details", {}).get("heatmap", {})
        for l_key, pos_dict in heatmap_data.items():
            all_lengths.add(l_key)
            for p_key in pos_dict:
                all_positions.add(p_key)
    # Sort lengths numerically, positions by percentage value
    lengths = sorted(all_lengths, key=lambda x: int(x))
    pos_order = {"0%": 0, "25%": 1, "50%": 2, "75%": 3, "100%": 4}
    positions = sorted(all_positions, key=lambda x: pos_order.get(x, 999))

    if not lengths:
        lengths = ["1000", "4000", "8000"]
    if not positions:
        positions = ["0%", "25%", "50%", "75%", "100%"]

    for idx, r in enumerate(needle):
        row_i, col_i = divmod(idx, cols)
        ax = axes[row_i][col_i]
        heatmap_data = r.get("details", {}).get("heatmap", {})

        matrix = np.full((len(positions), len(lengths)), np.nan)
        if heatmap_data:
            for li, l in enumerate(lengths):
                for pi, p in enumerate(positions):
                    matrix[pi][li] = heatmap_data.get(l, {}).get(p, np.nan)
        else:
            # Build from aggregate metrics
            for li, l in enumerate(lengths):
                key = f"accuracy_len_{l}"
                if key in r["metrics"]:
                    for pi in range(len(positions)):
                        matrix[pi][li] = r["metrics"][key]

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([f"{int(l)//1000}K" for l in lengths], fontsize=8)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(positions, fontsize=8)
        ax.set_xlabel("Doc Length", fontsize=8)
        ax.set_ylabel("Needle Position", fontsize=8)

        acc = r["metrics"].get("overall_accuracy", 0)
        ax.set_title(f"{short_name(r['model'])} (acc={acc:.2f})", fontsize=9)

        # Annotate cells
        for pi in range(len(positions)):
            for li in range(len(lengths)):
                val = matrix[pi][li]
                if np.isnan(val):
                    ax.text(li, pi, "—", ha="center", va="center", fontsize=7, color="gray")
                else:
                    color = "white" if val < 0.5 else "black"
                    ax.text(li, pi, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")

    # Hide unused axes
    for idx in range(n, rows * cols):
        row_i, col_i = divmod(idx, cols)
        axes[row_i][col_i].set_visible(False)

    fig.suptitle("Needle-in-Haystack: Accuracy by Document Length and Needle Position",
                 fontsize=14, y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Accuracy", pad=0.04)
    fig.savefig(FIG_DIR / "needle_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> needle_heatmap.png")


# ===================================================================
# CHART 5: Needle Bar Chart — Overall accuracy ranked
# ===================================================================
def chart_needle_bar():
    fig, ax = plt.subplots(figsize=(12, 7))
    needle = by_task("needle_in_haystack")

    entries = []
    for r in needle:
        acc = r["metrics"].get("overall_accuracy", 0)
        deg = r["metrics"].get("degradation_rate", 0)
        entries.append((short_name(r["model"]), acc, deg, r["model"]))

    entries.sort(key=lambda x: x[1], reverse=True)
    names = [e[0] for e in entries]
    accs = [e[1] for e in entries]
    degs = [e[2] for e in entries]
    colors = [model_color(e[3]) for e in entries]

    bars = ax.barh(range(len(names)), accs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Overall Accuracy")
    ax.set_title("Needle-in-Haystack: Models Ranked by Overall Accuracy")
    ax.set_xlim(0, 1.15)

    for i, (bar, e) in enumerate(zip(bars, entries)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{e[1]:.3f} (deg={e[2]:.2f})", va="center", fontsize=8)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "needle_bar_chart.png")
    plt.close(fig)
    print("  -> needle_bar_chart.png")


# ===================================================================
# MARKDOWN REPORT
# ===================================================================
def generate_markdown() -> str:
    lines: list[str] = []
    L = lines.append

    # ------------------------------------------------------------------
    # Executive Summary
    # ------------------------------------------------------------------
    L("# Multimodal Embedding Model Evaluation Report\n")
    L(f"**Date**: 2026-03-12  ")
    L(f"**Data source**: `{', '.join(rf.name for rf in RESULTS_FILES if rf.exists())}`\n")

    mrl = by_task("mrl_stress")
    cm = by_task("cross_modal_retrieval")
    needle = by_task("needle_in_haystack")

    n_models = len(set(r["model"] for r in results))
    L("## 1. Executive Summary\n")
    L(f"This report evaluates **{n_models} embedding models** across **3 tasks**:\n")
    L("| Task | Description | # Models Tested |")
    L("|------|-------------|-----------------|")
    L(f"| MRL Stress | Semantic similarity with dimension compression (Spearman ρ on STS-B) | {len(mrl)} |")
    L(f"| Cross-Modal Retrieval | Bidirectional text↔image retrieval (COCO images + hard negatives) | {len(cm)} |")
    L(f"| Needle-in-Haystack | Long-document fact retrieval at varying lengths and positions | {len(needle)} |")
    L("")

    L("**Key Findings:**\n")
    # Best MRL
    best_mrl = max(mrl, key=lambda r: r.get("details", {}).get("full_spearman", 0))
    L(f"- **Best semantic similarity**: {short_name(best_mrl['model'])} "
      f"(ρ={best_mrl['details']['full_spearman']:.3f} at {best_mrl['details']['full_dim']}d)")
    # Best Needle
    best_needle = max(needle, key=lambda r: r["metrics"].get("overall_accuracy", 0))
    perfect_needle = [r for r in needle if r["metrics"].get("overall_accuracy", 0) >= 1.0]
    if perfect_needle:
        perfect_names = ", ".join(short_name(r["model"]) for r in perfect_needle)
        L(f"- **Perfect long-text retrieval** (accuracy=1.000): {perfect_names}")
    else:
        L(f"- **Best long-text retrieval**: {short_name(best_needle['model'])} "
          f"(accuracy={best_needle['metrics']['overall_accuracy']:.3f})")
    # Best Cross-modal
    if cm:
        best_cm = max(cm, key=lambda r: r["metrics"].get("avg_recall@1", 0))
        L(f"- **Best cross-modal**: {short_name(best_cm['model'])} "
          f"(avg R@1={best_cm['metrics']['avg_recall@1']:.3f})")
    L("- **Quantization impact**: Full-precision BGE-M3 (GPU) scores 0.973 on needle vs 0.693 for "
      "Ollama quantized — a **28 percentage point gap** showing quantization severely hurts long-text tasks")
    L("- **API vs Local**: New API models (Cohere v4, Jina v4, Voyage MM-3.5, Doubao) all achieve "
      "perfect needle accuracy (1.000), while all Ollama models collapse beyond 1K characters")
    L("- **Local multimodal breakthroughs**: Qwen3-VL-2B and SigLIP2-400M achieve perfect R@1=1.000 "
      "on cross-modal retrieval (though on smaller test pools — 20 and 50 samples respectively)")
    L("")

    # ------------------------------------------------------------------
    # Task 1: MRL Stress
    # ------------------------------------------------------------------
    L("## 2. Task 1: MRL Stress (Semantic Similarity & Dimension Compression)\n")
    L("Evaluates how well embeddings capture semantic similarity using Spearman rank correlation "
      "on STS-B sentence pairs, and how quality degrades "
      "when truncating to lower dimensions (Matryoshka Representation Learning).\n")

    # Table: all models × all dimensions
    all_dims_set: set[int] = set()
    for r in mrl:
        for k in r["metrics"]:
            if k.startswith("spearman_dim_"):
                all_dims_set.add(int(k.replace("spearman_dim_", "")))
    all_dims = sorted(all_dims_set)

    header = "| Model | Type | " + " | ".join(f"d={d}" for d in all_dims) + " | Full Dim | Samples |"
    sep = "|-------|------|" + "|".join("------" for _ in all_dims) + "|----------|---------|"
    L(header)
    L(sep)
    for r in sorted(mrl, key=lambda x: x.get("details", {}).get("full_spearman", 0), reverse=True):
        m = r["metrics"]
        fd = r.get("details", {}).get("full_dim", "?")
        n_pairs = int(m.get("n_pairs", 0))
        cols = []
        for d in all_dims:
            key = f"spearman_dim_{d}"
            if key in m:
                val = m[key]
                # Bold if this is the best across models for this dim
                best_at_dim = max(rr["metrics"].get(key, -1) for rr in mrl)
                if abs(val - best_at_dim) < 1e-6:
                    cols.append(f"**{val:.3f}**")
                else:
                    cols.append(f"{val:.3f}")
            else:
                cols.append("—")
        row = f"| {short_name(r['model'])} | {model_type(r['model'])} | " + " | ".join(cols) + f" | {fd} | {n_pairs} |"
        L(row)
    L("")
    L("*Note: Local models (Jina v3, Qwen3-VL-2B) were evaluated on 200 STS-B pairs (vs 1379 for others) to save compute time.*\n")

    L("![MRL Dimension Reduction](figures/mrl_line_chart.png)\n")
    L("![MRL Ranking](figures/mrl_bar_chart.png)\n")

    L("### Analysis\n")
    L("- **MxbAI Embed Large** achieves the highest Spearman ρ (0.893) despite being a local "
      "quantized model via Ollama, surpassing all API providers including Jina v4 API (0.887) and DashScope v3 (0.875).")
    L("- **Jina v4 (API)** ranks second (0.887 at 2048d), closely followed by DashScope v3 (0.875 at 1024d).")
    L("- **Voyage MM-3.5** (0.884 at 1024d) performs comparably to DashScope v3, and shows solid "
      "MRL compression (min_viable_dim=32).")
    L("- **DashScope text-embedding-v3** shows the best dimension compression resilience: "
      "only a 0.028 drop from 1024d to 32d, indicating strong Matryoshka training.")
    L("- **Jina v3 (Local)** via sentence-transformers (0.836) performs comparably to OpenAI 3-large (0.836), "
      "with excellent compression resilience (0.026 drop to 32d).")
    L("- **Qwen3-VL-2B** (0.820 at 2048d) is strong for a 2B multimodal model, with good MRL compression "
      "(only 0.022 drop to 32d).")
    L("- **CLIP models** (ViT-B-32: ρ=0.615, ViT-L-14: ρ=0.655) score well below text-specialized "
      "models on pure text similarity, as expected for vision-language models.")
    L("")

    # ------------------------------------------------------------------
    # Task 2: Cross-Modal Retrieval
    # ------------------------------------------------------------------
    L("## 3. Task 2: Cross-Modal Retrieval\n")
    L("Tests bidirectional text↔image retrieval using COCO val2017 images with "
      "GPT-4o-mini generated captions. Hard negative evaluation adds wrong-but-plausible "
      "captions (3 per image) to the text pool.\n")

    L("| Model | Type | T→I R@1 | T→I R@5 | I→T R@1 | I→T R@5 | Hard R@1 | Hard R@5 | Gap | N pairs |")
    L("|-------|------|---------|---------|---------|---------|----------|----------|-----|---------|")
    for r in sorted(cm, key=lambda x: x["metrics"].get("avg_recall@1", 0), reverse=True):
        m = r["metrics"]
        n_pairs = r.get("details", {}).get("n_pairs", "?")
        L(f"| {short_name(r['model'])} | {model_type(r['model'])} "
          f"| {m.get('t2i_recall@1', 0):.3f} | {m.get('t2i_recall@5', 0):.3f} "
          f"| {m.get('i2t_recall@1', 0):.3f} | {m.get('i2t_recall@5', 0):.3f} "
          f"| {m.get('i2t_hard_recall@1', 0):.3f} | {m.get('i2t_hard_recall@5', 0):.3f} "
          f"| {m.get('modality_gap', 0):.2f} | {n_pairs} |")
    L("")

    L("![Cross-Modal Comparison](figures/cross_modal_bar.png)\n")

    L("### Analysis\n")
    L("- **Qwen3-VL-2B, SigLIP2-400M, and Voyage MM-3.5** all achieve perfect R@1=1.000 on standard "
      "retrieval. However, sample sizes differ significantly (20, 50, and 10 pairs respectively), "
      "so direct comparison requires caution.")
    L("- **Voyage MM-3.5 caveat**: Tested on only 10 image-text pairs due to API rate limits (3 RPM). "
      "Its perfect R@1=1.000 is likely inflated by the small pool — the hard-negative score "
      "(i2t_hard_R@1=0.900) is a more reliable signal, and even that had only 30 hard negatives vs 600 for CLIP/DashScope.")
    L("- **Hard-negative discrimination** separates models clearly: DashScope MM-v1 (0.770 on 600 negatives) "
      "> Voyage (0.900 on 30) ≈ Qwen3-VL-2B (0.850 on 60) > SigLIP2 (0.680 on 150) > CLIP-L (0.615 on 600) > CLIP-B (0.570 on 600).")
    L("- **Modality gap** varies widely: Qwen3-VL-2B has the smallest gap (0.28), while SigLIP2 has "
      "a large gap (1.00) yet still achieves perfect R@1 — gap alone doesn't predict performance.")
    L("- **Qwen3-8B (Q5) via Ollama** completely fails at cross-modal retrieval "
      "(R@1=0.005, effectively random). The massive modality gap (1.24) confirms that "
      "Q5 quantization destroys multimodal alignment.")
    L("")

    # ------------------------------------------------------------------
    # Task 3: Needle-in-Haystack
    # ------------------------------------------------------------------
    L("## 4. Task 3: Needle-in-Haystack\n")
    L("Tests whether an embedding model can find a specific fact (\"needle\") embedded within "
      "a long document (\"haystack\") of varying length and position. A correct retrieval means "
      "sim(query, doc_with_needle) > sim(query, doc_without_needle).\n")

    L("| Model | Type | Overall Acc | Deg Rate | 1K | 4K | 8K | Test Cases |")
    L("|-------|------|-------------|----------|-----|-----|-----|------------|")
    for r in sorted(needle, key=lambda x: x["metrics"].get("overall_accuracy", 0), reverse=True):
        m = r["metrics"]
        n_cases = r.get("details", {}).get("n_test_cases", "?")
        L(f"| {short_name(r['model'])} | {model_type(r['model'])} "
          f"| {m.get('overall_accuracy', 0):.3f} | {m.get('degradation_rate', 0):.3f} "
          f"| {m.get('accuracy_len_1000', 0):.2f} | {m.get('accuracy_len_4000', 0):.2f} "
          f"| {m.get('accuracy_len_8000', '—')} | {n_cases} |")
    L("")

    L("![Needle Heatmap](figures/needle_heatmap.png)\n")
    L("![Needle Ranking](figures/needle_bar_chart.png)\n")

    L("### Analysis\n")
    L("- **Seven models achieve perfect accuracy** (1.000): OpenAI 3-large, Cohere v4, Jina v4 (API), "
      "Voyage MM-3.5, Doubao (ARK), Jina v3 (Local), and Qwen3-VL-2B. All API models tested are perfect.")
    L("- **Local models can match APIs**: Jina v3 (local, full-precision) and Qwen3-VL-2B both achieve "
      "1.000 accuracy, though on a smaller test set (60 vs 150 cases, 2 lengths vs 3).")
    L("- **Full-precision GPU changes everything**: BGE-M3 on GPU (sentence-transformers, 0.973) "
      "dramatically outperforms its Ollama quantized counterpart (0.693). "
      "This 28pp gap is the single most important finding for practitioners.")
    L("- **All Ollama models collapse at 4K+**: Every quantized model shows severe degradation "
      "beyond 1K characters (all achieve ~1.0 at 1K but drop to 0.40–0.72 at 4K). "
      "Quantization disproportionately affects long-context understanding.")
    L("- **DashScope v3** is the only API model not achieving perfect accuracy (0.993), "
      "with a slight dip at 8K/75% position (0.9), but still far ahead of any Ollama model.")
    L("")

    # ------------------------------------------------------------------
    # Model Summary Card
    # ------------------------------------------------------------------
    L("## 5. Model Summary Card\n")

    L("| Model | Provider | Type | Dims | Modalities | MRL ρ (full) | Needle Acc | Cross-Modal R@1 | Best At |")
    L("|-------|----------|------|------|------------|-------------|------------|-----------------|---------|")

    all_models = {}
    for r in results:
        if r.get("error"):
            continue
        key = r["model"]
        if key not in all_models:
            all_models[key] = {"mrl": "—", "needle": "—", "cross": "—", "dims": "—"}
        if r["task"] == "mrl_stress":
            fd = r.get("details", {}).get("full_dim", "?")
            fs = r.get("details", {}).get("full_spearman", 0)
            all_models[key]["mrl"] = f"{fs:.3f}"
            all_models[key]["dims"] = str(fd)
        elif r["task"] == "needle_in_haystack":
            all_models[key]["needle"] = f"{r['metrics']['overall_accuracy']:.3f}"
        elif r["task"] == "cross_modal_retrieval":
            all_models[key]["cross"] = f"{r['metrics']['avg_recall@1']:.3f}"

    # Determine best-at labels
    for model, d in all_models.items():
        bests = []
        if d["mrl"] != "—":
            mrl_val = float(d["mrl"])
            all_mrl = [float(v["mrl"]) for v in all_models.values() if v["mrl"] != "—"]
            if mrl_val == max(all_mrl):
                bests.append("MRL")
        if d["needle"] != "—":
            n_val = float(d["needle"])
            all_n = [float(v["needle"]) for v in all_models.values() if v["needle"] != "—"]
            if n_val == max(all_n):
                bests.append("Needle")
        if d["cross"] != "—":
            c_val = float(d["cross"])
            all_c = [float(v["cross"]) for v in all_models.values() if v["cross"] != "—"]
            if c_val == max(all_c):
                bests.append("Cross-Modal")
        d["best"] = ", ".join(bests) if bests else "—"

    modality_map = {
        "text-embedding-v3": "Text",
        "multimodal-embedding-v1": "Text+Image",
        "text-embedding-3-large": "Text",
        "embed-v4.0": "Text",
        "jina-embeddings-v4": "Text",
        "voyage-multimodal-3.5": "Text+Image",
        "doubao-embedding-text-240715": "Text",
        "nomic-embed-text": "Text",
        "mxbai-embed-large": "Text",
        "bge-m3": "Text",
        "snowflake-arctic-embed:335m": "Text",
        "dengcao/Qwen3-Embedding-8B:Q5_K_M": "Text+Image*",
        "BAAI/bge-m3": "Text",
        "clip-ViT-B-32": "Text+Image",
        "clip-ViT-L-14": "Text+Image",
        "jinaai/jina-embeddings-v3": "Text",
        "Qwen/Qwen3-VL-Embedding-2B": "Text+Image",
        "google/siglip2-so400m-patch14-384": "Text+Image",
    }

    provider_map = {
        "text-embedding-v3": "DashScope",
        "multimodal-embedding-v1": "DashScope",
        "text-embedding-3-large": "OpenAI",
        "embed-v4.0": "Cohere",
        "jina-embeddings-v4": "Jina (API)",
        "voyage-multimodal-3.5": "Voyage",
        "doubao-embedding-text-240715": "Volcengine/ARK",
        "nomic-embed-text": "Ollama",
        "mxbai-embed-large": "Ollama",
        "bge-m3": "Ollama",
        "snowflake-arctic-embed:335m": "Ollama",
        "dengcao/Qwen3-Embedding-8B:Q5_K_M": "Ollama",
        "BAAI/bge-m3": "SentenceTransformers",
        "clip-ViT-B-32": "SentenceTransformers",
        "clip-ViT-L-14": "SentenceTransformers",
        "jinaai/jina-embeddings-v3": "SentenceTransformers",
        "Qwen/Qwen3-VL-Embedding-2B": "Transformers",
        "google/siglip2-so400m-patch14-384": "Transformers",
    }

    for model in all_models:
        d = all_models[model]
        best_str = f"**{d['best']}**" if d["best"] != "—" else "—"
        L(f"| {short_name(model)} | {provider_map.get(model, '?')} | {model_type(model)} "
          f"| {d['dims']} | {modality_map.get(model, '?')} "
          f"| {d['mrl']} | {d['needle']} | {d['cross']} | {best_str} |")
    L("")
    L("\\* Qwen3-Embedding-8B supports images in theory, but the quantized Ollama version "
      "produces non-functional cross-modal embeddings.  ")
    L("Note: CLIP models support text+image natively but have weak text-only similarity "
      "(ρ 0.62–0.66) compared to text-specialized models (ρ 0.85+).  ")
    L("Note: Cross-modal R@1 for Qwen3-VL-2B (n=20), SigLIP2 (n=50), and Voyage (n=10) "
      "were tested on smaller pools than DashScope/CLIP (n=200). Hard-negative scores are "
      "more informative for comparing across different pool sizes.\n")

    # ------------------------------------------------------------------
    # Key Takeaways
    # ------------------------------------------------------------------
    L("## 6. Key Takeaways & Recommendations\n")

    L("### When to Use Which Model\n")
    L("| Use Case | Recommended Model | Why |")
    L("|----------|-------------------|-----|")
    L("| Short text similarity / search | MxbAI Embed Large (Ollama) | Best Spearman ρ (0.893), free, local |")
    L("| Long document search (>4K chars) | OpenAI 3-large or Cohere v4 | Perfect accuracy (1.000), robust at all positions |")
    L("| Cross-modal (text↔image), best quality | DashScope MM-v1 | R@1=0.985 on 200 pairs, best hard-negative resistance (0.770 on 600) |")
    L("| Cross-modal, local/free | Qwen3-VL-2B (Transformers) | R@1=1.000 (n=20), hard R@1=0.850, smallest modality gap (0.28) |")
    L("| Cross-modal, CLIP baseline | CLIP ViT-L-14 (sentence-transformers) | R@1=0.958 on 200 pairs, open-source, well-studied |")
    L("| Budget-friendly with dimension flexibility | DashScope text-embedding-v3 | Best compression resilience, cheap API |")
    L("| Local deployment, long text | Jina v3 (sentence-transformers, GPU) | Perfect needle accuracy (1.000), strong MRL, open-source |")
    L("| Local deployment, short text | MxbAI Embed Large (Ollama) | Best quality, runs on CPU |")
    L("| MRL dimension compression | DashScope v3 or Jina v3 Local | Both retain >96% quality at half-dim, >80% at 32d |")
    L("")

    L("### API Models Comparison\n")
    L("| Provider | Model | MRL ρ | Needle | Cross-Modal | Strengths |")
    L("|----------|-------|-------|--------|-------------|-----------|")
    L("| DashScope | text-embedding-v3 | 0.875 | 0.993 | — | Best MRL compression resilience |")
    L("| DashScope | multimodal-embedding-v1 | — | — | 0.985 | Best hard-negative discrimination (n=200) |")
    L("| OpenAI | text-embedding-3-large | 0.836 | 1.000 | — | Perfect needle, large dim (3072) |")
    L("| Cohere | embed-v4.0 | — | 1.000 | — | Perfect needle |")
    L("| Jina | jina-embeddings-v4 | 0.887 | 1.000 | — | Best MRL ρ among APIs, perfect needle |")
    L("| Voyage | voyage-multimodal-3.5 | 0.884 | 1.000 | 1.000* | Multimodal + perfect needle |")
    L("| Volcengine | doubao-embedding-text-240715 | — | 1.000 | — | Perfect needle, Chinese-optimized |")
    L("")
    L("\\* Voyage cross-modal tested on only 10 pairs.\n")

    L("### Quantization Impact\n")
    L("The most striking finding is the **severe degradation from quantization on long-text tasks**:\n")
    L("| Model | Quantized (Ollama) | Full Precision (GPU) | Delta |")
    L("|-------|-------------------|---------------------|-------|")
    L("| BGE-M3 MRL ρ | 0.848 | 0.848 | 0.000 |")
    L("| BGE-M3 Needle Acc | 0.693 | 0.973 | **+0.280** |")
    L("")
    L("- **Short-text similarity is unaffected** by quantization (identical MRL scores).")
    L("- **Long-text understanding is devastated** by quantization (28pp accuracy drop).")
    L("- **Recommendation**: For any task involving documents longer than ~1K characters, "
      "use full-precision models or API providers. Reserve quantized models for short-text "
      "search and similarity tasks only.\n")

    L("### Local Multimodal Models\n")
    L("- **Qwen3-VL-2B** is the standout local multimodal model: perfect needle accuracy, strong MRL "
      "(0.820 at 2048d), and excellent cross-modal retrieval with the smallest modality gap (0.28). "
      "At 2B parameters, it's practical to run on a single GPU.")
    L("- **SigLIP2-400M** is excellent for cross-modal retrieval (R@1=1.000 on 50 pairs, hard R@1=0.680 on 150) "
      "and very lightweight (~400M params), but only supports image+text, not text-only tasks.")
    L("- **Jina v3 (Local)** delivers API-level needle accuracy (1.000) with good MRL compression, "
      "making it the best open-source text-only embedding model in our evaluation.\n")

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================
def main():
    print("Generating charts...")
    chart_mrl_line()
    chart_mrl_bar()
    chart_cross_modal()
    chart_needle_heatmap()
    chart_needle_bar()

    # Note: evaluation_report.md is maintained manually in Chinese.
    # Only regenerate it if --with-report is passed.
    import sys
    if "--with-report" in sys.argv:
        print("Generating markdown report...")
        md = generate_markdown()
        REPORT_MD.write_text(md, encoding="utf-8")
        print(f"Report saved to {REPORT_MD}")

    print(f"Figures saved to {FIG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
