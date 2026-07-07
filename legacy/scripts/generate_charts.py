"""Generate evaluation charts for blog post using matplotlib.

Creates 5 PNG charts in reports/blog_assets/:
  1. Radar/spider chart: 6 models across 4 tasks
  2. Cross-modal horizontal bar chart: 5 models
  3. MRL horizontal bar chart: 6 models
  4. Crosslingual horizontal bar chart: 8 models
  5. MRL degradation grouped bar chart: 4 models x 4 dimensions

Usage:
    uv run python scripts/generate_charts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Output directory ──────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "blog_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

# Try to load a Chinese-capable font
_CN_FONT = None
for family in ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei"]:
    try:
        from matplotlib.font_manager import FontProperties
        fp = FontProperties(family=family)
        if fp.get_name() != family:
            continue
        _CN_FONT = family
        break
    except Exception:
        continue

if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ── Color palette ─────────────────────────────────────────────────────────
COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B3",
    "#CCB974", "#64B5CD", "#E8A0BF", "#76C7C0",
]


# ══════════════════════════════════════════════════════════════════════════
# Chart 1: Radar / Spider chart — 6 models × 4 tasks
# ══════════════════════════════════════════════════════════════════════════
def chart1_radar():
    categories = ["MRL\n(Spearman ρ)", "Cross-Modal\n(hard R@1)", "Crosslingual\n(hard R@1)", "Needle\n(accuracy)"]
    n = len(categories)

    # model: [MRL, CrossModal, Crosslingual, Needle]
    # Use None for tasks the model doesn't support
    models = {
        "Gemini Embed 2":       [0.683, 0.928, 0.973, 0.917],
        "Voyage MM-3.5":        [0.880, 0.900, 0.983, 0.833],
        "Jina v4":              [0.833, None,  0.978, 0.933],
        "OpenAI 3-large":       [0.767, None,  0.975, 1.000],
        "DashScope v3":         [0.779, None,  0.968, 1.000],
        "Qwen3-VL-2B (local)":  [0.776, 0.945, 0.988, 0.833],
    }

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], categories, fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=9, color="grey")

    for i, (name, vals) in enumerate(models.items()):
        plot_vals = []
        for v in vals:
            plot_vals.append(v if v is not None else 0)
        plot_vals += plot_vals[:1]
        color = COLORS[i % len(COLORS)]
        ax.plot(angles, plot_vals, "o-", linewidth=2, label=name, color=color, markersize=5)
        ax.fill(angles, plot_vals, alpha=0.08, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=10, frameon=True)
    ax.set_title("Multi-Task Embedding Model Comparison", fontsize=14, fontweight="bold", pad=25)

    fig.tight_layout()
    path = OUT_DIR / "chart1_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Chart 2: Cross-modal horizontal bar — 5 models
# ══════════════════════════════════════════════════════════════════════════
def chart2_crossmodal():
    data = [
        ("Qwen3-VL-2B",      0.945),
        ("Gemini Embed 2",    0.928),
        ("Voyage MM-3.5*",    0.900),
        ("Jina CLIP v2",      0.873),
        ("CLIP ViT-L-14",     0.768),
    ]
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, vals, color=COLORS[:len(data)][::-1], height=0.55, edgecolor="white")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("hard_avg_R@1", fontsize=12)
    ax.set_title("Cross-Modal Retrieval Ranking (Text ↔ Image)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)

    # Footnote
    ax.annotate("* Voyage: only 10 test pairs (rate limit)",
                xy=(0.99, 0.02), xycoords="axes fraction", ha="right",
                fontsize=9, fontstyle="italic", color="grey")

    fig.tight_layout()
    path = OUT_DIR / "chart2_crossmodal.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Chart 3: MRL horizontal bar — 6 models
# ══════════════════════════════════════════════════════════════════════════
def chart3_mrl():
    data = [
        ("Voyage MM-3.5",     0.880),
        ("Jina v4",           0.833),
        ("mxbai-embed-large", 0.815),
        ("DashScope v3",      0.788),
        ("Qwen3-VL-2B",      0.774),
        ("OpenAI 3-large",    0.760),
        ("Gemini Embed 2",    0.668),
    ]
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, vals, color=COLORS[:len(data)][::-1], height=0.55, edgecolor="white")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0.6, 0.95)
    ax.set_xlabel("Spearman ρ (rank correlation with full-dim)", fontsize=12)
    ax.set_title("MRL Dimension Compression Resilience (150-pair Hard Mode)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    path = OUT_DIR / "chart3_mrl.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Chart 4: Crosslingual horizontal bar — 8 models
# ══════════════════════════════════════════════════════════════════════════
def chart4_crosslingual():
    data = [
        ("Gemini Embed 2",     0.997),
        ("Qwen3-VL-2B",       0.988),
        ("Jina v4",            0.985),
        ("Voyage MM-3.5",      0.982),
        ("OpenAI 3-large",     0.967),
        ("Cohere v4",          0.955),
        ("BGE-M3 (568M)",      0.940),
        ("Jina CLIP v2",       0.934),
        ("nomic (137M)",       0.154),
        ("mxbai (335M)",       0.120),
    ]
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]

    # 10 colors needed — extend palette
    colors = (COLORS + ["#D4A76A", "#8FBC8F"])[:len(data)][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(names, vals, color=colors, height=0.55, edgecolor="white")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0.0, 1.10)
    ax.set_xlabel("hard_avg_R@1", fontsize=12)
    ax.set_title("Crosslingual Retrieval Ranking (Chinese ↔ English)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)

    fig.tight_layout()
    path = OUT_DIR / "chart4_crosslingual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Chart 5: MRL degradation grouped bar — 4 models × 4 dimensions
# ══════════════════════════════════════════════════════════════════════════
def chart5_mrl_degradation():
    dims = ["256", "512", "1024", "Full"]
    models = {
        "Voyage MM-3.5":   [0.874, 0.882, 0.880, 0.880],
        "Jina v4":         [0.828, 0.824, 0.836, 0.833],
        "OpenAI 3-large":  [0.762, 0.761, 0.760, 0.767],
        "Gemini Embed 2":  [0.689, 0.661, 0.668, 0.683],
    }

    x = np.arange(len(dims))
    width = 0.18
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, vals) in enumerate(models.items()):
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=COLORS[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=12)
    ax.set_xlabel("Embedding Dimension", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_ylim(0.6, 0.95)
    ax.set_title("MRL Degradation Across Dimensions (150-pair Hard Mode)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    path = OUT_DIR / "chart5_mrl_degradation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Chart 6a: MRL dumbbell — full dim vs 256 dim
# ══════════════════════════════════════════════════════════════════════════
def chart6a_mrl_dumbbell():
    # (model, dim_256, full_dim) — sorted by full_dim descending
    raw = [
        ("Voyage MM-3.5",      0.874, 0.880),
        ("Jina v4",            0.828, 0.833),
        ("mxbai-embed-large",  0.795, 0.815),
        ("nomic-embed-text",   0.774, 0.781),
        ("Qwen3-VL-2B",       0.776, 0.774),
        ("OpenAI 3-large",     0.762, 0.767),
        ("Gemini Embed 2",     0.689, 0.683),
    ]

    # Reverse so highest is at the top in horizontal bar layout
    raw = raw[::-1]
    names = [r[0] for r in raw]
    dim256 = [r[1] for r in raw]
    full = [r[2] for r in raw]

    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Connecting lines
    for i in range(len(names)):
        lo = min(dim256[i], full[i])
        hi = max(dim256[i], full[i])
        ax.plot([lo, hi], [y[i], y[i]], color="#bbbbbb", linewidth=3, zorder=1)

    # Dots
    ax.scatter(dim256, y, color="#e07b39", s=120, zorder=2, label="256 dimensions", edgecolors="white", linewidths=1)
    ax.scatter(full, y, color="#3a7ebf", s=120, zorder=2, label="Full dimensions", edgecolors="white", linewidths=1)

    # Gap annotations
    for i in range(len(names)):
        gap = full[i] - dim256[i]
        mid = (dim256[i] + full[i]) / 2
        sign = "+" if gap >= 0 else ""
        ax.text(max(dim256[i], full[i]) + 0.006, y[i], f"{sign}{gap * 100:.1f}%",
                va="center", fontsize=10, color="#555555", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlim(0.60, 0.95)
    ax.set_xlabel("Spearman ρ", fontsize=13)
    ax.set_title("MRL: Full Dimension vs 256 Dimension Quality",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "chart_mrl_dumbbell.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Chart 6b: MRL dimension degradation — line chart
# ══════════════════════════════════════════════════════════════════════════
def chart6b_mrl_lines():
    dims = ["256", "512", "1024", "Full"]

    # Data from results/eval_rerun_bugfix_20260315.json mrl_stress entries.
    # "Full" = highest available dimension for each model.
    models = {
        # label:         [dim_256, dim_512, dim_1024, full_dim]
        "Voyage MM-3.5":   [0.8740, 0.8823, 0.8804, 0.8804],  # full=1024
        "Jina v4":         [0.8284, 0.8242, 0.8363, 0.8330],  # full=2048
        "mxbai-embed-large": [0.7945, 0.8046, 0.8148, 0.8148],  # full=1024
        "nomic-embed-text": [0.7744, 0.7783, None,   0.7805],  # full=768, no 1024
        "OpenAI 3-large":  [0.7622, 0.7610, 0.7605, 0.7668],  # full=3072
        "Qwen3-VL-2B":    [0.7760, 0.7750, 0.7741, 0.7736],  # full=2048
        "Gemini Embed 2":  [0.6889, 0.6610, 0.6683, 0.6834],  # full=3072
    }

    markers = ["o", "s", "D", "^", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = [0, 1, 2, 3]

    for i, (name, vals) in enumerate(models.items()):
        color = COLORS[i % len(COLORS)]
        marker = markers[i % len(markers)]
        # Filter out None values (nomic has no 1024)
        xs = [x_pos[j] for j, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, f"-{marker}", linewidth=2.2, markersize=8,
                label=name, color=color, markeredgecolor="white", markeredgewidth=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(dims, fontsize=13)
    ax.set_xlabel("Embedding Dimension", fontsize=13)
    ax.set_ylabel("Spearman ρ", fontsize=13)
    ax.set_ylim(0.55, 0.95)
    ax.set_title("MRL Dimension Compression: Spearman ρ by Dimension",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "chart_mrl_lines.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Chart 7: Needle-in-a-Haystack heatmap — 13 models × 5 lengths
# ══════════════════════════════════════════════════════════════════════════
def chart6_needle_heatmap():
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    lengths = ["1K", "4K", "8K", "16K", "32K"]

    # (model_name, [1K, 4K, 8K, 16K, 32K])  — None = not tested
    raw = [
        ("Gemini Embed 2",    [1.000, 1.000, 1.000, 1.000, 1.000]),
        ("OpenAI 3-large",    [1.000, 1.000, 1.000, None,  None]),
        ("Jina v4",           [1.000, 1.000, 1.000, None,  None]),
        ("Cohere v4",         [1.000, 1.000, 1.000, None,  None]),
        ("Qwen3-VL-2B",       [1.000, 1.000, None,  None,  None]),
        ("Voyage MM-3.5",     [1.000, 1.000, None,  None,  None]),
        ("Jina CLIP v2",      [1.000, 1.000, 1.000, None,  None]),
        ("BGE-M3 (568M)",     [1.000, 1.000, 0.920, None,  None]),
        ("mxbai (335M)",      [0.980, 0.600, 0.400, None,  None]),
        ("nomic (137M)",      [1.000, 0.460, 0.440, None,  None]),
    ]

    model_names = [r[0] for r in raw]
    n_rows = len(raw)
    n_cols = len(lengths)

    # Build numeric matrix; use NaN for missing
    data = np.full((n_rows, n_cols), np.nan)
    for i, (_, vals) in enumerate(raw):
        for j, v in enumerate(vals):
            if v is not None:
                data[i, j] = v

    # Custom colormap: red/orange → yellow → green
    cmap = LinearSegmentedColormap.from_list(
        "needle",
        [(0.0, "#d73027"), (0.5, "#fee08b"), (0.85, "#a6d96a"), (1.0, "#1a9850")],
    )
    cmap.set_bad(color="#e8e8e8")  # gray for NaN

    fig, ax = plt.subplots(figsize=(12, 7))

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.3, vmax=1.0)

    # Grid lines — draw cell borders
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(lengths, fontsize=13, fontweight="bold")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(model_names, fontsize=12)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", size=0)

    # Cell text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                continue  # no text on missing cells
            # Choose text color: white on dark cells, black on light cells
            text_color = "white" if v < 0.65 else "black"
            label = f"{v:.3f}" if v < 1.0 else "1.000"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    ax.set_xlabel("Document Length", fontsize=13, labelpad=10)
    ax.set_title("Needle-in-a-Haystack: Accuracy by Document Length",
                 fontsize=15, fontweight="bold", pad=15)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Accuracy", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    path = OUT_DIR / "chart_needle_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    print(f"Generating charts in {OUT_DIR} ...")
    chart1_radar()
    chart2_crossmodal()
    chart3_mrl()
    chart4_crosslingual()
    chart5_mrl_degradation()
    chart6a_mrl_dumbbell()
    chart6b_mrl_lines()
    chart6_needle_heatmap()
    print("Done! All 8 charts generated.")


if __name__ == "__main__":
    main()
