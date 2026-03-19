"""Interactive data viewer for mm-embedding-bench datasets.

Usage:
    uv run streamlit run scripts/data_viewer_app.py --server.port 8501
"""

from __future__ import annotations

import difflib
import html as html_mod
import json
import math
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

STSB_PATH = DATA_DIR / "mrl_stress" / "stsb_test.jsonl"
CROSS_PATH = DATA_DIR / "cross_modal" / "metadata.jsonl"
NEEDLES_PATH = DATA_DIR / "needle_haystack" / "needles.jsonl"
HAYSTACKS_PATH = DATA_DIR / "needle_haystack" / "haystacks.jsonl"
IMAGES_DIR = DATA_DIR / "cross_modal" / "images"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Diff highlighting for hard negatives
# ---------------------------------------------------------------------------
def highlight_diff(caption: str, hard_neg: str) -> str:
    """Return HTML of hard_neg with changed words highlighted in bold red."""
    cap_words = caption.split()
    hn_words = hard_neg.split()
    sm = difflib.SequenceMatcher(None, cap_words, hn_words)

    parts: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        hn_chunk = " ".join(hn_words[j1:j2])
        escaped = html_mod.escape(hn_chunk)
        if op == "equal":
            parts.append(escaped)
        elif op in ("replace", "insert"):
            parts.append(
                f"<span style='color:#dc2626;font-weight:700;background:#fef2f2;"
                f"padding:1px 3px;border-radius:3px'>{escaped}</span>"
            )
        # 'delete' means words in caption but not in hard_neg — nothing to render

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MM-Embedding-Bench 数据查看器",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for score badges and card styling
st.markdown("""
<style>
.score-high { background: #f0fdf4; color: #15803d; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
.score-mid  { background: #fefce8; color: #a16207; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
.score-low  { background: #fef2f2; color: #dc2626; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
.caption-box { background: #f0fdf4; border-left: 3px solid #22c55e; padding: 8px 12px; border-radius: 0 8px 8px 0; margin: 4px 0; font-size: 14px; }
.hardneg-box { background: #fef2f2; border-left: 3px solid #ef4444; padding: 6px 12px; border-radius: 0 8px 8px 0; margin: 3px 0; font-size: 13px; color: #6b7280; }
.hardneg-label { font-size: 10px; font-weight: 600; color: #ef4444; text-transform: uppercase; }
.needle-box { background: #fefce8; border-left: 3px solid #eab308; padding: 8px 12px; border-radius: 0 8px 8px 0; margin: 4px 0; }
.query-box  { background: #dbeafe; border-left: 3px solid #3b82f6; padding: 8px 12px; border-radius: 0 8px 8px 0; margin: 4px 0; }
.cat-tag { display: inline-block; background: #f3f4f6; color: #6b7280; font-size: 11px; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
stsb_data = load_jsonl(str(STSB_PATH))
cross_data = load_jsonl(str(CROSS_PATH))
needles_data = load_jsonl(str(NEEDLES_PATH))
haystacks_data = load_jsonl(str(HAYSTACKS_PATH))

# ---------------------------------------------------------------------------
# Sidebar — dataset stats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 数据查看器")
    st.caption("MM-Embedding-Bench 评测数据集")
    st.divider()
    st.metric("STS-B 句对", len(stsb_data))
    st.metric("COCO 图文对", len(cross_data))
    st.metric("Needle 事实", len(needles_data))
    st.metric("Haystack 文档", len(haystacks_data))
    st.divider()

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    f"📊 MRL Stress (STS-B) [{len(stsb_data)}]",
    f"🖼️ 跨模态检索 (COCO) [{len(cross_data)}]",
    f"🪡 大海捞针 [{len(needles_data)}+{len(haystacks_data)}]",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: STS-B
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    # --- Score distribution chart ---
    scores = [item["score"] for item in stsb_data]
    avg_score = sum(scores) / len(scores)

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("总句对数", len(stsb_data))
    col_stat2.metric("平均分数", f"{avg_score:.2f}")
    col_stat3.metric("分数范围", f"{min(scores):.1f} – {max(scores):.1f}")

    # Histogram
    n_bins = 20
    bin_edges = [i * 5 / n_bins for i in range(n_bins + 1)]
    bin_counts = [0] * n_bins
    for s in scores:
        idx = min(int(s / 5 * n_bins), n_bins - 1)
        bin_counts[idx] += 1

    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
    bar_colors = []
    for c in bin_centers:
        if c >= 4:
            bar_colors.append("#22c55e")
        elif c >= 1:
            bar_colors.append("#eab308")
        else:
            bar_colors.append("#ef4444")

    fig_hist = go.Figure(go.Bar(
        x=bin_centers,
        y=bin_counts,
        marker_color=bar_colors,
        width=0.22,
    ))
    fig_hist.update_layout(
        height=180,
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis=dict(title="Score", dtick=1),
        yaxis=dict(title="Count"),
        plot_bgcolor="white",
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Filters ---
    with st.sidebar:
        st.subheader("STS-B 过滤器")
        score_range = st.slider(
            "分数范围",
            min_value=0.0,
            max_value=5.0,
            value=(0.0, 5.0),
            step=0.1,
            key="stsb_score_range",
        )

    search_q = st.text_input("🔍 搜索句子内容", key="stsb_search", placeholder="输入关键词...")

    # Filter
    filtered = stsb_data
    if score_range != (0.0, 5.0):
        filtered = [x for x in filtered if score_range[0] <= x["score"] <= score_range[1]]
    if search_q:
        q = search_q.lower()
        filtered = [x for x in filtered if q in x["text_a"].lower() or q in x["text_b"].lower()]

    st.caption(f"显示 {len(filtered)} / {len(stsb_data)} 条")

    # Pagination
    per_page = 50
    total_pages = max(1, math.ceil(len(filtered) / per_page))

    col_prev, col_page, col_next, col_info = st.columns([1, 2, 1, 3])
    if "stsb_page" not in st.session_state:
        st.session_state.stsb_page = 1
    # Reset page when filter changes
    if st.session_state.stsb_page > total_pages:
        st.session_state.stsb_page = 1

    with col_page:
        page = st.number_input(
            "页码", min_value=1, max_value=total_pages, value=st.session_state.stsb_page,
            key="stsb_page_input", label_visibility="collapsed",
        )
        st.session_state.stsb_page = page
    with col_info:
        st.caption(f"第 {page} / {total_pages} 页")

    start_idx = (page - 1) * per_page
    page_data = filtered[start_idx : start_idx + per_page]

    # Build colored table
    if page_data:
        rows_html = []
        for i, item in enumerate(page_data):
            idx = start_idx + i + 1
            s = item["score"]
            if s >= 4:
                cls = "score-high"
            elif s >= 1:
                cls = "score-mid"
            else:
                cls = "score-low"
            rows_html.append(
                f"<tr>"
                f"<td style='color:#9ca3af;width:40px'>{idx}</td>"
                f"<td>{item['text_a']}</td>"
                f"<td>{item['text_b']}</td>"
                f"<td><span class='{cls}'>{s:.1f}</span></td>"
                f"</tr>"
            )

        table_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
            "<thead><tr style='border-bottom:2px solid #e5e7eb;text-align:left'>"
            "<th style='padding:8px;color:#6b7280;font-size:12px'>#</th>"
            "<th style='padding:8px;color:#6b7280;font-size:12px'>Sentence A</th>"
            "<th style='padding:8px;color:#6b7280;font-size:12px'>Sentence B</th>"
            "<th style='padding:8px;color:#6b7280;font-size:12px;width:70px'>Score</th>"
            "</tr></thead><tbody>"
        )
        for row in rows_html:
            table_html += row
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("没有匹配的数据")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Cross-Modal
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    # --- Sidebar filters ---
    categories = sorted(set(item.get("category", "?") for item in cross_data))
    with st.sidebar:
        st.subheader("跨模态过滤器")
        selected_cat = st.selectbox("类别", ["全部"] + categories, key="cross_cat")

    cross_search = st.text_input("🔍 搜索描述内容", key="cross_search", placeholder="输入关键词...")

    # Filter
    cm_filtered = cross_data
    if selected_cat != "全部":
        cm_filtered = [x for x in cm_filtered if x.get("category") == selected_cat]
    if cross_search:
        q = cross_search.lower()
        cm_filtered = [
            x for x in cm_filtered
            if q in x.get("caption", "").lower()
            or q in x.get("original_caption", "").lower()
            or any(q in hn.lower() for hn in x.get("hard_negatives", []))
        ]

    st.caption(f"显示 {len(cm_filtered)} / {len(cross_data)} 项")

    # Pagination
    cm_per_page = 10
    cm_total_pages = max(1, math.ceil(len(cm_filtered) / cm_per_page))

    if "cross_page" not in st.session_state:
        st.session_state.cross_page = 1
    if st.session_state.cross_page > cm_total_pages:
        st.session_state.cross_page = 1

    cm_col1, cm_col2, cm_col3 = st.columns([2, 1, 3])
    with cm_col1:
        cm_page = st.number_input(
            "页码", min_value=1, max_value=cm_total_pages,
            value=st.session_state.cross_page, key="cross_page_input",
            label_visibility="collapsed",
        )
        st.session_state.cross_page = cm_page
    with cm_col3:
        st.caption(f"第 {cm_page} / {cm_total_pages} 页")

    cm_start = (cm_page - 1) * cm_per_page
    cm_slice = cm_filtered[cm_start : cm_start + cm_per_page]

    # Render cards in 2-column layout
    for row_start in range(0, len(cm_slice), 2):
        cols = st.columns(2)
        for col_idx in range(2):
            item_idx = row_start + col_idx
            if item_idx >= len(cm_slice):
                break
            item = cm_slice[item_idx]
            with cols[col_idx]:
                with st.container(border=True):
                    img_path = IMAGES_DIR / item["image_path"].replace("images/", "")

                    ic, tc = st.columns([1, 1.3])
                    with ic:
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.warning(f"图片未找到: {item['image_path']}")

                        cat = item.get("category", "?")
                        st.markdown(f"<span class='cat-tag'>{cat}</span> &nbsp; <span style='color:#9ca3af;font-size:11px'>#{item['id']} · COCO {item['coco_id']}</span>", unsafe_allow_html=True)

                    with tc:
                        # Caption
                        caption = item["caption"]
                        st.markdown(f"<div class='caption-box'>{html_mod.escape(caption)}</div>", unsafe_allow_html=True)
                        # Hard negatives with diff highlighting
                        for hi, hn in enumerate(item.get("hard_negatives", []), 1):
                            hn_html = highlight_diff(caption, hn)
                            st.markdown(
                                f"<div class='hardneg-box'><span class='hardneg-label'>Hard Negative #{hi}</span><br>{hn_html}</div>",
                                unsafe_allow_html=True,
                            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Needle-in-Haystack
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🪡 Needle 事实 & 查询")

    # Needles in 2-column grid
    for row_start in range(0, len(needles_data), 2):
        cols = st.columns(2)
        for col_idx in range(2):
            ni = row_start + col_idx
            if ni >= len(needles_data):
                break
            n = needles_data[ni]
            with cols[col_idx]:
                with st.container(border=True):
                    cat = n.get("category", "general")
                    st.markdown(f"**#{ni + 1}** · <span class='cat-tag'>{cat}</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='needle-box'>{n['needle']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='query-box'><b>Query:</b> {n['query']}</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("📄 Haystack 文档")

    hay_search = st.text_input("🔍 搜索文档内容", key="hay_search", placeholder="输入关键词...")

    hay_filtered = haystacks_data
    if hay_search:
        q = hay_search.lower()
        hay_filtered = [h for h in hay_filtered if q in h.get("text", "").lower()]

    st.caption(f"显示 {len(hay_filtered)} / {len(haystacks_data)} 篇文档")

    for i, h in enumerate(hay_filtered):
        length = h.get("length", 0)
        actual = h.get("actual_length", len(h.get("text", "")))
        if length >= 8000:
            len_label, len_color = "8K", "🔴"
        elif length >= 4000:
            len_label, len_color = "4K", "🟡"
        else:
            len_label, len_color = "1K", "🟢"

        preview = h.get("text", "")[:150]
        with st.expander(f"{len_color} **{len_label}** ({actual:,} chars) — {preview}..."):
            st.text(h.get("text", "")[:8000])
            if len(h.get("text", "")) > 8000:
                st.caption("(仅显示前 8000 字符)")
