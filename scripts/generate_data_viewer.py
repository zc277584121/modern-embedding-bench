"""Generate an interactive HTML data viewer for all 3 evaluation datasets.

Reads JSONL data files and embeds them into a single self-contained HTML file.

Usage:
    uv run python scripts/generate_data_viewer.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT = ROOT / "reports" / "data_viewer.html"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    print("Loading data...")

    stsb = load_jsonl(DATA_DIR / "mrl_stress" / "stsb_test.jsonl")
    cross_modal = load_jsonl(DATA_DIR / "cross_modal" / "metadata.jsonl")
    needles = load_jsonl(DATA_DIR / "needle_haystack" / "needles.jsonl")
    haystacks = load_jsonl(DATA_DIR / "needle_haystack" / "haystacks.jsonl")

    print(f"  STS-B: {len(stsb)} pairs")
    print(f"  Cross-modal: {len(cross_modal)} items")
    print(f"  Needles: {len(needles)}, Haystacks: {len(haystacks)}")

    # Truncate haystack text for embedding (keep first 500 chars for preview)
    haystacks_slim = []
    for h in haystacks:
        haystacks_slim.append({
            "length": h["length"],
            "actual_length": h.get("actual_length", len(h.get("text", ""))),
            "text_preview": h["text"][:500],
            "text_full": h["text"][:8000],  # cap at 8K to keep file size reasonable
        })

    # Serialize as JSON for embedding
    stsb_json = json.dumps(stsb, ensure_ascii=False)
    cross_modal_json = json.dumps(cross_modal, ensure_ascii=False)
    needles_json = json.dumps(needles, ensure_ascii=False)
    haystacks_json = json.dumps(haystacks_slim, ensure_ascii=False)

    html = generate_html(stsb_json, cross_modal_json, needles_json, haystacks_json)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"\nSaved to {OUTPUT} ({size_kb:.0f} KB)")


def generate_html(
    stsb_json: str,
    cross_modal_json: str,
    needles_json: str,
    haystacks_json: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MM-Embedding-Bench 数据查看器</title>
<style>
/* ── Reset & Base ─────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg: #ffffff;
  --bg2: #f9fafb;
  --bg3: #f3f4f6;
  --fg: #111827;
  --fg2: #6b7280;
  --border: #e5e7eb;
  --accent: #3b82f6;
  --accent-light: #dbeafe;
  --green: #22c55e;
  --green-bg: #f0fdf4;
  --yellow: #eab308;
  --yellow-bg: #fefce8;
  --red: #ef4444;
  --red-bg: #fef2f2;
  --radius: 8px;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
}}

[data-theme="dark"] {{
  --bg: #111827;
  --bg2: #1f2937;
  --bg3: #374151;
  --fg: #f9fafb;
  --fg2: #9ca3af;
  --border: #4b5563;
  --accent-light: #1e3a5f;
  --green-bg: #052e16;
  --yellow-bg: #422006;
  --red-bg: #450a0a;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.6;
}}

/* ── Header ───────────────────────────────────────── */
.header {{
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}}

.header h1 {{
  font-size: 18px;
  font-weight: 700;
}}

.header-right {{
  display: flex;
  gap: 12px;
  align-items: center;
}}

.theme-toggle {{
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 12px;
  cursor: pointer;
  font-size: 14px;
  color: var(--fg);
}}

/* ── Tabs ─────────────────────────────────────────── */
.tabs {{
  display: flex;
  gap: 0;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
}}

.tab {{
  padding: 12px 20px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  color: var(--fg2);
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
  user-select: none;
}}

.tab:hover {{ color: var(--fg); }}
.tab.active {{
  color: var(--accent);
  border-bottom-color: var(--accent);
}}

.tab .badge {{
  background: var(--bg3);
  color: var(--fg2);
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 10px;
  margin-left: 6px;
}}

/* ── Content Panels ───────────────────────────────── */
.panel {{
  display: none;
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}}

.panel.active {{ display: block; }}

/* ── Controls ─────────────────────────────────────── */
.controls {{
  display: flex;
  gap: 16px;
  align-items: center;
  margin-bottom: 20px;
  flex-wrap: wrap;
}}

.search-box {{
  flex: 1;
  min-width: 200px;
  padding: 8px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-size: 14px;
  background: var(--bg);
  color: var(--fg);
  outline: none;
}}

.search-box:focus {{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-light); }}

.filter-label {{
  font-size: 13px;
  color: var(--fg2);
  white-space: nowrap;
}}

.range-slider {{
  width: 200px;
  accent-color: var(--accent);
}}

.stat-bar {{
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}}

.stat-card {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
  min-width: 120px;
}}

.stat-card .label {{ font-size: 12px; color: var(--fg2); }}
.stat-card .value {{ font-size: 20px; font-weight: 700; }}

/* ── STS-B Histogram ──────────────────────────────── */
.histogram {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 20px;
}}

.histogram canvas {{
  width: 100%;
  height: 120px;
}}

/* ── Table ─────────────────────────────────────────── */
.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}}

.data-table th {{
  background: var(--bg2);
  padding: 10px 12px;
  text-align: left;
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  color: var(--fg2);
  border-bottom: 2px solid var(--border);
  position: sticky;
  top: 60px;
  z-index: 10;
}}

.data-table td {{
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}}

.data-table tr:hover {{ background: var(--bg2); }}

.score-badge {{
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 13px;
}}

.score-high {{ background: var(--green-bg); color: #15803d; }}
.score-mid {{ background: var(--yellow-bg); color: #a16207; }}
.score-low {{ background: var(--red-bg); color: #dc2626; }}

/* ── Pagination ───────────────────────────────────── */
.pagination {{
  display: flex;
  gap: 4px;
  align-items: center;
  justify-content: center;
  margin-top: 20px;
  flex-wrap: wrap;
}}

.page-btn {{
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg);
  color: var(--fg);
  cursor: pointer;
  font-size: 13px;
}}

.page-btn:hover {{ background: var(--bg2); }}
.page-btn.active {{ background: var(--accent); color: white; border-color: var(--accent); }}
.page-btn:disabled {{ opacity: 0.4; cursor: default; }}

.page-info {{
  font-size: 13px;
  color: var(--fg2);
  padding: 0 12px;
}}

/* ── Cross-Modal Cards ────────────────────────────── */
.card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
  gap: 20px;
}}

.cm-card {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow);
  display: flex;
  min-height: 200px;
}}

.cm-card .img-side {{
  width: 240px;
  min-width: 240px;
  background: var(--bg3);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}}

.cm-card .img-side img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
}}

.cm-card .img-side .cat-tag {{
  position: absolute;
  top: 8px;
  left: 8px;
  background: rgba(0,0,0,0.6);
  color: white;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
}}

.cm-card .text-side {{
  flex: 1;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow: hidden;
}}

.cm-card .caption {{
  background: var(--green-bg);
  border-left: 3px solid var(--green);
  padding: 8px 12px;
  border-radius: 0 var(--radius) var(--radius) 0;
  font-size: 13px;
  line-height: 1.5;
}}

.cm-card .hard-neg {{
  background: var(--red-bg);
  border-left: 3px solid var(--red);
  padding: 6px 12px;
  border-radius: 0 var(--radius) var(--radius) 0;
  font-size: 12px;
  line-height: 1.4;
  color: var(--fg2);
}}

.cm-card .hard-neg-label {{
  font-size: 10px;
  font-weight: 600;
  color: var(--red);
  text-transform: uppercase;
  margin-bottom: 2px;
}}

.cm-card .item-id {{
  font-size: 11px;
  color: var(--fg2);
}}

/* ── Needle Cards ─────────────────────────────────── */
.needle-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}}

.needle-card {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  box-shadow: var(--shadow);
}}

.needle-card .needle-cat {{
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 8px;
}}

.needle-card .needle-text {{
  font-size: 13px;
  background: var(--yellow-bg);
  border-left: 3px solid var(--yellow);
  padding: 8px 12px;
  border-radius: 0 var(--radius) var(--radius) 0;
  margin-bottom: 8px;
  line-height: 1.5;
}}

.needle-card .query-text {{
  font-size: 13px;
  background: var(--accent-light);
  border-left: 3px solid var(--accent);
  padding: 8px 12px;
  border-radius: 0 var(--radius) var(--radius) 0;
  line-height: 1.5;
}}

.haystack-section {{
  margin-top: 24px;
}}

.haystack-item {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 12px;
  overflow: hidden;
}}

.haystack-header {{
  padding: 12px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  user-select: none;
}}

.haystack-header:hover {{ background: var(--bg3); }}

.haystack-meta {{
  display: flex;
  gap: 12px;
  font-size: 12px;
  color: var(--fg2);
}}

.haystack-meta .len-badge {{
  background: var(--bg3);
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
}}

.haystack-body {{
  display: none;
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  font-size: 13px;
  line-height: 1.7;
  max-height: 400px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
}}

.haystack-body.open {{ display: block; }}

.expand-btn {{
  font-size: 12px;
  color: var(--accent);
  cursor: pointer;
}}

/* ── Responsive ───────────────────────────────────── */
@media (max-width: 768px) {{
  .card-grid {{ grid-template-columns: 1fr; }}
  .cm-card {{ flex-direction: column; }}
  .cm-card .img-side {{ width: 100%; min-width: unset; height: 200px; }}
  .needle-grid {{ grid-template-columns: 1fr; }}
  .controls {{ flex-direction: column; }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>MM-Embedding-Bench 数据查看器</h1>
  <div class="header-right">
    <button class="theme-toggle" onclick="toggleTheme()">🌙 / ☀️</button>
  </div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('stsb')">
    MRL Stress (STS-B) <span class="badge" id="stsb-count"></span>
  </div>
  <div class="tab" onclick="switchTab('cross')">
    跨模态检索 (COCO) <span class="badge" id="cross-count"></span>
  </div>
  <div class="tab" onclick="switchTab('needle')">
    大海捞针 <span class="badge" id="needle-count"></span>
  </div>
</div>

<!-- ═══ Tab 1: STS-B ═══════════════════════════════ -->
<div class="panel active" id="panel-stsb">
  <div class="stat-bar">
    <div class="stat-card">
      <div class="label">总句对数</div>
      <div class="value" id="stsb-total">—</div>
    </div>
    <div class="stat-card">
      <div class="label">当前显示</div>
      <div class="value" id="stsb-filtered">—</div>
    </div>
    <div class="stat-card">
      <div class="label">平均分数</div>
      <div class="value" id="stsb-avg">—</div>
    </div>
  </div>

  <div class="histogram">
    <canvas id="stsb-histogram" height="120"></canvas>
  </div>

  <div class="controls">
    <input type="text" class="search-box" id="stsb-search" placeholder="搜索句子内容...">
    <span class="filter-label">分数范围:</span>
    <input type="range" class="range-slider" id="stsb-min" min="0" max="5" step="0.1" value="0">
    <span id="stsb-range-label">0.0 – 5.0</span>
    <input type="range" class="range-slider" id="stsb-max" min="0" max="5" step="0.1" value="5">
  </div>

  <table class="data-table" id="stsb-table">
    <thead>
      <tr>
        <th style="width:50px">#</th>
        <th>Sentence A</th>
        <th>Sentence B</th>
        <th style="width:80px">Score</th>
      </tr>
    </thead>
    <tbody id="stsb-tbody"></tbody>
  </table>

  <div class="pagination" id="stsb-pagination"></div>
</div>

<!-- ═══ Tab 2: Cross-Modal ═════════════════════════ -->
<div class="panel" id="panel-cross">
  <div class="stat-bar">
    <div class="stat-card">
      <div class="label">总图文对</div>
      <div class="value" id="cross-total">—</div>
    </div>
    <div class="stat-card">
      <div class="label">当前显示</div>
      <div class="value" id="cross-filtered">—</div>
    </div>
  </div>

  <div class="controls">
    <input type="text" class="search-box" id="cross-search" placeholder="搜索描述或类别...">
    <select id="cross-cat-filter" style="padding:8px 12px; border:1px solid var(--border); border-radius:var(--radius); background:var(--bg); color:var(--fg); font-size:14px;">
      <option value="">所有类别</option>
    </select>
  </div>

  <div class="card-grid" id="cross-grid"></div>
  <div class="pagination" id="cross-pagination"></div>
</div>

<!-- ═══ Tab 3: Needle ══════════════════════════════ -->
<div class="panel" id="panel-needle">
  <div class="stat-bar">
    <div class="stat-card">
      <div class="label">Needle 数量</div>
      <div class="value" id="needle-total">—</div>
    </div>
    <div class="stat-card">
      <div class="label">Haystack 数量</div>
      <div class="value" id="haystack-total">—</div>
    </div>
  </div>

  <h3 style="margin-bottom:12px; font-size:16px;">Needle 事实 & 查询</h3>
  <div class="needle-grid" id="needle-grid"></div>

  <h3 style="margin-bottom:12px; font-size:16px;">Haystack 文档</h3>
  <div class="controls">
    <input type="text" class="search-box" id="haystack-search" placeholder="搜索文档内容...">
  </div>
  <div class="haystack-section" id="haystack-list"></div>
</div>

<!-- ═══ Inline Data ════════════════════════════════ -->
<script>
const STSB_DATA = {stsb_json};
const CROSS_DATA = {cross_modal_json};
const NEEDLES_DATA = {needles_json};
const HAYSTACKS_DATA = {haystacks_json};
</script>

<script>
// ── Tab switching ────────────────────────────────
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t, i) => {{
    const panels = ['stsb', 'cross', 'needle'];
    t.classList.toggle('active', panels[i] === name);
  }});
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
}}

// ── Theme toggle ─────────────────────────────────
function toggleTheme() {{
  const d = document.documentElement;
  d.setAttribute('data-theme', d.getAttribute('data-theme') === 'dark' ? '' : 'dark');
}}

// ── Utility ──────────────────────────────────────
function escHtml(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

// ═══════════════════════════════════════════════════
// TAB 1: STS-B
// ═══════════════════════════════════════════════════
const STSB_PER_PAGE = 50;
let stsbFiltered = [...STSB_DATA];
let stsbPage = 0;

function scoreClass(s) {{
  if (s >= 4) return 'score-high';
  if (s >= 1) return 'score-mid';
  return 'score-low';
}}

function renderStsbHistogram() {{
  const canvas = document.getElementById('stsb-histogram');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 120 * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = 120;

  // Bin scores into 20 bins (0-0.25, 0.25-0.5, ...)
  const nBins = 20;
  const bins = new Array(nBins).fill(0);
  for (const item of STSB_DATA) {{
    const idx = Math.min(Math.floor(item.score / 5 * nBins), nBins - 1);
    bins[idx]++;
  }}
  const maxBin = Math.max(...bins);

  const pad = 40;
  const barW = (W - pad * 2) / nBins;

  ctx.clearRect(0, 0, W, H);

  for (let i = 0; i < nBins; i++) {{
    const barH = (bins[i] / maxBin) * (H - 30);
    const x = pad + i * barW;
    const y = H - 20 - barH;
    const score = (i / nBins) * 5;

    if (score >= 4) ctx.fillStyle = '#22c55e';
    else if (score >= 1) ctx.fillStyle = '#eab308';
    else ctx.fillStyle = '#ef4444';

    ctx.fillRect(x + 1, y, barW - 2, barH);
  }}

  // X axis labels
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('color') || '#666';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {{
    ctx.fillText(i.toString(), pad + (i / 5) * (W - pad * 2), H - 4);
  }}
  ctx.textAlign = 'left';
  ctx.fillText('Score 分布', 4, 12);
}}

function filterStsb() {{
  const q = document.getElementById('stsb-search').value.toLowerCase();
  const mn = parseFloat(document.getElementById('stsb-min').value);
  const mx = parseFloat(document.getElementById('stsb-max').value);

  document.getElementById('stsb-range-label').textContent = mn.toFixed(1) + ' – ' + mx.toFixed(1);

  stsbFiltered = STSB_DATA.filter(item => {{
    if (item.score < mn || item.score > mx) return false;
    if (q && !item.text_a.toLowerCase().includes(q) && !item.text_b.toLowerCase().includes(q)) return false;
    return true;
  }});

  document.getElementById('stsb-filtered').textContent = stsbFiltered.length;
  stsbPage = 0;
  renderStsbTable();
}}

function renderStsbTable() {{
  const start = stsbPage * STSB_PER_PAGE;
  const slice = stsbFiltered.slice(start, start + STSB_PER_PAGE);

  const tbody = document.getElementById('stsb-tbody');
  tbody.innerHTML = slice.map((item, i) => {{
    const idx = start + i + 1;
    const cls = scoreClass(item.score);
    return `<tr>
      <td style="color:var(--fg2)">${{idx}}</td>
      <td>${{escHtml(item.text_a)}}</td>
      <td>${{escHtml(item.text_b)}}</td>
      <td><span class="score-badge ${{cls}}">${{item.score.toFixed(1)}}</span></td>
    </tr>`;
  }}).join('');

  renderStsbPagination();
}}

function renderStsbPagination() {{
  const total = Math.ceil(stsbFiltered.length / STSB_PER_PAGE);
  const el = document.getElementById('stsb-pagination');
  if (total <= 1) {{ el.innerHTML = ''; return; }}

  let html = `<button class="page-btn" onclick="stsbPage=0;renderStsbTable()" ${{stsbPage===0?'disabled':''}}>&laquo;</button>`;
  html += `<button class="page-btn" onclick="stsbPage--;renderStsbTable()" ${{stsbPage===0?'disabled':''}}>‹</button>`;

  const start = Math.max(0, stsbPage - 3);
  const end = Math.min(total, start + 7);
  for (let i = start; i < end; i++) {{
    html += `<button class="page-btn ${{i===stsbPage?'active':''}}" onclick="stsbPage=${{i}};renderStsbTable()">${{i+1}}</button>`;
  }}

  html += `<span class="page-info">${{stsbPage+1}} / ${{total}}</span>`;
  html += `<button class="page-btn" onclick="stsbPage++;renderStsbTable()" ${{stsbPage>=total-1?'disabled':''}}>›</button>`;
  html += `<button class="page-btn" onclick="stsbPage=${{total-1}};renderStsbTable()" ${{stsbPage>=total-1?'disabled':''}}>&raquo;</button>`;
  el.innerHTML = html;
}}

// ═══════════════════════════════════════════════════
// TAB 2: Cross-Modal
// ═══════════════════════════════════════════════════
const CROSS_PER_PAGE = 12;
let crossFiltered = [...CROSS_DATA];
let crossPage = 0;

function filterCross() {{
  const q = document.getElementById('cross-search').value.toLowerCase();
  const cat = document.getElementById('cross-cat-filter').value;

  crossFiltered = CROSS_DATA.filter(item => {{
    if (cat && item.category !== cat) return false;
    if (q) {{
      const haystack = (item.caption + ' ' + item.original_caption + ' ' + (item.hard_negatives||[]).join(' ')).toLowerCase();
      if (!haystack.includes(q)) return false;
    }}
    return true;
  }});

  document.getElementById('cross-filtered').textContent = crossFiltered.length;
  crossPage = 0;
  renderCrossGrid();
}}

function renderCrossGrid() {{
  const start = crossPage * CROSS_PER_PAGE;
  const slice = crossFiltered.slice(start, start + CROSS_PER_PAGE);
  const grid = document.getElementById('cross-grid');

  grid.innerHTML = slice.map(item => {{
    const imgPath = '../data/cross_modal/' + item.image_path;
    const hns = (item.hard_negatives || []).map((hn, i) =>
      `<div class="hard-neg"><div class="hard-neg-label">Hard Negative #${{i+1}}</div>${{escHtml(hn)}}</div>`
    ).join('');

    return `<div class="cm-card">
      <div class="img-side">
        <img src="${{imgPath}}" alt="COCO ${{item.coco_id}}" loading="lazy"
             onerror="this.style.display='none';this.parentElement.innerHTML='<div style=\\'padding:20px;color:var(--fg2);font-size:12px\\'>Image not found<br>${{item.image_path}}</div>'">
        <span class="cat-tag">${{item.category || '?'}}</span>
      </div>
      <div class="text-side">
        <div class="item-id">#${{item.id}} · COCO ${{item.coco_id}}</div>
        <div class="caption">${{escHtml(item.caption)}}</div>
        ${{hns}}
      </div>
    </div>`;
  }}).join('');

  renderCrossPagination();
}}

function renderCrossPagination() {{
  const total = Math.ceil(crossFiltered.length / CROSS_PER_PAGE);
  const el = document.getElementById('cross-pagination');
  if (total <= 1) {{ el.innerHTML = ''; return; }}

  let html = `<button class="page-btn" onclick="crossPage=0;renderCrossGrid()" ${{crossPage===0?'disabled':''}}>&laquo;</button>`;
  html += `<button class="page-btn" onclick="crossPage--;renderCrossGrid()" ${{crossPage===0?'disabled':''}}>‹</button>`;

  const start = Math.max(0, crossPage - 3);
  const end = Math.min(total, start + 7);
  for (let i = start; i < end; i++) {{
    html += `<button class="page-btn ${{i===crossPage?'active':''}}" onclick="crossPage=${{i}};renderCrossGrid()">${{i+1}}</button>`;
  }}

  html += `<span class="page-info">${{crossPage+1}} / ${{total}}</span>`;
  html += `<button class="page-btn" onclick="crossPage++;renderCrossGrid()" ${{crossPage>=total-1?'disabled':''}}>›</button>`;
  html += `<button class="page-btn" onclick="crossPage=${{total-1}};renderCrossGrid()" ${{crossPage>=total-1?'disabled':''}}>&raquo;</button>`;
  el.innerHTML = html;
}}

// ═══════════════════════════════════════════════════
// TAB 3: Needle
// ═══════════════════════════════════════════════════
function renderNeedles() {{
  const grid = document.getElementById('needle-grid');
  grid.innerHTML = NEEDLES_DATA.map((n, i) => `
    <div class="needle-card">
      <div class="needle-cat">#${{i+1}} · ${{n.category || 'general'}}</div>
      <div class="needle-text">${{escHtml(n.needle)}}</div>
      <div class="query-text"><strong>Query:</strong> ${{escHtml(n.query)}}</div>
    </div>
  `).join('');
}}

function renderHaystacks() {{
  const q = (document.getElementById('haystack-search').value || '').toLowerCase();
  const list = document.getElementById('haystack-list');

  const filtered = q
    ? HAYSTACKS_DATA.filter(h => h.text_preview.toLowerCase().includes(q) || h.text_full.toLowerCase().includes(q))
    : HAYSTACKS_DATA;

  list.innerHTML = filtered.map((h, i) => {{
    const lenLabel = h.length >= 8000 ? '8K' : h.length >= 4000 ? '4K' : '1K';
    const lenColor = h.length >= 8000 ? 'var(--red)' : h.length >= 4000 ? 'var(--yellow)' : 'var(--green)';
    return `<div class="haystack-item">
      <div class="haystack-header" onclick="this.nextElementSibling.classList.toggle('open');this.querySelector('.expand-btn').textContent=this.nextElementSibling.classList.contains('open')?'收起':'展开'">
        <div class="haystack-meta">
          <span class="len-badge" style="color:${{lenColor}}">${{lenLabel}} (${{h.actual_length}} chars)</span>
          <span>${{escHtml(h.text_preview.slice(0, 120))}}...</span>
        </div>
        <span class="expand-btn">展开</span>
      </div>
      <div class="haystack-body">${{escHtml(h.text_full)}}</div>
    </div>`;
  }}).join('');
}}

// ═══════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {{
  // Badges
  document.getElementById('stsb-count').textContent = STSB_DATA.length;
  document.getElementById('cross-count').textContent = CROSS_DATA.length;
  document.getElementById('needle-count').textContent = NEEDLES_DATA.length + '+' + HAYSTACKS_DATA.length;

  // STS-B
  document.getElementById('stsb-total').textContent = STSB_DATA.length;
  document.getElementById('stsb-filtered').textContent = STSB_DATA.length;
  const avg = STSB_DATA.reduce((s, x) => s + x.score, 0) / STSB_DATA.length;
  document.getElementById('stsb-avg').textContent = avg.toFixed(2);
  renderStsbHistogram();
  filterStsb();

  document.getElementById('stsb-search').addEventListener('input', filterStsb);
  document.getElementById('stsb-min').addEventListener('input', filterStsb);
  document.getElementById('stsb-max').addEventListener('input', filterStsb);

  // Cross-modal
  document.getElementById('cross-total').textContent = CROSS_DATA.length;
  document.getElementById('cross-filtered').textContent = CROSS_DATA.length;

  // Populate category filter
  const cats = [...new Set(CROSS_DATA.map(x => x.category).filter(Boolean))].sort();
  const sel = document.getElementById('cross-cat-filter');
  cats.forEach(c => {{
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  }});

  filterCross();
  document.getElementById('cross-search').addEventListener('input', filterCross);
  document.getElementById('cross-cat-filter').addEventListener('change', filterCross);

  // Needle
  document.getElementById('needle-total').textContent = NEEDLES_DATA.length;
  document.getElementById('haystack-total').textContent = HAYSTACKS_DATA.length;
  renderNeedles();
  renderHaystacks();
  document.getElementById('haystack-search').addEventListener('input', renderHaystacks);

  // Resize histogram on window resize
  window.addEventListener('resize', renderStsbHistogram);
}});
</script>

</body>
</html>"""


if __name__ == "__main__":
    main()
