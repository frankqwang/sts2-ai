"""Build a standalone HTML viewer for LLM step_trace.jsonl files."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


RUN_RE = re.compile(
    r"run:\s+char=(?P<char>\S+)\s+act=(?P<act>\S+)\s+floor=(?P<floor>\S+)\s+encounter=(?P<encounter>\S+)\s+round=(?P<round>\S+)\s+gold=(?P<gold>\S+)"
)
PLAYER_RE = re.compile(r"player:\s+hp=(?P<hp>\S+)\s+block=(?P<block>\S+)\s+energy=(?P<energy>\S+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="step_trace.jsonl")
    parser.add_argument("--out", default="", help="output .html path")
    parser.add_argument("--title", default="", help="viewer title")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    return parser.parse_args()


def read_jsonl(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _first_match(pattern: re.Pattern[str], text: str) -> dict[str, str]:
    match = pattern.search(text)
    return match.groupdict() if match else {}


def _short_action(row: dict[str, Any]) -> str:
    chosen = row.get("chosen_action") if isinstance(row.get("chosen_action"), dict) else {}
    action = str(chosen.get("action") or chosen.get("type") or "?")
    card = str(chosen.get("card_id") or "")
    card_index = chosen.get("card_index")
    target = chosen.get("target_id")
    parts = [action]
    if card:
        parts.append(card)
    elif card_index is not None:
        parts.append(f"hand[{card_index}]")
    if target not in (None, "", -1):
        parts.append(f"-> {target}")
    return " ".join(parts)


def _reason(row: dict[str, Any]) -> str:
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    return str(decoded.get("reason") or "")


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    user = str(row.get("user_message") or "")
    run = _first_match(RUN_RE, user)
    player = _first_match(PLAYER_RE, user)
    quality = row.get("quality_report") if isinstance(row.get("quality_report"), dict) else {}
    metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    return {
        "step": row.get("step"),
        "route": row.get("route"),
        "adapter": row.get("adapter_name") or row.get("adapter_key"),
        "action_mode": row.get("action_mode"),
        "gen_ms": row.get("gen_ms"),
        "invalid": bool(row.get("invalid_output")),
        "fallback": bool(decoded.get("used_fallback")),
        "flags": row.get("quality_flags") or [],
        "action": _short_action(row),
        "reason": _reason(row),
        "run": run,
        "player": player,
        "metrics": metrics,
        "enabled_count": row.get("enabled_count"),
        "retrieved_knowledge_count": len(row.get("retrieved_knowledge") or []),
    }


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    routes: dict[str, int] = {}
    flags: dict[str, int] = {}
    invalid = 0
    for row in rows:
        route = str(row.get("route") or "-")
        routes[route] = routes.get(route, 0) + 1
        if row.get("invalid_output"):
            invalid += 1
        for flag in row.get("quality_flags") or []:
            key = str(flag)
            flags[key] = flags.get(key, 0) + 1
    return {"steps": len(rows), "invalid": invalid, "routes": routes, "flags": flags}


def _default_out(trace: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return trace.parent / f"{trace.stem}_viewer_{stamp}.html"


def build_html(*, trace_path: Path, rows: list[dict[str, Any]], title: str) -> str:
    summaries = [_summary(row) for row in rows]
    payload = {
        "trace_path": str(trace_path),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "title": title or trace_path.parent.name,
        "stats": _stats(rows),
        "summaries": summaries,
        "rows": rows,
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page_title = escape(title or trace_path.parent.name)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --line: #d9dde4;
  --muted: #657083;
  --text: #141820;
  --accent: #2364aa;
  --bad: #b42318;
  --warn: #b05a00;
  --ok: #147a3f;
  --code: #0f172a;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }}
header {{ height: 58px; display: flex; align-items: center; gap: 18px; padding: 0 18px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 4; }}
h1 {{ font-size: 18px; margin: 0; white-space: nowrap; }}
.meta {{ color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.layout {{ display: grid; grid-template-columns: 380px minmax(0, 1fr); min-height: calc(100vh - 58px); }}
aside {{ border-right: 1px solid var(--line); background: var(--panel); overflow: auto; max-height: calc(100vh - 58px); }}
main {{ overflow: auto; max-height: calc(100vh - 58px); padding: 16px; }}
.toolbar {{ padding: 12px; border-bottom: 1px solid var(--line); display: grid; gap: 8px; }}
input, select, button {{ font: inherit; border: 1px solid var(--line); background: #fff; color: var(--text); border-radius: 6px; padding: 7px 9px; }}
button {{ cursor: pointer; }}
.stats {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; color: var(--muted); background: #fafbfc; }}
.step-list {{ display: grid; }}
.step-item {{ padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }}
.step-item:hover, .step-item.active {{ background: #eef5ff; }}
.step-top {{ display: flex; justify-content: space-between; gap: 10px; font-weight: 650; }}
.step-sub {{ color: var(--muted); font-size: 12px; margin-top: 3px; display: flex; gap: 8px; flex-wrap: wrap; }}
.step-reason {{ margin-top: 5px; color: #344054; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bad {{ color: var(--bad); }}
.warn {{ color: var(--warn); }}
.ok {{ color: var(--ok); }}
.cards {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-height: 74px; }}
.card b {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }}
.card strong {{ font-size: 18px; }}
.split {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; align-items: start; }}
.section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; margin-bottom: 12px; }}
.section h2 {{ margin: 0; padding: 10px 12px; font-size: 13px; border-bottom: 1px solid var(--line); background: #fbfcfd; display: flex; justify-content: space-between; align-items: center; }}
pre {{ margin: 0; padding: 12px; overflow: auto; white-space: pre-wrap; word-break: break-word; color: var(--code); font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; max-height: 62vh; }}
.json pre {{ max-height: 34vh; }}
.reason {{ padding: 12px; white-space: pre-wrap; }}
.hidden {{ display: none; }}
@media (max-width: 1100px) {{
  .layout {{ grid-template-columns: 1fr; }}
  aside {{ max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--line); }}
  main {{ max-height: none; }}
  .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .split {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>{page_title}</h1>
  <div class="meta" id="tracePath"></div>
</header>
<div class="layout">
  <aside>
    <div class="toolbar">
      <input id="q" placeholder="搜索 step / action / reason / prompt">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <select id="route"></select>
        <select id="flag"></select>
      </div>
      <div class="stats" id="stats"></div>
    </div>
    <div class="step-list" id="stepList"></div>
  </aside>
  <main>
    <div class="cards">
      <div class="card"><b>step</b><strong id="cStep">-</strong></div>
      <div class="card"><b>position</b><strong id="cPos">-</strong></div>
      <div class="card"><b>player</b><strong id="cPlayer">-</strong></div>
      <div class="card"><b>route</b><strong id="cRoute">-</strong></div>
      <div class="card"><b>quality</b><strong id="cQuality">-</strong></div>
    </div>
    <div class="section"><h2>Chosen / Reason</h2><div class="reason" id="reason"></div></div>
    <div class="section"><h2>Retrieved Knowledge</h2><pre id="knowledge"></pre></div>
    <div class="split">
      <div class="section"><h2>Input: user_message <button data-copy="input">copy</button></h2><pre id="input"></pre></div>
      <div class="section"><h2>Output: raw_generation / decoded <button data-copy="output">copy</button></h2><pre id="output"></pre></div>
    </div>
    <div class="split">
      <div class="section json"><h2>Quality</h2><pre id="quality"></pre></div>
      <div class="section json"><h2>Full Row JSON</h2><pre id="rowJson"></pre></div>
    </div>
  </main>
</div>
<script id="trace-json" type="application/json">{data}</script>
<script>
const DATA = JSON.parse(document.getElementById('trace-json').textContent);
const rows = DATA.rows || [];
const sums = DATA.summaries || [];
let active = 0;

const el = (id) => document.getElementById(id);
const fmt = (v) => v === null || v === undefined || v === '' ? '-' : String(v);
const pretty = (v) => JSON.stringify(v ?? null, null, 2);

function init() {{
  el('tracePath').textContent = DATA.trace_path + ' | built ' + DATA.built_at;
  renderStats();
  fillFilters();
  renderList();
  selectStep(0);
  el('q').addEventListener('input', renderList);
  el('route').addEventListener('change', renderList);
  el('flag').addEventListener('change', renderList);
  document.querySelectorAll('button[data-copy]').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const target = btn.getAttribute('data-copy') === 'input' ? el('input') : el('output');
      navigator.clipboard?.writeText(target.textContent || '');
    }});
  }});
}}

function renderStats() {{
  const stats = DATA.stats || {{}};
  const routeBits = Object.entries(stats.routes || {{}}).map(([k,v]) => `${{k}}:${{v}}`).join(' ');
  el('stats').innerHTML = [
    `<span class="pill">steps ${{stats.steps || 0}}</span>`,
    `<span class="pill">invalid ${{stats.invalid || 0}}</span>`,
    `<span class="pill">${{routeBits || 'routes -'}}</span>`
  ].join('');
}}

function fillFilters() {{
  const routes = [...new Set(sums.map(s => s.route || '-'))].sort();
  el('route').innerHTML = '<option value="">all routes</option>' + routes.map(r => `<option>${{escapeHtml(r)}}</option>`).join('');
  const flags = [...new Set(sums.flatMap(s => s.flags || []))].sort();
  el('flag').innerHTML = '<option value="">all flags</option>' + flags.map(f => `<option>${{escapeHtml(f)}}</option>`).join('');
}}

function matches(i) {{
  const q = el('q').value.trim().toLowerCase();
  const route = el('route').value;
  const flag = el('flag').value;
  const s = sums[i] || {{}};
  const r = rows[i] || {{}};
  if (route && (s.route || '-') !== route) return false;
  if (flag && !(s.flags || []).includes(flag)) return false;
  if (!q) return true;
  const hay = [
    s.step, s.route, s.adapter, s.action, s.reason,
    r.user_message, r.raw_generation,
    JSON.stringify(r.decoded || {{}}),
    JSON.stringify(r.chosen_action || {{}}),
    JSON.stringify(r.retrieved_knowledge || [])
  ].join('\\n').toLowerCase();
  return hay.includes(q);
}}

function renderList() {{
  const list = el('stepList');
  list.innerHTML = '';
  sums.forEach((s, i) => {{
    if (!matches(i)) return;
    const item = document.createElement('div');
    item.className = 'step-item' + (i === active ? ' active' : '');
    item.onclick = () => selectStep(i);
    const hp = s.player?.hp || '-';
    const pos = `F${{s.run?.floor || '?'}} R${{s.run?.round || '?'}}`;
    const flags = (s.flags || []).length ? `<span class="bad">${{escapeHtml((s.flags || []).join(','))}}</span>` : '<span class="ok">ok</span>';
    item.innerHTML = `
      <div class="step-top"><span>#${{fmt(s.step)}} ${{escapeHtml(s.action || '-')}}</span><span>${{escapeHtml(pos)}}</span></div>
      <div class="step-sub"><span>${{escapeHtml(s.route || '-')}}</span><span>hp ${{escapeHtml(hp)}}</span><span>${{flags}}</span></div>
      <div class="step-reason">${{escapeHtml(s.reason || '-')}}</div>`;
    list.appendChild(item);
  }});
}}

function selectStep(i) {{
  active = Math.max(0, Math.min(i, rows.length - 1));
  const row = rows[active] || {{}};
  const s = sums[active] || {{}};
  el('cStep').textContent = fmt(s.step);
  el('cPos').textContent = `F${{s.run?.floor || '?'}} R${{s.run?.round || '?'}}`;
  el('cPlayer').textContent = `${{s.player?.hp || '-'}} / b${{s.player?.block || '-'}} / e${{s.player?.energy || '-'}}`;
  el('cRoute').textContent = `${{s.route || '-'}} / ${{s.adapter || '-'}}`;
  const loss = s.metrics?.current_hp_loss;
  el('cQuality').textContent = `${{(s.flags || []).length ? (s.flags || []).join(',') : 'ok'}}${{loss ? ' loss=' + loss : ''}}`;
  el('reason').textContent = `${{s.action || '-'}}\\n${{s.reason || '-'}}`;
  el('knowledge').textContent = pretty(row.retrieved_knowledge || []);
  el('input').textContent = row.user_message || '';
  el('output').textContent = [
    'raw_generation:',
    row.raw_generation || '',
    '',
    'decoded:',
    pretty(row.decoded),
    '',
    'chosen_action:',
    pretty(row.chosen_action)
  ].join('\\n');
  el('quality').textContent = pretty({{quality_flags: row.quality_flags || [], quality_report: row.quality_report || null, stats: row.stats || null}});
  el('rowJson').textContent = pretty(row);
  renderList();
}}

function escapeHtml(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}

init();
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace).resolve()
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    rows = read_jsonl(trace_path, max_rows=max(0, args.max_rows))
    out = Path(args.out).resolve() if args.out else _default_out(trace_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(trace_path=trace_path, rows=rows, title=args.title), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
