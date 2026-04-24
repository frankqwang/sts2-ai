"""读 llm_policy 写的 step_trace.jsonl，把 LLM 的每步 input / output /
思考过程渲染成可读文本。

两种用法：

1) 观战实时跟进（另开一个终端）：

    python llm\\scripts\\replay_trace.py --trace "<run>\\step_trace.jsonl" --follow

2) 战斗结束后回看：

    python llm\\scripts\\replay_trace.py --trace "<run>\\step_trace.jsonl"

默认每步只显示状态摘要，`--full` 打印完整 user 消息；`--raw-only` 只打
LLM 原始输出（最适合看模型有没有乱说话）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, type=str, help="step_trace.jsonl 路径")
    p.add_argument("--follow", action="store_true", help="tail -f 模式，有新行就打")
    p.add_argument("--full", action="store_true", help="展开完整 user 消息（默认只摘要）")
    p.add_argument("--raw-only", action="store_true", help="只打 LLM raw_generation，调试格式错误时用")
    p.add_argument("--no-color", action="store_true", help="禁用 ANSI 颜色")
    return p.parse_args()


def iter_jsonl(path: Path, *, follow: bool, poll_s: float = 0.2) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                if not follow:
                    return
                time.sleep(poll_s)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class Styler:
    def __init__(self, *, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    dim = lambda self, t: self._wrap("2", t)
    bold = lambda self, t: self._wrap("1", t)
    cyan = lambda self, t: self._wrap("36", t)
    green = lambda self, t: self._wrap("32", t)
    yellow = lambda self, t: self._wrap("33", t)
    red = lambda self, t: self._wrap("31", t)
    magenta = lambda self, t: self._wrap("35", t)


_RUN_LINE = re.compile(r"^run: .*encounter=(\S+)")
_PLAYER_LINE = re.compile(r"^player: hp=(\S+) block=(\S+) energy=(\S+)")


def summarize_user(user_msg: str) -> str:
    encounter = "?"
    hp = energy = block = "?"
    for line in user_msg.splitlines():
        m = _RUN_LINE.match(line)
        if m:
            encounter = m.group(1)
        m = _PLAYER_LINE.match(line)
        if m:
            hp, block, energy = m.group(1), m.group(2), m.group(3)
    enemies = [ln.strip() for ln in user_msg.splitlines() if ln.strip().startswith("id=")]
    hand = [ln.strip() for ln in user_msg.splitlines() if ln.strip().startswith("[") and "cost=" in ln]
    return (
        f"encounter={encounter} hp={hp} block={block} energy={energy} "
        f"enemies={len(enemies)} hand={len(hand)}"
    )


def render_step(entry: dict, *, style: Styler, full: bool, raw_only: bool) -> str:
    step = entry.get("step", "?")
    gen_ms = entry.get("gen_ms", 0.0)
    decoded = entry.get("decoded", {})
    chosen = entry.get("chosen_action", {})
    enabled_count = entry.get("enabled_count", "?")
    raw = str(entry.get("raw_generation", "")).strip()
    user_msg = str(entry.get("user_message", "")).strip()

    if raw_only:
        header = style.dim(f"--- step {step} ({gen_ms:.0f} ms) ---")
        return header + "\n" + raw + "\n"

    fallback = bool(decoded.get("used_fallback"))
    fb_reason = decoded.get("fallback_reason", "")
    action_idx = decoded.get("action_index", "?")
    llm_reason = str(decoded.get("reason", "")).strip()

    status_icon = style.red("✗") if fallback else style.green("✓")
    header_parts = [
        style.bold(f"[step {step:>3}]"),
        f"gen={gen_ms/1000:.1f}s",
        f"legal={enabled_count}",
        f"idx={action_idx} {status_icon}",
    ]
    if fallback:
        header_parts.append(style.red(f"FALLBACK:{fb_reason}"))
    header = " ".join(header_parts)

    lines = [header]

    # 状态摘要（或完整）
    if full and user_msg:
        lines.append(style.dim("  user message:"))
        for ln in user_msg.splitlines():
            lines.append(style.dim("    " + ln))
    else:
        lines.append(style.dim("  state: ") + summarize_user(user_msg))

    # LLM 原始输出
    if raw:
        # 标出 JSON 部分
        json_match = re.search(r"\{[^{}]*\"action_index\"[^{}]*\}", raw)
        if json_match:
            pre = raw[: json_match.start()].rstrip()
            json_str = raw[json_match.start() : json_match.end()]
            post = raw[json_match.end() :].strip()
            if pre:
                lines.append(style.magenta("  thinking: ") + pre.replace("\n", " "))
            lines.append(style.cyan("  JSON:     ") + json_str)
            if post:
                lines.append(style.dim("  (trailing: " + post[:60] + ")"))
        else:
            lines.append(style.yellow("  raw (no JSON): ") + raw[:200])

    # 解析 reason + sim 实际执行
    if llm_reason and not fallback:
        lines.append(style.green("  reason: ") + llm_reason)
    action_type = chosen.get("action") or chosen.get("type") or "?"
    card = chosen.get("card_id") or ""
    target = chosen.get("target_id")
    act_parts = [action_type]
    if card:
        act_parts.append(f"card={card}")
    if target:
        act_parts.append(f"target={target}")
    lines.append(style.bold("  action: ") + " ".join(act_parts))

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace)
    if not trace_path.exists() and not args.follow:
        print(f"trace not found: {trace_path}", file=sys.stderr)
        return 2
    if not trace_path.exists() and args.follow:
        print(f"waiting for trace file: {trace_path}")
        while not trace_path.exists():
            time.sleep(0.5)

    style = Styler(enabled=not args.no_color and sys.stdout.isatty())
    print(style.dim(f"=== trace: {trace_path} ==="))
    for entry in iter_jsonl(trace_path, follow=args.follow):
        print(render_step(entry, style=style, full=args.full, raw_only=args.raw_only))
        print()  # 每步后空一行
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
