#!/usr/bin/env python3
"""Render a human-readable report from offline non-combat branch rollout JSONL data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sanitize(text: Any) -> str:
    value = str(text or "").replace('"', "'").replace("\n", " ").strip()
    return value[:80] if len(value) > 80 else value


def _branch_label(branch: dict[str, Any], option_index: int, best_idx: int) -> str:
    label = _sanitize(branch.get("branch_label") or branch.get("option_metadata", {}).get("card_id") or f"option_{option_index}")
    score = float(branch.get("score") or 0.0)
    suffix = " (best)" if option_index == best_idx else ""
    return f"{label} | {score:.4f}{suffix}"


def render_report(rows: list[dict[str, Any]], seed: str) -> str:
    ordered = sorted(
        (row for row in rows if str(row.get("seed") or "") == seed),
        key=lambda row: (int(row.get("floor") or 0), int(row.get("sample_index") or 0)),
    )
    if not ordered:
        raise ValueError(f"seed not found: {seed}")

    lines: list[str] = []
    lines.append(f"# 离线分支可视化：{seed}")
    lines.append("")
    lines.append(f"- 决策点数：{len(ordered)}")
    lines.append(f"- 样本类型：{', '.join(sorted({str(row.get('sample_type') or 'unknown') for row in ordered}))}")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    lines.append('  start["Start"]')

    previous_best_node = "start"
    for row_index, row in enumerate(ordered):
        sample_type = str(row.get("sample_type") or "unknown")
        floor = int(row.get("floor") or 0)
        best_idx = int(row.get("best_idx") or 0)
        tree_summary = row.get("tree_summary") or {}
        decision_node = f"d{row_index}"
        decision_label = f"f{floor:02d} {sample_type} s{int(row.get('sample_index') or 0):04d}"
        if tree_summary:
            summary_bits = []
            if "max_reward_depth" in tree_summary:
                summary_bits.append(f"depth={tree_summary['max_reward_depth']}")
            if "beam_width" in tree_summary:
                summary_bits.append(f"beam={tree_summary['beam_width']}")
            if summary_bits:
                decision_label = f"{decision_label} | {' '.join(summary_bits)}"
        lines.append(f'  {decision_node}["{_sanitize(decision_label)}"]')
        lines.append(f"  {previous_best_node} --> {decision_node}")

        best_option_node = None
        for option_index, branch in enumerate(row.get("branch_rollouts") or []):
            option_node = f"d{row_index}_o{option_index}"
            option_label = _branch_label(branch, option_index, best_idx)
            lines.append(f'  {option_node}["{_sanitize(option_label)}"]')
            lines.append(f"  {decision_node} --> {option_node}")
            if option_index == best_idx:
                best_option_node = option_node

        previous_best_node = best_option_node or decision_node

    lines.append("```")
    lines.append("")
    lines.append("## 决策明细")
    lines.append("")
    for row in ordered:
        sample_type = str(row.get("sample_type") or "unknown")
        floor = int(row.get("floor") or 0)
        best_idx = int(row.get("best_idx") or 0)
        lines.append(f"### floor {floor} · {sample_type} · sample {int(row.get('sample_index') or 0)}")
        lines.append("")
        for option_index, branch in enumerate(row.get("branch_rollouts") or []):
            option_label = _sanitize(branch.get("branch_label") or branch.get("option_metadata", {}).get("card_id") or option_index)
            score = float(branch.get("score") or 0.0)
            terminal = branch.get("terminal_summary") or {}
            bits = [
                f"score={score:.4f}",
                f"hp_after={int(terminal.get('hp_after') or 0)}",
                f"boss_reached={bool(terminal.get('boss_reached'))}",
                f"reason={_sanitize(terminal.get('terminal_reason') or terminal.get('terminal_state_type') or '')}",
            ]
            prefix = "*" if option_index == best_idx else "-"
            lines.append(f"{prefix} {option_label}: {'; '.join(bits)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render one offline branch episode as Markdown + Mermaid")
    parser.add_argument("--input", required=True, help="Path to raw_branch_rollout.jsonl")
    parser.add_argument("--seed", required=True, help="Episode seed to render")
    parser.add_argument("--output", required=True, help="Markdown output path")
    args = parser.parse_args()

    rows = _load_rows(Path(args.input))
    markdown = render_report(rows, str(args.seed))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
