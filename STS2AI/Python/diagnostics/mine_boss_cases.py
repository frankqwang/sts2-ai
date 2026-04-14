from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from analyze_iteration_replays import SkadaNameResolver
from analyze_training_window import _load_summaries
from analyze_training_window import _combat_entries


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return median(values) if values else 0.0


def _boss_combats(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        combat for combat in _combat_entries(summary)
        if str(combat.get("room_type") or "").lower() == "boss"
    ]


def _boss_case_label(summary: dict[str, Any], boss_combat: dict[str, Any]) -> str:
    if bool(summary.get("act1_cleared")):
        return "boss_clear"
    end_reason = str(boss_combat.get("end_reason") or summary.get("end_reason") or "unknown")
    if end_reason == "combat_pending_stall":
        return "boss_stall"
    if end_reason == "max_steps":
        return "boss_max_steps"
    if str(summary.get("outcome") or "") == "death":
        return "boss_loss"
    return "boss_unresolved"


def _build_report(summaries: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    resolver = SkadaNameResolver()
    boss_cases = []
    case_counts = Counter()
    enemy_counts = Counter()
    end_reason_counts = Counter()
    start_hp_by_case: dict[str, list[float]] = {}
    start_hp_pct_by_case: dict[str, list[float]] = {}
    potion_use_by_case: dict[str, list[float]] = {}
    action_count_by_case: dict[str, list[float]] = {}

    for summary in summaries:
        boss_combat_list = _boss_combats(summary)
        if boss_combat_list:
            boss_combat = boss_combat_list[-1]
            enemy_group = str(boss_combat.get("enemy_group") or "UNKNOWN")
            end_reason = str(boss_combat.get("end_reason") or summary.get("end_reason") or "unknown")
            start_hp = float(boss_combat.get("start_hp", 0) or 0)
            start_max_hp = max(1.0, float(boss_combat.get("start_max_hp", 1) or 1))
            potion_uses = int(boss_combat.get("potion_uses", 0) or 0)
            repeat_hits = int(boss_combat.get("repeat_hits", 0) or 0)
            action_count = len(boss_combat.get("actions") or [])
            label = _boss_case_label(summary, boss_combat)
        elif bool(summary.get("boss_reached")):
            enemy_group = "UNKNOWN"
            end_reason = str(summary.get("end_reason") or "unknown")
            start_hp = 0.0
            start_max_hp = 1.0
            potion_uses = 0
            repeat_hits = int(summary.get("repeat_max", 0) or 0)
            action_count = 0
            if bool(summary.get("act1_cleared")):
                label = "boss_clear"
            elif end_reason == "combat_pending_stall":
                label = "boss_stall"
            elif end_reason == "max_steps":
                label = "boss_max_steps"
            elif str(summary.get("outcome") or "") == "death":
                label = "boss_loss"
            else:
                label = "boss_unresolved"
        else:
            continue

        case_counts[label] += 1
        enemy_counts[enemy_group] += 1
        end_reason_counts[end_reason] += 1
        start_hp_by_case.setdefault(label, []).append(start_hp)
        start_hp_pct_by_case.setdefault(label, []).append(start_hp / start_max_hp)
        potion_use_by_case.setdefault(label, []).append(float(potion_uses))
        action_count_by_case.setdefault(label, []).append(float(action_count))
        boss_cases.append(
            {
                "iteration": int(summary.get("iteration", -1)),
                "episode": int(summary.get("episode", -1)),
                "label": label,
                "boss_enemy_group": enemy_group,
                "boss_enemy_display": resolver.enemy_group(enemy_group),
                "start_hp": int(start_hp),
                "start_max_hp": int(start_max_hp),
                "start_hp_pct": round(start_hp / start_max_hp, 4),
                "potion_uses": potion_uses,
                "repeat_hits": repeat_hits,
                "action_count": action_count,
                "end_reason": end_reason,
                "trace_path": str(summary.get("trace_path") or ""),
                "summary_path": str(summary.get("_path") or ""),
            }
        )

    boss_cases.sort(key=lambda item: (item["label"], -item["start_hp_pct"], item["iteration"], item["episode"]))
    case_summary = []
    for name, count in case_counts.most_common():
        case_summary.append(
            {
                "name": name,
                "count": count,
                "start_hp_mean": round(_mean(start_hp_by_case.get(name, [])), 4),
                "start_hp_pct_mean": round(_mean(start_hp_pct_by_case.get(name, [])), 4),
                "start_hp_pct_median": round(_median(start_hp_pct_by_case.get(name, [])), 4),
                "potion_uses_mean": round(_mean(potion_use_by_case.get(name, [])), 4),
                "action_count_mean": round(_mean(action_count_by_case.get(name, [])), 4),
                "examples": [case for case in boss_cases if case["label"] == name][:top_k],
            }
        )

    return {
        "boss_case_count": len(boss_cases),
        "case_counts": dict(case_counts),
        "boss_enemy_top": [
            {"name": name, "display": resolver.enemy_group(name), "count": count}
            for name, count in enemy_counts.most_common(top_k)
        ],
        "boss_end_reasons": dict(end_reason_counts),
        "case_summary": case_summary,
    }


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Boss Case 专项分析")
    lines.append("")
    lines.append(f"- Boss case 数量: `{report.get('boss_case_count', 0)}`")
    lines.append(f"- Case 分布: `{json.dumps(report.get('case_counts', {}), ensure_ascii=False)}`")
    lines.append(f"- Boss end_reason: `{json.dumps(report.get('boss_end_reasons', {}), ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Boss 遭遇 Top")
    for item in report.get("boss_enemy_top") or []:
        lines.append(f"- `{item['display']}`: {item['count']}")
    if not (report.get("boss_enemy_top") or []):
        lines.append("- 无")
    lines.append("")

    for case in report.get("case_summary") or []:
        lines.append(f"## {case['name']}")
        lines.append(f"- 数量: `{case['count']}`")
        lines.append(f"- 进 boss 起始 HP 均值: `{case['start_hp_mean']:.4f}`")
        lines.append(f"- 进 boss 起始 HP 比例均值 / 中位: `{case['start_hp_pct_mean']:.4f}` / `{case['start_hp_pct_median']:.4f}`")
        lines.append(f"- 药水使用均值: `{case['potion_uses_mean']:.4f}`")
        lines.append(f"- Boss 动作步数均值: `{case['action_count_mean']:.4f}`")
        lines.append("- 样例:")
        examples = case.get("examples") or []
        if not examples:
            lines.append("  - 无")
        for sample in examples:
            trace = sample.get("trace_path") or ""
            trace_link = f"[replay](/" + trace.replace("\\", "/") + ")" if trace else ""
            lines.append(
                f"  - iter `{sample['iteration']}` ep `{sample['episode']}` "
                f"{sample['boss_enemy_display']} hp `{sample['start_hp']}/{sample['start_max_hp']}` "
                f"potions `{sample['potion_uses']}` end `{sample['end_reason']}` {trace_link}".rstrip()
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="按 boss case 聚合训练 summary，分析清关/卡死/失败模式。")
    parser.add_argument("training_dir", help="训练输出目录")
    parser.add_argument("--iter-start", type=int, default=None, help="起始 iteration")
    parser.add_argument("--iter-end", type=int, default=None, help="结束 iteration")
    parser.add_argument("--top-k", type=int, default=10, help="每类保留样例数量")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    analysis_dir = training_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summaries = _load_summaries(training_dir, args.iter_start, args.iter_end)
    if not summaries:
        raise SystemExit("未找到可用的 *.summary.json。")

    report = _build_report(summaries, top_k=args.top_k)

    suffix = []
    if args.iter_start is not None:
        suffix.append(f"from_{args.iter_start:05d}")
    if args.iter_end is not None:
        suffix.append(f"to_{args.iter_end:05d}")
    name = "boss_cases" + ("_" + "_".join(suffix) if suffix else "")

    json_path = analysis_dir / f"{name}.json"
    md_path = analysis_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
