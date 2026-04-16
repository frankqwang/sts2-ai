"""Classify episode failures into categories (boss death, early death, stall, loop, etc.)."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_iteration_replays import SkadaNameResolver
from analyze_training_window import _load_summaries
from analyze_training_window import _combat_entries


def _boss_combats(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        combat for combat in _combat_entries(summary)
        if str(combat.get("room_type") or "").lower() == "boss"
    ]


def _classify_episode(summary: dict[str, Any], *, early_floor_threshold: int, pending_stall_threshold: int) -> tuple[str, dict[str, Any]]:
    outcome = str(summary.get("outcome") or "unknown")
    end_reason = str(summary.get("end_reason") or "unknown")
    final_floor = int(summary.get("final_floor", 0) or 0)
    boss_reached = bool(summary.get("boss_reached"))
    act1_cleared = bool(summary.get("act1_cleared"))
    repeat_max = int(summary.get("repeat_max", 0) or 0)
    pending_spans = summary.get("combat_pending_spans") or []
    long_pending = max((int(span.get("count", 0) or 0) for span in pending_spans), default=0)
    boss_combats = _boss_combats(summary)
    boss_combat = boss_combats[-1] if boss_combats else None

    if act1_cleared or outcome == "victory":
        return "act1_clear", {"final_floor": final_floor}
    if end_reason == "combat_pending_stall":
        return "combat_pending_stall", {"max_pending_span": long_pending}
    if end_reason == "repeat_loop":
        return "repeat_loop", {"repeat_max": repeat_max}
    if end_reason == "timeout":
        return "timeout", {}
    if boss_combat is not None:
        boss_end = str(boss_combat.get("end_reason") or end_reason or "unknown")
        boss_hp = int(boss_combat.get("start_hp", 0) or 0)
        if boss_end == "combat_pending_stall" or long_pending >= pending_stall_threshold:
            return "boss_stall", {"boss_start_hp": boss_hp, "max_pending_span": long_pending}
        if outcome == "death":
            return "boss_loss", {"boss_start_hp": boss_hp}
        if end_reason == "max_steps":
            return "boss_max_steps", {"boss_start_hp": boss_hp}
        return "boss_unresolved", {"boss_start_hp": boss_hp}
    if boss_reached:
        if end_reason == "combat_pending_stall" or long_pending >= pending_stall_threshold:
            return "boss_stall", {"max_pending_span": long_pending}
        if outcome == "death":
            return "boss_loss", {"final_floor": final_floor}
        if end_reason == "max_steps":
            return "boss_max_steps", {"max_pending_span": long_pending}
        return "boss_unresolved", {"final_floor": final_floor}
    if outcome == "death" and final_floor <= early_floor_threshold:
        return "early_death", {"final_floor": final_floor}
    if outcome == "death":
        return "preboss_death", {"final_floor": final_floor}
    if end_reason == "max_steps":
        return "max_steps_nonboss", {"max_pending_span": long_pending}
    return "other", {"final_floor": final_floor}


def _build_report(
    summaries: list[dict[str, Any]],
    *,
    early_floor_threshold: int,
    pending_stall_threshold: int,
    top_k: int,
) -> dict[str, Any]:
    resolver = SkadaNameResolver()
    counts = Counter()
    per_class_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    death_enemy_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    boss_enemy_by_class: dict[str, Counter[str]] = defaultdict(Counter)

    for summary in summaries:
        label, extra = _classify_episode(
            summary,
            early_floor_threshold=early_floor_threshold,
            pending_stall_threshold=pending_stall_threshold,
        )
        counts[label] += 1
        death_enemy = str(summary.get("death_enemy") or "")
        if death_enemy:
            death_enemy_by_class[label][death_enemy] += 1
        for combat in _boss_combats(summary):
            enemy_group = str(combat.get("enemy_group") or "UNKNOWN")
            boss_enemy_by_class[label][enemy_group] += 1

        if len(per_class_examples[label]) < top_k:
            per_class_examples[label].append(
                {
                    "iteration": int(summary.get("iteration", -1)),
                    "episode": int(summary.get("episode", -1)),
                    "final_floor": int(summary.get("final_floor", 0) or 0),
                    "outcome": str(summary.get("outcome") or "unknown"),
                    "end_reason": str(summary.get("end_reason") or "unknown"),
                    "repeat_max": int(summary.get("repeat_max", 0) or 0),
                    "death_enemy": death_enemy,
                    "death_enemy_display": resolver.enemy_group(death_enemy) if death_enemy else "",
                    "trace_path": str(summary.get("trace_path") or ""),
                    "summary_path": str(summary.get("_path") or ""),
                    **extra,
                }
            )

    classes = []
    for name, count in counts.most_common():
        classes.append(
            {
                "name": name,
                "count": count,
                "death_enemy_top": [
                    {
                        "name": enemy,
                        "display": resolver.enemy_group(enemy),
                        "count": enemy_count,
                    }
                    for enemy, enemy_count in death_enemy_by_class[name].most_common(top_k)
                ],
                "boss_enemy_top": [
                    {
                        "name": enemy,
                        "display": resolver.enemy_group(enemy),
                        "count": enemy_count,
                    }
                    for enemy, enemy_count in boss_enemy_by_class[name].most_common(top_k)
                ],
                "examples": per_class_examples[name],
            }
        )

    return {
        "episode_count": len(summaries),
        "class_counts": dict(counts),
        "classes": classes,
        "config": {
            "early_floor_threshold": early_floor_threshold,
            "pending_stall_threshold": pending_stall_threshold,
        },
    }


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Episode Case 分型")
    lines.append("")
    lines.append(f"- 样本局数: `{report.get('episode_count', 0)}`")
    lines.append(f"- 分型分布: `{json.dumps(report.get('class_counts', {}), ensure_ascii=False)}`")
    lines.append("")

    for cls in report.get("classes") or []:
        lines.append(f"## {cls['name']}")
        lines.append(f"- 数量: `{cls['count']}`")
        death_top = cls.get("death_enemy_top") or []
        boss_top = cls.get("boss_enemy_top") or []
        if death_top:
            lines.append("- 死亡敌人 Top:")
            for item in death_top:
                lines.append(f"  - `{item['display']}`: {item['count']}")
        if boss_top:
            lines.append("- Boss 遭遇 Top:")
            for item in boss_top:
                lines.append(f"  - `{item['display']}`: {item['count']}")
        examples = cls.get("examples") or []
        if examples:
            lines.append("- 样例:")
            for sample in examples:
                trace = sample.get("trace_path") or ""
                trace_link = f"[replay](/" + trace.replace("\\", "/") + ")" if trace else ""
                lines.append(
                    f"  - iter `{sample['iteration']}` ep `{sample['episode']}` floor `{sample['final_floor']}` "
                    f"outcome `{sample['outcome']}` end `{sample['end_reason']}` "
                    f"{trace_link}".rstrip()
                )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="基于 *.summary.json 对 episode 做失败/异常 case 分型。")
    parser.add_argument("training_dir", help="训练输出目录")
    parser.add_argument("--iter-start", type=int, default=None, help="起始 iteration")
    parser.add_argument("--iter-end", type=int, default=None, help="结束 iteration")
    parser.add_argument("--early-floor-threshold", type=int, default=8, help="多少层以下算 early_death")
    parser.add_argument("--pending-stall-threshold", type=int, default=20, help="combat_pending 连续多少步算 stall")
    parser.add_argument("--top-k", type=int, default=8, help="每类保留多少样例")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    analysis_dir = training_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summaries = _load_summaries(training_dir, args.iter_start, args.iter_end)
    if not summaries:
        raise SystemExit("未找到可用的 *.summary.json。")

    report = _build_report(
        summaries,
        early_floor_threshold=args.early_floor_threshold,
        pending_stall_threshold=args.pending_stall_threshold,
        top_k=args.top_k,
    )

    suffix = []
    if args.iter_start is not None:
        suffix.append(f"from_{args.iter_start:05d}")
    if args.iter_end is not None:
        suffix.append(f"to_{args.iter_end:05d}")
    name = "episode_case_classification" + ("_" + "_".join(suffix) if suffix else "")

    json_path = analysis_dir / f"{name}.json"
    md_path = analysis_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
