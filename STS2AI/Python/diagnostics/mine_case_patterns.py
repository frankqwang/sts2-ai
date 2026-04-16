"""Mine per-combat case patterns (damage, block, HP loss) and correlate with episode outcomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analyze_iteration_replays import SkadaNameResolver
from analyze_training_window import _combat_entries, _load_summaries, _shop_offer_affordable
from classify_episode_failures import _classify_episode


def _enemy_damage(action: dict[str, Any]) -> int:
    total = 0
    result = action.get("result") or {}
    for item in result.get("enemy_changes") or []:
        pre_hp = int(item.get("pre_hp", 0) or 0)
        post_hp = int(item.get("post_hp", 0) or 0)
        total += max(0, pre_hp - post_hp)
    return total


def _block_gain(action: dict[str, Any]) -> int:
    result = action.get("result") or {}
    return int(result.get("post_block", 0) or 0) - int(result.get("pre_block", 0) or 0)


def _hp_ratio(action: dict[str, Any], combat: dict[str, Any]) -> float:
    result = action.get("result") or {}
    pre_hp = float(result.get("pre_hp", 0) or 0)
    start_max_hp = float(combat.get("start_max_hp", 0) or 0)
    return pre_hp / max(1.0, start_max_hp)


def _route_prefix(summary: dict[str, Any], n: int = 5) -> tuple[str, ...]:
    out: list[str] = []
    for choice in summary.get("route_choices") or []:
        if len(out) >= n:
            break
        out.append(str(choice.get("node_type") or choice.get("label") or "unknown").strip().lower())
    return tuple(out)


def _shop_summary(summary: dict[str, Any]) -> dict[str, float]:
    visits = 0
    empty = 0
    affordable_skip = 0
    remove = 0
    remove_target = 0
    for session in summary.get("shop_sessions") or []:
        visits += 1
        actions = session.get("actions") or []
        non_proceed = [a for a in actions if str(a.get("action") or "").strip().lower() != "proceed"]
        if not non_proceed:
            empty += 1
            if _shop_offer_affordable(session):
                affordable_skip += 1
        for action in actions:
            name = str(action.get("action") or "").strip().lower()
            label = str(action.get("label") or "").strip().lower()
            if name == "remove_card" or (name == "shop_purchase" and label == "remove_card"):
                remove += 1
            elif name == "remove_target":
                remove_target += 1
    return {
        "visits": visits,
        "empty": empty,
        "affordable_skip": affordable_skip,
        "remove": remove,
        "remove_target": remove_target,
    }


def _register_case(store: dict[str, list[dict[str, Any]]], key: str, item: dict[str, Any], *, limit: int) -> None:
    bucket = store.setdefault(key, [])
    if len(bucket) < limit:
        bucket.append(item)


def _build_report(
    summaries: list[dict[str, Any]],
    *,
    top_k: int,
    low_hp_threshold: float,
) -> dict[str, Any]:
    resolver = SkadaNameResolver()
    pattern_counts: Counter[str] = Counter()
    pattern_examples: dict[str, list[dict[str, Any]]] = {}
    class_counts: Counter[str] = Counter()
    route_counter_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    killer_counter: Counter[str] = Counter()
    boss_hp_loss_ratios: list[float] = []

    for summary in summaries:
        label, _ = _classify_episode(summary, early_floor_threshold=8, pending_stall_threshold=20)
        class_counts[label] += 1
        route_counter_by_class[label][" -> ".join(_route_prefix(summary, 5)) or "无路线"] += 1

        combats = _combat_entries(summary)
        shop_info = _shop_summary(summary)
        if label == "boss_loss":
            boss = [c for c in combats if str(c.get("room_type") or "").strip().lower() == "boss"]
            if boss:
                start_hp = float(boss[-1].get("start_hp", 0) or 0)
                start_max = float(boss[-1].get("start_max_hp", 0) or 0)
                ratio = start_hp / max(1.0, start_max)
                boss_hp_loss_ratios.append(ratio)
                if ratio >= 0.9:
                    pattern_counts["high_hp_boss_loss"] += 1
                    _register_case(
                        pattern_examples,
                        "high_hp_boss_loss",
                        {
                            "iteration": int(summary.get("iteration", -1)),
                            "episode": int(summary.get("episode", -1)),
                            "floor": int(summary.get("final_floor", 0) or 0),
                            "boss_enemy": str(boss[-1].get("enemy_group") or "UNKNOWN"),
                            "boss_start_hp": int(start_hp),
                            "boss_start_max_hp": int(start_max),
                            "trace_zh_path": str(summary.get("trace_zh_path") or ""),
                        },
                        limit=top_k,
                    )

        if label == "early_death":
            prefix = _route_prefix(summary, 4)
            if prefix and all(item == "monster" for item in prefix):
                pattern_counts["early_death_all_monster_route"] += 1
                _register_case(
                    pattern_examples,
                    "early_death_all_monster_route",
                    {
                        "iteration": int(summary.get("iteration", -1)),
                        "episode": int(summary.get("episode", -1)),
                        "floor": int(summary.get("final_floor", 0) or 0),
                        "cards_taken": list(summary.get("cards_taken") or []),
                        "trace_zh_path": str(summary.get("trace_zh_path") or ""),
                    },
                    limit=top_k,
                )

        if label in {"boss_loss", "preboss_death"} and shop_info["visits"] > 0 and shop_info["empty"] == shop_info["visits"]:
            pattern_counts["empty_shop_then_die"] += 1
            _register_case(
                pattern_examples,
                "empty_shop_then_die",
                {
                    "iteration": int(summary.get("iteration", -1)),
                    "episode": int(summary.get("episode", -1)),
                    "floor": int(summary.get("final_floor", 0) or 0),
                    "class": label,
                    "shop_visits": int(shop_info["visits"]),
                    "trace_zh_path": str(summary.get("trace_zh_path") or ""),
                },
                    limit=top_k,
                )

        if label in {"boss_loss", "preboss_death"} and shop_info["affordable_skip"] > 0:
            pattern_counts["affordable_shop_skip"] += 1
            _register_case(
                pattern_examples,
                "affordable_shop_skip",
                {
                    "iteration": int(summary.get("iteration", -1)),
                    "episode": int(summary.get("episode", -1)),
                    "floor": int(summary.get("final_floor", 0) or 0),
                    "class": label,
                    "affordable_shop_skip": int(shop_info["affordable_skip"]),
                    "trace_zh_path": str(summary.get("trace_zh_path") or ""),
                },
                limit=top_k,
            )

        if label == "act1_clear" and int(summary.get("final_floor", 0) or 0) == 18:
            pattern_counts["clear_then_next_room_die"] += 1
            _register_case(
                pattern_examples,
                "clear_then_next_room_die",
                {
                    "iteration": int(summary.get("iteration", -1)),
                    "episode": int(summary.get("episode", -1)),
                    "floor": int(summary.get("final_floor", 0) or 0),
                    "trace_zh_path": str(summary.get("trace_zh_path") or ""),
                },
                limit=top_k,
            )

        for combat in combats:
            room_type = str(combat.get("room_type") or "").strip().lower()
            enemy_group = str(combat.get("enemy_group") or "UNKNOWN")
            if label in {"early_death", "preboss_death"} and room_type in {"monster", "elite"} and combat.get("won") is False:
                killer_counter[enemy_group] += 1

            for action in combat.get("actions") or []:
                result = action.get("result") or {}
                pre_intent_zh = str(result.get("pre_intent_zh") or "")
                damage = _enemy_damage(action)
                block_gain = _block_gain(action)
                ratio = _hp_ratio(action, combat)
                action_name = str(action.get("action_name") or "").strip().lower()
                action_label_zh = str(action.get("action_label_zh") or "")

                if (
                    ratio <= low_hp_threshold
                    and damage <= 0
                    and block_gain <= 0
                    and action_name not in {"end_turn", "use_potion"}
                ):
                    pattern_counts["low_hp_setup_action"] += 1
                    _register_case(
                        pattern_examples,
                        "low_hp_setup_action",
                        {
                            "iteration": int(summary.get("iteration", -1)),
                            "episode": int(summary.get("episode", -1)),
                            "floor": int(combat.get("floor", 0) or 0),
                            "class": label,
                            "pre_hp_ratio": round(ratio, 4),
                            "action_label_zh": action_label_zh,
                            "pre_intent_zh": pre_intent_zh,
                            "trace_zh_path": str(summary.get("trace_zh_path") or ""),
                        },
                        limit=top_k,
                    )

                if "意图 Buff" in pre_intent_zh and damage <= 0 and action_name not in {"use_potion"}:
                    pattern_counts["buff_window_no_pressure"] += 1
                    _register_case(
                        pattern_examples,
                        "buff_window_no_pressure",
                        {
                            "iteration": int(summary.get("iteration", -1)),
                            "episode": int(summary.get("episode", -1)),
                            "floor": int(combat.get("floor", 0) or 0),
                            "class": label,
                            "action_label_zh": action_label_zh,
                            "pre_intent_zh": pre_intent_zh,
                            "trace_zh_path": str(summary.get("trace_zh_path") or ""),
                        },
                        limit=top_k,
                    )

                enemy_changes = result.get("enemy_changes") or []
                spike = any(
                    int(change.get("post_hp", 0) or 0) >= 999999
                    or int(change.get("post_hp", 0) or 0) > int(change.get("pre_hp", 0) or 0) + 100
                    for change in enemy_changes
                )
                if spike:
                    pattern_counts["enemy_hp_spike_anomaly"] += 1
                    _register_case(
                        pattern_examples,
                        "enemy_hp_spike_anomaly",
                        {
                            "iteration": int(summary.get("iteration", -1)),
                            "episode": int(summary.get("episode", -1)),
                            "floor": int(combat.get("floor", 0) or 0),
                            "class": label,
                            "action_label_zh": action_label_zh,
                            "enemy_changes": enemy_changes,
                            "trace_zh_path": str(summary.get("trace_zh_path") or ""),
                        },
                        limit=top_k,
                    )

                if action_name == "end_turn" and int(result.get("post_hp", 0) or 0) <= 0:
                    defeated = [item for item in enemy_changes if bool(item.get("defeated"))]
                    if defeated:
                        pattern_counts["death_on_end_turn_with_defeat"] += 1
                        _register_case(
                            pattern_examples,
                            "death_on_end_turn_with_defeat",
                            {
                                "iteration": int(summary.get("iteration", -1)),
                                "episode": int(summary.get("episode", -1)),
                                "floor": int(combat.get("floor", 0) or 0),
                                "class": label,
                                "pre_intent_zh": pre_intent_zh,
                                "defeated": defeated,
                                "trace_zh_path": str(summary.get("trace_zh_path") or ""),
                            },
                            limit=top_k,
                        )

    return {
        "episode_count": len(summaries),
        "class_counts": dict(class_counts),
        "pattern_counts": dict(pattern_counts),
        "avg_boss_loss_entry_hp_ratio": round(mean(boss_hp_loss_ratios), 4) if boss_hp_loss_ratios else 0.0,
        "preboss_killer_top": [
            {
                "name": name,
                "display": resolver.enemy_group(name),
                "count": count,
            }
            for name, count in killer_counter.most_common(top_k)
        ],
        "route_top_by_class": {
            cls: [{"name": name, "count": count} for name, count in counter.most_common(top_k)]
            for cls, counter in route_counter_by_class.items()
        },
        "pattern_examples": pattern_examples,
    }


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# 代表局模式挖掘")
    lines.append("")
    lines.append(f"- 样本局数: `{report.get('episode_count', 0)}`")
    lines.append(f"- 分型分布: `{json.dumps(report.get('class_counts', {}), ensure_ascii=False)}`")
    lines.append(f"- Boss 失败入场 HP 比例均值: `{report.get('avg_boss_loss_entry_hp_ratio', 0):.4f}`")
    lines.append("")

    lines.append("## 模式计数")
    for name, count in (report.get("pattern_counts") or {}).items():
        lines.append(f"- `{name}`: {count}")
    lines.append("")

    lines.append("## boss 前死亡杀手 Top")
    killers = report.get("preboss_killer_top") or []
    if not killers:
        lines.append("- 无")
    else:
        for item in killers:
            lines.append(f"- `{item.get('display')}`: {item.get('count', 0)}")
    lines.append("")

    for cls, routes in (report.get("route_top_by_class") or {}).items():
        lines.append(f"## {cls} 路线 Top")
        if not routes:
            lines.append("- 无")
        else:
            for item in routes:
                lines.append(f"- `{item.get('name')}`: {item.get('count', 0)}")
        lines.append("")

    for name, examples in (report.get("pattern_examples") or {}).items():
        lines.append(f"## {name} 代表局")
        if not examples:
            lines.append("- 无")
        else:
            for item in examples:
                trace = str(item.get("trace_zh_path") or "")
                trace_link = f"[回放](/" + trace.replace("\\", "/") + ")" if trace else ""
                lines.append(
                    f"- iter `{item.get('iteration')}` ep `{item.get('episode')}` floor `{item.get('floor')}` "
                    f"class `{item.get('class', '')}` action `{item.get('action_label_zh', '')}` "
                    f"{trace_link}".rstrip()
                )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="挖掘代表局模式与可疑失败路径。")
    parser.add_argument("training_dir", help="训练输出目录")
    parser.add_argument("--iter-start", type=int, default=None, help="起始 iteration")
    parser.add_argument("--iter-end", type=int, default=None, help="结束 iteration")
    parser.add_argument("--top-k", type=int, default=10, help="每类保留多少个 Top/代表局")
    parser.add_argument("--low-hp-threshold", type=float, default=0.25, help="低血阈值（相对最大生命）")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    analysis_dir = training_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summaries = _load_summaries(training_dir, args.iter_start, args.iter_end)
    if not summaries:
        raise SystemExit("未找到可用的 *.summary.json。")

    report = _build_report(summaries, top_k=args.top_k, low_hp_threshold=args.low_hp_threshold)

    suffix: list[str] = []
    if args.iter_start is not None:
        suffix.append(f"from_{args.iter_start:05d}")
    if args.iter_end is not None:
        suffix.append(f"to_{args.iter_end:05d}")
    name = "case_patterns" + ("_" + "_".join(suffix) if suffix else "")

    json_path = analysis_dir / f"{name}.json"
    md_path = analysis_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
