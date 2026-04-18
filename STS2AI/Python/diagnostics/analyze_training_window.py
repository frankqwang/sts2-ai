"""训练窗口分析：分析特定训练窗口的指标趋势。"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from analyze_iteration_replays import SkadaNameResolver


def _pct(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return median(values) if values else 0.0


def _top(counter: Counter[str], resolver, category: str, limit: int = 12) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name, count in counter.most_common(limit):
        if category == "card":
            display = resolver.card(name)
        elif category == "encounter":
            display = resolver.enemy_group(name) if "+" in name else resolver.encounter(name)
        else:
            display = resolver.generic(name)
        items.append({"name": name, "display": display, "count": count})
    return items


def _combat_entries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("combat_details", "combat_records", "combat_rooms", "combat_entries"):
        value = summary.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _shop_offer_affordable(session: dict[str, Any]) -> bool:
    enter_gold = int(session.get("enter_gold", 0) or 0)
    offers = session.get("offers") or []
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        if not bool(offer.get("is_stocked", True)):
            continue
        cost = int(offer.get("cost", 0) or 0)
        can_afford = offer.get("can_afford")
        if can_afford is True or enter_gold >= max(0, cost):
            return True
    return False


def _boss_episode_end_reason(summary: dict[str, Any]) -> str:
    if bool(summary.get("act1_cleared")) or str(summary.get("outcome") or "") == "victory":
        return "boss_clear"
    end_reason = str(summary.get("end_reason") or "unknown")
    if end_reason == "combat_pending_stall":
        return "boss_stall"
    if end_reason == "max_steps":
        return "boss_max_steps"
    if str(summary.get("outcome") or "") == "death":
        return "boss_loss"
    return "boss_unresolved"


def _load_metrics(training_dir: Path, iter_start: int | None, iter_end: int | None) -> list[dict[str, Any]]:
    metrics_path = training_dir / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if not metrics_path.exists():
        return rows
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        iteration = int(row.get("iteration", -1))
        if iter_start is not None and iteration < iter_start:
            continue
        if iter_end is not None and iteration > iter_end:
            continue
        rows.append(row)
    return rows


def _load_summaries(training_dir: Path, iter_start: int | None, iter_end: int | None) -> list[dict[str, Any]]:
    replay_dir = training_dir / "replays"
    summaries: list[dict[str, Any]] = []
    for path in sorted(replay_dir.glob("*.summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        iteration = int(payload.get("iteration", -1))
        if iter_start is not None and iteration < iter_start:
            continue
        if iter_end is not None and iteration > iter_end:
            continue
        payload["_path"] = str(path.resolve())
        summaries.append(payload)
    return summaries


def _build_report(
    training_dir: Path,
    metrics_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    pending_stall_threshold: int,
    top_k: int,
) -> dict[str, Any]:
    resolver = SkadaNameResolver()
    iterations = [int(row.get("iteration", -1)) for row in metrics_rows]
    floors = [float(summary.get("final_floor", 0) or 0) for summary in summaries]
    outcomes = Counter(str(summary.get("outcome") or "unknown") for summary in summaries)
    end_reasons = Counter(str(summary.get("end_reason") or "unknown") for summary in summaries)
    death_enemies = Counter(str(summary.get("death_enemy") or "UNKNOWN") for summary in summaries if summary.get("death_enemy"))
    route_counts = Counter()
    early_route_counts = Counter()
    event_counts = Counter()
    card_pick_counts = Counter()
    auto_action_counts = Counter()
    shop_action_counts = Counter()
    rest_action_counts = Counter()
    room_type_counts = Counter()
    boss_end_reasons = Counter()
    boss_enemy_groups = Counter()
    suspicious_cases: list[dict[str, Any]] = []
    stall_cases: list[dict[str, Any]] = []
    empty_shop_count = 0
    affordable_shop_skip_count = 0
    unaffordable_empty_shop_count = 0
    shop_visit_count = 0
    card_reward_total = 0
    card_reward_skip_count = 0
    map_override_steps = 0
    wait_steps_total = 0
    combat_pending_steps_total = 0
    combat_pending_wait_steps_total = 0
    repeat_max_values: list[int] = []

    for summary in summaries:
        counters = summary.get("counters") or {}
        wait_steps_total += int(counters.get("wait_steps", 0) or 0)
        combat_pending_steps_total += int(counters.get("combat_pending_steps", 0) or 0)
        combat_pending_wait_steps_total += int(counters.get("combat_pending_wait_steps", 0) or 0)
        map_override_steps += int(counters.get("map_override_steps", 0) or 0)
        repeat_max_values.append(int(summary.get("repeat_max", 0) or 0))

        for choice in summary.get("route_choices") or []:
            label = str(choice.get("node_type") or choice.get("label") or "unknown")
            route_counts[label] += 1
            if int(choice.get("floor", 0) or 0) <= 8:
                early_route_counts[label] += 1

        for event in summary.get("event_choices") or []:
            event_id = str(event.get("event_id") or event.get("label") or "unknown")
            event_counts[event_id] += 1

        for reward in summary.get("card_rewards") or []:
            card_reward_total += 1
            if bool(reward.get("skipped")):
                card_reward_skip_count += 1
            else:
                card_pick_counts[str(reward.get("label") or "unknown")] += 1

        for session in summary.get("shop_sessions") or []:
            shop_visit_count += 1
            actions = session.get("actions") or []
            only_proceed = len(actions) == 1 and str(actions[0].get("action") or "") == "proceed"
            if only_proceed:
                empty_shop_count += 1
                if _shop_offer_affordable(session):
                    affordable_shop_skip_count += 1
                else:
                    unaffordable_empty_shop_count += 1
            for action in actions:
                shop_action_counts[str(action.get("action") or action.get("label") or "unknown")] += 1

        for session in summary.get("rest_sessions") or []:
            for action in session.get("actions") or []:
                rest_action_counts[str(action.get("action") or action.get("label") or "unknown")] += 1

        for action in summary.get("auto_actions") or []:
            auto_action_counts[str(action.get("kind") or action.get("action") or "unknown")] += 1

        pending_spans = summary.get("combat_pending_spans") or []
        long_spans = [span for span in pending_spans if int(span.get("count", 0) or 0) >= pending_stall_threshold]
        if long_spans:
            stall_cases.append(
                {
                    "iteration": int(summary.get("iteration", -1)),
                    "episode": int(summary.get("episode", -1)),
                    "floor": int(summary.get("final_floor", 0) or 0),
                    "end_reason": str(summary.get("end_reason") or "unknown"),
                    "max_span": max(int(span.get("count", 0) or 0) for span in long_spans),
                    "summary_path": summary.get("_path", ""),
                    "trace_path": summary.get("trace_path", ""),
                }
            )

        combat_entries = _combat_entries(summary)
        for combat in combat_entries:
            room_type = str(combat.get("room_type") or "unknown")
            room_type_counts[room_type] += 1
            if room_type == "boss":
                boss_end_reasons[str(combat.get("end_reason") or "unknown")] += 1
                boss_enemy_groups[str(combat.get("enemy_group") or "UNKNOWN")] += 1

        if not combat_entries and bool(summary.get("boss_reached")):
            room_type_counts["boss_episode"] += 1
            boss_end_reasons[_boss_episode_end_reason(summary)] += 1

        if (
            str(summary.get("end_reason") or "") in {"max_steps", "repeat_loop", "timeout", "combat_pending_stall"}
            or long_spans
            or int(summary.get("repeat_max", 0) or 0) >= 3
        ):
            suspicious_cases.append(
                {
                    "iteration": int(summary.get("iteration", -1)),
                    "episode": int(summary.get("episode", -1)),
                    "floor": int(summary.get("final_floor", 0) or 0),
                    "outcome": str(summary.get("outcome") or "unknown"),
                    "end_reason": str(summary.get("end_reason") or "unknown"),
                    "repeat_max": int(summary.get("repeat_max", 0) or 0),
                    "combat_pending_steps": int(counters.get("combat_pending_steps", 0) or 0),
                    "wait_steps": int(counters.get("wait_steps", 0) or 0),
                    "summary_path": summary.get("_path", ""),
                    "trace_path": summary.get("trace_path", ""),
                }
            )

    metric_summary = {}
    if metrics_rows:
        metric_summary = {
            "iteration_start": min(iterations),
            "iteration_end": max(iterations),
            "iteration_count": len(metrics_rows),
            "avg_floor_mean": round(_mean([float(row.get("avg_floor", 0) or 0) for row in metrics_rows]), 4),
            "boss_reach_rate_mean": round(_mean([float(row.get("boss_reach_rate", 0) or 0) for row in metrics_rows]), 4),
            "act1_clear_rate_mean": round(_mean([float(row.get("act1_clear_rate", 0) or 0) for row in metrics_rows]), 4),
            "boss_hp_fraction_mean": round(_mean([float(row.get("boss_hp_fraction_dealt_mean", 0) or 0) for row in metrics_rows]), 4),
            "card_reward_skip_rate_mean": round(_mean([float(row.get("card_reward_skip_rate", 0) or 0) for row in metrics_rows]), 4),
            "ppo_vloss_mean": round(_mean([float(row.get("ppo_vloss", 0) or 0) for row in metrics_rows]), 4),
            "combat_vloss_mean": round(_mean([float(row.get("combat_ppo_vloss", 0) or 0) for row in metrics_rows]), 4),
        }

    return {
        "training_dir": str(training_dir.resolve()),
        "episode_count": len(summaries),
        "metrics_window": metric_summary,
        "episodes": {
            "avg_floor": round(_mean(floors), 4),
            "median_floor": round(_median(floors), 4),
            "outcomes": dict(sorted(outcomes.items())),
            "end_reasons": dict(sorted(end_reasons.items())),
            "repeat_max_mean": round(_mean([float(v) for v in repeat_max_values]), 4),
        },
        "route": {
            "map_override_steps": map_override_steps,
            "map_override_rate": round(_pct(map_override_steps, sum(route_counts.values())), 4),
            "route_top": _top(route_counts, resolver, "generic", top_k),
            "early_route_top": _top(early_route_counts, resolver, "generic", top_k),
            "event_top": _top(event_counts, resolver, "generic", top_k),
        },
        "rewards": {
            "card_reward_total": card_reward_total,
            "card_reward_skip_count": card_reward_skip_count,
            "card_reward_skip_rate": round(_pct(card_reward_skip_count, card_reward_total), 4),
            "card_pick_top": _top(card_pick_counts, resolver, "card", top_k),
        },
        "shop": {
            "shop_visit_count": shop_visit_count,
            "empty_shop_count": empty_shop_count,
            "empty_shop_rate": round(_pct(empty_shop_count, shop_visit_count), 4),
            "affordable_shop_skip_count": affordable_shop_skip_count,
            "affordable_shop_skip_rate": round(_pct(affordable_shop_skip_count, shop_visit_count), 4),
            "unaffordable_empty_shop_count": unaffordable_empty_shop_count,
            "shop_action_top": _top(shop_action_counts, resolver, "generic", top_k),
            "rest_action_top": _top(rest_action_counts, resolver, "generic", top_k),
        },
        "combat": {
            "room_type_top": _top(room_type_counts, resolver, "generic", top_k),
            "death_enemy_top": _top(death_enemies, resolver, "encounter", top_k),
            "boss_end_reasons": dict(sorted(boss_end_reasons.items())),
            "boss_enemy_groups": _top(boss_enemy_groups, resolver, "encounter", top_k),
            "wait_steps_total": wait_steps_total,
            "combat_pending_steps_total": combat_pending_steps_total,
            "combat_pending_wait_steps_total": combat_pending_wait_steps_total,
            "stall_case_count": len(stall_cases),
            "stall_cases": stall_cases[:top_k],
        },
        "auto_actions": {
            "top": _top(auto_action_counts, resolver, "generic", top_k),
        },
        "suspicious_cases": suspicious_cases[:top_k],
    }


def _format_top(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 无"]
    return [f"- `{item['display']}`: {item['count']}" for item in items]


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    metrics = report.get("metrics_window") or {}
    episodes = report.get("episodes") or {}
    route = report.get("route") or {}
    rewards = report.get("rewards") or {}
    shop = report.get("shop") or {}
    combat = report.get("combat") or {}
    auto_actions = report.get("auto_actions") or {}

    lines.append("# 训练窗口全量分析")
    lines.append("")
    lines.append(f"- 样本局数: `{report.get('episode_count', 0)}`")
    if metrics:
        lines.append(
            f"- iteration 范围: `{metrics.get('iteration_start')}` - `{metrics.get('iteration_end')}` "
            f"(`{metrics.get('iteration_count')}` 轮)"
        )
        lines.append(f"- `avg_floor` 均值: `{metrics.get('avg_floor_mean', 0):.4f}`")
        lines.append(f"- `boss_reach_rate` 均值: `{metrics.get('boss_reach_rate_mean', 0) * 100:.2f}%`")
        lines.append(f"- `act1_clear_rate` 均值: `{metrics.get('act1_clear_rate_mean', 0) * 100:.2f}%`")
        lines.append(f"- `boss_hp_fraction_dealt_mean` 均值: `{metrics.get('boss_hp_fraction_mean', 0) * 100:.2f}%`")
        lines.append(f"- `ppo_vloss` 均值: `{metrics.get('ppo_vloss_mean', 0):.4f}`")
    lines.append("")

    lines.append("## Episode 总览")
    lines.append(f"- 平均层数: `{episodes.get('avg_floor', 0):.4f}`")
    lines.append(f"- 中位层数: `{episodes.get('median_floor', 0):.4f}`")
    lines.append(f"- 结局分布: `{json.dumps(episodes.get('outcomes', {}), ensure_ascii=False)}`")
    lines.append(f"- 结束原因: `{json.dumps(episodes.get('end_reasons', {}), ensure_ascii=False)}`")
    lines.append(f"- `repeat_max` 均值: `{episodes.get('repeat_max_mean', 0):.4f}`")
    lines.append("")

    lines.append("## 路线 / 非战斗")
    lines.append(
        f"- 地图 override 次数: `{route.get('map_override_steps', 0)}` "
        f"({route.get('map_override_rate', 0) * 100:.2f}%)"
    )
    lines.append("### 路线 Top")
    lines.extend(_format_top(route.get("route_top") or []))
    lines.append("")
    lines.append("### 前 8 层路线 Top")
    lines.extend(_format_top(route.get("early_route_top") or []))
    lines.append("")
    lines.append("### 事件 Top")
    lines.extend(_format_top(route.get("event_top") or []))
    lines.append("")

    lines.append("## 卡奖 / 商店")
    lines.append(
        f"- 卡奖总数: `{rewards.get('card_reward_total', 0)}`，"
        f"跳过 `{rewards.get('card_reward_skip_count', 0)}` "
        f"({rewards.get('card_reward_skip_rate', 0) * 100:.2f}%)"
    )
    lines.append("### 选牌 Top")
    lines.extend(_format_top(rewards.get("card_pick_top") or []))
    lines.append("")
    lines.append(
        f"- 进店次数: `{shop.get('shop_visit_count', 0)}`，"
        f"空店直接离开 `{shop.get('empty_shop_count', 0)}` "
        f"({shop.get('empty_shop_rate', 0) * 100:.2f}%)"
    )
    lines.append(
        f"- `affordable_shop_skip`: `{shop.get('affordable_shop_skip_count', 0)}` "
        f"({shop.get('affordable_shop_skip_rate', 0) * 100:.2f}%)；"
        f"`unaffordable_empty_shop`: `{shop.get('unaffordable_empty_shop_count', 0)}`"
    )
    lines.append("### 商店动作 Top")
    lines.extend(_format_top(shop.get("shop_action_top") or []))
    lines.append("")
    lines.append("### 火堆动作 Top")
    lines.extend(_format_top(shop.get("rest_action_top") or []))
    lines.append("")

    lines.append("## 战斗 / Stall")
    lines.append(f"- `wait_steps_total`: `{combat.get('wait_steps_total', 0)}`")
    lines.append(f"- `combat_pending_steps_total`: `{combat.get('combat_pending_steps_total', 0)}`")
    lines.append(f"- `combat_pending_wait_steps_total`: `{combat.get('combat_pending_wait_steps_total', 0)}`")
    lines.append(f"- stall case 数: `{combat.get('stall_case_count', 0)}`")
    lines.append("### 房间类型 Top")
    lines.extend(_format_top(combat.get("room_type_top") or []))
    lines.append("")
    lines.append("### 死亡敌人 Top")
    lines.extend(_format_top(combat.get("death_enemy_top") or []))
    lines.append("")
    lines.append(f"- Boss 结束原因: `{json.dumps(combat.get('boss_end_reasons', {}), ensure_ascii=False)}`")
    lines.append("### Boss 遭遇 Top")
    lines.extend(_format_top(combat.get("boss_enemy_groups") or []))
    lines.append("")

    lines.append("## Auto Actions")
    lines.extend(_format_top(auto_actions.get("top") or []))
    lines.append("")

    lines.append("## 可疑 Case")
    suspicious = report.get("suspicious_cases") or []
    if suspicious:
        for case in suspicious:
            trace_path = str(case.get("trace_path") or "")
            trace_link = f"[replay](/" + trace_path.replace("\\", "/") + ")" if trace_path else ""
            lines.append(
                f"- iter `{case['iteration']}` ep `{case['episode']}` floor `{case['floor']}` "
                f"outcome `{case['outcome']}` end `{case['end_reason']}` repeat `{case['repeat_max']}` "
                f"pending `{case['combat_pending_steps']}` wait `{case['wait_steps']}` {trace_link}".rstrip()
            )
    else:
        lines.append("- 无")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="按 iteration 窗口聚合训练 replay summary，输出问题分析报告。")
    parser.add_argument("training_dir", help="训练输出目录")
    parser.add_argument("--iter-start", type=int, default=None, help="起始 iteration")
    parser.add_argument("--iter-end", type=int, default=None, help="结束 iteration")
    parser.add_argument("--top-k", type=int, default=12, help="Top 列表保留数量")
    parser.add_argument("--pending-stall-threshold", type=int, default=20, help="combat_pending 连续多少步算 stall")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    analysis_dir = training_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = _load_metrics(training_dir, args.iter_start, args.iter_end)
    summaries = _load_summaries(training_dir, args.iter_start, args.iter_end)
    if not summaries:
        raise SystemExit("未找到可用的 *.summary.json，先用新版本训练落结构化 replay。")

    report = _build_report(
        training_dir,
        metrics_rows,
        summaries,
        pending_stall_threshold=args.pending_stall_threshold,
        top_k=args.top_k,
    )

    suffix = []
    if args.iter_start is not None:
        suffix.append(f"from_{args.iter_start:05d}")
    if args.iter_end is not None:
        suffix.append(f"to_{args.iter_end:05d}")
    name = "training_window" + ("_" + "_".join(suffix) if suffix else "")

    json_path = analysis_dir / f"{name}.json"
    md_path = analysis_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
