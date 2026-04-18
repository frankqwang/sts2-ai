"""路线和商店模式挖掘：分析路线选择和商店购买模式。"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analyze_iteration_replays import SkadaNameResolver
from analyze_training_window import _load_summaries, _shop_offer_affordable


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _split_groups(summaries: list[dict[str, Any]], top_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(summaries, key=lambda s: (float(s.get("final_floor", 0) or 0), bool(s.get("act1_cleared"))))
    if not ordered:
        return [], []
    n = max(1, int(len(ordered) * top_ratio))
    return ordered[-n:], ordered[:n]


def _route_signature(summary: dict[str, Any], *, first_n: int) -> str:
    parts: list[str] = []
    for choice in summary.get("route_choices") or []:
        if len(parts) >= first_n:
            break
        label = str(choice.get("node_type") or choice.get("label") or "unknown").strip().lower()
        parts.append(label or "unknown")
    return " -> ".join(parts) if parts else "无地图选择"


def _shop_profile(summary: dict[str, Any]) -> dict[str, int]:
    out = {
        "remove_card": 0,
        "buy_card": 0,
        "buy_potion": 0,
        "empty_shop": 0,
        "affordable_shop_skip": 0,
        "unaffordable_empty_shop": 0,
    }
    for session in summary.get("shop_sessions") or []:
        actions = session.get("actions") or []
        if len(actions) == 1 and str(actions[0].get("action") or "") == "proceed":
            out["empty_shop"] += 1
            if _shop_offer_affordable(session):
                out["affordable_shop_skip"] += 1
            else:
                out["unaffordable_empty_shop"] += 1
        for action in actions:
            action_name = str(action.get("action") or "").strip().lower()
            label = str(action.get("label") or "").strip().lower()
            if action_name == "remove_card" or (action_name == "shop_purchase" and label == "remove_card"):
                out["remove_card"] += 1
            elif "potion" in action_name or "potion" in label:
                out["buy_potion"] += 1
            elif action_name not in {"proceed", ""}:
                out["buy_card"] += 1
    return out


def _card_profile(summary: dict[str, Any], *, first_n: int) -> list[str]:
    picks: list[str] = []
    for reward in summary.get("card_rewards") or []:
        if bool(reward.get("skipped")):
            continue
        picks.append(str(reward.get("label") or "unknown"))
        if len(picks) >= first_n:
            break
    return picks


def _aggregate_group(summaries: list[dict[str, Any]], resolver: SkadaNameResolver, *, first_route_n: int, first_card_n: int, top_k: int) -> dict[str, Any]:
    route_counter = Counter()
    first_node_counter = Counter()
    event_counter = Counter()
    card_counter = Counter()
    first_card_combo_counter = Counter()
    shop_counter = Counter()
    empty_shop = 0
    shop_visits = 0
    skip_rates: list[float] = []
    map_override_steps = 0
    boss_reached = 0
    act1_cleared = 0
    final_floors = [float(summary.get("final_floor", 0) or 0) for summary in summaries]

    for summary in summaries:
        route_counter[_route_signature(summary, first_n=first_route_n)] += 1
        route_choices = summary.get("route_choices") or []
        if route_choices:
            first_label = str(route_choices[0].get("node_type") or route_choices[0].get("label") or "unknown")
            first_node_counter[first_label] += 1
        for event in summary.get("event_choices") or []:
            event_counter[str(event.get("event_id") or event.get("label") or "unknown")] += 1
        picks = _card_profile(summary, first_n=first_card_n)
        for pick in picks:
            card_counter[pick] += 1
        first_card_combo_counter[" + ".join(picks) if picks else "无前期拿牌"] += 1
        profile = _shop_profile(summary)
        for key, value in profile.items():
            shop_counter[key] += value
        empty_shop += profile["empty_shop"]
        shop_visits += len(summary.get("shop_sessions") or [])
        card_rewards = summary.get("card_rewards") or []
        if card_rewards:
            skip_rates.append(sum(1 for reward in card_rewards if reward.get("skipped")) / len(card_rewards))
        map_override_steps += int((summary.get("counters") or {}).get("map_override_steps", 0) or 0)
        boss_reached += int(bool(summary.get("boss_reached")))
        act1_cleared += int(bool(summary.get("act1_cleared")))

    return {
        "episode_count": len(summaries),
        "avg_floor": round(_mean(final_floors), 4),
        "boss_reach_rate": round(boss_reached / max(1, len(summaries)), 4),
        "act1_clear_rate": round(act1_cleared / max(1, len(summaries)), 4),
        "card_reward_skip_rate": round(_mean(skip_rates), 4),
        "map_override_steps": map_override_steps,
        "route_signature_top": [{"name": name, "count": count} for name, count in route_counter.most_common(top_k)],
        "first_node_top": [{"name": name, "display": resolver.generic(name), "count": count} for name, count in first_node_counter.most_common(top_k)],
        "event_top": [{"name": name, "display": resolver.generic(name), "count": count} for name, count in event_counter.most_common(top_k)],
        "card_pick_top": [{"name": name, "display": resolver.card(name), "count": count} for name, count in card_counter.most_common(top_k)],
        "first_card_combo_top": [{"name": name, "count": count} for name, count in first_card_combo_counter.most_common(top_k)],
        "shop_profile": {
            "shop_visits": shop_visits,
            "empty_shop": empty_shop,
            "empty_shop_rate": round(empty_shop / max(1, shop_visits), 4),
            "affordable_shop_skip": shop_counter["affordable_shop_skip"],
            "unaffordable_empty_shop": shop_counter["unaffordable_empty_shop"],
            "remove_card": shop_counter["remove_card"],
            "buy_card": shop_counter["buy_card"],
            "buy_potion": shop_counter["buy_potion"],
        },
    }


def _build_report(summaries: list[dict[str, Any]], *, top_ratio: float, first_route_n: int, first_card_n: int, top_k: int) -> dict[str, Any]:
    resolver = SkadaNameResolver()
    high_group, low_group = _split_groups(summaries, top_ratio)
    return {
        "episode_count": len(summaries),
        "config": {
            "top_ratio": top_ratio,
            "first_route_n": first_route_n,
            "first_card_n": first_card_n,
        },
        "high_floor_group": _aggregate_group(high_group, resolver, first_route_n=first_route_n, first_card_n=first_card_n, top_k=top_k),
        "low_floor_group": _aggregate_group(low_group, resolver, first_route_n=first_route_n, first_card_n=first_card_n, top_k=top_k),
    }


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    def _render_group(lines: list[str], title: str, group: dict[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append(f"- 局数: `{group.get('episode_count', 0)}`")
        lines.append(f"- 平均层数: `{group.get('avg_floor', 0):.4f}`")
        lines.append(f"- `boss_reach_rate`: `{group.get('boss_reach_rate', 0) * 100:.2f}%`")
        lines.append(f"- `act1_clear_rate`: `{group.get('act1_clear_rate', 0) * 100:.2f}%`")
        lines.append(f"- `card_reward_skip_rate`: `{group.get('card_reward_skip_rate', 0) * 100:.2f}%`")
        lines.append(f"- `map_override_steps`: `{group.get('map_override_steps', 0)}`")
        shop_profile = group.get("shop_profile") or {}
        lines.append(
            f"- 商店: visits `{shop_profile.get('shop_visits', 0)}` / empty `{shop_profile.get('empty_shop', 0)}` "
            f"({shop_profile.get('empty_shop_rate', 0) * 100:.2f}%) / remove `{shop_profile.get('remove_card', 0)}` "
            f"/ buy_card `{shop_profile.get('buy_card', 0)}` / buy_potion `{shop_profile.get('buy_potion', 0)}` "
            f"/ affordable_skip `{shop_profile.get('affordable_shop_skip', 0)}` "
            f"/ unaffordable_empty `{shop_profile.get('unaffordable_empty_shop', 0)}`"
        )
        for key, header in [
            ("route_signature_top", "路线签名 Top"),
            ("first_node_top", "首个地图节点 Top"),
            ("event_top", "事件 Top"),
            ("card_pick_top", "前期拿牌 Top"),
            ("first_card_combo_top", "前期拿牌组合 Top"),
        ]:
            lines.append(f"### {header}")
            items = group.get(key) or []
            if not items:
                lines.append("- 无")
            else:
                for item in items:
                    label = item.get("display") or item.get("name")
                    lines.append(f"- `{label}`: {item.get('count', 0)}")
            lines.append("")

    lines: list[str] = []
    lines.append("# 路线 / 商店相关性分析")
    lines.append("")
    lines.append(f"- 样本局数: `{report.get('episode_count', 0)}`")
    lines.append(f"- 配置: `{json.dumps(report.get('config', {}), ensure_ascii=False)}`")
    lines.append("")
    _render_group(lines, "高层组", report.get("high_floor_group") or {})
    _render_group(lines, "低层组", report.get("low_floor_group") or {})
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="分析路线、商店、卡奖与高低层结果的相关性。")
    parser.add_argument("training_dir", help="训练输出目录")
    parser.add_argument("--iter-start", type=int, default=None, help="起始 iteration")
    parser.add_argument("--iter-end", type=int, default=None, help="结束 iteration")
    parser.add_argument("--top-ratio", type=float, default=0.25, help="取前后多少比例做高低层对比")
    parser.add_argument("--first-route-n", type=int, default=5, help="路线签名保留前几个 map choice")
    parser.add_argument("--first-card-n", type=int, default=3, help="统计前几张拿牌")
    parser.add_argument("--top-k", type=int, default=10, help="每类保留 Top 数量")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    analysis_dir = training_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summaries = _load_summaries(training_dir, args.iter_start, args.iter_end)
    if not summaries:
        raise SystemExit("未找到可用的 *.summary.json。")

    report = _build_report(
        summaries,
        top_ratio=args.top_ratio,
        first_route_n=args.first_route_n,
        first_card_n=args.first_card_n,
        top_k=args.top_k,
    )

    suffix = []
    if args.iter_start is not None:
        suffix.append(f"from_{args.iter_start:05d}")
    if args.iter_end is not None:
        suffix.append(f"to_{args.iter_end:05d}")
    name = "route_shop_patterns" + ("_" + "_".join(suffix) if suffix else "")

    json_path = analysis_dir / f"{name}.json"
    md_path = analysis_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
