"""Analyze deck-building and shop patterns correlated with episode win/loss outcomes."""
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


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _top_cards(counter: Counter[str], resolver: SkadaNameResolver, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "name": card_id,
            "display": resolver.card(card_id),
            "count": count,
        }
        for card_id, count in counter.most_common(limit)
    ]


def _shop_counts(summary: dict[str, Any]) -> dict[str, int]:
    out = {
        "shop_visits": 0,
        "empty_shop": 0,
        "affordable_shop_skip": 0,
        "unaffordable_empty_shop": 0,
        "remove_card": 0,
        "buy_card": 0,
        "buy_potion": 0,
    }
    for session in summary.get("shop_sessions") or []:
        out["shop_visits"] += 1
        actions = session.get("actions") or []
        non_proceed = [a for a in actions if str(a.get("action") or "").strip().lower() != "proceed"]
        if not non_proceed:
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
            elif action_name not in {"", "proceed"}:
                out["buy_card"] += 1
    return out


def _rest_counts(summary: dict[str, Any]) -> dict[str, int]:
    out = {"rest": 0, "smith": 0, "other": 0}
    for session in summary.get("rest_sessions") or []:
        for action in session.get("actions") or []:
            action_name = str(action.get("action") or "").strip().lower()
            label = str(action.get("label") or "").strip().lower()
            if action_name != "choose_rest_option":
                continue
            if label == "rest":
                out["rest"] += 1
            elif label == "smith":
                out["smith"] += 1
            else:
                out["other"] += 1
    return out


def _boss_combat(summary: dict[str, Any]) -> dict[str, Any] | None:
    boss_combats = [
        combat for combat in _combat_entries(summary)
        if str(combat.get("room_type") or "").strip().lower() == "boss"
    ]
    return boss_combats[-1] if boss_combats else None


def _card_prefix(summary: dict[str, Any], limit: int) -> list[str]:
    picks: list[str] = []
    for reward in summary.get("card_rewards") or []:
        if bool(reward.get("skipped")):
            continue
        picks.append(str(reward.get("label") or "unknown"))
        if len(picks) >= limit:
            break
    return picks


def _group_stats(
    summaries: list[dict[str, Any]],
    resolver: SkadaNameResolver,
    *,
    top_k: int,
    early_card_n: int,
) -> dict[str, Any]:
    floors: list[float] = []
    cards_taken_total: list[float] = []
    cards_skipped_total: list[float] = []
    shop_visits: list[float] = []
    empty_shop_rate: list[float] = []
    remove_per_ep: list[float] = []
    affordable_shop_skip_per_ep: list[float] = []
    buy_card_per_ep: list[float] = []
    buy_potion_per_ep: list[float] = []
    rest_per_ep: list[float] = []
    smith_per_ep: list[float] = []
    boss_start_hp: list[float] = []
    boss_start_hp_ratio: list[float] = []
    boss_start_deck_size: list[float] = []
    boss_start_potions: list[float] = []
    boss_action_count: list[float] = []
    boss_enemy_counter: Counter[str] = Counter()
    card_counter: Counter[str] = Counter()
    early_card_counter: Counter[str] = Counter()
    early_combo_counter: Counter[str] = Counter()

    for summary in summaries:
        floors.append(float(summary.get("final_floor", 0) or 0))
        cards_taken = list(summary.get("cards_taken") or [])
        cards_taken_total.append(float(len(cards_taken)))
        cards_skipped_total.append(float(summary.get("cards_skipped", 0) or 0))
        for card_id in cards_taken:
            card_counter[str(card_id)] += 1
        prefix = _card_prefix(summary, early_card_n)
        for card_id in prefix:
            early_card_counter[card_id] += 1
        early_combo_counter[" + ".join(prefix) if prefix else "无前期拿牌"] += 1

        shop = _shop_counts(summary)
        shop_visits.append(float(shop["shop_visits"]))
        empty_shop_rate.append(float(shop["empty_shop"]) / max(1.0, float(shop["shop_visits"])))
        remove_per_ep.append(float(shop["remove_card"]))
        affordable_shop_skip_per_ep.append(float(shop["affordable_shop_skip"]))
        buy_card_per_ep.append(float(shop["buy_card"]))
        buy_potion_per_ep.append(float(shop["buy_potion"]))

        rest = _rest_counts(summary)
        rest_per_ep.append(float(rest["rest"]))
        smith_per_ep.append(float(rest["smith"]))

        boss_combat = _boss_combat(summary)
        if boss_combat is not None:
            start_hp = float(boss_combat.get("start_hp", 0) or 0)
            start_max_hp = float(boss_combat.get("start_max_hp", 0) or 0)
            boss_start_hp.append(start_hp)
            boss_start_hp_ratio.append(start_hp / max(1.0, start_max_hp))
            boss_start_deck_size.append(float(boss_combat.get("start_deck_size", 0) or 0))
            boss_start_potions.append(float(boss_combat.get("start_potion_count", 0) or 0))
            boss_action_count.append(float(boss_combat.get("action_count", 0) or 0))
            boss_enemy_counter[str(boss_combat.get("enemy_group") or "UNKNOWN")] += 1

    return {
        "episode_count": len(summaries),
        "avg_floor": round(_mean(floors), 4),
        "avg_cards_taken": round(_mean(cards_taken_total), 4),
        "avg_cards_skipped": round(_mean(cards_skipped_total), 4),
        "shop": {
            "visits_per_ep": round(_mean(shop_visits), 4),
            "empty_rate": round(_mean(empty_shop_rate), 4),
            "remove_per_ep": round(_mean(remove_per_ep), 4),
            "affordable_shop_skip_per_ep": round(_mean(affordable_shop_skip_per_ep), 4),
            "buy_card_per_ep": round(_mean(buy_card_per_ep), 4),
            "buy_potion_per_ep": round(_mean(buy_potion_per_ep), 4),
        },
        "campfire": {
            "rest_per_ep": round(_mean(rest_per_ep), 4),
            "smith_per_ep": round(_mean(smith_per_ep), 4),
        },
        "boss_entry": {
            "avg_start_hp": round(_mean(boss_start_hp), 4),
            "avg_start_hp_ratio": round(_mean(boss_start_hp_ratio), 4),
            "avg_deck_size": round(_mean(boss_start_deck_size), 4),
            "avg_potion_count": round(_mean(boss_start_potions), 4),
            "avg_action_count": round(_mean(boss_action_count), 4),
            "boss_enemy_top": [
                {
                    "name": name,
                    "display": resolver.enemy_group(name),
                    "count": count,
                }
                for name, count in boss_enemy_counter.most_common(top_k)
            ],
        },
        "cards_taken_top": _top_cards(card_counter, resolver, top_k),
        "early_cards_top": _top_cards(early_card_counter, resolver, top_k),
        "early_card_combo_top": [
            {"name": name, "count": count}
            for name, count in early_combo_counter.most_common(top_k)
        ],
        "card_counter": dict(card_counter),
        "early_card_counter": dict(early_card_counter),
    }


def _diff_cards(
    left: dict[str, Any],
    right: dict[str, Any],
    resolver: SkadaNameResolver,
    *,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    left_episodes = max(1, int(left.get("episode_count", 0)))
    right_episodes = max(1, int(right.get("episode_count", 0)))
    left_counter = Counter(left.get("card_counter") or {})
    right_counter = Counter(right.get("card_counter") or {})
    all_cards = set(left_counter) | set(right_counter)
    rows: list[dict[str, Any]] = []
    for card_id in all_cards:
        left_rate = left_counter[card_id] / left_episodes
        right_rate = right_counter[card_id] / right_episodes
        rows.append(
            {
                "name": card_id,
                "display": resolver.card(card_id),
                "left_rate": round(left_rate, 4),
                "right_rate": round(right_rate, 4),
                "delta": round(left_rate - right_rate, 4),
            }
        )
    rows.sort(key=lambda item: item["delta"], reverse=True)
    return {
        "left_over_indexed": rows[:limit],
        "right_over_indexed": list(reversed(rows[-limit:])),
    }


def _build_report(
    summaries: list[dict[str, Any]],
    *,
    early_floor_threshold: int,
    pending_stall_threshold: int,
    top_k: int,
    early_card_n: int,
) -> dict[str, Any]:
    resolver = SkadaNameResolver()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        label, _ = _classify_episode(
            summary,
            early_floor_threshold=early_floor_threshold,
            pending_stall_threshold=pending_stall_threshold,
        )
        grouped[label].append(summary)

    focus_order = [
        "act1_clear",
        "boss_loss",
        "preboss_death",
        "combat_pending_stall",
        "early_death",
    ]
    groups = {
        name: _group_stats(grouped.get(name, []), resolver, top_k=top_k, early_card_n=early_card_n)
        for name in focus_order
        if grouped.get(name)
    }

    comparisons: dict[str, Any] = {}
    if "act1_clear" in groups and "boss_loss" in groups:
        comparisons["act1_clear_vs_boss_loss"] = {
            "cards_taken_delta": _diff_cards(groups["act1_clear"], groups["boss_loss"], resolver, limit=top_k)
        }
    if "act1_clear" in groups and "preboss_death" in groups:
        comparisons["act1_clear_vs_preboss_death"] = {
            "cards_taken_delta": _diff_cards(groups["act1_clear"], groups["preboss_death"], resolver, limit=top_k)
        }

    return {
        "episode_count": len(summaries),
        "config": {
            "early_floor_threshold": early_floor_threshold,
            "pending_stall_threshold": pending_stall_threshold,
            "early_card_n": early_card_n,
        },
        "groups": groups,
        "comparisons": comparisons,
    }


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Build / 结果关联分析")
    lines.append("")
    lines.append(f"- 样本局数: `{report.get('episode_count', 0)}`")
    lines.append(f"- 配置: `{json.dumps(report.get('config', {}), ensure_ascii=False)}`")
    lines.append("")

    for name, group in (report.get("groups") or {}).items():
        lines.append(f"## {name}")
        lines.append(f"- 局数: `{group.get('episode_count', 0)}`")
        lines.append(f"- 平均层数: `{group.get('avg_floor', 0):.4f}`")
        lines.append(f"- 平均拿牌数: `{group.get('avg_cards_taken', 0):.4f}`")
        lines.append(f"- 平均跳牌数: `{group.get('avg_cards_skipped', 0):.4f}`")
        shop = group.get("shop") or {}
        lines.append(
            f"- 商店: visits/ep `{shop.get('visits_per_ep', 0):.4f}` / empty_rate `{shop.get('empty_rate', 0) * 100:.2f}%`"
            f" / remove/ep `{shop.get('remove_per_ep', 0):.4f}` / affordable_skip/ep `{shop.get('affordable_shop_skip_per_ep', 0):.4f}` / buy_card/ep `{shop.get('buy_card_per_ep', 0):.4f}`"
            f" / buy_potion/ep `{shop.get('buy_potion_per_ep', 0):.4f}`"
        )
        campfire = group.get("campfire") or {}
        lines.append(
            f"- 营火: rest/ep `{campfire.get('rest_per_ep', 0):.4f}` / smith/ep `{campfire.get('smith_per_ep', 0):.4f}`"
        )
        boss_entry = group.get("boss_entry") or {}
        lines.append(
            f"- Boss 入场: HP `{boss_entry.get('avg_start_hp', 0):.2f}` / HP 比例 `{boss_entry.get('avg_start_hp_ratio', 0) * 100:.2f}%`"
            f" / 牌组大小 `{boss_entry.get('avg_deck_size', 0):.2f}` / 药水数 `{boss_entry.get('avg_potion_count', 0):.2f}`"
            f" / Boss 动作步数 `{boss_entry.get('avg_action_count', 0):.2f}`"
        )
        for key, header in [
            ("cards_taken_top", "拿牌 Top"),
            ("early_cards_top", "前期拿牌 Top"),
            ("early_card_combo_top", "前期拿牌组合 Top"),
            ("boss_enemy_top", "Boss 遭遇 Top"),
        ]:
            items = boss_entry.get(key) if key == "boss_enemy_top" else group.get(key)
            lines.append(f"### {header}")
            if not items:
                lines.append("- 无")
            else:
                for item in items:
                    label = item.get("display") or item.get("name")
                    lines.append(f"- `{label}`: {item.get('count', 0)}")
            lines.append("")

    for name, comp in (report.get("comparisons") or {}).items():
        lines.append(f"## {name}")
        diff = comp.get("cards_taken_delta") or {}
        lines.append("### 左侧更常见")
        for item in diff.get("left_over_indexed") or []:
            lines.append(
                f"- `{item.get('display')}`: left `{item.get('left_rate', 0):.4f}` vs right `{item.get('right_rate', 0):.4f}`"
            )
        lines.append("")
        lines.append("### 右侧更常见")
        for item in diff.get("right_over_indexed") or []:
            lines.append(
                f"- `{item.get('display')}`: left `{item.get('left_rate', 0):.4f}` vs right `{item.get('right_rate', 0):.4f}`"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="分析 build / 商店 / 营火与结果的关系。")
    parser.add_argument("training_dir", help="训练输出目录")
    parser.add_argument("--iter-start", type=int, default=None, help="起始 iteration")
    parser.add_argument("--iter-end", type=int, default=None, help="结束 iteration")
    parser.add_argument("--early-floor-threshold", type=int, default=8, help="多少层以下算 early_death")
    parser.add_argument("--pending-stall-threshold", type=int, default=20, help="多少步 pending 算 stall")
    parser.add_argument("--top-k", type=int, default=12, help="每类保留多少个 Top")
    parser.add_argument("--early-card-n", type=int, default=5, help="统计前几张拿牌")
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
        early_card_n=args.early_card_n,
    )

    suffix: list[str] = []
    if args.iter_start is not None:
        suffix.append(f"from_{args.iter_start:05d}")
    if args.iter_end is not None:
        suffix.append(f"to_{args.iter_end:05d}")
    name = "build_outcomes" + ("_" + "_".join(suffix) if suffix else "")

    json_path = analysis_dir / f"{name}.json"
    md_path = analysis_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
