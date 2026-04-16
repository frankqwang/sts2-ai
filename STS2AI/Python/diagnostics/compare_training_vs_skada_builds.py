"""Compare AI training deck builds against Skada community win-rate statistics."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from analyze_iteration_replays import SkadaNameResolver
from analyze_training_window import _load_summaries
from classify_episode_failures import _classify_episode


DB_PATH = Path(r"C:\Users\Administrator\Desktop\sts2Raw2\STS2AI\Assets\datasets\skada\skada_analytics.sqlite")


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _top(counter: Counter[str], resolver: SkadaNameResolver, category: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, count in counter.most_common(limit):
        if category in {"card", "shop_target"}:
            display = resolver.card(name)
        else:
            display = resolver.generic(name)
        rows.append({"name": name, "display": display, "count": count})
    return rows


def _training_profile(
    summaries: list[dict[str, Any]],
    resolver: SkadaNameResolver,
    *,
    label: str,
    top_k: int,
) -> dict[str, Any]:
    floors: list[float] = []
    cards_taken_counter: Counter[str] = Counter()
    early_card_counter: Counter[str] = Counter()
    shop_action_counter: Counter[str] = Counter()
    rest_action_counter: Counter[str] = Counter()
    shop_visits = 0
    empty_shops = 0

    for summary in summaries:
        floors.append(float(summary.get("final_floor", 0) or 0))
        for card_id in summary.get("cards_taken") or []:
            cards_taken_counter[str(card_id)] += 1
        picked = 0
        for reward in summary.get("card_rewards") or []:
            if bool(reward.get("skipped")):
                continue
            early_card_counter[str(reward.get("label") or "unknown")] += 1
            picked += 1
            if picked >= 5:
                break
        for session in summary.get("shop_sessions") or []:
            shop_visits += 1
            actions = session.get("actions") or []
            non_proceed = [a for a in actions if str(a.get("action") or "").strip().lower() != "proceed"]
            if not non_proceed:
                empty_shops += 1
            for action in actions:
                action_name = str(action.get("action") or "").strip().lower() or "unknown"
                shop_action_counter[action_name] += 1
        for session in summary.get("rest_sessions") or []:
            for action in session.get("actions") or []:
                if str(action.get("action") or "").strip().lower() == "choose_rest_option":
                    rest_action_counter[str(action.get("label") or "unknown").strip().lower()] += 1

    return {
        "name": label,
        "episode_count": len(summaries),
        "avg_floor": round(_mean(floors), 4),
        "shop_visits_per_ep": round(shop_visits / max(1, len(summaries)), 4),
        "empty_shop_rate": round(empty_shops / max(1, shop_visits), 4),
        "cards_taken_top": _top(cards_taken_counter, resolver, "card", top_k),
        "early_cards_top": _top(early_card_counter, resolver, "card", top_k),
        "shop_action_top": _top(shop_action_counter, resolver, "generic", top_k),
        "campfire_action_top": _top(rest_action_counter, resolver, "generic", top_k),
    }


def _skada_profile(
    conn: sqlite3.Connection,
    resolver: SkadaNameResolver,
    *,
    character: str,
    min_ascension: int,
    top_k: int,
    label: str,
) -> dict[str, Any]:
    run_rows = conn.execute(
        """
        SELECT run_id, floor_reached
        FROM runs
        WHERE character = ? AND ascension >= ? AND is_victory = 1
        """,
        (character, min_ascension),
    ).fetchall()
    run_ids = [int(row[0]) for row in run_rows]
    avg_floor = _mean([float(row[1] or 0) for row in run_rows])
    if not run_ids:
        return {
            "name": label,
            "episode_count": 0,
            "avg_floor": 0.0,
            "cards_taken_top": [],
            "early_cards_top": [],
            "shop_action_top": [],
            "shop_remove_target_top": [],
            "campfire_action_top": [],
        }

    placeholder = ",".join("?" for _ in run_ids)
    final_deck_counter: Counter[str] = Counter()
    early_card_counter: Counter[str] = Counter()
    shop_action_counter: Counter[str] = Counter()
    shop_remove_target_counter: Counter[str] = Counter()
    campfire_counter: Counter[str] = Counter()

    for row in conn.execute(
        f"""
        SELECT card_id, count
        FROM run_final_deck
        WHERE run_id IN ({placeholder})
        """,
        run_ids,
    ):
        final_deck_counter[str(row[0])] += int(row[1] or 0)

    for row in conn.execute(
        f"""
        SELECT card_id
        FROM run_floor_card_choices
        WHERE run_id IN ({placeholder}) AND was_picked = 1 AND floor <= 8
        """,
        run_ids,
    ):
        early_card_counter[str(row[0])] += 1

    for row in conn.execute(
        f"""
        SELECT action_type, item_id
        FROM run_floor_shop_actions
        WHERE run_id IN ({placeholder})
        """,
        run_ids,
    ):
        action_type = str(row[0] or "unknown").strip().lower()
        shop_action_counter[action_type] += 1
        if action_type == "remove":
            shop_remove_target_counter[str(row[1] or "unknown")] += 1

    for row in conn.execute(
        f"""
        SELECT campfire_choice
        FROM run_floor_timeline
        WHERE run_id IN ({placeholder}) AND campfire_choice IS NOT NULL
        """,
        run_ids,
    ):
        campfire_counter[str(row[0] or "unknown").strip().lower()] += 1

    return {
        "name": label,
        "episode_count": len(run_ids),
        "avg_floor": round(avg_floor, 4),
        "cards_taken_top": _top(final_deck_counter, resolver, "card", top_k),
        "early_cards_top": _top(early_card_counter, resolver, "card", top_k),
        "shop_action_top": _top(shop_action_counter, resolver, "generic", top_k),
        "shop_remove_target_top": _top(shop_remove_target_counter, resolver, "shop_target", top_k),
        "campfire_action_top": _top(campfire_counter, resolver, "generic", top_k),
    }


def _build_report(
    training_dir: Path,
    *,
    iter_start: int | None,
    iter_end: int | None,
    character: str,
    min_ascension: int,
    top_k: int,
) -> dict[str, Any]:
    resolver = SkadaNameResolver()
    summaries = _load_summaries(training_dir, iter_start, iter_end)
    clear_group: list[dict[str, Any]] = []
    boss_loss_group: list[dict[str, Any]] = []

    for summary in summaries:
        label, _ = _classify_episode(summary, early_floor_threshold=8, pending_stall_threshold=20)
        if label == "act1_clear":
            clear_group.append(summary)
        elif label == "boss_loss":
            boss_loss_group.append(summary)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        all_wins = _skada_profile(
            conn,
            resolver,
            character=character,
            min_ascension=0,
            top_k=top_k,
            label=f"skada_{character}_all_wins",
        )
        high_wins = _skada_profile(
            conn,
            resolver,
            character=character,
            min_ascension=min_ascension,
            top_k=top_k,
            label=f"skada_{character}_a{min_ascension}_wins",
        )
    finally:
        conn.close()

    return {
        "training_dir": str(training_dir),
        "iter_start": iter_start,
        "iter_end": iter_end,
        "character": character,
        "min_ascension": min_ascension,
        "training_act1_clear": _training_profile(clear_group, resolver, label="training_act1_clear", top_k=top_k),
        "training_boss_loss": _training_profile(boss_loss_group, resolver, label="training_boss_loss", top_k=top_k),
        "skada_all_wins": all_wins,
        "skada_high_ascension_wins": high_wins,
    }


def _render_profile(lines: list[str], title: str, profile: dict[str, Any], *, include_remove_targets: bool = False) -> None:
    lines.append(f"## {title}")
    lines.append(f"- 样本数: `{profile.get('episode_count', 0)}`")
    lines.append(f"- 平均层数: `{profile.get('avg_floor', 0):.4f}`")
    if "shop_visits_per_ep" in profile:
        lines.append(f"- 商店 visits/ep: `{profile.get('shop_visits_per_ep', 0):.4f}`")
        lines.append(f"- 空店率: `{profile.get('empty_shop_rate', 0) * 100:.2f}%`")
    for key, header in [
        ("cards_taken_top", "拿牌 / 最终牌组 Top"),
        ("early_cards_top", "前 8 层拿牌 Top"),
        ("shop_action_top", "商店动作 Top"),
        ("campfire_action_top", "营火动作 Top"),
    ]:
        lines.append(f"### {header}")
        items = profile.get(key) or []
        if not items:
            lines.append("- 无")
        else:
            for item in items:
                label = item.get("display") or item.get("name")
                lines.append(f"- `{label}`: {item.get('count', 0)}")
        lines.append("")
    if include_remove_targets:
        lines.append("### 商店删牌目标 Top")
        items = profile.get("shop_remove_target_top") or []
        if not items:
            lines.append("- 无")
        else:
            for item in items:
                label = item.get("display") or item.get("name")
                lines.append(f"- `{label}`: {item.get('count', 0)}")
        lines.append("")


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# 训练窗口 vs Skada 构筑对照")
    lines.append("")
    lines.append(f"- 训练目录: `{report.get('training_dir', '')}`")
    lines.append(f"- iteration: `{report.get('iter_start')}` - `{report.get('iter_end')}`")
    lines.append(f"- 角色: `{report.get('character')}` / 高难参考下限: `A{report.get('min_ascension')}`")
    lines.append("")
    _render_profile(lines, "训练窗口：Act1 通关组", report.get("training_act1_clear") or {})
    _render_profile(lines, "训练窗口：Boss 失败组", report.get("training_boss_loss") or {})
    _render_profile(lines, "Skada：所有胜利", report.get("skada_all_wins") or {}, include_remove_targets=True)
    _render_profile(lines, f"Skada：A{report.get('min_ascension')}+ 胜利", report.get("skada_high_ascension_wins") or {}, include_remove_targets=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="将训练窗口构筑与 Skada 胜利样本做对照。")
    parser.add_argument("training_dir", help="训练输出目录")
    parser.add_argument("--iter-start", type=int, default=None, help="起始 iteration")
    parser.add_argument("--iter-end", type=int, default=None, help="结束 iteration")
    parser.add_argument("--character", type=str, default="IRONCLAD", help="角色 ID")
    parser.add_argument("--min-ascension", type=int, default=10, help="高难参考的最低难度")
    parser.add_argument("--top-k", type=int, default=12, help="每类保留多少个 Top")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    analysis_dir = training_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report = _build_report(
        training_dir,
        iter_start=args.iter_start,
        iter_end=args.iter_end,
        character=args.character,
        min_ascension=args.min_ascension,
        top_k=args.top_k,
    )

    suffix: list[str] = []
    if args.iter_start is not None:
        suffix.append(f"from_{args.iter_start:05d}")
    if args.iter_end is not None:
        suffix.append(f"to_{args.iter_end:05d}")
    name = "training_vs_skada_builds" + ("_" + "_".join(suffix) if suffix else "")

    json_path = analysis_dir / f"{name}.json"
    md_path = analysis_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"JSON: {json_path}")
    print(f"MD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
