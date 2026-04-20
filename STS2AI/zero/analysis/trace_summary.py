from __future__ import annotations

"""把详细轨迹转成便于人工检查的中文摘要。

当前支持两类来源：
- raw_runs/iter_xxxx.jsonl：训练 collect 轨迹
- eval/iter_xxxx_candidate_eval.jsonl：评估轨迹

摘要目标：
- raw 更偏排查：带手牌、合法动作、选择结果
- eval 更偏结论：只看评估时模型实际做了什么
- 每步用 3 行左右描述，便于快速扫读
- 单行尽量不超过 200 列
- 每个 iter 随机抽若干把 fight，聚焦最近结果

这是一个离线分析工具，不绑定训练主循环：
- 训练只负责把 raw_runs / eval 日志落盘
- 需要复盘时，再单独运行这个脚本抽样生成中文摘要
"""

import argparse
import json
import random
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..paths import STS2AI_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", choices=["raw", "eval", "both"], default="both")
    parser.add_argument("--iters", type=int, default=3, help="最近多少个 iter")
    parser.add_argument("--samples-per-iter", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--max-steps-per-fight", type=int, default=80)
    args = parser.parse_args()

    output_dir = args.run_root / "analysis" / "trace_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    name_catalog = _load_name_catalog()

    if args.source in {"raw", "both"}:
        _summarize_recent_raw_iters(
            run_root=args.run_root,
            output_dir=output_dir,
            recent_iters=args.iters,
            samples_per_iter=args.samples_per_iter,
            max_steps=args.max_steps_per_fight,
            seed=args.seed,
            name_catalog=name_catalog,
        )
    if args.source in {"eval", "both"}:
        _summarize_recent_eval_iters(
            run_root=args.run_root,
            output_dir=output_dir,
            recent_iters=args.iters,
            samples_per_iter=args.samples_per_iter,
            max_steps=args.max_steps_per_fight,
            seed=args.seed,
            name_catalog=name_catalog,
        )

    # 这里只返回输入输出位置，方便外层脚本或人工二次调用。
    print(
        json.dumps(
            {
                "run_root": str(args.run_root),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


def _summarize_recent_raw_iters(
    *,
    run_root: Path,
    output_dir: Path,
    recent_iters: int,
    samples_per_iter: int,
    max_steps: int,
    seed: int,
    name_catalog: dict[str, str],
) -> None:
    raw_root = run_root / "raw_runs"
    if not raw_root.exists():
        return
    files = sorted(raw_root.glob("iter_*.jsonl"))
    for file_path in files[-recent_iters:]:
        rows = _read_jsonl(file_path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("fight_id") or "")].append(row)
        chosen = _sample_groups(grouped, samples_per_iter, seed=seed + _iter_no(file_path))
        lines: list[str] = []
        for group_key in chosen:
            transitions = sorted(grouped[group_key], key=lambda row: int(row.get("step_idx") or 0))
            lines.extend(_render_raw_fight(group_key, transitions[:max_steps], name_catalog=name_catalog))
            lines.append("")
        (output_dir / f"{file_path.stem}_raw_zh.txt").write_text("\n".join(lines), encoding="utf-8")


def _summarize_recent_eval_iters(
    *,
    run_root: Path,
    output_dir: Path,
    recent_iters: int,
    samples_per_iter: int,
    max_steps: int,
    seed: int,
    name_catalog: dict[str, str],
) -> None:
    eval_root = run_root / "eval"
    if not eval_root.exists():
        return
    files = sorted(eval_root.glob("iter_*_candidate_eval.jsonl"))
    for file_path in files[-recent_iters:]:
        rows = _read_jsonl(file_path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = f"{row.get('case_id','')}|ep{row.get('episode_index',0)}"
            grouped[key].append(row)
        chosen = _sample_groups(grouped, samples_per_iter, seed=seed + 1000 + _iter_no(file_path))
        lines: list[str] = []
        for group_key in chosen:
            events = sorted(grouped[group_key], key=lambda row: (int(row.get("episode_index") or 0), int(row.get("step_idx") or -1)))
            lines.extend(_render_eval_fight(group_key, events[: max_steps + 1], name_catalog=name_catalog))
            lines.append("")
        (output_dir / f"{file_path.stem}_eval_zh.txt").write_text("\n".join(lines), encoding="utf-8")


def _render_raw_fight(
    fight_id: str,
    transitions: list[dict[str, Any]],
    *,
    name_catalog: dict[str, str],
) -> list[str]:
    if not transitions:
        return [f"=== collect fight {fight_id}（空） ==="]
    first = transitions[0]
    state = first["state"]
    context = state["context"]
    header = (
        f"=== collect fight={fight_id} case={context['metadata'].get('skada_case_id','')} "
        f"seed={context['metadata'].get('seed','')} floor={context.get('encounter_id','')} ==="
    )
    lines = [_wrap(header)]
    for row in transitions:
        state = row["state"]
        next_state = row["next_state"]
        action = row["action"]
        meta = row.get("metadata", {})
        legal_actions = state.get("legal_actions") or []
        hand = state.get("hand") or []
        line1 = (
            f"[Step {row.get('step_idx', 0):>3}] HP {state['player']['hp']:.0f}/{state['player']['max_hp']:.0f} "
            f"格挡 {state['player']['block']:.0f} 能量 {state['player']['energy']:.0f} | "
            f"敌方 {_enemy_brief(state.get('enemies') or [], name_catalog=name_catalog)}"
        )
        line2 = f"手牌：{_hand_brief(hand, name_catalog=name_catalog)}"
        legal_action_lines = _legal_action_lines(legal_actions, name_catalog=name_catalog)
        line3 = (
            f"选择：{_format_action(action, name_catalog=name_catalog)} | "
            f"unc={float(meta.get('uncertainty', 0.0) or 0.0):.3f} "
            f"gap={float(meta.get('top2_gap', 0.0) or 0.0):.3f} | "
            f"prog={'Y' if bool(meta.get('made_progress', False)) else 'N'} "
            f"enemy_hp_delta={float(meta.get('enemy_hp_delta', 0.0) or 0.0):.1f}"
        )
        line4 = (
            f"结果 HP {next_state['player']['hp']:.0f}/{next_state['player']['max_hp']:.0f} "
            f"格挡 {next_state['player']['block']:.0f} 能量 {next_state['player']['energy']:.0f} | "
            f"敌方 {_enemy_brief(next_state.get('enemies') or [], name_catalog=name_catalog)}"
        )
        lines.extend([_wrap(line1), _wrap(line2), f"合法动作（{len(legal_actions)} 个）：", *legal_action_lines, _wrap(line3), _wrap(line4), ""])
    last = transitions[-1]
    lines.append(_wrap(f"--- fight 结束标记：done={last.get('done')} outcome={last.get('run_outcome')} ---"))
    return lines


def _render_eval_fight(group_key: str, events: list[dict[str, Any]], *, name_catalog: dict[str, str]) -> list[str]:
    header = f"=== eval fight={group_key} ==="
    lines = [_wrap(header)]
    fight_end = None
    for row in events:
        if row.get("event") == "fight_end":
            fight_end = row
            continue
        if row.get("event") != "step":
            continue
        if isinstance(row.get("state"), dict) and isinstance(row.get("next_state"), dict):
            lines.extend(_render_eval_step_detailed(row, name_catalog=name_catalog))
            continue
        line1 = (
            f"[Step {int(row.get('step_idx', 0)):>3}] HP {float(row.get('player_hp', 0.0)):.0f} "
            f"格挡 {float(row.get('player_block', 0.0)):.0f} 能量 {float(row.get('player_energy', 0.0)):.0f} | "
            f"敌方 {_enemy_hp_list_brief(row.get('enemy_hp') or [], row.get('enemy_block') or [])}"
        )
        teacher_bits: list[str] = []
        teacher_best = row.get("teacher_best_action_index")
        action_index = row.get("action_index")
        if teacher_best is not None and action_index is not None:
            teacher_bits.append("teacher=一致" if int(teacher_best) == int(action_index) else "teacher=不一致")
        teacher_topk = row.get("teacher_topk_indices") or []
        if teacher_topk and action_index is not None:
            teacher_bits.append("topk=命中" if int(action_index) in {int(v) for v in teacher_topk} else "topk=未命中")
        line2 = (
            f"选择：{_format_action(row, name_catalog=name_catalog)}"
            + (f" | {' '.join(teacher_bits)}" if teacher_bits else "")
        )
        line3 = (
            f"prog={'Y' if bool(row.get('made_progress', False)) else 'N'} "
            f"enemy_hp_delta={float(row.get('enemy_hp_delta', 0.0) or 0.0):.1f} "
            f"enemy_count_delta={int(row.get('enemy_count_delta', 0) or 0)}"
        )
        lines.extend([_wrap(line1), _wrap(line2), _wrap(line3), ""])
    if fight_end is not None:
        lines.append(
            _wrap(
                f"--- 结算 outcome={fight_end.get('outcome')} steps={fight_end.get('steps')} "
                f"耗时={float(fight_end.get('duration_s', 0.0) or 0.0):.3f}s "
                f"step/s={float(fight_end.get('step_throughput', 0.0) or 0.0):.1f} "
                f"无进展占比={float(fight_end.get('no_progress_ratio', 0.0) or 0.0):.3f} "
                f"最长无进展={fight_end.get('max_no_progress_streak')}"
            )
        )
    return lines


def _render_eval_step_detailed(row: dict[str, Any], *, name_catalog: dict[str, str]) -> list[str]:
    state = row["state"]
    next_state = row["next_state"]
    action = row.get("action") or {}
    legal_actions = state.get("legal_actions") or []
    teacher_bits: list[str] = []
    teacher_best = row.get("teacher_best_action_index")
    action_index = row.get("action_index")
    if teacher_best is not None and action_index is not None:
        teacher_bits.append("teacher=一致" if int(teacher_best) == int(action_index) else "teacher=不一致")
    teacher_topk = row.get("teacher_topk_indices") or []
    if teacher_topk and action_index is not None:
        teacher_bits.append("topk=命中" if int(action_index) in {int(v) for v in teacher_topk} else "topk=未命中")
    line1 = (
        f"[Step {int(row.get('step_idx', 0)):>3}] HP {float(state['player']['hp']):.0f}/{float(state['player']['max_hp']):.0f} "
        f"格挡 {float(state['player']['block']):.0f} 能量 {float(state['player']['energy']):.0f} | "
        f"敌方 {_enemy_brief(state.get('enemies') or [], name_catalog=name_catalog)}"
    )
    line2 = f"手牌：{_hand_brief(state.get('hand') or [], name_catalog=name_catalog)}"
    legal_action_lines = _legal_action_lines(legal_actions, name_catalog=name_catalog)
    line3 = (
        f"选择：{_format_action(action, name_catalog=name_catalog)}"
        + (f" | {' '.join(teacher_bits)}" if teacher_bits else "")
    )
    line4 = (
        f"结果 HP {float(next_state['player']['hp']):.0f}/{float(next_state['player']['max_hp']):.0f} "
        f"格挡 {float(next_state['player']['block']):.0f} 能量 {float(next_state['player']['energy']):.0f} | "
        f"敌方 {_enemy_brief(next_state.get('enemies') or [], name_catalog=name_catalog)}"
    )
    line5 = (
        f"prog={'Y' if bool(row.get('made_progress', False)) else 'N'} "
        f"enemy_hp_delta={float(row.get('enemy_hp_delta', 0.0) or 0.0):.1f} "
        f"enemy_count_delta={int(row.get('enemy_count_delta', 0) or 0)}"
    )
    return [
        _wrap(line1),
        _wrap(line2),
        f"合法动作（{len(legal_actions)} 个）：",
        *legal_action_lines,
        _wrap(line3),
        _wrap(line4),
        _wrap(line5),
        "",
    ]


def _enemy_brief(enemies: list[dict[str, Any]], *, name_catalog: dict[str, str]) -> str:
    parts = []
    for enemy in enemies:
        parts.append(
            f"{_display_name(enemy.get('enemy_id','?'), name_catalog)} {float(enemy.get('hp', 0.0)):.0f}/{float(enemy.get('max_hp', 0.0)):.0f}"
            f"(盾{float(enemy.get('block', 0.0)):.0f},意图={enemy.get('intent_id','')})"
        )
    return " ; ".join(parts) if parts else "无"


def _enemy_hp_list_brief(hps: list[Any], blocks: list[Any]) -> str:
    parts = []
    for idx, hp in enumerate(hps):
        block = blocks[idx] if idx < len(blocks) else 0.0
        parts.append(f"#{idx} {float(hp):.0f}(盾{float(block):.0f})")
    return " ; ".join(parts) if parts else "无"


def _hand_brief(hand: list[dict[str, Any]], *, name_catalog: dict[str, str]) -> str:
    parts = []
    for card in hand:
        notes: list[str] = [f"费{float(card.get('cost_now', 0.0)):.0f}"]
        tags = set(card.get("tags") or [])
        if "requires_target" in tags:
            notes.append("需目标")
        if "retain" in tags or bool(card.get("retain")):
            notes.append("保留")
        if "ethereal" in tags or bool(card.get("ethereal")):
            notes.append("消逝")
        if "exhaust" in tags or bool(card.get("exhaust")):
            notes.append("消耗")
        parts.append(f"{_display_name(card.get('card_id','?'), name_catalog)}({','.join(notes)})")
    return "； ".join(parts) if parts else "空手"


def _legal_action_lines(
    actions: list[dict[str, Any]],
    *,
    name_catalog: dict[str, str],
    limit: int = 8,
) -> list[str]:
    lines: list[str] = []
    for action in actions[:limit]:
        lines.append(f"  - {_wrap(_format_action(action, name_catalog=name_catalog), width=168)}")
    if len(actions) > limit:
        lines.append(f"  - ...（其余 {len(actions) - limit} 个未展开）")
    return lines or ["  - 无"]


def _format_action(action: dict[str, Any], *, name_catalog: dict[str, str]) -> str:
    action_type = str(action.get("action_type", "") or "").strip()
    if not action_type:
        action_id = str(action.get("action_id", "") or "").strip()
        if "|" in action_id:
            action_type = action_id.split("|", 1)[0]
        elif action_id:
            action_type = action_id
    card_id = str(action.get("card_id", "") or "").strip()
    special_id = str(action.get("special_id", "") or "").strip()
    target_id = str(action.get("target_id", "") or "").strip()
    can_execute = action.get("can_execute")
    if action_type == "play_card" and card_id:
        text = f"出牌 {_display_name(card_id, name_catalog)}"
    elif action_type == "end_turn":
        text = "结束回合"
    elif action_type == "select_hand_card" and card_id:
        text = f"选手牌 {_display_name(card_id, name_catalog)}"
    elif action_type == "confirm_selection":
        text = "确认选择"
    elif special_id:
        text = f"{action_type or '动作'} {special_id}"
    elif card_id:
        text = f"{action_type or '动作'} {_display_name(card_id, name_catalog)}"
    else:
        fallback = str(action.get("action_id", "") or "").strip()
        text = action_type or fallback or "未知动作"
    if target_id:
        text += f" -> 目标{target_id}"
    if can_execute is False:
        text += " [no]"
    return text


def _display_name(raw_id: str, name_catalog: dict[str, str]) -> str:
    key = str(raw_id or "").strip()
    if not key:
        return "?"
    return name_catalog.get(key.upper()) or name_catalog.get(key) or key


def _load_name_catalog() -> dict[str, str]:
    """优先读取权威文本映射；没有时退回原始 id。"""
    catalog: dict[str, str] = {}
    sqlite_path = STS2AI_ROOT / "data" / "game_wiki" / "game_catalog.sqlite"
    if sqlite_path.exists():
        try:
            with sqlite3.connect(sqlite_path) as conn:
                for table, id_col in (("cards", "id"), ("relics", "id"), ("potions", "id"), ("monsters", "id")):
                    try:
                        rows = conn.execute(
                            f"SELECT {id_col}, "
                            "COALESCE(name_zh, ''), COALESCE(name_en, ''), payload_json "
                            f"FROM {table}"
                        ).fetchall()
                    except sqlite3.DatabaseError:
                        continue
                    for item_id, name_zh, name_en, payload_json in rows:
                        if name_zh:
                            catalog[str(item_id).upper()] = str(name_zh)
                            continue
                        if name_en:
                            catalog[str(item_id).upper()] = str(name_en)
                            continue
                        try:
                            payload = json.loads(payload_json or "{}")
                        except json.JSONDecodeError:
                            payload = {}
                        zh_name = (
                            payload.get("name_zh")
                            or payload.get("display_name_zh")
                            or payload.get("localized_name")
                            or payload.get("name")
                        )
                        if zh_name:
                            catalog[str(item_id).upper()] = str(zh_name)
        except sqlite3.DatabaseError:
            pass
    return catalog


def _wrap(text: str, width: int = 180) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)) or text


def _sample_groups(grouped: dict[str, list[dict[str, Any]]], count: int, *, seed: int) -> list[str]:
    keys = sorted(grouped.keys())
    if len(keys) <= count:
        return keys
    rng = random.Random(seed)
    return sorted(rng.sample(keys, count))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _iter_no(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.isdigit():
            return int(part)
    return 0


if __name__ == "__main__":
    main()
