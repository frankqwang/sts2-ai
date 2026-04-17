"""Boss 战出牌顺序合理性分析。

用法：
  python -m networkV2.s7_diagnostics.card_play_analysis ../Artifacts/runs/co16
  python -m networkV2.s7_diagnostics.card_play_analysis ../Artifacts/runs/co16 --n 3 --outcome defeat

功能：
  - 挑最新 N 场 boss trajectory（可按 outcome 过滤 defeat/victory）
  - 每 step 的 chosen_card 从 sqlite 查 card 机制（cost/type/tags/powers）
  - 按 turn 打印打牌序列 + 机制注释 + HP/block/enemy 状态
  - 统计可疑行为（不合理牌序）

产出：
  - 终端打印（人类阅读）
  - <dump>/analysis/card_play_analysis.txt（人类阅读版本）
  - <dump>/analysis/card_play_analysis.json（结构化机械可读）
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "source_knowledge.sqlite"
_CATALOG_JSONL = (
    Path(__file__).resolve().parents[3] / "Artifacts" / "game_knowledge" / "cards.jsonl"
)


def _load_card_catalog() -> dict[str, dict]:
    """优先从 Artifacts/game_knowledge/cards.jsonl（带描述）加载，
    fallback 到老的 sqlite（无描述）。返回 card_id (upper) → meta dict。"""
    catalog: dict[str, dict] = {}
    # 优先：导出的 JSONL catalog（含 description_en / title_en）
    if _CATALOG_JSONL.exists():
        with _CATALOG_JSONL.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                cid = str(d.get("id", "")).upper()
                catalog[cid] = {
                    "id": d.get("id"),
                    "title_en": d.get("title_en") or "",
                    "title_zhs": d.get("title_zhs") or "",
                    "description_en": d.get("description_en") or "",
                    "description_zhs": d.get("description_zhs") or "",
                    "cost": d.get("cost"),
                    "type": d.get("card_type"),
                    "rarity": d.get("rarity"),
                    "target": d.get("target_type"),
                    "tags": d.get("tags") or [],
                    "keywords": d.get("keywords") or [],
                    "powers": d.get("powers") or [],
                    "commands": d.get("commands") or [],
                }
        return catalog
    # Fallback: sqlite（无 description）
    if not _DB_PATH.exists():
        return {}
    db = sqlite3.connect(str(_DB_PATH))
    c = db.cursor()
    for row in c.execute(
        "SELECT id, cost, card_type, rarity, target_type, "
        "tags_json, keywords_json, powers_json, commands_json "
        "FROM cards"
    ):
        cid, cost, ct, rarity, target, tags, kw, powers, cmds = row
        catalog[cid.upper()] = {
            "id": cid, "title_en": "", "description_en": "",
            "cost": cost, "type": ct, "rarity": rarity, "target": target,
            "tags": json.loads(tags or "[]"),
            "keywords": json.loads(kw or "[]"),
            "powers": json.loads(powers or "[]"),
            "commands": json.loads(cmds or "[]"),
        }
    db.close()
    return catalog


def _annotate_card(meta: dict | None) -> str:
    """一行简短 card 机制描述。"""
    if not meta:
        return "(unknown)"
    tag_set = set(t.lower() for t in meta["tags"]) | set(k.lower() for k in meta["keywords"])
    flags = []
    if "damage" in tag_set: flags.append("ATK")
    if "block" in tag_set: flags.append("BLK")
    if "vulnerable" in tag_set: flags.append("VULN")
    if "weak" in tag_set: flags.append("WEAK")
    if "frail" in tag_set: flags.append("FRAIL")
    if "strength" in tag_set: flags.append("STR+")
    if "dexterity" in tag_set: flags.append("DEX+")
    if "draw" in tag_set or "draw" in " ".join(meta["commands"]).lower(): flags.append("DRAW")
    if "exhaust" in tag_set: flags.append("EXH")
    if "aoe" in tag_set or "all_enemies" in meta.get("target", "").lower(): flags.append("AOE")
    if meta["type"] == "Power": flags.append("PWR")
    if meta["type"] == "Skill" and not flags: flags.append("SKL")
    tag_str = "/".join(flags) if flags else meta["type"]
    return f"cost {meta['cost']} [{tag_str}]"


def load_trajectories(
    dump_dir: Path,
    n: int = 5,
    outcome_filter: str | None = None,
    room_type_filter: str | None = None,
) -> list[dict]:
    """从最新 iter 往回收集 trajectories。支持按 room_type 和 outcome 过滤。"""
    files = sorted(dump_dir.glob("iter*_trajectories.jsonl"), reverse=True)
    results: list[dict] = []
    for p in files:
        for line in p.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            s = d.get("summary") or {}
            rt = str(s.get("room_type", "")).lower()
            outcome = str(s.get("outcome", "")).lower()
            if room_type_filter and rt != room_type_filter.lower():
                continue
            if outcome_filter and outcome != outcome_filter.lower():
                continue
            results.append({
                "iter": int(p.stem[4:8]),
                "summary": s,
                "trajectory": d.get("trajectory", []),
            })
            if len(results) >= n:
                return results
    return results


# 向后兼容
load_boss_trajectories = load_trajectories


def analyze(trajectory: list[dict], catalog: dict) -> dict:
    """对一场战斗做 per-turn 分析，返回报告。"""
    by_turn: dict[int, list] = defaultdict(list)
    for step in trajectory:
        by_turn[step["turn"]].append(step)

    turns_report = []
    suspects: list[str] = []  # 可疑行为

    for turn_id in sorted(by_turn):
        steps_in_turn = by_turn[turn_id]
        first = steps_in_turn[0]
        last = steps_in_turn[-1]
        hp_start = first["hp"]
        hp_end = last["hp"]
        block_start = first["block"]
        block_max_in_turn = max(s["block"] for s in steps_in_turn)
        block_end = last["block"]
        enemy_start = first["enemy_hp_total"]
        enemy_end = last["enemy_hp_after"]
        energy_start = first["energy"]
        damage_dealt = enemy_start - enemy_end
        hp_lost_this_turn = hp_start - hp_end

        # 逐步卡序列
        play_seq = []
        for s in steps_in_turn:
            if s.get("chosen_action") == "play_card" and s.get("chosen_card"):
                card = s["chosen_card"].upper()
                meta = catalog.get(card)
                play_seq.append({
                    "card": card, "cost": meta["cost"] if meta else "?",
                    "type": meta["type"] if meta else "?",
                    "annotated": _annotate_card(meta),
                    "description": (meta or {}).get("description_en", ""),
                    "before_enemy_hp": s["enemy_hp_total"],
                    "after_enemy_hp": s["enemy_hp_after"],
                    "before_block": s["block"],
                })
            elif s.get("chosen_action") in ("end_turn", "end"):
                play_seq.append({"card": None, "type": "END_TURN",
                                 "annotated": f"END (energy left: {s['energy']})"})

        # 可疑行为检测
        # 1) 结束回合时还有 energy + 手里可能有可打的
        end_steps = [s for s in steps_in_turn if s.get("chosen_action") in ("end_turn", "end")]
        if end_steps and end_steps[0]["energy"] > 0:
            suspects.append(f"T{turn_id}: end_turn with {end_steps[0]['energy']} energy left")
        # 2) Vuln-Atk 顺序：本 turn 打了 VULN 卡后面没跟 ATK 卡
        cards_ordered = [p for p in play_seq if p.get("card")]
        for i, p in enumerate(cards_ordered):
            meta = catalog.get(p["card"])
            if not meta:
                continue
            tags = set(t.lower() for t in meta["tags"]) | set(k.lower() for k in meta["keywords"])
            if "vulnerable" in tags:
                # 检查后面有没有 ATK
                later_atk = any(
                    "damage" in (set(t.lower() for t in (catalog.get(p2["card"], {}).get("tags") or []))
                                 | set(k.lower() for k in (catalog.get(p2["card"], {}).get("keywords") or [])))
                    for p2 in cards_ordered[i+1:] if p2.get("card")
                )
                if not later_atk:
                    suspects.append(f"T{turn_id}: played VULN card {p['card']} but no ATK followed")
        # 3) 无 block 且掉血巨多：本 turn block=0 结束（没用防御资源）且下回合可能要挨打
        if block_end == 0 and block_max_in_turn == 0 and hp_lost_this_turn == 0:
            # 可能是 boss 在"charge"回合没出攻击；skip
            pass
        # 4) HP 较低 (<30%) 时仍在攻击而非防御
        hp_ratio = hp_start / max(first["max_hp"], 1)
        if hp_ratio < 0.3 and block_max_in_turn == 0:
            played_any_block = any(
                "block" in (set(t.lower() for t in (catalog.get(p["card"], {}).get("tags") or []))
                            | set(k.lower() for k in (catalog.get(p["card"], {}).get("keywords") or [])))
                for p in cards_ordered
            )
            if not played_any_block:
                suspects.append(f"T{turn_id}: HP {hp_start}/{first['max_hp']} ({int(100*hp_ratio)}%) low but no block used")

        turns_report.append({
            "turn": turn_id,
            "hp_start": hp_start, "hp_end": hp_end,
            "block_start": block_start, "block_max": block_max_in_turn, "block_end": block_end,
            "enemy_hp_start": enemy_start, "enemy_hp_end": enemy_end,
            "damage_dealt": damage_dealt,
            "hp_lost": hp_lost_this_turn,
            "energy_start": energy_start,
            "plays": play_seq,
        })

    # ---- 整场 aggregate 分析 ----
    total_energy_start = sum(
        steps_in_turn[0]["energy"] for steps_in_turn in by_turn.values()
    )
    total_energy_wasted = 0
    turns_all_end = 0  # 整回合只 end_turn 没打任何牌
    for steps_in_turn in by_turn.values():
        turn_plays = [s for s in steps_in_turn
                      if s.get("chosen_action") == "play_card" and s.get("chosen_card")]
        end_step = next((s for s in steps_in_turn
                         if s.get("chosen_action") in ("end_turn", "end")), None)
        if end_step:
            total_energy_wasted += end_step.get("energy", 0)
        if not turn_plays:
            turns_all_end += 1

    # Deck 利用：所有出过的卡 vs 常见基础卡
    cards_played = set()
    play_count = 0
    for step in trajectory:
        if step.get("chosen_action") == "play_card" and step.get("chosen_card"):
            cards_played.add(step["chosen_card"].upper())
            play_count += 1

    aggregate = {
        "n_turns": len(by_turn),
        "total_plays": play_count,
        "unique_cards_played": sorted(cards_played),
        "n_unique_played": len(cards_played),
        "turns_all_end_turn": turns_all_end,         # 整回合啥都不打
        "turn_degeneracy_rate": turns_all_end / max(len(by_turn), 1),
        "total_energy_wasted": total_energy_wasted,  # end_turn 时剩余 energy 之和
        "total_energy_start": total_energy_start,
        "energy_waste_rate": total_energy_wasted / max(total_energy_start, 1),
    }

    # 严重异常检测
    if aggregate["turn_degeneracy_rate"] >= 0.5:
        suspects.append(
            f"[SEVERE] {turns_all_end}/{len(by_turn)} turns 完全不打牌（degenerate 策略）"
        )
    if aggregate["energy_waste_rate"] >= 0.4:
        suspects.append(
            f"[SEVERE] 总能量浪费率 {100*aggregate['energy_waste_rate']:.0f}% "
            f"({total_energy_wasted}/{total_energy_start})"
        )

    return {
        "turns": turns_report,
        "suspects": suspects,
        "aggregate": aggregate,
    }


def format_report(case: dict, analysis: dict) -> str:
    s = case["summary"]
    agg = analysis.get("aggregate", {})
    out = [
        "=" * 80,
        f"iter {case['iter']} | {s['encounter_id']} | outcome={s['outcome']} | steps={s['steps']} | final_hp={s['final_hp']}/{s['max_hp']}",
        f"  [agg] {agg.get('n_turns','?')} turns | {agg.get('total_plays','?')} plays | "
        f"{agg.get('n_unique_played','?')} unique cards | "
        f"deg_turns={agg.get('turns_all_end_turn','?')}/{agg.get('n_turns','?')} "
        f"({100*agg.get('turn_degeneracy_rate',0):.0f}%) | "
        f"energy_waste={agg.get('total_energy_wasted','?')}/{agg.get('total_energy_start','?')} "
        f"({100*agg.get('energy_waste_rate',0):.0f}%)",
        f"  cards played: {agg.get('unique_cards_played', [])}",
        "=" * 80,
    ]
    for t in analysis["turns"]:
        out.append(
            f"\nT{t['turn']}: HP {t['hp_start']}→{t['hp_end']} (lost {t['hp_lost']})  "
            f"block {t['block_start']}→{t['block_max']}→{t['block_end']}  "
            f"enemy {t['enemy_hp_start']}→{t['enemy_hp_end']} (dealt {t['damage_dealt']})  "
            f"energy start={t['energy_start']}"
        )
        for p in t["plays"]:
            if p.get("card"):
                desc = p.get("description", "")
                out.append(f"    -> {p['card']:<22s} {p['annotated']:<22s}  "
                          f"enemy {p['before_enemy_hp']}->{p['after_enemy_hp']}")
                if desc:
                    out.append(f"         \"{desc[:110]}\"")
            else:
                out.append(f"    -> {p['annotated']}")
    if analysis["suspects"]:
        out.append("\n!!! SUSPECT BEHAVIOR:")
        for s in analysis["suspects"]:
            out.append(f"  - {s}")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--n", type=int, default=5, help="要分析的场数")
    ap.add_argument("--outcome", type=str, default=None,
                    choices=["victory", "defeat"], help="过滤结果类型")
    ap.add_argument("--room-type", type=str, default=None,
                    choices=["monster", "elite", "boss"],
                    help="过滤战斗类型（默认不过滤）")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        tag = f"_{args.room_type}" if args.room_type else ""
        tag += f"_{args.outcome}" if args.outcome else ""
        args.out = args.dump_dir / "analysis" / f"card_play_analysis{tag}.txt"

    catalog = _load_card_catalog()
    if not catalog:
        print("warning: card catalog empty (no sqlite?)")
    print(f"Loaded {len(catalog)} cards")

    cases = load_trajectories(
        args.dump_dir, n=args.n,
        outcome_filter=args.outcome,
        room_type_filter=args.room_type,
    )
    print(f"Collected {len(cases)} trajectories (room={args.room_type}, outcome={args.outcome})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for case in cases:
            analysis = analyze(case["trajectory"], catalog)
            report = format_report(case, analysis)
            print(report)
            f.write(report + "\n")
    print(f"\n[card_play_analysis] wrote {args.out}")


if __name__ == "__main__":
    main()
