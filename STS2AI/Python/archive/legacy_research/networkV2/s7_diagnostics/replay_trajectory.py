"""Trajectory 文字观战模式——逐步慢速打印 agent 决策。

用法：
  python -m networkV2.s7_diagnostics.replay_trajectory ../Artifacts/runs/co21 \\
      --room-type boss --outcome victory --delay 0.5

  # 没 delay 则一次打印全部
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


_CATALOG_JSONL = (
    Path(__file__).resolve().parents[3] / "Artifacts" / "game_knowledge" / "cards.jsonl"
)


def _load_cards() -> dict:
    cat = {}
    if _CATALOG_JSONL.exists():
        with _CATALOG_JSONL.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                cat[d["id"].upper()] = d
    return cat


def _hp_bar(cur: int, maxv: int, width: int = 20, filled="█", empty="░") -> str:
    if maxv <= 0:
        return empty * width
    ratio = max(0.0, min(1.0, cur / maxv))
    n = int(round(ratio * width))
    return filled * n + empty * (width - n)


def _find_first(dump_dir: Path, outcome: str | None, room_type: str | None) -> dict | None:
    for p in sorted(dump_dir.glob("iter*_trajectories.jsonl"), reverse=True):
        for line in p.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            s = d.get("summary", {})
            if room_type and s.get("room_type", "").lower() != room_type.lower():
                continue
            if outcome and s.get("outcome", "").lower() != outcome.lower():
                continue
            return {"iter": int(p.stem[4:8]), "summary": s, "trajectory": d.get("trajectory", [])}
    return None


def replay(case: dict, catalog: dict, delay: float = 0.0) -> None:
    s = case["summary"]
    traj = case["trajectory"]
    enc_id = s.get("encounter_id", "?").upper().replace("_BOSS", "").replace("_", " ")
    max_hp = s.get("max_hp", 80)
    print()
    print("=" * 72)
    print(f" 🎬  观战: iter {case['iter']}  |  {enc_id}  |  {s['outcome'].upper()}")
    print(f"     {s['steps']} steps, final_hp={s['final_hp']}/{max_hp}")
    print("=" * 72)

    if not traj:
        print("  (no trajectory)")
        return

    enemy_max = traj[0].get("enemy_hp_total", 0) or 1
    cur_turn = -1

    for step in traj:
        if step["turn"] != cur_turn:
            cur_turn = step["turn"]
            print(f"\n  — Turn {cur_turn} —")

        hp = step["hp"]; block = step["block"]; energy = step["energy"]
        e_hp = step["enemy_hp_total"]; e_hp_after = step["enemy_hp_after"]
        card = step.get("chosen_card", "")
        action = step.get("chosen_action", "")

        # 决策前状态
        hp_bar = _hp_bar(hp, max_hp)
        enemy_bar = _hp_bar(e_hp, enemy_max)

        state_line = (
            f"    👤 HP {hp_bar} {hp}/{max_hp}  🛡 {block}  ⚡ {energy}  "
            f"│  👹 {enemy_bar} {e_hp}"
        )
        print(state_line)

        if action == "play_card" and card:
            meta = catalog.get(card.upper(), {})
            title = meta.get("title_en", card)
            desc = (meta.get("description_en", "") or "").replace("\n", " ")[:65]
            cost = meta.get("cost", "?")
            ctype = meta.get("card_type", "")
            damage_dealt = max(0, e_hp - e_hp_after)
            dmg_note = f"  → 伤 {damage_dealt}" if damage_dealt else ""
            print(f"      🎴 {title}  [{cost}⚡ {ctype}]{dmg_note}")
            if desc:
                print(f"           \"{desc}\"")
        elif action in ("end_turn", "end"):
            print(f"      ⏭  END TURN (energy left: {energy})")
        else:
            print(f"      ▶ {action}")

        if delay > 0:
            time.sleep(delay)

    # 结局
    result_emoji = "🏆" if s["outcome"] == "victory" else "💀"
    print()
    print("=" * 72)
    print(f" {result_emoji}  {s['outcome'].upper()}  final HP {s['final_hp']}/{max_hp}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--room-type", type=str, default="boss",
                    choices=["monster", "elite", "boss"])
    ap.add_argument("--outcome", type=str, default=None,
                    choices=["victory", "defeat"])
    ap.add_argument("--delay", type=float, default=0.0,
                    help="每 step 间隔秒数；0 = 一次全打印")
    args = ap.parse_args()

    case = _find_first(args.dump_dir, args.outcome, args.room_type)
    if case is None:
        print(f"no matching trajectory (room={args.room_type}, outcome={args.outcome})")
        return
    catalog = _load_cards()
    replay(case, catalog, args.delay)


if __name__ == "__main__":
    main()
