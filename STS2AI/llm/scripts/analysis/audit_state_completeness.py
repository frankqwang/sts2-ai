"""把 sim 回包里所有字段 dump 出来，对照当前 state_renderer 渲染的内容，
把缺失信息枚举清楚，供补全规划用。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_STS2AI_ROOT = Path(__file__).resolve().parents[3]
for p in (_STS2AI_ROOT, _STS2AI_ROOT / "bridge"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from game_bridge.session import create_game_session
from llm.data_pipeline.encounter_pool import load_skada_case_pool
from llm.data_pipeline.state_renderer import render_state_text
from llm.training.grpo_rollout import _inject_spec_context


def _summarize_keys(obj, depth=0, max_depth=3):
    """递归摘要字段结构，不打印完整值。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and depth < max_depth:
                out[k] = _summarize_keys(v, depth + 1, max_depth)
            elif isinstance(v, list):
                out[k] = f"<list len={len(v)}>"
            elif isinstance(v, dict):
                out[k] = f"<dict keys={sorted(v.keys())}>"
            elif isinstance(v, str) and len(v) > 40:
                out[k] = f"<str len={len(v)}>: {v[:40]}..."
            else:
                out[k] = v
        return out
    elif isinstance(obj, list):
        if not obj:
            return "<empty list>"
        return [_summarize_keys(obj[0], depth + 1, max_depth), f"... +{len(obj)-1}"]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", required=True, help="Skada single-combat cases.jsonl")
    parser.add_argument("--case-character", default="IRONCLAD")
    parser.add_argument("--case-floor-min", type=int, default=1)
    parser.add_argument("--case-floor-max", type=int, default=17)
    parser.add_argument("--case-sample-seed", type=int, default=0)
    parser.add_argument("--case-sample-mode", choices=["file", "random", "stratified"], default="file")
    parser.add_argument("--include-lost-cases", action="store_true")
    parser.add_argument("--seed", default="audit-1")
    parser.add_argument("--port", type=int, default=15710)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _STS2AI_ROOT / "Artifacts" / "llm" / "diagnostics" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_skada_case_pool(
        args.case_index,
        character_id=args.case_character,
        floor_min=args.case_floor_min,
        floor_max=args.case_floor_max,
        won_only=not args.include_lost_cases,
        limit=1,
        sample_seed=args.case_sample_seed,
        sample_mode=args.case_sample_mode,
    )
    if not specs:
        raise SystemExit("no Skada cases matched audit filters")
    spec = specs[0]

    session = create_game_session(mode="combat", transport="pipe_proto", backend="sim", port=args.port, auto_launch=True)
    try:
        state = session.reset(
            character_id=args.case_character,
            encounter_id=spec.encounter_id,
            build=spec.build,
            seed=args.seed,
        )
        state = _inject_spec_context(state, spec, seed=args.seed)
        # 走一步，让 state 里有真实 buff / deck 初始化完整
        legal = [a for a in state.get("legal_actions", []) if isinstance(a, dict) and a.get("is_enabled") is not False]
        if legal:
            # 找个 play_card 试试（拿到 buff 后的 state）
            for a in legal:
                if a.get("action") == "play_card" and a.get("target_id"):
                    try:
                        step_out = session.act_gym(a)
                        if isinstance(step_out, tuple):
                            state = step_out[0]
                        else:
                            state = step_out
                        state = _inject_spec_context(state, spec, seed=args.seed)
                        break
                    except Exception:
                        break
    finally:
        try:
            session.close()
        except Exception:
            pass

    # ==== 完整 raw state dump ====
    raw_path = out_dir / "raw_state.json"
    raw_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"raw state saved to: {raw_path} ({raw_path.stat().st_size} bytes)")

    # ==== 字段结构摘要 ====
    summary_path = out_dir / "state_structure.json"
    summary_path.write_text(json.dumps(_summarize_keys(state, max_depth=4), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"structure summary saved to: {summary_path}")

    # ==== 当前 renderer 输出 ====
    legal = state.get("legal_actions") or []
    rendered = render_state_text(state, legal, encounter_id=spec.encounter_id)
    rendered_path = out_dir / "rendered_prompt.txt"
    rendered_path.write_text(rendered, encoding="utf-8")
    print(f"rendered prompt saved to: {rendered_path} ({len(rendered)} chars)")

    # ==== 缺失项逐个确认 ====
    print("\n=== MISSING FIELD AUDIT ===")
    top = state
    battle = state.get("battle") or {}
    player_top = state.get("player") or {}
    battle_player = battle.get("player") or {}

    checklist = []

    # 玩家 relics
    relics = player_top.get("relics") or battle_player.get("relics") or []
    checklist.append(("player.relics", len(relics) if isinstance(relics, list) else "N/A",
                     "YES" if "relic" in rendered.lower() else "NO - 缺失"))

    # 玩家 potions
    potions = player_top.get("potions") or []
    checklist.append(("player.potions", len(potions) if isinstance(potions, list) else "N/A",
                     "YES" if "potion" in rendered.lower() else "NO - 缺失"))

    # 牌堆内容
    draw = battle.get("draw_pile_cards") or []
    discard = battle.get("discard_pile_cards") or []
    exhaust = battle.get("exhaust_pile_cards") or []
    checklist.append((f"battle.draw_pile_cards (内容)", f"{len(draw)} cards",
                     "YES" if "draw_cards:" in rendered else "NO - 缺失"))
    checklist.append((f"battle.discard_pile_cards (内容)", f"{len(discard)} cards",
                     "YES" if "discard_cards:" in rendered else "NO - 缺失"))
    checklist.append((f"battle.exhaust_pile_cards (内容)", f"{len(exhaust)} cards",
                     "YES" if "exhaust_cards:" in rendered else "NO - 缺失"))

    # 完整 deck
    deck = player_top.get("deck") or []
    checklist.append(("player.deck (全牌组)", f"{len(deck)} cards", "NO - 缺失"))

    # 玩家 powers
    ppowers = battle_player.get("powers") or player_top.get("powers") or []
    checklist.append(("player.powers", f"{len(ppowers)}",
                     "YES (id+amount) 但无描述" if ppowers else "-"))

    # 敌人 powers
    enemies = state.get("enemies") or battle.get("enemies") or []
    enemy_powers_count = sum(len(e.get("powers", []) or []) for e in enemies if isinstance(e, dict))
    checklist.append((f"enemies[*].powers", f"共 {enemy_powers_count}",
                     "YES (id+amount) 但无描述"))

    # 手牌 preview_damage / preview_block
    hand = battle.get("hand") or []
    has_preview = any(
        ("preview_damage" in c or "damage_now" in c or "effective_damage" in c)
        for c in hand if isinstance(c, dict)
    )
    checklist.append(("hand[*].preview_damage (实时伤害)", f"{len(hand)} cards",
                     "YES" if has_preview else "NO - 必须 sim C# 改"))

    # 手牌 description
    has_desc = any(
        c.get("description") or c.get("text")
        for c in hand if isinstance(c, dict)
    )
    checklist.append(("hand[*].description", f"{len(hand)} cards",
                     "YES" if has_desc else "NO - 缺失（静态表可补）"))

    # 顶层字段有什么
    top_keys = sorted(top.keys())
    checklist.append(("state top-level keys", ", ".join(top_keys), "-"))

    for name, value, status in checklist:
        print(f"  {name:<45} {str(value):<30} [{status}]")

    # ==== 敌人一个完整 dump ====
    if enemies:
        print("\n=== ENEMY[0] RAW ===")
        print(json.dumps(enemies[0], ensure_ascii=False, indent=2))

    # ==== 手牌一个完整 dump ====
    if hand:
        print("\n=== HAND[0] RAW ===")
        print(json.dumps(hand[0], ensure_ascii=False, indent=2))

    # ==== 玩家顶层字段 ====
    print("\n=== PLAYER (top) keys ===")
    print(sorted(player_top.keys()))
    print("\n=== PLAYER (battle) keys ===")
    print(sorted(battle_player.keys()))

    if relics:
        print("\n=== RELIC[0] RAW ===")
        print(json.dumps(relics[0], ensure_ascii=False, indent=2))

    if potions:
        print("\n=== POTION[0] RAW ===")
        print(json.dumps(potions[0], ensure_ascii=False, indent=2))

    if deck:
        print("\n=== DECK[0] RAW ===")
        print(json.dumps(deck[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
