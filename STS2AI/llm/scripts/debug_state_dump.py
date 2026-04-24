"""只启一场，把 raw state 完整 dump 出来看字段结构。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_STS2AI_ROOT = Path(__file__).resolve().parents[2]
for p in (_STS2AI_ROOT, _STS2AI_ROOT / "bridge"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from game_bridge.session import create_game_session
from llm.data_pipeline.encounter_pool import ACT1_POOL


def main() -> None:
    spec = ACT1_POOL[0]  # CHOMPERS_NORMAL starter
    with create_game_session(mode="combat", transport="pipe_proto", backend="sim", port=15560, auto_launch=True) as session:
        state = session.reset(character_id="IRONCLAD", encounter_id=spec.encounter_id, build=spec.build, seed="debug-0")
        print("--- top-level keys ---")
        print(sorted(state.keys()))
        print("\n--- player keys ---")
        print(sorted((state.get("player") or {}).keys()))
        print("\n--- battle keys ---")
        print(sorted((state.get("battle") or {}).keys()))
        battle = state.get("battle") or {}
        print("\n--- battle.player keys ---")
        print(sorted((battle.get("player") or {}).keys()))
        print("\n--- first hand card ---")
        hand = battle.get("hand") or (battle.get("player") or {}).get("hand") or []
        if hand:
            print(json.dumps(hand[0], ensure_ascii=False, indent=2))
        else:
            print("(hand empty)")
        print("\n--- first legal_action ---")
        la = state.get("legal_actions") or []
        if la:
            print(json.dumps(la[0], ensure_ascii=False, indent=2))
        print("\n--- first enemy ---")
        enemies = state.get("enemies") or battle.get("enemies") or []
        if enemies:
            print(json.dumps(enemies[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
