"""只启一场，把 raw state 完整 dump 出来看字段结构。"""
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
from llm.training.grpo_rollout import _inject_spec_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", required=True, help="Skada single-combat cases.jsonl")
    parser.add_argument("--case-character", default="IRONCLAD")
    parser.add_argument("--case-floor-min", type=int, default=1)
    parser.add_argument("--case-floor-max", type=int, default=17)
    parser.add_argument("--case-sample-seed", type=int, default=0)
    parser.add_argument("--case-sample-mode", choices=["file", "random", "stratified"], default="file")
    parser.add_argument("--include-lost-cases", action="store_true")
    parser.add_argument("--seed", default="debug-0")
    parser.add_argument("--port", type=int, default=15560)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        raise SystemExit("no Skada cases matched debug filters")
    spec = specs[0]
    with create_game_session(mode="combat", transport="pipe_proto", backend="sim", port=args.port, auto_launch=True) as session:
        state = session.reset(character_id=args.case_character, encounter_id=spec.encounter_id, build=spec.build, seed=args.seed)
        state = _inject_spec_context(state, spec, seed=args.seed)
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
