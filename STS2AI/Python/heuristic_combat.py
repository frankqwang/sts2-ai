"""Heuristic combat strategy — rule-based card play for data collection.

Not used in production training (pure PPO mode). Kept as utility for:
- Warm-start data collection (if NN is too weak to explore)
- Baseline comparison (heuristic vs NN win rate)
- Debugging (manually run a game with heuristic to verify env works)

Usage:
    from heuristic_combat import heuristic_combat_action
    action_idx, action = heuristic_combat_action(legal_actions, state)
"""
from __future__ import annotations
from typing import Any


def heuristic_combat_action(legal: list[dict], state: dict) -> tuple[int, dict]:
    """Simple rule-based combat action selection.

    Rules (priority order):
    1. Play attack cards first (kill enemies = win)
    2. Play block cards if enemy intends to attack
    3. Play any remaining playable card (powers, skills)
    4. End turn

    Returns (action_index, action_dict).
    """
    import random as _rng

    player = state.get("player") or {}
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or []

    attacks = []
    blocks = []
    others = []
    end_turn_idx = None

    for i, a in enumerate(legal):
        act = (a.get("action") or "").lower()
        label = (a.get("label") or "").lower()

        if act == "end_turn":
            end_turn_idx = i
            continue
        if act != "play_card":
            others.append(i)
            continue

        is_attack = any(k in label for k in (
            "打击", "痛击", "重击", "旋风斩", "猛击",
            "strike", "bash", "heavy", "whirlwind", "pummel",
            "铁浪", "暴怒", "怒火", "撕裂", "劈砍",
            "anger", "cleave", "iron wave", "carnage", "rampage",
            "blood", "body slam", "headbutt", "clothesline",
        ))
        is_block = any(k in label for k in (
            "防御", "护甲", "格挡", "铁壁",
            "defend", "block", "armor", "shrug", "sentinel",
            "耸肩", "真步", "自我修复",
        ))

        if is_attack:
            attacks.append(i)
        elif is_block:
            blocks.append(i)
        else:
            others.append(i)

    enemy_attacking = any(
        "attack" in (intent.get("type") or "").lower()
        for e in enemies
        for intent in (e.get("intents") or [])
    )

    if attacks:
        return _rng.choice(attacks), legal[_rng.choice(attacks)]
    if blocks and enemy_attacking:
        return _rng.choice(blocks), legal[_rng.choice(blocks)]
    if others:
        return _rng.choice(others), legal[_rng.choice(others)]
    if blocks:
        return _rng.choice(blocks), legal[_rng.choice(blocks)]
    if end_turn_idx is not None:
        return end_turn_idx, legal[end_turn_idx]

    idx = _rng.randrange(len(legal))
    return idx, legal[idx]
