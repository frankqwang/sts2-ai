from __future__ import annotations

from ..domain import BattleState, TransitionDelta


def compute_transition_delta(state: BattleState, next_state: BattleState) -> TransitionDelta:
    enemy_hp = []
    enemy_block = []
    enemy_buffs = []
    for index, enemy in enumerate(state.enemies):
        next_enemy = next_state.enemies[index] if index < len(next_state.enemies) else enemy
        enemy_hp.append(_safe_ratio_delta(next_enemy.hp - enemy.hp, enemy.max_hp))
        enemy_block.append(_safe_ratio_delta(next_enemy.block - enemy.block, max(enemy.max_hp, 20.0)))
        enemy_buffs.append(_diff_mapping(enemy.buffs, next_enemy.buffs))

    max_energy = float(state.player.resources.get("max_energy", max(state.player.energy, next_state.player.energy, 3.0)) or 3.0)
    deck_span = _deck_span(state)

    return TransitionDelta(
        self_hp=_safe_ratio_delta(next_state.player.hp - state.player.hp, state.player.max_hp),
        self_block=_safe_ratio_delta(next_state.player.block - state.player.block, max(state.player.max_hp, 20.0)),
        self_energy=_safe_ratio_delta(next_state.player.energy - state.player.energy, max_energy),
        enemy_hp=enemy_hp,
        enemy_block=enemy_block,
        self_buffs=_diff_mapping(state.player.buffs, next_state.player.buffs),
        enemy_buffs=enemy_buffs,
        hand_size=_safe_ratio_delta(len(next_state.hand) - len(state.hand), max(len(state.hand), 1)),
        draw_pile_size=_safe_ratio_delta(next_state.piles.draw_pile_size - state.piles.draw_pile_size, deck_span),
        discard_pile_size=_safe_ratio_delta(next_state.piles.discard_pile_size - state.piles.discard_pile_size, deck_span),
    )


def _diff_mapping(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in keys}


def _safe_ratio_delta(delta: float, scale: float) -> float:
    if abs(scale) <= 1e-6:
        return 0.0
    return float(delta) / float(scale)


def _deck_span(state: BattleState) -> float:
    total = (
        len(state.hand)
        + state.piles.draw_pile_size
        + state.piles.discard_pile_size
        + state.piles.exhaust_pile_size
    )
    return float(max(total, 1))
