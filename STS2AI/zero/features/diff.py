from __future__ import annotations

from ..domain import BattleState, TransitionDelta


def compute_transition_delta(state: BattleState, next_state: BattleState) -> TransitionDelta:
    enemy_hp = []
    enemy_block = []
    enemy_buffs = []
    for index, enemy in enumerate(state.enemies):
        next_enemy = next_state.enemies[index] if index < len(next_state.enemies) else enemy
        enemy_hp.append(float(next_enemy.hp - enemy.hp))
        enemy_block.append(float(next_enemy.block - enemy.block))
        enemy_buffs.append(_diff_mapping(enemy.buffs, next_enemy.buffs))

    return TransitionDelta(
        self_hp=float(next_state.player.hp - state.player.hp),
        self_block=float(next_state.player.block - state.player.block),
        self_energy=float(next_state.player.energy - state.player.energy),
        enemy_hp=enemy_hp,
        enemy_block=enemy_block,
        self_buffs=_diff_mapping(state.player.buffs, next_state.player.buffs),
        enemy_buffs=enemy_buffs,
        hand_size=float(len(next_state.hand) - len(state.hand)),
        draw_pile_size=float(next_state.piles.draw_pile_size - state.piles.draw_pile_size),
        discard_pile_size=float(next_state.piles.discard_pile_size - state.piles.discard_pile_size),
    )


def _diff_mapping(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in keys}
