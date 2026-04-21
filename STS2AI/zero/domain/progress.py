from __future__ import annotations

"""战斗推进信号的共享 helper。

collector 和 evaluator 都会关心“这一手有没有让战斗往结束方向推进”，
这里集中定义，避免两处各写一套 no-progress 统计。
"""

from dataclasses import dataclass

from .battle import BattleState


@dataclass(slots=True)
class ProgressSignal:
    enemy_hp_delta: float = 0.0
    enemy_count_delta: int = 0
    player_hp_delta: float = 0.0
    made_progress: bool = False


def assess_transition_progress(previous: BattleState, current: BattleState) -> ProgressSignal:
    previous_enemy_hp = _total_enemy_hp(previous)
    current_enemy_hp = _total_enemy_hp(current)
    enemy_hp_delta = previous_enemy_hp - current_enemy_hp
    enemy_count_delta = len(previous.living_enemies) - len(current.living_enemies)
    player_hp_delta = current.player.hp - previous.player.hp
    made_progress = bool(
        enemy_hp_delta > 1e-6
        or enemy_count_delta > 0
        or (current.terminal and str(current.run_outcome).strip().lower() in {"victory", "win"})
    )
    return ProgressSignal(
        enemy_hp_delta=float(enemy_hp_delta),
        enemy_count_delta=int(enemy_count_delta),
        player_hp_delta=float(player_hp_delta),
        made_progress=made_progress,
    )


def _total_enemy_hp(state: BattleState) -> float:
    return float(sum(max(0.0, enemy.hp) for enemy in state.enemies))
