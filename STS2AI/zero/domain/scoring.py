from __future__ import annotations

"""战斗训练用的分层评分 helper。

目标不是把 zero 训练直接改成 RL，而是先把后验质量信号接进：
- sample_weight
- keep_score
- teacher queue priority

这样模型不再只是“复制采到的动作”，而会更偏向保留、学习和重标
那些真正让战斗往胜利推进的状态。
"""

from .battle import BattleState
from .labels import FightLabel
from .progress import assess_transition_progress


def compute_step_progress_score(previous: BattleState, current: BattleState) -> float:
    """量化单步转移是否在推进战斗。

    正分代表更接近赢：
    - 打掉敌方血量
    - 击杀敌人
    - 直接赢下战斗

    负分代表局面变差：
    - 自己掉血
    - 没有任何有效推进
    """
    progress = assess_transition_progress(previous, current)
    enemy_max_hp = sum(max(1.0, enemy.max_hp) for enemy in previous.enemies) or 1.0
    self_max_hp = max(1.0, previous.player.max_hp)
    enemy_hp_component = progress.enemy_hp_delta / enemy_max_hp
    enemy_kill_component = 0.35 * max(0, progress.enemy_count_delta)
    self_hp_penalty = max(0.0, -progress.player_hp_delta) / self_max_hp
    stagnation_penalty = 0.08 if not progress.made_progress else 0.0
    victory_bonus = 0.35 if current.terminal and str(current.run_outcome).strip().lower() in {"victory", "win"} else 0.0
    score = (
        0.90 * enemy_hp_component
        + enemy_kill_component
        + victory_bonus
        - 0.45 * self_hp_penalty
        - stagnation_penalty
    )
    return max(-1.0, min(1.5, float(score)))


def compute_fight_score(
    label: FightLabel,
    *,
    encounter_class: str,
    truncated: bool,
    no_progress_ratio: float,
    max_no_progress_streak: int,
) -> float:
    """量化整场战斗质量。

    普通战更看重剩余血量；
    boss 战允许“拼光血过战”，因此降低 HP 维度权重。

    timeout 和长时间 no-progress 会被显式惩罚，避免“活着但不推进”的
    策略拿到虚高分。
    """
    encounter = (encounter_class or "").strip().lower()
    hp_weight = 0.15 if encounter == "boss" else 0.35
    encounter_bonus = 0.12 if encounter == "elite" else 0.20 if encounter == "boss" else 0.0
    win_component = 1.20 * label.fight_win
    enemy_component = 0.70 * label.enemy_hp_fraction_dealt
    hp_component = hp_weight * label.self_hp_fraction_remaining
    timeout_penalty = 0.85 if truncated else 0.0
    no_progress_penalty = 0.60 * max(0.0, min(1.0, no_progress_ratio))
    streak_penalty = 0.25 * min(1.0, float(max_no_progress_streak) / 64.0)
    score = (
        win_component
        + enemy_component
        + hp_component
        + encounter_bonus * label.fight_win
        - timeout_penalty
        - no_progress_penalty
        - streak_penalty
    )
    return max(-1.0, min(2.5, float(score)))


def compute_episode_score_proxy(
    *,
    fight_score: float,
    floor: int,
    encounter_class: str,
) -> float:
    """在还没有完整 run return 前，给出一个可训练用的整局进度代理分。

    当前 collect 单位还是 combat，因此这里不伪装成真实 episode return。
    先用：
    - 当前战斗质量
    - 所处楼层深度
    - elite / boss 的长期价值

    作为更长程的“整局进展”代理信号。
    """
    floor_component = min(1.0, max(0.0, float(floor) / 30.0))
    encounter = (encounter_class or "").strip().lower()
    encounter_component = 0.15 if encounter == "elite" else 0.25 if encounter == "boss" else 0.0
    score = 0.75 * fight_score + 0.25 * floor_component + encounter_component
    return max(-1.0, min(2.5, float(score)))
