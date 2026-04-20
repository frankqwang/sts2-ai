from __future__ import annotations

"""战斗训练用的分层评分 helper。

目标不是把 zero 训练直接改成 RL，而是先把后验质量信号接进：
- sample_weight
- keep_score
- teacher queue priority

这样模型不再只是“复制采到的动作”，而会更偏向保留、学习和重标
那些真正让战斗往胜利推进的状态。
"""

from .battle import BattleState, LegalAction
from .labels import FightLabel
from .progress import assess_transition_progress


def compute_hp_quality_score(
    label: FightLabel,
    *,
    encounter_class: str,
) -> float:
    """把 HP 状态映射成更符合战斗直觉的质量分。

    设计目标：
    - 普通战对前 10 点掉血宽容，避免“为了少掉几点血无限防御”
    - 超过安全区后惩罚非线性变陡，让低血区真正变成高风险
    - boss 战允许“拼血过战”，因此显著降低 HP 惩罚强度
    """
    encounter = (encounter_class or "").strip().lower()
    max_hp = max(1.0, float(label.player_max_hp or 0.0) or 1.0)
    current_hp = max(0.0, min(max_hp, float(label.player_hp or 0.0)))
    hp_fraction = max(0.0, min(1.0, float(label.self_hp_fraction_remaining)))
    hp_loss = max(0.0, max_hp - current_hp)

    if encounter == "boss":
        base_quality = 0.92 if label.fight_win >= 0.5 else 0.55 * hp_fraction
        overflow = max(0.0, hp_loss - 20.0)
        penalty = 0.12 * min(1.0, (overflow / max(max_hp * 0.50, 1.0)) ** 1.4)
        return max(0.0, min(1.0, base_quality - penalty))

    safe_loss = min(10.0, max_hp * 0.18)
    if hp_loss <= safe_loss:
        return 1.0

    overflow = (hp_loss - safe_loss) / max(max_hp - safe_loss, 1.0)
    penalty = min(1.0, 1.6 * (overflow**1.7))
    quality = 1.0 - penalty
    if label.fight_win < 0.5:
        quality *= max(0.35, 0.55 + 0.45 * hp_fraction)
    return max(0.0, min(1.0, float(quality)))


def compute_step_progress_score(
    previous: BattleState,
    current: BattleState,
    *,
    chosen_action: LegalAction | None = None,
) -> float:
    """量化单步转移是否在推进战斗。

    正分代表更接近赢：
    - 打掉敌方血量
    - 击杀敌人
    - 直接赢下战斗

    负分代表局面变差：
    - 自己掉血
    - 没有任何有效推进
    - 有明确输出机会却选择保守/结束回合
    """
    progress = assess_transition_progress(previous, current)
    enemy_max_hp = sum(max(1.0, enemy.max_hp) for enemy in previous.enemies) or 1.0
    self_max_hp = max(1.0, previous.player.max_hp)
    enemy_hp_component = progress.enemy_hp_delta / enemy_max_hp
    enemy_kill_component = 0.35 * max(0, progress.enemy_count_delta)
    self_hp_penalty = max(0.0, -progress.player_hp_delta) / self_max_hp
    stagnation_penalty = 0.08 if not progress.made_progress else 0.0
    opportunity_penalty = _compute_opportunity_penalty(previous, chosen_action)
    victory_bonus = 0.35 if current.terminal and str(current.run_outcome).strip().lower() in {"victory", "win"} else 0.0
    score = (
        0.90 * enemy_hp_component
        + enemy_kill_component
        + victory_bonus
        - 0.45 * self_hp_penalty
        - stagnation_penalty
        - opportunity_penalty
    )
    return max(-1.0, min(1.5, float(score)))


def compute_fight_score(
    label: FightLabel,
    *,
    encounter_class: str,
    truncated: bool,
    no_progress_ratio: float,
    max_no_progress_streak: int,
    step_count: int,
) -> float:
    """量化整场战斗质量。

    timeout 和长时间 no-progress 会被显式惩罚，避免“活着但不推进”的
    策略拿到虚高分；同时普通战会明确鼓励更快结束，避免只惩罚 timeout。
    """
    encounter = (encounter_class or "").strip().lower()
    hp_quality = compute_hp_quality_score(label, encounter_class=encounter_class)
    speed_quality = _compute_speed_quality(
        step_count=step_count,
        encounter_class=encounter_class,
        won=label.fight_win >= 0.5,
    )
    encounter_bonus = 0.12 if encounter == "elite" else 0.20 if encounter == "boss" else 0.0
    win_component = 1.10 * label.fight_win
    enemy_component = 0.55 * label.enemy_hp_fraction_dealt
    hp_component = 0.28 * hp_quality
    speed_component = 0.32 * speed_quality
    timeout_penalty = 0.85 if truncated else 0.0
    no_progress_penalty = 0.60 * max(0.0, min(1.0, no_progress_ratio))
    streak_penalty = 0.25 * min(1.0, float(max_no_progress_streak) / 64.0)
    score = (
        win_component
        + enemy_component
        + hp_component
        + speed_component
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


def _compute_speed_quality(*, step_count: int, encounter_class: str, won: bool) -> float:
    encounter = (encounter_class or "").strip().lower()
    budget = 34 if encounter == "boss" else 24 if encounter == "elite" else 16
    if step_count <= 0:
        return 0.0
    if step_count <= budget:
        fast_finish = 1.0 - 0.25 * max(0.0, float(step_count - 1) / max(budget, 1))
        return max(0.0, min(1.0, fast_finish))
    overflow = (float(step_count) - float(budget)) / max(float(budget), 1.0)
    slow_penalty = min(1.0, overflow**1.2)
    base = 0.75 if won else 0.45
    return max(0.0, min(1.0, base - 0.55 * slow_penalty))


def _compute_opportunity_penalty(previous: BattleState, chosen_action: LegalAction | None) -> float:
    if chosen_action is None:
        return 0.0

    executable_actions = [action for action in previous.legal_actions if action.can_execute]
    if not executable_actions:
        return 0.0

    damaging_actions = [action for action in executable_actions if float(action.damage_now) > 0.0]
    if not damaging_actions:
        return 0.0

    penalty = 0.0
    chosen_damage = max(0.0, float(chosen_action.damage_now))
    chosen_is_progress = chosen_damage > 0.0

    if chosen_action.action_type == "end_turn":
        penalty += 0.12

    if not chosen_is_progress and chosen_action.action_type != "play_potion":
        penalty += 0.08

    lethal_available = False
    for enemy in previous.enemies:
        if not enemy.alive:
            continue
        lethal_threshold = max(0.0, float(enemy.hp) + float(enemy.block))
        if any(float(action.damage_now) >= lethal_threshold for action in damaging_actions):
            lethal_available = True
            break
    if lethal_available and chosen_damage <= 0.0:
        penalty += 0.18

    if previous.player.energy >= 1.0 and chosen_action.action_type == "end_turn":
        non_terminal_choices = [action for action in executable_actions if action.action_type != "end_turn"]
        if non_terminal_choices:
            penalty += 0.10

    return min(0.35, penalty)
