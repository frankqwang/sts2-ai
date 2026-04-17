"""networkV2 战斗 reward 入口。

目的：V2 trainer 的 combat reward 集中到此文件；V1 老库 `core/rl_reward_shaping.py`
仅作为基础函数出处（PBRS / tactical / terminal），由本文件 re-export。
未来如要完全解耦 V1，可把基础函数迁移进来，V1 侧改 import。

组成：
- **共享层**：PBRS + tactical + terminal（来自 core.rl_reward_shaping）
- **co-trainer 独有**：dense damage/block shaping + boss-aware final + boss debuff setup bonus

Co-trainer boss-aware 强化（方向 2）：
- boss win 终局 reward × BOSS_WIN_BONUS_MULT（放大 boss 胜利信号）
- boss 败局 damage_ratio > BOSS_NEAR_LOSS_THRESHOLD → 用 BOSS_NEAR_LOSS_REWARD
  替代默认 -1.0（近胜宽容，防"一败涂地"让 agent 学成"不打 boss"）
- boss 战里动作成功挂载 Vuln/Weak 到 boss → BOSS_DEBUFF_SETUP_BONUS 额外奖励
"""

from __future__ import annotations

from typing import Any

from core.rl_reward_shaping import (
    combat_potential,
    combat_step_reward,
    combat_local_tactical_reward,
    combat_terminal_reward,
    shaped_reward,
    terminal_reward,
)

__all__ = [
    # re-exports
    "combat_potential",
    "combat_step_reward",
    "combat_local_tactical_reward",
    "combat_terminal_reward",
    "shaped_reward",
    "terminal_reward",
    # co-trainer boss-aware
    "dense_combat_shaping",
    "co_trainer_boss_damage_bonus",
    "co_trainer_boss_debuff_bonus",
    "co_trainer_final_reward",
    "boss_damage_ratio",
    # 常数
    "BOSS_WIN_BONUS_MULT",
    "BOSS_NEAR_LOSS_THRESHOLD",
    "BOSS_NEAR_LOSS_REWARD",
    "BOSS_DEBUFF_SETUP_BONUS",
    "WIN_REWARD",
    "LOSE_REWARD",
]

# ---- 常数 ----
WIN_REWARD = 1.0
LOSE_REWARD = -1.0

BOSS_WIN_BONUS_MULT: float = 2.0
BOSS_NEAR_LOSS_THRESHOLD: float = 0.7
BOSS_NEAR_LOSS_REWARD: float = -0.3
BOSS_DEBUFF_SETUP_BONUS: float = 0.02


# ---------------------------------------------------------------------------
# Helpers（和 combat_cotrainer._enemies_total_hp / _player_block 同款逻辑）
# ---------------------------------------------------------------------------

def _enemies_total_hp(state: dict) -> int:
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    return sum(
        int(e.get("hp", 0) or 0)
        for e in enemies
        if isinstance(e, dict) and e.get("is_alive", True)
    )


def _enemies_total_max_hp(state: dict) -> int:
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    total = sum(
        int(e.get("max_hp", 0) or 0)
        for e in enemies
        if isinstance(e, dict)
    )
    return max(total, 1)


def _player_block(state: dict) -> int:
    battle = state.get("battle") or {}
    player = state.get("player") or {}
    return int(battle.get("block", player.get("block", 0)) or 0)


# ---------------------------------------------------------------------------
# co-trainer 独有 shaping
# ---------------------------------------------------------------------------

def dense_combat_shaping(
    prev_state: dict,
    next_state: dict,
    player_max_hp: int,
) -> float:
    """Dense step shaping: attack + block。

    每 step 加:
      + 0.05 × damage_dealt / enemy_total_max_hp      (造伤害) [co20: 0.02 → 0.05]
      + 0.02 × block_gained / player_max_hp            (获得格挡) [co20: 0.015 → 0.02]

    量级：单场累计 +0.3~0.6，作为 zero-positive-adv trap 的主力 signal。
    对 boss (HP 170-300) 尤其重要：agent 打 10% 血也能拿到连续 reward 梯度，
    避免 "打 70% 才有 -0.3 soft reward" 的断崖式 reward landscape。
    """
    damage = max(0, _enemies_total_hp(prev_state) - _enemies_total_hp(next_state))
    enemy_max = _enemies_total_max_hp(prev_state)
    damage_reward = 0.05 * damage / enemy_max

    prev_block = _player_block(prev_state)
    next_block = _player_block(next_state)
    block_gained = max(0, next_block - prev_block)
    block_reward = 0.02 * block_gained / max(player_max_hp, 1)
    return damage_reward + block_reward


def co_trainer_boss_damage_bonus(
    prev_state: dict,
    next_state: dict,
    room_type: str,
) -> float:
    """Boss 战每 step 造成的 damage 额外奖励（按掉血百分比）。

    说明：`dense_combat_shaping` 已经给所有战斗通用的 damage reward (0.05×pct)。
    此处 boss 专属再叠加 **0.10 × damage_ratio**，让 boss 战的每 damage 信号
    比 monster 战强 3× (共 0.15 vs monster 的 0.05)。

    目的：boss HP 170-300，每点 damage 珍贵，PPO 需要更强即时信号去推 agent
    "努力打" 而非 "绝望 end_turn"。对 THE_KIN 307 HP:
      - 打掉 30% (92 dmg) → bonus 累积 +0.045（叠加 dense +0.015 共 +0.06）
      - 打掉 50% → bonus +0.075
    """
    if room_type != "boss":
        return 0.0
    damage = max(0, _enemies_total_hp(prev_state) - _enemies_total_hp(next_state))
    enemy_max = _enemies_total_max_hp(prev_state)
    return 0.10 * damage / enemy_max


def co_trainer_boss_debuff_bonus(
    state: dict,
    action: dict | None,
    room_type: str,
) -> float:
    """Boss 战内，play_card 动作成功挂载 Vulnerable/Weak 到 boss → +0.02。

    说明：`combat_local_tactical_reward` 已经给 Vuln setup 一个 +0.03 的 "顺序奖励"
    （Vuln→Attack 套路）；此处额外的 +0.02 **只在 boss 战**给，进一步放大 boss
    战里上 debuff 的价值信号。非 boss 战不给，避免 full_run 的战斗被过度偏向 debuff 套路。
    """
    if room_type != "boss":
        return 0.0
    if not isinstance(action, dict):
        return 0.0
    if str(action.get("action", "")).strip().lower() != "play_card":
        return 0.0

    card = action.get("card") or {}
    tags = card.get("tags") or []
    keywords = card.get("keywords") or []
    tag_set = {str(t).lower() for t in tags} | {str(k).lower() for k in keywords}
    if "vulnerable" in tag_set or "weak" in tag_set:
        return BOSS_DEBUFF_SETUP_BONUS
    return 0.0


def boss_damage_ratio(
    final_state: dict,
    enemy_max_hp_at_start: int,
) -> float:
    """敌人被打掉的 HP 比例（用于 boss 败局 near-win 判定）。

    final_state: 战斗结束时 state
    enemy_max_hp_at_start: 战斗开始时 enemy 总 max HP
    返回 [0, 1]，0=毫发未损，1=打掉全部
    """
    enemy_max = max(int(enemy_max_hp_at_start or 0), 1)
    remaining = _enemies_total_hp(final_state)
    dealt = max(0, enemy_max - remaining)
    return min(1.0, dealt / enemy_max)


def co_trainer_final_reward(
    won: bool,
    room_type: str,
    *,
    boss_damage_ratio: float = 0.0,
) -> float:
    """Co-trainer 终局额外 final_r（叠加在 combat_step_reward 的 terminal 之上）。

    注意：co-trainer rollout 里 terminal 奖励会加两层——一层来自
    combat_step_reward(combat_won=True/False)，另一层来自本函数的 final_r。
    这种"双层放大"是 co6+ 系列沿用的设计，让 GAE 末端有强 signal。

    Boss-aware（co20 改为连续渐变，解决 reward landscape 断崖）：
      - boss win: +2.0（= WIN_REWARD × BOSS_WIN_BONUS_MULT）
      - **boss 败**: -1.0 + 0.9 × damage_ratio  连续渐变
          - 0% 打掉: -1.0（完全没打到）
          - 30% 打掉: -0.73
          - 70% 打掉: -0.37
          - 100% 打掉: -0.1
        vs co19 旧版本: 只在 damage_ratio>0.7 才 -0.3，<70% 全 -1.0 无梯度
      - 其它（monster/elite win or loss）: ±1.0
    """
    if won:
        if room_type == "boss":
            return WIN_REWARD * BOSS_WIN_BONUS_MULT
        return WIN_REWARD
    # lost
    if room_type == "boss":
        # 连续渐变，对弱 agent 保持 reward gradient
        clamped = max(0.0, min(1.0, float(boss_damage_ratio)))
        return -1.0 + 0.9 * clamped
    return LOSE_REWARD
