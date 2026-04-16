"""Memory 编译器：将 memory 对象转为 token 数值特征。

TurnPrefixMemory、CombatMemory、RunBuildMemory 的状态由 env wrapper 维护。
此编译器只负责将已有的 memory 对象"翻译"成数值特征向量。
"""

from __future__ import annotations

from networkV2.s1_schema.memory import (
    TurnPrefixMemory,
    CombatMemory,
    RunBuildMemory,
    PlayedAction,
)


# 归一化常量
_MAX_TURN = 30
_MAX_HP_LOSS = 100
_MAX_CARDS = 20
_MAX_ENERGY = 10
_MAX_DAMAGE = 200
_MAX_BLOCK = 200
_MAX_DECK = 50


class MemoryCompiler:
    """将 memory 对象编译为数值特征向量。"""

    def compile_turn_prefix_summary(self, prefix: TurnPrefixMemory) -> list[float]:
        """编译 turn prefix 摘要为数值向量。"""
        return [
            prefix.cards_played / 10.0,
            prefix.attacks_played / 10.0,
            prefix.skills_played / 10.0,
            prefix.powers_played / 5.0,
            prefix.energy_spent / _MAX_ENERGY,
            prefix.total_damage_dealt / _MAX_DAMAGE,
            prefix.total_block_gained / _MAX_BLOCK,
            prefix.total_cards_drawn / _MAX_CARDS,
            prefix.total_exhaust / 5.0,
            prefix.potions_used / 3.0,
            # 打牌比例
            prefix.attacks_played / max(prefix.cards_played, 1),
            prefix.skills_played / max(prefix.cards_played, 1),
        ]

    def compile_played_action(self, action: PlayedAction) -> list[float]:
        """编译单个已打出动作为数值向量。"""
        return [
            float(action.action_type == "play_card"),
            float(action.action_type == "use_potion"),
            float(action.card_type == "attack"),
            float(action.card_type == "skill"),
            float(action.card_type == "power"),
            action.cost / _MAX_ENERGY,
            action.damage_dealt / _MAX_DAMAGE,
            action.block_gained / _MAX_BLOCK,
            action.cards_drawn / 5.0,
            float(action.energy_delta) / _MAX_ENERGY,
            float(action.was_exhaust),
            float(action.was_retain),
        ]

    def compile_combat_memory(self, mem: CombatMemory) -> list[float]:
        """编译战斗长程记忆为数值向量。

        注意：fight_mode 的 5 维 one-hot 已被移除 —— 该字段没有任何写入路径，
        过去恒等于 UNKNOWN（即 4 维恒 0 + 1 维恒 1），属于典型死通道。
        FightMode 分类本质是基于 HP 趋势 / 敌人残血 / intents / powers 的衍生信号，
        这些原始信号都在网络输入里，让网络自学比人拍阈值更稳妥。

        behavior_history / transition_count：追踪敌人行为切换序列，
        当前以 next_move_id 变化作为代理信号（bridge 暂未暴露 phase_id）。
        """
        # 本回合敌方最新伤害（时间序列尾部）
        last_enemy_hp_loss = mem.recent_enemy_hp_loss_window[-1] if mem.recent_enemy_hp_loss_window else 0
        last_intent_damage = mem.recent_intent_damage_window[-1] if mem.recent_intent_damage_window else 0
        return [
            mem.turn_index / _MAX_TURN,
            mem.cumulative_hp_loss / _MAX_HP_LOSS,
            mem.recent_hp_loss_avg / 20.0,
            mem.potions_used / 5.0,
            len(mem.behavior_history) / 5.0,
            mem.transition_count / 5.0,
            mem.reshuffle_count / 10.0,
            mem.exhaust_total / _MAX_CARDS,
            mem.total_damage_dealt / 500.0,
            mem.total_block_gained / 500.0,
            mem.max_single_turn_damage / _MAX_DAMAGE,
            # 敌方时序（新增 4 维）
            mem.recent_enemy_hp_loss_avg / 30.0,
            last_enemy_hp_loss / 30.0,
            mem.recent_intent_damage_avg / _MAX_DAMAGE,
            last_intent_damage / _MAX_DAMAGE,
            # 防守效率（1 维）：主动挡掉的伤害比例
            mem.block_efficiency,
        ]

    def compile_run_build_memory(self, mem: RunBuildMemory) -> list[float]:
        """编译整局慢变量为数值向量。"""
        return [
            # 构筑画像
            mem.deck_size / _MAX_DECK,
            mem.frontload,
            mem.block,
            mem.draw,
            mem.scaling,
            mem.aoe,
            mem.heal,
            mem.curse_density,
            mem.high_cost_density,
            mem.zero_cost_density,
            mem.x_cost_density,
            mem.consistency,
            # 目标上下文
            mem.survival_priority,
            mem.resource_priority,
            mem.preserve_hp_bias,
            mem.boss_pressure,
            mem.elite_pressure,
            # 局信息
            mem.act / 4.0,
            mem.floor / 60.0,
            mem.gold / 500.0,
            mem.relic_count / 25.0,
            mem.potion_count / 5.0,
            # 累计统计
            mem.combats_seen / 20.0,
            mem.elites_seen / 5.0,
            mem.bosses_seen / 3.0,
            mem.total_hp_lost / _MAX_HP_LOSS,
            mem.potions_used_total / 10.0,
            # 跨战斗/跨房间历史（新增 5 维 → 保持 token numeric ≤ 32）
            mem.unique_enemy_types_seen / 15.0,      # 敌人多样性
            mem.room_type_ratio("monster"),          # 战斗房占比（combat 侧已有 combats_seen 做绝对值）
            mem.room_type_ratio("elite"),            # elite 占比
            mem.room_type_ratio("rest"),             # 非战斗 rest 占比（反映 HP 恢复机会）
            mem.unique_events_seen / 10.0,           # 事件多样性
        ]
