"""时间记忆层定义：三个时间尺度的记忆。

TurnPrefixMemory: 本回合高分辨率历史（快变量）
CombatMemory: 本战斗长程摘要（中变量）
RunBuildMemory: 整局慢变量背景（慢变量）

TurnPrefix 和 CombatMemory 的状态由 CombatEnvWrapper 维护，
Compiler 只读取，不修改。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# TurnPrefixMemory - 本回合高分辨率历史
# ---------------------------------------------------------------------------

@dataclass
class PlayedAction:
    """本回合内一次已执行的动作。"""
    action_type: str = ""     # "play_card" / "use_potion" / "end_turn"
    card_id: str = ""
    card_type: str = ""       # "attack" / "skill" / "power"
    target_id: str = ""
    cost: int = 0
    damage_dealt: float = 0.0
    block_gained: float = 0.0
    cards_drawn: int = 0
    energy_delta: int = 0
    was_exhaust: bool = False
    was_retain: bool = False


@dataclass
class TurnPrefixMemory:
    """本回合高分辨率历史。"""
    played_actions: list[PlayedAction] = field(default_factory=list)
    # 累计统计
    cards_played: int = 0
    attacks_played: int = 0
    skills_played: int = 0
    powers_played: int = 0
    energy_spent: int = 0
    total_damage_dealt: float = 0.0
    total_block_gained: float = 0.0
    total_cards_drawn: int = 0
    total_exhaust: int = 0
    potions_used: int = 0

    def record_action(self, action: PlayedAction) -> None:
        """记录一次动作并更新累计统计。

        注意：cards_played / attacks_played / skills_played / powers_played / energy_spent / total_exhaust
        仅对 play_card 动作累加；potion 动作只更新 potions_used 与效果量 (damage/block/draw)，
        避免污染 turn-prefix 的卡牌统计。
        """
        self.played_actions.append(action)
        if action.action_type == "play_card":
            self.cards_played += 1
            if action.card_type == "attack":
                self.attacks_played += 1
            elif action.card_type == "skill":
                self.skills_played += 1
            elif action.card_type == "power":
                self.powers_played += 1
            self.energy_spent += action.cost
            if action.was_exhaust:
                self.total_exhaust += 1
        elif action.action_type in ("use_potion", "drink_potion"):
            self.potions_used += 1
        # 效果量对所有动作通用累加（potion 也可能造成 damage/block/draw）
        self.total_damage_dealt += action.damage_dealt
        self.total_block_gained += action.block_gained
        self.total_cards_drawn += action.cards_drawn

    def reset(self) -> None:
        """新回合开始时清零。"""
        self.played_actions.clear()
        self.cards_played = 0
        self.attacks_played = 0
        self.skills_played = 0
        self.powers_played = 0
        self.energy_spent = 0
        self.total_damage_dealt = 0.0
        self.total_block_gained = 0.0
        self.total_cards_drawn = 0
        self.total_exhaust = 0
        self.potions_used = 0


# ---------------------------------------------------------------------------
# CombatMemory - 本战斗长程摘要
# ---------------------------------------------------------------------------

# U4 修复：原 FightMode enum（RACE/STABILIZE/ATTRITION/BURST_PREP/UNKNOWN）和
# `fight_mode` 字段是死通道——运行时没有任何写入路径，恒为 UNKNOWN。
# memory_encoder.py 已经把它从 token encoding 里移除。现在彻底删掉避免误导后续开发。
# 若后续需要 FightMode 分类，应从 HP 趋势/敌人残血/intent 等原始信号通过网络自学，
# 不应再加显式分类字段。


@dataclass
class CombatMemory:
    """本战斗长程摘要。由 CombatEnvWrapper 每步更新。"""
    turn_index: int = 0
    cumulative_hp_loss: int = 0
    recent_hp_loss_window: list[int] = field(default_factory=list)  # 近 N 回合玩家掉血
    # 敌方视角时序：和 recent_hp_loss_window 对称，让网络看到"这场仗的 DPS 是否达标"
    recent_enemy_hp_loss_window: list[int] = field(default_factory=list)
    recent_intent_damage_window: list[int] = field(default_factory=list)  # 每回合敌人总 intent 伤害
    potions_used: int = 0
    # 敌人行为序列：存 "敌人id:move_id" 这样的字符串，用于追踪"进入新阶段"事件
    # 原名 phase_history；现用 move_id 变化作为代理信号（bridge 暂不暴露 phase_id）
    behavior_history: list[str] = field(default_factory=list)
    reshuffle_count: int = 0
    exhaust_total: int = 0
    # 辅助统计
    total_damage_dealt: float = 0.0
    total_block_gained: float = 0.0
    max_single_turn_damage: float = 0.0

    _WINDOW = 5

    def on_new_turn(
        self,
        hp_loss_this_turn: int,
        enemy_hp_loss_this_turn: int = 0,
        intent_damage_this_turn: int = 0,
    ) -> None:
        """每回合结束时调用。"""
        self.turn_index += 1
        self.cumulative_hp_loss += hp_loss_this_turn
        self.recent_hp_loss_window.append(hp_loss_this_turn)
        self.recent_enemy_hp_loss_window.append(enemy_hp_loss_this_turn)
        self.recent_intent_damage_window.append(intent_damage_this_turn)
        for w in (self.recent_hp_loss_window,
                  self.recent_enemy_hp_loss_window,
                  self.recent_intent_damage_window):
            if len(w) > self._WINDOW:
                w.pop(0)

    def on_behavior_change(self, behavior_id: str) -> None:
        """敌人行为切换时调用（move_id 变化或真正的 phase 变化都走这里）。"""
        if not self.behavior_history or self.behavior_history[-1] != behavior_id:
            self.behavior_history.append(behavior_id)

    def reset(self) -> None:
        """新战斗开始时清零。"""
        self.turn_index = 0
        self.cumulative_hp_loss = 0
        self.recent_hp_loss_window.clear()
        self.recent_enemy_hp_loss_window.clear()
        self.recent_intent_damage_window.clear()
        self.potions_used = 0
        self.behavior_history.clear()
        self.reshuffle_count = 0
        self.exhaust_total = 0
        self.total_damage_dealt = 0.0
        self.total_block_gained = 0.0
        self.max_single_turn_damage = 0.0

    @staticmethod
    def _avg(w: list[int]) -> float:
        return sum(w) / len(w) if w else 0.0

    @property
    def recent_hp_loss_avg(self) -> float:
        return self._avg(self.recent_hp_loss_window)

    @property
    def recent_enemy_hp_loss_avg(self) -> float:
        return self._avg(self.recent_enemy_hp_loss_window)

    @property
    def recent_intent_damage_avg(self) -> float:
        return self._avg(self.recent_intent_damage_window)

    @property
    def block_efficiency(self) -> float:
        """防守效率：本战斗主动挡掉的伤害 / (主动 block + 实际掉血)。

        = 1.0 → 完美挡伤害（zero HP loss）
        = 0.5 → 挡掉一半伤害
        = 0.0 → 完全没 block，硬吃伤害
        """
        total = self.total_block_gained + self.cumulative_hp_loss
        if total <= 0:
            return 0.5
        return self.total_block_gained / total

    @property
    def transition_count(self) -> int:
        """行为切换次数 = len(behavior_history) - 1（首次进入不算切换）。"""
        return max(len(self.behavior_history) - 1, 0)


# ---------------------------------------------------------------------------
# RunBuildMemory - 整局慢变量
# ---------------------------------------------------------------------------

@dataclass
class RunBuildMemory:
    """整局慢变量背景。

    复用现有 run_memory / build_profile / objective_context 的语义，
    但重新组织为正式 slow memory 输入。
    """
    # 构筑画像
    build_identity: str = ""    # "frontload_aggro" / "block_turtle" / "scaling" / ...
    deck_size: int = 0
    frontload: float = 0.0
    block: float = 0.0
    draw: float = 0.0
    scaling: float = 0.0
    aoe: float = 0.0
    heal: float = 0.0
    curse_density: float = 0.0
    high_cost_density: float = 0.0
    zero_cost_density: float = 0.0
    x_cost_density: float = 0.0
    consistency: float = 0.0

    # 目标上下文
    survival_priority: float = 0.0
    resource_priority: float = 0.0
    preserve_hp_bias: float = 0.0
    boss_pressure: float = 0.0
    elite_pressure: float = 0.0

    # 局信息
    act: int = 1
    floor: int = 0
    gold: int = 0
    relic_count: int = 0
    potion_count: int = 0

    # 累计统计
    combats_seen: int = 0
    elites_seen: int = 0
    bosses_seen: int = 0
    total_hp_lost: int = 0
    potions_used_total: int = 0

    # 跨战斗/跨房间历史（帮助网络感知"已经历什么路径、遇过什么敌人、做过什么选择"）
    enemy_types_seen: dict[str, int] = field(default_factory=dict)  # enemy_id → 遇到次数
    room_type_history: list[str] = field(default_factory=list)      # monster/elite/boss/shop/rest/event/map
    event_history: list[str] = field(default_factory=list)          # event_id 序列

    def register_combat(self, enemy_ids: list[str], room_type: str) -> None:
        """战斗开始时登记：遇到的敌人 + 房间类型。"""
        for eid in enemy_ids:
            if not eid:
                continue
            self.enemy_types_seen[eid] = self.enemy_types_seen.get(eid, 0) + 1
        if room_type:
            self.room_type_history.append(room_type)

    def register_room(self, room_type: str) -> None:
        """非战斗房间登记（shop/rest/event/map）。"""
        if room_type:
            self.room_type_history.append(room_type)

    def register_event(self, event_id: str) -> None:
        """事件登记。"""
        if event_id:
            self.event_history.append(event_id)

    @property
    def unique_enemy_types_seen(self) -> int:
        return len(self.enemy_types_seen)

    @property
    def unique_events_seen(self) -> int:
        return len(set(self.event_history))

    def room_type_ratio(self, room_type: str) -> float:
        if not self.room_type_history:
            return 0.0
        return sum(1 for r in self.room_type_history if r == room_type) / len(self.room_type_history)
