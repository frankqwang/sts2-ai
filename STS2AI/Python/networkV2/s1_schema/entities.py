"""实体定义：EntitySemantics（本体语义）和 RuntimeInstances（运行时状态）。

EntitySemantics: 实体"是什么"，不含运行时状态。来源于 vocab / source_knowledge。
RuntimeInstances: 实体"当前状态"，来源于 bridge 运行时数据。

Buff 三级分层:
  Level 1 (通用 power): 存在 powers dict 里，如 strength/weak/vulnerable
  Level 2 (特殊行为 power): 由 auto_modifier_rules 映射为 RuleModifier
  Level 3 (boss 复杂机制): 由 mechanism_config 配置
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# EntitySemantics - 实体本体（静态属性，不含运行时状态）
# ---------------------------------------------------------------------------

@dataclass
class CardSemantics:
    """卡牌本体语义。"""
    entity_id: str = ""
    card_type: str = ""       # "attack" / "skill" / "power" / "status" / "curse"
    rarity: str = ""          # "common" / "uncommon" / "rare" / "basic" / "special"
    base_cost: int = 0
    is_upgraded: bool = False
    tags: list[str] = field(default_factory=list)       # 功能标签
    keywords: list[str] = field(default_factory=list)    # 关键词 (exhaust, ethereal, retain, innate...)


@dataclass
class RelicSemantics:
    """遗物本体语义。"""
    entity_id: str = ""
    relic_tags: list[str] = field(default_factory=list)
    functional_signals: dict[str, float] = field(default_factory=dict)
    # signals: energy, draw, strength, dexterity, vigor, thorns,
    #          intangible, artifact, poison, defense, regen


@dataclass
class PotionSemantics:
    """药水本体语义。"""
    entity_id: str = ""
    potion_type: str = ""     # "attack" / "block" / "buff" / "heal" / "energy" / "special"
    tags: list[str] = field(default_factory=list)


@dataclass
class EnemySemantics:
    """敌人本体语义。"""
    entity_id: str = ""
    is_boss: bool = False
    is_elite: bool = False
    is_minion: bool = False


# ---------------------------------------------------------------------------
# RuntimeInstances - 运行时实例状态
# ---------------------------------------------------------------------------

@dataclass
class IntentInfo:
    """敌人意图信息。"""
    intent_type: str = ""     # "attack" / "defend" / "buff" / "debuff" / "unknown"
    damage: int = 0           # 单次伤害
    total_damage: int = 0     # 总伤害
    repeats: int = 1          # 攻击次数


@dataclass
class PlayerRuntime:
    """玩家运行时状态。"""
    hp: int = 0
    max_hp: int = 1
    block: int = 0
    energy: int = 0
    max_energy: int = 3
    # Level 1 通用 power: name → stacks
    powers: dict[str, int] = field(default_factory=dict)

    @property
    def hp_ratio(self) -> float:
        return self.hp / max(self.max_hp, 1)


@dataclass
class HandCardRuntime:
    """手牌运行时状态。"""
    card_id: str = ""
    hand_index: int = 0
    current_cost: int = 0
    card_type: str = ""
    is_upgraded: bool = False
    upgrade_count: int = 0        # 升级次数（某些卡可多次升级，is_upgraded 是 >0 的压扁）
    rarity: str = ""              # "basic" / "common" / "uncommon" / "rare" / "special" / "curse"
    target_type: str = ""         # "none" / "enemy" / "self" / "all_enemies" / "random_enemy" / ...
    can_play: bool = False
    requires_target: bool = False
    valid_target_ids: list[int] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # preview 数值（来自 bridge 或 action preview）
    damage_est: float = 0.0
    block_est: float = 0.0
    draw_est: float = 0.0
    # 特殊标记
    retain: bool = False
    ethereal: bool = False
    exhaust: bool = False


@dataclass
class EnemyRuntime:
    """敌人运行时状态。"""
    entity_id: str = ""       # 唯一标识: "{enemy_id}_{combat_id}"
    enemy_id: str = ""        # 类型标识: "jaw_worm" / "hexaghost"
    combat_id: int = -1       # 战斗内实例 ID
    name: str = ""
    hp: int = 0
    max_hp: int = 1
    block: int = 0
    is_alive: bool = True
    is_hittable: bool = True
    intends_to_attack: bool = False
    next_move_id: str | None = None
    intents: list[IntentInfo] = field(default_factory=list)
    # Level 1 通用 power: name → stacks
    powers: dict[str, int] = field(default_factory=dict)

    @property
    def hp_ratio(self) -> float:
        return self.hp / max(self.max_hp, 1)

    def has_buff(self, buff_name: str) -> bool:
        """检查是否拥有某个 buff（不区分大小写）。"""
        lower = buff_name.lower()
        return any(k.lower() == lower for k in self.powers if self.powers[k] > 0)

    def buff_stacks(self, buff_name: str) -> int:
        """获取某个 buff 的层数。"""
        lower = buff_name.lower()
        for k, v in self.powers.items():
            if k.lower() == lower:
                return v
        return 0

    @property
    def total_intent_damage(self) -> int:
        return sum(
            i.total_damage if i.total_damage > 0 else i.damage * i.repeats
            for i in self.intents
            if i.intent_type == "attack"
        )

    @property
    def minion_count(self) -> int:
        """本实例是否是 minion（需要外部传入，这里只看 power）。"""
        return self.powers.get("minion", 0)


@dataclass
class PileSummary:
    """牌堆摘要。"""
    pile_type: str = ""       # "draw" / "discard" / "exhaust" / "hand" / "deck"
    size: int = 0
    attack_count: int = 0
    skill_count: int = 0
    power_count: int = 0
    curse_count: int = 0
    status_count: int = 0
    zero_cost_count: int = 0
    # 关键牌追踪
    key_card_ids: list[str] = field(default_factory=list)

    @property
    def attack_ratio(self) -> float:
        return self.attack_count / max(self.size, 1)

    @property
    def skill_ratio(self) -> float:
        return self.skill_count / max(self.size, 1)

    @property
    def zero_cost_density(self) -> float:
        return self.zero_cost_count / max(self.size, 1)

    @property
    def reshuffle_proximity(self) -> float:
        """接近洗牌的程度：draw pile 越小越接近 1.0。"""
        if self.pile_type != "draw":
            return 0.0
        return 1.0 - min(self.size / 20.0, 1.0)
