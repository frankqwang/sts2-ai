"""合法动作候选定义。

RawActionCandidates 表示当前合法动作集合，每个动作带有：
  - 动作类型和目标
  - preview 数值估计
  - 语义签名（family, target_scope, roles）

这些信息最终会被编译成 action_bank 中的 token。
ActionHypothesis 不在此处定义——它是网络 Action Contextualizer 的产物。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 动作语义枚举（字符串常量，避免枚举膨胀）
# ---------------------------------------------------------------------------

# 动作 family
ACTION_FAMILIES = [
    "play_card", "use_potion", "discard_potion",
    "end_turn", "proceed",
    "card_selection", "hand_selection",
    "confirm", "cancel",
    "map", "reward", "card_reward", "shop", "rest", "smith",
    "event_option", "other",
]

# 目标 scope
TARGET_SCOPES = [
    "none", "self", "single_enemy", "all_enemies",
    "choice", "map", "shop", "event", "other",
]

# 语义角色（可多选）
SEMANTIC_ROLES = [
    "attack", "block", "draw", "debuff", "buff", "heal",
    "aoe", "x_cost", "setup", "scaling", "resource", "terminal",
    "exhaust", "retain", "ethereal",
]


# ---------------------------------------------------------------------------
# ActionCandidate
# ---------------------------------------------------------------------------

@dataclass
class ActionCandidate:
    """一个合法动作候选。"""
    # 基础标识
    action_type: str = ""         # "play_card" / "end_turn" / "use_potion" / ...
    action_index: int = 0         # 在 legal_actions 列表中的原始索引
    label: str = ""               # 可读标签

    # 源与目标
    source_card_id: str = ""      # 打出的卡牌 ID（如果是 play_card）
    source_card_type: str = ""    # "attack" / "skill" / "power"
    source_potion_id: str = ""    # 使用的药水 ID（如果是 use_potion）
    target_enemy_id: str = ""     # 目标敌人 entity_id
    target_card_id: str = ""      # 目标卡牌 ID（如果是 selection）
    hand_index: int = -1
    target_combat_id: int = -1

    # 语义签名
    family: str = "other"         # ACTION_FAMILIES 中的一个
    target_scope: str = "none"    # TARGET_SCOPES 中的一个
    roles: list[str] = field(default_factory=list)  # SEMANTIC_ROLES 子集

    # Preview 数值估计
    cost: int = 0
    damage_est: float = 0.0
    block_est: float = 0.0
    draw_est: float = 0.0
    heal_est: float = 0.0
    energy_delta: int = 0

    # 标记
    is_zero_cost: bool = False
    is_x_cost: bool = False
    exhausts: bool = False
    retains: bool = False
    ends_turn: bool = False

    @property
    def is_play_card(self) -> bool:
        return self.action_type == "play_card"

    @property
    def is_end_turn(self) -> bool:
        return self.action_type == "end_turn"

    @property
    def has_target(self) -> bool:
        return self.target_scope == "single_enemy" and self.target_enemy_id != ""
