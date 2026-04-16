"""Token Bank 容器定义。

三层结构：
  SharedWorldBanks: 战斗/非战斗共享的世界级 bank (6 组)
  CombatBanks: 仅战斗时的 bank (5 组)
  UnifiedTokenBanks: 统一输出容器 (shared + combat/option + action)

Token 是 Compiler → Network 的接口层，由 net/tokenizer.py 转为 tensor。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Token 类型常量
# ---------------------------------------------------------------------------

# --- shared: build_bank ---
TK_DECK_CARD = "deck_card"
TK_BUILD_PROFILE = "build_profile"

# --- shared: inventory_bank ---
TK_RELIC = "relic"
TK_POTION = "potion"

# --- shared: economy_bank ---
TK_ECONOMY = "economy"

# --- shared: route_bank ---
TK_ROUTE_SUMMARY = "route_summary"
TK_ROUTE_NODE = "route_node"

# --- shared: objective_bank ---
TK_OBJECTIVE = "objective"

# --- shared: forecast_bank ---
TK_COMBAT_FORECAST = "combat_forecast"

# --- combat: board_bank ---
TK_PLAYER = "player"
TK_HAND_CARD = "hand_card"
TK_ENEMY_CORE = "enemy_core"
TK_ENEMY_INTENT = "enemy_intent"
TK_PILE_SUMMARY = "pile_summary"

# --- combat: mechanism_bank ---
TK_MECHANISM = "mechanism"

# --- combat: modifier_bank ---
TK_MODIFIER = "modifier"

# --- combat: turn_prefix_bank ---
TK_PLAYED_ACTION = "played_action"
TK_TURN_SUMMARY = "turn_summary"

# --- combat: combat_memory_bank ---
TK_COMBAT_SUMMARY = "combat_summary"

# --- action_bank (combat + non-combat 共用) ---
TK_ACTION_CANDIDATE = "action_candidate"

# --- non-combat option 类型 ---
TK_CARD_REWARD_OPTION = "card_reward_option"
TK_SHOP_OPTION = "shop_option"
TK_ROUTE_OPTION = "route_option"
TK_REST_OPTION = "rest_option"
TK_EVENT_OPTION = "event_option"
TK_SELECTION_OPTION = "selection_option"


# 所有 token 类型列表（用于 type embedding）
ALL_TOKEN_TYPES = [
    "pad",  # index 0 = padding
    # shared banks
    TK_DECK_CARD, TK_BUILD_PROFILE,
    TK_RELIC, TK_POTION,
    TK_ECONOMY,
    TK_ROUTE_SUMMARY, TK_ROUTE_NODE,
    TK_OBJECTIVE,
    TK_COMBAT_FORECAST,
    # combat banks
    TK_PLAYER, TK_HAND_CARD, TK_ENEMY_CORE, TK_ENEMY_INTENT, TK_PILE_SUMMARY,
    TK_MECHANISM,
    TK_MODIFIER,
    TK_PLAYED_ACTION, TK_TURN_SUMMARY,
    TK_COMBAT_SUMMARY,
    # action
    TK_ACTION_CANDIDATE,
    # non-combat options
    TK_CARD_REWARD_OPTION, TK_SHOP_OPTION, TK_ROUTE_OPTION,
    TK_REST_OPTION, TK_EVENT_OPTION, TK_SELECTION_OPTION,
]
TOKEN_TYPE_TO_IDX = {t: i for i, t in enumerate(ALL_TOKEN_TYPES)}
NUM_TOKEN_TYPES = len(ALL_TOKEN_TYPES)

# 时间尺度
TIME_SCALES = ["slow", "medium", "fast"]
TIME_SCALE_TO_IDX = {t: i for i, t in enumerate(TIME_SCALES)}

TOKEN_TIME_SCALE = {
    # shared - slow
    TK_DECK_CARD: "slow", TK_BUILD_PROFILE: "slow",
    TK_RELIC: "slow", TK_POTION: "slow",
    TK_ECONOMY: "slow",
    TK_ROUTE_SUMMARY: "slow", TK_ROUTE_NODE: "slow",
    TK_OBJECTIVE: "slow",
    TK_COMBAT_FORECAST: "slow",
    # combat - fast
    TK_PLAYER: "fast", TK_HAND_CARD: "fast",
    TK_ENEMY_CORE: "fast", TK_ENEMY_INTENT: "fast", TK_PILE_SUMMARY: "fast",
    TK_PLAYED_ACTION: "fast", TK_TURN_SUMMARY: "fast",
    # combat - medium
    TK_MECHANISM: "medium", TK_MODIFIER: "medium", TK_COMBAT_SUMMARY: "medium",
    # action
    TK_ACTION_CANDIDATE: "fast",
    # non-combat options
    TK_CARD_REWARD_OPTION: "fast", TK_SHOP_OPTION: "fast",
    TK_ROUTE_OPTION: "fast", TK_REST_OPTION: "fast",
    TK_EVENT_OPTION: "fast", TK_SELECTION_OPTION: "fast",
}

# 决策域
DECISION_DOMAINS = [
    "combat", "card_reward", "shop", "route", "rest", "event", "selection",
]


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass
class Token:
    """Compiler 输出的单个 token。"""
    numeric: list[float] = field(default_factory=list)
    token_type: str = "pad"
    owner_id: str = ""
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def type_idx(self) -> int:
        return TOKEN_TYPE_TO_IDX.get(self.token_type, 0)

    @property
    def time_scale(self) -> str:
        return TOKEN_TIME_SCALE.get(self.token_type, "fast")

    @property
    def time_scale_idx(self) -> int:
        return TIME_SCALE_TO_IDX.get(self.time_scale, 0)


# ---------------------------------------------------------------------------
# TokenBank
# ---------------------------------------------------------------------------

@dataclass
class TokenBank:
    """一组同类 token 的容器。"""
    bank_name: str = ""
    tokens: list[Token] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)

    def add(self, token: Token) -> None:
        self.tokens.append(token)

    @property
    def is_empty(self) -> bool:
        return len(self.tokens) == 0


def _bank(name: str) -> TokenBank:
    return field(default_factory=lambda n=name: TokenBank(bank_name=n))


# ---------------------------------------------------------------------------
# SharedWorldBanks - 战斗/非战斗共享
# ---------------------------------------------------------------------------

@dataclass
class SharedWorldBanks:
    """战斗和非战斗共享的世界级 bank (6 组)。

    始终编译，不管当前决策域是什么。
    """
    build_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="build"))
    inventory_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="inventory"))
    economy_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="economy"))
    route_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="route"))
    objective_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="objective"))
    forecast_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="forecast"))

    def all_banks(self) -> list[TokenBank]:
        return [
            self.build_bank, self.inventory_bank, self.economy_bank,
            self.route_bank, self.objective_bank, self.forecast_bank,
        ]


# ---------------------------------------------------------------------------
# CombatBanks - 仅战斗时
# ---------------------------------------------------------------------------

@dataclass
class CombatBanks:
    """仅战斗时的 bank (5 组)。"""
    board_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="board"))
    mechanism_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="mechanism"))
    modifier_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="modifier"))
    turn_prefix_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="turn_prefix"))
    combat_memory_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="combat_memory"))

    def all_banks(self) -> list[TokenBank]:
        return [
            self.board_bank, self.mechanism_bank, self.modifier_bank,
            self.turn_prefix_bank, self.combat_memory_bank,
        ]


# ---------------------------------------------------------------------------
# UnifiedTokenBanks - Compiler 的最终输出
# ---------------------------------------------------------------------------

@dataclass
class UnifiedTokenBanks:
    """Compiler 统一输出容器。

    shared: 6 组世界级 bank（始终有）
    combat: 5 组战斗 bank（仅战斗时有，非战斗时为 None）
    action_bank: 动作/选项候选（始终有）
    decision_domain: 当前决策域
    """
    shared: SharedWorldBanks = field(default_factory=SharedWorldBanks)
    combat: CombatBanks | None = None
    action_bank: TokenBank = field(default_factory=lambda: TokenBank(bank_name="action"))
    decision_domain: str = "combat"

    def all_banks(self) -> list[TokenBank]:
        banks = list(self.shared.all_banks())
        if self.combat is not None:
            banks.extend(self.combat.all_banks())
        banks.append(self.action_bank)
        return banks

    @property
    def total_tokens(self) -> int:
        return sum(len(b) for b in self.all_banks())

    def summary(self) -> dict[str, int]:
        return {b.bank_name: len(b) for b in self.all_banks()}

    @property
    def is_combat(self) -> bool:
        return self.decision_domain == "combat"


# ---------------------------------------------------------------------------
# 向后兼容别名
# ---------------------------------------------------------------------------

CombatTokenBanks = UnifiedTokenBanks
