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

# R2.1: 非战斗 option 的语义 bucket（event 类型 / reward 类型）
# bank_assembler._action_token 会把 event_kind 做成 9 维 one-hot，补齐非战斗 option token
# 原先只靠 family/roles 区分的信号缺口（所有 event option 都是 "resource"、card_reward 选卡
# 不区分稀有度等）。
EVENT_KINDS = [
    "gain_gold",
    "gain_relic",
    "gain_potion",
    "gain_hp",       # heal / max_hp up
    "lose_hp",       # 自扣血换收益
    "gain_curse",
    "remove_card",
    "upgrade_card",
    "unknown",
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

    # R2.1: 非战斗 option 专用 numeric 特征（combat action 默认 0）
    # rarity_weight: 选卡/shop 买卡的稀有度 → 一维 soft 权重
    #   basic=0.0, curse/status=-0.3, common=0.25, uncommon=0.5, rare=1.0, special=0.5
    # price_ratio: shop 物品价格 / max(player_gold, 1)，反映"买下占钱包比例"
    # can_afford:  shop 是否买得起（0/1），和 price_ratio 组合给网络"买得起但贵" vs "便宜但买不起" 信号
    # event_kind:  EVENT_KINDS 中的一个字符串标签，bank_assembler 做 one-hot
    rarity_weight: float = 0.0
    price_ratio: float = 0.0
    can_afford: float = 0.0
    event_kind: str = ""

    # U8 修复：route 节点专属 risk/value。原先 route_compiler 把 [0,1] risk/value
    # 硬塞进 damage_est/block_est 字段（再乘 30/50 对抗 _action_token 的归一化）——
    # 是个"字段语义黑客"。现在独立字段承载原始 [0,1] 值，bank_assembler 编码时不
    # 走 _DMG/_BLK 归一化路径，保留 raw 量级。仅 map family 填，其他 action 默认 0。
    route_risk: float = 0.0   # [0,1]：该节点的潜在威胁（boss=1, elite=0.7, rest=0）
    route_value: float = 0.0  # [0,1]：该节点的潜在价值（rest=0.7, shop=0.6, boss=0）

    # 路径规划信号:从此 child 到 boss 的**全局路径统计**(覆盖下游所有可达子路径)
    # 目的:选下一步时等价看到了整条路线的 type 分布,不仅是近邻
    # 仅 map family 填;所有计数按 "路径长度" 归一化到 [0,1] 防值爆炸
    route_path_rest_rate: float = 0.0      # 所有下游路径上 rest 占比
    route_path_shop_rate: float = 0.0      # shop 占比
    route_path_elite_rate: float = 0.0     # elite 占比
    route_path_treasure_rate: float = 0.0  # treasure 占比
    route_path_event_rate: float = 0.0     # event 占比
    route_path_monster_rate: float = 0.0   # monster 占比
    route_best_rest_count: float = 0.0     # 所有下游路径里最大 rest 数(归一化 /5)
    route_path_length_norm: float = 0.0    # 到 boss 最短距离(归一化 /17,约等于 act 长度)

    # Skada victory-runs 挖出的路径先验(build_path_priors.py 产出,loader 查表填):
    # frequency: 该 fingerprint 在 victory 玩家(同 character+asc)里的出现频率 [0,1]
    # efficiency: 1 - normalized(avg_duration),越高 = 越快赢;未查到时 0.5(中性)
    route_prior_frequency: float = 0.0
    route_prior_efficiency: float = 0.5

    # Skada community priors（social inductive bias,from skada_analytics.sqlite）
    # 只在 non-combat option 上填;combat 动作默认 0.0(不适用)。
    # loader 构造 option 时查 SkadaPriors,让 token-level 就含"玩家群体行为"先验。
    # 关键价值:冷启动就有 baseline,不用靠 long-horizon reward 从零学"好卡"先验。
    pick_rate_prior: float = 0.0       # [0,1]  全群体选卡率
    win_rate_delta_prior: float = 0.0  # [-0.5, 0.5] 选 vs 跳的胜率差(近似)
    deck_win_rate_prior: float = 0.0   # [0,1]  拿过这张卡的 run 胜率均值
    # TODO(cleanup): synergy_prior 当前永远是 0.0——填值来源 build_card_synergy_matrix.py
    # + skada_index_dataset.deck_card_synergy() 已于 2026-04-19 删除(死代码)。
    # 字段本身和 bank_assembler.py:829 的打包逻辑保留,避免改动网络 max_numeric_dim=58
    # 导致旧 checkpoint 不兼容。下次重训时一起删:
    #   - 本字段
    #   - bank_assembler.py skada_prior_axes 去掉这一维
    #   - network_config.py max_numeric_dim 58→57
    synergy_prior: float = 0.0         # [-1, 1] 与当前 deck 的 synergy 提升

    @property
    def is_play_card(self) -> bool:
        return self.action_type == "play_card"

    @property
    def is_end_turn(self) -> bool:
        return self.action_type == "end_turn"

    @property
    def has_target(self) -> bool:
        return self.target_scope == "single_enemy" and self.target_enemy_id != ""
