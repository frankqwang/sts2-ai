"""Shop 硬编码规则:只删卡,不买。

设计理由:
  - skada 数据里 shop_actions 类型分布:remove 1600 (66%) / buy_relic 414 /
    buy_card 162 / buy_potion 166 —— remove 是主流,高手普遍以删卡为主
  - shop buy 决策依赖 "具体在卖什么",skada 记录只存实际动作不含 offered items,
    无法学习 "vs 其他可买 item" 的对比,BC 无法 learn policy
  - 删卡是少数能从数据驱动的决策,且规则简单可靠
  - 训练期不产 shop sample,推理期调本模块

规则优先级(从高到低):
  1. 诅咒/状态卡(curse / status / ASCENDERS_BANE)→ 立即删
  2. 超量的 Strike(超过 4 张基础 Strike)→ 删一张
  3. 超量的 Defend(超过 3 张基础 Defend)→ 删一张
  4. 规则不匹配 → leave shop(离店)

不覆盖:
  - buy_card / buy_relic / buy_potion 一律 skip
  - 未来可拓展:遇到 skada priors 高胜率卡/遗物时才买
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ShopDecision:
    """输出给 runtime 的 shop 决策。

    action: "remove" / "leave"(未来可能加 "buy_card" / "buy_relic" / "buy_potion")
    target_card_id: 当 action="remove" 时,要删的卡 id(lower_snake)
    reason: 决策的文字解释,便于 log/诊断
    """
    action: str
    target_card_id: str = ""
    reason: str = ""


# 判定 "垃圾卡" 的 id 正则/关键词(lower_snake 规范化后)
_CURSE_OR_STATUS_INDICATORS = (
    "curse_", "ascenders_bane", "_status",
    "wound", "dazed", "slimed", "burn", "void",
    # STS2 可能的 status 卡(按 card_type=status)
)

# 角色基础卡(strike_* / defend_*)
_BASE_STRIKE_PREFIX = "strike_"
_BASE_DEFEND_PREFIX = "defend_"


def _is_curse_or_status(card_id: str) -> bool:
    cid = str(card_id or "").lower()
    return any(ind in cid for ind in _CURSE_OR_STATUS_INDICATORS)


def _is_base_strike(card_id: str) -> bool:
    return str(card_id or "").lower().startswith(_BASE_STRIKE_PREFIX)


def _is_base_defend(card_id: str) -> bool:
    return str(card_id or "").lower().startswith(_BASE_DEFEND_PREFIX)


def decide_shop_action(
    deck: list[str],
    *,
    gold: int = 0,
    remove_price: int = 75,
    strike_keep_count: int = 4,
    defend_keep_count: int = 3,
) -> ShopDecision:
    """给定当前 deck,返回 shop 动作。

    deck:      当前 run 的 deck(list of card_id,lower_snake 规范化,含重复)
    gold:      当前金币(skada 里删卡价格约 50-75,按 asc 浮动)
    remove_price:  估计的删卡价格(金币不够就不删,直接离店)
    strike_keep_count / defend_keep_count: 保留的基础卡下限
    """
    # 0. 金币不足 → 离店
    if gold < remove_price:
        return ShopDecision(action="leave", reason=f"gold {gold} < remove price {remove_price}")

    # 1. 优先删诅咒/状态卡
    for cid in deck:
        if _is_curse_or_status(cid):
            return ShopDecision(
                action="remove",
                target_card_id=str(cid).lower(),
                reason="curse/status card",
            )

    # 2. 删超量 Strike
    strikes = [c for c in deck if _is_base_strike(c)]
    if len(strikes) > strike_keep_count:
        return ShopDecision(
            action="remove",
            target_card_id=strikes[0].lower(),
            reason=f"strike count {len(strikes)} > {strike_keep_count}",
        )

    # 3. 删超量 Defend
    defends = [c for c in deck if _is_base_defend(c)]
    if len(defends) > defend_keep_count:
        return ShopDecision(
            action="remove",
            target_card_id=defends[0].lower(),
            reason=f"defend count {len(defends)} > {defend_keep_count}",
        )

    # 4. 无删卡目标 → 离店
    return ShopDecision(action="leave", reason="nothing to remove")


def resolve_to_legal_action(
    decision: ShopDecision,
    legal_actions: list[dict[str, Any]],
) -> int | None:
    """把 ShopDecision 对齐到 runtime 的 legal_actions 列表,返回 action_index。

    legal_actions 形式(来自 sim bridge):
      [{"action": "remove", "item_id": "strike_ironclad", ...},
       {"action": "buy_card", "item_id": "...", ...},
       {"action": "leave_shop", ...}]

    返回 None 表示 legal_actions 里没这个动作(异常情况,调用方 fallback)。
    """
    if decision.action == "leave":
        # 找 leave_shop / exit / proceed 之类的动作
        for i, a in enumerate(legal_actions):
            atype = str(a.get("action", "") or "").lower()
            if atype in ("leave_shop", "exit_shop", "leave", "proceed", "skip"):
                return i
        return None

    if decision.action == "remove":
        target = decision.target_card_id.lower()
        for i, a in enumerate(legal_actions):
            atype = str(a.get("action", "") or "").lower()
            iid = str(a.get("item_id", "") or "").lower()
            if atype == "remove" and iid == target:
                return i
        # 没找到精确 target,退而求其次:选任意 remove 动作
        for i, a in enumerate(legal_actions):
            if str(a.get("action", "") or "").lower() == "remove":
                return i
        return None

    return None
