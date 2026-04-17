"""Shop Compiler: 商店购买/remove/skip。

游戏数据格式 (shop state):
  {
    "items": [
      {"category": "card"/"relic"/"potion"/"remove_card",
       "cost": int, "can_afford": bool, "id": str, ...},
    ]
  }
"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.actions import ActionCandidate
from networkV2.s4_compiler.noncombat.card_reward_compiler import _rarity_weight


class ShopCompiler:
    """编译 shop 选项为 ActionCandidate 列表。"""

    def compile(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        shop = obs.get("shop") or {}
        items_by_index = {
            item.get("index", i): item
            for i, item in enumerate(shop.get("items", []) or [])
            if isinstance(item, dict)
        }
        # R2.3: 读取玩家当前金币用于 price_ratio / can_afford（原先这两个字段 ActionCandidate
        # 都是 0，shop 买 7g 卡和 150g rare relic 在 token 层完全无区别）
        player = obs.get("player") or (obs.get("battle") or {}).get("player") or {}
        player_gold = int(player.get("gold", 0) or 0)
        gold_denom = max(player_gold, 1)

        for i, action in enumerate(legal_actions):
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action", "") or "").lower()
            label = str(action.get("label", "") or "")

            if action_type in ("shop_purchase", "claim_reward"):
                idx = action.get("index", -1)
                item = items_by_index.get(idx, {})
                category = str(item.get("category", "") or "").lower()
                cost = int(item.get("cost", item.get("price", 0)) or 0)
                item_id = str(item.get("id", action.get("id", "")) or "").lower()
                rarity = str(item.get("rarity", "") or "").lower()

                # can_afford 优先读 obs，没有再按 cost vs gold 推
                raw_afford = item.get("can_afford")
                if isinstance(raw_afford, bool):
                    can_afford = 1.0 if raw_afford else 0.0
                else:
                    can_afford = 1.0 if cost <= player_gold else 0.0

                price_ratio = min(cost / gold_denom, 2.0)  # clip 到 2.0（买不起时的极端值）

                roles = []
                if category == "card":
                    roles.append("build")
                elif category == "relic":
                    roles.append("buff")
                elif category == "potion":
                    roles.append("resource")
                elif category == "remove_card":
                    roles.append("setup")

                # R2.3: rarity_weight 对 card / relic / potion 都适用（remove_card 没 rarity）
                if category in ("card", "relic", "potion"):
                    rw = _rarity_weight(rarity)
                elif category == "remove_card":
                    rw = 0.6  # 去基础卡是 STS 公认高价值，固定给 0.6
                else:
                    rw = 0.0

                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label,
                    family="shop",
                    source_card_id=item_id,
                    cost=cost,
                    roles=roles,
                    target_scope="shop",
                    # R2.3 non-combat option 信号
                    rarity_weight=rw,
                    price_ratio=price_ratio,
                    can_afford=can_afford,
                ))

            elif action_type in ("shop_exit", "proceed"):
                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label or "Exit Shop",
                    family="shop",
                    roles=["terminal"],
                    ends_turn=True,
                ))

        return candidates
