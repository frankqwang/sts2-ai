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

                roles = []
                if category == "card":
                    roles.append("build")
                elif category == "relic":
                    roles.append("buff")
                elif category == "potion":
                    roles.append("resource")
                elif category == "remove_card":
                    roles.append("setup")

                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label,
                    family="shop",
                    source_card_id=item_id,
                    cost=cost,
                    roles=roles,
                    target_scope="shop",
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
