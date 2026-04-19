"""Card Reward Option Builder: 选牌/三选一/发现。

游戏数据格式 (card_reward state):
  {
    "can_skip": bool,
    "cards": [
      {"id": str, "cost": int, "type": str, "rarity": str, "is_upgraded": bool, ...},
      ...
    ]
  }

输出: list[ActionCandidate]，每个候选代表"选这张牌"或"跳过"。
"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.actions import ActionCandidate


# R2.3: 稀有度 → soft 权重（token_bank_builder._action_token 会编进 rarity_weight 通道）
# 原先 card_reward / shop 只给 family+roles，同 cost+type+rarity 的选项 token 完全一致。
# 现在把稀有度用一个连续 scalar 编码，让网络在 token 层就能分清 "Strike vs Demon Form"。
_RARITY_WEIGHT = {
    "basic":    0.0,
    "common":   0.25,
    "uncommon": 0.5,
    "rare":     1.0,
    "special":  0.5,
    "curse":   -0.3,
    "status":  -0.2,
}


def _rarity_weight(rarity: str) -> float:
    return _RARITY_WEIGHT.get(str(rarity or "").strip().lower(), 0.0)


def _pick(raw: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if not isinstance(raw, dict):
        return default
    for key in keys:
        if key in raw:
            return raw[key]
    return default


class CardRewardOptionBuilder:
    """构建 card_reward 选项为 ActionCandidate 列表。"""

    def build(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []

        # 从 legal_actions 构建（更可靠，因为 legal_actions 是真正可执行的）
        for i, action in enumerate(legal_actions):
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action", "") or "").lower()

            if action_type in ("select_card_reward", "claim_reward"):
                card_id = str(action.get("card_id", action.get("id", "")) or "").lower()
                label = str(action.get("label", "") or "")

                # 尝试从 card_reward.cards 中找到对应卡牌的详细信息
                card_info = self._find_card_info(obs, card_id)
                if card_info:
                    candidates.append(ActionCandidate(
                        action_type=action_type,
                        action_index=i,
                        label=label,
                        family="card_reward",
                        source_card_id=card_id,
                        source_card_type=card_info.get("type", ""),
                        cost=card_info.get("cost", 0),
                        roles=self._infer_roles(card_info),
                        target_scope="none",
                        # R2.3: 稀有度进 token，区分 "Strike vs Demon Form"
                        rarity_weight=_rarity_weight(card_info.get("rarity", "")),
                    ))
                    continue

                reward_info = self._find_reward_item_info(obs, action)
                if reward_info:
                    candidates.append(ActionCandidate(
                        action_type=action_type,
                        action_index=i,
                        label=label,
                        family="card_reward",
                        source_card_id=reward_info["id"],
                        source_card_type=reward_info["card_type"],
                        cost=reward_info["cost"],
                        roles=reward_info["roles"],
                        target_scope="none",
                        rarity_weight=_rarity_weight(reward_info["rarity"]),
                        event_kind=reward_info["event_kind"],
                        can_afford=1.0,
                    ))

            elif action_type in ("skip", "skip_card_reward"):
                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=str(action.get("label", "Skip") or "Skip"),
                    family="card_reward",
                    roles=["terminal"],
                    target_scope="none",
                    ends_turn=True,
                ))

            elif action_type == "proceed":
                candidates.append(ActionCandidate(
                    action_type="proceed",
                    action_index=i,
                    label="Proceed",
                    family="card_reward",
                    roles=["terminal"],
                    ends_turn=True,
                ))

        return candidates

    def _find_card_info(self, obs: dict[str, Any], card_id: str) -> dict[str, Any]:
        """从 obs 的 card_reward 区域查找卡牌详情。"""
        cr = obs.get("card_reward") or {}
        for card in cr.get("cards", []) or []:
            if isinstance(card, dict):
                cid = str(_pick(card, "id", default="") or "").lower()
                if cid == card_id:
                    return {
                        "type": str(_pick(card, "type", "card_type", default="") or "").lower(),
                        "cost": int(_pick(card, "cost", "energy_cost", default=0) or 0),
                        "rarity": str(_pick(card, "rarity", default="") or "").lower(),
                        "is_upgraded": bool(_pick(card, "is_upgraded", default=False)),
                    }
        return {}

    def _find_reward_item_info(
        self,
        obs: dict[str, Any],
        action: dict[str, Any],
    ) -> dict[str, Any]:
        rewards = obs.get("rewards") or {}
        wanted_index = action.get("index", None)
        wanted_id = str(action.get("id", "") or "").lower()
        for idx, item in enumerate(rewards.get("items", []) or []):
            if not isinstance(item, dict):
                continue
            item_index = item.get("index", idx)
            item_id = str(_pick(item, "id", default="") or "").lower()
            if wanted_id and item_id != wanted_id:
                continue
            if wanted_index is not None and int(item_index or 0) != int(wanted_index or 0):
                continue
            item_type = str(_pick(item, "category", "type", default="") or "").lower()
            card_type = str(_pick(item, "card_type", default=item_type) or "").lower()
            rarity = str(_pick(item, "rarity", default="") or "").lower()
            event_kind = {
                "gold": "gain_gold",
                "relic": "gain_relic",
                "potion": "gain_potion",
                "heal": "gain_hp",
                "max_hp": "gain_hp",
                "remove_card": "remove_card",
                "upgrade_card": "upgrade_card",
                "curse": "gain_curse",
            }.get(item_type, "unknown")
            if item_type in ("remove_card", "upgrade_card"):
                roles = ["setup"]
            elif item_type == "card":
                roles = self._infer_roles({"type": card_type})
            else:
                roles = ["resource"]
            return {
                "id": item_id,
                "card_type": card_type if item_type == "card" else "",
                "cost": int(_pick(item, "cost", "price", default=0) or 0),
                "rarity": rarity,
                "roles": roles,
                "event_kind": event_kind,
            }
        return {}

    def _infer_roles(self, card_info: dict[str, Any]) -> list[str]:
        roles: list[str] = []
        ct = card_info.get("type", "").lower()
        if ct == "attack":
            roles.append("attack")
        elif ct == "skill":
            roles.append("block")
        elif ct == "power":
            roles.append("buff")
        return roles
