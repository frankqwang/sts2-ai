"""Card Reward Compiler: 选牌/三选一/发现。

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


# R2.3: 稀有度 → soft 权重（bank_assembler._action_token 会编进 rarity_weight 通道）
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


class CardRewardCompiler:
    """编译 card_reward 选项为 ActionCandidate 列表。"""

    def compile(
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
