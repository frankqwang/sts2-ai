"""Selection Option Builder: treasure / relic_select / 其他奖励选择屏。"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.actions import ActionCandidate
from networkV2.s4_featurization.noncombat.card_reward_options import _rarity_weight


_PICK_ACTION_TYPES = {
    "claim_treasure_relic",
    "select_relic",
    "select_reward",
    "claim_reward",
}
_TERMINAL_ACTION_TYPES = {
    "skip_relic_selection",
    "skip_reward_selection",
    "proceed",
    "advance_dialogue",
}


def _pick(raw: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if not isinstance(raw, dict):
        return default
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _default_item_type(state_type: str) -> str:
    if state_type in ("treasure", "relic_select", "relic_reward"):
        return "relic"
    return ""


def _event_kind(item_type: str) -> str:
    item_type = str(item_type or "").lower()
    if item_type == "gold":
        return "gain_gold"
    if item_type == "relic":
        return "gain_relic"
    if item_type == "potion":
        return "gain_potion"
    if item_type in ("heal", "max_hp", "max_health", "hp"):
        return "gain_hp"
    if item_type in ("curse", "gain_curse"):
        return "gain_curse"
    if item_type in ("remove_card", "purge"):
        return "remove_card"
    if item_type in ("upgrade_card", "smith"):
        return "upgrade_card"
    return "unknown"


def _roles(item_type: str, item: dict[str, Any]) -> list[str]:
    item_type = str(item_type or "").lower()
    if item_type in ("gold", "relic", "potion", "heal", "max_hp", "max_health", "hp"):
        return ["resource"]
    if item_type in ("remove_card", "purge", "upgrade_card", "smith"):
        return ["setup"]
    if item_type == "card":
        card_type = str(_pick(item, "card_type", "type", default="") or "").lower()
        if card_type == "attack":
            return ["attack"]
        if card_type == "skill":
            return ["block"]
        if card_type == "power":
            return ["buff"]
    return ["resource"]


class SelectionOptionBuilder:
    """构建 treasure / relic_select 等“选一个奖励”类决策。"""

    def build(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[ActionCandidate]:
        state_type = str(obs.get("state_type", "") or "").lower()
        items_by_index = self._collect_items(obs, state_type)
        candidates: list[ActionCandidate] = []

        for i, action in enumerate(legal_actions):
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action", "") or "").lower()
            label = str(action.get("label", "") or "")

            if action_type in _PICK_ACTION_TYPES:
                idx = int(action.get("index", -1) or -1)
                item = items_by_index.get(idx, {})
                item_type = str(
                    _pick(item, "category", "type", default=_default_item_type(state_type)) or ""
                ).lower()
                item_id = str(_pick(item, "id", default=action.get("id", "")) or "").lower()
                rarity = str(_pick(item, "rarity", default="") or "").lower()
                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label,
                    family="reward",
                    source_card_id=item_id,
                    target_scope="choice",
                    roles=_roles(item_type, item),
                    event_kind=_event_kind(item_type),
                    rarity_weight=_rarity_weight(rarity),
                    can_afford=1.0,
                ))
            elif action_type in _TERMINAL_ACTION_TYPES:
                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label or "Continue",
                    family="reward",
                    target_scope="choice",
                    roles=["terminal"],
                    ends_turn=True,
                ))

        return candidates

    def _collect_items(self, obs: dict[str, Any], state_type: str) -> dict[int, dict[str, Any]]:
        if state_type == "treasure":
            raw_items = (obs.get("treasure") or {}).get("relics") or []
        elif state_type in ("relic_select", "relic_reward"):
            raw_items = (
                (obs.get("relic_select") or {}).get("relics")
                or (obs.get("relic_reward") or {}).get("relics")
                or []
            )
        else:
            raw_items = (obs.get("rewards") or {}).get("items") or []

        return {
            int(item.get("index", idx) or 0): item
            for idx, item in enumerate(raw_items)
            if isinstance(item, dict)
        }
