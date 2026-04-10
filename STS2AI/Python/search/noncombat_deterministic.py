from __future__ import annotations

from typing import Any


_BASIC_CARD_PRIORITY = {
    "ASCENDERS_BANE": 100.0,
    "STRIKE_IRONCLAD": 90.0,
    "DEFEND_IRONCLAD": 80.0,
    "STRIKE": 70.0,
    "DEFEND": 60.0,
}

_RARITY_SCORE = {
    "rare": 40.0,
    "uncommon": 25.0,
    "common": 10.0,
    "basic": -20.0,
    "curse": -80.0,
    "status": -100.0,
}

_TYPE_SCORE = {
    "power": 20.0,
    "skill": 12.0,
    "attack": 8.0,
    "curse": -100.0,
    "status": -100.0,
}


def _norm(value: Any) -> str:
    return str(value or "").strip().upper()


def _card_from_index(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    card_select = state.get("card_select") or {}
    cards = card_select.get("cards") or []
    try:
        idx = int(action.get("index", -1))
    except Exception:
        idx = -1
    if 0 <= idx < len(cards) and isinstance(cards[idx], dict):
        return cards[idx]
    return {}


def _remove_priority(card: dict[str, Any]) -> float:
    card_id = _norm(card.get("id") or card.get("name"))
    if card_id in _BASIC_CARD_PRIORITY:
        return _BASIC_CARD_PRIORITY[card_id]
    rarity = str(card.get("rarity") or "").strip().lower()
    card_type = str(card.get("type") or "").strip().lower()
    score = 0.0
    score -= _RARITY_SCORE.get(rarity, 5.0)
    score -= _TYPE_SCORE.get(card_type, 0.0)
    score -= float(card.get("cost") or 0) * 0.5
    if bool(card.get("is_upgraded")):
        score -= 25.0
    return score


def _upgrade_priority(card: dict[str, Any]) -> float:
    if bool(card.get("is_upgraded")):
        return -1_000_000.0
    card_id = _norm(card.get("id") or card.get("name"))
    rarity = str(card.get("rarity") or "").strip().lower()
    card_type = str(card.get("type") or "").strip().lower()
    score = _RARITY_SCORE.get(rarity, 5.0) + _TYPE_SCORE.get(card_type, 0.0)
    score += float(card.get("cost") or 0) * 1.5
    if card_id in _BASIC_CARD_PRIORITY:
        score -= 20.0
    return score


def choose_deterministic_card_select_action(state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
    card_select = state.get("card_select") or {}
    screen_type = str(card_select.get("screen_type") or "").strip().lower()
    prompt = str(card_select.get("prompt") or "").strip().lower()

    confirm = next(
        (
            action for action in legal
            if str(action.get("action") or "").strip().lower() in {"confirm_selection", "combat_confirm_selection"}
        ),
        None,
    )
    if confirm is not None and bool(card_select.get("can_confirm")):
        return confirm

    selectable = [
        action for action in legal
        if str(action.get("action") or "").strip().lower() in {"select_card", "combat_select_card"}
    ]
    if not selectable:
        return confirm

    if "upgrade" in screen_type or "upgrade" in prompt:
        return max(selectable, key=lambda action: _upgrade_priority(_card_from_index(state, action)))

    if "remove" in screen_type or "purge" in screen_type or "remove" in prompt or "purge" in prompt:
        return max(selectable, key=lambda action: _remove_priority(_card_from_index(state, action)))

    return max(selectable, key=lambda action: _upgrade_priority(_card_from_index(state, action)))


def choose_deterministic_rest_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    *,
    hp_rest_threshold: float = 0.5,
) -> dict[str, Any] | None:
    player = state.get("player") or {}
    hp = float(player.get("hp") or 0.0)
    max_hp = float(player.get("max_hp") or 1.0)
    hp_ratio = hp / max(max_hp, 1.0)
    options_by_label: dict[str, dict[str, Any]] = {}
    for action in legal:
        if str(action.get("action") or "").strip().lower() != "choose_rest_option":
            continue
        label = str(action.get("label") or action.get("type") or action.get("id") or "").strip().lower()
        options_by_label[label] = action

    rest = next((action for label, action in options_by_label.items() if "rest" in label or "heal" in label), None)
    smith = next((action for label, action in options_by_label.items() if "smith" in label or "upgrade" in label), None)
    if hp_ratio < hp_rest_threshold and rest is not None:
        return rest
    if smith is not None:
        return smith
    return rest or next(iter(options_by_label.values()), None)


def choose_deterministic_shop_action(state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
    shop = state.get("shop") or {}
    items = shop.get("items") or []
    item_by_index: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            item_by_index[int(item.get("index", -1))] = item
        except Exception:
            continue

    remove_action: dict[str, Any] | None = None
    for action in legal:
        if str(action.get("action") or "").strip().lower() != "shop_purchase":
            continue
        try:
            item = item_by_index.get(int(action.get("index", -1)))
        except Exception:
            item = None
        category = str((item or {}).get("category") or "").strip().lower()
        if category == "remove_card":
            remove_action = action
            break

    if remove_action is not None:
        return remove_action
    return next(
        (
            action for action in legal
            if str(action.get("action") or "").strip().lower() in {"proceed", "shop_exit", "skip"}
        ),
        None,
    )

