from __future__ import annotations

from typing import Any


def normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    normalized = {"type": str(action.get("type", "")).lower()}
    if "hand_index" in action and action["hand_index"] is not None:
        normalized["hand_index"] = int(action["hand_index"])
    if "option_index" in action and action["option_index"] is not None:
        normalized["option_index"] = int(action["option_index"])
    if normalized["type"] == "select_card_option" and "card_index" in action and action["card_index"] is not None:
        normalized.setdefault("option_index", int(action["card_index"]))
    if "slot" in action and action["slot"] is not None:
        normalized["slot"] = int(action["slot"])
    if "target_id" in action and action["target_id"] is not None:
        normalized["target_id"] = int(action["target_id"])
    return normalized


def action_to_key(action: dict[str, Any]) -> tuple[Any, ...]:
    normalized = normalize_action(action)
    return (
        normalized.get("type"),
        normalized.get("hand_index"),
        normalized.get("option_index"),
        normalized.get("slot"),
        normalized.get("target_id"),
    )


def _selection_entries(selection: dict[str, Any], preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    for key in preferred_keys:
        raw_entries = selection.get(key)
        if isinstance(raw_entries, list):
            return [entry for entry in raw_entries if isinstance(entry, dict)]
    return []


def _selection_can_confirm(selection: dict[str, Any]) -> bool:
    return bool(selection.get("can_confirm") or selection.get("confirmable"))


def _selection_can_cancel(selection: dict[str, Any]) -> bool:
    return bool(selection.get("cancelable") or selection.get("can_cancel") or selection.get("can_cancel_selection"))


def _selection_option_index(entry: dict[str, Any]) -> int | None:
    for key in ("option_index", "index", "hand_index", "card_index"):
        if key not in entry or entry.get(key) is None:
            continue
        try:
            return int(entry[key])
        except (TypeError, ValueError):
            continue
    return None


def enumerate_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    card_selection = state.get("card_selection")
    if state.get("is_card_selection_active") and isinstance(card_selection, dict):
        for selectable_option in _selection_entries(
            card_selection,
            ("selectable_options", "selectable_cards", "options", "cards"),
        ):
            option_index = _selection_option_index(selectable_option)
            if option_index is None:
                continue
            actions.append({"type": "select_card_option", "option_index": option_index})

        if _selection_can_confirm(card_selection):
            actions.append({"type": "confirm_selection"})
        if _selection_can_cancel(card_selection):
            actions.append({"type": "cancel_selection"})
        return _dedupe_actions(actions)

    hand_selection = state.get("hand_selection")

    if state.get("is_hand_selection_active") and isinstance(hand_selection, dict):
        for selectable_card in hand_selection.get("selectable_cards", []):
            if "hand_index" not in selectable_card:
                continue
            actions.append(
                {
                    "type": "select_hand_card",
                    "hand_index": int(selectable_card["hand_index"]),
                }
            )

        if hand_selection.get("can_confirm"):
            actions.append({"type": "confirm_selection"})
        if hand_selection.get("cancelable"):
            actions.append({"type": "cancel_selection"})
        return _dedupe_actions(actions)

    for card in state.get("hand_cards", []):
        if not card.get("can_play"):
            continue
        hand_index = int(card["hand_index"])
        if card.get("requires_target"):
            for target_id in card.get("valid_target_ids", []):
                actions.append(
                    {
                        "type": "play_card",
                        "hand_index": hand_index,
                        "target_id": int(target_id),
                    }
                )
        else:
            actions.append({"type": "play_card", "hand_index": hand_index})

    for potion in state.get("potions", []):
        if not potion.get("can_use_in_combat"):
            continue
        slot = int(potion.get("slot", -1))
        if potion.get("requires_target"):
            for target_id in potion.get("valid_target_ids", []):
                actions.append({"type": "use_potion", "slot": slot, "target_id": int(target_id)})
        else:
            actions.append({"type": "use_potion", "slot": slot})

    if state.get("legal_end_turn"):
        actions.append({"type": "end_turn"})

    return _dedupe_actions(actions)


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for action in actions:
        key = action_to_key(action)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalize_action(action))
    return deduped
