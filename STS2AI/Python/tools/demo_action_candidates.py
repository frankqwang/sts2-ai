from __future__ import annotations

import _path_init  # noqa: F401

from typing import Any

from sts2_singleplayer_env import translate_combat_action_for_v1


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _append_unique_action(actions: list[dict[str, Any]], action: dict[str, Any]) -> None:
    if isinstance(action, dict) and action not in actions:
        actions.append(action)


def _menu_actions_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    menu = state.get("menu") if isinstance(state.get("menu"), dict) else {}
    discovered: list[dict[str, Any]] = []
    for key in ("actions", "available_actions"):
        raw_actions = menu.get(key)
        if not isinstance(raw_actions, list):
            continue
        for raw_action in raw_actions:
            if isinstance(raw_action, str):
                action_id = _lower(raw_action)
                if action_id:
                    discovered.append({"action": action_id})
                continue
            if not isinstance(raw_action, dict):
                continue
            action_id = _lower(raw_action.get("id") or raw_action.get("action"))
            if not action_id:
                continue
            if raw_action.get("is_enabled") is False or raw_action.get("enabled") is False:
                continue
            discovered.append({"action": action_id})
    return discovered


def _extract_env_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    raw_actions = state.get("legal_actions")
    if not isinstance(raw_actions, list):
        return []

    discovered: list[dict[str, Any]] = []
    int_keys = {"index", "card_index", "target_id", "ascension", "timeout_ms"}
    for raw_action in raw_actions:
        if isinstance(raw_action, str):
            action_id = _lower(raw_action)
            if action_id:
                _append_unique_action(discovered, {"action": action_id})
            continue
        if not isinstance(raw_action, dict):
            continue
        if raw_action.get("is_enabled") is False or raw_action.get("enabled") is False:
            continue
        action_id = _lower(raw_action.get("action") or raw_action.get("id"))
        if not action_id:
            continue
        action_payload: dict[str, Any] = {"action": action_id}
        for key, value in raw_action.items():
            if key in {"action", "id"}:
                continue
            if key in int_keys:
                try:
                    action_payload[key] = int(value)
                    continue
                except (TypeError, ValueError):
                    pass
            action_payload[key] = value
        _append_unique_action(discovered, action_payload)
    return discovered


def _should_prefer_env_legal_actions(state_type: str) -> bool:
    return state_type in {"menu", "overlay", "game_over"}


def _combat_candidate_actions(adapted_state: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    card_selection = adapted_state.get("card_selection") or {}
    hand_selection = adapted_state.get("hand_selection") or {}
    if adapted_state.get("is_card_selection_active"):
        selectable_options = card_selection.get("selectable_options") or card_selection.get("selectable_cards") or []
        for selectable in selectable_options:
            option_index = selectable.get("option_index")
            if option_index is None:
                option_index = selectable.get("index", selectable.get("hand_index", selectable.get("card_index")))
            if option_index is None:
                continue
            policy_action = {"type": "select_card_option", "option_index": int(option_index)}
            try:
                _append_unique_action(candidates, translate_combat_action_for_v1(adapted_state, policy_action))
            except Exception:
                continue
        if card_selection.get("can_confirm"):
            try:
                _append_unique_action(
                    candidates,
                    translate_combat_action_for_v1(adapted_state, {"type": "confirm_selection"}),
                )
            except Exception:
                pass
        return candidates

    if adapted_state.get("is_hand_selection_active"):
        for selectable in hand_selection.get("selectable_cards", []):
            policy_action = {"type": "select_hand_card", "hand_index": int(selectable.get("hand_index", -1))}
            try:
                _append_unique_action(candidates, translate_combat_action_for_v1(adapted_state, policy_action))
            except Exception:
                continue
        if hand_selection.get("can_confirm"):
            try:
                _append_unique_action(
                    candidates,
                    translate_combat_action_for_v1(adapted_state, {"type": "confirm_selection"}),
                )
            except Exception:
                pass
        return candidates

    for card in adapted_state.get("hand_cards", []):
        if not card.get("can_play"):
            continue
        base_action = {"type": "play_card", "hand_index": int(card.get("hand_index", -1))}
        target_ids = list(card.get("valid_target_ids") or [])
        if card.get("requires_target") and target_ids:
            for target_id in target_ids:
                candidate_action = dict(base_action)
                candidate_action["target_id"] = int(target_id)
                try:
                    _append_unique_action(candidates, translate_combat_action_for_v1(adapted_state, candidate_action))
                except Exception:
                    continue
            continue

        try:
            _append_unique_action(candidates, translate_combat_action_for_v1(adapted_state, base_action))
        except Exception:
            continue

    for potion in adapted_state.get("potions", []):
        if not potion.get("can_use_in_combat"):
            continue
        base_action = {"type": "use_potion", "slot": int(potion.get("slot", -1))}
        target_ids = list(potion.get("valid_target_ids") or [])
        if potion.get("requires_target") and target_ids:
            for target_id in target_ids:
                candidate_action = dict(base_action)
                candidate_action["target_id"] = int(target_id)
                try:
                    _append_unique_action(candidates, translate_combat_action_for_v1(adapted_state, candidate_action))
                except Exception:
                    continue
            continue
        try:
            _append_unique_action(candidates, translate_combat_action_for_v1(adapted_state, base_action))
        except Exception:
            continue

    if adapted_state.get("legal_end_turn"):
        try:
            _append_unique_action(candidates, translate_combat_action_for_v1(adapted_state, {"type": "end_turn"}))
        except Exception:
            pass
    return candidates


def _non_combat_candidate_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    state_type = _lower(state.get("state_type"))
    candidates: list[dict[str, Any]] = []
    env_legal_actions = _extract_env_legal_actions(state)
    if env_legal_actions and _should_prefer_env_legal_actions(state_type):
        return env_legal_actions

    if state_type == "map":
        for option in (state.get("map") or {}).get("next_options", []):
            _append_unique_action(candidates, {"action": "choose_map_node", "index": int(option.get("index", -1))})
        return candidates

    if state_type == "combat_rewards":
        rewards = state.get("rewards") or {}
        for item in rewards.get("items", []):
            _append_unique_action(candidates, {"action": "claim_reward", "index": int(item.get("index", -1))})
        if rewards.get("can_proceed"):
            _append_unique_action(candidates, {"action": "proceed"})
        return candidates

    if state_type == "card_reward":
        reward_state = state.get("card_reward") or {}
        for card in reward_state.get("cards", []):
            _append_unique_action(candidates, {"action": "select_card_reward", "card_index": int(card.get("index", -1))})
        if reward_state.get("can_skip"):
            _append_unique_action(candidates, {"action": "skip_card_reward"})
        return candidates

    if state_type == "rest_site":
        rest_site = state.get("rest_site") or {}
        for option in rest_site.get("options", []):
            if option.get("is_enabled") is False:
                continue
            _append_unique_action(candidates, {"action": "choose_rest_option", "index": int(option.get("index", -1))})
        if rest_site.get("can_proceed"):
            _append_unique_action(candidates, {"action": "proceed"})
        return candidates

    if state_type == "event":
        event_state = state.get("event") or {}
        if event_state.get("in_dialogue"):
            _append_unique_action(candidates, {"action": "advance_dialogue"})
        unlocked_event_index = 0
        for option in event_state.get("options", []):
            if option.get("is_locked") or option.get("is_chosen"):
                continue
            _append_unique_action(candidates, {"action": "choose_event_option", "index": unlocked_event_index})
            unlocked_event_index += 1
        if not candidates:
            _append_unique_action(candidates, {"action": "proceed"})
            _append_unique_action(candidates, {"action": "advance_dialogue"})
        return candidates

    if state_type == "shop":
        shop_state = state.get("shop") or {}
        for item in shop_state.get("items", []):
            if item.get("is_stocked") is False or item.get("can_afford") is False:
                continue
            _append_unique_action(candidates, {"action": "shop_purchase", "index": int(item.get("index", -1))})
        _append_unique_action(candidates, {"action": "proceed"})
        return candidates

    if state_type == "card_select":
        select_state = state.get("card_select") or {}
        for card in select_state.get("cards", []):
            _append_unique_action(candidates, {"action": "select_card", "index": int(card.get("index", -1))})
        if select_state.get("can_confirm"):
            _append_unique_action(candidates, {"action": "confirm_selection"})
        if select_state.get("can_cancel"):
            _append_unique_action(candidates, {"action": "cancel_selection"})
        return candidates

    if state_type == "relic_select":
        relic_state = state.get("relic_select") or {}
        for relic in relic_state.get("relics", []):
            _append_unique_action(candidates, {"action": "select_relic", "index": int(relic.get("index", -1))})
        if relic_state.get("can_skip"):
            _append_unique_action(candidates, {"action": "skip_relic_selection"})
        return candidates

    if state_type == "treasure":
        treasure_state = state.get("treasure") or {}
        for relic in treasure_state.get("relics", []):
            _append_unique_action(candidates, {"action": "claim_treasure_relic", "index": int(relic.get("index", -1))})
        if treasure_state.get("can_proceed"):
            _append_unique_action(candidates, {"action": "proceed"})
        return candidates

    if state_type == "menu":
        for action in _menu_actions_from_state(state):
            _append_unique_action(candidates, action)
        _append_unique_action(candidates, {"action": "start_run"})
        return candidates

    if state_type == "overlay":
        overlay = state.get("overlay") if isinstance(state.get("overlay"), dict) else {}
        for key in ("actions", "available_actions"):
            raw_actions = overlay.get(key)
            if not isinstance(raw_actions, list):
                continue
            for item in raw_actions:
                if isinstance(item, str):
                    action_id = _lower(item)
                    if action_id:
                        _append_unique_action(candidates, {"action": action_id})
                    continue
                if not isinstance(item, dict):
                    continue
                action_id = _lower(item.get("id") or item.get("action"))
                if not action_id:
                    continue
                if item.get("is_enabled") is False or item.get("enabled") is False:
                    continue
                action_payload = {"action": action_id}
                if "index" in item:
                    action_payload["index"] = int(item["index"])
                _append_unique_action(candidates, action_payload)
        return candidates

    if state_type == "game_over":
        game_over = state.get("game_over") if isinstance(state.get("game_over"), dict) else {}
        for item in game_over.get("available_actions", []):
            if isinstance(item, str):
                action_id = _lower(item)
                if action_id:
                    _append_unique_action(candidates, {"action": action_id})
                continue
            if not isinstance(item, dict):
                continue
            action_id = _lower(item.get("action") or item.get("id"))
            if not action_id:
                continue
            action_payload = {"action": action_id}
            if "index" in item:
                action_payload["index"] = int(item["index"])
            if "is_confirm" in item:
                action_payload["is_confirm"] = bool(item["is_confirm"])
            if "is_cancel" in item:
                action_payload["is_cancel"] = bool(item["is_cancel"])
            if "text" in item:
                action_payload["text"] = item["text"]
            _append_unique_action(candidates, action_payload)
        if not candidates:
            for button in game_over.get("buttons", []):
                if not isinstance(button, dict) or button.get("is_enabled") is False:
                    continue
                action_payload = {"action": "overlay_press"}
                if "index" in button:
                    action_payload["index"] = int(button["index"])
                if "is_confirm" in button:
                    action_payload["is_confirm"] = bool(button["is_confirm"])
                if "is_cancel" in button:
                    action_payload["is_cancel"] = bool(button["is_cancel"])
                if "text" in button:
                    action_payload["text"] = button["text"]
                _append_unique_action(candidates, action_payload)
        return candidates

    for key in ("overlay", "actions", "available_actions"):
        raw_value = state.get(key)
        if not isinstance(raw_value, list):
            continue
        for item in raw_value:
            if not isinstance(item, dict):
                continue
            action_id = _lower(item.get("id") or item.get("action"))
            if not action_id:
                continue
            if item.get("is_enabled") is False or item.get("enabled") is False:
                continue
            _append_unique_action(candidates, {"action": action_id})
    return candidates
