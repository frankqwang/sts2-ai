from __future__ import annotations

import json
import re
import time
import http.client
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class SingleplayerAutomationError(RuntimeError):
    pass


class SingleplayerConnectionError(SingleplayerAutomationError):
    pass


class SingleplayerApiError(SingleplayerAutomationError):
    pass


class SingleplayerTimeoutError(SingleplayerAutomationError):
    pass


COMBAT_STATE_TYPES = {"monster", "elite", "boss", "hand_select"}


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _parse_cost(raw_cost: Any) -> int:
    text = str(raw_cost or "").strip().upper()
    if text == "X":
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _intent_total_damage(raw_intent: dict[str, Any]) -> int:
    for value in (raw_intent.get("label"), raw_intent.get("title"), raw_intent.get("description")):
        text = str(value or "")
        multiplied = re.search(r"(\d+)\s*[xX×]\s*(\d+)", text)
        if multiplied:
            return int(multiplied.group(1)) * int(multiplied.group(2))
        repeated_zh = re.search(r"(\d+)\s*点伤害\s*(\d+)\s*次", text)
        if repeated_zh:
            return int(repeated_zh.group(1)) * int(repeated_zh.group(2))
        number = re.search(r"(\d+)", text)
        if number:
            return int(number.group(1))
    return 0


def _normalize_powers(raw_powers: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for power in raw_powers or []:
        normalized.append(
            {
                "id": power.get("id") or power.get("name") or power.get("title"),
                "name": power.get("name") or power.get("title") or power.get("id"),
                "amount": int(power.get("amount") or power.get("counter") or 0),
            }
        )
    return normalized


def _normalize_enemy(enemy: dict[str, Any]) -> dict[str, Any]:
    intents = []
    total_attack = 0
    for raw_intent in enemy.get("intents", []):
        intent_type = _lower(raw_intent.get("type"))
        total_damage = _intent_total_damage(raw_intent)
        intents.append(
            {
                "intent_type": intent_type,
                "label": raw_intent.get("label"),
                "title": raw_intent.get("title"),
                "description": raw_intent.get("description"),
                "total_damage": total_damage,
            }
        )
        if intent_type in ("attack", "deathblow"):
            total_attack += total_damage

    return {
        "entity_id": enemy.get("entity_id"),
        "combat_id": int(enemy.get("combat_id", -1)),
        "name": enemy.get("name"),
        "current_hp": int(enemy.get("hp", 0)),
        "max_hp": int(enemy.get("max_hp", 1)),
        "block": int(enemy.get("block", 0)),
        "is_alive": int(enemy.get("hp", 0)) > 0,
        "powers": _normalize_powers(enemy.get("status")),
        "intends_to_attack": total_attack > 0,
        "intents": intents,
    }


def _normalize_hand_card(card: dict[str, Any], enemies: list[dict[str, Any]], energy: int, is_play_phase: bool) -> dict[str, Any]:
    target_type = _lower(card.get("target_type"))
    requires_target = target_type in {"enemy", "anyenemy", "any_enemy"}
    valid_target_ids = [enemy["combat_id"] for enemy in enemies if enemy.get("is_alive")] if requires_target else []
    energy_cost = _parse_cost(card.get("cost"))
    raw_can_play = card.get("can_play")
    if raw_can_play is None:
        can_play = bool(is_play_phase and energy_cost <= energy)
    else:
        can_play = bool(raw_can_play)
    return {
        "hand_index": int(card.get("index", -1)),
        "id": card.get("id"),
        "title": card.get("name"),
        "type": card.get("type"),
        "target_type": card.get("target_type"),
        "energy_cost": energy_cost,
        "requires_target": requires_target,
        "valid_target_ids": valid_target_ids,
        "can_play": can_play,
        "unplayable_reason": card.get("unplayable_reason"),
        "keywords": card.get("keywords", []),
        "description": card.get("description"),
        "is_upgraded": bool(card.get("is_upgraded")),
        "gains_block": bool(card.get("gains_block")),
        "card_type": card.get("card_type") or card.get("type"),
    }


def _normalize_potion(potion: dict[str, Any], enemies: list[dict[str, Any]]) -> dict[str, Any]:
    target_type = _lower(potion.get("target_type"))
    requires_target = target_type in {"enemy", "anyenemy", "any_enemy"}
    valid_target_ids = [enemy["combat_id"] for enemy in enemies if enemy.get("is_alive")] if requires_target else []
    return {
        "slot": int(potion.get("slot", -1)),
        "id": potion.get("id"),
        "title": potion.get("name"),
        "description": potion.get("description"),
        "target_type": potion.get("target_type"),
        "can_use_in_combat": bool(potion.get("can_use_in_combat")),
        "requires_target": requires_target,
        "valid_target_ids": valid_target_ids,
        "keywords": potion.get("keywords", []),
    }


def _normalize_selection_option(option: dict[str, Any], index_field: str) -> dict[str, Any]:
    index_value = option.get(index_field)
    if index_value is None:
        for fallback_field in ("index", "hand_index", "option_index", "card_index"):
            if option.get(fallback_field) is not None:
                index_value = option.get(fallback_field)
                break
    try:
        selection_index = int(index_value if index_value is not None else -1)
    except (TypeError, ValueError):
        selection_index = -1
    return {
        index_field: selection_index,
        "id": option.get("id"),
        "title": option.get("name"),
        "type": option.get("type"),
        "energy_cost": _parse_cost(option.get("cost")),
        "description": option.get("description"),
    }


def _normalize_card_selection(selection: dict[str, Any]) -> dict[str, Any]:
    selectable_source = (
        selection.get("selectable_options")
        or selection.get("selectable_cards")
        or selection.get("options")
        or selection.get("cards")
        or []
    )
    selected_source = (
        selection.get("selected_options")
        or selection.get("selected_cards")
        or []
    )
    return {
        "prompt_text": selection.get("prompt") or selection.get("prompt_text"),
        "selectable_options": [
            _normalize_selection_option(option, "option_index")
            for option in selectable_source
            if isinstance(option, dict)
        ],
        "selected_options": [
            _normalize_selection_option(option, "option_index")
            for option in selected_source
            if isinstance(option, dict)
        ],
        "can_confirm": bool(selection.get("can_confirm") or selection.get("confirmable")),
        "cancelable": bool(selection.get("cancelable") or selection.get("can_cancel") or selection.get("can_cancel_selection")),
    }


def adapt_v1_state_for_combat_policy(state: dict[str, Any]) -> dict[str, Any]:
    state_type = _lower(state.get("state_type"))
    battle = state.get("battle") or {}
    run = state.get("run") or {}
    # Player may be at root level (new MCP format) or inside battle (old format)
    player = battle.get("player") or state.get("player") or {}
    enemies = [_normalize_enemy(enemy) for enemy in battle.get("enemies", [])]
    is_play_phase = bool(battle.get("is_play_phase"))
    energy = int(player.get("energy", 0))
    potions = [_normalize_potion(potion, enemies) for potion in player.get("potions", [])]
    hand_cards = [
        _normalize_hand_card(card, enemies, energy=energy, is_play_phase=is_play_phase)
        for card in player.get("hand", [])
    ]

    hand_select = state.get("hand_select") if state_type == "hand_select" else None
    hand_selection = None
    if isinstance(hand_select, dict):
        selectable_cards = []
        for selectable in hand_select.get("cards", []):
            selectable_cards.append(
                {
                    "hand_index": int(selectable.get("index", -1)),
                    "id": selectable.get("id"),
                    "title": selectable.get("name"),
                    "type": selectable.get("type"),
                    "energy_cost": _parse_cost(selectable.get("cost")),
                    "description": selectable.get("description"),
                }
            )
        hand_selection = {
            "prompt_text": hand_select.get("prompt"),
            "selectable_cards": selectable_cards,
            "selected_cards": hand_select.get("selected_cards", []),
            "can_confirm": bool(hand_select.get("can_confirm")),
            "cancelable": False,
        }

    raw_card_selection = state.get("card_selection") if isinstance(state.get("card_selection"), dict) else battle.get("card_selection")
    card_selection = None
    if isinstance(raw_card_selection, dict):
        card_selection = _normalize_card_selection(raw_card_selection)

    return {
        "is_done": False,
        "is_combat_active": state_type in COMBAT_STATE_TYPES,
        "round_number": int(battle.get("round", 0)),
        "current_side": battle.get("turn"),
        "is_play_phase": is_play_phase,
        "can_end_turn": bool(is_play_phase and state_type != "hand_select"),
        "legal_end_turn": bool(is_play_phase and state_type != "hand_select"),
        "player_actions_disabled": False,
        "is_action_queue_running": False,
        "player": {
            "character": player.get("character"),
            "current_hp": int(player.get("hp", 0)),
            "max_hp": int(player.get("max_hp", 1)),
            "block": int(player.get("block", 0)),
            "energy": energy,
            "max_energy": int(player.get("max_energy", max(energy, 0))),
            "gold": int(player.get("gold", 0)),
            "powers": _normalize_powers(player.get("status")),
            "potions": potions,
        },
        "run": {
            "act": int(run.get("act", 0) or 0),
            "floor": int(run.get("floor", 0) or 0),
            "seed": run.get("seed"),
            "character_id": run.get("character_id"),
            "ascension": int(run.get("ascension", run.get("ascension_level", 0)) or 0),
        },
        "piles": {
            "draw": int(player.get("draw_pile_count", 0)),
            "discard": int(player.get("discard_pile_count", 0)),
            "exhaust": int(player.get("exhaust_pile_count", 0)),
        },
        "enemies": enemies,
        "potions": potions,
        "hand_cards": hand_cards,
        "hand_selection": hand_selection,
        "is_hand_selection_active": bool(hand_selection),
        "card_selection": card_selection,
        "is_card_selection_active": bool(card_selection),
    }


def translate_combat_action_for_v1(adapted_state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    normalized_type = _lower(action.get("type"))
    if normalized_type == "end_turn":
        return {"action": "end_turn"}
    if normalized_type == "select_hand_card":
        return {"action": "combat_select_card", "card_index": int(action["hand_index"])}
    if normalized_type == "select_card_option":
        option_index = action.get("option_index", action.get("card_index"))
        return {"action": "combat_select_card", "card_index": int(option_index)}
    if normalized_type == "confirm_selection":
        return {"action": "combat_confirm_selection"}
    if normalized_type == "use_potion":
        payload: dict[str, Any] = {"action": "use_potion", "slot": int(action["slot"])}
        target_id = action.get("target_id")
        if target_id is not None:
            for enemy in adapted_state.get("enemies", []):
                if int(enemy.get("combat_id", -1)) == int(target_id):
                    payload["target"] = enemy.get("entity_id")
                    break
        return payload
    if normalized_type != "play_card":
        raise SingleplayerApiError(f"Unsupported combat action '{normalized_type}' for v1 automation.")

    payload: dict[str, Any] = {
        "action": "play_card",
        "card_index": int(action["hand_index"]),
    }
    target_id = action.get("target_id")
    if target_id is not None:
        for enemy in adapted_state.get("enemies", []):
            if int(enemy.get("combat_id", -1)) == int(target_id):
                payload["target"] = enemy.get("entity_id")
                break
    return payload


@dataclass(slots=True)
class SingleplayerClient:
    base_url: str = "http://127.0.0.1:15526"
    poll_interval_s: float = 0.15
    request_timeout_s: float = 10.0
    ready_timeout_s: float = 20.0
    request_retry_count: int = 2
    request_retry_delay_s: float = 0.2

    def get_state(self) -> dict[str, Any]:
        return self._request_json("GET", "/api/v1/singleplayer")

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        # The v1 API returns an action status object on POST, not the updated game state.
        # Follow the action with a fresh GET so callers always receive a state snapshot.
        self._request_json("POST", "/api/v1/singleplayer", payload)
        state = self.get_state()
        if self._should_wait_for_post_action_combat_settle(state):
            state = self.wait_until(
                lambda current: self._is_post_action_combat_settled(current),
                timeout_s=self.ready_timeout_s,
                initial_state=state,
            )
        return state

    def wait_until(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout_s: float | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timeout_s = self.ready_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        state = initial_state if initial_state is not None else self.get_state()
        while time.monotonic() < deadline:
            if predicate(state):
                return state
            time.sleep(self.poll_interval_s)
            state = self.get_state()
        raise SingleplayerTimeoutError(
            "Singleplayer API did not reach the requested state before timeout. "
            f"Last state: {json.dumps(state, ensure_ascii=True)}"
        )

    def wait_for_state_change(
        self,
        previous_state: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        previous_signature = json.dumps(previous_state, ensure_ascii=True, sort_keys=True)
        return self.wait_until(
            lambda current: json.dumps(current, ensure_ascii=True, sort_keys=True) != previous_signature,
            timeout_s=timeout_s,
            initial_state=previous_state,
        )

    def wait_until_actionable_combat(
        self,
        *,
        timeout_s: float | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.wait_until(
            lambda current: self._is_actionable_combat_state(current),
            timeout_s=timeout_s,
            initial_state=initial_state,
        )

    def close(self) -> None:
        return None

    def _should_wait_for_post_action_combat_settle(self, state: dict[str, Any]) -> bool:
        return self._is_combat_state(state) and not self._is_actionable_combat_state(state)

    def _is_post_action_combat_settled(self, state: dict[str, Any]) -> bool:
        return not self._is_combat_state(state) or self._is_actionable_combat_state(state)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        retry_budget = max(0, int(self.request_retry_count)) if method.upper() == "GET" else 0
        body = ""
        for attempt in range(retry_budget + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                    body = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                # Retry transient server errors (500) on GET requests
                if exc.code >= 500 and attempt < retry_budget:
                    time.sleep(max(0.0, float(self.request_retry_delay_s)))
                    continue
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("error"):
                    raise SingleplayerApiError(parsed["error"]) from exc
                raise SingleplayerApiError(f"HTTP {exc.code}: {body}") from exc
            except urllib.error.URLError as exc:
                if attempt < retry_budget:
                    time.sleep(max(0.0, float(self.request_retry_delay_s)))
                    continue
                raise SingleplayerConnectionError(
                    f"Could not connect to singleplayer API at {url}. Make sure STS2MCP is loaded."
                ) from exc
            except (ConnectionResetError, ConnectionAbortedError, http.client.RemoteDisconnected, TimeoutError, OSError) as exc:
                if attempt < retry_budget:
                    time.sleep(max(0.0, float(self.request_retry_delay_s)))
                    continue
                raise SingleplayerConnectionError(
                    f"Singleplayer API request to {url} was interrupted."
                ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SingleplayerApiError(f"Invalid JSON response from {url}: {body}") from exc

        if isinstance(parsed, dict) and parsed.get("status") == "error":
            raise SingleplayerApiError(parsed.get("error", "Unknown singleplayer API error"))
        if not isinstance(parsed, dict):
            raise SingleplayerApiError(f"Expected JSON object from {url}, got {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _is_actionable_combat_state(state: dict[str, Any]) -> bool:
        state_type = _lower(state.get("state_type"))
        if state_type not in COMBAT_STATE_TYPES:
            return False
        if state_type == "hand_select":
            return True
        if isinstance(state.get("card_selection"), dict):
            return True
        battle = state.get("battle") or {}
        if not (bool(battle.get("is_play_phase")) and _lower(battle.get("turn")) == "player"):
            return False

        player = battle.get("player") or state.get("player") or {}
        hand = list(player.get("hand") or [])
        if not hand:
            return True

        if any(bool(card.get("can_play")) for card in hand):
            return True

        reasons = {_lower(card.get("unplayable_reason")) for card in hand if card.get("unplayable_reason") is not None}
        if reasons and reasons.issubset({"playeractionsdisabled", "disabled", "none"}):
            return False

        return True

    @staticmethod
    def _is_combat_state(state: dict[str, Any]) -> bool:
        return _lower(state.get("state_type")) in COMBAT_STATE_TYPES
