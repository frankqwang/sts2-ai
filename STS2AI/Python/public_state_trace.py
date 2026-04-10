from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


def normalize_public_legal_action(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("action"),
        action.get("index"),
        action.get("card_index"),
        action.get("target_id"),
        action.get("col"),
        action.get("row"),
        action.get("slot"),
        bool(action.get("is_enabled", True)),
    )


def normalize_trace_action(action: dict[str, Any] | None) -> dict[str, Any]:
    if not action:
        return {}
    return {
        "action": action.get("action"),
        "index": action.get("index"),
        "label": action.get("label") or "",
        "card_index": action.get("card_index"),
        "target_id": action.get("target_id") or action.get("target"),
        "col": action.get("col"),
        "row": action.get("row"),
        "slot": action.get("slot"),
    }


def extract_battle(state: dict[str, Any]) -> dict[str, Any]:
    return state.get("battle") or {}


def extract_player(state: dict[str, Any]) -> dict[str, Any]:
    battle = extract_battle(state)
    return battle.get("player") or state.get("player") or {}


def extract_hand_labels(state: dict[str, Any]) -> list[str]:
    player = extract_player(state)
    hand = player.get("hand") or []
    labels = []
    for card in hand:
        label = card.get("id") or card.get("label") or card.get("name") or "?"
        labels.append(str(label))
    return sorted(labels)


def extract_player_statuses(state: dict[str, Any]) -> list[tuple[str, int]]:
    player = extract_player(state)
    result = []
    for status in player.get("status") or player.get("powers") or []:
        name = status.get("id") or status.get("name") or ""
        amount = status.get("amount")
        if amount is None:
            amount = status.get("stacks") or 0
        result.append((str(name), int(amount or 0)))
    return sorted(result)


def extract_enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = extract_battle(state)
    result: list[dict[str, Any]] = []
    for enemy in battle.get("enemies") or []:
        status_names = []
        for status in enemy.get("status") or enemy.get("powers") or []:
            status_names.append(str(status.get("id") or status.get("name") or ""))
        intents = enemy.get("intents") or []
        result.append(
            {
                "key": enemy.get("combat_id")
                or enemy.get("entity_id")
                or enemy.get("slot")
                or enemy.get("index")
                or "",
                "name": enemy.get("id") or enemy.get("name") or "",
                "hp": enemy.get("hp") or enemy.get("current_hp") or 0,
                "max_hp": enemy.get("max_hp") or 0,
                "block": enemy.get("block") or 0,
                "is_alive": bool(enemy.get("is_alive", True)),
                "intent": str(intents[0].get("type", "") if intents else "").strip().lower(),
                "statuses": sorted(status_names),
            }
        )
    return sorted(
        result,
        key=lambda item: (
            int(item.get("key") or -1),
            str(item.get("name") or ""),
            int(item.get("hp") or 0),
            int(item.get("block") or 0),
            str(item.get("intent") or ""),
        ),
    )


def build_public_observation(state: dict[str, Any]) -> dict[str, Any]:
    run = state.get("run") or {}
    battle = extract_battle(state)
    player = extract_player(state)
    legal_mask = [
        normalize_public_legal_action(action)
        for action in (state.get("legal_actions") or [])
    ]
    return {
        "state_type": str(state.get("state_type") or "").lower(),
        "terminal": bool(state.get("terminal", False)),
        "run_outcome": str(state.get("run_outcome") or "").lower() or None,
        "floor": run.get("floor"),
        "act": run.get("act"),
        "turn": battle.get("round_number") or battle.get("round") or 0,
        "player": {
            "hp": player.get("hp") or player.get("current_hp") or 0,
            "max_hp": player.get("max_hp") or 0,
            "block": player.get("block") or 0,
            "energy": player.get("energy") or battle.get("energy") or 0,
            "draw_count": player.get("draw_pile_count") or battle.get("draw_pile_count") or 0,
            "discard_count": player.get("discard_pile_count") or battle.get("discard_pile_count") or 0,
            "exhaust_count": player.get("exhaust_pile_count") or battle.get("exhaust_pile_count") or 0,
            "hand_labels": extract_hand_labels(state),
            "statuses": extract_player_statuses(state),
        },
        "enemies": extract_enemies(state),
        "legal_mask": legal_mask,
    }


def public_state_hash(state: dict[str, Any]) -> str:
    observation = build_public_observation(state)
    hashable = json.loads(json.dumps(observation, ensure_ascii=True, default=str))
    for enemy in hashable.get("enemies") or []:
        enemy.pop("name", None)
    payload = json.dumps(hashable, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


@dataclass
class PublicStateTraceEntry:
    step: int
    turn: int
    state_type: str
    terminal: bool
    floor: int | None
    act: int | None
    action: dict[str, Any]
    public_state_hash: str
    player_hp: int
    player_block: int
    energy: int
    hand_labels: list[str]
    draw_count: int
    discard_count: int
    exhaust_count: int
    enemies: list[dict[str, Any]]
    player_statuses: list[tuple[str, int]]
    legal_mask: list[tuple[Any, ...]]
    legal_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_trace_entry(
    state: dict[str, Any],
    *,
    step: int,
    action: dict[str, Any] | None = None,
) -> PublicStateTraceEntry:
    observation = build_public_observation(state)
    player = observation["player"]
    return PublicStateTraceEntry(
        step=step,
        turn=int(observation["turn"] or 0),
        state_type=str(observation["state_type"] or ""),
        terminal=bool(observation["terminal"]),
        floor=observation.get("floor"),
        act=observation.get("act"),
        action=normalize_trace_action(action),
        public_state_hash=public_state_hash(state),
        player_hp=int(player["hp"] or 0),
        player_block=int(player["block"] or 0),
        energy=int(player["energy"] or 0),
        hand_labels=list(player["hand_labels"] or []),
        draw_count=int(player["draw_count"] or 0),
        discard_count=int(player["discard_count"] or 0),
        exhaust_count=int(player["exhaust_count"] or 0),
        enemies=list(observation["enemies"] or []),
        player_statuses=list(player["statuses"] or []),
        legal_mask=list(observation["legal_mask"] or []),
        legal_action_count=len(observation["legal_mask"] or []),
    )
