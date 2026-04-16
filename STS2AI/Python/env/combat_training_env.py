"""基于管道的战斗训练环境，用于隔离的战斗遭遇 rollout。"""
from __future__ import annotations


import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from env.headless_sim_runner import DEFAULT_DLL_PATH, DEFAULT_REPO_ROOT, start_headless_sim, stop_process
from env.pipe_client import PipeClient
from env.simulator_api_error import SimulatorApiError

from env.full_run_env import _normalize_build_spec


_CARD_TYPE_NAMES = {
    0: "UNKNOWN",
    1: "ATTACK",
    2: "SKILL",
    3: "POWER",
    4: "STATUS",
    5: "CURSE",
    6: "QUEST",
}
_TARGET_TYPE_NAMES = {
    0: None,
    1: "None",
    2: "Self",
    3: "AnyEnemy",
    4: "AnyPlayer",
    5: "AnyAlly",
    6: "TargetedNoCreature",
    7: "AllEnemies",
    8: "RandomEnemy",
    9: "AllAllies",
    10: "Osty",
}
_ROOM_TYPES = {"monster", "elite", "boss"}


def _pick(raw: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if not isinstance(raw, dict):
        return default
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _room_type_from_catalog(catalog: dict[str, str], encounter_id: str | None) -> str:
    room_type = catalog.get(str(encounter_id or "").strip().upper(), "monster")
    return room_type if room_type in _ROOM_TYPES else "monster"


def _normalize_power(power: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(power, dict):
        return {"id": "", "amount": 0}
    return {
        "id": str(_pick(power, "id", "Id", default="") or ""),
        "amount": int(_pick(power, "amount", "Amount", default=0) or 0),
    }


def _normalize_enemy(raw_enemy: dict[str, Any], room_type: str) -> dict[str, Any]:
    combat_id = _pick(raw_enemy, "combat_id", "CombatId")
    enemy_id = str(_pick(raw_enemy, "id", "Id", default="") or "")
    intents: list[dict[str, Any]] = []
    for intent in _pick(raw_enemy, "intents", "Intents", default=[]) or []:
        if not isinstance(intent, dict):
            continue
        damage = _pick(intent, "damage", "Damage")
        total_damage = _pick(intent, "total_damage", "TotalDamage")
        intents.append(
            {
                "intent_type": str(_pick(intent, "intent_type", "IntentType", default="") or "").lower(),
                "damage": int(damage or 0) if damage is not None else None,
                "total_damage": int(total_damage or 0) if total_damage is not None else None,
                "repeats": int(_pick(intent, "repeats", "Repeats", default=1) or 1),
            }
        )
    return {
        "entity_id": f"{enemy_id}_{combat_id}" if combat_id is not None and enemy_id else enemy_id,
        "combat_id": int(combat_id or -1),
        "id": enemy_id,
        "name": str(_pick(raw_enemy, "name", "Name", default=enemy_id) or enemy_id),
        "hp": int(_pick(raw_enemy, "hp", "current_hp", "CurrentHp", default=0) or 0),
        "current_hp": int(_pick(raw_enemy, "current_hp", "CurrentHp", default=0) or 0),
        "max_hp": int(_pick(raw_enemy, "max_hp", "MaxHp", default=1) or 1),
        "block": int(_pick(raw_enemy, "block", "Block", default=0) or 0),
        "is_alive": bool(_pick(raw_enemy, "is_alive", "IsAlive", default=True)),
        "is_hittable": bool(_pick(raw_enemy, "is_hittable", "IsHittable", default=True)),
        "intends_to_attack": bool(_pick(raw_enemy, "intends_to_attack", "IntendsToAttack", default=False)),
        "next_move_id": _pick(raw_enemy, "next_move_id", "NextMoveId"),
        "enemy_type": room_type,
        "intents": intents,
        "powers": [_normalize_power(power) for power in (_pick(raw_enemy, "powers", "Powers", default=[]) or [])],
    }


def _normalize_hand_card(raw_card: dict[str, Any]) -> dict[str, Any]:
    hand_index = int(_pick(raw_card, "hand_index", "HandIndex", default=0) or 0)
    raw_target_type = _pick(raw_card, "target_type", "TargetType", default=0)
    target_type = raw_target_type if isinstance(raw_target_type, str) else _TARGET_TYPE_NAMES.get(int(raw_target_type or 0))
    raw_card_type = _pick(raw_card, "card_type", "CardType", default=0)
    card_type = str(raw_card_type).upper() if isinstance(raw_card_type, str) and raw_card_type else _CARD_TYPE_NAMES.get(int(raw_card_type or 0), "UNKNOWN")
    return {
        "hand_index": hand_index,
        "card_index": hand_index,
        "id": str(_pick(raw_card, "id", "Id", default="") or ""),
        "name": str(_pick(raw_card, "title", "Title", "id", "Id", default="") or ""),
        "title": str(_pick(raw_card, "title", "Title", "id", "Id", default="") or ""),
        "energy_cost": int(_pick(raw_card, "energy_cost", "EnergyCost", default=0) or 0),
        "cost": int(_pick(raw_card, "energy_cost", "EnergyCost", default=0) or 0),
        "cost_for_turn": int(_pick(raw_card, "energy_cost", "EnergyCost", default=0) or 0),
        "card_type": card_type,
        "type": card_type,
        "target_type": target_type,
        "is_upgraded": bool(_pick(raw_card, "is_upgraded", "IsUpgraded", default=False)),
        "can_play": bool(_pick(raw_card, "can_play", "CanPlay", default=False)),
        "requires_target": bool(_pick(raw_card, "requires_target", "RequiresTarget", default=False)),
        "valid_target_ids": [int(target_id) for target_id in (_pick(raw_card, "valid_target_ids", "ValidTargetIds", default=[]) or [])],
        "gains_block": bool(_pick(raw_card, "gains_block", "GainsBlock", default=False)),
        "keywords": [str(keyword) for keyword in (_pick(raw_card, "keywords", "Keywords", default=[]) or [])],
        "description": _pick(raw_card, "description", "Description"),
    }


def build_combat_legal_actions(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if bool(_pick(snapshot, "episode_done", "EpisodeDone", "terminal", default=False)):
        return []

    actions: list[dict[str, Any]] = []
    hand_selection = _pick(snapshot, "hand_selection", "HandSelection")
    if bool(_pick(snapshot, "is_hand_selection_active", "IsHandSelectionActive", default=False)) and isinstance(hand_selection, dict):
        for card in _pick(hand_selection, "selectable_cards", "SelectableCards", default=[]) or []:
            if not isinstance(card, dict):
                continue
            hand_index = int(_pick(card, "hand_index", "HandIndex", default=0) or 0)
            actions.append(
                {
                    "action": "select_hand_card",
                    "hand_index": hand_index,
                    "index": hand_index,
                    "label": str(_pick(card, "title", "Title", "id", "Id", default="") or ""),
                    "card_id": str(_pick(card, "id", "Id", default="") or ""),
                }
            )
        if bool(_pick(hand_selection, "can_confirm", "CanConfirm", default=False)):
            actions.append({"action": "confirm_selection", "label": "Confirm"})
        if bool(_pick(hand_selection, "cancelable", "Cancelable", default=False)):
            actions.append({"action": "cancel_selection", "label": "Cancel"})
        return actions

    card_selection = _pick(snapshot, "card_selection", "CardSelection")
    if bool(_pick(snapshot, "is_card_selection_active", "IsCardSelectionActive", default=False)) and isinstance(card_selection, dict):
        for option in _pick(card_selection, "selectable_cards", "SelectableCards", default=[]) or []:
            if not isinstance(option, dict):
                continue
            choice_index = int(_pick(option, "choice_index", "ChoiceIndex", default=0) or 0)
            actions.append(
                {
                    "action": "select_card_option",
                    "choice_index": choice_index,
                    "index": choice_index,
                    "label": str(_pick(option, "title", "Title", "id", "Id", default="") or ""),
                    "card_id": str(_pick(option, "id", "Id", default="") or ""),
                }
            )
        if bool(_pick(card_selection, "can_confirm", "CanConfirm", default=False)):
            actions.append({"action": "confirm_selection", "label": "Confirm"})
        if bool(_pick(card_selection, "cancelable", "Cancelable", default=False)):
            actions.append({"action": "cancel_selection", "label": "Cancel"})
        return actions

    hand_cards = snapshot.get("hand")
    if not isinstance(hand_cards, list):
        hand_cards = ((snapshot.get("battle") or {}).get("hand") or [])
    for raw_card in hand_cards or []:
        if not isinstance(raw_card, dict):
            continue
        if not bool(_pick(raw_card, "can_play", "CanPlay", default=False)):
            continue
        hand_index = int(_pick(raw_card, "hand_index", "HandIndex", default=0) or 0)
        card_id = str(_pick(raw_card, "id", "Id", default="") or "")
        label = str(_pick(raw_card, "title", "Title", default=card_id) or card_id)
        valid_targets = [int(target_id) for target_id in (_pick(raw_card, "valid_target_ids", "ValidTargetIds", default=[]) or [])]
        if bool(_pick(raw_card, "requires_target", "RequiresTarget", default=False)) and valid_targets:
            for target_id in valid_targets:
                actions.append(
                    {
                        "action": "play_card",
                        "hand_index": hand_index,
                        "card_index": hand_index,
                        "index": hand_index,
                        "card_id": card_id,
                        "label": label,
                        "target_id": target_id,
                    }
                )
        else:
            actions.append(
                {
                    "action": "play_card",
                    "hand_index": hand_index,
                    "card_index": hand_index,
                    "index": hand_index,
                    "card_id": card_id,
                    "label": label,
                }
            )
    if bool(_pick(snapshot, "can_end_turn", "CanEndTurn", default=False)):
        actions.append({"action": "end_turn", "label": "End Turn"})
    return actions


def adapt_combat_snapshot(
    snapshot: dict[str, Any],
    *,
    current_build: dict[str, Any] | None,
    room_type_lookup: dict[str, str],
) -> dict[str, Any]:
    encounter_id = str(snapshot.get("encounter_id") or "").strip()
    room_type = _room_type_from_catalog(room_type_lookup, encounter_id)
    if bool(snapshot.get("episode_done")):
        state_type = "game_over"
    elif bool(snapshot.get("is_hand_selection_active")):
        state_type = "hand_select"
    elif bool(snapshot.get("is_card_selection_active")):
        state_type = "card_select"
    else:
        state_type = room_type

    raw_player = snapshot.get("player") if isinstance(snapshot.get("player"), dict) else {}
    build_payload = current_build or {}
    player = {
        "hp": int(_pick(raw_player, "hp", "current_hp", "CurrentHp", default=0) or 0),
        "current_hp": int(_pick(raw_player, "current_hp", "CurrentHp", default=0) or 0),
        "max_hp": int(_pick(raw_player, "max_hp", "MaxHp", default=1) or 1),
        "block": int(_pick(raw_player, "block", "Block", default=0) or 0),
        "energy": int(_pick(raw_player, "energy", "Energy", default=0) or 0),
        "max_energy": int(_pick(raw_player, "max_energy", "MaxEnergy", default=0) or 0),
        "powers": [_normalize_power(power) for power in (_pick(raw_player, "powers", "Powers", default=[]) or [])],
        "deck": list(build_payload.get("deck") or []),
        "cards": list(build_payload.get("deck") or []),
        "relics": list(build_payload.get("relics") or []),
        "potions": [],
        "gold": int(build_payload.get("gold") or 0),
    }

    enemies = [
        _normalize_enemy(raw_enemy, room_type)
        for raw_enemy in (snapshot.get("enemies") or [])
        if isinstance(raw_enemy, dict)
    ]
    hand = [
        _normalize_hand_card(raw_card)
        for raw_card in (snapshot.get("hand") or [])
        if isinstance(raw_card, dict)
    ]
    piles = snapshot.get("piles") if isinstance(snapshot.get("piles"), dict) else {}
    battle = {
        "player": player,
        "hand": hand,
        "enemies": enemies,
        "energy": int(_pick(raw_player, "energy", "Energy", default=0) or 0),
        "max_energy": int(_pick(raw_player, "max_energy", "MaxEnergy", default=0) or 0),
        "round_number": int(snapshot.get("round_number") or 0),
        "is_play_phase": bool(snapshot.get("is_play_phase")),
        "draw_pile_count": int(_pick(piles, "draw", "Draw", default=0) or 0),
        "discard_pile_count": int(_pick(piles, "discard", "Discard", default=0) or 0),
        "exhaust_pile_count": int(_pick(piles, "exhaust", "Exhaust", default=0) or 0),
        "draw_pile": [{"id": card_id} for card_id in (_pick(piles, "draw_card_ids", "DrawCardIds", default=[]) or [])],
        "discard_pile": [{"id": card_id} for card_id in (_pick(piles, "discard_card_ids", "DiscardCardIds", default=[]) or [])],
        "exhaust_pile": [{"id": card_id} for card_id in (_pick(piles, "exhaust_card_ids", "ExhaustCardIds", default=[]) or [])],
    }
    state = {
        "state_type": state_type,
        "room_type": room_type,
        "character_id": snapshot.get("character_id"),
        "encounter_id": encounter_id,
        "round_number": int(snapshot.get("round_number") or 0),
        "player": player,
        "hand": hand,
        "battle": battle,
        "enemies": enemies,
        "can_end_turn": bool(_pick(snapshot, "can_end_turn", "CanEndTurn", default=False)),
        "is_hand_selection_active": bool(_pick(snapshot, "is_hand_selection_active", "IsHandSelectionActive", default=False)),
        "is_card_selection_active": bool(_pick(snapshot, "is_card_selection_active", "IsCardSelectionActive", default=False)),
        "legal_actions": [],
        "run": {
            "floor": 0,
            "act": 1,
            "seed": snapshot.get("seed"),
        },
        "terminal": bool(snapshot.get("episode_done")),
        "run_outcome": "victory" if snapshot.get("victory") is True else "defeat" if snapshot.get("victory") is False else None,
        "_combat_snapshot": snapshot,
    }
    if isinstance(snapshot.get("hand_selection"), dict):
        state["hand_selection"] = snapshot.get("hand_selection")
        state["hand_select"] = snapshot.get("hand_selection")
    if isinstance(snapshot.get("card_selection"), dict):
        state["card_selection"] = snapshot.get("card_selection")
        state["card_select"] = snapshot.get("card_selection")
    state["legal_actions"] = build_combat_legal_actions(state)
    return state


@dataclass(slots=True)
class PipeBackedCombatTrainingClient:
    port: int = 15527
    connect_timeout_s: float = 10.0
    auto_launch: bool = False
    repo_root: str | Path = DEFAULT_REPO_ROOT
    dll_path: str | Path = DEFAULT_DLL_PATH
    _pipe: PipeClient = field(init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)
    _owned_host_proc: Any | None = field(default=None, init=False, repr=False)
    _room_type_lookup: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _current_build: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._pipe = PipeClient(port=self.port)

    def _start_owned_host(self) -> None:
        if self._owned_host_proc is not None:
            return
        self._owned_host_proc = start_headless_sim(
            port=self.port,
            repo_root=self.repo_root,
            dll_path=self.dll_path,
            connect_timeout_s=max(15.0, float(self.connect_timeout_s)),
            protocol="json",
            extra_host_args=["--combat-sim-server"],
        )

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        try:
            self._pipe.connect(timeout_s=self.connect_timeout_s)
        except Exception:
            if not self.auto_launch:
                raise
            self._start_owned_host()
            self._pipe = PipeClient(port=self.port)
            self._pipe.connect(timeout_s=max(15.0, float(self.connect_timeout_s)))
        self._connected = True

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_connected()
        result = self._pipe.call(method, params)
        if not isinstance(result, dict):
            raise SimulatorApiError(f"{method} response is not a dict")
        return result

    def _adapt(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return adapt_combat_snapshot(
            snapshot,
            current_build=self._current_build,
            room_type_lookup=self._room_type_lookup,
        )

    def combat_catalog(self) -> dict[str, Any]:
        result = self._call("combat_catalog")
        encounters = result.get("encounters") or []
        lookup: dict[str, str] = {}
        for item in encounters:
            if not isinstance(item, dict):
                continue
            encounter_id = str(item.get("encounter_id") or "").strip().upper()
            room_type = str(item.get("room_type") or "").strip().lower()
            if encounter_id and room_type in _ROOM_TYPES:
                lookup[encounter_id] = room_type
        self._room_type_lookup = lookup
        return {"encounters": list(encounters)}

    def reset(
        self,
        *,
        character_id: str = "IRONCLAD",
        encounter_id: str,
        ascension_level: int = 0,
        seed: str | None = None,
        build: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_build = _normalize_build_spec(build)
        params: dict[str, Any] = {
            "character_id": str(character_id),
            "encounter_id": str(encounter_id),
            "ascension_level": int(ascension_level),
        }
        if seed:
            params["seed"] = str(seed)
        if normalized_build is not None:
            params["build"] = normalized_build
        snapshot = self._call("combat_reset", params)
        self._current_build = normalized_build
        if not self._room_type_lookup:
            try:
                self.combat_catalog()
            except Exception:
                self._room_type_lookup = {}
        return self._adapt(snapshot)

    def get_state(self) -> dict[str, Any]:
        snapshot = self._call("combat_state")
        return self._adapt(snapshot)

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        result = self._call("combat_step", action)
        if not bool(result.get("accepted", False)):
            error = SimulatorApiError(str(result.get("error") or "Combat step rejected"))
            state = result.get("state")
            if isinstance(state, dict):
                setattr(error, "latest_state", self._adapt(state))
            raise error
        snapshot = result.get("state")
        if not isinstance(snapshot, dict):
            raise SimulatorApiError("Combat step response did not include a state payload")
        state = self._adapt(snapshot)
        done = bool(state.get("terminal"))
        reward = 0.0
        if done:
            reward = 1.0 if state.get("run_outcome") == "victory" else -1.0
        info = {
            "accepted": True,
            "state_type": state.get("state_type"),
            "run_outcome": state.get("run_outcome"),
            "room_type": state.get("room_type"),
        }
        return state, reward, done, info

    def close(self) -> None:
        try:
            self._pipe.close()
        except Exception:
            pass
        self._connected = False
        stop_process(self._owned_host_proc)
        self._owned_host_proc = None


def sample_weighted_room_type(
    rng: random.Random,
    *,
    monster_weight: int,
    elite_weight: int,
    boss_weight: int,
) -> str:
    population = ["monster", "elite", "boss"]
    weights = [max(0, int(monster_weight)), max(0, int(elite_weight)), max(0, int(boss_weight))]
    if sum(weights) <= 0:
        raise ValueError("At least one room weight must be positive.")
    return rng.choices(population, weights=weights, k=1)[0]
