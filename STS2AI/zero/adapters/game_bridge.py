from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..config import ZERO_RUNTIME_DEFAULTS
from ..domain import (
    BattleState,
    EnemyState,
    HandCardState,
    LegalAction,
    PileSummary,
    PlayerState,
    StaticContext,
    TargetSummary,
)


def convert_game_bridge_state(
    raw: dict[str, Any],
    *,
    fallback_encounter_id: str = "",
    fallback_seed: str | None = None,
    fallback_encounter_class: str = "",
) -> BattleState:
    battle_raw = _as_dict(raw.get("battle"))
    run_raw = _as_dict(raw.get("run"))
    top_player_raw = _as_dict(raw.get("player"))
    battle_player_raw = _as_dict(battle_raw.get("player"))

    enemies_raw = _as_list(raw.get("enemies")) or _as_list(battle_raw.get("enemies"))
    hand_raw = _as_list(battle_raw.get("hand")) or _as_list(raw.get("hand"))
    legal_actions_raw = _as_list(raw.get("legal_actions"))

    player_buffs = _powers_to_mapping(_as_list(battle_player_raw.get("powers")) or _as_list(top_player_raw.get("powers")))
    player = PlayerState(
        hp=float(_pick(top_player_raw, "hp", "current_hp", default=0.0)),
        max_hp=float(_pick(top_player_raw, "max_hp", default=0.0)),
        block=float(_pick(battle_player_raw, "block", default=_pick(top_player_raw, "block", default=0.0))),
        energy=float(_pick(battle_raw, "energy", default=_pick(top_player_raw, "energy", default=0.0))),
        potions=_normalize_named_items(_as_list(top_player_raw.get("potions"))),
        buffs=player_buffs,
        resources={
            "gold": float(_pick(top_player_raw, "gold", default=0.0)),
            "stars": float(_pick(battle_player_raw, "stars", default=_pick(top_player_raw, "stars", default=0.0))),
            "max_energy": float(_pick(battle_raw, "max_energy", default=_pick(top_player_raw, "max_energy", default=0.0))),
            "open_potion_slots": float(_pick(top_player_raw, "open_potion_slots", default=0.0)),
        },
    )

    enemies = [
        EnemyState(
            enemy_id=str(_pick(item, "monster_id", "entity_id", "enemy_id", "id", default=f"enemy_{index}")),
            hp=float(_pick(item, "hp", "current_hp", default=0.0)),
            max_hp=float(_pick(item, "max_hp", default=0.0)),
            block=float(_pick(item, "block", default=0.0)),
            intent_id=str(_pick(item, "intent_type", "intent_id", "next_move_id", default="")),
            alive=bool(_pick(item, "alive", "is_alive", default=True)),
            buffs=_powers_to_mapping(_as_list(item.get("buffs")) or _as_list(item.get("powers"))),
            tags=[str(tag) for tag in _as_list(item.get("tags"))],
        )
        for index, item in enumerate(enemies_raw)
    ]

    hand = [
        HandCardState(
            card_id=str(_pick(item, "card_id", "id", default="")),
            cost_now=float(_pick(item, "cost_now", "cost", default=0.0)),
            damage_now=float(_pick(item, "damage_now", "damage", default=0.0)),
            block_now=float(_pick(item, "block_now", "block", default=0.0)),
            magic_now=float(_pick(item, "magic_now", "magic", default=0.0)),
            is_upgraded=bool(_pick(item, "is_upgraded", default=False)),
            retain=bool(_pick(item, "retain", default=False)),
            exhaust=bool(_pick(item, "exhaust", default=False)),
            ethereal=bool(_pick(item, "ethereal", default=False)),
            tags=_build_hand_tags(item),
        )
        for item in hand_raw
    ]

    piles = PileSummary(
        draw_pile_size=int(len(_as_list(battle_raw.get("draw_pile_cards"))) or _pick(top_player_raw, "draw_pile_count", default=0)),
        discard_pile_size=int(len(_as_list(battle_raw.get("discard_pile_cards"))) or _pick(top_player_raw, "discard_pile_count", default=0)),
        exhaust_pile_size=int(len(_as_list(battle_raw.get("exhaust_pile_cards"))) or _pick(top_player_raw, "exhaust_pile_count", default=0)),
        attack_count=sum(1 for item in hand_raw if str(_pick(item, "type", "card_type", default="")).lower() == "attack"),
        skill_count=sum(1 for item in hand_raw if str(_pick(item, "type", "card_type", default="")).lower() == "skill"),
        power_count=sum(1 for item in hand_raw if str(_pick(item, "type", "card_type", default="")).lower() == "power"),
    )

    encounter_id = str(_pick(raw, "encounter_id", default="") or fallback_encounter_id or "")
    encounter_class = fallback_encounter_class or _resolve_encounter_class(encounter_id) or "normal"
    act_value = int(_pick(run_raw, "act", default=0))
    if act_value <= 0:
        act_value = 1
    context = StaticContext(
        character_id=str(_pick(raw, "character_id", default=ZERO_RUNTIME_DEFAULTS.default_character_id)),
        act=act_value,
        floor=int(_pick(run_raw, "floor", default=0)),
        encounter_class=encounter_class,
        encounter_id=encounter_id,
        relics=_normalize_named_items(_as_list(top_player_raw.get("relics"))),
        fixed_powers=list(player_buffs.keys()),
        metadata={
            "seed": str(fallback_seed or ""),
            "state_type": str(_pick(raw, "state_type", default="")),
            "round_number_raw": _pick(raw, "round_number_raw", default=_pick(battle_raw, "round_number_raw", default=0)),
            "turn_side": str(_pick(battle_raw, "turn_side", default="")),
        },
    )

    legal_actions = [
        _convert_action(item, hand_raw=hand_raw, enemies_raw=enemies_raw)
        for item in legal_actions_raw
    ]
    return BattleState(
        player=player,
        enemies=enemies,
        hand=hand,
        piles=piles,
        context=context,
        legal_actions=legal_actions,
        terminal=bool(_pick(raw, "terminal", default=False)),
        run_outcome=str(_pick(raw, "run_outcome", default="")),
        raw=raw,
    )


class GameBridgeCombatRuntime:
    def __init__(
        self,
        *,
        port: int = ZERO_RUNTIME_DEFAULTS.default_port,
        auto_launch: bool = True,
        connect_timeout_s: float = ZERO_RUNTIME_DEFAULTS.default_connect_timeout_s,
        character_id: str = ZERO_RUNTIME_DEFAULTS.default_character_id,
        encounter_id: str = "",
        seed: str | None = None,
        build: dict[str, Any] | None = None,
    ):
        _ensure_python_bridge_path()
        from game_bridge.session import create_combat_session

        self._session = create_combat_session(
            port=port,
            auto_launch=auto_launch,
            connect_timeout_s=connect_timeout_s,
        )
        self._character_id = character_id
        self._encounter_id = encounter_id
        self._seed = seed
        self._build = build
        self._encounter_class = _resolve_encounter_class(encounter_id)
        self._latest_state: BattleState | None = None

    def configure(
        self,
        *,
        character_id: str | None = None,
        encounter_id: str | None = None,
        seed: str | None = None,
        build: dict[str, Any] | None = None,
    ) -> None:
        if character_id is not None:
            self._character_id = character_id
        if encounter_id is not None:
            self._encounter_id = encounter_id
            self._encounter_class = _resolve_encounter_class(encounter_id)
        if seed is not None:
            self._seed = seed
        if build is not None:
            self._build = build

    def reset(self, *, seed: str | None = None) -> BattleState:
        resolved_seed = seed or self._seed
        raw = self._session.reset(
            character_id=self._character_id,
            encounter_id=self._encounter_id,
            seed=resolved_seed,
            build=self._build,
        )
        self._latest_state = convert_game_bridge_state(
            raw,
            fallback_encounter_id=self._encounter_id,
            fallback_seed=resolved_seed,
            fallback_encounter_class=self._encounter_class,
        )
        return self._latest_state

    def get_state(self) -> BattleState:
        raw = self._session.get_state()
        self._latest_state = convert_game_bridge_state(
            raw,
            fallback_encounter_id=self._encounter_id,
            fallback_seed=self._seed,
            fallback_encounter_class=self._encounter_class,
        )
        return self._latest_state

    def step(self, action_index: int) -> BattleState:
        if self._latest_state is None:
            raise RuntimeError("必须先 reset 再 step")
        raw_action = self._latest_state.raw.get("legal_actions")
        action_rows = raw_action if isinstance(raw_action, list) else []
        action = action_rows[action_index]
        next_raw, _, _, _ = self._session.step(action)
        self._latest_state = convert_game_bridge_state(
            next_raw,
            fallback_encounter_id=self._encounter_id,
            fallback_seed=self._seed,
            fallback_encounter_class=self._encounter_class,
        )
        return self._latest_state

    def close(self) -> None:
        self._session.close()


def _convert_action(raw: dict[str, Any], *, hand_raw: list[dict[str, Any]], enemies_raw: list[dict[str, Any]]) -> LegalAction:
    action_name = str(_pick(raw, "action", "type", default=""))
    action_index = _pick(raw, "index", default=None)
    card_index = _pick(raw, "card_index", default=None)
    target_id = _pick(raw, "target_id", default=None)
    hand_card = _lookup_by_index(hand_raw, card_index)
    target_enemy = _lookup_enemy(enemies_raw, target_id)
    card_id = str(_pick(raw, "card_id", default="") or _pick(hand_card, "id", default=""))
    if not card_id and action_name == "play_card":
        card_id = str(_pick(raw, "label", default=""))
    return LegalAction(
        action_id=_build_action_instance_id(raw),
        action_type=action_name,
        can_execute=bool(_pick(raw, "is_enabled", "can_execute", default=True)),
        card_id=card_id,
        potion_id=str(_pick(raw, "potion_id", default="")),
        special_id=str(_pick(raw, "special_id", default="")),
        target_id=str(target_id or ""),
        cost_now=float(_pick(raw, "cost_now", "cost", default=_pick(hand_card, "cost", default=0.0))),
        damage_now=float(_pick(raw, "damage_now", "damage", default=0.0)),
        block_now=float(_pick(raw, "block_now", "block", default=0.0)),
        magic_now=float(_pick(raw, "magic_now", "magic", default=0.0)),
        tags=_build_action_tags(raw, hand_card=hand_card, action_index=action_index),
        target_summary=_build_target_summary(target_enemy),
        raw=raw,
    )


def _ensure_python_bridge_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    candidate_roots = [
        repo_root / "Python",
        repo_root / "bridge",
    ]
    for root in candidate_roots:
        path = str(root)
        if path not in sys.path:
            sys.path.insert(0, path)


def _pick(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_named_items(items: list[Any]) -> list[str]:
    values: list[str] = []
    for item in items:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            values.append(str(_pick(item, "id", "name", default="")))
    return [value for value in values if value]


def _powers_to_mapping(items: list[Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        if isinstance(item, dict):
            power_id = str(_pick(item, "id", default=""))
            if power_id:
                result[power_id] = float(_pick(item, "amount", default=0.0))
    return result


def _build_hand_tags(card: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    card_type = str(_pick(card, "type", "card_type", default="")).lower()
    if card_type:
        tags.append(card_type)
    if bool(_pick(card, "requires_target", default=False)):
        tags.append("requires_target")
    if bool(_pick(card, "can_play", default=False)):
        tags.append("can_play")
    return tags


def _lookup_by_index(items: list[dict[str, Any]], index: Any) -> dict[str, Any]:
    if not isinstance(index, int) or index < 0:
        return {}
    if index < len(items) and isinstance(items[index], dict):
        return items[index]
    for item in items:
        if isinstance(item, dict) and _pick(item, "index", default=None) == index:
            return item
    return {}


def _lookup_enemy(items: list[dict[str, Any]], target_id: Any) -> dict[str, Any]:
    if target_id is None:
        return {}
    for item in items:
        if isinstance(item, dict) and _pick(item, "target_id", "combat_id", default=None) == target_id:
            return item
    return {}


def _build_target_summary(raw: dict[str, Any]) -> TargetSummary | None:
    if not raw:
        return None
    return TargetSummary(
        hp=float(_pick(raw, "hp", "current_hp", default=0.0)),
        max_hp=float(_pick(raw, "max_hp", default=0.0)),
        block=float(_pick(raw, "block", default=0.0)),
        intent_id=str(_pick(raw, "intent_type", "intent_id", "next_move_id", default="")),
        alive=bool(_pick(raw, "alive", "is_alive", default=True)),
        buffs=_powers_to_mapping(_as_list(raw.get("buffs")) or _as_list(raw.get("powers"))),
    )


def _build_action_tags(raw: dict[str, Any], *, hand_card: dict[str, Any], action_index: Any) -> list[str]:
    tags: list[str] = []
    action_name = str(_pick(raw, "action", default=""))
    if action_name:
        tags.append(action_name)
    if isinstance(action_index, int):
        tags.append(f"index:{action_index}")
    tags.extend(_build_hand_tags(hand_card))
    return tags


def _build_action_instance_id(raw: dict[str, Any]) -> str:
    return "|".join(
        [
            str(_pick(raw, "action", default="")),
            str(_pick(raw, "index", default="")),
            str(_pick(raw, "card_index", default="")),
            str(_pick(raw, "target_id", default="")),
            str(_pick(raw, "col", default="")),
            str(_pick(raw, "row", default="")),
            str(_pick(raw, "slot", default="")),
        ]
    )


def _resolve_encounter_class(encounter_id: str) -> str:
    if not encounter_id:
        return ""
    _ensure_python_bridge_path()
    from game_bridge.catalog.sim_catalog import GAME_CATALOG

    for item in GAME_CATALOG.encounters():
        if str(item.get("encounter_id", "")).lower() == encounter_id.lower():
            return str(item.get("room_type", "") or "")
    return ""
