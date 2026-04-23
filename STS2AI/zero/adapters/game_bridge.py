from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Iterable

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

_ENGINE_POWER_IDS = {
    "BARRICADE",
    "CORRUPTION",
    "DARK_EMBRACE",
    "DEMON_FORM",
    "EVOLVE",
    "FEEL_NO_PAIN",
    "INFLAME",
    "METALLICIZE",
    "PYRE",
    "RUPTURE",
}
_EXHAUST_PAYOFF_IDS = {
    "DARK_EMBRACE",
    "FEEL_NO_PAIN",
    "PACTS_END",
    "PYRE",
}
_RESOURCE_CARD_IDS = {
    "BLOODLETTING",
    "BURNING_PACT",
    "INFERNAL_BLADE",
    "OFFERING",
    "POMMEL_STRIKE",
    "SHRUG_IT_OFF",
}


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
    selection_raw = _resolve_selection_state(raw, battle_raw)

    enemies_raw = _as_list(raw.get("enemies")) or _as_list(battle_raw.get("enemies"))
    hand_raw = (
        _as_list(battle_raw.get("hand"))
        or _as_list(battle_player_raw.get("hand"))
        or _as_list(raw.get("hand"))
        or _as_list(top_player_raw.get("hand"))
    )
    selection_cards_raw = _as_list(selection_raw.get("cards"))
    legal_actions_raw = _as_list(raw.get("legal_actions"))
    draw_pile_cards = _normalize_named_items(_as_list(battle_raw.get("draw_pile_cards")))
    discard_pile_cards = _normalize_named_items(_as_list(battle_raw.get("discard_pile_cards")))
    exhaust_pile_cards = _normalize_named_items(_as_list(battle_raw.get("exhaust_pile_cards")))
    deck_cards = _normalize_deck_cards(_as_list(top_player_raw.get("deck")))

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
            target_key=str(_pick(item, "target_id", "combat_id", "entity_id", "enemy_id", "id", default="")),
        )
        for index, item in enumerate(enemies_raw)
    ]

    hand = [
        HandCardState(
            card_id=str(_pick(item, "card_id", "id", default="")),
            cost_now=float(_pick(item, "cost_now", "cost", default=0.0)),
            damage_now=_resolve_hand_card_damage(item),
            block_now=_resolve_hand_card_block(item),
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
        draw_pile_size=int(len(draw_pile_cards) or _pick(top_player_raw, "draw_pile_count", default=0)),
        discard_pile_size=int(len(discard_pile_cards) or _pick(top_player_raw, "discard_pile_count", default=0)),
        exhaust_pile_size=int(len(exhaust_pile_cards) or _pick(top_player_raw, "exhaust_pile_count", default=0)),
        draw_cards=draw_pile_cards,
        discard_cards=discard_pile_cards,
        exhaust_cards=exhaust_pile_cards,
        attack_count=sum(1 for item in hand_raw if str(_pick(item, "type", "card_type", default="")).lower() == "attack"),
        skill_count=sum(1 for item in hand_raw if str(_pick(item, "type", "card_type", default="")).lower() == "skill"),
        power_count=sum(1 for item in hand_raw if str(_pick(item, "type", "card_type", default="")).lower() == "power"),
    )

    encounter_id = str(_pick(raw, "encounter_id", default="") or fallback_encounter_id or "")
    encounter_class = fallback_encounter_class or _resolve_encounter_class(encounter_id) or "normal"
    act_value = int(_pick(run_raw, "act", default=0))
    if act_value <= 0:
        act_value = 1
    round_number_raw = _pick(
        raw,
        "round_number_raw",
        default=_pick(battle_raw, "round_number_raw", "round", default=0),
    )
    try:
        turn_id = int(round_number_raw or 0)
    except (TypeError, ValueError):
        turn_id = 0
    context = StaticContext(
        character_id=str(_pick(raw, "character_id", default=ZERO_RUNTIME_DEFAULTS.default_character_id)),
        act=act_value,
        floor=int(_pick(run_raw, "floor", default=0)),
        encounter_class=encounter_class,
        encounter_id=encounter_id,
        deck_cards=deck_cards,
        relics=_normalize_named_items(_as_list(top_player_raw.get("relics"))),
        fixed_powers=list(player_buffs.keys()),
        metadata={
            "seed": str(fallback_seed or ""),
            "state_type": str(_pick(raw, "state_type", default="")),
            "round_number_raw": round_number_raw,
            "turn_id": turn_id,
            "turn_side": str(_pick(battle_raw, "turn_side", default="")),
            **_selection_metadata(selection_raw, hand_raw),
        },
    )

    legal_actions = [
        _convert_action(
            item,
            hand_raw=hand_raw,
            selection_cards_raw=selection_cards_raw,
            enemies_raw=enemies_raw,
        )
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
        self._last_reset_timing: dict[str, float] = {}
        self._last_step_timing: dict[str, float] = {}

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
        session_reset_started_at = time.perf_counter()
        raw = self._session.reset(
            character_id=self._character_id,
            encounter_id=self._encounter_id,
            seed=resolved_seed,
            build=self._build,
        )
        session_reset_duration_s = time.perf_counter() - session_reset_started_at
        transport_metrics = self._get_last_transport_metrics()
        convert_started_at = time.perf_counter()
        self._latest_state = convert_game_bridge_state(
            raw,
            fallback_encounter_id=self._encounter_id,
            fallback_seed=resolved_seed,
            fallback_encounter_class=self._encounter_class,
        )
        state_convert_duration_s = time.perf_counter() - convert_started_at
        self._last_reset_timing = {
            "session_call_duration_s": float(session_reset_duration_s),
            "transport_duration_s": float(transport_metrics.get("total_duration_s", 0.0) or 0.0),
            "transport_write_duration_s": float(transport_metrics.get("write_duration_s", 0.0) or 0.0),
            "transport_read_duration_s": float(transport_metrics.get("read_duration_s", 0.0) or 0.0),
            "transport_decode_duration_s": float(transport_metrics.get("decode_duration_s", 0.0) or 0.0),
            "state_convert_duration_s": float(state_convert_duration_s),
        }
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
        legal_actions = self._latest_state.legal_actions
        if action_index < 0 or action_index >= len(action_rows):
            if action_index < 0 or action_index >= len(legal_actions):
                raise IndexError(
                    f"action_index 越界: index={action_index} raw={len(action_rows)} legal={len(legal_actions)}"
                )
            resolved_index = _resolve_raw_action_index(legal_actions[action_index], action_rows)
            if resolved_index is None:
                raise IndexError(
                    f"无法把 legal action 映射回 raw action: index={action_index} "
                    f"action_id={legal_actions[action_index].action_id} raw={len(action_rows)} legal={len(legal_actions)}"
                )
            action_index = resolved_index
        action = action_rows[action_index]
        session_step_started_at = time.perf_counter()
        next_raw, _, _, _ = self._session.step(action)
        session_step_duration_s = time.perf_counter() - session_step_started_at
        transport_metrics = self._get_last_transport_metrics()
        convert_started_at = time.perf_counter()
        self._latest_state = convert_game_bridge_state(
            next_raw,
            fallback_encounter_id=self._encounter_id,
            fallback_seed=self._seed,
            fallback_encounter_class=self._encounter_class,
        )
        state_convert_duration_s = time.perf_counter() - convert_started_at
        self._last_step_timing = {
            "session_call_duration_s": float(session_step_duration_s),
            "transport_duration_s": float(transport_metrics.get("total_duration_s", 0.0) or 0.0),
            "transport_write_duration_s": float(transport_metrics.get("write_duration_s", 0.0) or 0.0),
            "transport_read_duration_s": float(transport_metrics.get("read_duration_s", 0.0) or 0.0),
            "transport_decode_duration_s": float(transport_metrics.get("decode_duration_s", 0.0) or 0.0),
            "state_convert_duration_s": float(state_convert_duration_s),
        }
        return self._latest_state

    def save_state(self) -> str:
        return str(self._session.save_state())

    def load_state(self, state_id: str) -> BattleState:
        raw = self._session.load_state(state_id)
        self._latest_state = convert_game_bridge_state(
            raw,
            fallback_encounter_id=self._encounter_id,
            fallback_seed=self._seed,
            fallback_encounter_class=self._encounter_class,
        )
        return self._latest_state

    def delete_state(self, state_id: str) -> None:
        delete_hook = getattr(self._session, "delete_state", None)
        if callable(delete_hook):
            delete_hook(state_id)

    def close(self) -> None:
        self._session.close()

    def get_last_reset_timing(self) -> dict[str, float]:
        return dict(self._last_reset_timing)

    def get_last_step_timing(self) -> dict[str, float]:
        return dict(self._last_step_timing)

    def _get_last_transport_metrics(self) -> dict[str, Any]:
        hook = getattr(self._session, "get_last_transport_metrics", None)
        if callable(hook):
            return dict(hook() or {})
        return {}


_HAND_CARD_ACTION_TYPES = {
    "play_card",
    "select_hand_card",
    "select_card",
    "select_card_option",
    "combat_select_card",
}


def _convert_action(
    raw: dict[str, Any],
    *,
    hand_raw: list[dict[str, Any]],
    selection_cards_raw: list[dict[str, Any]],
    enemies_raw: list[dict[str, Any]],
) -> LegalAction:
    action_name = str(_pick(raw, "action", "type", default=""))
    action_index = _pick(raw, "index", default=None)
    card_index = _pick(raw, "card_index", default=None)
    target_id = _pick(raw, "target_id", default=None)
    if action_name in _HAND_CARD_ACTION_TYPES:
        hand_card = (
            _lookup_by_index(selection_cards_raw, card_index)
            or _lookup_by_index(hand_raw, card_index)
        )
    else:
        hand_card = {}
    target_enemy = _lookup_enemy(enemies_raw, target_id)
    card_id = str(_pick(raw, "card_id", default="") or _pick(hand_card, "id", default=""))
    if not card_id and action_name == "play_card":
        card_id = str(_pick(raw, "label", default=""))
    damage_now = _resolve_action_damage(raw, hand_card=hand_card, card_id=card_id)
    block_now = _resolve_action_block(raw, hand_card=hand_card, card_id=card_id)
    return LegalAction(
        action_id=_build_action_instance_id(raw),
        action_type=action_name,
        can_execute=bool(_pick(raw, "is_enabled", "can_execute", default=True)),
        card_id=card_id,
        potion_id=str(_pick(raw, "potion_id", default="")),
        special_id=str(_pick(raw, "special_id", default="")),
        target_id=str(target_id or ""),
        cost_now=float(_pick(raw, "cost_now", "cost", default=_pick(hand_card, "cost", default=0.0) if hand_card else 0.0)),
        damage_now=damage_now,
        block_now=block_now,
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


def _pick_number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _resolve_selection_state(raw: dict[str, Any], battle_raw: dict[str, Any]) -> dict[str, Any]:
    return (
        _as_dict(raw.get("hand_select"))
        or _as_dict(raw.get("card_select"))
        or _as_dict(battle_raw.get("card_selection"))
    )


def _selection_metadata(selection_raw: dict[str, Any], hand_raw: list[dict[str, Any]]) -> dict[str, str | float | int | bool]:
    if not selection_raw:
        return {
            "submenu_selected_count": 0,
            "submenu_max_select": 0,
            "submenu_remaining_slots": 0,
            "submenu_can_confirm": False,
            "submenu_can_cancel": False,
            "submenu_selected_engine_count": 0.0,
            "submenu_selected_payoff_count": 0.0,
            "submenu_selected_resource_count": 0.0,
        }
    selected_cards_raw = _as_list(selection_raw.get("selected_cards"))
    selected_card_ids = _normalize_named_items(selected_cards_raw)
    max_select = int(_pick(selection_raw, "max_select", default=0) or 0)
    selected_count = int(_pick(selection_raw, "selected_count", default=len(selected_card_ids)) or 0)
    if max_select <= 0 and selected_count > 0:
        selectable_count = len(_as_list(selection_raw.get("cards"))) or len(hand_raw)
        max_select = selected_count + selectable_count
    semantic_counts = _semantic_counts_for_cards(selected_card_ids)
    remaining_slots = max(0, max_select - selected_count) if max_select > 0 else 0
    return {
        "submenu_selected_count": int(selected_count),
        "submenu_max_select": int(max_select),
        "submenu_remaining_slots": int(remaining_slots),
        "submenu_can_confirm": bool(_pick(selection_raw, "can_confirm", default=False)),
        "submenu_can_cancel": bool(_pick(selection_raw, "can_cancel", default=False)),
        "submenu_selected_engine_count": float(semantic_counts["engine"]),
        "submenu_selected_payoff_count": float(semantic_counts["payoff"]),
        "submenu_selected_resource_count": float(semantic_counts["resource"]),
    }


def _normalize_named_items(items: list[Any]) -> list[str]:
    values: list[str] = []
    for item in items:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            values.append(str(_pick(item, "id", "name", default="")))
    return [value for value in values if value]


def _normalize_deck_cards(items: list[Any]) -> list[str]:
    cards: list[str] = []
    for item in items:
        if isinstance(item, str):
            card_id = item
        elif isinstance(item, dict):
            card_id = str(_pick(item, "id", "card_id", "name", default=""))
            upgrades = int(_pick(item, "upgrade_level", "upgrades", default=0) or 0)
            if upgrades > 0 and card_id:
                card_id = f"{card_id}+{upgrades}"
        else:
            continue
        if card_id:
            cards.append(card_id)
    return cards


def _semantic_counts_for_cards(card_ids: Iterable[str]) -> dict[str, float]:
    counts = {"engine": 0.0, "payoff": 0.0, "resource": 0.0}
    for raw_id in card_ids:
        normalized_id = str(raw_id or "").upper().replace("+", "").strip()
        if normalized_id in _ENGINE_POWER_IDS:
            counts["engine"] += 1.0
        if normalized_id in _EXHAUST_PAYOFF_IDS:
            counts["payoff"] += 1.0
        if normalized_id in _RESOURCE_CARD_IDS:
            counts["resource"] += 1.0
    return counts


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


def _resolve_hand_card_damage(card: dict[str, Any]) -> float:
    explicit = _pick_number(card, "damage_now", "damage")
    if explicit is not None:
        return explicit
    damage, _ = _infer_card_effects(
        card_id=str(_pick(card, "card_id", "id", default="")),
        card_type=str(_pick(card, "type", "card_type", default="")),
        cost_now=float(_pick(card, "cost_now", "cost", default=0.0)),
        tags=_build_hand_tags(card),
    )
    return damage


def _resolve_hand_card_block(card: dict[str, Any]) -> float:
    explicit = _pick_number(card, "block_now", "block")
    if explicit is not None:
        return explicit
    _, block = _infer_card_effects(
        card_id=str(_pick(card, "card_id", "id", default="")),
        card_type=str(_pick(card, "type", "card_type", default="")),
        cost_now=float(_pick(card, "cost_now", "cost", default=0.0)),
        tags=_build_hand_tags(card),
    )
    return block


def _resolve_action_damage(raw: dict[str, Any], *, hand_card: dict[str, Any], card_id: str) -> float:
    explicit = _pick_number(raw, "damage_now", "damage")
    if explicit is not None:
        return explicit
    hand_explicit = _pick_number(hand_card, "damage_now", "damage")
    if hand_explicit is not None:
        return hand_explicit
    damage, _ = _infer_card_effects(
        card_id=card_id,
        card_type=str(_pick(hand_card, "type", "card_type", default="")),
        cost_now=float(_pick(raw, "cost_now", "cost", default=_pick(hand_card, "cost", default=0.0) if hand_card else 0.0)),
        tags=_build_action_tags(raw, hand_card=hand_card, action_index=_pick(raw, "index", default=None)),
    )
    return damage


def _resolve_action_block(raw: dict[str, Any], *, hand_card: dict[str, Any], card_id: str) -> float:
    explicit = _pick_number(raw, "block_now", "block")
    if explicit is not None:
        return explicit
    hand_explicit = _pick_number(hand_card, "block_now", "block")
    if hand_explicit is not None:
        return hand_explicit
    _, block = _infer_card_effects(
        card_id=card_id,
        card_type=str(_pick(hand_card, "type", "card_type", default="")),
        cost_now=float(_pick(raw, "cost_now", "cost", default=_pick(hand_card, "cost", default=0.0) if hand_card else 0.0)),
        tags=_build_action_tags(raw, hand_card=hand_card, action_index=_pick(raw, "index", default=None)),
    )
    return block


def _infer_card_effects(*, card_id: str, card_type: str, cost_now: float, tags: list[str]) -> tuple[float, float]:
    normalized_id = str(card_id or "").strip().upper()
    normalized_type = str(card_type or "").strip().lower()
    tag_set = {str(tag).strip().lower() for tag in tags}
    is_attack = normalized_type == "attack" or "attack" in tag_set
    is_skill = normalized_type == "skill" or "skill" in tag_set
    energy_cost = max(0.0, float(cost_now or 0.0))

    damage = 0.0
    block = 0.0
    if normalized_id.startswith("STRIKE"):
        damage = 6.0
    elif normalized_id.startswith("DEFEND"):
        block = 5.0
    elif any(token in normalized_id for token in ("BODYGUARD", "SHROUD", "GUARD", "WARD", "PROTECT", "BARRIER", "SHIELD")):
        block = max(5.0, 4.0 + 2.0 * energy_cost)
    elif is_attack:
        damage = max(4.0, 4.0 + 2.0 * energy_cost)
    elif is_skill and any(token in normalized_id for token in ("BLOCK", "LEAP", "GLACIER")):
        block = max(5.0, 4.0 + 2.0 * energy_cost)
    return float(damage), float(block)


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


def _resolve_raw_action_index(action: LegalAction, raw_actions: list[dict[str, Any]]) -> int | None:
    for index, raw_action in enumerate(raw_actions):
        if _build_action_instance_id(raw_action) == action.action_id:
            return index
    for index, raw_action in enumerate(raw_actions):
        raw_type = str(_pick(raw_action, "action", "type", default=""))
        raw_card_id = str(_pick(raw_action, "card_id", default="") or _pick(raw_action, "label", default=""))
        raw_target_id = str(_pick(raw_action, "target_id", default="") or "")
        if raw_type == action.action_type and raw_card_id == action.card_id and raw_target_id == action.target_id:
            return index
    for index, raw_action in enumerate(raw_actions):
        raw_type = str(_pick(raw_action, "action", "type", default=""))
        raw_card_id = str(_pick(raw_action, "card_id", default="") or _pick(raw_action, "label", default=""))
        if raw_type == action.action_type and raw_card_id == action.card_id:
            return index
    return None


def _resolve_encounter_class(encounter_id: str) -> str:
    if not encounter_id:
        return ""
    _ensure_python_bridge_path()
    from game_bridge.catalog.sim_catalog import GAME_CATALOG

    for item in GAME_CATALOG.encounters():
        if str(item.get("encounter_id", "")).lower() == encounter_id.lower():
            return str(item.get("room_type", "") or "")
    return ""
