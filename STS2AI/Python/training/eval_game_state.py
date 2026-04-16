"""Game state tracking — progress extraction, loop detection, auto-progress actions.

Extracted from evaluate_ai.py. These functions track game state transitions,
detect infinite loops, compute state signatures, and handle automatic progression
through non-decision screens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from network.state_features import _lower, _safe_float, _safe_int

SELECTION_SCREENS = {"card_select", "hand_select", "relic_select"}


def _combat_rewards_state(state: dict) -> dict:
    rewards_state = state.get("combat_rewards")
    if isinstance(rewards_state, dict):
        return rewards_state
    rewards_state = state.get("rewards")
    if isinstance(rewards_state, dict):
        return rewards_state
    return {}
SELECTION_ACTION_NAMES = {
    "select_card", "combat_select_card", "combat_confirm_selection",
    "confirm_selection", "cancel_selection", "select_card_option",
}



def _legal_action_name_set(legal: list[dict[str, Any]]) -> set[str]:
    return {
        str(action.get("action") or "")
        for action in legal
        if isinstance(action, dict)
    }



def _combat_loop_progress_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") if isinstance(battle.get("player"), dict) else (_extract_player_snapshot(state) or {})
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
    player_hand = player.get("hand") if isinstance(player.get("hand"), list) else []
    draw_pile = player.get("draw_pile") if isinstance(player.get("draw_pile"), list) else []
    discard_pile = player.get("discard_pile") if isinstance(player.get("discard_pile"), list) else []
    exhaust_pile = player.get("exhaust_pile") if isinstance(player.get("exhaust_pile"), list) else []
    total_enemy_hp = 0
    total_enemy_block = 0
    alive_enemy_count = 0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        block = _safe_int(enemy.get("block"), 0)
        if hp > 0:
            alive_enemy_count += 1
            total_enemy_hp += hp
            total_enemy_block += block
    return (
        _safe_int(player.get("hp", player.get("current_hp")), 0),
        _safe_int(player.get("block"), 0),
        _safe_int(player.get("energy", player.get("current_energy")), 0),
        len(player_hand),
        len(draw_pile),
        len(discard_pile),
        len(exhaust_pile),
        alive_enemy_count,
        total_enemy_hp,
        total_enemy_block,
    )



def _loop_progress_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    progress = _extract_progress(state)
    state_type = progress["state_type"]
    if state_type in COMBAT_SCREENS:
        return (
            state_type,
            progress["act"],
            progress["floor"],
            *_combat_loop_progress_signature(state),
        )
    return (
        state_type,
        progress["act"],
        progress["floor"],
        progress["hp"],
        progress["gold"],
        progress["deck_count"],
        progress["relic_count"],
        progress["potion_count"],
    )



def _is_selection_screen(state_type: str, legal: list[dict[str, Any]]) -> bool:
    st = (state_type or "").strip().lower()
    return st in SELECTION_SCREENS or bool(
        _legal_action_name_set(legal) & SELECTION_ACTION_NAMES
    )



def _choose_auto_progress_action(
    state: dict[str, Any],
    state_type: str,
    legal: list[dict[str, Any]],
    last_reward_claim_sig: str | None = None,
) -> dict[str, Any] | None:
    st = (state_type or "").strip().lower()
    last_reward_claim_sig = str(last_reward_claim_sig or "").strip().lower()

    if _is_selection_screen(st, legal):
        for action in legal:
            action_name = str(action.get("action") or "")
            if "confirm" in action_name or "skip" in action_name:
                return action
        for action in legal:
            if "select" in str(action.get("action") or ""):
                return action

    if st == "combat_rewards":
        claim_action = _choose_claimable_reward_action(state, legal)
        repeated_claim = False
        if claim_action is not None:
            claim_sig = _reward_claim_signature(state, claim_action)
            repeated_claim = bool(claim_sig and claim_sig == last_reward_claim_sig)
        if repeated_claim:
            for action in legal:
                if action.get("action") in ("proceed", "skip"):
                    return action
        if claim_action is not None:
            return claim_action
        for action in legal:
            if action.get("action") in ("proceed", "skip"):
                return action

    return None



def _reward_item_claimable(state: dict[str, Any], reward_item: dict[str, Any] | None) -> bool:
    if not isinstance(reward_item, dict):
        return True
    explicit = reward_item.get("claimable")
    if explicit is not None:
        return bool(explicit)
    reward_type = str(reward_item.get("type") or "").strip().lower()
    if reward_type != "potion":
        return True
    rewards_state = _combat_rewards_state(state)
    player = rewards_state.get("player") or state.get("player") or {}
    try:
        open_slots = int(player.get("open_potion_slots", 0) or 0)
    except Exception:
        open_slots = 0
    return open_slots > 0



def _choose_claimable_reward_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> dict[str, Any] | None:
    rewards_state = _combat_rewards_state(state)
    items = rewards_state.get("items")
    indexed_items: dict[int, dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except Exception:
                continue
            indexed_items[idx] = item
    fallback: dict[str, Any] | None = None
    for action in legal:
        if action.get("action") != "claim_reward":
            continue
        reward_item = indexed_items.get(int(action.get("index", -1)))
        enriched_action = dict(action)
        for src_key, dst_key in (
            ("reward_type", "reward_type"),
            ("reward_id", "reward_id"),
            ("reward_key", "reward_key"),
            ("reward_source", "reward_source"),
            ("claimable", "claimable"),
            ("claim_block_reason", "claim_block_reason"),
        ):
            if action.get(src_key) is not None and enriched_action.get(dst_key) is None:
                enriched_action[dst_key] = action.get(src_key)
        if isinstance(reward_item, dict):
            for src_key, dst_key in (
                ("type", "reward_type"),
                ("id", "reward_id"),
                ("reward_key", "reward_key"),
                ("reward_source", "reward_source"),
                ("claimable", "claimable"),
                ("claim_block_reason", "claim_block_reason"),
            ):
                if reward_item.get(src_key) is not None and not enriched_action.get(dst_key):
                    enriched_action[dst_key] = reward_item.get(src_key)
        if fallback is None:
            fallback = enriched_action
        explicit_claimable = enriched_action.get("claimable")
        if explicit_claimable is not None:
            if bool(explicit_claimable):
                return enriched_action
            continue
        if _reward_item_claimable(state, reward_item):
            return enriched_action
    return fallback



def _reward_claim_signature(state: dict[str, Any], action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    if str(action.get("action") or "").strip().lower() != "claim_reward":
        return ""
    remaining_claim_actions = 0
    try:
        remaining_claim_actions = sum(
            1
            for legal_action in (state.get("legal_actions") or [])
            if isinstance(legal_action, dict)
            and str(legal_action.get("action") or "").strip().lower() == "claim_reward"
            and legal_action.get("is_enabled") is not False
        )
    except Exception:
        remaining_claim_actions = 0
    parts = [
        str(action.get("action") or "").strip().lower(),
        str(action.get("label") or "").strip().lower(),
        str(action.get("reward_type") or "").strip().lower(),
        str(action.get("reward_id") or action.get("id") or "").strip().lower(),
        str(action.get("reward_key") or "").strip().lower(),
        str(remaining_claim_actions),
    ]
    return "|".join(parts)



def _next_reward_claim_signature(
    state_type: str,
    state: dict[str, Any],
    action: dict[str, Any] | None,
) -> str:
    if (state_type or "").strip().lower() != "combat_rewards":
        return ""
    return _reward_claim_signature(state, action)



@dataclass
class RepeatLoopTracker:
    trigger_count: int = 3
    max_repeats: int = 20
    last_transition_key: tuple[Any, ...] | None = None
    last_action_signature: str = ""
    repeat_count: int = 0

    def observe_transition(
        self,
        before_state: dict[str, Any],
        before_legal: list[dict[str, Any]],
        action: dict[str, Any] | None,
        after_state: dict[str, Any],
    ) -> int:
        action_sig = _action_signature(action)
        before_sig = _loop_state_signature(before_state, before_legal)
        after_sig = _loop_state_signature(after_state, _enabled_legal_actions(after_state))
        before_progress = _loop_progress_signature(before_state)
        after_progress = _loop_progress_signature(after_state)
        current_key = (before_sig, action_sig, after_sig)
        stagnant = before_sig == after_sig and before_progress == after_progress
        repeated_action = bool(action_sig) and action_sig == self.last_action_signature
        if stagnant and repeated_action and current_key == self.last_transition_key:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.last_transition_key = current_key
        self.last_action_signature = action_sig
        return self.repeat_count

    def choose_escape_action(
        self,
        legal: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if self.repeat_count < self.trigger_count:
            return None
        return _choose_repeat_escape_action(legal, self.last_action_signature)

    def should_abort(self) -> bool:
        return self.repeat_count >= self.max_repeats



def _extract_progress(state: dict[str, Any]) -> dict[str, Any]:
    run_state = state.get("run") if isinstance(state.get("run"), dict) else {}
    player = _extract_player_snapshot(state) or {}
    deck = player.get("deck") if isinstance(player.get("deck"), list) else []
    relics = player.get("relics") if isinstance(player.get("relics"), list) else []
    potions = player.get("potions") if isinstance(player.get("potions"), list) else []
    return {
        "state_type": _lower(state.get("state_type")),
        "act": _safe_int(run_state.get("act"), 0),
        "floor": _safe_int(run_state.get("floor"), 0),
        "hp": _safe_int(player.get("hp", player.get("current_hp")), 0),
        "max_hp": max(1, _safe_int(player.get("max_hp"), 1)),
        "gold": _safe_int(player.get("gold"), 0),
        "deck_count": len(deck),
        "relic_count": len(relics),
        "potion_count": len(potions),
    }



def _compute_delta(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    before_progress: dict[str, Any],
    after_progress: dict[str, Any],
) -> dict[str, Any]:
    before_safe = _json_safe_tree(before_state)
    after_safe = _json_safe_tree(after_state)
    changed_keys: list[str] = []
    before_keys = set(before_safe.keys()) if isinstance(before_safe, dict) else set()
    after_keys = set(after_safe.keys()) if isinstance(after_safe, dict) else set()
    for key in sorted(before_keys | after_keys):
        before_value = before_safe.get(key) if isinstance(before_safe, dict) else None
        after_value = after_safe.get(key) if isinstance(after_safe, dict) else None
        if before_value != after_value:
            changed_keys.append(key)

    return {
        "state_changed": before_safe != after_safe,
        "state_type_changed": before_progress["state_type"] != after_progress["state_type"],
        "changed_top_level_keys": changed_keys,
        "act_delta": after_progress["act"] - before_progress["act"],
        "floor_delta": after_progress["floor"] - before_progress["floor"],
        "hp_delta": after_progress["hp"] - before_progress["hp"],
        "max_hp_delta": after_progress["max_hp"] - before_progress["max_hp"],
        "gold_delta": after_progress["gold"] - before_progress["gold"],
        "deck_count_delta": after_progress["deck_count"] - before_progress["deck_count"],
        "relic_count_delta": after_progress["relic_count"] - before_progress["relic_count"],
        "potion_count_delta": after_progress["potion_count"] - before_progress["potion_count"],
    }

