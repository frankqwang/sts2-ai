"""session 共用的状态/流程语义 helper。"""

from __future__ import annotations

from typing import Any

_OUTCOME_ALIASES = {
    "victory": "victory",
    "win": "victory",
    "won": "victory",
    "success": "victory",
    "defeat": "defeat",
    "loss": "defeat",
    "lost": "defeat",
    "death": "defeat",
    "dead": "defeat",
    "failure": "defeat",
}

# In-combat states. ``card_select`` covers the UI screens triggered by
# HEADBUTT / ARMAMENTS / EXHUME / DUAL_WIELD / etc. — sim sets state_type to
# ``card_select`` and clears the top-level ``enemies`` list (it's rendering a
# pile-selection screen) but the fight is *not* over. Without including
# ``card_select`` here, rollout's left_combat detector aborts every episode
# that plays one of these trigger cards (observed: THE_KIN_BOSS 7-8 step
# left_combat, LAGAVULIN_MATRIARCH same pattern).
COMBAT_STATE_TYPES = frozenset({"monster", "elite", "boss", "hand_select", "card_select"})


def lower_text(value: Any) -> str:
    return str(value or "").lower()


def state_type(state: dict[str, Any] | None) -> str:
    return lower_text((state or {}).get("state_type"))


def is_combat_state(state: dict[str, Any] | None) -> bool:
    return state_type(state) in COMBAT_STATE_TYPES


def is_actionable_combat_state(state: dict[str, Any] | None) -> bool:
    current = state or {}
    current_type = state_type(current)
    if current_type not in COMBAT_STATE_TYPES:
        return False
    # hand_select / card_select are pile-pick UI screens (HEADBUTT, ARMAMENTS,
    # EXHUME, DUAL_WIELD, …). The sim still expects an action — typically
    # ``select_hand_card`` / ``select_card_option`` / ``confirm_selection`` —
    # so we treat them as actionable combat steps even though the top-level
    # ``enemies`` list is empty during these screens.
    if current_type in {"hand_select", "card_select"}:
        return True
    if isinstance(current.get("card_selection"), dict):
        return True

    battle = current.get("battle") or {}
    if not (bool(battle.get("is_play_phase")) and lower_text(battle.get("turn")) == "player"):
        return False

    player = battle.get("player") or current.get("player") or {}
    hand = list(player.get("hand") or [])
    if not hand:
        return True

    if any(bool(card.get("can_play")) for card in hand):
        return True

    reasons = {lower_text(card.get("unplayable_reason")) for card in hand if card.get("unplayable_reason") is not None}
    if reasons and reasons.issubset({"playeractionsdisabled", "disabled", "none"}):
        return False
    return True


def should_wait_for_post_action_combat_settle(state: dict[str, Any] | None) -> bool:
    return is_combat_state(state) and not is_actionable_combat_state(state)


def is_post_action_combat_settled(state: dict[str, Any] | None) -> bool:
    return not is_combat_state(state) or is_actionable_combat_state(state)


def is_menu_ready_for_v2_reset(state: dict[str, Any] | None) -> bool:
    current = state or {}
    if state_type(current) != "menu":
        return True
    menu = current.get("menu")
    if not isinstance(menu, dict):
        return True
    return bool(menu.get("is_main_menu_visible"))


def looks_like_missing_endpoint(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    return any(
        marker in text
        for marker in (
            "http 404",
            "not found",
            "unsupported full run env",
            "unknown api",
        )
    )


def normalize_run_outcome(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return _OUTCOME_ALIASES.get(text, text)


def is_victory_outcome(value: Any) -> bool:
    return normalize_run_outcome(value) == "victory"


def is_failure_outcome(value: Any) -> bool:
    return normalize_run_outcome(value) == "defeat"


__all__ = [
    "COMBAT_STATE_TYPES",
    "is_failure_outcome",
    "is_actionable_combat_state",
    "is_combat_state",
    "is_menu_ready_for_v2_reset",
    "is_post_action_combat_settled",
    "is_victory_outcome",
    "looks_like_missing_endpoint",
    "lower_text",
    "normalize_run_outcome",
    "should_wait_for_post_action_combat_settle",
    "state_type",
]
