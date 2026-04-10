from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable

import numpy as np

from test_simulator_consistency import COMBAT_TYPES
from verify_save_load import choose_default_action

SELECTION_ACTION_NAMES = {
    "select_card",
    "combat_select_card",
    "combat_confirm_selection",
    "confirm_selection",
    "cancel_selection",
    "skip_relic_selection",
}
SELECTION_SCREENS = {"card_select", "hand_select", "relic_select"}
POST_CARD_REWARD_ACTIONS = {"select_card_reward", "skip_card_reward"}


@dataclass(slots=True)
class RolloutDecision:
    action: dict[str, Any]
    source: str


def legal_action_name_set(legal: list[dict[str, Any]]) -> set[str]:
    return {
        str(action.get("action") or "").strip().lower()
        for action in legal
        if isinstance(action, dict)
    }


def is_selection_screen(state_type: str, legal: list[dict[str, Any]]) -> bool:
    st = (state_type or "").strip().lower()
    return st in SELECTION_SCREENS or bool(legal_action_name_set(legal) & SELECTION_ACTION_NAMES)


def combat_rewards_state(state: dict[str, Any]) -> dict[str, Any]:
    rewards_state = state.get("combat_rewards")
    if isinstance(rewards_state, dict):
        return rewards_state
    rewards_state = state.get("rewards")
    if isinstance(rewards_state, dict):
        return rewards_state
    return {}


def reward_item_claimable(state: dict[str, Any], reward_item: dict[str, Any] | None) -> bool:
    if not isinstance(reward_item, dict):
        return True
    explicit = reward_item.get("claimable")
    if explicit is not None:
        return bool(explicit)
    reward_type = str(reward_item.get("type") or "").strip().lower()
    if reward_type != "potion":
        return True
    rewards_state = combat_rewards_state(state)
    player = rewards_state.get("player") or state.get("player") or {}
    try:
        open_slots = int(player.get("open_potion_slots", 0) or 0)
    except Exception:
        open_slots = 0
    return open_slots > 0


def choose_claimable_reward_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    *,
    prefer_highest_index: bool = False,
) -> dict[str, Any] | None:
    rewards_state = combat_rewards_state(state)
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
    claimable_actions: list[dict[str, Any]] = []
    for action in legal:
        if str(action.get("action") or "").strip().lower() != "claim_reward":
            continue
        try:
            reward_item = indexed_items.get(int(action.get("index", -1)))
        except Exception:
            reward_item = None
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
                claimable_actions.append(enriched_action)
            continue
        if reward_item_claimable(state, reward_item):
            claimable_actions.append(enriched_action)
    if claimable_actions:
        key_fn = lambda action: int(action.get("index") or (-1 if prefer_highest_index else 999))
        return max(claimable_actions, key=key_fn) if prefer_highest_index else min(claimable_actions, key=key_fn)
    return fallback


def claim_reward_action_count(legal: list[dict[str, Any]]) -> int:
    try:
        return sum(
            1
            for action in legal
            if isinstance(action, dict)
            and str(action.get("action") or "").strip().lower() == "claim_reward"
            and action.get("is_enabled") is not False
        )
    except Exception:
        return 0


def reward_claim_signature(state: dict[str, Any], action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    if str(action.get("action") or "").strip().lower() != "claim_reward":
        return ""
    remaining_claim_actions = claim_reward_action_count(state.get("legal_actions") or [])
    parts = [
        str(action.get("action") or "").strip().lower(),
        str(action.get("label") or "").strip().lower(),
        str(action.get("reward_type") or "").strip().lower(),
        str(action.get("reward_id") or action.get("id") or "").strip().lower(),
        str(action.get("reward_key") or "").strip().lower(),
        str(remaining_claim_actions),
    ]
    return "|".join(parts)


def next_reward_claim_signature(
    state_type: str,
    state: dict[str, Any],
    action: dict[str, Any] | None,
) -> str:
    if (state_type or "").strip().lower() != "combat_rewards":
        return ""
    return reward_claim_signature(state, action)


def choose_selection_action(state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not legal:
        return None
    if not is_selection_screen(str(state.get("state_type") or ""), legal):
        return None

    confirm = next(
        (
            action
            for action in legal
            if str(action.get("action") or "").strip().lower()
            in {"confirm_selection", "combat_confirm_selection", "skip_relic_selection", "cancel_selection"}
        ),
        None,
    )
    if confirm is not None:
        return confirm

    select = next(
        (
            action
            for action in legal
            if "select" in str(action.get("action") or "").strip().lower()
        ),
        None,
    )
    if select is not None:
        return select
    return legal[0]


def choose_reward_progress_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> dict[str, Any] | None:
    st = (state.get("state_type") or "").strip().lower()
    if st != "combat_rewards":
        return None
    claim_action = choose_claimable_reward_action(state, legal, prefer_highest_index=False)
    if claim_action is not None:
        return claim_action
    for action in legal:
        if str(action.get("action") or "").strip().lower() in {"proceed", "skip"}:
            return action
    return None


def choose_auto_progress_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    last_action_name: str | None = None,
    last_reward_claim_sig: str | None = None,
    last_reward_claim_count: int | None = None,
    reward_chain_card_reward_seen: bool = False,
) -> dict[str, Any] | None:
    st = (state.get("state_type") or "").strip().lower()
    last_action_name = str(last_action_name or "").strip().lower()
    last_reward_claim_sig = str(last_reward_claim_sig or "").strip().lower()

    selection_action = choose_selection_action(state, legal)
    if selection_action is not None:
        return selection_action

    if st == "combat_rewards":
        current_claim_count = claim_reward_action_count(legal)
        claim_action = choose_claimable_reward_action(
            state,
            legal,
            prefer_highest_index=not reward_chain_card_reward_seen,
        )
        repeated_claim = False
        if claim_action is not None:
            claim_sig = reward_claim_signature(state, claim_action)
            repeated_claim = bool(claim_sig and claim_sig == last_reward_claim_sig)
        stalled_after_card_reward = bool(
            reward_chain_card_reward_seen
            and claim_action is not None
            and last_reward_claim_count is not None
            and current_claim_count >= max(0, int(last_reward_claim_count))
        )
        if last_action_name in POST_CARD_REWARD_ACTIONS:
            for action in legal:
                if str(action.get("action") or "").strip().lower() in {"proceed", "skip"}:
                    return action
        if stalled_after_card_reward or (reward_chain_card_reward_seen and repeated_claim):
            for action in legal:
                if str(action.get("action") or "").strip().lower() in {"proceed", "skip"}:
                    return action
        if claim_action is not None:
            return claim_action
        for action in legal:
            if str(action.get("action") or "").strip().lower() in {"proceed", "skip"}:
                return action

    if st == "event":
        event_state = state.get("event") or {}
        if event_state.get("in_dialogue"):
            for action in legal:
                if str(action.get("action") or "").strip().lower() == "advance_dialogue":
                    return action
        if last_action_name in POST_CARD_REWARD_ACTIONS and (
            event_state.get("can_proceed") or event_state.get("is_finished")
        ):
            for action in legal:
                if str(action.get("action") or "").strip().lower() == "proceed":
                    return action
        has_explicit_event_choice = any(
            str(action.get("action") or "").strip().lower() == "choose_event_option"
            for action in legal
        )
        if not has_explicit_event_choice:
            for action_name in ("advance_dialogue", "proceed"):
                for action in legal:
                    if str(action.get("action") or "").strip().lower() == action_name:
                        return action

    return None


def _combat_rollout_decision(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    rng: random.Random,
    combat_evaluator: Any | None = None,
) -> RolloutDecision:
    if combat_evaluator is not None:
        try:
            policy, _ = combat_evaluator.evaluate(state, legal)
            idx = int(np.argmax(policy)) if len(policy) > 0 else 0
            return RolloutDecision(action=legal[idx], source="combat_net")
        except Exception:
            pass

    play_cards = [a for a in legal if a.get("action") == "play_card"]
    potions = [a for a in legal if a.get("action") == "use_potion"]
    end_turns = [a for a in legal if a.get("action") == "end_turn"]
    if play_cards:
        return RolloutDecision(action=rng.choice(play_cards), source="combat_fallback")
    if potions:
        return RolloutDecision(action=rng.choice(potions), source="combat_fallback")
    if end_turns:
        return RolloutDecision(action=end_turns[0], source="combat_fallback")
    return RolloutDecision(action=legal[0], source="combat_fallback")


def choose_rollout_decision(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    rng: random.Random,
    *,
    combat_evaluator: Any | None = None,
    ppo_policy: Any | None = None,
    fallback_action_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> RolloutDecision:
    if not legal:
        return RolloutDecision(action={"action": "wait"}, source="wait")

    selection_action = choose_selection_action(state, legal)
    if selection_action is not None:
        return RolloutDecision(action=selection_action, source="selection_rule")

    reward_action = choose_reward_progress_action(state, legal)
    if reward_action is not None:
        return RolloutDecision(action=reward_action, source="reward_progression")

    st = str(state.get("state_type") or "").strip().lower()
    if st in COMBAT_TYPES:
        return _combat_rollout_decision(state, legal, rng, combat_evaluator)

    if ppo_policy is not None:
        try:
            action = ppo_policy.choose_action(state, legal)
            if action:
                return RolloutDecision(action=action, source="ppo_policy")
        except Exception:
            pass

    fallback = (fallback_action_fn or choose_default_action)(state)
    return RolloutDecision(action=fallback, source="fallback")

