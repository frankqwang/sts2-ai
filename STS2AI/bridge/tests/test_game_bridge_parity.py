from __future__ import annotations

import sys
from pathlib import Path

_python_root = Path(__file__).resolve().parents[1]
if str(_python_root) not in sys.path:
    sys.path.insert(0, str(_python_root))

from game_bridge.parity import (
    compare_states,
    canonicalize_state,
    _choose_default_action,
    _find_matching_action,
    _resolve_parity_seed,
)


def test_find_matching_action_ignores_label_only():
    reference = {
        "action": "play_card",
        "card_index": 0,
        "target_id": 1,
        "label": "Strike",
    }
    legal_actions = [
        {
            "action": "play_card",
            "card_index": 0,
            "target_id": 1,
            "label": "Strike+",
        }
    ]

    matched = _find_matching_action(legal_actions, reference)

    assert matched is not None
    assert matched["target_id"] == 1


def test_canonicalize_state_summarizes_event_screen():
    state = {
        "state_type": "event",
        "terminal": False,
        "run_outcome": None,
        "run": {"act": 1, "floor": 3},
        "event": {
            "event_id": "EVENT.BigFish",
            "player": {"hp": 50, "max_hp": 80, "gold": 99},
            "in_dialogue": True,
            "is_finished": False,
            "options": [
                {"index": 0, "id": "take", "label": "Take", "text": "Take the relic", "is_locked": False, "is_proceed": False}
            ],
        },
        "legal_actions": [{"action": "choose_event_option", "index": 0, "label": "Take"}],
    }

    canonical = canonicalize_state(state)

    assert canonical["state_type"] == "event"
    assert canonical["payload"]["event_id"] == "BigFish"
    assert canonical["legal_actions"][0]["action"] == "choose_event_option"
    assert canonical["player"]["hp"] == 50


def test_compare_states_reports_screen_payload_diff():
    left = {
        "state_type": "event",
        "terminal": False,
        "run_outcome": None,
        "run": {"act": 1, "floor": 3},
        "player": {},
        "legal_actions": [{"action": "choose_event_option", "index": 0, "label": "Take"}],
        "event": {
            "event_id": "BigFish",
            "in_dialogue": True,
            "is_finished": False,
            "options": [{"index": 0, "id": "take", "label": "Take", "text": "Take", "is_locked": False, "is_proceed": False}],
        },
    }
    right = {
        **left,
        "event": {
            "event_id": "BigFish",
            "in_dialogue": True,
            "is_finished": False,
            "options": [{"index": 0, "id": "leave", "label": "Leave", "text": "Leave", "is_locked": False, "is_proceed": False}],
        },
    }

    diffs = compare_states(left, right)

    assert any("payload.options[0].id" in diff for diff in diffs)


def test_resolve_parity_seed_uses_shared_default():
    assert _resolve_parity_seed(None) == "123456"
    assert _resolve_parity_seed("") == "123456"
    assert _resolve_parity_seed(" 789 ") == "789"


def test_choose_default_action_prefers_confirm():
    action = _choose_default_action(
        [
            {"action": "combat_select_card", "card_index": 0},
            {"action": "combat_confirm_selection"},
        ]
    )

    assert action == {"action": "combat_confirm_selection"}


def test_canonicalize_state_uses_enemy_status_as_power_fallback():
    state = {
        "state_type": "monster",
        "terminal": False,
        "run_outcome": None,
        "run": {"act": 1, "floor": 1},
        "battle": {
            "player": {"hp": 80, "max_hp": 80, "gold": 99, "block": 0, "energy": 3, "max_energy": 3},
            "enemies": [
                {
                    "combat_id": 1,
                    "hp": 20,
                    "max_hp": 20,
                    "block": 0,
                    "is_alive": True,
                    "is_hittable": True,
                    "status": [{"id": "RAVENOUS_POWER", "amount": 4}],
                    "intents": [{"type": "attack", "damage": 6, "total_damage": 6, "repeats": 1}],
                }
            ],
        },
        "legal_actions": [{"action": "end_turn"}],
    }

    canonical = canonicalize_state(state)

    assert canonical["payload"]["enemies"][0]["powers"] == [{"id": "RAVENOUS_POWER", "amount": 4}]
