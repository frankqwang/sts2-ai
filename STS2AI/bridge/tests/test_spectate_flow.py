from __future__ import annotations

import sys
from pathlib import Path

_python_root = Path(__file__).resolve().parents[1]
if str(_python_root) not in sys.path:
    sys.path.insert(0, str(_python_root))

from game_bridge.spectate.controller import SpectatorController
from game_bridge.spectate.zero_external_policy import ZeroExternalPolicyAdapter


class _DummySession:
    def __init__(self) -> None:
        self.reset_kwargs: dict[str, object] | None = None
        self._state = {"terminal": True, "run_outcome": "victory", "legal_actions": []}

    def reset(self, **kwargs):
        self.reset_kwargs = dict(kwargs)
        return dict(self._state)

    def get_state(self):
        return dict(self._state)


class _DummyPolicy:
    def select_action(self, state, legal, context):
        raise AssertionError("terminal 初始态不应该走到策略选择")


class _DummyZeroPolicy:
    def __init__(self) -> None:
        self.selected_state = None
        self.reset_calls = 0

    def reset_episode(self) -> None:
        self.reset_calls += 1

    def observe_transition(self, *_args, **_kwargs) -> None:
        return None

    def select_action(self, state) -> int:
        self.selected_state = state
        return 1


def test_spectator_controller_passes_encounter_id_to_reset():
    session = _DummySession()
    controller = SpectatorController(session=session, policy=_DummyPolicy())

    result = controller.play_episode(
        character_id="IRONCLAD",
        encounter_id="CHOMPERS_NORMAL",
        seed="TESTSEED",
        build={"deck": [{"id": "STRIKE_IRONCLAD"}]},
        max_steps=4,
    )

    assert result["run_outcome"] == "victory"
    assert session.reset_kwargs == {
        "character_id": "IRONCLAD",
        "encounter_id": "CHOMPERS_NORMAL",
        "seed": "TESTSEED",
        "build": {"deck": [{"id": "STRIKE_IRONCLAD"}]},
    }


def test_zero_external_policy_treats_visible_monster_state_as_actionable_combat():
    adapter = ZeroExternalPolicyAdapter.__new__(ZeroExternalPolicyAdapter)
    adapter._policy = _DummyZeroPolicy()
    adapter._previous_state = None
    adapter._previous_action_index = None
    adapter._convert_enabled_combat_state = lambda raw_state, enabled_legal: {
        "raw_state": raw_state,
        "enabled_legal": enabled_legal,
    }

    state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "player": {
                "hand": [
                    {"id": "STRIKE_IRONCLAD", "can_play": True},
                ]
            },
        },
    }
    legal_actions = [
        {"action": "play_card", "card_index": 0, "is_enabled": True},
        {"action": "end_turn", "is_enabled": True},
    ]

    chosen = adapter.select_action(state, legal_actions, None)

    assert chosen == {"action": "end_turn", "is_enabled": True}
    assert adapter._policy.selected_state == {
        "raw_state": state,
        "enabled_legal": legal_actions,
    }
    assert adapter._previous_action_index == 1
