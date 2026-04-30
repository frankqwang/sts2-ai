from __future__ import annotations

import sys
from pathlib import Path
_python_root = Path(__file__).resolve().parents[1]
if str(_python_root) not in sys.path:
    sys.path.insert(0, str(_python_root))

from game_bridge.spectate.controller import SpectatorController


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


