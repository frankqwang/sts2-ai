"""策略无关观战控制器。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from game_bridge.spectate.overlay import OverlayWriter
from game_bridge.spectate.policy import PolicyAdapter
from game_bridge.types import PolicyContext, StateView


@dataclass
class SpectatorController:
    session: Any
    policy: PolicyAdapter
    overlay: OverlayWriter | None = None
    step_delay: float = 0.0
    idle_poll_interval_s: float = 0.25
    max_idle_polls: int = 40

    def play_episode(
        self,
        *,
        character_id: str = "IRONCLAD",
        seed: str | None = None,
        build: dict[str, Any] | None = None,
        max_steps: int = 800,
    ) -> dict[str, Any]:
        state = self.session.reset(character_id=character_id, seed=seed, build=build)
        idle_polls = 0
        steps_taken = 0
        for step_index in range(max_steps):
            legal = [
                action
                for action in (state.get("legal_actions") or [])
                if isinstance(action, dict) and action.get("is_enabled") is not False
            ]
            if not legal:
                state_view = StateView(state)
                if state_view.terminal or state_view.run_outcome:
                    break
                idle_polls += 1
                if idle_polls > self.max_idle_polls:
                    break
                time.sleep(self.idle_poll_interval_s)
                state = self.session.get_state()
                continue
            idle_polls = 0
            context = PolicyContext(step_index=step_index, character_id=character_id, seed=seed)
            action = self.policy.select_action(state, legal, context)
            if self.overlay is not None:
                self.overlay.publish({
                    "step_index": step_index,
                    "state_type": state.get("state_type"),
                    "legal_actions": legal,
                    "chosen_action": action,
                })
            if action is None:
                return {
                    "steps": steps_taken,
                    "stopped": True,
                    "state_type": state.get("state_type"),
                    "run_outcome": state.get("run_outcome"),
                }
            state = self.session.act(action)
            steps_taken = step_index + 1
            if self.step_delay > 0:
                time.sleep(self.step_delay)
        return {
            "steps": steps_taken,
            "stopped": False,
            "state_type": state.get("state_type"),
            "run_outcome": state.get("run_outcome"),
            "terminal": bool(state.get("terminal", False)),
        }
