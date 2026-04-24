"""策略无关观战控制器。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from game_bridge.session.singleplayer_api import SingleplayerApiError
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
    # sim 在 state 流转边界有时会把一个 action 当成 legal 给出，但实际执行时拒绝
    # （比如地图节点切换瞬间 proceed 短暂不可用）。这类错误不应当让整场 run 崩。
    max_recoverable_step_errors: int = 5

    def play_episode(
        self,
        *,
        character_id: str = "IRONCLAD",
        encounter_id: str | None = None,
        seed: str | None = None,
        build: dict[str, Any] | None = None,
        floor: int | None = None,
        max_steps: int = 800,
    ) -> dict[str, Any]:
        reset_kwargs: dict[str, Any] = {
            "character_id": character_id,
            "encounter_id": encounter_id,
            "seed": seed,
            "build": build,
        }
        if floor is not None:
            reset_kwargs["floor"] = floor
        state = self.session.reset(**reset_kwargs)
        idle_polls = 0
        steps_taken = 0
        recoverable_errors_seen = 0
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
            try:
                state = self.session.act(action)
            except SingleplayerApiError as exc:
                # sim 拒绝了一个它之前标 legal 的动作，通常是状态边界竞态。
                # 记录后刷新 state 继续，不拖垮整个 episode。
                print(f"[spectate] step {step_index} act rejected: {exc}. Refreshing state.", flush=True)
                recoverable_errors_seen += 1
                if recoverable_errors_seen > self.max_recoverable_step_errors:
                    return {
                        "steps": steps_taken,
                        "stopped": True,
                        "state_type": state.get("state_type"),
                        "run_outcome": state.get("run_outcome"),
                        "abort_reason": "too_many_step_rejections",
                    }
                try:
                    state = self.session.get_state()
                except Exception:
                    # 刷新也挂，放弃
                    return {
                        "steps": steps_taken,
                        "stopped": True,
                        "state_type": state.get("state_type"),
                        "run_outcome": state.get("run_outcome"),
                        "abort_reason": "state_refresh_failed_after_reject",
                    }
                continue
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
