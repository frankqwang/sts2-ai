from __future__ import annotations

import uuid
from typing import Callable

from ..domain import RawTransition
from ..ports import BattleRuntime, Policy


class TrajectoryCollector:
    """Roll out a policy and emit lossless-enough combat transitions.

    We intentionally record both:
    - `action_index`: the categorical choice relative to that state's legal set
    - `action`: the concrete LegalAction snapshot with action/card/target data

    This keeps behavior-cloning labels simple while preserving the chosen
    action's explicit semantics for later feature building and debugging.
    """

    def collect(
        self,
        *,
        runtime_factory: Callable[[], BattleRuntime],
        policy: Policy,
        episodes: int,
        max_steps: int = 200,
    ) -> list[RawTransition]:
        transitions: list[RawTransition] = []
        for _ in range(episodes):
            run_id = uuid.uuid4().hex
            fight_id = uuid.uuid4().hex
            runtime = runtime_factory()
            try:
                state = runtime.reset()
                reset_hook = getattr(policy, "reset_episode", None)
                if callable(reset_hook):
                    reset_hook()
                for step_idx in range(max_steps):
                    if state.terminal or not state.legal_actions:
                        break
                    scores = policy.score_actions(state)
                    action_index = policy.select_action(state)
                    next_state = runtime.step(action_index)
                    action = state.legal_actions[action_index]
                    transitions.append(
                        RawTransition(
                            run_id=run_id,
                            fight_id=fight_id,
                            step_idx=step_idx,
                            seed=str(state.context.metadata.get("seed", "")),
                            action_index=action_index,
                            state=state,
                            action=action,
                            next_state=next_state,
                            done=next_state.terminal,
                            fight_outcome=next_state.run_outcome,
                            run_outcome=next_state.run_outcome,
                            metadata={
                                "action_id": action.action_id,
                                "uncertainty": float(policy.estimate_uncertainty(state)),
                                "top2_gap": _top2_gap(scores),
                            },
                        )
                    )
                    observe_hook = getattr(policy, "observe_transition", None)
                    if callable(observe_hook):
                        observe_hook(state, action_index, next_state)
                    state = next_state
                    if state.terminal:
                        break
            finally:
                runtime.close()
        return transitions


def _top2_gap(scores: list[float]) -> float:
    if len(scores) < 2:
        return 1.0
    ordered = sorted(scores, reverse=True)
    return float(ordered[0] - ordered[1])
