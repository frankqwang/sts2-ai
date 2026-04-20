from __future__ import annotations

import math
import random
import time
import uuid
from typing import Callable

from ..domain import RawTransition, assess_transition_progress
from ..ports import BattleRuntime, Policy


class TrajectoryCollector:
    """Roll out a policy and emit lossless-enough combat transitions.

    Collector now supports:
    - single-pass policy inference when the policy exposes `infer(state)`
    - rollout-time exploration (`temperature` / `epsilon_greedy`)
    - deterministic RNG control via `seed`
    """

    def collect(
        self,
        *,
        runtime_factory: Callable[[], BattleRuntime],
        policy: Policy,
        episodes: int,
        max_steps: int = 200,
        epsilon_greedy: float = 0.0,
        temperature: float = 0.0,
        seed: int | None = None,
        on_episode_start: Callable[[dict[str, object]], None] | None = None,
        on_transition: Callable[[RawTransition], None] | None = None,
        on_episode_end: Callable[[dict[str, object]], None] | None = None,
    ) -> list[RawTransition]:
        transitions: list[RawTransition] = []
        rng = random.Random(seed)
        for episode_index in range(episodes):
            run_id = uuid.uuid4().hex
            fight_id = uuid.uuid4().hex
            runtime = runtime_factory()
            episode_started_at = time.perf_counter()
            emitted_steps = 0
            last_state = None
            progress_steps = 0
            no_progress_steps = 0
            max_no_progress_streak = 0
            current_no_progress_streak = 0
            if on_episode_start is not None:
                on_episode_start(
                    {
                        "episode_index": episode_index,
                        "run_id": run_id,
                        "fight_id": fight_id,
                    }
                )
            try:
                state = runtime.reset()
                last_state = state
                reset_hook = getattr(policy, "reset_episode", None)
                if callable(reset_hook):
                    reset_hook()
                for step_idx in range(max_steps):
                    if state.terminal or not state.legal_actions:
                        break
                    inference = self._infer_policy(policy, state)
                    scores = inference["scores"]
                    greedy_action = inference["action_index"]
                    uncertainty = inference["uncertainty"]
                    action_index = _sample_action(
                        scores=scores,
                        greedy_action=greedy_action,
                        epsilon_greedy=epsilon_greedy,
                        temperature=temperature,
                        rng=rng,
                    )
                    next_state = runtime.step(action_index)
                    action = state.legal_actions[action_index]
                    progress = assess_transition_progress(state, next_state)
                    if progress.made_progress:
                        progress_steps += 1
                        current_no_progress_streak = 0
                    else:
                        no_progress_steps += 1
                        current_no_progress_streak += 1
                        max_no_progress_streak = max(max_no_progress_streak, current_no_progress_streak)
                    transition = RawTransition(
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
                            "uncertainty": float(uncertainty),
                            "top2_gap": _top2_gap(scores),
                            "made_progress": bool(progress.made_progress),
                            "enemy_hp_delta": float(progress.enemy_hp_delta),
                            "enemy_count_delta": int(progress.enemy_count_delta),
                        },
                    )
                    transitions.append(transition)
                    if on_transition is not None:
                        on_transition(transition)
                    emitted_steps = step_idx + 1
                    observe_hook = getattr(policy, "observe_transition", None)
                    if callable(observe_hook):
                        observe_hook(state, action_index, next_state)
                    state = next_state
                    last_state = state
                    if state.terminal:
                        break
            finally:
                runtime.close()
                if on_episode_end is not None:
                    truncated = bool(last_state is not None and not last_state.terminal and emitted_steps >= max_steps)
                    on_episode_end(
                        {
                            "episode_index": episode_index,
                            "run_id": run_id,
                            "fight_id": fight_id,
                            "duration_s": round(time.perf_counter() - episode_started_at, 6),
                            "steps": emitted_steps,
                            "terminal": bool(last_state.terminal) if last_state is not None else False,
                            "truncated": truncated,
                            "outcome": "timeout" if truncated else (str(last_state.run_outcome) if last_state is not None else ""),
                            "encounter_id": str(last_state.context.encounter_id) if last_state is not None else "",
                            "progress_steps": progress_steps,
                            "no_progress_steps": no_progress_steps,
                            "no_progress_ratio": round(no_progress_steps / max(emitted_steps, 1), 6),
                            "max_no_progress_streak": max_no_progress_streak,
                        }
                    )
        return transitions

    def _infer_policy(self, policy: Policy, state) -> dict[str, float | int | list[float]]:
        infer_hook = getattr(policy, "infer", None)
        if callable(infer_hook):
            result = infer_hook(state)
            return {
                "scores": list(result.get("scores", [])),
                "action_index": int(result.get("action_index", 0) or 0),
                "uncertainty": float(result.get("uncertainty", 0.0) or 0.0),
            }
        scores = policy.score_actions(state)
        return {
            "scores": scores,
            "action_index": int(policy.select_action(state)),
            "uncertainty": float(policy.estimate_uncertainty(state)),
        }


def _top2_gap(scores: list[float]) -> float:
    if len(scores) < 2:
        return 1.0
    ordered = sorted(scores, reverse=True)
    return float(ordered[0] - ordered[1])


def _sample_action(
    *,
    scores: list[float],
    greedy_action: int,
    epsilon_greedy: float,
    temperature: float,
    rng: random.Random,
) -> int:
    if not scores:
        return 0
    greedy_action = min(max(greedy_action, 0), len(scores) - 1)
    if epsilon_greedy > 0.0 and rng.random() < epsilon_greedy:
        return rng.randrange(len(scores))
    if temperature <= 0.0 or len(scores) == 1:
        return greedy_action
    max_score = max(scores)
    logits = [(score - max_score) / max(temperature, 1e-6) for score in scores]
    weights = [math.exp(min(50.0, value)) for value in logits]
    if sum(weights) <= 0.0:
        return greedy_action
    return rng.choices(range(len(scores)), weights=weights, k=1)[0]
