from __future__ import annotations

import logging
import math
import random
import time
import uuid
from collections import deque
from typing import Callable

from ..domain import HistoryStep, RawTransition, TransitionDelta, assess_transition_progress, compact_raw_transition
from ..ports import BattleRuntime, Policy


_logger = logging.getLogger(__name__)


class TrajectoryCollector:
    """Roll out a policy and emit lossless-enough combat transitions.

    Collector supports:
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
            history_window: deque[HistoryStep] = deque()
            prefix_action_indices: list[int] = []
            episode_started_at = time.perf_counter()
            reset_duration_s = 0.0
            policy_infer_duration_s = 0.0
            policy_collate_duration_s = 0.0
            policy_forward_duration_s = 0.0
            policy_postprocess_duration_s = 0.0
            intent_forward_duration_s = 0.0
            action_forward_duration_s = 0.0
            batch_wait_duration_s = 0.0
            batch_size_total = 0
            batch_size_samples = 0
            env_step_duration_s = 0.0
            runtime_reset_call_duration_s = 0.0
            runtime_reset_transport_duration_s = 0.0
            runtime_reset_transport_write_duration_s = 0.0
            runtime_reset_transport_read_duration_s = 0.0
            runtime_reset_transport_decode_duration_s = 0.0
            runtime_reset_state_convert_duration_s = 0.0
            runtime_step_call_duration_s = 0.0
            runtime_step_transport_duration_s = 0.0
            runtime_step_transport_write_duration_s = 0.0
            runtime_step_transport_read_duration_s = 0.0
            runtime_step_transport_decode_duration_s = 0.0
            runtime_step_state_convert_duration_s = 0.0
            observe_duration_s = 0.0
            emit_duration_s = 0.0
            emitted_steps = 0
            last_state = None
            progress_steps = 0
            no_progress_steps = 0
            max_no_progress_streak = 0
            current_no_progress_streak = 0
            turn_id_fallback_used_steps = 0
            last_turn_id: int | None = None
            turn_id_fallback_warned = False
            if on_episode_start is not None:
                on_episode_start(
                    {
                        "episode_index": episode_index,
                        "run_id": run_id,
                        "fight_id": fight_id,
                    }
                )
            try:
                reset_started_at = time.perf_counter()
                state = runtime.reset()
                reset_duration_s = time.perf_counter() - reset_started_at
                reset_timing = _runtime_timing(runtime, "get_last_reset_timing")
                runtime_reset_call_duration_s = float(reset_timing.get("session_call_duration_s", 0.0) or 0.0)
                runtime_reset_transport_duration_s = float(reset_timing.get("transport_duration_s", 0.0) or 0.0)
                runtime_reset_transport_write_duration_s = float(
                    reset_timing.get("transport_write_duration_s", 0.0) or 0.0
                )
                runtime_reset_transport_read_duration_s = float(
                    reset_timing.get("transport_read_duration_s", 0.0) or 0.0
                )
                runtime_reset_transport_decode_duration_s = float(
                    reset_timing.get("transport_decode_duration_s", 0.0) or 0.0
                )
                runtime_reset_state_convert_duration_s = float(
                    reset_timing.get("state_convert_duration_s", 0.0) or 0.0
                )
                last_state = state
                reset_hook = getattr(policy, "reset_episode", None)
                if callable(reset_hook):
                    reset_hook()
                for step_idx in range(max_steps):
                    if state.terminal or not state.legal_actions:
                        break
                    infer_started_at = time.perf_counter()
                    inference = self._infer_policy(policy, state)
                    turn_id_fallback_used = bool(inference.get("turn_id_fallback_used", False))
                    current_turn_id = int(inference.get("turn_id", 0) or 0)
                    if current_turn_id > 0:
                        if last_turn_id is not None and current_turn_id < last_turn_id:
                            raise ValueError(
                                f"turn_id 非单调: last_turn_id={last_turn_id} current_turn_id={current_turn_id} "
                                f"episode_index={episode_index} run_id={run_id} fight_id={fight_id}"
                            )
                        last_turn_id = current_turn_id
                    if turn_id_fallback_used:
                        turn_id_fallback_used_steps += 1
                        if not turn_id_fallback_warned:
                            _logger.warning(
                                "zero.collector turn_id fallback used episode_index=%s run_id=%s fight_id=%s",
                                episode_index,
                                run_id,
                                fight_id,
                            )
                            turn_id_fallback_warned = True
                    policy_infer_duration_s += time.perf_counter() - infer_started_at
                    policy_collate_duration_s += float(inference.get("policy_collate_duration_s", 0.0) or 0.0)
                    intent_forward_duration_s += float(inference.get("intent_forward_duration_s", 0.0) or 0.0)
                    action_forward_duration_s += float(inference.get("action_forward_duration_s", 0.0) or 0.0)
                    policy_forward_duration_s += float(inference.get("policy_forward_duration_s", 0.0) or 0.0)
                    policy_postprocess_duration_s += float(inference.get("policy_postprocess_duration_s", 0.0) or 0.0)
                    batch_wait_duration_s += float(inference.get("batch_wait_duration_s", 0.0) or 0.0)
                    batch_size_total += int(inference.get("batch_size_effective", 1) or 1)
                    batch_size_samples += 1
                    is_turn_start = bool(inference.get("is_turn_start", False))
                    chosen_intent = int(inference.get("active_intent", 0) or 0)
                    behavior_intent_logprob = 0.0
                    scores = [float(value) for value in (inference["scores"] or [])]
                    greedy_action = int(inference["action_index"])
                    intent_scores_raw = inference.get("intent_scores", [])
                    has_intent_scores = bool(intent_scores_raw)
                    if is_turn_start and has_intent_scores:
                        intent_scores = [float(value) for value in intent_scores_raw]
                        greedy_intent = chosen_intent
                        chosen_intent, behavior_intent_logprob = _sample_action(
                            scores=intent_scores,
                            greedy_action=greedy_intent,
                            epsilon_greedy=epsilon_greedy,
                            temperature=temperature,
                            rng=rng,
                        )
                        intent_hook = getattr(policy, "observe_intent_choice", None)
                        if callable(intent_hook):
                            intent_hook(state, chosen_intent)
                        action_scores_by_intent = inference.get("action_scores_by_intent", [])
                        action_indices_by_intent = inference.get("action_index_by_intent", [])
                        action_intent_count = len(action_scores_by_intent)
                        if 0 <= chosen_intent < action_intent_count:
                            chosen_scores = action_scores_by_intent[chosen_intent]
                            scores = [float(value) for value in chosen_scores]
                        action_index_count = len(action_indices_by_intent)
                        if 0 <= chosen_intent < action_index_count:
                            greedy_action = int(action_indices_by_intent[chosen_intent])
                    model_log_probs = _compute_model_log_probs(scores)
                    action_index, behavior_logprob = _sample_action(
                        scores=scores,
                        greedy_action=greedy_action,
                        epsilon_greedy=epsilon_greedy,
                        temperature=temperature,
                        rng=rng,
                    )
                    env_step_started_at = time.perf_counter()
                    next_state = runtime.step(action_index)
                    env_step_duration_s += time.perf_counter() - env_step_started_at
                    step_timing = _runtime_timing(runtime, "get_last_step_timing")
                    runtime_step_call_duration_s += float(step_timing.get("session_call_duration_s", 0.0) or 0.0)
                    runtime_step_transport_duration_s += float(step_timing.get("transport_duration_s", 0.0) or 0.0)
                    runtime_step_transport_write_duration_s += float(
                        step_timing.get("transport_write_duration_s", 0.0) or 0.0
                    )
                    runtime_step_transport_read_duration_s += float(
                        step_timing.get("transport_read_duration_s", 0.0) or 0.0
                    )
                    runtime_step_transport_decode_duration_s += float(
                        step_timing.get("transport_decode_duration_s", 0.0) or 0.0
                    )
                    runtime_step_state_convert_duration_s += float(
                        step_timing.get("state_convert_duration_s", 0.0) or 0.0
                    )
                    action = state.legal_actions[action_index]
                    progress = assess_transition_progress(state, next_state)
                    reward = _compute_reward(state, action, next_state)
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
                        reward=float(reward),
                        metadata={
                            "action_id": action.action_id,
                            # PPO 分母应对应“旧模型本身”的策略概率，而不是 rollout
                            # 采样器经 temperature / epsilon 扰动后的行为概率。
                            "old_logprob": float(model_log_probs[action_index]) if model_log_probs else 0.0,
                            "behavior_logprob": float(behavior_logprob),
                            "value_pred": float(inference.get("ppo_value", 0.0) or 0.0),
                            "old_intent_logprob": float(inference.get("old_intent_logprob", 0.0) or 0.0),
                            "behavior_intent_logprob": float(behavior_intent_logprob),
                            "old_intent_value": float(inference.get("intent_value", 0.0) or 0.0),
                            "active_intent": int(chosen_intent),
                            "turn_start_mask": 1 if is_turn_start else 0,
                            "turn_id": int(inference.get("turn_id", 0) or 0),
                            "turn_id_fallback_used": turn_id_fallback_used,
                            "reward": float(reward),
                            "top2_gap": _top2_gap(scores),
                            "made_progress": bool(progress.made_progress),
                            "enemy_hp_delta": float(progress.enemy_hp_delta),
                            "enemy_count_delta": int(progress.enemy_count_delta),
                        },
                    )
                    if on_transition is not None:
                        emit_started_at = time.perf_counter()
                        on_transition(transition)
                        emit_duration_s += time.perf_counter() - emit_started_at
                    transition = compact_raw_transition(transition)
                    transitions.append(transition)
                    emitted_steps = step_idx + 1
                    observe_hook = getattr(policy, "observe_transition", None)
                    if callable(observe_hook):
                        observe_started_at = time.perf_counter()
                        observe_hook(state, action_index, next_state)
                        observe_duration_s += time.perf_counter() - observe_started_at
                    history_window.append(
                        HistoryStep(
                            state=None,
                            action=None,
                            delta=TransitionDelta(),
                            history_token=[],
                        )
                    )
                    prefix_action_indices.append(action_index)
                    state = transition.next_state
                    last_state = state
                    if state.terminal:
                        break
            finally:
                runtime.close()
                if on_episode_end is not None:
                    duration_s = time.perf_counter() - episode_started_at
                    truncated = bool(last_state is not None and not last_state.terminal and emitted_steps >= max_steps)
                    accounted_duration_s = (
                        reset_duration_s
                        + policy_infer_duration_s
                        + env_step_duration_s
                        + observe_duration_s
                        + emit_duration_s
                    )
                    overhead_duration_s = max(0.0, duration_s - accounted_duration_s)
                    on_episode_end(
                        {
                            "episode_index": episode_index,
                            "run_id": run_id,
                            "fight_id": fight_id,
                            "duration_s": round(duration_s, 6),
                            "steps": emitted_steps,
                            "step_throughput": round(emitted_steps / max(duration_s, 1e-6), 6),
                            "core_step_throughput": round(
                                emitted_steps / max(policy_infer_duration_s + env_step_duration_s, 1e-6),
                                6,
                            ),
                            "reset_duration_s": round(reset_duration_s, 6),
                            "policy_infer_duration_s": round(policy_infer_duration_s, 6),
                            "policy_collate_duration_s": round(policy_collate_duration_s, 6),
                            "intent_forward_duration_s": round(intent_forward_duration_s, 6),
                            "action_forward_duration_s": round(action_forward_duration_s, 6),
                            "policy_forward_duration_s": round(policy_forward_duration_s, 6),
                            "policy_postprocess_duration_s": round(policy_postprocess_duration_s, 6),
                            "avg_batch_wait_duration_s": round(
                                batch_wait_duration_s / max(batch_size_samples, 1),
                                6,
                            ),
                            "avg_batch_size_effective": round(
                                batch_size_total / max(batch_size_samples, 1),
                                4,
                            ),
                            "env_step_duration_s": round(env_step_duration_s, 6),
                            "runtime_reset_call_duration_s": round(runtime_reset_call_duration_s, 6),
                            "runtime_reset_transport_duration_s": round(runtime_reset_transport_duration_s, 6),
                            "runtime_reset_transport_write_duration_s": round(runtime_reset_transport_write_duration_s, 6),
                            "runtime_reset_transport_read_duration_s": round(runtime_reset_transport_read_duration_s, 6),
                            "runtime_reset_transport_decode_duration_s": round(
                                runtime_reset_transport_decode_duration_s,
                                6,
                            ),
                            "runtime_reset_state_convert_duration_s": round(
                                runtime_reset_state_convert_duration_s,
                                6,
                            ),
                            "runtime_step_call_duration_s": round(runtime_step_call_duration_s, 6),
                            "runtime_step_transport_duration_s": round(runtime_step_transport_duration_s, 6),
                            "runtime_step_transport_write_duration_s": round(runtime_step_transport_write_duration_s, 6),
                            "runtime_step_transport_read_duration_s": round(runtime_step_transport_read_duration_s, 6),
                            "runtime_step_transport_decode_duration_s": round(
                                runtime_step_transport_decode_duration_s,
                                6,
                            ),
                            "runtime_step_state_convert_duration_s": round(
                                runtime_step_state_convert_duration_s,
                                6,
                            ),
                            "observe_duration_s": round(observe_duration_s, 6),
                            "emit_duration_s": round(emit_duration_s, 6),
                            "overhead_duration_s": round(overhead_duration_s, 6),
                            "terminal": bool(last_state.terminal) if last_state is not None else False,
                            "truncated": truncated,
                            "outcome": "timeout" if truncated else (str(last_state.run_outcome) if last_state is not None else ""),
                            "encounter_id": str(last_state.context.encounter_id) if last_state is not None else "",
                            "progress_steps": progress_steps,
                            "no_progress_steps": no_progress_steps,
                            "no_progress_ratio": round(no_progress_steps / max(emitted_steps, 1), 6),
                            "max_no_progress_streak": max_no_progress_streak,
                            "turn_id_fallback_used_steps": int(turn_id_fallback_used_steps),
                            "turn_id_fallback_used": bool(turn_id_fallback_used_steps > 0),
                            "max_turn_id": int(last_turn_id or 0),
                        }
                    )
        return transitions

    def _infer_policy(self, policy: Policy, state) -> dict[str, float | int | list[float]]:
        infer_hook = getattr(policy, "infer", None)
        if callable(infer_hook):
            result = infer_hook(state)
            return {
                "scores": result.get("scores", []),
                "action_index": int(result.get("action_index", 0) or 0),
                "intent_scores": result.get("intent_scores", []),
                "action_scores_by_intent": result.get("action_scores_by_intent", []),
                "action_index_by_intent": result.get("action_index_by_intent", []),
                "is_turn_start": bool(result.get("is_turn_start", False)),
                "turn_id": int(result.get("turn_id", 0) or 0),
                "turn_id_fallback_used": bool(result.get("turn_id_fallback_used", False)),
                "active_intent": int(result.get("active_intent", 0) or 0),
                "old_intent_logprob": float(result.get("old_intent_logprob", 0.0) or 0.0),
                "intent_value": float(result.get("intent_value", 0.0) or 0.0),
                "fight_win_prob": float(result.get("fight_win_prob", 0.0) or 0.0),
                "enemy_hp_fraction_dealt": float(result.get("enemy_hp_fraction_dealt", 0.0) or 0.0),
                "self_hp_fraction_remaining": float(result.get("self_hp_fraction_remaining", 0.0) or 0.0),
                "ppo_value": float(result.get("ppo_value", 0.0) or 0.0),
                "policy_collate_duration_s": float(result.get("policy_collate_duration_s", 0.0) or 0.0),
                "intent_forward_duration_s": float(result.get("intent_forward_duration_s", 0.0) or 0.0),
                "action_forward_duration_s": float(result.get("action_forward_duration_s", 0.0) or 0.0),
                "policy_forward_duration_s": float(result.get("policy_forward_duration_s", 0.0) or 0.0),
                "policy_postprocess_duration_s": float(result.get("policy_postprocess_duration_s", 0.0) or 0.0),
                "batch_wait_duration_s": float(result.get("batch_wait_duration_s", 0.0) or 0.0),
                "batch_size_effective": int(result.get("batch_size_effective", 1) or 1),
            }
        scores = policy.score_actions(state)
        return {
            "scores": scores,
            "action_index": int(policy.select_action(state)),
            "intent_scores": [],
            "action_scores_by_intent": [],
            "action_index_by_intent": [],
            "is_turn_start": False,
            "turn_id": 0,
            "turn_id_fallback_used": False,
            "active_intent": 0,
            "old_intent_logprob": 0.0,
            "intent_value": 0.0,
            "fight_win_prob": 0.0,
            "enemy_hp_fraction_dealt": 0.0,
            "self_hp_fraction_remaining": 0.0,
            "ppo_value": 0.0,
            "policy_collate_duration_s": 0.0,
            "intent_forward_duration_s": 0.0,
            "action_forward_duration_s": 0.0,
            "policy_forward_duration_s": 0.0,
            "policy_postprocess_duration_s": 0.0,
            "batch_wait_duration_s": 0.0,
            "batch_size_effective": 1,
        }


def _runtime_timing(runtime, method_name: str) -> dict[str, float]:
    hook = getattr(runtime, method_name, None)
    if callable(hook):
        payload = hook()
        if isinstance(payload, dict):
            return payload
    return {}


def _top2_gap(scores: list[float]) -> float:
    if len(scores) < 2:
        return 1.0
    ordered = sorted(scores, reverse=True)
    return float(ordered[0] - ordered[1])


def _compute_reward(state, action, next_state) -> float:
    current_enemy_hp = sum(float(enemy.hp) for enemy in state.enemies if enemy.alive)
    next_enemy_hp = sum(float(enemy.hp) for enemy in next_state.enemies if enemy.alive)
    current_enemy_max_hp = max(1.0, sum(float(enemy.max_hp) for enemy in state.enemies))
    combat_start_hp = max(
        1.0,
        float(state.context.metadata.get("combat_start_hp") or 0.0)
        or float(state.player.max_hp)
        or float(state.player.hp)
        or 1.0,
    )
    enemy_progress = max(0.0, current_enemy_hp - next_enemy_hp) / current_enemy_max_hp
    self_loss = max(0.0, float(state.player.hp) - float(next_state.player.hp)) / combat_start_hp
    engine_gain = max(0.0, _engine_buff_total(next_state.player.buffs) - _engine_buff_total(state.player.buffs))
    exhaust_delta = max(0.0, float(next_state.piles.exhaust_pile_size) - float(state.piles.exhaust_pile_size))
    engine_active = max(_engine_buff_total(state.player.buffs), _engine_buff_total(next_state.player.buffs))

    reward = 0.15 * enemy_progress - 0.45 * self_loss - 0.003
    reward += 0.06 * engine_gain
    if exhaust_delta > 0.0:
        if engine_active > 0.0:
            reward += 0.04 * exhaust_delta * min(engine_active, 2.0)
        else:
            reward -= 0.01 * exhaust_delta
    if getattr(action, "action_type", "") == "end_turn":
        if any(legal.can_execute and legal.action_type != "end_turn" for legal in state.legal_actions):
            reward -= 0.02 * max(0.0, float(state.player.energy))
        if enemy_progress <= 0.0 and self_loss <= 0.0:
            reward -= 0.03
    outcome = str(next_state.run_outcome or "").strip().lower()
    if next_state.terminal:
        target_hp_after = max(0.0, float(state.context.metadata.get("combat_target_hp_after") or 0.0))
        hp_target_gap_ratio = min(1.0, abs(max(0.0, float(next_state.player.hp)) - target_hp_after) / combat_start_hp)
        if outcome in {"victory", "win"}:
            reward += 2.0 + max(0.0, float(next_state.player.hp)) / combat_start_hp
            reward += 0.5 * (1.0 - hp_target_gap_ratio)
        else:
            reward -= 2.0
            reward -= 0.25 * hp_target_gap_ratio
    return float(reward)


def _engine_buff_total(buffs: dict[str, float]) -> float:
    return (
        float(buffs.get("FEEL_NO_PAIN_POWER", 0.0) or 0.0)
        + float(buffs.get("DARK_EMBRACE_POWER", 0.0) or 0.0)
        + float(buffs.get("PYRE_POWER", 0.0) or 0.0)
    )


def _sample_action(
    *,
    scores: list[float],
    greedy_action: int,
    epsilon_greedy: float,
    temperature: float,
    rng: random.Random,
) -> tuple[int, float]:
    if not scores:
        return 0, 0.0
    greedy_action = min(max(greedy_action, 0), len(scores) - 1)
    action_count = len(scores)
    if temperature <= 0.0 or action_count == 1:
        base_probs = [0.0 for _ in scores]
        base_probs[greedy_action] = 1.0
    else:
        max_score = max(scores)
        logits = [(score - max_score) / max(temperature, 1e-6) for score in scores]
        weights = [math.exp(min(50.0, value)) for value in logits]
        total = sum(weights)
        if total <= 0.0:
            base_probs = [0.0 for _ in scores]
            base_probs[greedy_action] = 1.0
        else:
            base_probs = [weight / total for weight in weights]

    epsilon = min(max(float(epsilon_greedy), 0.0), 1.0)
    uniform_prob = 1.0 / float(action_count)
    behavior_probs = [((1.0 - epsilon) * prob) + (epsilon * uniform_prob) for prob in base_probs]
    total = sum(behavior_probs)
    if total <= 0.0:
        return greedy_action, 0.0
    normalized = [prob / total for prob in behavior_probs]
    action_index = int(rng.choices(range(action_count), weights=normalized, k=1)[0])
    return action_index, float(math.log(max(normalized[action_index], 1e-8)))


def _compute_model_log_probs(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    shifted = [score - max_score for score in scores]
    weights = [math.exp(max(-50.0, min(50.0, value))) for value in shifted]
    total = sum(weights)
    if total <= 0.0:
        uniform_logprob = float(-math.log(len(scores)))
        return [uniform_logprob for _ in scores]
    probs = [weight / total for weight in weights]
    return [float(math.log(max(prob, 1e-8))) for prob in probs]
