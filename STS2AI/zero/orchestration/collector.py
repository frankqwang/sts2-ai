from __future__ import annotations

import math
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable

from ..domain import FightLabel, HistoryStep, RawTransition, TeacherRequest, TrainingSample, TransitionDelta, assess_transition_progress, compact_raw_transition
from ..ports import BattleRuntime, Policy, SearchTeacher
from .teacher import TeacherQueueBuilder


@dataclass(slots=True)
class SearchDecision:
    action_index: int
    priority: float
    reason_tags: list[str]
    search_policy: list[float]
    teacher_topk: list[int]
    teacher_best_action_index: int
    teacher_ranking_margin: float
    teacher_value: float
    teacher_search_trace: list[dict[str, float | int | str | bool]]
    source: str = "search"
    search_duration_s: float = 0.0
    search_simulations: int = 0
    search_cache_hit: bool = False


class SearchGuidedActionSelector:
    """在 collect 时按优先级让 search teacher 接管动作选择。"""

    def __init__(
        self,
        *,
        search_teacher: SearchTeacher,
        queue_builder: TeacherQueueBuilder,
        policy: Policy | None = None,
        priority_threshold: float,
        max_guided_steps_per_episode: int,
        target_encounters: tuple[str, ...] = (),
    ):
        self._search_teacher = search_teacher
        self._queue_builder = queue_builder
        self._policy = policy
        self._priority_threshold = float(priority_threshold)
        self._max_guided_steps_per_episode = max(0, int(max_guided_steps_per_episode))
        self._target_encounters = {value.strip().upper() for value in target_encounters if value}
        self._guided_steps = 0

    def reset_episode(self) -> None:
        self._guided_steps = 0

    def maybe_select_action(
        self,
        *,
        run_id: str,
        fight_id: str,
        step_idx: int,
        state,
        history: list[HistoryStep],
        prefix_action_indices: list[int],
        student_scores: list[float],
        student_uncertainty: float,
        student_fight_win_prob: float,
        student_enemy_hp_fraction_dealt: float,
        student_self_hp_fraction_remaining: float,
        greedy_action: int,
    ) -> SearchDecision | None:
        if not state.legal_actions or self._guided_steps >= self._max_guided_steps_per_episode:
            return None
        sample = _build_guidance_sample(
            run_id=run_id,
            fight_id=fight_id,
            step_idx=step_idx,
            state=state,
            history=history,
            prefix_action_indices=prefix_action_indices,
            student_uncertainty=student_uncertainty,
            student_scores=student_scores,
            student_fight_win_prob=student_fight_win_prob,
            student_enemy_hp_fraction_dealt=student_enemy_hp_fraction_dealt,
            student_self_hp_fraction_remaining=student_self_hp_fraction_remaining,
            greedy_action=greedy_action,
        )
        priority, reason_tags = self._queue_builder.score_sample(sample)
        encounter_id = str(state.context.encounter_id or "").upper()
        force_guidance = bool(encounter_id and encounter_id in self._target_encounters)
        if not force_guidance and priority < self._priority_threshold:
            return None
        label = _label_with_optional_policy(
            self._search_teacher,
            TeacherRequest(
                request_id=sample.sample_id,
                sample=sample,
                priority=max(priority, self._priority_threshold),
                reason_tags=(["search_guided_target"] if force_guidance else []) + list(reason_tags),
            ),
            seed=str(state.context.metadata.get("seed", "")),
            policy=self._policy,
        )
        action_index = int(label.best_action_index)
        if action_index < 0 or action_index >= len(state.legal_actions):
            return None
        self._guided_steps += 1
        return SearchDecision(
            action_index=action_index,
            priority=max(priority, self._priority_threshold),
            reason_tags=(["search_guided_target"] if force_guidance else []) + list(reason_tags),
            search_policy=list(label.policy),
            teacher_topk=list(label.topk_indices),
            teacher_best_action_index=int(label.best_action_index),
            teacher_ranking_margin=float(label.ranking_margin),
            teacher_value=float(label.teacher_value),
            teacher_search_trace=list(label.search_trace),
            search_duration_s=float(label.metadata.get("search_duration_s", 0.0) or 0.0),
            search_simulations=int(label.metadata.get("search_simulations", 0) or 0),
            search_cache_hit=bool(label.metadata.get("search_cache_hit", False)),
            source="search_guided",
        )


class SearchSelfPlaySelector:
    """每个决策点都调用搜索，并直接按搜索分布出动作。"""

    def __init__(self, *, search_teacher: SearchTeacher, policy: Policy | None = None):
        self._search_teacher = search_teacher
        self._policy = policy

    def select_action(
        self,
        *,
        run_id: str,
        fight_id: str,
        step_idx: int,
        state,
        history: list[HistoryStep],
        prefix_action_indices: list[int],
        student_scores: list[float],
        student_uncertainty: float,
        student_fight_win_prob: float,
        student_enemy_hp_fraction_dealt: float,
        student_self_hp_fraction_remaining: float,
        greedy_action: int,
        temperature: float,
        epsilon_greedy: float,
        rng: random.Random,
    ) -> SearchDecision | None:
        if not state.legal_actions:
            return None
        sample = _build_guidance_sample(
            run_id=run_id,
            fight_id=fight_id,
            step_idx=step_idx,
            state=state,
            history=history,
            prefix_action_indices=prefix_action_indices,
            student_uncertainty=student_uncertainty,
            student_scores=student_scores,
            student_fight_win_prob=student_fight_win_prob,
            student_enemy_hp_fraction_dealt=student_enemy_hp_fraction_dealt,
            student_self_hp_fraction_remaining=student_self_hp_fraction_remaining,
            greedy_action=greedy_action,
        )
        label = _label_with_optional_policy(
            self._search_teacher,
            TeacherRequest(
                request_id=sample.sample_id,
                sample=sample,
                priority=1.0,
                reason_tags=["search_self_play"],
            ),
            seed=str(state.context.metadata.get("seed", "")),
            policy=self._policy,
        )
        if not label.policy:
            return None
        action_index = _sample_policy_action(
            policy=label.policy,
            best_action=label.best_action_index,
            epsilon_greedy=epsilon_greedy,
            temperature=temperature,
            rng=rng,
        )
        if action_index < 0 or action_index >= len(state.legal_actions):
            return None
        return SearchDecision(
            action_index=action_index,
            priority=1.0,
            reason_tags=["search_self_play"],
            search_policy=list(label.policy),
            teacher_topk=list(label.topk_indices),
            teacher_best_action_index=int(label.best_action_index),
            teacher_ranking_margin=float(label.ranking_margin),
            teacher_value=float(label.teacher_value),
            teacher_search_trace=list(label.search_trace),
            search_duration_s=float(label.metadata.get("search_duration_s", 0.0) or 0.0),
            search_simulations=int(label.metadata.get("search_simulations", 0) or 0),
            search_cache_hit=bool(label.metadata.get("search_cache_hit", False)),
            source="search_self_play",
        )


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
        search_guidance_factory: Callable[[int | None], SearchGuidedActionSelector | None] | None = None,
        search_self_play_factory: Callable[[int | None], SearchSelfPlaySelector | None] | None = None,
        on_episode_start: Callable[[dict[str, object]], None] | None = None,
        on_transition: Callable[[RawTransition], None] | None = None,
        on_episode_end: Callable[[dict[str, object]], None] | None = None,
    ) -> list[RawTransition]:
        transitions: list[RawTransition] = []
        rng = random.Random(seed)
        search_guidance = search_guidance_factory(None) if search_guidance_factory is not None else None
        search_self_play = search_self_play_factory(None) if search_self_play_factory is not None else None
        for episode_index in range(episodes):
            run_id = uuid.uuid4().hex
            fight_id = uuid.uuid4().hex
            runtime = runtime_factory()
            if search_guidance is not None:
                search_guidance.reset_episode()
            history_window: deque[HistoryStep] = deque()
            prefix_action_indices: list[int] = []
            episode_started_at = time.perf_counter()
            reset_duration_s = 0.0
            policy_infer_duration_s = 0.0
            env_step_duration_s = 0.0
            observe_duration_s = 0.0
            emit_duration_s = 0.0
            search_duration_s = 0.0
            search_simulations = 0
            search_cache_hits = 0
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
                reset_started_at = time.perf_counter()
                state = runtime.reset()
                reset_duration_s = time.perf_counter() - reset_started_at
                last_state = state
                reset_hook = getattr(policy, "reset_episode", None)
                if callable(reset_hook):
                    reset_hook()
                for step_idx in range(max_steps):
                    if state.terminal or not state.legal_actions:
                        break
                    infer_started_at = time.perf_counter()
                    inference = self._infer_policy(policy, state)
                    policy_infer_duration_s += time.perf_counter() - infer_started_at
                    scores = inference["scores"]
                    greedy_action = inference["action_index"]
                    uncertainty = inference["uncertainty"]
                    search_decision = None
                    if search_self_play is not None:
                        search_decision = search_self_play.select_action(
                            run_id=run_id,
                            fight_id=fight_id,
                            step_idx=step_idx,
                            state=state,
                            history=list(history_window),
                            prefix_action_indices=prefix_action_indices,
                            student_scores=scores,
                            student_uncertainty=uncertainty,
                            student_fight_win_prob=float(inference.get("fight_win_prob", 0.0) or 0.0),
                            student_enemy_hp_fraction_dealt=float(inference.get("enemy_hp_fraction_dealt", 0.0) or 0.0),
                            student_self_hp_fraction_remaining=float(inference.get("self_hp_fraction_remaining", 0.0) or 0.0),
                            greedy_action=greedy_action,
                            temperature=temperature,
                            epsilon_greedy=epsilon_greedy,
                            rng=rng,
                        )
                    elif search_guidance is not None:
                        search_decision = search_guidance.maybe_select_action(
                            run_id=run_id,
                            fight_id=fight_id,
                            step_idx=step_idx,
                            state=state,
                            history=list(history_window),
                            prefix_action_indices=prefix_action_indices,
                            student_scores=scores,
                            student_uncertainty=uncertainty,
                            student_fight_win_prob=float(inference.get("fight_win_prob", 0.0) or 0.0),
                            student_enemy_hp_fraction_dealt=float(inference.get("enemy_hp_fraction_dealt", 0.0) or 0.0),
                            student_self_hp_fraction_remaining=float(inference.get("self_hp_fraction_remaining", 0.0) or 0.0),
                            greedy_action=greedy_action,
                        )
                    if search_decision is not None:
                        search_duration_s += float(search_decision.search_duration_s)
                        search_simulations += int(search_decision.search_simulations)
                        search_cache_hits += int(search_decision.search_cache_hit)
                    if search_self_play is not None:
                        action_index = search_decision.action_index if search_decision is not None else greedy_action
                    else:
                        sampled_action_index = _sample_action(
                            scores=scores,
                            greedy_action=greedy_action,
                            epsilon_greedy=epsilon_greedy,
                            temperature=temperature,
                            rng=rng,
                        )
                        action_index = search_decision.action_index if search_decision is not None else sampled_action_index
                    raw_actions = state.raw.get("legal_actions") if isinstance(state.raw, dict) else []
                    raw_actions = raw_actions if isinstance(raw_actions, list) else []
                    if search_decision is not None and action_index >= len(raw_actions):
                        search_decision = None
                        action_index = greedy_action if search_self_play is not None else sampled_action_index
                    env_step_started_at = time.perf_counter()
                    next_state = runtime.step(action_index)
                    env_step_duration_s += time.perf_counter() - env_step_started_at
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
                            "search_guided": bool(search_decision is not None and search_decision.source == "search_guided"),
                            "search_collected": bool(search_decision is not None and search_decision.source == "search_self_play"),
                            "search_guidance_priority": float(search_decision.priority) if search_decision else 0.0,
                            "search_guidance_tags": "|".join(search_decision.reason_tags) if search_decision else "",
                            "search_policy": list(search_decision.search_policy) if search_decision else [],
                            "search_teacher_topk": list(search_decision.teacher_topk) if search_decision else [],
                            "search_teacher_best_action_index": int(search_decision.teacher_best_action_index) if search_decision else -1,
                            "search_teacher_ranking_margin": float(search_decision.teacher_ranking_margin) if search_decision else 0.0,
                            "search_teacher_value": float(search_decision.teacher_value) if search_decision else 0.0,
                            "search_teacher_trace": list(search_decision.teacher_search_trace) if search_decision else [],
                            "search_source": str(search_decision.source) if search_decision else "",
                            "search_duration_s": float(search_decision.search_duration_s) if search_decision else 0.0,
                            "search_simulations": int(search_decision.search_simulations) if search_decision else 0,
                            "search_cache_hit": bool(search_decision.search_cache_hit) if search_decision else False,
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
                        + search_duration_s
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
                            "env_step_duration_s": round(env_step_duration_s, 6),
                            "observe_duration_s": round(observe_duration_s, 6),
                            "emit_duration_s": round(emit_duration_s, 6),
                            "search_duration_s": round(search_duration_s, 6),
                            "search_simulations": int(search_simulations),
                            "search_cache_hits": int(search_cache_hits),
                            "overhead_duration_s": round(overhead_duration_s, 6),
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
                "fight_win_prob": float(result.get("fight_win_prob", 0.0) or 0.0),
                "enemy_hp_fraction_dealt": float(result.get("enemy_hp_fraction_dealt", 0.0) or 0.0),
                "self_hp_fraction_remaining": float(result.get("self_hp_fraction_remaining", 0.0) or 0.0),
            }
        scores = policy.score_actions(state)
        return {
            "scores": scores,
            "action_index": int(policy.select_action(state)),
            "uncertainty": float(policy.estimate_uncertainty(state)),
            "fight_win_prob": 0.0,
            "enemy_hp_fraction_dealt": 0.0,
            "self_hp_fraction_remaining": 0.0,
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


def _sample_policy_action(
    *,
    policy: list[float],
    best_action: int,
    epsilon_greedy: float,
    temperature: float,
    rng: random.Random,
) -> int:
    if not policy:
        return max(0, best_action)
    best_action = min(max(best_action, 0), len(policy) - 1)
    if epsilon_greedy > 0.0 and rng.random() < epsilon_greedy:
        return rng.randrange(len(policy))
    if temperature <= 0.0 or len(policy) == 1:
        return best_action
    logits = [max(1e-8, float(value)) for value in policy]
    adjusted = [pow(value, 1.0 / max(temperature, 1e-6)) for value in logits]
    total = sum(adjusted)
    if total <= 0.0:
        return best_action
    return rng.choices(range(len(policy)), weights=adjusted, k=1)[0]


def _label_with_optional_policy(search_teacher: SearchTeacher, request: TeacherRequest, *, seed: str, policy: Policy | None):
    if policy is None:
        return search_teacher.label_request(request, seed=seed)
    try:
        return search_teacher.label_request(request, seed=seed, policy=policy)
    except TypeError:
        return search_teacher.label_request(request, seed=seed)


def _build_guidance_sample(
    *,
    run_id: str,
    fight_id: str,
    step_idx: int,
    state,
    history: list[HistoryStep],
    prefix_action_indices: list[int],
    student_uncertainty: float,
    student_scores: list[float],
    student_fight_win_prob: float,
    student_enemy_hp_fraction_dealt: float,
    student_self_hp_fraction_remaining: float,
    greedy_action: int,
) -> TrainingSample:
    return TrainingSample(
        sample_id=f"guided:{fight_id}:{step_idx}",
        run_id=run_id,
        fight_id=fight_id,
        step_idx=step_idx,
        state=state,
        history=history,
        legal_actions=state.legal_actions,
        behavior_action_index=max(0, min(greedy_action, max(len(state.legal_actions) - 1, 0))),
        behavior_action_id=state.legal_actions[max(0, min(greedy_action, max(len(state.legal_actions) - 1, 0)))].action_id if state.legal_actions else "",
        delta=TransitionDelta(),
        fight_label=FightLabel(
            fight_win=0.0,
            enemy_hp_fraction_dealt=0.0,
            self_hp_fraction_remaining=1.0,
            player_hp=float(state.player.hp),
            player_max_hp=float(max(state.player.max_hp, 1.0)),
        ),
        bucket_key=f"guided|{state.context.encounter_class}|floor{state.context.floor}",
        pool_name="recent_online",
        main_card_id=state.legal_actions[max(0, min(greedy_action, max(len(state.legal_actions) - 1, 0)))].card_id if state.legal_actions else "",
        risk_band="normal",
        step_progress_score=0.0,
        fight_score=float(state.context.metadata.get("fight_score_hint", 0.0) or 0.0),
        episode_score_proxy=0.0,
        sample_weight=1.0,
        keep_score=0.0,
        metadata={
            "prefix_action_indices": list(prefix_action_indices),
            "uncertainty": float(student_uncertainty),
            "top2_gap": _top2_gap(student_scores),
            "student_policy_scores": list(student_scores),
            "student_fight_win_prob": float(student_fight_win_prob),
            "student_enemy_hp_fraction_dealt": float(student_enemy_hp_fraction_dealt),
            "student_self_hp_fraction_remaining": float(student_self_hp_fraction_remaining),
            "fight_timeout": False,
            "fight_no_progress_ratio": 0.0,
            "hp_quality_score": 1.0,
            "fight_score": 0.0,
        },
    )
