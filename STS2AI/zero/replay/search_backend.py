from __future__ import annotations

"""基于 same-seed root MCTS 的 combat search backend。

V1 目标：
- 直接用 root MCTS 产出搜索分布，不再依赖弱搜索先验聚合
- 统一复用 `compute_fight_score(...)` / `compute_hp_quality_score(...)`
- 为后续 branching / 类 MCTS 预留预算和 trace 结构
"""

from collections import OrderedDict
from dataclasses import dataclass
import math
import time
from typing import Iterable, TypedDict

from ..config import SearchConfig
from ..domain import FightLabel, HistoryStep, SearchLabel, SearchRequest, TransitionDelta, assess_transition_progress, compute_fight_score, compute_hp_quality_score
from ..features import compute_transition_delta
from .skada import SkadaCombatCase, SkadaReplayRuntime


@dataclass(slots=True)
class SearchBudget:
    max_root_actions: int
    rollouts_per_action: int
    max_branch_steps: int
    allow_branching: bool
    trace_topk: int
    puct_exploration: float
    enable_snapshot_restore: bool
    leaf_eval_horizon: int
    leaf_value_weight: float
    root_cache_size: int


@dataclass(slots=True)
class CachedSearchResult:
    label: SearchLabel
    budget_signature: tuple[int, int, int, bool]


class RootActionStats(TypedDict):
    visits: int
    value_sum: float
    best_score: float
    hp_sum: float
    speed_sum: float
    outcomes: list[str]


class BranchResult(TypedDict):
    fight_quality_score: float
    hp_quality_score: float
    speed_quality: float
    step_count: int
    truncated: bool
    outcome: str


class CombatSearchBackend:
    """在固定 replay case 上做 same-seed root MCTS。"""

    def __init__(
        self,
        case: SkadaCombatCase,
        *,
        config: SearchConfig,
        port: int = 15527,
        auto_launch: bool = True,
        connect_timeout_s: float = 30.0,
    ):
        self._case = case
        self._config = config
        self._port = port
        self._auto_launch = auto_launch
        self._connect_timeout_s = connect_timeout_s
        self._budget = SearchBudget(
            max_root_actions=max(1, int(config.max_root_actions)),
            rollouts_per_action=max(1, int(config.rollouts_per_action)),
            max_branch_steps=max(1, int(config.max_branch_steps)),
            allow_branching=bool(config.allow_branching),
            trace_topk=max(1, int(config.trace_topk)),
            puct_exploration=1.25,
            enable_snapshot_restore=bool(config.enable_snapshot_restore),
            leaf_eval_horizon=max(0, int(config.leaf_eval_horizon)),
            leaf_value_weight=max(0.0, min(1.0, float(config.leaf_value_weight))),
            root_cache_size=max(0, int(config.root_cache_size)),
        )
        self._runtime: SkadaReplayRuntime | None = None
        self._cached_labels: OrderedDict[tuple[object, ...], CachedSearchResult] = OrderedDict()

    def label_request(self, request: SearchRequest, runtime_factory=None, seed: str | None = None, policy=None) -> SearchLabel:
        sample = request.sample
        if not sample.legal_actions:
            return SearchLabel(search_value=0.0, metadata={"search_backend": "CombatSearchBackend"})
        started_at = time.perf_counter()
        cache_key = self._build_cache_key(sample)
        cached = self._cached_labels.get(cache_key)
        if cached is not None and cached.budget_signature == self._budget_signature():
            self._cached_labels.move_to_end(cache_key)
            return self._with_search_metadata(
                cached.label,
                search_duration_s=time.perf_counter() - started_at,
                search_simulations=0,
                search_cache_hit=True,
                snapshot_used=False,
                prefix_replay_count=0,
            )

        runtime = self._get_runtime()
        trace_rows: list[dict[str, float | int | str | bool]] = []
        search_simulations = 0
        snapshot_used = False
        prefix_replay_count = 0
        root_state_id: str | None = None
        try:
            replay_ok, current_state = self._replay_prefix(runtime, sample)
            prefix_replay_count += 1
            if not replay_ok or not current_state.legal_actions:
                return self._fallback_label(sample, backend_name="CombatSearchBackendFallback")
            candidate_indices = self._candidate_root_indices(sample)
            priors = self._root_priors(sample, candidate_indices)
            total_simulations = max(1, self._budget.max_root_actions * self._budget.rollouts_per_action)
            if self._budget.enable_snapshot_restore:
                root_state_id = self._try_save_state(runtime)
                snapshot_used = root_state_id is not None
            stats, search_simulations, replay_replays = self._run_root_simulations(
                runtime=runtime,
                sample=sample,
                candidate_indices=candidate_indices,
                priors=priors,
                total_simulations=total_simulations,
                root_state_id=root_state_id,
                policy=policy,
            )
            prefix_replay_count += replay_replays
            if search_simulations <= 0 and root_state_id is not None:
                snapshot_used = False
                stats, search_simulations, replay_replays = self._run_root_simulations(
                    runtime=runtime,
                    sample=sample,
                    candidate_indices=candidate_indices,
                    priors=priors,
                    total_simulations=total_simulations,
                    root_state_id=None,
                    policy=policy,
                )
                prefix_replay_count += replay_replays
        finally:
            if root_state_id is not None:
                runtime.delete_state(root_state_id)

        if not candidate_indices:
            return self._fallback_label(sample, backend_name="CombatSearchBackendFallback")

        visit_policy = [0.0] * len(sample.legal_actions)
        total_visits = sum(int(stats[index]["visits"]) for index in candidate_indices)
        if total_visits > 0:
            for action_index in candidate_indices:
                visit_policy[action_index] = float(stats[action_index]["visits"]) / float(total_visits)
        else:
            for action_index, prior in priors.items():
                visit_policy[action_index] = prior

        average_scores = [float("-inf")] * len(sample.legal_actions)
        for action_index in candidate_indices:
            visits = int(stats[action_index]["visits"])
            average_scores[action_index] = (
                float(stats[action_index]["value_sum"]) / float(max(visits, 1))
                if visits > 0
                else 0.0
            )
            visits = max(1, int(stats[action_index]["visits"]))
            trace_rows.append(
                {
                    "action_index": int(action_index),
                    "action_id": sample.legal_actions[action_index].action_id,
                    "card_id": sample.legal_actions[action_index].card_id,
                    "prior": round(float(priors.get(action_index, 0.0)), 6),
                    "visits": int(stats[action_index]["visits"]),
                    "score_avg": round(average_scores[action_index], 6),
                    "score_best": round(float(stats[action_index]["best_score"]), 6),
                    "hp_quality_avg": round(float(stats[action_index]["hp_sum"]) / float(visits), 6),
                    "speed_quality_avg": round(float(stats[action_index]["speed_sum"]) / float(visits), 6),
                    "outcome_mode": _mode(list(stats[action_index]["outcomes"])),
                }
            )

        ordered = sorted(
            range(len(sample.legal_actions)),
            key=lambda idx: (visit_policy[idx], average_scores[idx]),
            reverse=True,
        )
        best_action_index = ordered[0]
        topk = ordered[: min(self._budget.trace_topk, len(ordered))]
        second_score = average_scores[ordered[1]] if len(ordered) > 1 else 0.0
        ranking_margin = max(0.05, float(average_scores[ordered[0]] - second_score))
        label = SearchLabel(
            policy=visit_policy,
            topk_indices=topk,
            best_action_index=best_action_index,
            ranking_margin=ranking_margin,
            search_value=float(max(0.0, average_scores[best_action_index])),
            search_trace=sorted(trace_rows, key=lambda item: (int(item["visits"]), float(item["score_avg"])), reverse=True),
            metadata={
                "search_backend": "CombatSearchBackend",
                "search_mode": "mcts_root_search",
                "search_budget_max_root_actions": self._budget.max_root_actions,
                "search_budget_rollouts_per_action": self._budget.rollouts_per_action,
                "search_budget_max_branch_steps": self._budget.max_branch_steps,
                "search_budget_total_simulations": total_simulations,
                "search_case_id": self._case.case_id,
            },
        )
        self._remember_cached_label(cache_key, label)
        return self._with_search_metadata(
            label,
            search_duration_s=time.perf_counter() - started_at,
            search_simulations=search_simulations,
            search_cache_hit=False,
            snapshot_used=snapshot_used,
            prefix_replay_count=prefix_replay_count,
        )

    def _candidate_root_indices(self, sample) -> list[int]:
        if not sample.legal_actions:
            return []
        prior_scores = self._extract_policy_scores(sample, len(sample.legal_actions))
        ordered = sorted(range(len(sample.legal_actions)), key=lambda idx: prior_scores[idx], reverse=True)
        limit = min(len(ordered), self._budget.max_root_actions)
        return ordered[:limit]

    def _empty_stats(self, candidate_indices: list[int]) -> dict[int, RootActionStats]:
        return {
            action_index: {
                "visits": 0,
                "value_sum": 0.0,
                "best_score": float("-inf"),
                "hp_sum": 0.0,
                "speed_sum": 0.0,
                "outcomes": [],
            }
            for action_index in candidate_indices
        }

    def _run_root_simulations(
        self,
        *,
        runtime,
        sample,
        candidate_indices: list[int],
        priors: dict[int, float],
        total_simulations: int,
        root_state_id: str | None,
        policy,
    ) -> tuple[dict[int, RootActionStats], int, int]:
        stats = self._empty_stats(candidate_indices)
        search_simulations = 0
        prefix_replays = 0
        for _simulation in range(total_simulations):
            selected_index = self._select_root_action(candidate_indices, priors, stats)
            sample_action = sample.legal_actions[selected_index]
            if root_state_id is not None:
                replay_ok, restored_state = self._restore_root_state(runtime, root_state_id)
                if not replay_ok or restored_state is None:
                    replay_ok, restored_state = self._replay_prefix(runtime, sample)
                    prefix_replays += 1
            else:
                replay_ok, restored_state = self._replay_prefix(runtime, sample)
                prefix_replays += 1
            if not replay_ok or not restored_state.legal_actions:
                continue
            runtime_action_index = (
                selected_index
                if 0 <= selected_index < len(restored_state.legal_actions)
                else _resolve_runtime_action_index(sample_action, restored_state.legal_actions)
            )
            if runtime_action_index is None:
                continue
            next_state = runtime.step(runtime_action_index)
            search_simulations += 1
            branch_result = self._continue_rollout(
                runtime,
                next_state,
                policy=policy,
                history=list(sample.history),
                root_action=sample.legal_actions[selected_index],
            )
            action_stats = stats[selected_index]
            action_stats["visits"] += 1
            action_stats["value_sum"] += float(branch_result["fight_quality_score"])
            action_stats["hp_sum"] += float(branch_result["hp_quality_score"])
            action_stats["speed_sum"] += float(branch_result["speed_quality"])
            action_stats["best_score"] = max(action_stats["best_score"], float(branch_result["fight_quality_score"]))
            action_stats["outcomes"].append(str(branch_result["outcome"]))
        return stats, search_simulations, prefix_replays

    def _get_runtime(self) -> SkadaReplayRuntime:
        if self._runtime is None:
            self._runtime = SkadaReplayRuntime(
                self._case,
                port=self._port,
                auto_launch=self._auto_launch,
                connect_timeout_s=self._connect_timeout_s,
            )
        return self._runtime

    def _budget_signature(self) -> tuple[int, int, int, bool]:
        return (
            self._budget.max_root_actions,
            self._budget.rollouts_per_action,
            self._budget.max_branch_steps,
            self._budget.allow_branching,
        )

    def _build_cache_key(self, sample) -> tuple[object, ...]:
        policy_scores = tuple(round(float(value), 3) for value in self._extract_policy_scores(sample, len(sample.legal_actions)))
        enemy_signature = tuple((enemy.enemy_id, round(float(enemy.hp), 1), round(float(enemy.block), 1)) for enemy in sample.state.enemies)
        hand_signature = tuple((card.card_id, round(float(card.cost_now), 1)) for card in sample.state.hand)
        prefix_signature = tuple(int(value) for value in (sample.metadata.get("prefix_action_indices") or []))
        return (
            self._case.case_id,
            sample.state.context.floor,
            sample.state.context.encounter_id,
            round(float(sample.state.player.hp), 1),
            round(float(sample.state.player.block), 1),
            round(float(sample.state.player.energy), 1),
            enemy_signature,
            hand_signature,
            prefix_signature,
            policy_scores,
        )

    def _remember_cached_label(self, cache_key: tuple[object, ...], label: SearchLabel) -> None:
        if self._budget.root_cache_size <= 0:
            return
        cached_label = SearchLabel(
            policy=list(label.policy),
            topk_indices=list(label.topk_indices),
            best_action_index=int(label.best_action_index),
            ranking_margin=float(label.ranking_margin),
            search_value=float(label.search_value),
            search_trace=[dict(item) for item in label.search_trace],
            metadata=dict(label.metadata),
        )
        self._cached_labels[cache_key] = CachedSearchResult(
            label=cached_label,
            budget_signature=self._budget_signature(),
        )
        self._cached_labels.move_to_end(cache_key)
        while len(self._cached_labels) > self._budget.root_cache_size:
            self._cached_labels.popitem(last=False)

    def _replay_prefix(self, runtime: SkadaReplayRuntime, sample) -> tuple[bool, object]:
        state = runtime.reset(seed=str(sample.state.context.metadata.get("seed", "")))
        prefix = sample.metadata.get("prefix_action_indices") or []
        if not isinstance(prefix, list):
            return False, state
        for raw_index in prefix:
            try:
                action_index = int(raw_index)
            except (TypeError, ValueError):
                return False, state
            if state.terminal or not state.legal_actions:
                return False, state
            if action_index < 0 or action_index >= len(state.legal_actions):
                return False, state
            state = runtime.step(action_index)
        return True, state

    def _try_save_state(self, runtime: SkadaReplayRuntime) -> str | None:
        try:
            return runtime.save_state()
        except Exception:
            return None

    def _restore_root_state(self, runtime: SkadaReplayRuntime, state_id: str) -> tuple[bool, object]:
        try:
            return True, runtime.load_state(state_id)
        except Exception:
            return False, None

    def _continue_rollout(self, runtime: SkadaReplayRuntime, state, *, policy=None, history: list[HistoryStep] | None = None, root_action=None) -> BranchResult:
        progress_steps = 0
        no_progress_steps = 0
        max_no_progress_streak = 0
        current_no_progress_streak = 0
        branch_history = list(history or [])
        if root_action is not None:
            branch_history.append(
                HistoryStep(
                    state=None,
                    action=root_action,
                    delta=TransitionDelta(),
                    history_token=[],
                )
            )
        step_count = 1
        rollout_horizon = min(self._budget.max_branch_steps, max(0, self._budget.leaf_eval_horizon))
        while step_count < rollout_horizon and not state.terminal and state.legal_actions:
            action_index = self._select_rollout_action(state, policy=policy, history=branch_history)
            next_state = runtime.step(action_index)
            progress = assess_transition_progress(state, next_state)
            delta = compute_transition_delta(state, next_state)
            branch_history.append(
                HistoryStep(
                    state=state,
                    action=state.legal_actions[action_index],
                    delta=delta,
                    history_token=[],
                )
            )
            if progress.made_progress:
                progress_steps += 1
                current_no_progress_streak = 0
            else:
                no_progress_steps += 1
                current_no_progress_streak += 1
                max_no_progress_streak = max(max_no_progress_streak, current_no_progress_streak)
            state = next_state
            step_count += 1
            if self._budget.allow_branching and step_count >= min(4, self._budget.max_branch_steps):
                break
        truncated = bool(not state.terminal and step_count >= self._budget.max_branch_steps)
        if state.terminal:
            label = _build_fight_label(state, truncated=truncated)
        else:
            label = self._estimate_leaf_label(policy, state, branch_history)
        no_progress_ratio = no_progress_steps / max(step_count, 1)
        fight_quality_score = compute_fight_score(
            label,
            encounter_class=state.context.encounter_class,
            truncated=truncated,
            no_progress_ratio=no_progress_ratio,
            max_no_progress_streak=max_no_progress_streak,
            step_count=step_count,
        )
        hp_quality_score = compute_hp_quality_score(
            label,
            encounter_class=state.context.encounter_class,
        )
        speed_quality = max(0.0, min(1.0, fight_quality_score - label.fight_win * 0.55))
        return {
            "fight_quality_score": float(fight_quality_score),
            "hp_quality_score": float(hp_quality_score),
            "speed_quality": float(speed_quality),
            "step_count": int(step_count),
            "truncated": bool(truncated),
            "outcome": "timeout" if truncated else str(state.run_outcome),
        }

    def _estimate_leaf_label(self, policy, state, history: list[HistoryStep]) -> FightLabel:
        evaluate_hook = getattr(policy, "evaluate_state", None)
        if callable(evaluate_hook):
            result = evaluate_hook(state, history)
            return FightLabel(
                fight_win=max(0.0, min(1.0, float(result.get("fight_win_prob", 0.0) or 0.0))),
                enemy_hp_fraction_dealt=max(0.0, min(1.0, float(result.get("enemy_hp_fraction_dealt", 0.0) or 0.0))),
                self_hp_fraction_remaining=max(0.0, min(1.0, float(result.get("self_hp_fraction_remaining", 0.0) or 0.0))),
                player_hp=max(0.0, float(state.player.hp)),
                player_max_hp=max(1.0, float(state.player.max_hp)),
            )
        return _build_fight_label(state, truncated=False)

    def _select_rollout_action(self, state, *, policy=None, history: list[HistoryStep] | None = None) -> int:
        evaluate_hook = getattr(policy, "evaluate_state", None)
        if callable(evaluate_hook):
            result = evaluate_hook(state, list(history or []))
            scores = list(result.get("scores", []))
            if scores:
                return max(range(len(scores)), key=lambda index: scores[index])
        scored: list[tuple[float, int]] = []
        for index, action in enumerate(state.legal_actions):
            immediate_score = 0.10 * float(action.damage_now) + 0.04 * float(action.block_now)
            if action.action_type == "end_turn":
                immediate_score -= 0.35
            if action.damage_now > 0 and action.target_summary is not None:
                remaining = max(0.0, action.target_summary.hp - action.damage_now)
                if remaining <= 0:
                    immediate_score += 0.9
            if action.can_execute:
                immediate_score += 0.05
            scored.append((float(immediate_score), index))
        scored.sort(reverse=True)
        return scored[0][1] if scored else 0

    def _fallback_label(self, sample, *, backend_name: str) -> SearchLabel:
        prior_scores = self._extract_policy_scores(sample, len(sample.legal_actions))
        policy = _scores_to_policy(prior_scores)
        ordered = sorted(range(len(policy)), key=lambda idx: policy[idx], reverse=True)
        best_action_index = ordered[0] if ordered else -1
        second = policy[ordered[1]] if len(ordered) > 1 else 0.0
        return SearchLabel(
            policy=policy,
            topk_indices=ordered[: min(self._budget.trace_topk, len(ordered))],
            best_action_index=best_action_index,
            ranking_margin=max(0.05, float(policy[best_action_index] - second)) if best_action_index >= 0 else 0.05,
            search_value=0.0,
            metadata={"search_backend": backend_name, "search_mode": "mcts_root_search"},
        )

    def _extract_policy_scores(self, sample, width: int) -> list[float]:
        raw_scores = sample.metadata.get("policy_scores") if isinstance(sample.metadata, dict) else None
        if isinstance(raw_scores, list) and raw_scores:
            values = [float(value) for value in raw_scores[:width]]
            if len(values) < width:
                values.extend([0.0] * (width - len(values)))
            return values
        return [0.0] * width

    def _root_priors(self, sample, candidate_indices: list[int]) -> dict[int, float]:
        width = len(sample.legal_actions)
        policy_scores = self._extract_policy_scores(sample, width)
        if not candidate_indices:
            return {}
        max_score = max(policy_scores[index] for index in candidate_indices) if candidate_indices else 0.0
        exp_scores = {
            index: math.exp(max(-50.0, min(50.0, policy_scores[index] - max_score)))
            for index in candidate_indices
        }
        total = sum(exp_scores.values()) or 1.0
        return {index: value / total for index, value in exp_scores.items()}

    def _select_root_action(self, candidate_indices: list[int], priors: dict[int, float], stats: dict[int, RootActionStats]) -> int:
        unvisited = [index for index in candidate_indices if int(stats[index]["visits"]) <= 0]
        if unvisited:
            return max(unvisited, key=lambda index: float(priors.get(index, 0.0)))
        total_visits = sum(int(stats[index]["visits"]) for index in candidate_indices)
        best_index = candidate_indices[0]
        best_score = -float("inf")
        for index in candidate_indices:
            action_stats = stats[index]
            visits = int(action_stats["visits"])
            q_value = float(action_stats["value_sum"]) / float(max(visits, 1)) if visits > 0 else 0.0
            prior = float(priors.get(index, 0.0))
            u_value = self._budget.puct_exploration * prior * math.sqrt(max(total_visits, 1)) / float(1 + visits)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _with_search_metadata(
        self,
        label: SearchLabel,
        *,
        search_duration_s: float,
        search_simulations: int,
        search_cache_hit: bool,
        snapshot_used: bool,
        prefix_replay_count: int,
    ) -> SearchLabel:
        metadata = dict(label.metadata)
        metadata.update(
            {
                "search_duration_s": round(float(search_duration_s), 6),
                "search_simulations": int(search_simulations),
                "search_cache_hit": bool(search_cache_hit),
                "search_snapshot_restore_used": bool(snapshot_used),
                "search_prefix_replay_count": int(prefix_replay_count),
            }
        )
        label.metadata = metadata
        return label


class MultiCaseSearchBackend:
    """按 replay case 分发 same-seed search backend。"""

    def __init__(
        self,
        cases: Iterable[SkadaCombatCase],
        *,
        config: SearchConfig,
        port: int = 15527,
        auto_launch: bool = True,
        connect_timeout_s: float = 30.0,
    ):
        case_list = list(cases)
        if not case_list:
            raise ValueError("MultiCaseSearchBackend 需要至少一个 case。")
        self._cases = case_list
        self._config = config
        self._port = port
        self._auto_launch = auto_launch
        self._connect_timeout_s = connect_timeout_s
        self._search_backends = {
            case.case_id: CombatSearchBackend(
                case,
                config=config,
                port=port,
                auto_launch=auto_launch,
                connect_timeout_s=connect_timeout_s,
            )
            for case in case_list
        }
        self._fallback = next(iter(self._search_backends.values()))

    def label_request(self, request: SearchRequest, runtime_factory=None, seed: str | None = None, policy=None) -> SearchLabel:
        case_id = str(request.sample.state.context.metadata.get("skada_case_id") or "")
        search_backend = self._search_backends.get(case_id, self._fallback)
        label = search_backend.label_request(request, runtime_factory=runtime_factory, seed=seed, policy=policy)
        if search_backend is self._fallback and case_id not in self._search_backends:
            metadata = dict(label.metadata)
            metadata["search_fallback_case_id"] = case_id
            label.metadata = metadata
        return label

    def clone_for_port(self, port: int) -> "MultiCaseSearchBackend":
        return MultiCaseSearchBackend(
            self._cases,
            config=self._config,
            port=port,
            auto_launch=self._auto_launch,
            connect_timeout_s=self._connect_timeout_s,
        )

    def label_requests(self, requests: list[SearchRequest], runtime_factory=None, policy=None) -> list[SearchLabel]:
        return [self.label_request(request, runtime_factory=runtime_factory, policy=policy) for request in requests]



def _scores_to_policy(scores: list[float]) -> list[float]:
    if not scores:
        return []
    floor = min(scores)
    shifted = [max(0.0, float(score) - floor + 1e-4) for score in scores]
    total = sum(shifted) or 1.0
    return [value / total for value in shifted]


def _build_fight_label(state, *, truncated: bool = False) -> FightLabel:
    enemy_max_hp = sum(enemy.max_hp for enemy in state.enemies) or 1.0
    enemy_remaining = sum(max(0.0, enemy.hp) for enemy in state.enemies)
    enemy_hp_fraction_dealt = max(0.0, min(1.0, 1.0 - enemy_remaining / enemy_max_hp))
    self_hp_fraction_remaining = 0.0
    if state.player.max_hp > 0:
        self_hp_fraction_remaining = max(0.0, min(1.0, state.player.hp / state.player.max_hp))
    if truncated:
        self_hp_fraction_remaining = 0.0
    fight_win = 1.0 if (not truncated and str(state.run_outcome).lower() in {"victory", "win"}) else 0.0
    return FightLabel(
        fight_win=fight_win,
        enemy_hp_fraction_dealt=enemy_hp_fraction_dealt,
        self_hp_fraction_remaining=self_hp_fraction_remaining,
        player_hp=max(0.0, float(state.player.hp)),
        player_max_hp=max(0.0, float(state.player.max_hp)),
    )


def _mode(values: list[str]) -> str:
    if not values:
        return ""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _resolve_runtime_action_index(sample_action, runtime_actions) -> int | None:
    runtime_by_id = {action.action_id: index for index, action in enumerate(runtime_actions)}
    runtime_index = runtime_by_id.get(sample_action.action_id)
    if runtime_index is not None:
        return runtime_index
    exact_candidates = [
        index
        for index, runtime_action in enumerate(runtime_actions)
        if runtime_action.action_type == sample_action.action_type
        and runtime_action.card_id == sample_action.card_id
        and runtime_action.target_id == sample_action.target_id
    ]
    if exact_candidates:
        return exact_candidates[0]
    fuzzy_candidates = [
        index
        for index, runtime_action in enumerate(runtime_actions)
        if runtime_action.action_type == sample_action.action_type
        and runtime_action.card_id == sample_action.card_id
    ]
    if fuzzy_candidates:
        return fuzzy_candidates[0]
    return None
