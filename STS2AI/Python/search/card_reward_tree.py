from __future__ import annotations

from dataclasses import dataclass, field
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class RewardTreeConfig:
    max_reward_depth: int = 3
    beam_width: int = 2
    rollout_max_combats: int = 3
    rollout_max_steps: int = 500
    rerun_low_spread_threshold: float = 0.0
    rerun_max_combats: int = 5
    rerun_max_steps: int = 900
    advance_max_steps: int = 900
    blend_local_weight: float = 0.6
    use_local_ort_rollout: bool = False
    local_ort_max_combat_steps: int = 600
    stop_floor: int | None = None
    max_option_seconds: float | None = 4.0
    recurse_only_when_spread_below: float | None = 0.25


@dataclass(slots=True)
class RewardTreeOptionValue:
    option_index: int
    local_score: float
    continuation_score: float
    final_score: float
    child_count: int
    truncated_reason: str | None = None


@dataclass(slots=True)
class RewardTreeResult:
    scores: list[float]
    root_outcomes: dict[int, Any]
    option_values: list[RewardTreeOptionValue]
    summary: dict[str, Any] = field(default_factory=dict)


def evaluate_card_reward_tree(
    *,
    client: Any,
    seed: str,
    state: dict[str, Any],
    root_options: list[dict[str, Any]],
    sample_index: int,
    debug_rollout_trace_dir: str | None,
    combat_evaluator: Any | None,
    ppo_policy: Any | None,
    config: RewardTreeConfig,
    apply_action: Callable[..., dict[str, Any]],
    settle_after_choice: Callable[..., dict[str, Any]],
    extract_floor: Callable[[dict[str, Any]], int],
    extract_player_hp: Callable[[dict[str, Any]], int],
    extract_card_reward_options: Callable[[dict[str, Any]], list[dict[str, Any]]],
    did_reach_boss: Callable[[dict[str, Any]], bool],
    choose_rollout_decision: Callable[..., Any],
    evaluate_branch_outcomes: Callable[..., dict[int, Any]],
    compute_option_scores: Callable[[dict[int, Any], int], list[float]],
) -> RewardTreeResult:
    root_floor = int(extract_floor(state))
    hp_before = int(extract_player_hp(state))
    root_scores: list[float] = []
    root_outcomes: dict[int, Any] = {}
    option_values: list[RewardTreeOptionValue] = []
    total_nodes = 0

    with tempfile.TemporaryDirectory(prefix=f"sts2_reward_tree_f{root_floor:02d}_") as tmpdir:
        root_snapshot = str(Path(tmpdir) / "root_snapshot.json")
        client.export_state(root_snapshot)

        def _restore_root(option_index: int) -> None:
            base_state = client.import_state(root_snapshot)
            branch_state = apply_action(client, base_state, root_options[option_index]["action"])
            settle_after_choice(
                client,
                branch_state,
                previous_state_type="card_reward",
                previous_floor=root_floor,
            )

        local_outcomes = evaluate_branch_outcomes(
            client=client,
            seed=seed,
            floor=root_floor,
            hp_before=hp_before,
            sample_index=sample_index,
            sample_type="card_reward_tree",
            options=root_options,
            restore_fn=_restore_root,
            combat_evaluator=combat_evaluator,
            ppo_policy=ppo_policy,
            debug_rollout_trace_dir=debug_rollout_trace_dir,
            max_combats=config.rollout_max_combats,
            max_steps=config.rollout_max_steps,
            use_local_ort_rollout=config.use_local_ort_rollout,
            local_ort_max_combat_steps=config.local_ort_max_combat_steps,
        )
        local_scores = compute_option_scores(local_outcomes, max_hp=max(hp_before, 1))
        local_spread = (max(local_scores) - min(local_scores)) if local_scores else 0.0

        for root_idx, _option in enumerate(root_options):
            outcome = local_outcomes.get(root_idx) or next(iter(local_outcomes.values()))
            root_outcomes[root_idx] = outcome
            local_score = float(local_scores[root_idx]) if root_idx < len(local_scores) else 0.0
            _restore_root(root_idx)
            continuation_score = 0.0
            child_count = 0
            truncated_reason: str | None = None
            nodes_visited = 0
            should_recurse = max(0, config.max_reward_depth - 1) > 0
            gate = config.recurse_only_when_spread_below
            if should_recurse and gate is not None and local_spread >= float(gate):
                should_recurse = False
                truncated_reason = "local_spread_gate"
            if should_recurse:
                deadline = (
                    time.monotonic() + float(config.max_option_seconds)
                    if config.max_option_seconds is not None and float(config.max_option_seconds) > 0.0
                    else None
                )
                continuation_score, child_count, truncated_reason, nodes_visited = _explore_reward_children(
                    client=client,
                    seed=seed,
                    depth_remaining=max(0, config.max_reward_depth - 1),
                    beam_width=max(1, config.beam_width),
                    sample_index=sample_index,
                    debug_rollout_trace_dir=debug_rollout_trace_dir,
                    combat_evaluator=combat_evaluator,
                    ppo_policy=ppo_policy,
                    config=config,
                    apply_action=apply_action,
                    settle_after_choice=settle_after_choice,
                    extract_floor=extract_floor,
                    extract_player_hp=extract_player_hp,
                    extract_card_reward_options=extract_card_reward_options,
                    did_reach_boss=did_reach_boss,
                    choose_rollout_decision=choose_rollout_decision,
                    evaluate_branch_outcomes=evaluate_branch_outcomes,
                    compute_option_scores=compute_option_scores,
                    deadline=deadline,
                )
            total_nodes += nodes_visited
            final_score = round(
                config.blend_local_weight * local_score
                + (1.0 - config.blend_local_weight) * continuation_score,
                4,
            ) if child_count > 0 else round(local_score, 4)
            root_scores.append(final_score)
            option_values.append(
                RewardTreeOptionValue(
                    option_index=root_idx,
                    local_score=round(local_score, 4),
                    continuation_score=round(continuation_score, 4),
                    final_score=final_score,
                    child_count=child_count,
                    truncated_reason=truncated_reason,
                )
            )

        return RewardTreeResult(
            scores=root_scores,
            root_outcomes=root_outcomes,
            option_values=option_values,
            summary={
                "label_source": "reward_tree",
                "max_reward_depth": int(config.max_reward_depth),
                "beam_width": int(config.beam_width),
                "total_reward_nodes": int(total_nodes),
            },
        )


def _explore_reward_children(
    *,
    client: Any,
    seed: str,
    depth_remaining: int,
    beam_width: int,
    sample_index: int,
    debug_rollout_trace_dir: str | None,
    combat_evaluator: Any | None,
    ppo_policy: Any | None,
    config: RewardTreeConfig,
    apply_action: Callable[..., dict[str, Any]],
    settle_after_choice: Callable[..., dict[str, Any]],
    extract_floor: Callable[[dict[str, Any]], int],
    extract_player_hp: Callable[[dict[str, Any]], int],
    extract_card_reward_options: Callable[[dict[str, Any]], list[dict[str, Any]]],
    did_reach_boss: Callable[[dict[str, Any]], bool],
    choose_rollout_decision: Callable[..., Any],
    evaluate_branch_outcomes: Callable[..., dict[int, Any]],
    compute_option_scores: Callable[[dict[int, Any], int], list[float]],
    deadline: float | None,
) -> tuple[float, int, str | None, int]:
    if deadline is not None and time.monotonic() >= deadline:
        return 0.0, 0, "time_budget", 0
    if depth_remaining <= 0:
        return 0.0, 0, "depth_limit", 0

    advance_result = _advance_to_next_reward(
        client=client,
        seed=seed,
        max_steps=config.advance_max_steps,
        did_reach_boss=did_reach_boss,
        choose_rollout_decision=choose_rollout_decision,
        apply_action=apply_action,
        combat_evaluator=combat_evaluator,
        ppo_policy=ppo_policy,
        use_local_ort_rollout=config.use_local_ort_rollout,
        local_ort_max_combat_steps=config.local_ort_max_combat_steps,
        stop_floor=config.stop_floor,
    )
    state = advance_result["state"]
    reason = str(advance_result["reason"])
    if reason != "next_card_reward":
        return 0.0, 0, reason, 0

    options = extract_card_reward_options(state)
    if len(options) < 2:
        return 0.0, 0, "insufficient_options", 0

    floor = int(extract_floor(state))
    hp_before = int(extract_player_hp(state))
    option_scores: list[tuple[float, int]] = []
    local_best: list[tuple[int, float, Any]] = []
    nodes_visited = len(options)

    with tempfile.TemporaryDirectory(prefix=f"sts2_reward_tree_child_f{floor:02d}_") as tmpdir:
        snapshot_path = str(Path(tmpdir) / "child_snapshot.json")
        client.export_state(snapshot_path)

        def _restore_child(option_index: int) -> None:
            base_state = client.import_state(snapshot_path)
            branch_state = apply_action(client, base_state, options[option_index]["action"])
            settle_after_choice(
                client,
                branch_state,
                previous_state_type="card_reward",
                previous_floor=floor,
            )

        outcomes = evaluate_branch_outcomes(
            client=client,
            seed=seed,
            floor=floor,
            hp_before=hp_before,
            sample_index=sample_index,
            sample_type="card_reward_tree",
            options=options,
            restore_fn=_restore_child,
            combat_evaluator=combat_evaluator,
            ppo_policy=ppo_policy,
            debug_rollout_trace_dir=debug_rollout_trace_dir,
            max_combats=config.rollout_max_combats,
            max_steps=config.rollout_max_steps,
            use_local_ort_rollout=config.use_local_ort_rollout,
            local_ort_max_combat_steps=config.local_ort_max_combat_steps,
        )
        scores = compute_option_scores(outcomes, max_hp=max(hp_before, 1))
        if (
            config.rerun_low_spread_threshold > 0
            and scores
            and max(scores) - min(scores) < config.rerun_low_spread_threshold
            and config.rerun_max_combats > config.rollout_max_combats
        ):
            outcomes = evaluate_branch_outcomes(
                client=client,
                seed=seed,
                floor=floor,
                hp_before=hp_before,
                sample_index=sample_index,
                sample_type="card_reward_tree",
                options=options,
                restore_fn=_restore_child,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                debug_rollout_trace_dir=debug_rollout_trace_dir,
                max_combats=config.rerun_max_combats,
                max_steps=config.rerun_max_steps,
                use_local_ort_rollout=config.use_local_ort_rollout,
                local_ort_max_combat_steps=config.local_ort_max_combat_steps,
            )
            scores = compute_option_scores(outcomes, max_hp=max(hp_before, 1))

        for idx, score in enumerate(scores):
            option_scores.append((float(score), idx))
            local_best.append((idx, float(score), outcomes.get(idx)))

        local_spread = (max(scores) - min(scores)) if scores else 0.0
        gate = config.recurse_only_when_spread_below
        if gate is not None and local_spread >= float(gate):
            return max(score for _, score, _ in local_best), len(local_best), "local_spread_gate", nodes_visited

        top_children = sorted(option_scores, reverse=True)[:beam_width]
        continuation_values: list[float] = []
        truncated_reason: str | None = None
        for _child_score, child_idx in top_children:
            if deadline is not None and time.monotonic() >= deadline:
                if truncated_reason is None:
                    truncated_reason = "time_budget"
                break
            _restore_child(child_idx)
            nested_score, _child_count, nested_reason, nested_nodes = _explore_reward_children(
                client=client,
                seed=seed,
                depth_remaining=depth_remaining - 1,
                beam_width=beam_width,
                sample_index=sample_index,
                debug_rollout_trace_dir=debug_rollout_trace_dir,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                config=config,
                apply_action=apply_action,
                settle_after_choice=settle_after_choice,
                extract_floor=extract_floor,
                extract_player_hp=extract_player_hp,
                extract_card_reward_options=extract_card_reward_options,
                did_reach_boss=did_reach_boss,
                choose_rollout_decision=choose_rollout_decision,
                evaluate_branch_outcomes=evaluate_branch_outcomes,
                compute_option_scores=compute_option_scores,
                deadline=deadline,
            )
            nodes_visited += nested_nodes
            blended = round(
                config.blend_local_weight * float(_child_score)
                + (1.0 - config.blend_local_weight) * nested_score,
                4,
            ) if _child_count > 0 else round(float(_child_score), 4)
            continuation_values.append(blended)
            if truncated_reason is None and nested_reason not in {None, "depth_limit"}:
                truncated_reason = nested_reason

        if continuation_values:
            return max(continuation_values), len(top_children), truncated_reason, nodes_visited
        if local_best:
            return max(score for _, score, _ in local_best), len(local_best), truncated_reason, nodes_visited
        return 0.0, 0, truncated_reason or "no_children", nodes_visited


def _advance_to_next_reward(
    *,
    client: Any,
    seed: str,
    max_steps: int,
    did_reach_boss: Callable[[dict[str, Any]], bool],
    choose_rollout_decision: Callable[..., Any],
    apply_action: Callable[..., dict[str, Any]],
    combat_evaluator: Any | None,
    ppo_policy: Any | None,
    use_local_ort_rollout: bool,
    local_ort_max_combat_steps: int,
    stop_floor: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(f"{seed}_tree_advance")
    state = client.get_state()
    for _ in range(max_steps):
        if state.get("terminal"):
            return {"reason": "terminal", "state": state}
        floor = int((state.get("run") or {}).get("floor") or 0)
        if stop_floor is not None and floor >= int(stop_floor):
            return {"reason": "floor_cap", "state": state}
        if did_reach_boss(state):
            return {"reason": "boss_reached", "state": state}
        if str(state.get("state_type") or "").lower() == "card_reward":
            return {"reason": "next_card_reward", "state": state}
        if (
            str(state.get("state_type") or "").lower() in {"monster", "elite", "boss", "combat"}
            and use_local_ort_rollout
            and getattr(client, "supports_local_ort", False)
        ):
            result = client.run_combat_local(max_steps=local_ort_max_combat_steps)
            post_state = result.get("state") if isinstance(result, dict) else None
            state = post_state if isinstance(post_state, dict) else client.get_state()
            continue
        legal = state.get("legal_actions") or []
        if not legal:
            state = apply_action(client, state, {"action": "wait"})
            continue
        decision = choose_rollout_decision(
            state,
            legal,
            rng,
            combat_evaluator=combat_evaluator,
            ppo_policy=ppo_policy,
        )
        state = apply_action(client, state, decision.action)
    return {"reason": "advance_timeout", "state": state}
