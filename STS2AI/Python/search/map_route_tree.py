from __future__ import annotations

from dataclasses import dataclass, field
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class MapRouteConfig:
    max_map_depth: int = 5
    beam_width: int = 2
    advance_max_steps: int = 1600
    boss_bonus: float = 5.0
    floor_weight: float = 0.2
    hp_weight: float = 0.8
    card_reward_weight: float = 0.2
    stop_floor: int | None = None
    max_option_seconds: float | None = 3.0


@dataclass(slots=True)
class MapRouteOptionValue:
    option_index: int
    route_score: float
    child_count: int
    terminal_reason: str | None = None
    end_floor: int = 0
    boss_reached: bool = False
    hp_after: int = 0


@dataclass(slots=True)
class MapRouteResult:
    scores: list[float]
    route_outcomes: dict[int, dict[str, Any]]
    option_values: list[MapRouteOptionValue]
    summary: dict[str, Any] = field(default_factory=dict)


def evaluate_map_route_tree(
    *,
    client: Any,
    seed: str,
    state: dict[str, Any],
    root_options: list[dict[str, Any]],
    config: MapRouteConfig,
    apply_action: Callable[..., dict[str, Any]],
    settle_after_choice: Callable[..., dict[str, Any]],
    extract_floor: Callable[[dict[str, Any]], int],
    extract_player_hp: Callable[[dict[str, Any]], int],
    did_reach_boss: Callable[[dict[str, Any]], bool],
    choose_rollout_decision: Callable[..., Any],
    choose_deterministic_screen_action: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any] | None],
    resolve_card_reward_choice: Callable[[Any, str, dict[str, Any]], tuple[dict[str, Any], float]],
    extract_map_options: Callable[[dict[str, Any]], list[dict[str, Any]]],
    use_local_ort_rollout: bool,
    local_ort_max_combat_steps: int,
) -> MapRouteResult:
    root_scores: list[float] = []
    route_outcomes: dict[int, dict[str, Any]] = {}
    option_values: list[MapRouteOptionValue] = []
    total_map_nodes = 0
    root_floor = int(extract_floor(state))

    with tempfile.TemporaryDirectory(prefix=f"sts2_map_tree_f{root_floor:02d}_") as tmpdir:
        root_snapshot = str(Path(tmpdir) / "map_root_snapshot.json")
        client.export_state(root_snapshot)
        for root_idx, option in enumerate(root_options):
            base_state = client.import_state(root_snapshot)
            branch_state = apply_action(client, base_state, option["action"])
            branch_state = settle_after_choice(
                client,
                branch_state,
                previous_state_type="map",
                previous_floor=root_floor,
            )
            deadline = (
                time.monotonic() + float(config.max_option_seconds)
                if config.max_option_seconds is not None and float(config.max_option_seconds) > 0.0
                else None
            )
            score, terminal_summary, child_count, explored = _advance_route_value(
                client=client,
                seed=f"{seed}_map{root_idx}",
                state=branch_state,
                depth_remaining=max(0, config.max_map_depth - 1),
                config=config,
                apply_action=apply_action,
                settle_after_choice=settle_after_choice,
                extract_floor=extract_floor,
                extract_player_hp=extract_player_hp,
                did_reach_boss=did_reach_boss,
                choose_rollout_decision=choose_rollout_decision,
                choose_deterministic_screen_action=choose_deterministic_screen_action,
                resolve_card_reward_choice=resolve_card_reward_choice,
                extract_map_options=extract_map_options,
                use_local_ort_rollout=use_local_ort_rollout,
                local_ort_max_combat_steps=local_ort_max_combat_steps,
                accumulated_reward_score=0.0,
                deadline=deadline,
            )
            total_map_nodes += explored
            score = round(float(score), 4)
            root_scores.append(score)
            route_outcomes[root_idx] = terminal_summary
            option_values.append(
                MapRouteOptionValue(
                    option_index=root_idx,
                    route_score=score,
                    child_count=child_count,
                    terminal_reason=str(terminal_summary.get("terminal_reason") or ""),
                    end_floor=int(terminal_summary.get("end_floor") or 0),
                    boss_reached=bool(terminal_summary.get("boss_reached")),
                    hp_after=int(terminal_summary.get("hp_after") or 0),
                )
            )
        client.import_state(root_snapshot)

    return MapRouteResult(
        scores=root_scores,
        route_outcomes=route_outcomes,
        option_values=option_values,
        summary={
            "label_source": "map_route_tree",
            "max_map_depth": int(config.max_map_depth),
            "beam_width": int(config.beam_width),
            "total_map_nodes": int(total_map_nodes),
        },
    )


def _advance_route_value(
    *,
    client: Any,
    seed: str,
    state: dict[str, Any],
    depth_remaining: int,
    config: MapRouteConfig,
    apply_action: Callable[..., dict[str, Any]],
    settle_after_choice: Callable[..., dict[str, Any]],
    extract_floor: Callable[[dict[str, Any]], int],
    extract_player_hp: Callable[[dict[str, Any]], int],
    did_reach_boss: Callable[[dict[str, Any]], bool],
    choose_rollout_decision: Callable[..., Any],
    choose_deterministic_screen_action: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any] | None],
    resolve_card_reward_choice: Callable[[Any, str, dict[str, Any]], tuple[dict[str, Any], float]],
    extract_map_options: Callable[[dict[str, Any]], list[dict[str, Any]]],
    use_local_ort_rollout: bool,
    local_ort_max_combat_steps: int,
    accumulated_reward_score: float,
    deadline: float | None,
) -> tuple[float, dict[str, Any], int, int]:
    explored_map_nodes = 0
    current = state
    for step_idx in range(config.advance_max_steps):
        if deadline is not None and time.monotonic() >= deadline:
            summary = _terminal_summary(
                current,
                extract_floor,
                extract_player_hp,
                did_reach_boss,
                accumulated_reward_score,
                config=config,
                reason="time_budget",
            )
            return float(summary["route_score"]), summary, 0, explored_map_nodes
        current_floor = int(extract_floor(current))
        if config.stop_floor is not None and current_floor >= int(config.stop_floor):
            summary = _terminal_summary(
                current,
                extract_floor,
                extract_player_hp,
                did_reach_boss,
                accumulated_reward_score,
                config=config,
                reason="floor_cap",
            )
            return float(summary["route_score"]), summary, 0, explored_map_nodes
        st = str(current.get("state_type") or "").strip().lower()
        if current.get("terminal") or did_reach_boss(current):
            summary = _terminal_summary(current, extract_floor, extract_player_hp, did_reach_boss, accumulated_reward_score, config=config)
            return float(summary["route_score"]), summary, 0, explored_map_nodes

        if st == "card_reward":
            action, reward_score = resolve_card_reward_choice(client, seed, current)
            current = apply_action(client, current, action)
            current = settle_after_choice(client, current, previous_state_type="card_reward", previous_floor=int(extract_floor(current)))
            accumulated_reward_score += float(reward_score)
            continue

        if st == "map":
            options = extract_map_options(current)
            if not options:
                summary = _terminal_summary(
                    current,
                    extract_floor,
                    extract_player_hp,
                    did_reach_boss,
                    accumulated_reward_score,
                    config=config,
                    reason="map_no_options",
                )
                return float(summary["route_score"]), summary, 0, explored_map_nodes
            explored_map_nodes += len(options)
            if depth_remaining <= 0:
                static_score = max(_static_route_score(current, option) for option in options)
                summary = _terminal_summary(
                    current,
                    extract_floor,
                    extract_player_hp,
                    did_reach_boss,
                    accumulated_reward_score + static_score,
                    config=config,
                    reason="map_depth_limit",
                )
                return float(summary["route_score"]), summary, len(options), explored_map_nodes

            ranked = sorted(options, key=lambda option: _static_route_score(current, option), reverse=True)
            ranked = ranked[: max(1, min(len(ranked), config.beam_width))]
            child_scores: list[tuple[float, dict[str, Any]]] = []
            with tempfile.TemporaryDirectory(prefix=f"sts2_map_tree_child_d{depth_remaining:02d}_") as tmpdir:
                snapshot = str(Path(tmpdir) / "map_child_snapshot.json")
                client.export_state(snapshot)
                for option in ranked:
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    base_state = client.import_state(snapshot)
                    branch_state = apply_action(client, base_state, option["action"])
                    branch_state = settle_after_choice(
                        client,
                        branch_state,
                        previous_state_type="map",
                        previous_floor=int(extract_floor(base_state)),
                    )
                    score, summary, _subchild_count, child_explored = _advance_route_value(
                        client=client,
                        seed=f"{seed}_d{depth_remaining}_i{int(option.get('index', 0))}",
                        state=branch_state,
                        depth_remaining=depth_remaining - 1,
                        config=config,
                        apply_action=apply_action,
                        settle_after_choice=settle_after_choice,
                        extract_floor=extract_floor,
                        extract_player_hp=extract_player_hp,
                        did_reach_boss=did_reach_boss,
                        choose_rollout_decision=choose_rollout_decision,
                        choose_deterministic_screen_action=choose_deterministic_screen_action,
                        resolve_card_reward_choice=resolve_card_reward_choice,
                        extract_map_options=extract_map_options,
                        use_local_ort_rollout=use_local_ort_rollout,
                        local_ort_max_combat_steps=local_ort_max_combat_steps,
                        accumulated_reward_score=accumulated_reward_score,
                        deadline=deadline,
                    )
                    explored_map_nodes += child_explored
                    child_scores.append((float(score), summary))
            if child_scores:
                best_score, best_summary = max(child_scores, key=lambda pair: pair[0])
                return best_score, best_summary, len(ranked), explored_map_nodes

        legal = current.get("legal_actions") or []
        deterministic = choose_deterministic_screen_action(current, legal)
        if deterministic is not None:
            current = apply_action(client, current, deterministic)
            current = settle_after_choice(client, current, previous_state_type=st, previous_floor=int(extract_floor(current)))
            continue

        if (
            st in {"monster", "elite", "boss", "combat"}
            and use_local_ort_rollout
            and getattr(client, "supports_local_ort", False)
        ):
            result = client.run_combat_local(max_steps=local_ort_max_combat_steps)
            post_state = result.get("state") if isinstance(result, dict) else None
            current = post_state if isinstance(post_state, dict) else client.get_state()
            continue

        if not legal:
            current = apply_action(client, current, {"action": "wait"})
            continue

        decision = choose_rollout_decision(current, legal, random.Random(f"{seed}:{step_idx}"))
        current = apply_action(client, current, decision.action)

    summary = _terminal_summary(current, extract_floor, extract_player_hp, did_reach_boss, accumulated_reward_score, config=config, reason="route_timeout")
    return float(summary["route_score"]), summary, 0, explored_map_nodes


def _terminal_summary(
    state: dict[str, Any],
    extract_floor: Callable[[dict[str, Any]], int],
    extract_player_hp: Callable[[dict[str, Any]], int],
    did_reach_boss: Callable[[dict[str, Any]], bool],
    accumulated_reward_score: float,
    *,
    config: MapRouteConfig,
    reason: str | None = None,
) -> dict[str, Any]:
    player = state.get("player") or {}
    hp_after = int(extract_player_hp(state))
    max_hp = max(1, int(player.get("max_hp") or 1))
    floor = int(extract_floor(state))
    boss_reached = bool(did_reach_boss(state))
    route_score = (
        (float(config.boss_bonus) if boss_reached else 0.0)
        + floor * float(config.floor_weight)
        + (hp_after / max_hp) * float(config.hp_weight)
        + float(accumulated_reward_score) * float(config.card_reward_weight)
    )
    return {
        "route_score": round(float(route_score), 4),
        "boss_reached": boss_reached,
        "end_floor": floor,
        "hp_after": hp_after,
        "max_hp": max_hp,
        "terminal_reason": reason or str(state.get("run_outcome") or state.get("state_type") or ""),
        "run_outcome": state.get("run_outcome"),
    }


def _static_route_score(state: dict[str, Any], option: dict[str, Any]) -> float:
    map_state = state.get("map") or {}
    nodes = map_state.get("nodes") or []
    player = state.get("player") or {}
    hp = float(player.get("hp") or 0.0)
    max_hp = max(1.0, float(player.get("max_hp") or 1.0))
    hp_ratio = hp / max_hp
    gold = float(player.get("gold") or 0.0)
    node_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        try:
            node_lookup[(int(node.get("col", 0)), int(node.get("row", 0)))] = node
        except Exception:
            continue

    def node_weight(node_type: str) -> float:
        norm = str(node_type or "").strip().lower()
        if norm == "boss":
            return 6.0
        if norm == "treasure":
            return 2.4
        if norm in {"event", "unknown"}:
            return 1.9
        if norm == "monster":
            return 1.4
        if norm == "shop":
            return 1.6 if gold >= 75 else 0.5
        if norm == "restsite":
            return 2.2 if hp_ratio < 0.5 else 0.9
        if norm == "elite":
            return 2.0 if hp_ratio >= 0.75 else 0.3
        return 0.8

    memo: dict[tuple[int, int], float] = {}

    def best_from(coord: tuple[int, int]) -> float:
        if coord in memo:
            return memo[coord]
        node = node_lookup.get(coord)
        if node is None:
            return 0.0
        children = node.get("children") or []
        child_values = []
        for child in children:
            if not isinstance(child, (list, tuple)) or len(child) != 2:
                continue
            child_coord = (int(child[0]), int(child[1]))
            child_node = node_lookup.get(child_coord) or {}
            child_values.append(node_weight(str(child_node.get("type") or "")) + best_from(child_coord))
        best = max(child_values) if child_values else 0.0
        memo[coord] = best
        return best

    try:
        coord = (int(option.get("col", 0)), int(option.get("row", 0)))
    except Exception:
        return 0.0
    option_type = str(option.get("type") or option.get("point_type") or "")
    return node_weight(option_type) + best_from(coord)
