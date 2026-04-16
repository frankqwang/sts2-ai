from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from search.combat_teacher_common import (
    COMBAT_STATE_TYPES,
    COMBAT_TURN_SOLUTION_SCHEMA_VERSION,
    aggregate_action_components,
    canonical_public_state_hash,
    combine_leaf_breakdown,
    compute_immediate_action_components,
    estimate_line_continuation_targets,
    immediate_expected_hp_loss_this_enemy_turn,
    is_action_supported_for_turn_solver,
    is_supported_solver_state,
    load_baseline_combat_policy,
    load_card_tags,
    sanitize_action,
    solver_support_diagnostics,
    unsupported_action_reason_for_turn_solver,
    _card_for_action,
    _card_slug,
)
from search.combat_turn_teacher_config import CombatTurnTeacherConfig
from ipc.full_run_env import create_full_run_client


class CombatTurnBranchEnv(Protocol):
    def save_state(self) -> str: ...

    def load_state(self, state_id: str) -> dict[str, Any]: ...

    def act(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def delete_state(self, state_id: str) -> bool: ...

    def clear_state_cache(self) -> bool: ...


@dataclass(slots=True)
class SolverSearchStats:
    nodes_expanded: int = 0
    cache_hits: int = 0
    evaluated_leaves: int = 0
    unsupported_branches: int = 0
    max_depth: int = 0
    search_mode: str = "dfs"
    beam_candidates: int = 0
    beam_pruned: int = 0
    max_nodes_exceeded: bool = False
    max_nodes_per_solve: int = 0


@dataclass(slots=True)
class SolverLineResult:
    total_value: float
    action_line: list[dict[str, Any]]
    component_totals: dict[str, float]
    leaf_state: dict[str, Any]
    leaf_baseline_value: float
    leaf_expected_hp_loss: float
    leaf_predicted_hp_loss_ratio: float = 0.0
    leaf_predicted_total_hp_loss: float = 0.0
    leaf_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CombatTurnActionEvaluation:
    action: dict[str, Any]
    score: float
    regret: float
    supported: bool = True
    component_score: float = 0.0
    leaf_score: float = 0.0
    expected_hp_loss: float = 0.0
    predicted_hp_loss_ratio: float = 0.0
    predicted_total_hp_loss: float = 0.0
    policy_prior: float = float("-inf")
    unsupported_reason: str = ""


@dataclass(slots=True)
class CombatTurnSolution:
    schema_version: str
    supported: bool
    unsupported_reason: str
    unsupported_details: dict[str, Any]
    best_first_action: dict[str, Any] | None
    best_full_turn_line: list[dict[str, Any]]
    per_action_score: list[dict[str, Any]]
    per_action_regret: list[dict[str, Any]]
    root_value: float
    continuation_targets: dict[str, float]
    leaf_breakdown: dict[str, float]
    search_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_LINE_SCORE_TIE_MARGIN = 1e-5
_POLICY_PRIOR_TIE_MARGIN = 1e-9


def _actions_are_semantically_equivalent(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> bool:
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if (first.get("action") or "") != (second.get("action") or ""):
        return False
    for key in ("label", "card_id", "screen_type", "node_type"):
        first_value = first.get(key)
        second_value = second.get(key)
        if first_value in (None, "") or second_value in (None, ""):
            continue
        if first_value != second_value:
            return False
    first_target = first.get("target_id") or first.get("target") or ""
    second_target = second.get("target_id") or second.get("target") or ""
    if first_target and second_target and first_target != second_target:
        return False
    return True


def _zero_components() -> dict[str, float]:
    return {
        "lethal_bonus": 0.0,
        "vulnerable_setup_bonus": 0.0,
        "body_slam_after_block_bonus": 0.0,
        "bad_end_turn_penalty": 0.0,
        "potion_waste_penalty": 0.0,
        "potion_cost": 0.0,
        # P0 Step 1 (2026-04-07 near-win analysis fix): rule-based bonuses /
        # penalties for play-pattern decisions. compute_immediate_action_components
        # in combat_teacher_common.py populates these.
        "power_card_early_bonus": 0.0,
        "damage_potion_clutch_bonus": 0.0,
        "x_cost_first_in_turn_bonus": 0.0,
        "early_game_defend_penalty": 0.0,
    }


def _component_score(components: dict[str, float]) -> float:
    return (
        float(components.get("lethal_bonus", 0.0))
        + float(components.get("vulnerable_setup_bonus", 0.0))
        + float(components.get("body_slam_after_block_bonus", 0.0))
        - float(components.get("bad_end_turn_penalty", 0.0))
        - float(components.get("potion_waste_penalty", 0.0))
        # P0 Step 1: signed-additive (compute_immediate_action_components bakes
        # the sign into the value).
        + float(components.get("power_card_early_bonus", 0.0))
        + float(components.get("damage_potion_clutch_bonus", 0.0))
        + float(components.get("x_cost_first_in_turn_bonus", 0.0))
        + float(components.get("early_game_defend_penalty", 0.0))
    )


def _is_terminal_combat_leaf(state: dict[str, Any]) -> bool:
    state_type = str(state.get("state_type") or "").strip().lower()
    return state_type not in COMBAT_STATE_TYPES or bool(state.get("terminal"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _player(state: dict[str, Any]) -> dict[str, Any]:
    player = state.get("player")
    return player if isinstance(player, dict) else {}


def _battle(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle")
    return battle if isinstance(battle, dict) else {}


def _alive_enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    enemies = state.get("enemies")
    if not isinstance(enemies, list):
        enemies = _battle(state).get("enemies")
    result: list[dict[str, Any]] = []
    if isinstance(enemies, list):
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
            if hp > 0 and not enemy.get("is_dead", False):
                result.append(enemy)
    return result


def _enemy_effective_hp_total(state: dict[str, Any]) -> float:
    total = 0.0
    for enemy in _alive_enemies(state):
        hp = _safe_float(enemy.get("hp", enemy.get("current_hp", 0)), 0.0)
        block = _safe_float(enemy.get("block", 0), 0.0)
        total += max(0.0, hp) + max(0.0, block)
    return total


def _player_hp_fraction(state: dict[str, Any]) -> float:
    player = _player(state)
    hp = _safe_float(player.get("current_hp", player.get("hp", 0)), 0.0)
    max_hp = max(1.0, _safe_float(player.get("max_hp", 1), 1.0))
    return max(0.0, min(1.0, hp / max_hp))


def _enemy_attack_pressure(state: dict[str, Any]) -> float:
    expected = float(immediate_expected_hp_loss_this_enemy_turn(state))
    player = _player(state)
    max_hp = max(1.0, _safe_float(player.get("max_hp", 1), 1.0))
    return max(0.0, min(1.5, expected / max_hp))


def _component_presence(value: float) -> float:
    if abs(float(value)) <= 1e-9:
        return 0.0
    return 1.0 if value > 0.0 else -1.0


class TacticalMechanismScorer:
    """Configurable full-turn leaf scorer used by the combat teacher."""

    def __init__(self, config: CombatTurnTeacherConfig):
        self.config = config

    def _mechanism_proxy(
        self,
        root_state: dict[str, Any],
        leaf_state: dict[str, Any],
        components: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        cfg = self.config
        room = cfg.room_kind(root_state)
        root_enemy_hp = _enemy_effective_hp_total(root_state)
        leaf_enemy_hp = _enemy_effective_hp_total(leaf_state)
        enemy_remaining_ratio = (leaf_enemy_hp / root_enemy_hp) if root_enemy_hp > 1e-6 else 0.0
        pressure = _enemy_attack_pressure(leaf_state)
        hp_frac = _player_hp_fraction(leaf_state)
        safety_gate = 1.0
        if hp_frac <= 0.35 or pressure >= 0.25:
            safety_gate = 0.35
        elif hp_frac <= 0.50 or pressure >= 0.16:
            safety_gate = 0.65
        if enemy_remaining_ratio <= 0.20:
            safety_gate *= 0.35

        power_raw = float(components.get("power_card_early_bonus", 0.0))
        if power_raw > 0.0:
            if room == "boss":
                power_score = float(cfg.power_early_bonus_boss)
            elif room == "elite":
                power_score = float(cfg.power_early_bonus_elite)
            else:
                power_score = 0.10
        else:
            power_score = 0.0

        vulnerable_score = float(cfg.vulnerable_setup_bonus) * max(
            0.0,
            _component_presence(float(components.get("vulnerable_setup_bonus", 0.0))),
        )
        x_cost_score = float(cfg.x_cost_first_bonus) * max(
            0.0,
            _component_presence(float(components.get("x_cost_first_in_turn_bonus", 0.0))),
        )
        body_slam_score = float(components.get("body_slam_after_block_bonus", 0.0))
        block_scaler_score = max(0.0, -float(components.get("early_game_defend_penalty", 0.0))) * 0.15
        attack_count_score = 0.0

        raw = power_score + vulnerable_score + x_cost_score + body_slam_score + block_scaler_score + attack_count_score
        score = float(cfg.mechanism_weight) * raw * safety_gate
        return score, {
            "mechanism_raw_score": float(raw),
            "mechanism_safety_gate": float(safety_gate),
            "mechanism_enemy_remaining_ratio": float(enemy_remaining_ratio),
            "mechanism_power_score": float(power_score),
            "mechanism_vulnerable_score": float(vulnerable_score),
            "mechanism_x_cost_score": float(x_cost_score),
            "mechanism_body_slam_score": float(body_slam_score),
            "mechanism_block_scaler_score": float(block_scaler_score),
            "mechanism_attack_count_score": float(attack_count_score),
        }

    def _rule_bonus(self, root_state: dict[str, Any], components: dict[str, float]) -> tuple[float, dict[str, float]]:
        cfg = self.config
        room = cfg.room_kind(root_state)
        lethal = float(cfg.lethal_bonus) * max(0.0, _component_presence(float(components.get("lethal_bonus", 0.0))))
        bad_end = -float(cfg.bad_end_turn_penalty) * max(
            0.0,
            _component_presence(float(components.get("bad_end_turn_penalty", 0.0))),
        )
        potion_waste = -float(cfg.potion_waste_penalty) * max(
            0.0,
            _component_presence(float(components.get("potion_waste_penalty", 0.0))),
        )
        vulnerable = float(cfg.vulnerable_setup_bonus) * max(
            0.0,
            _component_presence(float(components.get("vulnerable_setup_bonus", 0.0))),
        )
        if float(components.get("power_card_early_bonus", 0.0)) > 0.0:
            power = float(cfg.power_early_bonus_boss if room == "boss" else cfg.power_early_bonus_elite if room == "elite" else 0.10)
        else:
            power = 0.0
        x_cost = float(cfg.x_cost_first_bonus) * max(
            0.0,
            _component_presence(float(components.get("x_cost_first_in_turn_bonus", 0.0))),
        )
        early_defend = -float(cfg.early_defend_penalty) * max(
            0.0,
            -_component_presence(float(components.get("early_game_defend_penalty", 0.0))),
        )
        body_slam = float(components.get("body_slam_after_block_bonus", 0.0))
        damage_potion = float(components.get("damage_potion_clutch_bonus", 0.0))
        total = lethal + bad_end + potion_waste + vulnerable + power + x_cost + early_defend + body_slam + damage_potion
        return total, {
            "rule_lethal_bonus": float(lethal),
            "rule_bad_end_turn_penalty": float(bad_end),
            "rule_potion_waste_penalty": float(potion_waste),
            "rule_vulnerable_setup_bonus": float(vulnerable),
            "rule_power_early_bonus": float(power),
            "rule_x_cost_first_bonus": float(x_cost),
            "rule_early_defend_penalty": float(early_defend),
            "rule_body_slam_bonus": float(body_slam),
            "rule_damage_potion_clutch_bonus": float(damage_potion),
        }

    def score(
        self,
        root_state: dict[str, Any],
        leaf_state: dict[str, Any],
        components: dict[str, float],
        *,
        combat_net_value: float,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.config
        root_enemy_hp = _enemy_effective_hp_total(root_state)
        leaf_enemy_hp = _enemy_effective_hp_total(leaf_state)
        enemy_damage_progress = 1.0 if root_enemy_hp <= 1e-6 else (root_enemy_hp - leaf_enemy_hp) / root_enemy_hp
        enemy_damage_progress = max(-0.5, min(1.5, enemy_damage_progress))

        player = _player(leaf_state)
        max_hp = max(1.0, _safe_float(player.get("max_hp", 1), 1.0))
        expected_hp_loss = float(immediate_expected_hp_loss_this_enemy_turn(leaf_state))
        expected_hp_loss_ratio = max(0.0, min(2.0, expected_hp_loss / max_hp))

        tactical_score = (
            float(cfg.damage_progress_weight) * enemy_damage_progress
            - float(cfg.hp_loss_weight) * expected_hp_loss_ratio
        )
        mechanism_score, mechanism_breakdown = self._mechanism_proxy(root_state, leaf_state, components)

        hp_safety = _player_hp_fraction(leaf_state)
        pressure = _enemy_attack_pressure(leaf_state)
        active_power_value = max(0.0, float(components.get("power_card_early_bonus", 0.0))) * 0.25
        continuation_score = 0.10 * hp_safety - 0.12 * pressure + active_power_value

        rule_bonus, rule_breakdown = self._rule_bonus(root_state, components)
        terminal_bonus = 0.0
        if _is_terminal_combat_leaf(leaf_state):
            state_type = str(leaf_state.get("state_type") or "").strip().lower()
            terminal_bonus = -5.0 if state_type == "game_over" else 2.0

        total = tactical_score + mechanism_score + continuation_score + rule_bonus + terminal_bonus
        breakdown: dict[str, float] = {
            "combat_net_value": float(combat_net_value),
            "enemy_damage_progress": float(enemy_damage_progress),
            "expected_hp_loss_ratio": float(expected_hp_loss_ratio),
            "immediate_expected_hp_loss_this_enemy_turn": float(expected_hp_loss),
            "tactical_score": float(tactical_score),
            "mechanism_score": float(mechanism_score),
            "continuation_score": float(continuation_score),
            "rule_bonus": float(rule_bonus),
            "terminal_bonus": float(terminal_bonus),
            "total": float(total),
        }
        breakdown.update({str(key): float(value) for key, value in components.items()})
        breakdown.update(mechanism_breakdown)
        breakdown.update(rule_breakdown)
        return float(total), breakdown


class CombatTurnSolver:
    """Solve the best public-information line from now until end turn."""

    def __init__(
        self,
        env: CombatTurnBranchEnv,
        baseline_policy,
        *,
        max_player_actions: int = 12,
        card_tags: dict[str, list[str]] | None = None,
        teacher_config: CombatTurnTeacherConfig | None = None,
    ):
        self.env = env
        self.baseline_policy = baseline_policy
        self.max_player_actions = max_player_actions
        self.card_tags = card_tags or load_card_tags()
        self.teacher_config = teacher_config
        self.scorer = (
            TacticalMechanismScorer(teacher_config)
            if teacher_config is not None and str(teacher_config.leaf_eval_mode).strip().lower() == "tactical_mechanism_v1"
            else None
        )
        self.stats = SolverSearchStats()
        self._cache: dict[str, SolverLineResult] = {}
        self._created_state_ids: set[str] = set()
        self._current_root_state: dict[str, Any] | None = None
        self._current_max_player_actions = int(max_player_actions)

    def _remember_state_id(self, state_id: str | None) -> str | None:
        if state_id:
            self._created_state_ids.add(str(state_id))
        return state_id

    def _delete_state_id(self, state_id: str | None) -> None:
        if not state_id:
            return
        sid = str(state_id)
        if sid in self._created_state_ids:
            try:
                self.env.delete_state(sid)
            except Exception:
                pass
            self._created_state_ids.discard(sid)

    def cleanup(self) -> None:
        for state_id in list(self._created_state_ids):
            self._delete_state_id(state_id)

    def _policy_logits_for_state(self, state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[float]:
        if not legal_actions:
            return []
        try:
            baseline = self.baseline_policy.score(state, legal_actions)
        except Exception:
            return []
        logits = baseline.get("logits")
        if logits is None:
            return []
        try:
            return [float(item) for item in list(logits)]
        except Exception:
            return []

    def _active_search_mode(self) -> str:
        if self.teacher_config is None:
            return "dfs"
        return str(self.teacher_config.search_mode or "dfs").strip().lower()

    def _node_budget_exceeded(self) -> bool:
        if self.teacher_config is None:
            return False
        budget = int(self.teacher_config.max_nodes_per_solve or 0)
        if budget <= 0:
            return False
        exceeded = int(self.stats.nodes_expanded) >= budget
        if exceeded:
            self.stats.max_nodes_exceeded = True
        return exceeded

    def _clone_line(self, line: SolverLineResult) -> SolverLineResult:
        return SolverLineResult(
            total_value=line.total_value,
            action_line=list(line.action_line),
            component_totals=dict(line.component_totals),
            leaf_state=line.leaf_state,
            leaf_baseline_value=line.leaf_baseline_value,
            leaf_expected_hp_loss=line.leaf_expected_hp_loss,
            leaf_predicted_hp_loss_ratio=line.leaf_predicted_hp_loss_ratio,
            leaf_predicted_total_hp_loss=line.leaf_predicted_total_hp_loss,
            leaf_breakdown=dict(line.leaf_breakdown),
        )

    def _compose_line(
        self,
        action: dict[str, Any],
        components: dict[str, float],
        child: SolverLineResult,
    ) -> SolverLineResult:
        component_totals = aggregate_action_components([components, child.component_totals])
        action_line = [sanitize_action(action) or {}] + list(child.action_line)
        if self.scorer is not None and self._current_root_state is not None:
            score, breakdown = self.scorer.score(
                self._current_root_state,
                child.leaf_state,
                component_totals,
                combat_net_value=float(child.leaf_breakdown.get("combat_net_value", child.leaf_baseline_value)),
            )
            expected_hp_loss = float(breakdown.get("immediate_expected_hp_loss_this_enemy_turn", child.leaf_expected_hp_loss))
            return SolverLineResult(
                total_value=score,
                action_line=action_line,
                component_totals=component_totals,
                leaf_state=child.leaf_state,
                leaf_baseline_value=float(child.leaf_breakdown.get("combat_net_value", child.leaf_baseline_value)),
                leaf_expected_hp_loss=expected_hp_loss,
                leaf_predicted_hp_loss_ratio=child.leaf_predicted_hp_loss_ratio,
                leaf_predicted_total_hp_loss=child.leaf_predicted_total_hp_loss,
                leaf_breakdown=breakdown,
            )
        return SolverLineResult(
            total_value=child.total_value + _component_score(components),
            action_line=action_line,
            component_totals=component_totals,
            leaf_state=child.leaf_state,
            leaf_baseline_value=child.leaf_baseline_value,
            leaf_expected_hp_loss=child.leaf_expected_hp_loss,
            leaf_predicted_hp_loss_ratio=child.leaf_predicted_hp_loss_ratio,
            leaf_predicted_total_hp_loss=child.leaf_predicted_total_hp_loss,
            leaf_breakdown=dict(child.leaf_breakdown),
        )

    def _action_tags(self, state: dict[str, Any], action: dict[str, Any]) -> set[str]:
        card = _card_for_action(state, action)
        slug = _card_slug(card)
        return set(str(tag).strip().lower() for tag in (self.card_tags or {}).get(slug, []) if str(tag).strip())

    def _cheap_action_score(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        action: dict[str, Any],
        policy_prior: float,
    ) -> float:
        action_name = str(action.get("action") or "").strip().lower()
        score = 0.02 * policy_prior if math.isfinite(policy_prior) else 0.0
        if action_name == "end_turn":
            score += 0.15
        tags = self._action_tags(state, action)
        if "power" in tags or "scaling" in tags:
            score += 0.55
        if "vulnerable" in tags or "setup" in tags:
            score += 0.35
        if "draw" in tags or "energy" in tags or "x_cost" in tags:
            score += 0.30
        if "attack" in tags or "damage" in tags:
            score += 0.20
        if "block" in tags or "defend" in tags:
            pressure = _enemy_attack_pressure(state)
            score += 0.40 if pressure >= 0.12 else 0.05
        components = compute_immediate_action_components(state, legal_actions, action, None, card_tags=self.card_tags)
        if float(components.get("bad_end_turn_penalty", 0.0)) > 0.0:
            score -= 0.60
        if float(components.get("potion_waste_penalty", 0.0)) > 0.0:
            score -= 0.30
        return float(score)

    def _search_action_indices(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        policy_logits: list[float],
    ) -> list[int]:
        if self._active_search_mode() != "beam" or self.teacher_config is None:
            return list(range(len(legal_actions)))
        beam_width = int(self.teacher_config.beam_width_for_state(state))
        cap = max(beam_width, int(self.teacher_config.candidate_cap_per_state or beam_width))
        required: set[int] = set()
        scored: list[tuple[float, int]] = []
        for idx, action in enumerate(legal_actions):
            if not is_action_supported_for_turn_solver(state, action, card_tags=self.card_tags):
                continue
            action_name = str(action.get("action") or "").strip().lower()
            tags = self._action_tags(state, action)
            if action_name == "end_turn" or {"power", "scaling", "vulnerable", "setup", "draw", "energy", "x_cost"} & tags:
                required.add(idx)
            prior = float(policy_logits[idx]) if idx < len(policy_logits) else float("-inf")
            scored.append((self._cheap_action_score(state, legal_actions, action, prior), idx))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected: list[int] = []
        seen: set[int] = set()
        for idx in sorted(required):
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
        for _score, idx in scored:
            if len(selected) >= cap:
                break
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
        supported_count = len(scored)
        self.stats.beam_candidates += len(selected)
        self.stats.beam_pruned += max(0, supported_count - len(selected))
        return selected

    def _line_beats_best(
        self,
        candidate: SolverLineResult,
        best: SolverLineResult | None,
        *,
        candidate_policy_prior: float,
        best_policy_prior: float,
    ) -> bool:
        if best is None:
            return True
        if candidate.total_value > best.total_value + _LINE_SCORE_TIE_MARGIN:
            return True
        if best.total_value > candidate.total_value + _LINE_SCORE_TIE_MARGIN:
            return False
        candidate_first_action = candidate.action_line[0] if candidate.action_line else None
        best_first_action = best.action_line[0] if best.action_line else None
        if _actions_are_semantically_equivalent(candidate_first_action, best_first_action):
            if candidate_policy_prior > best_policy_prior + _POLICY_PRIOR_TIE_MARGIN:
                return True
            if best_policy_prior > candidate_policy_prior + _POLICY_PRIOR_TIE_MARGIN:
                return False
        if candidate.leaf_expected_hp_loss + _POLICY_PRIOR_TIE_MARGIN < best.leaf_expected_hp_loss:
            return True
        if best.leaf_expected_hp_loss + _POLICY_PRIOR_TIE_MARGIN < candidate.leaf_expected_hp_loss:
            return False
        candidate_component_score = _component_score(candidate.component_totals)
        best_component_score = _component_score(best.component_totals)
        if candidate_component_score > best_component_score + _POLICY_PRIOR_TIE_MARGIN:
            return True
        return False

    def _evaluate_leaf(self, state: dict[str, Any]) -> SolverLineResult:
        if _is_terminal_combat_leaf(state):
            state_type = str(state.get("state_type") or "").strip().lower()
            if state_type == "game_over":
                baseline_value = -1.0
                expected_hp_loss = 999.0
            else:
                baseline_value = 1.0
                expected_hp_loss = 0.0
            leaf_state = state
            total = baseline_value - 0.15 * expected_hp_loss
            breakdown = combine_leaf_breakdown(
                state,
                baseline_value=baseline_value,
                total_action_components=_zero_components(),
            )
            if self.scorer is not None and self._current_root_state is not None:
                total, breakdown = self.scorer.score(
                    self._current_root_state,
                    state,
                    _zero_components(),
                    combat_net_value=baseline_value,
                )
                expected_hp_loss = float(breakdown.get("immediate_expected_hp_loss_this_enemy_turn", expected_hp_loss))
            self.stats.evaluated_leaves += 1
            return SolverLineResult(
                total_value=total,
                action_line=[],
                component_totals=_zero_components(),
                leaf_state=leaf_state,
                leaf_baseline_value=baseline_value,
                leaf_expected_hp_loss=expected_hp_loss,
                leaf_predicted_hp_loss_ratio=0.0,
                leaf_predicted_total_hp_loss=0.0,
                leaf_breakdown={str(key): float(value) for key, value in breakdown.items()},
            )

        legal_actions = [action for action in state.get("legal_actions") or [] if isinstance(action, dict)]
        baseline = self.baseline_policy.score(state, legal_actions)
        baseline_value = float(baseline["value"])
        expected_hp_loss = float(
            combine_leaf_breakdown(
                state,
                baseline_value=baseline_value,
                total_action_components=_zero_components(),
            )["immediate_expected_hp_loss_this_enemy_turn"]
        )
        total = baseline_value - 0.15 * expected_hp_loss
        breakdown = combine_leaf_breakdown(
            state,
            baseline_value=baseline_value,
            total_action_components=_zero_components(),
        )
        if self.scorer is not None and self._current_root_state is not None:
            total, breakdown = self.scorer.score(
                self._current_root_state,
                state,
                _zero_components(),
                combat_net_value=baseline_value,
            )
            expected_hp_loss = float(breakdown.get("immediate_expected_hp_loss_this_enemy_turn", expected_hp_loss))
        self.stats.evaluated_leaves += 1
        return SolverLineResult(
            total_value=total,
            action_line=[],
            component_totals=_zero_components(),
            leaf_state=state,
            leaf_baseline_value=baseline_value,
            leaf_expected_hp_loss=expected_hp_loss,
            leaf_predicted_hp_loss_ratio=0.0,
            leaf_predicted_total_hp_loss=0.0,
            leaf_breakdown={str(key): float(value) for key, value in breakdown.items()},
        )

    def _transition_action(
        self,
        state_id: str,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        self.env.load_state(state_id)
        next_state = self.env.act(action)
        next_state_type = str(next_state.get("state_type") or "").strip().lower()
        if next_state_type in COMBAT_STATE_TYPES and is_supported_solver_state(next_state, card_tags=self.card_tags):
            next_state_id = self._remember_state_id(self.env.save_state())
            return next_state, next_state_id
        return next_state, None

    def _search(
        self,
        state: dict[str, Any],
        state_id: str,
        *,
        actions_used: int,
    ) -> SolverLineResult | None:
        self.stats.max_depth = max(self.stats.max_depth, actions_used)
        cache_key = canonical_public_state_hash(state, action_budget_used=actions_used)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.stats.cache_hits += 1
            return self._clone_line(cached)

        if actions_used >= self._current_max_player_actions or self._node_budget_exceeded():
            leaf = self._evaluate_leaf(state)
            self._cache[cache_key] = leaf
            return leaf

        if not is_supported_solver_state(state, card_tags=self.card_tags):
            return None

        legal_actions = [action for action in state.get("legal_actions") or [] if isinstance(action, dict)]
        policy_logits = self._policy_logits_for_state(state, legal_actions)
        self.stats.nodes_expanded += 1
        if self._node_budget_exceeded():
            leaf = self._evaluate_leaf(state)
            self._cache[cache_key] = leaf
            return leaf
        best: SolverLineResult | None = None
        best_policy_prior = float("-inf")
        for action_index in self._search_action_indices(state, legal_actions, policy_logits):
            action = legal_actions[action_index]
            if not is_action_supported_for_turn_solver(state, action, card_tags=self.card_tags):
                self.stats.unsupported_branches += 1
                continue

            action_name = str(action.get("action") or "").strip().lower()
            if action_name == "end_turn":
                components = compute_immediate_action_components(
                    state,
                    legal_actions,
                    action,
                    None,
                    card_tags=self.card_tags,
                )
                leaf = self._evaluate_leaf(state)
                candidate = self._compose_line(action, components, leaf)
            else:
                next_state_id: str | None = None
                try:
                    next_state, next_state_id = self._transition_action(state_id, action)
                except Exception:
                    self.stats.unsupported_branches += 1
                    continue
                components = compute_immediate_action_components(
                    state,
                    legal_actions,
                    action,
                    next_state,
                    card_tags=self.card_tags,
                )
                if _is_terminal_combat_leaf(next_state):
                    child = self._evaluate_leaf(next_state)
                elif next_state_id is not None:
                    child = self._search(next_state, next_state_id, actions_used=actions_used + 1)
                else:
                    child = None
                self._delete_state_id(next_state_id)
                if child is None:
                    self.stats.unsupported_branches += 1
                    continue
                candidate = self._compose_line(action, components, child)

            candidate_policy_prior = (
                float(policy_logits[action_index])
                if action_index < len(policy_logits)
                else float("-inf")
            )
            if self._line_beats_best(
                candidate,
                best,
                candidate_policy_prior=candidate_policy_prior,
                best_policy_prior=best_policy_prior,
            ):
                best = candidate
                best_policy_prior = candidate_policy_prior

        if best is None:
            return None

        cached_result = self._clone_line(best)
        self._cache[cache_key] = cached_result
        return best

    def solve(
        self,
        state: dict[str, Any],
        *,
        root_state_id: str | None = None,
    ) -> CombatTurnSolution:
        search_mode = self._active_search_mode()
        max_nodes = int(self.teacher_config.max_nodes_per_solve) if self.teacher_config is not None else 0
        self.stats = SolverSearchStats(search_mode=search_mode, max_nodes_per_solve=max_nodes)
        self._cache = {}
        self._current_root_state = state
        self._current_max_player_actions = (
            self.teacher_config.max_actions_for_state(state, fallback=self.max_player_actions)
            if self.teacher_config is not None
            else int(self.max_player_actions)
        )
        root_state_id = root_state_id or self.env.save_state()
        self._remember_state_id(root_state_id)
        unsupported_reason = ""
        try:
            support = solver_support_diagnostics(state, card_tags=self.card_tags)
            if not bool(support.get("supported", False)):
                unsupported_reason = str(support.get("reason") or "unsupported_state")
                return CombatTurnSolution(
                    schema_version=COMBAT_TURN_SOLUTION_SCHEMA_VERSION,
                    supported=False,
                    unsupported_reason=unsupported_reason,
                    unsupported_details=support,
                    best_first_action=None,
                    best_full_turn_line=[],
                    per_action_score=[],
                    per_action_regret=[],
                    root_value=0.0,
                    continuation_targets={},
                    leaf_breakdown={},
                    search_stats=asdict(self.stats),
                )

            root_legal = [action for action in state.get("legal_actions") or [] if isinstance(action, dict)]
            policy_logits = self._policy_logits_for_state(state, root_legal)
            evaluations: list[CombatTurnActionEvaluation] = []
            best_line: SolverLineResult | None = None
            best_policy_prior = float("-inf")
            root_action_indices = set(self._search_action_indices(state, root_legal, policy_logits))
            for action_index, action in enumerate(root_legal):
                if not is_action_supported_for_turn_solver(state, action, card_tags=self.card_tags):
                    action_reason = unsupported_action_reason_for_turn_solver(state, action, card_tags=self.card_tags) or "unsupported_action"
                    evaluations.append(
                        CombatTurnActionEvaluation(
                            action=sanitize_action(action) or {},
                            score=float("-inf"),
                            regret=float("inf"),
                            supported=False,
                            unsupported_reason=action_reason,
                        )
                    )
                    self.stats.unsupported_branches += 1
                    continue
                if action_index not in root_action_indices:
                    evaluations.append(
                        CombatTurnActionEvaluation(
                            action=sanitize_action(action) or {},
                            score=float("-inf"),
                            regret=float("inf"),
                            supported=False,
                            unsupported_reason="beam_pruned",
                            policy_prior=float(policy_logits[action_index]) if action_index < len(policy_logits) else float("-inf"),
                        )
                    )
                    continue

                action_name = str(action.get("action") or "").strip().lower()
                if action_name == "end_turn":
                    components = compute_immediate_action_components(
                        state,
                        root_legal,
                        action,
                        None,
                        card_tags=self.card_tags,
                    )
                    leaf = self._evaluate_leaf(state)
                    line = self._compose_line(action, components, leaf)
                else:
                    next_state_id: str | None = None
                    try:
                        next_state, next_state_id = self._transition_action(root_state_id, action)
                    except Exception:
                        evaluations.append(
                            CombatTurnActionEvaluation(
                                action=sanitize_action(action) or {},
                                score=float("-inf"),
                                regret=float("inf"),
                                supported=False,
                                unsupported_reason="transition_failed",
                            )
                        )
                        self.stats.unsupported_branches += 1
                        continue
                    components = compute_immediate_action_components(
                        state,
                        root_legal,
                        action,
                        next_state,
                        card_tags=self.card_tags,
                    )
                    if _is_terminal_combat_leaf(next_state):
                        child = self._evaluate_leaf(next_state)
                    elif next_state_id is not None:
                        child = self._search(next_state, next_state_id, actions_used=1)
                    else:
                        child = None
                    self._delete_state_id(next_state_id)
                    if child is None:
                        evaluations.append(
                            CombatTurnActionEvaluation(
                                action=sanitize_action(action) or {},
                                score=float("-inf"),
                                regret=float("inf"),
                                supported=False,
                                unsupported_reason="child_search_failed",
                            )
                        )
                        self.stats.unsupported_branches += 1
                        continue
                    line = self._compose_line(action, components, child)

                evaluations.append(
                    CombatTurnActionEvaluation(
                        action=sanitize_action(action) or {},
                        score=float(line.total_value),
                        regret=0.0,
                        supported=True,
                        component_score=float(_component_score(line.component_totals)),
                        leaf_score=float(line.leaf_baseline_value),
                        expected_hp_loss=float(line.leaf_expected_hp_loss),
                        predicted_hp_loss_ratio=float(line.leaf_predicted_hp_loss_ratio),
                        predicted_total_hp_loss=float(line.leaf_predicted_total_hp_loss),
                        policy_prior=float(policy_logits[action_index]) if action_index < len(policy_logits) else float("-inf"),
                    )
                )
                candidate_policy_prior = (
                    float(policy_logits[action_index])
                    if action_index < len(policy_logits)
                    else float("-inf")
                )
                if self._line_beats_best(
                    line,
                    best_line,
                    candidate_policy_prior=candidate_policy_prior,
                    best_policy_prior=best_policy_prior,
                ):
                    best_line = line
                    best_policy_prior = candidate_policy_prior

            if best_line is None:
                unsupported_reason = "no_supported_actions"
                return CombatTurnSolution(
                    schema_version=COMBAT_TURN_SOLUTION_SCHEMA_VERSION,
                    supported=False,
                    unsupported_reason=unsupported_reason,
                    unsupported_details={
                        "supported": False,
                        "reason": unsupported_reason,
                        "state_type": str(state.get("state_type") or "").strip().lower(),
                        "enabled_legal_action_count": len(root_legal),
                        "unsupported_action_count": sum(0 if item.supported else 1 for item in evaluations),
                        "unsupported_action_reasons": [
                            {
                                "reason": item.unsupported_reason,
                                "action": item.action,
                            }
                            for item in evaluations
                            if not item.supported
                        ],
                    },
                    best_first_action=None,
                    best_full_turn_line=[],
                    per_action_score=[asdict(item) for item in evaluations],
                    per_action_regret=[asdict(item) for item in evaluations],
                    root_value=0.0,
                    continuation_targets={},
                    leaf_breakdown={},
                    search_stats=asdict(self.stats),
                )

            best_score = best_line.total_value
            for item in evaluations:
                if math.isfinite(item.score):
                    item.regret = max(0.0, float(best_score - item.score))

            leaf_breakdown = dict(best_line.leaf_breakdown) if best_line.leaf_breakdown else combine_leaf_breakdown(
                best_line.leaf_state,
                baseline_value=best_line.leaf_baseline_value,
                total_action_components=best_line.component_totals,
            )
            continuation_baseline_value = float(leaf_breakdown.get("combat_net_value", best_line.leaf_baseline_value))
            continuation_targets = estimate_line_continuation_targets(
                terminal_state=best_line.leaf_state,
                baseline_value=continuation_baseline_value,
                total_potions_used=int(round(best_line.component_totals.get("potion_cost", 0.0))),
            )
            return CombatTurnSolution(
                schema_version=COMBAT_TURN_SOLUTION_SCHEMA_VERSION,
                supported=True,
                unsupported_reason="",
                unsupported_details={},
                best_first_action=best_line.action_line[0] if best_line.action_line else None,
                best_full_turn_line=best_line.action_line,
                per_action_score=[
                    {
                        "action": item.action,
                        "score": item.score,
                        "total_score": item.score,
                        "component_score": item.component_score,
                        "leaf_score": item.leaf_score,
                        "expected_hp_loss": item.expected_hp_loss,
                        "predicted_hp_loss_ratio": item.predicted_hp_loss_ratio,
                        "predicted_total_hp_loss": item.predicted_total_hp_loss,
                        "policy_prior": item.policy_prior,
                        "supported": item.supported,
                        "unsupported_reason": item.unsupported_reason,
                    }
                    for item in evaluations
                ],
                per_action_regret=[
                    {
                        "action": item.action,
                        "regret": item.regret,
                        "supported": item.supported,
                    }
                    for item in evaluations
                ],
                root_value=float(best_score),
                continuation_targets=continuation_targets,
                leaf_breakdown=leaf_breakdown,
                search_stats=asdict(self.stats),
            )
        finally:
            try:
                self.env.load_state(root_state_id)
            except Exception:
                pass
            self.cleanup()


def _solve_current_state_from_env(args: argparse.Namespace) -> dict[str, Any]:
    client = create_full_run_client(
        use_pipe=True,
        transport=args.transport,
        port=args.port,
        ready_timeout_s=30.0,
        request_timeout_s=30.0,
    )
    try:
        if args.seed:
            state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=args.seed, timeout_s=30.0)
            max_steps = max(1, int(args.sample_limit))
            for _ in range(max_steps):
                if str(state.get("state_type") or "").strip().lower() in COMBAT_STATE_TYPES:
                    break
                legal = [action for action in state.get("legal_actions") or [] if isinstance(action, dict)]
                if not legal:
                    state = client.get_state()
                    continue
                state = client.act(legal[0])
        else:
            state = client.get_state()

        root_state_id = client.save_state()
        baseline = load_baseline_combat_policy(args.combat_checkpoint)
        solver = CombatTurnSolver(
            client,
            baseline,
            max_player_actions=args.max_player_actions,
        )
        solution = solver.solve(state, root_state_id=root_state_id)
        return {
            "state_type": str(state.get("state_type") or "").strip().lower(),
            "floor": int(((state.get("run") or {}).get("floor") or 0)),
            "solution": solution.to_dict(),
        }
    finally:
        try:
            client.clear_state_cache()
        except Exception:
            pass
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Combat end-of-turn solver V1")
    parser.add_argument("--combat-checkpoint", required=True, help="Combat baseline checkpoint used for value/policy")
    parser.add_argument("--transport", default="pipe-binary", help="pureSim transport (default: pipe-binary)")
    parser.add_argument("--port", type=int, default=15527, help="PureSim port")
    parser.add_argument("--seed", default=None, help="Optional seed to auto-drive until first combat state")
    parser.add_argument("--sample-limit", type=int, default=200, help="Max auto-drive actions before giving up on --seed mode")
    parser.add_argument("--max-player-actions", type=int, default=12, help="Max player actions before forced leaf evaluation")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    payload = _solve_current_state_from_env(args)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
