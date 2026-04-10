"""MultiTurnSolverPlanner: cross-turn lookahead built on top of CombatTurnSolver.

================================================================================
RESEARCH-ONLY (frozen 2026-04-07, do NOT use in production)
================================================================================

This module is **research-only**. The production champion uses the single-turn
solver (`turn_solver_planner.py::TurnSolverPlanner`), NOT this multi-turn one.

Why research-only:
  - n=200 cohort: P4 (this file) = 2.5% act1_clear; P3 (single-turn) = 4.0%
  - Runtime: P4 = 20.92 s/game; P3 = 6.29 s/game (3.3× slower)
  - Per-boss bucket: P4 worse than P3 on ceremonial_beast (17.4% vs 34.8%);
    P4 finds 1 the_kin win that P3 misses → future per-boss expert signal,
    NOT a current production lever.
  - Bug 4 (real cause = env state pollution after lookahead loop) is fixed.
    "Line truncation" hypothesis was disproven by telemetry (P4 line len 3.73
    > P3 line len 3.16). See task3_bug4_forensic_20260407.md.

Score contract (R6 compliance, frozen 2026-04-07):
  - Full doc: docs/benchmarks/multi_turn_score_contract.md
  - Key invariant: cross_value = cand.line_components_score + γ * solution2.root_value
                                = components_only(turn1) + 0.9 * full_value(turn2)
  - DO NOT add leaf1 to turn 1 — leaf1 is an approximation of "value at
    start of turn 2", and solution2.root_value already computes that quantity
    explicitly. Adding both = double counting.

Hard gates to promote this file out of research (per multi_turn_score_contract.md §8):
  1. cross-turn act1_clear ≥ P3 (currently 2.5% < 4.0%)
  2. runtime < 2× P3 (currently 3.3×)
  3. no per-boss bucket strictly worse than P3 (ceremonial_beast fails)
  4. updated cohort CSV + sha256 in manifest
  5. canonical_eval.ps1 production command updated

Until ALL hard gates pass: production stays on TurnSolverPlanner (single-turn),
and any PR wiring this file into the canonical eval will be rejected per v2
Rule 6 ("无接口量纲说明，不准推进 multi-turn production 化").

================================================================================

The single-turn solver finds the best action LINE for the current turn. This
planner does a 2-turn lookahead at boss decision points:

  1. Run single-turn solver from the current state. Get top-K candidate first
     actions (by per_action_score) instead of only the single best.
  2. For each top-K candidate, simulate the FULL line that starts with that
     action (the solver returns a complete turn line). At end_turn, simulate
     the enemy's turn by calling step({"action": "end_turn"}).
  3. From the resulting "start of next turn" state, run the single-turn solver
     again to get its root_value (score of the best line for that next turn).
  4. The cross-turn score for the candidate (P4 fix, post double-counting):
        cross_value = cand.line_components_score + gamma * solution2.root_value
                    = components_only(turn1)     + 0.9 * full_value(turn2)
     NOT (cand.score + gamma * solution2.root_value), because cand.score
     includes leaf1 which is an approximation of solution2.root_value's start
     state. See multi_turn_score_contract.md §5.2 for the full justification.
  5. Pick the candidate with the best cross-turn score and return its first
     action; cache the rest of the line.

This is essentially "depth-2 alpha pruning at the turn level" using the
single-turn solver as the leaf evaluator. It addresses the failure mode where
the single-turn solver picks a locally optimal line that leaves the player in
a bad position for the next turn.
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from combat_nn import CombatPolicyValueNetwork
from combat_teacher_common import BaselineCombatPolicy
from rl_encoder_v2 import Vocab
from turn_solver_planner import (
    _CachedTurnPlan,
    _PipeEnvAdapter,
    _action_matches_legal,
    _heuristic_state_value,
    _state_signature,
    _TunedCombatTurnSolver,
)

logger = logging.getLogger("multi_turn_solver_planner")


class MultiTurnSolverPlanner:
    """2-turn lookahead planner driven by the existing single-turn solver."""

    def __init__(
        self,
        baseline_policy: BaselineCombatPolicy,
        *,
        mode: str = "boss",
        max_player_actions: int = 12,
        hp_loss_weight: float = 0.05,
        heuristic_blend_alpha: float = 0.5,
        topk_candidates: int = 3,
        gamma: float = 0.9,
        heuristic_enemy_hp_weight: float = 1.0,
        heuristic_use_absolute_hp: bool = False,
    ):
        self.baseline_policy = baseline_policy
        self._mode = mode
        self.max_player_actions = max_player_actions
        self.hp_loss_weight = hp_loss_weight
        self.heuristic_blend_alpha = heuristic_blend_alpha
        self.topk_candidates = max(1, int(topk_candidates))
        self.gamma = float(gamma)
        self.heuristic_enemy_hp_weight = heuristic_enemy_hp_weight
        self.heuristic_use_absolute_hp = heuristic_use_absolute_hp
        self._cache: _CachedTurnPlan | None = None
        # Telemetry
        self.calls = 0
        self.solver_calls = 0
        self.cache_hits = 0
        self.unsupported = 0
        self.fallbacks = 0
        self.errors = 0
        self.lookahead_attempts = 0
        self.lookahead_successes = 0
        # P4 forensic telemetry (Task 3): track line lengths and where they
        # come from so we can isolate Bug 4 (line truncation).
        self.forensic = {
            "base_solver_line_lengths": [],     # solution1.best_full_turn_line lengths (= P3 single-turn)
            "candidate_line_lengths": [],       # _solve_line_for_first_action returned line lengths
            "sub_solution_line_lengths": [],    # sub_solver.solve from mid_state line lengths (excludes first_action)
            "candidate_short_circuit_count": 0, # cand returned [first_action] only because mid_st not in combat
            "candidate_unsupported_count": 0,   # sub_solution unsupported -> [first_action]
            "candidate_supported_count": 0,     # sub_solution found a real continuation
            "best_chosen_line_lengths": [],     # length of the line that was actually chosen as best
            "topk_filter_dropped": 0,           # number of candidates dropped by topk filter (always 0 now since we union)
            "n_candidates_per_call": [],        # how many candidates we evaluated per call
            # Decision quality forensic
            "best_chosen_eq_p3_best": 0,        # times P4 picked the same first action as P3 base solver
            "best_chosen_neq_p3_best": 0,       # times P4 picked a DIFFERENT first action
            "p4_best_score_when_diff": [],      # cross_value of P4 chosen when != P3 best
            "p3_best_score_when_diff": [],      # cross_value of P3 best (computed by P4 if available)
            "score_p4_minus_p3_when_diff": [],  # how much "better" P4 thought its choice was
            # P4 line vs P3 line element-wise diff
            "p4_line_eq_p3_line": 0,            # times P4 cached line == P3 base line element-wise
            "p4_line_neq_p3_line": 0,           # times P4 cached line != P3 base line (post-first-action)
            "p4_line_diff_first_diverge_idx": [],  # at which index does P4 line first diverge from P3
        }

    def invalidate_cache(self) -> None:
        self._cache = None

    def _make_solver(self, env: _PipeEnvAdapter) -> _TunedCombatTurnSolver:
        return _TunedCombatTurnSolver(
            env=env,
            baseline_policy=self.baseline_policy,
            max_player_actions=self.max_player_actions,
            hp_loss_weight=self.hp_loss_weight,
            heuristic_blend_alpha=self.heuristic_blend_alpha,
            heuristic_enemy_hp_weight=self.heuristic_enemy_hp_weight,
            heuristic_use_absolute_hp=self.heuristic_use_absolute_hp,
        )

    def _try_cache(
        self,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
    ) -> tuple[int, str] | None:
        if self._cache is None or self._cache.is_exhausted():
            return None
        sig = _state_signature(state)
        if sig != self._cache.state_hash:
            self._cache = None
            return None
        next_action = self._cache.actions[self._cache.next_idx]
        for i, la in enumerate(legal):
            if _action_matches_legal(next_action, la):
                self._cache.next_idx += 1
                self.cache_hits += 1
                return i, "multi_turn_cache"
        self._cache = None
        return None

    def _replay_action_line(
        self,
        env: _PipeEnvAdapter,
        root_state_id: str,
        actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Replay an action line from the root state and return the final state."""
        env.load_state(root_state_id)
        state: dict[str, Any] = {}
        for action in actions:
            state = env.act(action)
            if not isinstance(state, dict):
                return {}
            st = (state.get("state_type") or "").lower()
            if state.get("terminal") or st == "game_over":
                return state
        return state

    def select_action(
        self,
        pipe_getter: Any,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
    ) -> tuple[int, str] | None:
        # P4 fix (2026-04-07): the previous formulation was
        #   cross_value = cand.score + gamma * future_value
        # where `cand.score = leaf1 + components1` (line.total_value at end of
        # turn 1) and `future_value = leaf2 + components2` (best line at end of
        # turn 2). This DOUBLE-COUNTED the start-of-turn-2 state value: leaf1
        # is an approximation of "value of being at start of turn 2", and
        # components2 + leaf2 is the actual computed value of playing turn 2
        # from that state. So leaf1 was being added on top of a value that
        # already included an approximation of itself.
        #
        # Corrected: drop leaf1, use only turn 1's components, and treat the
        # full turn 2 value as the discounted future term:
        #   cross_value = components1 + gamma * (components2 + leaf2)
        #              = cand.line_components_score + gamma * solution2.root_value
        #
        # combat_turn_solver now exposes `line_components_score` per
        # per_action_score entry.
        self.calls += 1
        # 1) Try cached plan first
        cached = self._try_cache(state, legal)
        if cached is not None:
            return cached

        try:
            raw_pipe = pipe_getter()
        except Exception as e:
            logger.debug("pipe_getter failed: %s", e)
            self.errors += 1
            return None
        env = _PipeEnvAdapter(raw_pipe)

        try:
            root_state_id = env.save_state()
        except Exception as e:
            logger.debug("save_state failed: %s", e)
            self.errors += 1
            return None
        if not root_state_id:
            self.errors += 1
            return None

        # 2) Run base solver to get per_action_score and best line
        solver1 = self._make_solver(env)
        try:
            solution1 = solver1.solve(state, root_state_id=root_state_id)
            self.solver_calls += 1
        except Exception as e:
            logger.debug("base solver failed: %s", e)
            self.errors += 1
            try:
                env.delete_state(root_state_id)
            except Exception:
                pass
            return None
        finally:
            try:
                solver1.cleanup()
            except Exception:
                pass

        if solution1 is None or not solution1.supported:
            self.unsupported += 1
            return None
        if not solution1.best_full_turn_line:
            self.fallbacks += 1
            return None

        # FORENSIC: record base solver line length (this is what single-turn
        # solver / P3 would return). Compare against candidate_line_lengths
        # below.
        self.forensic["base_solver_line_lengths"].append(len(solution1.best_full_turn_line))

        # Base solver's cleanup() deleted the root state. Re-save for lookahead.
        try:
            root_state_id = env.save_state()
        except Exception as e:
            logger.debug("re-save after base solver failed: %s", e)
            self.errors += 1
            return None
        if not root_state_id:
            self.errors += 1
            return None

        # 3) For each of the top-K candidates, look ahead.
        # P4 fix #2 (2026-04-07): the candidate ranking used to sort by
        # `score` (leaf + components), which is biased by the leaf eval.
        # The lookahead re-ranks by `line_components_score + gamma * future`,
        # so the topk filter must respect BOTH metrics. We take the union of
        # top-K by `score` and top-K by `line_components_score` to make sure
        # the lookahead gets a chance at all aggressive candidates.
        # Always include the base solver's best_first_action as candidate #0
        # to guarantee we never pick worse than the base solver.
        per_action = solution1.per_action_score or []
        supported = [a for a in per_action if a.get("supported", True)]
        scored_by_score = sorted(
            supported, key=lambda a: -float(a.get("score", -1e9)),
        )
        scored_by_components = sorted(
            supported, key=lambda a: -float(a.get("line_components_score", -1e9)),
        )
        # Union: take topk from each ranking, dedup by action signature.
        seen: set[str] = set()
        scored: list[dict[str, Any]] = []
        topk = max(1, int(self.topk_candidates))
        for src in (scored_by_score[:topk], scored_by_components[:topk]):
            for a in src:
                key = json.dumps(a.get("action") or {}, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                scored.append(a)
        # Promote best_first_action to position 0 if not already there
        best_first_action = solution1.best_first_action or {}
        if scored and best_first_action:
            best_first_idx = next(
                (i for i, a in enumerate(scored)
                 if _action_matches_legal(best_first_action, a.get("action") or {})),
                None,
            )
            if best_first_idx is not None and best_first_idx != 0:
                scored.insert(0, scored.pop(best_first_idx))

        # If only one candidate or no per-action scores, fall back to single-turn
        if len(scored) <= 1:
            best_first = solution1.best_full_turn_line[0]
            self._cache = _CachedTurnPlan(
                actions=list(solution1.best_full_turn_line),
                next_idx=0,
                state_hash=_state_signature(state),
            )
            for i, la in enumerate(legal):
                if _action_matches_legal(best_first, la):
                    self._cache.next_idx = 1
                    try:
                        env.delete_state(root_state_id)
                    except Exception:
                        pass
                    return i, "multi_turn_single"
            try:
                env.delete_state(root_state_id)
            except Exception:
                pass
            return None

        self.lookahead_attempts += 1
        self.forensic["n_candidates_per_call"].append(len(scored))
        # 4) Evaluate each candidate with a 1-turn lookahead.
        # P4 fix #2: `scored` is already the union of top-K by score AND
        # top-K by components (capped at 2*topk), so we iterate over ALL of
        # it. The previous `scored[:topk]` slice silently dropped the
        # components-sorted candidates.
        candidate_results: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
        for cand_idx, cand in enumerate(scored):
            cand_action = cand.get("action") or {}
            if not isinstance(cand_action, dict):
                continue
            # P4 fix: use components-only score for turn 1 (excludes leaf eval
            # at end of turn 1, which is double-counted by turn 2's full value).
            # Fall back to `score` if line_components_score is missing (old
            # solver versions).
            cand_components_only = float(
                cand.get("line_components_score", cand.get("score", 0.0))
            )
            # Keep cand_turn_score for the special-case branches below (game
            # over, combat ended cleanly) where future_value isn't computed.
            cand_turn_score = float(cand.get("score", 0.0))
            # We need the full line that STARTS with this candidate action.
            # Re-run solver with a forced first action.
            try:
                line, _ = self._solve_line_for_first_action(
                    env, root_state_id, state, cand_action,
                )
            except Exception as e:
                logger.debug("lookahead cand %d solve raised: %s", cand_idx, e)
                continue
            if line is None:
                logger.debug("lookahead cand %d returned None line", cand_idx)
                continue
            # FORENSIC: record candidate line length
            self.forensic["candidate_line_lengths"].append(len(line))
            # Replay the line to reach end_turn state
            try:
                end_state = self._replay_action_line(env, root_state_id, line)
            except Exception as e:
                logger.debug("replay failed: %s", e)
                continue
            if not isinstance(end_state, dict):
                continue
            end_st = (end_state.get("state_type") or "").lower()
            if end_state.get("terminal") or end_st == "game_over":
                # Game ended during this line — score with massive bonus/penalty
                if "victory" in (end_state.get("game_over") or {}).get("outcome", "").lower():
                    cross_value = 100.0  # huge bonus for actually winning
                else:
                    cross_value = -100.0  # huge penalty for dying
                candidate_results.append((cross_value, cand_action, line))
                continue
            if end_st not in {"combat", "monster", "elite", "boss"}:
                # Combat ended cleanly (won this combat)
                cross_value = cand_turn_score + 50.0
                candidate_results.append((cross_value, cand_action, line))
                continue

            # 5) From the next-turn state, run another solver to get root value
            try:
                next_state_id = env.save_state()
            except Exception:
                next_state_id = ""
            if not next_state_id:
                # Can't lookahead — just use the candidate's turn score
                candidate_results.append((cand_turn_score, cand_action, line))
                continue
            solver2 = self._make_solver(env)
            try:
                solution2 = solver2.solve(end_state, root_state_id=next_state_id)
                self.solver_calls += 1
            except Exception as e:
                logger.debug("lookahead solver failed: %s", e)
                solution2 = None
            finally:
                try:
                    solver2.cleanup()
                except Exception:
                    pass
                try:
                    env.delete_state(next_state_id)
                except Exception:
                    pass

            if solution2 is None or not solution2.supported:
                # No turn 2 lookahead — fall back to single-turn cand score.
                candidate_results.append((cand_turn_score, cand_action, line))
                continue
            future_value = float(solution2.root_value)
            # P4 fix: components_only(turn1) + gamma * full_value(turn2)
            cross_value = cand_components_only + self.gamma * future_value
            candidate_results.append((cross_value, cand_action, line))

        # P4 BUG 4 FIX (Task 3, 2026-04-07): the lookahead loop above executed
        # `env.load_state(root) → env.act(cand) → env.save_state(mid) →
        # sub_solver.solve(mid)` for each candidate. After the last candidate
        # the env is at some sub_solver leaf state, NOT at the root state. The
        # main game loop then calls `client.act(legal[idx])` based on the root
        # state's legal_actions, but the sim is in a different state, causing
        # the action to either fail or apply to the wrong state. This is the
        # actual cause of P4 multi-turn divergence (P3 single-turn doesn't hit
        # this because its DFS naturally converges and the final env state is
        # closer to root, AND because P3's `solver.solve` is the LAST sim
        # operation before return, so it doesn't trigger this multi-cand
        # accumulation).
        # Fix: explicitly load_state(root_state_id) before deleting it, so the
        # sim is back at the canonical root state when select_action returns.
        try:
            env.load_state(root_state_id)
        except Exception:
            pass
        try:
            env.delete_state(root_state_id)
        except Exception:
            pass

        if not candidate_results:
            # All lookahead attempts failed — fall back to base solver result
            best_first = solution1.best_full_turn_line[0]
            self._cache = _CachedTurnPlan(
                actions=list(solution1.best_full_turn_line),
                next_idx=0,
                state_hash=_state_signature(state),
            )
            for i, la in enumerate(legal):
                if _action_matches_legal(best_first, la):
                    self._cache.next_idx = 1
                    return i, "multi_turn_fallback"
            self.fallbacks += 1
            return None

        self.lookahead_successes += 1
        # 6) Pick best candidate, cache its line
        best = max(candidate_results, key=lambda r: r[0])
        best_value, best_first_action, best_line = best
        # FORENSIC: record the length of the line we actually chose
        self.forensic["best_chosen_line_lengths"].append(len(best_line))

        # FORENSIC: did P4 pick the same first action as P3 base solver?
        p3_best_first = solution1.best_first_action or {}
        p3_eq_p4 = _action_matches_legal(best_first_action, p3_best_first)
        if p3_eq_p4:
            self.forensic["best_chosen_eq_p3_best"] += 1
        else:
            self.forensic["best_chosen_neq_p3_best"] += 1
            # Find the P3 best's cross_value in candidate_results
            p3_cross = None
            for cv, ca, _ in candidate_results:
                if _action_matches_legal(ca, p3_best_first):
                    p3_cross = cv
                    break
            self.forensic["p4_best_score_when_diff"].append(float(best_value))
            if p3_cross is not None:
                self.forensic["p3_best_score_when_diff"].append(float(p3_cross))
                self.forensic["score_p4_minus_p3_when_diff"].append(float(best_value - p3_cross))

        # FORENSIC: when first action matches, is the REST of the line the same?
        # P3 line = solution1.best_full_turn_line
        # P4 line = best_line (from candidate_results)
        if p3_eq_p4:
            p3_line = list(solution1.best_full_turn_line or [])
            min_len = min(len(p3_line), len(best_line))
            divergence_idx = -1
            for k in range(min_len):
                if not _action_matches_legal(p3_line[k], best_line[k]):
                    divergence_idx = k
                    break
            if divergence_idx == -1 and len(p3_line) == len(best_line):
                self.forensic["p4_line_eq_p3_line"] += 1
            else:
                self.forensic["p4_line_neq_p3_line"] += 1
                if divergence_idx == -1:
                    divergence_idx = min_len  # length mismatch
                self.forensic["p4_line_diff_first_diverge_idx"].append(int(divergence_idx))
        self._cache = _CachedTurnPlan(
            actions=list(best_line),
            next_idx=0,
            state_hash=_state_signature(state),
        )
        for i, la in enumerate(legal):
            if _action_matches_legal(best_first_action, la):
                self._cache.next_idx = 1
                return i, "multi_turn"
        self.fallbacks += 1
        return None

    def _solve_line_for_first_action(
        self,
        env: _PipeEnvAdapter,
        root_state_id: str,
        state: dict[str, Any],
        first_action: dict[str, Any],
    ) -> tuple[list[dict[str, Any]] | None, float]:
        """Find the best action line that STARTS with first_action.

        Implementation: load root, play first_action, then run a fresh solver
        from the resulting state to find the best continuation. The full line
        is [first_action] + continuation.
        """
        env.load_state(root_state_id)
        try:
            mid_state = env.act(first_action)
        except Exception as e:
            logger.debug("first action act failed: %s", e)
            return None, -1e9
        if not isinstance(mid_state, dict):
            return None, -1e9
        mid_st = (mid_state.get("state_type") or "").lower()
        if mid_st not in {"combat", "monster", "elite", "boss"}:
            # First action ended the turn / combat — line is just [first_action]
            self.forensic["candidate_short_circuit_count"] += 1
            return [first_action], 0.0

        try:
            mid_state_id = env.save_state()
        except Exception:
            mid_state_id = ""
        if not mid_state_id:
            return None, -1e9

        sub_solver = self._make_solver(env)
        try:
            sub_solution = sub_solver.solve(mid_state, root_state_id=mid_state_id)
            self.solver_calls += 1
        except Exception as e:
            logger.debug("sub-solver failed: %s", e)
            return None, -1e9
        finally:
            try:
                sub_solver.cleanup()
            except Exception:
                pass
            try:
                env.delete_state(mid_state_id)
            except Exception:
                pass

        if sub_solution is None or not sub_solution.supported:
            self.forensic["candidate_unsupported_count"] += 1
            return [first_action], 0.0

        line = [first_action] + list(sub_solution.best_full_turn_line)
        line_value = float(sub_solution.root_value)
        # FORENSIC: record sub_solution line length (excluding first_action)
        self.forensic["sub_solution_line_lengths"].append(len(sub_solution.best_full_turn_line))
        self.forensic["candidate_supported_count"] += 1
        return line, line_value


def build_multi_turn_solver_planner(
    combat_net: CombatPolicyValueNetwork,
    vocab: Vocab,
    device: torch.device,
    *,
    mode: str = "boss",
    max_player_actions: int = 12,
    hp_loss_weight: float = 0.05,
    heuristic_blend_alpha: float = 0.5,
    topk_candidates: int = 3,
    gamma: float = 0.9,
    heuristic_enemy_hp_weight: float = 1.0,
    heuristic_use_absolute_hp: bool = False,
) -> MultiTurnSolverPlanner:
    baseline = BaselineCombatPolicy(network=combat_net, vocab=vocab, device=device)
    return MultiTurnSolverPlanner(
        baseline_policy=baseline,
        mode=mode,
        max_player_actions=max_player_actions,
        hp_loss_weight=hp_loss_weight,
        heuristic_blend_alpha=heuristic_blend_alpha,
        topk_candidates=topk_candidates,
        gamma=gamma,
        heuristic_enemy_hp_weight=heuristic_enemy_hp_weight,
        heuristic_use_absolute_hp=heuristic_use_absolute_hp,
    )
