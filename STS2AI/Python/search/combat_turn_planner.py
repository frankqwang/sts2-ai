#!/usr/bin/env python3
"""Policy-guided turn-level combat planner v1.

Searches complete turn sequences (all actions from turn start to end_turn),
not individual card plays. Uses current combat NN policy as proposal
distribution and evaluates afterstates (next player decision point).

Architecture:
  - TurnPlanner: top-level controller with plan caching
  - PolicyBeamCandidateGenerator: beam search over complete turn sequences
  - AfterstateEvaluator: abstract base for scoring afterstates
  - RolloutCombatEvaluator: rollout to combat end for ground-truth scoring
  - ActionSequence / CandidateAfterstate: data structures

Usage in evaluate_ai.py:
  --combat-turn-planner --planner-mode boss_elite
"""
from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActionSequence:
    """A complete turn sequence with metadata."""
    actions: list[dict] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    total_logp: float = 0.0
    is_complete: bool = False  # reached end_turn / terminal / no legal

    def normalized_logp(self) -> float:
        n = max(len(self.log_probs), 1)
        return self.total_logp / n


@dataclass
class CandidateAfterstate:
    """A candidate turn sequence with its afterstate evaluation."""
    sequence: ActionSequence
    afterstate: dict | None = None
    is_terminal: bool = False
    player_won: bool = False
    # Evaluator outputs
    win_prob: float = 0.5
    hp_loss: float = 0.0
    utility: float = 0.0
    # Combined score (utility + prior bonus)
    final_score: float = 0.0


@dataclass
class PlannedTurnCache:
    """Cached plan for the current turn."""
    sequence: ActionSequence
    next_action_idx: int = 0
    root_state_hash: str = ""

    def next_action(self) -> dict | None:
        if self.next_action_idx < len(self.sequence.actions):
            action = self.sequence.actions[self.next_action_idx]
            self.next_action_idx += 1
            return action
        return None

    def is_exhausted(self) -> bool:
        return self.next_action_idx >= len(self.sequence.actions)


# ---------------------------------------------------------------------------
# Evaluator interface
# ---------------------------------------------------------------------------

class AfterstateEvaluator(ABC):
    """Abstract base for scoring combat afterstates."""

    @abstractmethod
    def evaluate(
        self,
        afterstate: dict[str, Any],
        is_terminal: bool,
        player_won: bool,
    ) -> tuple[float, float, float]:
        """Evaluate an afterstate.

        Returns:
            (win_prob, hp_loss, utility)
        """
        ...


class RolloutCombatEvaluator(AfterstateEvaluator):
    """Evaluate afterstate by rolling out to combat end with direct policy.

    For v1 this uses heuristic scoring from the afterstate directly,
    without actual rollout (which would require keeping the FM alive).
    """

    def __init__(self, hp_loss_weight: float = 0.2):
        self.hp_loss_weight = hp_loss_weight

    def evaluate(
        self,
        afterstate: dict[str, Any],
        is_terminal: bool,
        player_won: bool,
    ) -> tuple[float, float, float]:
        if is_terminal:
            if player_won:
                return 1.0, 0.0, 1.0
            else:
                return 0.0, 1.0, -1.0

        # Extract state info for heuristic scoring
        player = afterstate.get("player", {}) or {}
        hp = int(player.get("hp", player.get("current_hp", 0)))
        max_hp = max(int(player.get("max_hp", 80)), 1)
        block = int(player.get("block", 0))

        battle = afterstate.get("battle", {}) or {}
        enemies = afterstate.get("enemies", []) or battle.get("enemies", [])
        alive_enemies = [
            e for e in enemies
            if isinstance(e, dict) and int(e.get("hp", e.get("current_hp", 1))) > 0
        ]

        # Heuristic scoring
        total_enemy_hp = sum(
            int(e.get("hp", e.get("current_hp", 0))) for e in alive_enemies
        )
        total_enemy_max_hp = max(
            sum(int(e.get("max_hp", 100)) for e in alive_enemies), 1
        )

        # Win proxy: how much enemy HP we've dealt
        enemy_damage_ratio = 1.0 - (total_enemy_hp / total_enemy_max_hp)

        # Survival proxy
        hp_ratio = hp / max_hp
        effective_hp = (hp + block) / max_hp

        # No enemies left = won this combat
        if not alive_enemies:
            return 1.0, max(0, 1.0 - hp_ratio), 1.0 + hp_ratio

        win_prob = min(0.95, enemy_damage_ratio * 0.7 + effective_hp * 0.3)
        hp_loss = max(0.0, 1.0 - hp_ratio)
        utility = (
            enemy_damage_ratio * 0.6
            + effective_hp * 0.3
            - hp_loss * self.hp_loss_weight
        )

        return win_prob, hp_loss, utility


# ---------------------------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------------------------

@dataclass
class BeamConfig:
    """Beam search configuration."""
    beam_width: int = 8
    expand_topk: int = 5
    max_turn_steps: int = 20
    max_final_sequences: int = 8
    prior_bonus: float = 0.05
    step_budget: int = 200


class PolicyBeamCandidateGenerator:
    """Generate candidate turn sequences via policy-guided beam search."""

    def __init__(
        self,
        combat_net: Any,
        vocab: Any,
        device: torch.device,
        config: BeamConfig | None = None,
    ):
        self.combat_net = combat_net
        self.vocab = vocab
        self.device = device
        self.config = config or BeamConfig()

    def _get_policy_logprobs(
        self,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
    ) -> np.ndarray:
        """Get log-probabilities from combat policy for legal actions."""
        from combat_nn import build_combat_features, build_combat_action_features

        sf = build_combat_features(state, self.vocab)
        af = build_combat_action_features(state, legal, self.vocab)

        sf_t: dict[str, torch.Tensor] = {}
        af_t: dict[str, torch.Tensor] = {}
        for k, v in sf.items():
            if not isinstance(v, np.ndarray):
                continue
            t = torch.tensor(v).unsqueeze(0)
            sf_t[k] = t.bool() if v.dtype == bool else (t.long() if v.dtype in (np.int64, np.int32) else t.float())
            sf_t[k] = sf_t[k].to(self.device)
        for k, v in af.items():
            if not isinstance(v, np.ndarray):
                continue
            t = torch.tensor(v).unsqueeze(0)
            af_t[k] = t.bool() if v.dtype == bool else (t.long() if v.dtype in (np.int64, np.int32) else t.float())
            af_t[k] = af_t[k].to(self.device)

        with torch.no_grad():
            logits, _ = self.combat_net(sf_t, af_t)

        mask = af_t["action_mask"].float().squeeze(0)
        logits = logits.squeeze(0) + (1.0 - mask) * (-1e9)
        logits = logits[:len(legal)]
        log_probs = torch.log_softmax(logits, dim=0).cpu().numpy()
        return log_probs

    def generate(
        self,
        fm: Any,  # PipeCombatForwardModel
    ) -> list[ActionSequence]:
        """Generate top-K complete turn sequences via beam search.

        Args:
            fm: Forward model at turn start (will be cloned internally)

        Returns:
            List of complete ActionSequence sorted by total_logp
        """
        cfg = self.config
        root_state = fm.get_state_dict()
        root_legal = fm.get_legal_actions()

        if not root_legal:
            return []

        # Initial log probs
        root_logp = self._get_policy_logprobs(root_state, root_legal)

        # Beam: list of (partial_sequence, clone_needed, last_state, last_legal)
        # We DON'T clone here — we'll clone when expanding
        beam: list[tuple[ActionSequence, dict, list]] = []

        # Seed beam with top-k root actions
        topk_indices = np.argsort(root_logp)[-cfg.expand_topk:][::-1]
        for idx in topk_indices:
            if idx >= len(root_legal):
                continue
            action = root_legal[idx]
            seq = ActionSequence(
                actions=[action],
                log_probs=[float(root_logp[idx])],
                total_logp=float(root_logp[idx]),
            )
            # Check if this is already end_turn
            if (action.get("action") or "").lower() == "end_turn":
                seq.is_complete = True
            beam.append((seq, root_state, root_legal))

        # Expand beam iteratively
        completed: list[ActionSequence] = []

        for step in range(cfg.max_turn_steps):
            if not beam:
                break

            next_beam: list[tuple[ActionSequence, dict, list]] = []

            for seq, _, _ in beam:
                if seq.is_complete:
                    completed.append(seq)
                    continue

                # Clone FM and replay this sequence
                clone = fm.clone()
                replay_ok = True
                for action in seq.actions:
                    try:
                        clone.step(action)
                    except Exception:
                        replay_ok = False
                        break

                if not replay_ok or clone.is_terminal:
                    seq.is_complete = True
                    completed.append(seq)
                    continue

                child_state = clone.get_state_dict()
                child_legal = clone.get_legal_actions()
                st = (child_state.get("state_type") or "").lower()

                if not child_legal or st not in COMBAT_SCREENS:
                    seq.is_complete = True
                    completed.append(seq)
                    continue

                # Get policy for next expansion
                try:
                    child_logp = self._get_policy_logprobs(child_state, child_legal)
                except Exception:
                    seq.is_complete = True
                    completed.append(seq)
                    continue

                # Expand top-k children
                topk = np.argsort(child_logp)[-cfg.expand_topk:][::-1]
                for cidx in topk:
                    if cidx >= len(child_legal):
                        continue
                    child_action = child_legal[cidx]
                    new_seq = ActionSequence(
                        actions=seq.actions + [child_action],
                        log_probs=seq.log_probs + [float(child_logp[cidx])],
                        total_logp=seq.total_logp + float(child_logp[cidx]),
                    )
                    if (child_action.get("action") or "").lower() == "end_turn":
                        new_seq.is_complete = True
                    next_beam.append((new_seq, child_state, child_legal))

            # Keep top beam_width by total_logp
            all_candidates = [(s, st, le) for s, st, le in next_beam if not s.is_complete]
            all_candidates.sort(key=lambda x: x[0].total_logp, reverse=True)
            beam = all_candidates[:cfg.beam_width]

            # Collect completed from this step
            for s, _, _ in next_beam:
                if s.is_complete:
                    completed.append(s)

            if len(completed) >= cfg.max_final_sequences:
                break

        # Add remaining beam as completed (truncated)
        for seq, _, _ in beam:
            seq.is_complete = True
            completed.append(seq)

        # Sort by total_logp and return top-K
        completed.sort(key=lambda s: s.total_logp, reverse=True)
        return completed[:cfg.max_final_sequences]


# ---------------------------------------------------------------------------
# Turn Planner
# ---------------------------------------------------------------------------

class TurnPlanner:
    """Top-level turn planner with plan caching."""

    def __init__(
        self,
        candidate_generator: PolicyBeamCandidateGenerator,
        evaluator: AfterstateEvaluator,
        config: BeamConfig | None = None,
    ):
        self.generator = candidate_generator
        self.evaluator = evaluator
        self.config = config or BeamConfig()
        self._cache: PlannedTurnCache | None = None

    def _state_hash(self, state: dict) -> str:
        """Quick hash for cache invalidation."""
        p = state.get("player", {}) or {}
        r = state.get("round_number") or (state.get("battle", {}) or {}).get("round_number", 0)
        hp = p.get("hp", p.get("current_hp", 0))
        return f"r{r}_hp{hp}"

    def plan_turn(
        self,
        pipe_getter: Any,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
    ) -> CandidateAfterstate | None:
        """Plan a complete turn and cache the result.

        Returns best candidate or None if planning fails.
        """
        from combat_mcts_agent import PipeCombatForwardModel

        fm = None
        try:
            fm = PipeCombatForwardModel.from_current_state(
                pipe_getter, max_step_budget=self.config.step_budget,
            )

            # Generate candidate sequences
            candidates_seqs = self.generator.generate(fm)
            if not candidates_seqs:
                return None

            # Evaluate each candidate's afterstate
            candidates: list[CandidateAfterstate] = []
            for seq in candidates_seqs:
                clone = fm.clone()
                try:
                    for action in seq.actions:
                        if clone.is_terminal:
                            break
                        clone.step(action)

                    afterstate = clone.get_state_dict()
                    is_term = clone.is_terminal
                    won = clone.player_won

                    win_prob, hp_loss, utility = self.evaluator.evaluate(
                        afterstate, is_term, won,
                    )

                    final_score = utility + self.config.prior_bonus * seq.normalized_logp()

                    candidates.append(CandidateAfterstate(
                        sequence=seq,
                        afterstate=afterstate,
                        is_terminal=is_term,
                        player_won=won,
                        win_prob=win_prob,
                        hp_loss=hp_loss,
                        utility=utility,
                        final_score=final_score,
                    ))
                except Exception as e:
                    logger.debug("Candidate eval failed: %s", e)
                    continue

            if not candidates:
                return None

            # Select best
            best = max(candidates, key=lambda c: c.final_score)

            # Cache the plan
            self._cache = PlannedTurnCache(
                sequence=best.sequence,
                next_action_idx=0,
                root_state_hash=self._state_hash(state),
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Planned turn: %d candidates, best utility=%.3f logp=%.2f actions=%d",
                    len(candidates), best.utility, best.sequence.total_logp,
                    len(best.sequence.actions),
                )

            return best

        except Exception as e:
            logger.debug("Turn planning failed: %s", e)
            return None
        finally:
            if fm is not None:
                try:
                    restored = fm.cleanup_and_restore()
                    if restored is None:
                        fm.cleanup()
                except Exception:
                    try:
                        fm.cleanup()
                    except Exception:
                        pass

    def get_cached_action(
        self,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
    ) -> tuple[int, str] | None:
        """Get next action from cached plan if still valid.

        Returns (action_idx, source) or None if cache is invalid.
        """
        if self._cache is None or self._cache.is_exhausted():
            return None

        # Check cache validity
        current_hash = self._state_hash(state)
        if current_hash != self._cache.root_state_hash:
            # State changed (e.g., unexpected damage), invalidate
            self._cache = None
            return None

        next_action = self._cache.next_action()
        if next_action is None:
            return None

        # Match to current legal actions
        for i, la in enumerate(legal):
            if (la.get("action") == next_action.get("action") and
                la.get("label", "") == next_action.get("label", "")):
                return i, "planner_cached"

        # Action no longer legal, invalidate cache
        self._cache = None
        return None

    def select_action(
        self,
        pipe_getter: Any,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
    ) -> tuple[int, str] | None:
        """Select action: use cache if valid, otherwise plan new turn.

        Returns (action_idx, source) or None to fall back to direct policy.
        """
        # Try cached plan first
        cached = self.get_cached_action(state, legal)
        if cached is not None:
            return cached

        # Plan new turn
        best = self.plan_turn(pipe_getter, state, legal)
        if best is None:
            return None

        # Return first action from plan
        cached = self.get_cached_action(state, legal)
        if cached is not None:
            return cached

        return None

    def invalidate_cache(self) -> None:
        """Force cache invalidation."""
        self._cache = None
