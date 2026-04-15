"""Shared full-run agent wrapper.

Purpose: collapse the action-selection logic that used to live in three places
(evaluate_ai.py / train_hybrid rollout / build_act1_combat_teacher_v2_dataset.py)
into one class, so downstream tools all play the game the same way. Concretely:

  - auto_progress (UI prompts: confirm/skip/claim)     ← rule
  - RepeatLoopTracker (escape argmax dead-loops)       ← rule
  - _select_action_nn (combat/noncombat NN + rerank)   ← NN

Before this wrapper, the live builder (search/build_act1_combat_teacher_v2_dataset.py)
reimplemented only a stripped-down NN argmax path, which caused most seeds to
stall at floor 2-6 on noncombat screens. evaluate_ai used the fuller pipeline
and reached boss fine. Sharing through `FullRunAgent` means there's literally
one place to change behaviour.

Usage:
    from full_run_agent import FullRunAgent, AgentConfig, load_agent_config

    cfg = load_agent_config("STS2AI/Python/configs/inference_config.toml")
    agent = FullRunAgent(
        ppo_net=ppo_net,
        combat_net=combat_net,
        vocab=vocab,
        device=device,
        cfg=cfg,
    )

    state = client.reset(...)
    while not state.get("terminal"):
        legal = [a for a in state.get("legal_actions", []) if a.get("is_enabled") is not False]
        pick = agent.select_action(state, legal)
        next_state = client.act(pick["action"])
        agent.after_step(before_state=state, before_legal=legal, action=pick["action"], next_state=next_state)
        state = next_state

This file deliberately does NOT reimplement the auto_progress / NN / rerank
logic. It imports them from evaluate_ai so there's exactly one copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Late imports to avoid circular dependencies. evaluate_ai.py pulls in heavy
# modules; by importing at call-time we keep this file light to import.


@dataclass
class AgentConfig:
    """Canonical inference config. Mirrors STS2AI/Python/configs/inference_config.toml.

    Downstream scripts should build this via `load_agent_config()` so they all
    agree on defaults.
    """

    # Combat decision path
    combat_safety_rerank: bool = True
    combat_safety_rerank_weight: float = 1.0
    combat_unsafe_mask: bool = False
    lethal_probe: bool = False
    beam_search: int = 0
    combat_turn_solver: bool = False
    combat_turn_planner: bool = False
    turn_solver_mode: str = "boss_elite"
    turn_planner_mode: str = "boss_elite"
    combat_teacher_rerank: bool = False
    combat_teacher_rerank_weight: float = 0.5

    # Non-combat helpers
    auto_progress: bool = True
    loop_escape: bool = True
    boss_conditioned_card_guidance: bool = True
    boss_conditioned_card_guidance_weight: float = 0.8

    # Sampling
    deterministic_policy: bool = True

    # Route policy (passed into _select_action_nn as act1_route_mode)
    act1_route_mode: str = "conservative"

    extra: dict[str, Any] = field(default_factory=dict)


def load_agent_config(path: str | Path | None) -> AgentConfig:
    """Load AgentConfig from the canonical TOML. Unknown TOML keys are ignored."""
    if not path:
        return AgentConfig()
    p = Path(path)
    if not p.exists():
        return AgentConfig()
    try:
        import tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return AgentConfig()

    cfg = AgentConfig()
    # flat mapping
    for section, key, attr in [
        ("combat", "combat_safety_rerank", "combat_safety_rerank"),
        ("combat", "combat_safety_rerank_weight", "combat_safety_rerank_weight"),
        ("combat", "combat_unsafe_mask", "combat_unsafe_mask"),
        ("combat", "lethal_probe", "lethal_probe"),
        ("combat", "beam_search", "beam_search"),
        ("combat", "combat_turn_solver", "combat_turn_solver"),
        ("combat", "combat_turn_planner", "combat_turn_planner"),
        ("combat", "turn_solver_mode", "turn_solver_mode"),
        ("combat", "turn_planner_mode", "turn_planner_mode"),
        ("combat", "combat_teacher_rerank", "combat_teacher_rerank"),
        ("combat", "combat_teacher_rerank_weight", "combat_teacher_rerank_weight"),
        ("noncombat", "auto_progress", "auto_progress"),
        ("noncombat", "loop_escape", "loop_escape"),
        ("noncombat", "boss_conditioned_card_guidance", "boss_conditioned_card_guidance"),
        ("noncombat", "boss_conditioned_card_guidance_weight", "boss_conditioned_card_guidance_weight"),
        ("sampling", "deterministic_policy", "deterministic_policy"),
    ]:
        sec = data.get(section)
        if isinstance(sec, dict) and key in sec:
            try:
                setattr(cfg, attr, type(getattr(cfg, attr))(sec[key]))
            except Exception:
                pass
    return cfg


def _sanitize_action(action: dict | None) -> tuple | None:
    """Stable hashable signature for comparing actions across the two codepaths."""
    if not isinstance(action, dict):
        return None
    keys = ("action", "type", "index", "card_index", "target_id", "col", "row", "slot")
    return tuple((k, action.get(k)) for k in keys)


def _match_action_idx(legal: list[dict], chosen: dict) -> int:
    sig = _sanitize_action(chosen)
    for i, a in enumerate(legal):
        if _sanitize_action(a) == sig:
            return i
    # Fallback: match by action name
    name = (chosen.get("action") or chosen.get("type") or "").lower() if isinstance(chosen, dict) else ""
    for i, a in enumerate(legal):
        if (a.get("action") or a.get("type") or "").lower() == name:
            return i
    return 0


class FullRunAgent:
    """Stateful agent that plays one full run.

    The agent owns:
      - `loop_tracker`: detects argmax dead-loops and pushes an escape action
      - `_last_reward_claim_sig`: prevents repeated claim_reward on the same item
    It reuses evaluate_ai's helper functions for the heavy work so there's no
    duplicate rule implementation.
    """

    def __init__(
        self,
        *,
        ppo_net,
        combat_net,
        vocab,
        device: torch.device,
        cfg: AgentConfig | None = None,
        combat_teacher_override=None,
        combat_bc_override=None,
        combat_mcts_agent=None,
        combat_pipe_getter=None,
        turn_planner=None,
    ):
        self.ppo_net = ppo_net
        self.combat_net = combat_net
        self.vocab = vocab
        self.device = device
        self.cfg = cfg or AgentConfig()
        # Optional shells — passed through to _select_action_nn
        self.combat_teacher_override = combat_teacher_override
        self.combat_bc_override = combat_bc_override
        self.combat_mcts_agent = combat_mcts_agent
        self.combat_pipe_getter = combat_pipe_getter
        self.turn_planner = turn_planner

        # evaluate_ai's RepeatLoopTracker instance
        import evaluate_ai as _eval_ai
        self._eval_ai = _eval_ai
        self.loop_tracker = _eval_ai.RepeatLoopTracker()
        self._last_reward_claim_sig = ""

        # Deploy-time teacher fusion: push the config value onto the combat net.
        # This is the hook that makes teacher-trained weights actually influence
        # argmax inference (before this, action_score_head was trained but never
        # consulted outside of the training loop).
        if combat_net is not None and self.cfg.combat_teacher_rerank:
            try:
                combat_net.teacher_rerank_weight = float(self.cfg.combat_teacher_rerank_weight)
            except Exception:
                pass

    # ---------------- main entry ----------------

    def select_action(
        self,
        state: dict,
        legal_actions: list[dict],
    ) -> dict:
        """Return {'action', 'action_idx', 'source', 'info'} for this state.

        Caller MUST invoke env.act(action) and then call `after_step()` so the
        agent's internal state (loop_tracker, reward_claim_sig) stays consistent.
        """
        if not legal_actions:
            return {
                "action": {"action": "wait"},
                "action_idx": 0,
                "source": "auto_wait",
                "info": {},
            }

        state_type = (state.get("state_type") or "").strip().lower()

        # 1. Rule-level auto progress (UI prompts)
        if self.cfg.auto_progress:
            auto = self._eval_ai._choose_auto_progress_action(
                state, state_type, legal_actions, self._last_reward_claim_sig
            )
            if auto is not None:
                idx = _match_action_idx(legal_actions, auto)
                return {
                    "action": auto,
                    "action_idx": idx,
                    "source": "auto_progress",
                    "info": {},
                }

        # 2. Loop-escape
        if self.cfg.loop_escape:
            escape = self.loop_tracker.choose_escape_action(legal_actions)
            if escape is not None:
                idx = _match_action_idx(legal_actions, escape)
                return {
                    "action": escape,
                    "action_idx": idx,
                    "source": "repeat_escape",
                    "info": {},
                }

        # 3. NN decision (combat / noncombat)
        idx, action, source, combat_trace = self._eval_ai._select_action_nn(
            state,
            legal_actions,
            self.ppo_net,
            self.combat_net,
            self.combat_teacher_override,
            self.combat_bc_override,
            self.combat_mcts_agent,
            self.combat_pipe_getter,
            self.vocab,
            self.device,
            lethal_probe=self.cfg.lethal_probe,
            combat_safety_rerank=self.cfg.combat_safety_rerank,
            beam_search=int(self.cfg.beam_search),
            turn_planner=self.turn_planner,
            act1_route_mode=self.cfg.act1_route_mode,
        )
        return {
            "action": action,
            "action_idx": idx,
            "source": source,
            "info": {"combat_trace": combat_trace},
        }

    # ---------------- post-step bookkeeping ----------------

    def after_step(
        self,
        *,
        before_state: dict,
        before_legal: list[dict],
        action: dict | None,
        next_state: dict,
    ) -> None:
        """Update loop_tracker + reward_claim dedup. Call after every env.act()."""
        self.loop_tracker.observe_transition(before_state, before_legal, action, next_state)
        before_state_type = (before_state.get("state_type") or "").strip().lower()
        if before_state_type == "combat_rewards" and action is not None:
            sig = self._eval_ai._reward_claim_signature(before_state, action)
            if sig:
                self._last_reward_claim_sig = sig
        else:
            # Non-reward screens reset dedup
            self._last_reward_claim_sig = ""

    def reset(self) -> None:
        """Call between runs / between seeds."""
        self.loop_tracker = self._eval_ai.RepeatLoopTracker()
        self._last_reward_claim_sig = ""
