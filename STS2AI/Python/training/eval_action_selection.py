"""Action selection strategies — NN inference, teacher override, MCTS integration.

Extracted from evaluate_ai.py. Contains the core action selection logic:
  _select_action_nn()     — main NN-based action selection with safety reranking
  _select_action_random() — random baseline
  _select_action_heuristic() — rule-based heuristic
  _build_ppo_tensors()    — tensor conversion for PPO network
  _build_combat_tensors() — tensor conversion for combat network
  _probe_direct_lethal_indices() — lethal detection for safety
  CombatMctsTrace, CombatTeacherOverride — dataclasses for MCTS/teacher info
  CombatMctsTacticalBlendEvaluator — blended MCTS evaluator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from core.rl_encoder_v2 import _lower, _safe_float, _safe_int, MAX_ACTIONS, build_structured_state
from core.combat_nn import CombatPolicyValueNetwork, CombatNNEvaluator, build_combat_features, build_combat_action_features

logger = logging.getLogger(__name__)

DEFAULT_ACT1_ROUTE_MODE = "conservative"




@dataclass(slots=True)
class CombatMctsTrace:
    chosen_action: dict[str, Any]
    top_actions: list[dict[str, Any]]
    sims: int
    root_value: float



@dataclass(slots=True)
class CombatTeacherOverride:
    network: CombatPolicyValueNetwork
    vocab: Vocab
    device: torch.device
    mode: str = "full_replace"
    lethal_logit_blend_alpha: float = 0.0
    direct_lethal_probe_top_k: int = 4
    direct_lethal_step_budget: int = 24



class CombatMctsTacticalBlendEvaluator:
    """Eval-time wrapper that nudges MCTS leaf values toward local tactical progress."""

    def __init__(self, base_evaluator: Any, config: CombatMctsTacticalBlendConfig):
        self.base_evaluator = base_evaluator
        self.config = config

    def _blend_value(self, state: dict[str, Any], nn_value: float) -> float:
        weight = max(0.0, min(1.0, float(self.config.weight)))
        if weight <= 0.0:
            return float(nn_value)
        tactical = _combat_tactical_leaf_value(state)
        return float(np.clip((1.0 - weight) * float(nn_value) + weight * tactical, -1.0, 1.0))

    def evaluate(self, state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
        policy, value = self.base_evaluator.evaluate(state, legal_actions)
        return policy, self._blend_value(state, value)

    def evaluate_batch(
        self,
        states: list[dict[str, Any]],
        legal_actions_list: list[list[dict[str, Any]]],
    ) -> list[tuple[np.ndarray, float]]:
        base_results = self.base_evaluator.evaluate_batch(states, legal_actions_list)
        return [
            (policy, self._blend_value(state, value))
            for (policy, value), state in zip(base_results, states)
        ]



# ---------------------------------------------------------------------------
# Tensor helpers (mirror train_hybrid.py patterns)
# ---------------------------------------------------------------------------

def _build_ppo_tensors(
    state: dict, legal: list[dict], vocab: Vocab, device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build PPO state/action tensors from raw game state."""
    ss = build_structured_state(state, vocab)
    sa = build_structured_actions(state, legal, vocab)

    state_t: dict[str, torch.Tensor] = {}
    for k, v in _structured_state_to_numpy_dict(ss).items():
        t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
        if "ids" in k or "idx" in k or "types" in k or "count" in k:
            t = t.long()
        elif "mask" in k:
            t = t.bool()
        else:
            t = t.float()
        state_t[k] = t.to(device)

    action_t: dict[str, torch.Tensor] = {}
    for k, v in _structured_actions_to_numpy_dict(sa).items():
        t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
        if "ids" in k or "types" in k or "indices" in k:
            t = t.long()
        elif "mask" in k:
            t = t.bool()
        else:
            t = t.float()
        action_t[k] = t.to(device)

    return state_t, action_t



def _build_combat_tensors(
    state: dict, legal: list[dict], vocab: Vocab, device: torch.device,
    ppo_net: FullRunPolicyNetworkV2 | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build combat NN state/action tensors from raw game state."""
    sf = build_combat_features(state, vocab)
    af = build_combat_action_features(state, legal, vocab)

    # Inject deck_repr from PPO brain if available (build_plan_z bridge)
    if ppo_net is not None and hasattr(ppo_net, "compute_deck_repr"):
        try:
            from rl_encoder_v2 import build_structured_state as _bss
            ss = _bss(state, vocab)
            deck_t = {
                "deck_ids": torch.tensor(ss.deck_ids).unsqueeze(0).to(device),
                "deck_aux": torch.tensor(ss.deck_aux).unsqueeze(0).float().to(device),
                "deck_mask": torch.tensor(ss.deck_mask).unsqueeze(0).bool().to(device),
            }
            with torch.no_grad():
                sf["deck_repr"] = ppo_net.compute_deck_repr(deck_t).squeeze(0).cpu().numpy()
        except Exception:
            pass

    sf_t: dict[str, torch.Tensor] = {}
    for k, v in sf.items():
        t = torch.tensor(v).unsqueeze(0)
        if v.dtype in (np.int64, np.int32):
            t = t.long()
        elif v.dtype == bool:
            t = t.bool()
        else:
            t = t.float()
        sf_t[k] = t.to(device)

    af_t: dict[str, torch.Tensor] = {}
    for k, v in af.items():
        t = torch.tensor(v).unsqueeze(0)
        if v.dtype in (np.int64, np.int32):
            t = t.long()
        elif v.dtype == bool:
            t = t.bool()
        else:
            t = t.float()
        af_t[k] = t.to(device)

    return sf_t, af_t



def _probe_direct_lethal_indices(
    *,
    legal: list[dict[str, Any]],
    pipe_getter: Any | None,
    candidate_indices: list[int],
    step_budget: int,
) -> set[int]:
    if pipe_getter is None or not candidate_indices:
        return set()
    fm = None
    lethal_indices: set[int] = set()
    try:
        fm = PipeCombatForwardModel.from_current_state(
            pipe_getter,
            max_step_budget=max(4, int(step_budget)),
        )
        for idx in candidate_indices:
            if idx < 0 or idx >= len(legal):
                continue
            child = fm.clone()
            try:
                child.step(legal[idx])
                if _combat_probe_is_victory(child.get_state_dict()):
                    lethal_indices.add(idx)
            except Exception:
                continue
        return lethal_indices
    except Exception:
        return set()
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



def _select_action_combat_teacher_rerank(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    baseline_logits: np.ndarray,
    combat_teacher_override: CombatTeacherOverride,
    pipe_getter: Any | None,
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    if not legal:
        return None, None, None
    masked_scores, teacher_logits = _combat_teacher_forward_arrays(
        state=state,
        legal=legal,
        combat_teacher_override=combat_teacher_override,
    )
    score_idx = int(np.argmax(masked_scores))
    baseline_idx = int(np.argmax(baseline_logits[:len(legal)])) if len(legal) > 0 else 0

    if pipe_getter is not None and combat_teacher_override.lethal_logit_blend_alpha > 0.0:
        candidate_indices = _combat_teacher_probe_candidate_indices(
            legal,
            masked_scores,
            baseline_logits[:len(legal)],
            top_k=combat_teacher_override.direct_lethal_probe_top_k,
        )
        lethal_indices = _probe_direct_lethal_indices(
            legal=legal,
            pipe_getter=pipe_getter,
            candidate_indices=candidate_indices,
            step_budget=combat_teacher_override.direct_lethal_step_budget,
        )
        if lethal_indices:
            best_idx = max(lethal_indices, key=lambda idx: float(baseline_logits[idx]))
            return int(best_idx), legal[int(best_idx)], "combat_teacher_rerank_direct_lethal"

    runtime_labels = set(detect_motif_labels(state, legal))
    override_source = _combat_teacher_runtime_override_source(
        state=state,
        legal=legal,
        baseline_idx=baseline_idx,
        teacher_idx=score_idx,
        runtime_labels=runtime_labels,
    )
    if override_source is not None:
        return score_idx, legal[score_idx], override_source
    return None, None, None



def _select_action_combat_teacher(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    combat_teacher_override: CombatTeacherOverride,
    pipe_getter: Any | None,
) -> tuple[int, dict[str, Any], str]:
    masked_scores, masked_logits = _combat_teacher_forward_arrays(
        state=state,
        legal=legal,
        combat_teacher_override=combat_teacher_override,
    )
    direct_lethal_probe = None
    if pipe_getter is not None and combat_teacher_override.lethal_logit_blend_alpha > 0.0:
        direct_lethal_probe = lambda candidate_indices: _probe_direct_lethal_indices(
            legal=legal,
            pipe_getter=pipe_getter,
            candidate_indices=candidate_indices,
            step_budget=combat_teacher_override.direct_lethal_step_budget,
        )
    action_idx, source = _select_combat_teacher_index(
        legal=legal,
        masked_scores=masked_scores,
        masked_logits=masked_logits,
        lethal_logit_blend_alpha=combat_teacher_override.lethal_logit_blend_alpha,
        direct_lethal_probe_top_k=combat_teacher_override.direct_lethal_probe_top_k,
        direct_lethal_probe=direct_lethal_probe,
    )
    if action_idx < len(legal):
        return action_idx, legal[action_idx], source
    return 0, legal[0], source



# ---------------------------------------------------------------------------
# Action selection strategies
# ---------------------------------------------------------------------------

def _select_action_nn(
    state: dict,
    legal: list[dict],
    ppo_net: FullRunPolicyNetworkV2,
    combat_net: CombatPolicyValueNetwork,
    combat_teacher_override: CombatTeacherOverride | None,
    combat_bc_override: CombatBcOverride | None,
    combat_mcts_agent: CombatMCTSAgent | None,
    combat_pipe_getter: Any | None,
    vocab: Vocab,
    device: torch.device,
    *,
    lethal_probe: bool = False,
    combat_safety_rerank: bool = False,
    beam_search: int = 0,
    turn_planner: Any | None = None,
    act1_route_mode: str = DEFAULT_ACT1_ROUTE_MODE,
) -> tuple[int, dict, str, CombatMctsTrace | None]:
    """Select action using NN (argmax / deterministic).

    Non-combat: PPO network argmax.
    Combat: Combat NN argmax.
    Falls back to heuristic on any error.
    """
    st = (state.get("state_type") or "").lower()

    try:
        if st in COMBAT_SCREENS:
            mcts_screen_mode = str(getattr(combat_mcts_agent, "_screen_mode", "always")) if combat_mcts_agent is not None else "always"
            mcts_min_floor = int(getattr(combat_mcts_agent, "_min_floor", 0)) if combat_mcts_agent is not None else 0
            run = state.get("run") if isinstance(state.get("run"), dict) else {}
            current_floor = _safe_int(run.get("floor", 0), 0)
            screen_match = (
                mcts_screen_mode == "always"
                or (mcts_screen_mode == "boss" and st == "boss")
                or (mcts_screen_mode == "elite" and st == "elite")
                or (mcts_screen_mode == "boss_elite" and st in ("boss", "elite"))
            )
            use_combat_mcts = (
                combat_mcts_agent is not None
                and combat_pipe_getter is not None
                and (screen_match or current_floor >= mcts_min_floor)
            )
            if use_combat_mcts:
                fm = None
                restored_state: dict[str, Any] | None = None
                try:
                    fm = PipeCombatForwardModel.from_current_state(
                        combat_pipe_getter,
                        max_step_budget=getattr(combat_mcts_agent, "_max_step_budget", 200),
                    )
                    action, root = combat_mcts_agent.choose_action(fm)
                    top_actions, root_value = _summarize_mcts_root(root)
                    trace_meta = CombatMctsTrace(
                        chosen_action=dict(action) if isinstance(action, dict) else {},
                        top_actions=top_actions,
                        sims=int(getattr(combat_mcts_agent.config, "num_simulations", 0)),
                        root_value=root_value,
                    )
                finally:
                    if fm is not None:
                        try:
                            restored = fm.cleanup_and_restore()
                            if restored is None:
                                fm.cleanup()
                            else:
                                restored_state = restored
                        except Exception:
                            try:
                                fm.cleanup()
                            except Exception:
                                pass
                effective_legal = legal
                if isinstance(restored_state, dict):
                    restored_legal = restored_state.get("legal_actions")
                    if isinstance(restored_legal, list) and restored_legal:
                        effective_legal = restored_legal
                action_idx = _match_legal_action_index(
                    effective_legal,
                    action,
                    allow_action_only_fallback=False,
                )
                if action_idx is not None and action_idx < len(effective_legal):
                    return action_idx, effective_legal[action_idx], "combat_mcts", trace_meta
                logger.debug("combat_mcts action no longer matches restored legal set; falling back to plain combat NN")
            if combat_teacher_override is not None and combat_teacher_override.mode == "full_replace":
                action_idx, action, action_source = _select_action_combat_teacher(
                    state=state,
                    legal=legal,
                    combat_teacher_override=combat_teacher_override,
                    pipe_getter=combat_pipe_getter,
                )
                return action_idx, action, action_source, None
            sf_t, af_t = _build_combat_tensors(state, legal, vocab, device, ppo_net=ppo_net)
            with torch.no_grad():
                logits, _value = combat_net(sf_t, af_t)
            # Mask invalid actions and argmax
            mask = af_t["action_mask"].float()
            logits = logits.squeeze(0)  # (MAX_ACTIONS,)
            logits = logits + (1.0 - mask.squeeze(0)) * (-1e9)
            base_logits = logits[:len(legal)].detach().cpu().numpy()
            decision_logits = np.asarray(base_logits, dtype=np.float32)
            if combat_bc_override is not None:
                bc_choice = _select_action_combat_bc(
                    state=state,
                    legal=legal,
                    combat_bc_override=combat_bc_override,
                    base_logits=base_logits,
                )
                if bc_choice is not None:
                    action_idx, action, action_source = bc_choice
                    return action_idx, action, action_source, None
            if combat_teacher_override is not None and combat_teacher_override.mode == "hard_override":
                teacher_choice = _select_action_combat_teacher_rerank(
                    state=state,
                    legal=legal,
                    baseline_logits=base_logits,
                    combat_teacher_override=combat_teacher_override,
                    pipe_getter=combat_pipe_getter,
                )
                if teacher_choice[0] is not None and teacher_choice[1] is not None and teacher_choice[2] is not None:
                    action_idx, action, action_source = teacher_choice
                    return action_idx, action, action_source, None
            # R1+R2 combat hard-safety mask (2026-04-15, lab feature).
            # Gated by `_COMBAT_UNSAFE_MASK_ENABLED`; see that constant's
            # comment in train_hybrid.py for the status note. Currently off
            # because apples-to-apples 5-iter training showed a -12pp
            # boss_reach regression with no working remedy.
            if _COMBAT_UNSAFE_MASK_ENABLED and len(decision_logits) > 0:
                _unsafe_mask = compute_combat_unsafe_mask(state, legal)
                decision_logits = np.asarray(decision_logits, dtype=np.float32).copy()
                decision_logits[: len(_unsafe_mask)] += (1.0 - _unsafe_mask) * (-1e9)
            if combat_safety_rerank and len(decision_logits) > 0:
                decision_logits, _safety_adjustments = rerank_combat_logits_with_safety(
                    state,
                    legal,
                    decision_logits,
                )
            # Standalone lethal probe — only fires when NN chose end_turn
            # but play_card options exist (minimal probe to catch missed lethals)
            if lethal_probe and combat_pipe_getter is not None:
                nn_choice = int(np.argmax(decision_logits)) if len(decision_logits) > 0 else 0
                chosen_action_type = (legal[nn_choice].get("action") or "").lower() if nn_choice < len(legal) else ""
                if chosen_action_type == "end_turn":
                    play_indices = [
                        i for i, a in enumerate(legal)
                        if (a.get("action") or "").lower() == "play_card"
                    ]
                    if play_indices:
                        raw_pipe = combat_pipe_getter()
                        best_lethal_idx: int | None = None
                        sid = None
                        try:
                            sid = raw_pipe.call("save_state").get("state_id", "")
                            for idx in play_indices[:4]:
                                raw_pipe.call("load_state", {"state_id": sid})
                                probe_result = raw_pipe.call("step", legal[idx])
                                probe_state = probe_result.get("state", probe_result)
                                if _combat_probe_is_victory(probe_state):
                                    if best_lethal_idx is None or base_logits[idx] > base_logits[best_lethal_idx]:
                                        best_lethal_idx = idx
                            raw_pipe.call("load_state", {"state_id": sid})
                        except Exception:
                            best_lethal_idx = None
                        finally:
                            if sid:
                                try:
                                    raw_pipe.call("delete_state", {"state_id": sid})
                                except Exception:
                                    pass
                        if best_lethal_idx is not None:
                            return best_lethal_idx, legal[best_lethal_idx], "nn_lethal", None
            # Turn-level planner (policy-guided beam search over complete turns,
            # OR DFS turn solver, OR multi-turn lookahead — see turn_solver_planner.py
            # and combat_turn_planner.py for the available implementations).
            if turn_planner is not None and combat_pipe_getter is not None:
                planner_mode = getattr(turn_planner, "_mode", "boss_elite")
                should_plan = (
                    planner_mode == "always"
                    or (planner_mode == "boss" and st == "boss")
                    or (planner_mode == "elite" and st == "elite")
                    or (planner_mode == "boss_elite" and st in ("boss", "elite"))
                )
                if should_plan:
                    try:
                        result = turn_planner.select_action(
                            combat_pipe_getter, state, legal,
                        )
                        if result is not None:
                            pidx, psource = result
                            if pidx < len(legal):
                                return pidx, legal[pidx], psource, None
                    except Exception as e:
                        logger.debug("Turn planner failed: %s", e)

            # NOTE 2026-04-08 (wizardly cleanup): legacy --beam-search branch
            # removed; beam_search_combat.py archived. See parser comment.
            base_choice = int(np.argmax(base_logits)) if len(base_logits) > 0 else 0
            action_idx = int(np.argmax(decision_logits)) if len(decision_logits) > 0 else 0
            action_source = "nn_safety" if combat_safety_rerank and action_idx != base_choice else "nn"
        else:
            state_t, action_t = _build_ppo_tensors(state, legal, vocab, device)
            with torch.no_grad():
                logits, _values, _dq, _boss_ready, _action_adv = ppo_net(state_t, action_t)
            # Argmax (deterministic)
            logits = logits.squeeze(0)  # (MAX_ACTIONS,)
            map_override = _choose_act1_safe_map_action(
                state,
                legal,
                logits.detach().cpu().numpy(),
                route_mode=act1_route_mode,
            )
            if map_override is not None:
                action_idx, action, action_source = map_override
                return action_idx, action, action_source, None
            action_idx = int(logits.argmax().item())
            action_source = "nn"

        if action_idx < len(legal):
            return action_idx, legal[action_idx], action_source, None
        return 0, legal[0], action_source, None

    except Exception as e:
        logger.debug("NN inference error, falling back to heuristic: %s", e)
        if st in COMBAT_SCREENS:
            action_idx, action = heuristic_combat_action(legal, state)
            return action_idx, action, "heuristic_fallback", None
        return 0, legal[0], "heuristic_fallback", None



def _select_action_random(
    state: dict, legal: list[dict],
) -> tuple[int, dict]:
    """Select a random legal action."""
    idx = random.randrange(len(legal))
    return idx, legal[idx]



def _select_action_heuristic(
    state: dict, legal: list[dict],
) -> tuple[int, dict]:
    """Select action using heuristic strategy.

    Combat: use the rule-based heuristic from train_hybrid.
    Non-combat: pick first legal action (usually a reasonable default).
    """
    st = (state.get("state_type") or "").lower()
    if st in COMBAT_SCREENS:
        return heuristic_combat_action(legal, state)
    candidate = None
    if st == "rest_site":
        candidate = choose_deterministic_rest_action(state, legal, hp_rest_threshold=0.5)
    elif st == "shop":
        candidate = choose_deterministic_shop_action(state, legal)
    elif st == "card_select":
        candidate = choose_deterministic_card_select_action(state, legal)
    if candidate is not None:
        idx = _match_legal_action_index(legal, candidate)
        return (idx if idx is not None else 0), candidate
    return 0, legal[0]

