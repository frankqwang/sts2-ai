#!/usr/bin/env python3
from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import numpy as np

from boss_leaf_evaluator import (
    LEAF_DATASET_SCHEMA_VERSION,
    LEAF_SCORE_V1_COEFFICIENTS,
    build_leaf_state_features,
    build_leaf_state_signature,
    score_targets_from_labels,
)
from build_combat_teacher_dataset import load_noncombat_policy
from combat_teacher_common import BaselineCombatPolicy, is_supported_solver_state, sanitize_action
from combat_turn_solver import CombatTurnSolver
from evaluate_ai import _choose_auto_progress_action, _next_reward_claim_signature
from full_run_env import create_full_run_client
from rl_encoder_v2 import build_structured_actions, build_structured_state, load_vocab
from rl_policy_v2 import FullRunPolicyNetworkV2, _structured_actions_to_numpy_dict, _structured_state_to_numpy_dict
from turn_solver_planner import _PipeEnvAdapter, _action_matches_legal
from vocab import Vocab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_boss_leaf_dataset")

COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}


def _load_seeds(seeds_file: Path, suite: str, limit: int) -> list[str]:
    payload = json.loads(seeds_file.read_text(encoding="utf-8-sig"))
    suite_entries = payload.get(suite) or []
    seeds: list[str] = []
    for item in suite_entries:
        if isinstance(item, dict):
            seed = str(item.get("seed") or "").strip()
            if seed:
                seeds.append(seed)
        elif item:
            seeds.append(str(item))
    return seeds[:limit] if limit > 0 else seeds


def _safe_load_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None)
    }
    model.load_state_dict(filtered, strict=False)


def _infer_combat_dims(state_dict: dict[str, Any]) -> tuple[int, int, int]:
    embed_dim = 48
    hidden_dim = 192
    deck_repr_dim = 0
    weight = state_dict.get("entity_emb.card_embed.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        embed_dim = int(weight.shape[1])
    for key, value in state_dict.items():
        if key.endswith("state_encoder.0.weight") and isinstance(value, torch.Tensor) and value.ndim == 2:
            hidden_dim = int(value.shape[0])
        if "deck_encoder.attn.in_proj_weight" in key and isinstance(value, torch.Tensor) and value.ndim == 2:
            deck_repr_dim = int(value.shape[0] // 3)
            break
    return embed_dim, hidden_dim, deck_repr_dim


def _load_combat_policy(checkpoint_path: str | Path, *, vocab: Vocab) -> BaselineCombatPolicy:
    from combat_nn import CombatPolicyValueNetwork

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    combat_state = checkpoint.get("mcts_model") or checkpoint.get("combat_model")
    if not isinstance(combat_state, dict):
        raise ValueError(f"Missing combat state dict in checkpoint: {checkpoint_path}")
    embed_dim, hidden_dim, deck_repr_dim = _infer_combat_dims(combat_state)
    network = CombatPolicyValueNetwork(
        vocab=vocab,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        deck_repr_dim=deck_repr_dim,
    )
    _safe_load_state_dict(network, combat_state)
    network.eval()
    return BaselineCombatPolicy(network=network, vocab=vocab, device=torch.device("cpu"))


def _build_ppo_tensors(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    vocab: Vocab,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    structured_state = build_structured_state(state, vocab)
    structured_actions = build_structured_actions(state, legal_actions, vocab)
    state_t: dict[str, torch.Tensor] = {}
    for key, value in _structured_state_to_numpy_dict(structured_state).items():
        tensor = torch.tensor(value).unsqueeze(0) if hasattr(value, "shape") else torch.tensor([value])
        if "ids" in key or "idx" in key or "types" in key or "count" in key:
            tensor = tensor.long()
        elif "mask" in key:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        state_t[key] = tensor
    action_t: dict[str, torch.Tensor] = {}
    for key, value in _structured_actions_to_numpy_dict(structured_actions).items():
        tensor = torch.tensor(value).unsqueeze(0) if hasattr(value, "shape") else torch.tensor([value])
        if "ids" in key or "types" in key or "indices" in key:
            tensor = tensor.long()
        elif "mask" in key:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        action_t[key] = tensor
    return state_t, action_t


def _select_noncombat_action(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    ppo_net: FullRunPolicyNetworkV2,
    *,
    vocab: Vocab,
) -> dict[str, Any]:
    state_t, action_t = _build_ppo_tensors(state, legal_actions, vocab)
    with torch.no_grad():
        logits, _value, _deck_q, _boss_ready, _action_adv = ppo_net(state_t, action_t)
    index = int(logits.squeeze(0)[: len(legal_actions)].argmax().item()) if legal_actions else 0
    return legal_actions[index] if 0 <= index < len(legal_actions) else legal_actions[0]


def _select_combat_action(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    baseline_policy: BaselineCombatPolicy,
    *,
    rng: random.Random | None = None,
    sample: bool = False,
    temperature: float = 1.0,
) -> dict[str, Any]:
    scored = baseline_policy.score(state, legal_actions)
    index = int(scored["best_index"]) if legal_actions else 0
    if sample and rng is not None and len(legal_actions) > 1:
        raw_probs = scored.get("probs")
        probs = np.asarray(raw_probs if raw_probs is not None else [], dtype=np.float64)
        if probs.shape[0] == len(legal_actions) and float(np.sum(probs)) > 0:
            temp = max(1e-3, float(temperature))
            logits = np.log(np.clip(probs, 1e-8, 1.0)) / temp
            weights = np.exp(logits - np.max(logits))
            total = float(np.sum(weights))
            if total > 0:
                norm = (weights / total).tolist()
                index = int(rng.choices(list(range(len(legal_actions))), weights=norm, k=1)[0])
    return legal_actions[index] if 0 <= index < len(legal_actions) else legal_actions[0]


def _sha1(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _enabled_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [action for action in (state.get("legal_actions") or []) if isinstance(action, dict) and action.get("is_enabled") is not False]


def _combat_round_number(state: dict[str, Any]) -> int:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    return int(
        battle.get("round_number", battle.get("round", state.get("round_number", state.get("round", 0)))) or 0
    )


def _resolve_candidate_line(
    env: _PipeEnvAdapter,
    baseline_policy: BaselineCombatPolicy,
    root_state_id: str,
    first_action: dict[str, Any],
    *,
    max_player_actions: int,
) -> list[dict[str, Any]] | None:
    env.load_state(root_state_id)
    try:
        next_state = env.act(first_action)
    except Exception:
        return None
    next_state_type = str(next_state.get("state_type") or "").strip().lower()
    first = sanitize_action(first_action) or {}
    if next_state.get("terminal") or next_state_type not in COMBAT_SCREENS:
        return [first]
    next_state_id = env.save_state()
    if not next_state_id:
        return None
    solver = CombatTurnSolver(env, baseline_policy, max_player_actions=max_player_actions)
    try:
        solution = solver.solve(next_state, root_state_id=next_state_id)
    finally:
        try:
            solver.cleanup()
        except Exception:
            pass
        try:
            env.delete_state(next_state_id)
        except Exception:
            pass
    if not solution.supported:
        return [first]
    return [first] + [dict(action) for action in solution.best_full_turn_line]


def _replay_line_to_leaf(env: _PipeEnvAdapter, root_state_id: str, line: list[dict[str, Any]], *, max_auto_steps: int = 32) -> dict[str, Any]:
    state = env.load_state(root_state_id)
    for action in line:
        state = env.act(action)
        state_type = str(state.get("state_type") or "").strip().lower()
        if state.get("terminal") or state_type not in COMBAT_SCREENS:
            return state
    for _ in range(max_auto_steps):
        state_type = str(state.get("state_type") or "").strip().lower()
        if state.get("terminal") or state_type not in COMBAT_SCREENS:
            return state
        legal = _enabled_legal_actions(state)
        if len(legal) > 1 or not legal:
            return state
        state = env.act(legal[0])
    return state


def _collect_group_records_from_replay_root(
    env: _PipeEnvAdapter,
    baseline_policy: BaselineCombatPolicy,
    replay_root_state_id: str,
    per_action: list[dict[str, Any]],
    per_action_regret: list[dict[str, Any]],
    parent_state: dict[str, Any],
    *,
    parent_id: str,
    seed: str,
    character: str,
    ascension: int,
    step: int,
    vocab: Vocab,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    group_records: list[dict[str, Any]] = []
    local_skips: Counter[str] = Counter()
    rollout_filtered_count = 0
    state_type = str(parent_state.get("state_type") or "").strip().lower()
    floor = int(((parent_state.get("run") or {}) if isinstance(parent_state.get("run"), dict) else {}).get("floor", 0) or 0)

    try:
        for candidate_rank, item in enumerate(per_action):
            candidate_action = item.get("action") or {}
            if not isinstance(candidate_action, dict):
                local_skips["candidate_action_invalid"] += 1
                continue
            line = _resolve_candidate_line(
                env,
                baseline_policy,
                replay_root_state_id,
                candidate_action,
                max_player_actions=args.max_player_actions,
            )
            if not line:
                local_skips["candidate_line_replay_failed"] += 1
                continue
            leaf_state = _replay_line_to_leaf(env, replay_root_state_id, line)
            leaf_state_type = str(leaf_state.get("state_type") or "").strip().lower()
            if (
                leaf_state.get("terminal")
                or leaf_state_type not in COMBAT_SCREENS
                or not is_supported_solver_state(leaf_state)
                or len(_enabled_legal_actions(leaf_state)) <= 1
            ):
                local_skips["leaf_state_not_supported"] += 1
                continue
            leaf_state_id = env.save_state()
            if not leaf_state_id:
                local_skips["leaf_snapshot_failed"] += 1
                continue
            try:
                rollout_seed = int(_sha1([parent_id, str(candidate_rank), str(step)])[:8], 16)
                rollout_stats = _aggregate_rollouts(
                    env,
                    baseline_policy,
                    leaf_state_id,
                    leaf_state,
                    rollouts_per_leaf=args.rollouts_per_leaf,
                    max_steps=args.rollout_max_steps,
                    rollout_policy=args.rollout_policy,
                    rollout_temperature=args.rollout_temperature,
                    rng_seed=rollout_seed,
                    solver_max_player_actions=getattr(args, "rollout_solver_max_actions", 12),
                    solver_hp_loss_weight=getattr(args, "rollout_solver_hp_loss_weight", 0.0),
                    solver_heuristic_blend=getattr(args, "rollout_solver_heuristic_blend", 1.0),
                )
            finally:
                try:
                    env.delete_state(leaf_state_id)
                except Exception:
                    pass
            if int(rollout_stats.get("rollouts", 0) or 0) < args.min_rollouts:
                rollout_filtered_count += 1
                local_skips["rollouts_below_min"] += 1
                continue
            signature = build_leaf_state_signature(leaf_state)
            features = build_leaf_state_features(leaf_state, vocab)
            boss_token = str(features.get("boss_token") or "")
            solver_regret = 0.0
            for regret_item in per_action_regret:
                if _action_matches_legal(candidate_action, regret_item.get("action") or {}):
                    solver_regret = float(regret_item.get("regret", 0.0) or 0.0)
                    break
            labels = {
                "win_prob": float(rollout_stats["win_prob"]),
                "boss_damage_ratio": float(rollout_stats["boss_damage_ratio"]),
                "hp_loss_ratio": float(rollout_stats["hp_loss_ratio"]),
                "survival_turns": float(rollout_stats["survival_turns"]),
                "rollouts": int(rollout_stats["rollouts"]),
            }
            group_records.append(
                {
                    "schema_version": LEAF_DATASET_SCHEMA_VERSION,
                    "sample_id": _sha1([parent_id, str(candidate_rank), boss_token, str(step)]),
                    "parent_id": parent_id,
                    "candidate_id": f"{parent_id}:c{candidate_rank}",
                    "seed": seed,
                    "character_id": character,
                    "ascension": ascension,
                    "floor": floor,
                    "encounter_kind": state_type,
                    "boss_token": boss_token,
                    "leaf_kind": "next_decision",
                    "round_number": signature["round_number"],
                    "solver_rank": candidate_rank,
                    "solver_score": float(item.get("score", 0.0) or 0.0),
                    "solver_regret": solver_regret,
                    "line": [sanitize_action(action) or {} for action in line],
                    "state_signature": signature,
                    "state_features": features,
                    "labels": labels,
                    "score_targets": score_targets_from_labels(labels),
                    "policy_used_for_rollouts": str(args.rollout_policy),
                }
            )
    finally:
        try:
            env.load_state(replay_root_state_id)
        except Exception:
            pass
        try:
            env.delete_state(replay_root_state_id)
        except Exception:
            pass

    # Phase 2.5 winnable-leaf filter (opt-in via --require-winnable-leaves).
    #
    # The canary on 2026-04-08 showed that naive sampling produces groups
    # where every candidate leaf rolls out to (win_prob=0, hp_loss=1.0)
    # because the boss state is structurally losing regardless of opener.
    # Such groups contribute no ranking signal (score_v1 clipped to -1 on
    # every candidate) so they dilute the dataset and can be safely
    # dropped. We keep a group iff at least one of its candidates hits
    # a minimum rollout-outcome threshold.
    if group_records and getattr(args, "require_winnable_leaves", False):
        threshold = float(getattr(args, "winnable_damage_threshold", 0.6))
        any_win = any(
            float(r["labels"].get("win_prob", 0.0) or 0.0) > 0.0
            for r in group_records
        )
        any_high_damage = any(
            float(r["labels"].get("boss_damage_ratio", 0.0) or 0.0) >= threshold
            for r in group_records
        )
        if not (any_win or any_high_damage):
            local_skips["non_winnable_leaf_group"] += len(group_records)
            group_records = []

    return group_records, local_skips, rollout_filtered_count


def _state_player_hp(state: dict[str, Any]) -> tuple[int, int]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") or state.get("player") or {}
    hp = int(player.get("hp", player.get("current_hp", 0)) or 0)
    max_hp = int(player.get("max_hp", 1) or 1)
    return hp, max(1, max_hp)


def _boss_damage_ratio(state: dict[str, Any]) -> float:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    total_hp = sum(float(enemy.get("hp", enemy.get("current_hp", 0)) or 0.0) for enemy in enemies if isinstance(enemy, dict))
    total_max_hp = sum(max(1.0, float(enemy.get("max_hp", 1) or 1.0)) for enemy in enemies if isinstance(enemy, dict))
    if total_max_hp <= 0:
        return 1.0 if str(state.get("state_type") or "").strip().lower() not in COMBAT_SCREENS else 0.0
    return float(max(0.0, min(1.0, 1.0 - total_hp / total_max_hp)))


def _boss_damage_ratio_with_fallback(state: dict[str, Any], *, fallback: float | None = None, assume_victory: bool = False) -> float:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    has_enemy_snapshot = any(isinstance(enemy, dict) for enemy in enemies)
    if not has_enemy_snapshot and fallback is not None:
        return float(max(0.0, min(1.0, fallback)))
    if not has_enemy_snapshot and assume_victory:
        return 1.0
    return _boss_damage_ratio(state)


def _rollout_leaf_once(
    env: _PipeEnvAdapter,
    baseline_policy: BaselineCombatPolicy,
    leaf_state_id: str,
    leaf_state: dict[str, Any],
    *,
    max_steps: int = 200,
    rollout_policy: str = "nn_sample",
    rng: random.Random | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    state = env.load_state(leaf_state_id)
    initial_hp, initial_max_hp = _state_player_hp(leaf_state)
    turns_survived = 0
    last_damage_ratio = _boss_damage_ratio_with_fallback(state, fallback=0.0)
    for _ in range(max_steps):
        state_type = str(state.get("state_type") or "").strip().lower()
        if state_type in COMBAT_SCREENS:
            last_damage_ratio = _boss_damage_ratio_with_fallback(state, fallback=last_damage_ratio)
        if state.get("terminal") or state_type == "game_over":
            return {
                "win_prob": 0.0,
                "boss_damage_ratio": _boss_damage_ratio_with_fallback(state, fallback=last_damage_ratio),
                "hp_loss_ratio": 1.0,
                "survival_turns": float(turns_survived),
            }
        if state_type not in COMBAT_SCREENS:
            final_hp, _ = _state_player_hp(state)
            hp_loss_ratio = max(0.0, min(1.0, (initial_hp - final_hp) / max(1, initial_max_hp)))
            return {
                "win_prob": 1.0,
                "boss_damage_ratio": _boss_damage_ratio_with_fallback(state, fallback=last_damage_ratio, assume_victory=True),
                "hp_loss_ratio": hp_loss_ratio,
                "survival_turns": float(turns_survived),
            }
        legal = _enabled_legal_actions(state)
        if not legal:
            break
        previous_round = _combat_round_number(state)
        action = _select_combat_action(
            state,
            legal,
            baseline_policy,
            rng=rng,
            sample=str(rollout_policy).strip().lower() == "nn_sample",
            temperature=temperature,
        )
        state = env.act(action)
        current_round = _combat_round_number(state)
        if current_round > previous_round:
            turns_survived += 1
    final_hp, _ = _state_player_hp(state)
    return {
        "win_prob": 0.0,
        "boss_damage_ratio": _boss_damage_ratio_with_fallback(state, fallback=last_damage_ratio),
        "hp_loss_ratio": max(0.0, min(1.0, (initial_hp - final_hp) / max(1, initial_max_hp))),
        "survival_turns": float(turns_survived),
    }


def _rollout_leaf_search_once(
    env: _PipeEnvAdapter,
    baseline_policy: BaselineCombatPolicy,
    leaf_state_id: str,
    leaf_state: dict[str, Any],
    *,
    max_steps: int = 200,
    solver_max_player_actions: int = 12,
    solver_hp_loss_weight: float = 0.0,
    solver_heuristic_blend: float = 1.0,
) -> dict[str, Any]:
    """Search-augmented rollout: every combat turn is solved via
    _TunedCombatTurnSolver and the full ``best_full_turn_line`` is
    played before yielding to the next turn.

    Matches the Phase 1 SOTA recipe (hp_loss_weight=0, heuristic_blend=1.0)
    so rollout outcomes reflect "what the production search would do"
    rather than "what the weakest PPO-only combat brain does". Designed
    to fix the 86% zero-spread tie on the legacy ``nn_sample`` rollouts.
    """
    # Import locally to avoid paying the torch import cost when the
    # search rollout is not used.
    from combat_turn_solver import CombatTurnSolution
    from turn_solver_planner import _TunedCombatTurnSolver

    state = env.load_state(leaf_state_id)
    initial_hp, initial_max_hp = _state_player_hp(leaf_state)
    turns_survived = 0
    last_damage_ratio = _boss_damage_ratio_with_fallback(state, fallback=0.0)

    # Single solver instance reused across all turns in this rollout.
    # Note we pass the env as the solver's branch env so it can
    # save/load around candidate lines.
    solver = _TunedCombatTurnSolver(
        env=env,
        baseline_policy=baseline_policy,
        max_player_actions=int(solver_max_player_actions),
        hp_loss_weight=float(solver_hp_loss_weight),
        heuristic_blend_alpha=float(solver_heuristic_blend),
    )

    # Hard cap on total step executions to guarantee forward progress
    # even if the solver's turn line somehow loops.
    step_budget = max(1, int(max_steps))
    steps_executed = 0

    while steps_executed < step_budget:
        state_type = str(state.get("state_type") or "").strip().lower()
        if state_type in COMBAT_SCREENS:
            last_damage_ratio = _boss_damage_ratio_with_fallback(state, fallback=last_damage_ratio)
        if state.get("terminal") or state_type == "game_over":
            return {
                "win_prob": 0.0,
                "boss_damage_ratio": _boss_damage_ratio_with_fallback(state, fallback=last_damage_ratio),
                "hp_loss_ratio": 1.0,
                "survival_turns": float(turns_survived),
            }
        if state_type not in COMBAT_SCREENS:
            # Combat exited cleanly — rollout has won this encounter.
            final_hp, _ = _state_player_hp(state)
            hp_loss_ratio = max(0.0, min(1.0, (initial_hp - final_hp) / max(1, initial_max_hp)))
            return {
                "win_prob": 1.0,
                "boss_damage_ratio": _boss_damage_ratio_with_fallback(
                    state, fallback=last_damage_ratio, assume_victory=True
                ),
                "hp_loss_ratio": hp_loss_ratio,
                "survival_turns": float(turns_survived),
            }

        legal = _enabled_legal_actions(state)
        if not legal:
            break
        previous_round = _combat_round_number(state)

        # Try to solve the whole turn. On any failure, fall back to
        # the baseline NN step so the rollout always makes forward
        # progress instead of spinning.
        solution: CombatTurnSolution | None = None
        try:
            # We need a state_id that matches the current live env state
            # so the solver can save/restore across its DFS branches.
            root_state_id = env.save_state()
            solution = solver.solve(state, root_state_id=root_state_id)
        except Exception:
            solution = None

        line: list[dict[str, Any]] = []
        if solution is not None and bool(getattr(solution, "supported", False)):
            line = list(getattr(solution, "best_full_turn_line", []) or [])

        if not line:
            # Solver unavailable — fall back to one baseline action step.
            action = _select_combat_action(
                state, legal, baseline_policy, rng=None, sample=False, temperature=1.0,
            )
            state = env.act(action)
            steps_executed += 1
            current_round = _combat_round_number(state)
            if current_round > previous_round:
                turns_survived += 1
            continue

        # Replay the solver's full-turn line action by action. We guard
        # against the line going off-rails by stopping on combat exit,
        # illegal action, or step budget exhaustion.
        line_consumed_a_turn = False
        for action in line:
            if steps_executed >= step_budget:
                break
            try:
                state = env.act(action)
            except Exception:
                break
            steps_executed += 1
            state_type = str(state.get("state_type") or "").strip().lower()
            if state.get("terminal") or state_type == "game_over":
                break
            if state_type not in COMBAT_SCREENS:
                # Combat exited mid-line — break out; the outer loop's
                # non-combat branch will treat this as a clean win.
                break
            if _combat_round_number(state) > previous_round:
                line_consumed_a_turn = True
                break

        if line_consumed_a_turn:
            turns_survived += 1
        elif steps_executed >= step_budget:
            break
        # else: still in the same turn (possibly because the solver line
        # ended without end_turn). The outer loop will re-solve from the
        # new state on the next iteration, which is the correct behavior
        # because something in the line may have changed the state (e.g.
        # non-attack cards adding cards to hand).

    final_hp, _ = _state_player_hp(state)
    return {
        "win_prob": 0.0,
        "boss_damage_ratio": _boss_damage_ratio_with_fallback(state, fallback=last_damage_ratio),
        "hp_loss_ratio": max(0.0, min(1.0, (initial_hp - final_hp) / max(1, initial_max_hp))),
        "survival_turns": float(turns_survived),
    }


def _aggregate_rollouts(
    env: _PipeEnvAdapter,
    baseline_policy: BaselineCombatPolicy,
    leaf_state_id: str,
    leaf_state: dict[str, Any],
    *,
    rollouts_per_leaf: int,
    max_steps: int,
    rollout_policy: str,
    rollout_temperature: float,
    rng_seed: int,
    solver_max_player_actions: int = 12,
    solver_hp_loss_weight: float = 0.0,
    solver_heuristic_blend: float = 1.0,
) -> dict[str, Any]:
    stats: list[dict[str, Any]] = []
    policy = str(rollout_policy).strip().lower()
    for rollout_idx in range(max(0, rollouts_per_leaf)):
        rng = random.Random(rng_seed + rollout_idx)
        if policy == "search_turn_solver":
            # Deterministic search: running it N times produces identical
            # outcomes (modulo sim determinism), so we short-circuit after
            # the first trial unless explicitly asked for more.
            stats.append(
                _rollout_leaf_search_once(
                    env,
                    baseline_policy,
                    leaf_state_id,
                    leaf_state,
                    max_steps=max_steps,
                    solver_max_player_actions=solver_max_player_actions,
                    solver_hp_loss_weight=solver_hp_loss_weight,
                    solver_heuristic_blend=solver_heuristic_blend,
                )
            )
            # One deterministic search is worth many sampled NN rollouts
            # for this policy; we still honor rollouts_per_leaf if caller
            # wants averaging.
        else:
            stats.append(
                _rollout_leaf_once(
                    env,
                    baseline_policy,
                    leaf_state_id,
                    leaf_state,
                    max_steps=max_steps,
                    rollout_policy=rollout_policy,
                    rng=rng,
                    temperature=rollout_temperature,
                )
            )
    if not stats:
        return {"rollouts": 0}
    return {
        "win_prob": float(np.mean([item["win_prob"] for item in stats])),
        "boss_damage_ratio": float(np.mean([item["boss_damage_ratio"] for item in stats])),
        "hp_loss_ratio": float(np.mean([item["hp_loss_ratio"] for item in stats])),
        "survival_turns": float(np.mean([item["survival_turns"] for item in stats])),
        "rollouts": len(stats),
    }


def _sampling_state_key(state: dict[str, Any], *, seed: str) -> str:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
    hand = battle.get("hand") or player.get("hand") or []
    enemies = battle.get("enemies") or state.get("enemies") or []
    payload = {
        "signature": build_leaf_state_signature(state),
        "hand_ids": [
            str(card.get("id") or card.get("card_id") or "")
            for card in hand
            if isinstance(card, dict)
        ],
        "enemy_rows": [
            {
                "id": str(enemy.get("entity_id") or enemy.get("id") or enemy.get("name") or ""),
                "hp": int(enemy.get("hp", enemy.get("current_hp", 0)) or 0),
                "block": int(enemy.get("block", 0) or 0),
            }
            for enemy in enemies
            if isinstance(enemy, dict)
        ],
        "legal_actions": [sanitize_action(action) or {} for action in _enabled_legal_actions(state)],
    }
    floor = int(((state.get("run") or {}) if isinstance(state.get("run"), dict) else {}).get("floor", 0) or 0)
    state_type = str(state.get("state_type") or "").strip().lower()
    return _sha1([seed, str(floor), state_type, json.dumps(payload, sort_keys=True)])


def _sampling_skip_reason(
    state: dict[str, Any],
    *,
    target_state_types: set[str],
    min_floor_for_elite: int,
    seen_turns: set[str],
    seed: str,
) -> str | None:
    state_type = str(state.get("state_type") or "").strip().lower()
    if state_type not in target_state_types:
        return None
    floor = int(((state.get("run") or {}) if isinstance(state.get("run"), dict) else {}).get("floor", 0) or 0)
    if state_type == "elite" and floor < min_floor_for_elite:
        return "elite_before_min_floor"
    turn_key = _sampling_state_key(state, seed=seed)
    if turn_key in seen_turns:
        return "duplicate_turn"
    if len(_enabled_legal_actions(state)) <= 1:
        return "insufficient_legal_actions"
    if not is_supported_solver_state(state):
        return "unsupported_solver_state"
    seen_turns.add(turn_key)
    return "ok"


def build_dataset(args: argparse.Namespace) -> int:
    vocab = load_vocab()
    baseline_policy = _load_combat_policy(args.checkpoint, vocab=vocab)
    noncombat_policy = load_noncombat_policy(args.checkpoint, vocab=vocab, device=torch.device("cpu"))
    client = create_full_run_client(
        use_pipe=True,
        transport=args.transport,
        port=args.port,
        ready_timeout_s=30.0,
        request_timeout_s=30.0,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out) if getattr(args, "summary_out", None) else out_path.with_suffix(".summary.json")
    manifest_path = Path(args.manifest_out) if getattr(args, "manifest_out", None) else out_path.with_suffix(".manifest.json")
    seeds = _load_seeds(Path(args.seeds_file), args.seed_suite, args.num_seeds)
    target_state_types = {item.strip().lower() for item in args.target_state_types}
    rows_written = 0
    sibling_groups = 0
    parent_attempts = 0
    boss_counter: Counter[str] = Counter()
    leaf_kind_counter: Counter[str] = Counter()
    encounter_counter: Counter[str] = Counter()
    skip_counter: Counter[str] = Counter()
    rollout_filtered_count = 0
    seed_stats: dict[str, dict[str, Any]] = {}
    start_time = time.monotonic()

    with out_path.open("w", encoding="utf-8") as handle:
        for seed_idx, seed in enumerate(seeds):
            logger.info("[%d/%d] seed=%s", seed_idx + 1, len(seeds), seed)
            state = client.reset(
                character_id=args.character,
                ascension_level=args.ascension,
                seed=seed,
                timeout_s=30.0,
            )
            seen_turns: set[str] = set()
            sampled_this_seed = 0
            last_reward_claim_sig = ""
            seed_diag = seed_stats.setdefault(
                seed,
                {
                    "rows": 0,
                    "sibling_groups": 0,
                    "parent_attempts": 0,
                    "target_state_hits": 0,
                    "skip_reasons": Counter(),
                    "boss_counts": Counter(),
                },
            )
            for step in range(args.max_steps_per_game):
                state_type = str(state.get("state_type") or "").strip().lower()
                if state.get("terminal") or state_type == "game_over":
                    break
                legal = _enabled_legal_actions(state)
                if not legal:
                    try:
                        state = client.act({"action": "wait"})
                    except Exception:
                        break
                    continue

                gate_reason = _sampling_skip_reason(
                    state,
                    target_state_types=target_state_types,
                    min_floor_for_elite=args.min_floor_for_elite,
                    seen_turns=seen_turns,
                    seed=seed,
                )
                if gate_reason is not None and sampled_this_seed >= args.max_parent_states_per_seed:
                    gate_reason = "max_parent_budget_exhausted"
                if gate_reason is not None and gate_reason != "ok":
                    skip_counter[gate_reason] += 1
                    seed_diag["skip_reasons"][gate_reason] += 1

                if gate_reason == "ok":
                    seed_diag["target_state_hits"] += 1
                if gate_reason == "ok" and sampled_this_seed < args.max_parent_states_per_seed:
                    raw_pipe = client._pipe if hasattr(client, "_pipe") else None
                    if raw_pipe is None:
                        skip_counter["pipe_env_missing"] += 1
                        seed_diag["skip_reasons"]["pipe_env_missing"] += 1
                    else:
                        env = _PipeEnvAdapter(raw_pipe)
                        solver_root_state_id = env.save_state()
                        if not solver_root_state_id:
                            skip_counter["solver_root_snapshot_failed"] += 1
                            seed_diag["skip_reasons"]["solver_root_snapshot_failed"] += 1
                        else:
                            parent_attempts += 1
                            seed_diag["parent_attempts"] += 1
                            solver = CombatTurnSolver(env, baseline_policy, max_player_actions=args.max_player_actions)
                            try:
                                solution = solver.solve(state, root_state_id=solver_root_state_id)
                            finally:
                                try:
                                    solver.cleanup()
                                except Exception:
                                    pass
                            if not solution.supported:
                                skip_counter["solver_unsupported"] += 1
                                seed_diag["skip_reasons"]["solver_unsupported"] += 1
                            elif not solution.per_action_score:
                                skip_counter["no_supported_candidates"] += 1
                                seed_diag["skip_reasons"]["no_supported_candidates"] += 1
                            else:
                                replay_root_state_id = env.save_state()
                                if not replay_root_state_id:
                                    skip_counter["replay_root_snapshot_failed"] += 1
                                    seed_diag["skip_reasons"]["replay_root_snapshot_failed"] += 1
                                else:
                                    parent_id = _sha1(
                                        [
                                            seed,
                                            str(((state.get("run") or {}) if isinstance(state.get("run"), dict) else {}).get("floor", 0)),
                                            str(_combat_round_number(state)),
                                            json.dumps(build_leaf_state_signature(state), sort_keys=True),
                                        ]
                                    )
                                    per_action = sorted(
                                        [item for item in solution.per_action_score if item.get("supported", True)],
                                        key=lambda item: -float(item.get("score", float("-inf"))),
                                    )[: args.topk_candidates]
                                    group_records, local_skips, local_rollout_filtered = _collect_group_records_from_replay_root(
                                        env,
                                        baseline_policy,
                                        replay_root_state_id,
                                        per_action,
                                        solution.per_action_regret,
                                        state,
                                        parent_id=parent_id,
                                        seed=seed,
                                        character=args.character,
                                        ascension=args.ascension,
                                        step=step,
                                        vocab=vocab,
                                        args=args,
                                    )
                                    rollout_filtered_count += local_rollout_filtered
                                    for reason, count in local_skips.items():
                                        skip_counter[reason] += count
                                        seed_diag["skip_reasons"][reason] += count
                                    if len(group_records) < args.min_siblings_per_parent:
                                        skip_counter["insufficient_siblings"] += 1
                                        seed_diag["skip_reasons"]["insufficient_siblings"] += 1
                                    else:
                                        for record in group_records:
                                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                        handle.flush()
                                        rows_written += len(group_records)
                                        sibling_groups += 1
                                        sampled_this_seed += 1
                                        seed_diag["rows"] += len(group_records)
                                        seed_diag["sibling_groups"] += 1
                                        for record in group_records:
                                            boss_key = str(record.get("boss_token") or "unknown").strip().lower() or "unknown"
                                            boss_counter[boss_key] += 1
                                            seed_diag["boss_counts"][boss_key] += 1
                                            leaf_kind_counter[str(record.get("leaf_kind") or "unknown")] += 1
                                            encounter_counter[str(record.get("encounter_kind") or "unknown")] += 1
                                    try:
                                        env.delete_state(replay_root_state_id)
                                    except Exception:
                                        pass
                            try:
                                env.delete_state(solver_root_state_id)
                            except Exception:
                                pass

                auto_action = _choose_auto_progress_action(state, state_type, legal, last_reward_claim_sig)
                if auto_action is not None:
                    action = auto_action
                else:
                    action = _select_combat_action(state, legal, baseline_policy) if state_type in COMBAT_SCREENS else _select_noncombat_action(state, legal, noncombat_policy, vocab=vocab)
                try:
                    last_reward_claim_sig = _next_reward_claim_signature(state_type, state, action)
                    state = client.act(action)
                except Exception:
                    break

    seed_diagnostics = {
        seed: {
            "rows": int(diag["rows"]),
            "sibling_groups": int(diag["sibling_groups"]),
            "parent_attempts": int(diag["parent_attempts"]),
            "target_state_hits": int(diag["target_state_hits"]),
            "skip_reasons": dict(sorted(diag["skip_reasons"].items())),
            "boss_counts": dict(sorted(diag["boss_counts"].items())),
        }
        for seed, diag in seed_stats.items()
    }
    summary = {
        "schema_version": LEAF_DATASET_SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint),
        "num_seeds": len(seeds),
        "rows": rows_written,
        "rows_written": rows_written,
        "sibling_groups": sibling_groups,
        "parent_attempts": parent_attempts,
        "seeds_with_rows": sum(1 for diag in seed_diagnostics.values() if int(diag["rows"]) > 0),
        "per_boss_counts": dict(sorted(boss_counter.items())),
        "per_leaf_kind_counts": dict(sorted(leaf_kind_counter.items())),
        "per_encounter_counts": dict(sorted(encounter_counter.items())),
        "skip_reasons": dict(sorted(skip_counter.items())),
        "rollouts_lt_min_filtered": rollout_filtered_count,
        "target_state_types": sorted(target_state_types),
        "elapsed_s": round(time.monotonic() - start_time, 2),
        "topk_candidates": args.topk_candidates,
        "score_softclip_temperature": 1.0,
        "seed_diagnostics": seed_diagnostics,
    }
    manifest = {
        "schema_version": LEAF_DATASET_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "output": str(out_path),
        "summary_path": str(summary_path),
        "checkpoint": str(args.checkpoint),
        "seeds_file": str(args.seeds_file),
        "seed_suite": str(args.seed_suite),
        "num_seeds_requested": int(args.num_seeds),
        "num_seeds_loaded": len(seeds),
        "character": str(args.character),
        "ascension": int(args.ascension),
        "transport": str(args.transport),
        "port": int(args.port),
        "target_state_types": sorted(target_state_types),
        "min_floor_for_elite": int(args.min_floor_for_elite),
        "topk_candidates": int(args.topk_candidates),
        "max_player_actions": int(args.max_player_actions),
        "max_parent_states_per_seed": int(args.max_parent_states_per_seed),
        "min_siblings_per_parent": int(args.min_siblings_per_parent),
        "max_steps_per_game": int(args.max_steps_per_game),
        "rollout_max_steps": int(args.rollout_max_steps),
        "rollouts_per_leaf": int(args.rollouts_per_leaf),
        "min_rollouts": int(args.min_rollouts),
        "rollout_policy": str(args.rollout_policy),
        "rollout_temperature": float(args.rollout_temperature),
        "score_v1_coefficients": LEAF_SCORE_V1_COEFFICIENTS,
        "score_softclip_temperature": 1.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d rows across %d sibling groups to %s", rows_written, sibling_groups, out_path)
    try:
        client.close()
    except Exception:
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build boss/final-elite leaf dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds-file", default="full_run_benchmark_seeds_200.json")
    parser.add_argument("--seed-suite", default="benchmark")
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--character", default="IRONCLAD")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--port", type=int, default=17120)
    parser.add_argument("--transport", default="pipe-binary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--target-state-types", nargs="+", default=["boss", "elite"])
    parser.add_argument("--min-floor-for-elite", type=int, default=14)
    parser.add_argument("--topk-candidates", type=int, default=4)
    parser.add_argument("--max-player-actions", type=int, default=12)
    parser.add_argument("--max-parent-states-per-seed", type=int, default=10)
    parser.add_argument("--min-siblings-per-parent", type=int, default=2)
    parser.add_argument("--max-steps-per-game", type=int, default=800)
    parser.add_argument("--rollout-max-steps", type=int, default=200)
    parser.add_argument("--rollouts-per-leaf", type=int, default=8)
    parser.add_argument("--min-rollouts", type=int, default=8)
    parser.add_argument(
        "--rollout-policy",
        choices=["nn_argmax", "nn_sample", "search_turn_solver"],
        default="nn_sample",
        help=(
            "Combat policy used during rollout. nn_argmax/nn_sample = baseline "
            "PPO combat (default, fast, but produces high zero-spread tie rate "
            "on leaf datasets). search_turn_solver = run _TunedCombatTurnSolver "
            "with heuristic_blend=1.0 on every turn, playing the full turn line "
            "per solve. Slower (3-5x per rollout due to DFS) but differentiates "
            "leaves because stronger combat extracts real signal from deck "
            "differences."
        ),
    )
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument(
        "--rollout-solver-max-actions", type=int, default=12,
        help=(
            "max_player_actions passed to _TunedCombatTurnSolver when "
            "--rollout-policy=search_turn_solver."
        ),
    )
    parser.add_argument(
        "--rollout-solver-hp-loss-weight", type=float, default=0.0,
        help=(
            "hp_loss_weight for the rollout solver. Default 0 matches the "
            "current Phase 1 SOTA recipe (act1 clear rate improved 3.5x at 200 seeds)."
        ),
    )
    parser.add_argument(
        "--rollout-solver-heuristic-blend", type=float, default=1.0,
        help=(
            "heuristic_blend_alpha for the rollout solver. Default 1.0 = "
            "fully heuristic leaf (current SOTA recipe). Set lower to blend "
            "in NN value."
        ),
    )
    parser.add_argument(
        "--require-winnable-leaves", action="store_true", default=False,
        help=(
            "Phase 2.5 filter: drop any sibling group where no candidate's "
            "rollout has win_prob>0 OR boss_damage_ratio above the winnable "
            "threshold. These groups contribute zero ranking signal "
            "because every candidate's score_v1 clips to -1. This is the "
            "mitigation for the structural 'losing boss entry' problem "
            "documented in the 2026-04-08 overnight canary."
        ),
    )
    parser.add_argument(
        "--winnable-damage-threshold", type=float, default=0.6,
        help=(
            "boss_damage_ratio threshold for --require-winnable-leaves. "
            "0.6 = a rollout must deal 60 percent of boss max HP to "
            "count as winnable signal. Lower threshold = keep more "
            "marginal groups."
        ),
    )
    args = parser.parse_args()
    return build_dataset(args)


if __name__ == "__main__":
    sys.exit(main())
