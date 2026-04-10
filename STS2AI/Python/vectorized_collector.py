"""Vectorized episode collector for parallel env stepping.

All N envs step in lockstep: parallel pipe I/O (ThreadPoolExecutor)
+ batched NN inference (single GPU forward pass).

This replaces the per-worker independent episode collection when
--vectorized is used, achieving ~3-4x throughput improvement.
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from combat_nn import CombatPolicyValueNetwork, build_combat_action_features, build_combat_features
from rl_encoder_v2 import (
    COMBAT_SCREENS,
    build_structured_actions,
    build_structured_state,
)
from rl_policy_v2 import (
    _structured_state_to_numpy_dict,
    _structured_actions_to_numpy_dict,
)
from rl_reward_shaping import (
    fight_summary,
    screen_local_delta_reward,
    shaped_reward,
)
from vocab import Vocab

logger = logging.getLogger(__name__)

# Screens where agent auto-picks first legal action (no NN inference needed)
AUTO_PROGRESS_SCREENS = {
    "combat_rewards", "card_select", "hand_select", "relic_select",
    "treasure", "advance_dialogue", "proceed", "confirm",
}

# Transient states where game is still processing — re-poll with get_state
WAIT_SCREENS = {"combat_pending", "pending", ""}

# Screens requiring non-combat NN inference
NC_INFERENCE_SCREENS = {
    "map", "card_reward", "shop", "rest_site", "campfire", "event",
}


@dataclass
class EnvState:
    """Per-env episode tracking state."""
    state: dict[str, Any] = field(default_factory=dict)
    prev_state: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    in_combat: bool = False
    combat_room_type: str = "monster"
    hp_at_combat_start: int = 0
    step_count: int = 0
    floors: int = 0
    combats: int = 0
    victories: int = 0
    boss_reached: bool = False
    boss_hp_peak: float = 0.0
    cards_taken: list[str] = field(default_factory=list)
    cards_skipped: int = 0
    deck_size_at_boss: list[int] = field(default_factory=list)
    outcome: str = "incomplete"
    error: str | None = None


def _np_to_tensor(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert numpy dict to tensor dict."""
    result = {}
    for k, v in arrays.items():
        t = torch.tensor(v)
        if v.dtype in (np.int64, np.int32):
            t = t.long()
        elif v.dtype == bool:
            t = t.bool()
        else:
            t = t.float()
        result[k] = t.to(device)
    return result


def _batch_np_to_tensor(
    feature_list: list[dict[str, np.ndarray]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack list of numpy feature dicts into batched tensors."""
    if not feature_list:
        return {}
    keys = feature_list[0].keys()
    result = {}
    for k in keys:
        arrays = [f[k] for f in feature_list]
        stacked = np.stack(arrays, axis=0)
        t = torch.tensor(stacked)
        if stacked.dtype in (np.int64, np.int32):
            t = t.long()
        elif stacked.dtype == bool:
            t = t.bool()
        else:
            t = t.float()
        result[k] = t.to(device)
    return result


def _get_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [a for a in (state.get("legal_actions") or []) if isinstance(a, dict)]


def _detect_room_type(st: str, state: dict[str, Any]) -> str:
    if st == "boss":
        return "boss"
    if st == "elite":
        return "elite"
    return "monster"


def _extract_deck_size(state: dict[str, Any]) -> int:
    player = state.get("player") or {}
    deck = player.get("deck") or player.get("cards") or []
    return len(deck)


def _estimate_boss_hp_fraction(state: dict[str, Any]) -> float:
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or []
    if not enemies:
        return 0.0
    total_frac = 0.0
    count = 0
    for e in enemies:
        hp = int(e.get("hp", e.get("current_hp", 0)) or 0)
        max_hp = int(e.get("max_hp", 1) or 1)
        if max_hp > 0:
            total_frac += 1.0 - (hp / max_hp)
            count += 1
    return total_frac / max(1, count)


def collect_vectorized_episodes(
    ppo_net,
    combat_net: CombatPolicyValueNetwork,
    vocab: Vocab,
    clients: list,
    *,
    character_id: str = "IRONCLAD",
    ascension_level: int = 0,
    max_steps: int = 600,
    episode_timeout: float = 90.0,
    screen_local_delta: bool = True,
    device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect episodes from N envs with synchronized stepping.

    Returns:
        ppo_data: list of dicts with PPO transition data (all envs merged)
        combat_data: list of dicts with combat PPO data (all envs merged)
        stats_list: list of per-env episode statistics
    """
    num_envs = len(clients)
    if device is None:
        device = next(ppo_net.parameters()).device

    pool = ThreadPoolExecutor(max_workers=num_envs)

    # --- Reset all envs in parallel ---
    reset_futures = {
        pool.submit(
            c.reset, character_id=character_id, ascension_level=ascension_level, timeout_s=30.0
        ): i
        for i, c in enumerate(clients)
    }
    envs = [EnvState() for _ in range(num_envs)]
    for f in as_completed(reset_futures):
        i = reset_futures[f]
        try:
            envs[i].state = f.result()
        except Exception as e:
            envs[i].done = True
            envs[i].error = str(e)
            logger.warning("Env %d reset failed: %s", i, e)

    # PPO and combat transition buffers
    ppo_transitions: list[dict[str, Any]] = []
    combat_transitions: list[dict[str, Any]] = []

    episode_start = time.monotonic()

    # --- Main step loop ---
    effective_steps = 0  # only count steps with actual decisions (not wait/pending)
    for step in range(max_steps * 3):  # allow extra iterations for wait cycles
        if time.monotonic() - episode_start > episode_timeout:
            break
        if effective_steps >= max_steps:
            break

        active = [i for i in range(num_envs) if not envs[i].done]
        if not active:
            break

        # Classify envs by screen type
        auto_ids = []
        combat_ids = []
        nc_ids = []
        terminal_ids = []
        wait_ids = []  # envs in transient state, need re-poll

        for i in active:
            st = (envs[i].state.get("state_type") or "").lower()
            legal = _get_legal_actions(envs[i].state)

            if st == "game_over" or envs[i].state.get("terminal"):
                terminal_ids.append(i)
            elif st in WAIT_SCREENS or (not legal and st not in AUTO_PROGRESS_SCREENS):
                wait_ids.append(i)  # transient state, re-poll
            elif not legal:
                auto_ids.append(i)
            elif st in COMBAT_SCREENS:
                combat_ids.append(i)
            elif st in NC_INFERENCE_SCREENS:
                nc_ids.append(i)
            else:
                auto_ids.append(i)  # unknown screen, auto-progress

        # --- Handle terminals ---
        for i in terminal_ids:
            state = envs[i].state
            go = state.get("game_over") or {}
            outcome = (go.get("run_outcome") or go.get("outcome") or "").lower()
            is_victory = "victory" in outcome or outcome == "win"
            envs[i].outcome = "victory" if is_victory else "death"
            envs[i].done = True

            # Terminal reward for last PPO step
            if envs[i].prev_state:
                terminal_val = 1.0 if is_victory else -1.0
                reward = shaped_reward(envs[i].prev_state, state, terminal_val, done=True)
                ppo_transitions.append({
                    "env_id": i, "type": "terminal", "reward": reward,
                })

            # Combat death feedback
            if envs[i].in_combat and not is_victory:
                room_type = envs[i].combat_room_type
                hp_start = envs[i].hp_at_combat_start
                player = state.get("player") or {}
                max_hp = int(player.get("max_hp", 80) or 80)
                boss_frac = max(envs[i].boss_hp_peak, _estimate_boss_hp_fraction(state)) if room_type == "boss" else 0.0
                feedback = fight_summary(hp_start, 0, max_hp, won=False, room_type=room_type, boss_hp_fraction_dealt=boss_frac)
                ppo_transitions.append({
                    "env_id": i, "type": "fight_death", "reward": feedback,
                })

        # --- Handle wait/pending screens (send "wait" action to advance game) ---
        if wait_ids:
            wait_futures = {
                pool.submit(clients[i].act, {"action": "wait"}): i
                for i in wait_ids
            }
            for f in as_completed(wait_futures):
                i = wait_futures[f]
                try:
                    envs[i].state = f.result()
                except Exception as e:
                    logger.debug("Env %d wait failed: %s", i, e)
            # wait steps don't count toward effective step budget
            if not (combat_ids or nc_ids or auto_ids or terminal_ids):
                continue  # pure wait iteration, skip effective step counting

        # --- Handle auto-progress screens ---
        auto_actions = {}
        for i in auto_ids:
            legal = _get_legal_actions(envs[i].state)
            if legal:
                auto_actions[i] = legal[0]
            else:
                # No legal actions on auto screen — send wait to advance
                auto_actions[i] = {"action": "wait"}

        # --- Batch combat inference ---
        combat_features = {}
        combat_action_features = {}
        if combat_ids:
            sf_list = []
            af_list = []
            for i in combat_ids:
                state = envs[i].state
                legal = _get_legal_actions(state)

                # Track combat entry
                st = (state.get("state_type") or "").lower()
                if not envs[i].in_combat:
                    envs[i].in_combat = True
                    envs[i].combats += 1
                    envs[i].combat_room_type = _detect_room_type(st, state)
                    player = state.get("player") or {}
                    envs[i].hp_at_combat_start = int(player.get("hp", player.get("current_hp", 0)) or 0)
                    if envs[i].combat_room_type == "boss":
                        envs[i].boss_reached = True
                        envs[i].deck_size_at_boss.append(_extract_deck_size(state))

                sf = build_combat_features(state, vocab)
                af = build_combat_action_features(state, legal, vocab)
                sf_list.append(sf)
                af_list.append(af)
                combat_features[i] = sf
                combat_action_features[i] = af

            # Use intersection of keys for batching (handle optional deck_ids)
            common_sf_keys = set(sf_list[0].keys())
            for sf in sf_list[1:]:
                common_sf_keys &= set(sf.keys())
            sf_filtered = [{k: sf[k] for k in common_sf_keys} for sf in sf_list]

            sf_batch = _batch_np_to_tensor(sf_filtered, device)
            af_batch = _batch_np_to_tensor(af_list, device)

            with torch.no_grad():
                logits_batch, values_batch = combat_net(sf_batch, af_batch)

            # Sample actions
            for batch_idx, i in enumerate(combat_ids):
                legal = _get_legal_actions(envs[i].state)
                af = combat_action_features[i]
                mask = torch.tensor(af["action_mask"], dtype=torch.float32, device=device)
                logits = logits_batch[batch_idx] + (1.0 - mask) * (-1e9)
                dist = torch.distributions.Categorical(logits=logits)
                action_idx_t = dist.sample()
                log_prob = dist.log_prob(action_idx_t).cpu().item()
                action_idx = action_idx_t.cpu().item()
                value = values_batch[batch_idx].cpu().item()

                if action_idx < len(legal):
                    action = legal[action_idx]
                else:
                    action = legal[0]; action_idx = 0

                auto_actions[i] = action

                # Store combat PPO data
                combat_transitions.append({
                    "env_id": i,
                    "state_features": {k: combat_features[i][k] for k in common_sf_keys},
                    "action_features": combat_action_features[i],
                    "action_idx": action_idx,
                    "log_prob": log_prob,
                    "value": value,
                })

        # --- Batch non-combat inference ---
        if nc_ids:
            ss_list = []
            sa_list = []
            nc_legal_actions = {}
            for i in nc_ids:
                state = envs[i].state
                legal = _get_legal_actions(state)
                nc_legal_actions[i] = legal

                # Combat exit handling
                if envs[i].in_combat:
                    envs[i].in_combat = False
                    # fight_summary for combat win
                    player = state.get("player") or {}
                    hp_now = int(player.get("hp", player.get("current_hp", 0)) or 0)
                    max_hp = int(player.get("max_hp", 80) or 80)
                    room_type = envs[i].combat_room_type
                    boss_frac = max(envs[i].boss_hp_peak, _estimate_boss_hp_fraction(state)) if room_type == "boss" else 0.0
                    feedback = fight_summary(envs[i].hp_at_combat_start, hp_now, max_hp, won=True, room_type=room_type, boss_hp_fraction_dealt=boss_frac)
                    ppo_transitions.append({
                        "env_id": i, "type": "fight_win", "reward": feedback,
                    })

                ss = build_structured_state(state, vocab)
                sa = build_structured_actions(state, legal, vocab)
                ss_list.append(ss)
                sa_list.append(sa)

            # Batch PPO forward — use existing serialization helpers
            ss_np_list = [_structured_state_to_numpy_dict(ss) for ss in ss_list]
            sa_np_list = [_structured_actions_to_numpy_dict(sa) for sa in sa_list]

            ss_batch = _batch_np_to_tensor(ss_np_list, device)
            sa_batch = _batch_np_to_tensor(sa_np_list, device)

            ppo_net.eval()
            with torch.no_grad():
                logits_batch, values_batch = ppo_net(ss_batch, sa_batch)[:2]

            for batch_idx, i in enumerate(nc_ids):
                legal = nc_legal_actions[i]
                n_legal = len(legal)
                logits = logits_batch[batch_idx, :n_legal]
                dist = torch.distributions.Categorical(logits=logits)
                action_idx_t = dist.sample()
                log_prob = dist.log_prob(action_idx_t).cpu().item()
                action_idx = action_idx_t.cpu().item()
                value = values_batch[batch_idx].cpu().item()

                if action_idx < len(legal):
                    action = legal[action_idx]
                else:
                    action = legal[0]; action_idx = 0

                auto_actions[i] = action

                # Track card reward behavior
                st = (envs[i].state.get("state_type") or "").lower()
                if st == "card_reward":
                    act_name = (action.get("action") or "").lower()
                    if "skip" in act_name:
                        envs[i].cards_skipped += 1
                    else:
                        label = action.get("label") or action.get("card_id") or ""
                        if label:
                            envs[i].cards_taken.append(label)

                # Reward shaping
                reward = 0.0
                if envs[i].prev_state:
                    reward = shaped_reward(envs[i].prev_state, envs[i].state, 0.0, done=False)
                    if screen_local_delta:
                        reward += screen_local_delta_reward(envs[i].prev_state, envs[i].state, st)

                ppo_transitions.append({
                    "env_id": i,
                    "type": "nc_step",
                    "state_features": ss_np_list[batch_idx],
                    "action_features": sa_np_list[batch_idx],
                    "action_idx": action_idx,
                    "log_prob": log_prob,
                    "value": value,
                    "reward": reward,
                    "screen_type": st,
                })

        # --- Update floor tracking ---
        for i in active:
            if i not in terminal_ids:
                run = envs[i].state.get("run") or {}
                envs[i].floors = max(envs[i].floors, int(run.get("floor", 0) or 0))
                if envs[i].combat_room_type == "boss" and envs[i].in_combat:
                    envs[i].boss_hp_peak = max(envs[i].boss_hp_peak, _estimate_boss_hp_fraction(envs[i].state))

        # Count effective steps (actual decisions, not waits)
        if combat_ids or nc_ids:
            effective_steps += 1

        # --- Parallel step all envs that have actions ---
        step_ids = [i for i in active if i in auto_actions and i not in terminal_ids]
        if step_ids:
            # Save prev state before stepping
            for i in step_ids:
                envs[i].prev_state = envs[i].state

            step_futures = {
                pool.submit(clients[i].act, auto_actions[i]): i
                for i in step_ids
            }
            for f in as_completed(step_futures):
                i = step_futures[f]
                try:
                    envs[i].state = f.result()
                    envs[i].step_count += 1
                except Exception as e:
                    envs[i].done = True
                    envs[i].error = str(e)
                    logger.warning("Env %d step failed: %s", i, e)

    pool.shutdown(wait=False)

    # --- Build per-env stats ---
    stats_list = []
    for i, env in enumerate(envs):
        stats_list.append({
            "env_id": i,
            "floors": env.floors,
            "combats": env.combats,
            "outcome": env.outcome,
            "boss_reached": env.boss_reached,
            "boss_hp_peak": env.boss_hp_peak,
            "deck_size_at_boss": env.deck_size_at_boss,
            "cards_taken": env.cards_taken,
            "cards_skipped": env.cards_skipped,
            "steps": env.step_count,
            "error": env.error,
        })

    return ppo_transitions, combat_transitions, stats_list
