#!/usr/bin/env python3
"""Unified PPO + MCTS hybrid training loop.

Trains both brains simultaneously in shared episodes:
- Non-combat screens → PPO inference + data collection
- Combat screens → MCTS search + data collection

Usage:
    # Single env (testing):
    python train_hybrid.py --pipe --num-envs 1 --start-port 15527 --max-iterations 5

    # 8 parallel envs (production):
    python train_hybrid.py --pipe --num-envs 8 --start-port 15527 --max-iterations 500

    # Resume from checkpoints:
    python train_hybrid.py --pipe --num-envs 8 --resume-ppo ppo_best.pt --resume-mcts mcts_best.pt

    # Or use the launch script:
    .\\scripts\\start-mcts-training.ps1  (update to call train_hybrid.py)
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds STS2AI/Python library dirs to sys.path)

import argparse
import atexit
import json
import logging
import random
import signal
import sys
import time
import traceback
from collections import deque
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vocab import load_vocab, Vocab
from rl_encoder_v2 import build_structured_state, build_structured_actions
from rl_policy_v2 import (
    FullRunPolicyNetworkV2,
    PPOTrainerV2,
    StructuredRolloutBuffer,
    _structured_state_to_numpy_dict,
    _structured_actions_to_numpy_dict,
)
from rl_reward_shaping import (
    boss_readiness_score,
    shaped_reward,
    combat_step_reward,
    combat_local_tactical_reward,
    compute_combat_feedback,
    screen_local_delta_reward,
    _extract_player,
    _safe_int,
)
from rl_segment_buffer import SegmentRolloutBuffer, Segment
from segment_collector import NonCombatSegmentCollector
from counterfactual_scoring import compute_counterfactual_reward
from combat_nn import (
    CombatPolicyValueNetwork,
    build_combat_features,
    build_combat_action_features,
    MAX_ACTIONS,
)
from mcts_core import MCTSConfig, mcts_search
from combat_mcts_agent import CombatMCTSAgent, PipeCombatForwardModel
from full_run_env import ApiBackedFullRunClient, PipeBackedFullRunClient, create_full_run_client
from headless_sim_runner import DEFAULT_DLL_PATH, start_headless_sim, stop_process
from sts2ai_paths import ARTIFACTS_ROOT, DATASETS_ROOT, MAINLINE_CHECKPOINT, REPO_ROOT
from training_health import TrainingHealthMonitor
from episode_data_saver import EpisodeDataSaver
from runtime.full_run_action_semantics import (
    legal_action_name_set as _shared_legal_action_name_set,
    is_selection_screen as _shared_is_selection_screen,
    choose_auto_progress_action as _shared_choose_auto_progress_action,
    combat_rewards_state as _shared_combat_rewards_state,
    reward_item_claimable as _shared_reward_item_claimable,
    choose_claimable_reward_action as _shared_choose_claimable_reward_action,
    claim_reward_action_count as _shared_claim_reward_action_count,
    reward_claim_signature as _shared_reward_claim_signature,
    next_reward_claim_signature as _shared_next_reward_claim_signature,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_shutdown_requested = False
DEFAULT_REPO_ROOT = REPO_ROOT
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "hybrid_training"
DEFAULT_MATCHUP_DATA_DIR = DATASETS_ROOT / "card_ranking_post_wizardly"
DEFAULT_COMBAT_TEACHER_DATA = DATASETS_ROOT / "combat_teacher_post_wizardly" / "teacher.jsonl"


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested (signal %d)...", signum)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}
ACT1_CLEAR_FLOOR = 17
SELECTION_ACTION_NAMES = {
    "select_card",
    "combat_select_card",
    "combat_confirm_selection",
    "confirm_selection",
    "cancel_selection",
    "skip_relic_selection",
}
SELECTION_SCREENS = {"card_select", "hand_select", "relic_select"}
POST_CARD_REWARD_ACTIONS = {"select_card_reward", "skip_card_reward"}
REWARD_FLOW_SCREENS = {"combat_rewards", "card_reward"}


def _legal_action_name_set(legal: list[dict[str, Any]]) -> set[str]:
    return _shared_legal_action_name_set(legal)


def _is_selection_screen(state_type: str, legal: list[dict[str, Any]]) -> bool:
    return _shared_is_selection_screen(state_type, legal)


def _choose_auto_progress_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    last_action_name: str | None = None,
    last_reward_claim_sig: str | None = None,
    last_reward_claim_count: int | None = None,
    reward_chain_card_reward_seen: bool = False,
) -> dict[str, Any] | None:
    return _shared_choose_auto_progress_action(
        state,
        legal,
        last_action_name=last_action_name,
        last_reward_claim_sig=last_reward_claim_sig,
        last_reward_claim_count=last_reward_claim_count,
        reward_chain_card_reward_seen=reward_chain_card_reward_seen,
    )


def _choose_empty_legal_recovery_action(
    state: dict[str, Any],
    last_action_name: str | None = None,
) -> dict[str, Any] | None:
    st = (state.get("state_type") or "").strip().lower()
    last_action_name = str(last_action_name or "").strip().lower()

    if st == "event":
        event_state = state.get("event") or {}
        if event_state.get("in_dialogue"):
            return {"action": "advance_dialogue"}
        if event_state.get("can_proceed") or event_state.get("is_finished"):
            return {"action": "proceed"}
        if last_action_name in POST_CARD_REWARD_ACTIONS and (
            event_state.get("can_proceed") or event_state.get("is_finished")
        ):
            return {"action": "proceed"}

    if st == "combat_rewards":
        rewards_state = state.get("combat_rewards") or state.get("rewards") or {}
        if rewards_state.get("can_proceed"):
            return {"action": "proceed"}

    if st in {"treasure", "rest", "rest_site", "shop"}:
        return {"action": "proceed"}

    return None


def _choose_rest_site_repeat_escape_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> dict[str, Any] | None:
    st = (state.get("state_type") or "").strip().lower()
    if st != "rest_site":
        return None

    rest_state = state.get("rest_site")
    if not isinstance(rest_state, dict):
        rest_state = {}

    option_by_index: dict[int, dict[str, Any]] = {}
    for option in rest_state.get("options") or []:
        if not isinstance(option, dict):
            continue
        try:
            option_index = int(option.get("index", -1))
        except Exception:
            continue
        option_by_index[option_index] = option

    def _action_name(action: dict[str, Any]) -> str:
        return str(action.get("action") or "").strip().lower()

    def _normalized_action_text(action: dict[str, Any]) -> str:
        parts: list[str] = []
        parts.append(str(action.get("label") or ""))
        try:
            option_index = int(action.get("index", -1))
        except Exception:
            option_index = -1
        option = option_by_index.get(option_index) or {}
        for key in ("id", "name", "description"):
            parts.append(str(option.get(key) or ""))
        return " ".join(part.strip().lower() for part in parts if str(part).strip())

    proceed = next((action for action in legal if _action_name(action) == "proceed"), None)
    rest_actions = [action for action in legal if _action_name(action) == "choose_rest_option"]
    if not rest_actions:
        return proceed

    for action in rest_actions:
        normalized = _normalized_action_text(action)
        if any(token in normalized for token in ("rest", "heal", "sleep", "recover")):
            return action

    if proceed is not None:
        return proceed

    return min(rest_actions, key=lambda action: int(action.get("index", 0)))


def _combat_rewards_state(state: dict[str, Any]) -> dict[str, Any]:
    return _shared_combat_rewards_state(state)


def _reward_item_claimable(state: dict[str, Any], reward_item: dict[str, Any] | None) -> bool:
    return _shared_reward_item_claimable(state, reward_item)


def _choose_claimable_reward_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    *,
    prefer_highest_index: bool = False,
) -> dict[str, Any] | None:
    return _shared_choose_claimable_reward_action(
        state,
        legal,
        prefer_highest_index=prefer_highest_index,
    )


def _claim_reward_action_count(legal: list[dict[str, Any]]) -> int:
    return _shared_claim_reward_action_count(legal)


def _reward_claim_signature(state: dict[str, Any], action: dict[str, Any] | None) -> str:
    return _shared_reward_claim_signature(state, action)


def _next_reward_claim_signature(
    state_type: str,
    state: dict[str, Any],
    action: dict[str, Any] | None,
) -> str:
    return _shared_next_reward_claim_signature(state_type, state, action)


def _combat_player_view(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle")
    if isinstance(battle, dict):
        player = battle.get("player")
        if isinstance(player, dict):
            return player
    player = state.get("player")
    return player if isinstance(player, dict) else {}


def _combat_hand_summary(state: dict[str, Any], max_cards: int = 6) -> str:
    battle = state.get("battle") or {}
    hand = battle.get("hand") or _combat_player_view(state).get("hand") or []
    if not isinstance(hand, list) or not hand:
        return "-"
    parts: list[str] = []
    for card in hand[:max_cards]:
        if not isinstance(card, dict):
            continue
        name = str(card.get("name") or card.get("label") or card.get("id") or "?")
        cost = card.get("cost")
        parts.append(f"{name}({cost})")
    more = "" if len(hand) <= max_cards else f"+{len(hand) - max_cards}"
    return ",".join(parts) + more


def _combat_enemy_intent_summary(state: dict[str, Any], max_enemies: int = 3) -> str:
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    if not isinstance(enemies, list) or not enemies:
        return "-"
    parts: list[str] = []
    for enemy in enemies[:max_enemies]:
        if not isinstance(enemy, dict):
            continue
        name = str(enemy.get("name") or enemy.get("id") or "?")
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        block = _safe_int(enemy.get("block", 0), 0)
        intents = enemy.get("intents") or []
        if isinstance(intents, list) and intents:
            intent0 = intents[0] if isinstance(intents[0], dict) else {}
        else:
            intent0 = {}
        intent_type = str(intent0.get("type") or intent0.get("label") or "?")
        dmg = _safe_int(intent0.get("damage", 0), 0)
        hits = max(1, _safe_int(intent0.get("hits", 1), 1))
        dmg_str = f"{dmg}x{hits}" if dmg > 0 else intent_type
        parts.append(f"{name}[{hp}/{block}:{dmg_str}]")
    more = "" if len(enemies) <= max_enemies else f"+{len(enemies) - max_enemies}"
    return ",".join(parts) + more


def _action_target_summary(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    for key in ("target", "target_id", "enemy_id", "slot"):
        value = action.get(key)
        if value not in (None, ""):
            return f" target={value}"
    return ""


def _topk_action_summary(
    legal: list[dict[str, Any]],
    logits_or_probs: np.ndarray | list[float] | torch.Tensor,
    k: int = 3,
    already_probs: bool = False,
) -> str:
    if not legal:
        return "-"
    if isinstance(logits_or_probs, torch.Tensor):
        arr = logits_or_probs.detach().float().cpu().numpy()
    else:
        arr = np.asarray(logits_or_probs, dtype=np.float64)
    arr = np.ravel(arr)
    if arr.size == 0:
        return "-"
    arr = arr[:len(legal)]
    if arr.size == 0:
        return "-"
    if already_probs:
        probs = arr
    else:
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        denom = np.sum(exp)
        probs = exp / denom if denom > 0 else np.zeros_like(arr)
    order = np.argsort(-probs)[: min(k, len(legal))]
    parts: list[str] = []
    for idx in order:
        action = legal[int(idx)]
        label = str(action.get("label") or action.get("action") or idx)
        parts.append(f"{label}:{float(probs[int(idx)]):.2f}")
    return " | ".join(parts)


def _resolve_counterfactual_runtime(
    use_segment_collector: bool,
    counterfactual_scoring: bool,
    counterfactual_weight: float,
) -> tuple[bool, float, list[str]]:
    """Resolve the actually effective counterfactual settings."""
    warnings: list[str] = []
    effective_scoring = bool(counterfactual_scoring and use_segment_collector)
    effective_weight = float(counterfactual_weight) if effective_scoring else 0.0

    if counterfactual_weight > 0 and not counterfactual_scoring:
        warnings.append(
            "counterfactual_weight > 0 but counterfactual_scoring is disabled; "
            "effective counterfactual weight is 0.0."
        )
    if counterfactual_scoring and not use_segment_collector:
        warnings.append(
            "counterfactual_scoring requires --use-segment-collector; legacy step-by-step PPO "
            "does not consume counterfactual reward, so effective counterfactual scoring is disabled."
        )

    return effective_scoring, effective_weight, warnings


def _configure_boss_aware_warmup(model: FullRunPolicyNetworkV2) -> tuple[int, int]:
    """Freeze PPO backbone and train only newly added boss-aware modules."""
    trainable_prefixes = (
        "entity_emb.text_token_embed",
        "boss_screen_adapter",
        "boss_readiness_head",
    )
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        allow = any(name.startswith(prefix) for prefix in trainable_prefixes)
        param.requires_grad = allow
        if allow:
            trainable_params += param.numel()
    return trainable_params, total_params


def _infer_retrieval_proj_dim(state_dict: dict[str, Any]) -> int:
    """Detect whether a checkpoint was trained with --retrieval-head.

    Returns the symbolic proj_dim if the head is present, else 0.
    Looks for `symbolic_head.out_proj.weight` which has shape (proj_dim, embed_dim).
    """
    if not isinstance(state_dict, dict):
        return 0
    out_proj = state_dict.get("symbolic_head.out_proj.weight")
    if isinstance(out_proj, torch.Tensor) and out_proj.ndim == 2:
        return int(out_proj.shape[0])
    return 0


def _checkpoint_retrieval_proj_dim(ckpt: dict[str, Any]) -> int:
    """Infer retrieval proj dim from any supported checkpoint payload layout."""
    if not isinstance(ckpt, dict):
        return 0
    for key in ("ppo_model", "model_state_dict", "mcts_model"):
        proj_dim = _infer_retrieval_proj_dim(ckpt.get(key, {}))
        if proj_dim > 0:
            return proj_dim
    return 0


def _extract_deck_size(state: dict[str, Any]) -> int:
    player = _extract_player(state)
    deck = player.get("deck") if isinstance(player.get("deck"), list) else []
    return len(deck)


def _detect_combat_room_type(state_type: str, state: dict[str, Any]) -> str:
    st = str(state_type or "").strip().lower()
    if st in ("boss", "elite", "monster"):
        return st
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or []
    for enemy in enemies:
        etype = str((enemy or {}).get("type", "")).strip().lower()
        if etype.startswith("boss"):
            return "boss"
        if etype.startswith("elite"):
            return "elite"
    return "monster"


def _estimate_boss_hp_fraction(state: dict[str, Any]) -> float:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or []
    total_hp = 0
    total_max_hp = 0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        total_hp += _safe_int(enemy.get("hp", enemy.get("current_hp", 0)))
        total_max_hp += max(1, _safe_int(enemy.get("max_hp", 1)))
    if total_max_hp <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - total_hp / total_max_hp))


# ---------------------------------------------------------------------------
# MCTS replay buffer (same as train_combat_mcts.py)
# ---------------------------------------------------------------------------

@dataclass
class MCTSTrainingExample:
    state_features: dict[str, np.ndarray]
    action_features: dict[str, np.ndarray]
    mcts_policy: np.ndarray
    outcome: float


class MCTSReplayBuffer:
    def __init__(self, max_size: int = 50000):
        self.buffer: deque[MCTSTrainingExample] = deque(maxlen=max_size)

    def add(self, ex: MCTSTrainingExample):
        self.buffer.append(ex)

    def sample(self, n: int) -> list[MCTSTrainingExample]:
        idx = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Combat PPO rollout buffer (per-step data for PPO training of combat NN)
# ---------------------------------------------------------------------------

@dataclass
class CombatRolloutBuffer:
    """Lightweight buffer for combat PPO steps.

    Stores per-step combat data: state/action features, chosen action index,
    log_prob from sampling, per-step shaped reward, value estimate, done flag.
    GAE is computed before training.
    """

    state_features: list[dict[str, np.ndarray]] = field(default_factory=list)
    action_features: list[dict[str, np.ndarray]] = field(default_factory=list)
    action_indices: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    screen_types: list[str] = field(default_factory=list)  # encounter type per step

    # Computed after collection
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def add(
        self,
        sf: dict[str, np.ndarray],
        af: dict[str, np.ndarray],
        action_idx: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        screen_type: str = "",
    ) -> None:
        self.state_features.append(sf)
        self.action_features.append(af)
        self.action_indices.append(action_idx)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.screen_types.append(screen_type)

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        """Compute GAE advantages and returns.

        Note: combat NN value head uses Tanh (output in [-1, 1]).
        GAE computation is standard — the bounded output just means
        value targets (returns) will naturally stay in a reasonable range.
        """
        n = len(self.rewards)
        self.advantages = [0.0] * n
        self.returns = [0.0] * n
        last_gae = 0.0

        for t in reversed(range(n)):
            if self.dones[t]:
                next_value = 0.0
                last_gae = 0.0
            elif t + 1 < n:
                next_value = self.values[t + 1]
            else:
                next_value = 0.0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * lam * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def to_tensors(self, device: torch.device | None = None) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Convert buffer to tensors for training."""
        n = len(self.rewards)

        # Stack state tensors
        state_tensors: dict[str, torch.Tensor] = {}
        if n > 0:
            for key in self.state_features[0]:
                arrays = [s[key] for s in self.state_features]
                arr = np.stack(arrays)
                if arr.dtype in (np.int64, np.int32):
                    state_tensors[key] = torch.tensor(arr, dtype=torch.long)
                elif arr.dtype == bool:
                    state_tensors[key] = torch.tensor(arr, dtype=torch.bool)
                else:
                    state_tensors[key] = torch.tensor(arr, dtype=torch.float32)

        # Stack action tensors
        action_tensors: dict[str, torch.Tensor] = {}
        if n > 0:
            for key in self.action_features[0]:
                arrays = [a[key] for a in self.action_features]
                arr = np.stack(arrays)
                if arr.dtype in (np.int64, np.int32):
                    action_tensors[key] = torch.tensor(arr, dtype=torch.long)
                elif arr.dtype == bool:
                    action_tensors[key] = torch.tensor(arr, dtype=torch.bool)
                else:
                    action_tensors[key] = torch.tensor(arr, dtype=torch.float32)

        result = {
            "state_tensors": state_tensors,
            "action_tensors": action_tensors,
            "actions": torch.tensor(self.action_indices, dtype=torch.long),
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "advantages": torch.tensor(self.advantages, dtype=torch.float32),
            "returns": torch.tensor(self.returns, dtype=torch.float32),
        }
        if device is not None:
            for k, v in result.items():
                if isinstance(v, dict):
                    result[k] = {kk: vv.to(device) for kk, vv in v.items()}
                else:
                    result[k] = v.to(device)
        return result

    def clear(self) -> None:
        for attr in ("state_features", "action_features", "action_indices",
                      "log_probs", "rewards", "values", "dones",
                      "advantages", "returns"):
            getattr(self, attr).clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Combat PPO Trainer
# ---------------------------------------------------------------------------

class CombatPPOTrainer:
    """PPO update for the combat neural network.

    Uses the same clipped surrogate + GAE approach as PPOTrainerV2,
    adapted for the combat NN's input format (combat features, not structured state).
    """

    def __init__(
        self,
        network: CombatPolicyValueNetwork,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.05,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    def update(self, buffer: CombatRolloutBuffer) -> dict[str, float]:
        """Run PPO update on the combat buffer. Returns loss metrics."""
        buffer.compute_gae()
        device = next(self.network.parameters()).device
        data = buffer.to_tensors(device)

        state_tensors = data["state_tensors"]
        action_tensors = data["action_tensors"]
        old_actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]

        # Normalize advantages
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        n = len(old_actions)
        total_ploss = 0.0
        total_vloss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _epoch in range(self.ppo_epochs):
            indices = torch.randperm(n, device=device)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                mb_idx = indices[start:end]

                # Slice minibatch
                mb_state = {k: v[mb_idx] for k, v in state_tensors.items()}
                mb_action = {k: v[mb_idx] for k, v in action_tensors.items()}
                mb_old_actions = old_actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Forward
                logits, values = self.network(mb_state, mb_action)

                # Compute new log_probs from Categorical
                mask = mb_action["action_mask"].float()
                logits_masked = logits + (1.0 - mask) * (-1e9)
                dist = torch.distributions.Categorical(logits=logits_masked)
                new_log_probs = dist.log_prob(mb_old_actions)
                entropy = dist.entropy().mean()

                # PPO clipped ratio
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clamp returns to [-1, 1] to match Tanh output)
                mb_returns_clamped = mb_returns.clamp(-1.0, 1.0)
                value_loss = F.mse_loss(values, mb_returns_clamped)

                # Combined loss
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        num_updates = max(num_updates, 1)
        return {
            "combat_ppo_ploss": total_ploss / num_updates,
            "combat_ppo_vloss": total_vloss / num_updates,
            "combat_entropy": total_entropy / num_updates,
        }


# ---------------------------------------------------------------------------
# Multi-process episode worker
# ---------------------------------------------------------------------------

def _mp_episode_worker(
    worker_id: int,
    port: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    inf_request_queue: mp.Queue,
    inf_result_queue: mp.Queue,
    character_id: str,
    episode_timeout: float,
    max_episode_steps: int,
    transport: str = "pipe",
    boss_entry_quality_weight: float = 0.0,
    early_damage_potion_penalty_weight: float = 0.0,
):
    """Worker process: collects episodes with its own pipe client.

    Runs in a loop: reads tasks from task_queue, collects episodes using
    InferenceClient for NN, sends results back via result_queue.
    """
    import logging
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger(f"worker-{worker_id}")

    from inference_server import InferenceClient
    from full_run_env import create_full_run_client

    try:
        client = create_full_run_client(port=port, use_pipe=True, transport=transport, ready_timeout_s=15.0)
        client._ensure_connected()
    except Exception as e:
        log.error("Worker %d: pipe connect failed: %s", worker_id, e)
        result_queue.put((worker_id, None, None, {"error": f"connect: {e}"}))
        return

    inf_client = InferenceClient(worker_id, inf_request_queue, inf_result_queue)
    vocab = load_vocab()

    while True:
        try:
            task = task_queue.get(timeout=5.0)
        except Exception:
            continue

        if task is None:  # shutdown sentinel
            break

        try:
            # task is just a signal to collect one episode
            ep_ppo, ep_mcts, ep_stats = collect_unified_episode(
                ppo_network=None,  # not used when inference_client is set
                mcts_agent=None,
                vocab=vocab,
                pipe=None,
                client=client,
                character_id=character_id,
                episode_timeout=episode_timeout,
                max_steps=max_episode_steps,
                inference_client=inf_client,
                boss_entry_quality_weight=boss_entry_quality_weight,
                early_damage_potion_penalty_weight=early_damage_potion_penalty_weight,
            )
            result_queue.put((worker_id, ep_ppo, ep_mcts, ep_stats))
        except Exception as e:
            log.warning("Worker %d episode failed: %s", worker_id, e)
            result_queue.put((worker_id, None, None, {"error": str(e)}))

    try:
        client.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Unified episode collection
# ---------------------------------------------------------------------------

def _export_ppo_actor_onnx(net, output_path: str, vocab) -> None:
    """Export PPO actor-only ONNX for ORT CPU inference."""
    import torch.nn as nn
    from rl_encoder_v2 import (SCALAR_DIM, MAX_DECK_SIZE, MAX_RELICS, MAX_POTIONS,
        MAX_HAND_SIZE, MAX_ENEMIES, MAX_MAP_NODES, MAX_CARD_REWARDS, MAX_SHOP_ITEMS,
        MAX_REST_OPTIONS, MAX_ACTIONS, CARD_AUX_DIM, ENEMY_AUX_DIM, NUM_RELIC_TAGS, MAP_ROUTE_DIM)

    class Wrapper(nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network
        def forward(self, scalars, deck_ids, deck_aux, deck_mask, relic_ids, relic_aux, relic_mask,
                    potion_ids, potion_mask, hand_ids, hand_aux, hand_mask,
                    enemy_ids, enemy_aux, enemy_mask,
                    screen_type_idx, next_boss_idx,
                    map_node_types, map_node_mask, map_route_features,
                    reward_card_ids, reward_card_aux, reward_card_mask,
                    shop_card_ids, shop_relic_ids, shop_potion_ids, shop_prices, shop_mask,
                    event_option_count, rest_option_ids, rest_option_mask,
                    action_type_ids, target_card_ids, target_enemy_ids,
                    target_node_types, target_indices, action_mask):
            ss = {'scalars': scalars, 'deck_ids': deck_ids, 'deck_aux': deck_aux,
                  'deck_mask': deck_mask.bool(), 'relic_ids': relic_ids, 'relic_aux': relic_aux,
                  'relic_mask': relic_mask.bool(), 'potion_ids': potion_ids,
                  'potion_mask': potion_mask.bool(), 'hand_ids': hand_ids, 'hand_aux': hand_aux,
                  'hand_mask': hand_mask.bool(), 'enemy_ids': enemy_ids, 'enemy_aux': enemy_aux,
                  'enemy_mask': enemy_mask.bool(), 'screen_type_idx': screen_type_idx.long(),
                  'next_boss_idx': next_boss_idx.long(),
                  'map_node_types': map_node_types, 'map_node_mask': map_node_mask.bool(),
                  'map_route_features': map_route_features,
                  'reward_card_ids': reward_card_ids, 'reward_card_aux': reward_card_aux,
                  'reward_card_mask': reward_card_mask.bool(),
                  'shop_card_ids': shop_card_ids, 'shop_relic_ids': shop_relic_ids,
                  'shop_potion_ids': shop_potion_ids, 'shop_prices': shop_prices,
                  'shop_mask': shop_mask.bool(),
                  'event_option_count': event_option_count.long(),
                  'rest_option_ids': rest_option_ids, 'rest_option_mask': rest_option_mask.bool()}
            sa = {'action_type_ids': action_type_ids, 'target_card_ids': target_card_ids,
                  'target_enemy_ids': target_enemy_ids, 'target_node_types': target_node_types,
                  'target_indices': target_indices, 'action_mask': action_mask.bool()}
            logits = self.network(ss, sa)[0]
            return logits

    wrapper = Wrapper(net).cpu().eval()
    B = 1
    inputs = [
        torch.randn(B, SCALAR_DIM), torch.zeros(B, MAX_DECK_SIZE, dtype=torch.int64),
        torch.zeros(B, MAX_DECK_SIZE, CARD_AUX_DIM), torch.ones(B, MAX_DECK_SIZE),
        torch.zeros(B, MAX_RELICS, dtype=torch.int64), torch.zeros(B, MAX_RELICS, NUM_RELIC_TAGS),
        torch.zeros(B, MAX_RELICS), torch.zeros(B, MAX_POTIONS, dtype=torch.int64),
        torch.zeros(B, MAX_POTIONS), torch.zeros(B, MAX_HAND_SIZE, dtype=torch.int64),
        torch.zeros(B, MAX_HAND_SIZE, CARD_AUX_DIM), torch.zeros(B, MAX_HAND_SIZE),
        torch.zeros(B, MAX_ENEMIES, dtype=torch.int64), torch.zeros(B, MAX_ENEMIES, ENEMY_AUX_DIM),
        torch.zeros(B, MAX_ENEMIES), torch.tensor([4], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64), torch.zeros(B, MAX_MAP_NODES, dtype=torch.int64),
        torch.ones(B, MAX_MAP_NODES), torch.zeros(B, MAX_MAP_NODES, MAP_ROUTE_DIM),
        torch.zeros(B, MAX_CARD_REWARDS, dtype=torch.int64), torch.zeros(B, MAX_CARD_REWARDS, CARD_AUX_DIM),
        torch.zeros(B, MAX_CARD_REWARDS), torch.zeros(B, MAX_SHOP_ITEMS, dtype=torch.int64),
        torch.zeros(B, MAX_SHOP_ITEMS, dtype=torch.int64), torch.zeros(B, MAX_SHOP_ITEMS, dtype=torch.int64),
        torch.zeros(B, MAX_SHOP_ITEMS), torch.zeros(B, MAX_SHOP_ITEMS),
        torch.tensor([0], dtype=torch.int64), torch.zeros(B, MAX_REST_OPTIONS, dtype=torch.int64),
        torch.zeros(B, MAX_REST_OPTIONS), torch.zeros(B, MAX_ACTIONS, dtype=torch.int64),
        torch.zeros(B, MAX_ACTIONS, dtype=torch.int64), torch.zeros(B, MAX_ACTIONS, dtype=torch.int64),
        torch.zeros(B, MAX_ACTIONS, dtype=torch.int64), torch.zeros(B, MAX_ACTIONS, dtype=torch.int64),
        torch.ones(B, MAX_ACTIONS),
    ]
    names = ['scalars', 'deck_ids', 'deck_aux', 'deck_mask', 'relic_ids', 'relic_aux', 'relic_mask',
             'potion_ids', 'potion_mask', 'hand_ids', 'hand_aux', 'hand_mask',
             'enemy_ids', 'enemy_aux', 'enemy_mask', 'screen_type_idx', 'next_boss_idx',
             'map_node_types', 'map_node_mask', 'map_route_features',
             'reward_card_ids', 'reward_card_aux', 'reward_card_mask',
             'shop_card_ids', 'shop_relic_ids', 'shop_potion_ids', 'shop_prices', 'shop_mask',
             'event_option_count', 'rest_option_ids', 'rest_option_mask',
             'action_type_ids', 'target_card_ids', 'target_enemy_ids',
             'target_node_types', 'target_indices', 'action_mask']
    torch.onnx.export(wrapper, tuple(inputs), output_path, input_names=names,
                      output_names=['policy_logits'], opset_version=17, do_constant_folding=True)


def collect_unified_episode(
    ppo_network: FullRunPolicyNetworkV2,
    mcts_agent: CombatMCTSAgent,
    vocab: Vocab,
    pipe,
    client,
    character_id: str = "IRONCLAD",
    ascension_level: int = 0,
    seed: str | None = None,
    episode_timeout: float = 90.0,
    max_steps: int = 600,
    use_mcts_combat: bool = False,
    combat_buffer: CombatRolloutBuffer | None = None,
    inference_client=None,
    # Phase 2-4 options
    use_segment_collector: bool = False,
    counterfactual_scoring: bool = False,
    counterfactual_weight: float = 0.3,
    screen_local_delta: bool = True,
    deterministic_policy: bool = False,
    episode_saver: EpisodeDataSaver | None = None,
    use_local_ort: bool = False,
    ppo_ort_session=None,  # ORT CPU session for non-combat actor (Branch C)
    # Step 2 / Phase 5 options
    boss_entry_quality_weight: float = 0.0,
    early_damage_potion_penalty_weight: float = 0.0,
) -> tuple[StructuredRolloutBuffer, list[MCTSTrainingExample], dict]:
    """Collect one episode with PPO for non-combat and MCTS/PPO for combat.

    Args:
        combat_buffer: If provided, NN combat steps are collected for PPO training.
            Only NN episodes (use_random_this_step=False) contribute PPO data.
            If None, a local buffer is created (returned via stats["_combat_buffer"]).

    Returns:
        ppo_buffer: non-combat steps for PPO training
        mcts_examples: combat decisions for behavior cloning training (outcomes backfilled)
        stats: episode statistics (includes "_combat_buffer" key for merging)
    """

    ppo_buffer = StructuredRolloutBuffer()
    segment_buffer = SegmentRolloutBuffer() if use_segment_collector else None
    segment_collector = NonCombatSegmentCollector() if use_segment_collector else None
    mcts_pending: list[dict] = []  # outcome=0 until episode end
    if combat_buffer is None:
        combat_buffer = CombatRolloutBuffer()
    stats: dict[str, Any] = {
        "floors": 0, "combats": 0, "combats_won": 0, "ppo_steps": 0,
        "mcts_decisions": 0, "mcts_searches": 0, "combat_ppo_steps": 0,
        "outcome": None, "error": None, "end_reason": None,
        "cards_taken": [], "cards_skipped": 0,
        "card_reward_screens": 0, "card_reward_skips": 0,
        "hp_timeline": [],  # HP after each combat
        "death_enemy": None,
        "boss_reached": False,
        "act1_cleared": False,
        "boss_hp_fraction_dealt": [],
        "deck_size_at_boss": [],
    }
    # Per-step timing diagnostics
    _slow_step_threshold = 1.0  # seconds
    _slow_steps = 0
    _max_step_ms = 0.0
    _timeout_count = 0
    # Episode trace for debugging (kept lightweight — only key events)
    _episode_trace: list[str] = []
    _last_action_key = ""
    _repeat_count = 0
    _MAX_REPEATS = 20  # bail out if same action repeated this many times
    _last_action_name = ""
    _last_reward_claim_sig = ""
    _last_reward_claim_count: int | None = None
    _reward_chain_card_reward_seen = False

    episode_start = time.monotonic()

    try:
        _t0 = time.monotonic()
        state = client.reset(character_id=character_id, ascension_level=ascension_level, seed=seed)
        _dt = time.monotonic() - _t0
        _max_step_ms = max(_max_step_ms, _dt * 1000)
        if _dt >= _slow_step_threshold:
            _slow_steps += 1
            logger.debug("Slow reset: %.0fms", _dt * 1000)
    except Exception as e:
        # Try reconnect + retry once
        try:
            if hasattr(client, "_reconnect"):
                client._reconnect()
            state = client.reset(character_id=character_id, ascension_level=ascension_level, seed=seed)
        except Exception as e2:
            stats["error"] = f"reset: {e2}"
            stats["end_reason"] = "error"
            return ppo_buffer, [], stats

    prev_state = state
    in_combat = False
    _prev_combat_state = state  # init; updated when entering/during combat
    _hp_at_combat_start = 80  # init; updated when entering combat
    _combat_room_type = "monster"
    _boss_hp_frac_peak = 0.0
    _pending_boss_deck_size: int | None = None
    _combat_ppo_pending = None  # pending combat PPO data awaiting next_state

    for step_i in range(max_steps):
        if time.monotonic() - episode_start > episode_timeout:
            stats["error"] = "timeout"
            stats["end_reason"] = "timeout"
            break
        if _shutdown_requested:
            break

        st = (state.get("state_type") or "").lower()
        if st == "card_reward":
            _reward_chain_card_reward_seen = True
        elif st not in REWARD_FLOW_SCREENS:
            _reward_chain_card_reward_seen = False
            _last_reward_claim_count = None

        # Terminal
        if st == "game_over" or state.get("terminal"):
            go = state.get("game_over") or {}
            outcome_str = (go.get("run_outcome") or go.get("outcome") or "").lower()
            terminal_value = 1.0 if ("victory" in outcome_str or outcome_str == "win") else -1.0
            stats["outcome"] = "victory" if terminal_value > 0 else "death"
            stats["end_reason"] = "terminal"
            _p = (state.get("player") or {})
            # Track death info
            if terminal_value < 0:
                _battle = state.get("battle") or {}
                _enemies = _battle.get("enemies") or []
                if _enemies:
                    stats["death_enemy"] = "+".join(
                        [e.get("id", "?")[:12] for e in _enemies[:4]])
            _episode_trace.append(
                f"[{step_i}] TERMINAL: {stats['outcome']} floor={stats['floors']} "
                f"hp={_p.get('hp',0)}/{_p.get('max_hp',0)} death_by={stats.get('death_enemy','N/A')}")

            # Final PPO step (done=True)
            reward = shaped_reward(
                prev_state, state, terminal_value, done=True,
                boss_entry_quality_weight=boss_entry_quality_weight,
                early_damage_potion_penalty_weight=early_damage_potion_penalty_weight,
            )

            if segment_collector is not None and segment_collector.is_open and segment_buffer is not None:
                # Phase 2: accumulate terminal reward and close segment
                segment_collector.add_reward(reward, tag="terminal", steps=1)
                seg = segment_collector.close_segment(done=True)
                if seg is not None:
                    segment_buffer.add(seg)
            elif len(ppo_buffer) > 0:
                # Legacy: mark last step done
                ppo_buffer.dones[-1] = True
                ppo_buffer.rewards[-1] = reward

            # Backfill MCTS outcomes
            for ex_data in mcts_pending:
                ex_data["outcome"] = terminal_value

            # Combat death feedback → non-combat decisions
            if in_combat and terminal_value < 0:
                # Determine room type and boss damage for shaped feedback
                _death_room_type = _combat_room_type or "monster"
                _boss_hp_frac_dealt = 0.0
                if _death_room_type == "boss":
                    _boss_hp_frac_dealt = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))
                    stats["boss_reached"] = True
                    stats["boss_hp_fraction_dealt"].append(_boss_hp_frac_dealt)
                _feedback = compute_combat_feedback(
                    _hp_at_combat_start, 0, _safe_int(_extract_player(state).get("max_hp", 80)),
                    won=False, room_type=_death_room_type,
                    boss_hp_fraction_dealt=_boss_hp_frac_dealt)
                if segment_collector is not None and segment_collector.is_open and segment_buffer is not None:
                    segment_collector.add_reward(_feedback, tag="fight_death", steps=1)
                    seg = segment_collector.close_segment(done=True)
                    if seg is not None:
                        segment_buffer.add(seg)
                elif len(ppo_buffer) > 0:
                    _fb_steps = min(10, len(ppo_buffer))
                    for _fi in range(len(ppo_buffer) - _fb_steps, len(ppo_buffer)):
                        ppo_buffer.rewards[_fi] += _feedback / max(1, _fb_steps)

            # Flush any pending combat PPO step before termination
            if in_combat and _combat_ppo_pending is not None and combat_buffer is not None:
                combat_won = terminal_value > 0
                # Partial credit for boss damage dealt on terminal loss (see rl_reward_shaping.combat_terminal_reward)
                _br_flush = 0.0
                if (not combat_won) and _combat_room_type == "boss":
                    _br_flush = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))
                step_reward = combat_step_reward(
                    _prev_combat_state, state, combat_won=combat_won,
                    hp_at_combat_start=_hp_at_combat_start,
                    boss_damage_ratio=_br_flush)
                step_reward += combat_local_tactical_reward(
                    _prev_combat_state,
                    action,
                    legal,
                )
                combat_buffer.add(
                    sf=_combat_ppo_pending["sf"],
                    af=_combat_ppo_pending["af"],
                    action_idx=_combat_ppo_pending["action_idx"],
                    log_prob=_combat_ppo_pending["log_prob"],
                    reward=step_reward,
                    value=_combat_ppo_pending["value"],
                    done=True,
                    screen_type=_combat_ppo_pending.get("screen_type", ""),
                )
                stats["combat_ppo_steps"] += 1
                _combat_ppo_pending = None
            # Mark last combat PPO step as done (death during combat)
            elif combat_buffer is not None and len(combat_buffer) > 0 and in_combat:
                combat_buffer.dones[-1] = True
                combat_won = terminal_value > 0
                _br_last = 0.0
                if (not combat_won) and _combat_room_type == "boss":
                    _br_last = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))
                combat_buffer.rewards[-1] = combat_step_reward(
                    prev_state, state, combat_won=combat_won,
                    hp_at_combat_start=_hp_at_combat_start,
                    boss_damage_ratio=_br_last)
            break

        run = state.get("run") or {}
        stats["floors"] = max(stats["floors"], int(run.get("floor", 0)))
        if _safe_int(run.get("act", 1), 1) > 1 or stats["floors"] >= ACT1_CLEAR_FLOOR:
            stats["act1_cleared"] = True

        legal = state.get("legal_actions", [])
        legal = [a for a in legal if isinstance(a, dict) and a.get("is_enabled") is not False]
        _current_reward_claim_count = _claim_reward_action_count(legal) if st == "combat_rewards" else None

        if not legal and st == "event":
            try:
                refreshed_state = client.get_state()
            except Exception:
                refreshed_state = None
            if isinstance(refreshed_state, dict):
                refreshed_legal = refreshed_state.get("legal_actions", [])
                refreshed_legal = [
                    a for a in refreshed_legal
                    if isinstance(a, dict) and a.get("is_enabled") is not False
                ]
                refreshed_event = refreshed_state.get("event") or {}
                original_event = state.get("event") or {}
                if (
                    refreshed_state.get("state_type") != st
                    or refreshed_legal
                    or refreshed_event != original_event
                ):
                    _episode_trace.append(f"[{step_i}] event: refresh-after-empty-legal")
                    state = refreshed_state
                    continue

        if not legal:
            recovery_action = _choose_empty_legal_recovery_action(state, _last_action_name)
            if recovery_action is not None:
                _episode_trace.append(
                    f"[{step_i}] {st}: empty-legal recovery via {recovery_action.get('action','?')}"
                )
                try:
                    prev_state = state
                    state = client.act(recovery_action)
                    _last_action_name = str(recovery_action.get("action") or "").strip().lower()
                    _last_reward_claim_sig = _next_reward_claim_signature(st, prev_state, recovery_action)
                except Exception:
                    break
                continue

        auto_action = _choose_auto_progress_action(
            state,
            legal,
            _last_action_name,
            _last_reward_claim_sig,
            _last_reward_claim_count,
            _reward_chain_card_reward_seen,
        )
        if auto_action is not None:
            auto_name = str(auto_action.get("action") or "?")
            auto_label = str(auto_action.get("label") or auto_name)
            _episode_trace.append(f"[{step_i}] {st}: auto-progress {auto_name} ({auto_label})")
            try:
                if _current_reward_claim_count is not None:
                    _last_reward_claim_count = _current_reward_claim_count
                prev_state = state
                state = client.act(auto_action)
                _last_action_name = auto_name.strip().lower()
                _last_reward_claim_sig = _next_reward_claim_signature(st, prev_state, auto_action)
            except Exception:
                break
            continue

        legal = state.get("legal_actions", [])
        legal = [a for a in legal if isinstance(a, dict) and a.get("is_enabled") is not False]

        if not legal:
            # No actions available — wait
            try:
                if _current_reward_claim_count is not None:
                    _last_reward_claim_count = _current_reward_claim_count
                state = client.act({"action": "wait"})
                _last_reward_claim_sig = ""
            except Exception:
                break
            continue

        # --- Auto-handle selection flows (select → confirm) ---
        # Detect by BOTH state_type AND legal_action content, because C# can
        # return select/confirm actions even under "elite"/"monster" state_type.
        _legal_action_names = {a.get("action", "") for a in legal}
        _is_selection = (
            st in ("card_select", "hand_select", "relic_select")
            or _legal_action_names & {"select_card", "combat_select_card",
                                      "combat_confirm_selection", "confirm_selection",
                                      "cancel_selection", "skip_relic_selection"}
        )
        if _is_selection:
            confirm = [a for a in legal if "confirm" in a.get("action", "") or "skip" in a.get("action", "")]
            if confirm:
                _episode_trace.append(f"[{step_i}] {st}: auto-confirm ({confirm[0].get('action')})")
                try:
                    state = client.act(confirm[0])
                    _last_reward_claim_sig = ""
                except Exception:
                    break
                continue
            select = [a for a in legal if "select" in a.get("action", "")]
            if select:
                _episode_trace.append(f"[{step_i}] {st}: auto-select {select[0].get('label','?')}")
                try:
                    state = client.act(select[0])
                    _last_reward_claim_sig = ""
                except Exception:
                    break
                continue

        # --- Auto-claim combat rewards (gold/potions always worth taking) ---
        # PPO decides card selection in card_reward screen. Here we just auto-claim
        # non-card rewards and auto-proceed to reach the card_reward screen.
        if st == "combat_rewards":
            claim = [a for a in legal if a.get("action") == "claim_reward"]
            if claim:
                _episode_trace.append(f"[{step_i}] combat_rewards: auto-claim {claim[0].get('label','?')}")
                try:
                    state = client.act(claim[0])
                except Exception:
                    break
                continue
            # All items claimed — auto-proceed to card_reward screen.
            proceed = [a for a in legal if a.get("action") in ("proceed", "skip")]
            if proceed:
                _episode_trace.append(f"[{step_i}] combat_rewards: auto-proceed to card_reward")
                try:
                    state = client.act(proceed[0])
                except Exception:
                    break
                continue

        # --- Repeat-action detection: try different action before abort ---
        _cur_action_key = f"{st}:{len(legal)}"
        if _cur_action_key == _last_action_key:
            _repeat_count += 1
            if _repeat_count >= 3:
                rest_escape = _choose_rest_site_repeat_escape_action(state, legal)
                if rest_escape is not None:
                    chosen = rest_escape
                else:
                # Try 1: proceed/skip
                    escape = [a for a in legal if a.get("action") in ("proceed", "skip", "cancel_selection")]
                    if not escape:
                        # Try 2: pick a different action than what PPO keeps choosing
                        import random as _rng
                        escape = [a for a in legal if a != legal[0]]  # anything different
                        if not escape:
                            escape = legal  # only one option, stuck
                    chosen = _rng.choice(escape) if len(escape) > 1 else escape[0]
                _episode_trace.append(f"[{step_i}] REPEAT x{_repeat_count}: escape via {chosen.get('action','?')}: {chosen.get('label','?')}")
                try:
                    if _current_reward_claim_count is not None:
                        _last_reward_claim_count = _current_reward_claim_count
                    state = client.act(chosen)
                    _last_reward_claim_sig = _next_reward_claim_signature(st, state, chosen)
                except Exception:
                    break
                if _repeat_count >= _MAX_REPEATS:
                    _episode_trace.append(f"[{step_i}] ABORT: still stuck after {_repeat_count} repeats")
                    stats["error"] = f"repeat_loop:{st}"
                    stats["end_reason"] = "repeat_loop"
                    break
                continue
        else:
            _repeat_count = 0
            _last_action_key = _cur_action_key

        # ----- COMBAT: LOCAL ORT (skip per-step Python inference) -----
        if st in COMBAT_SCREENS and use_local_ort and pipe is not None:
            if not in_combat:
                in_combat = True
                stats["combats"] += 1
                _combat_room_type = _detect_combat_room_type(st, state)
                if _pending_boss_deck_size is not None:
                    _combat_room_type = "boss"
                player = state.get("player") or {}
                _hp_at_combat_start = int(player.get("hp", player.get("current_hp", 0)) or 0)
                if _combat_room_type == "boss":
                    stats["boss_reached"] = True
                    stats["deck_size_at_boss"].append(
                        _pending_boss_deck_size if _pending_boss_deck_size is not None
                        else _extract_deck_size(state))
                    _pending_boss_deck_size = None

            # Run entire combat in C# with local ORT actor
            raw_pipe = pipe() if callable(pipe) else pipe
            try:
                _ort_t0 = time.monotonic()
                result = raw_pipe.call("run_combat_local", {"max_steps": 600})
                combat_steps = result.get("combat_steps", 0)
                post_state = result.get("state", result)
                _ort_elapsed = time.monotonic() - _ort_t0
                stats.setdefault("_ort_combat_time", 0.0)
                stats["_ort_combat_time"] += _ort_elapsed
                stats.setdefault("_ort_combat_calls", 0)
                stats["_ort_combat_calls"] += 1
                mcts_decisions += combat_steps
                combat_ppo_steps += combat_steps

                # Compute fight summary reward for non-combat PPO
                post_player = post_state.get("player") or {}
                _hp_now = int(post_player.get("hp", post_player.get("current_hp", 0)) or 0)
                _max_hp = int(post_player.get("max_hp", 80) or 80)
                post_st = (post_state.get("state_type") or "").lower()
                won = post_st not in COMBAT_SCREENS and post_st != "game_over"
                _boss_hp_frac = 0.0
                if _combat_room_type == "boss":
                    _boss_hp_frac = _estimate_boss_hp_fraction(post_state if not won else state)
                    stats["boss_hp_fracs"].append(max(_boss_hp_frac, _boss_hp_frac_peak if '_boss_hp_frac_peak' in dir() else 0))
                _feedback = compute_combat_feedback(
                    _hp_at_combat_start, _hp_now if won else 0, _max_hp,
                    won=won, room_type=_combat_room_type,
                    boss_hp_fraction_dealt=_boss_hp_frac)
                if len(ppo_buffer) > 0:
                    ppo_buffer.rewards[-1] += _feedback

                prev_state = state
                state = post_state
                in_combat = False
                _episode_trace.append(
                    f"[{step_i}] ORT_COMBAT #{stats['combats']} {_combat_room_type} "
                    f"steps={combat_steps} won={won} hp={_hp_at_combat_start}->{_hp_now}")
                continue
            except Exception as ort_err:
                _episode_trace.append(f"[{step_i}] ORT_COMBAT_ERROR: {ort_err}")
                # Fall through to normal combat path

        # ----- COMBAT: MCTS/PPO -----
        if st in COMBAT_SCREENS:
            if not in_combat:
                in_combat = True
                stats["combats"] += 1
                _combat_room_type = _detect_combat_room_type(st, state)
                if _pending_boss_deck_size is not None:
                    _combat_room_type = "boss"
                _boss_hp_frac_peak = 0.0
                _prev_combat_state = state  # track for combat step reward
                player = state.get("player") or {}
                battle = state.get("battle") or {}
                enemies = battle.get("enemies") or []
                _hp_at_combat_start = int(player.get("hp", player.get("current_hp", 0)) or 0)
                if _combat_room_type == "boss":
                    stats["boss_reached"] = True
                    stats["deck_size_at_boss"].append(
                        _pending_boss_deck_size
                        if _pending_boss_deck_size is not None
                        else _extract_deck_size(state)
                    )
                    _pending_boss_deck_size = None
                _episode_trace.append(
                    f"[{step_i}] COMBAT #{stats['combats']} floor={run.get('floor',0)} "
                    f"hp={player.get('hp',0)} blk={player.get('block',0)} e={player.get('energy',0)} "
                    f"hand=[{_combat_hand_summary(state)}] "
                    f"intent=[{_combat_enemy_intent_summary(state)}]"
                )

                # Flush PPO's last non-combat step as "entering combat"
                if len(ppo_buffer) > 0:
                    reward = shaped_reward(
                        prev_state, state, 0.0, done=False,
                        boss_entry_quality_weight=boss_entry_quality_weight,
                        early_damage_potion_penalty_weight=early_damage_potion_penalty_weight,
                    )
                    ppo_buffer.rewards[-1] = reward

            # Combat decision: MCTS search (if pipe available) + Combat PPO data
            import random as _random
            try:
                sf = build_combat_features(state, vocab)
                af = build_combat_action_features(state, legal, vocab)

                # Note: deck_ids/aux/mask are now included in build_combat_features()
                # automatically. CombatPolicyValueNetwork with deck_repr_dim > 0
                # encodes them internally via its own deck_encoder.

                # Try MCTS search first (high-quality decision + behavior cloning target)
                mcts_used = False
                if pipe is not None and use_mcts_combat:
                    try:
                        fm = PipeCombatForwardModel.from_current_state(
                            pipe() if callable(pipe) else pipe)
                        action, root = mcts_agent.choose_action(fm)
                        from combat_mcts_agent import _reconcile_action
                        action = _reconcile_action(action, legal)
                        # Find action_idx in legal list
                        action_idx = 0
                        action_label = action.get("label", "")
                        for ai, la in enumerate(legal):
                            if la.get("label") == action_label and la.get("action") == action.get("action"):
                                action_idx = ai
                                break

                        # Collect MCTS behavior cloning target (visit distribution)
                        if af["action_mask"].any():
                            _, mcts_policy = root.visit_distribution()
                            padded = np.zeros(MAX_ACTIONS, dtype=np.float32)
                            padded[:len(mcts_policy)] = mcts_policy
                            mcts_pending.append({
                                "state_features": sf,
                                "action_features": af,
                                "mcts_policy": padded,
                                "outcome": 0.0,
                            })
                        mcts_used = True
                        stats["mcts_searches"] += 1
                    except Exception as e:
                        logger.debug("MCTS search failed at step %d: %s", step_i, e)
                    finally:
                        try:
                            if 'fm' in dir() and fm is not None:
                                fm.cleanup_and_restore()
                        except Exception:
                            try:
                                if 'fm' in dir() and fm is not None:
                                    fm.cleanup()
                            except Exception:
                                pass

                # Fallback: NN PPO sampling (also used for Combat PPO data)
                if not mcts_used:
                    if inference_client is not None:
                        logits_np, value = inference_client.combat_inference(sf, af)
                        mask = af["action_mask"].astype(np.float32)
                        logits_masked = logits_np + (1.0 - mask) * (-1e9)
                        logits_t = torch.tensor(logits_masked)
                        dist = torch.distributions.Categorical(logits=logits_t)
                        if deterministic_policy:
                            action_idx_t = logits_t.argmax(dim=-1)
                        else:
                            action_idx_t = dist.sample()
                        log_prob = dist.log_prob(action_idx_t).item()
                        action_idx = action_idx_t.item()
                    else:
                        combat_device = next(mcts_agent.network.parameters()).device
                        sf_t = {}
                        for k, v in sf.items():
                            t = torch.tensor(v).unsqueeze(0)
                            if v.dtype in (np.int64, np.int32): t = t.long()
                            elif v.dtype == bool: t = t.bool()
                            else: t = t.float()
                            sf_t[k] = t.to(combat_device)
                        af_t = {}
                        for k, v in af.items():
                            t = torch.tensor(v).unsqueeze(0)
                            if v.dtype in (np.int64, np.int32): t = t.long()
                            elif v.dtype == bool: t = t.bool()
                            else: t = t.float()
                            af_t[k] = t.to(combat_device)
                        with torch.no_grad():
                            logits, value_t = mcts_agent.network(sf_t, af_t)
                        mask = af_t["action_mask"].float()
                        logits_masked = logits + (1.0 - mask) * (-1e9)
                        dist = torch.distributions.Categorical(logits=logits_masked.squeeze(0))
                        if deterministic_policy:
                            action_idx_t = logits_masked.squeeze(0).argmax(dim=-1)
                        else:
                            action_idx_t = dist.sample()
                        log_prob = dist.log_prob(action_idx_t).cpu().item()
                        action_idx = action_idx_t.cpu().item()
                        value = value_t.squeeze(0).cpu().item()

                    if action_idx < len(legal):
                        action = legal[action_idx]
                    else:
                        action = legal[0]; action_idx = 0

                # Combat PPO data: only from NN episodes (not MCTS/heuristic).
                # MCTS actions have extremely low NN log_prob → ratio explosion in PPO.
                stats["mcts_decisions"] += 1
                _act_label = action.get("label", action.get("action", "?"))
                _src = "mcts" if mcts_used else "nn"
                _target_suffix = _action_target_summary(action)
                _player_view = _combat_player_view(state)
                _combat_ctx = (
                    f"hp={_player_view.get('hp',0)} blk={_player_view.get('block',0)} "
                    f"e={_player_view.get('energy',0)} "
                    f"hand=[{_combat_hand_summary(state)}] "
                    f"intent=[{_combat_enemy_intent_summary(state)}]"
                )

                if not mcts_used:
                    _top_actions = _topk_action_summary(legal, logits_masked, k=3)
                    _episode_trace.append(
                        f"[{step_i}] COMBAT {_src}: {_act_label}{_target_suffix} "
                        f"(idx={action_idx} v={value:.2f} lp={log_prob:.2f}) "
                        f"{_combat_ctx} top=[{_top_actions}]"
                    )
                    _combat_ppo_pending = {
                        "sf": sf, "af": af,
                        "action_idx": action_idx,
                        "log_prob": log_prob,
                        "value": value,
                        "screen_type": st,
                    }
                else:
                    _episode_trace.append(
                        f"[{step_i}] COMBAT {_src}: {_act_label}{_target_suffix} "
                        f"(idx={action_idx}) {_combat_ctx}"
                    )
                    _combat_ppo_pending = None  # MCTS data goes to mcts_pending only

            except Exception as e:
                logger.warning("Combat eval failed at step %d: %s", step_i, e)
                action = _random.choice(legal)
                _combat_ppo_pending = None

            # Execute action via client (pipe or HTTP).
            _t0 = time.monotonic()
            _act_desc = action.get("action", "?") if isinstance(action, dict) else "?"
            try:
                state = client.act(action)
            except Exception:
                try:
                    fallback = {k: v for k, v in action.items()
                                if k not in ("target_id", "slot", "target")}
                    state = client.act(fallback)
                except Exception:
                    try:
                        state = client.act({"action": "end_turn"})
                    except Exception:
                        try:
                            state = client.get_state()
                        except Exception as e2:
                            stats["error"] = f"combat step: {e2}"
                            _timeout_count += 1
                            break
            _dt = time.monotonic() - _t0
            _max_step_ms = max(_max_step_ms, _dt * 1000)
            if _dt >= _slow_step_threshold:
                _slow_steps += 1
                logger.debug("Slow combat step %d (%s): %.0fms", step_i, _act_desc, _dt * 1000)
            if _combat_room_type == "boss":
                _boss_hp_frac_peak = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))

            # Add combat PPO step now that we have next_state
            if (combat_buffer is not None
                    and _combat_ppo_pending is not None):
                next_st = (state.get("state_type") or "").lower()
                # Check if combat just ended
                combat_just_ended = next_st not in COMBAT_SCREENS
                combat_won = None
                if combat_just_ended:
                    # If next state is game_over with death, combat was lost
                    if next_st == "game_over" or state.get("terminal"):
                        go = state.get("game_over") or {}
                        outcome_str = (go.get("run_outcome") or go.get("outcome") or "").lower()
                        combat_won = "victory" in outcome_str or outcome_str == "win"
                        if not combat_won:
                            combat_won = False
                    else:
                        combat_won = True  # survived combat → reward screen or map
                # Partial credit for boss damage dealt on terminal loss (see rl_reward_shaping.combat_terminal_reward)
                _br_main = 0.0
                if combat_just_ended and combat_won is False and _combat_room_type == "boss":
                    _br_main = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))
                step_reward = combat_step_reward(
                    _prev_combat_state, state, combat_won=combat_won,
                    hp_at_combat_start=_hp_at_combat_start,
                    boss_damage_ratio=_br_main)
                combat_buffer.add(
                    sf=_combat_ppo_pending["sf"],
                    af=_combat_ppo_pending["af"],
                    action_idx=_combat_ppo_pending["action_idx"],
                    log_prob=_combat_ppo_pending["log_prob"],
                    reward=step_reward,
                    value=_combat_ppo_pending["value"],
                    done=bool(combat_just_ended),
                    screen_type=_combat_ppo_pending.get("screen_type", ""),
                )
                stats["combat_ppo_steps"] += 1
                _combat_ppo_pending = None

            _prev_combat_state = state  # update for next step's reward

        # ----- NON-COMBAT: PPO -----
        else:
            if in_combat:
                in_combat = False
                stats["combats_won"] += 1
                player = state.get("player") or {}
                _hp_now = int(player.get("hp", player.get("current_hp", 0)))
                _max_hp = int(player.get("max_hp", 80))
                stats["hp_timeline"].append(_hp_now)
                _episode_trace.append(f"[{step_i}] COMBAT WON -> {st} hp={_hp_now}/{_max_hp}")
                _boss_hp_frac_dealt = 1.0 if _combat_room_type == "boss" else 0.0
                if _combat_room_type == "boss":
                    stats["boss_reached"] = True
                    stats["act1_cleared"] = True
                    stats["boss_hp_fraction_dealt"].append(max(_boss_hp_frac_peak, _boss_hp_frac_dealt))

                # --- Combat feedback → non-combat decisions ---
                _feedback = compute_combat_feedback(
                    _hp_at_combat_start, _hp_now, _max_hp, won=True,
                    room_type=_combat_room_type,
                    boss_hp_fraction_dealt=_boss_hp_frac_dealt,
                )
                if segment_collector is not None and segment_collector.is_open:
                    # Phase 2: accumulate into current segment (direct attribution)
                    segment_collector.add_reward(_feedback, tag="fight_summary", steps=5)
                    _episode_trace.append(
                        f"[{step_i}] COMBAT FEEDBACK: {_feedback:+.3f} → segment")
                else:
                    # Legacy: spread to last N non-combat PPO steps
                    _fb_steps = min(10, len(ppo_buffer))
                    for _fi in range(len(ppo_buffer) - _fb_steps, len(ppo_buffer)):
                        ppo_buffer.rewards[_fi] += _feedback / max(1, _fb_steps)
                    if _fb_steps > 0:
                        _episode_trace.append(
                            f"[{step_i}] COMBAT FEEDBACK: {_feedback:+.3f} spread to {_fb_steps} PPO steps")
                # Combat ended → backfill pending MCTS examples with HP-based value
                if mcts_pending:
                    player = state.get("player") or {}
                    hp = float(player.get("hp", player.get("current_hp", 0)))
                    max_hp = max(1.0, float(player.get("max_hp", 1)))
                    combat_value = hp / max_hp
                    for ex_data in mcts_pending:
                        if ex_data["outcome"] == 0.0:
                            ex_data["outcome"] = combat_value

            # PPO inference — pending-step pattern:
            # 1. Observe state, sample action
            # 2. Execute action → get next_state
            # 3. Compute reward = shaped(state, next_state)
            # 4. Write (state, action, reward) to buffer
            _ppo_pending = None
            try:
                ss = build_structured_state(state, vocab)
                sa = build_structured_actions(state, legal, vocab)

                sf_np = _structured_state_to_numpy_dict(ss)
                af_np = _structured_actions_to_numpy_dict(sa)

                _nc_fwd_t0 = time.monotonic()
                if inference_client is not None:
                    action_idx, log_prob, _, value = inference_client.ppo_inference(sf_np, af_np)
                elif ppo_ort_session is not None:
                    # Branch C: ORT CPU actor-only (0.5ms vs PyTorch 6.5ms)
                    ort_inputs = {}
                    for inp in ppo_ort_session.get_inputs():
                        name = inp.name
                        arr = sf_np.get(name, af_np.get(name))
                        if arr is None:
                            continue
                        arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
                        if arr.dtype == bool:
                            arr = arr.astype(np.float32)
                        elif arr.dtype in (np.int64, np.int32):
                            arr = arr.astype(np.int64)
                        else:
                            arr = arr.astype(np.float32)
                        ort_inputs[name] = arr.reshape(1, *arr.shape) if arr.ndim > 0 else arr.reshape(1)
                    ort_logits = ppo_ort_session.run(None, ort_inputs)[0][0]  # (MAX_ACTIONS,)
                    # Mask and sample
                    mask = af_np["action_mask"].astype(np.float32)
                    masked = ort_logits + (1.0 - mask) * (-1e9)
                    if deterministic_policy:
                        action_idx = int(np.argmax(masked))
                    else:
                        # Stable softmax + sample
                        shifted = masked - masked.max()
                        probs = np.exp(shifted)
                        probs = probs / probs.sum()
                        action_idx = int(np.random.choice(len(probs), p=probs))
                    # log_prob and value will be recomputed after rollout (Phase 3)
                    # For now use placeholder values
                    log_prob = float(np.log(max(probs[action_idx], 1e-8))) if not deterministic_policy else 0.0
                    value = 0.0  # placeholder, recomputed in batch GPU after rollout
                else:
                    ppo_device = next(ppo_network.parameters()).device
                    state_t = {}
                    for k, v in sf_np.items():
                        t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
                        if "ids" in k or "idx" in k or "types" in k or "count" in k:
                            t = t.long()
                        elif "mask" in k:
                            t = t.bool()
                        else:
                            t = t.float()
                        state_t[k] = t.to(ppo_device)
                    action_t = {}
                    for k, v in af_np.items():
                        t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
                        if "ids" in k or "types" in k or "indices" in k:
                            t = t.long()
                        elif "mask" in k:
                            t = t.bool()
                        else:
                            t = t.float()
                        action_t[k] = t.to(ppo_device)
                    with torch.no_grad():
                        action_idx_t, log_prob_t, _, value_t = ppo_network.get_action_and_value(
                            state_t, action_t, deterministic=deterministic_policy)
                    action_idx = action_idx_t.cpu().item()
                    log_prob = log_prob_t.cpu().item()
                    value = value_t.cpu().item()

                _nc_fwd_elapsed = time.monotonic() - _nc_fwd_t0
                stats.setdefault("_nc_forward_time", 0.0)
                stats["_nc_forward_time"] += _nc_fwd_elapsed
                stats.setdefault("_nc_forward_calls", 0)
                stats["_nc_forward_calls"] += 1

                if action_idx < len(legal):
                    action = legal[action_idx]
                else:
                    action = legal[0]

                # Track card selections
                _act_name = action.get("action", "")
                _act_label = action.get("label", _act_name)
                if st == "map" and _act_name == "choose_map_node":
                    _node_hint = str(
                        action.get("node_type")
                        or action.get("note")
                        or action.get("label")
                        or ""
                    ).strip().lower()
                    if "boss" in _node_hint:
                        _pending_boss_deck_size = _extract_deck_size(state)
                if st == "card_reward":
                    stats["card_reward_screens"] += 1
                if _act_name == "select_card_reward":
                    stats["cards_taken"].append(_act_label)
                elif _act_name in ("skip", "skip_card_reward"):
                    stats["cards_skipped"] += 1
                    if st == "card_reward":
                        stats["card_reward_skips"] += 1

                # Extract player HP from nested state (player may be under event/map/shop etc.)
                _p = state.get("player") or {}
                if not _p.get("hp"):
                    for _container_key in ("event", "map", "shop", "rest_site", "rewards",
                                           "card_reward", "treasure", "combat_rewards"):
                        _container = state.get(_container_key)
                        if isinstance(_container, dict) and isinstance(_container.get("player"), dict):
                            _p = _container["player"]
                            break
                _evt_id = ""
                if st == "event":
                    _evt = state.get("event") or {}
                    _evt_id = f" [{_evt.get('event_id', '?')}]"
                _episode_trace.append(
                    f"[{step_i}] {st}{_evt_id}: {_act_label} "
                    f"(idx={action_idx} v={value:.2f}) "
                    f"floor={run.get('floor',0)} hp={_p.get('hp', _p.get('current_hp', '?'))}")

                # Save pending — reward computed AFTER act()
                _ppo_pending = {
                    "ss": ss, "sa": sa,
                    "action_idx": action_idx,
                    "log_prob": log_prob,
                    "value": value,
                    "pre_state": state,  # state BEFORE action
                    "screen_type": st,
                }

            except Exception as e:
                logger.warning("PPO inference failed at step %d: %s", step_i, e)
                action = legal[0]

            prev_state = state

            # Execute action via client (pipe or HTTP)
            _t0 = time.monotonic()
            _act_desc = action.get("action", "?") if isinstance(action, dict) else "?"
            try:
                if _current_reward_claim_count is not None:
                    _last_reward_claim_count = _current_reward_claim_count
                state = client.act(action)
                _last_action_name = str(_act_desc or "").strip().lower()
                _last_reward_claim_sig = _next_reward_claim_signature(st, prev_state, action)
            except Exception as e1:
                err_str = str(e1)
                _episode_trace.append(f"[{step_i}] ACT FAILED ({_act_desc}): {err_str[:80]}")
                # Retry once for transient errors (file rename race condition)
                if "rename" in err_str.lower() or "file" in err_str.lower():
                    import time as _t
                    _t.sleep(0.1)
                    try:
                        if _current_reward_claim_count is not None:
                            _last_reward_claim_count = _current_reward_claim_count
                        state = client.act(action)
                        _last_action_name = str(_act_desc or "").strip().lower()
                        _last_reward_claim_sig = _next_reward_claim_signature(st, prev_state, action)
                        _episode_trace.append(f"[{step_i}] RETRY OK")
                    except Exception:
                        stats["error"] = f"rename_retry_failed: {err_str[:50]}"
                        stats["end_reason"] = "step_error"
                        break
                else:
                    pass
                try:
                    state = client.get_state()
                    fresh_legal = [a for a in state.get("legal_actions", [])
                                   if isinstance(a, dict) and a.get("is_enabled") is not False]
                    if fresh_legal:
                        fallback_action = (
                            _choose_auto_progress_action(
                                state,
                                fresh_legal,
                                _last_action_name,
                                _last_reward_claim_sig,
                                _last_reward_claim_count,
                                _reward_chain_card_reward_seen,
                            )
                            or fresh_legal[0]
                        )
                        fallback_claim_count = _claim_reward_action_count(fresh_legal) if st == "combat_rewards" else None
                        if fallback_claim_count is not None:
                            _last_reward_claim_count = fallback_claim_count
                        prev_state = state
                        state = client.act(fallback_action)
                        _last_action_name = str(fallback_action.get("action") or "").strip().lower()
                        _last_reward_claim_sig = _next_reward_claim_signature(
                            st,
                            prev_state,
                            fallback_action,
                        )
                    else:
                        try:
                            recovery_action = _choose_empty_legal_recovery_action(state, _last_action_name)
                            if recovery_action is not None:
                                prev_state = state
                                state = client.act(recovery_action)
                                _last_action_name = str(recovery_action.get("action") or "").strip().lower()
                                _last_reward_claim_sig = _next_reward_claim_signature(
                                    st,
                                    prev_state,
                                    recovery_action,
                                )
                            else:
                                state = client.act({"action": "wait"})
                                _last_action_name = "wait"
                                _last_reward_claim_sig = ""
                        except Exception:
                            state = client.get_state()
                except Exception as e2:
                    _episode_trace.append(f"[{step_i}] FALLBACK FAILED: {str(e2)[:80]}")
                    stats["error"] = f"non-combat step: {e2}"
                    stats["end_reason"] = "step_error"
                    _timeout_count += 1
                    break
            _dt = time.monotonic() - _t0
            _max_step_ms = max(_max_step_ms, _dt * 1000)
            if _dt >= _slow_step_threshold:
                _slow_steps += 1
                logger.debug("Slow non-combat step %d (%s): %.0fms", step_i, _act_desc, _dt * 1000)

            # Now compute reward with CORRECT (pre_state → post_state) and add to buffer
            if _ppo_pending is not None:
                reward = shaped_reward(
                    _ppo_pending["pre_state"], state, 0.0, done=False,
                    boss_entry_quality_weight=boss_entry_quality_weight,
                    action=action,
                    early_damage_potion_penalty_weight=early_damage_potion_penalty_weight,
                )
                local_delta = 0.0
                if screen_local_delta and segment_collector is None:
                    local_delta = screen_local_delta_reward(
                        _ppo_pending["pre_state"], state, _ppo_pending.get("screen_type", ""),
                    )
                    reward += local_delta

                if segment_collector is not None and segment_buffer is not None:
                    # Phase 2: Close previous segment (if any) and open new one
                    if segment_collector.is_open:
                        seg = segment_collector.close_segment(done=False)
                        if seg is not None:
                            segment_buffer.add(seg)
                    # Compute counterfactual reward (Phase 3)
                    _cf_teacher = None
                    if counterfactual_scoring:
                        try:
                            _cf_reward, _cf_teacher = compute_counterfactual_reward(
                                st, _ppo_pending["pre_state"], legal,
                                _ppo_pending["action_idx"],
                            )
                        except Exception:
                            _cf_reward = 0.0
                    else:
                        _cf_reward = 0.0
                    # Open new segment
                    segment_collector.open_segment(
                        state=_structured_state_to_numpy_dict(_ppo_pending["ss"]),
                        actions=_structured_actions_to_numpy_dict(_ppo_pending["sa"]),
                        action_idx=_ppo_pending["action_idx"],
                        log_prob=_ppo_pending["log_prob"],
                        value=_ppo_pending["value"],
                        screen_type_idx=_ppo_pending["ss"].screen_type_idx,
                        teacher_logits=_cf_teacher,
                    )
                    # Add PBRS reward + counterfactual
                    segment_collector.add_reward(reward, tag="pbrs", steps=1)
                    if _cf_reward != 0.0:
                        segment_collector.add_reward(
                            _cf_reward * counterfactual_weight, tag="counterfactual", steps=0)
                    stats["ppo_steps"] += 1
                else:
                    # Legacy step-by-step path
                    ppo_buffer.add(
                        _ppo_pending["ss"], _ppo_pending["sa"],
                        _ppo_pending["action_idx"], _ppo_pending["log_prob"],
                        reward, _ppo_pending["value"], done=False,
                        boss_readiness_target=boss_readiness_score(_ppo_pending["pre_state"]))
                    stats["ppo_steps"] += 1
                    # Save for offline data
                    if episode_saver is not None:
                        episode_saver.add_step(
                            _structured_state_to_numpy_dict(_ppo_pending["ss"]),
                            _structured_actions_to_numpy_dict(_ppo_pending["sa"]),
                            _ppo_pending["action_idx"], reward,
                            _ppo_pending.get("screen_type", "unknown"),
                            _ppo_pending["log_prob"], _ppo_pending["value"],
                        )

    # Close any remaining open segment
    if segment_collector is not None and segment_collector.is_open and segment_buffer is not None:
        seg = segment_collector.close_segment(done=True)
        if seg is not None:
            segment_buffer.add(seg)

    # Set floor targets for PPO deck quality head
    # Richer target: progress + win bonus + combat efficiency
    final_floor = stats["floors"]
    total_floor = float(final_floor)
    won = stats.get("outcome") == "victory"
    combats_won = stats.get("combats_won", 0)
    combats_total = max(1, stats.get("combats", 1))
    dq_target = (
        (total_floor / 51.0) * 0.5                      # progress [0, 0.5]
        + (1.0 if won else 0.0) * 0.3                   # win bonus [0, 0.3]
        + (combats_won / combats_total) * 0.2            # combat efficiency [0, 0.2]
    )
    dq_target = min(dq_target, 1.0)
    if len(ppo_buffer) > 0:
        ppo_buffer.set_floor_targets(dq_target)
    if segment_buffer is not None and len(segment_buffer) > 0:
        segment_buffer.set_floor_targets(dq_target)

    # Build MCTS examples
    mcts_examples = [MCTSTrainingExample(**d) for d in mcts_pending]

    # End-of-episode reason
    if not stats.get("error") and not stats.get("outcome"):
        stats["end_reason"] = "max_steps"
        _episode_trace.append(f"[END] max_steps reached (steps={step_i}, st={st})")
    stats["episode_time_s"] = time.monotonic() - episode_start
    stats["slow_steps"] = _slow_steps
    stats["max_step_ms"] = _max_step_ms
    stats["timeout_count"] = _timeout_count
    # Save high-quality episodes for offline RL
    if episode_saver is not None:
        episode_saver.finish_episode(
            floor=final_floor,
            outcome=stats.get("outcome"),
            combats_won=combats_won,
            extra_stats=stats,
        )

    stats["_combat_buffer"] = combat_buffer  # for merging in main loop
    stats["_episode_trace"] = _episode_trace  # for replay dump
    stats["_segment_buffer"] = segment_buffer  # for Phase 2 merging
    if segment_buffer is not None and len(segment_buffer) > 0:
        stats["segment_stats"] = segment_buffer.get_segment_stats()
    return ppo_buffer, mcts_examples, stats


# ---------------------------------------------------------------------------
# MCTS train step (from train_combat_mcts.py)
# ---------------------------------------------------------------------------

def mcts_train_step(
    network: CombatPolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[MCTSTrainingExample],
    device: torch.device | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    if device is None:
        device = next(network.parameters()).device

    state_tensors = {}
    action_tensors = {}
    for k in batch[0].state_features:
        arr = np.stack([ex.state_features[k] for ex in batch])
        if arr.dtype in (np.int64, np.int32):
            state_tensors[k] = torch.tensor(arr, dtype=torch.long, device=device)
        elif arr.dtype == bool:
            state_tensors[k] = torch.tensor(arr, dtype=torch.bool, device=device)
        else:
            state_tensors[k] = torch.tensor(arr, dtype=torch.float32, device=device)
    for k in batch[0].action_features:
        arr = np.stack([ex.action_features[k] for ex in batch])
        if arr.dtype in (np.int64, np.int32):
            action_tensors[k] = torch.tensor(arr, dtype=torch.long, device=device)
        elif arr.dtype == bool:
            action_tensors[k] = torch.tensor(arr, dtype=torch.bool, device=device)
        else:
            action_tensors[k] = torch.tensor(arr, dtype=torch.float32, device=device)

    target_policy = torch.tensor(np.stack([ex.mcts_policy for ex in batch]),
                                  dtype=torch.float32, device=device)
    target_value = torch.tensor([ex.outcome for ex in batch],
                                 dtype=torch.float32, device=device)

    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, value = network.forward(state_tensors, action_tensors)
            logits_safe = logits.float().clamp(min=-30.0)
            log_probs = F.log_softmax(logits_safe, dim=-1)
            mask = action_tensors["action_mask"].float()
            policy_loss = -(target_policy * (log_probs * mask)).sum(dim=-1).mean()
            value_loss = F.mse_loss(value.float(), target_value)
            loss = policy_loss + value_loss
    else:
        logits, value = network.forward(state_tensors, action_tensors)
        logits_safe = logits.clamp(min=-30.0)
        log_probs = F.log_softmax(logits_safe, dim=-1)
        mask = action_tensors["action_mask"].float()
        policy_loss = -(target_policy * (log_probs * mask)).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, target_value)
        loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()

    return {"mcts_ploss": policy_loss.item(), "mcts_vloss": value_loss.item()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Unified PPO + MCTS hybrid training")
    # Environment
    parser.add_argument("--pipe", action="store_true", help="Use pipe for MCTS (recommended)")
    parser.add_argument(
        "--transport",
        choices=["http", "pipe", "pipe-binary"],
        default=None,
        help="Simulator transport override. Defaults to 'pipe' when --pipe is set, otherwise 'http'.",
    )
    parser.add_argument("--auto-launch", action="store_true",
                        help="Auto-launch one fresh Sim host per env port for pipe transports.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                        help="Repo root used when auto-launching Sim hosts.")
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_DLL_PATH,
                        help="Path to headless_sim_host_0991.exe/.dll (or legacy HeadlessSim.dll) for auto-launch.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--start-port", type=int, default=15527)
    parser.add_argument("--character-id", default="IRONCLAD")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional global seed for Python/NumPy/Torch and per-episode env reset seeds")

    # Training
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--episodes-per-iter", type=int, default=0,
                        help="0 = auto (same as num-envs)")
    parser.add_argument("--episode-timeout", type=float, default=90.0)
    parser.add_argument("--max-episode-steps", type=int, default=600)

    # PPO hyperparams
    parser.add_argument("--ppo-lr", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch", type=int, default=32)
    parser.add_argument("--ppo-entropy-coeff", type=float, default=0.1)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--boss-readiness-coeff", type=float, default=0.05,
                        help="Auxiliary loss weight for boss-aware build readiness head")
    parser.add_argument("--boss-aware-warmup-only", action="store_true", default=False,
                        help="Freeze old PPO backbone and train only new boss-aware adapter/head params")

    # MCTS hyperparams
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--mcts-lr", type=float, default=1e-3)
    parser.add_argument("--mcts-batch-size", type=int, default=128)
    parser.add_argument("--mcts-train-steps", type=int, default=10)
    parser.add_argument("--mcts-warmup-iters", type=int, default=50,
                        help="Use random combat actions for first N iters (untrained NN is worse than random)")
    parser.add_argument("--mcts-replay-size", type=int, default=50000)

    # Combat PPO hyperparams
    parser.add_argument("--combat-ppo-lr", type=float, default=3e-4,
                        help="Combat PPO learning rate (default: 3e-4)")
    parser.add_argument("--combat-ppo-epochs", type=int, default=4,
                        help="Combat PPO epochs per update (default: 4)")
    parser.add_argument("--combat-ppo-minibatch", type=int, default=64,
                        help="Combat PPO minibatch size (default: 64)")
    parser.add_argument("--combat-ppo-entropy-coeff", type=float, default=0.05,
                        help="Combat PPO entropy coefficient (default: 0.05)")
    parser.add_argument("--combat-ppo-clip", type=float, default=0.2,
                        help="Combat PPO clip epsilon (default: 0.2)")

    # Network architecture
    parser.add_argument("--embed-dim", type=int, default=48,
                        help="Entity embedding dimension (default: 48)")
    parser.add_argument("--combat-hidden-dim", type=int, default=192,
                        help="Combat NN hidden dimension (default: 192)")
    parser.add_argument("--deck-repr-dim", type=int, default=0,
                        help="Deck embedding dimension for build_plan_z bridge (0=disabled, 64=recommended)")
    parser.add_argument("--vectorized", action="store_true", default=False,
                        help="Use vectorized episode collection (parallel pipe I/O + batch NN inference)")
    parser.add_argument("--local-ort", action="store_true", default=False,
                        help="Use C# local ORT CPU actor for combat (3x+ throughput, requires ONNX model loaded in sim)")
    parser.add_argument("--ort-model-path", type=str, default=None,
                        help="Path to ONNX actor model (auto-loads into each HeadlessSim on startup)")
    parser.add_argument("--iter-time-budget", type=float, default=0,
                        help="Max seconds per iter for episode collection (0=no limit, 4.0=recommended with --local-ort)")
    parser.add_argument("--zero-cuda-collector", action="store_true", default=False,
                        help="Use CPU policy snapshots for rollout workers (zero CUDA in collector threads)")
    parser.add_argument("--residual-adapter", action="store_true", default=False,
                        help="Use residual adapter mode: freeze backbone, train only deck adapter heads")
    parser.add_argument("--freeze-embeddings", action="store_true", default=False,
                        help="Freeze shared entity embeddings to prevent combat/ranking gradient conflict")
    parser.add_argument("--retrieval-head", action="store_true", default=False,
                        help="Enable SymbolicFeaturesHead cross-attention over "
                             "source_knowledge.sqlite. Adds ~18K trainable params "
                             "and a zero-shot symbolic prior for rare entities. "
                             "Default off so the champion is untouched. See "
                             "docs/HANDOFF_2026-04-09.md §7.2.D for design.")
    parser.add_argument("--retrieval-proj-dim", type=int, default=16,
                        help="Output projection dim of SymbolicFeaturesHead "
                             "(default: 16). Only used when --retrieval-head is set.")
    parser.add_argument("--freeze-combat", action="store_true", default=False,
                        help="Freeze entire combat brain (mcts_net). Only train PPO/non-combat side.")
    parser.add_argument("--freeze-ppo", action="store_true", default=False,
                        help="Freeze entire PPO brain (ppo_net). Only train combat side.")
    parser.add_argument("--combat-boss-only", action="store_true", default=False,
                        help="Only train combat on boss/elite encounters (skip hallway). "
                             "Requires --freeze-ppo.")
    parser.add_argument("--combat-monster-reward-weight", type=float, default=1.0,
                        help="Reward weight for monster combat data during merge (default 1.0). "
                             "Setting <1.0 keeps monster data in buffer but dampens its loss contribution. "
                             "Ignored when --combat-boss-only is set. Recommended: 0.1 for focus on boss/elite.")
    parser.add_argument("--matchup-loss-decay-tau", type=float, default=0.0,
                        help="Exponential decay tau for matchup_loss_weight (0=no decay, 300=recommended)")

    # Checkpoints
    parser.add_argument("--resume", type=str, default=str(MAINLINE_CHECKPOINT),
                        help="Resume both networks from a hybrid checkpoint (hybrid_XXXXX.pt)")
    parser.add_argument("--resume-ppo", type=str, default=None,
                        help="Resume PPO only (standalone PPO checkpoint)")
    parser.add_argument("--resume-mcts", type=str, default=None,
                        help="Resume MCTS only (standalone MCTS checkpoint)")
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--multi-process", action="store_true",
                        help="Use multi-process workers + batch inference (bypasses GIL)")
    parser.add_argument("--deterministic-policy", action="store_true", default=False,
                        help="Use argmax action selection for PPO/combat policy (audit/demo only)")
    parser.add_argument("--batch-timeout-ms", type=float, default=5.0,
                        help="Batch inference server timeout in ms")
    parser.add_argument("--mcts", action="store_true",
                        help="Enable MCTS combat search (slower but higher quality decisions)")
    parser.add_argument("--no-mcts", action="store_true",
                        help="(deprecated, MCTS off by default)")

    # --- GPT-design optimizations (Phase 1-4) ---
    parser.add_argument("--screen-value-heads", action="store_true", default=True,
                        help="Use screen-specific value heads (Phase 1A, default: True)")
    parser.add_argument("--no-screen-value-heads", dest="screen_value_heads", action="store_false")
    parser.add_argument("--per-screen-adv-norm", action="store_true", default=True,
                        help="Normalize advantages per screen type (Phase 1B)")
    parser.add_argument("--no-per-screen-adv-norm", dest="per_screen_adv_norm", action="store_false")
    parser.add_argument("--weighted-screen-sampling", action="store_true", default=True,
                        help="Weight minibatch sampling by screen frequency (Phase 1C)")
    parser.add_argument("--no-weighted-screen-sampling", dest="weighted_screen_sampling", action="store_false")
    parser.add_argument("--use-segment-collector", action="store_true", default=False,
                        help="Use semi-MDP segment collector for non-combat (Phase 2)")
    parser.add_argument("--counterfactual-scoring", action="store_true", default=False,
                        help="Enable screen-local counterfactual scoring (Phase 3)")
    parser.add_argument("--counterfactual-weight", type=float, default=0.0,
                        help="Blend weight for counterfactual reward (Phase 3)")
    parser.add_argument("--kl-warmstart", action="store_true", default=False,
                        help="Enable KL warm-start from heuristic teacher (Phase 4)")
    parser.add_argument("--kl-beta-start", type=float, default=0.5,
                        help="Initial KL loss coefficient (Phase 4)")
    parser.add_argument("--kl-beta-decay", type=int, default=2000,
                        help="Number of iterations for KL beta decay (Phase 4)")

    # --- Offline data saving ---
    parser.add_argument("--save-offline-data", action="store_true", default=True,
                        help="Save high-quality episodes for offline RL (default: True)")
    parser.add_argument("--no-save-offline-data", dest="save_offline_data", action="store_false")
    parser.add_argument("--offline-min-floor", type=int, default=14,
                        help="Min floor to save episode (default: 14)")
    parser.add_argument("--save-replay-traces", action="store_true", default=True,
                        help="Write per-episode replay trace files (default: True)")
    parser.add_argument("--no-save-replay-traces", dest="save_replay_traces", action="store_false")
    parser.add_argument("--save-metrics-log", action="store_true", default=True,
                        help="Append per-iteration metrics to metrics.jsonl (default: True)")
    parser.add_argument("--no-save-metrics-log", dest="save_metrics_log", action="store_false")
    parser.add_argument("--screen-local-delta", action="store_true", default=True,
                        help="Add small immediate screen-local delta reward on legacy PPO path (default: True)")
    parser.add_argument("--no-screen-local-delta", dest="screen_local_delta", action="store_false")

    # --- Matchup ranking data ---
    parser.add_argument("--matchup-data-dir", type=str, default=str(DEFAULT_MATCHUP_DATA_DIR),
                        help="Directory containing offline card ranking data (JSONL + NPZ)")
    parser.add_argument("--matchup-batch-size", type=int, default=32,
                        help="Batch size for matchup ranking loss (default: 32)")
    parser.add_argument("--matchup-loss-weight", type=float, default=0.1,
                        help="Weight for matchup ranking loss (default: 0.1)")
    parser.add_argument("--matchup-warmup-iters", type=int, default=100,
                        help="Skip matchup loss for first N iterations (default: 100)")
    parser.add_argument("--matchup-blend-beta", type=float, default=0.0,
                        help="Blend matchup_score_head into teacher signal (0=off, 0.3=recommended)")
    parser.add_argument("--matchup-min-spread", type=float, default=0.001,
                        help="Filter out ranking samples with score spread below this (default: 0.001)")

    # --- Skada community priors ---
    parser.add_argument("--skada-prior-weight", type=float, default=0.15,
                        help="Blend weight for Skada community priors in counterfactual scoring (0=off, 0.15=recommended)")
    parser.add_argument("--skada-boss-weights", action="store_true", default=False,
                        help="Use Skada boss wipe rates to scale boss-entry quality bonus")
    parser.add_argument("--skada-db", type=str, default=None,
                        help="Path to Skada analytics SQLite DB (default: auto-detect)")

    # --- Combat teacher data (offline turn-solver teacher) ---
    parser.add_argument("--combat-teacher-data-dir", type=str, default=str(DEFAULT_COMBAT_TEACHER_DATA),
                        help="JSONL file or directory containing combat teacher dataset (from build_combat_teacher_dataset.py)")
    parser.add_argument("--combat-teacher-loss-weight", type=float, default=0.1,
                        help="Weight for combat teacher loss (default: 0.1)")
    parser.add_argument("--combat-teacher-batch-size", type=int, default=32,
                        help="Batch size for combat teacher loss (default: 32)")
    parser.add_argument("--combat-teacher-warmup-iters", type=int, default=0,
                        help="Skip combat teacher loss for first N iterations (default: 0)")

    # --- Step 2 / Phase 5: Macro Milestone PPO (boss-entry build quality) ---
    parser.add_argument("--boss-entry-quality-weight", type=float, default=0.0,
                        help="Step 2 / Phase 5: scale boss-entry quality milestone bonus "
                             "(potions/HP/relics, fires once when crossing floor 14->15+). "
                             "0.0 = disabled (default), 1.0 = full weight (~+0.95 max).")
    parser.add_argument("--early-damage-potion-penalty-weight", type=float, default=0.0,
                        help="Step 2 / Phase 5: scale per-use penalty for using a damage potion "
                             "before reaching the boss zone. 0.0 = disabled (default), 1.0 = -0.05/use.")

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if args.deterministic_policy and args.multi_process:
        logger.warning("Config: --deterministic-policy forces single-process inference for audit stability.")
        args.multi_process = False
    if args.episodes_per_iter == 0:
        args.episodes_per_iter = max(args.num_envs, 2)

    effective_counterfactual_scoring, effective_counterfactual_weight, cf_warnings = (
        _resolve_counterfactual_runtime(
            use_segment_collector=args.use_segment_collector,
            counterfactual_scoring=args.counterfactual_scoring,
            counterfactual_weight=args.counterfactual_weight,
        )
    )
    for warning_msg in cf_warnings:
        logger.warning("Config: %s", warning_msg)
    if args.ppo_minibatch != 32:
        logger.warning(
            "Config: ppo_minibatch=%d is outside the current ACT1 first-win profile (recommended: 32).",
            args.ppo_minibatch,
        )
    if args.use_segment_collector or args.counterfactual_scoring or args.kl_warmstart:
        logger.warning(
            "Config: running outside the current ACT1 first-win profile "
            "(segment=%s counterfactual=%s kl_warmstart=%s).",
            args.use_segment_collector,
            args.counterfactual_scoring,
            args.kl_warmstart,
        )
    if args.boss_entry_quality_weight != 0.0:
        logger.warning(
            "Config: Step 2 / Phase 5 boss-entry quality milestone enabled "
            "(weight=%.2f, max bonus ~%.2f at floor crossing 14->15+, "
            "early-dmg-pot penalty weight=%.2f)",
            args.boss_entry_quality_weight,
            0.95 * args.boss_entry_quality_weight,
            args.early_damage_potion_penalty_weight,
        )

    vocab = load_vocab()
    env_ports = [args.start_port + i for i in range(args.num_envs)]
    env_urls = [f"http://127.0.0.1:{p}" for p in env_ports]

    # Output
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / f"hybrid_{args.num_envs}env_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = vars(args).copy()
    for _key, _value in list(config_payload.items()):
        if isinstance(_value, Path):
            config_payload[_key] = str(_value)
    config_payload["effective_counterfactual_scoring"] = effective_counterfactual_scoring
    config_payload["effective_counterfactual_weight"] = effective_counterfactual_weight
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    metrics_log = output_dir / "metrics.jsonl" if args.save_metrics_log else None
    metrics_history: list[dict[str, Any]] = []
    health_monitor = TrainingHealthMonitor()
    health_check_interval = 25

    # Matchup ranking dataset (offline card ranking data)
    matchup_dataset = None
    if args.matchup_data_dir:
        from matchup_dataset import MatchupRankingDataset
        matchup_dataset = MatchupRankingDataset(
            args.matchup_data_dir,
            min_spread=args.matchup_min_spread,
        )
        logger.info("Matchup ranking dataset: %d samples from %s (filtered %d with spread < %.4f)",
                     len(matchup_dataset), args.matchup_data_dir,
                     matchup_dataset._filtered_count, args.matchup_min_spread)
        if len(matchup_dataset) > 0:
            stats = matchup_dataset.get_stats()
            logger.info("  avg_options=%.1f score_spread=%.4f skip_best=%.1f%%",
                         stats.get("avg_options", 0),
                         stats.get("score_std", 0),
                         stats.get("skip_best_rate", 0) * 100)

    # Combat teacher dataset (offline turn-solver teacher data)
    combat_teacher_dataset = None
    if args.combat_teacher_data_dir:
        from combat_teacher_dataset import load_combat_teacher_samples
        from train_combat_teacher import CombatTeacherTorchDataset
        ct_path = Path(args.combat_teacher_data_dir)
        ct_samples: list = []
        if ct_path.is_file():
            ct_samples = load_combat_teacher_samples(ct_path)
        elif ct_path.is_dir():
            for jsonl_file in sorted(ct_path.glob("*.jsonl")):
                ct_samples.extend(load_combat_teacher_samples(jsonl_file))
        if ct_samples:
            # Only use train-split samples
            ct_samples = [s for s in ct_samples if str(s.split or "train") == "train"]
            combat_teacher_dataset = CombatTeacherTorchDataset(ct_samples, vocab=vocab)
            logger.info("Combat teacher dataset: %d samples from %s",
                         len(combat_teacher_dataset), args.combat_teacher_data_dir)
        else:
            logger.warning("Combat teacher dataset: 0 samples found in %s", args.combat_teacher_data_dir)

    # Offline data saver
    episode_saver = None
    if args.save_offline_data:
        offline_dir = output_dir / "offline_data"
        episode_saver = EpisodeDataSaver(
            output_dir=offline_dir,
            min_floor=args.offline_min_floor,
        )
        logger.info("Offline data saver: floor >= %d → %s", args.offline_min_floor, offline_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info("Device: %s (AMP: %s)", device, use_amp)

    # For the STS2AI mainline, champion checkpoints are authoritative. If a
    # resume checkpoint carries retrieval-head metadata, automatically adopt the
    # matching retrieval architecture instead of requiring the caller to repeat
    # those flags on every invocation.
    use_symbolic_features = getattr(args, "retrieval_head", False)
    symbolic_proj_dim = getattr(args, "retrieval_proj_dim", 16)
    _resume_sources = [
        ("--resume", args.resume),
        ("--resume-ppo", args.resume_ppo),
        ("--resume-mcts", args.resume_mcts),
    ]
    for _arg_name, _ckpt_path in _resume_sources:
        if not _ckpt_path:
            continue
        try:
            _peek_ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
        except Exception as _e:
            logger.warning("Could not inspect %s for retrieval-head metadata: %s", _ckpt_path, _e)
            continue
        _ckpt_retrieval_proj_dim = _checkpoint_retrieval_proj_dim(_peek_ckpt)
        if _ckpt_retrieval_proj_dim > 0 and not use_symbolic_features:
            use_symbolic_features = True
            symbolic_proj_dim = _ckpt_retrieval_proj_dim
            logger.info(
                "Auto-enabled retrieval-head from %s checkpoint metadata (proj_dim=%d).",
                _arg_name,
                _ckpt_retrieval_proj_dim,
            )
        if _ckpt_retrieval_proj_dim > 0 and symbolic_proj_dim != _ckpt_retrieval_proj_dim:
            raise SystemExit(
                f"{_arg_name} checkpoint '{_ckpt_path}' expects retrieval proj_dim="
                f"{_ckpt_retrieval_proj_dim}, but this run requested "
                f"--retrieval-proj-dim {symbolic_proj_dim}. Re-run with "
                f"--retrieval-proj-dim {_ckpt_retrieval_proj_dim}."
            )

    # Load networks with shared embeddings
    ppo_net = FullRunPolicyNetworkV2(
        vocab=vocab,
        embed_dim=args.embed_dim,
        use_symbolic_features=use_symbolic_features,
        symbolic_proj_dim=symbolic_proj_dim,
    )
    deck_repr_dim = getattr(args, "deck_repr_dim", 0)

    # Auto-detect deck_repr_dim from checkpoint if --resume is provided so we
    # build a network architecture that matches the checkpoint exactly. Without
    # this, --resume of a deck-aware checkpoint silently drops the deck/pile
    # encoders and the state_encoder mismatches.
    if args.resume and deck_repr_dim == 0:
        try:
            _peek_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            _peek_state = _peek_ckpt.get("mcts_model", {})
            if isinstance(_peek_state, dict):
                for _k, _v in _peek_state.items():
                    if "deck_encoder" in _k and "attn.in_proj_weight" in _k:
                        deck_repr_dim = int(_v.shape[0]) // 3
                        logger.info("Auto-detected deck_repr_dim=%d from checkpoint %s",
                                    deck_repr_dim, args.resume)
                        break
            del _peek_ckpt, _peek_state
        except Exception as _e:
            logger.warning("Could not peek checkpoint for deck_repr_dim auto-detect: %s", _e)

    mcts_net = CombatPolicyValueNetwork(
        vocab=vocab, embed_dim=args.embed_dim,
        hidden_dim=args.combat_hidden_dim,
        entity_embeddings=ppo_net.entity_emb,  # shared embeddings
        deck_repr_dim=deck_repr_dim,
        residual_adapter=getattr(args, "residual_adapter", False),
        symbolic_head=ppo_net.symbolic_head,  # shared (may be None)
    )
    if use_symbolic_features:
        sym_param_count = sum(p.numel() for p in ppo_net.symbolic_head.parameters())
        logger.info(
            "SymbolicFeaturesHead enabled: %d trainable params (owned by PPO optimizer, "
            "combat optimizer will exclude symbolic_head.* to avoid double-step)",
            sym_param_count,
        )
        if getattr(args, "freeze_embeddings", False):
            logger.info(
                "SymbolicFeaturesHead: query side (entity_emb) is frozen via "
                "--freeze-embeddings; symbol side (symbol_embed + cross_attn) "
                "remains trainable."
            )
    start_iter = 0

    def _safe_load_state_dict(model, state_dict, label="model"):
        """Load state dict, handling shape mismatches with partial copy.

        For Linear weights where model dim > checkpoint dim (e.g. deck_repr_dim
        expansion), copies checkpoint columns into the first N columns of the
        model weight and zero-inits the rest.
        """
        current = model.state_dict()
        filtered = {}
        skipped = []
        partial = []
        for k, v in state_dict.items():
            if k in current and current[k].shape == v.shape:
                filtered[k] = v
            elif k in current and v.dim() == 2 and current[k].dim() == 2:
                # Linear weight: (out, in_new) vs (out, in_old)
                if current[k].shape[0] == v.shape[0] and current[k].shape[1] > v.shape[1]:
                    # Partial copy: old columns + zero-init new columns
                    new_w = torch.zeros_like(current[k])
                    new_w[:, :v.shape[1]] = v
                    filtered[k] = new_w
                    partial.append(f"{k}: {list(v.shape)}->{list(current[k].shape)}")
                else:
                    skipped.append(f"{k}: ckpt={list(v.shape)} vs model={list(current[k].shape)}")
            elif k in current:
                skipped.append(f"{k}: ckpt={list(v.shape)} vs model={list(current[k].shape)}")
        if partial:
            logger.info("Partial-loaded %d expanded params in %s: %s", len(partial), label, "; ".join(partial[:5]))
        if skipped:
            logger.warning("Skipped %d mismatched keys in %s: %s", len(skipped), label, "; ".join(skipped[:5]))
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            logger.info("New params in %s (randomly init): %d keys", label, len(missing))

    # --resume: load hybrid checkpoint (both networks)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "ppo_model" in ckpt:
            _safe_load_state_dict(ppo_net, ckpt["ppo_model"], "PPO")
            logger.info("Loaded PPO from hybrid checkpoint")
        if "mcts_model" in ckpt:
            _safe_load_state_dict(mcts_net, ckpt["mcts_model"], "combat")
            logger.info("Loaded combat policy from hybrid checkpoint")
        start_iter = ckpt.get("iteration", 0) + 1

    # --resume-ppo / --resume-mcts: load standalone checkpoints (override)
    if args.resume_ppo:
        ckpt = torch.load(args.resume_ppo, map_location="cpu", weights_only=False)
        if "ppo_model" in ckpt:
            _safe_load_state_dict(ppo_net, ckpt["ppo_model"], "PPO")
        elif "model_state_dict" in ckpt:
            _safe_load_state_dict(ppo_net, ckpt["model_state_dict"], "PPO")
        logger.info("Loaded PPO from %s", args.resume_ppo)

    if args.resume_mcts:
        ckpt = torch.load(args.resume_mcts, map_location="cpu", weights_only=False)
        if "mcts_model" in ckpt:
            _safe_load_state_dict(mcts_net, ckpt["mcts_model"], "combat")
        elif "model_state_dict" in ckpt:
            _safe_load_state_dict(mcts_net, ckpt["model_state_dict"], "combat")
        logger.info("Loaded combat policy from %s", args.resume_mcts)

    # Initialize combat deck_encoder from PPO deck_encoder (transfer learned representation)
    if deck_repr_dim > 0 and hasattr(mcts_net, 'deck_encoder') and hasattr(ppo_net, 'deck_encoder'):
        try:
            ppo_deck_sd = {k: v for k, v in ppo_net.deck_encoder.state_dict().items()}
            combat_deck_sd = mcts_net.deck_encoder.state_dict()
            matched = 0
            for k in combat_deck_sd:
                if k in ppo_deck_sd and combat_deck_sd[k].shape == ppo_deck_sd[k].shape:
                    combat_deck_sd[k] = ppo_deck_sd[k]
                    matched += 1
            if matched > 0:
                mcts_net.deck_encoder.load_state_dict(combat_deck_sd)
                logger.info("Initialized combat deck_encoder from PPO deck_encoder (%d/%d params copied)",
                            matched, len(combat_deck_sd))
        except Exception as e:
            logger.debug("Could not copy deck_encoder weights: %s", e)

    if args.boss_aware_warmup_only:
        trainable_params, total_params = _configure_boss_aware_warmup(ppo_net)
        logger.info(
            "Boss-aware warmup: training only new PPO boss-aware params (%d / %d trainable)",
            trainable_params,
            total_params,
        )

    ppo_net.to(device)
    mcts_net.to(device)

    ppo_trainer = PPOTrainerV2(
        network=ppo_net, lr=args.ppo_lr, ppo_epochs=args.ppo_epochs,
        minibatch_size=args.ppo_minibatch, entropy_coeff=args.ppo_entropy_coeff,
        clip_epsilon=args.ppo_clip, max_grad_norm=1.0,
        boss_readiness_coeff=args.boss_readiness_coeff,
    )

    mcts_config = MCTSConfig(num_simulations=args.mcts_sims, c_puct=1.5,
                              temperature=1.0, dirichlet_alpha=0.3, dirichlet_fraction=0.25)
    mcts_agent = CombatMCTSAgent(network=mcts_net, vocab=vocab, config=mcts_config,
                                  training=True, device=device)
    # Exclude shared symbolic_head params from the combat optimizer — the PPO
    # optimizer owns them. Combat's backward still accumulates gradients on
    # those params via autograd; they are consumed at PPO step time. This
    # avoids two Adam states updating the same parameter with independent
    # moving averages, which would cause thrashing.
    def _combat_trainable_params():
        return [
            p for n, p in mcts_net.named_parameters()
            if not n.startswith("symbolic_head.")
        ]
    if use_symbolic_features:
        excluded_combat = sum(
            p.numel() for n, p in mcts_net.named_parameters()
            if n.startswith("symbolic_head.")
        )
        logger.info(
            "SymbolicFeaturesHead: excluded %d params from combat optimizer "
            "(owned by PPO optimizer instead)", excluded_combat,
        )
    mcts_optimizer = torch.optim.Adam(
        _combat_trainable_params() if use_symbolic_features else mcts_net.parameters(),
        lr=args.mcts_lr,
        weight_decay=1e-4,
    )
    mcts_replay = MCTSReplayBuffer(max_size=args.mcts_replay_size)

    # Combat PPO trainer (shares mcts_net, uses its own optimizer with separate lr)
    # NOTE: CombatPPOTrainer builds its own optimizer from network.parameters()
    # in its __init__. When symbolic features are enabled we need to rebuild
    # that optimizer with the filtered param list. Done immediately below.
    combat_ppo_trainer = CombatPPOTrainer(
        network=mcts_net,
        lr=args.combat_ppo_lr,
        clip_epsilon=args.combat_ppo_clip,
        entropy_coeff=args.combat_ppo_entropy_coeff,
        ppo_epochs=args.combat_ppo_epochs,
        minibatch_size=args.combat_ppo_minibatch,
    )
    if use_symbolic_features:
        # Swap CombatPPOTrainer's internal optimizer to exclude symbolic_head.*
        # See _combat_trainable_params() above for the rationale. Match the
        # Adam defaults from CombatPPOTrainer.__init__ (line ~727).
        combat_ppo_trainer.optimizer = torch.optim.Adam(
            _combat_trainable_params(),
            lr=args.combat_ppo_lr,
        )

    # Residual adapter: freeze backbone, only train adapter heads
    if getattr(args, "residual_adapter", False) and deck_repr_dim > 0:
        adapter_prefixes = ("deck_encoder.", "delta_logits_head.", "delta_value_head.",
                            "adapter_alpha", "adapter_beta")
        frozen_count = 0
        trainable_count = 0
        for name, param in mcts_net.named_parameters():
            if any(name.startswith(p) for p in adapter_prefixes):
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        logger.info("Residual adapter: frozen %d params, trainable %d params (adapter only)",
                    frozen_count, trainable_count)

    # Freeze shared entity embeddings (GPT Pro: prevent combat/ranking gradient conflict)
    if getattr(args, "freeze_embeddings", False):
        frozen_emb = 0
        for name, param in ppo_net.named_parameters():
            if name.startswith("entity_emb."):
                param.requires_grad = False
                frozen_emb += param.numel()
        # Combat net shares entity_emb, so freezing ppo_net's also freezes combat's
        logger.info("Frozen shared entity embeddings: %d params", frozen_emb)

    # Freeze entire combat brain (splice diagnostic finding: combat475 > combat600)
    if getattr(args, "freeze_combat", False):
        frozen_combat = 0
        for name, param in mcts_net.named_parameters():
            param.requires_grad = False
            frozen_combat += param.numel()
        logger.info("Frozen entire combat brain: %d params (PPO-only training)", frozen_combat)

    # Freeze entire PPO brain (combat-only training to improve boss conversion)
    if getattr(args, "freeze_ppo", False):
        frozen_ppo = 0
        for name, param in ppo_net.named_parameters():
            if not name.startswith("entity_emb."):  # embeddings already handled above
                param.requires_grad = False
                frozen_ppo += param.numel()
        logger.info("Frozen entire PPO brain: %d params (combat-only training)", frozen_ppo)

    logger.info("PPO params: %d | MCTS params: %d | Envs: %d",
                ppo_net.param_count(), mcts_net.param_count(), len(env_ports))

    # Pipe clients — reuse from PipeBackedFullRunClient (single session per port)
    transport = args.transport or ("pipe-binary" if args.pipe else "http")
    use_pipe_transport = transport in {"pipe", "pipe-binary"}
    pipe_clients: dict[int, Any] = {}
    spawned_env_procs: list[Any] = []

    logger.info("Starting hybrid training from iter %d (output: %s)", start_iter, output_dir)

    def _cleanup_spawned_envs() -> None:
        while spawned_env_procs:
            stop_process(spawned_env_procs.pop())

    if args.auto_launch:
        if not use_pipe_transport:
            logger.warning("--auto-launch is only supported for pipe transports; ignoring for transport=%s", transport)
        else:
            launch_protocol = "bin" if transport == "pipe-binary" else "json"
            atexit.register(_cleanup_spawned_envs)
            logger.info(
                "Auto-launching %d fresh Sim hosts from %s on ports %s (%s)",
                len(env_ports),
                Path(args.headless_dll).resolve(),
                ",".join(str(port) for port in env_ports),
                launch_protocol,
            )
            for port in env_ports:
                spawned_env_procs.append(
                    start_headless_sim(
                        port=port,
                        repo_root=args.repo_root,
                        dll_path=args.headless_dll,
                        protocol=launch_protocol,
                    )
                )

    # Create persistent clients (one per env). PipeBackedFullRunClient
    # owns the pipe connection; MCTS reuses it via client._pipe.
    env_clients: dict[int, Any] = {}
    for port in env_ports:
        if use_pipe_transport:
            client = create_full_run_client(port=port, use_pipe=True, transport=transport, ready_timeout_s=15.0)
            client._ensure_connected()
            env_clients[port] = client
            pipe_clients[port] = client._pipe  # share pipe for MCTS
            logger.info("Pipe client ready: port %d transport=%s", port, transport)
        else:
            url = f"http://127.0.0.1:{port}"
            env_clients[port] = ApiBackedFullRunClient(
                base_url=url, poll_interval_s=0.005, request_timeout_s=60.0)

    if use_pipe_transport and not pipe_clients:
        logger.error("No pipe connections!")
        _cleanup_spawned_envs()
        return 1

    # --- Load ORT model into each HeadlessSim (if --local-ort) ---
    if args.local_ort and args.ort_model_path and use_pipe_transport:
        import os
        ort_abs = os.path.abspath(args.ort_model_path)
        for port, raw_pipe in pipe_clients.items():
            try:
                result = raw_pipe.call("load_ort_model", {"path": ort_abs})
                loaded = result.get("loaded", False)
                logger.info("ORT model loaded on port %d: %s", port, "OK" if loaded else "FAILED")
            except Exception as e:
                logger.warning("ORT model load failed on port %d: %s", port, e)

    # --- Multi-process batch inference setup ---
    inf_server = None
    inf_clients: dict[int, Any] = {}
    if args.multi_process and not args.local_ort:
        from inference_server import InferenceServer, InferenceClient
        inf_server = InferenceServer(
            ppo_net=ppo_net, combat_net=mcts_net, device=device,
            num_workers=len(env_ports),
            max_batch=len(env_ports),
            timeout_s=args.batch_timeout_ms / 1000.0,
        )
        inf_server.start()
        for i in range(len(env_ports)):
            inf_clients[i] = InferenceClient(
                worker_id=i,
                request_queue=inf_server.request_queue,
                result_queue=inf_server.get_result_queue(i),
            )
        logger.info("Multi-process batch inference enabled (%d workers)", len(env_ports))
    elif args.local_ort and args.multi_process:
        # ORT mode: combat in C#, non-combat uses CPU policy snapshot.
        # Zero CUDA in worker threads — eliminates multi-thread CUDA contention.
        logger.info("Local ORT mode: combat in C#, non-combat CPU snapshot (%d workers)", len(env_ports))

    def _collect(env_idx: int,
                 combat_buf: CombatRolloutBuffer | None = None):
        port = env_ports[env_idx % len(env_ports)]
        episode_seed = None if args.seed is None else f"audit-{args.seed}-iter{iteration:05d}-ep{env_idx:03d}"
        client = env_clients.get(port)
        # Skip dead envs immediately
        if client is not None and hasattr(client, 'is_dead') and client.is_dead:
            return StructuredRolloutBuffer(), [], {"error": f"port {port} is dead", "floors": 0}
        pipe_getter = (lambda c=client: getattr(c, "_pipe", None)) if hasattr(client, "_pipe") else None
        pipe = pipe_getter if pipe_getter is not None else pipe_clients.get(port)
        # Use inference client for multi-process mode
        inf_client = inf_clients.get(env_idx % len(env_ports)) if args.multi_process else None
        # Use CPU snapshots if zero-CUDA mode (no CUDA in worker threads)
        _ppo = _cpu_ppo_net if _use_zero_cuda else ppo_net
        _mcts = _cpu_mcts_agent if _use_zero_cuda else mcts_agent
        return collect_unified_episode(
            _ppo, _mcts, vocab, pipe, client,
            character_id=args.character_id,
            seed=episode_seed,
            episode_timeout=args.episode_timeout,
            max_steps=args.max_episode_steps,
            use_mcts_combat=args.mcts,
            combat_buffer=combat_buf,
            inference_client=inf_client,
            use_segment_collector=args.use_segment_collector,
            counterfactual_scoring=effective_counterfactual_scoring,
            counterfactual_weight=effective_counterfactual_weight,
            screen_local_delta=args.screen_local_delta,
            deterministic_policy=args.deterministic_policy,
            episode_saver=episode_saver,
            use_local_ort=args.local_ort,
            ppo_ort_session=_ppo_ort_session,
            boss_entry_quality_weight=args.boss_entry_quality_weight,
            early_damage_potion_penalty_weight=args.early_damage_potion_penalty_weight,
        )

    # Load Skada community priors (card quality, synergies, boss difficulty)
    _skada_priors_obj = None
    if args.skada_prior_weight > 0 or args.skada_boss_weights:
        try:
            from skada.skada_priors import SkadaPriors
            _skada_priors_obj = SkadaPriors(args.skada_db)
            if _skada_priors_obj.loaded:
                logger.info("Skada priors loaded: %d cards, %d relics, %d synergies, %d bosses",
                            _skada_priors_obj.num_cards, _skada_priors_obj.num_relics,
                            _skada_priors_obj.num_synergies, _skada_priors_obj.num_bosses)
                if args.skada_boss_weights:
                    from rl_reward_shaping import load_skada_boss_difficulty
                    load_skada_boss_difficulty(_skada_priors_obj)
            else:
                logger.warning("Skada DB not found — skada priors disabled")
                _skada_priors_obj = None
        except Exception as e:
            logger.warning("Failed to load Skada priors: %s", e)
            _skada_priors_obj = None

    # Register learned card evaluator for counterfactual scoring
    if effective_counterfactual_scoring or args.matchup_blend_beta > 0 or args.skada_prior_weight > 0:
        from counterfactual_scoring import set_learned_evaluator
        # Blend alpha ramps up over training: start with heuristic, gradually trust learned
        _initial_alpha = 0.3 if start_iter > 200 else 0.0
        _matchup_beta = args.matchup_blend_beta
        _skada_gamma = args.skada_prior_weight if _skada_priors_obj is not None else 0.0
        set_learned_evaluator(
            ppo_net, vocab,
            alpha=_initial_alpha,
            matchup_beta=_matchup_beta,
            skada_priors=_skada_priors_obj,
            skada_gamma=_skada_gamma,
        )
        logger.info("Learned card evaluator registered (alpha=%.2f, matchup_beta=%.2f, skada_gamma=%.2f)",
                     _initial_alpha, _matchup_beta, _skada_gamma)

    # --- Zero-CUDA collector: CPU policy snapshot for worker threads ---
    _cpu_ppo_net = None
    _cpu_mcts_net = None
    _cpu_mcts_agent = None
    _use_zero_cuda = args.zero_cuda_collector and args.multi_process
    _ppo_ort_session = None  # ORT CPU session for non-combat (Branch C)
    if _use_zero_cuda:
        import copy
        _cpu_ppo_net = copy.deepcopy(ppo_net).cpu().eval()
        _cpu_mcts_net = copy.deepcopy(mcts_net).cpu().eval()
        _cpu_mcts_agent = CombatMCTSAgent(
            network=_cpu_mcts_net, vocab=vocab, config=mcts_config,
            training=False, device=torch.device("cpu"))

        # Export PPO actor to ONNX for ORT CPU inference (Branch C)
        try:
            import onnxruntime as ort
            _ppo_onnx_path = str(output_dir / "ppo_actor.onnx")
            _export_ppo_actor_onnx(_cpu_ppo_net, _ppo_onnx_path, vocab)
            _ppo_ort_opts = ort.SessionOptions()
            _ppo_ort_opts.intra_op_num_threads = 1
            _ppo_ort_opts.inter_op_num_threads = 1
            _ppo_ort_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            _ppo_ort_session = ort.InferenceSession(_ppo_onnx_path, _ppo_ort_opts, providers=["CPUExecutionProvider"])
            logger.info("Zero-CUDA collector: PPO ORT CPU session created (0.5ms vs PyTorch 6.5ms)")
        except Exception as _ort_err:
            logger.warning("PPO ORT export failed, falling back to CPU PyTorch: %s", _ort_err)

        logger.info("Zero-CUDA collector: CPU policy snapshots created (no CUDA in worker threads)")

    try:
        end_iter = start_iter + args.max_iterations
        for iteration in range(start_iter, end_iter):
            if _shutdown_requested:
                break
            iter_start = time.monotonic()

            # Refresh CPU snapshots + ORT sessions every iteration (zero-CUDA mode)
            if _use_zero_cuda and _cpu_ppo_net is not None:
                _cpu_ppo_net.load_state_dict({k: v.cpu() for k, v in ppo_net.state_dict().items()})
                _cpu_mcts_net.load_state_dict({k: v.cpu() for k, v in mcts_net.state_dict().items()})
                # Re-export PPO ORT every 25 iter (amortize ~800ms export cost)
                if _ppo_ort_session is not None and iteration % 25 == 0 and iteration > 0:
                    try:
                        _export_ppo_actor_onnx(_cpu_ppo_net, _ppo_onnx_path, vocab)
                        _ppo_ort_session = ort.InferenceSession(_ppo_onnx_path, _ppo_ort_opts, providers=["CPUExecutionProvider"])
                    except Exception:
                        pass

                # Export fresh combat ONNX and hot-reload into C# sims
                # Only export every 25 iterations to amortize overhead (~1-2s per export)
                _ort_refresh_interval = 25
                if args.local_ort and use_pipe_transport and iteration % _ort_refresh_interval == 0:
                    import os
                    from export_actor_onnx import export_from_training_snapshot
                    _onnx_path = str(output_dir / f"actor_v{iteration:05d}.onnx")
                    try:
                        _export_ms = export_from_training_snapshot(
                            ppo_net.state_dict(), mcts_net.state_dict(), vocab, _onnx_path,
                            policy_version=iteration)
                        for _port, _pipe in pipe_clients.items():
                            _pipe.call("load_ort_model", {"path": os.path.abspath(_onnx_path)})
                        if iteration == 0 or iteration % 25 == 0:
                            logger.info("ORT v%d exported (%.0fms) + loaded into %d sims",
                                        iteration, _export_ms, len(pipe_clients))
                        # Clean up old ONNX files (keep only current)
                        if iteration > 0:
                            _old = str(output_dir / f"actor_v{iteration-1:05d}.onnx")
                            try:
                                os.remove(_old)
                            except OSError:
                                pass
                    except Exception as _e:
                        logger.warning("ORT export/reload failed iter %d: %s", iteration, _e)

            # --- Collect episodes ---

            # Combat: MCTS search (high-quality actions) + Combat PPO (per-step reward learning)
            if iteration == 0:
                if args.mcts:
                    logger.info("Combat search mode: MCTS selects actions, PPO learns value/aux targets")
                else:
                    logger.info("Pure PPO mode: combat NN selects actions directly (no search)")
            ppo_net.eval()
            mcts_net.eval()
            ppo_buffer = StructuredRolloutBuffer()
            combat_buffer = CombatRolloutBuffer()
            _ep_counter = 0
            new_mcts = 0
            total_floors = 0
            total_combats = 0
            victories = 0
            ppo_steps = 0
            mcts_decisions = 0
            combat_ppo_steps = 0
            boss_reached_eps = 0
            act1_cleared_eps = 0
            boss_hp_fracs: list[float] = []
            _iter_ort_combat_time = 0.0
            _iter_ort_combat_calls = 0
            _iter_nc_forward_time = 0.0
            _iter_nc_forward_calls = 0
            deck_sizes_at_boss: list[int] = []
            card_reward_screens = 0
            card_reward_skips = 0
            ep_times = []
            iter_slow_steps = 0
            iter_max_step_ms = 0.0
            iter_timeout_count = 0

            def _merge_episode(ep_ppo, ep_mcts, ep_stats):
                """Merge single episode results into iteration-level accumulators."""
                nonlocal new_mcts, total_floors, total_combats, victories
                nonlocal ppo_steps, mcts_decisions, combat_ppo_steps
                nonlocal boss_reached_eps, act1_cleared_eps
                nonlocal boss_hp_fracs, deck_sizes_at_boss, card_reward_screens, card_reward_skips
                nonlocal iter_slow_steps, iter_max_step_ms, iter_timeout_count
                nonlocal _ep_counter
                nonlocal _iter_ort_combat_time, _iter_ort_combat_calls
                nonlocal _iter_nc_forward_time, _iter_nc_forward_calls

                # Merge PPO buffer — weight rewards by floor reached
                # Higher floor episodes get amplified rewards (learn from better runs)
                _ep_floor = ep_stats.get("floors", 0)
                _floor_weight = 1.0 + max(0, _ep_floor - 3) * 0.15
                # floor 3: 1.0x, floor 5: 1.3x, floor 8: 1.75x, floor 10: 2.05x
                for i in range(len(ep_ppo)):
                    ppo_buffer.states.append(ep_ppo.states[i])
                    ppo_buffer.actions_data.append(ep_ppo.actions_data[i])
                    ppo_buffer.action_indices.append(ep_ppo.action_indices[i])
                    ppo_buffer.log_probs.append(ep_ppo.log_probs[i])
                    ppo_buffer.rewards.append(ep_ppo.rewards[i] * _floor_weight)
                    ppo_buffer.values.append(ep_ppo.values[i])
                    ppo_buffer.dones.append(ep_ppo.dones[i])
                    ppo_buffer.floor_targets.append(ep_ppo.floor_targets[i])
                    ppo_buffer.boss_readiness_targets.append(ep_ppo.boss_readiness_targets[i])

                # Merge combat PPO buffer
                ep_combat = ep_stats.get("_combat_buffer")
                boss_only = getattr(args, "combat_boss_only", False)
                monster_reward_weight = float(getattr(args, "combat_monster_reward_weight", 1.0))
                boss_screens = {"boss", "elite"}
                if ep_combat is not None:
                    for i in range(len(ep_combat)):
                        # Encounter-gated: skip hallway if --combat-boss-only
                        step_screen = ep_combat.screen_types[i] if ep_combat.screen_types and i < len(ep_combat.screen_types) else ""
                        is_boss_or_elite = step_screen in boss_screens
                        if boss_only and ep_combat.screen_types and not is_boss_or_elite:
                            continue
                        combat_buffer.state_features.append(ep_combat.state_features[i])
                        combat_buffer.action_features.append(ep_combat.action_features[i])
                        combat_buffer.action_indices.append(ep_combat.action_indices[i])
                        combat_buffer.log_probs.append(ep_combat.log_probs[i])
                        # Apply reward weight: monster fights get down-weighted if < 1.0
                        step_reward = ep_combat.rewards[i]
                        if (not boss_only) and (monster_reward_weight != 1.0) and (not is_boss_or_elite) and step_screen:
                            step_reward = step_reward * monster_reward_weight
                        combat_buffer.rewards.append(step_reward)
                        combat_buffer.values.append(ep_combat.values[i])
                        combat_buffer.dones.append(ep_combat.dones[i])
                        if ep_combat.screen_types:
                            combat_buffer.screen_types.append(step_screen)

                # Merge segment buffer (Phase 2) — segment data goes into ppo_buffer
                # by converting segments to step-equivalent entries
                ep_seg = ep_stats.get("_segment_buffer")
                if ep_seg is not None and len(ep_seg) > 0:
                    for seg in ep_seg.segments:
                        ppo_buffer.states.append(seg.state)
                        ppo_buffer.actions_data.append(seg.actions)
                        ppo_buffer.action_indices.append(seg.action_idx)
                        ppo_buffer.log_probs.append(seg.log_prob)
                        ppo_buffer.rewards.append(seg.reward_sum * _floor_weight)
                        ppo_buffer.values.append(seg.value)
                        ppo_buffer.dones.append(seg.done)
                        ppo_buffer.floor_targets.append(seg.floor_target)
                        ppo_buffer.boss_readiness_targets.append(0.0)

                for ex in ep_mcts:
                    mcts_replay.add(ex)
                    new_mcts += 1

                total_floors += ep_stats.get("floors", 0)
                total_combats += ep_stats.get("combats", 0)
                ppo_steps += ep_stats.get("ppo_steps", 0)
                mcts_decisions += ep_stats.get("mcts_decisions", 0)
                combat_ppo_steps += ep_stats.get("combat_ppo_steps", 0)
                boss_reached_eps += int(bool(ep_stats.get("boss_reached")))
                act1_cleared_eps += int(bool(ep_stats.get("act1_cleared")))
                boss_hp_fracs.extend(ep_stats.get("boss_hp_fraction_dealt", []))
                deck_sizes_at_boss.extend(ep_stats.get("deck_size_at_boss", []))
                card_reward_screens += ep_stats.get("card_reward_screens", 0)
                card_reward_skips += ep_stats.get("card_reward_skips", 0)

                # Per-episode summary (sample log every 50 iter)
                if iteration % 50 == 0:
                    _fl = ep_stats.get("floors", 0)
                    _cw = ep_stats.get("combats_won", 0)
                    _ct = ep_stats.get("combats", 0)
                    _cards = ep_stats.get("cards_taken", [])
                    _skip = ep_stats.get("cards_skipped", 0)
                    _de = ep_stats.get("death_enemy", "N/A")
                    _hp = ep_stats.get("hp_timeline", [])
                    _out = ep_stats.get("outcome", "?")
                    logger.info("  EP: %s f%d | %dW/%d combats | cards: %s skip:%d | hp:%s | died_to:%s",
                                _out, _fl, _cw, _ct,
                                ",".join(_cards[:5]) if _cards else "none", _skip,
                                "->".join(str(h) for h in _hp[-5:]) if _hp else "?",
                                _de)
                ep_times.append(ep_stats.get("episode_time_s", 0))
                iter_slow_steps += ep_stats.get("slow_steps", 0)
                iter_max_step_ms = max(iter_max_step_ms, ep_stats.get("max_step_ms", 0))
                iter_timeout_count += ep_stats.get("timeout_count", 0)
                if ep_stats.get("outcome") == "victory":
                    victories += 1

                # Accumulate timing stats from zero-CUDA paths
                _iter_ort_combat_time += ep_stats.get("_ort_combat_time", 0.0)
                _iter_ort_combat_calls += ep_stats.get("_ort_combat_calls", 0)
                _iter_nc_forward_time += ep_stats.get("_nc_forward_time", 0.0)
                _iter_nc_forward_calls += ep_stats.get("_nc_forward_calls", 0)

                # --- Episode trace: dump EVERY episode to file ---
                trace = ep_stats.get("_episode_trace", [])
                ep_error = ep_stats.get("error")
                ep_outcome = ep_stats.get("outcome", "unknown")
                ep_end_reason = ep_stats.get("end_reason")
                ep_floor = ep_stats.get("floors", 0)
                ep_combats = ep_stats.get("combats", 0)
                ep_time = ep_stats.get("episode_time_s", 0)

                if ep_error and trace:
                    logger.warning("Episode ERROR (%s) trace:\n  %s",
                                   ep_error, "\n  ".join(trace[-30:]))

                # Write every episode trace to replays/ subdirectory
                if args.save_replay_traces and trace:
                    replay_dir = output_dir / "replays"
                    replay_dir.mkdir(exist_ok=True)
                    if ep_error:
                        tag = "ERR"
                    elif ep_outcome:
                        tag = str(ep_outcome).upper()[:3]
                    elif ep_end_reason == "max_steps":
                        tag = "MAX"
                    else:
                        tag = "UNK"
                    trace_path = replay_dir / f"i{iteration:05d}_e{_ep_counter:03d}_{tag}_f{ep_floor}.txt"
                    try:
                        with open(trace_path, "w", encoding="utf-8") as f:
                            f.write(f"# Iter {iteration} episode {_ep_counter}\n")
                            f.write(f"# outcome={ep_outcome} floor={ep_floor} "
                                    f"combats={ep_combats} time={ep_time:.1f}s "
                                    f"end_reason={ep_end_reason} error={ep_error}\n\n")
                            f.write("\n".join(trace))
                    except Exception:
                        pass
                _ep_counter += 1

            num_workers = min(len(env_ports), args.episodes_per_iter)
            if args.vectorized:
                # Vectorized: all envs step in lockstep (parallel pipe I/O + batch NN)
                from vectorized_collector import collect_vectorized_episodes
                vec_clients = [env_clients[p] for p in env_ports]
                vec_ppo, vec_combat, vec_stats = collect_vectorized_episodes(
                    ppo_net=ppo_net,
                    combat_net=mcts_net,
                    vocab=vocab,
                    clients=vec_clients,
                    character_id=args.character_id,
                    max_steps=args.max_episode_steps,
                    episode_timeout=args.episode_timeout,
                    screen_local_delta=args.screen_local_delta,
                    device=device,
                )
                # Merge vectorized results into iteration buffers
                env_floor_map = {s["env_id"]: s["floors"] / 16.0 for s in vec_stats}
                for t in vec_ppo:
                    if t.get("type") == "nc_step" and "state_features" in t:
                        floor_target = env_floor_map.get(t.get("env_id", -1), 0.0)
                        ppo_buffer.states.append(t["state_features"])
                        ppo_buffer.actions_data.append(t["action_features"])
                        ppo_buffer.action_indices.append(t["action_idx"])
                        ppo_buffer.log_probs.append(t["log_prob"])
                        ppo_buffer.rewards.append(t["reward"])
                        ppo_buffer.values.append(t["value"])
                        ppo_buffer.dones.append(False)
                        ppo_buffer.floor_targets.append(floor_target)
                        ppo_buffer.boss_readiness_targets.append(0.0)
                        ppo_steps += 1
                for t in vec_combat:
                    if "state_features" in t:
                        combat_buffer.add(
                            sf=t["state_features"],
                            af=t["action_features"],
                            action_idx=t["action_idx"],
                            log_prob=t["log_prob"],
                            reward=0.0,
                            value=t["value"],
                            done=False,
                        )
                        combat_ppo_steps += 1
                for s in vec_stats:
                    total_floors += s["floors"]
                    total_combats += s["combats"]
                    if s["outcome"] == "victory":
                        victories += 1
                        act1_cleared_eps += 1
                    if s["boss_reached"]:
                        boss_reached_eps += 1
                        if s["boss_hp_peak"] > 0:
                            boss_hp_fracs.append(s["boss_hp_peak"])
                        for ds in s["deck_size_at_boss"]:
                            deck_sizes_at_boss.append(ds)
                    card_reward_skips += s["cards_skipped"]
                    card_reward_screens += s["cards_skipped"] + len(s["cards_taken"])
                    ep_times.append(0.0)
            elif num_workers > 1:
                # Time-budget collection: submit all episodes, harvest completed
                # ones within a time limit. Prevents straggler episodes from
                # blocking the entire iteration (critical for --local-ort mode
                # where combat is coarse-grained).
                iter_budget_s = args.iter_time_budget if hasattr(args, 'iter_time_budget') and args.iter_time_budget > 0 else 0
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    env_idx = 0
                    futures = {}
                    for _ in range(min(num_workers, args.episodes_per_iter)):
                        futures[executor.submit(_collect, env_idx)] = env_idx
                        env_idx += 1
                    remaining = args.episodes_per_iter - len(futures)
                    harvest_deadline = time.monotonic() + iter_budget_s if iter_budget_s > 0 else float("inf")
                    harvested = 0
                    skipped = 0

                    while futures:
                        if _shutdown_requested:
                            break
                        # Check time budget
                        time_left = harvest_deadline - time.monotonic()
                        if iter_budget_s > 0 and time_left <= 0 and harvested >= 4:
                            # Time's up and we have enough episodes — skip remaining
                            skipped = len(futures)
                            for f in futures:
                                f.cancel()
                            break
                        budget_timeout = max(0.1, min(time_left, 5.0)) if iter_budget_s > 0 else None
                        try:
                            for f in as_completed(futures, timeout=budget_timeout):
                                fidx = futures.pop(f)
                                try:
                                    ep_ppo, ep_mcts, ep_stats = f.result()
                                except Exception as e:
                                    logger.warning("Episode %d failed: %s", fidx, e)
                                    ep_ppo, ep_mcts, ep_stats = StructuredRolloutBuffer(), [], {"error": str(e)}

                                _merge_episode(ep_ppo, ep_mcts, ep_stats)
                                harvested += 1

                                if remaining > 0 and not _shutdown_requested and time.monotonic() < harvest_deadline:
                                    futures[executor.submit(_collect, env_idx)] = env_idx
                                    env_idx += 1
                                    remaining -= 1
                                break
                        except TimeoutError:
                            # Time budget exceeded — skip remaining futures
                            if harvested >= 4:
                                skipped = len(futures)
                                for f in futures:
                                    f.cancel()
                                futures.clear()
                    if skipped > 0:
                        iter_timeout_count += skipped
            else:
                for ep in range(args.episodes_per_iter):
                    if _shutdown_requested:
                        break
                    ep_ppo, ep_mcts, ep_stats = _collect(ep)
                    _merge_episode(ep_ppo, ep_mcts, ep_stats)

            _collect_end = time.monotonic()

            # --- Train PPO ---
            ppo_metrics = {"ppo_ploss": 0, "ppo_vloss": 0, "ppo_entropy": 0, "boss_readiness_loss": 0}
            if len(ppo_buffer) >= 4 and not getattr(args, "freeze_ppo", False):
                ppo_net.train()
                ppo_buffer.compute_gae()
                # Phase 4: KL warm-start beta schedule
                _kl_beta = 0.0
                if args.kl_warmstart:
                    _progress = min(iteration / max(1, args.kl_beta_decay), 1.0)
                    _kl_beta = args.kl_beta_start * max(0, 1.0 - _progress) + 0.05 * _progress
                ppo_metrics = ppo_trainer.update(
                    ppo_buffer,
                    per_screen_adv_norm=args.per_screen_adv_norm,
                    weighted_screen_sampling=args.weighted_screen_sampling,
                    kl_beta=_kl_beta,
                )
                ppo_buffer.clear()

            # --- Train MCTS (behavior cloning from MCTS visit distributions) ---
            mcts_metrics = {"mcts_ploss": 0, "mcts_vloss": 0}
            if len(mcts_replay) >= args.mcts_batch_size and new_mcts > 0:
                mcts_net.train()
                for _ in range(args.mcts_train_steps):
                    batch = mcts_replay.sample(args.mcts_batch_size)
                    mcts_metrics = mcts_train_step(mcts_net, mcts_optimizer, batch,
                                                       device=device, use_amp=use_amp)

            # --- Train Combat PPO ---
            combat_ppo_metrics = {"combat_ppo_ploss": 0, "combat_ppo_vloss": 0, "combat_entropy": 0}
            if len(combat_buffer) >= 32 and not getattr(args, "freeze_combat", False):
                mcts_net.train()
                combat_ppo_metrics = combat_ppo_trainer.update(combat_buffer)
                combat_buffer.clear()
            elif len(combat_buffer) >= 32:
                combat_buffer.clear()  # discard combat data when frozen

            # --- Train matchup ranking (offline card ranking data) ---
            matchup_rank_loss_val = 0.0
            if (matchup_dataset is not None
                    and iteration >= args.matchup_warmup_iters
                    and len(matchup_dataset) > 0
                    and not getattr(args, "freeze_ppo", False)):
                ppo_net.train()
                mb = matchup_dataset.sample_batch(args.matchup_batch_size, device=device)
                if mb is not None and "state_tensors" in mb and "action_tensors" in mb:
                    pred_scores_full = ppo_net.compute_matchup_scores(
                        mb["state_tensors"], mb["action_tensors"])
                    # pred_scores is (B, MAX_ACTIONS=30), target is (B, MAX_OPTIONS=4)
                    # Slice to match option count
                    n_opts = mb["target_scores"].shape[1]
                    pred_scores = pred_scores_full[:, :n_opts]
                    from ranking_loss import listwise_ranking_loss
                    rank_loss = listwise_ranking_loss(
                        pred_scores, mb["target_scores"], mb["option_mask"])
                    decay_tau = getattr(args, "matchup_loss_decay_tau", 0.0)
                    if decay_tau > 0:
                        import math
                        matchup_w = args.matchup_loss_weight * math.exp(-iteration / decay_tau)
                    else:
                        matchup_w = args.matchup_loss_weight
                    total_rloss = matchup_w * rank_loss
                    ppo_trainer.optimizer.zero_grad()
                    total_rloss.backward()
                    ppo_trainer.optimizer.step()
                    matchup_rank_loss_val = rank_loss.item()

            # --- Train combat teacher (offline turn-solver teacher data) ---
            ct_loss_val = 0.0
            ct_ce_val = 0.0
            ct_rank_val = 0.0
            ct_cont_val = 0.0
            if (combat_teacher_dataset is not None
                    and iteration >= args.combat_teacher_warmup_iters
                    and len(combat_teacher_dataset) > 0
                    and not getattr(args, "freeze_combat", False)):
                from train_combat_teacher import (
                    _regret_weighted_pairwise_ranking,
                    _stack_batch as _ct_stack_batch,
                )
                mcts_net.train()
                # Sample a random batch
                ct_bs = min(args.combat_teacher_batch_size, len(combat_teacher_dataset))
                ct_indices = random.sample(range(len(combat_teacher_dataset)), ct_bs)
                ct_raw_batch = [combat_teacher_dataset[i] for i in ct_indices]
                ct_batch = _ct_stack_batch(ct_raw_batch, device)

                ct_logits, _ct_value, ct_action_scores, ct_continuation = mcts_net.forward_teacher(
                    ct_batch["state_features"], ct_batch["action_features"])
                ct_action_mask = ct_batch["action_features"]["action_mask"]
                ct_masked_scores = ct_action_scores.masked_fill(~ct_action_mask, -1e9)

                # Teacher best-action CE (on action_score head)
                ct_ce = F.cross_entropy(ct_masked_scores, ct_batch["teacher_best_action_index"])

                # Regret-weighted pairwise ranking (clamp regrets to avoid numerical explosion)
                ct_regrets_clamped = ct_batch["regrets"].clamp(max=10.0)
                ct_rank = _regret_weighted_pairwise_ranking(
                    ct_masked_scores, ct_regrets_clamped,
                    ct_batch["teacher_best_action_index"], ct_action_mask,
                    ct_batch["sample_weight"])

                # Continuation value regression (win_prob, hp_loss, potion_cost)
                ct_cont = F.mse_loss(ct_continuation, ct_batch["continuation_targets"])

                ct_total = args.combat_teacher_loss_weight * (ct_ce + ct_rank + ct_cont)
                combat_ppo_trainer.optimizer.zero_grad()
                ct_total.backward()
                torch.nn.utils.clip_grad_norm_(mcts_net.parameters(), 1.0)
                combat_ppo_trainer.optimizer.step()

                ct_loss_val = ct_total.item()
                ct_ce_val = ct_ce.item()
                ct_rank_val = ct_rank.item()
                ct_cont_val = ct_cont.item()

            _update_end = time.monotonic()
            iter_time = _update_end - iter_start
            _collect_time = _collect_end - iter_start
            _update_time = _update_end - _collect_end
            avg_floor = total_floors / max(1, args.episodes_per_iter)
            avg_ep = sum(ep_times) / max(1, len(ep_times))
            boss_reach_rate = boss_reached_eps / max(1, args.episodes_per_iter)
            act1_clear_rate = act1_cleared_eps / max(1, args.episodes_per_iter)
            boss_hp_fraction_mean = (
                float(np.mean(boss_hp_fracs)) if boss_hp_fracs else 0.0
            )
            deck_size_at_boss_mean = (
                float(np.mean(deck_sizes_at_boss)) if deck_sizes_at_boss else 0.0
            )
            card_reward_skip_rate = (
                card_reward_skips / max(1, card_reward_screens)
            )

            entry = {
                "iteration": iteration,
                "avg_floor": avg_floor,
                "victories": victories,
                "episodes": args.episodes_per_iter,
                "combats": total_combats,
                "ppo_steps": ppo_steps,
                "mcts_decisions": mcts_decisions,
                "combat_decisions": mcts_decisions,
                "mcts_replay": len(mcts_replay),
                "combat_search_replay": len(mcts_replay),
                "new_mcts": new_mcts,
                "new_combat_search_samples": new_mcts,
                "ppo_ploss": ppo_metrics.get("policy_loss", ppo_metrics.get("ppo_ploss", 0)),
                "ppo_vloss": ppo_metrics.get("value_loss", ppo_metrics.get("ppo_vloss", 0)),
                "ppo_entropy": ppo_metrics.get("entropy", ppo_metrics.get("ppo_entropy", 0)),
                "boss_readiness_loss": ppo_metrics.get("boss_readiness_loss", 0),
                "mcts_ploss": mcts_metrics.get("mcts_ploss", 0),
                "mcts_vloss": mcts_metrics.get("mcts_vloss", 0),
                "combat_search_ploss": mcts_metrics.get("mcts_ploss", 0),
                "combat_search_vloss": mcts_metrics.get("mcts_vloss", 0),
                "combat_ppo_steps": combat_ppo_steps,
                "combat_ppo_ploss": combat_ppo_metrics.get("combat_ppo_ploss", 0),
                "combat_ppo_vloss": combat_ppo_metrics.get("combat_ppo_vloss", 0),
                "combat_entropy": combat_ppo_metrics.get("combat_entropy", 0),
                "matchup_rank_loss": round(matchup_rank_loss_val, 6),
                "combat_teacher_loss": round(ct_loss_val, 6),
                "combat_teacher_ce": round(ct_ce_val, 6),
                "combat_teacher_rank": round(ct_rank_val, 6),
                "combat_teacher_cont": round(ct_cont_val, 6),
                "avg_ep_time": avg_ep,
                "iter_time_s": iter_time,
                "collect_time_s": round(_collect_time, 3),
                "update_time_s": round(_update_time, 3),
                "ort_combat_time_s": round(_iter_ort_combat_time, 3),
                "ort_combat_calls": _iter_ort_combat_calls,
                "nc_forward_time_s": round(_iter_nc_forward_time, 3),
                "nc_forward_calls": _iter_nc_forward_calls,
                "slow_steps": iter_slow_steps,
                "max_step_ms": round(iter_max_step_ms, 1),
                "timeout_count": iter_timeout_count,
                "boss_reach_rate": round(boss_reach_rate, 4),
                "act1_clear_rate": round(act1_clear_rate, 4),
                "boss_hp_fraction_dealt_mean": round(boss_hp_fraction_mean, 4),
                "deck_size_at_boss_mean": round(deck_size_at_boss_mean, 2),
                "card_reward_skip_rate": round(card_reward_skip_rate, 4),
            }
            metrics_history.append(entry)
            if metrics_log is not None:
                with open(metrics_log, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            logger.info(
                "Iter %3d | floor %.1f | vic %d/%d | boss %.0f%% act1 %.0f%% boss_hp %.2f deck@boss %.1f skip %.0f%% | ppo %d combat %d cppo %d | "
                "ppo_pl %.4f ppo_vl %.4f ppo_ent %.3f boss_r %.4f | search_pl %.4f search_vl %.4f | "
                "cppo_pl %.4f cppo_vl %.4f cbt_ent %.3f | %.0fs",
                iteration, avg_floor, victories, args.episodes_per_iter,
                boss_reach_rate * 100.0, act1_clear_rate * 100.0,
                boss_hp_fraction_mean, deck_size_at_boss_mean, card_reward_skip_rate * 100.0,
                ppo_steps, mcts_decisions, combat_ppo_steps,
                entry["ppo_ploss"], entry["ppo_vloss"], entry.get("ppo_entropy", 0),
                entry["boss_readiness_loss"],
                entry["mcts_ploss"], entry["mcts_vloss"],
                entry["combat_ppo_ploss"], entry["combat_ppo_vloss"],
                entry["combat_entropy"],
                iter_time,
            )

            # Health check
            # Ramp up learned card evaluator blend (every 100 iter)
            if effective_counterfactual_scoring and iteration > 0 and iteration % 100 == 0:
                from counterfactual_scoring import set_learned_evaluator
                _alpha = min(0.7, 0.1 + iteration * 0.3 / 1000)  # 0→0.7 over 2000 iter
                set_learned_evaluator(
                    ppo_net, vocab, alpha=_alpha,
                    skada_priors=_skada_priors_obj,
                    skada_gamma=args.skada_prior_weight if _skada_priors_obj is not None else 0.0,
                )

            if iteration > 0 and iteration % health_check_interval == 0:
                try:
                    alerts = health_monitor.check_all(metrics_history)
                    for alert in alerts:
                        logger.warning("HEALTH: %s", alert)
                except Exception:
                    pass

            # Periodic diagnostic dump
            if iteration > 0 and iteration % 50 == 0:
                try:
                    recent = metrics_history[-50:] if len(metrics_history) >= 50 else metrics_history
                    floors = [e.get("avg_floor", 0) for e in recent]
                    search_plosses = [
                        e.get("combat_search_ploss", e.get("mcts_ploss", 0))
                        for e in recent
                        if e.get("combat_search_ploss", e.get("mcts_ploss", 0)) > 0
                    ]
                    total_slow = sum(e.get("slow_steps", 0) for e in recent)
                    total_to = sum(e.get("timeout_count", 0) for e in recent)
                    ppo_buf_sizes = [e.get("ppo_steps", 0) for e in recent]
                    logger.info("=== DIAGNOSTIC iter %d ===", iteration)
                    logger.info("  Floor trend (last %d): avg=%.1f min=%.0f max=%.0f",
                                len(recent), np.mean(floors) if floors else 0,
                                min(floors) if floors else 0, max(floors) if floors else 0)
                    if search_plosses and len(search_plosses) >= 2:
                        logger.info("  Combat search ploss trend: %.3f -> %.3f",
                                    search_plosses[0], search_plosses[-1])
                    cppo_plosses = [e.get("combat_ppo_ploss", 0) for e in recent if e.get("combat_ppo_ploss", 0) > 0]
                    cppo_entropies = [e.get("combat_entropy", 0) for e in recent if e.get("combat_entropy", 0) > 0]
                    if cppo_plosses and len(cppo_plosses) >= 2:
                        logger.info("  Combat PPO ploss trend: %.3f -> %.3f",
                                    cppo_plosses[0], cppo_plosses[-1])
                    if cppo_entropies and len(cppo_entropies) >= 2:
                        logger.info("  Combat entropy trend: %.3f -> %.3f",
                                    cppo_entropies[0], cppo_entropies[-1])
                    logger.info("  Slow steps total: %d, timeouts: %d", total_slow, total_to)
                    logger.info("  PPO buffer avg size: %d",
                                int(np.mean(ppo_buf_sizes)) if ppo_buf_sizes else 0)
                    cppo_buf_sizes = [e.get("combat_ppo_steps", 0) for e in recent]
                    logger.info("  Combat PPO buffer avg size: %d",
                                int(np.mean(cppo_buf_sizes)) if cppo_buf_sizes else 0)

                    # --- Feature activity monitor ---
                    # Check if any feature groups are all-zero (dead features = schema bug)
                    if len(ppo_buffer) > 0:
                        dead_features = []
                        sample_ss = ppo_buffer.states[0]
                        for key in ["deck_mask", "relic_mask", "map_node_mask",
                                     "hand_mask", "enemy_mask"]:
                            vals = [getattr(s, key, None) for s in ppo_buffer.states[-20:]]
                            vals = [v for v in vals if v is not None]
                            if vals and not any(v.any() for v in vals):
                                dead_features.append(key)
                        if dead_features:
                            logger.warning("  DEAD FEATURES (all-zero): %s — schema bug?",
                                           ", ".join(dead_features))
                        else:
                            logger.info("  Feature activity: all feature groups active")

                    # --- Reward distribution monitor ---
                    if len(ppo_buffer) > 4:
                        rewards = np.array(ppo_buffer.rewards[-50:])
                        r_mean, r_std = rewards.mean(), rewards.std()
                        logger.info("  PPO reward: mean=%.4f std=%.4f min=%.4f max=%.4f",
                                    r_mean, r_std, rewards.min(), rewards.max())
                        if r_std < 1e-6:
                            logger.warning("  PPO REWARD FLAT (std=0) — reward shaping broken?")

                    if combat_buffer is not None and len(combat_buffer) > 4:
                        c_rewards = np.array(combat_buffer.rewards[-50:])
                        logger.info("  Combat reward: mean=%.4f std=%.4f min=%.4f max=%.4f",
                                    c_rewards.mean(), c_rewards.std(),
                                    c_rewards.min(), c_rewards.max())

                    # --- Advantage monitor ---
                    if len(ppo_buffer) > 4 and ppo_buffer.advantages:
                        advs = np.array(ppo_buffer.advantages[-50:])
                        logger.info("  PPO advantages: mean=%.4f std=%.4f",
                                    advs.mean(), advs.std())
                        if advs.std() < 1e-6:
                            logger.warning("  PPO ADVANTAGES FLAT — value function not learning?")

                except Exception:
                    pass

            # Save
            if iteration % args.save_interval == 0:
                torch.save({
                    "ppo_model": ppo_net.state_dict(),
                    "mcts_model": mcts_net.state_dict(),
                    "iteration": iteration,
                    "ppo_config": {"embed_dim": args.embed_dim},
                    "mcts_config": {"embed_dim": args.embed_dim, "hidden_dim": args.combat_hidden_dim},
                }, output_dir / f"hybrid_{iteration:05d}.pt")

    except Exception as e:
        logger.error("Crash: %s\n%s", e, traceback.format_exc())
        torch.save({
            "ppo_model": ppo_net.state_dict(),
            "mcts_model": mcts_net.state_dict(),
            "crash": str(e),
        }, output_dir / "hybrid_crash.pt")
        raise
    finally:
        if inf_server is not None:
            inf_server.stop()
        for client in env_clients.values():
            try:
                if hasattr(client, '_pipe') and client._pipe is not None:
                    client._pipe.call("delete_state", {"clear_all": True})
                elif hasattr(client, 'act'):
                    pass  # HTTP client — no cleanup needed
            except Exception:
                pass
            try:
                if hasattr(client, 'close'):
                    client.close()
            except Exception:
                pass
        _cleanup_spawned_envs()

    # Final save — use the SAME config keys as the periodic save above so the
    # downstream loader doesn't trip the "config disagrees with weights"
    # warning. The previous version hardcoded embed_dim=32 / hidden_dim=128
    # (legacy defaults from a much earlier training era) which is now wrong
    # because the current defaults are embed_dim=48 / hidden_dim=192.
    torch.save({
        "ppo_model": ppo_net.state_dict(),
        "mcts_model": mcts_net.state_dict(),
        "iteration": end_iter - 1,
        "ppo_config": {"embed_dim": args.embed_dim},
        "mcts_config": {"embed_dim": args.embed_dim, "hidden_dim": args.combat_hidden_dim},
    }, output_dir / "hybrid_final.pt")

    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
