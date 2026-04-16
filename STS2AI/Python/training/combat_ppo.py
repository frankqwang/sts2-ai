"""战斗 PPO 训练器：CombatRolloutBuffer + CombatPPOTrainer + mcts_train_step。"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.combat_network import CombatPolicyValueNetwork

logger = logging.getLogger(__name__)


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
    sample_weights: list[float] = field(default_factory=list)
    hard_state_tags: list[list[str]] = field(default_factory=list)

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
        sample_weight: float = 1.0,
        hard_state_tags: list[str] | None = None,
    ) -> None:
        self.state_features.append(sf)
        self.action_features.append(af)
        self.action_indices.append(action_idx)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.screen_types.append(screen_type)
        self.sample_weights.append(float(sample_weight))
        self.hard_state_tags.append(list(hard_state_tags or []))

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
            "sample_weights": torch.tensor(self.sample_weights, dtype=torch.float32),
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
                      "advantages", "returns", "screen_types",
                      "sample_weights", "hard_state_tags"):
            getattr(self, attr).clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Combat PPO Trainer
# ---------------------------------------------------------------------------


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
        target_kl: float = 0.0,
    ):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl

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
        sample_weights = data["sample_weights"]

        # Normalize advantages
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        n = len(old_actions)
        total_ploss = 0.0
        total_vloss = 0.0
        total_entropy = 0.0
        total_ratio_mean = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        num_updates = 0
        early_stop = False

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
                mb_sample_weights = sample_weights[mb_idx]
                mb_sample_weights = mb_sample_weights / mb_sample_weights.mean().clamp_min(1e-6)

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
                policy_loss = -(torch.min(surr1, surr2) * mb_sample_weights).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                # Value loss (clamp returns to [-1, 1] to match Tanh output)
                mb_returns_clamped = mb_returns.clamp(-1.0, 1.0)
                value_loss = F.mse_loss(values, mb_returns_clamped, reduction="none")
                value_loss = (value_loss * mb_sample_weights).mean()

                # Combined loss
                entropy = (dist.entropy() * mb_sample_weights).mean()
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                total_entropy += entropy.item()
                total_ratio_mean += ratio.mean().item()
                total_clip_fraction += clip_fraction.item()
                approx_kl = (mb_old_log_probs - new_log_probs).mean().abs()
                total_approx_kl += approx_kl.item()
                num_updates += 1

                if self.target_kl > 0 and approx_kl.item() > self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        num_updates = max(num_updates, 1)
        return {
            "combat_ppo_ploss": total_ploss / num_updates,
            "combat_ppo_vloss": total_vloss / num_updates,
            "combat_entropy": total_entropy / num_updates,
            "combat_ppo_ratio_mean": total_ratio_mean / num_updates,
            "combat_ppo_clip_fraction": total_clip_fraction / num_updates,
            "combat_ppo_approx_kl": total_approx_kl / num_updates,
            "combat_ppo_early_stop": float(early_stop),
        }


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

