from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import LossWeights
from .network import ZeroNetOutput


@dataclass(slots=True)
class LossBreakdown:
    policy: torch.Tensor
    value: torch.Tensor
    delta: torch.Tensor
    uncertainty: torch.Tensor
    total: torch.Tensor


def compute_losses(output: ZeroNetOutput, batch, weights: LossWeights) -> LossBreakdown:
    """组装单个 batch 的多头训练目标。

    policy-only 路线下，我们直接让模型对行为动作做加权 CE，
    不再引入搜索 KL 等外部监督信号。
    """
    policy_loss = _compute_policy_loss(output, batch, weights)
    value_loss = _compute_value_loss(output, batch)
    delta_loss = F.mse_loss(_select_delta_targets(output.delta_pred, batch.behavior_action_index), batch.delta_targets)
    uncertainty_loss = F.binary_cross_entropy_with_logits(
        output.uncertainty,
        batch.uncertainty_target.clamp(0.0, 1.0),
    )
    total = (
        weights.policy * policy_loss
        + weights.value * value_loss
        + weights.delta * delta_loss
        + weights.uncertainty * uncertainty_loss
    )
    return LossBreakdown(
        policy=policy_loss,
        value=value_loss,
        delta=delta_loss,
        uncertainty=uncertainty_loss,
        total=total,
    )


def _compute_policy_loss(output: ZeroNetOutput, batch, weights: LossWeights) -> torch.Tensor:
    """对行为动作做加权 cross-entropy 监督。"""
    if weights.policy_behavior_ce_weight <= 0.0:
        return torch.zeros((), device=output.policy_logits.device)
    imitation_loss = F.cross_entropy(
        _masked_policy_logits(output.policy_logits, batch.action_mask),
        batch.behavior_action_index,
        reduction="none",
    )
    ce_scale = batch.behavior_ce_scale
    if torch.any(ce_scale > 0.0):
        weighted = (imitation_loss * batch.sample_weight * ce_scale).mean()
        return weights.policy_behavior_ce_weight * weighted
    return torch.zeros((), device=output.policy_logits.device)


def _compute_value_loss(output: ZeroNetOutput, batch) -> torch.Tensor:
    """把二值胜负监督和连续结果分数混合成 value loss。"""
    fight_win = F.binary_cross_entropy_with_logits(output.fight_win, batch.fight_targets[:, 0])
    enemy = F.mse_loss(output.enemy_hp_fraction_dealt, batch.fight_targets[:, 1])
    hp = F.mse_loss(output.self_hp_fraction_remaining, batch.fight_targets[:, 2])
    fraction_loss = 0.5 * (enemy + hp)
    return 0.5 * fight_win + 0.5 * fraction_loss


def _masked_policy_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """在 softmax / CE 前稳定地屏蔽非法动作。"""
    fill_value = -float(torch.finfo(logits.dtype).max)
    return logits.masked_fill(action_mask <= 0, fill_value)


def _select_delta_targets(delta_pred: torch.Tensor, behavior_action_index: torch.Tensor) -> torch.Tensor:
    if delta_pred.dim() == 2:
        return delta_pred
    if delta_pred.dim() != 3:
        raise ValueError(f"delta_pred 维度不支持: shape={tuple(delta_pred.shape)}")
    gather_index = behavior_action_index.view(-1, 1, 1).expand(-1, 1, delta_pred.size(-1))
    return delta_pred.gather(1, gather_index).squeeze(1)
