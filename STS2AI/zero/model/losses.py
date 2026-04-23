from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import LossWeights
from .network import (
    HierarchicalPolicyOutput,
    ZeroNetOutput,
    select_action_logits,
    select_action_value,
    select_confirm_logit,
    select_future_summary,
)


@dataclass(slots=True)
class LossBreakdown:
    policy: torch.Tensor
    policy_align: torch.Tensor
    value: torch.Tensor
    delta: torch.Tensor
    future_summary: torch.Tensor
    policy_entropy: torch.Tensor
    total: torch.Tensor


def compute_losses(output: ZeroNetOutput, batch, weights: LossWeights) -> LossBreakdown:
    """默认行为克隆 / ranking 路线的多头 loss。"""
    action_logits = select_action_logits(output, batch.active_intent if isinstance(output, HierarchicalPolicyOutput) else None)
    action_value = select_action_value(output, batch.active_intent if isinstance(output, HierarchicalPolicyOutput) else None)
    masked_logits = _masked_policy_logits(action_logits, batch.action_mask)
    policy_loss = _compute_policy_loss(masked_logits, batch, weights)
    policy_align_loss = compute_policy_value_alignment_loss(
        masked_logits,
        action_value,
        batch.action_mask,
        temperature=weights.policy_value_align_temperature,
    )
    value_loss = _compute_value_loss(output, action_value, batch)
    future_summary_loss = F.mse_loss(
        select_future_summary(output, batch.active_intent if isinstance(output, HierarchicalPolicyOutput) else None),
        batch.chosen_action_future_targets,
    )
    submenu_confirm_loss = _compute_submenu_confirm_loss(
        select_confirm_logit(output, batch.active_intent if isinstance(output, HierarchicalPolicyOutput) else None),
        batch,
    )
    total = (
        weights.policy * policy_loss
        + weights.policy_align * policy_align_loss
        + weights.value * value_loss
        + weights.future_summary * future_summary_loss
        + weights.submenu_confirm * submenu_confirm_loss
    )
    return LossBreakdown(
        policy=policy_loss + weights.submenu_confirm * submenu_confirm_loss,
        policy_align=policy_align_loss,
        value=value_loss,
        delta=torch.zeros((), device=masked_logits.device),
        future_summary=future_summary_loss,
        policy_entropy=torch.zeros((), device=masked_logits.device),
        total=total,
    )


def _compute_policy_loss(masked_logits: torch.Tensor, batch, weights: LossWeights) -> torch.Tensor:
    if weights.policy_behavior_ce_weight <= 0.0:
        return torch.zeros((), device=masked_logits.device)
    imitation_loss = F.cross_entropy(
        masked_logits,
        batch.behavior_action_index,
        reduction="none",
    )
    weighted = imitation_loss * batch.sample_weight * batch.behavior_ce_scale
    return weights.policy_behavior_ce_weight * weighted.mean()


def _compute_value_loss(output: ZeroNetOutput, action_value: torch.Tensor, batch) -> torch.Tensor:
    chosen_action_value = _gather_action_scalar(action_value, batch.behavior_action_index)
    state_value_loss = F.mse_loss(output.state_value, batch.ppo_return)
    action_value_loss = F.mse_loss(chosen_action_value, batch.ppo_return)
    if isinstance(output, HierarchicalPolicyOutput):
        intent_value_loss = F.mse_loss(output.intent_value, batch.turn_return)
        return state_value_loss + 0.5 * action_value_loss + 0.5 * intent_value_loss
    return state_value_loss + 0.5 * action_value_loss


def _compute_submenu_confirm_loss(confirm_logit: torch.Tensor, batch) -> torch.Tensor:
    target_mask = batch.submenu_has_confirm > 0.5
    if not bool(target_mask.any().item()):
        return torch.zeros((), device=confirm_logit.device)
    loss = F.binary_cross_entropy_with_logits(
        confirm_logit[target_mask],
        batch.submenu_confirm_target[target_mask],
        reduction="none",
    )
    return (loss * batch.sample_weight[target_mask].clamp_min(0.1)).mean()


def compute_policy_value_alignment_loss(
    policy_logits: torch.Tensor,
    action_value: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-3)
    masked_value = _masked_policy_logits(action_value, action_mask)
    target_probs = torch.softmax(masked_value / safe_temperature, dim=-1).detach()
    target_probs = target_probs * (action_mask > 0).to(target_probs.dtype)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


def _masked_policy_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    fill_value = -float(torch.finfo(logits.dtype).max)
    return logits.masked_fill(action_mask <= 0, fill_value)


def _gather_action_scalar(values: torch.Tensor, action_index: torch.Tensor) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"expected [batch, actions], got shape={tuple(values.shape)}")
    return values.gather(1, action_index.unsqueeze(1)).squeeze(1)
