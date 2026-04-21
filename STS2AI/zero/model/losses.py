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
    ranking: torch.Tensor
    delta: torch.Tensor
    uncertainty: torch.Tensor
    total: torch.Tensor


def compute_losses(output: ZeroNetOutput, batch, weights: LossWeights) -> LossBreakdown:
    """组装单个 batch 的多头训练目标。

    这里应当作为“最终生效 loss 组合”的唯一入口。
    如果后续调整任一分支语义，要在同一次改动里同步更新下面辅助函数的注释，
    保证训练意图能直接从代码里读出来。
    """
    policy_loss = _compute_policy_loss(output, batch, weights)
    value_loss = _compute_value_loss(output, batch)
    ranking_loss = _compute_ranking_loss(output, batch)
    delta_loss = F.mse_loss(output.delta_pred, batch.delta_targets)
    uncertainty_loss = F.binary_cross_entropy_with_logits(
        output.uncertainty,
        batch.uncertainty_target.clamp(0.0, 1.0),
    )
    total = (
        weights.policy * policy_loss
        + weights.value * value_loss
        + weights.ranking * ranking_loss
        + weights.delta * delta_loss
        + weights.uncertainty * uncertainty_loss
    )
    return LossBreakdown(
        policy=policy_loss,
        value=value_loss,
        ranking=ranking_loss,
        delta=delta_loss,
        uncertainty=uncertainty_loss,
        total=total,
    )


def _compute_policy_loss(output: ZeroNetOutput, batch, weights: LossWeights) -> torch.Tensor:
    """按 search / non-search 显式分流策略监督。

    当前约定：
    - search 样本：只通过 KL 拟合归一化后的 search policy
    - non-search 样本：只通过加权 CE 拟合行为动作
    - 低质量 rollout：额外乘上 `policy_bad_rollout_ce_scale`，避免“拖回合坏行为”
      长时间继续以行为克隆的方式主导训练。

    这是刻意做的监督分流；如果后续调整路由、正则或权重，
    必须同步更新这里的注释。
    """
    search_mask = batch.search_policy_mask > 0
    losses = []
    if search_mask.any():
        # search 样本只跟 search policy 对齐；先把非法动作 mask 掉，
        # 避免目标分布把概率落到不可执行动作上。
        safe_logits = _masked_policy_logits(output.policy_logits[search_mask], batch.action_mask[search_mask])
        valid_mask = batch.action_mask[search_mask] > 0
        search_log_probs = F.log_softmax(safe_logits, dim=-1)
        search_policy = batch.search_policy[search_mask] * valid_mask.to(batch.search_policy.dtype)
        search_policy = search_policy / search_policy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        if weights.policy_search_kl_weight > 0.0:
            losses.append(weights.policy_search_kl_weight * F.kl_div(search_log_probs, search_policy, reduction="batchmean"))

    non_search_mask = ~search_mask
    if non_search_mask.any() and weights.policy_behavior_ce_weight > 0.0:
        # 在线 / 未打 search 标签的样本继续走行为克隆；
        # sample_weight 决定这些样本在混池训练中的实际影响力。
        imitation_loss = F.cross_entropy(
            _masked_policy_logits(output.policy_logits[non_search_mask], batch.action_mask[non_search_mask]),
            batch.behavior_action_index[non_search_mask],
            reduction="none",
        )
        ce_scale = batch.behavior_ce_scale[non_search_mask]
        bad_rollout_mask = batch.fight_quality_score[non_search_mask] < 0.55
        if bad_rollout_mask.any():
            ce_scale = torch.where(
                bad_rollout_mask,
                ce_scale * weights.policy_bad_rollout_ce_scale,
                ce_scale,
            )
        if torch.any(ce_scale > 0.0):
            weighted = (imitation_loss * batch.sample_weight[non_search_mask] * ce_scale).mean()
            losses.append(weights.policy_behavior_ce_weight * weighted)
    if not losses:
        return torch.zeros((), device=output.policy_logits.device)
    return sum(losses)


def _compute_value_loss(output: ZeroNetOutput, batch) -> torch.Tensor:
    """把二值胜负监督和连续结果分数混合成 value loss。"""
    fight_win = F.binary_cross_entropy_with_logits(output.fight_win, batch.fight_targets[:, 0])
    enemy = F.mse_loss(output.enemy_hp_fraction_dealt, batch.fight_targets[:, 1])
    hp = F.mse_loss(output.self_hp_fraction_remaining, batch.fight_targets[:, 2])
    fraction_loss = 0.5 * (enemy + hp)
    return 0.5 * fight_win + 0.5 * fraction_loss


def _compute_ranking_loss(output: ZeroNetOutput, batch) -> torch.Tensor:
    """要求 search 最优动作的分数至少压过平均合法备选动作。"""
    mask = batch.search_best_action_index >= 0
    if not mask.any():
        return torch.zeros((), device=output.policy_logits.device)
    logits = _masked_policy_logits(output.policy_logits[mask], batch.action_mask[mask])
    if logits.size(1) <= 1:
        return torch.zeros((), device=output.policy_logits.device)
    best_idx = batch.search_best_action_index[mask]
    best = logits.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    exclude_mask = F.one_hot(best_idx, num_classes=logits.size(1)).bool() | ~(batch.action_mask[mask] > 0)
    valid_other_counts = (~exclude_mask).sum(dim=1).clamp_min(1)
    mean_other = logits.masked_fill(exclude_mask, 0.0).sum(dim=1) / valid_other_counts
    margin = batch.search_ranking_margin[mask].clamp_min(0.05)
    return torch.relu(margin - (best - mean_other)).mean()


def _masked_policy_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """在 softmax / CE 前稳定地屏蔽非法动作。"""
    fill_value = -float(torch.finfo(logits.dtype).max)
    return logits.masked_fill(action_mask <= 0, fill_value)
