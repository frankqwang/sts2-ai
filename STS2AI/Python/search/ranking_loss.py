#!/usr/bin/env python3
"""Ranking losses for training from offline matchup/counterfactual data.

Two loss functions:
  - listwise_ranking_loss: KL divergence between softmax distributions
  - pairwise_ranking_loss: Bradley-Terry margin loss over all ordered pairs

Both take predicted scores vs target scores with a validity mask.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def listwise_ranking_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence between softmax of predicted and target scores.

    Args:
        predicted: (B, K) raw scores from NN head
        target: (B, K) ground-truth scores from offline evaluation
        mask: (B, K) bool — valid options (True = valid)
        temperature: softmax temperature (lower = sharper)

    Returns:
        Scalar loss.
    """
    # Mask invalid options with large negative
    neg_inf = torch.tensor(-1e9, device=predicted.device, dtype=predicted.dtype)
    pred_masked = torch.where(mask, predicted / temperature, neg_inf)
    tgt_masked = torch.where(mask, target / temperature, neg_inf)

    pred_log_probs = F.log_softmax(pred_masked, dim=-1)
    tgt_probs = F.softmax(tgt_masked, dim=-1)

    # Zero out invalid positions for clean KL
    pred_log_probs = pred_log_probs * mask.float()
    tgt_probs = tgt_probs * mask.float()

    # Re-normalize target after masking (safety)
    tgt_sum = tgt_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    tgt_probs = tgt_probs / tgt_sum

    return F.kl_div(pred_log_probs, tgt_probs, reduction="batchmean")


def pairwise_ranking_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Bradley-Terry style pairwise ranking loss.

    For each pair (i, j) where target[i] > target[j], maximizes
    P(i > j) = sigmoid(predicted[i] - predicted[j] - margin).

    Args:
        predicted: (B, K) raw scores from NN head
        target: (B, K) ground-truth scores from offline evaluation
        mask: (B, K) bool — valid options
        margin: minimum score difference to enforce

    Returns:
        Scalar loss.
    """
    B, K = predicted.shape
    device = predicted.device

    if K < 2:
        return torch.tensor(0.0, device=device)

    total_loss = torch.tensor(0.0, device=device)
    pair_count = 0

    # Enumerate all pairs — K is small (3-4), so this is fine
    for i in range(K):
        for j in range(i + 1, K):
            # Both must be valid
            both_valid = mask[:, i] & mask[:, j]  # (B,)
            if not both_valid.any():
                continue

            # Target ordering: higher target score should have higher predicted score
            t_diff = target[:, i] - target[:, j]  # (B,)
            p_diff = predicted[:, i] - predicted[:, j]  # (B,)

            # Sign: if t_diff > 0, we want p_diff > margin
            #        if t_diff < 0, we want p_diff < -margin
            # Use: -log(sigmoid(sign * (p_diff - sign * margin)))
            sign = t_diff.sign()
            # Skip tied pairs
            non_tied = (t_diff.abs() > 1e-6) & both_valid
            if not non_tied.any():
                continue

            # Hinge-like: -log sigmoid(sign * p_diff)
            logit = sign[non_tied] * (p_diff[non_tied] - sign[non_tied] * margin)
            loss = F.binary_cross_entropy_with_logits(
                logit,
                torch.ones_like(logit),
                reduction="mean",
            )
            total_loss = total_loss + loss
            pair_count += 1

    return total_loss / max(pair_count, 1)
