"""Policy Head: 输出每个动作的 logit。

使用 bilinear scorer: score = action^T W decision + action^T b
即动作分数同时看自身语义和全局决策上下文。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networkV2.s5_net.tokenizer import BankTensor


class PolicyHead(nn.Module):
    def __init__(self, d_model: int = 384):
        super().__init__()
        self.bilinear = nn.Bilinear(d_model, d_model, 1)
        self.action_score = nn.Linear(d_model, 1)

    def forward(
        self,
        decision_repr: torch.Tensor,        # (B, d_model)
        action_refined: BankTensor,          # (B, L_action, d_model) + mask
    ) -> torch.Tensor:
        """
        Returns: logits (B, L_action)，mask 外的位置填 -inf
        """
        B, L, d = action_refined.features.shape
        mask = action_refined.mask  # (B, L)

        # 扩展 decision_repr 到每个 action
        dec = decision_repr.unsqueeze(1).expand(B, L, d)  # (B, L, d)

        # Bilinear + action-only score
        bilinear_score = self.bilinear(action_refined.features, dec).squeeze(-1)  # (B, L)
        action_score = self.action_score(action_refined.features).squeeze(-1)      # (B, L)
        logits = bilinear_score + action_score

        # Mask invalid actions
        logits = logits.masked_fill(~mask, float("-inf"))

        return logits
