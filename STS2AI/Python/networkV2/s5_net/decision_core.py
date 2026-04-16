"""Decision Core: 把所有动作放在一起比较。

输入: action_hypothesis_tokens + learnable decision_token
输出:
  - decision_token_refined: 全局决策摘要 → Value/LeafEvaluator
  - action_hypotheses_refined: 每个动作的最终表示 → Policy
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networkV2.s5_net.encoders.common import TransformerBlock
from networkV2.s5_net.tokenizer import BankTensor


class DecisionCore(nn.Module):
    """Decision Core: decision_token + action_hypotheses → self-attention reasoning。"""

    def __init__(self, d_model: int = 384, n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        # Learnable decision token
        self.decision_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self, action_hypothesis_bt: BankTensor,
    ) -> tuple[torch.Tensor, BankTensor]:
        """
        Args:
            action_hypothesis_bt: (B, L_action, d_model) + mask

        Returns:
            decision_repr: (B, d_model) — 全局决策摘要
            action_refined: BankTensor (B, L_action, d_model) — 动作最终表示
        """
        B = action_hypothesis_bt.features.size(0)
        ah = action_hypothesis_bt.features  # (B, L, d)
        ah_mask = action_hypothesis_bt.mask  # (B, L)

        # Prepend decision token
        dt = self.decision_token.expand(B, -1, -1)       # (B, 1, d)
        dt_mask = torch.ones(B, 1, dtype=torch.bool, device=ah.device)

        seq = torch.cat([dt, ah], dim=1)                  # (B, 1+L, d)
        mask = torch.cat([dt_mask, ah_mask], dim=1)        # (B, 1+L)

        # Self-attention reasoning
        for layer in self.layers:
            seq = layer(seq, mask)

        # 拆分
        decision_repr = seq[:, 0, :]                       # (B, d)
        action_refined = seq[:, 1:, :]                     # (B, L, d)

        return decision_repr, BankTensor(action_refined, ah_mask, "action_refined")
