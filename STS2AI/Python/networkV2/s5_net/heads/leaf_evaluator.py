"""Leaf Evaluator: 搜索用的独立叶子节点评估器。

输入: decision_repr + board + combat_memory + mechanism + modifier
输出: LeafOutputs (leaf_score, transition_risk, survival_margin, resource_retention)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from networkV2.s5_net.tokenizer import BankTensor


@dataclass
class LeafOutputs:
    leaf_score: torch.Tensor
    transition_risk: torch.Tensor
    survival_margin: torch.Tensor
    resource_retention: torch.Tensor


class LeafEvaluator(nn.Module):
    def __init__(self, d_model: int = 384, hidden_dim: int = 256):
        super().__init__()
        # 4 个 bank 的 pool 拼接: board + combat_memory + mechanism + modifier
        self.context_proj = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim),
            nn.GELU(),
        )
        self.eval_proj = nn.Sequential(
            nn.Linear(d_model + hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.leaf_score_head = nn.Linear(hidden_dim, 1)
        self.transition_risk_head = nn.Linear(hidden_dim, 1)
        self.survival_head = nn.Linear(hidden_dim, 1)
        self.resource_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        decision_repr: torch.Tensor,
        board_bt: BankTensor,
        combat_memory_bt: BankTensor,
        mechanism_bt: BankTensor,
        modifier_bt: BankTensor,
    ) -> LeafOutputs:
        board_pool = self._masked_mean(board_bt)
        cm_pool = self._masked_mean(combat_memory_bt)
        mech_pool = self._masked_mean(mechanism_bt)
        mod_pool = self._masked_mean(modifier_bt)

        ctx = self.context_proj(torch.cat([board_pool, cm_pool, mech_pool, mod_pool], dim=-1))
        h = self.eval_proj(torch.cat([decision_repr, ctx], dim=-1))

        return LeafOutputs(
            leaf_score=torch.tanh(self.leaf_score_head(h).squeeze(-1)),
            transition_risk=torch.sigmoid(self.transition_risk_head(h).squeeze(-1)),
            survival_margin=torch.sigmoid(self.survival_head(h).squeeze(-1)),
            resource_retention=torch.sigmoid(self.resource_head(h).squeeze(-1)),
        )

    @staticmethod
    def _masked_mean(bt: BankTensor) -> torch.Tensor:
        mask = bt.mask.unsqueeze(-1).float()
        summed = (bt.features * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count
