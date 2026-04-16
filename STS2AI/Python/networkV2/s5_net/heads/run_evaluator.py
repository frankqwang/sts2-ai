"""Run Evaluator: 非战斗长程价值评估。

判断的不是"这回合能不能活"，而是：
  - 当前 run 的整体胜面
  - 当前 build 对未来战斗的 readiness
  - 当前资源余量（HP/gold/potion）
  - 当前牌库质量趋势

输入: decision_repr + build/inventory/objective/forecast pool
输出: RunEvalOutputs
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from networkV2.s5_net.tokenizer import BankTensor


@dataclass
class RunEvalOutputs:
    run_win_prob: torch.Tensor        # (B,) 整局胜率 [0,1]
    boss_readiness: torch.Tensor      # (B,) boss 准备度 [0,1]
    resource_health: torch.Tensor     # (B,) 资源健康度 [0,1]
    deck_quality: torch.Tensor        # (B,) 牌库质量 [-1,1]


class RunEvaluator(nn.Module):
    def __init__(self, d_model: int = 384, hidden_dim: int = 256):
        super().__init__()
        # 4 个 shared bank pool 拼接
        self.context_proj = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim),
            nn.GELU(),
        )
        self.eval_proj = nn.Sequential(
            nn.Linear(d_model + hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.run_win_head = nn.Linear(hidden_dim, 1)
        self.boss_ready_head = nn.Linear(hidden_dim, 1)
        self.resource_head = nn.Linear(hidden_dim, 1)
        self.deck_quality_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        decision_repr: torch.Tensor,
        build_bt: BankTensor,
        inventory_bt: BankTensor,
        objective_bt: BankTensor,
        forecast_bt: BankTensor,
    ) -> RunEvalOutputs:
        build_pool = self._masked_mean(build_bt)
        inv_pool = self._masked_mean(inventory_bt)
        obj_pool = self._masked_mean(objective_bt)
        fcast_pool = self._masked_mean(forecast_bt)

        ctx = self.context_proj(torch.cat([build_pool, inv_pool, obj_pool, fcast_pool], dim=-1))
        h = self.eval_proj(torch.cat([decision_repr, ctx], dim=-1))

        return RunEvalOutputs(
            run_win_prob=torch.sigmoid(self.run_win_head(h).squeeze(-1)),
            boss_readiness=torch.sigmoid(self.boss_ready_head(h).squeeze(-1)),
            resource_health=torch.sigmoid(self.resource_head(h).squeeze(-1)),
            deck_quality=torch.tanh(self.deck_quality_head(h).squeeze(-1)),
        )

    @staticmethod
    def _masked_mean(bt: BankTensor) -> torch.Tensor:
        mask = bt.mask.unsqueeze(-1).float()
        summed = (bt.features * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count
