"""Run Evaluator: 非战斗长程价值评估(7 head,全部可监督)。

设计原则:每个 head 对应一个**独立可监督**的真值信号源(来自 skada 轨迹或启发式)。
多 head 监督共享 encoder,让 RunEvaluator 成为"通用构筑理解模块"。

Heads:
  1. run_win_prob         sigmoid [0,1]  — 整局胜率(skada is_victory 真值)
  2. boss_readiness       sigmoid [0,1]  — boss 准备度(skada boss 战真实胜负 × hp)
  3. resource_health      sigmoid [0,1]  — 资源余量(hp/gold/potion 组合真值)
  4. deck_quality         tanh    [-1,1] — 牌库质量(skada cards.win_rate_delta 聚合)
  5. expected_hp_loss     softplus [0,∞) — 未来每场平均掉血(skada combat_stats.dmg_taken)
  6. expected_dmg_output  softplus [0,∞) — 未来每场平均输出(skada combat_stats.dmg_dealt)
  7. floor_clear_prob     sigmoid [0,1]  — 本层到下个 checkpoint 前通关概率

输入: decision_repr + 4 个 shared bank 的 masked_mean
输出: RunEvalOutputs(7 attr)

向后兼容:
  - RunEvalOutputs 原 4 个字段保留;新字段默认 None(老 loss 不强制要求)
  - 老 checkpoint load 时新 head 的参数随机初始化(load_compatible_params 已支持)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from networkV2.s5_net.tokenizer import BankTensor


@dataclass
class RunEvalOutputs:
    run_win_prob: torch.Tensor         # (B,) 整局胜率 [0,1]
    boss_readiness: torch.Tensor       # (B,) boss 准备度 [0,1]
    resource_health: torch.Tensor      # (B,) 资源健康度 [0,1]
    deck_quality: torch.Tensor         # (B,) 牌库质量 [-1,1]
    # 新增 3 个 head
    expected_hp_loss: torch.Tensor     # (B,) 未来每战掉血期望 [0,+inf)
    expected_dmg_output: torch.Tensor  # (B,) 未来每战输出期望 [0,+inf)
    floor_clear_prob: torch.Tensor     # (B,) 本层通关概率 [0,1]


class RunEvaluator(nn.Module):
    def __init__(self, d_model: int = 384, hidden_dim: int = 256):
        super().__init__()
        # 4 个 shared bank pool 拼接 → context
        self.context_proj = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim),
            nn.GELU(),
        )
        self.eval_proj = nn.Sequential(
            nn.Linear(d_model + hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # 7 个 head(每个只 hidden_dim → 1,参数量可忽略)
        self.run_win_head = nn.Linear(hidden_dim, 1)
        self.boss_ready_head = nn.Linear(hidden_dim, 1)
        self.resource_head = nn.Linear(hidden_dim, 1)
        self.deck_quality_head = nn.Linear(hidden_dim, 1)
        # 新增
        self.hp_loss_head = nn.Linear(hidden_dim, 1)
        self.dmg_output_head = nn.Linear(hidden_dim, 1)
        self.floor_clear_head = nn.Linear(hidden_dim, 1)

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
            expected_hp_loss=F.softplus(self.hp_loss_head(h).squeeze(-1)),
            expected_dmg_output=F.softplus(self.dmg_output_head(h).squeeze(-1)),
            floor_clear_prob=torch.sigmoid(self.floor_clear_head(h).squeeze(-1)),
        )

    @staticmethod
    def _masked_mean(bt: BankTensor) -> torch.Tensor:
        mask = bt.mask.unsqueeze(-1).float()
        summed = (bt.features * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count
