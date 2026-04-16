"""Value Heads: 4 个独立的价值评估头，各自独立投影。

1. fight_win_value: 战斗赢面 (sigmoid → [0, 1])
2. expected_hp_loss: 期望掉血 (softplus → [0, +inf))
3. survival_2turn: 近 2 回合生存概率 (sigmoid → [0, 1])
4. tempo_value: 节奏优势 (tanh → [-1, 1])
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ValueOutputs:
    fight_win: torch.Tensor
    expected_hp_loss: torch.Tensor
    survival_2turn: torch.Tensor
    tempo: torch.Tensor


class _SingleValueHead(nn.Module):
    """单个 value head：独立 proj + 输出激活。"""
    def __init__(self, d_model: int, hidden_dim: int, activation: str = "sigmoid"):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x).squeeze(-1)
        if self.activation == "sigmoid":
            return torch.sigmoid(out)
        elif self.activation == "softplus":
            return F.softplus(out)
        elif self.activation == "tanh":
            return torch.tanh(out)
        return out


class ValueHeads(nn.Module):
    def __init__(self, d_model: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.fight_win = _SingleValueHead(d_model, hidden_dim, "sigmoid")
        self.hp_loss = _SingleValueHead(d_model, hidden_dim, "softplus")
        self.survival = _SingleValueHead(d_model, hidden_dim, "sigmoid")
        self.tempo = _SingleValueHead(d_model, hidden_dim, "tanh")

    def forward(self, decision_repr: torch.Tensor) -> ValueOutputs:
        return ValueOutputs(
            fight_win=self.fight_win(decision_repr),
            expected_hp_loss=self.hp_loss(decision_repr),
            survival_2turn=self.survival(decision_repr),
            tempo=self.tempo(decision_repr),
        )
