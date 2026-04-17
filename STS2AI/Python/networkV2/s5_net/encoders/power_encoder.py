"""Power Encoder: 表示"战场上所有 active power 的状态"。

输入: power_bank tokens (每个 active enemy/player power 一个 token)
输出: power_tokens (B, L_power, d_model)

每个 power token 已经在 bank_assembler 里带了 owner identity snapshot
(hp_ratio / max_hp / block / is_hittable)，网络通过 self-attention 在 power
之间交互，也通过下游 cross-attention 把 power 关联到对应 enemy_core / player_token。

此 encoder 结构和 MechanismEncoder / ModifierEncoder 一致（TransformerBlock × N）。
"""

from __future__ import annotations

import torch.nn as nn

from networkV2.s5_net.encoders.common import TransformerBlock
from networkV2.s5_net.tokenizer import BankTensor


class PowerEncoder(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, power_bt: BankTensor) -> BankTensor:
        x = power_bt.features
        mask = power_bt.mask
        for layer in self.layers:
            x = layer(x, mask)
        return BankTensor(x, mask, power_bt.bank_name)
