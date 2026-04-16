"""Combat Memory Encoder: 表示"这场战斗到现在的长程态势"。

输入: combat_memory_bank tokens
输出: combat_memory_tokens (B, L_cm, d_model)
"""

from __future__ import annotations

import torch.nn as nn

from networkV2.s5_net.encoders.common import TransformerBlock
from networkV2.s5_net.tokenizer import BankTensor


class CombatMemoryEncoder(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, cm_bt: BankTensor) -> BankTensor:
        x = cm_bt.features
        mask = cm_bt.mask
        for layer in self.layers:
            x = layer(x, mask)
        return BankTensor(x, mask, cm_bt.bank_name)
