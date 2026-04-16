"""Modifier Encoder: 表示"当前规则被怎么改了"。

输入: modifier_bank tokens
输出: modifier_tokens (B, L_mod, d_model)
"""

from __future__ import annotations

import torch.nn as nn

from networkV2.s5_net.encoders.common import TransformerBlock
from networkV2.s5_net.tokenizer import BankTensor


class ModifierEncoder(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, mod_bt: BankTensor) -> BankTensor:
        x = mod_bt.features
        mask = mod_bt.mask
        for layer in self.layers:
            x = layer(x, mask)
        return BankTensor(x, mask, mod_bt.bank_name)
