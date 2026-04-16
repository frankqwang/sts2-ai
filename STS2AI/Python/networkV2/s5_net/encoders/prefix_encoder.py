"""Turn Prefix Encoder: 表示"本回合已经怎么打了"。

输入: turn_prefix_bank tokens (played_action sequence + turn_summary)
输出: prefix_tokens (B, L_prefix, d_model)

使用 causal transformer（有序序列）。
"""

from __future__ import annotations

import torch.nn as nn

from networkV2.s5_net.encoders.common import TransformerBlock
from networkV2.s5_net.tokenizer import BankTensor


class TurnPrefixEncoder(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, prefix_bt: BankTensor) -> BankTensor:
        x = prefix_bt.features
        mask = prefix_bt.mask
        for layer in self.layers:
            x = layer(x, mask, is_causal=True)
        return BankTensor(x, mask, prefix_bt.bank_name)
