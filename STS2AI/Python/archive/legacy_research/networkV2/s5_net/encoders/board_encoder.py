"""Board Encoder: 表示"当前战场是什么"。

输入: board_bank tokens (player, hand_cards, enemies, piles)
输出: board_tokens (B, L_board, d_model)

使用 self-attention，token 之间自然交互。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networkV2.s5_net.encoders.common import TransformerBlock
from networkV2.s5_net.tokenizer import BankTensor


class BoardEncoder(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        ffn_dim = d_model * 2
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, board_bt: BankTensor) -> BankTensor:
        x = board_bt.features
        mask = board_bt.mask
        for layer in self.layers:
            x = layer(x, mask)
        return BankTensor(x, mask, board_bt.bank_name)
