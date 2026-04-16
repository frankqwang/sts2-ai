"""Build Memory Encoder: 表示"这套牌怎么赢"。

输入: build_bank + inventory_bank tokens
输出: build_memory_slots (B, n_slots, d_model) — 固定长度的 slow memory

使用 Slot Attention 把变长的 deck/relic/potion/profile tokens
压缩为固定数量的 latent slots。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networkV2.s5_net.encoders.common import SlotAttention
from networkV2.s5_net.tokenizer import BankTensor


class BuildMemoryEncoder(nn.Module):
    def __init__(self, d_model: int = 384, n_slots: int = 8, n_iters: int = 3, n_heads: int = 4):
        super().__init__()
        self.slot_attn = SlotAttention(d_model, n_slots, n_iters, n_heads)

    def forward(
        self,
        build_bt: BankTensor,
        inventory_bt: BankTensor | None = None,
    ) -> torch.Tensor:
        """
        Returns: (B, n_slots, d_model)
        """
        # 拼接 build + inventory tokens
        features = build_bt.features
        mask = build_bt.mask

        if inventory_bt is not None and not inventory_bt.mask.sum() == 0:
            features = torch.cat([features, inventory_bt.features], dim=1)
            mask = torch.cat([mask, inventory_bt.mask], dim=1)

        return self.slot_attn(features, mask)
