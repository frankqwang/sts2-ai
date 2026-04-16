"""Encoder 共享组件：TransformerBlock, CrossAttentionBlock, SlotAttention。"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """标准 Pre-LN Transformer block (self-attention + FFN)。"""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            mask: (B, L) bool, True = valid token
            is_causal: 使用 causal mask（用于 turn prefix 等有序序列）
        """
        kpm = ~mask if mask is not None else None
        h = self.norm1(x)
        # causal mask: 显式构造上三角 mask
        attn_mask = None
        if is_causal:
            L = x.size(1)
            attn_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )  # True = ignore
        h, _ = self.attn(h, h, h, key_padding_mask=kpm, attn_mask=attn_mask)
        h = torch.nan_to_num(h, nan=0.0)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-LN Cross-Attention block: query attends to key/value。"""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Lq, d_model)
            kv: (B, Lkv, d_model)
            kv_mask: (B, Lkv) bool, True = valid
        """
        # 如果 kv 全部被 mask 掉（空 bank），跳过 attention 直接走 FFN
        if kv_mask is not None and not kv_mask.any():
            return query + self.ffn(self.norm2(query))
        kpm = ~kv_mask if kv_mask is not None else None
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        h, _ = self.attn(q, k, k, key_padding_mask=kpm)
        # 安全网：batch 内部分样本的 kv 可能全被 mask，产生 NaN
        h = torch.nan_to_num(h, nan=0.0)
        query = query + h
        query = query + self.ffn(self.norm2(query))
        return query


class SlotAttention(nn.Module):
    """Slot Attention: learnable latent slots attend to input tokens。

    用于 Build Memory Encoder：把 variable-length deck/relic/potion tokens
    压缩为固定数量的 latent slots。
    """

    def __init__(
        self, d_model: int, n_slots: int, n_iters: int = 3,
        n_heads: int = 4, ffn_dim: int = 0,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters

        # Learnable slot initialization
        self.slot_init = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)

        # Cross-attention blocks for iterative refinement
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, ffn_dim or d_model * 2)
            for _ in range(n_iters)
        ])

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, L, d_model)
            mask: (B, L) bool

        Returns:
            slots: (B, n_slots, d_model)
        """
        B = inputs.size(0)
        slots = self.slot_init.expand(B, -1, -1)

        for cross_block in self.cross_blocks:
            slots = cross_block(slots, inputs, kv_mask=mask)

        return slots
