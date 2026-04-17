"""Tokenizer: Token Bank → Tensor 转换。

将 Compiler 输出的 Python Token 对象转换为 PyTorch tensor，
加上 type/time_scale embedding，统一投影到 d_model 维。

每个 bank 独立 pad 到各自的 max_len，输出 (B, L, d_model) + mask。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networkV2.s1_schema.token_banks import (
    TokenBank, UnifiedTokenBanks, NUM_TOKEN_TYPES, Token,
)


class BankTensor:
    """一个 bank 张量化后的结果。"""
    __slots__ = ("features", "mask", "bank_name")

    def __init__(self, features: torch.Tensor, mask: torch.Tensor, bank_name: str):
        self.features = features   # (B, L, d_model)
        self.mask = mask           # (B, L) bool, True = valid
        self.bank_name = bank_name


class BankTokenizer(nn.Module):
    """将 TokenBank 转为 tensor 并投影到 d_model。

    每个 token:
      [numeric features] + type_embedding + time_scale_embedding
      → linear projection → d_model
    """

    def __init__(self, d_model: int = 384, max_numeric_dim: int = 48):
        super().__init__()
        self.d_model = d_model
        self.max_numeric_dim = max_numeric_dim

        # Embeddings
        self.type_embed = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.time_scale_embed = nn.Embedding(3, d_model)  # slow/medium/fast

        # 数值投影：把变长 numeric 先 pad 到 max_numeric_dim，再投影
        self.numeric_proj = nn.Linear(max_numeric_dim, d_model)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.type_embed.weight, std=0.02)
        nn.init.normal_(self.time_scale_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.numeric_proj.weight)
        nn.init.zeros_(self.numeric_proj.bias)

    def tokenize_bank(
        self, bank: TokenBank, batch_size: int = 1, device: torch.device | None = None,
    ) -> BankTensor:
        """将单个 TokenBank 转为 BankTensor。

        当前实现：batch_size=1（单样本），后续扩展 batched 版本。
        """
        if device is None:
            device = self.numeric_proj.weight.device

        tokens = bank.tokens
        seq_len = max(len(tokens), 1)  # 至少 1（避免空 bank）

        # 构建 numeric tensor: (1, L, max_numeric_dim)
        numeric = torch.zeros(1, seq_len, self.max_numeric_dim, device=device)
        type_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        ts_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)

        for i, tok in enumerate(tokens):
            n = min(len(tok.numeric), self.max_numeric_dim)
            numeric[0, i, :n] = torch.tensor(tok.numeric[:n], dtype=torch.float32)
            type_ids[0, i] = tok.type_idx
            ts_ids[0, i] = tok.time_scale_idx
            mask[0, i] = True

        # 投影: numeric → d_model, 加上 type + time_scale embedding
        h = self.numeric_proj(numeric)                    # (1, L, d_model)
        h = h + self.type_embed(type_ids)                 # + type embedding
        h = h + self.time_scale_embed(ts_ids)             # + time scale embedding
        h = self.norm(h)

        return BankTensor(features=h, mask=mask, bank_name=bank.bank_name)

    def tokenize_banks(
        self, banks: UnifiedTokenBanks, device: torch.device | None = None,
    ) -> dict[str, BankTensor]:
        """将 UnifiedTokenBanks 的所有 bank 转为 BankTensor dict。"""
        result: dict[str, BankTensor] = {}
        for bank in banks.all_banks():
            if not bank.is_empty:
                result[bank.bank_name] = self.tokenize_bank(bank, device=device)
        return result

    def tokenize_padded_bank(self, padded: "PaddedBank", device: torch.device | None = None) -> BankTensor:
        """将 PaddedBank (已 batched) 转为 BankTensor。

        Args:
            padded: 来自 training/batch.py 的 PaddedBank
        """
        from networkV2.s6_training.batch import PaddedBank
        if device is None:
            device = self.numeric_proj.weight.device

        # numeric pad 到 max_numeric_dim
        B, L, ndim = padded.numeric.shape
        if ndim < self.max_numeric_dim:
            pad = torch.zeros(B, L, self.max_numeric_dim - ndim, device=device)
            numeric = torch.cat([padded.numeric.to(device), pad], dim=-1)
        elif ndim > self.max_numeric_dim:
            numeric = padded.numeric[:, :, :self.max_numeric_dim].to(device)
        else:
            numeric = padded.numeric.to(device)

        type_ids = padded.type_ids.to(device)
        ts_ids = padded.ts_ids.to(device)
        mask = padded.mask.to(device)

        h = self.numeric_proj(numeric)
        h = h + self.type_embed(type_ids)
        h = h + self.time_scale_embed(ts_ids)
        h = self.norm(h)

        return BankTensor(features=h, mask=mask, bank_name=padded.bank_name)

    def tokenize_batched_banks(
        self, batched_banks: dict[str, "PaddedBank"], device: torch.device | None = None,
    ) -> dict[str, BankTensor]:
        """将 BatchedBanks.banks dict 转为 BankTensor dict。"""
        return {
            name: self.tokenize_padded_bank(pb, device=device)
            for name, pb in batched_banks.items()
        }
