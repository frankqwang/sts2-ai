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


def _valid_token_prefix(tokens: list[Token]) -> list[Token]:
    """返回连续前缀里的有效 token。

    约定：GraphRunner 的 padding 只会把 `token_type='pad'` 追加在尾部。
    这里取 leading valid prefix，避免每步为整段 padded tail 做无意义拷贝。
    """
    valid_len = 0
    for tok in tokens:
        if tok.token_type == "pad":
            break
        valid_len += 1
    return tokens[:valid_len]


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

    def __init__(self, d_model: int = 384, max_numeric_dim: int = 58):
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
        valid_tokens = _valid_token_prefix(tokens)
        n_valid = len(valid_tokens)

        # 构建 numeric tensor: (1, L, max_numeric_dim)
        numeric = torch.zeros(1, seq_len, self.max_numeric_dim, device=device)
        type_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        ts_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)

        if n_valid > 0:
            numeric_width = min(
                max((len(tok.numeric) for tok in valid_tokens), default=0),
                self.max_numeric_dim,
            )
            if numeric_width > 0:
                numeric_rows: list[list[float]] = []
                for tok in valid_tokens:
                    row = list(tok.numeric[:numeric_width])
                    if len(row) < numeric_width:
                        row.extend([0.0] * (numeric_width - len(row)))
                    numeric_rows.append(row)
                numeric[0, :n_valid, :numeric_width] = torch.tensor(
                    numeric_rows, dtype=torch.float32, device=device,
                )
            type_ids[0, :n_valid] = torch.tensor(
                [tok.type_idx for tok in valid_tokens],
                dtype=torch.long, device=device,
            )
            ts_ids[0, :n_valid] = torch.tensor(
                [tok.time_scale_idx for tok in valid_tokens],
                dtype=torch.long, device=device,
            )
            mask[0, :n_valid] = True

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

    # ------------------------------------------------------------------
    # Static buffer path (for CUDA graph support)
    # ------------------------------------------------------------------

    def fill_static_buffers(
        self,
        banks: UnifiedTokenBanks,
        host_buffers: dict[str, dict[str, torch.Tensor]],
        gpu_buffers: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """把 banks 的内容写进 pre-allocated host pinned buffers,然后 async copy 到 GPU。

        host_buffers/gpu_buffers: {bank_name: {numeric/type_ids/ts_ids/mask: tensor}}

        host buffer 是 pinned CPU tensor(pin_memory=True),支持 non_blocking DMA 到 GPU。
        这个 copy **必须在 CUDA graph capture 之外**调用(capture 内会炸)。
        replay 之前 fill_static_buffers → graph.replay() 自动用 GPU static buffer 里的新数据。
        """
        from networkV2.s5_net.bank_max_spec import BankOverflowError

        for bank in banks.all_banks():
            name = bank.bank_name
            if name not in host_buffers:
                # 未在 max_spec 声明的 bank — 跳过(不会进 graph 输入)
                continue
            h = host_buffers[name]
            g = gpu_buffers[name]
            max_len = h["numeric"].shape[1]
            if len(bank.tokens) > max_len:
                raise BankOverflowError(
                    f"bank '{name}' has {len(bank.tokens)} tokens > max_len {max_len}. "
                    f"调大 BankMaxSpec.{name} 或分析是否有异常膨胀。"
                )

            # Zero 整个 host buffer(避免上次残留)
            h["numeric"].zero_()
            h["type_ids"].zero_()
            h["ts_ids"].zero_()
            h["mask"].zero_()
            valid_tokens = _valid_token_prefix(bank.tokens)
            n_tokens = len(valid_tokens)
            if n_tokens > 0:
                numeric_width = min(
                    max((len(tok.numeric) for tok in valid_tokens), default=0),
                    self.max_numeric_dim,
                )
                if numeric_width > 0:
                    numeric_rows: list[list[float]] = []
                    for tok in valid_tokens:
                        row = list(tok.numeric[:numeric_width])
                        if len(row) < numeric_width:
                            row.extend([0.0] * (numeric_width - len(row)))
                        numeric_rows.append(row)
                    h["numeric"][0, :n_tokens, :numeric_width].copy_(
                        torch.tensor(numeric_rows, dtype=torch.float32),
                    )
                h["type_ids"][0, :n_tokens].copy_(
                    torch.tensor([tok.type_idx for tok in valid_tokens], dtype=torch.long),
                )
                h["ts_ids"][0, :n_tokens].copy_(
                    torch.tensor([tok.time_scale_idx for tok in valid_tokens], dtype=torch.long),
                )
                h["mask"][0, :n_tokens] = True

            # Async DMA to GPU static buffers(不阻塞 CPU,graph replay 前 sync)
            for key in ("numeric", "type_ids", "ts_ids", "mask"):
                g[key].copy_(h[key], non_blocking=True)

    def project_static(
        self,
        gpu_buffers: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, BankTensor]:
        """用 static GPU buffer 跑 embedding + projection + norm。

        返回 dict[bank_name, BankTensor] 结构与 `tokenize_banks()` 对齐,
        但所有 tensor 都在 **pre-alloc GPU buffer** 里,不新建。
        **这个方法 CUDA graph-capturable**。
        """
        result: dict[str, BankTensor] = {}
        for name, bufs in gpu_buffers.items():
            numeric = bufs["numeric"]
            ndim = numeric.shape[-1]
            if ndim < self.max_numeric_dim:
                pad = torch.zeros(
                    numeric.shape[0], numeric.shape[1], self.max_numeric_dim - ndim,
                    device=numeric.device, dtype=numeric.dtype,
                )
                numeric = torch.cat([numeric, pad], dim=-1)
            elif ndim > self.max_numeric_dim:
                numeric = numeric[:, :, : self.max_numeric_dim]

            h = self.numeric_proj(numeric)
            h = h + self.type_embed(bufs["type_ids"])
            h = h + self.time_scale_embed(bufs["ts_ids"])
            h = self.norm(h)
            result[name] = BankTensor(h, bufs["mask"], name)
        return result

    def fill_static_padded_buffers(
        self,
        batched_banks: dict[str, "PaddedBank"],
        host_buffers: dict[str, dict[str, torch.Tensor]],
        gpu_buffers: dict[str, dict[str, torch.Tensor]],
        *,
        batch_size: int | None = None,
    ) -> None:
        """把 batched PaddedBank 写进 pre-allocated static buffers。

        用于 rollout 集中 batch inference 的 CUDA graph bucket。
        host/gpu buffer 的 batch 维固定为 bucket 大小；若实际 batch 更小，尾部会被清零。
        """
        from networkV2.s5_net.bank_max_spec import BankOverflowError

        for name, h in host_buffers.items():
            g = gpu_buffers[name]
            h["numeric"].zero_()
            h["type_ids"].zero_()
            h["ts_ids"].zero_()
            h["mask"].zero_()

            pb = batched_banks.get(name)
            if pb is not None:
                src_batch = int(pb.numeric.shape[0])
                dst_batch = int(h["numeric"].shape[0])
                if batch_size is not None:
                    src_batch = min(src_batch, int(batch_size))
                if src_batch > dst_batch:
                    raise BankOverflowError(
                        f"bank '{name}' batch {src_batch} > static batch {dst_batch}. "
                        f"调大 graph bucket 或回退 eager。"
                    )
                if pb.numeric.shape[1] > h["numeric"].shape[1]:
                    raise BankOverflowError(
                        f"bank '{name}' len {pb.numeric.shape[1]} > static len {h['numeric'].shape[1]}."
                    )
                ndim = min(int(pb.numeric.shape[2]), int(h["numeric"].shape[2]))
                if src_batch > 0:
                    h["numeric"][:src_batch, : pb.numeric.shape[1], :ndim].copy_(
                        pb.numeric[:src_batch, :, :ndim],
                    )
                    h["type_ids"][:src_batch, : pb.type_ids.shape[1]].copy_(
                        pb.type_ids[:src_batch],
                    )
                    h["ts_ids"][:src_batch, : pb.ts_ids.shape[1]].copy_(
                        pb.ts_ids[:src_batch],
                    )
                    h["mask"][:src_batch, : pb.mask.shape[1]].copy_(
                        pb.mask[:src_batch],
                    )

            for key in ("numeric", "type_ids", "ts_ids", "mask"):
                g[key].copy_(h[key], non_blocking=True)


def alloc_static_bank_buffers(
    bank_names: list[str],
    max_spec,
    device: torch.device,
    max_numeric_dim: int = 58,
    batch: int = 1,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    """预分配 host(pinned) + GPU static buffer 对。

    Returns: (host_buffers, gpu_buffers) 两个 dict,键为 bank_name。
    """
    host: dict[str, dict[str, torch.Tensor]] = {}
    gpu: dict[str, dict[str, torch.Tensor]] = {}
    for name in bank_names:
        max_len = max_spec.get(name)
        # Host pinned
        host[name] = {
            "numeric": torch.zeros(batch, max_len, max_numeric_dim, dtype=torch.float32, pin_memory=True),
            "type_ids": torch.zeros(batch, max_len, dtype=torch.long, pin_memory=True),
            "ts_ids": torch.zeros(batch, max_len, dtype=torch.long, pin_memory=True),
            "mask": torch.zeros(batch, max_len, dtype=torch.bool, pin_memory=True),
        }
        # GPU
        gpu[name] = {
            "numeric": torch.zeros(batch, max_len, max_numeric_dim, device=device, dtype=torch.float32),
            "type_ids": torch.zeros(batch, max_len, device=device, dtype=torch.long),
            "ts_ids": torch.zeros(batch, max_len, device=device, dtype=torch.long),
            "mask": torch.zeros(batch, max_len, device=device, dtype=torch.bool),
        }
    return host, gpu
