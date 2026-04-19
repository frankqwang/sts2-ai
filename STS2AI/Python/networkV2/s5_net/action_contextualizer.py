"""Action Contextualizer: 动作上下文化。

支持三种模式（通过 NetworkConfig.contextualizer_mode 切换）:

  "full"   : 原设计 6 段独立 cross-attention
    - 并行 3 段: action → board, modifier, mechanism （增量 merge）
    - 串行 3 段: action → prefix, combat_memory, build
    - attention ops: 6

  "merged" : 并行 3→1 合并 + 串行 3→1 合并
    - 1 个 cross: action → concat(board, modifier, mechanism)（加 bank_type emb）
    - 1 个 cross: action → concat(prefix, combat_memory, build_slots)
    - attention ops: 2
    - 能力: 网络需要自己从 concat kv 里分辨 bank 类型（靠 bank_type_emb）

  "minimal": 全合并
    - 1 个 cross: action → concat(all 6 banks)
    - attention ops: 1
    - 能力: 最弱，仅用于调试/tiny preset

配置保留原因：
  训练速度 vs 能力的 trade-off。slim 版 2 ops 比 full 版 6 ops 快 3x，
  但需要 ~1.5-2x 样本达到同等水平。详见 network_config.py 顶部文档。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networkV2.s5_net.encoders.common import CrossAttentionBlock
from networkV2.s5_net.tokenizer import BankTensor


# bank_type_embedding 索引（用于 merged/minimal 模式区分来源）
_BANK_TYPE_BOARD = 0
_BANK_TYPE_MODIFIER = 1
_BANK_TYPE_MECHANISM = 2
_BANK_TYPE_POWER = 3            # v2: power_bank（每 active power 一个 token）
_BANK_TYPE_PREFIX = 4
_BANK_TYPE_COMBAT_MEMORY = 5
_BANK_TYPE_BUILD = 6
_N_BANK_TYPES = 7


class ActionContextualizer(nn.Module):

    def __init__(
        self, d_model: int = 384, n_heads: int = 8, dropout: float = 0.1,
        mode: str = "full",
    ):
        super().__init__()
        self.mode = mode
        ffn_dim = d_model * 2

        if mode == "full":
            # 原版 7 段 cross（v2：并行 4 段 board/mod/mech/power + 串行 3 段 prefix/cm/build）
            self.cross_board = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.cross_modifier = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.cross_mechanism = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.cross_power = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.merge_gate = nn.Sequential(
                nn.Linear(d_model * 4, d_model), nn.GELU(),
            )
            self.merge_norm = nn.LayerNorm(d_model)
            self.cross_prefix = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.cross_combat_memory = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.cross_build = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
        elif mode in ("merged", "minimal"):
            # 合并模式：用 bank_type_emb 标记 kv 来源，少数几次 cross
            self.bank_type_emb = nn.Embedding(_N_BANK_TYPES, d_model)
            # 预计算 bank_type id tensor 作为 buffer(而非每次 torch.tensor(int))
            # 避免 CPU→GPU copy,CUDA graph capture 要求。
            self.register_buffer(
                "_bank_type_ids",
                torch.arange(_N_BANK_TYPES, dtype=torch.long),
                persistent=False,
            )
            if mode == "merged":
                self.cross_fast = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)  # board+mod+mech
                self.cross_slow = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)  # prefix+cm+build
            else:  # minimal
                self.cross_all = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
        else:
            raise ValueError(f"Unknown contextualizer mode: {mode}")

    def _tag(self, bt: BankTensor, bank_type_id: int) -> BankTensor:
        """给 bank tensor 加上 bank_type_embedding，供合并模式区分来源。

        不用 torch.tensor(int, device=...) —— 会 CPU→GPU copy 破坏 CUDA graph。
        从预分配的 _bank_type_ids buffer(已经在 GPU 上)取对应 index。
        """
        type_emb = self.bank_type_emb(self._bank_type_ids[bank_type_id])
        return BankTensor(bt.features + type_emb, bt.mask, bt.bank_name)

    @staticmethod
    def _concat(bts: list[BankTensor]) -> tuple[torch.Tensor, torch.Tensor]:
        feats = torch.cat([b.features for b in bts], dim=1)
        masks = torch.cat([b.mask for b in bts], dim=1)
        return feats, masks

    def forward(
        self,
        action_bt: BankTensor,
        board_bt: BankTensor,
        modifier_bt: BankTensor,
        mechanism_bt: BankTensor,
        power_bt: BankTensor,
        prefix_bt: BankTensor,
        combat_memory_bt: BankTensor,
        build_slots: torch.Tensor,
    ) -> BankTensor:
        action = action_bt.features

        if self.mode == "full":
            # 并行 4 段（v2：加 power）+ 增量 merge
            out_board = self.cross_board(action, board_bt.features, board_bt.mask)
            out_mod = self.cross_modifier(action, modifier_bt.features, modifier_bt.mask)
            out_mech = self.cross_mechanism(action, mechanism_bt.features, mechanism_bt.mask)
            out_power = self.cross_power(action, power_bt.features, power_bt.mask)
            delta = torch.cat(
                [out_board - action, out_mod - action, out_mech - action, out_power - action],
                dim=-1,
            )
            merged = self.merge_gate(delta)
            action = self.merge_norm(action + merged)
            # 串行 3 段
            action = self.cross_prefix(action, prefix_bt.features, prefix_bt.mask)
            action = self.cross_combat_memory(action, combat_memory_bt.features, combat_memory_bt.mask)
            action = self.cross_build(action, build_slots, kv_mask=None)

        elif self.mode == "merged":
            # 把 build_slots 包成 BankTensor 方便统一处理
            build_bt = BankTensor(
                features=build_slots,
                mask=torch.ones(build_slots.shape[:2], dtype=torch.bool, device=build_slots.device),
                bank_name="build_slots",
            )
            # 并行 4 bank concat（v2：加 power）→ 1 次 cross
            fast_bts = [
                self._tag(board_bt, _BANK_TYPE_BOARD),
                self._tag(modifier_bt, _BANK_TYPE_MODIFIER),
                self._tag(mechanism_bt, _BANK_TYPE_MECHANISM),
                self._tag(power_bt, _BANK_TYPE_POWER),
            ]
            fast_kv, fast_mask = self._concat(fast_bts)
            action = self.cross_fast(action, fast_kv, fast_mask)
            # 串行 3 bank concat → 1 次 cross
            slow_bts = [
                self._tag(prefix_bt, _BANK_TYPE_PREFIX),
                self._tag(combat_memory_bt, _BANK_TYPE_COMBAT_MEMORY),
                self._tag(build_bt, _BANK_TYPE_BUILD),
            ]
            slow_kv, slow_mask = self._concat(slow_bts)
            action = self.cross_slow(action, slow_kv, slow_mask)

        else:  # minimal: 全部 concat 做单次 cross
            build_bt = BankTensor(
                features=build_slots,
                mask=torch.ones(build_slots.shape[:2], dtype=torch.bool, device=build_slots.device),
                bank_name="build_slots",
            )
            all_bts = [
                self._tag(board_bt, _BANK_TYPE_BOARD),
                self._tag(modifier_bt, _BANK_TYPE_MODIFIER),
                self._tag(mechanism_bt, _BANK_TYPE_MECHANISM),
                self._tag(power_bt, _BANK_TYPE_POWER),
                self._tag(prefix_bt, _BANK_TYPE_PREFIX),
                self._tag(combat_memory_bt, _BANK_TYPE_COMBAT_MEMORY),
                self._tag(build_bt, _BANK_TYPE_BUILD),
            ]
            all_kv, all_mask = self._concat(all_bts)
            action = self.cross_all(action, all_kv, all_mask)

        return BankTensor(action, action_bt.mask, "action_hypothesis")
