"""Option Contextualizer: 非战斗选项上下文化。

支持三种模式（通过 NetworkConfig.option_contextualizer_mode 切换）:

  "full"   : 原设计 6 段独立 cross-attention
    - 并行 5 段 (build/inventory/economy/forecast/objective) + 串行 1 段 (build_slots)
    - attention ops: 6

  "merged" : 并行 5→1 + 串行保留
    - 1 个 cross: option → concat(build, inventory, economy, forecast, objective)
    - 1 个 cross: option → build_slots
    - attention ops: 2

  "minimal": 全合并
    - 1 个 cross: option → concat(all 6 inputs)
    - attention ops: 1

详细说明见 action_contextualizer.py 和 network_config.py。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networkV2.s5_net.encoders.common import CrossAttentionBlock
from networkV2.s5_net.tokenizer import BankTensor


_BANK_BUILD = 0
_BANK_INVENTORY = 1
_BANK_ECONOMY = 2
_BANK_FORECAST = 3
_BANK_OBJECTIVE = 4
_BANK_BUILD_SLOTS = 5
_N_BANKS = 6


class OptionContextualizer(nn.Module):

    def __init__(
        self, d_model: int = 384, n_heads: int = 8, dropout: float = 0.1,
        mode: str = "full",
    ):
        super().__init__()
        self.mode = mode
        ffn_dim = d_model * 2

        if mode == "full":
            self.cross_build = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.cross_inventory = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.cross_economy = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.cross_forecast = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.cross_objective = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            self.merge_gate = nn.Sequential(
                nn.Linear(d_model * 5, d_model), nn.GELU(),
            )
            self.merge_norm = nn.LayerNorm(d_model)
            self.cross_build_slots = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
        elif mode in ("merged", "minimal"):
            self.bank_type_emb = nn.Embedding(_N_BANKS, d_model)
            self.register_buffer(
                "_bank_type_ids",
                torch.arange(_N_BANKS, dtype=torch.long),
                persistent=False,
            )
            if mode == "merged":
                self.cross_shared = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
                self.cross_build_slots = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            else:
                self.cross_all = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
        else:
            raise ValueError(f"Unknown option contextualizer mode: {mode}")

    def _tag(self, bt: BankTensor, type_id: int) -> BankTensor:
        type_emb = self.bank_type_emb(self._bank_type_ids[type_id])
        return BankTensor(bt.features + type_emb, bt.mask, bt.bank_name)

    @staticmethod
    def _concat(bts: list[BankTensor]) -> tuple[torch.Tensor, torch.Tensor]:
        feats = torch.cat([b.features for b in bts], dim=1)
        masks = torch.cat([b.mask for b in bts], dim=1)
        return feats, masks

    def forward(
        self,
        option_bt: BankTensor,
        build_bt: BankTensor,
        inventory_bt: BankTensor,
        economy_bt: BankTensor,
        forecast_bt: BankTensor,
        objective_bt: BankTensor,
        build_slots: torch.Tensor,
    ) -> BankTensor:
        option = option_bt.features

        if self.mode == "full":
            o_build = self.cross_build(option, build_bt.features, build_bt.mask)
            o_inv = self.cross_inventory(option, inventory_bt.features, inventory_bt.mask)
            o_econ = self.cross_economy(option, economy_bt.features, economy_bt.mask)
            o_fcast = self.cross_forecast(option, forecast_bt.features, forecast_bt.mask)
            o_obj = self.cross_objective(option, objective_bt.features, objective_bt.mask)
            delta = torch.cat([o_build - option, o_inv - option, o_econ - option,
                               o_fcast - option, o_obj - option], dim=-1)
            merged = self.merge_gate(delta)
            option = self.merge_norm(option + merged)
            option = self.cross_build_slots(option, build_slots, kv_mask=None)

        elif self.mode == "merged":
            shared_bts = [
                self._tag(build_bt, _BANK_BUILD),
                self._tag(inventory_bt, _BANK_INVENTORY),
                self._tag(economy_bt, _BANK_ECONOMY),
                self._tag(forecast_bt, _BANK_FORECAST),
                self._tag(objective_bt, _BANK_OBJECTIVE),
            ]
            shared_kv, shared_mask = self._concat(shared_bts)
            option = self.cross_shared(option, shared_kv, shared_mask)
            option = self.cross_build_slots(option, build_slots, kv_mask=None)

        else:  # minimal
            build_slots_bt = BankTensor(
                features=build_slots,
                mask=torch.ones(build_slots.shape[:2], dtype=torch.bool, device=build_slots.device),
                bank_name="build_slots",
            )
            all_bts = [
                self._tag(build_bt, _BANK_BUILD),
                self._tag(inventory_bt, _BANK_INVENTORY),
                self._tag(economy_bt, _BANK_ECONOMY),
                self._tag(forecast_bt, _BANK_FORECAST),
                self._tag(objective_bt, _BANK_OBJECTIVE),
                self._tag(build_slots_bt, _BANK_BUILD_SLOTS),
            ]
            all_kv, all_mask = self._concat(all_bts)
            option = self.cross_all(option, all_kv, all_mask)

        return BankTensor(option, option_bt.mask, "option_hypothesis")
