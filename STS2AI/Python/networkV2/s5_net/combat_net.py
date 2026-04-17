"""CombatNetV2: 战斗网络主入口。

完整流水线:
  UnifiedTokenBanks
  → BankTokenizer (Layer 1)
  → Memory Encoders (Layer 2)
  → Action Contextualizer (Layer 3) — 含 shared world context 注入
  → Decision Core (Layer 4)
  → Policy/Value/LeafEvaluator (Layer 5)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from networkV2.s1_schema.token_banks import UnifiedTokenBanks
from networkV2.s5_net.tokenizer import BankTokenizer, BankTensor
from networkV2.s5_net.encoders.build_encoder import BuildMemoryEncoder
from networkV2.s5_net.encoders.board_encoder import BoardEncoder
from networkV2.s5_net.encoders.mechanism_encoder import MechanismEncoder
from networkV2.s5_net.encoders.modifier_encoder import ModifierEncoder
from networkV2.s5_net.encoders.prefix_encoder import TurnPrefixEncoder
from networkV2.s5_net.encoders.combat_memory_encoder import CombatMemoryEncoder
from networkV2.s5_net.encoders.common import CrossAttentionBlock
from networkV2.s5_net.action_contextualizer import ActionContextualizer
from networkV2.s5_net.decision_core import DecisionCore
from networkV2.s5_net.heads.policy_head import PolicyHead
from networkV2.s5_net.heads.value_heads import ValueHeads, ValueOutputs
from networkV2.s5_net.heads.leaf_evaluator import LeafEvaluator, LeafOutputs


@dataclass
class CombatNetOutput:
    logits: torch.Tensor              # (B, L_action) policy logits
    values: ValueOutputs              # 4 value heads
    leaf: LeafOutputs                 # leaf evaluator
    action_mask: torch.Tensor         # (B, L_action) bool


class CombatNetV2(nn.Module):

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_build_slots: int = 8,
        max_numeric_dim: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Layer 1: Tokenizer
        self.tokenizer = BankTokenizer(d_model, max_numeric_dim)

        # Layer 2: Memory Encoders
        self.build_encoder = BuildMemoryEncoder(d_model, n_build_slots, n_iters=3, n_heads=max(n_heads // 2, 1))
        self.board_encoder = BoardEncoder(d_model, n_heads, n_layers=3, dropout=dropout)
        self.mechanism_encoder = MechanismEncoder(d_model, max(n_heads // 2, 1), n_layers=2, dropout=dropout)
        self.modifier_encoder = ModifierEncoder(d_model, max(n_heads // 2, 1), n_layers=2, dropout=dropout)
        self.prefix_encoder = TurnPrefixEncoder(d_model, max(n_heads // 2, 1), n_layers=2, dropout=dropout)
        self.combat_memory_encoder = CombatMemoryEncoder(d_model, max(n_heads // 2, 1), n_layers=1, dropout=dropout)

        # Shared world context: objective + forecast → 注入 action contextualizer
        self.shared_world_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.shared_world_gate = nn.Parameter(torch.tensor(0.1))

        # Layer 3: Action Contextualizer
        self.action_contextualizer = ActionContextualizer(d_model, n_heads, dropout)

        # Layer 4: Decision Core
        self.decision_core = DecisionCore(d_model, n_heads, n_layers=3, dropout=dropout)

        # Layer 5: Heads
        self.policy_head = PolicyHead(d_model)
        self.value_heads = ValueHeads(d_model)
        self.leaf_evaluator = LeafEvaluator(d_model)

    def forward(
        self,
        banks: UnifiedTokenBanks | None = None,
        batched_banks: dict | None = None,
    ) -> CombatNetOutput:
        device = next(self.parameters()).device

        # Layer 1: Tokenize
        if batched_banks is not None:
            bank_tensors = self.tokenizer.tokenize_batched_banks(batched_banks, device=device)
        else:
            assert banks is not None, "Need banks or batched_banks"
            bank_tensors = self.tokenizer.tokenize_banks(banks, device=device)

        # 推断 batch size
        B = 1
        for bt in bank_tensors.values():
            B = bt.features.size(0)
            break

        _e = lambda: self._empty_bt(device, B)
        build_bt = bank_tensors.get("build", _e())
        inventory_bt = bank_tensors.get("inventory")
        board_bt = bank_tensors.get("board", _e())
        mechanism_bt = bank_tensors.get("mechanism", _e())
        modifier_bt = bank_tensors.get("modifier", _e())
        prefix_bt = bank_tensors.get("turn_prefix", _e())
        combat_mem_bt = bank_tensors.get("combat_memory", _e())
        action_bt = bank_tensors.get("action", _e())
        # Shared world banks
        objective_bt = bank_tensors.get("objective", _e())
        forecast_bt = bank_tensors.get("forecast", _e())

        # Layer 2: Memory Encoders (空 bank 跳过)
        build_slots = self.build_encoder(build_bt, inventory_bt)
        board_enc = self.board_encoder(board_bt) if board_bt.mask.any() else board_bt
        mech_enc = self.mechanism_encoder(mechanism_bt) if mechanism_bt.mask.any() else mechanism_bt
        mod_enc = self.modifier_encoder(modifier_bt) if modifier_bt.mask.any() else modifier_bt
        prefix_enc = self.prefix_encoder(prefix_bt) if prefix_bt.mask.any() else prefix_bt
        cm_enc = self.combat_memory_encoder(combat_mem_bt) if combat_mem_bt.mask.any() else combat_mem_bt

        # Shared world context: objective + forecast → 注入到 action tokens
        obj_pool = self._masked_mean(objective_bt)
        fcast_pool = self._masked_mean(forecast_bt)
        world_ctx = self.shared_world_proj(torch.cat([obj_pool, fcast_pool], dim=-1))  # (B, d)
        # gate 后加到 action tokens 上
        action_features = action_bt.features + self.shared_world_gate * world_ctx.unsqueeze(1)
        action_bt = BankTensor(action_features, action_bt.mask, action_bt.bank_name)

        # Layer 3: Action Contextualizer
        action_hyp = self.action_contextualizer(
            action_bt, board_enc, mod_enc, mech_enc,
            prefix_enc, cm_enc, build_slots,
        )

        # Layer 4: Decision Core
        decision_repr, action_refined = self.decision_core(action_hyp)

        # Layer 5: Heads
        logits = self.policy_head(decision_repr, action_refined)
        values = self.value_heads(decision_repr)
        leaf = self.leaf_evaluator(decision_repr, board_enc, cm_enc, mech_enc, mod_enc)

        return CombatNetOutput(
            logits=logits,
            values=values,
            leaf=leaf,
            action_mask=action_bt.mask,
        )

    def _empty_bt(self, device: torch.device, batch_size: int = 1) -> BankTensor:
        return BankTensor(
            features=torch.zeros(batch_size, 1, self.d_model, device=device),
            mask=torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            bank_name="empty",
        )

    @staticmethod
    def _masked_mean(bt: BankTensor) -> torch.Tensor:
        mask = bt.mask.unsqueeze(-1).float()
        summed = (bt.features * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count
