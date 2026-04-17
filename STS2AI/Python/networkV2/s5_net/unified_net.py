"""UnifiedNet: 战斗 + 非战斗统一网络入口。

根据 decision_domain 自动路由：
  combat → Action Contextualizer → Decision Core → Policy/Value/Leaf
  non-combat → Option Contextualizer → Decision Core → Policy/RunEvaluator

配置通过 NetworkConfig 控制所有层数，支持 slim/full 切换。
详见 network_config.py。

兼容旧接口：构造时可直接传 d_model/n_heads 等参数（会自动转成 NetworkConfig）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from networkV2.s1_schema.token_banks import UnifiedTokenBanks
from networkV2.s5_net.network_config import NetworkConfig, preset_full
from networkV2.s5_net.tokenizer import BankTokenizer, BankTensor
from networkV2.s5_net.encoders.build_encoder import BuildMemoryEncoder
from networkV2.s5_net.encoders.board_encoder import BoardEncoder
from networkV2.s5_net.encoders.mechanism_encoder import MechanismEncoder
from networkV2.s5_net.encoders.modifier_encoder import ModifierEncoder
from networkV2.s5_net.encoders.power_encoder import PowerEncoder
from networkV2.s5_net.encoders.prefix_encoder import TurnPrefixEncoder
from networkV2.s5_net.encoders.combat_memory_encoder import CombatMemoryEncoder
from networkV2.s5_net.action_contextualizer import ActionContextualizer
from networkV2.s5_net.option_contextualizer import OptionContextualizer
from networkV2.s5_net.decision_core import DecisionCore
from networkV2.s5_net.heads.policy_head import PolicyHead
from networkV2.s5_net.heads.value_heads import ValueHeads, ValueOutputs
from networkV2.s5_net.heads.leaf_evaluator import LeafEvaluator, LeafOutputs
from networkV2.s5_net.heads.run_evaluator import RunEvaluator, RunEvalOutputs


@dataclass
class UnifiedNetOutput:
    logits: torch.Tensor
    action_mask: torch.Tensor
    decision_domain: str
    values: ValueOutputs | None = None
    leaf: LeafOutputs | None = None
    run_eval: RunEvalOutputs | None = None


class UnifiedNet(nn.Module):
    """战斗 + 非战斗统一网络。"""

    def __init__(
        self,
        config: NetworkConfig | None = None,
        # --- 兼容旧 API（如果传了旧参数就用旧参数覆盖默认 config）---
        d_model: int | None = None,
        n_heads: int | None = None,
        n_build_slots: int | None = None,
        max_numeric_dim: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()

        # 兼容：旧调用方式传的散参数合成 config
        if config is None:
            config = preset_full()
        if d_model is not None: config.d_model = d_model
        if n_heads is not None: config.n_heads = n_heads
        if n_build_slots is not None: config.n_build_slots = n_build_slots
        if max_numeric_dim is not None: config.max_numeric_dim = max_numeric_dim
        if dropout is not None: config.dropout = dropout

        self.config = config
        d = config.d_model
        nh = config.n_heads
        nh_half = max(nh // 2, 1)
        drop = config.dropout
        self.d_model = d

        # ---- Shared ----
        self.tokenizer = BankTokenizer(d, config.max_numeric_dim)
        self.build_encoder = BuildMemoryEncoder(
            d, config.n_build_slots, n_iters=config.build_n_iters, n_heads=nh_half)
        self.decision_core = DecisionCore(d, nh, n_layers=config.decision_n_layers, dropout=drop)
        self.policy_head = PolicyHead(d)

        # Shared world context injection
        self.shared_world_proj = nn.Sequential(nn.Linear(d * 2, d), nn.GELU())
        self.shared_world_gate = nn.Parameter(torch.tensor(0.1))

        # ---- Combat branch ----
        self.board_encoder = BoardEncoder(d, nh, n_layers=config.board_n_layers, dropout=drop)
        self.mechanism_encoder = MechanismEncoder(d, nh_half, n_layers=config.mechanism_n_layers, dropout=drop)
        self.modifier_encoder = ModifierEncoder(d, nh_half, n_layers=config.modifier_n_layers, dropout=drop)
        self.power_encoder = PowerEncoder(d, nh_half, n_layers=config.power_n_layers, dropout=drop)
        self.prefix_encoder = TurnPrefixEncoder(d, nh_half, n_layers=config.prefix_n_layers, dropout=drop)
        self.combat_memory_encoder = CombatMemoryEncoder(
            d, nh_half, n_layers=config.combat_memory_n_layers, dropout=drop)
        self.action_contextualizer = ActionContextualizer(
            d, nh, drop, mode=config.contextualizer_mode)
        self.value_heads = ValueHeads(d)
        self.leaf_evaluator = LeafEvaluator(d)

        # ---- Non-combat branch ----
        self.option_contextualizer = OptionContextualizer(
            d, nh, drop, mode=config.option_contextualizer_mode)
        self.run_evaluator = RunEvaluator(d)

        # ---- Encounter Conditioning (方案 A: Conditional Policy) ----
        # 给 decision_repr 注入 encounter-specific bias，所有下游 head（policy /
        # value / leaf / run_eval）自动继承 boss context。
        #
        # 2026-04-17 co17 诊断: 原 init (gate=0.1, embed std=0.02) 导致注入量级仅
        # ~0.002，相对 decision_repr (~0.5-1.0) 的信号比 1e-5，PPO 梯度
        # ∂L/∂gate 几乎为 0 → 死锁：gate 不更新 → 注入持续小 → 永远没梯度信号。
        # co17 iter 40 时 gate 仍然 0.108（几乎没动），embed norm 也没分化。
        #
        # 修复：gate init 1.0 + embed std 0.3 → 初始注入 ~0.3（decision_repr 30%），
        # PPO 能明显感受到 conditioning 的 causal effect，梯度正常流动。
        # 代价：部分破坏继承 checkpoint 的 policy 表示，前几 iter 胜率会掉，
        # 但很快会适应 + 真正学会 conditioning。
        self.enable_encounter_conditioning = bool(
            getattr(config, "enable_encounter_conditioning", False))
        if self.enable_encounter_conditioning:
            n_enc = int(getattr(config, "n_encounters", 128))
            self.encounter_embed = nn.Embedding(n_enc, d)
            nn.init.normal_(self.encounter_embed.weight, mean=0.0, std=0.3)
            self.encounter_gate = nn.Parameter(torch.tensor(1.0))
        else:
            self.encounter_embed = None
            self.encounter_gate = None

    def _apply_encounter_conditioning(
        self,
        decision_repr: torch.Tensor,
        encounter_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        """给 decision_repr 加上 encounter-specific bias。"""
        if not self.enable_encounter_conditioning or encounter_idx is None:
            return decision_repr
        idx = encounter_idx.to(device=decision_repr.device, dtype=torch.long)
        idx = idx.clamp(min=0, max=self.encounter_embed.num_embeddings - 1)
        boss_bias = self.encounter_embed(idx)
        return decision_repr + self.encounter_gate * boss_bias

    def forward(
        self,
        banks: UnifiedTokenBanks | None = None,
        batched_banks: dict | None = None,
        decision_domain: str = "combat",
        encounter_idx: torch.Tensor | None = None,
    ) -> UnifiedNetOutput:
        """
        encounter_idx: (B,) long tensor。若 enable_encounter_conditioning=True，
        decision_repr 会加上 encounter-specific bias。调用方从
        `networkV2.s1_schema.encounter_vocab.encounter_to_index(encounter_id)` 得到。
        传 None 或 全 0 → UNKNOWN，不加 bias。
        """
        device = next(self.parameters()).device

        if batched_banks is not None:
            bt = self.tokenizer.tokenize_batched_banks(batched_banks, device=device)
        else:
            assert banks is not None
            bt = self.tokenizer.tokenize_banks(banks, device=device)
            decision_domain = banks.decision_domain

        B = next(iter(bt.values())).features.size(0) if bt else 1
        _e = lambda: self._empty_bt(device, B)

        build_bt = bt.get("build", _e())
        inventory_bt = bt.get("inventory")
        economy_bt = bt.get("economy", _e())
        objective_bt = bt.get("objective", _e())
        forecast_bt = bt.get("forecast", _e())
        action_bt = bt.get("action", _e())

        build_slots = self.build_encoder(build_bt, inventory_bt)

        obj_pool = self._masked_mean(objective_bt)
        fcast_pool = self._masked_mean(forecast_bt)
        world_ctx = self.shared_world_proj(torch.cat([obj_pool, fcast_pool], dim=-1))
        action_features = action_bt.features + self.shared_world_gate * world_ctx.unsqueeze(1)
        action_bt = BankTensor(action_features, action_bt.mask, action_bt.bank_name)

        if decision_domain == "combat":
            return self._combat_forward(bt, B, device, action_bt, build_slots,
                                        build_bt, inventory_bt, objective_bt, forecast_bt,
                                        encounter_idx=encounter_idx)
        else:
            return self._noncombat_forward(bt, B, device, action_bt, build_slots,
                                           build_bt, inventory_bt, economy_bt,
                                           objective_bt, forecast_bt, decision_domain,
                                           encounter_idx=encounter_idx)

    def _combat_forward(self, bt, B, device, action_bt, build_slots,
                        build_bt, inventory_bt, objective_bt, forecast_bt,
                        encounter_idx=None):
        _e = lambda: self._empty_bt(device, B)
        board_bt = bt.get("board", _e())
        mechanism_bt = bt.get("mechanism", _e())
        modifier_bt = bt.get("modifier", _e())
        power_bt = bt.get("power", _e())
        prefix_bt = bt.get("turn_prefix", _e())
        combat_mem_bt = bt.get("combat_memory", _e())

        board_enc = self.board_encoder(board_bt) if board_bt.mask.any() else board_bt
        mech_enc = self.mechanism_encoder(mechanism_bt) if mechanism_bt.mask.any() else mechanism_bt
        mod_enc = self.modifier_encoder(modifier_bt) if modifier_bt.mask.any() else modifier_bt
        power_enc = self.power_encoder(power_bt) if power_bt.mask.any() else power_bt
        prefix_enc = self.prefix_encoder(prefix_bt) if prefix_bt.mask.any() else prefix_bt
        cm_enc = self.combat_memory_encoder(combat_mem_bt) if combat_mem_bt.mask.any() else combat_mem_bt

        action_hyp = self.action_contextualizer(
            action_bt, board_enc, mod_enc, mech_enc, power_enc, prefix_enc, cm_enc, build_slots)

        decision_repr, action_refined = self.decision_core(action_hyp)
        decision_repr = self._apply_encounter_conditioning(decision_repr, encounter_idx)

        logits = self.policy_head(decision_repr, action_refined)
        values = self.value_heads(decision_repr)
        leaf = self.leaf_evaluator(decision_repr, board_enc, cm_enc, mech_enc, mod_enc, power_enc)

        return UnifiedNetOutput(
            logits=logits, action_mask=action_bt.mask,
            decision_domain="combat", values=values, leaf=leaf)

    def _noncombat_forward(self, bt, B, device, option_bt, build_slots,
                           build_bt, inventory_bt, economy_bt, objective_bt, forecast_bt,
                           domain, encounter_idx=None):
        _e = lambda: self._empty_bt(device, B)
        inv_bt = inventory_bt or _e()

        option_hyp = self.option_contextualizer(
            option_bt, build_bt, inv_bt, economy_bt, forecast_bt, objective_bt, build_slots)

        decision_repr, option_refined = self.decision_core(option_hyp)
        decision_repr = self._apply_encounter_conditioning(decision_repr, encounter_idx)

        logits = self.policy_head(decision_repr, option_refined)
        run_eval = self.run_evaluator(decision_repr, build_bt, inv_bt, objective_bt, forecast_bt)

        return UnifiedNetOutput(
            logits=logits, action_mask=option_bt.mask,
            decision_domain=domain, run_eval=run_eval)

    def _empty_bt(self, device: torch.device, batch_size: int = 1) -> BankTensor:
        return BankTensor(
            features=torch.zeros(batch_size, 1, self.d_model, device=device),
            mask=torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            bank_name="empty")

    @staticmethod
    def _masked_mean(bt: BankTensor) -> torch.Tensor:
        mask = bt.mask.unsqueeze(-1).float()
        summed = (bt.features * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count

    # ------------------------------------------------------------------
    # Checkpoint 跨配置兼容加载
    # ------------------------------------------------------------------

    def load_compatible_params(self, state_dict: dict, strict_shapes: bool = True) -> dict:
        """从 checkpoint 加载 shape 匹配的参数（跨配置部分继承）。

        典型用法:
            # 用 slim 预设训了一版 checkpoint_slim.pt
            # 切换到 full 预设继续训练
            net_full = UnifiedNet(config=preset_full())
            report = net_full.load_compatible_params(torch.load("checkpoint_slim.pt"))
            print(report)  # 看加载了多少参数、跳过多少

        Args:
            state_dict: 旧 checkpoint 的 state_dict
            strict_shapes: True 时只加载 shape 完全匹配的；False 时允许部分 tensor 的前 N 个切片

        Returns:
            加载报告 dict: {loaded, skipped_shape, missing}
        """
        own_state = self.state_dict()
        loaded: list[str] = []
        skipped_shape: list[str] = []
        missing: list[str] = [k for k in own_state if k not in state_dict]

        new_state = dict(own_state)
        for k, v in state_dict.items():
            if k not in own_state:
                continue
            target = own_state[k]
            if target.shape == v.shape:
                new_state[k] = v
                loaded.append(k)
            elif not strict_shapes:
                # 允许部分切片（比如 full 版比 slim 多几层，但共享层 shape 一致的还能用）
                try:
                    slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, v.shape))
                    new_state[k][slices] = v[slices]
                    loaded.append(f"{k}(partial)")
                except Exception:
                    skipped_shape.append(f"{k}: got {tuple(v.shape)} expected {tuple(target.shape)}")
            else:
                skipped_shape.append(f"{k}: got {tuple(v.shape)} expected {tuple(target.shape)}")

        self.load_state_dict(new_state, strict=False)

        return {
            "loaded": len(loaded),
            "skipped_shape": len(skipped_shape),
            "missing": len(missing),
            "loaded_keys_sample": loaded[:5],
            "skipped_sample": skipped_shape[:5],
        }
