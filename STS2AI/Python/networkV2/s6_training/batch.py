"""Batch 工具：多个 UnifiedTokenBanks 拼成 batched tensor。

核心函数:
  collate_training_samples(samples) → BatchedBanks（含全部 multi-head targets）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from networkV2.s1_schema.token_banks import UnifiedTokenBanks, TokenBank, Token


@dataclass
class PaddedBank:
    """一个 bank 的 batched 结果。"""
    numeric: torch.Tensor    # (B, L_max, numeric_dim)
    type_ids: torch.Tensor   # (B, L_max)
    ts_ids: torch.Tensor     # (B, L_max)
    mask: torch.Tensor       # (B, L_max) bool
    bank_name: str = ""


@dataclass
class BatchedBanks:
    """多个样本 batch 后的结果。"""
    banks: dict[str, PaddedBank] = field(default_factory=dict)
    batch_size: int = 0
    decision_domains: list[str] = field(default_factory=list)
    # PPO labels
    action_indices: torch.Tensor | None = None      # (B,) long
    old_log_probs: torch.Tensor | None = None       # (B,)
    advantages: torch.Tensor | None = None           # (B,)
    returns: torch.Tensor | None = None              # (B,)
    # Multi-head value targets
    fight_win_targets: torch.Tensor | None = None    # (B,)
    hp_loss_targets: torch.Tensor | None = None      # (B,)
    survival_targets: torch.Tensor | None = None     # (B,)
    turn_damage_targets: torch.Tensor | None = None  # (B,) <0 = invalid (skip in loss)
    # Leaf targets
    leaf_targets: torch.Tensor | None = None         # (B,)
    transition_risk_targets: torch.Tensor | None = None    # (B,)
    resource_retention_targets: torch.Tensor | None = None  # (B,)
    # RunEvaluator auxiliary targets
    boss_readiness_targets: torch.Tensor | None = None      # (B,)
    resource_health_targets: torch.Tensor | None = None     # (B,)
    deck_quality_targets: torch.Tensor | None = None        # (B,)
    # Sample weights
    sample_weights: torch.Tensor | None = None       # (B,)


def _pad_bank(samples: list[TokenBank], max_numeric_dim: int = 48) -> PaddedBank:
    B = len(samples)
    max_len = max((len(bank) for bank in samples), default=1)
    max_len = max(max_len, 1)

    numeric = torch.zeros(B, max_len, max_numeric_dim)
    type_ids = torch.zeros(B, max_len, dtype=torch.long)
    ts_ids = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    bank_name = samples[0].bank_name if samples else ""

    for b, bank in enumerate(samples):
        for i, tok in enumerate(bank.tokens):
            n = min(len(tok.numeric), max_numeric_dim)
            numeric[b, i, :n] = torch.tensor(tok.numeric[:n], dtype=torch.float32)
            type_ids[b, i] = tok.type_idx
            ts_ids[b, i] = tok.time_scale_idx
            mask[b, i] = True

    return PaddedBank(numeric=numeric, type_ids=type_ids, ts_ids=ts_ids,
                      mask=mask, bank_name=bank_name)


def collate_banks(
    samples: list[UnifiedTokenBanks],
    max_numeric_dim: int = 48,
) -> BatchedBanks:
    B = len(samples)
    if B == 0:
        return BatchedBanks()

    all_bank_names: set[str] = set()
    for s in samples:
        for bank in s.all_banks():
            if not bank.is_empty:
                all_bank_names.add(bank.bank_name)

    banks: dict[str, PaddedBank] = {}
    for name in sorted(all_bank_names):
        bank_list = []
        for s in samples:
            found = None
            for bank in s.all_banks():
                if bank.bank_name == name:
                    found = bank
                    break
            bank_list.append(found or TokenBank(bank_name=name))
        banks[name] = _pad_bank(bank_list, max_numeric_dim)

    return BatchedBanks(
        banks=banks,
        decision_domains=[s.decision_domain for s in samples],
        batch_size=B,
    )


@dataclass
class TrainingSample:
    """一个训练样本：banks + 全部 labels。"""
    banks: UnifiedTokenBanks
    action_index: int = 0
    reward: float = 0.0
    # PPO
    value_target: float = 0.0       # GAE return（fight_win head 的监督主信号）
    advantage: float = 0.0          # GAE advantage
    old_log_prob: float = 0.0
    value_estimate: float = 0.0     # rollout 时网络的 value 输出（仅用于 GAE bootstrap，不做监督）
    # Multi-head value targets
    # fight_win_target: ≥0 时视为显式监督（如终局 0/1 硬标签）；<0（如 -1）视为"无显式监督"，
    # loss 会改用 returns (value_target) 作为目标 —— 避免 value head 自蒸馏
    fight_win_target: float = -1.0
    hp_loss_target: float = 0.0     # [0,+inf) 期望掉血
    survival_target: float = 1.0    # [0,1] 近期生存概率
    # 1-turn lookahead：从本步起到回合结束累计的实际造成伤害（含本步动作伤害）。
    # combo / 牌序学习的关键监督信号。<0 = 无效（非战斗或回合未关闭），loss 跳过。
    turn_damage_target: float = -1.0
    # Leaf head targets（leaf_evaluator 4 个 head 全监督）
    leaf_target: float = 0.0                  # leaf_score ∈ [-1,1]（= 2*value_target - 1）
    transition_risk_target: float = 0.0       # ∈ [0,1] 敌方行为切换频率
    resource_retention_target: float = 0.5    # ∈ [0,1] 资源保留度
    # RunEvaluator head targets（non-combat 样本用）
    boss_readiness_target: float = 0.5        # ∈ [0,1]
    resource_health_target: float = 0.5       # ∈ [0,1]
    deck_quality_target: float = 0.0          # ∈ [-1,1]
    # Meta
    sample_weight: float = 1.0
    encounter_id: str = ""
    room_type: str = "monster"


def collate_training_samples(
    samples: list[TrainingSample],
    max_numeric_dim: int = 48,
) -> BatchedBanks:
    """将 TrainingSample 列表拼成 BatchedBanks，含全部 multi-head targets。"""
    batched = collate_banks([s.banks for s in samples], max_numeric_dim)
    batched.action_indices = torch.tensor([s.action_index for s in samples], dtype=torch.long)
    batched.old_log_probs = torch.tensor([s.old_log_prob for s in samples], dtype=torch.float32)
    batched.advantages = torch.tensor([s.advantage for s in samples], dtype=torch.float32)
    batched.returns = torch.tensor([s.value_target for s in samples], dtype=torch.float32)
    batched.fight_win_targets = torch.tensor([s.fight_win_target for s in samples], dtype=torch.float32)
    batched.hp_loss_targets = torch.tensor([s.hp_loss_target for s in samples], dtype=torch.float32)
    batched.survival_targets = torch.tensor([s.survival_target for s in samples], dtype=torch.float32)
    batched.turn_damage_targets = torch.tensor([s.turn_damage_target for s in samples], dtype=torch.float32)
    batched.leaf_targets = torch.tensor([s.leaf_target for s in samples], dtype=torch.float32)
    batched.transition_risk_targets = torch.tensor([s.transition_risk_target for s in samples], dtype=torch.float32)
    batched.resource_retention_targets = torch.tensor([s.resource_retention_target for s in samples], dtype=torch.float32)
    batched.boss_readiness_targets = torch.tensor([s.boss_readiness_target for s in samples], dtype=torch.float32)
    batched.resource_health_targets = torch.tensor([s.resource_health_target for s in samples], dtype=torch.float32)
    batched.deck_quality_targets = torch.tensor([s.deck_quality_target for s in samples], dtype=torch.float32)
    batched.sample_weights = torch.tensor([s.sample_weight for s in samples], dtype=torch.float32)
    # Encounter conditioning index（方案 A: Conditional Policy）
    from networkV2.s1_schema.encounter_vocab import encounter_to_index
    batched.encounter_indices = torch.tensor(
        [encounter_to_index(s.encounter_id) for s in samples], dtype=torch.long,
    )
    return batched
