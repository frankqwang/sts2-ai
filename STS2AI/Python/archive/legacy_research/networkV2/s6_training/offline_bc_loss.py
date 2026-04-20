"""Offline BC + 7-head Value Regression Loss(skada 离线监督)。

和 PPO NonCombatLoss 的区别:
  - 无 PPO clip / GAE / advantage(纯监督)
  - 无 old_log_probs(loader 产的 sample 没有 behavior policy)
  - 7 个 value head 全部有真值监督(skada 轨迹提供)
  - Policy CE with label_smoothing + action_mask
  - per-sample weight(is_victory × ascension)

Heads 对应的 loss:
  policy                   → weighted CE(to was_picked chosen_index)
  run_win_prob             → MSE(is_victory)
  boss_readiness           → MSE(启发式/真值)
  resource_health          → MSE(启发式)
  deck_quality             → MSE(启发式)
  expected_hp_loss         → smooth_l1(skada combat_stats.dmg_taken future)
  expected_dmg_output      → smooth_l1(skada combat_stats.dmg_dealt future)
  floor_clear_prob         → MSE(是否过本 checkpoint)

训练流程(train_noncombat_offline.py 负责):按 decision_domain 分组 forward,
本 loss 模块接受单个 domain 的 batch(UnifiedNetOutput + 对应 targets)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from networkV2.s5_net.unified_net import UnifiedNetOutput


logger = logging.getLogger(__name__)


@dataclass
class OfflineBCLossConfig:
    # Policy
    policy_coef: float = 1.0
    entropy_coef: float = 0.02
    label_smoothing: float = 0.05
    mask_invalid_actions: bool = True

    # RunEvaluator 7 head weights(按信号质量配比)
    run_win_coef: float = 1.0           # 真值来自 is_victory(最强信号)
    boss_readiness_coef: float = 0.5    # boss 战真值
    resource_health_coef: float = 0.3   # hp/gold/potion 可直接算
    deck_quality_coef: float = 0.5      # skada win_rate_delta 聚合
    # 新加 3 个 head:
    expected_hp_loss_coef: float = 0.02       # softplus 输出量级大(0-80),权重小防压其他
    expected_dmg_output_coef: float = 0.02    # 同 hp_loss
    floor_clear_coef: float = 0.3             # 层通关率 [0,1],sigmoid

    # Pairwise margin loss(档 3):显式学 chosen > rejected,不依赖 is_victory
    # 对 card_reward / relic 等 policy batch 格外有效,因 N 候选里 chosen 只有 1 个。
    pairwise_margin: float = 0.3              # margin size(logit space)
    pairwise_coef: float = 0.5                # 权重:补充主 CE,不压制它

    # future_deck_quality head(档 2):走到下个 boss 入口时 deck 质量的 value
    # 监督 target 从 loader 回填,-2 表示无效(跳过 loss);
    # advantage-weighted policy reweighting:credit 给真正贡献 build 提升的 choice
    future_dq_coef: float = 0.3               # MSE weight
    awr_advantage_coef: float = 1.0           # 0 = 不启用 AWR;>0 = 启用,policy CE × (1 + coef × tanh(adv))


class OfflineBCLoss(nn.Module):
    """7 head + per-domain policy 监督训练的 combined loss。"""

    def __init__(self, config: OfflineBCLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or OfflineBCLossConfig()

    def forward(
        self,
        output: UnifiedNetOutput,
        *,
        action_indices: torch.Tensor,                       # (B,)
        run_win_targets: torch.Tensor | None = None,        # (B,) [0,1]
        boss_readiness_targets: torch.Tensor | None = None, # (B,) [0,1]
        resource_health_targets: torch.Tensor | None = None,# (B,) [0,1]
        deck_quality_targets: torch.Tensor | None = None,   # (B,) [-1,1]
        # 新 3 个 target(可选,None → skip 该 head 的 loss)
        expected_hp_loss_targets: torch.Tensor | None = None,    # (B,) [0,+inf)
        expected_dmg_output_targets: torch.Tensor | None = None, # (B,) [0,+inf)
        floor_clear_targets: torch.Tensor | None = None,         # (B,) [0,1]
        # 档 2:future_deck_quality target,-2 表示无效(loss skip 该样本)
        future_dq_targets: torch.Tensor | None = None,           # (B,) ∈ [-1,1] 或 -2
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert output.run_eval is not None, "OfflineBCLoss requires run_eval head"

        device = action_indices.device
        B = action_indices.size(0)
        w = sample_weights if sample_weights is not None else torch.ones(B, device=device)
        w_norm = w / w.sum().clamp(min=1e-8)
        metrics: dict[str, float] = {}

        # ---- Policy: weighted CE(with label smoothing + action mask) ----
        logits = torch.nan_to_num(output.logits, nan=0.0)
        if self.cfg.mask_invalid_actions and output.action_mask is not None:
            mask = output.action_mask.to(logits.device)
            logits = logits.masked_fill(~mask, -1e9)

        log_probs_all = F.log_softmax(logits, dim=-1)
        n_classes = logits.size(-1)

        if self.cfg.label_smoothing > 0:
            ls = self.cfg.label_smoothing
            with torch.no_grad():
                target_dist = torch.full_like(log_probs_all, ls / max(n_classes - 1, 1))
                target_dist.scatter_(1, action_indices.unsqueeze(1), 1.0 - ls)
                if self.cfg.mask_invalid_actions and output.action_mask is not None:
                    mask = output.action_mask.to(target_dist.device)
                    target_dist = target_dist * mask.float()
                    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            ce = -(target_dist * log_probs_all).sum(dim=-1)
        else:
            ce = -log_probs_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # ---- AWR 加权(档 2 核心):credit 给 future_dq 提升大的 choice ----
        # adv_weight = 1 + coef × tanh(future_dq_target - baseline_dq)
        # baseline 取 batch 内 future_dq mean 作中心化
        awr_w = torch.ones_like(ce)
        if (future_dq_targets is not None) and self.cfg.awr_advantage_coef > 0:
            valid = (future_dq_targets > -1.5).float()          # -2 表示无效
            if valid.sum() > 0:
                # advantage = future_dq_target - mean(valid 样本) 近似 policy gradient 的 centered adv
                fdq = future_dq_targets.clamp(-1, 1)
                mean_fdq = (fdq * valid).sum() / valid.sum().clamp(min=1e-6)
                adv = (fdq - mean_fdq) * valid                  # invalid 样本 adv=0
                awr_w = 1.0 + self.cfg.awr_advantage_coef * torch.tanh(adv)
                metrics["bc_awr_mean_weight"] = awr_w.mean().item()
                metrics["bc_awr_adv_std"] = adv.std().item() if valid.sum() > 1 else 0.0

        policy_loss = (ce * w_norm * awr_w).sum()
        metrics["bc_policy_loss"] = policy_loss.item()

        # ---- Pairwise margin loss(档 3):chosen > rejected 每对 margin hinge ----
        # 对每个 sample:chosen 的 logit 必须比任何一个合法 rejected logit 至少高 margin
        # 不依赖 is_victory;纯比较 offer 内部"chosen vs the others"
        pairwise_loss = torch.tensor(0.0, device=device)
        if self.cfg.pairwise_coef > 0:
            # 取 chosen 的 logit
            chosen_logit = logits.gather(1, action_indices.unsqueeze(1))   # (B, 1)
            # 计算每个 rejected 的 violation: max(0, rej - chosen + margin)
            # 对 invalid action(已 masked 到 -1e9)自动被 relu clamp 为 0(margin + rej - chosen 为极大负)
            diff = logits - chosen_logit + self.cfg.pairwise_margin        # (B, C)
            # mask 掉 chosen 本身(不跟自己比)
            self_mask = torch.zeros_like(diff, dtype=torch.bool)
            self_mask.scatter_(1, action_indices.unsqueeze(1), True)
            diff = diff.masked_fill(self_mask, 0.0)                        # self pos 不贡献
            violation = F.relu(diff)                                       # (B, C)
            # 每 sample 平均 violation(考虑 valid action 数)
            if output.action_mask is not None:
                valid_cnt = output.action_mask.to(diff.device).float().sum(dim=-1).clamp(min=1.0)
            else:
                valid_cnt = torch.full((B,), float(n_classes), device=device)
            per_sample = violation.sum(dim=-1) / valid_cnt                  # (B,)
            pairwise_loss = (per_sample * w_norm).sum()
        metrics["bc_pairwise_loss"] = pairwise_loss.item()

        # Entropy reg
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        entropy_mean = (entropy * w_norm).sum()
        metrics["bc_entropy"] = entropy_mean.item()

        # Top-1 诊断
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == action_indices).float()
            top1 = (correct * w).sum() / w.sum().clamp(min=1e-8)
            metrics["bc_top1_acc"] = top1.item()

        # ---- 7 value heads ----
        re = output.run_eval

        vl_win = torch.tensor(0.0, device=device)
        if run_win_targets is not None:
            vl_win = ((re.run_win_prob - run_win_targets.clamp(0, 1)).pow(2) * w_norm).sum()
        metrics["bc_vl_run_win"] = vl_win.item()

        vl_br = torch.tensor(0.0, device=device)
        if boss_readiness_targets is not None:
            vl_br = ((re.boss_readiness - boss_readiness_targets.clamp(0, 1)).pow(2) * w_norm).sum()
        metrics["bc_vl_boss_ready"] = vl_br.item()

        vl_rh = torch.tensor(0.0, device=device)
        if resource_health_targets is not None:
            vl_rh = ((re.resource_health - resource_health_targets.clamp(0, 1)).pow(2) * w_norm).sum()
        metrics["bc_vl_resource_health"] = vl_rh.item()

        vl_dq = torch.tensor(0.0, device=device)
        if deck_quality_targets is not None:
            vl_dq = ((re.deck_quality - deck_quality_targets.clamp(-1, 1)).pow(2) * w_norm).sum()
        metrics["bc_vl_deck_quality"] = vl_dq.item()

        # 新 3 个 head
        vl_hp = torch.tensor(0.0, device=device)
        if expected_hp_loss_targets is not None:
            tgt = expected_hp_loss_targets.clamp(min=0.0)
            vl_hp = (F.smooth_l1_loss(re.expected_hp_loss, tgt, reduction="none") * w_norm).sum()
        metrics["bc_vl_exp_hp_loss"] = vl_hp.item()

        vl_dmg = torch.tensor(0.0, device=device)
        if expected_dmg_output_targets is not None:
            tgt = expected_dmg_output_targets.clamp(min=0.0)
            vl_dmg = (F.smooth_l1_loss(re.expected_dmg_output, tgt, reduction="none") * w_norm).sum()
        metrics["bc_vl_exp_dmg_output"] = vl_dmg.item()

        vl_fc = torch.tensor(0.0, device=device)
        if floor_clear_targets is not None:
            vl_fc = ((re.floor_clear_prob - floor_clear_targets.clamp(0, 1)).pow(2) * w_norm).sum()
        metrics["bc_vl_floor_clear"] = vl_fc.item()

        # future_deck_quality head 监督(档 2a):只对 target > -1.5 的有效样本计 loss
        vl_fdq = torch.tensor(0.0, device=device)
        if future_dq_targets is not None:
            valid_mask = (future_dq_targets > -1.5).float()
            if valid_mask.sum() > 0:
                fdq_tgt = future_dq_targets.clamp(-1, 1)
                per_sample = (re.future_deck_quality_at_boss - fdq_tgt).pow(2)
                vl_fdq = (per_sample * w_norm * valid_mask).sum()
        metrics["bc_vl_future_dq"] = vl_fdq.item()

        total = (
            self.cfg.policy_coef * policy_loss
            + self.cfg.pairwise_coef * pairwise_loss
            - self.cfg.entropy_coef * entropy_mean
            + self.cfg.run_win_coef * vl_win
            + self.cfg.boss_readiness_coef * vl_br
            + self.cfg.resource_health_coef * vl_rh
            + self.cfg.deck_quality_coef * vl_dq
            + self.cfg.expected_hp_loss_coef * vl_hp
            + self.cfg.expected_dmg_output_coef * vl_dmg
            + self.cfg.floor_clear_coef * vl_fc
            + self.cfg.future_dq_coef * vl_fdq
        )
        metrics["bc_total_loss"] = total.item()
        return total, metrics
