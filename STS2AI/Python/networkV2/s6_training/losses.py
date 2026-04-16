"""Loss 函数：PPO policy + 4 value heads + leaf evaluator，全部 multi-head 监督。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from networkV2.s5_net.combat_net import CombatNetOutput
from networkV2.s5_net.unified_net import UnifiedNetOutput


@dataclass
class LossConfig:
    clip_eps: float = 0.15  # 原 0.2 对 slim 网络偏大，导致 policy 突变后策略崩溃
    # entropy_coef 0.01 太低 → long1 iter 10 监测到战斗 policy 偏好动作 0
    # 升到 0.03 强制保留更多探索；KL 早停的存在让我们可以激进点
    entropy_coef: float = 0.03
    # policy_coef: 1.0 正常训练；0.0 = 冻结 policy 只训 value head（warmup 用）
    policy_coef: float = 1.0
    value_coef: float = 0.5
    fight_win_coef: float = 1.0
    # hp_loss raw 数值大（target ∈ [0,80]，softplus 输出可达 100+），
    # smooth_l1 出来 ~10。原 coef 0.5 → 贡献 4.9 ≫ 其他 head 总和 0.3，
    # 导致 value_loss 通过 shared encoder 推飞 policy（KL 爆炸主因）。
    # 调到 0.02 → 贡献 ~0.2，与 fight_win 同量级。
    hp_loss_coef: float = 0.02
    survival_coef: float = 0.5
    tempo_coef: float = 0.3
    leaf_coef: float = 0.2
    # leaf_evaluator 其余 3 个 head 的权重（小：避免盖过主 head，同时消除"白训"）
    transition_risk_coef: float = 0.1
    survival_margin_coef: float = 0.1
    resource_retention_coef: float = 0.1
    # advantages 归一化：True = loss 内对 minibatch 做 (adv-mean)/std；
    # False = advantages 已被外部（ppo.py）全局归一化好，loss 内不再重算。
    # 历史上 True 会导致每 minibatch 内 mean(adv)=0，ratio≈1 时 policy_loss 永远=0。
    normalize_adv: bool = True


class CombatLoss(nn.Module):
    """战斗综合 loss：PPO + 4 value heads + leaf + entropy。"""

    def __init__(self, config: LossConfig | None = None):
        super().__init__()
        self.cfg = config or LossConfig()

    def forward(
        self,
        output: CombatNetOutput,
        action_indices: torch.Tensor,           # (B,)
        old_log_probs: torch.Tensor,            # (B,)
        advantages: torch.Tensor,               # (B,)
        returns: torch.Tensor,                  # (B,)
        *,
        fight_win_targets: torch.Tensor | None = None,
        hp_loss_targets: torch.Tensor | None = None,
        survival_targets: torch.Tensor | None = None,
        leaf_targets: torch.Tensor | None = None,
        transition_risk_targets: torch.Tensor | None = None,
        resource_retention_targets: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        device = action_indices.device
        B = action_indices.size(0)
        w = sample_weights if sample_weights is not None else torch.ones(B, device=device)
        w = w / w.sum().clamp(min=1)  # 归一化
        metrics: dict[str, float] = {}

        # ---- Policy loss (PPO clipped) ----
        logits = torch.nan_to_num(output.logits, nan=0.0)
        log_probs_all = F.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # log_ratio 收紧到 [-3, 3]：原 [-10, 10] 让 ratio 范围 [4.5e-5, 22026]，
        # 一旦极端值参与 surr1/surr2 就把 minibatch 梯度吹爆（NaN 主源）。
        # [-3, 3] → ratio ∈ [0.05, 20]，对 PPO 更新足够宽松。
        log_ratio = (log_probs - old_log_probs).clamp(-3, 3)
        ratio = torch.exp(log_ratio)
        # advantages 归一化：默认在 loss 内做（向后兼容）；
        # 关键 bug 修复：当 normalize_adv=False 时假设外部已全局归一化，loss 内不再
        # 重算——否则每 minibatch mean(adv)=0，ratio≈1 时 policy_loss 恒为 0。
        if self.cfg.normalize_adv:
            adv_std = advantages.std()
            adv = (advantages - advantages.mean()) / (adv_std + 1e-8) if adv_std > 1e-8 else advantages * 0.0
        else:
            adv = advantages
        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
        policy_loss = -(torch.min(surr1, surr2) * w).sum()
        metrics["policy_loss"] = policy_loss.item()

        # Approx KL divergence (Schulman): http://joschu.net/blog/kl-approx.html
        # KL ≈ E[(ratio - 1) - log_ratio]，无偏且非负
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        metrics["approx_kl"] = approx_kl

        # Entropy
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        entropy_mean = (entropy * w).sum()
        metrics["entropy"] = entropy_mean.item()

        # ---- Value losses (4 heads, 全部接监督) ----

        # fight_win: MSE
        #   - fight_win_targets >= 0 : 显式监督（终局 0/1 硬标签）
        #   - fight_win_targets <  0 : 哨值 → 改用 returns (GAE return) 作目标
        # 这样非终局 step 不再以网络旧 value 为目标（避免自蒸馏）
        returns_clamped = returns.clamp(0, 1)
        if fight_win_targets is None:
            win_t = returns_clamped
        else:
            win_t = torch.where(fight_win_targets >= 0.0, fight_win_targets.clamp(0, 1), returns_clamped)
        vl_win = ((output.values.fight_win - win_t).pow(2) * w).sum()
        metrics["vl_fight_win"] = vl_win.item()

        # hp_loss: smooth L1
        vl_hp = torch.tensor(0.0, device=device)
        if hp_loss_targets is not None:
            vl_hp = (F.smooth_l1_loss(output.values.expected_hp_loss, hp_loss_targets, reduction="none") * w).sum()
        metrics["vl_hp_loss"] = vl_hp.item()

        # survival: MSE (不用 BCE，避免 target 超界问题)
        vl_surv = torch.tensor(0.0, device=device)
        if survival_targets is not None:
            vl_surv = ((output.values.survival_2turn - survival_targets.clamp(0, 1)).pow(2) * w).sum()
        metrics["vl_survival"] = vl_surv.item()

        # tempo: MSE, target = tanh(advantages) 作为粗略节奏信号
        tempo_t = torch.tanh(advantages)
        vl_tempo = ((output.values.tempo - tempo_t).pow(2) * w).sum()
        metrics["vl_tempo"] = vl_tempo.item()

        value_loss = (
            self.cfg.fight_win_coef * vl_win
            + self.cfg.hp_loss_coef * vl_hp
            + self.cfg.survival_coef * vl_surv
            + self.cfg.tempo_coef * vl_tempo
        )
        metrics["value_loss"] = value_loss.item()

        # ---- Leaf evaluator：全 4 个 head 监督（消除饥饿 head）----
        leaf_loss = torch.tensor(0.0, device=device)
        if leaf_targets is not None:
            # leaf_score ∈ [-1,1]；target 用 2*value_target-1 映射到相同范围
            leaf_loss = (F.smooth_l1_loss(output.leaf.leaf_score, leaf_targets, reduction="none") * w).sum()
        metrics["leaf_loss"] = leaf_loss.item()

        # transition_risk ∈ [0,1]: 敌方行为切换频率
        tr_loss = torch.tensor(0.0, device=device)
        if transition_risk_targets is not None:
            tr_loss = ((output.leaf.transition_risk - transition_risk_targets.clamp(0, 1)).pow(2) * w).sum()
        metrics["leaf_transition_risk"] = tr_loss.item()

        # survival_margin ∈ [0,1]: 复用 survival_target
        sm_loss = torch.tensor(0.0, device=device)
        if survival_targets is not None:
            sm_loss = ((output.leaf.survival_margin - survival_targets.clamp(0, 1)).pow(2) * w).sum()
        metrics["leaf_survival_margin"] = sm_loss.item()

        # resource_retention ∈ [0,1]
        rr_loss = torch.tensor(0.0, device=device)
        if resource_retention_targets is not None:
            rr_loss = ((output.leaf.resource_retention - resource_retention_targets.clamp(0, 1)).pow(2) * w).sum()
        metrics["leaf_resource_retention"] = rr_loss.item()

        # ---- Total ----
        total = (
            self.cfg.policy_coef * policy_loss
            - self.cfg.entropy_coef * entropy_mean
            + self.cfg.value_coef * value_loss
            + self.cfg.leaf_coef * leaf_loss
            + self.cfg.transition_risk_coef * tr_loss
            + self.cfg.survival_margin_coef * sm_loss
            + self.cfg.resource_retention_coef * rr_loss
        )
        metrics["total_loss"] = total.item()

        return total, metrics


@dataclass
class NonCombatLossConfig:
    clip_eps: float = 0.15
    # entropy_coef 0.01 → 0.05：long1 监测到 card_reward 100% skip / shop 100% leave 的
    # entropy collapse。non-combat 选项少（card_reward 4 个、shop 10 个），collapse 风险更高，
    # 比 combat 更激进。
    entropy_coef: float = 0.05
    policy_coef: float = 1.0   # 0.0 = 冻结 policy 只训 value head（warmup）
    # RunEvaluator 4 head 的权重：run_win_prob 为主，其余 3 个小权重避免饥饿
    run_win_coef: float = 1.0
    boss_readiness_coef: float = 0.2
    resource_health_coef: float = 0.2
    deck_quality_coef: float = 0.2
    # 同 LossConfig.normalize_adv：默认 True（loss 内归一化），ppo.py 全局归一化时设 False。
    normalize_adv: bool = True


class NonCombatLoss(nn.Module):
    """非战斗 loss：PPO policy + run_evaluator.run_win_prob 监督 + entropy。

    设计要点：
    - `run_win_prob` 的监督目标 = GAE returns（与 PPO 一致），不使用网络自己的旧预测
      避免出现 P1-2 里 combat head 的自蒸馏问题。
    - 其他 run_eval head（boss_readiness / resource_health / deck_quality）暂不监督，
      留作纯特征输出；若后续想监督可再加。
    """

    def __init__(self, config: NonCombatLossConfig | None = None):
        super().__init__()
        self.cfg = config or NonCombatLossConfig()

    def forward(
        self,
        output: UnifiedNetOutput,
        action_indices: torch.Tensor,       # (B,)
        old_log_probs: torch.Tensor,        # (B,)
        advantages: torch.Tensor,           # (B,)
        returns: torch.Tensor,              # (B,) ∈ [0,1] GAE return
        *,
        run_win_targets: torch.Tensor | None = None,  # (B,) >=0 显式监督，<0 fallback 到 returns
        boss_readiness_targets: torch.Tensor | None = None,   # (B,) ∈ [0,1]
        resource_health_targets: torch.Tensor | None = None,  # (B,) ∈ [0,1]
        deck_quality_targets: torch.Tensor | None = None,     # (B,) ∈ [-1,1]
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert output.run_eval is not None, "NonCombatLoss requires UnifiedNetOutput with run_eval"

        device = action_indices.device
        B = action_indices.size(0)
        w = sample_weights if sample_weights is not None else torch.ones(B, device=device)
        w = w / w.sum().clamp(min=1)
        metrics: dict[str, float] = {}

        # ---- Policy loss (PPO clipped) ----
        logits = torch.nan_to_num(output.logits, nan=0.0)
        log_probs_all = F.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # log_ratio 收紧到 [-3, 3]：原 [-10, 10] 让 ratio 范围 [4.5e-5, 22026]，
        # 一旦极端值参与 surr1/surr2 就把 minibatch 梯度吹爆（NaN 主源）。
        # [-3, 3] → ratio ∈ [0.05, 20]，对 PPO 更新足够宽松。
        log_ratio = (log_probs - old_log_probs).clamp(-3, 3)
        ratio = torch.exp(log_ratio)
        # 见 CombatLoss 同段注释：normalize_adv=False 时由 ppo.py 全局归一化。
        if self.cfg.normalize_adv:
            adv_std = advantages.std()
            adv = (advantages - advantages.mean()) / (adv_std + 1e-8) if adv_std > 1e-8 else advantages * 0.0
        else:
            adv = advantages
        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
        policy_loss = -(torch.min(surr1, surr2) * w).sum()
        metrics["nc_policy_loss"] = policy_loss.item()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        metrics["nc_approx_kl"] = approx_kl

        # Entropy
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        entropy_mean = (entropy * w).sum()
        metrics["nc_entropy"] = entropy_mean.item()

        # ---- Run-win head（同 combat fight_win 的监督策略）----
        returns_clamped = returns.clamp(0, 1)
        if run_win_targets is None:
            win_t = returns_clamped
        else:
            win_t = torch.where(run_win_targets >= 0.0, run_win_targets.clamp(0, 1), returns_clamped)
        vl_win = ((output.run_eval.run_win_prob - win_t).pow(2) * w).sum()
        metrics["nc_vl_run_win"] = vl_win.item()

        # ---- 其余 3 个 head 加启发式监督，消除饥饿 head ----
        br_loss = torch.tensor(0.0, device=device)
        if boss_readiness_targets is not None:
            br_loss = ((output.run_eval.boss_readiness - boss_readiness_targets.clamp(0, 1)).pow(2) * w).sum()
        metrics["nc_vl_boss_ready"] = br_loss.item()

        rh_loss = torch.tensor(0.0, device=device)
        if resource_health_targets is not None:
            rh_loss = ((output.run_eval.resource_health - resource_health_targets.clamp(0, 1)).pow(2) * w).sum()
        metrics["nc_vl_resource_health"] = rh_loss.item()

        dq_loss = torch.tensor(0.0, device=device)
        if deck_quality_targets is not None:
            # deck_quality ∈ [-1,1]（tanh 输出）
            dq_loss = ((output.run_eval.deck_quality - deck_quality_targets.clamp(-1, 1)).pow(2) * w).sum()
        metrics["nc_vl_deck_quality"] = dq_loss.item()

        total = (
            self.cfg.policy_coef * policy_loss
            - self.cfg.entropy_coef * entropy_mean
            + self.cfg.run_win_coef * vl_win
            + self.cfg.boss_readiness_coef * br_loss
            + self.cfg.resource_health_coef * rh_loss
            + self.cfg.deck_quality_coef * dq_loss
        )
        metrics["nc_total_loss"] = total.item()
        return total, metrics
