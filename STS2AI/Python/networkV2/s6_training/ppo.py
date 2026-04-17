"""PPO Training Step：单步 PPO 梯度更新，完整 multi-head 监督。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from networkV2.s5_net.combat_net import CombatNetV2
from networkV2.s5_net.unified_net import UnifiedNet
from networkV2.s6_training.batch import BatchedBanks, TrainingSample, collate_training_samples
from networkV2.s6_training.losses import CombatLoss, LossConfig, NonCombatLoss, NonCombatLossConfig


def _globally_normalize_advantages(samples: list[TrainingSample]) -> list[TrainingSample]:
    """对一组 TrainingSample 的 advantages 做一次全局 (mean=0, std=1) 归一化，
    返回新的样本列表（不修改原样本，避免跨 train_step 累积归一化）。

    必须在 train_step 入口调用一次而不是每 minibatch 调一次——否则每个 minibatch 内
    mean(adv)=0 会让 PPO 的 ratio≈1 时 policy_loss 恒等于 0（已知 bug）。
    """
    if len(samples) < 2:
        # n<2 时 `torch.std` 默认 unbiased=True 会返回 NaN（N-1=0 分母），
        # 后续除法会把所有 advantage 污染成 NaN。batch 太小无统计意义，直接返回原值。
        return samples
    advs = torch.tensor([s.advantage for s in samples], dtype=torch.float64)
    mean = advs.mean().item()
    # unbiased=False 使 n=1 时 std=0 而非 NaN；虽然我们上面已 early-return，
    # 这里也保持和"总体 std"语义一致。
    std = advs.std(unbiased=False).item()
    if std <= 1e-8:
        # 全 batch adv 几乎相同 → 无信号，留原值
        return samples
    return [replace(s, advantage=(s.advantage - mean) / (std + 1e-8)) for s in samples]


@dataclass
class PPOConfig:
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_numeric_dim: int = 48
    # Value warmup: 前 N 轮 train_step 调用时 policy_coef=0，只训 value head
    value_warmup_iters: int = 0
    # KL 早停：一个 epoch 内 minibatch 的平均 approx_kl 超过阈值就结束当前 epoch
    # 防止 PPO 策略更新过大导致 catastrophic forgetting
    # 0 = 禁用；典型值 0.015-0.03
    target_kl: float = 0.02
    loss_config: LossConfig | None = None


class CombatPPOTrainerV2:
    def __init__(self, net: CombatNetV2, config: PPOConfig | None = None):
        self.net = net
        self.cfg = config or PPOConfig()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.cfg.lr)
        # 关键：advantages 在 ppo.py 入口做了一次全局归一化，loss 内不再重算
        loss_cfg = self.cfg.loss_config or LossConfig()
        loss_cfg.normalize_adv = False
        self.loss_fn = CombatLoss(loss_cfg)

    def train_step(self, samples: list[TrainingSample]) -> dict[str, float]:
        """对一组 TrainingSample 做 PPO 更新。全部 multi-head targets 传入 loss。"""
        if not samples:
            return {}

        # 全局归一化 advantages（每个 train_step 调一次，不是每 minibatch）
        samples = _globally_normalize_advantages(samples)

        self.net.train()
        device = next(self.net.parameters()).device
        all_metrics: list[dict[str, float]] = []
        nan_skip_count = 0  # 诊断：被 NaN 跳过的 minibatch 数
        epochs_done = 0  # 实际跑完的 epoch 数（考虑 KL 早停）

        for _epoch in range(self.cfg.ppo_epochs):
            indices = torch.randperm(len(samples))
            # 和 UnifiedPPOTrainer 同款 KL 早停：最近 5 个 minibatch 滑窗均值超 1.5×target
            # 立即 break；PPOConfig.target_kl=0 禁用。原先 CombatPPOTrainerV2 配了
            # target_kl 但从没消费过，KL 爆炸时会白白跑完 4 epoch 把 policy 炸飞。
            epoch_kls: list[float] = []
            for start in range(0, len(samples), self.cfg.mini_batch_size):
                end = min(start + self.cfg.mini_batch_size, len(samples))
                batch_idx = indices[start:end]
                batch_samples = [samples[i] for i in batch_idx]

                batched = collate_training_samples(batch_samples, self.cfg.max_numeric_dim)
                enc_idx = getattr(batched, "encounter_indices", None)
                if enc_idx is not None:
                    enc_idx = enc_idx.to(device)
                output = self.net(batched_banks=batched.banks, encounter_idx=enc_idx)

                # 完整 multi-head loss（全部 target 传齐，避免 head 白训）
                loss, metrics = self.loss_fn(
                    output=output,
                    action_indices=batched.action_indices.to(device),
                    old_log_probs=batched.old_log_probs.to(device),
                    advantages=batched.advantages.to(device),
                    returns=batched.returns.to(device),
                    fight_win_targets=batched.fight_win_targets.to(device),
                    hp_loss_targets=batched.hp_loss_targets.to(device),
                    survival_targets=batched.survival_targets.to(device),
                    leaf_targets=batched.leaf_targets.to(device),
                    transition_risk_targets=(
                        batched.transition_risk_targets.to(device)
                        if batched.transition_risk_targets is not None else None),
                    resource_retention_targets=(
                        batched.resource_retention_targets.to(device)
                        if batched.resource_retention_targets is not None else None),
                    turn_damage_targets=(
                        batched.turn_damage_targets.to(device)
                        if batched.turn_damage_targets is not None else None),
                    sample_weights=batched.sample_weights.to(device),
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_skip_count += 1
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                # NaN 处理：原来整 batch 丢，导致一旦有 NaN 整轮 metrics 全空。
                # 改为 per-param nan_to_num（NaN→0），保留 batch；同时计数用于诊断。
                had_nan_grad = False
                for p in self.net.parameters():
                    if p.grad is None:
                        continue
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        had_nan_grad = True
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                if had_nan_grad:
                    nan_skip_count += 1  # 仍计数，但梯度已清零继续 step

                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                all_metrics.append(metrics)

                # Per-minibatch KL 早停
                kl = metrics.get("approx_kl", 0.0)
                if kl:
                    epoch_kls.append(float(kl))
                if self.cfg.target_kl > 0 and len(epoch_kls) >= 5:
                    recent_kl = sum(epoch_kls[-5:]) / 5
                    if recent_kl > 1.5 * self.cfg.target_kl:
                        break

            epochs_done += 1
            # Per-epoch KL 早停（双重保险）
            if self.cfg.target_kl > 0 and epoch_kls:
                mean_kl = sum(epoch_kls) / len(epoch_kls)
                if mean_kl > self.cfg.target_kl:
                    break

        if not all_metrics:
            return {
                "nan_skip_count": float(nan_skip_count),
                "epochs_done": float(epochs_done),
            }
        avg: dict[str, float] = {}
        all_keys: set[str] = set()
        for m in all_metrics:
            all_keys.update(m.keys())
        for key in all_keys:
            avg[key] = sum(m.get(key, 0.0) for m in all_metrics) / len(all_metrics)
        avg["nan_skip_count"] = float(nan_skip_count)
        avg["epochs_done"] = float(epochs_done)
        return avg


class UnifiedPPOTrainer:
    """统一训练器：混合 combat / non-combat 样本的 PPO 更新。

    按 sample 的 `banks.decision_domain` 把每个 mini-batch 拆成 combat / non-combat
    两个子批，分别 forward（UnifiedNet 按 decision_domain 走对应分支）并算 loss，
    两个子 loss 按样本数加权相加后反向。

    这样 non-combat 样本真的会流梯度到 `run_evaluator`；combat head 只接收 combat 样本；
    避免了 "full-run 跑 combat-only trainer 导致 non-combat 结构性失训" 的问题。
    """

    def __init__(
        self,
        net: UnifiedNet,
        config: PPOConfig | None = None,
        nc_loss_config: NonCombatLossConfig | None = None,
    ):
        self.net = net
        self.cfg = config or PPOConfig()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.cfg.lr)
        # 关键：advantages 在 train_step 入口做一次全局归一化，loss 内不再重算
        # （否则每 minibatch mean(adv)=0，ratio≈1 时 policy_loss=0）
        loss_cfg = self.cfg.loss_config or LossConfig()
        loss_cfg.normalize_adv = False
        self.combat_loss = CombatLoss(loss_cfg)
        nc_cfg = nc_loss_config or NonCombatLossConfig()
        nc_cfg.normalize_adv = False
        self.nc_loss = NonCombatLoss(nc_cfg)
        # 记录 train_step 调用次数，用于 value warmup
        self._step_count: int = 0
        # 保存原始 policy_coef，warmup 结束后恢复
        self._orig_combat_pc: float = self.combat_loss.cfg.policy_coef
        self._orig_nc_pc: float = self.nc_loss.cfg.policy_coef

    @staticmethod
    def _split_by_domain(samples: list[TrainingSample]) -> tuple[list[TrainingSample], list[TrainingSample]]:
        combat, noncombat = [], []
        for s in samples:
            if s.banks.decision_domain == "combat":
                combat.append(s)
            else:
                noncombat.append(s)
        return combat, noncombat

    def _combat_forward_loss(self, samples: list[TrainingSample], device) -> tuple[torch.Tensor, dict[str, float]]:
        batched = collate_training_samples(samples, self.cfg.max_numeric_dim)
        enc_idx = getattr(batched, "encounter_indices", None)
        if enc_idx is not None:
            enc_idx = enc_idx.to(device)
        output = self.net(batched_banks=batched.banks, decision_domain="combat", encounter_idx=enc_idx)
        return self.combat_loss(
            output=output,
            action_indices=batched.action_indices.to(device),
            old_log_probs=batched.old_log_probs.to(device),
            advantages=batched.advantages.to(device),
            returns=batched.returns.to(device),
            fight_win_targets=batched.fight_win_targets.to(device),
            hp_loss_targets=batched.hp_loss_targets.to(device),
            survival_targets=batched.survival_targets.to(device),
            leaf_targets=batched.leaf_targets.to(device),
            transition_risk_targets=batched.transition_risk_targets.to(device) if batched.transition_risk_targets is not None else None,
            resource_retention_targets=batched.resource_retention_targets.to(device) if batched.resource_retention_targets is not None else None,
            turn_damage_targets=batched.turn_damage_targets.to(device) if batched.turn_damage_targets is not None else None,
            sample_weights=batched.sample_weights.to(device),
        )

    def _noncombat_forward_loss(self, samples: list[TrainingSample], device) -> tuple[torch.Tensor, dict[str, float]]:
        batched = collate_training_samples(samples, self.cfg.max_numeric_dim)
        # Non-combat 分支对所有非战斗 domain 走同一计算图；output.decision_domain 只是标签
        # 用第一个样本的 domain 作为代表（label 不影响梯度）
        domain = samples[0].banks.decision_domain or "event"
        enc_idx = getattr(batched, "encounter_indices", None)
        if enc_idx is not None:
            enc_idx = enc_idx.to(device)
        output = self.net(batched_banks=batched.banks, decision_domain=domain, encounter_idx=enc_idx)
        # non-combat 样本暂用 fight_win_target 作为 run_win_target 的信号源：
        # full-run rollout 里终局时写 0/1 硬标签到 fight_win_target，其他为 -1（哨值）
        # 语义完全一致（整局胜率）
        return self.nc_loss(
            output=output,
            action_indices=batched.action_indices.to(device),
            old_log_probs=batched.old_log_probs.to(device),
            advantages=batched.advantages.to(device),
            returns=batched.returns.to(device),
            run_win_targets=batched.fight_win_targets.to(device),
            boss_readiness_targets=batched.boss_readiness_targets.to(device) if batched.boss_readiness_targets is not None else None,
            resource_health_targets=batched.resource_health_targets.to(device) if batched.resource_health_targets is not None else None,
            deck_quality_targets=batched.deck_quality_targets.to(device) if batched.deck_quality_targets is not None else None,
            sample_weights=batched.sample_weights.to(device),
        )

    def train_step(self, samples: list[TrainingSample]) -> dict[str, float]:
        if not samples:
            return {}

        # 全局归一化 advantages（每个 train_step 一次，不是每 minibatch；修 policy_loss=0 bug）
        # 按 domain 分别归一化：combat / non-combat 的 reward scale 不同，混在一起会被对方拉偏
        combat_pre, nc_pre = self._split_by_domain(samples)
        combat_pre = _globally_normalize_advantages(combat_pre)
        nc_pre = _globally_normalize_advantages(nc_pre)
        samples = combat_pre + nc_pre

        # --- Value warmup: 前 N 轮把 policy_coef 置 0，只让 value head 学 ---
        self._step_count += 1
        in_warmup = self._step_count <= self.cfg.value_warmup_iters
        if in_warmup:
            self.combat_loss.cfg.policy_coef = 0.0
            self.nc_loss.cfg.policy_coef = 0.0
        else:
            self.combat_loss.cfg.policy_coef = self._orig_combat_pc
            self.nc_loss.cfg.policy_coef = self._orig_nc_pc

        self.net.train()
        device = next(self.net.parameters()).device
        all_metrics: list[dict[str, float]] = []
        epochs_done = 0  # 实际完成的 epoch 数（含早停）
        nan_skip_count = 0  # 诊断：NaN 影响的 minibatch 数

        for _epoch in range(self.cfg.ppo_epochs):
            indices = torch.randperm(len(samples))
            # 收集本 epoch 各 minibatch 的 approx_kl，用于早停判断
            epoch_kls: list[float] = []
            for start in range(0, len(samples), self.cfg.mini_batch_size):
                end = min(start + self.cfg.mini_batch_size, len(samples))
                batch_idx = indices[start:end]
                mini = [samples[i] for i in batch_idx]
                combat_samples, nc_samples = self._split_by_domain(mini)

                n = max(len(mini), 1)
                total = torch.tensor(0.0, device=device)
                metrics: dict[str, float] = {}

                if combat_samples:
                    cl, cm = self._combat_forward_loss(combat_samples, device)
                    if torch.isfinite(cl):
                        total = total + cl * (len(combat_samples) / n)
                    metrics.update(cm)

                if nc_samples:
                    nl, nm = self._noncombat_forward_loss(nc_samples, device)
                    if torch.isfinite(nl):
                        total = total + nl * (len(nc_samples) / n)
                    metrics.update(nm)

                if not torch.isfinite(total) or total.requires_grad is False:
                    nan_skip_count += 1
                    continue

                self.optimizer.zero_grad()
                total.backward()
                # NaN 处理：原来"任意 grad NaN 就丢整 batch"会让 metrics 全空、
                # 上层报 policy_loss=0。改为 per-param nan_to_num，保留 batch 继续 step。
                had_nan_grad = False
                for p in self.net.parameters():
                    if p.grad is None:
                        continue
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        had_nan_grad = True
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                if had_nan_grad:
                    nan_skip_count += 1
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                # 收集 KL（warmup 时 policy 不更新也收集，用于观察）
                kl_c = metrics.get("approx_kl", 0.0)
                kl_nc = metrics.get("nc_approx_kl", 0.0)
                if kl_c or kl_nc:
                    # 混合 batch 时按样本数加权
                    kl_weighted = (kl_c * len(combat_samples) + kl_nc * len(nc_samples)) / n
                    epoch_kls.append(kl_weighted)

                metrics["combat_batch"] = float(len(combat_samples))
                metrics["noncombat_batch"] = float(len(nc_samples))
                metrics["total_loss"] = float(total.item())
                all_metrics.append(metrics)

                # Per-minibatch KL 早停：原来等整个 epoch 跑完才检查，KL 已经飞到 5.0+。
                # 改为最近 N=5 个 minibatch 滑窗均值超 1.5×target 立即 break。
                # 在 epoch 内、warmup 之外触发，让 policy 不至于在单个 epoch 里跑飞。
                if not in_warmup and self.cfg.target_kl > 0 and len(epoch_kls) >= 5:
                    recent_kl = sum(epoch_kls[-5:]) / 5
                    if recent_kl > 1.5 * self.cfg.target_kl:
                        break

            epochs_done += 1
            # 跨 epoch 早停（双重保险）：本 epoch 平均 KL 超阈值就停止剩余 epoch
            if not in_warmup and self.cfg.target_kl > 0 and epoch_kls:
                mean_kl = sum(epoch_kls) / len(epoch_kls)
                if mean_kl > self.cfg.target_kl:
                    break

        if not all_metrics:
            # 全部 minibatch 被跳过 → 暴露 nan_skip_count，便于上层定位"假 0"
            return {
                "warmup": float(in_warmup),
                "epochs_done": float(epochs_done),
                "nan_skip_count": float(nan_skip_count),
            }
        # 对所有 minibatch 可能出现的 key 求平均（不只是第一个 batch 的 key）
        all_keys: set[str] = set()
        for m in all_metrics:
            all_keys.update(m.keys())
        avg: dict[str, float] = {}
        for key in all_keys:
            avg[key] = sum(m.get(key, 0.0) for m in all_metrics) / len(all_metrics)
        avg["warmup"] = float(in_warmup)
        avg["epochs_done"] = float(epochs_done)
        avg["nan_skip_count"] = float(nan_skip_count)
        return avg
