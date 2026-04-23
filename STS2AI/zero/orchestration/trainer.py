from __future__ import annotations

import json
import math
import os
import random
import tempfile
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..buffers import SamplePoolSet
from ..config import LossWeights, TrainConfig
from ..domain import BattleState, HistoryStep, TrainingSummary
from ..features import BatchCollator, FeatureExtractor, compute_transition_delta
from ..model import (
    HierarchicalPolicyOutput,
    ZeroNet,
    compute_losses,
    compute_policy_value_alignment_loss,
    is_hierarchical_output,
    select_action_logits,
    select_action_value,
    select_confirm_logit,
    select_future_summary,
)
from ..ports import Policy


@dataclass(slots=True)
class _PpoLiteLosses:
    policy: torch.Tensor
    policy_align: torch.Tensor
    value: torch.Tensor
    delta: torch.Tensor
    future_summary: torch.Tensor
    policy_entropy: torch.Tensor
    total: torch.Tensor


@dataclass(slots=True)
class _RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1)
        for value_tensor in flat:
            value = float(value_tensor.item())
            self.count += 1
            delta = value - self.mean
            self.mean += delta / float(self.count)
            delta2 = value - self.mean
            self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 1.0
        return max(self.m2 / float(self.count - 1), 1e-6)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if self.count <= 1:
            return values
        return (values - self.mean) / self.std

    def to_dict(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "mean": float(self.mean),
            "m2": float(self.m2),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "_RunningMoments":
        if not payload:
            return cls()
        return cls(
            count=int(payload.get("count") or 0),
            mean=float(payload.get("mean") or 0.0),
            m2=float(payload.get("m2") or 0.0),
        )


def _compute_ppo_lite_losses(
    output,
    batch,
    config: TrainConfig,
    *,
    step_return_stats: _RunningMoments,
    turn_return_stats: _RunningMoments,
) -> _PpoLiteLosses:
    active_intent = batch.active_intent if is_hierarchical_output(output) else None
    action_logits = select_action_logits(output, active_intent)
    action_value = select_action_value(output, active_intent)
    safe_logits = action_logits.masked_fill(batch.action_mask <= 0, -float(torch.finfo(action_logits.dtype).max))
    log_probs = F.log_softmax(safe_logits, dim=-1)
    probs = torch.softmax(safe_logits, dim=-1)
    chosen_logprob = log_probs.gather(1, batch.behavior_action_index.unsqueeze(1)).squeeze(1)
    advantage = batch.ppo_advantage
    if bool(config.ppo_advantage_norm) and advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    ratio = torch.exp(chosen_logprob - batch.old_logprob)
    clipped_ratio = torch.clamp(ratio, 1.0 - config.ppo_clip_ratio, 1.0 + config.ppo_clip_ratio)
    ppo_policy = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
    imitation_loss = F.cross_entropy(
        safe_logits,
        batch.behavior_action_index,
        reduction="none",
    )
    behavior_policy = (imitation_loss * batch.sample_weight * batch.behavior_ce_scale).mean()
    submenu_confirm_loss = _compute_submenu_confirm_loss(
        select_confirm_logit(output, active_intent),
        batch,
    )
    policy = ppo_policy + config.ppo_behavior_ce_coef * behavior_policy + config.ppo_submenu_confirm_coef * submenu_confirm_loss

    chosen_action_value = _gather_action_scalar(action_value, batch.behavior_action_index)
    step_target = batch.ppo_return
    old_step_value = batch.old_value
    if bool(config.normalize_step_returns):
        step_mean = step_return_stats.mean
        step_std = step_return_stats.std
        step_target = (batch.ppo_return - step_mean) / step_std
        old_step_value = (batch.old_value - step_mean) / step_std
        state_value_pred = (output.state_value - step_mean) / step_std
        chosen_action_value_pred = (chosen_action_value - step_mean) / step_std
    else:
        state_value_pred = output.state_value
        chosen_action_value_pred = chosen_action_value
    state_value_loss = _compute_clipped_value_loss(
        prediction=state_value_pred,
        target=step_target,
        old_prediction=old_step_value,
        clip_range=config.ppo_value_clip,
    )
    action_value_loss = F.mse_loss(chosen_action_value_pred, step_target)
    value = state_value_loss + 0.5 * action_value_loss

    intent_entropy = torch.zeros((), device=safe_logits.device)
    if isinstance(output, HierarchicalPolicyOutput):
        turn_mask = batch.turn_start_mask > 0.5
        if bool(turn_mask.any().item()):
            intent_logits = output.intent_logits[turn_mask]
            turn_log_probs = F.log_softmax(intent_logits, dim=-1)
            chosen_intent_logprob = turn_log_probs.gather(
                1,
                batch.active_intent[turn_mask].unsqueeze(1),
            ).squeeze(1)
            intent_advantage = batch.turn_advantage[turn_mask]
            if bool(config.ppo_advantage_norm) and intent_advantage.numel() > 1:
                intent_advantage = (intent_advantage - intent_advantage.mean()) / intent_advantage.std(unbiased=False).clamp_min(1e-6)
            intent_ratio = torch.exp(chosen_intent_logprob - batch.old_intent_logprob[turn_mask])
            clipped_intent_ratio = torch.clamp(
                intent_ratio,
                1.0 - config.ppo_clip_ratio,
                1.0 + config.ppo_clip_ratio,
            )
            high_level_policy = -torch.min(
                intent_ratio * intent_advantage,
                clipped_intent_ratio * intent_advantage,
            ).mean()
            turn_target = batch.turn_return[turn_mask]
            if bool(config.normalize_turn_returns):
                turn_mean = turn_return_stats.mean
                turn_std = turn_return_stats.std
                turn_target = (turn_target - turn_mean) / turn_std
                intent_value_pred = (output.intent_value[turn_mask] - turn_mean) / turn_std
                old_intent_value = (batch.old_intent_value[turn_mask] - turn_mean) / turn_std
            else:
                intent_value_pred = output.intent_value[turn_mask]
                old_intent_value = batch.old_intent_value[turn_mask]
            intent_value_loss = _compute_clipped_value_loss(
                prediction=intent_value_pred,
                target=turn_target,
                old_prediction=old_intent_value,
                clip_range=config.ppo_value_clip,
            )
            intent_probs = torch.softmax(intent_logits, dim=-1)
            intent_entropy = -(intent_probs * turn_log_probs).sum(dim=-1).mean()
            policy = policy + config.ppo_turn_intent_coef * high_level_policy
            value = value + config.ppo_turn_value_coef * intent_value_loss

    policy_align = compute_policy_value_alignment_loss(
        safe_logits,
        action_value,
        batch.action_mask,
        temperature=config.ppo_policy_align_temperature,
    )
    future_summary = F.mse_loss(
        select_future_summary(output, active_intent),
        batch.chosen_action_future_targets,
    )
    action_entropy = -(probs * log_probs).sum(dim=-1).mean()
    delta = torch.zeros((), device=safe_logits.device)

    total = (
        policy
        + config.ppo_policy_align_coef * policy_align
        + config.ppo_value_coef * value
        + config.ppo_future_summary_coef * future_summary
        - config.ppo_action_entropy_coef * action_entropy
        - config.ppo_intent_entropy_coef * intent_entropy
    )
    return _PpoLiteLosses(
        policy=policy,
        policy_align=policy_align,
        value=value,
        delta=delta,
        future_summary=future_summary,
        policy_entropy=action_entropy + intent_entropy,
        total=total,
    )


def _compute_submenu_confirm_loss(confirm_logit: torch.Tensor, batch) -> torch.Tensor:
    target_mask = batch.submenu_has_confirm > 0.5
    if not bool(target_mask.any().item()):
        return torch.zeros((), device=confirm_logit.device)
    submenu_loss = F.binary_cross_entropy_with_logits(
        confirm_logit[target_mask],
        batch.submenu_confirm_target[target_mask],
        reduction="none",
    )
    weights = batch.sample_weight[target_mask].clamp_min(0.1)
    return (submenu_loss * weights).mean()


def _gather_action_scalar(values: torch.Tensor, action_index: torch.Tensor) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"expected [batch, actions] action values, got shape={tuple(values.shape)}")
    return values.gather(1, action_index.unsqueeze(1)).squeeze(1)

def _compute_clipped_value_loss(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    old_prediction: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    if clip_range <= 0:
        return F.mse_loss(prediction, target)
    clipped_prediction = old_prediction + (prediction - old_prediction).clamp(-clip_range, clip_range)
    unclipped = (prediction - target) ** 2
    clipped = (clipped_prediction - target) ** 2
    return torch.max(unclipped, clipped).mean()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _is_amp_enabled(config: TrainConfig, device: torch.device) -> bool:
    return bool(config.amp_enabled and device.type == "cuda")


class ZeroTrainer:
    def __init__(self, model: ZeroNet, config: TrainConfig, loss_weights: LossWeights, collator: BatchCollator):
        self._model = model
        self._config = config
        self._loss_weights = loss_weights
        self._collator = collator
        self._device = _resolve_device(config.device)
        self._amp_enabled = _is_amp_enabled(config, self._device)
        self._model.to(self._device)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._global_step = 0
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=self._lr_multiplier)
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
        self._step_return_stats = _RunningMoments()
        self._turn_return_stats = _RunningMoments()

    @property
    def model(self) -> ZeroNet:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def state_dict(self) -> dict[str, Any]:
        scaler_state = self._scaler.state_dict() if self._amp_enabled else None
        return {
            "optimizer_state": self._optimizer.state_dict(),
            "scheduler_state": self._scheduler.state_dict(),
            "scaler_state": scaler_state,
            "global_step": self._global_step,
            "device": str(self._device),
            "step_return_stats": self._step_return_stats.to_dict(),
            "turn_return_stats": self._turn_return_stats.to_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            self._optimizer.load_state_dict(optimizer_state)
            for state in self._optimizer.state.values():
                for key, value in list(state.items()):
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self._device)
        scheduler_state = payload.get("scheduler_state")
        if scheduler_state:
            self._scheduler.load_state_dict(scheduler_state)
        scaler_state = payload.get("scaler_state")
        if scaler_state and self._amp_enabled:
            self._scaler.load_state_dict(scaler_state)
        self._global_step = int(payload.get("global_step") or 0)
        self._step_return_stats = _RunningMoments.from_dict(payload.get("step_return_stats"))
        self._turn_return_stats = _RunningMoments.from_dict(payload.get("turn_return_stats"))

    def train_iteration(self, pools: SamplePoolSet) -> TrainingSummary:
        if self._config.algorithm == "ppo_lite":
            return self._train_iteration_ppo_lite(pools)
        self._model.train()
        metrics = TrainingSummary()
        prefetched = deque(self._prefetch_samples(pools))
        for _ in range(self._config.steps_per_iteration):
            if not prefetched:
                prefetched.extend(self._prefetch_samples(pools))
            if not prefetched:
                break
            samples = prefetched.popleft()
            if not samples:
                continue
            while len(prefetched) < max(1, self._config.prefetch_batches - 1):
                extra = self._sample_batch(pools)
                if not extra:
                    break
                prefetched.append(extra)

            batch = self._collator.collate(samples).to(self._device)
            pool_counts = Counter(sample.pool_name for sample in samples)
            self._optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self._device.type, enabled=self._amp_enabled):
                output = self._model(batch)
                losses = (
                    _compute_ppo_lite_losses(output, batch, self._config)
                    if self._config.algorithm == "ppo_lite"
                    else compute_losses(output, batch, self._loss_weights)
                )

            if not torch.isfinite(losses.total):
                metrics.skipped_non_finite_steps += 1
                continue

            if self._amp_enabled:
                self._scaler.scale(losses.total).backward()
                self._scaler.unscale_(self._optimizer)
            else:
                losses.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.grad_clip_norm)
            if not math.isfinite(float(grad_norm)):
                metrics.skipped_non_finite_steps += 1
                self._optimizer.zero_grad(set_to_none=True)
                if self._amp_enabled:
                    self._scaler.update()
                continue

            if self._amp_enabled:
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                self._optimizer.step()
            self._scheduler.step()
            self._global_step += 1

            metrics.steps += 1
            metrics.policy_loss += float(losses.policy.detach())
            metrics.policy_align_loss += float(losses.policy_align.detach())
            metrics.value_loss += float(losses.value.detach())
            metrics.delta_loss += float(losses.delta.detach())
            metrics.future_summary_loss += float(losses.future_summary.detach())
            metrics.policy_entropy += float(losses.policy_entropy.detach())
            metrics.total_loss += float(losses.total.detach())
            metrics.grad_norm += float(grad_norm)
            metrics.learning_rate += float(self._optimizer.param_groups[0]["lr"])
            for pool_name, count in pool_counts.items():
                metrics.pool_usage[pool_name] = metrics.pool_usage.get(pool_name, 0) + int(count)

        if metrics.steps > 0:
            metrics.policy_loss /= metrics.steps
            metrics.policy_align_loss /= metrics.steps
            metrics.value_loss /= metrics.steps
            metrics.delta_loss /= metrics.steps
            metrics.future_summary_loss /= metrics.steps
            metrics.policy_entropy /= metrics.steps
            metrics.total_loss /= metrics.steps
            metrics.grad_norm /= metrics.steps
            metrics.learning_rate /= metrics.steps
        else:
            metrics.zero_step = True
        return metrics

    def _train_iteration_ppo_lite(self, pools: SamplePoolSet) -> TrainingSummary:
        self._model.train()
        metrics = TrainingSummary()
        dataset = list(pools.pool_items("recent_online"))
        if not dataset:
            metrics.zero_step = True
            return metrics

        batch_size = max(1, int(self._config.batch_size))
        max_steps = max(1, int(self._config.steps_per_iteration))
        max_epochs = max(1, int(self._config.ppo_epochs))

        for _epoch in range(max_epochs):
            if metrics.steps >= max_steps:
                break
            random.shuffle(dataset)
            for start in range(0, len(dataset), batch_size):
                if metrics.steps >= max_steps:
                    break
                samples = dataset[start : start + batch_size]
                if not samples:
                    continue

                batch = self._collator.collate(samples).to(self._device)
                self._step_return_stats.update(batch.ppo_return)
                turn_mask = batch.turn_start_mask > 0.5
                if bool(turn_mask.any().item()):
                    self._turn_return_stats.update(batch.turn_return[turn_mask])
                pool_counts = Counter(sample.pool_name for sample in samples)
                self._optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self._device.type, enabled=self._amp_enabled):
                    output = self._model(batch)
                    losses = _compute_ppo_lite_losses(
                        output,
                        batch,
                        self._config,
                        step_return_stats=self._step_return_stats,
                        turn_return_stats=self._turn_return_stats,
                    )

                if not torch.isfinite(losses.total):
                    metrics.skipped_non_finite_steps += 1
                    continue

                if self._amp_enabled:
                    self._scaler.scale(losses.total).backward()
                    self._scaler.unscale_(self._optimizer)
                else:
                    losses.total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.grad_clip_norm)
                if not math.isfinite(float(grad_norm)):
                    metrics.skipped_non_finite_steps += 1
                    self._optimizer.zero_grad(set_to_none=True)
                    if self._amp_enabled:
                        self._scaler.update()
                    continue

                if self._amp_enabled:
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    self._optimizer.step()
                self._scheduler.step()
                self._global_step += 1

                metrics.steps += 1
                metrics.policy_loss += float(losses.policy.detach())
                metrics.policy_align_loss += float(losses.policy_align.detach())
                metrics.value_loss += float(losses.value.detach())
                metrics.delta_loss += float(losses.delta.detach())
                metrics.future_summary_loss += float(losses.future_summary.detach())
                metrics.policy_entropy += float(losses.policy_entropy.detach())
                metrics.total_loss += float(losses.total.detach())
                metrics.grad_norm += float(grad_norm)
                metrics.learning_rate += float(self._optimizer.param_groups[0]["lr"])
                for pool_name, count in pool_counts.items():
                    metrics.pool_usage[pool_name] = metrics.pool_usage.get(pool_name, 0) + int(count)

        if metrics.steps > 0:
            metrics.policy_loss /= metrics.steps
            metrics.policy_align_loss /= metrics.steps
            metrics.value_loss /= metrics.steps
            metrics.delta_loss /= metrics.steps
            metrics.future_summary_loss /= metrics.steps
            metrics.policy_entropy /= metrics.steps
            metrics.total_loss /= metrics.steps
            metrics.grad_norm /= metrics.steps
            metrics.learning_rate /= metrics.steps
        else:
            metrics.zero_step = True
        return metrics

    def _sample_batch(self, pools: SamplePoolSet) -> list:
        return pools.mixed_sample(self._config.batch_size)

    def _prefetch_samples(self, pools: SamplePoolSet) -> list[list]:
        batches: list[list] = []
        for _ in range(max(1, self._config.prefetch_batches)):
            samples = self._sample_batch(pools)
            if not samples:
                break
            batches.append(samples)
        return batches

    def _lr_multiplier(self, step: int) -> float:
        if self._config.algorithm == "ppo_lite":
            return 1.0
        warmup_steps = max(0, int(self._config.warmup_steps))
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-6, float(step + 1) / float(warmup_steps))
        adjusted = max(step - warmup_steps, 0) + 1
        decay = 1.0 / math.sqrt(float(adjusted))
        return max(float(self._config.min_lr_ratio), decay)


@dataclass(slots=True)
class _InferenceRequest:
    state: BattleState
    history: list[HistoryStep]
    created_at: float
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BaseException | None = None


class _BatchedInferenceCoordinator:
    def __init__(self, model: ZeroNet, collator: BatchCollator):
        self._model = model
        self._collator = collator
        self._device = next(model.parameters()).device
        self._window_s = max(0.0, float(collator._config.microbatch_window_ms)) / 1000.0
        self._requests: list[_InferenceRequest] = []
        self._condition = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(target=self._run_loop, name="zero-batched-infer", daemon=True)
        self._worker.start()

    def submit(self, state: BattleState, history: list[HistoryStep]) -> dict[str, Any]:
        request = _InferenceRequest(state=state, history=history, created_at=time.perf_counter())
        with self._condition:
            if self._closed:
                raise RuntimeError("batched inference coordinator 已关闭")
            self._requests.append(request)
            self._condition.notify()
        request.done.wait()
        if request.error is not None:
            raise request.error
        return dict(request.result or {})

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=1.0)

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while not self._requests and not self._closed:
                    self._condition.wait()
                if self._closed and not self._requests:
                    return
                batch_requests = [self._requests.pop(0)]
                deadline = time.perf_counter() + self._window_s
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    if not self._requests:
                        self._condition.wait(timeout=remaining)
                    if self._requests:
                        batch_requests.extend(self._requests)
                        self._requests = []
            self._execute_batch(batch_requests)

    def _execute_batch(self, requests: list[_InferenceRequest]) -> None:
        try:
            self._model.eval()
            collate_started_at = time.perf_counter()
            batch = self._collator.collate_inference_batch(
                [(request.state, request.history, request.state.legal_actions) for request in requests]
            ).to(self._device)
            policy_collate_duration_s = time.perf_counter() - collate_started_at
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self._device.type,
                    enabled=bool(self._device.type == "cuda"),
                ):
                    intent_started_at = time.perf_counter()
                    encoded_state = self._model.encode_state(batch)
                    intent_forward_duration_s = time.perf_counter() - intent_started_at
                    action_started_at = time.perf_counter()
                    action_logits, action_value = self._model.score_actions(batch, encoded_state)
                    action_forward_duration_s = time.perf_counter() - action_started_at
            policy_forward_duration_s = intent_forward_duration_s + action_forward_duration_s
            action_logits_cpu = action_logits.float().cpu()
            action_value_cpu = action_value.float().cpu()
            action_mask_cpu = batch.action_mask.float().cpu()
            state_value_cpu = encoded_state.state_value.float().cpu()
            flat_future_cpu = (
                encoded_state.future_summary.float().cpu()
                if encoded_state.future_summary is not None
                else None
            )
            flat_confirm_cpu = (
                encoded_state.confirm_now_logit.float().cpu()
                if encoded_state.confirm_now_logit is not None
                else None
            )
            intent_logits_cpu = (
                encoded_state.intent_logits.float().cpu()
                if encoded_state.intent_logits is not None
                else None
            )
            intent_value_cpu = (
                encoded_state.intent_value.float().cpu()
                if encoded_state.intent_value is not None
                else None
            )
            future_by_intent_cpu = (
                encoded_state.future_summary_by_intent.float().cpu()
                if encoded_state.future_summary_by_intent is not None
                else None
            )
            confirm_by_intent_cpu = (
                encoded_state.confirm_now_logit_by_intent.float().cpu()
                if encoded_state.confirm_now_logit_by_intent is not None
                else None
            )
            for row, request in enumerate(requests):
                valid = int(action_mask_cpu[row].sum().item())
                common_result = {
                    "policy_arch": self._model.policy_arch,
                    "state_value": float(state_value_cpu[row].item()),
                    "policy_collate_duration_s": float(policy_collate_duration_s / max(len(requests), 1)),
                    "intent_forward_duration_s": float(intent_forward_duration_s / max(len(requests), 1)),
                    "action_forward_duration_s": float(action_forward_duration_s / max(len(requests), 1)),
                    "policy_forward_duration_s": float(policy_forward_duration_s / max(len(requests), 1)),
                    "policy_postprocess_duration_s": 0.0,
                    "batch_wait_duration_s": max(0.0, float(collate_started_at - request.created_at)),
                    "batch_size_effective": int(len(requests)),
                }
                if self._model.policy_arch == "flat":
                    request.result = {
                        **common_result,
                        "action_logits": action_logits_cpu[row, :valid].contiguous(),
                        "action_values": action_value_cpu[row, :valid].contiguous(),
                        "future_summary": (
                            flat_future_cpu[row].contiguous()
                            if flat_future_cpu is not None
                            else torch.zeros(3, dtype=torch.float32)
                        ),
                        "confirm_now_logit": float(flat_confirm_cpu[row].item()) if flat_confirm_cpu is not None else 0.0,
                    }
                else:
                    if intent_logits_cpu is None or intent_value_cpu is None or future_by_intent_cpu is None or confirm_by_intent_cpu is None:
                        raise RuntimeError("hierarchical batched inference 缺少 intent 分支输出。")
                    intent_logits_row = intent_logits_cpu[row].contiguous()
                    if intent_logits_row.numel() > 0:
                        intent_log_probs_row = torch.log_softmax(intent_logits_row, dim=0)
                    else:
                        intent_log_probs_row = torch.zeros_like(intent_logits_row)
                    action_logits_row = action_logits_cpu[row, :, :valid].contiguous()
                    action_value_row = action_value_cpu[row, :, :valid].contiguous()
                    if action_logits_row.numel() > 0:
                        action_index_by_intent_row = action_logits_row.argmax(dim=1)
                    else:
                        action_index_by_intent_row = torch.zeros(
                            action_logits_cpu.size(1),
                            dtype=torch.long,
                        )
                    request.result = {
                        **common_result,
                        "intent_logits": intent_logits_row,
                        "intent_log_probs": intent_log_probs_row,
                        "intent_value": float(intent_value_cpu[row].item()),
                        "future_by_intent": future_by_intent_cpu[row].contiguous(),
                        "confirm_now_by_intent": confirm_by_intent_cpu[row].contiguous(),
                        "action_logits_by_intent": action_logits_row,
                        "action_values_by_intent": action_value_row,
                        "action_index_by_intent": action_index_by_intent_row,
                    }
                request.done.set()
        except BaseException as exc:  # pragma: no cover - failure fan-out
            for request in requests:
                request.error = exc
                request.done.set()


class ModelPolicyAdapter(Policy):
    INTENT_LABELS = ("burst", "survive", "setup", "dig")
    _REPLAN_EXIT_ACTION_TYPES = {"confirm_selection", "combat_confirm_selection", "cancel_selection"}

    def __init__(
        self,
        model: ZeroNet,
        collator: BatchCollator,
        history_steps: int,
        *,
        batcher: _BatchedInferenceCoordinator | None = None,
    ):
        self._model = model
        self._collator = collator
        self._history_extractor = FeatureExtractor(collator._config)
        self._history: deque = deque(maxlen=history_steps)
        self._device = next(model.parameters()).device
        self._cached_state_ref = None
        self._cached_inference: dict[str, Any] | None = None
        self._current_turn_id: int | None = None
        self._active_intent: int = 0
        self._batcher = batcher or _BatchedInferenceCoordinator(model, collator)
        self._force_replan: bool = False
        self._fallback_turn_id: int = 1

    def reset_episode(self) -> None:
        self._history.clear()
        self._cached_state_ref = None
        self._cached_inference = None
        self._current_turn_id = None
        self._active_intent = 0
        self._force_replan = False
        self._fallback_turn_id = 1

    def observe_intent_choice(self, state: BattleState, intent_index: int) -> None:
        self._current_turn_id = _state_turn_id(state, fallback=self._fallback_turn_id)
        self._active_intent = int(intent_index)
        self._cached_state_ref = None
        self._cached_inference = None
        self._force_replan = False

    def observe_transition(self, state: BattleState, action_index: int, next_state: BattleState) -> None:
        if not (0 <= action_index < len(state.legal_actions)):
            return
        chosen_action = state.legal_actions[action_index]
        delta = compute_transition_delta(state, next_state)
        self._history.append(
            HistoryStep(
                state=None,
                action=None,
                delta=delta,
                history_token=self._history_extractor.encode_history_step_token(
                    state,
                    chosen_action,
                    delta,
                ),
            )
        )
        explicit_next_turn = _extract_explicit_turn_id(next_state)
        if explicit_next_turn is not None:
            self._fallback_turn_id = explicit_next_turn
        elif str(chosen_action.action_type or "").lower() in {"end_turn", *self._REPLAN_EXIT_ACTION_TYPES}:
            self._fallback_turn_id += 1
        self._force_replan = self._should_replan(state, chosen_action, next_state)
        self._cached_state_ref = None
        self._cached_inference = None

    def _should_replan(self, state: BattleState, chosen_action, next_state: BattleState) -> bool:
        if self._model.policy_arch != "hierarchical_intent":
            return False
        action_type = str(chosen_action.action_type or "").strip().lower()
        if action_type in self._REPLAN_EXIT_ACTION_TYPES:
            return True
        if abs(len(next_state.hand) - len(state.hand)) >= 2:
            return True
        if bool(state.player.energy > 0) != bool(next_state.player.energy > 0):
            return True
        if len(next_state.living_enemies) != len(state.living_enemies):
            return True
        prev_state_type = str((state.context.metadata or {}).get("state_type", "") or "").lower()
        next_state_type = str((next_state.context.metadata or {}).get("state_type", "") or "").lower()
        if prev_state_type != next_state_type and {prev_state_type, next_state_type} & {"hand_select", "card_select"}:
            return True
        return False

    def infer(self, state: BattleState) -> dict[str, Any]:
        if self._cached_state_ref is state and self._cached_inference is not None:
            return self._cached_inference
        result = self.evaluate_state(state, list(self._history))
        self._cached_state_ref = state
        self._cached_inference = result
        return result

    def evaluate_state(self, state: BattleState, history: list[HistoryStep]) -> dict[str, Any]:
        if not state.legal_actions:
            return {
                "scores": [],
                "action_index": 0,
                "action_values": [],
                "intent_scores": [],
                "action_scores_by_intent": [],
                "action_values_by_intent": [],
                "action_index_by_intent": [],
                "is_turn_start": True,
                "turn_id": 0,
                "turn_id_fallback_used": False,
                "active_intent": 0,
                "active_intent_label": "flat" if self._model.policy_arch == "flat" else self.INTENT_LABELS[0],
                "old_intent_logprob": 0.0,
                "intent_value": 0.0,
                "death_risk_2t": 0.0,
                "next_turn_power": 0.0,
                "setup_value": 0.0,
                "ppo_value": 0.0,
                "policy_collate_duration_s": 0.0,
                "intent_forward_duration_s": 0.0,
                "action_forward_duration_s": 0.0,
                "policy_forward_duration_s": 0.0,
                "policy_postprocess_duration_s": 0.0,
                "batch_wait_duration_s": 0.0,
                "batch_size_effective": 1,
            }
        postprocess_started_at = time.perf_counter()
        raw = self._batcher.submit(state, history)
        turn_id = _state_turn_id(state, fallback=self._fallback_turn_id)
        is_turn_start = self._current_turn_id != turn_id or self._force_replan
        policy_arch = str(raw.get("policy_arch") or self._model.policy_arch)
        confirm_index = _find_confirm_action_index(state)
        policy_postprocess_duration_s = time.perf_counter() - postprocess_started_at
        if policy_arch == "flat":
            action_scores = _tensor_to_python_list(raw.get("action_logits"))
            action_values = _tensor_to_python_list(raw.get("action_values"))
            confirm_now_logit = float(raw.get("confirm_now_logit", 0.0) or 0.0)
            if confirm_index is not None and confirm_index < len(action_scores):
                action_scores[confirm_index] += confirm_now_logit
            action_index = _argmax_index(action_scores)
            future_summary = _tensor_to_python_list(raw.get("future_summary"))
            return {
                "scores": action_scores,
                "action_index": int(action_index),
                "action_values": action_values,
                "intent_scores": [],
                "action_scores_by_intent": [],
                "action_values_by_intent": [],
                "action_index_by_intent": [],
                "is_turn_start": True,
                "turn_id": int(turn_id),
                "turn_id_fallback_used": bool(_extract_explicit_turn_id(state) is None),
                "active_intent": 0,
                "active_intent_label": "flat",
                "old_intent_logprob": 0.0,
                "intent_value": 0.0,
                "death_risk_2t": float(future_summary[0] if len(future_summary) > 0 else 0.0),
                "next_turn_power": float(future_summary[1] if len(future_summary) > 1 else 0.0),
                "setup_value": float(future_summary[2] if len(future_summary) > 2 else 0.0),
                "ppo_value": float(raw.get("state_value", 0.0) or 0.0),
                "policy_collate_duration_s": float(raw.get("policy_collate_duration_s", 0.0) or 0.0),
                "intent_forward_duration_s": float(raw.get("intent_forward_duration_s", 0.0) or 0.0),
                "action_forward_duration_s": float(raw.get("action_forward_duration_s", 0.0) or 0.0),
                "policy_forward_duration_s": float(raw.get("policy_forward_duration_s", 0.0) or 0.0),
                "policy_postprocess_duration_s": float(
                    (raw.get("policy_postprocess_duration_s", 0.0) or 0.0) + policy_postprocess_duration_s
                ),
                "batch_wait_duration_s": float(raw.get("batch_wait_duration_s", 0.0) or 0.0),
                "batch_size_effective": int(raw.get("batch_size_effective", 1) or 1),
            }

        intent_scores = raw.get("intent_logits")
        if not isinstance(intent_scores, torch.Tensor):
            intent_scores = torch.tensor(intent_scores or [], dtype=torch.float32)
        else:
            intent_scores = intent_scores.to(dtype=torch.float32)
        intent_log_probs = raw.get("intent_log_probs")
        if not isinstance(intent_log_probs, torch.Tensor):
            intent_log_probs = torch.tensor(intent_log_probs or [], dtype=torch.float32)
        else:
            intent_log_probs = intent_log_probs.to(dtype=torch.float32)
        action_scores_by_intent = raw.get("action_logits_by_intent")
        if not isinstance(action_scores_by_intent, torch.Tensor):
            action_scores_by_intent = torch.tensor(action_scores_by_intent or [], dtype=torch.float32)
        else:
            action_scores_by_intent = action_scores_by_intent.to(dtype=torch.float32)
        action_values_by_intent = raw.get("action_values_by_intent")
        if not isinstance(action_values_by_intent, torch.Tensor):
            action_values_by_intent = torch.tensor(action_values_by_intent or [], dtype=torch.float32)
        else:
            action_values_by_intent = action_values_by_intent.to(dtype=torch.float32)
        future_by_intent = raw.get("future_by_intent")
        if not isinstance(future_by_intent, torch.Tensor):
            future_by_intent = torch.tensor(future_by_intent or [], dtype=torch.float32)
        else:
            future_by_intent = future_by_intent.to(dtype=torch.float32)
        confirm_now_by_intent = raw.get("confirm_now_by_intent")
        if not isinstance(confirm_now_by_intent, torch.Tensor):
            confirm_now_by_intent = torch.tensor(confirm_now_by_intent or [], dtype=torch.float32)
        else:
            confirm_now_by_intent = confirm_now_by_intent.to(dtype=torch.float32)

        active_intent = int(intent_scores.argmax().item()) if is_turn_start and intent_scores.numel() > 0 else self._active_intent
        active_intent = max(0, min(active_intent, max(intent_scores.numel() - 1, 0)))
        action_scores_by_intent_python = []
        action_values_by_intent_python = []
        action_index_by_intent_python = []
        for intent_index in range(action_scores_by_intent.size(0)):
            scores = _sanitize_score_list(action_scores_by_intent[intent_index].tolist())
            if confirm_index is not None and confirm_index < len(scores):
                scores[confirm_index] += float(confirm_now_by_intent[intent_index].item()) if confirm_now_by_intent.numel() > intent_index else 0.0
            action_scores_by_intent_python.append(_sanitize_score_list(scores))
            values = action_values_by_intent[intent_index].tolist() if action_values_by_intent.ndim >= 2 else []
            action_values_by_intent_python.append([float(value) for value in values])
            action_index_by_intent_python.append(_argmax_index(scores))
        scores = (
            action_scores_by_intent_python[active_intent]
            if 0 <= active_intent < len(action_scores_by_intent_python)
            else []
        )
        action_values = (
            action_values_by_intent_python[active_intent]
            if 0 <= active_intent < len(action_values_by_intent_python)
            else []
        )
        action_index = (
            action_index_by_intent_python[active_intent]
            if 0 <= active_intent < len(action_index_by_intent_python)
            else 0
        )
        old_intent_logprob = float(intent_log_probs[active_intent].item()) if intent_log_probs.numel() > active_intent else 0.0
        future_row = (
            future_by_intent[active_intent].tolist()
            if future_by_intent.ndim >= 2 and future_by_intent.size(0) > active_intent
            else [0.0, 0.0, 0.0]
        )
        return {
            "scores": _sanitize_score_list(scores),
            "action_index": int(action_index),
            "action_values": [float(value) for value in action_values],
            "intent_scores": [float(value) for value in intent_scores.tolist()],
            "action_scores_by_intent": action_scores_by_intent_python,
            "action_values_by_intent": action_values_by_intent_python,
            "action_index_by_intent": action_index_by_intent_python,
            "is_turn_start": bool(is_turn_start),
            "turn_id": int(turn_id),
            "turn_id_fallback_used": bool(_extract_explicit_turn_id(state) is None),
            "active_intent": int(active_intent),
            "active_intent_label": self.INTENT_LABELS[active_intent] if active_intent < len(self.INTENT_LABELS) else str(active_intent),
            "old_intent_logprob": float(old_intent_logprob),
            "intent_value": float(raw.get("intent_value", 0.0) or 0.0),
            "death_risk_2t": float(future_row[0] if len(future_row) > 0 else 0.0),
            "next_turn_power": float(future_row[1] if len(future_row) > 1 else 0.0),
            "setup_value": float(future_row[2] if len(future_row) > 2 else 0.0),
            "ppo_value": float(raw.get("state_value", 0.0) or 0.0),
            "policy_collate_duration_s": float(raw.get("policy_collate_duration_s", 0.0) or 0.0),
            "intent_forward_duration_s": float(raw.get("intent_forward_duration_s", 0.0) or 0.0),
            "action_forward_duration_s": float(raw.get("action_forward_duration_s", 0.0) or 0.0),
            "policy_forward_duration_s": float(raw.get("policy_forward_duration_s", 0.0) or 0.0),
            "policy_postprocess_duration_s": float(
                (raw.get("policy_postprocess_duration_s", 0.0) or 0.0) + policy_postprocess_duration_s
            ),
            "batch_wait_duration_s": float(raw.get("batch_wait_duration_s", 0.0) or 0.0),
            "batch_size_effective": int(raw.get("batch_size_effective", 1) or 1),
        }

    def select_action(self, state: BattleState) -> int:
        inference = self.infer(state)
        if self._model.policy_arch == "hierarchical_intent" and inference.get("is_turn_start", False):
            self.observe_intent_choice(state, int(inference.get("active_intent", 0) or 0))
        return int(inference["action_index"])

    def score_actions(self, state: BattleState) -> list[float]:
        inference = self.infer(state)
        if self._model.policy_arch == "hierarchical_intent" and inference.get("is_turn_start", False):
            self.observe_intent_choice(state, int(inference.get("active_intent", 0) or 0))
        scores = inference["scores"]
        return list(scores)

    def clone_for_rollout(self) -> "ModelPolicyAdapter":
        return ModelPolicyAdapter(
            self._model,
            self._collator,
            self._history.maxlen or 0,
            batcher=self._batcher,
        )


def _extract_explicit_turn_id(state: BattleState) -> int | None:
    metadata = state.context.metadata or {}
    value = metadata.get("turn_id", metadata.get("round_number_raw"))
    if value in {None, "", 0, "0"}:
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _state_turn_id(state: BattleState, *, fallback: int) -> int:
    explicit = _extract_explicit_turn_id(state)
    return explicit if explicit is not None else int(max(fallback, 1))


def _find_confirm_action_index(state: BattleState) -> int | None:
    for index, action in enumerate(state.legal_actions):
        action_type = str(action.action_type or "").strip().lower()
        if action_type in {"confirm_selection", "combat_confirm_selection"}:
            return index
    return None


def _argmax_index(values: list[float]) -> int:
    if not values:
        return 0
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _sanitize_score_list(values: list[float], *, fill: float = -1.0e9) -> list[float]:
    sanitized: list[float] = []
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric):
            numeric = fill
        sanitized.append(numeric)
    return sanitized


def _tensor_to_python_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return _sanitize_score_list([float(item) for item in value.tolist()])
    if isinstance(value, list):
        return _sanitize_score_list([float(item) for item in value])
    return []


@dataclass(slots=True)
class LocalCheckpointStore:
    root: Path
    active_pointer_name: str = "active.json"

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._staging_root.mkdir(parents=True, exist_ok=True)

    @property
    def _staging_root(self) -> Path:
        return self.root / "_staging"

    @property
    def _active_pointer_path(self) -> Path:
        return self.root / self.active_pointer_name

    @property
    def _baseline_pointer_path(self) -> Path:
        return self.root / "baseline_eval.json"

    def save(self, version: str, payload: dict[str, object]) -> Path:
        path = self.root / f"{version}.pt"
        self._atomic_torch_save(path, payload)
        return path

    def save_candidate(self, version: str, payload: dict[str, object]) -> Path:
        path = self._staging_root / f"{version}.pt"
        self._atomic_torch_save(path, payload)
        return path

    def discard(self, path: Path | None) -> None:
        if path and path.exists():
            path.unlink()

    def load(self, version: str) -> dict[str, object]:
        path = self.root / f"{version}.pt"
        return torch.load(path, map_location="cpu")

    def write_active_version(self, version: str) -> Path:
        payload = {"active_version": version}
        self._atomic_write_text(self._active_pointer_path, json.dumps(payload, ensure_ascii=False, indent=2))
        return self._active_pointer_path

    def read_active_version(self) -> str | None:
        path = self._active_pointer_path
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("active_version") or "") or None

    def write_baseline_eval(self, rows: list[dict[str, object]]) -> Path:
        self._atomic_write_text(
            self._baseline_pointer_path,
            json.dumps(rows, ensure_ascii=False, indent=2),
        )
        return self._baseline_pointer_path

    def read_baseline_eval(self) -> list[dict[str, object]] | None:
        path = self._baseline_pointer_path
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else None

    def _atomic_torch_save(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
