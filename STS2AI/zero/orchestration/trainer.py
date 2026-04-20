from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..buffers import SamplePoolSet
from ..config import LossWeights, TrainConfig
from ..domain import BattleState, HistoryStep, TrainingSummary
from ..features import BatchCollator, FeatureExtractor, compute_transition_delta
from ..model import ZeroNet, compute_losses
from ..ports import Policy


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

    def train_iteration(self, pools: SamplePoolSet) -> TrainingSummary:
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
            teacher_ratio = sum(1 for sample in samples if sample.teacher_label is not None) / max(1, len(samples))
            self._optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self._device.type, enabled=self._amp_enabled):
                output = self._model(batch)
                losses = compute_losses(output, batch, self._loss_weights)

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
            metrics.value_loss += float(losses.value.detach())
            metrics.ranking_loss += float(losses.ranking.detach())
            metrics.delta_loss += float(losses.delta.detach())
            metrics.uncertainty_loss += float(losses.uncertainty.detach())
            metrics.total_loss += float(losses.total.detach())
            metrics.grad_norm += float(grad_norm)
            metrics.learning_rate += float(self._optimizer.param_groups[0]["lr"])
            metrics.teacher_sample_ratio += float(teacher_ratio)
            for pool_name, count in pool_counts.items():
                metrics.pool_usage[pool_name] = metrics.pool_usage.get(pool_name, 0) + int(count)

        if metrics.steps > 0:
            metrics.policy_loss /= metrics.steps
            metrics.value_loss /= metrics.steps
            metrics.ranking_loss /= metrics.steps
            metrics.delta_loss /= metrics.steps
            metrics.uncertainty_loss /= metrics.steps
            metrics.total_loss /= metrics.steps
            metrics.grad_norm /= metrics.steps
            metrics.learning_rate /= metrics.steps
            metrics.teacher_sample_ratio /= metrics.steps
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
        warmup_steps = max(0, int(self._config.warmup_steps))
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-6, float(step + 1) / float(warmup_steps))
        adjusted = max(step - warmup_steps, 0) + 1
        decay = 1.0 / math.sqrt(float(adjusted))
        return max(float(self._config.min_lr_ratio), decay)


class ModelPolicyAdapter(Policy):
    def __init__(self, model: ZeroNet, collator: BatchCollator, history_steps: int):
        self._model = model
        self._collator = collator
        self._history_extractor = FeatureExtractor(collator._config)
        self._history: deque = deque(maxlen=history_steps)
        self._device = next(model.parameters()).device
        self._cached_state_ref = None
        self._cached_inference: dict[str, Any] | None = None

    def reset_episode(self) -> None:
        self._history.clear()
        self._cached_state_ref = None
        self._cached_inference = None

    def observe_transition(self, state: BattleState, action_index: int, next_state: BattleState) -> None:
        if not (0 <= action_index < len(state.legal_actions)):
            return
        delta = compute_transition_delta(state, next_state)
        self._history.append(
            HistoryStep(
                state=None,
                action=None,
                delta=delta,
                history_token=self._history_extractor.encode_history_step_token(
                    state,
                    state.legal_actions[action_index],
                    delta,
                ),
            )
        )
        self._cached_state_ref = None
        self._cached_inference = None

    def infer(self, state: BattleState) -> dict[str, Any]:
        if self._cached_state_ref is state and self._cached_inference is not None:
            return self._cached_inference
        if not state.legal_actions:
            result = {"scores": [], "action_index": 0, "uncertainty": 0.0}
            self._cached_state_ref = state
            self._cached_inference = result
            return result
        self._model.eval()
        batch = self._collator.collate_inference(state, list(self._history), state.legal_actions).to(self._device)
        with torch.no_grad():
            output = self._model(batch)
        logits = output.policy_logits[0]
        valid = int(batch.action_mask[0].sum().item())
        scores = logits[:valid].tolist()
        action_index = max(range(len(scores)), key=lambda index: scores[index]) if scores else 0
        uncertainty = float(torch.sigmoid(output.uncertainty)[0].item())
        result = {
            "scores": scores,
            "action_index": action_index,
            "uncertainty": uncertainty,
        }
        self._cached_state_ref = state
        self._cached_inference = result
        return result

    def select_action(self, state: BattleState) -> int:
        return int(self.infer(state)["action_index"])

    def score_actions(self, state: BattleState) -> list[float]:
        return list(self.infer(state)["scores"])

    def estimate_uncertainty(self, state: BattleState) -> float:
        return float(self.infer(state)["uncertainty"])

    def clone_for_rollout(self) -> "ModelPolicyAdapter":
        return ModelPolicyAdapter(self._model, self._collator, self._history.maxlen or 0)


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
