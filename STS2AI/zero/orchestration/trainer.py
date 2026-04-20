from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from ..buffers import SamplePoolSet
from ..config import LossWeights, TrainConfig
from ..domain import BattleState, FightLabel, TrainingSample, TrainingSummary, TransitionDelta
from ..features import BatchCollator, compute_transition_delta
from ..model import ZeroNet, compute_losses
from ..ports import Policy


class ZeroTrainer:
    def __init__(self, model: ZeroNet, config: TrainConfig, loss_weights: LossWeights, collator: BatchCollator):
        self._model = model
        self._config = config
        self._loss_weights = loss_weights
        self._collator = collator
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train_iteration(self, pools: SamplePoolSet) -> TrainingSummary:
        self._model.train()
        metrics = TrainingSummary()
        for _ in range(self._config.steps_per_iteration):
            samples = pools.mixed_sample(self._config.batch_size)
            if not samples:
                break
            batch = self._collator.collate(samples)
            output = self._model(batch)
            losses = compute_losses(output, batch, self._loss_weights)

            self._optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.grad_clip_norm)
            self._optimizer.step()

            metrics.steps += 1
            metrics.policy_loss += float(losses.policy.detach())
            metrics.value_loss += float(losses.value.detach())
            metrics.ranking_loss += float(losses.ranking.detach())
            metrics.delta_loss += float(losses.delta.detach())
            metrics.uncertainty_loss += float(losses.uncertainty.detach())
            metrics.total_loss += float(losses.total.detach())

        if metrics.steps > 0:
            metrics.policy_loss /= metrics.steps
            metrics.value_loss /= metrics.steps
            metrics.ranking_loss /= metrics.steps
            metrics.delta_loss /= metrics.steps
            metrics.uncertainty_loss /= metrics.steps
            metrics.total_loss /= metrics.steps
        return metrics


class ModelPolicyAdapter(Policy):
    def __init__(self, model: ZeroNet, collator: BatchCollator, history_steps: int):
        self._model = model
        self._collator = collator
        self._history: deque = deque(maxlen=history_steps)

    def reset_episode(self) -> None:
        self._history.clear()

    def observe_transition(self, state: BattleState, action_index: int, next_state: BattleState) -> None:
        if not (0 <= action_index < len(state.legal_actions)):
            return
        from ..domain import HistoryStep

        self._history.append(
            HistoryStep(
                state=state,
                action=state.legal_actions[action_index],
                delta=compute_transition_delta(state, next_state),
            )
        )

    def select_action(self, state: BattleState) -> int:
        scores = self.score_actions(state)
        if not scores:
            return 0
        return max(range(len(scores)), key=lambda index: scores[index])

    def score_actions(self, state: BattleState) -> list[float]:
        if not state.legal_actions:
            return []
        self._model.eval()
        sample = TrainingSample(
            sample_id="inference",
            run_id="inference",
            fight_id="inference",
            step_idx=0,
            state=state,
            history=list(self._history),
            legal_actions=state.legal_actions,
            behavior_action_index=0,
            delta=TransitionDelta(),
            fight_label=FightLabel(
                fight_win=0.0,
                enemy_hp_fraction_dealt=0.0,
                self_hp_fraction_remaining=0.0,
            ),
        )
        batch = self._collator.collate([sample])
        with torch.no_grad():
            output = self._model(batch)
        logits = output.policy_logits[0]
        valid = int(batch.action_mask[0].sum().item())
        return logits[:valid].tolist()

    def estimate_uncertainty(self, state: BattleState) -> float:
        if not state.legal_actions:
            return 0.0
        self._model.eval()
        sample = TrainingSample(
            sample_id="inference",
            run_id="inference",
            fight_id="inference",
            step_idx=0,
            state=state,
            history=list(self._history),
            legal_actions=state.legal_actions,
            behavior_action_index=0,
            delta=TransitionDelta(),
            fight_label=FightLabel(
                fight_win=0.0,
                enemy_hp_fraction_dealt=0.0,
                self_hp_fraction_remaining=0.0,
            ),
        )
        batch = self._collator.collate([sample])
        with torch.no_grad():
            output = self._model(batch)
        return float(torch.sigmoid(output.uncertainty)[0].item())


@dataclass(slots=True)
class LocalCheckpointStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, version: str, model_state: dict[str, object]) -> Path:
        path = self.root / f"{version}.pt"
        torch.save(model_state, path)
        return path

    def load(self, version: str) -> dict[str, object]:
        path = self.root / f"{version}.pt"
        return torch.load(path, map_location="cpu")
