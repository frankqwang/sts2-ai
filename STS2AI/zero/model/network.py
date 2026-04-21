from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import EncoderConfig
from .components import ActionEncoder, CurrentStateEncoder, HistoryEncoder, MlpBlock


@dataclass(slots=True)
class ZeroNetOutput:
    policy_logits: torch.Tensor
    fight_win: torch.Tensor
    enemy_hp_fraction_dealt: torch.Tensor
    self_hp_fraction_remaining: torch.Tensor
    delta_pred: torch.Tensor
    uncertainty: torch.Tensor


class ZeroNet(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self._config = config
        self.current_state = CurrentStateEncoder(config)
        self.history = HistoryEncoder(config)
        self.action = ActionEncoder(config)
        self.context_fusion = MlpBlock(config.hidden_dim + config.history_dim, config.hidden_dim, config.hidden_dim)
        self.policy_head = MlpBlock(config.hidden_dim + config.action_dim, config.hidden_dim, 1)
        self.value_head = MlpBlock(config.hidden_dim, config.hidden_dim, 3)
        delta_dim = 3 + config.max_enemies * 2 + 3
        self.delta_head = MlpBlock(config.hidden_dim, config.hidden_dim, delta_dim)
        self.uncertainty_head = MlpBlock(config.hidden_dim, config.hidden_dim, 1)

    def forward(self, batch) -> ZeroNetOutput:
        state_hidden = self.current_state(batch)
        history_hidden = self.history(batch)
        context_hidden = self.context_fusion(torch.cat([state_hidden, history_hidden], dim=-1))

        action_hidden = self.action(batch)
        expanded_context = context_hidden.unsqueeze(1).expand(-1, action_hidden.size(1), -1)
        policy_logits = self.policy_head(torch.cat([expanded_context, action_hidden], dim=-1)).squeeze(-1)
        policy_logits = policy_logits.masked_fill(batch.action_mask <= 0, float("-inf"))

        values = self.value_head(context_hidden)
        delta_pred = self.delta_head(context_hidden)
        uncertainty = self.uncertainty_head(context_hidden).squeeze(-1)
        return ZeroNetOutput(
            policy_logits=policy_logits,
            fight_win=values[:, 0],
            enemy_hp_fraction_dealt=values[:, 1],
            self_hp_fraction_remaining=values[:, 2],
            delta_pred=delta_pred,
            uncertainty=uncertainty,
        )
