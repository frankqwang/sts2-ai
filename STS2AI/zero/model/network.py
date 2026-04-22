from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import EncoderConfig
from .components import ActionEncoder, CurrentStateEncoder, HistoryEncoder, MlpBlock, RecurrentHistoryEncoder, ResidualHistoryFusion


@dataclass(slots=True)
class ZeroNetOutput:
    policy_logits: torch.Tensor
    fight_win: torch.Tensor
    enemy_hp_fraction_dealt: torch.Tensor
    self_hp_fraction_remaining: torch.Tensor
    ppo_value: torch.Tensor
    delta_pred: torch.Tensor
    uncertainty: torch.Tensor


class ZeroNet(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self._config = config
        self._variant = _normalize_model_variant(config.model_variant)
        self.current_state = CurrentStateEncoder(config)
        self.history = _build_history_encoder(config, self._variant)
        self.action = ActionEncoder(config)
        self.context_fusion = (
            MlpBlock(config.hidden_dim, config.hidden_dim, config.hidden_dim)
            if self.history is None
            else ResidualHistoryFusion(config)
        )
        self.policy_head = MlpBlock(config.hidden_dim + config.action_dim, config.hidden_dim, 1)
        self.value_head = MlpBlock(config.hidden_dim, config.hidden_dim, 3)
        self.ppo_value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        delta_dim = 3 + config.max_enemies * 2 + 3
        self.delta_head = MlpBlock(config.hidden_dim, config.hidden_dim, delta_dim)
        self.uncertainty_head = MlpBlock(config.hidden_dim, config.hidden_dim, 1)

    def forward(self, batch) -> ZeroNetOutput:
        state_hidden = self.current_state(batch)
        if self.history is None:
            context_hidden = self.context_fusion(state_hidden)
        else:
            history_hidden = self.history(batch)
            context_hidden = self.context_fusion(state_hidden, history_hidden)

        action_hidden = self.action(batch)
        expanded_context = context_hidden.unsqueeze(1).expand(-1, action_hidden.size(1), -1)
        policy_logits = self.policy_head(torch.cat([expanded_context, action_hidden], dim=-1)).squeeze(-1)
        policy_logits = policy_logits.masked_fill(batch.action_mask <= 0, float("-inf"))

        values = self.value_head(context_hidden)
        ppo_value = self.ppo_value_head(context_hidden).squeeze(-1)
        delta_pred = self.delta_head(context_hidden)
        uncertainty = self.uncertainty_head(context_hidden).squeeze(-1)
        return ZeroNetOutput(
            policy_logits=policy_logits,
            fight_win=values[:, 0],
            enemy_hp_fraction_dealt=values[:, 1],
            self_hp_fraction_remaining=values[:, 2],
            ppo_value=ppo_value,
            delta_pred=delta_pred,
            uncertainty=uncertainty,
        )


def _normalize_model_variant(name: str) -> str:
    normalized = str(name or "history_transformer").strip().lower().replace("-", "_")
    aliases = {
        "default": "history_transformer",
        "history": "history_transformer",
        "transformer": "history_transformer",
        "history_transformer": "history_transformer",
        "stateless": "stateless",
        "no_history": "stateless",
        "gru": "recurrent_gru",
        "recurrent": "recurrent_gru",
        "recurrent_gru": "recurrent_gru",
    }
    if normalized not in aliases:
        raise ValueError(f"不支持的 model_variant={name}")
    return aliases[normalized]


def _build_history_encoder(config: EncoderConfig, variant: str) -> nn.Module | None:
    if variant == "stateless":
        return None
    if variant == "history_transformer":
        return HistoryEncoder(config)
    if variant == "recurrent_gru":
        return RecurrentHistoryEncoder(config)
    raise ValueError(f"不支持的 model_variant={variant}")
