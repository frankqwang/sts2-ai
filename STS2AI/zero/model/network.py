from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ..config import EncoderConfig
from .components import (
    ActionEncoder,
    CurrentStateEncoder,
    HistoryEncoder,
    MlpHiddenBlock,
    MlpOutputBlock,
    RecurrentHistoryEncoder,
    ResidualHistoryFusion,
)


PolicyArch = Literal["flat", "hierarchical_intent"]
HistoryVariant = Literal["stateless", "recurrent_gru", "history_transformer"]


@dataclass(slots=True)
class FlatPolicyOutput:
    state_value: torch.Tensor
    action_logits: torch.Tensor
    action_value: torch.Tensor
    death_risk_2t: torch.Tensor
    next_turn_power: torch.Tensor
    setup_value: torch.Tensor
    confirm_now_logit: torch.Tensor


@dataclass(slots=True)
class HierarchicalPolicyOutput:
    state_value: torch.Tensor
    intent_logits: torch.Tensor
    intent_value: torch.Tensor
    action_logits: torch.Tensor
    action_value: torch.Tensor
    death_risk_2t: torch.Tensor
    next_turn_power: torch.Tensor
    setup_value: torch.Tensor
    confirm_now_logit: torch.Tensor


ZeroNetOutput = FlatPolicyOutput | HierarchicalPolicyOutput


@dataclass(slots=True)
class EncodedStateOutput:
    context_hidden: torch.Tensor
    hand_tokens: torch.Tensor
    hand_mask: torch.Tensor
    enemy_tokens: torch.Tensor
    enemy_mask: torch.Tensor
    state_value: torch.Tensor
    future_summary: torch.Tensor | None = None
    confirm_now_logit: torch.Tensor | None = None
    intent_logits: torch.Tensor | None = None
    intent_value: torch.Tensor | None = None
    future_summary_by_intent: torch.Tensor | None = None
    confirm_now_logit_by_intent: torch.Tensor | None = None


class ZeroNet(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self._config = config
        self._policy_arch, self._history_variant = _resolve_encoder_variants(config)
        self.current_state = CurrentStateEncoder(config)
        self.history = _build_history_encoder(config, self._history_variant)
        self.context_fusion = (
            MlpHiddenBlock(config.hidden_dim, config.hidden_dim, config.hidden_dim)
            if self.history is None
            else ResidualHistoryFusion(config)
        )
        self.state_value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.flat_future_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.future_summary_dim),
        )
        self.flat_confirm_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.action = ActionEncoder(config)

        if self._policy_arch == "hierarchical_intent":
            self.intent_embedding = nn.Embedding(config.intent_vocab_size, config.intent_dim)
            nn.init.normal_(self.intent_embedding.weight, mean=0.0, std=0.01)
            self.intent_policy_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.intent_vocab_size),
            )
            self.intent_value_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )
            self.intent_future_proj = nn.Sequential(
                nn.Linear(config.hidden_dim + config.intent_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.intent_future_head = nn.Linear(config.hidden_dim, config.future_summary_dim)
            self.intent_confirm_head = nn.Linear(config.hidden_dim, 1)
            action_input_dim = config.hidden_dim * 3 + config.intent_dim + config.action_dim
        else:
            self.intent_embedding = None
            self.intent_policy_head = None
            self.intent_value_head = None
            self.intent_future_proj = None
            self.intent_future_head = None
            self.intent_confirm_head = None
            action_input_dim = config.hidden_dim * 3 + config.action_dim

        self.action_policy_head = MlpOutputBlock(
            action_input_dim,
            config.hidden_dim,
            1,
        )
        self.action_value_head = MlpOutputBlock(action_input_dim, config.hidden_dim, 1)

    @property
    def policy_arch(self) -> PolicyArch:
        return self._policy_arch

    @property
    def history_variant(self) -> HistoryVariant:
        return self._history_variant

    def encode_state(self, batch) -> EncodedStateOutput:
        state_tokens = self.current_state(batch)
        state_hidden = state_tokens.context_hidden
        if self.history is None:
            context_hidden = self.context_fusion(state_hidden)
        else:
            history_hidden = self.history(batch)
            context_hidden = self.context_fusion(state_hidden, history_hidden)
        state_value = self.state_value_head(context_hidden).squeeze(-1)
        if self._policy_arch == "flat":
            future_summary = self.flat_future_head(context_hidden)
            confirm_now_logit = self.flat_confirm_head(context_hidden).squeeze(-1)
            return EncodedStateOutput(
                context_hidden=context_hidden,
                hand_tokens=state_tokens.hand_tokens,
                hand_mask=state_tokens.hand_mask,
                enemy_tokens=state_tokens.enemy_tokens,
                enemy_mask=state_tokens.enemy_mask,
                state_value=state_value,
                future_summary=future_summary,
                confirm_now_logit=confirm_now_logit,
            )
        assert self.intent_embedding is not None
        assert self.intent_policy_head is not None
        assert self.intent_value_head is not None
        assert self.intent_future_proj is not None
        assert self.intent_future_head is not None
        assert self.intent_confirm_head is not None
        intent_logits = self.intent_policy_head(context_hidden)
        intent_value = self.intent_value_head(context_hidden).squeeze(-1)
        intent_embed = self.intent_embedding.weight.unsqueeze(0).expand(context_hidden.size(0), -1, -1)
        expanded_context = context_hidden.unsqueeze(1).expand(-1, self._config.intent_vocab_size, -1)
        future_hidden = self.intent_future_proj(torch.cat([expanded_context, intent_embed], dim=-1))
        future_summary_by_intent = self.intent_future_head(future_hidden)
        confirm_now_logit_by_intent = self.intent_confirm_head(future_hidden).squeeze(-1)
        return EncodedStateOutput(
            context_hidden=context_hidden,
            hand_tokens=state_tokens.hand_tokens,
            hand_mask=state_tokens.hand_mask,
            enemy_tokens=state_tokens.enemy_tokens,
            enemy_mask=state_tokens.enemy_mask,
            state_value=state_value,
            intent_logits=intent_logits,
            intent_value=intent_value,
            future_summary_by_intent=future_summary_by_intent,
            confirm_now_logit_by_intent=confirm_now_logit_by_intent,
        )

    def score_actions(self, batch, encoded_state: EncodedStateOutput) -> tuple[torch.Tensor, torch.Tensor]:
        action_hidden = self.action(batch)
        batch_size, action_count, _ = action_hidden.shape
        hand_context = _build_action_hand_context(
            batch=batch,
            hand_tokens=encoded_state.hand_tokens,
            hand_mask=encoded_state.hand_mask,
        )
        enemy_context = _build_action_enemy_context(
            batch=batch,
            enemy_tokens=encoded_state.enemy_tokens,
            enemy_mask=encoded_state.enemy_mask,
        )
        if self._policy_arch == "flat":
            expanded_context = encoded_state.context_hidden.unsqueeze(1).expand(-1, action_count, -1)
            combined = torch.cat([expanded_context, hand_context, enemy_context, action_hidden], dim=-1)
            action_logits = self.action_policy_head(combined).squeeze(-1)
            action_value = self.action_value_head(combined).squeeze(-1)
            action_logits = action_logits.masked_fill(batch.action_mask <= 0, float("-inf"))
            action_value = action_value.masked_fill(batch.action_mask <= 0, 0.0)
            return action_logits, action_value
        assert self.intent_embedding is not None
        intent_embed = self.intent_embedding.weight.view(1, self._config.intent_vocab_size, 1, self._config.intent_dim)
        expanded_context = encoded_state.context_hidden.view(batch_size, 1, 1, self._config.hidden_dim).expand(
            -1,
            self._config.intent_vocab_size,
            action_count,
            -1,
        )
        expanded_hand = hand_context.unsqueeze(1).expand(-1, self._config.intent_vocab_size, -1, -1)
        expanded_enemy = enemy_context.unsqueeze(1).expand(-1, self._config.intent_vocab_size, -1, -1)
        expanded_actions = action_hidden.unsqueeze(1).expand(-1, self._config.intent_vocab_size, -1, -1)
        expanded_intents = intent_embed.expand(batch_size, -1, action_count, -1)
        combined = torch.cat([expanded_context, expanded_hand, expanded_enemy, expanded_intents, expanded_actions], dim=-1)
        action_logits = self.action_policy_head(combined).squeeze(-1)
        action_value = self.action_value_head(combined).squeeze(-1)
        action_mask = batch.action_mask.unsqueeze(1)
        action_logits = action_logits.masked_fill(action_mask <= 0, float("-inf"))
        action_value = action_value.masked_fill(action_mask <= 0, 0.0)
        return action_logits, action_value

    def forward(self, batch) -> ZeroNetOutput:
        encoded_state = self.encode_state(batch)
        action_logits, action_value = self.score_actions(batch, encoded_state)
        if self._policy_arch == "flat":
            future_summary = encoded_state.future_summary
            confirm_now_logit = encoded_state.confirm_now_logit
            if future_summary is None or confirm_now_logit is None:
                raise RuntimeError("flat policy 缺少未来摘要或 confirm 输出。")
            return FlatPolicyOutput(
                state_value=encoded_state.state_value,
                action_logits=action_logits,
                action_value=action_value,
                death_risk_2t=future_summary[:, 0],
                next_turn_power=future_summary[:, 1],
                setup_value=future_summary[:, 2],
                confirm_now_logit=confirm_now_logit,
            )
        if (
            encoded_state.intent_logits is None
            or encoded_state.intent_value is None
            or encoded_state.future_summary_by_intent is None
            or encoded_state.confirm_now_logit_by_intent is None
        ):
            raise RuntimeError("hierarchical_intent 输出缺少必要分支。")
        return HierarchicalPolicyOutput(
            state_value=encoded_state.state_value,
            intent_logits=encoded_state.intent_logits,
            intent_value=encoded_state.intent_value,
            action_logits=action_logits,
            action_value=action_value,
            death_risk_2t=encoded_state.future_summary_by_intent[:, :, 0],
            next_turn_power=encoded_state.future_summary_by_intent[:, :, 1],
            setup_value=encoded_state.future_summary_by_intent[:, :, 2],
            confirm_now_logit=encoded_state.confirm_now_logit_by_intent,
        )


def _build_action_hand_context(batch, hand_tokens: torch.Tensor, hand_mask: torch.Tensor) -> torch.Tensor:
    batch_size, action_count = batch.action_card_ids.shape
    hidden_dim = hand_tokens.size(-1)
    if hand_tokens.size(1) == 0:
        return torch.zeros(batch_size, action_count, hidden_dim, device=hand_tokens.device, dtype=hand_tokens.dtype)
    explicit_index = batch.action_card_indices
    clamped_index = explicit_index.clamp(min=0, max=max(hand_tokens.size(1) - 1, 0))
    gathered = hand_tokens.gather(1, clamped_index.unsqueeze(-1).expand(-1, -1, hidden_dim))
    explicit_valid = (explicit_index >= 0) & (explicit_index < hand_tokens.size(1)) & (batch.action_mask > 0)
    matched_cards = (
        (batch.action_card_ids.unsqueeze(-1) == batch.hand_card_ids.unsqueeze(1))
        & (batch.action_card_ids.unsqueeze(-1) > 0)
        & (hand_mask.unsqueeze(1) > 0)
    )
    fallback = _masked_action_token_reduce(hand_tokens, matched_cards)
    selected = torch.where(explicit_valid.unsqueeze(-1), gathered, fallback)
    return selected * batch.action_mask.to(dtype=selected.dtype).unsqueeze(-1)


def _build_action_enemy_context(batch, enemy_tokens: torch.Tensor, enemy_mask: torch.Tensor) -> torch.Tensor:
    batch_size, action_count = batch.action_target_ids.shape
    hidden_dim = enemy_tokens.size(-1)
    if enemy_tokens.size(1) == 0:
        return torch.zeros(batch_size, action_count, hidden_dim, device=enemy_tokens.device, dtype=enemy_tokens.dtype)
    target_matches = (
        (batch.action_target_ids.unsqueeze(-1) == batch.enemy_target_ids.unsqueeze(1))
        & (batch.action_target_ids.unsqueeze(-1) > 0)
        & (enemy_mask.unsqueeze(1) > 0)
    )
    selected = _masked_action_token_reduce(enemy_tokens, target_matches)
    return selected * batch.action_mask.to(dtype=selected.dtype).unsqueeze(-1)


def _masked_action_token_reduce(tokens: torch.Tensor, match_mask: torch.Tensor) -> torch.Tensor:
    weights = match_mask.to(dtype=tokens.dtype)
    total = (tokens.unsqueeze(1) * weights.unsqueeze(-1)).sum(dim=2)
    denom = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
    reduced = total / denom
    empty = weights.sum(dim=2) <= 0
    if empty.any():
        reduced = reduced.clone()
        reduced[empty] = 0.0
    return reduced


def _resolve_encoder_variants(config: EncoderConfig) -> tuple[PolicyArch, HistoryVariant]:
    policy_arch = str(config.policy_arch or "flat").strip().lower().replace("-", "_")
    history_variant = str(config.history_variant or "recurrent_gru").strip().lower().replace("-", "_")
    legacy = str(config.model_variant or "").strip().lower().replace("-", "_")

    if legacy:
        if legacy == "hierarchical_intent":
            if policy_arch == "flat":
                policy_arch = "hierarchical_intent"
            if history_variant == "recurrent_gru":
                history_variant = "recurrent_gru"
        elif legacy in {"stateless", "recurrent_gru", "history_transformer"}:
            if history_variant == "recurrent_gru":
                history_variant = legacy
        else:
            raise ValueError(f"不支持的 model_variant={config.model_variant}")

    valid_policy_arch = {"flat", "hierarchical_intent"}
    if policy_arch not in valid_policy_arch:
        raise ValueError(f"不支持的 policy_arch={config.policy_arch}")
    valid_history_variant = {"stateless", "recurrent_gru", "history_transformer"}
    if history_variant not in valid_history_variant:
        raise ValueError(f"不支持的 history_variant={config.history_variant}")
    return policy_arch, history_variant


def _build_history_encoder(config: EncoderConfig, variant: HistoryVariant) -> nn.Module | None:
    if variant == "stateless":
        return None
    if variant == "history_transformer":
        return HistoryEncoder(config)
    if variant == "recurrent_gru":
        return RecurrentHistoryEncoder(config)
    raise ValueError(f"不支持的 history_variant={variant}")


def is_hierarchical_output(output: ZeroNetOutput) -> bool:
    return isinstance(output, HierarchicalPolicyOutput)


def select_action_logits(output: ZeroNetOutput, active_intent: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(output, FlatPolicyOutput):
        return output.action_logits
    if active_intent is None:
        raise ValueError("hierarchical 输出需要 active_intent。")
    gather_index = active_intent.view(-1, 1, 1).expand(-1, 1, output.action_logits.size(-1))
    return output.action_logits.gather(1, gather_index).squeeze(1)


def select_action_value(output: ZeroNetOutput, active_intent: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(output, FlatPolicyOutput):
        return output.action_value
    if active_intent is None:
        raise ValueError("hierarchical 输出需要 active_intent。")
    gather_index = active_intent.view(-1, 1, 1).expand(-1, 1, output.action_value.size(-1))
    return output.action_value.gather(1, gather_index).squeeze(1)


def select_future_summary(output: ZeroNetOutput, active_intent: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(output, FlatPolicyOutput):
        return torch.stack(
            [output.death_risk_2t, output.next_turn_power, output.setup_value],
            dim=-1,
        )
    if active_intent is None:
        raise ValueError("hierarchical 输出需要 active_intent。")
    death = output.death_risk_2t.gather(1, active_intent.unsqueeze(1)).squeeze(1)
    power = output.next_turn_power.gather(1, active_intent.unsqueeze(1)).squeeze(1)
    setup = output.setup_value.gather(1, active_intent.unsqueeze(1)).squeeze(1)
    return torch.stack([death, power, setup], dim=-1)


def select_confirm_logit(output: ZeroNetOutput, active_intent: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(output, FlatPolicyOutput):
        return output.confirm_now_logit
    if active_intent is None:
        raise ValueError("hierarchical 输出需要 active_intent。")
    return output.confirm_now_logit.gather(1, active_intent.unsqueeze(1)).squeeze(1)
