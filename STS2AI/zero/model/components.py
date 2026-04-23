from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import EncoderConfig
from ..features.extractor import ACTION_SEMANTIC_DIM, HAND_SEMANTIC_DIM, HISTORY_TOKEN_DIM, PILE_SEMANTIC_DIM, PLAYER_SEMANTIC_DIM, STATIC_NUMERIC_DIM


class MlpHiddenBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MlpOutputBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        safe_x = x
        safe_mask = mask
        empty_rows = mask.sum(dim=1) <= 0
        if empty_rows.any():
            # MultiheadAttention does not handle "all keys masked" rows well.
            # Keep one zero token alive, then zero the pooled result back out.
            safe_x = x.clone()
            safe_mask = mask.clone()
            safe_x[empty_rows, 0, :] = 0.0
            safe_mask[empty_rows, 0] = 1.0
        key_padding_mask = safe_mask <= 0
        query = self.query.expand(x.size(0), -1, -1)
        pooled, _ = self.attention(query, safe_x, safe_x, key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)
        if empty_rows.any():
            pooled = pooled.clone()
            pooled[empty_rows] = 0.0
        return pooled


class ResidualCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        query_mask: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_tokens.size(1) == 0:
            return query_tokens
        attended, _ = self.attention(
            query_tokens,
            context_tokens,
            context_tokens,
            key_padding_mask=context_mask <= 0,
        )
        output = self.norm(query_tokens + attended)
        if query_mask.numel() > 0:
            output = output * query_mask.to(dtype=output.dtype).unsqueeze(-1)
        return output


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_steps: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_steps, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


@dataclass(slots=True)
class CurrentStateTokens:
    context_hidden: torch.Tensor
    summary_tokens: torch.Tensor
    summary_mask: torch.Tensor
    enemy_tokens: torch.Tensor
    enemy_mask: torch.Tensor
    hand_tokens: torch.Tensor
    hand_mask: torch.Tensor


class CurrentStateEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.id_embedding = nn.Embedding(config.id_hash_buckets, 32)
        self.token_type_embedding = nn.Embedding(16, config.hidden_dim)
        player_input_dim = 6 + config.buff_slots + 3 + PLAYER_SEMANTIC_DIM
        hand_input_dim = 9 + HAND_SEMANTIC_DIM + 32
        enemy_input_dim = 6 + config.buff_slots + 32 + 32 + 64
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.global_token = MlpHiddenBlock(player_input_dim + STATIC_NUMERIC_DIM + 96, config.hidden_dim, config.hidden_dim)
        self.id_set_pool = MaskedAttentionPool(32)
        self.set_summary = MlpHiddenBlock(32, 128, config.hidden_dim)
        self.pile_focus_token = MlpHiddenBlock(32, 128, config.hidden_dim)
        self.buff_summary = MlpHiddenBlock(33, 128, 64)
        self.buff_pool = MaskedAttentionPool(64)
        self.buff_token_proj = MlpHiddenBlock(64, 128, config.hidden_dim)
        self.pile_token = MlpHiddenBlock(8 + PILE_SEMANTIC_DIM, config.hidden_dim, config.hidden_dim)
        self.enemy_token = MlpHiddenBlock(enemy_input_dim, config.hidden_dim, config.hidden_dim)
        self.hand_token = MlpHiddenBlock(hand_input_dim, config.hidden_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.token_backbone_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            dropout=config.history_dropout,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.token_backbone_layers)
        self.hand_to_context = ResidualCrossAttention(config.hidden_dim, config.token_backbone_heads)
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, batch) -> CurrentStateTokens:
        batch_size = batch.player_numeric.size(0)
        static_id_embed = self.id_embedding(batch.static_ids).flatten(start_dim=1)
        global_token = self.global_token(torch.cat([batch.player_numeric, batch.static_numeric, static_id_embed], dim=-1))
        pile_token = self.pile_token(batch.pile_numeric)
        summary_tokens = [
            global_token,
            pile_token,
            self._summarize_id_set(batch.relic_ids, batch.relic_mask),
            self._summarize_id_set(batch.deck_card_ids, batch.deck_card_mask),
            self._summarize_id_set(batch.potion_ids, batch.potion_mask),
            self._summarize_id_set(batch.draw_pile_ids, batch.draw_pile_mask),
            self._summarize_id_set(batch.discard_pile_ids, batch.discard_pile_mask),
            self._summarize_id_set(batch.exhaust_pile_ids, batch.exhaust_pile_mask),
            self.buff_token_proj(self._summarize_buffs(batch.player_buff_ids, batch.player_buff_values, batch.player_buff_mask)),
        ]
        summary_type_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        summary_tokens = [
            token + self.token_type_embedding.weight[type_id].view(1, -1)
            for token, type_id in zip(summary_tokens, summary_type_ids, strict=True)
        ]
        summary_tensor = torch.stack(summary_tokens, dim=1)
        summary_mask = torch.ones(batch_size, summary_tensor.size(1), device=summary_tensor.device, dtype=torch.float32)
        pile_focus_groups = [
            self._build_focus_tokens(batch.deck_card_ids, batch.deck_card_mask, token_type_id=12),
            self._build_focus_tokens(batch.draw_pile_ids, batch.draw_pile_mask, token_type_id=13),
            self._build_focus_tokens(batch.discard_pile_ids, batch.discard_pile_mask, token_type_id=14),
            self._build_focus_tokens(batch.exhaust_pile_ids, batch.exhaust_pile_mask, token_type_id=15),
        ]
        focus_tensors = [tokens for tokens, _ in pile_focus_groups]
        focus_masks = [mask for _, mask in pile_focus_groups]
        summary_tensor = torch.cat([summary_tensor, *focus_tensors], dim=1)
        summary_mask = torch.cat([summary_mask, *focus_masks], dim=1)

        enemy_embed = torch.cat(
            [
                self.id_embedding(batch.enemy_ids),
                self.id_embedding(batch.enemy_intent_ids),
            ],
            dim=-1,
        )
        enemy_buff_hidden = self._summarize_enemy_buffs(batch.enemy_buff_ids, batch.enemy_buff_values, batch.enemy_buff_mask)
        enemy_tokens = self.enemy_token(torch.cat([batch.enemy_numeric, enemy_embed, enemy_buff_hidden], dim=-1))
        enemy_tokens = enemy_tokens + self.token_type_embedding.weight[10].view(1, 1, -1)

        hand_embed = self.id_embedding(batch.hand_card_ids)
        hand_tokens = self.hand_token(torch.cat([batch.hand_numeric, hand_embed], dim=-1))
        hand_tokens = hand_tokens + self.token_type_embedding.weight[11].view(1, 1, -1)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, summary_tensor, enemy_tokens, hand_tokens], dim=1)
        token_mask = torch.cat(
            [
                torch.ones(batch_size, 1, device=tokens.device, dtype=torch.float32),
                summary_mask,
                batch.enemy_mask,
                batch.hand_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=token_mask <= 0)
        cls_hidden = self.output_norm(encoded[:, 0, :])
        summary_width = summary_tensor.size(1)
        enemy_width = enemy_tokens.size(1)
        hand_width = hand_tokens.size(1)
        summary_hidden = encoded[:, 1 : 1 + summary_width, :]
        enemy_hidden = encoded[:, 1 + summary_width : 1 + summary_width + enemy_width, :]
        hand_hidden = encoded[:, 1 + summary_width + enemy_width : 1 + summary_width + enemy_width + hand_width, :]
        context_tokens = torch.cat([cls_hidden.unsqueeze(1), summary_hidden, enemy_hidden], dim=1)
        context_mask = torch.cat(
            [
                torch.ones(batch_size, 1, device=tokens.device, dtype=torch.float32),
                summary_mask,
                batch.enemy_mask,
            ],
            dim=1,
        )
        contextual_hand = self.hand_to_context(hand_hidden, batch.hand_mask, context_tokens, context_mask)
        return CurrentStateTokens(
            context_hidden=cls_hidden,
            summary_tokens=summary_hidden,
            summary_mask=summary_mask,
            enemy_tokens=enemy_hidden * batch.enemy_mask.to(dtype=enemy_hidden.dtype).unsqueeze(-1),
            enemy_mask=batch.enemy_mask,
            hand_tokens=contextual_hand,
            hand_mask=batch.hand_mask,
        )

    def _summarize_id_set(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.id_embedding(ids)
        pooled = self.id_set_pool(embedded, mask)
        return self.set_summary(pooled)

    def _build_focus_tokens(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        *,
        token_type_id: int,
        top_k: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.size(1) == 0 or top_k <= 0:
            empty_tokens = torch.zeros(ids.size(0), 0, self.hidden_dim, device=ids.device, dtype=self.cls_token.dtype)
            empty_mask = torch.zeros(ids.size(0), 0, device=ids.device, dtype=torch.float32)
            return empty_tokens, empty_mask
        width = min(top_k, ids.size(1))
        selected_ids = ids[:, :width]
        selected_mask = mask[:, :width].to(dtype=torch.float32)
        embedded = self.id_embedding(selected_ids)
        tokens = self.pile_focus_token(embedded)
        tokens = tokens + self.token_type_embedding.weight[token_type_id].view(1, 1, -1)
        tokens = tokens * selected_mask.unsqueeze(-1)
        return tokens, selected_mask

    def _summarize_buffs(self, ids: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.id_embedding(ids)
        hidden = self.buff_summary(torch.cat([embedded, values.unsqueeze(-1)], dim=-1))
        return self.buff_pool(hidden, mask)

    def _summarize_enemy_buffs(self, ids: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, enemy_count, buff_width = ids.shape
        flat_ids = ids.reshape(batch_size * enemy_count, buff_width)
        flat_values = values.reshape(batch_size * enemy_count, buff_width)
        flat_mask = mask.reshape(batch_size * enemy_count, buff_width)
        flat_hidden = self._summarize_buffs(flat_ids, flat_values, flat_mask)
        return flat_hidden.reshape(batch_size, enemy_count, -1)


class HistoryEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.input_proj = nn.Linear(HISTORY_TOKEN_DIM, config.history_dim)
        self.input_norm = nn.LayerNorm(config.history_dim)
        self.position = LearnedPositionalEncoding(config.history_steps, config.history_dim)
        self.dropout = nn.Dropout(config.history_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.history_dim,
            nhead=config.history_heads,
            dim_feedforward=config.history_dim * 4,
            batch_first=True,
            activation="gelu",
            dropout=config.history_dropout,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.history_layers)
        self.pool = MaskedAttentionPool(config.history_dim)
        self.output_norm = nn.LayerNorm(config.history_dim)

    def forward(self, batch) -> torch.Tensor:
        row_has_history = batch.history_mask.sum(dim=1) > 0
        if not bool(row_has_history.any().item()):
            return torch.zeros(
                batch.history_numeric.size(0),
                self.input_proj.out_features,
                device=batch.history_numeric.device,
                dtype=batch.history_numeric.dtype,
            )
        hidden = self.input_proj(batch.history_numeric)
        hidden = self.input_norm(hidden)
        hidden = self.position(hidden)
        hidden = self.dropout(hidden)
        safe_hidden = hidden
        safe_mask = batch.history_mask
        empty_rows = ~row_has_history
        if empty_rows.any():
            # Keep one zero token alive for empty-history rows so the transformer
            # does not see an all-masked sequence and emit NaNs.
            safe_hidden = hidden.clone()
            safe_mask = batch.history_mask.clone()
            safe_hidden[empty_rows, 0, :] = 0.0
            safe_mask[empty_rows, 0] = 1.0
        encoded = self.encoder(safe_hidden, src_key_padding_mask=safe_mask <= 0)
        pooled = self.pool(encoded, safe_mask)
        pooled = self.output_norm(pooled)
        if empty_rows.any():
            pooled = pooled.clone()
            pooled[empty_rows] = 0.0
        return pooled


class RecurrentHistoryEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.input_proj = nn.Linear(HISTORY_TOKEN_DIM, config.history_dim)
        self.input_norm = nn.LayerNorm(config.history_dim)
        self.position = LearnedPositionalEncoding(config.history_steps, config.history_dim)
        self.dropout = nn.Dropout(config.history_dropout)
        self.encoder = nn.GRU(
            input_size=config.history_dim,
            hidden_size=config.history_dim,
            batch_first=True,
        )

    def forward(self, batch) -> torch.Tensor:
        row_lengths = batch.history_mask.sum(dim=1).to(dtype=torch.long)
        if not bool((row_lengths > 0).any().item()):
            return torch.zeros(
                batch.history_numeric.size(0),
                self.input_proj.out_features,
                device=batch.history_numeric.device,
                dtype=batch.history_numeric.dtype,
            )
        hidden = self.input_proj(batch.history_numeric)
        hidden = self.input_norm(hidden)
        hidden = self.position(hidden)
        hidden = self.dropout(hidden)
        outputs = torch.zeros(
            batch.history_numeric.size(0),
            self.input_proj.out_features,
            device=batch.history_numeric.device,
            dtype=batch.history_numeric.dtype,
        )
        non_empty = row_lengths > 0
        # history_steps 很短（默认 8），直接走固定长度 GRU 前向并按有效步 gather，
        # 比 pack_padded_sequence(... lengths.cpu()) 更适合 rollout 热路径，
        # 也能避免每次推理额外的 CPU 参与。
        encoded, _ = self.encoder(hidden[non_empty])
        last_indices = (row_lengths[non_empty] - 1).clamp_min(0)
        gathered = encoded[
            torch.arange(encoded.size(0), device=encoded.device),
            last_indices,
        ]
        outputs[non_empty] = gathered.to(dtype=outputs.dtype)
        return outputs


class ResidualHistoryFusion(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.state_norm = nn.LayerNorm(config.hidden_dim)
        self.history_norm = nn.LayerNorm(config.history_dim)
        self.history_proj = nn.Linear(config.history_dim, config.hidden_dim)
        self.gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        nn.init.constant_(self.gate.bias, config.history_gate_bias)

    def forward(self, state_hidden: torch.Tensor, history_hidden: torch.Tensor) -> torch.Tensor:
        normed_state = self.state_norm(state_hidden)
        normed_history = self.history_norm(history_hidden)
        projected_history = self.history_proj(normed_history)
        gate = torch.sigmoid(self.gate(torch.cat([normed_state, projected_history], dim=-1)))
        return self.output_norm(state_hidden + gate * projected_history)


class ActionEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.id_embedding = nn.Embedding(config.id_hash_buckets, 32)
        self.encoder = MlpHiddenBlock(13 + ACTION_SEMANTIC_DIM + 64, config.hidden_dim, config.action_dim)

    def forward(self, batch) -> torch.Tensor:
        embedded = torch.cat(
            [
                self.id_embedding(batch.action_type_ids),
                self.id_embedding(batch.action_card_ids),
            ],
            dim=-1,
        )
        return self.encoder(torch.cat([batch.action_numeric, embedded], dim=-1))
