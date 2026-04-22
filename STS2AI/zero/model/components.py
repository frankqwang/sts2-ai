from __future__ import annotations

import torch
from torch import nn

from ..config import EncoderConfig
from ..features.extractor import ACTION_SEMANTIC_DIM, HAND_SEMANTIC_DIM, HISTORY_TOKEN_DIM, PILE_SEMANTIC_DIM, PLAYER_SEMANTIC_DIM, STATIC_NUMERIC_DIM


class MlpBlock(nn.Module):
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


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_steps: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_steps, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


class CurrentStateEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.id_embedding = nn.Embedding(config.id_hash_buckets, 32)

        player_input_dim = 6 + config.buff_slots + 3 + PLAYER_SEMANTIC_DIM
        enemy_input_dim = 6 + config.buff_slots + 64 + 64
        hand_input_dim = 9 + HAND_SEMANTIC_DIM + 32

        self.player_encoder = MlpBlock(player_input_dim, config.hidden_dim, config.hidden_dim)
        self.static_numeric_encoder = MlpBlock(STATIC_NUMERIC_DIM + 96, 128, 64)
        self.id_list_encoder = MlpBlock(32, 96, 64)
        self.id_list_pool = MaskedAttentionPool(64)
        self.buff_encoder = MlpBlock(33, 96, 64)
        self.buff_pool = MaskedAttentionPool(64)
        self.enemy_encoder = MlpBlock(enemy_input_dim, config.hidden_dim, config.hidden_dim)
        self.enemy_pool = MaskedAttentionPool(config.hidden_dim)
        self.hand_encoder = MlpBlock(hand_input_dim, config.hidden_dim, config.hidden_dim)
        self.hand_pool = MaskedAttentionPool(config.hidden_dim)
        self.pile_encoder = MlpBlock(8 + PILE_SEMANTIC_DIM, 64, 64)
        fusion_input_dim = config.hidden_dim + 64 + config.hidden_dim + config.hidden_dim + 64 + 64 + 64 + 64 + 64 + 64 + 64 + 64
        self.fusion = MlpBlock(fusion_input_dim, config.hidden_dim, config.hidden_dim)

    def forward(self, batch) -> torch.Tensor:
        player_hidden = self.player_encoder(batch.player_numeric)

        static_id_embed = self.id_embedding(batch.static_ids).flatten(start_dim=1)
        static_hidden = self.static_numeric_encoder(torch.cat([batch.static_numeric, static_id_embed], dim=-1))
        relic_hidden = self._pool_id_list(batch.relic_ids, batch.relic_mask)
        deck_hidden = self._pool_id_list(batch.deck_card_ids, batch.deck_card_mask)
        potion_hidden = self._pool_id_list(batch.potion_ids, batch.potion_mask)
        draw_hidden = self._pool_id_list(batch.draw_pile_ids, batch.draw_pile_mask)
        discard_hidden = self._pool_id_list(batch.discard_pile_ids, batch.discard_pile_mask)
        exhaust_hidden = self._pool_id_list(batch.exhaust_pile_ids, batch.exhaust_pile_mask)
        player_buff_hidden = self._pool_buffs(batch.player_buff_ids, batch.player_buff_values, batch.player_buff_mask)

        enemy_embed = torch.cat(
            [
                self.id_embedding(batch.enemy_ids),
                self.id_embedding(batch.enemy_intent_ids),
            ],
            dim=-1,
        )
        enemy_buff_hidden = self._pool_enemy_buffs(batch.enemy_buff_ids, batch.enemy_buff_values, batch.enemy_buff_mask)
        enemy_hidden = self.enemy_encoder(torch.cat([batch.enemy_numeric, enemy_embed, enemy_buff_hidden], dim=-1))
        enemy_pooled = self.enemy_pool(enemy_hidden, batch.enemy_mask)

        hand_embed = self.id_embedding(batch.hand_card_ids)
        hand_hidden = self.hand_encoder(torch.cat([batch.hand_numeric, hand_embed], dim=-1))
        hand_pooled = self.hand_pool(hand_hidden, batch.hand_mask)

        pile_hidden = self.pile_encoder(batch.pile_numeric)
        return self.fusion(
            torch.cat(
                [
                    player_hidden,
                    static_hidden,
                    enemy_pooled,
                    hand_pooled,
                    pile_hidden,
                    relic_hidden,
                    deck_hidden,
                    potion_hidden,
                    draw_hidden,
                    discard_hidden,
                    exhaust_hidden,
                    player_buff_hidden,
                ],
                dim=-1,
            )
        )

    def _pool_id_list(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.id_list_encoder(self.id_embedding(ids))
        return self.id_list_pool(hidden, mask)

    def _pool_buffs(self, ids: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.id_embedding(ids)
        hidden = self.buff_encoder(torch.cat([embedded, values.unsqueeze(-1)], dim=-1))
        return self.buff_pool(hidden, mask)

    def _pool_enemy_buffs(self, ids: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, enemy_count, buff_width = ids.shape
        flat_ids = ids.reshape(batch_size * enemy_count, buff_width)
        flat_values = values.reshape(batch_size * enemy_count, buff_width)
        flat_mask = mask.reshape(batch_size * enemy_count, buff_width)
        flat_hidden = self._pool_buffs(flat_ids, flat_values, flat_mask)
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
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden[non_empty],
            lengths=row_lengths[non_empty].cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, final_hidden = self.encoder(packed)
        outputs[non_empty] = final_hidden[-1].to(dtype=outputs.dtype)
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
        self.encoder = MlpBlock(13 + ACTION_SEMANTIC_DIM + 64, config.hidden_dim, config.action_dim)

    def forward(self, batch) -> torch.Tensor:
        embedded = torch.cat(
            [
                self.id_embedding(batch.action_type_ids),
                self.id_embedding(batch.action_card_ids),
            ],
            dim=-1,
        )
        return self.encoder(torch.cat([batch.action_numeric, embedded], dim=-1))
