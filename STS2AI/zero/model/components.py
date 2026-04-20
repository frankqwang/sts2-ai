from __future__ import annotations

import torch
from torch import nn

from ..config import EncoderConfig


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


class CurrentStateEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.id_embedding = nn.Embedding(config.id_hash_buckets, 32)

        player_input_dim = 6 + config.buff_slots + 3
        enemy_input_dim = 6 + config.buff_slots + 64
        hand_input_dim = 9 + 32

        self.player_encoder = MlpBlock(player_input_dim, config.hidden_dim, config.hidden_dim)
        self.static_numeric_encoder = MlpBlock(6 + 96, 128, 64)
        self.enemy_encoder = MlpBlock(enemy_input_dim, config.hidden_dim, config.hidden_dim)
        self.enemy_pool = MaskedAttentionPool(config.hidden_dim)
        self.hand_encoder = MlpBlock(hand_input_dim, config.hidden_dim, config.hidden_dim)
        self.hand_pool = MaskedAttentionPool(config.hidden_dim)
        self.pile_encoder = MlpBlock(8, 64, 64)
        self.fusion = MlpBlock(config.hidden_dim + 64 + config.hidden_dim + config.hidden_dim + 64, config.hidden_dim, config.hidden_dim)

    def forward(self, batch) -> torch.Tensor:
        player_hidden = self.player_encoder(batch.player_numeric)

        static_id_embed = self.id_embedding(batch.static_ids).flatten(start_dim=1)
        static_hidden = self.static_numeric_encoder(torch.cat([batch.static_numeric, static_id_embed], dim=-1))

        enemy_embed = torch.cat(
            [
                self.id_embedding(batch.enemy_ids),
                self.id_embedding(batch.enemy_intent_ids),
            ],
            dim=-1,
        )
        enemy_hidden = self.enemy_encoder(torch.cat([batch.enemy_numeric, enemy_embed], dim=-1))
        enemy_pooled = self.enemy_pool(enemy_hidden, batch.enemy_mask)

        hand_embed = self.id_embedding(batch.hand_card_ids)
        hand_hidden = self.hand_encoder(torch.cat([batch.hand_numeric, hand_embed], dim=-1))
        hand_pooled = self.hand_pool(hand_hidden, batch.hand_mask)

        pile_hidden = self.pile_encoder(batch.pile_numeric)
        return self.fusion(torch.cat([player_hidden, static_hidden, enemy_pooled, hand_pooled, pile_hidden], dim=-1))


class HistoryEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.input_proj = nn.Linear(17, config.history_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.history_dim,
            nhead=config.history_heads,
            dim_feedforward=config.history_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.history_layers)
        self.pool = MaskedAttentionPool(config.history_dim)

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
        if empty_rows.any():
            pooled = pooled.clone()
            pooled[empty_rows] = 0.0
        return pooled


class ActionEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.id_embedding = nn.Embedding(config.id_hash_buckets, 32)
        self.encoder = MlpBlock(13 + 64, config.hidden_dim, config.action_dim)

    def forward(self, batch) -> torch.Tensor:
        embedded = torch.cat(
            [
                self.id_embedding(batch.action_type_ids),
                self.id_embedding(batch.action_card_ids),
            ],
            dim=-1,
        )
        return self.encoder(torch.cat([batch.action_numeric, embedded], dim=-1))
