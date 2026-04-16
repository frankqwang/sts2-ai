"""Neural network encoder modules — shared building blocks for all policy networks.

This file contains ONLY nn.Module definitions used by both CombatPolicyValueNetwork
and FullRunPolicyNetworkV2. Feature engineering (state/action building, entity feature
extraction) lives in state_features.py.

Modules:
  EntityEmbeddings    - Learned embedding tables for cards, relics, potions, monsters, etc.
  SetEncoder          - Multi-head self-attention + masked mean pooling for variable-length sets
  SharedTrunk         - 2-layer MLP combining scalar + set representations
  ScreenHead          - Cross-attention head (trunk queries screen-specific entities)
  SimpleScreenHead    - Lightweight MLP for fixed-option screens (rest, event)
  BilinearActionScorer - Bilinear state-action scoring with masking
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vocab import Vocab

# Import constants and features from state_features (the canonical home)
from core.state_features import (
    # Dimension constants needed by NN modules
    SCALAR_DIM, CARD_AUX_DIM, ENEMY_AUX_DIM,
    MAX_DECK_SIZE, MAX_HAND_SIZE, MAX_RELICS, MAX_POTIONS,
    MAX_ENEMIES, MAX_ACTIONS,
    MAX_MAP_NODES, MAX_CARD_REWARDS, MAX_SHOP_ITEMS,
    MAX_EVENT_OPTIONS, MAX_REST_OPTIONS,
    NUM_FUNCTIONAL_TAGS, NUM_RELIC_TAGS,
    NODE_TYPES, ACTION_TYPES, TEXT_TOKEN_BUCKETS,
    MAP_ROUTE_DIM, SCREEN_TYPE_TO_IDX, COMBAT_SCREENS,
    MAX_EVENT_OPTIONS,
    # Dataclasses
    StructuredState, StructuredActions,
    # Feature building functions
    build_structured_state, build_structured_actions,
    # Auxiliary functions (re-exported for callers)
    _card_aux_features, _cached_card_encoding, _cached_card_idx,
    _cached_relic_idx, _cached_potion_idx, _cached_monster_idx,
    _enemy_aux_features, _lower, _safe_float, _safe_int,
    _extract_player, _compute_route_features, _extract_map_paths,
    _card_reward_cards_from_state_or_actions,
)


# ---------------------------------------------------------------------------
# Neural network modules
# ---------------------------------------------------------------------------

class EntityEmbeddings(nn.Module):
    """Learned embedding tables for all entity types."""

    def __init__(self, vocab: Vocab, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.card_embed = nn.Embedding(vocab.card_vocab_size, embed_dim, padding_idx=0)
        self.relic_embed = nn.Embedding(vocab.relic_vocab_size, embed_dim, padding_idx=0)
        self.potion_embed = nn.Embedding(vocab.potion_vocab_size, embed_dim, padding_idx=0)
        self.monster_embed = nn.Embedding(vocab.monster_vocab_size, embed_dim, padding_idx=0)
        # Map node type embedding (8 types)
        self.node_type_embed = nn.Embedding(len(NODE_TYPES), embed_dim)
        # Action type embedding
        self.action_type_embed = nn.Embedding(len(ACTION_TYPES), embed_dim)
        # Hashed text token embedding for boss-aware planning context
        self.text_token_embed = nn.Embedding(TEXT_TOKEN_BUCKETS, embed_dim, padding_idx=0)
        # Rest option embedding (7 options: rest, smith, recall, dig, lift, toke, other)
        self.rest_option_embed = nn.Embedding(8, embed_dim)
        # Event option embedding (simple learned per-index since events are contextual)
        self.event_option_embed = nn.Embedding(MAX_EVENT_OPTIONS, embed_dim)
        # Generic index embedding — distinguishes option 0/1/2/... for any action type
        self.index_embed = nn.Embedding(20, embed_dim)


class SetEncoder(nn.Module):
    """Encode a variable-length set of entity embeddings via self-attention + pool.

    Input: (B, max_len, dim) + mask (B, max_len)
    Output: (B, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4,
                 force_linear: bool = False):
        super().__init__()
        # Project input to attention dim if needed. `force_linear=True` skips
        # the nn.Identity fast-path even when input_dim == output_dim — used
        # by retrieval-enabled encoders in rl_policy_v2 / combat_nn so that
        # checkpoint partial-copy + [I|0] init can work uniformly.
        if force_linear or input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = nn.Identity()
        self.attn = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) entity representations
            mask: (B, L) bool — True for valid elements
        Returns:
            (B, output_dim) aggregated representation
        """
        x = self.proj(x)  # (B, L, output_dim)

        # Self-attention (key_padding_mask wants True for positions to IGNORE)
        attn_mask = ~mask  # invert
        # For fully-masked samples, unmask position 0 to prevent NaN in attention.
        # Results will be zeroed out via masked mean pooling anyway.
        # All ops are ONNX-trace-compatible (no in-place indexing, no data-dependent branching).
        fully_masked = attn_mask.all(dim=-1, keepdim=True)  # (B, 1)
        # Build unmask_first without in-place ops: [True, False, False, ...]
        unmask_first = torch.arange(x.shape[1], device=x.device).unsqueeze(0) == 0  # (1, L) bool
        safe_attn_mask = attn_mask & ~(fully_masked & unmask_first)

        attn_out, _ = self.attn(x, x, x, key_padding_mask=safe_attn_mask)
        attn_out = self.norm(attn_out + x)  # residual + norm

        # Masked mean pooling (fully-masked samples get zero naturally)
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return pooled  # (B, output_dim)


class SharedTrunk(nn.Module):
    """MLP that combines scalar features with set-encoded representations."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScreenHead(nn.Module):
    """Per-screen-type context encoder.

    Takes trunk output and screen-specific entity representations,
    produces screen context vector.
    """

    def __init__(self, trunk_dim: int, entity_dim: int, output_dim: int = 128,
                 num_heads: int = 4):
        super().__init__()
        # Cross-attention: trunk queries screen entities
        self.trunk_proj = nn.Linear(trunk_dim, output_dim)
        self.entity_proj = nn.Linear(entity_dim, output_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(
        self,
        trunk: torch.Tensor,         # (B, trunk_dim)
        entities: torch.Tensor,       # (B, L, entity_dim)
        mask: torch.Tensor,           # (B, L) bool
    ) -> torch.Tensor:
        """Returns (B, output_dim) screen context."""
        projected_trunk = self.trunk_proj(trunk)  # (B, output_dim)
        query = projected_trunk.unsqueeze(1)       # (B, 1, output_dim)
        kv = self.entity_proj(entities)             # (B, L, output_dim)

        attn_mask = ~mask  # True = ignore

        # For fully-masked samples, unmask position 0 to prevent NaN.
        # All ops are ONNX-trace-compatible (no in-place indexing, no branching).
        fully_masked = attn_mask.all(dim=-1, keepdim=True)  # (B, 1)
        unmask_first = torch.arange(entities.shape[1], device=entities.device).unsqueeze(0) == 0
        safe_attn_mask = attn_mask & ~(fully_masked & unmask_first)

        ctx, _ = self.cross_attn(query, kv, kv, key_padding_mask=safe_attn_mask)
        ctx = self.norm(ctx.squeeze(1) + projected_trunk)  # residual

        # Replace fully-masked samples with projected trunk (branchless)
        ctx = torch.where(fully_masked, projected_trunk, ctx)

        return ctx  # (B, output_dim)


class SimpleScreenHead(nn.Module):
    """Simple MLP head for screens with few fixed options (rest, event)."""

    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BilinearActionScorer(nn.Module):
    """Scores actions via bilinear interaction between state and action embeddings.

    score_i = state^T W action_i + b

    Uses manual matmul instead of nn.Bilinear for ONNX compatibility.
    Weight layout matches nn.Bilinear(state_dim, action_dim, 1) exactly.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Same parameterization as nn.Bilinear(state_dim, action_dim, 1)
        self.bilinear = nn.Bilinear(state_dim, action_dim, 1)

    def forward(
        self,
        state: torch.Tensor,     # (B, state_dim)
        actions: torch.Tensor,   # (B, A, action_dim)
        mask: torch.Tensor,      # (B, A) bool
    ) -> torch.Tensor:
        """Returns (B, A) logits, masked to -inf for invalid actions."""
        B, A, _ = actions.shape
        # Manual bilinear: score = state @ W @ action + bias
        # W shape: (1, state_dim, action_dim), bias shape: (1,)
        W = self.bilinear.weight  # (1, state_dim, action_dim)
        bias = self.bilinear.bias  # (1,)
        # state @ W: (B, 1, state_dim) @ (state_dim, action_dim) -> (B, 1, action_dim)
        sW = torch.matmul(state, W.squeeze(0))  # (B, action_dim)
        # sW * actions -> (B, A)
        scores = (sW.unsqueeze(1) * actions).sum(dim=-1) + bias  # (B, A)

        scores = scores.masked_fill(~mask, float("-inf"))
        return scores
