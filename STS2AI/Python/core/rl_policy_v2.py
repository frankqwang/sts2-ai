"""V2 Policy Network: entity embeddings + attention + pointer action scoring.

Architecture:
  Entity Embeddings (card/relic/potion/monster/node)
          ↓
  Set Encoders (deck, relics, potions — self-attention + pool)
          ↓
  Shared Trunk (concat scalars + set reprs → MLP → trunk_repr)
          ↓
  Screen Heads (cross-attention: trunk queries screen entities)
          ↓
  ├─ Value Head: V(s) = MLP(trunk, screen_ctx)
  └─ Action Scorer: bilinear(state_repr, action_embed) per legal action

Compared to V1 (flat 300-dim features → MLP):
  - Learns entity-level representations (specific cards, relics, enemies)
  - Attention captures set interactions (card synergies, build coherence)
  - Pointer-style action scoring: same card embedding whether in hand, reward, or shop
  - Variable-length action masking (no fixed 15-slot padding waste)
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from vocab import Vocab, load_vocab
from rl_encoder_v2 import (
    CARD_AUX_DIM,
    ENEMY_AUX_DIM,
    MAX_ACTIONS,
    MAX_CARD_REWARDS,
    MAX_DECK_SIZE,
    MAX_ENEMIES,
    MAX_HAND_SIZE,
    MAX_MAP_NODES,
    MAP_ROUTE_DIM,
    MAX_POTIONS,
    MAX_RELICS,
    MAX_REST_OPTIONS,
    MAX_SHOP_ITEMS,
    SCALAR_DIM,
    SCREEN_TYPE_TO_IDX,
    COMBAT_SCREENS,
    BilinearActionScorer,
    EntityEmbeddings,
    ScreenHead,
    SetEncoder,
    SharedTrunk,
    SimpleScreenHead,
    StructuredActions,
    StructuredState,
    build_structured_actions,
    build_structured_state,
)
from relic_tags import NUM_RELIC_TAGS
from symbolic_features_head import SymbolicFeaturesHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# V2 Network
# ---------------------------------------------------------------------------

class FullRunPolicyNetworkV2(nn.Module):
    """V2 actor-critic with entity embeddings and pointer action scoring."""

    def __init__(
        self,
        vocab: Vocab,
        embed_dim: int = 32,
        set_encoder_dim: int = 64,
        trunk_hidden: int = 256,
        trunk_output: int = 128,
        screen_head_dim: int = 128,
        num_attn_heads: int = 4,
        # --- Symbolic features head (sqlite-backed cross-attention) ---
        # When use_symbolic_features=True, a shared SymbolicFeaturesHead is
        # constructed and concatenated into every card/relic/monster/potion
        # encoder input, giving the policy a zero-shot prior over rare entities.
        # See plan: C:/Users/Administrator/.claude/plans/async-snacking-tome.md
        use_symbolic_features: bool = False,
        symbolic_proj_dim: int = 16,
        symbolic_db_path: Path | str | None = None,
        symbolic_head: SymbolicFeaturesHead | None = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.set_encoder_dim = set_encoder_dim
        self.trunk_output = trunk_output
        self.screen_head_dim = screen_head_dim

        # --- Entity embeddings ---
        self.entity_emb = EntityEmbeddings(vocab, embed_dim)

        # --- Symbolic features head (optional) ---
        # If a pre-built head is passed in, reuse it (shared with combat brain).
        # Otherwise construct one iff use_symbolic_features is set.
        if symbolic_head is not None:
            self.symbolic_head = symbolic_head
            self.use_symbolic_features = True
            self.symbolic_proj_dim = symbolic_head.proj_dim
        elif use_symbolic_features:
            self.symbolic_head = SymbolicFeaturesHead(
                vocab=vocab,
                db_path=symbolic_db_path,
                embed_dim=embed_dim,
                proj_dim=symbolic_proj_dim,
            )
            self.use_symbolic_features = True
            self.symbolic_proj_dim = symbolic_proj_dim
        else:
            self.symbolic_head = None
            self.use_symbolic_features = False
            self.symbolic_proj_dim = 0

        # Encoder input dims get a +proj_dim chunk at the end when retrieval
        # is enabled. The new columns sit AFTER the existing base+aux layout,
        # so checkpoint partial-copy keeps old cols bit-identical and the new
        # cols multiply zero (SymbolicFeaturesHead.out_proj is zero-init).
        sp = self.symbolic_proj_dim

        # --- Set encoders ---
        # Deck: card_embed + card_aux (+ symbolic) → set_encoder_dim
        # force_linear=True when retrieval is enabled so that even if
        # embed_dim + sp coincidentally equals set_encoder_dim (e.g.
        # 48 + 16 = 64 for potion_encoder), the encoder has a real
        # nn.Linear projection that our partial-copy + [I|0] repair can
        # handle uniformly. No-op when retrieval is off.
        fl = self.use_symbolic_features
        card_input_dim = embed_dim + CARD_AUX_DIM + sp
        self.deck_encoder = SetEncoder(card_input_dim, set_encoder_dim, num_attn_heads, force_linear=fl)
        self.relic_encoder = SetEncoder(embed_dim + NUM_RELIC_TAGS + sp, set_encoder_dim, num_attn_heads, force_linear=fl)
        self.potion_encoder = SetEncoder(embed_dim + sp, set_encoder_dim, num_attn_heads, force_linear=fl)

        # Combat set encoders
        self.hand_encoder = SetEncoder(card_input_dim, set_encoder_dim, num_attn_heads, force_linear=fl)
        self.enemy_encoder = SetEncoder(embed_dim + ENEMY_AUX_DIM + sp, set_encoder_dim, num_attn_heads, force_linear=fl)

        # --- Shared trunk ---
        # Input: scalars + deck_repr + relic_repr + potion_repr + hand_repr + enemy_repr
        # + screen_type_onehot
        trunk_input_dim = (
            SCALAR_DIM
            + set_encoder_dim * 5  # deck, relic, potion, hand, enemy
            + len(SCREEN_TYPE_TO_IDX)  # screen type one-hot
        )
        self.trunk = SharedTrunk(trunk_input_dim, trunk_hidden, trunk_output)

        # --- Screen heads ---
        # Map: cross-attention trunk → node embeddings
        self.map_head = ScreenHead(trunk_output, embed_dim + MAP_ROUTE_DIM, screen_head_dim, num_attn_heads)
        # Card reward: cross-attention trunk → reward card embeddings
        # (card_input_dim already includes +sp when retrieval enabled)
        self.card_reward_head = ScreenHead(trunk_output, card_input_dim, screen_head_dim, num_attn_heads)
        # Shop: cross-attention trunk → shop item embeddings
        # entity_embed + symbolic features (additive blend across modalities) + price
        shop_item_dim = embed_dim + sp + 1
        self.shop_head = ScreenHead(trunk_output, shop_item_dim, screen_head_dim, num_attn_heads)
        # Rest: simple MLP (trunk + option embeddings pooled)
        self.rest_head = SimpleScreenHead(trunk_output + embed_dim, screen_head_dim)
        # Event: simple MLP (trunk + option count)
        self.event_head = SimpleScreenHead(trunk_output + 1, screen_head_dim)
        # Combat: already encoded in trunk via hand/enemy set encoders
        self.combat_head = SimpleScreenHead(trunk_output, screen_head_dim)
        # Default fallback
        self.default_head = SimpleScreenHead(trunk_output, screen_head_dim)

        # --- Value heads (screen-specific) ---
        # Each screen type gets its own value head for lower variance.
        # Index mapping: combat(0-3)→"combat", map(4)→"map", card_reward(5)→"card_reward",
        # rest_site(6)→"campfire", shop(7)→"shop", event(8)→"event", others→"default"
        def _make_value_head():
            return nn.Sequential(
                nn.Linear(screen_head_dim, screen_head_dim // 2),
                nn.ReLU(),
                nn.Linear(screen_head_dim // 2, 1),
            )
        self.value_heads = nn.ModuleDict({
            "combat": _make_value_head(),
            "map": _make_value_head(),
            "card_reward": _make_value_head(),
            "campfire": _make_value_head(),
            "shop": _make_value_head(),
            "event": _make_value_head(),
            "default": _make_value_head(),
        })
        # Backward compat alias — old checkpoints have self.value_head
        self.value_head = self.value_heads["default"]
        # Screen idx → value head key
        self._screen_idx_to_vhead = {
            0: "combat", 1: "combat", 2: "combat", 3: "combat",
            4: "map", 5: "card_reward", 6: "campfire", 7: "shop",
            8: "event",
        }  # anything else → "default"

        # --- Deck quality auxiliary head ---
        # Predicts run outcome from deck + context (act/floor/scalars).
        # Used for: (1) auxiliary loss to train deck encoder,
        #           (2) learned counterfactual card evaluation at card_reward screens.
        # Input: deck_repr + scalars (act, floor, hp, gold, etc.)
        dq_input_dim = set_encoder_dim + SCALAR_DIM
        self.deck_quality_head = nn.Sequential(
            nn.Linear(dq_input_dim, set_encoder_dim),
            nn.ReLU(),
            nn.Linear(set_encoder_dim, set_encoder_dim // 2),
            nn.ReLU(),
            nn.Linear(set_encoder_dim // 2, 1),
            nn.Sigmoid(),  # output in [0, 1] — normalized run progress
        )

        # Upcoming boss adapter: zero-init preserves old checkpoint behavior
        self.boss_screen_adapter = nn.Linear(embed_dim, screen_head_dim, bias=False)
        nn.init.zeros_(self.boss_screen_adapter.weight)

        # Boss-aware build readiness auxiliary head
        readiness_input_dim = set_encoder_dim + SCALAR_DIM + embed_dim
        self.boss_readiness_head = nn.Sequential(
            nn.Linear(readiness_input_dim, set_encoder_dim),
            nn.ReLU(),
            nn.Linear(set_encoder_dim, set_encoder_dim // 2),
            nn.ReLU(),
            nn.Linear(set_encoder_dim // 2, 1),
            nn.Sigmoid(),
        )

        # --- Action scorer ---
        # Action representation: action_type_embed + target_entity_embed
        action_repr_dim = embed_dim * 2  # action_type + target_entity
        self.action_proj = nn.Linear(action_repr_dim, screen_head_dim)
        self.action_scorer = BilinearActionScorer(screen_head_dim, screen_head_dim)

        # --- Dueling: per-action advantage for card_reward screens ---
        # Enables Q(s,a) = V(s) + A(s,a) so different card choices get different values.
        # Without this, all cards at a card_reward screen share the same V(s).
        self.action_advantage = nn.Sequential(
            nn.Linear(screen_head_dim * 2, screen_head_dim),  # concat(ctx, action) → 128
            nn.ReLU(),
            nn.Linear(screen_head_dim, 1),
        )

        # --- Matchup-grounded option scorer ---
        # Trained by offline ranking data (combat simulation outcomes), NOT by PPO returns.
        # Used for card_reward and shop screens to learn simulation-grounded card values.
        # Zero-init final layer: new head starts at 0, preserving old checkpoint behavior.
        self.matchup_score_head = nn.Sequential(
            nn.Linear(screen_head_dim * 2, screen_head_dim),  # concat(ctx, action_repr)
            nn.ReLU(),
            nn.Linear(screen_head_dim, screen_head_dim // 2),
            nn.ReLU(),
            nn.Linear(screen_head_dim // 2, 1),
        )
        nn.init.zeros_(self.matchup_score_head[-1].weight)
        nn.init.zeros_(self.matchup_score_head[-1].bias)

        # --- Expanded encoder proj repair ---
        # When symbolic features are enabled, some encoders whose proj was
        # nn.Identity in the baseline (input dim equal to output dim) now have
        # a fresh nn.Linear. Baseline checkpoints have no saved weight for
        # those, so a naive load would leave them randomly initialized — which
        # would destroy the corresponding pathway. Fix each by initializing to
        # [I | 0]: first columns = identity (so baseline input passes through
        # unchanged), new columns = zero (so symbolic features contribute zero
        # at init, consistent with out_proj zero-init).
        if self.use_symbolic_features:
            self._repair_expanded_projs(sp)

    def _repair_expanded_projs(self, sp: int):
        """Initialize encoder projections expanded from nn.Identity to [I | 0].

        Only matters for encoders whose pre-retrieval input dim equaled the
        set_encoder_dim (identity fast-path). In the current baseline that's
        only `relic_encoder.proj` (input 64 == output 64). Other encoders
        already had nn.Linear so _safe_load_state_dict's partial-copy preserves
        their old columns on load — we rely on that path for them.

        This helper is also future-proof: it iterates over all set encoders
        and checks whether their proj matches the 'was Identity, now Linear'
        pattern (input_dim == output_dim + sp) and applies the fix.
        """
        encoders = [
            ("deck_encoder", self.deck_encoder),
            ("relic_encoder", self.relic_encoder),
            ("potion_encoder", self.potion_encoder),
            ("hand_encoder", self.hand_encoder),
            ("enemy_encoder", self.enemy_encoder),
        ]
        repaired = []
        for name, enc in encoders:
            proj = enc.proj
            if not isinstance(proj, nn.Linear):
                # Guards against a future refactor that replaces Linear with
                # something else — catches regressions loudly at init time
                # instead of silently corrupting weights at load time.
                raise RuntimeError(
                    f"FullRunPolicyNetworkV2: {name}.proj is {type(proj).__name__}, "
                    "expected nn.Linear after enabling symbolic features. "
                    "Check the encoder input-dim arithmetic."
                )
            out_dim = proj.out_features
            in_dim = proj.in_features
            # Was this Linear "born from Identity"? i.e., would in_dim == out_dim
            # have produced Identity under SetEncoder's pre-retrieval ctor logic?
            baseline_in_dim = in_dim - sp  # subtract the symbolic chunk we added
            if baseline_in_dim == out_dim:
                # Baseline would have been Identity. Initialize to [I | 0].
                with torch.no_grad():
                    proj.weight.zero_()
                    proj.weight[:, :baseline_in_dim] = torch.eye(out_dim)
                    proj.bias.zero_()
                repaired.append(name)
        if repaired:
            logger.info(
                "SymbolicFeaturesHead: repaired %d encoder proj layers to [I|0] "
                "(baseline had Identity fast-path): %s",
                len(repaired), ", ".join(repaired),
            )

    def _encode_state(
        self,
        ss: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode structured state into trunk repr, screen context, deck repr, and boss embedding.

        Args:
            ss: dict of batched tensors from StructuredState

        Returns:
            trunk_repr: (B, trunk_output)
            screen_ctx: (B, screen_head_dim)
            deck_repr: (B, set_encoder_dim) — for auxiliary deck quality prediction
            boss_emb: (B, embed_dim) — hashed upcoming boss embedding
        """
        B = ss["scalars"].shape[0]
        device = ss["scalars"].device

        # --- Entity embeddings ---
        # deck / hand: [card_embed | card_aux | (optional) symbolic]
        deck_base = self.entity_emb.card_embed(ss["deck_ids"])  # (B, MAX_DECK, embed)
        deck_parts = [deck_base, ss["deck_aux"]]
        if self.symbolic_head is not None:
            deck_parts.append(self.symbolic_head.card(ss["deck_ids"], deck_base))
        deck_emb = torch.cat(deck_parts, dim=-1)

        relic_base = self.entity_emb.relic_embed(ss["relic_ids"])  # (B, MAX_RELICS, embed)
        relic_parts = [relic_base, ss["relic_aux"]]
        if self.symbolic_head is not None:
            relic_parts.append(self.symbolic_head.relic(ss["relic_ids"], relic_base))
        relic_emb = torch.cat(relic_parts, dim=-1)

        potion_base = self.entity_emb.potion_embed(ss["potion_ids"])  # (B, MAX_POTIONS, embed)
        if self.symbolic_head is not None:
            potion_emb = torch.cat(
                [potion_base, self.symbolic_head.potion(ss["potion_ids"], potion_base)],
                dim=-1,
            )
        else:
            potion_emb = potion_base

        hand_base = self.entity_emb.card_embed(ss["hand_ids"])
        hand_parts = [hand_base, ss["hand_aux"]]
        if self.symbolic_head is not None:
            hand_parts.append(self.symbolic_head.card(ss["hand_ids"], hand_base))
        hand_emb = torch.cat(hand_parts, dim=-1)

        enemy_base = self.entity_emb.monster_embed(ss["enemy_ids"])
        enemy_parts = [enemy_base, ss["enemy_aux"]]
        if self.symbolic_head is not None:
            enemy_parts.append(self.symbolic_head.monster(ss["enemy_ids"], enemy_base))
        enemy_emb = torch.cat(enemy_parts, dim=-1)

        boss_emb = self.entity_emb.text_token_embed(ss["next_boss_idx"])

        # --- Set encoding ---
        deck_repr = self.deck_encoder(deck_emb, ss["deck_mask"])    # (B, set_dim)
        relic_repr = self.relic_encoder(relic_emb, ss["relic_mask"])
        potion_repr = self.potion_encoder(potion_emb, ss["potion_mask"])
        hand_repr = self.hand_encoder(hand_emb, ss["hand_mask"])
        enemy_repr = self.enemy_encoder(enemy_emb, ss["enemy_mask"])

        # --- Screen type one-hot ---
        screen_onehot = F.one_hot(
            ss["screen_type_idx"], num_classes=len(SCREEN_TYPE_TO_IDX),
        ).float()  # (B, num_screen_types)

        # --- Trunk ---
        trunk_input = torch.cat([
            ss["scalars"],
            deck_repr, relic_repr, potion_repr,
            hand_repr, enemy_repr,
            screen_onehot,
        ], dim=-1)
        trunk_repr = self.trunk(trunk_input)  # (B, trunk_output)

        # --- Screen head (per sample, but batched) ---
        # We compute ALL screen heads and select per-sample.
        # This is slightly wasteful but avoids complex per-sample branching.
        # In practice the extra compute is negligible vs the attention costs.

        # Map head
        map_node_emb = self.entity_emb.node_type_embed(ss["map_node_types"])  # (B, MAX_MAP, embed)
        # Concatenate route lookahead features (min_elite, max_shop, max_rest, avg_monster, rows_to_boss)
        map_route = ss.get("map_route_features")
        if map_route is not None:
            map_node_emb = torch.cat([map_node_emb, map_route.float()], dim=-1)  # (B, MAX_MAP, embed+5)
        else:
            map_node_emb = torch.cat([map_node_emb, torch.zeros(*map_node_emb.shape[:2], MAP_ROUTE_DIM, device=map_node_emb.device)], dim=-1)
        map_ctx = self.map_head(trunk_repr, map_node_emb, ss["map_node_mask"])

        # Card reward head
        reward_base = self.entity_emb.card_embed(ss["reward_card_ids"])
        reward_parts = [reward_base, ss["reward_card_aux"]]
        if self.symbolic_head is not None:
            reward_parts.append(self.symbolic_head.card(ss["reward_card_ids"], reward_base))
        reward_emb = torch.cat(reward_parts, dim=-1)
        reward_ctx = self.card_reward_head(trunk_repr, reward_emb, ss["reward_card_mask"])

        # Shop head
        shop_card_emb = self.entity_emb.card_embed(ss["shop_card_ids"])
        shop_relic_emb = self.entity_emb.relic_embed(ss["shop_relic_ids"])
        shop_potion_emb = self.entity_emb.potion_embed(ss["shop_potion_ids"])
        # Combine: additive blend — only one modality is non-pad per slot, so
        # pad rows contribute zero to the sum.
        shop_entity_emb = shop_card_emb + shop_relic_emb + shop_potion_emb
        if self.symbolic_head is not None:
            # Each modality contributes its own symbolic features; pad rows'
            # symbolic output is zero (fully-masked rows zeroed in _attend),
            # so additive blend is valid in symbolic space too.
            shop_card_sym = self.symbolic_head.card(ss["shop_card_ids"], shop_card_emb)
            shop_relic_sym = self.symbolic_head.relic(ss["shop_relic_ids"], shop_relic_emb)
            shop_potion_sym = self.symbolic_head.potion(ss["shop_potion_ids"], shop_potion_emb)
            shop_sym = shop_card_sym + shop_relic_sym + shop_potion_sym
            shop_entity_emb = torch.cat([shop_entity_emb, shop_sym], dim=-1)
        shop_entity_emb = torch.cat([shop_entity_emb, ss["shop_prices"].unsqueeze(-1)], dim=-1)
        shop_ctx = self.shop_head(trunk_repr, shop_entity_emb, ss["shop_mask"])

        # Rest head
        rest_emb = self.entity_emb.rest_option_embed(ss["rest_option_ids"])  # (B, MAX_REST, embed)
        rest_mask_f = ss["rest_option_mask"].unsqueeze(-1).float()
        rest_pooled = (rest_emb * rest_mask_f).sum(dim=1) / rest_mask_f.sum(dim=1).clamp(min=1)
        rest_ctx = self.rest_head(torch.cat([trunk_repr, rest_pooled], dim=-1))

        # Event head
        event_count = ss["event_option_count"].unsqueeze(-1).float() / 5.0
        event_ctx = self.event_head(torch.cat([trunk_repr, event_count], dim=-1))

        # Combat head
        combat_ctx = self.combat_head(trunk_repr)

        # Default head
        default_ctx = self.default_head(trunk_repr)

        # --- Select screen context per sample ---
        # Stack all contexts: (B, num_heads, screen_head_dim)
        # Indices: map=4, card_reward=5, rest_site=6, shop=7, event=8
        # combat screens (0-3), combat_rewards=9, treasure=10, other=11
        all_ctx = torch.stack([
            combat_ctx,     # 0: combat
            combat_ctx,     # 1: monster
            combat_ctx,     # 2: elite
            combat_ctx,     # 3: boss
            map_ctx,        # 4: map
            reward_ctx,     # 5: card_reward
            rest_ctx,       # 6: rest_site
            shop_ctx,       # 7: shop
            event_ctx,      # 8: event
            default_ctx,    # 9: combat_rewards
            default_ctx,    # 10: treasure
            default_ctx,    # 11: other
        ], dim=1)  # (B, 12, screen_head_dim)

        # Gather per-sample screen context
        idx = ss["screen_type_idx"].clamp(0, all_ctx.shape[1] - 1)  # (B,)
        idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.screen_head_dim)
        screen_ctx = all_ctx.gather(1, idx_expanded).squeeze(1)  # (B, screen_head_dim)
        screen_ctx = screen_ctx + self.boss_screen_adapter(boss_emb)

        return trunk_repr, screen_ctx, deck_repr, boss_emb

    def _compute_screen_values(
        self,
        screen_ctx: torch.Tensor,
        screen_type_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute values using screen-specific value heads.

        Routes each sample to its appropriate value head based on screen_type_idx.
        """
        B = screen_ctx.shape[0]
        device = screen_ctx.device

        # Compute values from all heads: (num_heads, B, 1) → gather per-sample
        head_keys = ["combat", "map", "card_reward", "campfire", "shop", "event", "default"]
        all_vals = torch.stack([
            self.value_heads[k](screen_ctx).squeeze(-1) for k in head_keys
        ], dim=1)  # (B, 7)

        # Map screen_type_idx → head index (0..6)
        idx_map = torch.full((max(len(SCREEN_TYPE_TO_IDX), 18),), 6, dtype=torch.long, device=device)  # default=6
        for sidx, hkey in self._screen_idx_to_vhead.items():
            idx_map[sidx] = head_keys.index(hkey)

        head_idx = idx_map[screen_type_idx.clamp(0, len(idx_map) - 1)]  # (B,)
        values = all_vals.gather(1, head_idx.unsqueeze(-1)).squeeze(-1)  # (B,)
        return values

    def _encode_actions(
        self,
        sa: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode structured actions into action representations.

        Each action is represented as: action_type_embed + target_entity_embed + target_enemy_embed
        - Card actions use card embedding as target
        - Map actions use node type embedding
        - Combat actions with enemy targets use monster embedding
        - The three are combined additively (only one is typically non-zero)

        Returns: (B, MAX_ACTIONS, screen_head_dim)
        """
        # Action type embedding
        atype_emb = self.entity_emb.action_type_embed(sa["action_type_ids"])  # (B, A, embed)

        # Card/entity target embedding
        card_target_emb = self.entity_emb.card_embed(sa["target_card_ids"])  # (B, A, embed)
        node_target_emb = self.entity_emb.node_type_embed(sa["target_node_types"])  # (B, A, embed)
        enemy_target_emb = self.entity_emb.monster_embed(sa["target_enemy_ids"])  # (B, A, embed)

        # Index embedding — distinguishes option 0 vs 1 vs 2 for events/rest/rewards
        idx_emb = self.entity_emb.index_embed(
            sa["target_indices"].clamp(0, 19))  # (B, A, embed), max 20 options

        # Combine targets: use card embed when available, else node, else enemy
        # (they're mutually exclusive in practice, so additive works)
        has_card = (sa["target_card_ids"] > 0).unsqueeze(-1).float()
        has_node = (sa["target_node_types"] > 0).unsqueeze(-1).float()
        has_enemy = (sa["target_enemy_ids"] > 0).unsqueeze(-1).float()
        has_specific = has_card + has_node + has_enemy  # >0 if any specific target
        has_specific = has_specific.clamp(0, 1)

        target_emb = (card_target_emb * has_card
                      + node_target_emb * has_node * (1 - has_card)
                      + enemy_target_emb * has_enemy
                      + idx_emb * (1 - has_specific))  # index fallback for generic actions

        # Combine action_type + target
        action_repr = torch.cat([atype_emb, target_emb], dim=-1)  # (B, A, embed*2)
        action_repr = self.action_proj(action_repr)  # (B, A, screen_head_dim)

        return action_repr

    def _compute_action_advantages(
        self,
        screen_ctx: torch.Tensor,
        action_repr: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean-centered per-action advantages (dueling head).

        Returns: (B, MAX_ACTIONS) — mean-centered advantages, masked positions = 0.
        """
        ctx_expanded = screen_ctx.unsqueeze(1).expand_as(action_repr)  # (B, A, dim)
        combined = torch.cat([ctx_expanded, action_repr], dim=-1)  # (B, A, dim*2)
        raw_adv = self.action_advantage(combined).squeeze(-1)  # (B, A)
        raw_adv = raw_adv.masked_fill(~action_mask, 0.0)

        # Mean-center over valid actions: A(s,a) - mean_a'(A(s,a'))
        n_valid = action_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
        mean_adv = (raw_adv * action_mask.float()).sum(dim=-1, keepdim=True) / n_valid
        return raw_adv - mean_adv

    def forward(
        self,
        state_tensors: dict[str, torch.Tensor],
        action_tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, MAX_ACTIONS) — raw scores, masked to -inf
            values: (B,) — state values V(s)
            deck_quality: (B,) — auxiliary deck quality prediction in [0, 1]
            boss_readiness: (B,) — boss-aware build readiness prediction in [0, 1]
            action_advantages: (B, MAX_ACTIONS) — per-action advantages A(s,a)
        """
        _, screen_ctx, deck_repr, boss_emb = self._encode_state(state_tensors)
        action_repr = self._encode_actions(action_tensors)
        action_mask = action_tensors["action_mask"]

        # Value (screen-specific heads)
        values = self._compute_screen_values(screen_ctx, state_tensors["screen_type_idx"])

        # Action scores via bilinear
        logits = self.action_scorer(screen_ctx, action_repr, action_mask)  # (B, A)

        # Per-action advantages (dueling)
        action_advantages = self._compute_action_advantages(
            screen_ctx, action_repr, action_mask)  # (B, A)

        # Deck quality auxiliary prediction (deck_repr + scalars context)
        dq_input = torch.cat([deck_repr, state_tensors["scalars"]], dim=-1)
        deck_quality = self.deck_quality_head(dq_input).squeeze(-1)  # (B,)
        readiness_input = torch.cat([deck_repr, state_tensors["scalars"], boss_emb], dim=-1)
        boss_readiness = self.boss_readiness_head(readiness_input).squeeze(-1)  # (B,)

        return logits, values, deck_quality, boss_readiness, action_advantages

    def get_action_and_value(
        self,
        state_tensors: dict[str, torch.Tensor],
        action_tensors: dict[str, torch.Tensor],
        action_idx: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value.

        For card_reward screens (screen_type_idx=5), returns Q(s,a) = V(s) + A(s,a)
        instead of V(s), so different card choices get different value estimates.
        """
        logits, values, _deck_quality, _boss_readiness, action_adv = self.forward(
            state_tensors, action_tensors)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action_idx is None:
            if deterministic:
                action_idx = logits.argmax(dim=-1)
            else:
                action_idx = dist.sample()

        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        # Dueling: for card_reward screens, use Q(s,a) = V(s) + A(s, a_taken)
        is_card_reward = (state_tensors["screen_type_idx"] == 5)
        if is_card_reward.any():
            a_adv = action_adv.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (B,)
            values = values.clone()
            values[is_card_reward] = values[is_card_reward] + a_adv[is_card_reward]

        return action_idx, log_prob, entropy, values

    @torch.no_grad()
    def evaluate_deck_quality(
        self,
        scalars: np.ndarray,
        deck_ids: np.ndarray,
        deck_aux: np.ndarray,
        deck_mask: np.ndarray,
    ) -> float:
        """Evaluate deck quality for a given deck + context.

        Used by counterfactual scoring to compute learned card marginal value:
          card_value = evaluate_deck_quality(deck + [card]) - evaluate_deck_quality(deck)

        Args:
            scalars: (SCALAR_DIM,) — act, floor, hp, gold, etc.
            deck_ids: (MAX_DECK_SIZE,) — card vocab indices
            deck_aux: (MAX_DECK_SIZE, CARD_AUX_DIM) — card auxiliary features
            deck_mask: (MAX_DECK_SIZE,) — boolean mask

        Returns:
            Quality score in [0, 1].
        """
        self.eval()
        device = next(self.parameters()).device

        # Batch dim
        s_t = torch.tensor(scalars, dtype=torch.float32, device=device).unsqueeze(0)
        d_ids = torch.tensor(deck_ids, dtype=torch.long, device=device).unsqueeze(0)
        d_aux = torch.tensor(deck_aux, dtype=torch.float32, device=device).unsqueeze(0)
        d_mask = torch.tensor(deck_mask, dtype=torch.bool, device=device).unsqueeze(0)

        # Encode deck
        deck_base = self.entity_emb.card_embed(d_ids)
        deck_parts = [deck_base, d_aux]
        if self.symbolic_head is not None:
            deck_parts.append(self.symbolic_head.card(d_ids, deck_base))
        deck_emb = torch.cat(deck_parts, dim=-1)
        deck_repr = self.deck_encoder(deck_emb, d_mask)  # (1, set_encoder_dim)

        # Deck quality with context
        dq_input = torch.cat([deck_repr, s_t], dim=-1)
        quality = self.deck_quality_head(dq_input).squeeze(-1).item()
        return quality

    def compute_deck_repr(
        self,
        state_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute deck_repr from state tensors for combat brain bridge.

        Returns:
            deck_repr: (B, set_encoder_dim) — deck embedding for build_plan_z.
        """
        with torch.no_grad():
            deck_base = self.entity_emb.card_embed(state_tensors["deck_ids"])
            deck_parts = [deck_base, state_tensors["deck_aux"]]
            if self.symbolic_head is not None:
                deck_parts.append(self.symbolic_head.card(state_tensors["deck_ids"], deck_base))
            deck_emb = torch.cat(deck_parts, dim=-1)
            return self.deck_encoder(deck_emb, state_tensors["deck_mask"])

    def compute_matchup_scores(
        self,
        state_tensors: dict[str, torch.Tensor],
        action_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute matchup-grounded scores for each legal action.

        Trained by offline ranking data from combat simulation outcomes.
        Separate from policy logits (action_scorer) and dueling advantages.

        Args:
            state_tensors: structured state batch
            action_tensors: structured action batch

        Returns:
            (B, MAX_ACTIONS) scores, invalid positions masked to -1e9.
        """
        _, screen_ctx, _, _ = self._encode_state(state_tensors)
        action_repr = self._encode_actions(action_tensors)
        action_mask = action_tensors["action_mask"]

        ctx_expanded = screen_ctx.unsqueeze(1).expand_as(action_repr)  # (B, A, dim)
        combined = torch.cat([ctx_expanded, action_repr], dim=-1)  # (B, A, dim*2)
        scores = self.matchup_score_head(combined).squeeze(-1)  # (B, A)
        return scores.masked_fill(~action_mask, -1e9)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Structured Rollout Buffer
# ---------------------------------------------------------------------------

@dataclass
class StructuredRolloutBuffer:
    """Stores episode data with structured state/action tensors for V2."""

    # Raw structured data (numpy)
    states: list[dict[str, np.ndarray]] = field(default_factory=list)
    actions_data: list[dict[str, np.ndarray]] = field(default_factory=list)

    # PPO scalars
    action_indices: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    # Deck quality auxiliary targets (final floor reached, set after episode)
    floor_targets: list[float] = field(default_factory=list)
    boss_readiness_targets: list[float] = field(default_factory=list)

    # Computed after collection
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def add(
        self,
        state: StructuredState,
        actions: StructuredActions,
        action_idx: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        boss_readiness_target: float = 0.0,
    ) -> None:
        self.states.append(_structured_state_to_numpy_dict(state))
        self.actions_data.append(_structured_actions_to_numpy_dict(actions))
        self.action_indices.append(action_idx)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        # Floor target placeholder — set by set_floor_targets() after episode
        self.floor_targets.append(0.0)
        self.boss_readiness_targets.append(float(boss_readiness_target))

    def set_floor_targets(self, final_floor: float) -> None:
        """Set deck quality target for all steps in the last episode.

        Called after an episode finishes. Sets all steps since the last
        done=True to the normalized final floor (floor / 20.0, clamped to [0,1]).
        """
        target = min(final_floor / 20.0, 1.0)
        for i in range(len(self.floor_targets) - 1, -1, -1):
            self.floor_targets[i] = target
            if i > 0 and self.dones[i - 1]:
                break  # stop at previous episode boundary

    def compute_gae(self, gamma: float = 0.999, lam: float = 0.95) -> None:
        n = len(self.rewards)
        self.advantages = [0.0] * n
        self.returns = [0.0] * n
        last_gae = 0.0

        for t in reversed(range(n)):
            if self.dones[t]:
                next_value = 0.0
                last_gae = 0.0
            elif t + 1 < n:
                next_value = self.values[t + 1]
            else:
                next_value = 0.0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * lam * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def to_tensors(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Convert buffer to tensors for training."""
        n = len(self.rewards)

        # Stack state tensors
        state_tensors = {}
        if n > 0:
            keys = self.states[0].keys()
            for key in keys:
                arrays = [s[key] for s in self.states]
                if arrays[0].dtype in (np.int64, np.int32):
                    state_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.long)
                elif arrays[0].dtype == bool:
                    state_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.bool)
                else:
                    state_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.float32)

        # Stack action tensors
        action_tensors = {}
        if n > 0:
            keys = self.actions_data[0].keys()
            for key in keys:
                arrays = [a[key] for a in self.actions_data]
                if arrays[0].dtype in (np.int64, np.int32):
                    action_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.long)
                elif arrays[0].dtype == bool:
                    action_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.bool)
                else:
                    action_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.float32)

        return {
            "state_tensors": state_tensors,
            "action_tensors": action_tensors,
            "actions": torch.tensor(self.action_indices, dtype=torch.long),
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "advantages": torch.tensor(self.advantages, dtype=torch.float32),
            "returns": torch.tensor(self.returns, dtype=torch.float32),
            "floor_targets": torch.tensor(self.floor_targets, dtype=torch.float32),
            "boss_readiness_targets": torch.tensor(self.boss_readiness_targets, dtype=torch.float32),
        }

    def clear(self) -> None:
        for attr in ("states", "actions_data", "action_indices",
                      "log_probs", "rewards", "values", "dones",
                      "advantages", "returns", "floor_targets", "boss_readiness_targets"):
            getattr(self, attr).clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Numpy conversion helpers
# ---------------------------------------------------------------------------

def _structured_state_to_numpy_dict(ss: StructuredState) -> dict[str, np.ndarray]:
    """Convert StructuredState dataclass to flat dict of numpy arrays."""
    return {
        "scalars": ss.scalars,
        "deck_ids": ss.deck_ids,
        "deck_aux": ss.deck_aux,
        "deck_mask": ss.deck_mask,
        "relic_ids": ss.relic_ids,
        "relic_aux": ss.relic_aux,
        "relic_mask": ss.relic_mask,
        "potion_ids": ss.potion_ids,
        "potion_mask": ss.potion_mask,
        "hand_ids": ss.hand_ids,
        "hand_aux": ss.hand_aux,
        "hand_mask": ss.hand_mask,
        "enemy_ids": ss.enemy_ids,
        "enemy_aux": ss.enemy_aux,
        "enemy_mask": ss.enemy_mask,
        "screen_type_idx": np.array(ss.screen_type_idx, dtype=np.int64),
        "next_boss_idx": np.array(ss.next_boss_idx, dtype=np.int64),
        "map_node_types": ss.map_node_types,
        "map_node_mask": ss.map_node_mask,
        "map_route_features": ss.map_route_features,
        "reward_card_ids": ss.reward_card_ids,
        "reward_card_aux": ss.reward_card_aux,
        "reward_card_mask": ss.reward_card_mask,
        "shop_card_ids": ss.shop_card_ids,
        "shop_relic_ids": ss.shop_relic_ids,
        "shop_potion_ids": ss.shop_potion_ids,
        "shop_prices": ss.shop_prices,
        "shop_mask": ss.shop_mask,
        "event_option_count": np.array(ss.event_option_count, dtype=np.int64),
        "rest_option_ids": ss.rest_option_ids,
        "rest_option_mask": ss.rest_option_mask,
    }


def _structured_actions_to_numpy_dict(sa: StructuredActions) -> dict[str, np.ndarray]:
    """Convert StructuredActions dataclass to flat dict of numpy arrays."""
    return {
        "action_type_ids": sa.action_type_ids,
        "target_card_ids": sa.target_card_ids,
        "target_enemy_ids": sa.target_enemy_ids,
        "target_node_types": sa.target_node_types,
        "target_indices": sa.target_indices,
        "action_mask": sa.action_mask,
    }


# ---------------------------------------------------------------------------
# PPO Trainer for V2
# ---------------------------------------------------------------------------

class PPOTrainerV2:
    """PPO trainer adapted for V2 structured inputs."""

    def __init__(
        self,
        network: FullRunPolicyNetworkV2,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.05,
        deck_quality_coeff: float = 0.1,
        boss_readiness_coeff: float = 0.05,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.deck_quality_coeff = deck_quality_coeff
        self.boss_readiness_coeff = boss_readiness_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    def update(
        self,
        buffer: StructuredRolloutBuffer,
        per_screen_adv_norm: bool = True,
        weighted_screen_sampling: bool = True,
        teacher_logits: list | None = None,
        kl_beta: float = 0.0,
    ) -> dict[str, float]:
        """Run PPO update on structured buffer data.

        Args:
            per_screen_adv_norm: Normalize advantages per screen type (Phase 1B).
            weighted_screen_sampling: Weight minibatch sampling by screen frequency (Phase 1C).
            teacher_logits: Optional teacher distribution per step for KL warm-start (Phase 4).
            kl_beta: KL loss coefficient for warm-start (0 = disabled).
        """
        data = buffer.to_tensors()
        n = len(buffer)
        if n == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "buffer_size": 0}

        # Move all tensors to network's device
        device = next(self.network.parameters()).device
        state_tensors = {k: v.to(device) for k, v in data["state_tensors"].items()}
        action_tensors = {k: v.to(device) for k, v in data["action_tensors"].items()}
        for k in ("actions", "old_log_probs", "advantages", "returns", "floor_targets", "boss_readiness_targets"):
            if k in data:
                data[k] = data[k].to(device)

        # Normalize advantages (per-screen or global)
        adv = data["advantages"]
        if per_screen_adv_norm and "screen_type_idx" in state_tensors:
            adv = self._normalize_advantages_per_screen(adv, state_tensors["screen_type_idx"])
        elif adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Compute sampling weights for rare screen upsampling (Phase 1C)
        sample_weights = None
        if weighted_screen_sampling and "screen_type_idx" in state_tensors:
            sample_weights = self._compute_screen_sample_weights(state_tensors["screen_type_idx"])

        # Prepare teacher logits tensor if provided (Phase 4)
        teacher_t = None
        if teacher_logits is not None and kl_beta > 0:
            teacher_t = torch.tensor(
                np.stack(teacher_logits), dtype=torch.float32, device=device
            )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_deck_loss = 0.0
        total_boss_readiness_loss = 0.0
        update_count = 0

        for _epoch in range(self.ppo_epochs):
            # Phase 1C: weighted sampling by screen frequency
            if sample_weights is not None:
                indices = torch.multinomial(sample_weights, n, replacement=True)
            else:
                indices = torch.randperm(n)

            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                mb_idx = indices[start:end]

                # Slice state tensors
                mb_state = {k: v[mb_idx] for k, v in state_tensors.items()}
                mb_action = {k: v[mb_idx] for k, v in action_tensors.items()}
                mb_actions = data["actions"][mb_idx]
                mb_old_lp = data["old_log_probs"][mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = data["returns"][mb_idx]
                mb_floor = data["floor_targets"][mb_idx]
                mb_boss_ready = data["boss_readiness_targets"][mb_idx]

                # Forward (get full output including deck quality and action advantages)
                logits, new_values, deck_quality, boss_readiness, new_action_adv = self.network.forward(
                    mb_state, mb_action,
                )

                # Dueling: for card_reward screens, use Q(s,a) = V(s) + A(s, a_taken)
                is_cr = (mb_state["screen_type_idx"] == 5)
                if is_cr.any():
                    cr_adv = new_action_adv[is_cr].gather(
                        1, mb_actions[is_cr].unsqueeze(1)).squeeze(1)
                    new_values = new_values.clone()
                    new_values[is_cr] = new_values[is_cr] + cr_adv

                # Recompute log_prob and entropy from logits
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_lp = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                # PPO losses
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                     1 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, mb_ret)
                entropy_loss = -entropy.mean()

                # Deck quality auxiliary loss
                deck_loss = F.mse_loss(deck_quality, mb_floor)
                boss_readiness_loss = F.mse_loss(boss_readiness, mb_boss_ready)

                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                    + self.deck_quality_coeff * deck_loss
                    + self.boss_readiness_coeff * boss_readiness_loss
                )

                # Phase 4: KL warm-start from heuristic teacher
                if teacher_t is not None and kl_beta > 0:
                    mb_teacher = teacher_t[mb_idx]
                    # Only apply KL to samples with valid teacher signal
                    has_teacher = mb_teacher.sum(dim=-1) > 0.5
                    if has_teacher.any():
                        student_log_probs = F.log_softmax(logits[has_teacher], dim=-1)
                        teacher_probs = mb_teacher[has_teacher]
                        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                        loss = loss + kl_beta * kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss.item())
                total_deck_loss += deck_loss.item()
                total_boss_readiness_loss += boss_readiness_loss.item()
                update_count += 1

        return {
            "policy_loss": total_policy_loss / max(1, update_count),
            "value_loss": total_value_loss / max(1, update_count),
            "entropy": total_entropy / max(1, update_count),
            "deck_quality_loss": total_deck_loss / max(1, update_count),
            "boss_readiness_loss": total_boss_readiness_loss / max(1, update_count),
            "buffer_size": n,
        }

    @staticmethod
    def _normalize_advantages_per_screen(
        adv: torch.Tensor, screen_type_idx: torch.Tensor, min_group_size: int = 4,
    ) -> torch.Tensor:
        """Normalize advantages per screen type. Falls back to global for small groups."""
        out = adv.clone()
        global_mean = adv.mean()
        global_std = adv.std().clamp(min=1e-8)
        unique_screens = screen_type_idx.unique()

        for st in unique_screens:
            mask = screen_type_idx == st
            count = mask.sum().item()
            if count >= min_group_size:
                group = adv[mask]
                std = group.std()
                if std > 1e-8:
                    out[mask] = (group - group.mean()) / (std + 1e-8)
                else:
                    out[mask] = 0.0
            else:
                # Fallback to global normalization for rare screens
                out[mask] = (adv[mask] - global_mean) / global_std
        return out

    @staticmethod
    def _compute_screen_sample_weights(screen_type_idx: torch.Tensor) -> torch.Tensor:
        """Compute per-sample weights inversely proportional to screen frequency.

        Ensures rare screens (shop, campfire, relic) appear more in minibatches.
        Weights are capped at 3x to avoid extreme oversampling.
        """
        n = len(screen_type_idx)
        weights = torch.ones(n, dtype=torch.float32, device=screen_type_idx.device)
        unique_screens, counts = screen_type_idx.unique(return_counts=True)
        for st, cnt in zip(unique_screens, counts):
            mask = screen_type_idx == st
            inv_freq = n / cnt.float()
            weights[mask] = inv_freq.clamp(max=3.0)
        # Normalize so weights sum to n
        weights = weights / weights.sum() * n
        return weights

    def save_checkpoint(self, path: str | Path, iteration: int, metrics: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": iteration,
            "metrics": metrics,
            "config": {
                "model_type": "encoder_v2",
                "embed_dim": self.network.embed_dim,
                "set_encoder_dim": self.network.set_encoder_dim,
                "trunk_output": self.network.trunk_output,
                "screen_head_dim": self.network.screen_head_dim,
            },
        }, path)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps({
            "model_type": "encoder_v2",
            "checkpoint_path": str(path),
            "embed_dim": self.network.embed_dim,
            "set_encoder_dim": self.network.set_encoder_dim,
            "trunk_output": self.network.trunk_output,
            "screen_head_dim": self.network.screen_head_dim,
            "param_count": self.network.param_count(),
            "iteration": iteration,
        }, indent=2))

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        vocab: Vocab | None = None,
    ) -> tuple[FullRunPolicyNetworkV2, dict]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if vocab is None:
            vocab = load_vocab()
        network = FullRunPolicyNetworkV2(
            vocab=vocab,
            embed_dim=config.get("embed_dim", 32),
            set_encoder_dim=config.get("set_encoder_dim", 64),
            trunk_output=config.get("trunk_output", 128),
            screen_head_dim=config.get("screen_head_dim", 128),
        )
        # Handle old checkpoints that have single value_head instead of value_heads
        state_dict = checkpoint["model_state_dict"]
        old_vhead_keys = [k for k in state_dict if k.startswith("value_head.") and not k.startswith("value_heads.")]
        if old_vhead_keys and not any(k.startswith("value_heads.") for k in state_dict):
            logger.warning("Old checkpoint with single value_head detected — broadcasting to all screen value heads")
            for old_key in old_vhead_keys:
                suffix = old_key[len("value_head."):]
                for head_name in network.value_heads:
                    new_key = f"value_heads.{head_name}.{suffix}"
                    state_dict[new_key] = state_dict[old_key].clone()
        network.load_state_dict(state_dict, strict=False)
        return network, checkpoint


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RLFullRunPolicyV2:
    """Wraps a trained V2 network for inference."""

    network: FullRunPolicyNetworkV2
    vocab: Vocab
    name: str = "rl_v2"
    deterministic: bool = True

    def choose_action(
        self,
        state: dict[str, Any],
        candidate_actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if candidate_actions is None or not candidate_actions:
            return None

        # Build structured inputs
        ss = build_structured_state(state, self.vocab)
        sa = build_structured_actions(state, candidate_actions, self.vocab)

        # Convert to batched tensors
        state_t = {k: torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray)
                   else torch.tensor([v])
                   for k, v in _structured_state_to_numpy_dict(ss).items()}
        action_t = {k: torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray)
                    else torch.tensor([v])
                    for k, v in _structured_actions_to_numpy_dict(sa).items()}

        # Fix dtypes
        for k, v in state_t.items():
            if "ids" in k or "idx" in k or "types" in k or "count" in k:
                state_t[k] = v.long()
            elif "mask" in k:
                state_t[k] = v.bool()
            else:
                state_t[k] = v.float()
        for k, v in action_t.items():
            if "ids" in k or "types" in k or "indices" in k:
                action_t[k] = v.long()
            elif "mask" in k:
                action_t[k] = v.bool()
            else:
                action_t[k] = v.float()

        # Forward
        self.network.eval()
        with torch.no_grad():
            logits, _values, _deck_q, _boss_ready, _action_adv = self.network.forward(state_t, action_t)
            if self.deterministic:
                action_idx = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                action_idx = Categorical(probs).sample().item()

        if 0 <= action_idx < len(candidate_actions):
            return candidate_actions[action_idx]
        return candidate_actions[0] if candidate_actions else None

    @classmethod
    def load(cls, path: str | Path, vocab: Vocab | None = None) -> RLFullRunPolicyV2:
        path = Path(path)
        if path.suffix == ".json":
            meta = json.loads(path.read_text())
            checkpoint_path = meta["checkpoint_path"]
        else:
            checkpoint_path = str(path)
        if vocab is None:
            vocab = load_vocab()
        network, _ = PPOTrainerV2.load_checkpoint(checkpoint_path, vocab)
        network.eval()
        return cls(network=network, vocab=vocab)
