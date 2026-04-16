"""Combat neural network — CombatPolicyValueNetwork and CombatNNEvaluator.

This file contains ONLY the network architecture and MCTS evaluation wrapper.
Feature engineering (state/action featurization) lives in combat_features.py.

Network architecture:
  Hand cards  -> self-attention -> hand_repr
  Enemies     -> self-attention -> enemy_repr
  Scalars     -> concat
  Turn prefix -> encoder
  -- state_encoder MLP -> state_repr (128-d)

  Policy path:
    state + actions -> joint self-attention -> BilinearScorer -> base_logits
    -> structured_action_logits (stop/continue -> family -> within-family)
    -> optional deck_delta, teacher fusion -> final logits

  Value path:
    state_repr -> value_head -> base_value in [-1, 1]
    state_repr -> continuation_head -> (win_prob, hp_loss, potion_cost)
    -> compose_room_conditioned_value() -> final V(s)

  Gates (all learnable scalars):
    action_aux_gate (0.1)    - action hand-crafted features strength
    turn_prefix_gate (0.0)   - intra-turn card history
    main_state/action_context_gate (0.0) - joint state-action attention
    stop_continue_gate (0.1) - end-turn vs continue decision
    action_family_gate (0.1) - play/potion/selection family bias
    resource_gate (0.1)      - potion vs non-potion resource bias
    teacher_action_context_gate (0.0) - teacher path attention (can freeze)
    deck_delta_gate (0.0)    - deck-conditioned action scoring
    adapter_alpha/beta (0.0) - residual adapter for fine-tuning
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vocab import Vocab, load_vocab
from network.shared_encoders import (
    CARD_AUX_DIM,
    ENEMY_AUX_DIM,
    MAX_ACTIONS,
    MAX_ENEMIES,
    MAX_HAND_SIZE,
    EntityEmbeddings,
    SetEncoder,
    BilinearActionScorer,
    build_structured_state,
)
from core.combat_features import (
    COMBAT_SCALAR_DIM,
    COMBAT_EXTRA_SCALAR_DIM,
    COMBAT_TURN_PREFIX_LEN,
    COMBAT_TURN_PREFIX_SCALAR_DIM,
    COMBAT_ACTION_AUX_DIM,
    COMBAT_ROOM_TYPE_DIM,
    COMBAT_ACTION_FAMILY_PLAY_CARD,
    COMBAT_ACTION_FAMILY_POTION,
    COMBAT_ACTION_FAMILY_END_TURN,
    COMBAT_ACTION_FAMILY_SELECTION,
    NUM_COMBAT_ACTION_FAMILIES,
    compose_room_conditioned_value,
    build_combat_features,
    build_combat_action_features,
)
from search.mcts_core import NNEvaluator, action_key
from core.symbolic_features_head import SymbolicFeaturesHead



# ---------------------------------------------------------------------------
# Combat Neural Network
# ---------------------------------------------------------------------------

NUM_COMBAT_ACTION_TYPES = 8  # play_card, end_turn, use_potion, select_hand_card, ...


class CombatPolicyValueNetwork(nn.Module):
    """Policy + Value network for MCTS-guided combat."""

    def __init__(
        self,
        vocab: Vocab,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_attn_heads: int = 4,
        entity_embeddings: EntityEmbeddings | None = None,
        deck_repr_dim: int = 0,
        residual_adapter: bool = False,
        pile_specific: bool | None = None,
        # --- Symbolic features head (sqlite-backed cross-attention) ---
        # Typically shared with the PPO brain (see FullRunPolicyNetworkV2 kwarg
        # of the same name). When provided, card/monster symbolic features are
        # concatenated into hand/enemy/deck/pile encoder inputs. Zero-init
        # SymbolicFeaturesHead.out_proj ensures baseline parity on init.
        symbolic_head: SymbolicFeaturesHead | None = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.deck_repr_dim = deck_repr_dim
        self.residual_adapter = residual_adapter
        # pile_specific=None means auto (True when deck_repr_dim > 0)
        # pile_specific=False forces no pile encoders even with deck
        self._pile_specific = pile_specific if pile_specific is not None else (deck_repr_dim > 0)

        # Entity embeddings (shared with non-combat V2 if provided)
        if entity_embeddings is not None:
            self.entity_emb = entity_embeddings
        else:
            self.entity_emb = EntityEmbeddings(vocab, embed_dim)

        # Symbolic features head (shared instance; owned by PPO optimizer —
        # combat optimizer excludes symbolic_head.* via name filter in
        # train_hybrid.py. Combat's backward still accumulates grads here.)
        self.symbolic_head = symbolic_head
        self.use_symbolic_features = symbolic_head is not None
        sp = symbolic_head.proj_dim if symbolic_head is not None else 0
        self.symbolic_proj_dim = sp

        # Combat action type embedding
        self.combat_action_type_embed = nn.Embedding(NUM_COMBAT_ACTION_TYPES, embed_dim)
        self.register_buffer(
            "action_type_to_family",
            torch.tensor([
                COMBAT_ACTION_FAMILY_PLAY_CARD,
                COMBAT_ACTION_FAMILY_END_TURN,
                COMBAT_ACTION_FAMILY_POTION,
                COMBAT_ACTION_FAMILY_SELECTION,
                COMBAT_ACTION_FAMILY_SELECTION,
                COMBAT_ACTION_FAMILY_SELECTION,
                COMBAT_ACTION_FAMILY_SELECTION,
                COMBAT_ACTION_FAMILY_SELECTION,
            ], dtype=torch.long),
        )

        # Set encoders
        # force_linear=True when retrieval is enabled so that any coincidental
        # match between input_dim and output_dim (e.g. embed_dim + sp happening
        # to equal hidden_dim/deck_repr_dim/pile_repr_dim for some config)
        # still gets a real nn.Linear that our checkpoint loader can handle.
        fl = self.use_symbolic_features
        card_input_dim = embed_dim + CARD_AUX_DIM + sp
        self.hand_encoder = SetEncoder(card_input_dim, hidden_dim, num_attn_heads, force_linear=fl)
        self.enemy_encoder = SetEncoder(embed_dim + ENEMY_AUX_DIM + sp, hidden_dim, num_attn_heads, force_linear=fl)

        # Optional deck encoder for build_plan_z
        if deck_repr_dim > 0:
            self.deck_encoder = SetEncoder(card_input_dim, deck_repr_dim, num_attn_heads, force_linear=fl)

        # Pile-specific encoders (draw/discard/exhaust) or fallback remain encoder
        self.pile_repr_dim = 32 if self._pile_specific else 0
        if self.pile_repr_dim > 0:
            self.draw_pile_encoder = SetEncoder(card_input_dim, self.pile_repr_dim, num_attn_heads, force_linear=fl)
            self.discard_pile_encoder = SetEncoder(card_input_dim, self.pile_repr_dim, num_attn_heads, force_linear=fl)
            self.exhaust_pile_encoder = SetEncoder(card_input_dim, self.pile_repr_dim, num_attn_heads, force_linear=fl)
            # Legacy fallback (for checkpoints without pile-specific data)
            self.remain_encoder = SetEncoder(card_input_dim, self.pile_repr_dim, num_attn_heads, force_linear=fl)

        # State encoder
        # Layout: [scalars(18) | hand_repr | enemy_repr | (deck) | (pile*3) | extra_scalars(14)]
        # extra_scalars are appended at END so old checkpoints can load via end-pad backward compat
        state_input_dim = COMBAT_SCALAR_DIM + hidden_dim * 2  # legacy scalars + hand + enemy
        if deck_repr_dim > 0 and not residual_adapter:
            state_input_dim += deck_repr_dim  # concat mode: deck in trunk
            state_input_dim += self.pile_repr_dim * 3  # draw + discard + exhaust (or remain fallback)
        state_input_dim += COMBAT_EXTRA_SCALAR_DIM  # v2 player powers (always at END)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Action encoder: action_type + card + enemy → hidden_dim
        action_repr_dim = embed_dim * 3  # type + card + enemy
        self.action_proj = nn.Linear(action_repr_dim, hidden_dim)
        self.action_aux_proj = nn.Sequential(
            nn.Linear(COMBAT_ACTION_AUX_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Gate init is parameterised via env vars so we can sweep without
        # editing source / re-running git diff. Unset → legacy default (0.1).
        import os as _os
        _g_aux = float(_os.getenv("COMBAT_ACTION_AUX_GATE_INIT", "0.1"))
        _g_fam = float(_os.getenv("COMBAT_ACTION_FAMILY_GATE_INIT", "0.1"))
        _g_stop = float(_os.getenv("COMBAT_STOP_CONTINUE_GATE_INIT", "0.1"))
        _g_res = float(_os.getenv("COMBAT_RESOURCE_GATE_INIT", "0.1"))
        self.action_aux_gate = nn.Parameter(torch.tensor(_g_aux))
        self.action_family_head = nn.Linear(hidden_dim, NUM_COMBAT_ACTION_FAMILIES)
        self.action_family_gate = nn.Parameter(torch.tensor(_g_fam))
        self.stop_continue_head = nn.Linear(hidden_dim, 2)
        self.stop_continue_gate = nn.Parameter(torch.tensor(_g_stop))
        self.resource_gate_head = nn.Linear(hidden_dim, 2)
        self.resource_gate = nn.Parameter(torch.tensor(_g_res))

        # Deck-conditioned action delta (GPT Pro #7: deck directly influences action scoring)
        # delta_logit(a) = f(deck_z, action_emb) — zero-init for safe start
        if deck_repr_dim > 0:
            self.deck_action_delta = nn.Sequential(
                nn.Linear(deck_repr_dim + hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            # Zero-init last layer
            nn.init.zeros_(self.deck_action_delta[-1].weight)
            nn.init.zeros_(self.deck_action_delta[-1].bias)
            self.deck_delta_gate = nn.Parameter(torch.tensor(0.0))

        # Policy head: bilinear(state, action) → score
        self.policy_scorer = BilinearActionScorer(hidden_dim, hidden_dim)

        # Value head: state → V(s) ∈ [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )
        self.main_action_context_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_attn_heads,
            batch_first=True,
        )
        self.main_action_context_norm = nn.LayerNorm(hidden_dim)
        self.main_action_context_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.main_action_context_ffn_norm = nn.LayerNorm(hidden_dim)
        self.main_action_context_gate = nn.Parameter(torch.tensor(0.0))
        self.main_state_context_gate = nn.Parameter(torch.tensor(0.0))
        self.turn_prefix_pos_embed = nn.Embedding(COMBAT_TURN_PREFIX_LEN, embed_dim)
        self.turn_prefix_token_proj = nn.Linear(embed_dim + CARD_AUX_DIM + embed_dim, hidden_dim)
        self.turn_prefix_encoder = nn.Sequential(
            nn.Linear(hidden_dim * COMBAT_TURN_PREFIX_LEN + COMBAT_TURN_PREFIX_SCALAR_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.turn_prefix_gate = nn.Parameter(torch.tensor(0.0))

        # Residual adapter: deck-conditioned delta heads (GPT Pro recommendation)
        # Frozen backbone computes base logits/value; adapter adds deck-aware residuals.
        # Last layers zero-initialized so initial output = pure base (safe warm start).
        if residual_adapter and deck_repr_dim > 0:
            self.delta_logits_head = nn.Sequential(
                nn.Linear(deck_repr_dim + hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            # Zero-init last layer for safe start
            nn.init.zeros_(self.delta_logits_head[-1].weight)
            nn.init.zeros_(self.delta_logits_head[-1].bias)

            self.delta_value_head = nn.Sequential(
                nn.Linear(deck_repr_dim + hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Tanh(),
            )
            # Zero-init last layer
            nn.init.zeros_(self.delta_value_head[-2].weight)
            nn.init.zeros_(self.delta_value_head[-2].bias)

            # Learnable gate scalars (init 0 → pure base at start)
            self.adapter_alpha = nn.Parameter(torch.tensor(0.0))
            self.adapter_beta = nn.Parameter(torch.tensor(0.0))

        # Offline teacher-stack auxiliary heads. They are intentionally separate
        # from the online PPO outputs so existing callers can keep using
        # forward() without any behavior change.
        #
        # The baseline teacher scorer is still a simple state/action MLP. We
        # keep it as the stable path, then add a lightweight attention-based
        # residual on top so older checkpoints remain compatible and the online
        # combat policy path stays untouched.
        self.action_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Deploy-time teacher fusion: when > 0, forward() additionally computes
        # forward_teacher's action_score_head output and blends it into the
        # policy logits. Before this, teacher training only affected online
        # inference via the shared encoder (indirect); action_score_head was
        # trained to convergence on solver answers but never consulted at deploy.
        # Typical usable range: 0.3 (light rerank) ~ 1.0 (teacher dominates).
        # Set via `combat_net.teacher_rerank_weight = x` from the calling script
        # (FullRunAgent / evaluate_ai flips this from AgentConfig).
        self.teacher_rerank_weight: float = 0.0
        self.teacher_action_context_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_attn_heads,
            batch_first=True,
        )
        self.teacher_action_context_norm = nn.LayerNorm(hidden_dim)
        self.teacher_action_context_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.teacher_action_context_ffn_norm = nn.LayerNorm(hidden_dim)
        self.teacher_action_context_gate = nn.Parameter(torch.tensor(0.0))
        # FREEZE_TEACHER_CONTEXT_GATE=1 keeps the gate pinned at its init value.
        # In v10 this gate ran away to -0.036 during teacher loss training and
        # silently pulled the shared encoder via residual attention, which
        # degraded online PPO policy. Freezing it lets teacher loss still
        # train action_score_head directly but stops the indirect encoder drift.
        if _os.getenv("FREEZE_TEACHER_CONTEXT_GATE", "0") in ("1", "true", "yes"):
            self.teacher_action_context_gate.requires_grad_(False)
        self.continuation_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # --- Encoder audit for symbolic features wiring ---
        # Combat encoder input/output dims never match in baseline (card input
        # 85 != hidden 128 / deck_repr 64 / pile 32), so SetEncoder.proj is
        # always nn.Linear here — no Identity fast-path to worry about like in
        # rl_policy_v2.py's relic_encoder. This audit is defensive: if a future
        # change introduces a matching dim, we want a loud failure at construct
        # time rather than silent weight corruption on checkpoint load.
        if self.use_symbolic_features:
            self._audit_encoder_projs()

    def _audit_encoder_projs(self):
        """Assert all encoder projs are nn.Linear (not Identity) when symbolic
        features are enabled. Combat currently never hits Identity but this is
        a regression guard for future refactors."""
        encoders: list[tuple[str, SetEncoder]] = [
            ("hand_encoder", self.hand_encoder),
            ("enemy_encoder", self.enemy_encoder),
        ]
        if self.deck_repr_dim > 0:
            encoders.append(("deck_encoder", self.deck_encoder))
        if self.pile_repr_dim > 0:
            encoders.extend([
                ("draw_pile_encoder", self.draw_pile_encoder),
                ("discard_pile_encoder", self.discard_pile_encoder),
                ("exhaust_pile_encoder", self.exhaust_pile_encoder),
                ("remain_encoder", self.remain_encoder),
            ])
        for name, enc in encoders:
            if not isinstance(enc.proj, nn.Linear):
                raise RuntimeError(
                    f"CombatPolicyValueNetwork: {name}.proj is "
                    f"{type(enc.proj).__name__}, expected nn.Linear after enabling "
                    "symbolic features. Check the encoder input-dim arithmetic."
                )

    def _encode_state_and_actions(
        self,
        state_features: dict[str, torch.Tensor],
        action_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Encode combat state and legal actions into hidden representations.

        Returns:
            state_repr: (B, hidden_dim)
            action_repr: (B, A, hidden_dim)
            deck_repr: (B, deck_repr_dim) or None if no deck encoder
            turn_prefix_repr: (B, hidden_dim)
        """
        # Hand encoding ([card_embed | card_aux | (optional) symbolic])
        hand_base = self.entity_emb.card_embed(state_features["hand_ids"])
        hand_parts = [hand_base, state_features["hand_aux"]]
        if self.symbolic_head is not None:
            hand_parts.append(self.symbolic_head.card(state_features["hand_ids"], hand_base))
        hand_emb = torch.cat(hand_parts, dim=-1)
        hand_repr = self.hand_encoder(hand_emb, state_features["hand_mask"])

        # Enemy encoding
        enemy_base = self.entity_emb.monster_embed(state_features["enemy_ids"])
        enemy_parts = [enemy_base, state_features["enemy_aux"]]
        if self.symbolic_head is not None:
            enemy_parts.append(self.symbolic_head.monster(state_features["enemy_ids"], enemy_base))
        enemy_emb = torch.cat(enemy_parts, dim=-1)
        enemy_repr = self.enemy_encoder(enemy_emb, state_features["enemy_mask"])

        # Deck encoding (if available)
        deck_repr: torch.Tensor | None = None
        if self.deck_repr_dim > 0:
            if "deck_ids" in state_features and "deck_mask" in state_features:
                deck_base = self.entity_emb.card_embed(state_features["deck_ids"])
                deck_parts = [deck_base, state_features["deck_aux"]]
                if self.symbolic_head is not None:
                    deck_parts.append(
                        self.symbolic_head.card(state_features["deck_ids"], deck_base)
                    )
                deck_emb = torch.cat(deck_parts, dim=-1)
                deck_repr = self.deck_encoder(deck_emb, state_features["deck_mask"])
            elif "deck_repr" in state_features:
                deck_repr = state_features["deck_repr"]
            else:
                batch_size = state_features["scalars"].shape[0]
                deck_repr = torch.zeros(batch_size, self.deck_repr_dim,
                                        device=state_features["scalars"].device)

        # Pile-specific encoding (draw/discard/exhaust) or fallback remain
        pile_reprs: list[torch.Tensor] = []
        if self.pile_repr_dim > 0:
            if "draw_pile_ids" in state_features:
                # Use actual pile card lists
                for pile_name, encoder in [
                    ("draw_pile", self.draw_pile_encoder),
                    ("discard_pile", self.discard_pile_encoder),
                    ("exhaust_pile", self.exhaust_pile_encoder),
                ]:
                    pile_ids = state_features[f"{pile_name}_ids"]
                    p_base = self.entity_emb.card_embed(pile_ids)
                    p_parts = [p_base, state_features[f"{pile_name}_aux"]]
                    if self.symbolic_head is not None:
                        p_parts.append(self.symbolic_head.card(pile_ids, p_base))
                    p_emb = torch.cat(p_parts, dim=-1)
                    pile_reprs.append(encoder(p_emb, state_features[f"{pile_name}_mask"]))
            elif "remain_ids" in state_features:
                # Fallback: single remain encoder replicated 3x
                remain_ids = state_features["remain_ids"]
                r_base = self.entity_emb.card_embed(remain_ids)
                r_parts = [r_base, state_features["remain_aux"]]
                if self.symbolic_head is not None:
                    r_parts.append(self.symbolic_head.card(remain_ids, r_base))
                r_emb = torch.cat(r_parts, dim=-1)
                remain_repr = self.remain_encoder(r_emb, state_features["remain_mask"])
                pile_reprs = [remain_repr, remain_repr, remain_repr]  # pad to 3x
            else:
                batch_size = state_features["scalars"].shape[0]
                dev = state_features["scalars"].device
                pile_reprs = [torch.zeros(batch_size, self.pile_repr_dim, device=dev)] * 3

        # State encoding: concat mode includes deck + piles, residual mode excludes them
        state_parts = [state_features["scalars"], hand_repr, enemy_repr]
        if self.deck_repr_dim > 0 and not self.residual_adapter and deck_repr is not None:
            state_parts.append(deck_repr)
        if self.pile_repr_dim > 0 and not self.residual_adapter and pile_reprs:
            state_parts.extend(pile_reprs)
        # v2 extra player power scalars (always appended LAST so old checkpoints
        # can load via end-pad backward compat). Tolerate absent key for callers
        # that haven't been migrated yet.
        if "extra_scalars" in state_features:
            state_parts.append(state_features["extra_scalars"])
        else:
            batch_size = state_features["scalars"].shape[0]
            dev = state_features["scalars"].device
            state_parts.append(
                torch.zeros(batch_size, COMBAT_EXTRA_SCALAR_DIM, device=dev, dtype=state_features["scalars"].dtype)
            )
        state_input = torch.cat(state_parts, dim=-1)
        state_repr = self.state_encoder(state_input)

        batch_size = state_features["scalars"].shape[0]
        dev = state_features["scalars"].device
        dtype = state_features["scalars"].dtype
        prefix_ids = state_features.get("turn_prefix_ids")
        prefix_aux = state_features.get("turn_prefix_aux")
        prefix_mask = state_features.get("turn_prefix_mask")
        prefix_scalars = state_features.get("turn_prefix_scalars")
        if prefix_ids is None:
            prefix_ids = torch.zeros(batch_size, COMBAT_TURN_PREFIX_LEN, dtype=torch.long, device=dev)
        if prefix_aux is None:
            prefix_aux = torch.zeros(
                batch_size, COMBAT_TURN_PREFIX_LEN, CARD_AUX_DIM, dtype=dtype, device=dev
            )
        if prefix_mask is None:
            prefix_mask = torch.zeros(batch_size, COMBAT_TURN_PREFIX_LEN, dtype=torch.bool, device=dev)
        if prefix_scalars is None:
            prefix_scalars = torch.zeros(
                batch_size, COMBAT_TURN_PREFIX_SCALAR_DIM, dtype=dtype, device=dev
            )
        prefix_card_emb = self.entity_emb.card_embed(prefix_ids)
        prefix_pos = self.turn_prefix_pos_embed(
            torch.arange(COMBAT_TURN_PREFIX_LEN, device=dev).unsqueeze(0).expand(batch_size, -1)
        )
        prefix_tokens = torch.cat([prefix_card_emb, prefix_aux, prefix_pos], dim=-1)
        prefix_tokens = self.turn_prefix_token_proj(prefix_tokens)
        prefix_tokens = prefix_tokens * prefix_mask.unsqueeze(-1).to(prefix_tokens.dtype)
        prefix_input = torch.cat([prefix_tokens.reshape(batch_size, -1), prefix_scalars], dim=-1)
        turn_prefix_repr = self.turn_prefix_encoder(prefix_input)

        # Action encoding
        atype_emb = self.combat_action_type_embed(action_features["action_type_ids"])
        card_emb = self.entity_emb.card_embed(action_features["target_card_ids"])
        enemy_emb_act = self.entity_emb.monster_embed(action_features["target_enemy_ids"])
        action_repr = torch.cat([atype_emb, card_emb, enemy_emb_act], dim=-1)
        action_repr = self.action_proj(action_repr)
        action_aux = action_features.get("action_aux")
        if action_aux is None:
            action_aux = torch.zeros(
                batch_size, MAX_ACTIONS, COMBAT_ACTION_AUX_DIM,
                dtype=dtype, device=dev,
            )
        action_repr = action_repr + self.action_aux_gate * self.action_aux_proj(action_aux)

        return state_repr, action_repr, deck_repr, turn_prefix_repr

    def _structured_action_logits(
        self,
        state_repr: torch.Tensor,
        base_logits: torch.Tensor,
        action_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compose flatter per-action scores into a more STS-like hierarchy.

        Decision flow:
        1. stop/continue gate
        2. continue-family gate (play_card / use_potion / selection)
        3. within-family action score

        We still emit a flat categorical over legal actions for PPO, but
        end_turn and use_potion no longer compete on exactly the same plane as
        ordinary card plays.
        """
        family_ids = action_features.get("action_family_ids")
        if family_ids is None:
            action_type_ids = action_features["action_type_ids"].clamp(
                min=0, max=self.action_type_to_family.shape[0] - 1,
            )
            family_ids = self.action_type_to_family[action_type_ids]
        action_mask = action_features["action_mask"]

        family_logits = self.action_family_gate * self.action_family_head(state_repr)
        stop_logits = self.stop_continue_gate * self.stop_continue_head(state_repr)
        resource_logits = self.resource_gate * self.resource_gate_head(state_repr)
        continue_bias = stop_logits[:, 0:1]
        end_bias = stop_logits[:, 1:2]

        gathered_family = torch.gather(family_logits, 1, family_ids.long())
        non_potion_bias = resource_logits[:, 0:1]
        potion_bias = resource_logits[:, 1:2]

        family_is_end = family_ids == COMBAT_ACTION_FAMILY_END_TURN
        family_is_potion = family_ids == COMBAT_ACTION_FAMILY_POTION

        resource_bias = torch.where(family_is_potion, potion_bias, non_potion_bias)
        continue_family_bias = continue_bias + gathered_family + resource_bias

        # Keep a small amount of per-action base score on end_turn so the model
        # can still rank multiple end-turn-like legal variants if they exist,
        # but make the stop head the dominant signal.
        end_logits = end_bias + 0.25 * base_logits
        continue_logits = continue_family_bias + base_logits
        structured_logits = torch.where(family_is_end, end_logits, continue_logits)
        return structured_logits.masked_fill(~action_mask, -1e9)

    def _apply_residual_action_context(
        self,
        *,
        state_repr: torch.Tensor,
        action_repr: torch.Tensor,
        action_mask: torch.Tensor,
        attn: nn.MultiheadAttention,
        norm: nn.LayerNorm,
        ffn: nn.Sequential,
        ffn_norm: nn.LayerNorm,
        state_gate: torch.Tensor,
        action_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a lightweight joint state/action attention residual."""
        joint_tokens = torch.cat([state_repr.unsqueeze(1), action_repr], dim=1)
        joint_padding_mask = torch.cat(
            [
                torch.zeros(
                    action_mask.shape[0],
                    1,
                    dtype=torch.bool,
                    device=action_mask.device,
                ),
                ~action_mask,
            ],
            dim=1,
        )
        attn_out, _ = attn(
            joint_tokens,
            joint_tokens,
            joint_tokens,
            key_padding_mask=joint_padding_mask,
            need_weights=False,
        )
        original_tokens = joint_tokens
        joint_tokens = norm(joint_tokens + attn_out)
        ffn_out = ffn(joint_tokens)
        joint_tokens = ffn_norm(joint_tokens + ffn_out)
        joint_delta = joint_tokens - original_tokens
        state_delta = joint_delta[:, 0, :]
        action_delta = joint_delta[:, 1:, :]
        contextual_state_repr = state_repr + state_gate * state_delta
        contextual_action_repr = action_repr + action_gate * action_delta
        return contextual_state_repr, contextual_action_repr

    def forward(
        self,
        state_features: dict[str, torch.Tensor],
        action_features: dict[str, torch.Tensor],
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state_features: dict of tensors (scalars, hand_*, enemy_*, deck_*, etc.)
            action_features: dict of tensors (action_type_ids, target_card_ids, ...)
            return_hidden: if True, also return (state_repr, action_repr) for Phase 4 Stage 6
                           boss expert consumption. Default False preserves old behavior.

        Returns:
            (policy_logits, value) if return_hidden=False (default)
            (policy_logits, value, state_repr, action_repr) if return_hidden=True
              where state_repr: (B, hidden_dim), action_repr: (B, A, hidden_dim)
        """
        state_repr, action_repr, deck_repr, turn_prefix_repr = self._encode_state_and_actions(state_features, action_features)
        state_repr = state_repr + self.turn_prefix_gate * turn_prefix_repr
        policy_state_repr, policy_action_repr = self._apply_residual_action_context(
            state_repr=state_repr,
            action_repr=action_repr,
            action_mask=action_features["action_mask"],
            attn=self.main_action_context_attn,
            norm=self.main_action_context_norm,
            ffn=self.main_action_context_ffn,
            ffn_norm=self.main_action_context_ffn_norm,
            state_gate=self.main_state_context_gate,
            action_gate=self.main_action_context_gate,
        )

        # Base policy + value (from backbone)
        base_logits = self.policy_scorer(policy_state_repr, policy_action_repr, action_features["action_mask"])
        logits = self._structured_action_logits(
            policy_state_repr,
            base_logits,
            action_features,
        )
        base_value = self.value_head(policy_state_repr).squeeze(-1)
        continuation_raw = self.continuation_value_head(policy_state_repr)
        continuation = torch.cat(
            [
                torch.sigmoid(continuation_raw[:, 0:1]),
                F.softplus(continuation_raw[:, 1:2]),
                F.softplus(continuation_raw[:, 2:3]),
            ],
            dim=-1,
        )
        value = compose_room_conditioned_value(
            base_value,
            continuation,
            state_features.get("room_type_onehot"),
        )

        # Deck-conditioned action delta: deck info directly influences per-action scoring
        if deck_repr is not None and hasattr(self, "deck_action_delta"):
            B, A, _ = policy_action_repr.shape
            deck_exp = deck_repr.unsqueeze(1).expand(-1, A, -1)  # (B, A, deck_dim)
            delta_in = torch.cat([deck_exp, policy_action_repr], dim=-1)  # (B, A, deck_dim+hidden)
            deck_delta = self.deck_action_delta(delta_in).squeeze(-1)  # (B, A)
            deck_delta = deck_delta.masked_fill(~action_features["action_mask"], 0.0)
            logits = logits + self.deck_delta_gate * deck_delta

        # Residual adapter: add deck-conditioned deltas
        if self.residual_adapter and deck_repr is not None and hasattr(self, "delta_logits_head"):
            B, A, _ = policy_action_repr.shape
            deck_expanded = deck_repr.unsqueeze(1).expand(-1, A, -1)  # (B, A, deck_dim)
            state_expanded = policy_state_repr.unsqueeze(1).expand(-1, A, -1)  # (B, A, hidden)
            delta_input = torch.cat([deck_expanded, state_expanded, policy_action_repr], dim=-1)
            delta_logits = self.delta_logits_head(delta_input).squeeze(-1)  # (B, A)
            delta_logits = delta_logits.masked_fill(~action_features["action_mask"], 0.0)
            logits = logits + self.adapter_alpha * delta_logits

            dv_input = torch.cat([deck_repr, policy_state_repr], dim=-1)
            delta_value = self.delta_value_head(dv_input).squeeze(-1)  # (B,)
            value = torch.tanh(value + self.adapter_beta * delta_value)

        # Deploy-time teacher fusion (teacher_rerank_weight > 0 activates).
        # The teacher path (action_score_head + teacher_action_context_attn) is
        # trained to match solver answers under the combat_teacher loss but is
        # normally unused at inference. When the caller flips the weight, we
        # add the teacher action_scores into the policy logits so the trained
        # teacher weights actually influence the deploy argmax.
        if self.teacher_rerank_weight > 0:
            action_mask = action_features["action_mask"]
            _, contextual_action_repr_t = self._apply_residual_action_context(
                state_repr=policy_state_repr,
                action_repr=policy_action_repr,
                action_mask=action_mask,
                attn=self.teacher_action_context_attn,
                norm=self.teacher_action_context_norm,
                ffn=self.teacher_action_context_ffn,
                ffn_norm=self.teacher_action_context_ffn_norm,
                state_gate=policy_state_repr.new_tensor(0.0),
                action_gate=self.teacher_action_context_gate,
            )
            state_exp = policy_state_repr.unsqueeze(1).expand(-1, policy_action_repr.shape[1], -1)
            score_input = torch.cat([state_exp, contextual_action_repr_t], dim=-1)
            teacher_scores = self.action_score_head(score_input).squeeze(-1)
            teacher_scores = teacher_scores.masked_fill(~action_mask, 0.0)  # invalid actions contribute 0
            logits = logits + float(self.teacher_rerank_weight) * teacher_scores

        if return_hidden:
            return logits, value, policy_state_repr, policy_action_repr
        return logits, value

    def forward_teacher(
        self,
        state_features: dict[str, torch.Tensor],
        action_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extended forward for the offline combat teacher stack."""
        state_repr, action_repr, deck_repr, turn_prefix_repr = self._encode_state_and_actions(state_features, action_features)
        state_repr = state_repr + self.turn_prefix_gate * turn_prefix_repr
        policy_state_repr, policy_action_repr = self._apply_residual_action_context(
            state_repr=state_repr,
            action_repr=action_repr,
            action_mask=action_features["action_mask"],
            attn=self.main_action_context_attn,
            norm=self.main_action_context_norm,
            ffn=self.main_action_context_ffn,
            ffn_norm=self.main_action_context_ffn_norm,
            state_gate=self.main_state_context_gate,
            action_gate=self.main_action_context_gate,
        )
        base_logits = self.policy_scorer(policy_state_repr, policy_action_repr, action_features["action_mask"])
        logits = self._structured_action_logits(
            policy_state_repr,
            base_logits,
            action_features,
        )
        base_value = self.value_head(policy_state_repr).squeeze(-1)

        action_mask = action_features["action_mask"]
        _, contextual_action_repr = self._apply_residual_action_context(
            state_repr=policy_state_repr,
            action_repr=policy_action_repr,
            action_mask=action_mask,
            attn=self.teacher_action_context_attn,
            norm=self.teacher_action_context_norm,
            ffn=self.teacher_action_context_ffn,
            ffn_norm=self.teacher_action_context_ffn_norm,
            state_gate=policy_state_repr.new_tensor(0.0),
            action_gate=self.teacher_action_context_gate,
        )

        state_expanded = policy_state_repr.unsqueeze(1).expand(-1, policy_action_repr.shape[1], -1)
        action_score_input = torch.cat([state_expanded, contextual_action_repr], dim=-1)
        raw_action_scores = self.action_score_head(action_score_input).squeeze(-1)
        action_scores = raw_action_scores.masked_fill(~action_mask, -1e9)

        continuation_raw = self.continuation_value_head(policy_state_repr)
        win_prob = torch.sigmoid(continuation_raw[:, 0:1])
        expected_hp_loss = F.softplus(continuation_raw[:, 1:2])
        expected_potion_cost = F.softplus(continuation_raw[:, 2:3])
        continuation = torch.cat([win_prob, expected_hp_loss, expected_potion_cost], dim=-1)
        value = compose_room_conditioned_value(
            base_value,
            continuation,
            state_features.get("room_type_onehot"),
        )
        return logits, value, action_scores, continuation

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# NN Evaluator wrapper for MCTS
# ---------------------------------------------------------------------------

def _auto_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _tensorize_features(features: dict[str, np.ndarray], device: torch.device,
                         unsqueeze: bool = True) -> dict[str, torch.Tensor]:
    """Convert numpy feature dict to tensors on device."""
    out = {}
    for k, v in features.items():
        a = np.array(v)
        if a.dtype in (np.int64, np.int32):
            t = torch.tensor(a).long()
        elif a.dtype == bool:
            t = torch.tensor(a).bool()
        else:
            t = torch.tensor(a).float()
        if unsqueeze:
            t = t.unsqueeze(0)
        out[k] = t.to(device)
    return out


class CombatNNEvaluator:
    """Wraps CombatPolicyValueNetwork as an NNEvaluator for MCTS."""

    def __init__(self, network: CombatPolicyValueNetwork, vocab: Vocab,
                 device: torch.device | None = None,
                 use_continuation_value: bool = False,
                 ppo_net: Any | None = None):
        self.device = device or _auto_device()
        self.network = network.to(self.device)
        self.vocab = vocab
        self.network.eval()
        self._use_amp = self.device.type == "cuda"
        self._use_continuation_value = use_continuation_value
        self._ppo_net = ppo_net

    def evaluate(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> tuple[np.ndarray, float]:
        """Evaluate combat state → (policy, value)."""
        sf = build_combat_features(state, self.vocab)
        af = build_combat_action_features(state, legal_actions, self.vocab)

        if self._ppo_net is not None and hasattr(self._ppo_net, "compute_deck_repr"):
            try:
                ss = build_structured_state(state, self.vocab)
                deck_t = {
                    "deck_ids": torch.tensor(ss.deck_ids).unsqueeze(0).to(self.device),
                    "deck_aux": torch.tensor(ss.deck_aux).unsqueeze(0).float().to(self.device),
                    "deck_mask": torch.tensor(ss.deck_mask).unsqueeze(0).bool().to(self.device),
                }
                with torch.no_grad():
                    sf["deck_repr"] = self._ppo_net.compute_deck_repr(deck_t).squeeze(0).detach().cpu().numpy()
            except Exception:
                pass

        state_t = _tensorize_features(sf, self.device)
        action_t = _tensorize_features(af, self.device)

        with torch.no_grad():
            if self._use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    if self._use_continuation_value:
                        logits, _value, _scores, continuation = self.network.forward_teacher(state_t, action_t)
                        value_scalar = compose_room_conditioned_value(
                            torch.zeros_like(_value),
                            continuation,
                            state_t.get("room_type_onehot"),
                        )[0].cpu().float().item()
                    else:
                        logits, value = self.network.forward(state_t, action_t)
                        value_scalar = value[0].cpu().float().item()
            else:
                if self._use_continuation_value:
                    logits, _value, _scores, continuation = self.network.forward_teacher(state_t, action_t)
                    value_scalar = compose_room_conditioned_value(
                        torch.zeros_like(_value),
                        continuation,
                        state_t.get("room_type_onehot"),
                    )[0].cpu().float().item()
                else:
                    logits, value = self.network.forward(state_t, action_t)
                    value_scalar = value[0].cpu().float().item()

        # Extract policy for legal actions only
        n = min(len(legal_actions), MAX_ACTIONS)
        raw_logits = logits[0, :n].cpu().float().numpy()
        policy = np.exp(raw_logits - raw_logits.max())
        policy = policy / policy.sum()

        return policy, value_scalar

    def evaluate_batch(
        self,
        states: list[dict[str, Any]],
        legal_actions_list: list[list[dict[str, Any]]],
    ) -> list[tuple[np.ndarray, float]]:
        """Batch evaluate multiple combat states in one forward pass.

        Returns:
            List of (policy, value) tuples, one per input state.
        """
        if not states:
            return []

        # Build feature dicts for each state
        all_sf = [build_combat_features(s, self.vocab) for s in states]
        all_af = [build_combat_action_features(s, la, self.vocab)
                  for s, la in zip(states, legal_actions_list)]

        if self._ppo_net is not None and hasattr(self._ppo_net, "compute_deck_repr"):
            for sf, state in zip(all_sf, states):
                try:
                    ss = build_structured_state(state, self.vocab)
                    deck_t = {
                        "deck_ids": torch.tensor(ss.deck_ids).unsqueeze(0).to(self.device),
                        "deck_aux": torch.tensor(ss.deck_aux).unsqueeze(0).float().to(self.device),
                        "deck_mask": torch.tensor(ss.deck_mask).unsqueeze(0).bool().to(self.device),
                    }
                    with torch.no_grad():
                        sf["deck_repr"] = self._ppo_net.compute_deck_repr(deck_t).squeeze(0).detach().cpu().numpy()
                except Exception:
                    pass

        # Stack along batch dimension
        batch_state: dict[str, torch.Tensor] = {}
        for k in all_sf[0]:
            arrs = [np.array(sf[k]) for sf in all_sf]
            stacked = np.stack(arrs, axis=0)
            if stacked.dtype in (np.int64, np.int32):
                batch_state[k] = torch.tensor(stacked).long().to(self.device)
            elif stacked.dtype == bool:
                batch_state[k] = torch.tensor(stacked).bool().to(self.device)
            else:
                batch_state[k] = torch.tensor(stacked).float().to(self.device)

        batch_action: dict[str, torch.Tensor] = {}
        for k in all_af[0]:
            arrs = [np.array(af[k]) for af in all_af]
            stacked = np.stack(arrs, axis=0)
            if stacked.dtype in (np.int64, np.int32):
                batch_action[k] = torch.tensor(stacked).long().to(self.device)
            elif stacked.dtype == bool:
                batch_action[k] = torch.tensor(stacked).bool().to(self.device)
            else:
                batch_action[k] = torch.tensor(stacked).float().to(self.device)

        with torch.no_grad():
            if self._use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, values = self.network.forward(batch_state, batch_action)
            else:
                logits, values = self.network.forward(batch_state, batch_action)

        logits = logits.cpu().float().numpy()
        values = values.cpu().float().numpy()

        results = []
        for i, la in enumerate(legal_actions_list):
            n = min(len(la), MAX_ACTIONS)
            raw = logits[i, :n]
            policy = np.exp(raw - raw.max())
            policy = policy / policy.sum()
            results.append((policy, float(values[i])))

        return results


# ---------------------------------------------------------------------------
# Backward-compat re-exports (canonical home: core.combat_features)
# ---------------------------------------------------------------------------
from core.combat_features import (  # noqa: F401
    COMBAT_TOTAL_SCALAR_DIM,
    _get_power_amount,
    _player_power_list,
)
