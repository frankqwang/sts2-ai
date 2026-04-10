"""Combat neural network for MCTS guidance.

Provides policy prior p(a|s) and value estimate V(s) for combat states.
Shares entity embeddings (card_embed, monster_embed) with the non-combat
V2 encoder for transfer learning.

Architecture:
  Hand cards → self-attention → hand_repr
  Enemies → self-attention → enemy_repr
  Scalars (hp, block, energy, round) → MLP → scalar_repr
  concat → MLP → combat_repr (128-d)
    ├─ Policy: bilinear(combat_repr, action_embed) → logits → softmax
    └─ Value: MLP → V(s) ∈ [-1, 1]
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import Vocab, load_vocab
from rl_encoder_v2 import (
    CARD_AUX_DIM,
    ENEMY_AUX_DIM,
    MAX_ACTIONS,
    MAX_ENEMIES,
    MAX_HAND_SIZE,
    EntityEmbeddings,
    SetEncoder,
    BilinearActionScorer,
    _card_aux_features,
    _cached_card_encoding,
    _cached_card_idx,
    _cached_monster_idx,
    _enemy_aux_features,
    _lower,
    _safe_float,
    _safe_int,
    _extract_player,
)
from mcts_core import NNEvaluator, action_key
from symbolic_features_head import SymbolicFeaturesHead


# ---------------------------------------------------------------------------
# Combat state featurization
# ---------------------------------------------------------------------------

COMBAT_SCALAR_DIM = 18  # FROZEN: 10 base + 8 player power features (legacy v1 layout)
COMBAT_EXTRA_SCALAR_DIM = 14  # NEW v2 player powers, appended at END of state_input
COMBAT_TOTAL_SCALAR_DIM = COMBAT_SCALAR_DIM + COMBAT_EXTRA_SCALAR_DIM  # 32


def _get_power_amount(powers: list, power_id: str) -> float:
    """Extract a specific power's stack count from a powers list."""
    for p in powers:
        if isinstance(p, dict):
            pid = _lower(p.get("id") or p.get("power_id", ""))
            if power_id in pid:
                return _safe_float(p.get("amount") or p.get("stacks"), 0)
    return 0.0


def _player_power_list(player: dict) -> list:
    """Single-source power list lookup. Avoids 3x double-count from pipe duplication."""
    for key in ("status", "powers", "power_list", "buffs", "debuffs"):
        v = player.get(key)
        if isinstance(v, list) and v:
            return v
    return []


def build_combat_features(
    state: dict, vocab: Vocab,
) -> dict[str, np.ndarray]:
    """Extract combat-specific features from state dict."""
    player = _extract_player(state)
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    run = state.get("run") if isinstance(state.get("run"), dict) else {}

    # Scalars
    scalars = np.zeros(COMBAT_SCALAR_DIM, dtype=np.float32)
    hp = _safe_float(player.get("hp", player.get("current_hp")))
    max_hp = max(1.0, _safe_float(player.get("max_hp"), 1))
    scalars[0] = hp / max_hp
    scalars[1] = max_hp / 100.0
    scalars[2] = _safe_float(player.get("block")) / 50.0
    scalars[3] = _safe_float(battle.get("energy") or player.get("energy")) / 5.0
    scalars[4] = _safe_float(battle.get("max_energy") or player.get("max_energy")) / 5.0
    scalars[5] = _safe_float(state.get("round_number") or battle.get("round_number")) / 20.0
    # Pile sizes
    scalars[6] = _safe_float(
        battle.get("draw_pile_count")
        or player.get("draw_pile_count")
        or len(player.get("draw_pile", []))
    ) / 30.0
    scalars[7] = _safe_float(
        battle.get("discard_pile_count")
        or player.get("discard_pile_count")
        or len(player.get("discard_pile", []))
    ) / 30.0
    scalars[8] = _safe_float(
        battle.get("exhaust_pile_count")
        or player.get("exhaust_pile_count")
        or len(player.get("exhaust_pile", []))
    ) / 20.0
    scalars[9] = _safe_int(run.get("floor")) / 20.0

    # Player powers/buffs/debuffs (single source — no double-count)
    player_powers = _player_power_list(player)
    scalars[10] = _get_power_amount(player_powers, "strength") / 10.0
    scalars[11] = _get_power_amount(player_powers, "dexterity") / 10.0
    scalars[12] = min(_get_power_amount(player_powers, "vulnerable") / 5.0, 1.0)
    scalars[13] = min(_get_power_amount(player_powers, "weak") / 5.0, 1.0)
    scalars[14] = min(_get_power_amount(player_powers, "frail") / 5.0, 1.0)
    scalars[15] = _get_power_amount(player_powers, "metallicize") / 10.0
    scalars[16] = _get_power_amount(player_powers, "regen") / 10.0
    scalars[17] = min(_get_power_amount(player_powers, "artifact") / 3.0, 1.0)

    # --- v2 extra player powers (appended at END of state_input, end-pad backward compat) ---
    extra_scalars = np.zeros(COMBAT_EXTRA_SCALAR_DIM, dtype=np.float32)
    extra_scalars[0]  = min(_get_power_amount(player_powers, "intangible") / 5.0, 1.0)
    extra_scalars[1]  = min(_get_power_amount(player_powers, "barricade") / 1.0, 1.0)
    extra_scalars[2]  = _get_power_amount(player_powers, "inflame") / 10.0
    extra_scalars[3]  = min(_get_power_amount(player_powers, "demon_form") / 5.0, 1.0)
    extra_scalars[4]  = min(_get_power_amount(player_powers, "flame_barrier") / 12.0, 1.0)
    extra_scalars[5]  = _get_power_amount(player_powers, "thorns") / 10.0
    extra_scalars[6]  = _get_power_amount(player_powers, "plated_armor") / 30.0
    extra_scalars[7]  = min(_get_power_amount(player_powers, "double_tap") / 3.0, 1.0)
    extra_scalars[8]  = min(_get_power_amount(player_powers, "energized") / 5.0, 1.0)
    extra_scalars[9]  = min(_get_power_amount(player_powers, "feel_no_pain") / 10.0, 1.0)
    extra_scalars[10] = min(_get_power_amount(player_powers, "dark_embrace") / 1.0, 1.0)
    extra_scalars[11] = min(_get_power_amount(player_powers, "evolve") / 3.0, 1.0)
    extra_scalars[12] = min(_get_power_amount(player_powers, "strength_up") / 3.0, 1.0)
    # [13] reserved for num_alive_enemies / num_minions ratio (computed below after enemies parsed)

    # Hand
    hand = battle.get("hand") or player.get("hand") or []
    hand_ids = np.zeros(MAX_HAND_SIZE, dtype=np.int64)
    hand_aux = np.zeros((MAX_HAND_SIZE, CARD_AUX_DIM), dtype=np.float32)
    hand_mask = np.zeros(MAX_HAND_SIZE, dtype=bool)
    for i, card in enumerate(hand[:MAX_HAND_SIZE]):
        if isinstance(card, dict):
            card_idx, card_aux = _cached_card_encoding(card, vocab)
            hand_ids[i] = card_idx
            hand_aux[i] = card_aux
            hand_mask[i] = True

    # Enemies
    enemies = state.get("enemies") or battle.get("enemies") or []
    alive = [e for e in enemies if isinstance(e, dict) and e.get("is_alive", True)]
    enemy_ids = np.zeros(MAX_ENEMIES, dtype=np.int64)
    enemy_aux = np.zeros((MAX_ENEMIES, ENEMY_AUX_DIM), dtype=np.float32)
    enemy_mask = np.zeros(MAX_ENEMIES, dtype=bool)
    n_minions = 0
    for i, enemy in enumerate(alive[:MAX_ENEMIES]):
        enemy_ids[i] = _cached_monster_idx(vocab, enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id", ""))
        enemy_aux[i] = _enemy_aux_features(enemy)
        enemy_mask[i] = True
        if _get_power_amount(_player_power_list(enemy), "minion") > 0:
            n_minions += 1

    # Fill the reserved extra_scalars[13] = num_minions / num_alive (cluster ratio)
    # Helps the model recognise multi-minion bosses (the_kin etc.)
    n_alive = max(1, len(alive))
    extra_scalars[13] = float(n_minions) / float(n_alive)

    features = {
        "scalars": scalars,
        "extra_scalars": extra_scalars,
        "hand_ids": hand_ids,
        "hand_aux": hand_aux,
        "hand_mask": hand_mask,
        "enemy_ids": enemy_ids,
        "enemy_aux": enemy_aux,
        "enemy_mask": enemy_mask,
    }

    # Optional: include full deck for build_plan_z bridge
    deck = player.get("deck") or player.get("cards") or state.get("deck") or []
    if deck:
        from rl_encoder_v2 import MAX_DECK_SIZE, CARD_AUX_DIM as NC_CARD_AUX_DIM
        deck_ids = np.zeros(MAX_DECK_SIZE, dtype=np.int64)
        deck_aux = np.zeros((MAX_DECK_SIZE, NC_CARD_AUX_DIM), dtype=np.float32)
        deck_mask = np.zeros(MAX_DECK_SIZE, dtype=bool)
        for i, card in enumerate(deck[:MAX_DECK_SIZE]):
            if isinstance(card, dict):
                card_idx, c_aux = _cached_card_encoding(card, vocab)
                deck_ids[i] = card_idx
                deck_aux[i, :CARD_AUX_DIM] = c_aux  # reuse combat card aux
                deck_mask[i] = True
        features["deck_ids"] = deck_ids
        features["deck_aux"] = deck_aux
        features["deck_mask"] = deck_mask

        # Pile-specific context: encode draw/discard/exhaust piles separately.
        # If actual pile card lists are available (from binary protocol), use them.
        # Otherwise fall back to computing remaining = master_deck - hand.
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        draw_cards = battle.get("draw_pile_cards") or []
        discard_cards = battle.get("discard_pile_cards") or []
        exhaust_cards = battle.get("exhaust_pile_cards") or []

        MAX_PILE = 30  # max cards per pile

        def _encode_pile(card_ids: list, prefix: str):
            pile_ids = np.zeros(MAX_PILE, dtype=np.int64)
            pile_aux = np.zeros((MAX_PILE, CARD_AUX_DIM), dtype=np.float32)
            pile_mask = np.zeros(MAX_PILE, dtype=bool)
            for pi, cid in enumerate(card_ids[:MAX_PILE]):
                if isinstance(cid, str) and cid:
                    idx = vocab.card_to_idx.get(cid, 0)
                    pile_ids[pi] = idx
                    # Minimal aux (no per-card cost/upgrade info from pile list)
                    pile_mask[pi] = True
                elif isinstance(cid, dict):
                    idx, aux = _cached_card_encoding(cid, vocab)
                    pile_ids[pi] = idx
                    pile_aux[pi] = aux
                    pile_mask[pi] = True
            features[f"{prefix}_ids"] = pile_ids
            features[f"{prefix}_aux"] = pile_aux
            features[f"{prefix}_mask"] = pile_mask

        if draw_cards or discard_cards or exhaust_cards:
            _encode_pile(draw_cards, "draw_pile")
            _encode_pile(discard_cards, "discard_pile")
            _encode_pile(exhaust_cards, "exhaust_pile")
        else:
            # Fallback: compute remaining = master_deck - hand,
            # and produce ALL pile keys (draw/discard/exhaust) with same data
            # so _stack_features doesn't get mismatched keys
            hand_set = set()
            for card in (hand[:MAX_HAND_SIZE]):
                if isinstance(card, dict):
                    hand_set.add(card.get("index", -1))

            remain_cards = [c for c in deck if isinstance(c, dict) and c.get("index", -1) not in hand_set]
            _encode_pile(remain_cards, "draw_pile")  # treat remaining as draw pile
            _encode_pile([], "discard_pile")  # empty discard
            _encode_pile([], "exhaust_pile")  # empty exhaust

    return features


def build_combat_action_features(
    state: dict,
    actions: list[dict],
    vocab: Vocab,
) -> dict[str, np.ndarray]:
    """Build action features for combat actions."""
    # Action types for combat
    COMBAT_ACTION_TYPES = [
        "play_card", "end_turn", "use_potion",
        "select_hand_card", "select_card_option",
        "confirm_selection", "cancel_selection", "other",
    ]
    atype_map = {a: i for i, a in enumerate(COMBAT_ACTION_TYPES)}

    n = min(len(actions), MAX_ACTIONS)
    action_type_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    target_card_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    target_enemy_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    action_mask = np.zeros(MAX_ACTIONS, dtype=bool)

    # Pre-extract enemies and hand for target resolution
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = _extract_player(state)
    hand = battle.get("hand") or player.get("hand") or []
    enemies = state.get("enemies") or battle.get("enemies") or []
    alive_enemies = [e for e in enemies if isinstance(e, dict) and e.get("is_alive", True)]

    for i, action in enumerate(actions[:MAX_ACTIONS]):
        action_mask[i] = True
        aname = _lower(action.get("action") or action.get("type", ""))
        action_type_ids[i] = atype_map.get(aname, atype_map["other"])

        idx = _safe_int(action.get("index") or action.get("card_index") or
                        action.get("hand_index"))

        # Card target
        if aname == "play_card":
            cidx = _safe_int(action.get("card_index") or action.get("hand_index") or
                             action.get("index"))
            if 0 <= cidx < len(hand) and isinstance(hand[cidx], dict):
                target_card_ids[i] = _cached_card_idx(vocab, hand[cidx].get("id"))

            # Enemy target
            target = action.get("target") or action.get("target_id")
            if target is not None:
                for e_idx, enemy in enumerate(alive_enemies):
                    eid = enemy.get("entity_id", enemy.get("combat_id", e_idx))
                    if eid == target or e_idx == _safe_int(target):
                        target_enemy_ids[i] = _cached_monster_idx(vocab, enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id", ""))
                        break

    return {
        "action_type_ids": action_type_ids,
        "target_card_ids": target_card_ids,
        "target_enemy_ids": target_enemy_ids,
        "action_mask": action_mask,
    }


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
        self.action_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode combat state and legal actions into hidden representations.

        Returns:
            state_repr: (B, hidden_dim)
            action_repr: (B, A, hidden_dim)
            deck_repr: (B, deck_repr_dim) or None if no deck encoder
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

        # Action encoding
        atype_emb = self.combat_action_type_embed(action_features["action_type_ids"])
        card_emb = self.entity_emb.card_embed(action_features["target_card_ids"])
        enemy_emb_act = self.entity_emb.monster_embed(action_features["target_enemy_ids"])
        action_repr = torch.cat([atype_emb, card_emb, enemy_emb_act], dim=-1)
        action_repr = self.action_proj(action_repr)

        return state_repr, action_repr, deck_repr

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
        state_repr, action_repr, deck_repr = self._encode_state_and_actions(state_features, action_features)

        # Base policy + value (from backbone)
        logits = self.policy_scorer(state_repr, action_repr, action_features["action_mask"])
        value = self.value_head(state_repr).squeeze(-1)

        # Deck-conditioned action delta: deck info directly influences per-action scoring
        if deck_repr is not None and hasattr(self, "deck_action_delta"):
            B, A, _ = action_repr.shape
            deck_exp = deck_repr.unsqueeze(1).expand(-1, A, -1)  # (B, A, deck_dim)
            delta_in = torch.cat([deck_exp, action_repr], dim=-1)  # (B, A, deck_dim+hidden)
            deck_delta = self.deck_action_delta(delta_in).squeeze(-1)  # (B, A)
            deck_delta = deck_delta.masked_fill(~action_features["action_mask"], 0.0)
            logits = logits + self.deck_delta_gate * deck_delta

        # Residual adapter: add deck-conditioned deltas
        if self.residual_adapter and deck_repr is not None and hasattr(self, "delta_logits_head"):
            B, A, _ = action_repr.shape
            deck_expanded = deck_repr.unsqueeze(1).expand(-1, A, -1)  # (B, A, deck_dim)
            state_expanded = state_repr.unsqueeze(1).expand(-1, A, -1)  # (B, A, hidden)
            delta_input = torch.cat([deck_expanded, state_expanded, action_repr], dim=-1)
            delta_logits = self.delta_logits_head(delta_input).squeeze(-1)  # (B, A)
            delta_logits = delta_logits.masked_fill(~action_features["action_mask"], 0.0)
            logits = logits + self.adapter_alpha * delta_logits

            dv_input = torch.cat([deck_repr, state_repr], dim=-1)
            delta_value = self.delta_value_head(dv_input).squeeze(-1)  # (B,)
            value = value + self.adapter_beta * delta_value

        if return_hidden:
            return logits, value, state_repr, action_repr
        return logits, value

    def forward_teacher(
        self,
        state_features: dict[str, torch.Tensor],
        action_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extended forward for the offline combat teacher stack."""
        state_repr, action_repr, deck_repr = self._encode_state_and_actions(state_features, action_features)
        logits = self.policy_scorer(state_repr, action_repr, action_features["action_mask"])
        value = self.value_head(state_repr).squeeze(-1)

        state_expanded = state_repr.unsqueeze(1).expand(-1, action_repr.shape[1], -1)
        action_score_input = torch.cat([state_expanded, action_repr], dim=-1)
        raw_action_scores = self.action_score_head(action_score_input).squeeze(-1)
        action_scores = raw_action_scores.masked_fill(~action_features["action_mask"], -1e9)

        continuation_raw = self.continuation_value_head(state_repr)
        win_prob = torch.sigmoid(continuation_raw[:, 0:1])
        expected_hp_loss = F.softplus(continuation_raw[:, 1:2])
        expected_potion_cost = F.softplus(continuation_raw[:, 2:3])
        continuation = torch.cat([win_prob, expected_hp_loss, expected_potion_cost], dim=-1)
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
                 use_continuation_value: bool = False):
        self.device = device or _auto_device()
        self.network = network.to(self.device)
        self.vocab = vocab
        self.network.eval()
        self._use_amp = self.device.type == "cuda"
        self._use_continuation_value = use_continuation_value

    def evaluate(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> tuple[np.ndarray, float]:
        """Evaluate combat state → (policy, value)."""
        sf = build_combat_features(state, self.vocab)
        af = build_combat_action_features(state, legal_actions, self.vocab)

        state_t = _tensorize_features(sf, self.device)
        action_t = _tensorize_features(af, self.device)

        with torch.no_grad():
            if self._use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    if self._use_continuation_value:
                        logits, _value, _scores, continuation = self.network.forward_teacher(state_t, action_t)
                        # Use win_prob (first column) as value, scaled to [-1, 1]
                        value_scalar = continuation[0, 0].cpu().float().item() * 2.0 - 1.0
                    else:
                        logits, value = self.network.forward(state_t, action_t)
                        value_scalar = value[0].cpu().float().item()
            else:
                if self._use_continuation_value:
                    logits, _value, _scores, continuation = self.network.forward_teacher(state_t, action_t)
                    value_scalar = continuation[0, 0].cpu().float().item() * 2.0 - 1.0
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
