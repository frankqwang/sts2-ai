"""SymbolicFeaturesHead — cross-attention over static symbol sets from source_knowledge.sqlite.

Per-entity cross-attention that gives the RL policy a zero-shot prior over rare
cards/relics/monsters/potions via the structured symbols (powers, commands, tags,
intents) stored in `tools/python/data/source_knowledge.sqlite`.

For each entity in a forward pass:
  - query = the existing (learned) entity embedding from EntityEmbeddings
  - key/value = per-symbol learned embeddings for the symbols attached to THIS entity
  - output = projected attention pooling → (B, L, proj_dim) concatenated into the
    existing encoder input alongside base embed + aux features

The output projection layer is **zero-initialized** so that the head contributes
exactly 0 at construction time. This means loading a baseline checkpoint into a
retrieval-enabled model produces bit-identical forward output at iter 0; training
then grows the symbolic contribution from zero.

Owned by the PPO network (FullRunPolicyNetworkV2). The combat network
(CombatPolicyValueNetwork) receives a reference to the same instance via its
`symbolic_head=` kwarg so both brains share one learned head. Optimizer ownership
is separated at train_hybrid.py construction time — only the PPO optimizer steps
these params; combat's backward still accumulates gradients on them via autograd.

See plan at C:/Users/Administrator/.claude/plans/async-snacking-tome.md §5 for
the full design rationale and the discarded multi-hot alternative.
"""

from __future__ import annotations

# Sys.path bootstrap so `import _path_init` is findable from direct CLI.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # noqa: E402

import _path_init  # noqa: F401,E402

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import Vocab
from source_knowledge_features import (
    KnowledgeMeta,
    build_all_symbol_tables,
)

logger = logging.getLogger(__name__)


class SymbolicFeaturesHead(nn.Module):
    """Cross-attention over per-entity symbol sets.

    Params (proj_dim=16, embed_dim=32, num_heads=4):
      symbol_embed.weight: ~401 × 32 = 12,832
      query_proj:          32*32 + 32 = 1,056
      cross_attn:          ~4,224
      out_proj:            32*16 + 16 = 528  (zero-init)
      TOTAL:               ~18,640 trainable

    Persistent buffers (~200 KB):
      card/relic/monster/potion symbol_ids  (int32)
      card/relic/monster/potion symbol_mask (bool)
      meta_json                              (ASCII bytes, drift detection)

    Forward signature:
      card(ids, query_embs) -> (B, L, proj_dim)
        ids: (B, L) long card vocab indices
        query_embs: (B, L, embed_dim) float — typically from entity_emb.card_embed(ids)
    """

    def __init__(
        self,
        vocab: Vocab,
        db_path: Path | str | None = None,
        embed_dim: int = 32,
        proj_dim: int = 16,
        num_heads: int = 4,
        card_max_len: int = 32,
        relic_max_len: int = 16,
        monster_max_len: int = 32,
        potion_max_len: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads

        # Build static symbol tables from sqlite.
        tables, meta = build_all_symbol_tables(
            vocab,
            db_path=db_path,
            card_max_len=card_max_len,
            relic_max_len=relic_max_len,
            monster_max_len=monster_max_len,
            potion_max_len=potion_max_len,
        )
        self._meta = meta  # for logging; not saved (meta_json buffer is the saved form)

        vocab_size = len(meta.global_symbol_vocab)
        if vocab_size < 2:
            raise RuntimeError(
                f"SymbolicFeaturesHead: global symbol vocab has only {vocab_size} "
                "entries; expected at least 2 (pad + unk). Is source_knowledge.sqlite "
                "empty or corrupt?"
            )

        # Learned symbol embedding table, pad=0.
        self.symbol_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Per-entity static id + mask buffers (persistent).
        def _register(name: str, arr):
            import numpy as np
            t = torch.from_numpy(np.ascontiguousarray(arr))
            self.register_buffer(name, t, persistent=True)

        _register("card_symbol_ids",    tables["card"][0])     # (V_card, M_card) int32
        _register("card_symbol_mask",   tables["card"][1])     # (V_card, M_card) bool
        _register("relic_symbol_ids",   tables["relic"][0])
        _register("relic_symbol_mask",  tables["relic"][1])
        _register("monster_symbol_ids", tables["monster"][0])
        _register("monster_symbol_mask",tables["monster"][1])
        _register("potion_symbol_ids",  tables["potion"][0])
        _register("potion_symbol_mask", tables["potion"][1])

        # Drift-detection blob (ASCII bytes of the KnowledgeMeta JSON).
        meta_bytes = meta.to_json().encode("ascii")
        self.register_buffer(
            "meta_json",
            torch.frombuffer(bytearray(meta_bytes), dtype=torch.uint8).clone(),
            persistent=True,
        )

        # Cross-attention stack (shared across entity types).
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = nn.Linear(embed_dim, proj_dim)

        # CRITICAL: zero-init out_proj so the head contributes exactly zero at
        # construction time. This is the invariant that makes loading a baseline
        # checkpoint into a retrieval-enabled model safe — forward output is
        # bit-identical at iter 0, symbolic learning grows from there.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        logger.info(
            "SymbolicFeaturesHead: global_vocab=%d, embed_dim=%d, proj_dim=%d, num_heads=%d",
            vocab_size, embed_dim, proj_dim, num_heads,
        )
        logger.info(
            "  coverage: card=%.1f%% relic=%.1f%% monster=%.1f%% potion=%.1f%%",
            meta.card_coverage * 100,
            meta.relic_coverage * 100,
            meta.monster_coverage * 100,
            meta.potion_coverage * 100,
        )

    # ------------------------------------------------------------------ core

    def _attend(
        self,
        ids: torch.Tensor,           # (B, L) long
        id_buf: torch.Tensor,        # (V, M) int32
        mask_buf: torch.Tensor,      # (V, M) bool
        query_embs: torch.Tensor,    # (B, L, embed_dim) float
    ) -> torch.Tensor:
        """Per-entity cross-attention → (B, L, proj_dim).

        Each (batch, slot) position performs an independent attention over that
        entity's symbol set. Fully-masked rows (pad entities, missing ids) are
        unmasked at position 0 to avoid softmax NaN, then zeroed out before the
        output projection.
        """
        B, L = ids.shape
        device = ids.device

        # Lookup symbol ids + mask for each entity slot.
        # id_buf[ids] produces (B, L, M) int32; cast to long for embedding lookup.
        sym_ids = id_buf[ids].long()      # (B, L, M)
        sym_mask = mask_buf[ids].bool()   # (B, L, M) — True = valid symbol
        M = sym_ids.shape[-1]

        # Symbol embeddings.
        sym_embs = self.symbol_embed(sym_ids)  # (B, L, M, E)

        # Flatten (B, L) → BL for batched single-query cross-attention.
        BL = B * L
        E = self.embed_dim
        q = self.query_proj(query_embs).reshape(BL, 1, E)  # (BL, 1, E)
        kv = sym_embs.reshape(BL, M, E)                    # (BL, M, E)
        key_padding = ~sym_mask.reshape(BL, M)             # True = ignore

        # Guard fully-masked rows to avoid softmax NaN. We unmask only position
        # 0 in those rows; the output for those rows is zeroed out below, so it
        # doesn't matter what attention returns.
        fully_masked = key_padding.all(dim=-1, keepdim=True)  # (BL, 1)
        unmask_first = (
            torch.arange(M, device=device).unsqueeze(0) == 0
        )  # (1, M)
        safe_km = key_padding & ~(fully_masked & unmask_first)

        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=safe_km)
        attn_out = attn_out.squeeze(1)  # (BL, E)

        # Zero out fully-masked positions so they contribute nothing downstream.
        any_valid = sym_mask.any(dim=-1).reshape(BL, 1).float()  # (BL, 1)
        attn_out = attn_out * any_valid

        out = self.out_proj(attn_out)  # (BL, proj_dim)
        return out.reshape(B, L, self.proj_dim)

    # ------------------------------------------------------------------ public lookups

    def card(self, ids: torch.Tensor, query_embs: torch.Tensor) -> torch.Tensor:
        return self._attend(ids, self.card_symbol_ids, self.card_symbol_mask, query_embs)

    def relic(self, ids: torch.Tensor, query_embs: torch.Tensor) -> torch.Tensor:
        return self._attend(ids, self.relic_symbol_ids, self.relic_symbol_mask, query_embs)

    def monster(self, ids: torch.Tensor, query_embs: torch.Tensor) -> torch.Tensor:
        return self._attend(ids, self.monster_symbol_ids, self.monster_symbol_mask, query_embs)

    def potion(self, ids: torch.Tensor, query_embs: torch.Tensor) -> torch.Tensor:
        return self._attend(ids, self.potion_symbol_ids, self.potion_symbol_mask, query_embs)

    # ------------------------------------------------------------------ introspection

    @property
    def meta(self) -> KnowledgeMeta:
        """Return the KnowledgeMeta from the build used to seed this instance.

        Note: this is the IN-MEMORY meta from construction. After load_state_dict
        the meta_json buffer may disagree — use `meta_from_buffer()` for that.
        """
        return self._meta

    def meta_from_buffer(self) -> KnowledgeMeta:
        """Decode the persistent meta_json buffer back to a KnowledgeMeta."""
        raw = bytes(self.meta_json.cpu().numpy().tobytes()).decode("ascii")
        return KnowledgeMeta.from_json(raw)


__all__ = ["SymbolicFeaturesHead"]
