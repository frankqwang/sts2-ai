"""Structured state/action encoder for encoder_v2 RL architecture.

Converts raw STS2 game state dicts into structured tensors suitable for
the V2 policy network with entity embeddings, attention-based set encoding,
and pointer-style action scoring.

Key design:
  - Every game entity (card, relic, potion, enemy, map node, event option)
    gets a learned embedding vector.
  - Variable-length sets (deck, hand, relics, etc.) are encoded via
    multi-head self-attention + mean pooling.
  - A shared trunk combines all set representations with scalar features.
  - Each screen type has a head that produces context for action scoring.
  - Actions are scored by comparing state representation against entity
    embeddings of the objects each action targets (pointer-style).

This module provides:
  1. `build_structured_state()` — raw state dict → StructuredState tensors
  2. `build_structured_actions()` — raw actions list → StructuredActions tensors
  3. `EntityEmbeddings` — nn.Module for all entity embedding tables
  4. `SetEncoder` — attention-based set encoder
  5. `SharedTrunk` — MLP that combines all representations
  6. `ScreenHeads` — per-screen-type context encoders
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import Vocab, load_vocab, _slugify
from card_tags import (
    load_card_tags, FUNCTIONAL_TAGS, FUNCTIONAL_TAG_TO_IDX, NUM_FUNCTIONAL_TAGS,
)
from relic_tags import (
    load_relic_tags, RELIC_FUNCTIONAL_TAGS, RELIC_TAG_TO_IDX, NUM_RELIC_TAGS,
)
from rl_reward_shaping import extract_next_boss_token

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DECK_SIZE = 50
MAX_HAND_SIZE = 12
MAX_RELICS = 25
MAX_POTIONS = 5
MAX_ENEMIES = 5
MAX_ACTIONS = 30  # max legal actions per step
MAX_MAP_NODES = 6
MAP_ROUTE_DIM = 5  # per-node route features: min_elite, max_shop, max_rest, avg_monster, rows_to_boss
MAX_SHOP_ITEMS = 12
MAX_CARD_REWARDS = 4
MAX_EVENT_OPTIONS = 5
MAX_REST_OPTIONS = 4
MAX_EVENT_CONTEXT = MAX_EVENT_OPTIONS + 1  # event id + option texts
TEXT_TOKEN_BUCKETS = 2048  # hash buckets for text-based embeddings

# Scalar feature count (act, floor, hp, max_hp, gold, energy, etc.)
SCALAR_DIM = 29  # 20 base + 9 problem_vector

# Card auxiliary features (cost_norm, type_onehot[7], rarity_onehot[10], is_upgraded, tags[32])
CARD_AUX_DIM = 19 + NUM_FUNCTIONAL_TAGS  # 51

# Enemy auxiliary features (hp_ratio, block_norm, intent features, powers)
# v3 layout (2026-04-08 PM, wizardly merge — no backward compat, retraining
# from scratch). Bumped from 32 → 40 to take the union of P0's boss-mechanic
# powers and QG's boss-state flags. The removed `ENEMY_AUX_LEGACY_DIM` /
# end-pad shim is gone — there is no Stage 5 / PPO900 checkpoint to load.
#
# Slot reservation table:
#  0      hp_ratio
#  1      max_hp_normalized          (max_hp / 200)
#  2      block_normalized           (block / 50)
#  3      has_attack_intent          (flag)
#  4      has_defend_intent          (flag)
#  5      has_buff_intent            (flag)
#  6      has_debuff_intent          (flag)
#  7      primary_intent_damage / 30 (per-hit, prefers `damage` from new DTO)
#  8      primary_intent_repeats / 5 (multi-hit count)
#  9      is_boss_heuristic          (max_hp >= 80)
# --- legacy 6 powers (kept for vocab continuity) ---
# 10     strength / 10
# 11     vulnerable / 5
# 12     weak / 5
# 13     poison / 20
# 14     artifact / 3
# 15     regen / 10
# --- boss-mechanic powers (P0 + QG union) ---
# 16     slippery / 9                  (Vantom — caps damage to 1, decrements per hit)
# 17     intangible / 5                (all incoming damage capped to 1)
# 18     hardtokill / 5                (cap = stack count, Act 2/3 boss)
# 19     minion / 1                    (flag — kin follower marker)
# 20     metallicize / 10              (passive block per turn)
# 21     barricade / 1                 (flag — block doesn't expire)
# 22     ritual / 5                    (+strength at turn end)
# 23     angry / 5                     (+strength on damage taken)
# 24     curl_up / 30                  (reflex shield trigger)
# 25     thorns / 10                   (recoil damage)
# 26     plated_armor / 30             (decrements on damage taken)
# 27     plating / 10                  (block retain — QG)
# 28     hardenedshell / 10            (initial shield — QG)
# 29     enrage / 5                    (+strength on card play)
# 30     mode_shift / 30               (phase shift threshold)
# 31     flight / 5                    (Byrd-style intangible variant)
# 32     spore_cloud / 5               (debuff on player on death)
# 33     plow / 5                      (ceremonial_beast scaling — QG)
# --- boss-state flags (QG additions, fed by binary protocol after wizardly merge) ---
# 34     is_hittable                   (1 if targetable by single-target cards)
# 35     intends_to_attack             (explicit attack flag from C# snapshot)
# 36     next_move_id_hash / 65536     (md5(next_move_id) % 65536, normalized)
# --- intent stats (P0 additions) ---
# 37     num_intents / 4               (multi-intent boss support)
# 38     total_intent_dmg / 30         (sum of all intent damage, multi-hit aware)
# 39     reserved                      (future expansion)
ENEMY_AUX_DIM = 40

# Map node types
NODE_TYPES = ["unknown", "monster", "elite", "boss", "restsite", "shop", "event", "treasure"]
NODE_TYPE_TO_IDX = {n: i for i, n in enumerate(NODE_TYPES)}

# Action types
ACTION_TYPES = [
    "play_card", "end_turn", "drink_potion", "use_potion",  # combat
    "choose_map_node", "select_card_reward", "skip", "skip_card_reward",  # non-combat
    "choose_rest_option", "shop_purchase", "shop_exit",
    "choose_event_option", "proceed", "advance_dialogue",
    "claim_reward", "select_card", "confirm_selection", "cancel_selection",
    "combat_select_card", "combat_confirm_selection", "select_card_option",
    "other",
]
ACTION_TYPE_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _lower(v: Any) -> str:
    return str(v).strip().lower() if v else ""


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _text_token_id(value: Any, buckets: int = TEXT_TOKEN_BUCKETS) -> int:
    """Map arbitrary text to a stable embedding bucket via hashing.

    Uses hash buckets instead of a closed vocabulary so that event ids and
    option labels can influence policy decisions without a pre-generated vocab.
    """
    if not value:
        return 0
    text = _slugify(str(value)).lower()
    if not text:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return 1 + (int.from_bytes(digest, "little") % (buckets - 1))


def _player_richness(player: dict[str, Any] | None) -> int:
    """Prefer the most informative player payload when multiple copies exist."""
    if not isinstance(player, dict):
        return -1
    score = len(player)
    for key in ("deck", "hand", "relics", "potions", "status"):
        value = player.get(key)
        if isinstance(value, list):
            score += 4 + len(value)
    for key in (
        "hp", "current_hp", "max_hp", "gold", "block", "energy", "max_energy",
        "draw_pile_count", "discard_pile_count", "exhaust_pile_count", "open_potion_slots",
    ):
        if player.get(key) is not None:
            score += 1
    return score


def _extract_player(state: dict) -> dict:
    """Extract player data from state (handles multiple formats).

    Godot sometimes exposes a lightweight top-level player plus a richer
    screen-local copy (for example map.player contains the deck while
    top-level player does not). Prefer the richest payload so the encoder
    sees consistent deck/relic/potion context across backends.
    """
    candidates: list[dict[str, Any]] = []
    top_player = state.get("player")
    if isinstance(top_player, dict):
        candidates.append(top_player)
    for key in ("battle", "map", "shop", "rest_site", "event",
                "rewards", "card_reward", "card_select", "relic_select", "treasure"):
        container = state.get(key)
        if isinstance(container, dict) and isinstance(container.get("player"), dict):
            candidates.append(container["player"])
    if not candidates:
        return {}
    return max(candidates, key=_player_richness)


def _compute_route_features(
    map_data: dict[str, Any],
    next_options: list[dict[str, Any]],
) -> np.ndarray:
    """For each next_option node, compute route lookahead features by BFS to boss.

    Returns (MAX_MAP_NODES, MAP_ROUTE_DIM) float32 array with:
      [0] min_elite: minimum elites on any path to boss (normalized /5)
      [1] max_shop: maximum shops on any path to boss (normalized /3)
      [2] max_rest: maximum rest sites on any path to boss (normalized /5)
      [3] avg_monster: average monsters on paths (normalized /10)
      [4] rows_to_boss: (boss_row - node_row) / 16
    """
    features = np.zeros((MAX_MAP_NODES, MAP_ROUTE_DIM), dtype=np.float32)
    nodes = map_data.get("nodes") or []
    boss = map_data.get("boss") or {}
    if not nodes or not next_options:
        return features

    boss_row = int(boss.get("row") or 0)

    # Build adjacency: (col, row) -> list of (child_col, child_row)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    node_type_map: dict[tuple[int, int], str] = {}
    for n in nodes:
        key = (int(n.get("col", -1)), int(n.get("row", -1)))
        node_type_map[key] = _lower(n.get("type"))
        children = n.get("children") or []
        adj[key] = [(int(c[0]), int(c[1])) for c in children if isinstance(c, (list, tuple)) and len(c) >= 2]

    for opt_idx, opt in enumerate(next_options[:MAX_MAP_NODES]):
        start = (int(opt.get("col", -1)), int(opt.get("row", -1)))
        if start not in node_type_map:
            continue

        # BFS/DFS to collect all paths from start to boss row
        # Use iterative DFS with path tracking, capped to prevent explosion
        stack = [(start, {"elite": 0, "shop": 0, "restsite": 0, "monster": 0})]
        path_stats: list[dict[str, int]] = []
        visited_states: set[tuple[int, int]] = set()
        max_paths = 50  # cap to prevent combinatorial explosion

        while stack and len(path_stats) < max_paths:
            node, counts = stack.pop()
            ntype = node_type_map.get(node, "unknown")
            new_counts = dict(counts)
            if ntype in new_counts:
                new_counts[ntype] += 1

            children = adj.get(node, [])
            if not children or node[1] >= boss_row:
                # Reached leaf or boss row
                path_stats.append(new_counts)
                continue

            for child in children:
                if child not in visited_states or len(path_stats) < 5:
                    stack.append((child, new_counts))
            visited_states.add(node)

        if not path_stats:
            continue

        elites = [p["elite"] for p in path_stats]
        shops = [p["shop"] for p in path_stats]
        rests = [p["restsite"] for p in path_stats]
        monsters = [p["monster"] for p in path_stats]
        rows_to_boss = max(0, boss_row - start[1])

        features[opt_idx, 0] = min(elites) / 5.0
        features[opt_idx, 1] = max(shops) / 3.0
        features[opt_idx, 2] = max(rests) / 5.0
        features[opt_idx, 3] = (sum(monsters) / len(monsters)) / 10.0
        features[opt_idx, 4] = rows_to_boss / 16.0

    return features


def _extract_map_paths(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect map node candidates from screen payload or legal actions."""
    map_state = state.get("map") if isinstance(state.get("map"), dict) else {}
    raw_paths = (
        map_state.get("next_options")
        or map_state.get("available_next_nodes")
        or map_state.get("paths")
        or []
    )
    paths = [node for node in raw_paths if isinstance(node, dict)]
    if paths:
        return paths

    fallback: list[dict[str, Any]] = []
    for action in state.get("legal_actions") or []:
        if not isinstance(action, dict):
            continue
        if _lower(action.get("action")) != "choose_map_node":
            continue
        fallback.append(
            {
                "index": action.get("index"),
                "col": action.get("col"),
                "row": action.get("row"),
                "type": action.get("type") or action.get("point_type") or action.get("node_type") or action.get("label"),
            }
        )
    return fallback


# ---------------------------------------------------------------------------
# Structured data containers
# ---------------------------------------------------------------------------

@dataclass
class StructuredState:
    """Pre-processed state tensors ready for the V2 network.

    All tensors are unbatched (no batch dim). Batching happens in the
    rollout buffer / training loop.
    """
    # Scalars
    scalars: np.ndarray  # (SCALAR_DIM,)

    # Entity ID indices (for embedding lookup)
    deck_ids: np.ndarray        # (MAX_DECK_SIZE,) int
    deck_aux: np.ndarray        # (MAX_DECK_SIZE, CARD_AUX_DIM) float
    deck_mask: np.ndarray       # (MAX_DECK_SIZE,) bool
    relic_ids: np.ndarray       # (MAX_RELICS,) int
    relic_aux: np.ndarray       # (MAX_RELICS, NUM_RELIC_TAGS) float — tag multi-hot
    relic_mask: np.ndarray      # (MAX_RELICS,) bool
    potion_ids: np.ndarray      # (MAX_POTIONS,) int
    potion_mask: np.ndarray     # (MAX_POTIONS,) bool

    # Combat-specific
    hand_ids: np.ndarray        # (MAX_HAND_SIZE,) int
    hand_aux: np.ndarray        # (MAX_HAND_SIZE, CARD_AUX_DIM) float
    hand_mask: np.ndarray       # (MAX_HAND_SIZE,) bool
    enemy_ids: np.ndarray       # (MAX_ENEMIES,) int
    enemy_aux: np.ndarray       # (MAX_ENEMIES, ENEMY_AUX_DIM) float
    enemy_mask: np.ndarray      # (MAX_ENEMIES,) bool

    # Screen type
    screen_type: str            # raw screen type string
    screen_type_idx: int        # index into SCREEN_TYPES
    next_boss_idx: int          # hashed token for upcoming boss identity

    # Screen-specific data (variable per screen)
    # Map
    map_node_types: np.ndarray   # (MAX_MAP_NODES,) int — node type index
    map_node_mask: np.ndarray    # (MAX_MAP_NODES,) bool
    map_route_features: np.ndarray  # (MAX_MAP_NODES, MAP_ROUTE_DIM) float — route lookahead per option
    # Card reward
    reward_card_ids: np.ndarray  # (MAX_CARD_REWARDS,) int
    reward_card_aux: np.ndarray  # (MAX_CARD_REWARDS, CARD_AUX_DIM) float
    reward_card_mask: np.ndarray # (MAX_CARD_REWARDS,) bool
    # Shop
    shop_card_ids: np.ndarray    # (MAX_SHOP_ITEMS,) int
    shop_relic_ids: np.ndarray   # (MAX_SHOP_ITEMS,) int
    shop_potion_ids: np.ndarray  # (MAX_SHOP_ITEMS,) int
    shop_prices: np.ndarray      # (MAX_SHOP_ITEMS,) float — normalized price
    shop_mask: np.ndarray        # (MAX_SHOP_ITEMS,) bool
    # Event
    event_option_count: int
    # Rest
    rest_option_ids: np.ndarray  # (MAX_REST_OPTIONS,) int — 0=rest, 1=smith, 2=other...
    rest_option_mask: np.ndarray # (MAX_REST_OPTIONS,) bool


@dataclass
class StructuredActions:
    """Pre-processed action tensors for pointer-style scoring."""
    # Action type indices
    action_type_ids: np.ndarray  # (MAX_ACTIONS,) int
    # Entity that each action targets (card/relic/potion/node/option index)
    target_card_ids: np.ndarray  # (MAX_ACTIONS,) int — card vocab idx or 0
    target_enemy_ids: np.ndarray # (MAX_ACTIONS,) int — monster vocab idx or 0 (combat targeting)
    target_node_types: np.ndarray  # (MAX_ACTIONS,) int — map node type or 0
    target_indices: np.ndarray   # (MAX_ACTIONS,) int — raw action index
    # Mask
    action_mask: np.ndarray      # (MAX_ACTIONS,) bool
    num_actions: int


# ---------------------------------------------------------------------------
# State building
# ---------------------------------------------------------------------------

COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}

# Screen type list (shared with legacy for compatibility)
SCREEN_TYPES = [
    "combat", "monster", "elite", "boss", "map", "card_reward",
    "rest_site", "shop", "event", "combat_rewards", "treasure",
    "card_select", "relic_select", "overlay", "combat_pending",
    "hand_select", "rewards", "other",
]
SCREEN_TYPE_TO_IDX = {s: i for i, s in enumerate(SCREEN_TYPES)}


def build_structured_state(state: dict, vocab: Vocab) -> StructuredState:
    """Convert raw game state dict into structured tensors."""

    state_type = _lower(state.get("state_type"))
    player = _extract_player(state)
    run = state.get("run") if isinstance(state.get("run"), dict) else {}

    # --- Scalars ---
    scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
    scalars[0] = _safe_int(run.get("act"), 1) / 4.0
    scalars[1] = _safe_int(run.get("floor"), 0) / 20.0
    hp = _safe_float(player.get("hp", player.get("current_hp")))
    max_hp = max(1.0, _safe_float(player.get("max_hp"), 1))
    scalars[2] = hp / max_hp  # hp_ratio
    scalars[3] = max_hp / 100.0
    scalars[4] = min(_safe_float(player.get("gold")) / 300.0, 1.0)
    # Combat-only scalars. Some backends leak lightweight combat fields into
    # non-combat screens, so keep them zero outside actual combat states.
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    if state_type in COMBAT_SCREENS:
        scalars[5] = _safe_float(battle.get("energy") or player.get("energy")) / 5.0
        scalars[6] = _safe_float(battle.get("max_energy") or player.get("max_energy")) / 5.0
        scalars[7] = _safe_float(player.get("block")) / 50.0
        scalars[8] = _safe_float(state.get("round_number") or battle.get("round_number")) / 20.0
    # Open potion slots
    potions_list = player.get("potions") if isinstance(player.get("potions"), list) else []
    max_potion_slots = _safe_int(player.get("max_potions"), 3)
    scalars[9] = len(potions_list) / max(1, max_potion_slots)
    # Deck size normalized
    deck_list = player.get("deck") if isinstance(player.get("deck"), list) else []
    scalars[10] = min(len(deck_list) / 40.0, 1.0)
    # Relic count normalized
    relic_list = player.get("relics") if isinstance(player.get("relics"), list) else []
    scalars[11] = min(len(relic_list) / 20.0, 1.0)

    # --- Deck statistics (STRATEGY KNOWLEDGE features) ---
    # These help NN understand deck composition without reading every card
    _n_deck = max(1, len(deck_list))
    _n_attacks = 0
    _n_skills = 0
    _n_powers = 0
    _n_basic = 0  # strike/defend
    _total_cost = 0
    _n_draw = 0
    _n_aoe = 0
    for c in deck_list:
        if not isinstance(c, dict):
            continue
        cid = _lower(c.get("id", ""))
        ctype = _lower(c.get("type", c.get("card_type", "")))
        cost = _safe_int(c.get("cost"), 1)
        _total_cost += cost
        if "strike" in cid or "defend" in cid:
            _n_basic += 1
        if ctype == "attack":
            _n_attacks += 1
        elif ctype == "skill":
            _n_skills += 1
        elif ctype == "power":
            _n_powers += 1
    scalars[12] = _n_attacks / _n_deck           # attack ratio
    scalars[13] = _n_skills / _n_deck            # skill ratio
    scalars[14] = min(_n_powers / 5.0, 1.0)      # power count normalized
    scalars[15] = _n_basic / _n_deck              # basic card ratio (lower = better deck)
    scalars[16] = _total_cost / max(1, _n_deck) / 3.0  # avg cost normalized
    scalars[17] = min(_n_deck / 30.0, 1.0)       # deck size (redundant with [10] but finer)

    # --- Problem vector (9 dims): deck capability assessment ---
    from rl_reward_shaping import compute_problem_vector
    pv = compute_problem_vector(state)
    scalars[20:29] = pv  # [frontload, aoe, block, draw, energy, scaling, consistency, elite_ready, boss_answer]
    scalars[18] = 0.0
    scalars[19] = 0.0

    # --- Deck ---
    deck_ids = np.zeros(MAX_DECK_SIZE, dtype=np.int64)
    deck_aux = np.zeros((MAX_DECK_SIZE, CARD_AUX_DIM), dtype=np.float32)
    deck_mask = np.zeros(MAX_DECK_SIZE, dtype=bool)
    for i, card in enumerate(deck_list[:MAX_DECK_SIZE]):
        if isinstance(card, dict):
            card_idx, card_aux = _cached_card_encoding(card, vocab)
            deck_ids[i] = card_idx
            deck_aux[i] = card_aux
            deck_mask[i] = True

    # --- Relics ---
    relic_ids = np.zeros(MAX_RELICS, dtype=np.int64)
    relic_aux = np.zeros((MAX_RELICS, NUM_RELIC_TAGS), dtype=np.float32)
    relic_mask = np.zeros(MAX_RELICS, dtype=bool)
    relic_tag_data = _get_relic_tags()
    for i, relic in enumerate(relic_list[:MAX_RELICS]):
        if isinstance(relic, dict):
            relic_ids[i] = _cached_relic_idx(vocab, relic.get("id"))
            relic_mask[i] = True
            # Relic tag features
            rid = relic.get("id")
            if rid:
                slug = _slugify(str(rid)).lower()
                for tag_name in relic_tag_data.get(slug, []):
                    tidx = RELIC_TAG_TO_IDX.get(tag_name)
                    if tidx is not None:
                        relic_aux[i, tidx] = 1.0

    # --- Potions ---
    potion_ids = np.zeros(MAX_POTIONS, dtype=np.int64)
    potion_mask = np.zeros(MAX_POTIONS, dtype=bool)
    for i, potion in enumerate(potions_list[:MAX_POTIONS]):
        if isinstance(potion, dict):
            potion_ids[i] = _cached_potion_idx(vocab, potion.get("id"))
            potion_mask[i] = True

    # --- Hand (combat) ---
    hand_ids = np.zeros(MAX_HAND_SIZE, dtype=np.int64)
    hand_aux = np.zeros((MAX_HAND_SIZE, CARD_AUX_DIM), dtype=np.float32)
    hand_mask = np.zeros(MAX_HAND_SIZE, dtype=bool)
    if state_type in COMBAT_SCREENS:
        hand = (battle.get("hand") or player.get("hand") or [])
        for i, card in enumerate(hand[:MAX_HAND_SIZE]):
            if isinstance(card, dict):
                card_idx, card_aux = _cached_card_encoding(card, vocab)
                hand_ids[i] = card_idx
                hand_aux[i] = card_aux
                hand_mask[i] = True

    # --- Enemies (combat) ---
    enemy_ids = np.zeros(MAX_ENEMIES, dtype=np.int64)
    enemy_aux = np.zeros((MAX_ENEMIES, ENEMY_AUX_DIM), dtype=np.float32)
    enemy_mask = np.zeros(MAX_ENEMIES, dtype=bool)
    if state_type in COMBAT_SCREENS:
        enemies = state.get("enemies") or battle.get("enemies") or []
        alive = [e for e in enemies if isinstance(e, dict) and e.get("is_alive", True)]
        for i, enemy in enumerate(alive[:MAX_ENEMIES]):
            enemy_ids[i] = _cached_monster_idx(vocab, enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id", ""))
            enemy_aux[i] = _enemy_aux_features(enemy)
            enemy_mask[i] = True

    # --- Screen type ---
    screen_key = state_type if state_type in SCREEN_TYPE_TO_IDX else "other"
    screen_type_idx = SCREEN_TYPE_TO_IDX.get(screen_key, SCREEN_TYPE_TO_IDX["other"])
    next_boss_idx = _text_token_id(extract_next_boss_token(state))

    # --- Map nodes ---
    map_node_types = np.zeros(MAX_MAP_NODES, dtype=np.int64)
    map_node_mask = np.zeros(MAX_MAP_NODES, dtype=bool)
    map_route_features = np.zeros((MAX_MAP_NODES, MAP_ROUTE_DIM), dtype=np.float32)
    if state_type == "map":
        paths = _extract_map_paths(state)
        for i, node in enumerate(paths[:MAX_MAP_NODES]):
            if isinstance(node, dict):
                ntype = _lower(node.get("type") or node.get("point_type") or node.get("label"))
                map_node_types[i] = NODE_TYPE_TO_IDX.get(ntype, NODE_TYPE_TO_IDX["unknown"])
                map_node_mask[i] = True
        # Route lookahead features from full map topology
        map_data = state.get("map") if isinstance(state.get("map"), dict) else {}
        if map_data.get("nodes"):
            map_route_features = _compute_route_features(map_data, paths)

    # --- Card reward ---
    reward_card_ids = np.zeros(MAX_CARD_REWARDS, dtype=np.int64)
    reward_card_aux = np.zeros((MAX_CARD_REWARDS, CARD_AUX_DIM), dtype=np.float32)
    reward_card_mask = np.zeros(MAX_CARD_REWARDS, dtype=bool)
    if state_type == "card_reward":
        cards = _card_reward_cards_from_state_or_actions(state)
        for i, card in enumerate(cards[:MAX_CARD_REWARDS]):
            if isinstance(card, dict):
                card_idx, card_aux = _cached_card_encoding(card, vocab)
                reward_card_ids[i] = card_idx
                reward_card_aux[i] = card_aux
                reward_card_mask[i] = True

    # --- Shop items ---
    shop_card_ids = np.zeros(MAX_SHOP_ITEMS, dtype=np.int64)
    shop_relic_ids = np.zeros(MAX_SHOP_ITEMS, dtype=np.int64)
    shop_potion_ids = np.zeros(MAX_SHOP_ITEMS, dtype=np.int64)
    shop_prices = np.zeros(MAX_SHOP_ITEMS, dtype=np.float32)
    shop_mask = np.zeros(MAX_SHOP_ITEMS, dtype=bool)
    if state_type == "shop":
        shop = state.get("shop") or {}
        items = shop.get("items") or []
        for i, item in enumerate(items[:MAX_SHOP_ITEMS]):
            if isinstance(item, dict):
                shop_mask[i] = True
                shop_prices[i] = min(_safe_float(item.get("price")) / 300.0, 1.0)
                cat = _lower(item.get("category"))
                if cat == "card" or "card_id" in item:
                    shop_card_ids[i] = _cached_card_idx(vocab, item.get("card_id") or item.get("id", ""))
                elif cat == "relic" or "relic_id" in item:
                    shop_relic_ids[i] = _cached_relic_idx(vocab, item.get("relic_id") or item.get("id", ""))
                elif cat == "potion" or "potion_id" in item:
                    shop_potion_ids[i] = _cached_potion_idx(vocab, item.get("potion_id") or item.get("id", ""))

    # --- Event options ---
    event_option_count = 0
    if state_type == "event":
        event = state.get("event") or {}
        event_option_count = len(event.get("options") or [])

    # --- Rest options ---
    rest_option_ids = np.zeros(MAX_REST_OPTIONS, dtype=np.int64)
    rest_option_mask = np.zeros(MAX_REST_OPTIONS, dtype=bool)
    REST_OPTION_MAP = {"rest": 0, "heal": 0, "smith": 1, "upgrade": 1,
                       "recall": 2, "dig": 3, "lift": 4, "toke": 5}
    if state_type == "rest_site":
        rs = state.get("rest_site") or {}
        options = rs.get("options") or []
        for i, opt in enumerate(options[:MAX_REST_OPTIONS]):
            if isinstance(opt, dict):
                opt_id = _lower(opt.get("id"))
                rest_option_ids[i] = REST_OPTION_MAP.get(opt_id, 6)
                rest_option_mask[i] = True

    return StructuredState(
        scalars=scalars,
        deck_ids=deck_ids, deck_aux=deck_aux, deck_mask=deck_mask,
        relic_ids=relic_ids, relic_aux=relic_aux, relic_mask=relic_mask,
        potion_ids=potion_ids, potion_mask=potion_mask,
        hand_ids=hand_ids, hand_aux=hand_aux, hand_mask=hand_mask,
        enemy_ids=enemy_ids, enemy_aux=enemy_aux, enemy_mask=enemy_mask,
        screen_type=state_type, screen_type_idx=screen_type_idx,
        next_boss_idx=next_boss_idx,
        map_node_types=map_node_types, map_node_mask=map_node_mask,
        map_route_features=map_route_features,
        reward_card_ids=reward_card_ids, reward_card_aux=reward_card_aux,
        reward_card_mask=reward_card_mask,
        shop_card_ids=shop_card_ids, shop_relic_ids=shop_relic_ids,
        shop_potion_ids=shop_potion_ids, shop_prices=shop_prices,
        shop_mask=shop_mask,
        event_option_count=event_option_count,
        rest_option_ids=rest_option_ids, rest_option_mask=rest_option_mask,
    )


# ---------------------------------------------------------------------------
# Action building
# ---------------------------------------------------------------------------

def build_structured_actions(
    state: dict,
    actions: list[dict],
    vocab: Vocab,
) -> StructuredActions:
    """Convert raw action list into structured tensors for pointer scoring."""
    n = min(len(actions), MAX_ACTIONS)

    action_type_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    target_card_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    target_enemy_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    target_node_types = np.zeros(MAX_ACTIONS, dtype=np.int64)
    target_indices = np.zeros(MAX_ACTIONS, dtype=np.int64)
    action_mask = np.zeros(MAX_ACTIONS, dtype=bool)

    state_type = _lower(state.get("state_type"))

    # Pre-extract enemies for combat target resolution
    enemies_list = []
    if state_type in COMBAT_SCREENS:
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        enemies_list = state.get("enemies") or battle.get("enemies") or []
        enemies_list = [e for e in enemies_list if isinstance(e, dict) and e.get("is_alive", True)]

    for i, action in enumerate(actions[:MAX_ACTIONS]):
        action_mask[i] = True
        action_name = _lower(action.get("action"))
        action_type_ids[i] = ACTION_TYPE_TO_IDX.get(action_name,
                                                      ACTION_TYPE_TO_IDX["other"])
        idx = _safe_int(action.get("index", action.get("card_index", 0)))
        target_indices[i] = idx

        # Resolve target entity based on action type
        if action_name == "play_card":
            # Card being played
            battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
            player = _extract_player(state)
            hand = battle.get("hand") or player.get("hand") or []
            if 0 <= idx < len(hand) and isinstance(hand[idx], dict):
                target_card_ids[i] = _cached_card_idx(vocab, hand[idx].get("id"))
            # Enemy target (if card targets a specific enemy)
            target = action.get("target")
            target_id = action.get("target_id")
            if target is not None or target_id is not None:
                tid = target if target is not None else target_id
                # Find matching enemy by entity_id or index
                for e_idx, enemy in enumerate(enemies_list):
                    eid = enemy.get("entity_id", enemy.get("combat_id", e_idx))
                    if eid == tid or e_idx == _safe_int(tid):
                        target_enemy_ids[i] = _cached_monster_idx(vocab, enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id", ""))
                        break

        elif action_name == "select_card_reward":
            cards = _card_reward_cards_from_state_or_actions(state)
            if 0 <= idx < len(cards) and isinstance(cards[idx], dict):
                target_card_ids[i] = _cached_card_idx(vocab, cards[idx].get("id"))
            elif action.get("card_id"):
                target_card_ids[i] = _cached_card_idx(vocab, action.get("card_id"))

        elif action_name == "choose_map_node":
            paths = _extract_map_paths(state)
            if 0 <= idx < len(paths) and isinstance(paths[idx], dict):
                ntype = _lower(paths[idx].get("type") or paths[idx].get("point_type") or paths[idx].get("label"))
                target_node_types[i] = NODE_TYPE_TO_IDX.get(ntype,
                                                             NODE_TYPE_TO_IDX["unknown"])

        elif action_name == "shop_purchase":
            shop = state.get("shop") or {}
            items = shop.get("items") or []
            if 0 <= idx < len(items) and isinstance(items[idx], dict):
                item = items[idx]
                cat = _lower(item.get("category"))
                if cat == "card" or "card_id" in item:
                    target_card_ids[i] = _cached_card_idx(vocab, item.get("card_id") or item.get("id", ""))

        elif action_name in ("select_card", "select_card_option"):
            # Card selection screen (upgrade, remove, etc.)
            cs = state.get("card_select") or {}
            cards = cs.get("cards") or []
            ci = _safe_int(action.get("card_index", action.get("index", -1)))
            if 0 <= ci < len(cards) and isinstance(cards[ci], dict):
                target_card_ids[i] = _cached_card_idx(vocab, cards[ci].get("id"))

        elif action_name in ("choose_event_option", "choose_rest_option"):
            # Use target_indices to distinguish options (index is the option number)
            target_indices[i] = idx

        elif action_name in ("drink_potion", "use_potion"):
            # Enemy target for targeted potions
            target = action.get("target") or action.get("target_id")
            if target is not None:
                for e_idx, enemy in enumerate(enemies_list):
                    eid = enemy.get("entity_id", enemy.get("combat_id", e_idx))
                    if eid == target or e_idx == _safe_int(target):
                        target_enemy_ids[i] = _cached_monster_idx(vocab, enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id", ""))
                        break

    return StructuredActions(
        action_type_ids=action_type_ids,
        target_card_ids=target_card_ids,
        target_enemy_ids=target_enemy_ids,
        target_node_types=target_node_types,
        target_indices=target_indices,
        action_mask=action_mask,
        num_actions=n,
    )


# ---------------------------------------------------------------------------
# Auxiliary feature builders
# ---------------------------------------------------------------------------

# Module-level tag caches (loaded lazily)
_CARD_TAG_CACHE: dict[str, list[str]] | None = None
_RELIC_TAG_CACHE: dict[str, list[str]] | None = None
_CARD_IDX_CACHE: dict[tuple[int, str], int] = {}
_CARD_ENCODING_CACHE: dict[tuple[int, str, int, str, str, bool], tuple[int, np.ndarray]] = {}
_RELIC_IDX_CACHE: dict[tuple[int, str], int] = {}
_POTION_IDX_CACHE: dict[tuple[int, str], int] = {}
_MONSTER_IDX_CACHE: dict[tuple[int, str], int] = {}


def _bounded_cache_put(cache: dict, key: Any, value: Any, *, max_size: int = 32768) -> Any:
    if len(cache) >= max_size:
        cache.clear()
    cache[key] = value
    return value


def _get_card_tags() -> dict[str, list[str]]:
    """Lazy-load card tags from card_tags.json."""
    global _CARD_TAG_CACHE
    if _CARD_TAG_CACHE is None:
        try:
            _CARD_TAG_CACHE = load_card_tags()
        except FileNotFoundError:
            _CARD_TAG_CACHE = {}
    return _CARD_TAG_CACHE


def _get_relic_tags() -> dict[str, list[str]]:
    """Lazy-load relic tags from relic_tags.json."""
    global _RELIC_TAG_CACHE
    if _RELIC_TAG_CACHE is None:
        try:
            _RELIC_TAG_CACHE = load_relic_tags()
        except FileNotFoundError:
            _RELIC_TAG_CACHE = {}
    return _RELIC_TAG_CACHE


def _cached_card_encoding(card: dict, vocab: Vocab) -> tuple[int, np.ndarray]:
    card_id = _lower(card.get("id"))
    key = (
        id(vocab),
        card_id,
        _safe_int(card.get("cost"), 0),
        _lower(card.get("type")),
        _lower(card.get("rarity")),
        bool(card.get("is_upgraded")),
    )
    cached = _CARD_ENCODING_CACHE.get(key)
    if cached is not None:
        return cached
    encoded = (vocab.card_idx(card_id), _card_aux_features(card, vocab))
    return _bounded_cache_put(_CARD_ENCODING_CACHE, key, encoded)


def _cached_card_idx(vocab: Vocab, card_id: Any) -> int:
    text = _lower(card_id)
    key = (id(vocab), text)
    cached = _CARD_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    return _bounded_cache_put(_CARD_IDX_CACHE, key, vocab.card_idx(text))


def _cached_relic_idx(vocab: Vocab, relic_id: Any) -> int:
    text = _lower(relic_id)
    key = (id(vocab), text)
    cached = _RELIC_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    return _bounded_cache_put(_RELIC_IDX_CACHE, key, vocab.relic_idx(text))


def _cached_potion_idx(vocab: Vocab, potion_id: Any) -> int:
    text = _lower(potion_id)
    key = (id(vocab), text)
    cached = _POTION_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    return _bounded_cache_put(_POTION_IDX_CACHE, key, vocab.potion_idx(text))


def _normalize_monster_token(monster_id: Any) -> str:
    text = _lower(monster_id)
    if not text:
        return ""
    # Runtime entity ids often append combat instance suffixes like `_0` / `-1`.
    normalized = re.sub(r"[-_]\d+$", "", text)
    return normalized or text


def _cached_monster_idx(vocab: Vocab, monster_id: Any) -> int:
    text = _normalize_monster_token(monster_id)
    key = (id(vocab), text)
    cached = _MONSTER_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    return _bounded_cache_put(_MONSTER_IDX_CACHE, key, vocab.monster_idx(text))


def _card_reward_cards_from_state_or_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    cr = state.get("card_reward") or {}
    cards = cr.get("cards") or []
    normalized_cards = [card for card in cards if isinstance(card, dict)]
    if normalized_cards:
        return normalized_cards

    synthesized: list[dict[str, Any]] = []
    for action in state.get("legal_actions") or []:
        if not isinstance(action, dict):
            continue
        if _lower(action.get("action")) != "select_card_reward":
            continue
        card_id = action.get("card_id")
        if not card_id:
            continue
        synthesized.append(
            {
                "index": action.get("index", action.get("card_index")),
                "id": card_id,
                "name": action.get("label"),
                "type": action.get("card_type"),
                "rarity": action.get("card_rarity"),
                "cost": action.get("cost"),
                "is_upgraded": bool(action.get("is_upgraded")),
            }
        )
    synthesized.sort(key=lambda card: _safe_int(card.get("index"), 999))
    return synthesized


def _card_aux_features(card: dict, vocab: Vocab) -> np.ndarray:
    """Build auxiliary features for a card (cost, type, rarity, upgraded, tags)."""
    feat = np.zeros(CARD_AUX_DIM, dtype=np.float32)

    # Cost (normalized, -1 for unplayable, -2 for X)
    cost = _safe_float(card.get("cost"), 0)
    feat[0] = cost / 5.0

    # Card type one-hot (7 types)
    ctype = _lower(card.get("type"))
    type_map = {"attack": 0, "skill": 1, "power": 2, "status": 3,
                "curse": 4, "quest": 5, "none": 6}
    tidx = type_map.get(ctype, 6)
    feat[1 + tidx] = 1.0

    # Rarity one-hot (10 rarities)
    rarity = _lower(card.get("rarity"))
    rarity_map = {"basic": 0, "common": 1, "uncommon": 2, "rare": 3,
                  "ancient": 4, "event": 5, "token": 6, "status": 7,
                  "curse": 8, "none": 9}
    ridx = rarity_map.get(rarity, 9)
    # If rarity not in state, try to get from vocab
    if rarity == "" and isinstance(card.get("id"), str):
        cidx = vocab.card_idx(_lower(card["id"]))
        if cidx >= 2 and cidx < len(vocab.card_props):
            ridx = vocab.card_props[cidx].get("rarity_idx", 9)
            tidx_from_vocab = vocab.card_props[cidx].get("type_idx", 6)
            # Backfill type if missing
            if ctype == "":
                feat[1:8] = 0
                feat[1 + tidx_from_vocab] = 1.0
    feat[8 + ridx] = 1.0

    # Is upgraded
    feat[18] = 1.0 if card.get("is_upgraded") else 0.0

    # Card tags — multi-hot functional tag vector (32 dims starting at index 19)
    card_id = card.get("id")
    if card_id:
        card_tags = _get_card_tags()
        # Normalize to slug: "AdaptiveStrike" → "adaptive_strike", "Strike_R" → "strike_r"
        slug = _slugify(str(card_id)).lower()
        tag_list = card_tags.get(slug, [])
        for tag_name in tag_list:
            tag_idx = FUNCTIONAL_TAG_TO_IDX.get(tag_name)
            if tag_idx is not None:
                feat[19 + tag_idx] = 1.0

    return feat


def _get_enemy_power(enemy: dict, power_id: str) -> float:
    """Extract a specific power's stack count from enemy.

    Reads ONLY the first non-empty source from (status, powers, power_list,
    buffs, debuffs) to avoid the 3x double-count caused by the pipe
    duplicating power data across multiple field names.
    """
    powers = None
    for key in ("status", "powers", "power_list", "buffs", "debuffs"):
        v = enemy.get(key)
        if isinstance(v, list) and v:
            powers = v
            break
    if not powers:
        return 0.0
    for p in powers:
        if isinstance(p, dict):
            pid = _lower(p.get("id") or p.get("power_id", ""))
            if power_id in pid:
                return _safe_float(p.get("amount") or p.get("stacks"), 0)
    return 0.0


def _enemy_is_minion(enemy: dict) -> bool:
    return _get_enemy_power(enemy, "minion") > 0


def _enemy_aux_features(enemy: dict) -> np.ndarray:
    """Build auxiliary features for an enemy. See `ENEMY_AUX_DIM` slot table
    above for the v3 layout (40 slots, 2026-04-08 PM wizardly merge)."""
    feat = np.zeros(ENEMY_AUX_DIM, dtype=np.float32)

    hp = _safe_float(enemy.get("current_hp", enemy.get("hp")))
    max_hp = max(1.0, _safe_float(enemy.get("max_hp"), 1))
    feat[0] = hp / max_hp  # HP ratio
    feat[1] = max_hp / 200.0  # Max HP normalized
    feat[2] = _safe_float(enemy.get("block")) / 50.0

    # Intent — C# sends nested intents[] with `damage` (per-hit) + `repeats`
    # post-2026-04-08 binary protocol. Falls back to legacy `total_damage` /
    # `hits` / `multiplier` for older sims and to flat `intent_type` /
    # `intent_damage` / `intent_hits` for the JSON-only path.
    intents = enemy.get("intents") or []
    num_intents = len(intents) if isinstance(intents, list) else 0
    total_intent_dmg = 0.0
    if intents and isinstance(intents[0], dict):
        intent_types = [_lower(intent.get("type", "")) for intent in intents if isinstance(intent, dict)]
        intent = " ".join(intent_types)
        primary_intent = next(
            (
                intent for intent in intents
                if isinstance(intent, dict)
                and (_safe_float(intent.get("damage", intent.get("total_damage", 0))) > 0)
            ),
            intents[0],
        )
        # Prefer per-hit `damage` (post-2026-04-08 DTO), fall back to total_damage
        intent_dmg = _safe_float(primary_intent.get("damage", primary_intent.get("total_damage", 0)))
        # Prefer explicit `repeats` (post-2026-04-08 DTO), fall back to legacy `hits`
        intent_hits = _safe_float(
            primary_intent.get(
                "repeats",
                primary_intent.get("hits", primary_intent.get("multiplier", 1)),
            ),
            1,
        )
        for it in intents:
            if isinstance(it, dict):
                total_intent_dmg += _safe_float(it.get("total_damage", it.get("damage", 0)))
    else:
        intent = _lower(enemy.get("intent_type", enemy.get("intent", "")))
        intent_dmg = _safe_float(enemy.get("intent_damage", 0))
        intent_hits = _safe_float(enemy.get("intent_hits", 1))
        total_intent_dmg = intent_dmg
    feat[3] = 1.0 if "attack" in intent else 0.0
    feat[4] = 1.0 if "defend" in intent or "block" in intent else 0.0
    feat[5] = 1.0 if ("buff" in intent and "debuff" not in intent) else 0.0
    feat[6] = 1.0 if "debuff" in intent else 0.0
    feat[7] = intent_dmg / 30.0
    feat[8] = intent_hits / 5.0

    # Is boss (heuristic: max_hp >= 80)
    feat[9] = 1.0 if max_hp >= 80 else 0.0

    # --- legacy 6 powers ---
    feat[10] = _get_enemy_power(enemy, "strength") / 10.0
    feat[11] = min(_get_enemy_power(enemy, "vulnerable") / 5.0, 1.0)
    feat[12] = min(_get_enemy_power(enemy, "weak") / 5.0, 1.0)
    feat[13] = min(_get_enemy_power(enemy, "poison") / 20.0, 1.0)
    feat[14] = min(_get_enemy_power(enemy, "artifact") / 3.0, 1.0)
    feat[15] = _get_enemy_power(enemy, "regen") / 10.0

    # --- boss-mechanic powers (P0 + QG union) ---
    feat[16] = min(_get_enemy_power(enemy, "slippery") / 9.0, 1.0)
    feat[17] = min(_get_enemy_power(enemy, "intangible") / 5.0, 1.0)
    feat[18] = min(_get_enemy_power(enemy, "hardtokill") / 5.0, 1.0)
    feat[19] = 1.0 if _enemy_is_minion(enemy) else 0.0
    feat[20] = min(_get_enemy_power(enemy, "metallicize") / 10.0, 1.0)
    feat[21] = min(_get_enemy_power(enemy, "barricade") / 1.0, 1.0)
    feat[22] = min(_get_enemy_power(enemy, "ritual") / 5.0, 1.0)
    feat[23] = min(_get_enemy_power(enemy, "angry") / 5.0, 1.0)
    feat[24] = min(_get_enemy_power(enemy, "curl_up") / 30.0, 1.0)
    feat[25] = min(_get_enemy_power(enemy, "thorns") / 10.0, 1.0)
    feat[26] = min(_get_enemy_power(enemy, "plated_armor") / 30.0, 1.0)
    feat[27] = min(_get_enemy_power(enemy, "plating") / 10.0, 1.0)
    feat[28] = min(_get_enemy_power(enemy, "hardenedshell") / 10.0, 1.0)
    feat[29] = min(_get_enemy_power(enemy, "enrage") / 5.0, 1.0)
    feat[30] = min(_get_enemy_power(enemy, "mode_shift") / 30.0, 1.0)
    feat[31] = min(_get_enemy_power(enemy, "flight") / 5.0, 1.0)
    feat[32] = min(_get_enemy_power(enemy, "spore_cloud") / 5.0, 1.0)
    feat[33] = min(_get_enemy_power(enemy, "plow") / 5.0, 1.0)

    # --- boss-state flags (QG additions, fed by binary protocol after wizardly merge) ---
    # `is_hittable` defaults to True if missing (legacy sim that doesn't send the field).
    is_hittable = enemy.get("is_hittable")
    feat[34] = 1.0 if (is_hittable is None or bool(is_hittable)) else 0.0
    # `intends_to_attack` defaults to mirroring the inferred attack flag for backward compat.
    intends_attack = enemy.get("intends_to_attack")
    if intends_attack is None:
        feat[35] = feat[3]  # mirror inferred attack flag
    else:
        feat[35] = 1.0 if bool(intends_attack) else 0.0
    # `next_move_id` cheap deterministic hash so different move ids produce different
    # feature values without exploding the dim. md5 first 4 hex chars → 0..65535 → /65536.
    move_id = enemy.get("next_move_id") or ""
    if move_id:
        import hashlib
        feat[36] = (int(hashlib.md5(str(move_id).encode("utf-8")).hexdigest()[:4], 16) % 65536) / 65536.0
    else:
        feat[36] = 0.0

    # --- intent stats (P0 additions) ---
    feat[37] = min(num_intents / 4.0, 1.0)
    feat[38] = min(total_intent_dmg / 30.0, 2.0)
    # feat[39] reserved

    return feat


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
