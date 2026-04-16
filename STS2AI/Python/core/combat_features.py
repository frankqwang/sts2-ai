"""Combat feature engineering — state and action featurization for combat NN.

Converts raw game state dicts into fixed-size numpy arrays consumed by
CombatPolicyValueNetwork. Separated from the network definition so that
network architecture code (combat_nn.py) stays focused and readable.

Feature schema:
  State features:
    scalars         (18d)  — hp, block, energy, round, pile sizes, 8 core powers
    extra_scalars   (14d)  — v2 extended powers (intangible, barricade, etc.)
    room_type_onehot (3d)  — hallway / elite / boss
    hand_ids/aux/mask       — up to 12 hand cards
    enemy_ids/aux/mask      — up to 5 enemies
    turn_prefix_ids/aux/mask/scalars — last 4 played cards + 32d turn summary
    deck/pile (optional)    — full deck and draw/discard/exhaust piles

  Action features:
    action_type_ids         — play_card / end_turn / use_potion / ...
    target_card_ids         — vocab index of played card
    target_enemy_ids        — vocab index of target enemy
    action_family_ids       — 4-family hierarchy (play / potion / end / selection)
    action_aux (13d)        — remaining energy, damage, block, draw, kill flags, etc.
    action_mask             — legal action mask

  Value composition:
    compose_room_conditioned_value() — room-type-aware aggregation of
    win-prob, hp-loss, potion-cost into a single V(s) in [-1, 1].
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core.vocab import Vocab
from core.rl_encoder_v2 import (
    CARD_AUX_DIM,
    ENEMY_AUX_DIM,
    MAX_ACTIONS,
    MAX_ENEMIES,
    MAX_HAND_SIZE,
    _cached_card_encoding,
    _cached_card_idx,
    _cached_monster_idx,
    _enemy_aux_features,
    _lower,
    _safe_float,
    _safe_int,
    _extract_player,
)


# ---------------------------------------------------------------------------
# Constants — combat feature dimensions
# ---------------------------------------------------------------------------

COMBAT_SCALAR_DIM = 18  # FROZEN: 10 base + 8 player power features (legacy v1 layout)
COMBAT_EXTRA_SCALAR_DIM = 14  # NEW v2 player powers, appended at END of state_input
COMBAT_TOTAL_SCALAR_DIM = COMBAT_SCALAR_DIM + COMBAT_EXTRA_SCALAR_DIM  # 32
COMBAT_TURN_PREFIX_LEN = 4
COMBAT_TURN_PREFIX_SCALAR_DIM = 32
COMBAT_ACTION_AUX_DIM = 13
COMBAT_ROOM_TYPE_DIM = 3

# Action family IDs — hierarchical action structure
COMBAT_ACTION_FAMILY_PLAY_CARD = 0
COMBAT_ACTION_FAMILY_POTION = 1
COMBAT_ACTION_FAMILY_END_TURN = 2
COMBAT_ACTION_FAMILY_SELECTION = 3
NUM_COMBAT_ACTION_FAMILIES = 4


# ---------------------------------------------------------------------------
# Helpers — power extraction, room type, turn prefix
# ---------------------------------------------------------------------------

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


def _combat_turn_prefix_payload(state: dict[str, Any]) -> dict[str, Any]:
    payload = state.get("_combat_turn_prefix")
    if isinstance(payload, dict):
        return payload
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    payload = battle.get("turn_prefix")
    return payload if isinstance(payload, dict) else {}


def _combat_room_type_onehot(state: dict[str, Any]) -> np.ndarray:
    st = _lower(state.get("state_type") or "")
    room_idx = 0  # hallway/monster
    if st == "elite":
        room_idx = 1
    elif st == "boss":
        room_idx = 2
    else:
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        enemies = battle.get("enemies") or state.get("enemies") or []
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            etype = _lower(enemy.get("type") or enemy.get("enemy_type") or "")
            if etype.startswith("boss"):
                room_idx = 2
                break
            if etype.startswith("elite"):
                room_idx = 1
    out = np.zeros(COMBAT_ROOM_TYPE_DIM, dtype=np.float32)
    out[room_idx] = 1.0
    return out


# ---------------------------------------------------------------------------
# Value composition — room-conditioned V(s) aggregation
# ---------------------------------------------------------------------------

def compose_room_conditioned_value(
    base_value: torch.Tensor,
    continuation: torch.Tensor,
    room_type_onehot: torch.Tensor | None,
) -> torch.Tensor:
    """Aggregate scalar combat value from win/cost continuation targets.

    hallway: low-cost win is preferred
    elite: survival and cost are balanced
    boss: survival dominates; HP-loss after victory matters far less
    """
    if room_type_onehot is None:
        room_type_onehot = torch.zeros(
            base_value.shape[0], COMBAT_ROOM_TYPE_DIM,
            dtype=base_value.dtype, device=base_value.device,
        )
        room_type_onehot[:, 0] = 1.0

    win_prob = continuation[:, 0]
    expected_hp_loss = continuation[:, 1]
    expected_potion_cost = continuation[:, 2]

    survival_value = win_prob * 2.0 - 1.0
    hp_cost = torch.tanh(expected_hp_loss / 15.0)
    potion_cost = torch.tanh(expected_potion_cost)

    hallway_mask = room_type_onehot[:, 0]
    elite_mask = room_type_onehot[:, 1]
    boss_mask = room_type_onehot[:, 2]

    hp_weight = hallway_mask * 0.35 + elite_mask * 0.20 + boss_mask * 0.00
    potion_weight = hallway_mask * 0.30 + elite_mask * 0.18 + boss_mask * 0.05

    conditioned_value = survival_value - hp_weight * hp_cost - potion_weight * potion_cost
    conditioned_value = conditioned_value.clamp(-1.0, 1.0)
    # base_value already comes from a Tanh head; keep the final blend linear in
    # range instead of applying a second squashing nonlinearity.
    return (0.5 * base_value + 0.5 * conditioned_value).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# State featurization — build_combat_features()
# ---------------------------------------------------------------------------

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
        "room_type_onehot": _combat_room_type_onehot(state),
        "hand_ids": hand_ids,
        "hand_aux": hand_aux,
        "hand_mask": hand_mask,
        "enemy_ids": enemy_ids,
        "enemy_aux": enemy_aux,
        "enemy_mask": enemy_mask,
    }

    prefix_payload = _combat_turn_prefix_payload(state)
    prefix_cards = prefix_payload.get("recent_cards") if isinstance(prefix_payload.get("recent_cards"), list) else []
    prefix_ids = np.zeros(COMBAT_TURN_PREFIX_LEN, dtype=np.int64)
    prefix_aux = np.zeros((COMBAT_TURN_PREFIX_LEN, CARD_AUX_DIM), dtype=np.float32)
    prefix_mask = np.zeros(COMBAT_TURN_PREFIX_LEN, dtype=bool)
    for i, card in enumerate(prefix_cards[-COMBAT_TURN_PREFIX_LEN:]):
        if not isinstance(card, dict):
            continue
        card_idx, card_aux = _cached_card_encoding(card, vocab)
        prefix_ids[i] = card_idx
        prefix_aux[i] = card_aux
        prefix_mask[i] = True

    prefix_scalars = np.zeros(COMBAT_TURN_PREFIX_SCALAR_DIM, dtype=np.float32)
    prefix_scalars[0] = min(_safe_float(prefix_payload.get("action_count"), 0.0) / 8.0, 1.0)
    prefix_scalars[1] = min(_safe_float(prefix_payload.get("cards_played"), 0.0) / 8.0, 1.0)
    prefix_scalars[2] = min(_safe_float(prefix_payload.get("attack_count"), 0.0) / 6.0, 1.0)
    prefix_scalars[3] = min(_safe_float(prefix_payload.get("skill_count"), 0.0) / 6.0, 1.0)
    prefix_scalars[4] = min(_safe_float(prefix_payload.get("power_count"), 0.0) / 4.0, 1.0)
    prefix_scalars[5] = min(_safe_float(prefix_payload.get("targeted_count"), 0.0) / 6.0, 1.0)
    prefix_scalars[6] = min(_safe_float(prefix_payload.get("non_card_count"), 0.0) / 4.0, 1.0)
    prefix_scalars[7] = min(_safe_float(prefix_payload.get("energy_spent"), 0.0) / 8.0, 1.0)
    prefix_scalars[8] = min(_safe_float(prefix_payload.get("potion_count"), 0.0) / 2.0, 1.0)
    prefix_scalars[9] = min(_safe_float(prefix_payload.get("selection_count"), 0.0) / 4.0, 1.0)
    prefix_scalars[10] = min(_safe_float(prefix_payload.get("damage_est"), 0.0) / 80.0, 1.0)
    prefix_scalars[11] = min(_safe_float(prefix_payload.get("block_est"), 0.0) / 60.0, 1.0)
    prefix_scalars[12] = min(_safe_float(prefix_payload.get("draw_est"), 0.0) / 8.0, 1.0)
    prefix_scalars[13] = min(_safe_float(prefix_payload.get("discard_count"), 0.0) / 5.0, 1.0)
    prefix_scalars[14] = min(_safe_float(prefix_payload.get("exhaust_count"), 0.0) / 5.0, 1.0)
    prefix_scalars[15] = min(_safe_float(prefix_payload.get("create_count"), 0.0) / 5.0, 1.0)
    prefix_scalars[16] = min(_safe_float(prefix_payload.get("last_action_attack"), 0.0), 1.0)
    prefix_scalars[17] = min(_safe_float(prefix_payload.get("last_action_skill"), 0.0), 1.0)
    prefix_scalars[18] = min(_safe_float(prefix_payload.get("last_action_power"), 0.0), 1.0)
    prefix_scalars[19] = min(_safe_float(prefix_payload.get("last_action_non_card"), 0.0), 1.0)
    # Richer chain-state summary: the model should know what kind of turn it
    # is currently executing, not only which cards were recently played.
    cards_played = _safe_float(prefix_payload.get("cards_played"), 0.0)
    attacks = _safe_float(prefix_payload.get("attack_count"), 0.0)
    skills = _safe_float(prefix_payload.get("skill_count"), 0.0)
    powers = _safe_float(prefix_payload.get("power_count"), 0.0)
    damage_est = _safe_float(prefix_payload.get("damage_est"), 0.0)
    block_est = _safe_float(prefix_payload.get("block_est"), 0.0)
    draw_est = _safe_float(prefix_payload.get("draw_est"), 0.0)
    energy_spent = _safe_float(prefix_payload.get("energy_spent"), 0.0)
    potion_count = _safe_float(prefix_payload.get("potion_count"), 0.0)
    targeted_count = _safe_float(prefix_payload.get("targeted_count"), 0.0)
    remaining_energy = max(0.0, _safe_float(battle.get("energy") or player.get("energy")) )
    hand_count = float(np.count_nonzero(hand_mask))
    alive_enemy_hp = 0.0
    for enemy in alive[:MAX_ENEMIES]:
        if not isinstance(enemy, dict):
            continue
        alive_enemy_hp += max(
            0.0,
            _safe_float(enemy.get("hp", enemy.get("current_hp", 0.0)))
            + _safe_float(enemy.get("block", 0.0)),
        )
    offense_score = damage_est + attacks * 4.0 + targeted_count * 2.0
    defense_score = block_est + skills * 2.0
    draw_score = draw_est * 3.0
    setup_score = powers * 4.0 + _safe_float(prefix_payload.get("create_count"), 0.0) * 2.0
    lethal_score = damage_est - alive_enemy_hp
    mode_scores = [offense_score, defense_score, draw_score, setup_score, lethal_score]
    mode_idx = int(np.argmax(mode_scores)) if any(score > 0 for score in mode_scores) else -1
    prefix_scalars[20] = min(damage_est / 120.0, 1.0)
    prefix_scalars[21] = min(block_est / 120.0, 1.0)
    prefix_scalars[22] = min(cards_played / 12.0, 1.0)
    prefix_scalars[23] = min(energy_spent / 10.0, 1.0)
    prefix_scalars[24] = min(draw_est / 12.0, 1.0)
    prefix_scalars[25] = min(1.0, potion_count)
    prefix_scalars[26] = 1.0 if mode_idx == 0 else 0.0  # offense
    prefix_scalars[27] = 1.0 if mode_idx == 1 else 0.0  # defense
    prefix_scalars[28] = 1.0 if mode_idx == 2 else 0.0  # draw
    prefix_scalars[29] = 1.0 if mode_idx == 3 else 0.0  # setup
    prefix_scalars[30] = 1.0 if mode_idx == 4 else 0.0  # lethal
    keeps_playing_possible = 1.0 if remaining_energy > 0 and hand_count > 0 else 0.0
    prefix_scalars[31] = keeps_playing_possible
    features["turn_prefix_ids"] = prefix_ids
    features["turn_prefix_aux"] = prefix_aux
    features["turn_prefix_mask"] = prefix_mask
    features["turn_prefix_scalars"] = prefix_scalars

    # Optional: include full deck for build_plan_z bridge
    deck = player.get("deck") or player.get("cards") or state.get("deck") or []
    if deck:
        from core.rl_encoder_v2 import MAX_DECK_SIZE, CARD_AUX_DIM as NC_CARD_AUX_DIM
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


# ---------------------------------------------------------------------------
# Action featurization — build_combat_action_features()
# ---------------------------------------------------------------------------

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
    action_family_ids = np.zeros(MAX_ACTIONS, dtype=np.int64)
    action_aux = np.zeros((MAX_ACTIONS, COMBAT_ACTION_AUX_DIM), dtype=np.float32)
    action_mask = np.zeros(MAX_ACTIONS, dtype=bool)

    # Pre-extract enemies and hand for target resolution
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = _extract_player(state)
    hand = battle.get("hand") or player.get("hand") or []
    enemies = state.get("enemies") or battle.get("enemies") or []
    alive_enemies = [e for e in enemies if isinstance(e, dict) and e.get("is_alive", True)]
    current_energy = _safe_float(battle.get("energy") or player.get("energy"))
    player_hp = _safe_float(player.get("hp", player.get("current_hp")))
    player_block = _safe_float(player.get("block"))

    def _enemy_intent_damage(enemy: dict[str, Any]) -> float:
        intents = enemy.get("intents") or []
        if isinstance(intents, list) and intents:
            total = 0.0
            for it in intents:
                if not isinstance(it, dict):
                    continue
                dmg = _safe_float(it.get("damage", it.get("total_damage", 0)))
                repeats = max(1.0, _safe_float(it.get("repeats", it.get("hits", 1)), 1.0))
                if "total_damage" in it:
                    total += dmg
                else:
                    total += dmg * repeats
            return total
        dmg = _safe_float(enemy.get("intent_damage", enemy.get("total_damage", 0)))
        hits = max(1.0, _safe_float(enemy.get("intent_hits", enemy.get("repeats", 1)), 1.0))
        return dmg if enemy.get("total_damage") is not None else dmg * hits

    total_enemy_intent_damage = sum(_enemy_intent_damage(enemy) for enemy in alive_enemies)

    def _card_effect_summary(card: dict[str, Any] | None) -> tuple[float, float, float, float, float, float, float]:
        """Per-card summary used to build action_aux features.

        IMPORTANT: the STS2 sim does NOT expose `damage` / `block` / `draw`
        fields on the card dict. Before this fallback was wired, damage/block
        always evaluated to 0, silently killing 5 of 13 action_aux dims
        (damage, block, kills_target, prevented_lethal, intent_pressure_delta).
        We now fall back to `card_base_stats.base_damage` / `base_block` /
        hits keyed on the card id; state modifiers (strength/weak/vulnerable)
        are applied later in the target-aware damage loop.
        """
        if not isinstance(card, dict):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        cost = _safe_float(card.get("cost_for_turn", card.get("cost", card.get("energy_cost", 0))))
        damage = _safe_float(card.get("damage", card.get("base_damage", card.get("attack_damage", 0))))
        block = _safe_float(card.get("block", card.get("base_block", 0)))
        draw = _safe_float(card.get("draw", card.get("cards_to_draw", card.get("draw_amount", 0))))
        magic = _safe_float(card.get("magic_number", card.get("magic", 0)))
        text = _lower(card.get("description") or card.get("raw_description") or card.get("text") or "")
        exhaust_flag = 1.0 if card.get("exhaust") or "exhaust" in text else 0.0
        discard_flag = 1.0 if any(tok in text for tok in ("discard", "put a card from your hand")) else 0.0
        create_flag = 1.0 if any(tok in text for tok in ("add ", "create ", "shuffle")) else 0.0
        if draw <= 0 and "draw" in text:
            draw = max(draw, magic)

        # Fallback damage/block from card_base_stats when sim didn't expose
        # them. This is the main path in STS2: sim only gives id/cost/type,
        # so we reconstruct per-hit damage * hits, and base block, from
        # curated card knowledge.
        if damage <= 0.0 or block <= 0.0:
            try:
                from card_base_stats import base_damage as _bd, base_hits as _bh, base_block as _bb
                cid = (card.get("id") or card.get("name") or card.get("label") or "").upper().replace(" ", "_").replace(".TITLE", "")
                if damage <= 0.0:
                    dv = _bd(cid)
                    if dv > 0:
                        damage = float(dv * max(1, _bh(cid)))
                if block <= 0.0:
                    bv = _bb(cid)
                    if bv > 0:
                        block = float(bv)
            except Exception:
                pass

        return cost, damage, block, draw, discard_flag, exhaust_flag, create_flag

    def _still_has_playable_followup(card_index: int | None, spent_energy: float) -> float:
        remaining_energy = max(0.0, current_energy - spent_energy)
        for h_idx, card in enumerate(hand):
            if card_index is not None and h_idx == card_index:
                continue
            if not isinstance(card, dict):
                continue
            card_cost = _safe_float(card.get("cost_for_turn", card.get("cost", card.get("energy_cost", 0))))
            if card_cost <= remaining_energy + 1e-6:
                return 1.0
        return 0.0

    for i, action in enumerate(actions[:MAX_ACTIONS]):
        action_mask[i] = True
        aname = _lower(action.get("action") or action.get("type", ""))
        action_type_ids[i] = atype_map.get(aname, atype_map["other"])
        if aname == "play_card":
            action_family_ids[i] = COMBAT_ACTION_FAMILY_PLAY_CARD
        elif aname == "use_potion":
            action_family_ids[i] = COMBAT_ACTION_FAMILY_POTION
        elif aname == "end_turn":
            action_family_ids[i] = COMBAT_ACTION_FAMILY_END_TURN
        else:
            action_family_ids[i] = COMBAT_ACTION_FAMILY_SELECTION

        idx = _safe_int(action.get("index") or action.get("card_index") or
                        action.get("hand_index"))

        # Card target
        if aname == "play_card":
            cidx = _safe_int(action.get("card_index") or action.get("hand_index") or
                             action.get("index"))
            played_card = hand[cidx] if 0 <= cidx < len(hand) and isinstance(hand[cidx], dict) else None
            spent_energy, est_damage, block_gain, draw_delta, discard_flag, exhaust_flag, create_flag = _card_effect_summary(played_card)
            if 0 <= cidx < len(hand) and isinstance(hand[cidx], dict):
                target_card_ids[i] = _cached_card_idx(vocab, hand[cidx].get("id"))

            # Enemy target
            target = action.get("target") or action.get("target_id")
            target_enemy: dict[str, Any] | None = None
            if target is not None:
                for e_idx, enemy in enumerate(alive_enemies):
                    eid = enemy.get("entity_id", enemy.get("combat_id", e_idx))
                    if eid == target or e_idx == _safe_int(target):
                        target_enemy_ids[i] = _cached_monster_idx(vocab, enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id", ""))
                        target_enemy = enemy
                        break
            target_hp_eff = _safe_float(target_enemy.get("hp", target_enemy.get("current_hp"))) + _safe_float(target_enemy.get("block")) if isinstance(target_enemy, dict) else 0.0
            kills_target = 1.0 if target_hp_eff > 0 and est_damage >= target_hp_eff - 1e-6 else 0.0
            target_intent_damage = _enemy_intent_damage(target_enemy) if isinstance(target_enemy, dict) else 0.0
            prevented_lethal = 1.0 if (player_hp + player_block) < total_enemy_intent_damage and (block_gain + target_intent_damage * kills_target) >= (total_enemy_intent_damage - (player_hp + player_block)) else 0.0
            intent_pressure_delta = min(1.0, (target_intent_damage * kills_target) / max(1.0, total_enemy_intent_damage))
            action_aux[i, 0] = max(0.0, current_energy - spent_energy) / 5.0
            action_aux[i, 1] = est_damage / 50.0
            action_aux[i, 2] = block_gain / 40.0
            action_aux[i, 3] = draw_delta / 5.0
            action_aux[i, 4] = discard_flag
            action_aux[i, 5] = exhaust_flag
            action_aux[i, 6] = create_flag
            action_aux[i, 7] = 0.0  # consumes_potion
            action_aux[i, 8] = kills_target
            action_aux[i, 9] = prevented_lethal
            action_aux[i, 10] = intent_pressure_delta
            action_aux[i, 12] = _still_has_playable_followup(cidx, spent_energy)
        elif aname == "use_potion":
            action_aux[i, 0] = current_energy / 5.0
            action_aux[i, 7] = 1.0  # consumes_potion
            action_aux[i, 12] = 1.0 if current_energy > 0 else 0.0
        elif aname == "end_turn":
            action_aux[i, 0] = current_energy / 5.0
            action_aux[i, 11] = 1.0  # ends_turn_immediately
        else:
            action_aux[i, 0] = current_energy / 5.0
            action_aux[i, 12] = 1.0 if current_energy > 0 else 0.0

    return {
        "action_type_ids": action_type_ids,
        "target_card_ids": target_card_ids,
        "target_enemy_ids": target_enemy_ids,
        "action_family_ids": action_family_ids,
        "action_aux": action_aux,
        "action_mask": action_mask,
    }
