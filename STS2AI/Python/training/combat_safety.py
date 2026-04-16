from __future__ import annotations

from typing import Any

import numpy as np


BLOCK_CARD_VALUES: dict[str, int] = {
    "DEFEND_IRONCLAD": 5,
    "SHRUG_IT_OFF": 8,
    "IRON_WAVE": 5,
    "BLOOD_WALL": 16,
    "FLAME_BARRIER": 12,
    "POWER_THROUGH": 15,
    "GHOSTLY_ARMOR": 10,
    "TRUE_GRIT": 7,
    "SENTINEL": 5,
}

DAMAGE_CARD_VALUES: dict[str, int] = {
    "STRIKE_IRONCLAD": 6,
    "BASH": 8,
    "ANGER": 6,
    "HEADBUTT": 9,
    "CLOTHESLINE": 12,
    "IRON_WAVE": 5,
    "BODY_SLAM": 0,
    "PUMMEL": 8,
    "CARNAGE": 20,
    "GRAPPLE": 8,
}

SETUP_CARD_IDS = {
    "FEEL_NO_PAIN",
    "INFLAME",
    "DEMON_FORM",
    "EVOLVE",
    "BARRICADE",
    "BRUTALITY",
    "DARK_EMBRACE",
    "CORRUPTION",
}

SELF_DAMAGE_CARD_IDS = {
    "BLOODLETTING",
    "OFFERING",
    "HEMOKINESIS",
}


# Self-damage per card in HP (for the R1 "would kill me this turn" mask).
# Only include cards that deal damage to self ON PLAY — not power-cards like
# COMBUST that self-damage on future turns, and not conditional cards like
# BLOOD_FOR_BLOOD whose cost reduces from being hurt (no self-damage on play).
SELF_DAMAGE_AMOUNT: dict[str, int] = {
    "BLOODLETTING": 3,
    "OFFERING": 6,
    "HEMOKINESIS": 2,
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_card_id(value: Any) -> str:
    token = str(value or "").strip().upper().replace(".TITLE", "")
    return token.replace(" ", "_")


def _battle(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle")
    return battle if isinstance(battle, dict) else {}


def _player(state: dict[str, Any]) -> dict[str, Any]:
    battle_player = _battle(state).get("player")
    if isinstance(battle_player, dict):
        return battle_player
    player = state.get("player")
    return player if isinstance(player, dict) else {}


def _alive_enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    enemies = _battle(state).get("enemies") or []
    return [
        enemy for enemy in enemies
        if isinstance(enemy, dict)
        and bool(enemy.get("is_alive", True))
        and _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0) > 0
    ]


def _enemy_attack_damage(enemy: dict[str, Any]) -> int:
    total = 0
    for intent in enemy.get("intents") or []:
        if _lower(intent.get("type")) == "attack":
            total += _safe_int(intent.get("total_damage", intent.get("damage", 0)), 0)
    return total


def _incoming_damage(state: dict[str, Any]) -> int:
    return sum(_enemy_attack_damage(enemy) for enemy in _alive_enemies(state))


def _card_for_action(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    hand = _player(state).get("hand") or []
    card_index = _safe_int(action.get("card_index"), -1)
    if 0 <= card_index < len(hand) and isinstance(hand[card_index], dict):
        return hand[card_index]
    return {}


def _card_id_for_action(state: dict[str, Any], action: dict[str, Any]) -> str:
    card = _card_for_action(state, action)
    raw = card.get("id") or action.get("card_id") or action.get("label") or ""
    return _normalize_card_id(raw)


def _card_type_for_action(state: dict[str, Any], action: dict[str, Any]) -> str:
    card = _card_for_action(state, action)
    return _lower(card.get("type"))


def _label_for_action(action: dict[str, Any]) -> str:
    return _lower(action.get("label"))


def _is_block_action(state: dict[str, Any], action: dict[str, Any]) -> bool:
    if _lower(action.get("action")) != "play_card":
        return False
    card_id = _card_id_for_action(state, action)
    label = _label_for_action(action)
    card_type = _card_type_for_action(state, action)
    return (
        card_id in BLOCK_CARD_VALUES
        or card_id.startswith("DEFEND")
        or "defend" in label
        or "block" in label
        or "armor" in label
        or "shrug" in label
        or card_type == "skill" and card_id in {"TRUE_GRIT", "GHOSTLY_ARMOR"}
    )


def _estimate_block_for_action(state: dict[str, Any], action: dict[str, Any]) -> int:
    if not _is_block_action(state, action):
        return 0
    card_id = _card_id_for_action(state, action)
    if card_id in BLOCK_CARD_VALUES:
        return BLOCK_CARD_VALUES[card_id]
    if card_id.startswith("DEFEND"):
        return 5
    return 5


def _estimate_damage_for_action(state: dict[str, Any], action: dict[str, Any]) -> int:
    """Estimate damage of a play_card action, including strength/vulnerable/weak modifiers.

    Source priority:
      1. BODY_SLAM → current player block (special card)
      2. card.damage / base_damage / attack_damage exposed by sim (usually absent in STS2)
      3. card_base_stats.total_damage_vs_target → base * hits * modifiers
         (the main working path — sim doesn't expose damage so we rebuild it)
      4. hardcoded DAMAGE_CARD_VALUES (legacy)
      5. type/label heuristic → 6
    """
    if _lower(action.get("action")) != "play_card":
        return 0
    card_id = _card_id_for_action(state, action)
    if card_id == "BODY_SLAM":
        return _safe_int(_player(state).get("block"), 0)

    # Path 2: sim-exposed damage field (rare in STS2, but respect if present)
    card = _card_for_action(state, action)
    if isinstance(card, dict):
        raw_dmg = card.get("damage")
        if raw_dmg is None:
            raw_dmg = card.get("base_damage")
        if raw_dmg is None:
            raw_dmg = card.get("attack_damage")
        if raw_dmg is not None:
            dmg = _safe_int(raw_dmg, 0)
            hits = max(1, _safe_int(card.get("hits", card.get("repeats", 1)), 1))
            if hits > 1 and (_lower(card.get("description") or "").find("time") >= 0
                             or card.get("multi_hit")):
                dmg = dmg * hits
            if dmg > 0:
                return dmg

    # Path 3: derive from card_base_stats + live state modifiers. This is the
    # main working path in STS2 since the sim doesn't expose damage fields.
    try:
        from card_base_stats import total_damage_vs_target, base_damage as _base_damage
    except Exception:
        _base_damage = None
        total_damage_vs_target = None
    if total_damage_vs_target is not None and isinstance(card, dict):
        target = _target_enemy(state, action)
        computed = total_damage_vs_target(card, target, _player(state))
        if computed > 0:
            return computed
        # AoE / target-less attacks: use base damage without a target (only strength matters)
        if _base_damage is not None and _base_damage(card) > 0:
            raw = _base_damage(card)
            # apply strength if any
            strength = 0
            for p in (_player(state).get("powers") or []):
                if isinstance(p, dict) and "strength" in _lower(p.get("id") or p.get("power_id") or ""):
                    strength += _safe_int(p.get("amount") or p.get("stacks"), 0)
            return max(0, (raw + strength))

    # Path 4/5: legacy fallback
    if card_id in DAMAGE_CARD_VALUES:
        return DAMAGE_CARD_VALUES[card_id]
    label = _label_for_action(action)
    card_type = _card_type_for_action(state, action)
    if card_type == "attack" or any(
        token in label for token in ("strike", "bash", "slam", "cleave", "pummel", "anger", "headbutt")
    ):
        return 6
    return 0


def _is_setup_action(state: dict[str, Any], action: dict[str, Any]) -> bool:
    if _lower(action.get("action")) != "play_card":
        return False
    card_id = _card_id_for_action(state, action)
    card_type = _card_type_for_action(state, action)
    return card_id in SETUP_CARD_IDS or card_type == "power"


def _is_self_damage_action(state: dict[str, Any], action: dict[str, Any]) -> bool:
    return _card_id_for_action(state, action) in SELF_DAMAGE_CARD_IDS


def _enemy_effective_hp(enemy: dict[str, Any]) -> int:
    hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
    blk = _safe_int(enemy.get("block"), 0)
    return max(0, hp) + max(0, blk)


def _card_cost_for_action(state: dict[str, Any], action: dict[str, Any]) -> int:
    card = _card_for_action(state, action)
    if not isinstance(card, dict):
        return 0
    # cost_for_turn reflects post-discount (e.g. Blood For Blood after damage)
    return max(0, _safe_int(card.get("cost_for_turn", card.get("cost", 0)), 0))


def turn_lethal_projection(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> dict[str, Any]:
    """Can the player kill all alive enemies THIS turn with the cards in hand?

    Greedy damage packing:
      1. Unique play_card actions by card_index (the same card appears N times
         in `legal` if it can target N enemies; still only one "use" of that card).
      2. Sort by (damage descending, cost ascending), pick until cost sum > energy.
      3. Lethal if total picked damage >= sum(enemy hp + block).

    Returns:
      {
        "lethal": bool,
        "total_enemy_eff_hp": int,
        "max_reachable_damage": int,
        "used_energy": int,
        "lethal_card_indices": set[int],
      }

    Callers use this to rerank: when lethal, end_turn / block-only / setup
    actions get a heavy penalty, and damage-dealing play_cards get a bonus.
    """
    enemies = _alive_enemies(state)
    if not enemies:
        return {
            "lethal": False,
            "total_enemy_eff_hp": 0,
            "max_reachable_damage": 0,
            "used_energy": 0,
            "lethal_card_indices": set(),
        }

    total_enemy_hp = sum(_enemy_effective_hp(e) for e in enemies)

    player = _player(state)
    energy = _safe_int(player.get("energy", player.get("max_energy", 0)), 0)

    # Collect unique plays by card_index (one "card" can have multiple target-
    # variants in legal list).
    by_card: dict[int, tuple[int, int]] = {}  # card_index -> (damage, cost)
    for action in legal:
        if _lower(action.get("action")) != "play_card":
            continue
        ci = _safe_int(action.get("card_index"), -1)
        if ci < 0:
            continue
        dmg = _estimate_damage_for_action(state, action)
        if dmg <= 0:
            continue
        cost = _card_cost_for_action(state, action)
        prev = by_card.get(ci)
        # Keep the highest-damage variant for this card (different targets can
        # have different effective damage due to vulnerable / weak / etc).
        if prev is None or dmg > prev[0]:
            by_card[ci] = (dmg, cost)

    if not by_card:
        return {
            "lethal": False,
            "total_enemy_eff_hp": total_enemy_hp,
            "max_reachable_damage": 0,
            "used_energy": 0,
            "lethal_card_indices": set(),
        }

    # Greedy pack
    ordered = sorted(
        ((ci, dmg, cost) for ci, (dmg, cost) in by_card.items()),
        key=lambda row: (-row[1], row[2]),
    )

    total_dmg = 0
    used_energy = 0
    lethal_cards: set[int] = set()
    for ci, dmg, cost in ordered:
        if used_energy + cost > energy:
            continue
        used_energy += cost
        total_dmg += dmg
        lethal_cards.add(ci)
        if total_dmg >= total_enemy_hp:
            return {
                "lethal": True,
                "total_enemy_eff_hp": total_enemy_hp,
                "max_reachable_damage": total_dmg,
                "used_energy": used_energy,
                "lethal_card_indices": lethal_cards,
            }

    return {
        "lethal": False,
        "total_enemy_eff_hp": total_enemy_hp,
        "max_reachable_damage": total_dmg,
        "used_energy": used_energy,
        "lethal_card_indices": lethal_cards,
    }


def _target_enemy(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    target_id = action.get("target_id", action.get("target"))
    if target_id is None:
        return None
    for enemy in _alive_enemies(state):
        combat_id = enemy.get("combat_id", enemy.get("entity_id"))
        if combat_id == target_id:
            return enemy
    return None


def score_combat_action_safety(state: dict[str, Any], legal: list[dict[str, Any]], action: dict[str, Any]) -> float:
    action_name = _lower(action.get("action"))
    play_actions = [candidate for candidate in legal if _lower(candidate.get("action")) == "play_card"]

    # Turn-lethal detection: if we can kill every alive enemy this turn with
    # the cards/energy in hand, the correct play is to go all-in on damage.
    # end_turn / block / setup should be heavily discouraged in that state.
    # This is the fix for the "boss left at 1-4 HP, AI played DEFEND" failure
    # seen in EVAL_005 / EVAL_001 / EVAL_017 close-loss boss fights.
    lethal = turn_lethal_projection(state, legal)
    lethal_turn = bool(lethal.get("lethal"))
    lethal_card_indices: set[int] = lethal.get("lethal_card_indices") or set()

    if action_name == "end_turn":
        if lethal_turn:
            # Much stronger than the old -1.5: NN logits can be very
            # confident about end_turn in late-game boss states, a small
            # nudge doesn't flip the argmax.
            return -6.0
        return -1.5 if play_actions else 0.0
    if action_name != "play_card":
        return 0.0

    player = _player(state)
    hp = _safe_int(player.get("current_hp", player.get("hp", 0)), 0)
    max_hp = max(1, _safe_int(player.get("max_hp", 1), 1))
    block = _safe_int(player.get("block"), 0)
    incoming = _incoming_damage(state)
    net_incoming = max(0, incoming - block)
    has_block_option = any(_estimate_block_for_action(state, candidate) > 0 for candidate in play_actions)

    enemies = _alive_enemies(state)
    attacking_enemies = [enemy for enemy in enemies if _enemy_attack_damage(enemy) > 0]
    lowest_hp = min((_safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0) for enemy in enemies), default=0)

    severe_danger = net_incoming >= max(10, int(max_hp * 0.2)) or hp <= max(18, int(max_hp * 0.35))
    moderate_danger = net_incoming >= 6

    block_gain = _estimate_block_for_action(state, action)
    damage = _estimate_damage_for_action(state, action)
    target = _target_enemy(state, action)
    score = 0.0

    target_is_attacker = False
    target_kill_ends_pressure = False
    if target is not None:
        target_hp = _safe_int(target.get("hp", target.get("current_hp", 0)), 0)
        target_is_attacker = _enemy_attack_damage(target) > 0
        kills_target = damage > 0 and target_hp > 0 and damage >= target_hp
        if target_hp == lowest_hp:
            score += 0.45
        if target_is_attacker:
            score += 0.35
        if kills_target:
            score += 1.25
            if target_is_attacker:
                score += 1.0
            target_kill_ends_pressure = len(attacking_enemies) <= 1 or len(enemies) <= 1
            if target_kill_ends_pressure:
                score += 1.5
        elif target_hp > lowest_hp:
            score -= 0.75

    if severe_danger and has_block_option:
        if block_gain > 0:
            score += 2.0 + 0.12 * min(block_gain, net_incoming)
            if block_gain >= net_incoming > 0:
                score += 1.0
        elif not target_kill_ends_pressure:
            score -= 1.4
    elif moderate_danger and has_block_option:
        if block_gain > 0:
            score += 0.6 + 0.08 * min(block_gain, net_incoming)
        elif not target_kill_ends_pressure:
            score -= 0.35

    if _is_setup_action(state, action):
        if severe_danger and has_block_option and not target_kill_ends_pressure:
            score -= 1.4
        elif moderate_danger and has_block_option and not target_kill_ends_pressure:
            score -= 0.5

    if _is_self_damage_action(state, action) and severe_danger:
        score -= 2.2

    # --- Turn-lethal rerank overrides ---
    # When we can kill all enemies this turn, flip the scoring toward damage:
    #   * if this action is in the lethal pack → big reward
    #   * if this action is block-only / setup / self-damage → big penalty
    # "play_card with damage>0" that's NOT in the greedy-best lethal pack is
    # also OK (maybe another packing works) so we only lightly bias those.
    if lethal_turn and action_name == "play_card":
        card_index = _safe_int(action.get("card_index"), -1)
        in_lethal_pack = card_index in lethal_card_indices
        has_damage = damage > 0
        is_block_only = block_gain > 0 and damage == 0
        is_setup = _is_setup_action(state, action)
        is_self_dmg = _is_self_damage_action(state, action)

        if in_lethal_pack and has_damage:
            # Dominant bonus: make sure this beats DEFEND/INFLAME even when
            # NN's logits strongly prefer those (e.g. logit gap of ~5 seen in
            # EVAL_005 where AI picked DEFEND over STRIKE on lethal turn).
            score += 6.0
        elif has_damage:
            # Off-pack damage card is still useful (alternative packing) but
            # less certain than greedy-picked ones.
            score += 1.5
        if is_block_only:
            # Block on a lethal turn is pure waste — incoming damage won't
            # land if all enemies are dead.
            score -= 4.0
        if is_setup:
            # Power / setup cards don't contribute damage this turn, so on
            # a lethal-is-available turn they're also wasted energy.
            score -= 3.0
        if is_self_dmg:
            # Self-damage / self-exhaust cards for draw/energy don't help us
            # close the game out faster than the already-lethal pack.
            score -= 3.0

    return score


def rerank_combat_logits_with_safety(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    base_logits: np.ndarray,
    *,
    weight: float = 1.0,
) -> tuple[np.ndarray, list[float]]:
    adjustments = [score_combat_action_safety(state, legal, action) * weight for action in legal]
    reranked = np.asarray(base_logits, dtype=np.float32).copy()
    reranked[: len(adjustments)] += np.asarray(adjustments, dtype=np.float32)
    return reranked, adjustments


def compute_combat_unsafe_mask(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> np.ndarray:
    """Return a float mask with 1.0 = safe action, 0.0 = obvious mistake.

    Callers multiply ``(1.0 - mask) * -1e9`` into the combat logits before any
    rerank / argmax, collapsing unsafe actions to ``-inf`` probability. The
    goal here is **zero false positives**: a rule that might be wrong in any
    realistic game state belongs in ``score_combat_action_safety`` (soft
    rerank), NOT here. If in doubt, let the policy decide.

    Currently covered:

    * **R1 self-kill self-damage.** Mask any self-damage card whose damage
      is strictly >= current HP. Playing it this turn drops HP to 0 regardless
      of what else is in hand, and no card draw / energy gain matters when
      you're dead. The softer case "survives this play but incoming damage
      next turn would kill" is *not* masked here — extra energy / draw may
      let you dig into a block; ``score_combat_action_safety`` already
      penalises it with -2.2 in severe_danger.

    * **R2 Phase 2 / invulnerable boss attack lock.** Detection mirrors
      ``rl_encoder_v2`` feat[39] ``boss_critical_state`` so the mask and the
      NN input feature stay in sync:

          max_hp > 10000         (WaterfallGiant TriggerAboutToBlowState
                                  sets max_hp = 999999999 as sentinel;
                                  no real boss is > 500 HP)
          is_hittable == False   (targetable block)
          intent contains "deathblow"

      When any alive enemy matches, **all** attack-type card plays are
      masked. Attacks do 0 useful damage in these phases; use the turn for
      block / power / draw instead. end_turn is never masked (stays safe).

    Non-``play_card`` actions (end_turn, use_potion, targeting, etc.) pass
    through as safe — their safety is handled by other layers.
    """
    mask = np.ones(len(legal), dtype=np.float32)
    player = _player(state)
    hp = _safe_int(player.get("current_hp", player.get("hp", 0)), 0)

    # R2 precheck: is any alive enemy in a Phase-2 / invulnerable regime?
    phase2_marker = False
    for enemy in _alive_enemies(state):
        if _safe_int(enemy.get("max_hp", 0), 0) > 10000:
            phase2_marker = True
            break
        is_hittable = enemy.get("is_hittable")
        if is_hittable is not None and not bool(is_hittable):
            phase2_marker = True
            break
        for intent in enemy.get("intents") or []:
            if "deathblow" in _lower(intent.get("type")):
                phase2_marker = True
                break
        if phase2_marker:
            break

    for idx, action in enumerate(legal):
        if _lower(action.get("action")) != "play_card":
            continue
        card_id = _card_id_for_action(state, action)

        # R1: would this self-damage kill me on play?
        sd = SELF_DAMAGE_AMOUNT.get(card_id, 0)
        if sd > 0 and hp > 0 and hp <= sd:
            mask[idx] = 0.0
            continue

        # R2: Phase 2 / invulnerable — attacks are wasted
        if phase2_marker and _card_type_for_action(state, action) == "attack":
            mask[idx] = 0.0
            continue

    return mask


def choose_heuristic_combat_action(legal: list[dict[str, Any]], state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if not legal:
        raise ValueError("legal actions must not be empty")

    best_index = 0
    best_score = float("-inf")
    for index, action in enumerate(legal):
        action_name = _lower(action.get("action"))
        score = score_combat_action_safety(state, legal, action)
        if action_name == "end_turn":
            score -= 0.5
        elif action_name == "play_card":
            score += 0.03 * _estimate_damage_for_action(state, action)
            score += 0.02 * _estimate_block_for_action(state, action)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index, legal[best_index]
