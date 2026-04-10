"""Reward shaping v5 — problem-vector + semi-MDP + screen-aware.

Architecture:
  1. problem_vector: 9-dim deck capability assessment, weighted by act
  2. survival_margin: threshold-based HP utility (not linear)
  3. economy_score: buying-power thresholds (not raw gold)
  4. milestone_reward: floor/act/elite/boss clear bonuses
  5. fight_summary: excess-hp-loss penalty (not raw hp loss)
  6. combat PBRS: unchanged from v4

Key changes from v4:
  - REMOVED: raw gold reward, raw potion reward, deck_size<=12 bonus
  - REPLACED: linear HP → threshold survival_margin
  - REPLACED: single deck_score → problem_vector with act-specific weights
  - ADDED: milestone events (floor advance, act clear, elite clear)
  - ADDED: fight_summary with excess-hp-loss model
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import math
import re
from typing import Any

import numpy as np

GAMMA = 0.999
COMBAT_GAMMA = 0.99

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _lower(v: Any) -> str:
    return str(v).strip().lower() if v else ""

def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _get_enemy_power(enemy: dict[str, Any], power_id: str) -> float:
    powers = enemy.get("status") or enemy.get("powers") or enemy.get("buffs") or []
    if not isinstance(powers, list):
        return 0.0
    for power in powers:
        if not isinstance(power, dict):
            continue
        pid = _lower(power.get("id") or power.get("power_id", ""))
        if power_id in pid:
            return _safe_float(power.get("amount") or power.get("stacks"), 0.0)
    return 0.0

def _player_richness(player: dict[str, Any] | None) -> int:
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


def _extract_player(state: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    top_player = state.get("player")
    if isinstance(top_player, dict):
        candidates.append(top_player)
    for key in ("battle", "map", "shop", "rest_site", "event", "rewards",
                "card_reward", "card_select", "relic_select", "treasure", "menu"):
        container = state.get(key)
        if isinstance(container, dict) and isinstance(container.get("player"), dict):
            candidates.append(container["player"])
    if not candidates:
        return {}
    return max(candidates, key=_player_richness)

def _extract_progress(state: dict[str, Any]) -> dict[str, Any]:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    player = _extract_player(state)
    relics = player.get("relics") if isinstance(player.get("relics"), list) else []
    deck = player.get("deck") if isinstance(player.get("deck"), list) else []
    upgraded = sum(1 for c in deck if isinstance(c, dict) and _safe_int(c.get("upgrades"), 0) > 0)
    basic_count = 0
    for c in deck:
        if isinstance(c, dict):
            cid = _lower(c.get("id", ""))
            if "strike" in cid or "defend" in cid:
                basic_count += 1
    return {
        "act": _safe_int(run.get("act"), 1),
        "floor": _safe_int(run.get("floor"), 0),
        "hp": _safe_int(player.get("hp", player.get("current_hp")), 0),
        "max_hp": max(1, _safe_int(player.get("max_hp"), 1)),
        "gold": _safe_int(player.get("gold"), 0),
        "relic_count": len(relics),
        "deck_size": max(1, len(deck)),
        "upgraded_count": upgraded,
        "basic_count": basic_count,
        "deck": deck,
    }


def _normalize_boss_text(value: Any) -> str:
    text = _lower(value)
    if not text:
        return "unknown"
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


_BOSS_TOKEN_ALIASES: dict[str, str] = {
    # Current STS2 boss ids / localized names
    "ceremonial_beast": "ceremonial_beast",
    "ceremonial_beast_boss": "ceremonial_beast",
    "仪式兽": "ceremonial_beast",
    "vantom": "vantom",
    "vantom_boss": "vantom",
    "墨影幻灵": "vantom",
    "the_kin": "the_kin",
    "the_kin_boss": "the_kin",
    "kin_follower": "the_kin",
    "kin_priest": "the_kin",
    "同族信徒": "the_kin",
    "同族神官": "the_kin",
    "kaiser_crab": "kaiser_crab",
    "kaiser_crab_boss": "kaiser_crab",
    "knowledge_demon": "knowledge_demon",
    "knowledge_demon_boss": "knowledge_demon",
    "the_insatiable": "the_insatiable",
    "the_insatiable_boss": "the_insatiable",
    "soul_fysh": "soul_fysh",
    "soul_fysh_boss": "soul_fysh",
    "lagavulin_matriarch": "lagavulin_matriarch",
    "lagavulin_matriarch_boss": "lagavulin_matriarch",
    "queen": "queen",
    "queen_boss": "queen",
    "test_subject": "test_subject",
    "test_subject_boss": "test_subject",
    "doormaker": "doormaker",
    "doormaker_boss": "doormaker",
    "waterfall_giant": "waterfall_giant",
    "waterfall_giant_boss": "waterfall_giant",
    # Legacy / vanilla aliases used by existing scorer
    "guardian": "guardian",
    "slime": "slime",
    "slime_boss": "slime",
    "hexaghost": "hexa",
    "hexa": "hexa",
    "champ": "champ",
    "collector": "collector",
    "automaton": "automaton",
    "bronze_automaton": "automaton",
    "time_eater": "time_eater",
    "awakened_one": "awakened",
    "awakened": "awakened",
    "donu": "donu_deca",
    "deca": "donu_deca",
    "donu_deca": "donu_deca",
}


def extract_next_boss_token(state: dict[str, Any]) -> str:
    """Return a stable boss token from run state or live battle enemies."""
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    for key in (
        "next_boss_archetype",
        "next_boss_id",
        "boss_archetype",
        "boss_id",
        "next_boss",
        "boss",
        "boss_name",
        "next_boss_name",
    ):
        token = canonicalize_boss_token(run.get(key))
        if token != "unknown":
            return token

    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        token = canonicalize_boss_token(enemy.get("id") or enemy.get("name"))
        if token != "unknown":
            return token
    return "unknown"


def canonicalize_boss_token(value: Any) -> str:
    token = _normalize_boss_text(value)
    if token == "unknown":
        return token
    if token in _BOSS_TOKEN_ALIASES:
        return _BOSS_TOKEN_ALIASES[token]
    # Fuzzy fallback for ids/titles not listed explicitly.
    for keyword, archetype in (
        ("ceremonial_beast", "ceremonial_beast"),
        ("仪式兽", "ceremonial_beast"),
        ("vantom", "vantom"),
        ("墨影幻灵", "vantom"),
        ("the_kin", "the_kin"),
        ("同族", "the_kin"),
        ("kin_", "the_kin"),
        ("kaiser_crab", "kaiser_crab"),
        ("knowledge_demon", "knowledge_demon"),
        ("insatiable", "the_insatiable"),
        ("soul_fysh", "soul_fysh"),
        ("lagavulin", "lagavulin_matriarch"),
        ("queen", "queen"),
        ("test_subject", "test_subject"),
        ("doormaker", "doormaker"),
        ("waterfall_giant", "waterfall_giant"),
        ("guardian", "guardian"),
        ("slime", "slime"),
        ("hexa", "hexa"),
        ("champ", "champ"),
        ("collector", "collector"),
        ("automaton", "automaton"),
        ("time_eater", "time_eater"),
        ("awakened", "awakened"),
        ("donu", "donu_deca"),
        ("deca", "donu_deca"),
    ):
        if keyword in token:
            return archetype
    return token


_BOSS_READINESS_WEIGHTS: dict[str, np.ndarray] = {
    "unknown": np.array([0.14, 0.08, 0.15, 0.13, 0.10, 0.15, 0.13, 0.05, 0.07], dtype=np.float32),
    "ceremonial_beast": np.array([0.06, 0.02, 0.18, 0.16, 0.10, 0.18, 0.14, 0.06, 0.10], dtype=np.float32),
    "vantom": np.array([0.18, 0.04, 0.18, 0.14, 0.10, 0.10, 0.16, 0.05, 0.05], dtype=np.float32),
    "the_kin": np.array([0.18, 0.18, 0.16, 0.12, 0.08, 0.07, 0.12, 0.05, 0.04], dtype=np.float32),
    "kaiser_crab": np.array([0.12, 0.04, 0.18, 0.12, 0.10, 0.20, 0.12, 0.04, 0.08], dtype=np.float32),
    "knowledge_demon": np.array([0.08, 0.04, 0.14, 0.18, 0.10, 0.20, 0.14, 0.04, 0.08], dtype=np.float32),
    "the_insatiable": np.array([0.16, 0.10, 0.14, 0.12, 0.08, 0.14, 0.12, 0.06, 0.08], dtype=np.float32),
    "soul_fysh": np.array([0.12, 0.14, 0.14, 0.14, 0.08, 0.12, 0.12, 0.06, 0.08], dtype=np.float32),
    "lagavulin_matriarch": np.array([0.10, 0.10, 0.16, 0.16, 0.10, 0.18, 0.12, 0.04, 0.04], dtype=np.float32),
    "queen": np.array([0.10, 0.16, 0.14, 0.16, 0.10, 0.12, 0.10, 0.04, 0.08], dtype=np.float32),
    "test_subject": np.array([0.14, 0.08, 0.16, 0.14, 0.10, 0.14, 0.12, 0.04, 0.08], dtype=np.float32),
    "doormaker": np.array([0.16, 0.06, 0.18, 0.14, 0.10, 0.12, 0.12, 0.04, 0.08], dtype=np.float32),
    "waterfall_giant": np.array([0.12, 0.06, 0.18, 0.14, 0.10, 0.16, 0.12, 0.04, 0.08], dtype=np.float32),
}


# Skada boss difficulty data — loaded at startup by training loop
_skada_boss_difficulty: dict[str, Any] | None = None


def load_skada_boss_difficulty(skada_priors) -> None:
    """Load Skada boss wipe rates to scale boss-entry quality bonus.

    Called by training loop at startup if Skada priors are available.
    """
    global _skada_boss_difficulty
    if skada_priors is None or not skada_priors.loaded:
        return
    _skada_boss_difficulty = {}
    for enc in ("ceremonial_beast", "vantom", "the_kin", "kaiser_crab",
                "knowledge_demon", "the_insatiable", "soul_fysh",
                "lagavulin_matriarch", "queen", "test_subject",
                "doormaker", "waterfall_giant"):
        # Try both lowercase and uppercase encounter IDs
        boss = skada_priors.boss(enc) or skada_priors.boss(enc.upper())
        if boss:
            _skada_boss_difficulty[enc] = boss.wipe_rate
    if _skada_boss_difficulty:
        import logging
        logging.getLogger(__name__).info(
            "Skada boss difficulty loaded: %d bosses (wipe rates: %s)",
            len(_skada_boss_difficulty),
            {k: f"{v:.1%}" for k, v in sorted(_skada_boss_difficulty.items(),
                                               key=lambda x: -x[1])[:5]},
        )


def skada_boss_difficulty_scale(boss_token: str) -> float:
    """Return a difficulty multiplier [0.5, 1.5] based on Skada wipe rate.

    Harder bosses (higher wipe_rate) get a larger multiplier so the
    boss_entry_quality_bonus matters more for them.
    Returns 1.0 if Skada data is unavailable.
    """
    if _skada_boss_difficulty is None:
        return 1.0
    wipe = _skada_boss_difficulty.get(boss_token, None)
    if wipe is None:
        return 1.0
    # Map wipe_rate [0, 1] to multiplier [0.5, 1.5]
    # Average wipe_rate ~0.3 → multiplier ~0.8
    return max(0.5, min(1.5, 0.5 + wipe))


def boss_readiness_score(state: dict[str, Any]) -> float:
    """Boss-aware build readiness in [0, 1]."""
    pv = compute_problem_vector(state)
    boss_token = extract_next_boss_token(state)
    weights = _BOSS_READINESS_WEIGHTS.get(boss_token, _BOSS_READINESS_WEIGHTS["unknown"])
    pv_score = float(np.dot(pv, weights))
    hp_score = (survival_margin(state) + 1.0) * 0.5  # [-1,1] -> [0,1]
    return float(max(0.0, min(1.0, pv_score * 0.9 + hp_score * 0.1)))


# ---------------------------------------------------------------------------
# 1. Problem Vector (9-dim deck capability)
# ---------------------------------------------------------------------------

# Use card_tags.json for capability detection (not hardcoded card names).
# card_tags maps card_id → list of functional tags like "damage", "block", "draw", etc.
_CARD_TAGS_CACHE: dict[str, list[str]] | None = None

def _get_card_tags() -> dict[str, list[str]]:
    global _CARD_TAGS_CACHE
    if _CARD_TAGS_CACHE is None:
        try:
            from card_tags import load_card_tags
            _CARD_TAGS_CACHE = load_card_tags()
        except Exception:
            _CARD_TAGS_CACHE = {}
    return _CARD_TAGS_CACHE


def _count_deck_tag(deck: list, tag: str) -> int:
    """Count cards in deck that have a specific functional tag."""
    tags_db = _get_card_tags()
    count = 0
    for c in deck:
        if not isinstance(c, dict):
            continue
        cid = c.get("id", "")
        card_tags = tags_db.get(cid, tags_db.get(cid.lower(), []))
        if tag in card_tags:
            count += 1
    return count


def compute_problem_vector(state: dict[str, Any]) -> np.ndarray:
    """9-dim deck capability vector using card_tags.

    [frontload, aoe, block_consistency, draw, energy,
     scaling, consistency, elite_readiness, boss_answer]

    Each in [0, 1] range. Uses functional tags from card_tags.json.
    """
    p = _extract_progress(state)
    deck = p.get("deck", [])
    ds = max(1, p["deck_size"])

    # Count capabilities using card_tags
    n_damage = _count_deck_tag(deck, "damage")
    n_aoe = _count_deck_tag(deck, "aoe")
    n_block = _count_deck_tag(deck, "block")
    n_draw = _count_deck_tag(deck, "draw")
    n_energy = _count_deck_tag(deck, "energy_gen")
    n_strength = _count_deck_tag(deck, "strength") + _count_deck_tag(deck, "strength_scaling")
    n_poison = _count_deck_tag(deck, "poison")
    n_vuln = _count_deck_tag(deck, "vulnerable")
    n_power = sum(1 for c in deck if isinstance(c, dict) and _lower(c.get("type", "")) == "power")

    frontload = min(1.0, (n_damage + n_vuln * 0.5) / ds)
    aoe = min(1.0, n_aoe / max(1, ds * 0.08))
    block_c = min(1.0, n_block / max(1, ds * 0.12))
    draw = min(1.0, n_draw / max(1, ds * 0.08))
    energy = min(1.0, n_energy / 2.0)
    scaling = min(1.0, (n_strength + n_poison + n_power * 0.5) / max(1, ds * 0.08))
    consistency = max(0.0, 1.0 - p["basic_count"] / ds)
    elite_ready = min(1.0, (frontload + block_c) / 2.0)
    boss_answer = min(1.0, (scaling + draw + block_c) / 3.0)

    return np.array([frontload, aoe, block_c, draw, energy,
                     scaling, consistency, elite_ready, boss_answer], dtype=np.float32)


# Act-specific weights for problem_vector
_ACT_WEIGHTS = {
    1: np.array([0.30, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05]),
    2: np.array([0.10, 0.20, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05]),
    3: np.array([0.05, 0.10, 0.15, 0.20, 0.10, 0.20, 0.10, 0.05, 0.05]),
}


def problem_score(state: dict[str, Any]) -> float:
    """Weighted problem-solving score [0, 1] based on act."""
    p = _extract_progress(state)
    act = min(max(p["act"], 1), 3)
    pv = compute_problem_vector(state)
    w = _ACT_WEIGHTS.get(act, _ACT_WEIGHTS[3])
    return float(np.dot(pv, w))


# ---------------------------------------------------------------------------
# 2. Survival Margin (threshold-based)
# ---------------------------------------------------------------------------

def survival_margin(state: dict[str, Any]) -> float:
    """Threshold-based HP utility [-1, 1].

    Low HP (below target): steep negative
    Above target: diminishing returns
    avoids "always rest to full" behavior.
    """
    p = _extract_progress(state)
    hp_ratio = p["hp"] / p["max_hp"]
    act = min(max(p["act"], 1), 3)
    # Higher targets = must maintain more HP as game gets harder
    target = {1: 0.6, 2: 0.65, 3: 0.7}.get(act, 0.65)
    # Wider temperature = smoother curve, less extreme values
    return math.tanh((hp_ratio - target) / 0.25)


# ---------------------------------------------------------------------------
# 3. Economy Score (buying power, not raw gold)
# ---------------------------------------------------------------------------

def economy_score(state: dict[str, Any]) -> float:
    """Buying power utility [0, 1]. Not raw gold."""
    p = _extract_progress(state)
    gold = p["gold"]
    if gold >= 150:
        return 1.0
    if gold >= 75:
        return 0.7
    if gold >= 50:
        return 0.4
    if gold >= 25:
        return 0.15
    return 0.0


# ---------------------------------------------------------------------------
# 4. Non-combat Potential (PBRS)
# ---------------------------------------------------------------------------

def potential(state: dict[str, Any]) -> float:
    """Potential Phi(s) for PBRS — v5.

    Components:
      - Floor progress:   [0, 1.2]  (unchanged)
      - Problem score:    [0, 0.3]  (replaces old deck_score)
      - Survival margin:  [-0.3, 0.3]  (threshold-based, replaces linear HP)
      - Economy:          [0, 0.1]  (buying power, replaces raw gold)
      - Relic collection: [0, 0.15]
      - Upgrade ratio:    [0, 0.1]
      - Dead draw burden: [-0.1, 0]

    Total range: ~[-0.4, 2.15]
    """
    p = _extract_progress(state)
    phi = 0.0

    # Floor progress (main forward signal)
    total_floor = (p["act"] - 1) * 17 + p["floor"]
    normalized = total_floor / 51.0
    phi += normalized * 0.8 + normalized ** 2 * 0.4

    # Problem-solving score (act-weighted deck quality)
    phi += problem_score(state) * 0.15

    # Survival margin (threshold-based, not linear HP)
    phi += survival_margin(state) * 0.15

    # Economy (buying power) — very small, just a nudge
    phi += economy_score(state) * 0.05

    # Relic collection
    phi += min(p["relic_count"] / 10.0, 1.0) * 0.15

    # Upgrade ratio
    phi += min(p["upgraded_count"] / p["deck_size"], 1.0) * 0.1

    # Dead draw burden (penalize excess basic cards beyond Act 1 allowance)
    act = min(max(p["act"], 1), 3)
    allowance = {1: 8, 2: 5, 3: 3}.get(act, 5)
    excess_basics = max(0, p["basic_count"] - allowance)
    phi -= excess_basics * 0.04

    return phi


# ---------------------------------------------------------------------------
# 5. Milestone Rewards (discrete events)
# ---------------------------------------------------------------------------

# Step 2 (Phase 5 Macro Milestone PPO, 2026-04-07): boss-entry quality
# milestone. Near-win analysis (docs/diagnostics/near_win_analysis_20260407.md)
# showed 5/6 near-win failures were caused by build issues at boss entry:
#   - 4/6 had 0 potions
#   - 1/6 had 1 healing potion (wrong type)
#   - 2/6 had HP < 50%
#   - 1/6 only 1 relic
# Step 1 (combat-side rules) was a null result. Step 2 fixes the upstream
# build quality so the agent ARRIVES at boss with the resources to win.

# Damage potion id substring matching (must match combat_teacher_common.py
# `damage_potion_keywords`).
_DAMAGE_POTION_KEYWORDS = (
    "fire", "explosive", "strength", "dexterity", "energy",
    "attack", "ancient", "blood", "essence_of_steel",
)


def _is_damage_potion(potion: Any) -> bool:
    """Heuristic: returns True if the potion id/name suggests a damage- or
    burst-oriented potion (Fire, Explosive, Strength, Dexterity, Energy, etc).
    Healing potions, swift potions, etc., return False."""
    if not isinstance(potion, dict):
        return False
    label = _lower(potion.get("id") or potion.get("name") or "")
    return any(kw in label for kw in _DAMAGE_POTION_KEYWORDS)


def _count_damage_potions(player: dict[str, Any]) -> int:
    potions = player.get("potions") if isinstance(player.get("potions"), list) else []
    return sum(1 for p in potions if _is_damage_potion(p))


def boss_entry_quality_bonus(player: dict[str, Any]) -> float:
    """Component bonus for the QUALITY of the build entering boss combat.

    Sums 4 sub-components, each capped, total in [0, ~0.95]:
      potion_count    : up to +0.30 (3 potions)
      damage_potion   : up to +0.20 (2 damage potions, half-credit each)
      hp_quality      : up to +0.20 (HP > 50% threshold, scaled to 100%)
      relic_count     : up to +0.30 (each relic above 2)

    The agent only sees this bonus once per boss-entry, so it's a true
    milestone (no double-counting via PBRS).
    """
    if not isinstance(player, dict):
        return 0.0

    # 1) Potion count: rewards keeping potions through Act 1.
    potions = player.get("potions") if isinstance(player.get("potions"), list) else []
    n_potions = min(3, len(potions))
    r_potion = 0.10 * n_potions  # up to +0.30

    # 2) Damage potion count: extra bonus for damage-typed potions.
    n_damage = _count_damage_potions(player)
    r_damage = 0.10 * min(2, n_damage)  # up to +0.20

    # 3) HP quality: linear bonus above the 50% threshold.
    hp = _safe_int(player.get("hp", player.get("current_hp", 0)), 0)
    max_hp = max(1, _safe_int(player.get("max_hp", 1), 0))
    hp_frac = hp / max_hp
    r_hp = 0.40 * max(0.0, hp_frac - 0.5)  # up to +0.20 at 100% HP

    # 4) Relic count above the baseline-2.
    relics = player.get("relics") if isinstance(player.get("relics"), list) else []
    r_relic = 0.10 * max(0, len(relics) - 2)  # up to +0.30 at 5 relics

    return r_potion + r_damage + r_hp + r_relic


def boss_entry_quality_breakdown(player: dict[str, Any]) -> dict[str, float]:
    """Detailed breakdown of the boss-entry quality bonus, for diagnostics."""
    if not isinstance(player, dict):
        return {"total": 0.0}
    potions = player.get("potions") if isinstance(player.get("potions"), list) else []
    relics = player.get("relics") if isinstance(player.get("relics"), list) else []
    n_potions = min(3, len(potions))
    n_damage = min(2, _count_damage_potions(player))
    hp = _safe_int(player.get("hp", player.get("current_hp", 0)), 0)
    max_hp = max(1, _safe_int(player.get("max_hp", 1), 0))
    hp_frac = hp / max_hp
    r_potion = 0.10 * n_potions
    r_damage = 0.10 * n_damage
    r_hp = 0.40 * max(0.0, hp_frac - 0.5)
    r_relic = 0.10 * max(0, len(relics) - 2)
    return {
        "potion_count": float(r_potion),
        "damage_potion": float(r_damage),
        "hp_quality": float(r_hp),
        "relic_count": float(r_relic),
        "total": float(r_potion + r_damage + r_hp + r_relic),
        "n_potions_raw": int(len(potions)),
        "n_damage_potions": int(_count_damage_potions(player)),
        "hp_frac": float(hp_frac),
        "n_relics": int(len(relics)),
    }


def early_damage_potion_use_penalty(action: dict[str, Any], state: dict[str, Any]) -> float:
    """Inverse of boss_entry_quality_bonus: penalize agent for using a damage
    potion before reaching boss zone. Encourages reservation."""
    if not isinstance(action, dict):
        return 0.0
    if _lower(action.get("action")) != "use_potion":
        return 0.0
    pp = _extract_progress(state)
    if pp.get("floor", 0) >= 14 or pp.get("act", 1) > 1:
        return 0.0  # boss zone reached, free to use
    label = _lower(action.get("label") or action.get("potion_id") or "")
    if any(kw in label for kw in _DAMAGE_POTION_KEYWORDS):
        return -0.05  # small per-use penalty
    return 0.0


def milestone_reward(
    prev_state: dict[str, Any],
    next_state: dict[str, Any],
    *,
    boss_entry_quality_weight: float = 1.0,
) -> float:
    """Discrete milestone bonuses for clear progress signals.

    Args:
        boss_entry_quality_weight: scales the boss-entry quality milestone
            (Step 2 Phase 5). Set to 0.0 to disable, 1.0 for full weight.
    """
    pp = _extract_progress(prev_state)
    np_ = _extract_progress(next_state)
    r = 0.0

    # Floor advance
    prev_floor = (pp["act"] - 1) * 17 + pp["floor"]
    next_floor = (np_["act"] - 1) * 17 + np_["floor"]
    if next_floor > prev_floor:
        r += 0.02

    # Boss floor reached (floor 15-16 in act 1) — dense signal for approaching victory
    if np_["floor"] >= 15 and pp["floor"] < 15 and np_["act"] == 1:
        r += 0.15  # significant bonus for reaching boss zone

        # Step 2: boss-entry build quality milestone (one-shot, paired with
        # the floor-15 crossing). Up to +0.95 if the agent enters with full
        # HP, 3+ potions (2 damage), 5 relics. Diagnostic studies show 0/0/0
        # builds correlate strongly with near-win losses.
        # Skada: scale bonus by boss difficulty (harder boss → larger bonus)
        if boss_entry_quality_weight != 0.0:
            player_next = _extract_player(next_state)
            boss_token = extract_next_boss_token(next_state)
            difficulty_scale = skada_boss_difficulty_scale(boss_token)
            r += boss_entry_quality_weight * difficulty_scale * boss_entry_quality_bonus(player_next)

    # Act clear (act number increased)
    if np_["act"] > pp["act"]:
        r += 0.3

    return r


# ---------------------------------------------------------------------------
# 6. Fight Summary (combat feedback for non-combat decisions)
# ---------------------------------------------------------------------------

_EXPECTED_HP_LOSS = {
    "monster": 0.15,
    "elite": 0.30,
    "boss": 0.40,
}

_CLEAR_BONUS = {
    "monster": 0.05,
    "elite": 0.20,
    "boss": 0.50,
}


_DEATH_PENALTY = {
    "monster": -0.3,
    "elite": -0.3,
    "boss": -0.1,  # Boss death is LESS punishing than regular death:
    # dying at boss means the agent made it far, which is good.
    # The real signal comes from _CLEAR_BONUS (0.50 for winning).
}


def fight_summary(
    hp_before: int, hp_after: int, max_hp: int,
    won: bool, room_type: str = "monster",
    boss_hp_fraction_dealt: float = 0.0,
) -> float:
    """Score a combat result for propagating to non-combat decisions.

    Penalizes excess HP loss (not any HP loss).
    Elite/boss clears get explicit bonus.
    Boss fights: partial credit for damage dealt even on loss.
    """
    max_hp = max(1, max_hp)
    room_type = _lower(room_type) if room_type else "monster"
    if room_type not in _CLEAR_BONUS:
        room_type = "monster"

    if not won:
        base_penalty = _DEATH_PENALTY.get(room_type, -0.3)
        # Boss partial credit: dealing 80% of boss HP is much better than 10%
        if room_type == "boss" and boss_hp_fraction_dealt > 0:
            partial_credit = 0.3 * boss_hp_fraction_dealt  # up to +0.3 for nearly killing boss
            return base_penalty + partial_credit
        return base_penalty

    clear_bonus = _CLEAR_BONUS[room_type]
    expected_loss = _EXPECTED_HP_LOSS[room_type]
    actual_loss = max(0, hp_before - hp_after) / max_hp
    excess = max(0.0, actual_loss - expected_loss)

    return clear_bonus - 0.3 * excess


# Backward compat alias
compute_combat_feedback = fight_summary


# ---------------------------------------------------------------------------
# 6b. Screen-Specific State Delta (for counterfactual scoring)
# ---------------------------------------------------------------------------

# Screen-specific mix weights: problem / survival / economy
# Tuned from codex/reward branch values
_SCREEN_MIX = {
    "map":             (0.68, 0.20, 0.12),
    "card_reward":     (0.72, 0.18, 0.10),
    "shop":            (0.54, 0.12, 0.34),
    "campfire":        (0.50, 0.42, 0.08),
    "rest_site":       (0.50, 0.42, 0.08),
    "relic":           (0.60, 0.22, 0.18),
    "relic_select":    (0.60, 0.22, 0.18),
    "event":           (0.54, 0.36, 0.10),
    "treasure":        (0.60, 0.22, 0.18),
    "combat_rewards":  (0.55, 0.25, 0.20),
}

LOCAL_DELTA_SCREENS = {"card_reward", "map", "shop"}


def generic_state_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    screen_type: str = "card_reward",
) -> float:
    """Compute state quality change using screen-specific mix of problem/survival/economy.

    Used by counterfactual scoring to evaluate hypothetical outcomes.
    """
    ps_before = problem_score(before)
    ps_after = problem_score(after)
    sm_before = survival_margin(before)
    sm_after = survival_margin(after)
    ec_before = economy_score(before)
    ec_after = economy_score(after)

    w_p, w_s, w_e = _SCREEN_MIX.get(screen_type, (0.50, 0.30, 0.20))
    delta = (
        w_p * (ps_after - ps_before)
        + w_s * (sm_after - sm_before)
        + w_e * (ec_after - ec_before)
    )
    return delta


def screen_local_delta_reward(
    before: dict[str, Any],
    after: dict[str, Any],
    screen_type: str,
    clip_range: float = 0.05,
) -> float:
    """Small immediate reward from screen-local utility change.

    This is intentionally conservative and only applies on a small set of
    non-combat screens where the action has an immediate deck/economy/survival
    implication. It is meant to complement PBRS, not replace it.
    """
    screen = _lower(screen_type)
    if screen not in LOCAL_DELTA_SCREENS:
        return 0.0
    delta = generic_state_delta(before, after, screen)
    return float(np.clip(delta, -clip_range, clip_range))


# ---------------------------------------------------------------------------
# 7. Shaped Reward (main non-combat per-step)
# ---------------------------------------------------------------------------

def shaped_reward(
    prev_state: dict[str, Any],
    next_state: dict[str, Any],
    raw_terminal_reward: float,
    done: bool,
    *,
    boss_entry_quality_weight: float = 1.0,
    action: dict[str, Any] | None = None,
    early_damage_potion_penalty_weight: float = 1.0,
) -> float:
    """Non-combat per-step reward = PBRS + milestones.

    Args:
        boss_entry_quality_weight: scales the Step 2 boss-entry quality
            milestone bonus. Default 1.0 for opt-in, 0.0 to disable.
        action: optional chosen action dict, used for early-damage-potion
            penalty (only fires when this is supplied).
        early_damage_potion_penalty_weight: scales the inverse penalty.
    """
    if done:
        won = raw_terminal_reward > 0
        return terminal_reward(next_state, won)

    phi_prev = potential(prev_state)
    phi_next = potential(next_state)
    pbrs = GAMMA * phi_next - phi_prev
    ms = milestone_reward(
        prev_state,
        next_state,
        boss_entry_quality_weight=boss_entry_quality_weight,
    )
    extra = 0.0
    if action is not None and early_damage_potion_penalty_weight != 0.0:
        extra += early_damage_potion_penalty_weight * early_damage_potion_use_penalty(action, prev_state)
    return pbrs + ms + extra


def terminal_reward(state: dict[str, Any], won: bool) -> float:
    """Terminal: victory +1.0, death scaled by floor."""
    if won:
        return 1.0
    p = _extract_progress(state)
    total_floor = (p["act"] - 1) * 17 + p["floor"]
    # Death at floor 3 = -0.82, floor 8 = -0.53, floor 12 = -0.29
    floor_bonus = min(total_floor / 17.0, 1.0)
    return -1.0 + floor_bonus


# ---------------------------------------------------------------------------
# 8. Combat reward shaping (unchanged from v4)
# ---------------------------------------------------------------------------

def _extract_combat_info(state: dict[str, Any]) -> dict[str, Any]:
    player = _extract_player(state)
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or []
    hand = battle.get("hand") or player.get("hand") or []
    total_enemy_hp = sum(_safe_int(e.get("hp"), 0) for e in enemies)
    total_enemy_max_hp = sum(max(1, _safe_int(e.get("max_hp"), 1)) for e in enemies)
    return {
        "hp": _safe_int(player.get("hp", player.get("current_hp")), 0),
        "max_hp": max(1, _safe_int(player.get("max_hp"), 1)),
        "block": _safe_int(player.get("block"), 0),
        "energy": _safe_int(player.get("energy"), 0),
        "enemy_hp": total_enemy_hp,
        "enemy_max_hp": total_enemy_max_hp,
        "num_enemies": len(enemies),
        "hand_size": len(hand) if isinstance(hand, list) else 0,
    }


def combat_potential(state: dict[str, Any]) -> float:
    """Combat state potential [0, 1.1]."""
    c = _extract_combat_info(state)
    phi = 0.0
    phi += (c["hp"] / c["max_hp"]) * 0.5
    phi += (1.0 - c["enemy_hp"] / max(1, c["enemy_max_hp"])) * 0.5
    phi += min(c["block"] / 30.0, 1.0) * 0.1
    return phi


def combat_terminal_reward(
    hp_before_combat: int, hp_after_combat: int, max_hp: int, won: bool,
    boss_damage_ratio: float = 0.0,
) -> float:
    """Combat terminal reward.

    Won: 1.0 - (hp_lost/max_hp)*0.8, minimum 0.2.
    Lost: -1.0 + 0.7 * boss_damage_ratio (gives a continuous gradient for
          near-wins so the combat brain can distinguish "barely lost to boss"
          from "got one-shot". boss_damage_ratio is clamped to [0,1] and is
          expected to be 0.0 for non-boss combats.)
    """
    if not won:
        # Partial credit for boss damage dealt: 0% → -1.0, 100% → -0.3
        shaped = max(0.0, min(1.0, float(boss_damage_ratio)))
        return -1.0 + 0.7 * shaped
    max_hp = max(1, max_hp)
    hp_lost = max(0, hp_before_combat - hp_after_combat)
    damage_ratio = hp_lost / max_hp
    return max(0.2, 1.0 - damage_ratio * 0.8)


def combat_step_reward(
    prev_state: dict[str, Any], next_state: dict[str, Any],
    combat_won: bool | None = None, hp_at_combat_start: int | None = None,
    boss_damage_ratio: float = 0.0,
) -> float:
    """Per-step combat reward: PBRS + draw bonus.

    boss_damage_ratio: only meaningful on terminal loss step for boss combats.
    Callers outside of boss fights should leave it at 0.0 (backward-compatible).
    """
    if combat_won is True:
        next_c = _extract_combat_info(next_state)
        hp_before = hp_at_combat_start if hp_at_combat_start is not None else next_c["max_hp"]
        return combat_terminal_reward(hp_before, next_c["hp"], next_c["max_hp"], won=True)
    if combat_won is False:
        # Route through combat_terminal_reward so near-win boss losses get partial credit.
        return combat_terminal_reward(0, 0, 1, won=False, boss_damage_ratio=boss_damage_ratio)

    phi_prev = combat_potential(prev_state)
    phi_next = combat_potential(next_state)
    reward = COMBAT_GAMMA * phi_next - phi_prev

    # Draw bonus
    prev_c = _extract_combat_info(prev_state)
    next_c = _extract_combat_info(next_state)
    energy_spent = prev_c["energy"] - next_c["energy"]
    if energy_spent > 0 and next_c["hand_size"] >= prev_c["hand_size"]:
        reward += 0.02

    return reward


def _combat_hand_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    player = _extract_player(state)
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    hand = battle.get("hand") or player.get("hand") or []
    return [card for card in hand if isinstance(card, dict)]


def _combat_alive_enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    alive: list[dict[str, Any]] = []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        if hp > 0:
            alive.append(enemy)
    return alive


def _card_slug(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    return _lower(card.get("id") or card.get("card_id") or card.get("name"))


def _card_tag_set(card: dict[str, Any] | None) -> set[str]:
    slug = _card_slug(card)
    if not slug:
        return set()
    tags_db = _get_card_tags()
    tags = tags_db.get(slug) or tags_db.get(slug.lower()) or []
    return {str(tag).strip().lower() for tag in tags}


def _resolve_action_card(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    idx = action.get("card_index", action.get("hand_index", action.get("index")))
    cidx = _safe_int(idx, -1)
    hand = _combat_hand_cards(state)
    if 0 <= cidx < len(hand):
        return hand[cidx]
    card_id = _lower(action.get("card_id") or action.get("id"))
    if card_id:
        for card in hand:
            if _card_slug(card) == card_id:
                return card
    return None


def _resolve_action_enemy(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    target = action.get("target")
    if target is None:
        target = action.get("target_id")
    enemies = _combat_alive_enemies(state)
    if target is None:
        return enemies[0] if len(enemies) == 1 else None
    target_str = str(target)
    target_idx = _safe_int(target, -999)
    for enemy in enemies:
        eid = enemy.get("entity_id", enemy.get("combat_id", enemy.get("id")))
        if str(eid) == target_str:
            return enemy
    if 0 <= target_idx < len(enemies):
        return enemies[target_idx]
    return None


def _enemy_has_power(enemy: dict[str, Any] | None, power_id: str) -> bool:
    if not isinstance(enemy, dict):
        return False
    return _get_enemy_power(enemy, power_id) > 0


def _card_estimated_damage(card: dict[str, Any] | None) -> int:
    if not isinstance(card, dict):
        return 0
    for key in ("damage", "base_damage", "display_damage"):
        value = _safe_int(card.get(key), -1)
        if value >= 0:
            return value
    return 0


def _is_body_slam_card(card: dict[str, Any] | None) -> bool:
    slug = _card_slug(card)
    return slug == "body_slam" or "body_slam" in slug


def combat_local_tactical_reward(
    state: dict[str, Any],
    action: dict[str, Any] | None,
    legal_actions: list[dict[str, Any]],
) -> float:
    """Tiny local combat reward for obvious sequencing patterns.

    Intended as a gentle nudge for common tactical mistakes:
    - Prefer applying Vulnerable before plain attacks when a follow-up attack exists.
    - Avoid zero-damage Body Slam when block cards are playable first.
    """
    if not isinstance(action, dict):
        return 0.0
    if _lower(action.get("action")) != "play_card":
        return 0.0

    player = _extract_player(state)
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    energy = _safe_int(battle.get("energy", player.get("energy", 0)), 0)
    player_block = _safe_int(player.get("block"), 0)
    chosen_card = _resolve_action_card(state, action)
    chosen_tags = _card_tag_set(chosen_card)
    if not chosen_card:
        return 0.0

    legal_cards = [
        _resolve_action_card(state, candidate)
        for candidate in legal_actions
        if isinstance(candidate, dict) and _lower(candidate.get("action")) == "play_card"
    ]
    legal_tags = [_card_tag_set(card) for card in legal_cards]
    has_block_option = any("block" in tags for tags in legal_tags)
    has_attack_option = any("damage" in tags for tags in legal_tags)
    has_body_slam_option = any(_is_body_slam_card(card) for card in legal_cards)

    chosen_enemy = _resolve_action_enemy(state, action)
    target_not_vulnerable = chosen_enemy is not None and not _enemy_has_power(chosen_enemy, "vulnerable")
    chosen_damage = _card_estimated_damage(chosen_card)
    target_hp = _safe_int((chosen_enemy or {}).get("hp", (chosen_enemy or {}).get("current_hp", 0)), 0)

    reward = 0.0

    if (
        _is_body_slam_card(chosen_card)
        and player_block <= 0
        and has_block_option
        and energy > 0
    ):
        reward -= 0.05
    elif (
        "block" in chosen_tags
        and player_block <= 0
        and has_body_slam_option
    ):
        reward += 0.02

    has_vuln_setup_option = any(
        ("vulnerable" in tags)
        for tags in legal_tags
    )

    if (
        "vulnerable" in chosen_tags
        and target_not_vulnerable
        and has_attack_option
    ):
        reward += 0.03
    elif (
        "damage" in chosen_tags
        and "vulnerable" not in chosen_tags
        and has_vuln_setup_option
        and target_not_vulnerable
        and chosen_damage < max(1, target_hp)
    ):
        reward -= 0.02

    return float(max(-0.05, min(0.05, reward)))
