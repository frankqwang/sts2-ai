"""卡牌基础属性：手工维护的伤害/格挡/命中数查找表（Ironclad）。"""

from __future__ import annotations

from typing import Any


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _upper(value: Any) -> str:
    return str(value or "").strip().upper().replace(".TITLE", "").replace(" ", "_")


# -----------------------------------------------------------------------
# BASE DAMAGE table — per-hit base damage, unupgraded.
# Multi-hit cards record `hits` separately so the caller can multiply.
# BODY_SLAM returns 0 here because its damage equals current block
# (callers already special-case it).
# -----------------------------------------------------------------------
BASE_DAMAGE: dict[str, int] = {
    # Starter
    "STRIKE_IRONCLAD": 6,
    "BASH": 8,
    # Common attacks
    "ANGER": 6,
    "BODY_SLAM": 0,          # damage = current block (special)
    "CLASH": 14,
    "CLEAVE": 8,             # AoE
    "CLOTHESLINE": 12,
    "FLEX": 0,
    "HAVOC": 0,              # plays top of draw pile; no direct damage
    "HEADBUTT": 9,
    "IRON_WAVE": 5,
    "PERFECTED_STRIKE": 6,   # +2 per "strike" card in deck; base 6 here
    "POMMEL_STRIKE": 9,
    "SHRUG_IT_OFF": 0,
    "SWORD_BOOMERANG": 3,    # 3 hits random
    "THUNDERCLAP": 4,        # AoE
    "TRUE_GRIT": 0,
    "TWIN_STRIKE": 5,        # 2 hits
    "WARCRY": 0,
    "WILD_STRIKE": 12,
    # Uncommon
    "BLOOD_FOR_BLOOD": 18,   # cost reduces by 1 each time you take damage
    "CARNAGE": 20,            # ethereal
    "COMBUST": 5,             # power — 5 damage AoE each turn, -1 HP self
    "DARK_EMBRACE": 0,
    "DISARM": 0,              # -2 enemy strength
    "DROPKICK": 5,            # +5 if vulnerable
    "DUAL_WIELD": 0,
    "ENTRENCH": 0,            # doubles block
    "EVOLVE": 0,
    "FEEL_NO_PAIN": 0,
    "FIRE_BREATHING": 0,
    "FLAME_BARRIER": 0,
    "GHOSTLY_ARMOR": 0,
    "HEMOKINESIS": 15,        # self-damage 2 HP
    "INFERNAL_BLADE": 0,      # skill — adds random attack to hand
    "INFLAME": 0,             # power
    "INTIMIDATE": 0,          # weak to all enemies
    "METALLICIZE": 0,         # power
    "POWER_THROUGH": 0,
    "PUMMEL": 2,              # 4 hits
    "RAGE": 0,                # power (this turn): gain 3 block per attack
    "RAMPAGE": 8,             # +5 damage permanently per play
    "REAPER": 4,              # AoE, heal for unblocked damage
    "RUPTURE": 0,
    "SEARING_BLOW": 12,       # scaling with upgrades
    "SECOND_WIND": 0,
    "SEEING_RED": 0,          # exhaust, +2 energy
    "SENTINEL": 0,
    "SEVER_SOUL": 16,
    "SHOCKWAVE": 0,           # weak + vulnerable AoE
    "SPOT_WEAKNESS": 0,       # conditional +3 strength
    "SWORD_BOOMERANG_UPGRADED": 3,  # placeholder
    "UPPERCUT": 13,
    "WHIRLWIND": 5,           # X-cost: 5 damage per enemy per X
    # Rare
    "BARRICADE": 0,           # power
    "BERSERK": 0,             # power
    "BLUDGEON": 32,
    "BRUTALITY": 0,           # power
    "CORRUPTION": 0,          # power
    "DEMON_FORM": 0,
    "DOUBLE_TAP": 0,          # skill
    "EXHUME": 0,
    "FEED": 10,               # +3 max HP on kill
    "FIEND_FIRE": 7,          # 7 per card exhausted, +3 upgraded
    "IMMOLATE": 21,           # AoE
    "IMPERVIOUS": 0,
    "JUGGERNAUT": 0,          # power: 5 damage per block gained
    "LIMIT_BREAK": 0,         # doubles strength
    "OFFERING": 0,            # +2 energy, +3 draw, 6 HP self
    "REAPER_UPGRADED": 5,     # placeholder
    # Statuses / basic skills still appear
    "DEFEND_IRONCLAD": 0,
    "ARMAMENTS": 0,
    "BATTLE_TRANCE": 0,
    "BLOODLETTING": 0,        # 2 energy + 3 cards / self 3 HP
    "BURNING_PACT": 0,        # exhaust + 2 draw
    "DISMANTLE": 0,
    "DOUBLE_TAP_SKILL": 0,
    "EXHUME_CARD": 0,
}

# Multi-hit counts (default 1).
BASE_HITS: dict[str, int] = {
    "TWIN_STRIKE": 2,
    "PUMMEL": 4,
    "INFERNO": 6,
    "FIEND_FIRE": 7,
    "FIEND_FIRE_UPGRADED": 7,
    "SWORD_BOOMERANG": 3,
    "RAMPAGE": 1,
    "THUNDERCLAP": 1,  # AoE but still 1 hit per enemy
    "CLEAVE": 1,
    "IMMOLATE": 1,
    "REAPER": 1,
    "WHIRLWIND": 1,    # X-cost handles multi-cast at cost level
}

# AoE flag: damage applies to all enemies (used for lethal projection).
AOE_ATTACKS: set[str] = {
    "CLEAVE", "THUNDERCLAP", "SHOCKWAVE", "IMMOLATE", "REAPER",
    "WHIRLWIND", "TEMPEST", "COMBUST",
}

# BASE BLOCK table (unupgraded).
BASE_BLOCK: dict[str, int] = {
    "DEFEND_IRONCLAD": 5,
    "SHRUG_IT_OFF": 8,
    "TRUE_GRIT": 7,
    "POWER_THROUGH": 15,
    "FLAME_BARRIER": 12,
    "GHOSTLY_ARMOR": 10,
    "SENTINEL": 5,
    "IRON_WAVE": 5,
    "IMPERVIOUS": 30,
    "SECOND_WIND": 0,      # archetype: N block per card discarded
    "BARRICADE": 0,        # power
    "BLOOD_WALL": 0,       # mod card; unknown
}

# Cards that apply vulnerable/weak/etc on hit — for future bonus scoring.
APPLIES_VULNERABLE: set[str] = {"BASH", "THUNDERCLAP", "UPPERCUT", "SHOCKWAVE"}
APPLIES_WEAK: set[str] = {"CLOTHESLINE", "INTIMIDATE", "UPPERCUT", "SHOCKWAVE"}

# Self-damage amounts (HP cost on play), mirrors combat_safety.SELF_DAMAGE_AMOUNT
SELF_DAMAGE: dict[str, int] = {
    "BLOODLETTING": 3,
    "OFFERING": 6,
    "HEMOKINESIS": 2,
    "COMBUST": 1,     # per turn (power)
    "BRUTALITY": 1,   # per turn (power)
}


# -----------------------------------------------------------------------
# Helpers: lookup + state modifiers
# -----------------------------------------------------------------------

def _normalize(card_or_id: Any) -> str:
    if isinstance(card_or_id, dict):
        return _upper(card_or_id.get("id") or card_or_id.get("name") or card_or_id.get("label"))
    return _upper(card_or_id)


def base_damage(card_or_id: Any) -> int:
    """Base damage per hit (unupgraded). Returns 0 if unknown."""
    return int(BASE_DAMAGE.get(_normalize(card_or_id), 0))


def base_hits(card_or_id: Any) -> int:
    """Number of damage-hit multi-cast (default 1)."""
    return int(BASE_HITS.get(_normalize(card_or_id), 1))


def is_aoe(card_or_id: Any) -> bool:
    return _normalize(card_or_id) in AOE_ATTACKS


def base_block(card_or_id: Any) -> int:
    return int(BASE_BLOCK.get(_normalize(card_or_id), 0))


def self_damage(card_or_id: Any) -> int:
    return int(SELF_DAMAGE.get(_normalize(card_or_id), 0))


def _get_power_amount(powers: list, power_ids: tuple[str, ...]) -> int:
    """Sum stacks for any power whose id contains any of `power_ids` (case-insensitive substring)."""
    total = 0
    for p in powers or []:
        if not isinstance(p, dict):
            continue
        pid = _lower(p.get("id") or p.get("power_id") or p.get("name", ""))
        if any(sub in pid for sub in power_ids):
            try:
                total += int(p.get("amount") or p.get("stacks") or 0)
            except (TypeError, ValueError):
                pass
    return total


def effective_damage_vs_target(
    card: dict[str, Any] | None,
    target: dict[str, Any] | None,
    player: dict[str, Any] | None,
) -> int:
    """Per-hit effective damage against this target including strength/weak/vulnerable.

    Formula (vanilla STS):
        dmg = base_damage + player_strength
        if player has "weak":    dmg *= 0.75
        if target has "vulnerable": dmg *= 1.5
    Block on target is NOT subtracted here — the caller decides whether to
    compare against hp only or hp+block. Returns rounded integer.
    """
    if not isinstance(card, dict):
        return 0
    base = base_damage(card)
    if base <= 0:
        return 0

    player_powers = (player or {}).get("powers") or []
    strength = _get_power_amount(player_powers, ("strength",))
    has_weak = _get_power_amount(player_powers, ("weak",)) > 0

    target_powers = (target or {}).get("powers") or []
    has_vuln = _get_power_amount(target_powers, ("vulnerable",)) > 0

    dmg = base + strength
    if has_weak:
        dmg = int(dmg * 0.75)
    if has_vuln:
        dmg = int(dmg * 1.5)
    return max(0, dmg)


def total_damage_vs_target(
    card: dict[str, Any] | None,
    target: dict[str, Any] | None,
    player: dict[str, Any] | None,
) -> int:
    """Total damage (per-hit × hits)."""
    per_hit = effective_damage_vs_target(card, target, player)
    if per_hit <= 0:
        return 0
    return per_hit * base_hits(card)
