"""Card tag extraction from C# source — semantic classification of every card.

Scans each card's .cs file to extract functional tags based on:
- DynamicVar types (DamageVar, BlockVar, PowerVar<T>, CardsVar, EnergyVar, ...)
- CardKeyword (Exhaust, Retain, Innate, Ethereal, ...)
- CardTag (Strike, Defend, Shiv, ...)
- TargetType (Self, AnyEnemy, AllEnemies, ...)
- OnPlay method body (PowerCmd.Apply<T>, CardPileCmd.Draw, DamageCmd.Attack, ...)

Produces a card_tags.json mapping slug → list of tags, plus a TAG_VOCAB
listing all possible tags and their indices.

Usage:
    # Generate card_tags.json from source
    python card_tags.py --repo-root /path/to/sts2

    # In code
    from card_tags import load_card_tags, TAG_VOCAB
    tags = load_card_tags()  # dict[str, list[str]]
    tag_indices = tags["offering"]  # e.g. ["draw", "energy_gen", "exhaust", "hp_loss", "self_target"]
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from vocab import _slugify

# ---------------------------------------------------------------------------
# Tag vocabulary — every possible tag a card can have
# ---------------------------------------------------------------------------

TAG_VOCAB: list[str] = [
    # --- Card type tags ---
    "attack",           # CardType.Attack
    "skill",            # CardType.Skill
    "power",            # CardType.Power
    "status",           # CardType.Status
    "curse",            # CardType.Curse

    # --- Target tags ---
    "self_target",      # TargetType.Self
    "single_target",    # TargetType.AnyEnemy
    "aoe",              # TargetType.AllEnemies
    "random_target",    # TargetType.RandomEnemy

    # --- Core mechanic tags ---
    "damage",           # Has DamageVar or CalculatedDamageVar
    "block",            # Has BlockVar or CalculatedBlockVar
    "multi_hit",        # .WithHitCount or RepeatVar
    "x_cost",           # HasEnergyCostX

    # --- Resource tags ---
    "draw",             # CardsVar or CardPileCmd.Draw
    "energy_gen",       # EnergyVar or PlayerCmd.GainEnergy
    "hp_loss",          # HpLossVar (self-damage)
    "heal",             # HealVar or CreatureCmd.Heal
    "gold_gen",         # GoldVar

    # --- Buff/scaling tags ---
    "strength",         # PowerVar<StrengthPower> or Apply<StrengthPower>
    "dexterity",        # PowerVar<DexterityPower> or Apply<DexterityPower>
    "strength_scaling",  # DemonForm-style: applies StrengthPower via a persistent power
    "block_scaling",    # Footwork-style: applies DexterityPower via a persistent power

    # --- Debuff tags ---
    "vulnerable",       # Apply<VulnerablePower>
    "weak",             # Apply<WeakPower>
    "poison",           # Apply<PoisonPower> or PoisonPerTurn
    "frail",            # Apply<FrailPower>

    # --- Keyword tags ---
    "exhaust",          # CardKeyword.Exhaust
    "ethereal",         # CardKeyword.Ethereal
    "innate",           # CardKeyword.Innate
    "retain",           # CardKeyword.Retain
    "unplayable",       # CardKeyword.Unplayable
    "sly",              # CardKeyword.Sly (stealth-related)

    # --- Card manipulation tags ---
    "discard",          # Forces discard (CardSelectCmd.FromHandForDiscard, CardCmd.Discard)
    "exhaust_other",    # Exhausts other cards
    "generate_card",    # Creates cards in hand (e.g. Shiv, wound)
    "upgrade_card",     # Upgrades cards

    # --- STS2 built-in tags ---
    "strike_tag",       # CardTag.Strike
    "defend_tag",       # CardTag.Defend
    "shiv_tag",         # CardTag.Shiv
    "minion_tag",       # CardTag.Minion

    # --- Advanced combat tags ---
    "summon",           # SummonVar
    "forge",            # ForgeVar (equipment/upgrade mechanic)
    "repeat",           # RepeatVar (multiple executions)
    "max_hp_change",    # MaxHpVar

    # --- Rarity-derived ---
    "basic",            # CardRarity.Basic
    "common",           # CardRarity.Common
    "uncommon",         # CardRarity.Uncommon
    "rare",             # CardRarity.Rare
]

TAG_TO_IDX: dict[str, int] = {tag: i for i, tag in enumerate(TAG_VOCAB)}
NUM_TAGS = len(TAG_VOCAB)

# ---------------------------------------------------------------------------
# Functional tags for NN features (excludes type/rarity already in aux)
# ---------------------------------------------------------------------------

FUNCTIONAL_TAGS: list[str] = [
    # Target
    "self_target", "single_target", "aoe", "random_target",
    # Core mechanics
    "damage", "block", "multi_hit", "x_cost",
    # Resources
    "draw", "energy_gen", "hp_loss", "heal",
    # Buffs / scaling
    "strength", "dexterity", "strength_scaling", "block_scaling",
    # Debuffs
    "vulnerable", "weak", "poison", "frail",
    # Keywords (synergy-relevant)
    "exhaust", "ethereal", "innate", "retain", "sly",
    # Card manipulation
    "discard", "exhaust_other", "generate_card", "upgrade_card",
    # Built-in tags
    "strike_tag", "defend_tag", "shiv_tag",
    # Advanced
    "summon", "forge",
]

FUNCTIONAL_TAG_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(FUNCTIONAL_TAGS)}
NUM_FUNCTIONAL_TAGS = len(FUNCTIONAL_TAGS)  # 32


# ---------------------------------------------------------------------------
# Tag extraction from a single card .cs file
# ---------------------------------------------------------------------------

def extract_tags_from_card(filepath: str | Path) -> list[str]:
    """Parse a card .cs file and return a sorted list of tag strings."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        text = f.read()

    tags: set[str] = set()

    # --- Card type from constructor ---
    m = re.search(r"base\((-?\d+),\s*CardType\.(\w+),\s*CardRarity\.(\w+),\s*TargetType\.(\w+)\)", text)
    if m:
        card_type = m.group(2)
        rarity = m.group(3)
        target = m.group(4)

        type_map = {"Attack": "attack", "Skill": "skill", "Power": "power",
                     "Status": "status", "Curse": "curse"}
        if card_type in type_map:
            tags.add(type_map[card_type])

        rarity_map = {"Basic": "basic", "Common": "common",
                       "Uncommon": "uncommon", "Rare": "rare"}
        if rarity in rarity_map:
            tags.add(rarity_map[rarity])

        target_map = {"Self": "self_target", "AnyEnemy": "single_target",
                       "AllEnemies": "aoe", "RandomEnemy": "random_target"}
        if target in target_map:
            tags.add(target_map[target])

    # --- X-cost ---
    if "HasEnergyCostX" in text and "true" in text.lower():
        tags.add("x_cost")
    if "EnergyCostType.X" in text:
        tags.add("x_cost")

    # --- DynamicVar types ---
    if re.search(r"new\s+DamageVar\b", text) or re.search(r"new\s+CalculatedDamageVar\b", text) or re.search(r"new\s+ExtraDamageVar\b", text) or re.search(r"new\s+OstyDamageVar\b", text):
        tags.add("damage")
    if re.search(r"new\s+BlockVar\b", text) or re.search(r"new\s+CalculatedBlockVar\b", text):
        tags.add("block")
    if re.search(r"new\s+CardsVar\b", text):
        tags.add("draw")
    if re.search(r"new\s+EnergyVar\b", text):
        tags.add("energy_gen")
    if re.search(r"new\s+HpLossVar\b", text):
        tags.add("hp_loss")
    if re.search(r"new\s+HealVar\b", text):
        tags.add("heal")
    if re.search(r"new\s+GoldVar\b", text):
        tags.add("gold_gen")
    if re.search(r"new\s+SummonVar\b", text):
        tags.add("summon")
    if re.search(r"new\s+ForgeVar\b", text):
        tags.add("forge")
    if re.search(r"new\s+RepeatVar\b", text):
        tags.add("repeat")
    if re.search(r"new\s+MaxHpVar\b", text):
        tags.add("max_hp_change")

    # --- PowerVar<T> detection ---
    for pm in re.finditer(r"new\s+PowerVar<(\w+)>", text):
        power_name = pm.group(1)
        if "Strength" in power_name:
            tags.add("strength")
        elif "Dexterity" in power_name:
            tags.add("dexterity")
        elif "Vulnerable" in power_name:
            tags.add("vulnerable")
        elif "Weak" in power_name:
            tags.add("weak")
        elif "Poison" in power_name:
            tags.add("poison")
        elif "Frail" in power_name:
            tags.add("frail")

    # --- OnPlay body: PowerCmd.Apply<T> ---
    for pm in re.finditer(r"PowerCmd\.Apply<(\w+)>", text):
        power_name = pm.group(1)
        if "Strength" in power_name:
            tags.add("strength")
            # If it's via a persistent power (DemonForm-style), mark as scaling
            if "Power" in power_name and power_name != "StrengthPower":
                tags.add("strength_scaling")
        elif "Dexterity" in power_name:
            tags.add("dexterity")
            if "Power" in power_name and power_name != "DexterityPower":
                tags.add("block_scaling")
        elif "Vulnerable" in power_name:
            tags.add("vulnerable")
        elif "Weak" in power_name:
            tags.add("weak")
        elif "Poison" in power_name:
            tags.add("poison")
        elif "Frail" in power_name:
            tags.add("frail")

    # Persistent power cards that grant strength/dexterity over turns
    if "power" in tags:
        if "strength" in tags:
            tags.add("strength_scaling")
        if "dexterity" in tags:
            tags.add("block_scaling")

    # --- Poison from DynamicVar named PoisonPerTurn ---
    if "Poison" in text and ("PoisonPerTurn" in text or "PoisonPower" in text):
        tags.add("poison")

    # --- Multi-hit ---
    if ".WithHitCount" in text or "new RepeatVar" in text:
        tags.add("multi_hit")

    # --- Card keywords ---
    if "CardKeyword.Exhaust" in text:
        tags.add("exhaust")
    if "CardKeyword.Ethereal" in text:
        tags.add("ethereal")
    if "CardKeyword.Innate" in text:
        tags.add("innate")
    if "CardKeyword.Retain" in text:
        tags.add("retain")
    if "CardKeyword.Unplayable" in text:
        tags.add("unplayable")
    if "CardKeyword.Sly" in text:
        tags.add("sly")

    # --- Card tags ---
    if "CardTag.Strike" in text:
        tags.add("strike_tag")
    if "CardTag.Defend" in text:
        tags.add("defend_tag")
    if "CardTag.Shiv" in text:
        tags.add("shiv_tag")
    if "CardTag.Minion" in text:
        tags.add("minion_tag")

    # --- Card manipulation from OnPlay ---
    if "CardPileCmd.Draw" in text or "CardCmd.Draw" in text:
        tags.add("draw")
    if "PlayerCmd.GainEnergy" in text:
        tags.add("energy_gen")
    if "CardSelectCmd.FromHandForDiscard" in text or "CardCmd.Discard" in text:
        tags.add("discard")
    if "CardCmd.Exhaust" in text and "exhaust" not in tags:
        tags.add("exhaust_other")
    if "CreateInHand" in text or "AddToHand" in text or "CardPileCmd.AddToHand" in text:
        tags.add("generate_card")
    if "CardCmd.Upgrade" in text or "UpgradeCard" in text:
        tags.add("upgrade_card")
    if "CreatureCmd.Heal" in text:
        tags.add("heal")

    return sorted(tags)


# ---------------------------------------------------------------------------
# Build full tag mapping for all cards
# ---------------------------------------------------------------------------

def build_card_tags(repo_root: str | Path) -> dict[str, list[str]]:
    """Scan all card .cs files and return slug → tags mapping."""
    card_dir = Path(repo_root) / "src" / "Core" / "Models" / "Cards"
    result: dict[str, list[str]] = {}

    for fn in sorted(card_dir.iterdir()):
        if fn.suffix == ".cs" and not fn.name.endswith(".uid"):
            slug = _slugify(fn.stem).lower()
            tags = extract_tags_from_card(fn)
            result[slug] = tags

    return result


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

# 2026-04-08 (wizardly cleanup): card_tags.py moved into tools/python/core/
# but card_tags.json stays at tools/python/card_tags.json so external tooling
# doesn't need to know about the move.
_DEFAULT_PATH = Path(__file__).parent.parent / "card_tags.json"


def save_card_tags(tags: dict[str, list[str]], path: str | Path | None = None) -> Path:
    path = Path(path) if path else _DEFAULT_PATH
    data = {
        "tag_vocab": TAG_VOCAB,
        "tag_to_idx": TAG_TO_IDX,
        "card_tags": tags,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def _find_repo_root() -> Path | None:
    """Auto-detect repo root by looking for src/Core/Models/Cards.

    After the 2026-04-08 wizardly cleanup, this file lives at
    tools/python/core/card_tags.py — one extra `parent` needed vs the
    pre-merge layout.
    """
    here = Path(__file__).resolve()
    for candidate in [
        here.parent.parent.parent.parent,  # core/ → tools/python/ → tools/ → repo/
        here.parent.parent.parent,         # historical fallback
        Path("."),
        here.parent,
    ]:
        if (candidate / "src" / "Core" / "Models" / "Cards").exists():
            return candidate
    return None


def load_card_tags(path: str | Path | None = None) -> dict[str, list[str]]:
    """Load card_tags.json → dict[slug, list[tag_name]]. Auto-builds if missing."""
    path = Path(path) if path else _DEFAULT_PATH
    if not path.exists():
        repo_root = _find_repo_root()
        if repo_root is not None:
            tags = build_card_tags(str(repo_root))
            save_card_tags(tags, path)
            data = json.loads(path.read_text())
            return data["card_tags"]
        raise FileNotFoundError(
            f"card_tags.json not found at {path} and cannot find repo root. "
            "Run `python card_tags.py --repo-root <path>` to generate it."
        )
    data = json.loads(path.read_text())
    return data["card_tags"]


def load_card_tag_indices(path: str | Path | None = None) -> dict[str, list[int]]:
    """Load card_tags.json → dict[slug, list[tag_idx]]."""
    tags = load_card_tags(path)
    return {slug: [TAG_TO_IDX[t] for t in tag_list if t in TAG_TO_IDX]
            for slug, tag_list in tags.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract card tags from C# source")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root
    if repo_root is None:
        here = Path(__file__).resolve()
        for candidate in [
            here.parent.parent.parent.parent,  # wizardly cleanup: core/ added a level
            here.parent.parent.parent,
            Path("."),
        ]:
            if (candidate / "src" / "Core" / "Models" / "Cards").exists():
                repo_root = str(candidate)
                break
        if repo_root is None:
            print("ERROR: Cannot find repo root.", file=sys.stderr)
            sys.exit(1)

    tags = build_card_tags(repo_root)
    out_path = save_card_tags(tags, args.output)

    # Stats
    print(f"Card tags saved to {out_path}")
    print(f"  Cards: {len(tags)}")
    print(f"  Tag vocabulary: {NUM_TAGS} tags")

    # Tag frequency
    from collections import Counter
    freq = Counter()
    for tag_list in tags.values():
        freq.update(tag_list)

    print(f"\n  Top 20 tags:")
    for tag, count in freq.most_common(20):
        print(f"    {tag:25s} {count:4d} cards ({100*count/len(tags):.0f}%)")

    # Show some examples
    print(f"\n  Examples:")
    for slug in ["offering", "demon_form", "whirlwind", "footwork",
                  "noxious_fumes", "blade_dance", "inflame", "acrobatics"]:
        if slug in tags:
            print(f"    {slug:25s} {tags[slug]}")


if __name__ == "__main__":
    main()
