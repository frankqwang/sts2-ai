"""Relic tag extraction from C# source — semantic classification of every relic.

Scans each relic's .cs file to extract functional tags based on:
- DynamicVar types (BlockVar, CardsVar, EnergyVar, HealVar, GoldVar, PowerVar<T>)
- Trigger methods (BeforeCombatStart, ModifyHandDraw, AfterPlayerTurnStart, ...)
- PowerCmd.Apply<T> calls
- RelicRarity

Usage:
    python relic_tags.py --repo-root /path/to/sts2

    from relic_tags import load_relic_tags, RELIC_FUNCTIONAL_TAGS
    tags = load_relic_tags()
    tags["bag_of_preparation"]  # ["draw", "combat_start", "common"]
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
# Relic tag vocabulary
# ---------------------------------------------------------------------------

RELIC_FUNCTIONAL_TAGS: list[str] = [
    # Effect types
    "damage",           # DamageVar or DamageCmd
    "block",            # BlockVar or GainBlock
    "draw",             # CardsVar or CardPileCmd.Draw or ModifyHandDraw
    "energy",           # EnergyVar or GainEnergy
    "heal",             # HealVar or CreatureCmd.Heal
    "gold",             # GoldVar or GainGold
    "max_hp",           # MaxHpVar or GainMaxHp

    # Buff/debuff application
    "strength",         # PowerVar<StrengthPower> or Apply<StrengthPower>
    "dexterity",        # PowerVar<DexterityPower>
    "vulnerable",       # PowerVar<VulnerablePower>
    "weak",             # PowerVar<WeakPower>
    "poison",           # PoisonPower
    "artifact",         # ArtifactPower

    # Card manipulation
    "upgrade_card",     # CardCmd.Upgrade
    "transform_card",   # CardCmd.Transform
    "generate_card",    # CreateInHand or AddToHand
    "exhaust_related",  # Exhaust references

    # Trigger timing
    "combat_start",     # BeforeCombatStart, AfterCombatStart
    "turn_start",       # BeforeSideTurnStart, AfterPlayerTurnStart
    "turn_end",         # AfterSideTurnEnd
    "on_attack",        # ModifyDamage, AfterAttack, OnAttack
    "on_block",         # ModifyBlock
    "on_card_play",     # AfterCardPlayed
    "on_rest",          # AfterCampfireOption, ModifyRestHeal
    "on_obtain",        # AfterObtained
    "passive",          # ModifyHandDraw, ModifyMaxEnergy, etc. (continuous modifiers)

    # Rarity
    "common",
    "uncommon",
    "rare",
    "boss",
    "shop",
    "event",
]

RELIC_TAG_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(RELIC_FUNCTIONAL_TAGS)}
NUM_RELIC_TAGS = len(RELIC_FUNCTIONAL_TAGS)


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------

def extract_relic_tags(filepath: str | Path) -> list[str]:
    """Parse a relic .cs file and return sorted list of tag strings."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        text = f.read()

    tags: set[str] = set()

    # --- Rarity ---
    m = re.search(r"RelicRarity\.(\w+)", text)
    if m:
        rarity_map = {"Common": "common", "Uncommon": "uncommon", "Rare": "rare",
                       "Boss": "boss", "Shop": "shop", "Event": "event",
                       "Ancient": "rare"}
        r = rarity_map.get(m.group(1))
        if r:
            tags.add(r)

    # --- DynamicVar types ---
    if re.search(r"new\s+(Damage|CalculatedDamage|ExtraDamage)Var\b", text):
        tags.add("damage")
    if re.search(r"new\s+(Block|CalculatedBlock)Var\b", text):
        tags.add("block")
    if re.search(r"new\s+CardsVar\b", text):
        tags.add("draw")
    if re.search(r"new\s+EnergyVar\b", text):
        tags.add("energy")
    if re.search(r"new\s+HealVar\b", text):
        tags.add("heal")
    if re.search(r"new\s+GoldVar\b", text):
        tags.add("gold")
    if re.search(r"new\s+MaxHpVar\b", text):
        tags.add("max_hp")

    # --- PowerVar<T> ---
    for pm in re.finditer(r"new\s+PowerVar<(\w+)>", text):
        power = pm.group(1)
        if "Strength" in power:
            tags.add("strength")
        elif "Dexterity" in power:
            tags.add("dexterity")
        elif "Vulnerable" in power:
            tags.add("vulnerable")
        elif "Weak" in power:
            tags.add("weak")
        elif "Poison" in power:
            tags.add("poison")
        elif "Artifact" in power:
            tags.add("artifact")

    # --- PowerCmd.Apply ---
    for pm in re.finditer(r"PowerCmd\.Apply<(\w+)>", text):
        power = pm.group(1)
        if "Strength" in power:
            tags.add("strength")
        elif "Dexterity" in power:
            tags.add("dexterity")
        elif "Vulnerable" in power:
            tags.add("vulnerable")
        elif "Weak" in power:
            tags.add("weak")
        elif "Poison" in power:
            tags.add("poison")
        elif "Artifact" in power:
            tags.add("artifact")

    # --- Card manipulation ---
    if "CardCmd.Upgrade" in text:
        tags.add("upgrade_card")
    if "CardCmd.Transform" in text:
        tags.add("transform_card")
    if "CreateInHand" in text or "AddToHand" in text:
        tags.add("generate_card")
    if "Exhaust" in text and ("exhaust" in text.lower()):
        tags.add("exhaust_related")

    # --- Commands in method body ---
    if "CardPileCmd.Draw" in text or "ModifyHandDraw" in text:
        tags.add("draw")
    if "GainEnergy" in text or "ModifyMaxEnergy" in text:
        tags.add("energy")
    if "GainBlock" in text:
        tags.add("block")
    if "GainGold" in text:
        tags.add("gold")
    if "Heal" in text and "CreatureCmd" in text:
        tags.add("heal")
    if "GainMaxHp" in text:
        tags.add("max_hp")

    # --- Trigger timing ---
    if re.search(r"(Before|After)CombatStart", text):
        tags.add("combat_start")
    if re.search(r"(Before|After)(Side|Player)TurnStart", text):
        tags.add("turn_start")
    if re.search(r"(Before|After)SideTurnEnd", text):
        tags.add("turn_end")
    if re.search(r"(ModifyDamage|AfterAttack|OnAttack|ModifyAttack)", text):
        tags.add("on_attack")
    if "ModifyBlock" in text:
        tags.add("on_block")
    if re.search(r"AfterCard(Played|Used)", text):
        tags.add("on_card_play")
    if re.search(r"(AfterCampfireOption|ModifyRestHeal|AfterRest)", text):
        tags.add("on_rest")
    if "AfterObtained" in text:
        tags.add("on_obtain")
    if re.search(r"Modify(HandDraw|MaxEnergy|CardCost|Block(?!$))", text):
        tags.add("passive")

    return sorted(tags)


# ---------------------------------------------------------------------------
# Build full mapping
# ---------------------------------------------------------------------------

def build_relic_tags(repo_root: str | Path) -> dict[str, list[str]]:
    """Scan all relic .cs files and return slug → tags mapping."""
    relic_dir = Path(repo_root) / "src" / "Core" / "Models" / "Relics"
    result: dict[str, list[str]] = {}
    for fn in sorted(relic_dir.iterdir()):
        if fn.suffix == ".cs" and not fn.name.endswith(".uid"):
            slug = _slugify(fn.stem).lower()
            result[slug] = extract_relic_tags(fn)
    return result


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

# 2026-04-08 (wizardly cleanup): relic_tags.py moved into tools/python/core/
# but relic_tags.json stays at tools/python/relic_tags.json.
_DEFAULT_PATH = Path(__file__).parent.parent / "relic_tags.json"


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for candidate in [
        here.parent.parent.parent.parent,  # wizardly cleanup: core/ adds a level
        here.parent.parent.parent,
        Path("."),
        here.parent,
    ]:
        if (candidate / "src" / "Core" / "Models" / "Relics").exists():
            return candidate
    return None


def save_relic_tags(tags: dict[str, list[str]], path: str | Path | None = None) -> Path:
    path = Path(path) if path else _DEFAULT_PATH
    data = {"relic_tags": tags, "tag_vocab": RELIC_FUNCTIONAL_TAGS}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def load_relic_tags(path: str | Path | None = None) -> dict[str, list[str]]:
    """Load relic_tags.json. Auto-builds if missing."""
    path = Path(path) if path else _DEFAULT_PATH
    if not path.exists():
        repo_root = _find_repo_root()
        if repo_root is not None:
            tags = build_relic_tags(str(repo_root))
            save_relic_tags(tags, path)
            return tags
        raise FileNotFoundError(f"relic_tags.json not found at {path}")
    data = json.loads(path.read_text())
    return data["relic_tags"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract relic tags from C# source")
    parser.add_argument("--repo-root", type=str, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root
    if repo_root is None:
        here = Path(__file__).resolve()
        for c in [
            here.parent.parent.parent.parent,  # wizardly cleanup
            here.parent.parent.parent,
            Path("."),
        ]:
            if (c / "src" / "Core" / "Models" / "Relics").exists():
                repo_root = str(c)
                break
        if repo_root is None:
            print("ERROR: Cannot find repo root.", file=sys.stderr)
            sys.exit(1)

    tags = build_relic_tags(repo_root)
    out = save_relic_tags(tags)

    print(f"Relic tags saved to {out}")
    print(f"  Relics: {len(tags)}")
    print(f"  Tag vocabulary: {NUM_RELIC_TAGS} tags")

    from collections import Counter
    freq = Counter()
    for tl in tags.values():
        freq.update(tl)
    print(f"\n  Top 15 tags:")
    for tag, count in freq.most_common(15):
        print(f"    {tag:20s} {count:4d} relics ({100*count/len(tags):.0f}%)")

    print(f"\n  Examples:")
    for slug in ["bag_of_preparation", "anchor", "bag_of_marbles",
                  "bellows", "archaic_tooth"]:
        if slug in tags:
            print(f"    {slug:25s} {tags[slug]}")


if __name__ == "__main__":
    main()
