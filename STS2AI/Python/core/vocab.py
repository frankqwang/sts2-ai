"""Game entity vocabulary for encoder_v2.

Builds and loads ID-to-index mappings for cards, relics, potions, and monsters
by scanning the C# source model definitions. Also extracts static card
properties (cost, type, rarity) for use as auxiliary features alongside learned
embeddings.

Usage:
    # Generate vocab from source (run once or whenever game data changes)
    python vocab.py --repo-root /path/to/sts2

    # In code
    from vocab import load_vocab
    v = load_vocab()
    idx = v.card_to_idx["strike_ironclad"]
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Slugify — replicates C# StringHelper.Slugify (CamelCase → UPPER_SNAKE_CASE)
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"\s+", "_", s.upper())
    s = re.sub(r"[^A-Z0-9_]", "", s)
    return s


# ---------------------------------------------------------------------------
# Card property extraction
# ---------------------------------------------------------------------------

_CARD_TYPE_MAP = {
    "Attack": 0, "Skill": 1, "Power": 2,
    "Status": 3, "Curse": 4, "Quest": 5, "None": 6,
}

_CARD_RARITY_MAP = {
    "Basic": 0, "Common": 1, "Uncommon": 2, "Rare": 3,
    "Ancient": 4, "Event": 5, "Token": 6, "Status": 7,
    "Curse": 8, "None": 9,
}


def _extract_card_props(filepath: str) -> dict[str, Any]:
    """Extract cost, type, rarity from a card .cs file."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        text = f.read()
    m = re.search(r"base\((-?\d+),\s*CardType\.(\w+),\s*CardRarity\.(\w+)", text)
    if m:
        return {
            "cost": int(m.group(1)),
            "type": m.group(2),
            "type_idx": _CARD_TYPE_MAP.get(m.group(2), 6),
            "rarity": m.group(3),
            "rarity_idx": _CARD_RARITY_MAP.get(m.group(3), 9),
        }
    # X-cost cards
    if "EnergyCostType.X" in text:
        return {"cost": -2, "type": "Attack", "type_idx": 0,
                "rarity": "Unknown", "rarity_idx": 9}
    return {"cost": 0, "type": "Unknown", "type_idx": 6,
            "rarity": "Unknown", "rarity_idx": 9}


# ---------------------------------------------------------------------------
# Vocabulary data class
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<unk>"]


@dataclass
class Vocab:
    """Maps entity IDs (lowercase) to indices and stores static properties."""

    # ID → index mappings (lowercase keys)
    card_to_idx: dict[str, int] = field(default_factory=dict)
    relic_to_idx: dict[str, int] = field(default_factory=dict)
    potion_to_idx: dict[str, int] = field(default_factory=dict)
    monster_to_idx: dict[str, int] = field(default_factory=dict)

    # Static card properties (indexed by card index)
    card_props: list[dict[str, Any]] = field(default_factory=list)

    # Enum size helpers
    card_type_count: int = len(_CARD_TYPE_MAP)
    card_rarity_count: int = len(_CARD_RARITY_MAP)

    @property
    def card_vocab_size(self) -> int:
        return len(self.card_to_idx)

    @property
    def relic_vocab_size(self) -> int:
        return len(self.relic_to_idx)

    @property
    def potion_vocab_size(self) -> int:
        return len(self.potion_to_idx)

    @property
    def monster_vocab_size(self) -> int:
        return len(self.monster_to_idx)

    def card_idx(self, card_id: str) -> int:
        """Look up card index, returning <unk>=1 for unknown IDs."""
        return self.card_to_idx.get(card_id.lower(), 1)

    def relic_idx(self, relic_id: str) -> int:
        return self.relic_to_idx.get(relic_id.lower(), 1)

    def potion_idx(self, potion_id: str) -> int:
        return self.potion_to_idx.get(potion_id.lower(), 1)

    def monster_idx(self, monster_id: str) -> int:
        return self.monster_to_idx.get(monster_id.lower(), 1)

    def to_dict(self) -> dict:
        return {
            "card_to_idx": self.card_to_idx,
            "relic_to_idx": self.relic_to_idx,
            "potion_to_idx": self.potion_to_idx,
            "monster_to_idx": self.monster_to_idx,
            "card_props": self.card_props,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Vocab:
        return cls(
            card_to_idx=d["card_to_idx"],
            relic_to_idx=d["relic_to_idx"],
            potion_to_idx=d["potion_to_idx"],
            monster_to_idx=d["monster_to_idx"],
            card_props=d.get("card_props", []),
        )


# ---------------------------------------------------------------------------
# Build vocab from source
# ---------------------------------------------------------------------------

def build_vocab(repo_root: str | Path) -> Vocab:
    """Scan C# model source files to build the full vocabulary."""
    root = Path(repo_root) / "src" / "Core" / "Models"

    def _scan_dir(subdir: str) -> list[str]:
        """Return sorted list of slugified IDs from .cs files."""
        d = root / subdir
        ids = []
        for fn in sorted(d.iterdir()):
            if fn.suffix == ".cs" and not fn.name.endswith(".uid"):
                ids.append(_slugify(fn.stem).lower())
        return ids

    # Cards (with properties)
    card_dir = root / "Cards"
    card_ids: list[str] = []
    card_props: list[dict[str, Any]] = []
    # Special tokens first
    for tok in SPECIAL_TOKENS:
        card_ids.append(tok)
        card_props.append({"cost": 0, "type": "None", "type_idx": 6,
                          "rarity": "None", "rarity_idx": 9})

    for fn in sorted(card_dir.iterdir()):
        if fn.suffix == ".cs" and not fn.name.endswith(".uid"):
            slug = _slugify(fn.stem).lower()
            card_ids.append(slug)
            card_props.append(_extract_card_props(str(fn)))

    card_to_idx = {cid: i for i, cid in enumerate(card_ids)}

    # Relics, potions, monsters (with special tokens)
    def _build_mapping(subdir: str) -> dict[str, int]:
        ids = list(SPECIAL_TOKENS) + _scan_dir(subdir)
        return {eid: i for i, eid in enumerate(ids)}

    relic_to_idx = _build_mapping("Relics")
    potion_to_idx = _build_mapping("Potions")
    monster_to_idx = _build_mapping("Monsters")

    return Vocab(
        card_to_idx=card_to_idx,
        relic_to_idx=relic_to_idx,
        potion_to_idx=potion_to_idx,
        monster_to_idx=monster_to_idx,
        card_props=card_props,
    )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

# 2026-04-08 (wizardly cleanup): vocab.py moved from tools/python/ to
# tools/python/core/, but vocab.json stayed at tools/python/vocab.json
# because that's where operators and older tools have always looked. We
# walk one extra `parent` to compensate.
_DEFAULT_VOCAB_PATH = Path(__file__).parent.parent / "vocab.json"


def save_vocab(vocab: Vocab, path: str | Path | None = None) -> Path:
    path = Path(path) if path else _DEFAULT_VOCAB_PATH
    path.write_text(json.dumps(vocab.to_dict(), indent=2, ensure_ascii=False))
    return path


def _find_repo_root() -> Path | None:
    """Auto-detect repo root by looking for src/Core/Models.

    vocab.py now lives in tools/python/core/. The repo root is 3 parents
    up from the file:  tools/python/core/ -> tools/python/ -> tools/ -> repo/
    """
    here = Path(__file__).resolve()
    for candidate in [
        here.parent.parent.parent.parent,  # tools/python/core/vocab.py → repo/
        here.parent.parent.parent,         # one less (historical fallback)
        Path("."),
        here.parent,
    ]:
        if (candidate / "src" / "Core" / "Models").exists():
            return candidate
    return None


def load_vocab(path: str | Path | None = None) -> Vocab:
    path = Path(path) if path else _DEFAULT_VOCAB_PATH
    if not path.exists():
        # Auto-build from C# source if available
        repo_root = _find_repo_root()
        if repo_root is not None:
            vocab = build_vocab(str(repo_root))
            save_vocab(vocab, path)
            return vocab
        raise FileNotFoundError(
            f"Vocab file not found at {path} and cannot find repo root. "
            "Run `python vocab.py --repo-root <path>` to generate it."
        )
    data = json.loads(path.read_text())
    return Vocab.from_dict(data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build game entity vocabulary")
    parser.add_argument("--repo-root", type=str, default=None,
                        help="Path to sts2 repo root")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: tools/python/vocab.json)")
    args = parser.parse_args()

    # Auto-detect repo root
    repo_root = args.repo_root
    if repo_root is None:
        # Try relative paths
        for candidate in [
            Path(__file__).parent.parent.parent,  # tools/python/../../
            Path("."),
        ]:
            if (candidate / "src" / "Core" / "Models").exists():
                repo_root = str(candidate)
                break
        if repo_root is None:
            print("ERROR: Cannot find repo root. Use --repo-root.", file=sys.stderr)
            sys.exit(1)

    vocab = build_vocab(repo_root)
    out_path = save_vocab(vocab, args.output)

    print(f"Vocab saved to {out_path}")
    print(f"  Cards:    {vocab.card_vocab_size} ({vocab.card_vocab_size - 2} + 2 special)")
    print(f"  Relics:   {vocab.relic_vocab_size} ({vocab.relic_vocab_size - 2} + 2 special)")
    print(f"  Potions:  {vocab.potion_vocab_size} ({vocab.potion_vocab_size - 2} + 2 special)")
    print(f"  Monsters: {vocab.monster_vocab_size} ({vocab.monster_vocab_size - 2} + 2 special)")

    # Sanity check
    assert vocab.card_idx("strike_ironclad") >= 2, "strike_ironclad should be a real card"
    assert vocab.card_idx("nonexistent_card") == 1, "<unk> should be 1"
    assert vocab.card_idx("<pad>") == 0, "<pad> should be 0"
    print("  Sanity checks passed!")


if __name__ == "__main__":
    main()
