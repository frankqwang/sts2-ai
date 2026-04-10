"""Symbolic feature extraction from source_knowledge.sqlite.

Reads the checked-in sqlite database (built by tools/python/data/build_source_database.py)
and produces:

  1. A global symbol vocabulary — the union of every C# class / tag / intent name
     appearing in any *_json column across cards / relics / monsters / potions.
     Sorted alphabetically, with reserved indices 0=<pad>, 1=<unk>.

  2. Per-entity padded symbol id tables — for each card/relic/monster/potion in
     the vocab, the (sorted) list of symbol indices it references, padded with 0.

These tables are consumed by SymbolicFeaturesHead (core/symbolic_features_head.py)
to seed a cross-attention pathway that gives the RL policy a zero-shot prior over
rare entities it has barely seen during training.

The database has NO natural-language text — every row is structured JSON arrays
of C# symbol names — so multi-hot / symbol-attention is the right shape. See
docs/HANDOFF_2026-04-09.md §7.2.D and the plan at
C:/Users/Administrator/.claude/plans/async-snacking-tome.md.

Design notes:
- Pure python + sqlite3 + numpy. No torch dependency.
- Deterministic: builder functions sort every list before returning.
- Special vocab tokens (`<pad>`, `<unk>`) get all-zero id rows and all-False masks.
- Missing ids (should be zero in practice — we verified 100% sqlite ↔ vocab overlap)
  also get all-zero/False rows so downstream attention ignores them.
- SHA1 drift check: on build we compute the sqlite file's SHA1 and compare against
  the value in `source_knowledge.manifest.json`. If the manifest lacks a sha1 field
  we write one back (first-run upgrade). If the sha differs we log a warning —
  this is the cheap "did someone regenerate the DB but forget to retrain?" detector.
"""

from __future__ import annotations

# Sys.path bootstrap so `import _path_init` (which lives at tools/python/) is
# findable whether this module is imported as `core.source_knowledge_features`
# or run directly as `python tools/python/core/source_knowledge_features.py`.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # noqa: E402

import _path_init  # noqa: F401,E402  (adds tools/python/{core,ipc,search} to sys.path)

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vocab import Vocab

logger = logging.getLogger(__name__)


# Default on-disk location of the sqlite + manifest.
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_DB_PATH = _THIS_DIR.parent / "data" / "source_knowledge.sqlite"
_DEFAULT_MANIFEST_PATH = _THIS_DIR.parent / "data" / "source_knowledge.manifest.json"


# Which JSON array columns to pull symbols from, per entity table.
# Scalar string columns like rarity / card_type / target_type are NOT included
# here — they are already covered by the existing card_aux / relic_aux feature
# vectors built in card_tags.py / relic_tags.py, and adding them again here
# would double-count.  Symbolic features should be orthogonal to the hand-
# maintained functional tag vectors.
_CARD_SYMBOL_COLS = [
    "powers_json",
    "commands_json",
    "tags_json",
    "card_tags_json",
    "keywords_json",
    "dynamic_vars_json",
]
_RELIC_SYMBOL_COLS = [
    "powers_json",
    "commands_json",
    "dynamic_vars_json",
]
_MONSTER_SYMBOL_COLS = [
    "powers_json",
    "commands_json",
    "intents_json",
]
_POTION_SYMBOL_COLS = [
    "powers_json",
    "commands_json",
]

PAD_SYMBOL_ID = 0
UNK_SYMBOL_ID = 1
_SPECIAL_SYMBOLS = ["<pad>", "<unk>"]


@dataclass
class KnowledgeMeta:
    """Schema + provenance for the built tables."""

    global_symbol_vocab: list[str]
    sqlite_sha1: str
    card_max_len: int
    relic_max_len: int
    monster_max_len: int
    potion_max_len: int
    # counts (for logging + tests)
    card_coverage: float = 0.0
    relic_coverage: float = 0.0
    monster_coverage: float = 0.0
    potion_coverage: float = 0.0

    def to_json(self) -> str:
        """Compact JSON for persistent storage as ASCII bytes."""
        return json.dumps(
            {
                "global_symbol_vocab": self.global_symbol_vocab,
                "sqlite_sha1": self.sqlite_sha1,
                "card_max_len": self.card_max_len,
                "relic_max_len": self.relic_max_len,
                "monster_max_len": self.monster_max_len,
                "potion_max_len": self.potion_max_len,
            },
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_json(cls, text: str) -> KnowledgeMeta:
        d = json.loads(text)
        return cls(
            global_symbol_vocab=list(d["global_symbol_vocab"]),
            sqlite_sha1=d.get("sqlite_sha1", ""),
            card_max_len=int(d.get("card_max_len", 32)),
            relic_max_len=int(d.get("relic_max_len", 16)),
            monster_max_len=int(d.get("monster_max_len", 32)),
            potion_max_len=int(d.get("potion_max_len", 16)),
        )


# ---------------------------------------------------------------------------
# sqlite helpers
# ---------------------------------------------------------------------------

def _sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_sqlite_drift(db_path: Path, manifest_path: Path) -> str:
    """Compute sqlite SHA1, compare against manifest.  Returns the current SHA1.

    On first run (no sha1 field in manifest) we write one back.  On mismatch we
    log a warning and keep going — the caller is responsible for deciding
    whether drift is fatal.
    """
    current = _sha1_of_file(db_path)
    if not manifest_path.exists():
        logger.debug("No manifest at %s; skipping drift check.", manifest_path)
        return current

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read manifest %s: %s", manifest_path, e)
        return current

    stored = manifest.get("sqlite_sha1")
    if stored is None:
        # First-run upgrade: stamp the SHA into the manifest.
        manifest["sqlite_sha1"] = current
        try:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Stamped sqlite_sha1=%s into %s (first-run)", current[:12], manifest_path.name)
        except Exception as e:
            logger.warning("Could not write sqlite_sha1 back to manifest: %s", e)
    elif stored != current:
        logger.warning(
            "source_knowledge.sqlite SHA drift: manifest=%s actual=%s — "
            "the sqlite was regenerated but nothing downstream was updated. "
            "If your symbolic_head checkpoint was built against the old DB, "
            "its persistent buffers may no longer match this build.",
            stored[:12], current[:12],
        )

    return current


def _load_json_column(raw: str | None) -> list[str]:
    """Parse a json array column, returning a list of string symbols.

    Robust to:
      - None (missing column)
      - '' (empty string)
      - invalid JSON (returns [] with a debug log)
      - non-string entries (filtered out)
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as e:
        logger.debug("Failed to parse JSON column (len=%d): %s", len(raw), e)
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x) for x in parsed if x is not None]


def _collect_symbols_per_entity(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    symbol_cols: list[str],
) -> dict[str, set[str]]:
    """Run one SELECT per (table, cols) and bucket symbols by entity id."""
    result: dict[str, set[str]] = {}
    cols_csv = ", ".join([id_col] + symbol_cols)
    for row in conn.execute(f"SELECT {cols_csv} FROM {table}").fetchall():
        entity_id = row[0]
        syms: set[str] = set()
        for v in row[1:]:
            for s in _load_json_column(v):
                syms.add(s)
        result[entity_id] = syms
    return result


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_global_symbol_vocab(
    db_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> tuple[list[str], str]:
    """Return the sorted global symbol vocabulary + sqlite SHA1.

    Vocab layout: [<pad>, <unk>, *sorted unique symbols across all tables/cols].
    The first two entries are reserved; consumers MUST treat index 0 as padding.
    """
    db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    manifest_path = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"source_knowledge sqlite not found: {db_path}")

    sqlite_sha1 = _check_sqlite_drift(db_path, manifest_path)

    all_syms: set[str] = set()
    with sqlite3.connect(str(db_path)) as conn:
        for table, cols in (
            ("cards", _CARD_SYMBOL_COLS),
            ("relics", _RELIC_SYMBOL_COLS),
            ("monsters", _MONSTER_SYMBOL_COLS),
            ("potions", _POTION_SYMBOL_COLS),
        ):
            cols_csv = ", ".join(cols)
            for row in conn.execute(f"SELECT {cols_csv} FROM {table}").fetchall():
                for v in row:
                    for s in _load_json_column(v):
                        all_syms.add(s)

    sorted_syms = sorted(all_syms)
    vocab = list(_SPECIAL_SYMBOLS) + sorted_syms
    return vocab, sqlite_sha1


def _build_entity_id_tables(
    entity_to_idx: dict[str, int],
    entity_to_symbols: dict[str, set[str]],
    global_symbol_vocab: list[str],
    max_len: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pack per-entity symbol sets into padded (V, max_len) int + bool tables.

    Returns (id_table, mask_table, coverage).
    - id_table[v, :] = sorted symbol indices referenced by vocab index v, padded with 0
    - mask_table[v, :] = True where id_table[v, :] is a real symbol (not pad)
    - coverage = fraction of non-special rows with >=1 real symbol
    """
    sym_to_idx = {s: i for i, s in enumerate(global_symbol_vocab)}

    V = len(entity_to_idx)
    id_table = np.zeros((V, max_len), dtype=np.int32)
    mask_table = np.zeros((V, max_len), dtype=bool)

    # Track how many entities actually got at least 1 symbol (for coverage stats).
    non_special_count = 0
    non_special_with_syms = 0
    overflow_count = 0

    for entity_id, row_idx in entity_to_idx.items():
        if entity_id in _SPECIAL_SYMBOLS:
            # <pad>/<unk> rows stay all zero / all False.
            continue
        non_special_count += 1

        syms = entity_to_symbols.get(entity_id, set())
        if not syms:
            # Missing from sqlite (shouldn't happen given 100% overlap verified)
            # or entity has genuinely zero symbols. Leave row all-zero/False.
            continue

        # Map to vocab indices, drop unknowns (shouldn't happen given global
        # vocab = union of all symbols), sort deterministically.
        sym_indices = sorted(
            sym_to_idx[s] for s in syms if s in sym_to_idx
        )

        if len(sym_indices) > max_len:
            overflow_count += 1
            sym_indices = sym_indices[:max_len]

        n = len(sym_indices)
        if n > 0:
            id_table[row_idx, :n] = sym_indices
            mask_table[row_idx, :n] = True
            non_special_with_syms += 1

    coverage = (non_special_with_syms / non_special_count) if non_special_count else 0.0
    if overflow_count:
        logger.warning(
            "%s: %d entities had >%d symbols and were truncated",
            label, overflow_count, max_len,
        )

    return id_table, mask_table, coverage


def build_card_symbol_ids(
    vocab: Vocab,
    db_path: Path | str | None = None,
    global_symbol_vocab: list[str] | None = None,
    max_len: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (id_table, mask_table) of shapes (card_vocab_size, max_len)."""
    db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    if global_symbol_vocab is None:
        global_symbol_vocab, _ = build_global_symbol_vocab(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        per_entity = _collect_symbols_per_entity(conn, "cards", "id", _CARD_SYMBOL_COLS)

    id_tbl, mask_tbl, _ = _build_entity_id_tables(
        vocab.card_to_idx, per_entity, global_symbol_vocab, max_len, "cards"
    )
    return id_tbl, mask_tbl


def build_relic_symbol_ids(
    vocab: Vocab,
    db_path: Path | str | None = None,
    global_symbol_vocab: list[str] | None = None,
    max_len: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    if global_symbol_vocab is None:
        global_symbol_vocab, _ = build_global_symbol_vocab(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        per_entity = _collect_symbols_per_entity(conn, "relics", "id", _RELIC_SYMBOL_COLS)

    id_tbl, mask_tbl, _ = _build_entity_id_tables(
        vocab.relic_to_idx, per_entity, global_symbol_vocab, max_len, "relics"
    )
    return id_tbl, mask_tbl


def build_monster_symbol_ids(
    vocab: Vocab,
    db_path: Path | str | None = None,
    global_symbol_vocab: list[str] | None = None,
    max_len: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    if global_symbol_vocab is None:
        global_symbol_vocab, _ = build_global_symbol_vocab(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        per_entity = _collect_symbols_per_entity(conn, "monsters", "id", _MONSTER_SYMBOL_COLS)

    id_tbl, mask_tbl, _ = _build_entity_id_tables(
        vocab.monster_to_idx, per_entity, global_symbol_vocab, max_len, "monsters"
    )
    return id_tbl, mask_tbl


def build_potion_symbol_ids(
    vocab: Vocab,
    db_path: Path | str | None = None,
    global_symbol_vocab: list[str] | None = None,
    max_len: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    if global_symbol_vocab is None:
        global_symbol_vocab, _ = build_global_symbol_vocab(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        per_entity = _collect_symbols_per_entity(conn, "potions", "id", _POTION_SYMBOL_COLS)

    id_tbl, mask_tbl, _ = _build_entity_id_tables(
        vocab.potion_to_idx, per_entity, global_symbol_vocab, max_len, "potions"
    )
    return id_tbl, mask_tbl


def build_all_symbol_tables(
    vocab: Vocab,
    db_path: Path | str | None = None,
    card_max_len: int = 32,
    relic_max_len: int = 16,
    monster_max_len: int = 32,
    potion_max_len: int = 16,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], KnowledgeMeta]:
    """One-shot build of everything `SymbolicFeaturesHead` needs.

    Returns:
        tables: {"card": (ids, mask), "relic": ..., "monster": ..., "potion": ...}
        meta:   KnowledgeMeta with global vocab, sqlite_sha1, per-table max_len,
                and coverage stats for debug logging.
    """
    db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    global_vocab, sqlite_sha1 = build_global_symbol_vocab(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        per_card = _collect_symbols_per_entity(conn, "cards", "id", _CARD_SYMBOL_COLS)
        per_relic = _collect_symbols_per_entity(conn, "relics", "id", _RELIC_SYMBOL_COLS)
        per_monster = _collect_symbols_per_entity(conn, "monsters", "id", _MONSTER_SYMBOL_COLS)
        per_potion = _collect_symbols_per_entity(conn, "potions", "id", _POTION_SYMBOL_COLS)

    card_ids, card_mask, card_cov = _build_entity_id_tables(
        vocab.card_to_idx, per_card, global_vocab, card_max_len, "cards"
    )
    relic_ids, relic_mask, relic_cov = _build_entity_id_tables(
        vocab.relic_to_idx, per_relic, global_vocab, relic_max_len, "relics"
    )
    monster_ids, monster_mask, monster_cov = _build_entity_id_tables(
        vocab.monster_to_idx, per_monster, global_vocab, monster_max_len, "monsters"
    )
    potion_ids, potion_mask, potion_cov = _build_entity_id_tables(
        vocab.potion_to_idx, per_potion, global_vocab, potion_max_len, "potions"
    )

    meta = KnowledgeMeta(
        global_symbol_vocab=global_vocab,
        sqlite_sha1=sqlite_sha1,
        card_max_len=card_max_len,
        relic_max_len=relic_max_len,
        monster_max_len=monster_max_len,
        potion_max_len=potion_max_len,
        card_coverage=card_cov,
        relic_coverage=relic_cov,
        monster_coverage=monster_cov,
        potion_coverage=potion_cov,
    )

    tables = {
        "card": (card_ids, card_mask),
        "relic": (relic_ids, relic_mask),
        "monster": (monster_ids, monster_mask),
        "potion": (potion_ids, potion_mask),
    }
    return tables, meta


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

def _main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from vocab import load_vocab
    v = load_vocab()
    tables, meta = build_all_symbol_tables(v)
    print(f"Global symbol vocab size: {len(meta.global_symbol_vocab)}")
    print(f"  Special tokens: {meta.global_symbol_vocab[:2]}")
    print(f"  First 5 symbols: {meta.global_symbol_vocab[2:7]}")
    print(f"  Last 5 symbols:  {meta.global_symbol_vocab[-5:]}")
    print(f"sqlite sha1: {meta.sqlite_sha1[:12]}")
    print()
    for name, (ids, mask) in tables.items():
        cov = getattr(meta, f"{name}_coverage")
        print(f"{name}: ids shape={ids.shape}, mask shape={mask.shape}, coverage={cov:.1%}")
    print()
    # Spot-check: first non-special card
    for card_id, idx in v.card_to_idx.items():
        if card_id in _SPECIAL_SYMBOLS:
            continue
        ids, mask = tables["card"]
        active_ids = ids[idx][mask[idx]]
        sym_names = [meta.global_symbol_vocab[i] for i in active_ids]
        print(f"Sample: card '{card_id}' (idx={idx}) -> {len(sym_names)} symbols: {sym_names}")
        break


if __name__ == "__main__":
    _main()
