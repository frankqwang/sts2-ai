#!/usr/bin/env python3
"""Build a source-derived STS2 knowledge database for training and analysis.

The database is intentionally source-first:
- scan C# model definitions under src/Core/Models
- reuse existing Python extractors where available
- export a SQLite database plus a small JSON manifest

This is meant to be a stable "knowledge base" layer that later training jobs
can query without repeatedly reparsing raw source files.
"""

from __future__ import annotations

# 2026-04-08 (wizardly cleanup): file moved into tools/python/data/.
# Sys.path bootstrap so `import _path_init` (which lives at tools/python/)
# can be found. Same pattern as runners/diagnostics/benchmarks/data_gen.
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # noqa: E402

import _path_init  # noqa: F401,E402  (adds tools/python/{core,ipc,search} to sys.path)

import argparse
import hashlib
import json
import re
import sqlite3
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 2026-04-08 (wizardly cleanup): card_tags / vocab moved into core/.
# `import _path_init` above puts core/ on sys.path so the flat imports below
# still resolve.
from card_tags import extract_tags_from_card
from vocab import _extract_card_props, _slugify


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _findall_sorted(pattern: str, text: str) -> list[str]:
    return sorted({match for match in re.findall(pattern, text, flags=re.MULTILINE)})


def _extract_ctor_enum_arg(text: str, enum_name: str, position: int) -> str | None:
    m = re.search(
        r"base\(([^)]*)\)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not m:
        return None
    args = [part.strip() for part in m.group(1).split(",")]
    if position >= len(args):
        return None
    target = args[position]
    enum_match = re.search(rf"{re.escape(enum_name)}\.(\w+)", target)
    return enum_match.group(1) if enum_match else None


def _extract_override_expr(text: str, property_name: str, return_type: str = "int") -> str | None:
    m = re.search(
        rf"public\s+override\s+{re.escape(return_type)}\s+{re.escape(property_name)}\s*=>\s*(.+?);",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if m:
        return " ".join(m.group(1).split())
    return None


def _extract_override_enum(text: str, property_name: str, enum_name: str) -> str | None:
    m = re.search(
        rf"public\s+override\s+{re.escape(enum_name)}\s+{re.escape(property_name)}\s*=>\s*{re.escape(enum_name)}\.(\w+)\s*;",
        text,
        flags=re.MULTILINE,
    )
    return m.group(1) if m else None


def _extract_dynamic_var_types(text: str) -> list[str]:
    return _findall_sorted(r"new\s+(\w+Var)(?:<[^>]+>)?\b", text)


def _extract_power_refs(text: str) -> list[str]:
    refs = set()
    for pattern in (
        r"PowerCmd\.Apply<(\w+)>",
        r"PowerVar<(\w+)>",
        r"HoverTipFactory\.FromPower<(\w+)>",
    ):
        refs.update(re.findall(pattern, text))
    return sorted(refs)


def _extract_keywords(text: str) -> list[str]:
    return _findall_sorted(r"CardKeyword\.(\w+)", text)


def _extract_card_tags(text: str) -> list[str]:
    return _findall_sorted(r"CardTag\.(\w+)", text)


def _extract_command_refs(text: str, prefixes: tuple[str, ...]) -> list[str]:
    refs: set[str] = set()
    for prefix in prefixes:
        refs.update(re.findall(rf"{re.escape(prefix)}\.(\w+)", text))
    return sorted(refs)


def _extract_monster_moves(text: str) -> list[dict[str, Any]]:
    """Extract monster moves as a single ordered list of {label, intent}.

    2026-04-08 (wizardly cleanup): the previous version returned two
    parallel lists (labels-with-order, intents-with-order) appended end
    to end. The downstream consumer had to assume `len(labels) ==
    len(intents)` and that they appeared in the same order in source —
    a brittle implicit contract.

    The new version walks a unified regex that captures `MoveState("Name")
    ... new XxxIntent(` blocks and pairs them by source-file order. If a
    block has only a label (no Intent yet) or only an Intent (no MoveState
    label), we still emit a row but with the missing field set to None.
    """
    moves: list[dict[str, Any]] = []
    label_iter = list(re.finditer(r'new\s+MoveState\("([^"]+)"', text))
    intent_iter = list(re.finditer(r"new\s+(\w+Intent)\(", text))

    # Sort both by file offset and walk them together. For each MoveState
    # label, attach the FIRST intent occurrence that appears AFTER the
    # label and BEFORE the next label. (This is the typical pattern in
    # the C# source: `new MoveState("Bash") { Intent = new BashIntent(...) }`
    # but the Intent constructor lives a few characters past the MoveState).
    used_intent_indices: set[int] = set()
    for li, lm in enumerate(label_iter):
        label = lm.group(1)
        next_label_start = label_iter[li + 1].start() if li + 1 < len(label_iter) else len(text)
        intent_for_label: str | None = None
        for ii, im in enumerate(intent_iter):
            if ii in used_intent_indices:
                continue
            if im.start() < lm.start():
                continue
            if im.start() >= next_label_start:
                break
            intent_for_label = im.group(1)
            used_intent_indices.add(ii)
            break
        moves.append({
            "order": len(moves),
            "label": label,
            "intent": intent_for_label,
        })

    # Any intents that didn't get paired with a label (e.g. monsters that
    # define intents inline without a MoveState wrapper) become standalone
    # moves with label=None.
    for ii, im in enumerate(intent_iter):
        if ii in used_intent_indices:
            continue
        moves.append({
            "order": len(moves),
            "label": None,
            "intent": im.group(1),
        })

    return moves


def _extract_all_possible_monsters(text: str) -> list[str]:
    return _findall_sorted(r"ModelDb\.Monster<(\w+)>\(\)", text)


@dataclass
class EntityRow:
    entity_id: str
    class_name: str
    file_path: str
    source_sha1: str
    payload: dict[str, Any]


def _iter_model_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    return sorted(
        path for path in model_dir.iterdir()
        if path.is_file() and path.suffix == ".cs" and not path.name.endswith(".uid")
    )


# 2026-04-08 (wizardly cleanup): per-file extraction errors are appended here
# instead of crashing the whole scan. Surfaced in the manifest at the end.
_BUILD_ERRORS: list[dict[str, str]] = []


def _safe_extract(label: str, path: Path, fn) -> Any:
    """Run a per-file extractor; on exception log to _BUILD_ERRORS and return None."""
    try:
        return fn()
    except Exception as exc:
        _BUILD_ERRORS.append({
            "phase": label,
            "file": str(path),
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:200],
            "trace": traceback.format_exc(limit=2)[-300:],
        })
        return None


def _build_card_rows(repo_root: Path) -> list[EntityRow]:
    model_dir = repo_root / "src" / "Core" / "Models" / "Cards"
    rows: list[EntityRow] = []
    for path in _iter_model_files(model_dir):
        try:
            text = _read_text(path)
            props = _extract_card_props(str(path))
            row = EntityRow(
                entity_id=_slugify(path.stem).lower(),
                class_name=path.stem,
                file_path=str(path),
                source_sha1=_sha1_text(text),
                payload={
                    "cost": props.get("cost"),
                    "card_type": props.get("type"),
                    "rarity": props.get("rarity"),
                    "target_type": _extract_ctor_enum_arg(text, "TargetType", 3),
                    "tags": extract_tags_from_card(path),
                    "keywords": _extract_keywords(text),
                    "card_tags": _extract_card_tags(text),
                    "powers": _extract_power_refs(text),
                    "dynamic_vars": _extract_dynamic_var_types(text),
                    "commands": _extract_command_refs(
                        text,
                        ("DamageCmd", "PowerCmd", "CardPileCmd", "PlayerCmd", "CardCmd"),
                    ),
                },
            )
            rows.append(row)
        except Exception as exc:
            _BUILD_ERRORS.append({
                "phase": "cards",
                "file": str(path),
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:200],
                "trace": traceback.format_exc(limit=2)[-300:],
            })
    return rows


def _build_monster_rows(repo_root: Path) -> list[EntityRow]:
    model_dir = repo_root / "src" / "Core" / "Models" / "Monsters"
    rows: list[EntityRow] = []
    for path in _iter_model_files(model_dir):
        try:
            text = _read_text(path)
            row = EntityRow(
                entity_id=_slugify(path.stem).lower(),
                class_name=path.stem,
                file_path=str(path),
                source_sha1=_sha1_text(text),
                payload={
                    "min_initial_hp_expr": _extract_override_expr(text, "MinInitialHp"),
                    "max_initial_hp_expr": _extract_override_expr(text, "MaxInitialHp"),
                    "death_sfx": _extract_override_expr(text, "DeathSfx", return_type="string"),
                    "intents": _findall_sorted(r"new\s+(\w+Intent)\(", text),
                    "move_labels": re.findall(r'new\s+MoveState\("([^"]+)"', text),
                    "moves": _extract_monster_moves(text),
                    "powers": _extract_power_refs(text),
                    "commands": _extract_command_refs(
                        text,
                        ("DamageCmd", "PowerCmd", "CreatureCmd", "Cmd", "SfxCmd", "VfxCmd"),
                    ),
                },
            )
            rows.append(row)
        except Exception as exc:
            _BUILD_ERRORS.append({
                "phase": "monsters",
                "file": str(path),
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:200],
                "trace": traceback.format_exc(limit=2)[-300:],
            })
    return rows


def _build_relic_rows(repo_root: Path) -> list[EntityRow]:
    model_dir = repo_root / "src" / "Core" / "Models" / "Relics"
    rows: list[EntityRow] = []
    for path in _iter_model_files(model_dir):
        try:
            text = _read_text(path)
            row = EntityRow(
                entity_id=_slugify(path.stem).lower(),
                class_name=path.stem,
                file_path=str(path),
                source_sha1=_sha1_text(text),
                payload={
                    "rarity": _extract_override_enum(text, "Rarity", "RelicRarity"),
                    "dynamic_vars": _extract_dynamic_var_types(text),
                    "powers": _extract_power_refs(text),
                    "commands": _extract_command_refs(
                        text,
                        ("PowerCmd", "DamageCmd", "PlayerCmd", "CardPileCmd", "RelicCmd"),
                    ),
                },
            )
            rows.append(row)
        except Exception as exc:
            _BUILD_ERRORS.append({
                "phase": "relics",
                "file": str(path),
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:200],
                "trace": traceback.format_exc(limit=2)[-300:],
            })
    return rows


def _build_potion_rows(repo_root: Path) -> list[EntityRow]:
    model_dir = repo_root / "src" / "Core" / "Models" / "Potions"
    rows: list[EntityRow] = []
    for path in _iter_model_files(model_dir):
        try:
            text = _read_text(path)
            row = EntityRow(
                entity_id=_slugify(path.stem).lower(),
                class_name=path.stem,
                file_path=str(path),
                source_sha1=_sha1_text(text),
                payload={
                    "rarity": _extract_override_enum(text, "Rarity", "PotionRarity"),
                    "usage": _extract_override_enum(text, "Usage", "PotionUsage"),
                    "target_type": _extract_override_enum(text, "TargetType", "TargetType"),
                    "powers": _extract_power_refs(text),
                    "commands": _extract_command_refs(
                        text,
                        ("PowerCmd", "DamageCmd", "CardPileCmd", "CardSelectCmd", "PlayerCmd"),
                    ),
                },
            )
            rows.append(row)
        except Exception as exc:
            _BUILD_ERRORS.append({
                "phase": "potions",
                "file": str(path),
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:200],
                "trace": traceback.format_exc(limit=2)[-300:],
            })
    return rows


def _extract_encounter_slots(text: str) -> list[str]:
    """Pull the `Slots` override expression's string literals if present.

    The original implementation called `re.search(...).group(1)` twice and
    crashed on the SECOND search if the first happened to find something
    truthy but the second didn't. Hoist the search once and guard.
    """
    m = re.search(r"Slots\s*=>\s*(.+?);", text, flags=re.DOTALL)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def _build_encounter_rows(repo_root: Path) -> list[EntityRow]:
    model_dir = repo_root / "src" / "Core" / "Models" / "Encounters"
    rows: list[EntityRow] = []
    for path in _iter_model_files(model_dir):
        try:
            text = _read_text(path)
            row = EntityRow(
                entity_id=_slugify(path.stem).lower(),
                class_name=path.stem,
                file_path=str(path),
                source_sha1=_sha1_text(text),
                payload={
                    "room_type": _extract_override_enum(text, "RoomType", "RoomType"),
                    "is_weak": "public override bool IsWeak => true;" in text,
                    "has_scene": "public override bool HasScene => true;" in text,
                    "slots": _extract_encounter_slots(text),
                    "tags": _findall_sorted(r"EncounterTag\.(\w+)", text),
                    "possible_monsters": _extract_all_possible_monsters(text),
                },
            )
            rows.append(row)
        except Exception as exc:
            _BUILD_ERRORS.append({
                "phase": "encounters",
                "file": str(path),
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:200],
                "trace": traceback.format_exc(limit=2)[-300:],
            })
    return rows


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cards (
            id TEXT PRIMARY KEY,
            class_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            source_sha1 TEXT NOT NULL,
            cost INTEGER,
            card_type TEXT,
            rarity TEXT,
            target_type TEXT,
            tags_json TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            card_tags_json TEXT NOT NULL,
            powers_json TEXT NOT NULL,
            dynamic_vars_json TEXT NOT NULL,
            commands_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS monsters (
            id TEXT PRIMARY KEY,
            class_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            source_sha1 TEXT NOT NULL,
            min_initial_hp_expr TEXT,
            max_initial_hp_expr TEXT,
            death_sfx_expr TEXT,
            intents_json TEXT NOT NULL,
            move_labels_json TEXT NOT NULL,
            moves_json TEXT NOT NULL,
            powers_json TEXT NOT NULL,
            commands_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS relics (
            id TEXT PRIMARY KEY,
            class_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            source_sha1 TEXT NOT NULL,
            rarity TEXT,
            dynamic_vars_json TEXT NOT NULL,
            powers_json TEXT NOT NULL,
            commands_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS potions (
            id TEXT PRIMARY KEY,
            class_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            source_sha1 TEXT NOT NULL,
            rarity TEXT,
            usage TEXT,
            target_type TEXT,
            powers_json TEXT NOT NULL,
            commands_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS encounters (
            id TEXT PRIMARY KEY,
            class_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            source_sha1 TEXT NOT NULL,
            room_type TEXT,
            is_weak INTEGER NOT NULL,
            has_scene INTEGER NOT NULL,
            slots_json TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            possible_monsters_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cards_rarity ON cards(rarity);
        CREATE INDEX IF NOT EXISTS idx_monsters_intents ON monsters(id);
        CREATE INDEX IF NOT EXISTS idx_encounters_room_type ON encounters(room_type);
        """
    )


def _insert_rows(conn: sqlite3.Connection, table: str, rows: list[EntityRow]) -> None:
    if table == "cards":
        conn.executemany(
            """
            INSERT OR REPLACE INTO cards (
                id, class_name, file_path, source_sha1, cost, card_type, rarity, target_type,
                tags_json, keywords_json, card_tags_json, powers_json, dynamic_vars_json, commands_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.entity_id,
                    row.class_name,
                    row.file_path,
                    row.source_sha1,
                    row.payload["cost"],
                    row.payload["card_type"],
                    row.payload["rarity"],
                    row.payload["target_type"],
                    _json(row.payload["tags"]),
                    _json(row.payload["keywords"]),
                    _json(row.payload["card_tags"]),
                    _json(row.payload["powers"]),
                    _json(row.payload["dynamic_vars"]),
                    _json(row.payload["commands"]),
                )
                for row in rows
            ],
        )
        return
    if table == "monsters":
        conn.executemany(
            """
            INSERT OR REPLACE INTO monsters (
                id, class_name, file_path, source_sha1, min_initial_hp_expr, max_initial_hp_expr,
                death_sfx_expr, intents_json, move_labels_json, moves_json, powers_json, commands_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.entity_id,
                    row.class_name,
                    row.file_path,
                    row.source_sha1,
                    row.payload["min_initial_hp_expr"],
                    row.payload["max_initial_hp_expr"],
                    row.payload["death_sfx"],
                    _json(row.payload["intents"]),
                    _json(row.payload["move_labels"]),
                    _json(row.payload["moves"]),
                    _json(row.payload["powers"]),
                    _json(row.payload["commands"]),
                )
                for row in rows
            ],
        )
        return
    if table == "relics":
        conn.executemany(
            """
            INSERT OR REPLACE INTO relics (
                id, class_name, file_path, source_sha1, rarity, dynamic_vars_json, powers_json, commands_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.entity_id,
                    row.class_name,
                    row.file_path,
                    row.source_sha1,
                    row.payload["rarity"],
                    _json(row.payload["dynamic_vars"]),
                    _json(row.payload["powers"]),
                    _json(row.payload["commands"]),
                )
                for row in rows
            ],
        )
        return
    if table == "potions":
        conn.executemany(
            """
            INSERT OR REPLACE INTO potions (
                id, class_name, file_path, source_sha1, rarity, usage, target_type, powers_json, commands_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.entity_id,
                    row.class_name,
                    row.file_path,
                    row.source_sha1,
                    row.payload["rarity"],
                    row.payload["usage"],
                    row.payload["target_type"],
                    _json(row.payload["powers"]),
                    _json(row.payload["commands"]),
                )
                for row in rows
            ],
        )
        return
    if table == "encounters":
        conn.executemany(
            """
            INSERT OR REPLACE INTO encounters (
                id, class_name, file_path, source_sha1, room_type, is_weak, has_scene, slots_json, tags_json, possible_monsters_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.entity_id,
                    row.class_name,
                    row.file_path,
                    row.source_sha1,
                    row.payload["room_type"],
                    int(bool(row.payload["is_weak"])),
                    int(bool(row.payload["has_scene"])),
                    _json(row.payload["slots"]),
                    _json(row.payload["tags"]),
                    _json(row.payload["possible_monsters"]),
                )
                for row in rows
            ],
        )
        return
    raise ValueError(f"Unsupported table: {table}")


def build_source_database(
    repo_root: str | Path,
    *,
    output_db: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_db = Path(output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = output_db.with_suffix(".manifest.json")
    manifest_path = Path(manifest_path)

    cards = _build_card_rows(repo_root)
    monsters = _build_monster_rows(repo_root)
    relics = _build_relic_rows(repo_root)
    potions = _build_potion_rows(repo_root)
    encounters = _build_encounter_rows(repo_root)

    conn = sqlite3.connect(output_db)
    try:
        _create_schema(conn)
        _insert_rows(conn, "cards", cards)
        _insert_rows(conn, "monsters", monsters)
        _insert_rows(conn, "relics", relics)
        _insert_rows(conn, "potions", potions)
        _insert_rows(conn, "encounters", encounters)

        metadata = {
            "schema_version": 1,
            "repo_root": str(repo_root),
            "counts": {
                "cards": len(cards),
                "monsters": len(monsters),
                "relics": len(relics),
                "potions": len(potions),
                "encounters": len(encounters),
            },
        }
        conn.execute("INSERT OR REPLACE INTO metadata (key, value_json) VALUES (?, ?)", ("build", _json(metadata)))
        conn.commit()
    finally:
        conn.close()

    manifest = {
        "schema_version": 1,
        "database_path": str(output_db),
        "repo_root": str(repo_root),
        "counts": {
            "cards": len(cards),
            "monsters": len(monsters),
            "relics": len(relics),
            "potions": len(potions),
            "encounters": len(encounters),
        },
        "sample_ids": {
            "cards": [row.entity_id for row in cards[:5]],
            "monsters": [row.entity_id for row in monsters[:5]],
            "relics": [row.entity_id for row in relics[:5]],
            "potions": [row.entity_id for row in potions[:5]],
            "encounters": [row.entity_id for row in encounters[:5]],
        },
        "extraction_errors": _BUILD_ERRORS,  # 2026-04-08: surface per-file failures
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a SQLite STS2 source knowledge database")
    parser.add_argument("--repo-root", type=str, default=None, help="Path to repo root")
    parser.add_argument(
        "--output-db",
        type=str,
        default=str(Path(__file__).with_name("source_knowledge.sqlite")),
        help="Output SQLite database path",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional JSON manifest path (default: alongside db)",
    )
    args = parser.parse_args()

    # 2026-04-08 (wizardly cleanup): file moved into tools/python/data/.
    # parents[3] lands at the worktree root (data/ -> python/ -> tools/ -> repo/).
    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[3]
    manifest = build_source_database(repo_root, output_db=args.output_db, manifest_path=args.manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
