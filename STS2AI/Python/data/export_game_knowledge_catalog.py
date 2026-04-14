#!/usr/bin/env python3
"""Export a query-friendly game knowledge catalog from source_knowledge + localization."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "source_knowledge.sqlite"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Assets" / "datasets" / "game_knowledge_catalog"
DEFAULT_RUNTIME_CARD_TEXT_PATH = Path(__file__).resolve().parent / "raw" / "card_runtime_texts.json"
SUPPORTED_LOCALES = ("eng", "zhs")
ENTITY_TABLES = ("cards", "relics", "potions", "monsters")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_locale_bundle(locale: str) -> dict[str, dict[str, Any]]:
    loc_dir = REPO_ROOT / "localization" / locale
    return {
        "cards": _read_json(loc_dir / "cards.json"),
        "card_keywords": _read_json(loc_dir / "card_keywords.json"),
        "relics": _read_json(loc_dir / "relics.json"),
        "potions": _read_json(loc_dir / "potions.json"),
        "monsters": _read_json(loc_dir / "monsters.json"),
    }


def _load_runtime_card_texts(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    by_card: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        card_id = str(row.get("id") or row.get("Id")).lower()
        locale = str(row.get("locale") or row.get("Locale")).lower()
        by_card.setdefault(card_id, {})[locale] = {
            "title": row.get("title") or row.get("Title") or "",
            "description_runtime": row.get("descriptionRuntime") or row.get("DescriptionRuntime") or "",
            "upgrade_preview_runtime": row.get("upgradePreviewRuntime") or row.get("UpgradePreviewRuntime") or "",
        }
    return by_card


def _normalize_source_path(path: str | None) -> str | None:
    if not path:
        return None
    match = re.search(r"(src[\\/].+)$", path, flags=re.IGNORECASE)
    if not match:
        return path.replace("\\", "/")
    return match.group(1).replace("\\", "/")


def _load_rows(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def _json_load(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"Unsupported JSON payload type: {type(raw)!r}")


def _entity_key(entity_id: str) -> str:
    return entity_id.upper()


def _detect_upgrade_tokens(text: str | None) -> bool:
    if not text:
        return False
    return "{IfUpgraded:" in text or "IfUpgraded" in text


_IF_UPGRADED_SHOW_RE = re.compile(r"\{IfUpgraded:show:([^{}|]*)\|([^{}]*)\}")
_IF_UPGRADED_SHOW_UNARY_RE = re.compile(r"\{IfUpgraded:show:([^|{}]+)\}")


def _render_static_upgrade_preview(text: str | None) -> str | None:
    """Resolve simple IfUpgraded:show tokens without a runtime CardModel instance.

    This intentionally only resolves the textual upgraded/non-upgraded branch.
    Dynamic vars such as `{Damage:diff()}` remain as-is because source-only export
    does not know upgraded combat values.
    """
    if not text:
        return None
    rendered = text
    while True:
        next_text = _IF_UPGRADED_SHOW_RE.sub(lambda m: m.group(1), rendered)
        next_text = _IF_UPGRADED_SHOW_UNARY_RE.sub(lambda m: m.group(1), next_text)
        if next_text == rendered:
            return rendered
        rendered = next_text


def _keyword_entries(keyword_ids: list[str], locale_bundles: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for keyword_id in keyword_ids:
        key = keyword_id.upper()
        entries.append(
            {
                "id": keyword_id,
                "title_en": locale_bundles["eng"]["card_keywords"].get(f"{key}.title"),
                "title_zhs": locale_bundles["zhs"]["card_keywords"].get(f"{key}.title"),
                "description_en": locale_bundles["eng"]["card_keywords"].get(f"{key}.description"),
                "description_zhs": locale_bundles["zhs"]["card_keywords"].get(f"{key}.description"),
            }
        )
    return entries


def _build_card_record(
    row: dict[str, Any],
    locale_bundles: dict[str, dict[str, dict[str, Any]]],
    runtime_card_texts: dict[str, dict[str, dict[str, str]]],
) -> dict[str, Any]:
    key = _entity_key(row["id"])
    desc_en = locale_bundles["eng"]["cards"].get(f"{key}.description")
    desc_zhs = locale_bundles["zhs"]["cards"].get(f"{key}.description")
    upgrade_preview_static_en = _render_static_upgrade_preview(desc_en)
    upgrade_preview_static_zhs = _render_static_upgrade_preview(desc_zhs)
    runtime_text = runtime_card_texts.get(row["id"], {})
    keywords = _json_load(row["keywords_json"]) or []
    return {
        "entity_type": "card",
        "id": row["id"],
        "class_name": row["class_name"],
        "source_path": _normalize_source_path(row.get("file_path")),
        "source_sha1": row["source_sha1"],
        "title_en": locale_bundles["eng"]["cards"].get(f"{key}.title"),
        "title_zhs": locale_bundles["zhs"]["cards"].get(f"{key}.title"),
        "description_en": desc_en,
        "description_zhs": desc_zhs,
        "upgrade_preview_static_en": upgrade_preview_static_en,
        "upgrade_preview_static_zhs": upgrade_preview_static_zhs,
        "description_runtime_en": runtime_text.get("eng", {}).get("description_runtime"),
        "description_runtime_zhs": runtime_text.get("zhs", {}).get("description_runtime"),
        "upgrade_preview_runtime_en": runtime_text.get("eng", {}).get("upgrade_preview_runtime"),
        "upgrade_preview_runtime_zhs": runtime_text.get("zhs", {}).get("upgrade_preview_runtime"),
        "has_upgrade_tokens": _detect_upgrade_tokens(desc_en) or _detect_upgrade_tokens(desc_zhs),
        "cost": row["cost"],
        "card_type": row["card_type"],
        "rarity": row["rarity"],
        "target_type": row["target_type"],
        "tags": _json_load(row["tags_json"]) or [],
        "keywords": keywords,
        "keyword_details": _keyword_entries(keywords, locale_bundles),
        "card_tags": _json_load(row["card_tags_json"]) or [],
        "powers": _json_load(row["powers_json"]) or [],
        "dynamic_vars": _json_load(row["dynamic_vars_json"]) or [],
        "commands": _json_load(row["commands_json"]) or [],
    }


def _build_relic_record(row: dict[str, Any], locale_bundles: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    key = _entity_key(row["id"])
    return {
        "entity_type": "relic",
        "id": row["id"],
        "class_name": row["class_name"],
        "source_path": _normalize_source_path(row.get("file_path")),
        "source_sha1": row["source_sha1"],
        "title_en": locale_bundles["eng"]["relics"].get(f"{key}.title"),
        "title_zhs": locale_bundles["zhs"]["relics"].get(f"{key}.title"),
        "description_en": locale_bundles["eng"]["relics"].get(f"{key}.description"),
        "description_zhs": locale_bundles["zhs"]["relics"].get(f"{key}.description"),
        "flavor_en": locale_bundles["eng"]["relics"].get(f"{key}.flavor"),
        "flavor_zhs": locale_bundles["zhs"]["relics"].get(f"{key}.flavor"),
        "rarity": row["rarity"],
        "powers": _json_load(row["powers_json"]) or [],
        "dynamic_vars": _json_load(row["dynamic_vars_json"]) or [],
        "commands": _json_load(row["commands_json"]) or [],
    }


def _build_potion_record(row: dict[str, Any], locale_bundles: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    key = _entity_key(row["id"])
    return {
        "entity_type": "potion",
        "id": row["id"],
        "class_name": row["class_name"],
        "source_path": _normalize_source_path(row.get("file_path")),
        "source_sha1": row["source_sha1"],
        "title_en": locale_bundles["eng"]["potions"].get(f"{key}.title"),
        "title_zhs": locale_bundles["zhs"]["potions"].get(f"{key}.title"),
        "description_en": locale_bundles["eng"]["potions"].get(f"{key}.description"),
        "description_zhs": locale_bundles["zhs"]["potions"].get(f"{key}.description"),
        "rarity": row["rarity"],
        "usage": row["usage"],
        "target_type": row["target_type"],
        "powers": _json_load(row["powers_json"]) or [],
        "commands": _json_load(row["commands_json"]) or [],
    }


def _build_monster_record(row: dict[str, Any], locale_bundles: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    key = _entity_key(row["id"])
    moves = _json_load(row["moves_json"]) or []
    localized_moves = []
    for move in moves:
        label = move.get("label")
        localized_moves.append(
            {
                "order": move.get("order"),
                "label": label,
                "intent": move.get("intent"),
                "title_en": locale_bundles["eng"]["monsters"].get(f"{key}.moves.{label}.title") if label else None,
                "title_zhs": locale_bundles["zhs"]["monsters"].get(f"{key}.moves.{label}.title") if label else None,
            }
        )
    return {
        "entity_type": "monster",
        "id": row["id"],
        "class_name": row["class_name"],
        "source_path": _normalize_source_path(row.get("file_path")),
        "source_sha1": row["source_sha1"],
        "name_en": locale_bundles["eng"]["monsters"].get(f"{key}.name"),
        "name_zhs": locale_bundles["zhs"]["monsters"].get(f"{key}.name"),
        "min_initial_hp_expr": row["min_initial_hp_expr"],
        "max_initial_hp_expr": row["max_initial_hp_expr"],
        "death_sfx_expr": row["death_sfx_expr"],
        "intents": _json_load(row["intents_json"]) or [],
        "move_labels": _json_load(row["move_labels_json"]) or [],
        "moves": localized_moves,
        "powers": _json_load(row["powers_json"]) or [],
        "commands": _json_load(row["commands_json"]) or [],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_sqlite(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        for table_name, rows in tables.items():
            if table_name == "cards":
                conn.execute(
                    f"""
                    CREATE TABLE {table_name} (
                        id TEXT PRIMARY KEY,
                        title_en TEXT,
                        title_zhs TEXT,
                        description_en TEXT,
                        description_zhs TEXT,
                        upgrade_preview_static_en TEXT,
                        upgrade_preview_static_zhs TEXT,
                        description_runtime_en TEXT,
                        description_runtime_zhs TEXT,
                        upgrade_preview_runtime_en TEXT,
                        upgrade_preview_runtime_zhs TEXT,
                        cost INTEGER,
                        card_type TEXT,
                        rarity TEXT,
                        target_type TEXT,
                        source_path TEXT,
                        tags_json TEXT,
                        keywords_json TEXT,
                        card_tags_json TEXT,
                        powers_json TEXT,
                        dynamic_vars_json TEXT,
                        commands_json TEXT,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
            elif table_name == "relics":
                conn.execute(
                    f"""
                    CREATE TABLE {table_name} (
                        id TEXT PRIMARY KEY,
                        title_en TEXT,
                        title_zhs TEXT,
                        description_en TEXT,
                        description_zhs TEXT,
                        flavor_en TEXT,
                        flavor_zhs TEXT,
                        rarity TEXT,
                        source_path TEXT,
                        powers_json TEXT,
                        dynamic_vars_json TEXT,
                        commands_json TEXT,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
            elif table_name == "potions":
                conn.execute(
                    f"""
                    CREATE TABLE {table_name} (
                        id TEXT PRIMARY KEY,
                        title_en TEXT,
                        title_zhs TEXT,
                        description_en TEXT,
                        description_zhs TEXT,
                        rarity TEXT,
                        usage TEXT,
                        target_type TEXT,
                        source_path TEXT,
                        powers_json TEXT,
                        commands_json TEXT,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
            elif table_name == "monsters":
                conn.execute(
                    f"""
                    CREATE TABLE {table_name} (
                        id TEXT PRIMARY KEY,
                        title_en TEXT,
                        title_zhs TEXT,
                        name_en TEXT,
                        name_zhs TEXT,
                        min_initial_hp_expr TEXT,
                        max_initial_hp_expr TEXT,
                        death_sfx_expr TEXT,
                        source_path TEXT,
                        intents_json TEXT,
                        move_labels_json TEXT,
                        moves_json TEXT,
                        powers_json TEXT,
                        commands_json TEXT,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
            else:
                conn.execute(
                    f"""
                    CREATE TABLE {table_name} (
                        id TEXT PRIMARY KEY,
                        title_en TEXT,
                        title_zhs TEXT,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
            for row in rows:
                if table_name == "cards":
                    conn.execute(
                        f"""
                        INSERT INTO {table_name} (
                            id,
                            title_en,
                            title_zhs,
                            description_en,
                            description_zhs,
                            upgrade_preview_static_en,
                            upgrade_preview_static_zhs,
                            description_runtime_en,
                            description_runtime_zhs,
                            upgrade_preview_runtime_en,
                            upgrade_preview_runtime_zhs,
                            cost,
                            card_type,
                            rarity,
                            target_type,
                            source_path,
                            tags_json,
                            keywords_json,
                            card_tags_json,
                            powers_json,
                            dynamic_vars_json,
                            commands_json,
                            payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["id"],
                            row.get("title_en") or row.get("name_en"),
                            row.get("title_zhs") or row.get("name_zhs"),
                            row.get("description_en"),
                            row.get("description_zhs"),
                            row.get("upgrade_preview_static_en"),
                            row.get("upgrade_preview_static_zhs"),
                            row.get("description_runtime_en"),
                            row.get("description_runtime_zhs"),
                            row.get("upgrade_preview_runtime_en"),
                            row.get("upgrade_preview_runtime_zhs"),
                            row.get("cost"),
                            row.get("card_type"),
                            row.get("rarity"),
                            row.get("target_type"),
                            row.get("source_path"),
                            json.dumps(row.get("tags") or [], ensure_ascii=False),
                            json.dumps(row.get("keywords") or [], ensure_ascii=False),
                            json.dumps(row.get("card_tags") or [], ensure_ascii=False),
                            json.dumps(row.get("powers") or [], ensure_ascii=False),
                            json.dumps(row.get("dynamic_vars") or [], ensure_ascii=False),
                            json.dumps(row.get("commands") or [], ensure_ascii=False),
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
                elif table_name == "relics":
                    conn.execute(
                        f"""
                        INSERT INTO {table_name} (
                            id,
                            title_en,
                            title_zhs,
                            description_en,
                            description_zhs,
                            flavor_en,
                            flavor_zhs,
                            rarity,
                            source_path,
                            powers_json,
                            dynamic_vars_json,
                            commands_json,
                            payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["id"],
                            row.get("title_en") or row.get("name_en"),
                            row.get("title_zhs") or row.get("name_zhs"),
                            row.get("description_en"),
                            row.get("description_zhs"),
                            row.get("flavor_en"),
                            row.get("flavor_zhs"),
                            row.get("rarity"),
                            row.get("source_path"),
                            json.dumps(row.get("powers") or [], ensure_ascii=False),
                            json.dumps(row.get("dynamic_vars") or [], ensure_ascii=False),
                            json.dumps(row.get("commands") or [], ensure_ascii=False),
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
                elif table_name == "potions":
                    conn.execute(
                        f"""
                        INSERT INTO {table_name} (
                            id,
                            title_en,
                            title_zhs,
                            description_en,
                            description_zhs,
                            rarity,
                            usage,
                            target_type,
                            source_path,
                            powers_json,
                            commands_json,
                            payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["id"],
                            row.get("title_en") or row.get("name_en"),
                            row.get("title_zhs") or row.get("name_zhs"),
                            row.get("description_en"),
                            row.get("description_zhs"),
                            row.get("rarity"),
                            row.get("usage"),
                            row.get("target_type"),
                            row.get("source_path"),
                            json.dumps(row.get("powers") or [], ensure_ascii=False),
                            json.dumps(row.get("commands") or [], ensure_ascii=False),
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
                elif table_name == "monsters":
                    conn.execute(
                        f"""
                        INSERT INTO {table_name} (
                            id,
                            title_en,
                            title_zhs,
                            name_en,
                            name_zhs,
                            min_initial_hp_expr,
                            max_initial_hp_expr,
                            death_sfx_expr,
                            source_path,
                            intents_json,
                            move_labels_json,
                            moves_json,
                            powers_json,
                            commands_json,
                            payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["id"],
                            row.get("title_en") or row.get("name_en"),
                            row.get("title_zhs") or row.get("name_zhs"),
                            row.get("name_en"),
                            row.get("name_zhs"),
                            row.get("min_initial_hp_expr"),
                            row.get("max_initial_hp_expr"),
                            row.get("death_sfx_expr"),
                            row.get("source_path"),
                            json.dumps(row.get("intents") or [], ensure_ascii=False),
                            json.dumps(row.get("move_labels") or [], ensure_ascii=False),
                            json.dumps(row.get("moves") or [], ensure_ascii=False),
                            json.dumps(row.get("powers") or [], ensure_ascii=False),
                            json.dumps(row.get("commands") or [], ensure_ascii=False),
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
                else:
                    conn.execute(
                        f"INSERT INTO {table_name} (id, title_en, title_zhs, payload_json) VALUES (?, ?, ?, ?)",
                        (
                            row["id"],
                            row.get("title_en") or row.get("name_en"),
                            row.get("title_zhs") or row.get("name_zhs"),
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
            conn.execute(f"CREATE INDEX idx_{table_name}_title_en ON {table_name}(title_en)")
            conn.execute(f"CREATE INDEX idx_{table_name}_title_zhs ON {table_name}(title_zhs)")
        conn.commit()
    finally:
        conn.close()


def build_catalog(db_path: Path, output_dir: Path, runtime_card_text_path: Path) -> dict[str, Any]:
    locale_bundles = {locale: _load_locale_bundle(locale) for locale in SUPPORTED_LOCALES}
    runtime_card_texts = _load_runtime_card_texts(runtime_card_text_path)
    conn = sqlite3.connect(db_path)
    try:
        cards = [_build_card_record(row, locale_bundles, runtime_card_texts) for row in _load_rows(conn, "cards")]
        relics = [_build_relic_record(row, locale_bundles) for row in _load_rows(conn, "relics")]
        potions = [_build_potion_record(row, locale_bundles) for row in _load_rows(conn, "potions")]
        monsters = [_build_monster_record(row, locale_bundles) for row in _load_rows(conn, "monsters")]
    finally:
        conn.close()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "cards": cards,
        "relics": relics,
        "potions": potions,
        "monsters": monsters,
    }
    for name, rows in tables.items():
        _write_jsonl(output_dir / f"{name}.jsonl", rows)
        _write_json(output_dir / f"{name}.sample.json", rows[:3])

    _write_sqlite(output_dir / "game_knowledge_catalog.sqlite", tables)

    manifest = {
        "db_source": str(db_path),
        "output_dir": str(output_dir),
        "runtime_card_text_path": str(runtime_card_text_path) if runtime_card_text_path.exists() else None,
        "entity_counts": {name: len(rows) for name, rows in tables.items()},
        "locales": list(SUPPORTED_LOCALES),
        "notes": {
            "cards": "卡牌记录同时包含本地化基础文本、静态升级预览文本，以及运行时精确描述/升级预览文本；payload_json 保留完整原始记录。",
            "monsters": "怪物导出包含名字、HP 表达式、intent、move label，以及按本地化表补齐的 move title。",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a query-friendly game knowledge catalog.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-card-texts", type=Path, default=DEFAULT_RUNTIME_CARD_TEXT_PATH)
    args = parser.parse_args()

    manifest = build_catalog(args.db, args.output_dir, args.runtime_card_texts)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
