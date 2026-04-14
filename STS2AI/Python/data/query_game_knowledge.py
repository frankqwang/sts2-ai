#!/usr/bin/env python3
"""Query the exported game knowledge catalog."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


DEFAULT_CATALOG_DIR = Path(__file__).resolve().parents[3] / "Assets" / "datasets" / "game_knowledge_catalog"
ENTITY_TABLES = ("cards", "relics", "potions", "monsters")


def _query_like(conn: sqlite3.Connection, table: str, query: str, limit: int) -> list[sqlite3.Row]:
    pattern = f"%{query}%"
    if table == "cards":
        sql = f"""
            SELECT id, title_en, title_zhs, payload_json
            FROM {table}
            WHERE id LIKE ?
               OR COALESCE(title_en, '') LIKE ?
               OR COALESCE(title_zhs, '') LIKE ?
               OR COALESCE(description_en, '') LIKE ?
               OR COALESCE(description_zhs, '') LIKE ?
               OR COALESCE(description_runtime_en, '') LIKE ?
               OR COALESCE(description_runtime_zhs, '') LIKE ?
               OR COALESCE(upgrade_preview_runtime_en, '') LIKE ?
               OR COALESCE(upgrade_preview_runtime_zhs, '') LIKE ?
            ORDER BY id
            LIMIT ?
        """
        params = (pattern, pattern, pattern, pattern, pattern, pattern, pattern, pattern, pattern, limit)
    elif table == "relics":
        sql = f"""
            SELECT id, title_en, title_zhs, payload_json
            FROM {table}
            WHERE id LIKE ?
               OR COALESCE(title_en, '') LIKE ?
               OR COALESCE(title_zhs, '') LIKE ?
               OR COALESCE(description_en, '') LIKE ?
               OR COALESCE(description_zhs, '') LIKE ?
               OR COALESCE(flavor_en, '') LIKE ?
               OR COALESCE(flavor_zhs, '') LIKE ?
            ORDER BY id
            LIMIT ?
        """
        params = (pattern, pattern, pattern, pattern, pattern, pattern, pattern, limit)
    elif table == "potions":
        sql = f"""
            SELECT id, title_en, title_zhs, payload_json
            FROM {table}
            WHERE id LIKE ?
               OR COALESCE(title_en, '') LIKE ?
               OR COALESCE(title_zhs, '') LIKE ?
               OR COALESCE(description_en, '') LIKE ?
               OR COALESCE(description_zhs, '') LIKE ?
            ORDER BY id
            LIMIT ?
        """
        params = (pattern, pattern, pattern, pattern, pattern, limit)
    else:
        sql = f"""
            SELECT id, title_en, title_zhs, payload_json
            FROM {table}
            WHERE id LIKE ?
               OR COALESCE(title_en, '') LIKE ?
               OR COALESCE(title_zhs, '') LIKE ?
               OR COALESCE(name_en, '') LIKE ?
               OR COALESCE(name_zhs, '') LIKE ?
               OR COALESCE(move_labels_json, '') LIKE ?
               OR COALESCE(moves_json, '') LIKE ?
            ORDER BY id
            LIMIT ?
        """
        params = (pattern, pattern, pattern, pattern, pattern, pattern, pattern, limit)
    return conn.execute(sql, params).fetchall()


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the exported game knowledge catalog.")
    parser.add_argument("query", help="要查询的关键字、id 或标题片段。")
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--entity", choices=ENTITY_TABLES + ("all",), default="all")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    db_path = args.catalog_dir / "game_knowledge_catalog.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = ENTITY_TABLES if args.entity == "all" else (args.entity,)
        result: dict[str, list[dict[str, object]]] = {}
        for table in tables:
            rows = _query_like(conn, table, args.query, args.limit)
            result[table] = [
                {
                    "id": row["id"],
                    "title_en": row["title_en"],
                    "title_zhs": row["title_zhs"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in rows
            ]
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
