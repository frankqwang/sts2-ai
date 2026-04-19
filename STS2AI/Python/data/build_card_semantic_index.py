#!/usr/bin/env python3
"""直接扫描源码卡牌定义，生成单源 card semantic index(jsonl + sqlite)。

设计约束：
  - 不依赖 `source_knowledge.sqlite`
  - 复杂记录落 `cards.jsonl`
  - sqlite 只保留训练/推理常用的轻量索引字段
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from constants import GAME_SEMANTIC_CARDS_JSONL, GAME_SEMANTIC_INDEX_DB, REPO_ROOT
from core.card_tags import FUNCTIONAL_TAG_TO_IDX, extract_tags_from_card
from core.vocab import _slugify


logger = logging.getLogger(__name__)

_BASE_RE = re.compile(
    r":\s*base\(\s*(-?\d+)\s*,\s*CardType\.(\w+)\s*,\s*CardRarity\.(\w+)\s*,\s*TargetType\.(\w+)\s*\)"
)
_KEYWORD_RE = re.compile(r"CardKeyword\.(\w+)")
_TAG_RE = re.compile(r"CardTag\.(\w+)")

_SCHEMA_SQL = """
DROP TABLE IF EXISTS cards;
DROP TABLE IF EXISTS metadata;

CREATE TABLE cards (
    id                    TEXT PRIMARY KEY,
    class_name            TEXT NOT NULL,
    card_type             TEXT NOT NULL,
    rarity                TEXT NOT NULL,
    target_type           TEXT NOT NULL,
    base_cost             INTEGER NOT NULL,
    is_x_cost             INTEGER NOT NULL,
    gains_block           INTEGER NOT NULL,
    source_tags_json      TEXT NOT NULL,
    source_keywords_json  TEXT NOT NULL,
    functional_tags_json  TEXT NOT NULL,
    source_path           TEXT NOT NULL,
    jsonl_path            TEXT NOT NULL,
    jsonl_line            INTEGER NOT NULL
);

CREATE TABLE metadata (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE INDEX idx_cards_type_rarity ON cards(card_type, rarity);
CREATE INDEX idx_cards_target_type ON cards(target_type);
"""


def _cards_root() -> Path:
    root = REPO_ROOT / "src" / "Core" / "Models" / "Cards"
    if not root.exists():
        raise FileNotFoundError(f"card source root not found: {root}")
    return root


def _normalize_card_id(name: str) -> str:
    return _slugify(name).lower()


def _parse_ctor(text: str) -> tuple[int, str, str, str]:
    m = _BASE_RE.search(text)
    if not m:
        raise ValueError("cannot parse card base(...) constructor")
    base_cost = int(m.group(1))
    card_type = m.group(2).lower()
    rarity = m.group(3).lower()
    target_type = m.group(4).lower()
    return base_cost, card_type, rarity, target_type


def _parse_keywords(text: str) -> list[str]:
    return sorted({m.group(1).lower() for m in _KEYWORD_RE.finditer(text)})


def _parse_tags(text: str) -> list[str]:
    return sorted({m.group(1).lower() for m in _TAG_RE.finditer(text)})


def _record_from_file(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8-sig")
    base_cost, card_type, rarity, target_type = _parse_ctor(text)
    class_name = path.stem
    card_id = _normalize_card_id(class_name)
    functional_tags = sorted(
        t for t in extract_tags_from_card(path)
        if t in FUNCTIONAL_TAG_TO_IDX
    )
    return {
        "id": card_id,
        "class_name": class_name,
        "card_type": card_type,
        "rarity": rarity,
        "target_type": target_type,
        "base_cost": base_cost,
        "is_x_cost": int("x_cost" in functional_tags or base_cost < 0),
        "gains_block": int("block" in functional_tags),
        "source_tags": _parse_tags(text),
        "source_keywords": _parse_keywords(text),
        "functional_tags": functional_tags,
        "source_path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def _build_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(_cards_root().rglob("*.cs")):
        try:
            records.append(_record_from_file(path))
        except Exception as exc:
            logger.warning(f"skip card semantic parse failed for {path}: {exc}")
    return records


def build_card_semantic_index(
    *,
    db_path: Path = GAME_SEMANTIC_INDEX_DB,
    cards_jsonl_path: Path = GAME_SEMANTIC_CARDS_JSONL,
) -> dict[str, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cards_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    records = _build_records()
    if not records:
        raise RuntimeError("no card semantic records parsed from source")

    with cards_jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(_SCHEMA_SQL)
        try:
            rel_jsonl = str(cards_jsonl_path.relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:
            rel_jsonl = str(cards_jsonl_path.resolve()).replace("\\", "/")
        rows = []
        for idx, rec in enumerate(records, start=1):
            rows.append((
                rec["id"],
                rec["class_name"],
                rec["card_type"],
                rec["rarity"],
                rec["target_type"],
                int(rec["base_cost"]),
                int(rec["is_x_cost"]),
                int(rec["gains_block"]),
                json.dumps(rec["source_tags"], ensure_ascii=False),
                json.dumps(rec["source_keywords"], ensure_ascii=False),
                json.dumps(rec["functional_tags"], ensure_ascii=False),
                rec["source_path"],
                rel_jsonl,
                idx,
            ))
        con.executemany(
            "INSERT INTO cards "
            "(id,class_name,card_type,rarity,target_type,base_cost,is_x_cost,gains_block,"
            " source_tags_json,source_keywords_json,functional_tags_json,source_path,jsonl_path,jsonl_line)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        meta_rows = [
            ("schema_version", "1"),
            ("source", "src_card_model_scan"),
            ("generated_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("cards_count", str(len(records))),
        ]
        con.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            meta_rows,
        )
        con.commit()
    finally:
        con.close()
    return {"cards": len(records)}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="从 src/Core/Models/Cards 生成 card semantic index")
    p.add_argument("--db-path", type=Path, default=GAME_SEMANTIC_INDEX_DB)
    p.add_argument("--cards-jsonl-path", type=Path, default=GAME_SEMANTIC_CARDS_JSONL)
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args()
    stats = build_card_semantic_index(
        db_path=args.db_path,
        cards_jsonl_path=args.cards_jsonl_path,
    )
    logger.info(f"built card semantic index: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
