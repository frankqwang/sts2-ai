#!/usr/bin/env python3
"""导出权威游戏数据到 `STS2AI/data/game_wiki/game_catalog.sqlite`。

设计目标：
1. runtime 权威层：直接从 HeadlessSim `game_catalog` RPC 拉卡牌/遗物/药水/怪物/遭遇/power。
2. source 权威层：直接扫描仓库 `src/Core/Models/**`，补角色模型、starter build、源码文件来源等结构。
3. authority sqlite：把两层数据写进一份 sqlite，后续查询统一以这份库为准。

当前不做的事情：
- 不在 Python 里硬编码角色、starter deck、starter relic、怪物/power 名单。
- 不假装 source 侧能 100% 解析所有运行时语义；解析不到的字段保留为空，并在 metadata 里标注来源。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


STS2AI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STS2AI_ROOT.parent
PYTHON_ROOT = STS2AI_ROOT / "Python"
BRIDGE_ROOT = STS2AI_ROOT / "bridge"
GAME_WIKI_ROOT = STS2AI_ROOT / "data" / "game_wiki"
DEFAULT_OUTPUT = GAME_WIKI_ROOT / "game_catalog.sqlite"
SOURCE_ROOT = REPO_ROOT / "src" / "Core" / "Models"
DEFAULT_SIM_HOST_CANDIDATES = (
    STS2AI_ROOT / "ENV" / "Sim" / "HeadlessSim" / "bin" / "Release" / "net9.0" / "HeadlessSim.exe",
    STS2AI_ROOT / "ENV" / "Sim" / "HeadlessSim" / "bin" / "Debug" / "net9.0" / "HeadlessSim.exe",
)
DEFAULT_PORT = 15527
SIM_READY_TIMEOUT = 30.0

if str(STS2AI_ROOT) not in sys.path:
    sys.path.insert(0, str(STS2AI_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))


RUNTIME_SCHEMA_SQL = """
DROP TABLE IF EXISTS cards;
DROP TABLE IF EXISTS relics;
DROP TABLE IF EXISTS potions;
DROP TABLE IF EXISTS monsters;
DROP TABLE IF EXISTS encounters;
DROP TABLE IF EXISTS powers;
DROP TABLE IF EXISTS characters;
DROP TABLE IF EXISTS source_models;
DROP TABLE IF EXISTS metadata;
DROP TABLE IF EXISTS skada_cards;
DROP TABLE IF EXISTS skada_relics;
DROP TABLE IF EXISTS skada_potions;

CREATE TABLE cards (
    id            TEXT PRIMARY KEY,
    class_name    TEXT,
    card_type     TEXT,
    rarity        TEXT,
    target_type   TEXT,
    base_cost     INTEGER,
    is_x_cost     INTEGER,
    gains_block   INTEGER,
    tags_json     TEXT,
    keywords_json TEXT,
    source_path   TEXT,
    payload_json  TEXT
);

CREATE TABLE relics (
    id           TEXT PRIMARY KEY,
    class_name   TEXT,
    rarity       TEXT,
    tags_json    TEXT,
    source_path  TEXT,
    payload_json TEXT
);

CREATE TABLE potions (
    id           TEXT PRIMARY KEY,
    class_name   TEXT,
    rarity       TEXT,
    source_path  TEXT,
    payload_json TEXT
);

CREATE TABLE monsters (
    id           TEXT PRIMARY KEY,
    class_name   TEXT,
    powers_json  TEXT,
    source_path  TEXT,
    payload_json TEXT
);

CREATE TABLE encounters (
    id                TEXT PRIMARY KEY,
    room_type         TEXT,
    act_index         INTEGER,
    monster_ids_json  TEXT,
    payload_json      TEXT
);

CREATE TABLE powers (
    class_name         TEXT PRIMARY KEY,
    power_id           TEXT,
    base_classes_json  TEXT,
    is_debuff_hint     INTEGER,
    source_path        TEXT,
    payload_json       TEXT
);

CREATE TABLE characters (
    id                         TEXT PRIMARY KEY,
    class_name                 TEXT,
    source_path                TEXT,
    starting_hp                INTEGER,
    starting_gold              INTEGER,
    max_energy                 INTEGER,
    card_pool_class            TEXT,
    relic_pool_class           TEXT,
    potion_pool_class          TEXT,
    unlocks_after_character_id TEXT,
    starting_deck_json         TEXT,
    starting_relics_json       TEXT,
    starting_potions_json      TEXT,
    payload_json               TEXT
);

CREATE TABLE source_models (
    kind        TEXT NOT NULL,
    class_name  TEXT NOT NULL,
    model_id    TEXT,
    base_class  TEXT,
    source_path TEXT,
    PRIMARY KEY (kind, class_name)
);

CREATE TABLE metadata (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

CREATE TABLE skada_cards (
    card_id      TEXT PRIMARY KEY,
    character    TEXT,
    card_type    TEXT,
    rarity       TEXT,
    base_cost    INTEGER,
    is_upgraded  INTEGER
);

CREATE TABLE skada_relics (
    relic_id  TEXT PRIMARY KEY,
    rarity    TEXT
);

CREATE TABLE skada_potions (
    potion_id TEXT PRIMARY KEY,
    rarity    TEXT
);

CREATE INDEX idx_cards_type_rarity ON cards(card_type, rarity);
CREATE INDEX idx_relics_rarity ON relics(rarity);
CREATE INDEX idx_encounters_room ON encounters(room_type);
CREATE INDEX idx_skada_cards_character ON skada_cards(character);
CREATE INDEX idx_source_models_kind ON source_models(kind);
"""


@dataclass(slots=True)
class SourceModelRecord:
    kind: str
    class_name: str
    model_id: str
    base_class: str
    source_path: str


@dataclass(slots=True)
class CharacterSourceRecord:
    character_id: str
    class_name: str
    source_path: str
    starting_hp: int
    starting_gold: int
    max_energy: int
    card_pool_class: str
    relic_pool_class: str
    potion_pool_class: str
    unlocks_after_character_id: str
    starting_deck: list[str]
    starting_relics: list[str]
    starting_potions: list[str]


def _resolve_sim_host(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"sim host not found: {path}")
    for candidate in DEFAULT_SIM_HOST_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("no HeadlessSim.exe found")


def _spawn_sim(exe: Path, port: int) -> subprocess.Popen:
    cmd = [str(exe), "--port", str(port), "--protocol", "json"]
    logger.info("spawning sim: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )


def _connect_client(port: int, timeout_s: float = SIM_READY_TIMEOUT):
    from game_bridge.transport.codec import JsonCodec
    from game_bridge.transport.connection import PipeConnection, PipeConnectionConfig

    class JsonClient:
        def __init__(self) -> None:
            self._conn = PipeConnection(
                PipeConnectionConfig(
                    port=port,
                    protocol="json",
                    codec=JsonCodec(),
                    connect_timeout_s=3.0,
                )
            )

        def connect(self, timeout_s: float = 3.0) -> None:
            self._conn.cfg.connect_timeout_s = float(timeout_s)
            self._conn.connect()

        def close(self) -> None:
            self._conn.close()

        def call(self, method: str, params: dict[str, Any] | None = None, timeout_s: float | None = None):
            return self._conn.safe_call(method, params, timeout_s=timeout_s)

    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client = JsonClient()
            client.connect(timeout_s=3.0)
            return client
        except Exception as exc:  # pragma: no cover - depends on local sim process timing
            last_err = exc
            time.sleep(0.3)
    raise ConnectionError(f"cannot connect to sim on port {port} within {timeout_s}s: {last_err}")


def _class_name_to_snake(name: str) -> str:
    if not name:
        return ""
    out = []
    for index, ch in enumerate(name):
        if ch.isupper() and index > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _snake_to_upper(name: str) -> str:
    return str(name or "").upper()


def _extract_single_line(text: str, field_name: str, *, default: str = "") -> str:
    pattern = rf"public override .* {re.escape(field_name)} => (?P<value>[^;]+);"
    match = re.search(pattern, text)
    if not match:
        return default
    return match.group("value").strip()


def _extract_int_expr(text: str, field_name: str, *, default: int = 0) -> int:
    raw = _extract_single_line(text, field_name, default="")
    match = re.search(r"(-?\d+)", raw)
    return int(match.group(1)) if match else default


def _extract_generic_type(text: str, field_name: str, type_name: str) -> str:
    pattern = rf"public override {re.escape(type_name)} {re.escape(field_name)} => ModelDb\.\w+<(?P<type>\w+)>\(\);"
    match = re.search(pattern, text)
    return match.group("type") if match else ""


def _extract_modeldb_list(text: str, accessor: str) -> list[str]:
    pattern = rf"ModelDb\.{re.escape(accessor)}<(?P<type>\w+)>\(\)"
    return [match.group("type") for match in re.finditer(pattern, text)]


def _extract_unlock_character(text: str) -> str:
    match = re.search(r"UnlocksAfterRunAs => ModelDb\.Character<(?P<type>\w+)>\(\);", text)
    if not match:
        return ""
    return match.group("type").upper()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_source_model_files(source_root: Path) -> Iterable[tuple[str, Path]]:
    mapping = {
        "card": source_root / "Cards",
        "relic": source_root / "Relics",
        "monster": source_root / "Monsters",
        "power": source_root / "Powers",
        "character": source_root / "Characters",
    }
    for kind, root in mapping.items():
        if not root.exists():
            continue
        for path in sorted(root.glob("*.cs")):
            yield kind, path


def _scan_source_models(
    *,
    source_root: Path,
    runtime_payload: dict[str, Any],
) -> tuple[list[SourceModelRecord], list[CharacterSourceRecord]]:
    card_ids_by_class = {
        str(card.get("class_name") or ""): _snake_to_upper(str(card.get("card_id") or ""))
        for card in runtime_payload.get("cards") or []
        if card.get("class_name") and card.get("card_id")
    }
    relic_ids_by_class = {
        str(relic.get("class_name") or ""): _snake_to_upper(str(relic.get("relic_id") or ""))
        for relic in runtime_payload.get("relics") or []
        if relic.get("class_name") and relic.get("relic_id")
    }
    potion_ids_by_class = {
        str(potion.get("class_name") or ""): _snake_to_upper(str(potion.get("potion_id") or ""))
        for potion in runtime_payload.get("potions") or []
        if potion.get("class_name") and potion.get("potion_id")
    }
    power_ids_by_class = {
        str(power.get("class_name") or ""): _snake_to_upper(str(power.get("class_name") or ""))
        for power in runtime_payload.get("powers") or []
        if power.get("class_name")
    }
    monster_ids_by_class = {
        str(monster.get("class_name") or ""): _class_name_to_snake(str(monster.get("class_name") or "")).upper()
        for monster in runtime_payload.get("monsters") or []
        if monster.get("class_name")
    }

    source_models: list[SourceModelRecord] = []
    characters: list[CharacterSourceRecord] = []
    for kind, path in _iter_source_model_files(source_root):
        text = _read_text(path)
        match = re.search(r"public\s+(?:sealed\s+|abstract\s+)?class\s+(?P<class>\w+)\s*:\s*(?P<base>\w+)", text)
        if not match:
            continue
        class_name = match.group("class")
        base_class = match.group("base")
        source_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        if kind == "card":
            model_id = card_ids_by_class.get(class_name, "")
        elif kind == "relic":
            model_id = relic_ids_by_class.get(class_name, "")
        elif kind == "monster":
            model_id = monster_ids_by_class.get(class_name, "")
        elif kind == "power":
            model_id = power_ids_by_class.get(class_name, "")
        else:
            model_id = class_name.upper()
        source_models.append(
            SourceModelRecord(
                kind=kind,
                class_name=class_name,
                model_id=model_id,
                base_class=base_class,
                source_path=source_path,
            )
        )

        if kind != "character":
            continue
        starting_deck = [card_ids_by_class.get(name, name.upper()) for name in _extract_modeldb_list(text, "Card")]
        starting_relics = [relic_ids_by_class.get(name, name.upper()) for name in _extract_modeldb_list(text, "Relic")]
        starting_potions = [potion_ids_by_class.get(name, name.upper()) for name in _extract_modeldb_list(text, "Potion")]
        characters.append(
            CharacterSourceRecord(
                character_id=class_name.upper(),
                class_name=class_name,
                source_path=source_path,
                starting_hp=_extract_int_expr(text, "StartingHp"),
                starting_gold=_extract_int_expr(text, "StartingGold", default=99),
                max_energy=_extract_int_expr(text, "MaxEnergy", default=3),
                card_pool_class=_extract_generic_type(text, "CardPool", "CardPoolModel"),
                relic_pool_class=_extract_generic_type(text, "RelicPool", "RelicPoolModel"),
                potion_pool_class=_extract_generic_type(text, "PotionPool", "PotionPoolModel"),
                unlocks_after_character_id=_extract_unlock_character(text),
                starting_deck=starting_deck,
                starting_relics=starting_relics,
                starting_potions=starting_potions,
            )
        )

    return source_models, characters


def _skada_character(runtime_card_id: str, character_ids: Iterable[str]) -> str:
    low = str(runtime_card_id or "").lower()
    normalized_ids = [item.lower() for item in character_ids if item]
    for character_id in normalized_ids:
        if low.endswith("_" + character_id) or low == character_id:
            return character_id.upper()
    tokens = re.split(r"[^a-z0-9]+", low)
    if tokens and tokens[-1] in normalized_ids:
        return tokens[-1].upper()
    return ""


def _write_db(
    *,
    runtime_payload: dict[str, Any],
    source_models: list[SourceModelRecord],
    source_characters: list[CharacterSourceRecord],
    out_path: Path,
    source_meta: dict[str, str],
) -> dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    character_ids = [record.character_id for record in source_characters]
    source_path_by_kind_class = {(record.kind, record.class_name): record.source_path for record in source_models}
    power_id_by_class = {(record.kind, record.class_name): record.model_id for record in source_models if record.kind == "power"}

    con = sqlite3.connect(str(out_path))
    try:
        con.executescript(RUNTIME_SCHEMA_SQL)
        stats: dict[str, int] = {}

        cards = runtime_payload.get("cards") or []
        card_rows = []
        skada_card_rows = []
        for card in cards:
            card_id = str(card.get("card_id") or "").lower()
            class_name = str(card.get("class_name") or "")
            if not card_id:
                continue
            source_path = source_path_by_kind_class.get(("card", class_name), "")
            card_rows.append(
                (
                    card_id,
                    class_name,
                    str(card.get("card_type") or "").lower(),
                    str(card.get("rarity") or "").lower(),
                    str(card.get("target_type") or "").lower(),
                    int(card.get("base_cost", 0) or 0),
                    1 if card.get("is_x_cost") else 0,
                    1 if card.get("gains_block") else 0,
                    json.dumps(card.get("tags") or [], ensure_ascii=False),
                    json.dumps(card.get("keywords") or [], ensure_ascii=False),
                    source_path,
                    json.dumps(card, ensure_ascii=False),
                )
            )
            skada_card_rows.append(
                (
                    _snake_to_upper(card_id),
                    _skada_character(card_id, character_ids),
                    str(card.get("card_type") or "").lower(),
                    str(card.get("rarity") or "").lower(),
                    int(card.get("base_cost", 0) or 0),
                    1 if str(card_id).endswith("+") else 0,
                )
            )
        con.executemany(
            "INSERT INTO cards (id,class_name,card_type,rarity,target_type,base_cost,is_x_cost,gains_block,tags_json,keywords_json,source_path,payload_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            card_rows,
        )
        con.executemany(
            "INSERT INTO skada_cards (card_id,character,card_type,rarity,base_cost,is_upgraded) VALUES (?,?,?,?,?,?)",
            skada_card_rows,
        )
        stats["cards"] = len(card_rows)

        relic_rows = []
        skada_relic_rows = []
        for relic in runtime_payload.get("relics") or []:
            relic_id = str(relic.get("relic_id") or "").lower()
            class_name = str(relic.get("class_name") or "")
            if not relic_id:
                continue
            relic_rows.append(
                (
                    relic_id,
                    class_name,
                    str(relic.get("rarity") or "").lower(),
                    json.dumps(relic.get("tags") or [], ensure_ascii=False),
                    source_path_by_kind_class.get(("relic", class_name), ""),
                    json.dumps(relic, ensure_ascii=False),
                )
            )
            skada_relic_rows.append((_snake_to_upper(relic_id), str(relic.get("rarity") or "").lower()))
        con.executemany(
            "INSERT INTO relics (id,class_name,rarity,tags_json,source_path,payload_json) VALUES (?,?,?,?,?,?)",
            relic_rows,
        )
        con.executemany("INSERT INTO skada_relics (relic_id,rarity) VALUES (?,?)", skada_relic_rows)
        stats["relics"] = len(relic_rows)

        potion_rows = []
        skada_potion_rows = []
        for potion in runtime_payload.get("potions") or []:
            potion_id = str(potion.get("potion_id") or "").lower()
            class_name = str(potion.get("class_name") or "")
            if not potion_id:
                continue
            potion_rows.append(
                (
                    potion_id,
                    class_name,
                    str(potion.get("rarity") or "").lower(),
                    source_path_by_kind_class.get(("potion", class_name), ""),
                    json.dumps(potion, ensure_ascii=False),
                )
            )
            skada_potion_rows.append((_snake_to_upper(potion_id), str(potion.get("rarity") or "").lower()))
        con.executemany(
            "INSERT INTO potions (id,class_name,rarity,source_path,payload_json) VALUES (?,?,?,?,?)",
            potion_rows,
        )
        con.executemany("INSERT INTO skada_potions (potion_id,rarity) VALUES (?,?)", skada_potion_rows)
        stats["potions"] = len(potion_rows)

        monster_rows = []
        for monster in runtime_payload.get("monsters") or []:
            class_name = str(monster.get("class_name") or "")
            monster_id = _class_name_to_snake(class_name)
            if not monster_id:
                continue
            monster_rows.append(
                (
                    monster_id,
                    class_name,
                    json.dumps(monster.get("powers") or [], ensure_ascii=False),
                    source_path_by_kind_class.get(("monster", class_name), ""),
                    json.dumps(monster, ensure_ascii=False),
                )
            )
        con.executemany(
            "INSERT INTO monsters (id,class_name,powers_json,source_path,payload_json) VALUES (?,?,?,?,?)",
            monster_rows,
        )
        stats["monsters"] = len(monster_rows)

        encounter_rows = []
        for encounter in runtime_payload.get("encounters") or []:
            encounter_id = str(encounter.get("encounter_id") or "").lower()
            if not encounter_id:
                continue
            encounter_rows.append(
                (
                    encounter_id,
                    str(encounter.get("room_type") or "").lower(),
                    int(encounter.get("act_index", -1) or -1),
                    json.dumps(encounter.get("monster_ids") or [], ensure_ascii=False),
                    json.dumps(encounter, ensure_ascii=False),
                )
            )
        con.executemany(
            "INSERT INTO encounters (id,room_type,act_index,monster_ids_json,payload_json) VALUES (?,?,?,?,?)",
            encounter_rows,
        )
        stats["encounters"] = len(encounter_rows)

        power_rows = []
        for power in runtime_payload.get("powers") or []:
            class_name = str(power.get("class_name") or "")
            if not class_name:
                continue
            power_rows.append(
                (
                    class_name,
                    power_id_by_class.get(("power", class_name), _snake_to_upper(class_name)),
                    json.dumps(power.get("base_classes") or [], ensure_ascii=False),
                    1 if power.get("is_debuff_hint") else 0,
                    source_path_by_kind_class.get(("power", class_name), ""),
                    json.dumps(power, ensure_ascii=False),
                )
            )
        con.executemany(
            "INSERT INTO powers (class_name,power_id,base_classes_json,is_debuff_hint,source_path,payload_json) VALUES (?,?,?,?,?,?)",
            power_rows,
        )
        stats["powers"] = len(power_rows)

        character_rows = []
        for character in source_characters:
            character_rows.append(
                (
                    character.character_id,
                    character.class_name,
                    character.source_path,
                    character.starting_hp,
                    character.starting_gold,
                    character.max_energy,
                    character.card_pool_class,
                    character.relic_pool_class,
                    character.potion_pool_class,
                    character.unlocks_after_character_id,
                    json.dumps(character.starting_deck, ensure_ascii=False),
                    json.dumps(character.starting_relics, ensure_ascii=False),
                    json.dumps(character.starting_potions, ensure_ascii=False),
                    json.dumps(
                        {
                            "character_id": character.character_id,
                            "class_name": character.class_name,
                            "starting_hp": character.starting_hp,
                            "starting_gold": character.starting_gold,
                            "max_energy": character.max_energy,
                        },
                        ensure_ascii=False,
                    ),
                )
            )
        con.executemany(
            "INSERT INTO characters (id,class_name,source_path,starting_hp,starting_gold,max_energy,card_pool_class,relic_pool_class,potion_pool_class,unlocks_after_character_id,starting_deck_json,starting_relics_json,starting_potions_json,payload_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            character_rows,
        )
        stats["characters"] = len(character_rows)

        source_rows = [
            (record.kind, record.class_name, record.model_id, record.base_class, record.source_path)
            for record in source_models
        ]
        con.executemany(
            "INSERT INTO source_models (kind,class_name,model_id,base_class,source_path) VALUES (?,?,?,?,?)",
            source_rows,
        )
        stats["source_models"] = len(source_rows)

        metadata_rows = [
            ("schema_version", "3"),
            ("authority_db", "game_wiki"),
            ("authority_source_runtime", "game_catalog_rpc"),
            ("authority_source_code", "src/Core/Models"),
            ("generated_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("runtime_cards", str(stats["cards"])),
            ("runtime_relics", str(stats["relics"])),
            ("runtime_potions", str(stats["potions"])),
            ("runtime_monsters", str(stats["monsters"])),
            ("runtime_encounters", str(stats["encounters"])),
            ("runtime_powers", str(stats["powers"])),
            ("source_models", str(stats["source_models"])),
            ("source_characters", str(stats["characters"])),
            ("sim_build_git_sha", source_meta.get("build_git_sha", "") or ""),
            ("sim_schema_id", source_meta.get("schema_id", "") or ""),
            ("sim_protocol_version", str(source_meta.get("protocol_version", "") or "")),
            ("source_root", str(SOURCE_ROOT)),
        ]
        con.executemany("INSERT INTO metadata (key,value) VALUES (?,?)", metadata_rows)
        stats["metadata"] = len(metadata_rows)

        con.commit()
        return stats
    finally:
        con.close()


def export_catalog(
    *,
    output: Path,
    sim_host: Path | None = None,
    port: int = DEFAULT_PORT,
    spawn_sim: bool = True,
    keep_sim: bool = False,
    source_root: Path = SOURCE_ROOT,
) -> dict[str, int]:
    proc: subprocess.Popen | None = None
    client = None
    try:
        if spawn_sim:
            proc = _spawn_sim(_resolve_sim_host(str(sim_host) if sim_host else None), port)

        logger.info("connecting to sim on port %s ...", port)
        client = _connect_client(port=port)
        logger.info("calling game_catalog RPC ...")
        payload = client.call("game_catalog", params={})
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected payload type: {type(payload).__name__}")

        logger.info(
            "runtime catalog received: cards=%d relics=%d potions=%d monsters=%d encounters=%d powers=%d",
            len(payload.get("cards") or []),
            len(payload.get("relics") or []),
            len(payload.get("potions") or []),
            len(payload.get("monsters") or []),
            len(payload.get("encounters") or []),
            len(payload.get("powers") or []),
        )

        logger.info("scanning source models from %s ...", source_root)
        source_models, source_characters = _scan_source_models(source_root=source_root, runtime_payload=payload)
        logger.info(
            "source scan complete: source_models=%d source_characters=%d",
            len(source_models),
            len(source_characters),
        )

        source_meta = {
            "protocol_version": getattr(client, "_protocol_version", None),
            "build_git_sha": getattr(client, "_server_build_git_sha", None),
            "schema_id": getattr(client, "_server_schema_id", None),
        }
        stats = _write_db(
            runtime_payload=payload,
            source_models=source_models,
            source_characters=source_characters,
            out_path=output,
            source_meta=source_meta,
        )
        logger.info("authority db written to %s", output)
        return stats
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if proc is not None and not keep_sim:
            logger.info("terminating sim host ...")
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as exc:  # pragma: no cover - best effort process cleanup
                logger.warning("sim terminate failed: %s", exc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出 runtime + source 双来源权威游戏库到 game_wiki sqlite")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sim-host", type=str, default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-spawn-sim", dest="spawn_sim", action="store_false", default=True)
    parser.add_argument("--keep-sim", action="store_true")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        stats = export_catalog(
            output=args.output,
            sim_host=Path(args.sim_host) if args.sim_host else None,
            port=args.port,
            spawn_sim=args.spawn_sim,
            keep_sim=args.keep_sim,
            source_root=args.source_root,
        )
    except Exception as exc:
        logger.error("export failed: %s: %s", type(exc).__name__, exc)
        return 1

    print("\n=== export summary ===")
    for key, value in stats.items():
        print(f"  {key:<15} {value}")
    print(f"  output         {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
