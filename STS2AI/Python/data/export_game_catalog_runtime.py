#!/usr/bin/env python3
"""从 headless sim runtime API 导出完整游戏 catalog 到 sqlite。

取代历史上基于 json 静态文件的 export_game_knowledge_catalog.py / build_source_database.py。
完全运行时驱动:启动 headless_sim_host → ProtoPipeClient 连接 → call("game_catalog") → 写 sqlite。

产出的 sqlite 有两层 schema:

A. 原始 runtime 层(保留 RPC 完整 payload,不做字段重命名)
   - `cards`    : id / class_name / card_type / rarity / target_type / base_cost / is_x_cost / gains_block / tags_json / keywords_json / payload_json
   - `relics`   : id / class_name / rarity / tags_json / payload_json
   - `potions`  : id / class_name / rarity / payload_json
   - `monsters` : id / class_name / powers_json / payload_json
   - `encounters`: id / room_type / act_index / monster_ids_json / payload_json
   - `powers`   : class_name / base_classes_json / is_debuff_hint / payload_json
   - `metadata` : key / value(生成时间 / 源版本 / schema_version 等)

B. Skada 兼容层(字段名/id 格式和 skada_analytics.sqlite 对齐,方便 join + 离线训练)
   - `skada_cards`   : card_id(UPPER_SNAKE) / character / card_type / rarity / base_cost / is_upgraded
   - `skada_relics`  : relic_id / rarity
   - `skada_potions` : potion_id / rarity

用法:
    # 自动起 sim、导出、关 sim(推荐)
    python -m data.export_game_catalog_runtime

    # 已有 sim 运行(自己起的)
    python -m data.export_game_catalog_runtime --no-spawn-sim --port 15527

    # 指定输出位置 / sim host exe
    python -m data.export_game_catalog_runtime \
        --output data/source_knowledge.sqlite \
        --sim-host ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

_PYTHON_ROOT = Path(__file__).resolve().parents[1]           # .../STS2AI/Python
_STS2AI_ROOT = _PYTHON_ROOT.parent                            # .../STS2AI
_DEFAULT_OUTPUT = _PYTHON_ROOT / "data" / "source_knowledge.sqlite"
_DEFAULT_SIM_HOST_CANDIDATES = (
    _STS2AI_ROOT / "ENV" / "Sim" / "HeadlessSim" / "bin" / "Release" / "net9.0" / "HeadlessSim.exe",
    _STS2AI_ROOT / "ENV" / "Sim" / "HeadlessSim" / "bin" / "Debug"   / "net9.0" / "HeadlessSim.exe",
)
_DEFAULT_PORT = 15527      # sim host 默认端口;pipe 名格式 "sts2_mcts_<port>"
_SIM_READY_TIMEOUT = 30.0  # 启动 sim 后 connect 的最大等待

# 保证 data/* 作为 module 运行时可 import networkV2 / env
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))


# ---------------------------------------------------------------------------
# Skada 兼容:character / is_upgraded 推导
# ---------------------------------------------------------------------------

# STS2 的 5 个角色。如果 card_id 末尾匹配到这些,character 字段取对应值。
_CHARACTERS = ("ironclad", "regent", "defect", "silent", "necrobinder")


def _skada_card_id(runtime_id: str) -> str:
    """runtime lower_snake → skada UPPER_SNAKE。"""
    return str(runtime_id or "").upper()


def _skada_character(runtime_card_id: str) -> str:
    """从 card_id 后缀推 character。跟 skada_analytics.cards.character 对齐(UPPER)。

    例:strike_ironclad → IRONCLAD / cleave → "" / ascenders_bane → ""
    """
    low = str(runtime_card_id or "").lower()
    for ch in _CHARACTERS:
        if low.endswith("_" + ch) or low.endswith(ch):
            if low == ch or low.endswith("_" + ch):
                return ch.upper()
    # 更严格:tail token 匹配
    tokens = re.split(r"[^a-z0-9]+", low)
    if tokens and tokens[-1] in _CHARACTERS:
        return tokens[-1].upper()
    return ""


def _is_upgraded(runtime_id: str) -> int:
    """skada 的 card_id 会用 '+' 后缀标识升级,runtime 一般不含。留 0。"""
    return 1 if str(runtime_id or "").endswith("+") else 0


# ---------------------------------------------------------------------------
# Sim 起进程 + 连接(复用 env/headless_sim_runner,协议固定 bin 以兼容 BinaryPipeClient)
# ---------------------------------------------------------------------------

_DEFAULT_PROTOCOL = "bin"    # bin 协议 + BinaryPipeClient 最稳定通用


def _resolve_sim_host(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"sim host not found: {p}")
    for c in _DEFAULT_SIM_HOST_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "no HeadlessSim.exe found. "
        "Tried: " + " ; ".join(str(c) for c in _DEFAULT_SIM_HOST_CANDIDATES)
    )


def _spawn_sim_via_runner(exe: Path, port: int, protocol: str = _DEFAULT_PROTOCOL) -> subprocess.Popen:
    """直接 Popen sim host,绕过 headless_sim_runner 的 fresh 校验。

    理由:本脚本用途是"读取游戏 catalog",用户更新游戏 + rebuild exe 后自然会跑;
    如果只 rebuild 了某个 .cs 而没 rebuild exe,runner 会强制中断 → 不适合这个命令。
    """
    cmd = [str(exe), "--port", str(port), "--protocol", protocol]
    logger.info(f"spawning sim: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )
    return proc


def _connect_client(port: int, protocol: str = _DEFAULT_PROTOCOL, timeout_s: float = _SIM_READY_TIMEOUT):
    """用匹配协议的 client 连上 sim,支持 bin / proto / json。"""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None

    if protocol == "bin":
        from env.binary_pipe_client import BinaryPipeClient
        client_cls = BinaryPipeClient
        client_kwargs: dict[str, Any] = {"port": port}
    elif protocol == "proto":
        from networkV2.s0_bridge.proto_pipe_client import ProtoPipeClient
        client_cls = ProtoPipeClient
        client_kwargs = {"port": port, "pipe_name": f"sts2_mcts_proto_{port}"}
    else:
        raise ValueError(f"protocol {protocol!r} not supported by export pipeline (use bin or proto)")

    while time.monotonic() < deadline:
        try:
            c = client_cls(**client_kwargs)
            c.connect(timeout_s=3.0)
            return c
        except Exception as e:
            last_err = e
            # protocol version 不匹配 = sim exe 过旧,不用再轮询,立刻报错提示 rebuild
            if "protocol version mismatch" in str(e).lower():
                raise ConnectionError(
                    f"sim protocol version mismatch: {e}\n"
                    "  → HeadlessSim.exe 是旧版,请 rebuild:\n"
                    "    dotnet build -c Release STS2AI/ENV/Sim/HeadlessSim/HeadlessSim.csproj\n"
                    "  rebuild 后重跑本脚本即可。"
                ) from e
            time.sleep(0.3)
    raise ConnectionError(
        f"cannot connect to sim on port {port}({protocol}) within {timeout_s}s: {last_err}"
    )


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
DROP TABLE IF EXISTS cards;
DROP TABLE IF EXISTS relics;
DROP TABLE IF EXISTS potions;
DROP TABLE IF EXISTS monsters;
DROP TABLE IF EXISTS encounters;
DROP TABLE IF EXISTS powers;
DROP TABLE IF EXISTS metadata;
DROP TABLE IF EXISTS skada_cards;
DROP TABLE IF EXISTS skada_relics;
DROP TABLE IF EXISTS skada_potions;

-- ======== A. Runtime 原始层 ========

CREATE TABLE cards (
    id            TEXT PRIMARY KEY,   -- lower_snake (match 旧 source_knowledge.cards.id)
    class_name    TEXT,               -- PascalCase
    card_type     TEXT,               -- attack / skill / power (lower)
    rarity        TEXT,               -- basic / common / uncommon / rare / special / curse / status (lower)
    target_type   TEXT,
    base_cost     INTEGER,
    is_x_cost     INTEGER,            -- 0/1
    gains_block   INTEGER,            -- 0/1
    tags_json     TEXT,               -- JSON array[str]
    keywords_json TEXT,               -- JSON array[str]
    payload_json  TEXT                -- 原 RPC 完整 dict(保证未来字段扩展不丢)
);

CREATE TABLE relics (
    id           TEXT PRIMARY KEY,    -- lower_snake
    class_name   TEXT,
    rarity       TEXT,                -- starter / common / uncommon / rare / ancient / ...
    tags_json    TEXT,
    payload_json TEXT
);

CREATE TABLE potions (
    id           TEXT PRIMARY KEY,
    class_name   TEXT,
    rarity       TEXT,
    payload_json TEXT
);

CREATE TABLE monsters (
    id           TEXT PRIMARY KEY,    -- lower_snake 从 class_name 转
    class_name   TEXT,                -- PascalCase
    powers_json  TEXT,
    payload_json TEXT
);

CREATE TABLE encounters (
    id                TEXT PRIMARY KEY,   -- lower (如 doormaker_boss)
    room_type         TEXT,
    act_index         INTEGER,            -- 0/1/2/... (-1 未分配)
    monster_ids_json  TEXT,
    payload_json      TEXT
);

CREATE TABLE powers (
    class_name         TEXT PRIMARY KEY,
    base_classes_json  TEXT,              -- 继承链 ["DirectParent", ..., "PowerModel"]
    is_debuff_hint     INTEGER,           -- 0/1(启发式判断)
    payload_json       TEXT
);

CREATE TABLE metadata (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

-- ======== B. Skada 兼容层(字段名 / id 格式对齐 skada_analytics.sqlite)========

CREATE TABLE skada_cards (
    card_id      TEXT PRIMARY KEY,   -- UPPER_SNAKE(match skada.cards.card_id)
    character    TEXT,               -- UPPER 角色名,空串 = 非角色专属
    card_type    TEXT,
    rarity       TEXT,
    base_cost    INTEGER,
    is_upgraded  INTEGER             -- runtime 一般 0(skada 数据才有 +)
);

CREATE TABLE skada_relics (
    relic_id  TEXT PRIMARY KEY,   -- UPPER_SNAKE
    rarity    TEXT
);

CREATE TABLE skada_potions (
    potion_id TEXT PRIMARY KEY,
    rarity    TEXT
);

-- 索引:下游查询最常用的路径
CREATE INDEX idx_cards_type_rarity ON cards(card_type, rarity);
CREATE INDEX idx_relics_rarity ON relics(rarity);
CREATE INDEX idx_encounters_room ON encounters(room_type);
CREATE INDEX idx_skada_cards_character ON skada_cards(character);
"""


def _class_name_to_snake(name: str) -> str:
    """PascalCase → lower_snake_case(monster id 规范化)。"""
    if not name:
        return ""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


# ---------------------------------------------------------------------------
# 写 sqlite
# ---------------------------------------------------------------------------

def _write_db(payload: dict[str, Any], out_path: Path, source_meta: dict[str, str]) -> dict[str, int]:
    """把 game_catalog payload 写进 sqlite。返回每表行数统计。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # 全量重建,避免 schema 遗留
    con = sqlite3.connect(str(out_path))
    try:
        con.executescript(_SCHEMA_SQL)

        stats: dict[str, int] = {}

        # ---- cards ----
        cards = payload.get("cards") or []
        card_rows = []
        skada_card_rows = []
        for c in cards:
            cid = str(c.get("card_id", "") or "").lower()
            if not cid:
                continue
            card_rows.append((
                cid,
                c.get("class_name", "") or "",
                str(c.get("card_type", "") or "").lower(),
                str(c.get("rarity", "") or "").lower(),
                str(c.get("target_type", "") or "").lower(),
                int(c.get("base_cost", 0) or 0),
                1 if c.get("is_x_cost") else 0,
                1 if c.get("gains_block") else 0,
                json.dumps(c.get("tags") or [], ensure_ascii=False),
                json.dumps(c.get("keywords") or [], ensure_ascii=False),
                json.dumps(c, ensure_ascii=False),
            ))
            skada_card_rows.append((
                _skada_card_id(cid),
                _skada_character(cid),
                str(c.get("card_type", "") or "").lower(),
                str(c.get("rarity", "") or "").lower(),
                int(c.get("base_cost", 0) or 0),
                _is_upgraded(cid),
            ))
        con.executemany(
            "INSERT OR REPLACE INTO cards "
            "(id,class_name,card_type,rarity,target_type,base_cost,is_x_cost,gains_block,"
            " tags_json,keywords_json,payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            card_rows,
        )
        con.executemany(
            "INSERT OR REPLACE INTO skada_cards "
            "(card_id,character,card_type,rarity,base_cost,is_upgraded) VALUES (?,?,?,?,?,?)",
            skada_card_rows,
        )
        stats["cards"] = len(card_rows)
        stats["skada_cards"] = len(skada_card_rows)

        # ---- relics ----
        relics = payload.get("relics") or []
        relic_rows, skada_relic_rows = [], []
        for r in relics:
            rid = str(r.get("relic_id", "") or "").lower()
            if not rid:
                continue
            relic_rows.append((
                rid,
                r.get("class_name", "") or "",
                str(r.get("rarity", "") or "").lower(),
                json.dumps(r.get("tags") or [], ensure_ascii=False),
                json.dumps(r, ensure_ascii=False),
            ))
            skada_relic_rows.append((
                rid.upper(),
                str(r.get("rarity", "") or "").lower(),
            ))
        con.executemany(
            "INSERT OR REPLACE INTO relics (id,class_name,rarity,tags_json,payload_json) VALUES (?,?,?,?,?)",
            relic_rows,
        )
        con.executemany(
            "INSERT OR REPLACE INTO skada_relics (relic_id,rarity) VALUES (?,?)",
            skada_relic_rows,
        )
        stats["relics"] = len(relic_rows)

        # ---- potions ----
        potions = payload.get("potions") or []
        potion_rows, skada_potion_rows = [], []
        for p in potions:
            pid = str(p.get("potion_id", "") or "").lower()
            if not pid:
                continue
            potion_rows.append((
                pid,
                p.get("class_name", "") or "",
                str(p.get("rarity", "") or "").lower(),
                json.dumps(p, ensure_ascii=False),
            ))
            skada_potion_rows.append((pid.upper(), str(p.get("rarity", "") or "").lower()))
        con.executemany(
            "INSERT OR REPLACE INTO potions (id,class_name,rarity,payload_json) VALUES (?,?,?,?)",
            potion_rows,
        )
        con.executemany(
            "INSERT OR REPLACE INTO skada_potions (potion_id,rarity) VALUES (?,?)",
            skada_potion_rows,
        )
        stats["potions"] = len(potion_rows)

        # ---- monsters ----
        monsters = payload.get("monsters") or []
        monster_rows = []
        for m in monsters:
            raw_id = str(m.get("monster_id", "") or "")
            class_name = str(m.get("class_name", "") or raw_id)
            # runtime 返回 monster_id 已是 PascalCase;我们再生成一份 lower_snake 主键
            snake = _class_name_to_snake(class_name) or raw_id.lower()
            monster_rows.append((
                snake,
                class_name,
                json.dumps(m.get("powers") or [], ensure_ascii=False),
                json.dumps(m, ensure_ascii=False),
            ))
        con.executemany(
            "INSERT OR REPLACE INTO monsters (id,class_name,powers_json,payload_json) VALUES (?,?,?,?)",
            monster_rows,
        )
        stats["monsters"] = len(monster_rows)

        # ---- encounters ----
        encounters = payload.get("encounters") or []
        enc_rows = []
        for e in encounters:
            eid = str(e.get("encounter_id", "") or "").lower()
            if not eid:
                continue
            enc_rows.append((
                eid,
                str(e.get("room_type", "") or "").lower(),
                int(e.get("act_index", -1) or -1),
                json.dumps(e.get("monster_ids") or [], ensure_ascii=False),
                json.dumps(e, ensure_ascii=False),
            ))
        con.executemany(
            "INSERT OR REPLACE INTO encounters (id,room_type,act_index,monster_ids_json,payload_json) VALUES (?,?,?,?,?)",
            enc_rows,
        )
        stats["encounters"] = len(enc_rows)

        # ---- powers ----
        powers = payload.get("powers") or []
        power_rows = []
        for pw in powers:
            cls = str(pw.get("class_name", "") or "")
            if not cls:
                continue
            power_rows.append((
                cls,
                json.dumps(pw.get("base_classes") or [], ensure_ascii=False),
                1 if pw.get("is_debuff_hint") else 0,
                json.dumps(pw, ensure_ascii=False),
            ))
        con.executemany(
            "INSERT OR REPLACE INTO powers (class_name,base_classes_json,is_debuff_hint,payload_json) VALUES (?,?,?,?)",
            power_rows,
        )
        stats["powers"] = len(power_rows)

        # ---- metadata ----
        meta_rows = [
            ("schema_version", "2"),
            ("source", "runtime_game_catalog_rpc"),
            ("generated_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("sim_build_git_sha", source_meta.get("build_git_sha", "") or ""),
            ("sim_schema_id", source_meta.get("schema_id", "") or ""),
            ("sim_protocol_version", str(source_meta.get("protocol_version", "") or "")),
        ]
        con.executemany(
            "INSERT OR REPLACE INTO metadata (key,value) VALUES (?,?)",
            meta_rows,
        )
        stats["metadata"] = len(meta_rows)

        con.commit()
        return stats
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def export_catalog(
    output: Path,
    sim_host: Path | None = None,
    port: int = _DEFAULT_PORT,
    spawn_sim: bool = True,
    keep_sim: bool = False,
    protocol: str = _DEFAULT_PROTOCOL,
) -> dict[str, int]:
    """主入口:起 sim(可选)→ call game_catalog → 写 sqlite。返回行数统计。"""
    proc: subprocess.Popen | None = None
    client = None
    try:
        if spawn_sim:
            exe = _resolve_sim_host(str(sim_host) if sim_host else None)
            proc = _spawn_sim_via_runner(exe, port=port, protocol=protocol)

        logger.info(f"connecting to sim on port {port} (protocol={protocol}) ...")
        client = _connect_client(port=port, protocol=protocol, timeout_s=_SIM_READY_TIMEOUT)
        source_meta = {
            "protocol_version": getattr(client, "_protocol_version", None),
            "build_git_sha": getattr(client, "_server_build_git_sha", None),
            "schema_id": getattr(client, "_server_schema_id", None),
        }
        logger.info(
            f"connected. sim_build_sha={source_meta['build_git_sha']} "
            f"schema={source_meta['schema_id']} proto_ver={source_meta['protocol_version']}"
        )

        logger.info("calling game_catalog RPC ...")
        payload = client.call("game_catalog", params={})
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected payload type: {type(payload).__name__}")
        logger.info(
            "received: cards=%d relics=%d potions=%d monsters=%d encounters=%d powers=%d",
            len(payload.get("cards") or []),
            len(payload.get("relics") or []),
            len(payload.get("potions") or []),
            len(payload.get("monsters") or []),
            len(payload.get("encounters") or []),
            len(payload.get("powers") or []),
        )

        logger.info(f"writing to {output} ...")
        stats = _write_db(payload, output, source_meta)
        logger.info(f"done. rows: {stats}")
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
            except Exception as e:
                logger.warning(f"sim terminate failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="从运行时 game_catalog RPC 导出 sqlite(不依赖 json 老文件)")
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                   help=f"输出 sqlite 路径(默认 {_DEFAULT_OUTPUT})")
    p.add_argument("--sim-host", type=str, default=None,
                   help="HeadlessSim.exe 路径(默认自动找 Release/Debug)")
    p.add_argument("--port", type=int, default=_DEFAULT_PORT,
                   help=f"sim pipe port(默认 {_DEFAULT_PORT})")
    p.add_argument("--no-spawn-sim", dest="spawn_sim", action="store_false", default=True,
                   help="不自动起 sim(假设外部已起好,用 --port 指向)")
    p.add_argument("--keep-sim", action="store_true",
                   help="导出完成后不 kill sim(调试用)")
    p.add_argument("--protocol", type=str, default=_DEFAULT_PROTOCOL, choices=["bin", "proto"],
                   help=f"sim 协议(默认 {_DEFAULT_PROTOCOL}),bin = BinaryPipeClient,proto = ProtoPipeClient")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        stats = export_catalog(
            output=Path(args.output),
            sim_host=Path(args.sim_host) if args.sim_host else None,
            port=args.port,
            spawn_sim=args.spawn_sim,
            keep_sim=args.keep_sim,
            protocol=args.protocol,
        )
    except Exception as e:
        logger.error(f"export failed: {type(e).__name__}: {e}")
        return 1
    print("\n=== export summary ===")
    for k, v in stats.items():
        print(f"  {k:<15} {v}")
    print(f"  output         {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
