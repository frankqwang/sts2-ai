#!/usr/bin/env python3
"""扫 skada 原始 jsonl,建 index sqlite(不存 payload)。

原始 jsonl 太大(4.7 GB / 51K runs),全部 copy 进 sqlite 无意义。
改用索引式:
  - jsonl 原文留在磁盘
  - sqlite 存每条 run 的 (file_path, line_offset, meta 字段, 清洗 flag)
  - 训练时按 run_id 查 index 得到 (file, offset) → seek → 只读那一条 record

Schema:
  runs:
    run_id            INT PRIMARY KEY
    character         TEXT
    ascension         INTEGER
    is_victory        INTEGER 0/1
    game_version      TEXT
    file_path         TEXT     -- 相对 repo root 的 jsonl path
    line_offset       INTEGER  -- byte offset 便于 seek
    line_number       INTEGER  -- 0-indexed(备用)
    floor_reached     INTEGER
    duration_sec      INTEGER
    has_map_acts      INTEGER 0/1
    has_final_deck    INTEGER 0/1
    has_combats       INTEGER 0/1
    n_card_choices    INTEGER
    n_relic_choices   INTEGER
    n_campfire        INTEGER
    n_shop            INTEGER
    asc_bucket        TEXT     -- low / mid / high / max
    is_clean          INTEGER 0/1   -- 经过清洗规则过滤后是否保留

  metadata:
    key / value(schema_version / generated_at / repo_root / ...)

用法:
  python -m networkV2.s6_training.build_skada_index \
      --dir data/skada/runs_victory/details \
      --output data/skada/derived/skada_runs.sqlite

清洗规则(实现在 _is_clean 里):
  - character ∈ KNOWN_CHARACTERS (5 个 STS2 角色)
  - ascension >= 0
  - duration_sec >= 60(排异常中止)
  - floor_reached > 0
  - has_floor_timeline(index 里已是全 true,隐含条件)
  - game_version 不是 v0.98.x(过老格式可能差异)
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_KNOWN_CHARACTERS = {"IRONCLAD", "REGENT", "DEFECT", "SILENT", "NECROBINDER"}
_MIN_VERSION_PARTS = (0, 99, 0)   # 严格低于 0.99.0 的 version 丢弃


def _parse_version(v: str) -> tuple[int, int, int]:
    s = str(v or "").lower().lstrip("v")
    parts = s.split(".")
    try:
        return tuple(int(p) for p in parts[:3]) + (0,) * (3 - len(parts[:3]))
    except Exception:
        return (0, 0, 0)


def _asc_bucket(asc: int) -> str:
    if asc < 5: return "low"
    if asc < 15: return "mid"
    if asc < 20: return "high"
    return "max"


def _is_clean(
    character: str,
    ascension: int,
    duration_sec: int,
    floor_reached: int,
    version: str,
    has_floor_timeline: bool,
) -> bool:
    if character not in _KNOWN_CHARACTERS:
        return False
    if ascension < 0:
        return False
    if duration_sec < 60:
        return False
    if floor_reached <= 0:
        return False
    if not has_floor_timeline:
        return False
    if _parse_version(version) < _MIN_VERSION_PARTS:
        return False
    return True


def build_index(input_dir: Path, output_db: Path, repo_root: Path) -> dict:
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()

    con = sqlite3.connect(str(output_db))
    con.execute("""
        CREATE TABLE runs (
            run_id           INTEGER PRIMARY KEY,
            character        TEXT,
            ascension        INTEGER,
            is_victory       INTEGER,
            game_version     TEXT,
            file_path        TEXT NOT NULL,
            line_offset      INTEGER NOT NULL,
            line_number      INTEGER NOT NULL,
            floor_reached    INTEGER,
            duration_sec     INTEGER,
            has_map_acts     INTEGER,
            has_final_deck   INTEGER,
            has_combats      INTEGER,
            n_card_choices   INTEGER,
            n_relic_choices  INTEGER,
            n_campfire       INTEGER,
            n_shop           INTEGER,
            asc_bucket       TEXT,
            is_clean         INTEGER
        )
    """)
    con.execute("CREATE INDEX idx_clean ON runs(is_clean, character, asc_bucket)")
    con.execute("CREATE INDEX idx_character_asc ON runs(character, ascension)")
    con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")

    total = clean_cnt = 0
    rows: list[tuple] = []
    for jf in sorted(input_dir.glob("*.jsonl")):
        if jf.name == "newSample.jsonl":
            continue
        rel_path = str(jf.relative_to(repo_root)).replace("\\", "/")
        line_no = 0
        with jf.open("rb") as f:  # 二进制读取拿真实 byte offset
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    line_no += 1
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    line_no += 1
                    continue
                run = rec.get("run") or {}
                rid = int(run.get("run_id", 0) or 0)
                if rid == 0:
                    line_no += 1
                    continue
                character = str(run.get("character", "")).upper()
                ascension = int(run.get("ascension", 0) or 0)
                is_vic = 1 if run.get("is_victory") else 0
                ver = str(run.get("game_version", "") or "")
                floor_reached = int(run.get("floor_reached", 0) or 0)
                duration_sec = int(run.get("duration_sec", 0) or 0)
                ft = rec.get("floor_timeline") or []
                has_ft = bool(ft)
                has_map_acts = bool(rec.get("map_acts"))
                has_final_deck = bool(rec.get("final_deck"))
                has_combats = bool(rec.get("combats"))
                n_card_choices = sum(1 for f_d in ft if f_d.get("card_choices"))
                n_relic_choices = sum(1 for f_d in ft if f_d.get("relic_choices"))
                n_campfire = sum(1 for f_d in ft if f_d.get("campfire_choice"))
                n_shop = sum(1 for f_d in ft if f_d.get("shop_actions"))
                clean = int(_is_clean(
                    character, ascension, duration_sec, floor_reached, ver, has_ft,
                ))
                clean_cnt += clean
                rows.append((
                    rid, character, ascension, is_vic, ver,
                    rel_path, offset, line_no,
                    floor_reached, duration_sec,
                    int(has_map_acts), int(has_final_deck), int(has_combats),
                    n_card_choices, n_relic_choices, n_campfire, n_shop,
                    _asc_bucket(ascension), clean,
                ))
                total += 1
                if total % 5000 == 0:
                    logger.info(f"indexed {total} runs ({clean_cnt} clean)")
                line_no += 1

    # 去重(file 可能跑过多次,保留最后一条)
    con.executemany(
        "INSERT OR REPLACE INTO runs VALUES (" + ",".join("?" * 19) + ")",
        rows,
    )

    # metadata
    from datetime import datetime, timezone
    con.executemany(
        "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
        [
            ("schema_version", "1"),
            ("generated_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("repo_root", str(repo_root)),
            ("input_dir", str(input_dir.relative_to(repo_root)).replace("\\", "/")),
            ("total_runs", str(total)),
            ("clean_runs", str(clean_cnt)),
        ],
    )
    con.commit()
    con.close()

    logger.info(f"done: {total} runs indexed, {clean_cnt} clean")
    return {
        "total_runs": total,
        "clean_runs": clean_cnt,
        "output_db": str(output_db),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path,
                   default=Path("data/skada/runs_victory/details"),
                   help="skada victory jsonl 目录(相对 cwd)")
    p.add_argument("--output", type=Path,
                   default=Path("data/skada/derived/skada_runs.sqlite"),
                   help="输出 sqlite 索引(覆盖写)")
    p.add_argument("--repo-root", type=Path, default=None,
                   help="repo 根目录,用于生成相对 file_path(默认 cwd 的上 2 级)")
    args = p.parse_args()

    repo_root = args.repo_root or Path.cwd().parents[1]  # 假设 cwd=STS2AI/Python
    input_dir = args.dir if args.dir.is_absolute() else Path.cwd() / args.dir

    stats = build_index(input_dir, args.output, repo_root)
    print()
    print("=== Index build summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
