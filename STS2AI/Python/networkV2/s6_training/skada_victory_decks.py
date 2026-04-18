"""从 skada 51K victory runs 的 final_deck 挖真实玩家 deck snapshot。

对比之前的 `real_boss_decks.py`(只用老 AI teacher 的 Artifacts/combat_teacher/),
本模块用 **真实 victory 玩家的 deck** — 多样性质量双赢:
  - 18K runs 有 final_deck(runs_victory 35.7% 覆盖)
  - 按 character × asc_bucket 分层(5×4 = 20 组,每组几百到上千 unique deck)
  - 含 final_relics(11-20 relics 是 act 3 末期,6-10 是 act 2,3-5 是 act 1 后段)

用法:
    from networkV2.s6_training.skada_victory_decks import load_skada_victory_decks
    decks = load_skada_victory_decks(character='IRONCLAD', min_asc=5, target_act=1)
    # → list[{deck: [...], relics: [...], character, asc, final_floor}]
    # cotrainer 随机抽用

缓存 sqlite → 只扫一次 jsonl。
"""
from __future__ import annotations

import json
import logging
import random
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)


_DEFAULT_INDEX_DB = Path(__file__).resolve().parents[2] / "data" / "skada" / "derived" / "skada_runs.sqlite"
_DEFAULT_CACHE_DB = Path(__file__).resolve().parents[2] / "data" / "skada" / "derived" / "skada_victory_decks.sqlite"


# act 通过 floor_reached 判断:
#   1-17 = act 1, 18-34 = act 2, 35-51 = act 3
def _floor_to_act(f: int) -> int:
    if f <= 17: return 1
    if f <= 34: return 2
    return 3


def _act_final_floor(act: int) -> int:
    return {1: 17, 2: 34, 3: 51}[act]


def build_cache(
    index_db: Path = _DEFAULT_INDEX_DB,
    cache_db: Path = _DEFAULT_CACHE_DB,
    *,
    require_map_acts: bool = False,
    min_deck_size: int = 8,
) -> dict:
    """扫 skada index 里所有 has_final_deck=1 的 runs,从 jsonl seek 提取 deck → cache。

    Output schema:
        decks(
            id          INTEGER PRIMARY KEY,
            character   TEXT,
            asc_bucket  TEXT,    -- low/mid/high/max
            ascension   INTEGER,
            act_reached INTEGER,
            floor_reached INTEGER,
            deck_json   TEXT,    -- list[str] lower snake card ids
            relics_json TEXT,
            deck_size   INTEGER,
            relic_count INTEGER,
            run_id      INTEGER
        )
    """
    from networkV2.s6_training.skada_index_dataset import SkadaIndexFetcher

    cache_db.parent.mkdir(parents=True, exist_ok=True)
    if cache_db.exists():
        cache_db.unlink()

    fetcher = SkadaIndexFetcher(index_db=index_db)

    # 查所有 has_final_deck=1 的 clean runs
    con = sqlite3.connect(str(index_db))
    con.row_factory = sqlite3.Row
    where = "is_clean=1 AND has_final_deck=1"
    if require_map_acts:
        where += " AND has_map_acts=1"
    rows = con.execute(
        f"SELECT run_id, character, ascension, floor_reached, asc_bucket, "
        f"       file_path, line_offset FROM runs WHERE {where}"
    ).fetchall()
    con.close()

    logger.info(f"found {len(rows)} runs with final_deck in index")

    out = sqlite3.connect(str(cache_db))
    out.execute("""
        CREATE TABLE decks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            character     TEXT NOT NULL,
            asc_bucket    TEXT NOT NULL,
            ascension     INTEGER,
            act_reached   INTEGER,
            floor_reached INTEGER,
            deck_json     TEXT NOT NULL,
            relics_json   TEXT,
            deck_size     INTEGER,
            relic_count   INTEGER,
            run_id        INTEGER
        )
    """)
    out.execute("CREATE INDEX idx_char_asc_act ON decks(character, asc_bucket, act_reached)")

    n_written = 0
    n_skipped_nodeck = 0
    n_skipped_smalldeck = 0
    for row in rows:
        # 直接用 SkadaIndexFetcher 的 row builder
        class _StubRow:
            def __init__(self, r):
                self.file_path = r["file_path"]
                self.line_offset = r["line_offset"]
                self.run_id = r["run_id"]
        try:
            rec = fetcher.fetch_record(_StubRow(row))
        except Exception as e:
            logger.debug(f"fetch failed run_id={row['run_id']}: {e}")
            continue

        # 提取 final_deck + final_relics
        fd = rec.get("final_deck") or []
        if not fd:
            n_skipped_nodeck += 1
            continue
        deck_ids: list[str] = []
        for c in fd:
            cid = str(c.get("card_id", "") or "").strip().lower()
            count = int(c.get("count", 1) or 1)
            if cid:
                deck_ids.extend([cid] * count)
        if len(deck_ids) < min_deck_size:
            n_skipped_smalldeck += 1
            continue

        fr = rec.get("final_relics") or []
        relic_ids = [str(r.get("relic_id", "") or "").strip().lower() for r in fr]
        relic_ids = [r for r in relic_ids if r]

        floor = int(row["floor_reached"] or 0)
        act = _floor_to_act(floor)

        out.execute(
            "INSERT INTO decks (character, asc_bucket, ascension, act_reached, "
            "floor_reached, deck_json, relics_json, deck_size, relic_count, run_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                row["character"],
                row["asc_bucket"],
                int(row["ascension"] or 0),
                act,
                floor,
                json.dumps(deck_ids, ensure_ascii=False),
                json.dumps(relic_ids, ensure_ascii=False),
                len(deck_ids),
                len(relic_ids),
                int(row["run_id"]),
            ),
        )
        n_written += 1
        if n_written % 1000 == 0:
            logger.info(f"  wrote {n_written} decks so far")

    # metadata
    from datetime import datetime, timezone
    out.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    out.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [
            ("source", "skada_runs_victory final_deck via index"),
            ("generated_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("n_written", str(n_written)),
            ("n_skipped_nodeck", str(n_skipped_nodeck)),
            ("n_skipped_smalldeck", str(n_skipped_smalldeck)),
        ],
    )
    out.commit()
    out.close()
    fetcher.close()

    return {
        "runs_with_final_deck_in_index": len(rows),
        "decks_written": n_written,
        "skipped_no_deck": n_skipped_nodeck,
        "skipped_small_deck": n_skipped_smalldeck,
        "cache_db": str(cache_db),
    }


def load_skada_victory_decks(
    cache_db: Path = _DEFAULT_CACHE_DB,
    *,
    character: str | None = None,
    asc_bucket: str | None = None,
    act_reached: int | None = None,
    min_deck_size: int = 8,
    max_decks: int | None = None,
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """从 cache 加载 deck 列表,按条件过滤。

    返回 list[{deck: [...], relics: [...], character, ascension, asc_bucket, floor_reached, act_reached}]
    ready for sim.reset(build=...)
    """
    if not cache_db.exists():
        logger.warning(f"skada_victory_decks cache not found at {cache_db}; run build_cache() first")
        return []

    con = sqlite3.connect(str(cache_db))
    con.row_factory = sqlite3.Row

    where = ["deck_size >= ?"]
    params: list[Any] = [int(min_deck_size)]
    if character:
        where.append("character = ?")
        params.append(character.upper())
    if asc_bucket:
        where.append("asc_bucket = ?")
        params.append(asc_bucket)
    if act_reached is not None:
        where.append("act_reached = ?")
        params.append(int(act_reached))

    q = "SELECT * FROM decks WHERE " + " AND ".join(where) + " ORDER BY id"
    if max_decks:
        q += f" LIMIT {int(max_decks)}"
    rows = con.execute(q, params).fetchall()
    con.close()

    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for r in rows:
        deck_raw = json.loads(r["deck_json"])
        relics_raw = json.loads(r["relics_json"] or "[]")
        if deduplicate:
            key = (tuple(sorted(Counter(deck_raw).items())), tuple(sorted(relics_raw)))
            if key in seen:
                continue
            seen.add(key)

        # sim 认 UPPER_SNAKE + 升级用 upgrade_level 字段,不认 lower/+suffix
        # skada 存 lower snake + "+" 后缀,这里转成 sim 格式
        deck_sim: list[dict[str, Any]] = []
        for cid in deck_raw:
            s = str(cid or "").strip().lower()
            upg = 0
            while s.endswith("+"):
                s = s[:-1]
                upg += 1
            if not s:
                continue
            deck_sim.append({"id": s.upper(), "upgrade_level": upg})

        relics_sim: list[dict[str, Any]] = []
        for rid in relics_raw:
            s = str(rid or "").strip().lower()
            if s:
                relics_sim.append({"id": s.upper()})

        out.append({
            "deck": deck_sim,
            "relics": relics_sim,
            "character": r["character"],
            "ascension": r["ascension"],
            "asc_bucket": r["asc_bucket"],
            "floor_reached": r["floor_reached"],
            "act_reached": r["act_reached"],
            "run_id": r["run_id"],
            # combat_cotrainer.build_chain_deck 期望字段
            "max_hp": 80,
            "current_hp": 80,
            "gold": 99,
            "max_energy": 3,
        })
    return out


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--index-db", type=Path, default=_DEFAULT_INDEX_DB)
    p.add_argument("--cache-db", type=Path, default=_DEFAULT_CACHE_DB)
    p.add_argument("--require-map-acts", action="store_true")
    args = p.parse_args()

    stats = build_cache(args.index_db, args.cache_db, require_map_acts=args.require_map_acts)
    print()
    print("=== skada_victory_decks cache build summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 展示每 character × act 分布
    con = sqlite3.connect(str(args.cache_db))
    print()
    print("character × act_reached 分布:")
    for r in con.execute(
        "SELECT character, act_reached, COUNT(*) c FROM decks "
        "GROUP BY character, act_reached ORDER BY character, act_reached"
    ):
        print(f"  {r[0]:<12} act{r[1]}: {r[2]}")


if __name__ == "__main__":
    main()
