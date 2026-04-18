#!/usr/bin/env python3
"""从 skada victory runs 挖路径先验,产出 priors.sqlite 供 loader 查表。

输入:`data/skada/runs_victory/details/*.jsonl`(~18K runs 带 map_acts)

两类先验(都不依赖 win_rate,因为数据全是 victory):

1. **frequency prior**:(character, asc_bucket, fingerprint) → 在 victory 玩家
   里出现频次归一化 [0,1]。高 = "这类路径是受欢迎的赢法"。
2. **efficiency prior**:同 key → avg_duration_sec(归一化)+ 避免平均 hp_taken
   (归一化)。低 duration / 低 hp_taken = "更高效的赢法"。

Fingerprint 设计(从 visited_coords 抽):
  - rest_count_discrete:   0 / 1 / 2 / 3+
  - elite_count_discrete:  0 / 1 / 2 / 3+
  - shop_count_discrete:   0 / 1 / 2+
  - treasure_count_discrete: 0 / 1 / 2+
  - length_bucket:         <13 / 13-15 / 16-17 / 18+

asc_bucket:  low(0-4) / mid(5-14) / high(15-20) / max(20+)

输出:sqlite 表 `path_priors`:
  (character, asc_bucket, fingerprint_key)
  → (freq, avg_duration_sec, avg_hp_taken_per_combat, n_samples)

用法:
  python -m networkV2.s6_training.build_path_priors \
      --dir data/skada/runs_victory/details \
      --output ../Artifacts/skada_path_priors.sqlite
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fingerprint 离散化
# ---------------------------------------------------------------------------

def _bucket_count(n: int, levels: tuple[int, ...]) -> int:
    """把 n 截到 levels 里的最大值(返回 bucket id)。"""
    for lv in levels:
        if n <= lv:
            return lv
    return levels[-1] + 1


def _asc_bucket(asc: int) -> str:
    if asc < 5: return "low"
    if asc < 15: return "mid"
    if asc < 20: return "high"
    return "max"


def _length_bucket(length: int) -> str:
    if length < 13: return "short"
    if length <= 15: return "med"
    if length <= 17: return "long"
    return "xlong"


@dataclass(frozen=True)
class Fingerprint:
    rest: int
    elite: int
    shop: int
    treasure: int
    length: str

    def key(self) -> str:
        return f"r{self.rest}_e{self.elite}_s{self.shop}_t{self.treasure}_{self.length}"


def _compute_fingerprint(visited_nodes_types: list[str]) -> Fingerprint:
    """从 visited 路径的 type 序列算 fingerprint。"""
    types = [t.upper() for t in visited_nodes_types]
    rest = types.count("R")
    elite = types.count("E")
    shop = types.count("S")
    treasure = types.count("T")
    length = len(types)
    return Fingerprint(
        rest=_bucket_count(rest, (0, 1, 2)),
        elite=_bucket_count(elite, (0, 1, 2)),
        shop=_bucket_count(shop, (0, 1)),
        treasure=_bucket_count(treasure, (0, 1)),
        length=_length_bucket(length),
    )


# ---------------------------------------------------------------------------
# 遍历 runs,抽 (act, visited_fingerprint, signals)
# ---------------------------------------------------------------------------

def iter_records(dir_path: Path) -> Iterator[dict]:
    """Iterate 所有 jsonl 文件的 records(跳过 newSample)。"""
    for jf in sorted(dir_path.glob("*.jsonl")):
        if jf.name == "newSample.jsonl":
            continue
        try:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"skip {jf}: {e}")


def extract_path_samples(rec: dict) -> Iterator[tuple[str, str, Fingerprint, dict[str, float]]]:
    """从一条 run 抽取 (character, asc_bucket, fingerprint, signals) 样本。

    每条 run 的每个 act 的完整 visited 路径作为一个样本。
    signals 含:
      - duration_sec:         本 run 总通关时间(act-level 拆不开,近似)
      - hp_taken_per_combat:  本 run 平均每战掉血
      - final_deck_size:      build 规模
      - final_relic_count:    遗物数
    """
    run = rec.get("run") or {}
    character = str(run.get("character", "")).upper()
    asc = int(run.get("ascension", 0) or 0)
    duration_sec = float(run.get("duration_sec", 0) or 0)

    map_acts = rec.get("map_acts") or []
    if not map_acts:
        return
    combats = rec.get("combats") or []
    n_combats = max(len(combats), 1)
    total_dmg_taken = sum(
        int(c.get("total_dmg_taken", 0) or 0) for c in combats
    )
    final_deck = rec.get("final_deck") or []
    final_relics = rec.get("final_relics") or []

    signals = {
        "duration_sec": duration_sec,
        "hp_taken_per_combat": total_dmg_taken / n_combats,
        "final_deck_size": float(sum(int(c.get("count", 1) or 1) for c in final_deck)),
        "final_relic_count": float(len(final_relics)),
    }

    asc_b = _asc_bucket(asc)
    for act in map_acts:
        visited = act.get("visited_coords") or []
        nodes = act.get("nodes") or []
        if not visited or not nodes:
            continue
        by_coord = {tuple(n.get("coord", [])): n for n in nodes if n.get("coord")}
        types = [
            str(by_coord.get(tuple(c), {}).get("type", "") or "")
            for c in visited
        ]
        fp = _compute_fingerprint([t for t in types if t])
        yield (character, asc_b, fp, signals)


# ---------------------------------------------------------------------------
# 聚合 + 写 sqlite
# ---------------------------------------------------------------------------

def build_priors(input_dir: Path, output_db: Path) -> dict:
    agg: dict[tuple[str, str, str], dict] = defaultdict(lambda: {
        "count": 0,
        "duration_sum": 0.0,
        "hp_taken_sum": 0.0,
        "deck_size_sum": 0.0,
        "relic_count_sum": 0.0,
    })

    per_group_totals: dict[tuple[str, str], int] = defaultdict(int)  # (char, asc) 总样本数,用于算 freq

    n_runs_scanned = 0
    n_samples = 0
    for rec in iter_records(input_dir):
        n_runs_scanned += 1
        for character, asc_b, fp, sig in extract_path_samples(rec):
            key = (character, asc_b, fp.key())
            rec_agg = agg[key]
            rec_agg["count"] += 1
            rec_agg["duration_sum"] += sig["duration_sec"]
            rec_agg["hp_taken_sum"] += sig["hp_taken_per_combat"]
            rec_agg["deck_size_sum"] += sig["final_deck_size"]
            rec_agg["relic_count_sum"] += sig["final_relic_count"]
            per_group_totals[(character, asc_b)] += 1
            n_samples += 1

        if n_runs_scanned % 5000 == 0:
            logger.info(f"scanned {n_runs_scanned} runs, {n_samples} samples, {len(agg)} unique fingerprints")

    logger.info(f"done scanning: {n_runs_scanned} runs, {n_samples} samples, {len(agg)} unique keys")

    # 写 sqlite
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    con = sqlite3.connect(str(output_db))
    con.execute("""
        CREATE TABLE path_priors (
            character       TEXT NOT NULL,
            asc_bucket      TEXT NOT NULL,
            fingerprint_key TEXT NOT NULL,
            n_samples       INTEGER NOT NULL,
            freq            REAL NOT NULL,      -- count / group_total(仅按 character+asc_bucket 归一化)
            avg_duration_sec REAL NOT NULL,
            avg_hp_taken     REAL NOT NULL,
            avg_deck_size    REAL NOT NULL,
            avg_relic_count  REAL NOT NULL,
            PRIMARY KEY (character, asc_bucket, fingerprint_key)
        )
    """)
    # 汇总表:每个 (character, asc_bucket) 组的全局均值 / 总量,loader 查询时用作归一化参考
    con.execute("""
        CREATE TABLE group_stats (
            character       TEXT NOT NULL,
            asc_bucket      TEXT NOT NULL,
            group_total     INTEGER NOT NULL,
            max_duration    REAL NOT NULL,
            min_duration    REAL NOT NULL,
            mean_duration   REAL NOT NULL,
            PRIMARY KEY (character, asc_bucket)
        )
    """)

    rows = []
    for (character, asc_b, fp_key), s in agg.items():
        total = per_group_totals[(character, asc_b)]
        rows.append((
            character, asc_b, fp_key,
            s["count"],
            s["count"] / max(total, 1),
            s["duration_sum"] / max(s["count"], 1),
            s["hp_taken_sum"] / max(s["count"], 1),
            s["deck_size_sum"] / max(s["count"], 1),
            s["relic_count_sum"] / max(s["count"], 1),
        ))
    con.executemany(
        "INSERT INTO path_priors VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )

    # group stats(用于归一化 duration)
    group_agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (character, asc_b, fp_key), s in agg.items():
        for _ in range(s["count"]):
            pass
        if s["count"] > 0:
            group_agg[(character, asc_b)].append(s["duration_sum"] / s["count"])

    group_rows = []
    for (character, asc_b), durations in group_agg.items():
        total = per_group_totals[(character, asc_b)]
        group_rows.append((
            character, asc_b, total,
            max(durations), min(durations),
            sum(durations) / len(durations),
        ))
    con.executemany(
        "INSERT INTO group_stats VALUES (?,?,?,?,?,?)",
        group_rows,
    )

    # metadata
    con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [
            ("source", "skada_runs_victory"),
            ("n_runs_scanned", str(n_runs_scanned)),
            ("n_samples", str(n_samples)),
            ("n_unique_fingerprints", str(len(agg))),
            ("schema_version", "1"),
        ],
    )

    con.commit()
    con.close()

    return {
        "runs_scanned": n_runs_scanned,
        "samples": n_samples,
        "unique_fingerprints": len(agg),
        "groups": len(per_group_totals),
        "output_db": str(output_db),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path,
                   default=Path("data/skada/runs_victory/details"),
                   help="skada victory jsonl 目录")
    p.add_argument("--output", type=Path,
                   default=Path("../Artifacts/skada_path_priors.sqlite"),
                   help="输出 sqlite(覆盖写)")
    args = p.parse_args()

    stats = build_priors(args.dir, args.output)
    print()
    print("=== Path priors build summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
