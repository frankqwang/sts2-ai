"""基于 skada_runs.sqlite 索引 + 原始 jsonl 的惰性数据集。

用法模式:
  1. 训练启动时:SkadaIndexFetcher(index_db, priors_db) — 连好两个 sqlite
  2. 抽样 runs:fetcher.sample_clean_runs(n=2000, balance_character=True, ...)
  3. 对每个 run_id:fetcher.fetch_record(run_id) → dict(从 jsonl seek 读一条)
  4. record → load_run_samples(带 priors 查询)→ TrainingSample 列表

避免把 4.7 GB payload 全 copy 进 sqlite。index 只存元信息(file + offset),
按需 seek 读原始 jsonl。训练主循环只关心 "一批 TrainingSample"。

索引 schema 见 build_skada_index.py;priors schema 见 build_path_priors.py。
"""
from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator


logger = logging.getLogger(__name__)


@dataclass
class RunIndexRow:
    run_id: int
    character: str
    ascension: int
    is_victory: int
    game_version: str
    file_path: str      # 相对 repo_root
    line_offset: int    # 字节偏移
    line_number: int
    floor_reached: int
    duration_sec: int
    has_map_acts: int
    has_final_deck: int
    has_combats: int
    n_card_choices: int
    n_relic_choices: int
    n_campfire: int
    n_shop: int
    asc_bucket: str
    is_clean: int


class SkadaIndexFetcher:
    """Index + 原始 jsonl seek 读取器;同时查 path priors。"""

    def __init__(
        self,
        index_db: Path,
        priors_db: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.index_db = Path(index_db)
        self.priors_db = Path(priors_db) if priors_db else None
        # repo_root 用于解析索引里的相对 file_path
        if repo_root is None:
            # 从 index 的 metadata 读
            con = sqlite3.connect(str(self.index_db))
            try:
                row = con.execute(
                    "SELECT value FROM metadata WHERE key='repo_root'"
                ).fetchone()
                repo_root = Path(row[0]) if row else Path.cwd()
            finally:
                con.close()
        self.repo_root = Path(repo_root)

        # 永久连 priors 常读
        self._priors_con: sqlite3.Connection | None = None
        if self.priors_db and self.priors_db.exists():
            self._priors_con = sqlite3.connect(str(self.priors_db))
            self._priors_con.row_factory = sqlite3.Row

        # 打开的 file handle 缓存(每个文件只 open 一次)
        self._file_handles: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 查 runs(按清洗规则 + 分层采样)
    # ------------------------------------------------------------------

    def sample_clean_runs(
        self,
        n: int = 2000,
        *,
        character: str | None = None,
        asc_bucket: str | None = None,
        require_map_acts: bool = False,
        seed: int = 0,
        version_prefix: str | None = "v0.103.2",
    ) -> list[RunIndexRow]:
        """从 clean runs 里抽 n 条。支持按 character / asc_bucket / 是否需要 map_acts 过滤。

        version_prefix: 只抽 game_version 以此为前缀的 run(LIKE prefix%)。
        默认 `v0.103.2` — 和当前 sim 版本精确对齐。skada 混了多个 patch
        (v0.99 / v0.102 / v0.103.0 / .1 / .2),非 .2 版本的 encounter balance 和
        card property 可能被 rebalance,replay 时 deck 合法性 / 胜率分布会偏。
        传 None 不过滤版本。
        """
        con = sqlite3.connect(str(self.index_db))
        con.row_factory = sqlite3.Row
        try:
            where = ["is_clean=1"]
            params: list = []
            if character:
                where.append("character = ?")
                params.append(character.upper())
            if asc_bucket:
                where.append("asc_bucket = ?")
                params.append(asc_bucket)
            if require_map_acts:
                where.append("has_map_acts = 1")
            if version_prefix:
                where.append("game_version LIKE ?")
                params.append(f"{version_prefix}%")
            q = (
                "SELECT * FROM runs "
                "WHERE " + " AND ".join(where) +
                " ORDER BY RANDOM() LIMIT ?"
            )
            rows = con.execute(q, params + [int(n)]).fetchall()
            return [RunIndexRow(**dict(r)) for r in rows]
        finally:
            con.close()

    def sample_balanced(
        self,
        n_per_group: int = 500,
        *,
        groups: Iterable[tuple[str, str]] | None = None,
        require_map_acts: bool = False,
        seed: int = 0,
    ) -> list[RunIndexRow]:
        """按 (character, asc_bucket) 分层采样,每层 n_per_group 条。

        groups 不给时自动用全部 5×4 = 20 组。
        """
        random.seed(seed)
        if groups is None:
            chars = ["IRONCLAD", "REGENT", "SILENT", "DEFECT", "NECROBINDER"]
            bkts = ["low", "mid", "high", "max"]
            groups = [(c, b) for c in chars for b in bkts]
        all_rows: list[RunIndexRow] = []
        for c, b in groups:
            rows = self.sample_clean_runs(
                n=n_per_group, character=c, asc_bucket=b,
                require_map_acts=require_map_acts,
            )
            all_rows.extend(rows)
        random.shuffle(all_rows)
        return all_rows

    # ------------------------------------------------------------------
    # Seek 读 jsonl 单行 record
    # ------------------------------------------------------------------

    def _get_handle(self, file_rel_path: str):
        """打开 jsonl 文件 handle,缓存复用。"""
        h = self._file_handles.get(file_rel_path)
        if h is not None:
            return h
        full = self.repo_root / file_rel_path
        if not full.exists():
            raise FileNotFoundError(f"jsonl not found: {full}")
        h = full.open("rb")
        self._file_handles[file_rel_path] = h
        return h

    def fetch_record(self, row: RunIndexRow) -> dict:
        """按索引里的 (file_path, line_offset) seek 读一条 run record。"""
        h = self._get_handle(row.file_path)
        h.seek(row.line_offset)
        line = h.readline()
        if not line:
            raise RuntimeError(
                f"empty line at run_id={row.run_id} {row.file_path}:{row.line_offset}"
            )
        return json.loads(line)

    def fetch_records(self, rows: Iterable[RunIndexRow]) -> Iterator[dict]:
        """批量拉取(按文件顺序化以减少 file-handle 切换,若行乱序也 OK)。"""
        for r in rows:
            try:
                yield self.fetch_record(r)
            except Exception as e:
                logger.warning(f"failed to fetch run_id={r.run_id}: {e}")

    def close(self) -> None:
        for h in self._file_handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._file_handles.clear()
        if self._priors_con is not None:
            self._priors_con.close()
            self._priors_con = None

    # ------------------------------------------------------------------
    # Path priors 查询(build_path_priors.py 产出的表)
    # ------------------------------------------------------------------

    @lru_cache(maxsize=4096)
    def lookup_path_prior(
        self, character: str, asc_bucket: str, fingerprint_key: str,
    ) -> tuple[float, float] | None:
        """返回 (freq, efficiency) ∈ ([0,1], [0,1]);未命中 None。"""
        if self._priors_con is None:
            return None
        row = self._priors_con.execute(
            "SELECT freq, avg_duration_sec FROM path_priors "
            "WHERE character = ? AND asc_bucket = ? AND fingerprint_key = ?",
            (character.upper(), asc_bucket, fingerprint_key),
        ).fetchone()
        if row is None:
            return None
        freq = float(row["freq"])
        # 归一化 duration 到 [0,1]:用 group_stats 的 min/max
        gs = self._priors_con.execute(
            "SELECT min_duration, max_duration FROM group_stats "
            "WHERE character = ? AND asc_bucket = ?",
            (character.upper(), asc_bucket),
        ).fetchone()
        if gs and gs["max_duration"] > gs["min_duration"]:
            dur_norm = (float(row["avg_duration_sec"]) - gs["min_duration"]) / (gs["max_duration"] - gs["min_duration"])
        else:
            dur_norm = 0.5
        efficiency = max(0.0, min(1.0, 1.0 - dur_norm))  # 越快 efficiency 越高
        return freq, efficiency

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        con = sqlite3.connect(str(self.index_db))
        con.row_factory = sqlite3.Row
        try:
            total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            clean = con.execute("SELECT COUNT(*) FROM runs WHERE is_clean=1").fetchone()[0]
            with_map = con.execute(
                "SELECT COUNT(*) FROM runs WHERE is_clean=1 AND has_map_acts=1"
            ).fetchone()[0]
            by_char = con.execute(
                "SELECT character, COUNT(*) c FROM runs WHERE is_clean=1 GROUP BY character"
            ).fetchall()
            by_bucket = con.execute(
                "SELECT asc_bucket, COUNT(*) c FROM runs WHERE is_clean=1 GROUP BY asc_bucket"
            ).fetchall()
            return {
                "total": total,
                "clean": clean,
                "clean_with_map_acts": with_map,
                "by_character": {r["character"]: r["c"] for r in by_char},
                "by_asc_bucket": {r["asc_bucket"]: r["c"] for r in by_bucket},
            }
        finally:
            con.close()
