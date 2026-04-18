"""Skada 数据校验脚本(独立 CLI)。

用途:跑一遍 skada 数据(jsonl 或 sqlite),输出统计报告,
     识别格式不兼容 / id 不匹配 / 字段缺失等问题。

在大批训练前**必须先跑一次**,确认:
  1. game_version 覆盖率(多少条是最新版 detail)
  2. character 分布(是否全部在 KNOWN_CHARACTERS 里)
  3. card_id / relic_id 和 source_knowledge 的匹配率
  4. map 节点 type 分布(是否有未揭示节点)
  5. event_text 分布(是否结构化)
  6. detail_expired 率 / 字段缺失率
  7. 每类决策点的数量(card_choices / relic_choices / ...)

输出:
  - stdout:人类可读报告
  - 可选 --json-out:机器可读完整统计(供 CI 阈值 check)

用法:
    python -m networkV2.s6_training.skada_data_validator \
        --jsonl-dir data/skada/runs/details \
        --sqlite STS2AI/Assets/datasets/skada/skada_analytics.sqlite
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from networkV2.s6_training.skada_id_mapping import (
    _load_source_card_ids, _load_source_relic_ids,
    normalize_card_id, normalize_relic_id,
    KNOWN_CHARACTERS, room_letter_to_domain, room_letter_to_name,
)
from networkV2.s6_training.skada_offline_loader import iter_records_from_jsonl


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 报告容器
# ---------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.total_runs = 0
        self.runs_by_version: Counter = Counter()
        self.runs_by_character: Counter = Counter()
        self.runs_by_detail_status: Counter = Counter()
        self.runs_expired: int = 0

        self.unknown_characters: Counter = Counter()
        self.unknown_card_ids: Counter = Counter()
        self.unknown_relic_ids: Counter = Counter()

        self.floor_count = 0
        self.floors_by_room_type: Counter = Counter()

        self.total_card_choices = 0
        self.card_choices_by_size: Counter = Counter()

        self.total_relic_choices = 0
        self.total_ancient_choices = 0

        self.total_shop_actions = 0
        self.shop_actions_by_type: Counter = Counter()

        self.total_campfire_choices = 0
        self.campfire_by_type: Counter = Counter()

        self.total_events = 0
        self.top_events: Counter = Counter()

        self.total_map_acts = 0
        self.map_node_types: Counter = Counter()
        self.map_has_unknown_nodes = 0   # 有多少 act 含有 type='?' 节点

        self.samples_from_run_estimate = 0

    def add_card_id(self, cid: str) -> None:
        base, _ = normalize_card_id(cid)
        if base and base not in _load_source_card_ids():
            self.unknown_card_ids[base] += 1

    def add_relic_id(self, rid: str) -> None:
        low = normalize_relic_id(rid)
        if low and low not in _load_source_relic_ids():
            self.unknown_relic_ids[low] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "runs_by_version": dict(self.runs_by_version),
            "runs_by_character": dict(self.runs_by_character),
            "runs_by_detail_status": dict(self.runs_by_detail_status),
            "runs_expired": self.runs_expired,
            "unknown_characters": dict(self.unknown_characters),
            "unknown_card_id_count": len(self.unknown_card_ids),
            "unknown_card_id_top": self.unknown_card_ids.most_common(20),
            "unknown_relic_id_count": len(self.unknown_relic_ids),
            "unknown_relic_id_top": self.unknown_relic_ids.most_common(20),
            "floor_count": self.floor_count,
            "floors_by_room_type": dict(self.floors_by_room_type),
            "total_card_choices": self.total_card_choices,
            "card_choices_by_size": dict(self.card_choices_by_size),
            "total_relic_choices": self.total_relic_choices,
            "total_ancient_choices": self.total_ancient_choices,
            "total_shop_actions": self.total_shop_actions,
            "shop_actions_by_type": dict(self.shop_actions_by_type),
            "total_campfire_choices": self.total_campfire_choices,
            "campfire_by_type": dict(self.campfire_by_type),
            "total_events": self.total_events,
            "top_events": self.top_events.most_common(30),
            "total_map_acts": self.total_map_acts,
            "map_node_types": dict(self.map_node_types),
            "map_has_unknown_nodes": self.map_has_unknown_nodes,
            "samples_from_run_estimate": self.samples_from_run_estimate,
        }

    def print_human(self) -> None:
        d = self.to_dict()
        print("=" * 70)
        print(f"SKADA DATA VALIDATION REPORT  (total runs: {d['total_runs']})")
        print("=" * 70)
        print()
        print("1. game_version 分布")
        for v, c in self.runs_by_version.most_common():
            pct = c * 100 / max(self.total_runs, 1)
            print(f"   {v!s:<20} {c:>6}  ({pct:.1f}%)")
        print()
        print("2. character 分布")
        for ch, c in self.runs_by_character.most_common():
            flag = "" if ch in KNOWN_CHARACTERS else "  [UNKNOWN]"
            print(f"   {ch!s:<15} {c:>6}{flag}")
        if self.unknown_characters:
            print(f"   → 未知 character: {dict(self.unknown_characters)}")
        print()
        print("3. detail 状态")
        for s, c in self.runs_by_detail_status.most_common():
            print(f"   status={s!s:<10} {c}")
        print(f"   detail_expired runs: {self.runs_expired}")
        print()
        print(f"4. floor 总数: {self.floor_count}")
        print("   room_type 分布:")
        for rt, c in self.floors_by_room_type.most_common():
            name = room_letter_to_name(rt)
            domain = room_letter_to_domain(rt)
            print(f"   {rt!s:<4} {c:>6} ({name}, domain={domain})")
        print()
        print("5. id 校验(对比 source_knowledge.sqlite)")
        n_unknown_cards = sum(self.unknown_card_ids.values())
        n_unknown_relics = sum(self.unknown_relic_ids.values())
        print(f"   unknown_card_id: {len(self.unknown_card_ids)} 种, 总出现 {n_unknown_cards} 次")
        if self.unknown_card_ids:
            for cid, cnt in self.unknown_card_ids.most_common(10):
                print(f"      {cid!r:<30} ×{cnt}")
        print(f"   unknown_relic_id: {len(self.unknown_relic_ids)} 种, 总出现 {n_unknown_relics} 次")
        if self.unknown_relic_ids:
            for rid, cnt in self.unknown_relic_ids.most_common(10):
                print(f"      {rid!r:<30} ×{cnt}")
        print()
        print("6. 决策点数量")
        print(f"   card_choices:    {self.total_card_choices} 次")
        if self.card_choices_by_size:
            print(f"     按候选数:     {dict(self.card_choices_by_size.most_common(6))}")
        print(f"   relic_choices:   {self.total_relic_choices}")
        print(f"   ancient_choices: {self.total_ancient_choices}")
        print(f"   shop_actions:    {self.total_shop_actions}")
        if self.shop_actions_by_type:
            print(f"     按 action_type: {dict(self.shop_actions_by_type.most_common())}")
        print(f"   campfire_choice: {self.total_campfire_choices}")
        if self.campfire_by_type:
            print(f"     按类型:       {dict(self.campfire_by_type.most_common())}")
        print(f"   events(event_text 非空): {self.total_events}")
        if self.top_events:
            print(f"     top 10:")
            for ev, cnt in self.top_events.most_common(10):
                # event_text 可能是中文,避免 stdout 编码错误
                try:
                    print(f"       {ev!r:<30} ×{cnt}")
                except UnicodeEncodeError:
                    print(f"       <non-ascii event> ×{cnt}")
        print()
        print("7. Map(路线)")
        print(f"   map_acts total: {self.total_map_acts}")
        print(f"   节点 type 分布: {dict(self.map_node_types.most_common())}")
        print(f"   有未揭示('?')节点的 act: {self.map_has_unknown_nodes} "
              f"(越少越说明 skada 是上帝视角,需要 mask_map_with_visibility 保护)")
        print()
        print(f"8. 估计可产出 sample 数(用 jsonl/sqlite 实际跑 loader 为准)")
        print(f"   ≈ {self.samples_from_run_estimate}")
        print()
        print("=" * 70)


# ---------------------------------------------------------------------------
# 扫 jsonl 目录
# ---------------------------------------------------------------------------

def validate_jsonl_dir(path: Path, max_runs: int | None = None) -> Report:
    rep = Report()
    for jf in sorted(Path(path).glob("*.jsonl")):
        for rec in iter_records_from_jsonl(jf):
            _absorb_record(rec, rep)
            if max_runs is not None and rep.total_runs >= max_runs:
                return rep
    return rep


# ---------------------------------------------------------------------------
# 扫 sqlite(兼容旧 19K runs 数据)
# ---------------------------------------------------------------------------

def validate_sqlite(db_path: Path, max_runs: int | None = None) -> Report:
    """从 skada_analytics.sqlite 直接采集统计(无需 JOIN 大量 detail)。

    对 run_details.raw_json 列的 record 做和 jsonl 相同的处理。
    """
    rep = Report()
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    # run-level 扫描
    q = "SELECT r.run_id, r.game_version, r.character, d.status, d.raw_json FROM runs r LEFT JOIN run_details d ON d.run_id=r.run_id"
    if max_runs is not None:
        q += f" LIMIT {int(max_runs)}"
    for row in con.execute(q):
        rec: dict[str, Any] = {}
        if row["raw_json"]:
            try:
                rec = json.loads(row["raw_json"])
            except Exception:
                rec = {}
        # 补一些从 runs 表直接拿
        rec.setdefault("run", {})
        rec["run"].setdefault("character", row["character"])
        rec["run"].setdefault("game_version", row["game_version"])
        rec["run"].setdefault("run_id", row["run_id"])
        rec["_sqlite_detail_status"] = row["status"]
        _absorb_record(rec, rep)

    con.close()
    return rep


# ---------------------------------------------------------------------------
# 核心:一条 record → report 累加
# ---------------------------------------------------------------------------

def _absorb_record(rec: dict[str, Any], rep: Report) -> None:
    rep.total_runs += 1
    run = rec.get("run", {})
    version = str(run.get("game_version", "") or "unknown")
    character = str(run.get("character", "") or "").upper()
    rep.runs_by_version[version] += 1
    rep.runs_by_character[character] += 1
    if character and character not in KNOWN_CHARACTERS:
        rep.unknown_characters[character] += 1

    # detail 状态(jsonl 用 detail_expired;sqlite 用 _sqlite_detail_status)
    if rec.get("_sqlite_detail_status") is not None:
        rep.runs_by_detail_status[rec["_sqlite_detail_status"]] += 1
    else:
        rep.runs_by_detail_status["jsonl"] += 1
    if rec.get("detail_expired"):
        rep.runs_expired += 1

    # floor_timeline
    floors = rec.get("floor_timeline") or []
    rep.floor_count += len(floors)
    n_cr = n_rc = n_an = n_sh = n_cf = n_ev = 0
    for f in floors:
        rt = str(f.get("room_type", "") or "")
        rep.floors_by_room_type[rt] += 1

        if f.get("card_choices"):
            n_cr += 1
            rep.total_card_choices += 1
            rep.card_choices_by_size[len(f["card_choices"])] += 1
            for c in f["card_choices"]:
                rep.add_card_id(c.get("card_id", ""))
        if f.get("relic_choices"):
            n_rc += 1
            rep.total_relic_choices += 1
            for c in f["relic_choices"]:
                rep.add_relic_id(c.get("relic_id", ""))
        if f.get("ancient_choices"):
            n_an += 1
            rep.total_ancient_choices += 1
            for c in f["ancient_choices"]:
                rep.add_relic_id(c.get("relic_id", ""))
        for act in f.get("shop_actions") or []:
            rep.total_shop_actions += 1
            rep.shop_actions_by_type[str(act.get("action_type", "") or "")] += 1
            iid = act.get("item_id", "")
            # item 可能是 card/relic/potion,按前缀粗分
            low = str(iid or "").lower()
            if "relic" in str(act.get("action_type", "")).lower():
                rep.add_relic_id(iid)
            # 其他当 card 验(会 False 但不累计错误)
        if f.get("campfire_choice"):
            n_cf += 1
            rep.total_campfire_choices += 1
            rep.campfire_by_type[str(f["campfire_choice"]).upper()] += 1
        for up in f.get("card_upgrades") or []:
            rep.add_card_id(up.get("card_id", ""))
        ev = f.get("event_text")
        if ev:
            n_ev += 1
            rep.total_events += 1
            rep.top_events[str(ev)[:30]] += 1

    # map acts
    for act in rec.get("map_acts") or []:
        rep.total_map_acts += 1
        nodes = act.get("nodes") or []
        has_unknown = False
        for n in nodes:
            t = str(n.get("type", "") or "")
            rep.map_node_types[t] += 1
            if t in ("?", "", "UNKNOWN"):
                has_unknown = True
        if has_unknown:
            rep.map_has_unknown_nodes += 1

    # 粗略样本数估计(参考之前 loader 行为:每 run ~64 samples)
    rep.samples_from_run_estimate += (
        n_cr + n_rc + n_an + n_cf + len(floors) +  # 决策 sample + value_only
        sum(max(0, len(a.get("visited_coords") or []) - 1) for a in (rec.get("map_acts") or []))
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Skada 数据校验器")
    p.add_argument("--jsonl-dir", type=Path, default=None,
                   help="扫 jsonl 目录(默认不扫)")
    p.add_argument("--sqlite", type=Path, default=None,
                   help="扫 skada_analytics.sqlite 文件(可和 jsonl 一起给,分别报告)")
    p.add_argument("--max-runs", type=int, default=None,
                   help="抽样 run 上限(smoke 用)")
    p.add_argument("--json-out", type=Path, default=None,
                   help="把完整统计写到 JSON 文件")
    return p


def main():
    args = _build_parser().parse_args()

    if args.jsonl_dir is None and args.sqlite is None:
        # 默认扫本地 jsonl
        args.jsonl_dir = Path("data/skada/runs/details")

    if args.jsonl_dir:
        print(f"\n>>> Scanning JSONL dir: {args.jsonl_dir}")
        rep_jsonl = validate_jsonl_dir(args.jsonl_dir, max_runs=args.max_runs)
        rep_jsonl.print_human()
        if args.json_out:
            out = Path(args.json_out)
            if args.sqlite:
                out = out.with_stem(out.stem + "_jsonl")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rep_jsonl.to_dict(), ensure_ascii=False, indent=2, default=str),
                           encoding="utf-8")
            print(f"wrote {out}")

    if args.sqlite:
        print(f"\n>>> Scanning SQLite: {args.sqlite}")
        rep_sql = validate_sqlite(args.sqlite, max_runs=args.max_runs)
        rep_sql.print_human()
        if args.json_out:
            out = Path(args.json_out)
            if args.jsonl_dir:
                out = out.with_stem(out.stem + "_sqlite")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rep_sql.to_dict(), ensure_ascii=False, indent=2, default=str),
                           encoding="utf-8")
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
