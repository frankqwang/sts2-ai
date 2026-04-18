#!/usr/bin/env python3
"""从 runs_victory + runs_failure 里筛出含完整 detail 的 run（新结构），
写到 runs_new/victory 和 runs_new/failure。

旧结构 vs 新结构：
- 老记录：detail 过期（detail_expired=true），只有基础 run 信息，缺 map_acts / final_deck / perspectives
- 新记录：完整详情，含 map_acts / final_deck / perspectives
- sts2log detail 窗口 = 3 天，窗口外的 run 过期

判定：记录有非空 map_acts (list) → 新结构 → 保留

输出结构：
  runs_new/victory/details/run_details_NNNNNN.jsonl  (每 shard 2000 行)
  runs_new/failure/details/run_details_NNNNNN.jsonl
  runs_new/SUMMARY.json  (源统计)

重跑时自动覆盖 runs_new。
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


SHARD_LINES = 2000


def is_new_structure(rec: dict[str, Any]) -> bool:
    if rec.get("detail_expired") is True:
        return False
    ma = rec.get("map_acts")
    if not isinstance(ma, list) or not ma:
        return False
    return True


def extract(src_details_dir: Path, dst_details_dir: Path) -> dict[str, int]:
    if dst_details_dir.exists():
        shutil.rmtree(dst_details_dir)
    dst_details_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    picked = 0
    shard_idx = 1
    count_in_shard = 0
    fh = None

    for src in sorted(src_details_dir.glob("run_details_*.jsonl")):
        with src.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not is_new_structure(rec):
                    continue
                if fh is None or count_in_shard >= SHARD_LINES:
                    if fh is not None:
                        fh.close()
                    shard_path = dst_details_dir / f"run_details_{shard_idx:06d}.jsonl"
                    fh = shard_path.open("w", encoding="utf-8", newline="\n")
                    shard_idx += 1
                    count_in_shard = 0
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_in_shard += 1
                picked += 1
    if fh is not None:
        fh.close()

    return {"total": total, "picked": picked, "shards": shard_idx - 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="runs_victory / runs_failure 所在根目录",
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()

    jobs = [
        ("victory", root / "runs_victory" / "details", root / "runs_new" / "victory" / "details"),
        ("failure", root / "runs_failure" / "details", root / "runs_new" / "failure" / "details"),
    ]
    summary: dict[str, Any] = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "rule": "is_new_structure: detail_expired != true AND map_acts is non-empty list",
        "shard_lines": SHARD_LINES,
        "stats": {},
    }
    for label, src, dst in jobs:
        if not src.exists():
            print(f"[{label}] src not found: {src}")
            summary["stats"][label] = {"error": "src not found"}
            continue
        stat = extract(src, dst)
        summary["stats"][label] = {
            **stat,
            "keep_ratio": round(stat["picked"] / stat["total"], 4) if stat["total"] else 0.0,
            "dst": str(dst),
        }
        print(
            f"[{label}] {stat['picked']}/{stat['total']} "
            f"({stat['picked'] / max(stat['total'], 1) * 100:.1f}%) → {stat['shards']} shards → {dst}"
        )

    summary_path = root / "runs_new" / "SUMMARY.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary written to {summary_path}")


if __name__ == "__main__":
    main()
