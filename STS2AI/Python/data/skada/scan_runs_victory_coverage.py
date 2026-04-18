#!/usr/bin/env python3
"""扫 runs_victory/details/*.jsonl,报告数据字段覆盖率。

每个 jsonl 文件包含多条 run record(每行一条)。输出:
- 总 run 数
- character / ascension / game_version 分布
- 每个 run-level 字段(combats / floor_timeline / final_deck / map_acts)覆盖率
- floor-level 字段(card_choices / relic_choices / ancient_choices / campfire_choice /
  shop_actions / event_text / combat)的平均每 run 出现次数
- 空/缺失 run 数

用法:
    python -m data.skada.scan_runs_victory_coverage \
        --dir STS2AI/Python/data/skada/runs_victory/details
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def scan(dir_path: Path, max_files: int | None = None) -> dict:
    files = sorted(dir_path.glob("*.jsonl"))
    # 排除 newSample.jsonl(pretty-printed 单条,不是 jsonl)
    files = [f for f in files if f.name != "newSample.jsonl"]
    if max_files:
        files = files[:max_files]

    total_runs = 0
    characters = Counter()
    ascensions = Counter()
    versions = Counter()
    is_victory_count = 0

    # run-level 字段存在率
    has_combats = has_floor_tl = has_final_deck = has_final_relics = has_map_acts = 0

    # floor-level 字段出现总次数
    n_card_choices = 0
    n_relic_choices = 0
    n_ancient_choices = 0
    n_campfire = 0
    n_shop_actions = 0
    n_event_text = 0
    n_combat_in_floor = 0
    card_choice_sizes = Counter()  # n_candidates → count

    # choice-level:多少 choice 有 was_picked=True
    n_card_picks = 0
    n_card_skips = 0

    # deck/relic size
    deck_sizes = []
    relic_sizes = []
    combat_counts = []
    floor_counts = []

    # map_acts 可用性
    n_map_acts_total = 0
    n_map_acts_with_visited = 0
    n_map_acts_with_nodes = 0

    # empty / broken
    n_empty_timeline = 0

    for jf in files:
        try:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total_runs += 1
                    run = rec.get("run", {}) or {}
                    characters[str(run.get("character", "")).upper()] += 1
                    ascensions[int(run.get("ascension", 0) or 0)] += 1
                    versions[str(run.get("game_version", "")) or "unknown"] += 1
                    if run.get("is_victory"):
                        is_victory_count += 1

                    combats = rec.get("combats") or []
                    ft = rec.get("floor_timeline") or []
                    fd = rec.get("final_deck") or []
                    fr = rec.get("final_relics") or []
                    ma = rec.get("map_acts") or []
                    if combats: has_combats += 1
                    if ft: has_floor_tl += 1
                    if fd: has_final_deck += 1
                    if fr: has_final_relics += 1
                    if ma: has_map_acts += 1
                    if not ft:
                        n_empty_timeline += 1
                    combat_counts.append(len(combats))
                    floor_counts.append(len(ft))
                    deck_sizes.append(len(fd))
                    relic_sizes.append(len(fr))

                    for f_data in ft:
                        if f_data.get("card_choices"):
                            n_card_choices += 1
                            cc = f_data["card_choices"]
                            card_choice_sizes[len(cc)] += 1
                            for c in cc:
                                if c.get("was_picked"):
                                    n_card_picks += 1
                                else:
                                    n_card_skips += 1
                        if f_data.get("relic_choices"):
                            n_relic_choices += 1
                        if f_data.get("ancient_choices"):
                            n_ancient_choices += 1
                        if f_data.get("campfire_choice"):
                            n_campfire += 1
                        if f_data.get("shop_actions"):
                            n_shop_actions += 1
                        if f_data.get("event_text"):
                            n_event_text += 1
                        if f_data.get("combat"):
                            n_combat_in_floor += 1

                    for act in ma:
                        n_map_acts_total += 1
                        if act.get("visited_coords"):
                            n_map_acts_with_visited += 1
                        if act.get("nodes"):
                            n_map_acts_with_nodes += 1
        except Exception as e:
            logger.warning(f"failed to scan {jf}: {e}")

    def _avg(xs):
        return sum(xs) / max(len(xs), 1)

    return {
        "files": len(files),
        "total_runs": total_runs,
        "is_victory_count": is_victory_count,
        "is_victory_rate": is_victory_count / max(total_runs, 1),
        "characters": dict(characters.most_common()),
        "ascensions_top": dict(ascensions.most_common(12)),
        "versions": dict(versions.most_common()),
        "run_level_coverage": {
            "has_combats": has_combats / max(total_runs, 1),
            "has_floor_timeline": has_floor_tl / max(total_runs, 1),
            "has_final_deck": has_final_deck / max(total_runs, 1),
            "has_final_relics": has_final_relics / max(total_runs, 1),
            "has_map_acts": has_map_acts / max(total_runs, 1),
            "empty_timeline_runs": n_empty_timeline,
        },
        "avg_per_run": {
            "combats": _avg(combat_counts),
            "floors": _avg(floor_counts),
            "final_deck_size": _avg(deck_sizes),
            "final_relic_count": _avg(relic_sizes),
            "card_choices_events": n_card_choices / max(total_runs, 1),
            "relic_choices_events": n_relic_choices / max(total_runs, 1),
            "ancient_choices_events": n_ancient_choices / max(total_runs, 1),
            "campfire_events": n_campfire / max(total_runs, 1),
            "shop_actions_events": n_shop_actions / max(total_runs, 1),
            "event_text_events": n_event_text / max(total_runs, 1),
        },
        "total_decision_points": {
            "card_choices": n_card_choices,
            "card_picks": n_card_picks,
            "card_skips": n_card_skips,
            "relic_choices": n_relic_choices,
            "ancient_choices": n_ancient_choices,
            "campfire": n_campfire,
            "shop_actions": n_shop_actions,
            "event_text": n_event_text,
        },
        "card_choice_size_distribution": dict(card_choice_sizes.most_common()),
        "map_acts": {
            "total": n_map_acts_total,
            "with_visited_coords": n_map_acts_with_visited,
            "with_nodes": n_map_acts_with_nodes,
        },
    }


def print_report(report: dict) -> None:
    print("=" * 72)
    print(f"Skada runs_victory 数据覆盖统计")
    print("=" * 72)
    print(f"files scanned: {report['files']}")
    print(f"total runs:    {report['total_runs']}")
    print(f"is_victory:    {report['is_victory_count']} ({report['is_victory_rate']*100:.1f}%)")
    print()
    print("character 分布:")
    for ch, n in report["characters"].items():
        print(f"  {ch:<15} {n}")
    print()
    print("ascension 分布 top12:")
    for a, n in report["ascensions_top"].items():
        print(f"  asc {a}: {n}")
    print()
    print("game_version 分布:")
    for v, n in report["versions"].items():
        print(f"  {v:<15} {n}")
    print()
    print("--- run-level 字段覆盖率 ---")
    for k, v in report["run_level_coverage"].items():
        if isinstance(v, float):
            print(f"  {k:<30} {v*100:.1f}%")
        else:
            print(f"  {k:<30} {v}")
    print()
    print("--- 每 run 平均 ---")
    for k, v in report["avg_per_run"].items():
        print(f"  {k:<30} {v:.2f}")
    print()
    print("--- 总决策点(供 BC 训练样本量估算)---")
    tdp = report["total_decision_points"]
    for k, v in tdp.items():
        print(f"  {k:<20} {v}")
    print(f"  (预估 sample 总量 ≈ card_choices + relic+ancient + campfire×SMITH/HEAL 过滤 + map*30 ≈ "
          f"{tdp['card_choices'] + tdp['relic_choices'] + tdp['ancient_choices'] + tdp['campfire']*0.95 + report['map_acts']['total']*25:.0f})")
    print()
    print("--- card_choices 候选数分布 ---")
    for sz, n in report["card_choice_size_distribution"].items():
        print(f"  {sz} 候选: {n}")
    print()
    print("--- map_acts 可用性 ---")
    for k, v in report["map_acts"].items():
        print(f"  {k:<22} {v}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path,
                   default=Path("STS2AI/Python/data/skada/runs_victory/details"))
    p.add_argument("--max-files", type=int, default=None,
                   help="最多扫前 N 个 shard(测试用)")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    logger.info(f"scanning {args.dir} ...")
    rep = scan(args.dir, max_files=args.max_files)
    print_report(rep)
    if args.json_out:
        args.json_out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
