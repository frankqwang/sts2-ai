#!/usr/bin/env python3
"""Build a merged teacher queue from multiple training run dirs.

This is a thin orchestration layer over build_offline_noncombat_teacher_queue.py:
- build per-run candidate entries
- merge all entries
- dedupe by seed (keep the highest-priority/best-floor record)
- apply the final category caps once across the merged pool
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    python_root = Path(__file__).resolve().parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))


import argparse
import json
from collections import Counter
from dataclasses import asdict

from search.build_offline_noncombat_teacher_queue import QueueEntry, build_queue


def _entry_sort_key(item: QueueEntry) -> tuple[int, int, str]:
    return (int(item.priority), int(item.baseline_end_floor), str(item.seed))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a merged offline non-combat teacher queue from multiple run dirs.")
    parser.add_argument("--run-dir", action="append", required=True, help="Training run directory containing replays/*.summary.json; repeat this flag")
    parser.add_argument("--output", required=True, help="Merged queue JSON output path")
    parser.add_argument("--seed-list-out", default="", help="Optional TXT output containing one selected seed per line")
    parser.add_argument("--min-preboss-floor", type=int, default=10)
    parser.add_argument("--max-total", type=int, default=24)
    parser.add_argument("--max-boss-reached-defeat", type=int, default=12)
    parser.add_argument("--max-preboss-death", type=int, default=10)
    parser.add_argument("--max-act1-clear-anchor", type=int, default=2)
    args = parser.parse_args()

    per_run_entries: list[QueueEntry] = []
    run_metadata: list[dict[str, object]] = []
    for run_dir_arg in args.run_dir:
        run_dir = Path(run_dir_arg)
        entries, metadata = build_queue(
            run_dir=run_dir,
            min_preboss_floor=int(args.min_preboss_floor),
            max_total=10**9,
            max_boss_reached_defeat=10**9,
            max_preboss_death=10**9,
            max_act1_clear_anchor=10**9,
        )
        per_run_entries.extend(entries)
        run_metadata.append(metadata)

    deduped_by_seed: dict[str, QueueEntry] = {}
    for item in per_run_entries:
        existing = deduped_by_seed.get(item.seed)
        if existing is None or _entry_sort_key(item) > _entry_sort_key(existing):
            deduped_by_seed[item.seed] = item

    merged_entries = sorted(deduped_by_seed.values(), key=_entry_sort_key, reverse=True)
    accepted: list[QueueEntry] = []
    category_seen: Counter[str] = Counter()
    per_category_limit = {
        "boss_reached_defeat": max(0, int(args.max_boss_reached_defeat)),
        "preboss_death": max(0, int(args.max_preboss_death)),
        "act1_clear_anchor": max(0, int(args.max_act1_clear_anchor)),
    }
    for item in merged_entries:
        if len(accepted) >= max(0, int(args.max_total)):
            break
        if category_seen[item.source_category] >= per_category_limit.get(item.source_category, 0):
            continue
        accepted.append(item)
        category_seen[item.source_category] += 1

    metadata = {
        "queue_version": "offline_noncombat_teacher_queue_multi.v1",
        "run_dirs": [str(Path(path).resolve()) for path in args.run_dir],
        "run_count": len(args.run_dir),
        "per_run_metadata": run_metadata,
        "merged_candidate_entries": len(per_run_entries),
        "merged_unique_seeds": len(deduped_by_seed),
        "selected_entries": len(accepted),
        "category_counts": dict(sorted(category_seen.items())),
        "selection_limits": per_category_limit,
        "min_preboss_floor": int(args.min_preboss_floor),
    }
    payload = {
        "queue_version": "offline_noncombat_teacher_queue.v1",
        "metadata": metadata,
        "entries": [asdict(item) for item in accepted],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.seed_list_out:
        seed_list_path = Path(args.seed_list_out)
        seed_list_path.parent.mkdir(parents=True, exist_ok=True)
        seed_list_path.write_text(
            "\n".join(item.seed for item in accepted) + ("\n" if accepted else ""),
            encoding="utf-8",
        )
    print(
        f"Merged queue selected {len(accepted)} seeds from {len(args.run_dir)} runs "
        f"({len(deduped_by_seed)} unique seeds, {len(per_run_entries)} candidates)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
