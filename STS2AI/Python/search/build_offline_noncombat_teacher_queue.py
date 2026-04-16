#!/usr/bin/env python3
"""Build a windowed teacher queue from training replay summaries.

This script reads `replays/*.summary.json` from a completed training window,
selects the highest-value seeds for offline non-combat teacher refresh, and
writes a compact queue JSON plus a plain seed list.
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
from dataclasses import dataclass, asdict
from typing import Any

from env.run_outcome_vocab import RUN_OUTCOME_VICTORY, is_failure_outcome, normalize_run_outcome


@dataclass
class QueueEntry:
    seed: str
    source_category: str
    priority: int
    baseline_end_floor: int
    baseline_outcome: str
    baseline_boss_reached: bool
    baseline_act1_cleared: bool
    replay_summary_path: str
    iteration: int
    episode: int


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _categorize_summary(summary: dict[str, Any], *, min_preboss_floor: int) -> tuple[str | None, int]:
    outcome = normalize_run_outcome(summary.get("outcome"), default="")
    normalized_defeat = is_failure_outcome(outcome) or outcome == "timeout"
    floor = _safe_int(summary.get("final_floor", 0), 0)
    boss_reached = bool(summary.get("boss_reached"))
    act1_cleared = bool(summary.get("act1_cleared"))
    if boss_reached and normalized_defeat:
        return "boss_reached_defeat", 20 + floor
    if normalized_defeat and floor >= int(min_preboss_floor):
        return "preboss_death", 30 + floor
    if outcome == RUN_OUTCOME_VICTORY or act1_cleared:
        return "act1_clear_anchor", 10 + floor
    return None, -1


def build_queue(
    *,
    run_dir: Path,
    min_preboss_floor: int,
    max_total: int,
    max_boss_reached_defeat: int,
    max_preboss_death: int,
    max_act1_clear_anchor: int,
) -> tuple[list[QueueEntry], dict[str, Any]]:
    replay_dir = run_dir / "replays"
    summary_paths = sorted(replay_dir.glob("*.summary.json"))
    entries: list[QueueEntry] = []
    missing_seed = 0
    category_seen: Counter[str] = Counter()
    for path in summary_paths:
        summary = _load_summary(path)
        seed = str(summary.get("seed") or "").strip()
        if not seed:
            missing_seed += 1
            continue
        category, priority = _categorize_summary(summary, min_preboss_floor=min_preboss_floor)
        if not category:
            continue
        entries.append(
            QueueEntry(
                seed=seed,
                source_category=category,
                priority=int(priority),
                baseline_end_floor=_safe_int(summary.get("final_floor", 0), 0),
                baseline_outcome=str(summary.get("outcome") or ""),
                baseline_boss_reached=bool(summary.get("boss_reached")),
                baseline_act1_cleared=bool(summary.get("act1_cleared")),
                replay_summary_path=str(path.resolve()),
                iteration=_safe_int(summary.get("iteration", 0), 0),
                episode=_safe_int(summary.get("episode", 0), 0),
            )
        )
    entries.sort(key=lambda item: (-item.priority, item.seed))
    accepted: list[QueueEntry] = []
    per_category_limit = {
        "boss_reached_defeat": max(0, int(max_boss_reached_defeat)),
        "preboss_death": max(0, int(max_preboss_death)),
        "act1_clear_anchor": max(0, int(max_act1_clear_anchor)),
    }
    seen_seeds: set[str] = set()
    for item in entries:
        if item.seed in seen_seeds:
            continue
        if len(accepted) >= max(0, int(max_total)):
            break
        if category_seen[item.source_category] >= per_category_limit.get(item.source_category, 0):
            continue
        accepted.append(item)
        seen_seeds.add(item.seed)
        category_seen[item.source_category] += 1
    metadata = {
        "run_dir": str(run_dir.resolve()),
        "summary_files": len(summary_paths),
        "missing_seed_summaries": missing_seed,
        "selected_entries": len(accepted),
        "category_counts": dict(sorted(category_seen.items())),
        "selection_limits": per_category_limit,
        "min_preboss_floor": int(min_preboss_floor),
    }
    return accepted, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a seed queue for offline non-combat teacher refresh.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing replays/*.summary.json")
    parser.add_argument("--output", required=True, help="Queue JSON output path")
    parser.add_argument("--seed-list-out", default="", help="Optional TXT output containing one selected seed per line")
    parser.add_argument("--min-preboss-floor", type=int, default=10)
    parser.add_argument("--max-total", type=int, default=12)
    parser.add_argument("--max-boss-reached-defeat", type=int, default=6)
    parser.add_argument("--max-preboss-death", type=int, default=4)
    parser.add_argument("--max-act1-clear-anchor", type=int, default=1)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output)
    entries, metadata = build_queue(
        run_dir=run_dir,
        min_preboss_floor=int(args.min_preboss_floor),
        max_total=int(args.max_total),
        max_boss_reached_defeat=int(args.max_boss_reached_defeat),
        max_preboss_death=int(args.max_preboss_death),
        max_act1_clear_anchor=int(args.max_act1_clear_anchor),
    )
    payload = {
        "queue_version": "offline_noncombat_teacher_queue.v1",
        "metadata": metadata,
        "entries": [asdict(item) for item in entries],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.seed_list_out:
        seed_list_path = Path(args.seed_list_out)
        seed_list_path.parent.mkdir(parents=True, exist_ok=True)
        seed_list_path.write_text(
            "\n".join(item.seed for item in entries) + ("\n" if entries else ""),
            encoding="utf-8",
        )
    print(
        f"Selected {len(entries)} seeds from {metadata['summary_files']} summaries "
        f"(missing_seed={metadata['missing_seed_summaries']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
