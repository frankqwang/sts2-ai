#!/usr/bin/env python3
"""
Build a cleaned campfire-choice dataset from scraped Skada run details.

Each sample is a single campfire decision with pre-floor deck context.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from skada_context import DeckTracker, dict_factory, fetch_many, safe_float, safe_int
from skada_db import DB_PATH
from skada_priors import SkadaPriors


def build_dataset(
    conn: sqlite3.Connection,
    priors: SkadaPriors,
    max_runs: int = 2000,
    characters: list[str] | None = None,
    min_floor_reached: int = 0,
    max_sample_floor: int = 0,
    min_ascension: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    where_parts = [
        """
        EXISTS (
            SELECT 1
            FROM run_floor_timeline t
            WHERE t.run_id = r.run_id
              AND t.room_type = 'R'
              AND t.campfire_choice IS NOT NULL
              AND t.campfire_choice <> ''
        )
        """
    ]
    params: list[Any] = []
    if characters:
        where_parts.append(f"upper(r.character) IN ({','.join('?' for _ in characters)})")
        params.extend([str(ch).strip().upper() for ch in characters])
    if min_floor_reached > 0:
        where_parts.append("(coalesce(r.is_victory, 0) = 1 OR coalesce(r.floor_reached, 0) >= ?)")
        params.append(int(min_floor_reached))
    if min_ascension > 0:
        where_parts.append("coalesce(r.ascension, 0) >= ?")
        params.append(int(min_ascension))
    where_sql = " AND ".join(where_parts)
    run_rows = conn.execute(
        f"""
        SELECT *
        FROM runs r
        WHERE {where_sql}
        ORDER BY r.run_id
        LIMIT ?
        """,
        [*params, max_runs],
    ).fetchall()
    run_ids = [int(row["run_id"]) for row in run_rows]

    timeline_rows = fetch_many(
        conn,
        """
        SELECT *
        FROM run_floor_timeline
        WHERE run_id IN ({placeholders})
        ORDER BY run_id, floor
        """,
        run_ids,
    )
    card_choice_rows = fetch_many(
        conn,
        """
        SELECT *
        FROM run_floor_card_choices
        WHERE run_id IN ({placeholders})
        ORDER BY run_id, floor, choice_index
        """,
        run_ids,
    )
    relic_choice_rows = fetch_many(
        conn,
        """
        SELECT *
        FROM run_floor_relic_choices
        WHERE run_id IN ({placeholders})
        ORDER BY run_id, floor, choice_index
        """,
        run_ids,
    )
    shop_action_rows = fetch_many(
        conn,
        """
        SELECT *
        FROM run_floor_shop_actions
        WHERE run_id IN ({placeholders})
        ORDER BY run_id, floor, action_index
        """,
        run_ids,
    )
    upgrade_rows = fetch_many(
        conn,
        """
        SELECT *
        FROM run_card_upgrades
        WHERE run_id IN ({placeholders})
        ORDER BY run_id, floor, upgrade_index
        """,
        run_ids,
    )

    timeline_by_run: dict[int, list[dict[str, Any]]] = defaultdict(list)
    choices_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    relics_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    shops_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    upgrades_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for row in timeline_rows:
        timeline_by_run[int(row["run_id"])].append(row)
    for row in card_choice_rows:
        choices_by_key[(int(row["run_id"]), int(row["floor"]))].append(row)
    for row in relic_choice_rows:
        relics_by_key[(int(row["run_id"]), int(row["floor"]))].append(row)
    for row in shop_action_rows:
        shops_by_key[(int(row["run_id"]), int(row["floor"]))].append(row)
    for row in upgrade_rows:
        upgrades_by_key[(int(row["run_id"]), int(row["floor"]))].append(row)

    samples: list[dict[str, Any]] = []
    char_counter: Counter[str] = Counter()
    choice_counter: Counter[str] = Counter()

    for run in run_rows:
        run_id = int(run["run_id"])
        character = str(run.get("character") or "UNKNOWN").upper()
        char_counter[character] += 1
        tracker = DeckTracker(character)
        recent_damage_taken: deque[float] = deque(maxlen=3)
        recent_damage_dealt: deque[float] = deque(maxlen=3)
        recent_rooms: deque[str] = deque(maxlen=8)

        for floor_row in timeline_by_run[run_id]:
            floor = int(floor_row["floor"])
            if max_sample_floor > 0 and floor > max_sample_floor:
                break
            key = (run_id, floor)
            room_type = str(floor_row.get("room_type") or "UNKNOWN")
            recent_shop_visits = sum(1 for room in recent_rooms if room == "S")
            recent_elites = sum(1 for room in recent_rooms if room == "A")
            shared = tracker.shared_context(
                priors=priors,
                floor=floor,
                hp_before=safe_float(floor_row.get("hp_before")),
                gold_before=safe_float(floor_row.get("gold_before")),
                recent_damage_taken=(sum(recent_damage_taken) / len(recent_damage_taken) if recent_damage_taken else 0.0),
                recent_damage_dealt=(sum(recent_damage_dealt) / len(recent_damage_dealt) if recent_damage_dealt else 0.0),
                recent_shop_visits=recent_shop_visits,
                recent_elites=recent_elites,
            )

            campfire_choice = str(floor_row.get("campfire_choice") or "").strip()
            if room_type == "R" and campfire_choice:
                choice_counter[campfire_choice] += 1
                hp_before = safe_float(floor_row.get("hp_before"))
                rest_value_proxy = max(0.0, min(1.0, 1.0 - min(1.0, hp_before / 80.0)))
                samples.append({
                    "sample_id": f"{run_id}:{floor}",
                    "run_id": run_id,
                    "floor": floor,
                    "character": character,
                    "ascension": safe_int(run.get("ascension")),
                    "is_victory": safe_int(run.get("is_victory")),
                    "floor_reached": safe_int(run.get("floor_reached")),
                    "duration_sec": safe_int(run.get("duration_sec")),
                    "room_type": room_type,
                    "campfire_choice": campfire_choice,
                    "campfire_name_en": floor_row.get("campfire_name_en"),
                    "campfire_name_zh": floor_row.get("campfire_name_zh"),
                    "rest_value_proxy": round(rest_value_proxy, 6),
                    "smith_targets": [str(item.get("card_id") or "") for item in upgrades_by_key.get(key, []) if item.get("card_id")],
                    **shared,
                })

            for option in choices_by_key.get(key, []):
                if safe_int(option.get("was_picked")) == 1:
                    tracker.add_card(option.get("card_id"))

            for relic in relics_by_key.get(key, []):
                if safe_int(relic.get("was_picked")) == 1:
                    tracker.add_relic(relic.get("relic_id"))

            for action in shops_by_key.get(key, []):
                action_type = str(action.get("action_type") or "")
                item_id = action.get("item_id")
                if action_type == "buy_card":
                    tracker.add_card(item_id)
                elif action_type == "buy_relic":
                    tracker.add_relic(item_id)
                elif action_type == "remove":
                    tracker.remove_card(item_id)

            for upgrade in upgrades_by_key.get(key, []):
                tracker.upgrade_card(upgrade.get("card_id"))

            dmg_taken = safe_float(floor_row.get("total_dmg_taken"))
            dmg_dealt = safe_float(floor_row.get("total_dmg_dealt"))
            if dmg_taken or dmg_dealt:
                recent_damage_taken.append(dmg_taken)
                recent_damage_dealt.append(dmg_dealt)
            recent_rooms.append(room_type)

    manifest = {
        "num_runs": len(run_ids),
        "num_samples": len(samples),
        "characters": dict(char_counter),
        "campfire_choices": dict(choice_counter),
        "avg_floor": round(sum(sample["floor"] for sample in samples) / max(1, len(samples)), 4),
        "victory_rate": round(sum(sample["is_victory"] for sample in samples) / max(1, len(samples)), 4),
    }
    return samples, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Skada campfire choice dataset")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--max-runs", type=int, default=2000, help="Max runs to include")
    parser.add_argument("--character", action="append", default=[],
                        help="Restrict to one character. Repeatable, but intended for per-character datasets.")
    parser.add_argument("--min-floor-reached", type=int, default=0,
                        help="Keep only runs that won or reached at least this floor.")
    parser.add_argument("--max-sample-floor", type=int, default=0,
                        help="Only emit samples up to this floor (e.g. 17 for act 1). 0 disables the cap.")
    parser.add_argument("--min-ascension", type=int, default=0,
                        help="Keep only runs with ascension at or above this threshold.")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON path")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Skada DB not found: {db_path}")

    normalized_characters = [str(ch).strip().upper() for ch in args.character if str(ch).strip()]
    char_suffix = ""
    if len(normalized_characters) == 1:
        char_suffix = f"{normalized_characters[0].lower()}_"
    elif len(normalized_characters) > 1:
        char_suffix = "multi_"

    output_path = Path(args.output) if args.output else (
        db_path.resolve().parents[3] / "artifacts" / "skada" / f"{char_suffix}campfire_choice.jsonl"
    )
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = dict_factory
    priors = SkadaPriors(db_path)
    samples, manifest = build_dataset(
        conn,
        priors,
        max_runs=args.max_runs,
        characters=normalized_characters,
        min_floor_reached=args.min_floor_reached,
        max_sample_floor=args.max_sample_floor,
        min_ascension=args.min_ascension,
    )
    conn.close()

    with output_path.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest.update({
        "db_path": str(db_path),
        "output_path": str(output_path),
        "max_runs": int(args.max_runs),
        "characters_filter": normalized_characters,
        "min_floor_reached": int(args.min_floor_reached),
        "max_sample_floor": int(args.max_sample_floor),
        "min_ascension": int(args.min_ascension),
    })
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
