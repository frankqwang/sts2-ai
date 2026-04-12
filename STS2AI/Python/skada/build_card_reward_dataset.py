#!/usr/bin/env python3
"""
Build a cleaned card-reward choice dataset from scraped Skada run details.

Each sample is a single reward-offer group:
- shared run/floor context
- 2-9 offered card options
- chosen_index label

Usage:
    python STS2AI/Python/skada/build_card_reward_dataset.py --max-runs 2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(_PYTHON_ROOT))
from skada_db import DB_PATH
from skada_priors import SkadaPriors
from sts2ai_paths import ARTIFACTS_ROOT


def _slug(card_id: str | None) -> str:
    text = str(card_id or "").strip().lower()
    if text.endswith("+"):
        text = text[:-1]
    return text


def _dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def _fetch_many(conn: sqlite3.Connection, query: str, run_ids: list[int]) -> list[dict[str, Any]]:
    if not run_ids:
        return []
    result: list[dict[str, Any]] = []
    chunk_size = 500
    for start in range(0, len(run_ids), chunk_size):
        chunk = run_ids[start:start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        result.extend(conn.execute(query.format(placeholders=placeholders), chunk).fetchall())
    return result


def _act_from_floor(floor: int) -> int:
    if floor <= 17:
        return 1
    if floor <= 33:
        return 2
    return 3


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def build_dataset(
    conn: sqlite3.Connection,
    priors: SkadaPriors,
    *,
    max_runs: int,
    min_choices: int,
    characters: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    where_parts = ["d.status = 'ok'"]
    params: list[Any] = []
    normalized_characters = [str(ch).strip().upper() for ch in (characters or []) if str(ch).strip()]
    if normalized_characters:
        placeholders = ",".join("?" for _ in normalized_characters)
        where_parts.append(f"upper(r.character) IN ({placeholders})")
        params.extend(normalized_characters)
    where_clause = " AND ".join(where_parts)
    run_rows = conn.execute(
        f"""
        SELECT r.run_id, r.character, r.ascension, r.is_victory, r.floor_reached,
               r.duration_sec, r.player_name
        FROM runs r
        JOIN run_details d ON d.run_id = r.run_id
        WHERE {where_clause}
        ORDER BY r.run_id
        LIMIT ?
        """,
        params + [max_runs],
    ).fetchall()
    run_ids = [int(r["run_id"]) for r in run_rows]
    if not run_ids:
        return [], {"num_runs": 0, "num_groups": 0, "num_option_rows": 0}

    timeline_rows = _fetch_many(
        conn,
        """
        SELECT * FROM run_floor_timeline
        WHERE run_id IN ({placeholders})
        ORDER BY run_id, floor
        """,
        run_ids,
    )
    card_choice_rows = _fetch_many(
        conn,
        """
        SELECT * FROM run_floor_card_choices
        WHERE run_id IN ({placeholders})
        ORDER BY run_id, floor, choice_index
        """,
        run_ids,
    )
    relic_choice_rows = _fetch_many(
        conn,
        """
        SELECT * FROM run_floor_relic_choices
        WHERE run_id IN ({placeholders})
        ORDER BY run_id, floor, choice_index
        """,
        run_ids,
    )
    shop_action_rows = _fetch_many(
        conn,
        """
        SELECT * FROM run_floor_shop_actions
        WHERE run_id IN ({placeholders})
        ORDER BY run_id, floor, action_index
        """,
        run_ids,
    )
    upgrade_rows = _fetch_many(
        conn,
        """
        SELECT * FROM run_card_upgrades
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

    groups: list[dict[str, Any]] = []
    room_counter: Counter[str] = Counter()
    card_counter: Counter[str] = Counter()
    char_counter: Counter[str] = Counter()

    for run in run_rows:
        run_id = int(run["run_id"])
        character = str(run["character"] or "UNKNOWN")
        char_counter[character] += 1
        prior_card_slugs: list[str] = []
        prior_relics: list[str] = []
        prior_card_gains = 0
        prior_shop_removes = 0
        prior_upgrades = 0

        for floor_row in timeline_by_run[run_id]:
            floor = int(floor_row["floor"])
            key = (run_id, floor)
            offered = choices_by_key.get(key, [])
            picked = [row for row in offered if _safe_int(row.get("was_picked")) == 1]
            if len(offered) >= min_choices and len(picked) == 1:
                chosen_choice = picked[0]
                chosen_index = next(
                    (idx for idx, row in enumerate(offered) if int(row["choice_index"]) == int(chosen_choice["choice_index"])),
                    -1,
                )
                if chosen_index >= 0:
                    floor_num = _safe_float(floor)
                    hp_before = _safe_float(floor_row.get("hp_before"))
                    gold_before = _safe_float(floor_row.get("gold_before"))
                    history_context = prior_card_slugs[-40:]
                    avg_history_score = (
                        sum(priors.card_score_for_context(card, floor, history_context) for card in history_context) / len(history_context)
                        if history_context else 0.5
                    )
                    options = []
                    for option in offered:
                        raw_card_id = str(option.get("card_id") or "")
                        card_counter[raw_card_id] += 1
                        raw_slug = _slug(raw_card_id)
                        prior = priors.card(raw_slug) or priors.card(raw_slug.rstrip("+"))
                        context_score = priors.card_score_for_context(raw_slug, floor, history_context)
                        repeat_count = sum(1 for item in history_context if item == raw_slug)
                        options.append({
                            "card_id": raw_card_id,
                            "card_slug": raw_slug,
                            "picked": _safe_int(option.get("was_picked")),
                            "repeat_count": repeat_count,
                            "skada_score_norm": prior.skada_score_norm if prior else 0.5,
                            "pick_rate": prior.pick_rate if prior else 0.0,
                            "win_rate_delta": prior.win_rate_delta if prior else 0.0,
                            "hold_rate": prior.hold_rate if prior else 0.0,
                            "floor_early": prior.floor_early if prior else 0.5,
                            "floor_mid": prior.floor_mid if prior else 0.5,
                            "floor_late": prior.floor_late if prior else 0.5,
                            "context_score": context_score,
                        })

                    room_type = str(floor_row.get("room_type") or "UNKNOWN")
                    room_counter[room_type] += 1
                    groups.append({
                        "offer_id": f"{run_id}:{floor}",
                        "run_id": run_id,
                        "floor": floor,
                        "act": _act_from_floor(floor),
                        "character": character,
                        "ascension": _safe_int(run.get("ascension")),
                        "is_victory": _safe_int(run.get("is_victory")),
                        "floor_reached": _safe_int(run.get("floor_reached")),
                        "duration_sec": _safe_int(run.get("duration_sec")),
                        "room_type": room_type,
                        "hp_before": hp_before,
                        "gold_before": gold_before,
                        "prior_card_count": len(history_context),
                        "prior_relic_count": len(prior_relics),
                        "prior_card_gains": prior_card_gains,
                        "prior_shop_removes": prior_shop_removes,
                        "prior_upgrades": prior_upgrades,
                        "avg_history_context_score": round(avg_history_score, 6),
                        "offer_size": len(options),
                        "chosen_index": chosen_index,
                        "options": options,
                    })

            for option in offered:
                if _safe_int(option.get("was_picked")) == 1:
                    prior_card_slugs.append(_slug(option.get("card_id")))
                    prior_card_gains += 1

            for relic in relics_by_key.get(key, []):
                if _safe_int(relic.get("was_picked")) == 1:
                    rid = str(relic.get("relic_id") or "").lower()
                    if rid:
                        prior_relics.append(rid)

            for action in shops_by_key.get(key, []):
                action_type = str(action.get("action_type") or "")
                item_id = str(action.get("item_id") or "")
                if action_type == "buy_card" and item_id:
                    prior_card_slugs.append(_slug(item_id))
                    prior_card_gains += 1
                elif action_type == "buy_relic" and item_id:
                    prior_relics.append(item_id.lower())
                elif action_type == "remove":
                    prior_shop_removes += 1

            if upgrades_by_key.get(key):
                prior_upgrades += len(upgrades_by_key[key])

    manifest = {
        "num_runs": len(run_ids),
        "num_groups": len(groups),
        "num_option_rows": sum(len(group["options"]) for group in groups),
        "characters": dict(char_counter),
        "room_types": dict(room_counter),
        "top_cards": card_counter.most_common(20),
        "avg_offer_size": round(
            sum(group["offer_size"] for group in groups) / max(1, len(groups)),
            4,
        ),
        "victory_rate": round(
            sum(group["is_victory"] for group in groups) / max(1, len(groups)),
            4,
        ),
    }
    return groups, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Skada card reward choice dataset")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--max-runs", type=int, default=2000, help="Max successful runs to include")
    parser.add_argument("--min-choices", type=int, default=2, help="Minimum cards offered in a reward group")
    parser.add_argument("--character", action="append", default=[],
                        help="Restrict to one character. Repeatable, but intended for per-character datasets.")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON path")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Skada DB not found: {db_path}")

    char_suffix = ""
    normalized_characters = [str(ch).strip().upper() for ch in args.character if str(ch).strip()]
    if len(normalized_characters) == 1:
        char_suffix = f"_{normalized_characters[0].lower()}"
    elif len(normalized_characters) > 1:
        char_suffix = "_multi"
    output_path = Path(args.output) if args.output else (
        ARTIFACTS_ROOT / "skada" / f"card_reward_pick{char_suffix}_{args.max_runs}.jsonl"
    )
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = _dict_factory
    priors = SkadaPriors(db_path)
    groups, manifest = build_dataset(
        conn,
        priors,
        max_runs=args.max_runs,
        min_choices=args.min_choices,
        characters=normalized_characters,
    )
    conn.close()

    with output_path.open("w", encoding="utf-8") as f:
        for row in groups:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest.update({
        "db_path": str(db_path),
        "output_path": str(output_path),
        "max_runs": int(args.max_runs),
        "min_choices": int(args.min_choices),
        "characters_filter": normalized_characters,
    })
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
