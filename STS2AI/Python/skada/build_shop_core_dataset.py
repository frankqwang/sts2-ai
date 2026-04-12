#!/usr/bin/env python3
"""
Build a cleaned shop-core dataset from scraped Skada run details.

Outputs two JSONL datasets:
- shop_visit: stage-1 action type labels
- shop_item_choice: stage-2 item choice labels for buy_card/buy_relic/remove
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
from skada_context import DeckTracker, dict_factory, fetch_many, safe_float, safe_int, slugify
from skada_db import DB_PATH
from skada_priors import SkadaPriors


ACTION_LABELS = {"none", "remove", "buy_card", "buy_relic", "buy_potion", "multi_action"}


def _shop_action_label(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "none"
    if len(actions) == 1:
        action_type = str(actions[0].get("action_type") or "")
        return action_type if action_type in ACTION_LABELS else "multi_action"
    return "multi_action"


def _shared_shop_row(
    run: dict[str, Any],
    floor_row: dict[str, Any],
    tracker: DeckTracker,
    priors: SkadaPriors,
    floor: int,
    recent_damage_taken: deque[float],
    recent_damage_dealt: deque[float],
    recent_rooms: deque[str],
    card_choices: list[dict[str, Any]],
    relic_choices: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    affordable_card_count: int,
    affordable_relic_count: int,
    affordable_potion_count: int,
    affordable_remove_available: int,
) -> dict[str, Any]:
    shared = tracker.shared_context(
        priors=priors,
        floor=floor,
        hp_before=safe_float(floor_row.get("hp_before")),
        gold_before=safe_float(floor_row.get("gold_before")),
        recent_damage_taken=(sum(recent_damage_taken) / len(recent_damage_taken) if recent_damage_taken else 0.0),
        recent_damage_dealt=(sum(recent_damage_dealt) / len(recent_damage_dealt) if recent_damage_dealt else 0.0),
        recent_shop_visits=sum(1 for room in recent_rooms if room == "S"),
        recent_elites=sum(1 for room in recent_rooms if room == "A"),
    )
    shared.update({
        "character": str(run.get("character") or "UNKNOWN").upper(),
        "ascension": safe_int(run.get("ascension")),
        "is_victory": safe_int(run.get("is_victory")),
        "floor_reached": safe_int(run.get("floor_reached")),
        "duration_sec": safe_int(run.get("duration_sec")),
        "room_type": str(floor_row.get("room_type") or "UNKNOWN"),
        "card_option_count": len(card_choices),
        "relic_option_count": len(relic_choices),
        "action_count": len(actions),
        "affordable_card_count": affordable_card_count,
        "affordable_relic_count": affordable_relic_count,
        "affordable_potion_count": affordable_potion_count,
        "affordable_remove_available": int(affordable_remove_available),
    })
    return shared


def build_dataset(
    conn: sqlite3.Connection,
    priors: SkadaPriors,
    max_runs: int = 2000,
    characters: list[str] | None = None,
    min_floor_reached: int = 0,
    max_sample_floor: int = 0,
    min_ascension: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    where_parts = [
        """
        EXISTS (
            SELECT 1
            FROM run_floor_timeline t
            WHERE t.run_id = r.run_id
              AND t.room_type = 'S'
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

    visit_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    visit_counter: Counter[str] = Counter()
    item_counter: Counter[str] = Counter()
    skipped_counter: Counter[str] = Counter()

    for run in run_rows:
        run_id = int(run["run_id"])
        character = str(run.get("character") or "UNKNOWN").upper()
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
            card_choices = choices_by_key.get(key, [])
            relic_choices = relics_by_key.get(key, [])
            actions = shops_by_key.get(key, [])

            if room_type == "S":
                gold_before = safe_float(floor_row.get("gold_before"))
                affordable_card_count = sum(
                    1 for option in card_choices
                    if safe_float(option.get("cost"), 10**9) <= gold_before
                )
                affordable_relic_count = sum(
                    1 for option in relic_choices
                    if safe_float(option.get("cost"), 10**9) <= gold_before
                )
                affordable_potion_count = sum(
                    1 for action in actions
                    if str(action.get("action_type") or "") == "buy_potion"
                )
                affordable_remove_available = any(
                    str(action.get("action_type") or "") == "remove"
                    for action in actions
                )
                shared = _shared_shop_row(
                    run=run,
                    floor_row=floor_row,
                    tracker=tracker,
                    priors=priors,
                    floor=floor,
                    recent_damage_taken=recent_damage_taken,
                    recent_damage_dealt=recent_damage_dealt,
                    recent_rooms=recent_rooms,
                    card_choices=card_choices,
                    relic_choices=relic_choices,
                    actions=actions,
                    affordable_card_count=affordable_card_count,
                    affordable_relic_count=affordable_relic_count,
                    affordable_potion_count=affordable_potion_count,
                    affordable_remove_available=int(affordable_remove_available),
                )
                visit_label = _shop_action_label(actions)
                visit_counter[visit_label] += 1
                visit_rows.append({
                    "visit_id": f"{run_id}:{floor}",
                    "run_id": run_id,
                    "floor": floor,
                    "action_type": visit_label,
                    **shared,
                })

                sibling_buy_cards = [str(action.get("item_id") or "") for action in actions if action.get("action_type") == "buy_card"]
                sibling_buy_relics = [str(action.get("item_id") or "") for action in actions if action.get("action_type") == "buy_relic"]
                history = tracker.current_history()

                for action in actions:
                    family = str(action.get("action_type") or "")
                    chosen_id = str(action.get("item_id") or "")
                    options: list[dict[str, Any]] = []
                    chosen_index = -1

                    if family == "buy_card":
                        ordered_ids: list[str] = []
                        seen_ids: set[str] = set()
                        for option in card_choices:
                            card_id = str(option.get("card_id") or "")
                            if card_id and card_id not in seen_ids:
                                ordered_ids.append(card_id)
                                seen_ids.add(card_id)
                        for card_id in sibling_buy_cards:
                            if card_id and card_id not in seen_ids:
                                ordered_ids.append(card_id)
                                seen_ids.add(card_id)
                        for idx, card_id in enumerate(ordered_ids):
                            slug = slugify(card_id)
                            prior = priors.card(slug)
                            context_score = priors.card_score_for_context(slug, floor, history)
                            options.append({
                                "item_id": card_id,
                                "baseline_score": context_score,
                                "context_score": context_score,
                                "pick_rate": prior.pick_rate if prior else 0.0,
                                "win_rate_delta": prior.win_rate_delta if prior else 0.0,
                                "hold_rate": prior.hold_rate if prior else 0.0,
                                "prior_score": prior.skada_score_norm if prior else 0.5,
                                "deck_synergy": priors.deck_synergy_boost(slug, history),
                                "count_in_deck": tracker.deck_counts.get(slug, 0),
                                "is_basic": 1 if slug.startswith("strike_") or slug.startswith("defend_") else 0,
                                "is_curse": 1 if slug.endswith("bane") else 0,
                            })
                            if card_id.upper() == chosen_id.upper():
                                chosen_index = idx
                    elif family == "buy_relic":
                        ordered_ids = []
                        seen_ids: set[str] = set()
                        for option in relic_choices:
                            relic_id = str(option.get("relic_id") or "")
                            if relic_id and relic_id not in seen_ids:
                                ordered_ids.append(relic_id)
                                seen_ids.add(relic_id)
                        for relic_id in sibling_buy_relics:
                            if relic_id and relic_id not in seen_ids:
                                ordered_ids.append(relic_id)
                                seen_ids.add(relic_id)
                        for idx, relic_id in enumerate(ordered_ids):
                            slug = slugify(relic_id)
                            prior = priors.relic(slug)
                            baseline_score = safe_float(prior.win_rate_owned, 0.5) if prior else 0.5
                            options.append({
                                "item_id": relic_id,
                                "baseline_score": baseline_score,
                                "context_score": baseline_score,
                                "pick_rate": prior.pick_rate if prior else 0.0,
                                "win_rate_delta": prior.win_rate_delta if prior else 0.0,
                                "hold_rate": prior.hold_rate if prior else 0.0,
                                "prior_score": safe_float(prior.win_rate_owned, 0.5) if prior else 0.5,
                                "deck_synergy": 0.0,
                                "count_in_deck": 0,
                                "is_basic": 0,
                                "is_curse": 0,
                            })
                            if relic_id.upper() == chosen_id.upper():
                                chosen_index = idx
                    elif family == "remove":
                        options = tracker.remove_candidates(priors, floor, chosen_card_id=chosen_id)
                        chosen_index = next(
                            (idx for idx, option in enumerate(options) if str(option["item_id"]).upper() == chosen_id.upper()),
                            -1,
                        )

                    if family in {"buy_card", "buy_relic", "remove"}:
                        if chosen_index >= 0 and options:
                            item_counter[family] += 1
                            item_rows.append({
                                "choice_id": f"{run_id}:{floor}:{family}:{safe_int(action.get('action_index'))}",
                                "run_id": run_id,
                                "floor": floor,
                                "choice_family": family,
                                "chosen_index": chosen_index,
                                "offer_size": len(options),
                                "chosen_item_id": chosen_id,
                                **shared,
                                "options": options,
                            })
                        else:
                            skipped_counter[family] += 1

            for option in card_choices:
                if safe_int(option.get("was_picked")) == 1:
                    tracker.add_card(option.get("card_id"))

            for relic in relic_choices:
                if safe_int(relic.get("was_picked")) == 1:
                    tracker.add_relic(relic.get("relic_id"))

            for action in actions:
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
        "num_visit_samples": len(visit_rows),
        "num_item_samples": len(item_rows),
        "visit_action_counts": dict(visit_counter),
        "item_family_counts": dict(item_counter),
        "skipped_item_samples": dict(skipped_counter),
        "avg_shop_floor": round(sum(row["floor"] for row in visit_rows) / max(1, len(visit_rows)), 4),
    }
    return visit_rows, item_rows, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Skada shop-core dataset")
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
    parser.add_argument("--visit-output", default=None, help="Output JSONL path for shop_visit samples")
    parser.add_argument("--item-output", default=None, help="Output JSONL path for shop_item_choice samples")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON path")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Skada DB not found: {db_path}")

    normalized_characters = [str(ch).strip().upper() for ch in args.character if str(ch).strip()]
    char_prefix = "multi" if len(normalized_characters) > 1 else (
        normalized_characters[0].lower() if len(normalized_characters) == 1 else "all"
    )
    base_dir = db_path.resolve().parents[3] / "artifacts" / "skada"
    visit_output = Path(args.visit_output) if args.visit_output else (base_dir / f"{char_prefix}_shop_visit.jsonl")
    item_output = Path(args.item_output) if args.item_output else (base_dir / f"{char_prefix}_shop_item_choice.jsonl")
    manifest_path = Path(args.manifest) if args.manifest else (base_dir / f"{char_prefix}_shop_core.manifest.json")
    visit_output.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = dict_factory
    priors = SkadaPriors(db_path)
    visit_rows, item_rows, manifest = build_dataset(
        conn,
        priors,
        max_runs=args.max_runs,
        characters=normalized_characters,
        min_floor_reached=args.min_floor_reached,
        max_sample_floor=args.max_sample_floor,
        min_ascension=args.min_ascension,
    )
    conn.close()

    with visit_output.open("w", encoding="utf-8") as f:
        for row in visit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with item_output.open("w", encoding="utf-8") as f:
        for row in item_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest.update({
        "db_path": str(db_path),
        "visit_output": str(visit_output),
        "item_output": str(item_output),
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
