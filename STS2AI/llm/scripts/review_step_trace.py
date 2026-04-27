"""Review LLM step traces into turn/combat lessons.

This is the no-API review pass. It summarizes what happened, extracts obvious
mechanism misses from existing quality flags, and writes compact lessons that
can be fed back into the local experience library.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.data_pipeline.experience_library import (  # noqa: E402
    DEFAULT_EXPERIENCE_PATH,
    ExperienceEntry,
    append_experience,
    load_experience,
)
from llm.paths import ARTIFACTS_ROOT, ensure_dirs  # noqa: E402
from llm.scripts.analyze_action_ordering import (  # noqa: E402
    _chosen,
    _energy,
    _enemies,
    _legal_actions,
    _round,
    analyze as analyze_ordering,
)


_MATH_GE_RE = re.compile(r"\b(-?\d+(?:\.\d+)?)\s*>=\s*(-?\d+(?:\.\d+)?)\b")
_FLOOR_RE = re.compile(r"\bfloor=(\d+|\?)\b")
_PLAYER_HP_RE = re.compile(r"\bplayer:\s+hp=(\d+)/(\d+)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="step_trace.jsonl")
    parser.add_argument("--out-dir", default="", help="default: Artifacts/llm/reviews/<trace-stem>_<timestamp>")
    parser.add_argument("--experience-path", default=str(DEFAULT_EXPERIENCE_PATH))
    parser.add_argument("--append-experience", action="store_true")
    parser.add_argument("--examples", type=int, default=12)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _reason(row: dict[str, Any]) -> str:
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    return str(decoded.get("reason") or "")


def _row_step(row: dict[str, Any]) -> int:
    try:
        return int(row.get("episode_step") if row.get("episode_step") is not None else row.get("step") or 0)
    except (TypeError, ValueError):
        return 0


def _floor(user: str) -> int | None:
    match = _FLOOR_RE.search(user)
    if not match or match.group(1) == "?":
        return None
    return int(match.group(1))


def _player_hp(user: str) -> tuple[int, int] | None:
    match = _PLAYER_HP_RE.search(user)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _action_label(chosen: dict[str, Any]) -> str:
    card = str(chosen.get("card_id") or "")
    target = str(chosen.get("target") or "")
    index = chosen.get("index")
    if card.lower() == "end_turn":
        return f"[{index}] end_turn"
    if card:
        return f"[{index}] {card}" + (f" -> {target}" if target else "")
    return f"[{index}] action"


def _reason_flags(row: dict[str, Any]) -> list[str]:
    reason = _reason(row)
    flags: list[str] = []
    for match in _MATH_GE_RE.finditer(reason):
        try:
            left = float(match.group(1))
            right = float(match.group(2))
        except ValueError:
            continue
        if left + 1e-9 < right:
            flags.append("reason_math_contradiction")
            break
    return flags


def _existing_lesson_keys(path: Path) -> set[tuple[str, str, str]]:
    return {
        (entry.applies_when, entry.advice, entry.avoid)
        for entry in load_experience(path)
    }


def _lesson(
    *,
    tags: list[str],
    applies_when: str,
    advice: str,
    avoid: str = "",
    source: str,
    confidence: float,
) -> ExperienceEntry:
    return ExperienceEntry(
        tags=tags,
        applies_when=applies_when,
        advice=advice,
        avoid=avoid,
        source=source,
        confidence=confidence,
    )


def _lesson_candidates_for_step(row: dict[str, Any], chosen: dict[str, Any]) -> list[ExperienceEntry]:
    flags = list(row.get("quality_flags") or [])
    flags.extend(_reason_flags(row))
    lessons: list[ExperienceEntry] = []
    source = f"trace_review:{row.get('episode_id')}:{row.get('episode_step', row.get('step'))}"
    if "missed_visible_lethal" in flags:
        lessons.append(_lesson(
            tags=["lethal", "targeting", "tempo"],
            applies_when="a legal action visibly kills an enemy this turn",
            advice="take the visible lethal before setup or end_turn unless another legal action wins more immediately",
            avoid="do not leave a killable enemy alive when the prompt lists lethal damage",
            source=source,
            confidence=0.75,
        ))
    if "dangerous_end_turn" in flags:
        lessons.append(_lesson(
            tags=["defense", "end_turn", "enemy_intent"],
            applies_when="enemy intent shows incoming damage and useful play_card actions are still legal",
            advice="use legal block, kill, or mitigation actions before ending the turn",
            avoid="do not end_turn into avoidable incoming damage",
            source=source,
            confidence=0.7,
        ))
    if "end_turn_with_playable_cards" in flags or "floating_energy_end_turn" in flags:
        lessons.append(_lesson(
            tags=["energy", "end_turn", "tempo"],
            applies_when="energy remains and playable cards are legal",
            advice="spend energy on useful damage, block, draw, or setup before end_turn",
            avoid="do not end_turn with unused energy and useful legal cards",
            source=source,
            confidence=0.65,
        ))
    if "reason_math_contradiction" in flags:
        lessons.append(_lesson(
            tags=["reason", "targeting", "lethal"],
            applies_when="explaining a selected action with damage and target numbers",
            advice="make the reason match the selected action, target, damage, and enemy HP",
            avoid="do not claim lethal or mention a different enemy when the chosen action does not support it",
            source=source,
            confidence=0.7,
        ))
    return lessons


def _turn_key(row: dict[str, Any]) -> tuple[str, int, int]:
    user = str(row.get("user_message") or "")
    floor_no = _floor(user)
    if floor_no is None:
        floor_no = -1
    round_no = _round(user)
    if round_no is None:
        round_no = -1
    return str(row.get("episode_id") or ""), int(floor_no), int(round_no)


def _step_summary(row: dict[str, Any]) -> dict[str, Any]:
    user = str(row.get("user_message") or "")
    actions = _legal_actions(row)
    chosen = _chosen(row, actions)
    flags = list(row.get("quality_flags") or [])
    flags.extend(_reason_flags(row))
    quality = row.get("quality_report") if isinstance(row.get("quality_report"), dict) else {}
    metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
    return {
        "step": row.get("episode_step", row.get("step")),
        "floor": _floor(user),
        "round": _round(user),
        "hp": _player_hp(user),
        "energy": _energy(user),
        "chosen": chosen,
        "action": _action_label(chosen),
        "reason": _reason(row),
        "flags": flags,
        "opportunities": quality.get("opportunities") or {},
        "misses": quality.get("misses") or {},
        "metrics": metrics,
    }


def _review_turns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_rows = sorted(
        rows,
        key=lambda row: (str(row.get("episode_id") or ""), _row_step(row)),
    )
    reviews: list[dict[str, Any]] = []
    i = 0
    while i < len(ordered_rows):
        episode_id, floor_no, round_no = _turn_key(ordered_rows[i])
        j = i + 1
        while j < len(ordered_rows) and _turn_key(ordered_rows[j]) == (episode_id, floor_no, round_no):
            j += 1
        ordered = ordered_rows[i:j]
        steps = [_step_summary(row) for row in ordered]
        flag_counts = Counter(flag for step in steps for flag in step["flags"])
        actions = [step["action"] for step in steps]
        start_hp = steps[0].get("hp") if steps else None
        next_hp = _player_hp(str(ordered_rows[j].get("user_message") or "")) if j < len(ordered_rows) else None
        end_hp = next_hp or (steps[-1].get("hp") if steps else None)
        observed_loss = 0
        if isinstance(start_hp, tuple) and isinstance(end_hp, tuple):
            observed_loss = max(0, int(start_hp[0]) - int(end_hp[0]))
        incoming_values = [
            float((step.get("metrics") or {}).get("incoming_damage") or 0.0)
            for step in steps
        ]
        current_loss_values = [
            float((step.get("metrics") or {}).get("current_hp_loss") or 0.0)
            for step in steps
        ]
        lessons: list[dict[str, Any]] = []
        for row, step in zip(ordered, steps):
            chosen = step["chosen"] if isinstance(step.get("chosen"), dict) else {}
            for entry in _lesson_candidates_for_step(row, chosen):
                lessons.append(entry.to_json())
        if observed_loss >= 8 and round_no >= 0:
            lessons.append(_lesson(
                tags=["defense", "turn_planning", "hp_loss"],
                applies_when="a whole combat turn is expected to lose significant HP",
                advice="compare the complete turn plan: lethal, block, mitigation, potion, then damage; prefer the sequence with the lowest survival risk",
                avoid="do not rank each attack independently while ignoring the turn's final HP loss",
                source=f"trace_turn_review:{episode_id}:floor{floor_no}:round{round_no}",
                confidence=0.7,
            ).to_json())
        reviews.append({
            "episode_id": episode_id,
            "floor": floor_no if floor_no >= 0 else None,
            "round": round_no if round_no >= 0 else None,
            "encounter_id": ordered[0].get("encounter_id"),
            "encounter_label": ordered[0].get("encounter_label"),
            "outcome": ordered[0].get("outcome"),
            "steps": len(steps),
            "start_hp": start_hp[0] if isinstance(start_hp, tuple) else None,
            "end_hp": end_hp[0] if isinstance(end_hp, tuple) else None,
            "observed_hp_loss": observed_loss,
            "incoming_damage_max": max(incoming_values) if incoming_values else 0.0,
            "current_hp_loss_max": max(current_loss_values) if current_loss_values else 0.0,
            "actions": actions,
            "flag_counts": {key: int(value) for key, value in flag_counts.most_common()},
            "lesson_candidates": lessons,
            "step_reviews": steps,
        })
        i = j
    return reviews


def _review_combats(rows: list[dict[str, Any]], turn_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("episode_id") or "")].append(row)
    turns_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in turn_reviews:
        turns_by_episode[str(review.get("episode_id") or "")].append(review)

    reviews: list[dict[str, Any]] = []
    for episode_id, episode_rows in sorted(grouped.items()):
        ordered = sorted(episode_rows, key=lambda row: int(row.get("episode_step") or row.get("step") or 0))
        flag_counts = Counter()
        lessons: list[dict[str, Any]] = []
        action_sequence: list[str] = []
        for row in ordered:
            summary = _step_summary(row)
            action_sequence.append(summary["action"])
            flag_counts.update(summary["flags"])
            for entry in _lesson_candidates_for_step(row, summary["chosen"]):
                lessons.append(entry.to_json())
        first = ordered[0] if ordered else {}
        reward = first.get("episode_reward") if isinstance(first.get("episode_reward"), dict) else {}
        reviews.append({
            "episode_id": episode_id,
            "encounter_id": first.get("encounter_id"),
            "encounter_label": first.get("encounter_label"),
            "seed": first.get("seed"),
            "outcome": first.get("outcome"),
            "reward_total": reward.get("total"),
            "steps": len(ordered),
            "turns": len(turns_by_episode.get(episode_id, [])),
            "flag_counts": {key: int(value) for key, value in flag_counts.most_common()},
            "action_sequence": action_sequence,
            "lesson_candidates": lessons,
        })
    return reviews


def _hard_cases(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        summary = _step_summary(row)
        flags = summary["flags"]
        if not flags:
            continue
        cases.append({
            "episode_id": row.get("episode_id"),
            "step": row.get("episode_step", row.get("step")),
            "encounter_id": row.get("encounter_id"),
            "round": summary["round"],
            "chosen": summary["chosen"],
            "reason": summary["reason"],
            "flags": flags,
            "user_message": row.get("user_message"),
            "raw_generation": row.get("raw_generation"),
        })
        if len(cases) >= limit:
            break
    return cases


def _damage_turn_hard_cases(turn_reviews: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    cases = [
        {
            "episode_id": review.get("episode_id"),
            "floor": review.get("floor"),
            "round": review.get("round"),
            "encounter_id": review.get("encounter_id"),
            "encounter_label": review.get("encounter_label"),
            "outcome": review.get("outcome"),
            "steps": review.get("steps"),
            "start_hp": review.get("start_hp"),
            "end_hp": review.get("end_hp"),
            "observed_hp_loss": review.get("observed_hp_loss"),
            "incoming_damage_max": review.get("incoming_damage_max"),
            "current_hp_loss_max": review.get("current_hp_loss_max"),
            "actions": review.get("actions") or [],
            "flag_counts": review.get("flag_counts") or {},
            "step_reviews": review.get("step_reviews") or [],
        }
        for review in turn_reviews
        if isinstance(review.get("observed_hp_loss"), (int, float))
        and float(review.get("observed_hp_loss") or 0) > 0
        and review.get("round") is not None
    ]
    cases.sort(
        key=lambda row: (
            float(row.get("observed_hp_loss") or 0),
            float(row.get("incoming_damage_max") or 0),
            int(row.get("steps") or 0),
        ),
        reverse=True,
    )
    return cases[:limit]


def _dedupe_lessons(entries: list[ExperienceEntry]) -> list[ExperienceEntry]:
    seen: set[tuple[str, str, str]] = set()
    out: list[ExperienceEntry] = []
    for entry in entries:
        key = (entry.applies_when, entry.advice, entry.avoid)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _collect_lessons(turn_reviews: list[dict[str, Any]], combat_reviews: list[dict[str, Any]]) -> list[ExperienceEntry]:
    entries: list[ExperienceEntry] = []
    for review in [*turn_reviews, *combat_reviews]:
        for raw in review.get("lesson_candidates") or []:
            if not isinstance(raw, dict):
                continue
            entries.append(ExperienceEntry(
                tags=[str(tag) for tag in (raw.get("tags") or [])],
                applies_when=str(raw.get("applies_when") or ""),
                advice=str(raw.get("advice") or ""),
                avoid=str(raw.get("avoid") or ""),
                source=str(raw.get("source") or ""),
                confidence=float(raw.get("confidence") or 0.5),
            ))
    return _dedupe_lessons(entries)


def _default_out_dir(trace_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent = trace_path.parent.name or trace_path.stem
    return ARTIFACTS_ROOT / "reviews" / f"{parent}_{stamp}"


def main() -> int:
    args = parse_args()
    ensure_dirs()
    trace_path = Path(args.trace).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_out_dir(trace_path)
    rows = _read_jsonl(trace_path)
    turn_reviews = _review_turns(rows)
    combat_reviews = _review_combats(rows, turn_reviews)
    hard_cases = _hard_cases(rows, limit=200)
    damage_turn_hard_cases = _damage_turn_hard_cases(turn_reviews, limit=200)
    ordering = analyze_ordering(rows, example_limit=max(0, args.examples))
    lessons = _collect_lessons(turn_reviews, combat_reviews)

    _write_jsonl(out_dir / "turn_reviews.jsonl", turn_reviews)
    _write_jsonl(out_dir / "combat_reviews.jsonl", combat_reviews)
    _write_jsonl(out_dir / "hard_cases.jsonl", hard_cases)
    _write_jsonl(out_dir / "damage_turn_hard_cases.jsonl", damage_turn_hard_cases)
    _write_jsonl(out_dir / "lessons.jsonl", [entry.to_json() for entry in lessons])
    _write_json(out_dir / "ordering_summary.json", ordering)

    experience_path = Path(args.experience_path)
    appended = 0
    if args.append_experience and lessons:
        existing = _existing_lesson_keys(experience_path)
        new_entries = [
            entry for entry in lessons
            if (entry.applies_when, entry.advice, entry.avoid) not in existing
        ]
        if new_entries:
            append_experience(new_entries, experience_path)
            appended = len(new_entries)

    flag_counts = Counter()
    for case in hard_cases:
        flag_counts.update(case.get("flags") or [])
    summary = {
        "kind": "step_trace_review",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "trace_path": str(trace_path),
        "out_dir": str(out_dir),
        "steps": len(rows),
        "turn_reviews": len(turn_reviews),
        "combat_reviews": len(combat_reviews),
        "hard_cases": len(hard_cases),
        "damage_turn_hard_cases": len(damage_turn_hard_cases),
        "lesson_candidates": len(lessons),
        "experience_appended": appended,
        "flag_counts": {key: int(value) for key, value in flag_counts.most_common()},
        "ordering_counts": ordering.get("counts") or {},
        "outputs": {
            "turn_reviews": str(out_dir / "turn_reviews.jsonl"),
            "combat_reviews": str(out_dir / "combat_reviews.jsonl"),
            "hard_cases": str(out_dir / "hard_cases.jsonl"),
            "damage_turn_hard_cases": str(out_dir / "damage_turn_hard_cases.jsonl"),
            "lessons": str(out_dir / "lessons.jsonl"),
            "ordering_summary": str(out_dir / "ordering_summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
