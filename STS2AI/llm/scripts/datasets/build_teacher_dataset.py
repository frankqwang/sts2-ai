"""Build teacher-repair SFT data from turn reviews and verified trace rules.

Inputs can be:
- turn-order review JSON files produced by Kimi or manual teacher review
- the matching episode_input.json files
- step_trace.jsonl files for conservative local repairs
- review_reselect_actions results

The output is a normal chat SFT/GRPO-lite dataset:
  train.jsonl / eval.jsonl

Every label is verified against the current prompt's listed legal_actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.action_quality import assess_action_quality_report  # noqa: E402


def _coerce_confidence(value: Any) -> float:
    """安全 cast confidence；bool 直接拒（避免 LLM 偶尔回 True 被错当 1.0）。"""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _coerce_tag_list(value: Any) -> list[str]:
    """tags 字段防御：LLM 偶尔回单 string，Python `[str(t) for t in s]` 会按字符拆开。"""
    if value is None:
        return []
    if isinstance(value, str):
        # 当成单元素，避免按字符拆开
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]
from llm.data_pipeline.experience_library import (  # noqa: E402
    DEFAULT_EXPERIENCE_PATH,
    ExperienceEntry,
    append_experience,
    load_experience,
)
from llm.paths import ARTIFACTS_ROOT, DATASETS_ROOT, ensure_dirs  # noqa: E402
from llm.prompts import load_system_prompt  # noqa: E402
from llm.scripts.analysis.analyze_action_ordering import (  # noqa: E402
    _chosen,
    _energy,
    _legal_actions,
)


_HAND_RE = re.compile(r"^\s*\[(\d+)]\s+([A-Z0-9_+]+)\s+cost=([0-9]+|X|\?)\b(.*)$", re.MULTILINE)
_ENEMY_RE = re.compile(
    r"^\s*(enemy\d+):\s+(\S+)\s+hp=(\d+)/(\d+)\s+block=(\d+)\s+intent=([^\s]+(?:\([^)]+\))?)\s+powers=(.*)$",
    re.MULTILINE,
)
_RETURN_LINE_RE = re.compile(r"^Return (?:one JSON line|strict JSON only): .*$", re.MULTILINE)
# combat policy 输出不再带 reason 字段——推理是 planner LoRA 的 job, combat 只选 action.
# 老的 SFT 数据（含 reason）通过下文 ``_normalize_user_message_schema`` 重写成新格式,
# 以避免训练时 mixed schema 让模型困惑.
_CURRENT_RETURN_LINE = (
    'Return strict JSON only: {"action_index":N,"confidence":0.0} '
    "using one listed action_index. Do not output multiple objects or candidates. "
    "Do not include a reason / plan / extra keys — strategy text belongs to the planner model."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", default=[], help="step_trace.jsonl for local verified repairs.")
    parser.add_argument("--review", action="append", default=[], help="turn_order_review/codex review JSON.")
    parser.add_argument("--episode-input", action="append", default=[], help="matching episode_input.json for review labels.")
    parser.add_argument(
        "--review-root",
        action="append",
        default=[],
        help="Directory containing per-episode turn_order_review.json and sibling episode_input.json files.",
    )
    parser.add_argument("--reselect-results", action="append", default=[], help="review_reselect_actions results JSONL.")
    parser.add_argument("--kimi-labels", action="append", default=[], help="filtered Kimi valid/kept labels JSONL.")
    parser.add_argument(
        "--keep-kimi-reasons",
        action="store_true",
        help="Use Kimi's original reason text in `_rows_from_kimi_labels` (filtered Kimi labels jsonl). "
             "By default Kimi labels are rewritten to short verified canonical reasons.",
    )
    parser.add_argument(
        "--no-kimi-reasons-in-review",
        action="store_true",
        help="Disable using Kimi's reason_en in turn_order_review SFT rows; fall back to canonical template "
             "('deal X damage to enemy'). Default is to use Kimi's mechanically-correct reason "
             "(fixes 'all cards described as Deal X damage' hallucination).",
    )
    parser.add_argument("--out-dir", default=str(DATASETS_ROOT / f"teacher_turn_repair_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--append-experience", action="store_true")
    parser.add_argument("--experience-path", default=str(DEFAULT_EXPERIENCE_PATH))
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _review_pairs_from_roots(roots: list[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if not root.exists():
            continue
        for review_path in sorted(root.rglob("turn_order_review.json")):
            episode_path = review_path.with_name("episode_input.json")
            if not episode_path.exists():
                continue
            key = (review_path.resolve(), episode_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def _user_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _score_note_for_action(user_message: str, action: dict[str, Any] | None) -> str:
    if not action:
        return "verified legal action"
    enemies = _enemies(user_message)
    target = str(action.get("target") or "")
    damage = action.get("damage")
    block = action.get("block")
    if isinstance(damage, int) and damage > 0 and target:
        enemy = enemies.get(target)
        if enemy and damage >= _enemy_effective_hp(enemy) > 0:
            return f"lethal {target}: damage={damage} target_hp={_enemy_effective_hp(enemy)}"
        return f"damage {target}: damage={damage}"
    if isinstance(block, int) and block > 0:
        return f"gain block={block}"
    if _is_end_turn(action):
        return "end_turn"
    card_id = str(action.get("card_id") or "").strip()
    return f"play {card_id}" if card_id else "verified legal action"


def _rough_action_score(user_message: str, action: dict[str, Any], selected_index: int) -> float:
    enemies = _enemies(user_message)
    index = int(action.get("index") or -1)
    if index == selected_index:
        return 10.0
    target = str(action.get("target") or "")
    damage = action.get("damage")
    block = action.get("block")
    if isinstance(damage, int) and damage > 0 and target:
        enemy = enemies.get(target)
        if enemy and damage >= _enemy_effective_hp(enemy) > 0:
            return 7.0
        return 3.0 + min(4.0, damage / 10.0)
    if isinstance(block, int) and block > 0:
        return 2.0 + min(3.0, block / 5.0)
    if _is_end_turn(action):
        return 0.0
    return 1.0


def _action_scores_for_label(user_message: str, action_index: int) -> list[dict[str, Any]]:
    actions = _legal_actions({"user_message": user_message})
    selected = _action_by_index(actions, action_index)
    ordered: list[dict[str, Any]] = []
    if selected is not None:
        ordered.append(selected)
    alternatives = [
        action for action in actions
        if isinstance(action.get("index"), int) and int(action["index"]) != action_index
    ]
    alternatives.sort(key=lambda action: _rough_action_score(user_message, action, action_index), reverse=True)
    ordered.extend(alternatives[: max(0, 4 - len(ordered))])

    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for action in ordered[:4]:
        raw_index = action.get("index")
        if not isinstance(raw_index, int) or raw_index in seen:
            continue
        seen.add(raw_index)
        out.append({
            "action_index": raw_index,
            "score": round(_rough_action_score(user_message, action, action_index), 2),
            "note": _score_note_for_action(user_message, action),
        })
    if not out:
        out.append({"action_index": int(action_index), "score": 10.0, "note": "verified legal action"})
    return out


def _json_action(
    action_index: int,
    *,
    user_message: str = "",
    confidence: float = 0.9,
) -> str:
    """Combat policy SFT label content.

    Reason / plan text is no longer part of the combat output schema —
    those belong to the planner LoRA. ``user_message`` is unused but
    kept for API compatibility with older callers (build_combat_training_pool).
    """
    _ = user_message  # kept for signature stability with legacy callers
    payload = {
        "action_index": int(action_index),
        "confidence": max(0.0, min(1.0, float(confidence))),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _normalize_user_message_schema(user_message: str) -> str:
    if not user_message:
        return user_message
    if _RETURN_LINE_RE.search(user_message):
        return _RETURN_LINE_RE.sub(_CURRENT_RETURN_LINE, user_message)
    return f"{user_message.rstrip()}\n{_CURRENT_RETURN_LINE}"


def _action_by_index(actions: list[dict[str, Any]], action_index: int | None) -> dict[str, Any] | None:
    if action_index is None:
        return None
    for action in actions:
        if action.get("index") == action_index:
            return action
    return None


def _is_end_turn(action: dict[str, Any] | None) -> bool:
    if not action:
        return False
    return str(action.get("card_id") or "").lower() == "end_turn" or "end_turn" in str(action.get("raw") or "").lower()


def _hand_info(user_message: str) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for match in _HAND_RE.finditer(user_message):
        hand, card, cost, rest = match.groups()
        parsed_cost = 99
        if cost.isdigit():
            parsed_cost = int(cost)
        text = rest.lower()
        info[hand] = {
            "card_id": card.replace("+", ""),
            "cost": parsed_cost,
            "draw": "draw" in text,
            "exhaust": "exhaust" in text,
            "text": rest.strip(),
        }
    return info


def _enemies(user_message: str) -> dict[str, dict[str, Any]]:
    enemies: dict[str, dict[str, Any]] = {}
    for match in _ENEMY_RE.finditer(user_message):
        label, name, hp, max_hp, block, intent, powers = match.groups()
        enemies[label] = {
            "label": label,
            "name": name,
            "hp": int(hp),
            "max_hp": int(max_hp),
            "block": int(block),
            "intent": intent,
            "attacking": "Attack" in intent,
            "powers": powers.strip(),
            "vulnerable": "VULNERABLE_POWER" in powers,
        }
    return enemies


def _action_cost(action: dict[str, Any], hand: dict[str, dict[str, Any]]) -> int:
    return int((hand.get(str(action.get("hand") or "")) or {}).get("cost", 99))


def _action_draws(action: dict[str, Any], hand: dict[str, dict[str, Any]]) -> bool:
    return bool((hand.get(str(action.get("hand") or "")) or {}).get("draw"))


def _enemy_effective_hp(enemy: dict[str, Any] | None) -> int:
    if not enemy:
        return 0
    return int(enemy.get("hp") or 0) + int(enemy.get("block") or 0)


def _is_lethal(action: dict[str, Any] | None, enemies: dict[str, dict[str, Any]]) -> bool:
    if not action:
        return False
    target = str(action.get("target") or "")
    damage = action.get("damage")
    return target in enemies and isinstance(damage, int) and damage >= _enemy_effective_hp(enemies[target]) > 0


def _lethal_actions(actions: list[dict[str, Any]], enemies: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [action for action in actions if _is_lethal(action, enemies)]


def _rank_lethal(action: dict[str, Any], enemies: dict[str, dict[str, Any]], hand: dict[str, dict[str, Any]]) -> tuple[int, int, int, int]:
    target = str(action.get("target") or "")
    enemy = enemies.get(target) or {}
    attacking = 0 if enemy.get("attacking") else 1
    cost = _action_cost(action, hand)
    damage = int(action.get("damage") or 0)
    overkill = max(0, damage - _enemy_effective_hp(enemy))
    draw_bonus = 0 if _action_draws(action, hand) else 1
    return attacking, cost, draw_bonus, overkill


def _original_index(row: dict[str, Any], actions: list[dict[str, Any]]) -> int | None:
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    if isinstance(decoded.get("action_index"), int):
        return int(decoded["action_index"])
    chosen = _chosen(row, actions)
    raw = chosen.get("index")
    return int(raw) if isinstance(raw, int) else None


def _sample(
    *,
    user_message: str,
    action_index: int,
    meta: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    """Build one combat SFT sample (no reason field).

    The combat policy outputs only ``{action_index, confidence}``; any
    teacher-supplied prose reason is discarded here so we don't smuggle
    it back into the supervision target. Diagnostic fields (the original
    teacher reason text, source rule, etc.) live in ``meta`` for trace
    inspection but never appear in the assistant message.
    """
    confidence = float(meta.get("confidence") or 0.9)
    user_message = _normalize_user_message_schema(user_message)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {
                "role": "assistant",
                "content": _json_action(
                    action_index,
                    user_message=user_message,
                    confidence=confidence,
                ),
            },
        ],
        "meta": {
            "advantage": 2.0,
            "teacher": "turn_repair",
            **meta,
            "teacher_action_index": int(action_index),
        },
    }


def _candidate_from_trace_row(row: dict[str, Any]) -> dict[str, Any] | None:
    user = str(row.get("user_message") or "")
    if not user:
        return None
    actions = _legal_actions({"user_message": user})
    if not actions:
        return None
    original_index = _original_index(row, actions)
    original = _action_by_index(actions, original_index)
    if original is None:
        return None

    hand = _hand_info(user)
    enemies = _enemies(user)
    lethal = _lethal_actions(actions, enemies)
    if not lethal:
        return None

    original_target = str((original or {}).get("target") or "")
    original_enemy = enemies.get(original_target)

    # 1. If the model kills a non-attacking enemy while an attacking enemy is killable,
    # teach attacking-target priority.
    if _is_lethal(original, enemies) and original_enemy and not original_enemy.get("attacking"):
        attacking_lethal = [action for action in lethal if (enemies.get(str(action.get("target") or "")) or {}).get("attacking")]
        if attacking_lethal:
            best = sorted(attacking_lethal, key=lambda action: _rank_lethal(action, enemies, hand))[0]
            if best.get("index") != original_index:
                return {
                    "action_index": int(best["index"]),
                    "rule": "attacking_lethal_priority",
                }

    # 2. Missed visible lethal: prefer the best lethal, especially attacking targets.
    if not _is_lethal(original, enemies):
        best = sorted(lethal, key=lambda action: _rank_lethal(action, enemies, hand))[0]
        if best.get("index") != original_index:
            return {
                "action_index": int(best["index"]),
                "rule": "visible_lethal",
            }

    # 3. Same-target cheap lethal beats high-cost overkill when it preserves energy/draws.
    if _is_lethal(original, enemies) and original_target:
        original_cost = _action_cost(original, hand)
        same_target = [action for action in lethal if str(action.get("target") or "") == original_target]
        cheaper = [
            action for action in same_target
            if _action_cost(action, hand) < original_cost
            and (_action_cost(action, hand) <= original_cost - 2 or _action_draws(action, hand))
        ]
        if cheaper:
            best = sorted(cheaper, key=lambda action: _rank_lethal(action, enemies, hand))[0]
            if best.get("index") != original_index:
                return {
                    "action_index": int(best["index"]),
                    "rule": "cheap_lethal_over_overkill",
                }
    return None


# NOTE: ``_reason_repair_from_trace_row`` was deleted intentionally.
# Its sole purpose was rewriting an inconsistent reason field to a
# canonical template; with the v4 design split (combat policy emits no
# reason, planner LoRA owns reasoning) the function had nothing to do.
# Step traces flagged with reason_math_contradiction etc. are still
# tagged in ``quality_flags`` and excluded from training via the existing
# ``TRAINING_BLOCKLIST_FLAGS`` blocklist.


def _validate_label(user: str, action_index: int) -> tuple[bool, dict[str, Any] | None]:
    actions = _legal_actions({"user_message": user})
    action = _action_by_index(actions, action_index)
    return action is not None and not _is_end_turn(action), action


def _preserves_visible_attacking_lethal(user: str, action: dict[str, Any]) -> bool:
    actions = _legal_actions({"user_message": user})
    enemies = _enemies(user)
    attacking_lethals = [
        candidate for candidate in _lethal_actions(actions, enemies)
        if (enemies.get(str(candidate.get("target") or "")) or {}).get("attacking")
    ]
    if not attacking_lethals:
        return True
    selected_target = str(action.get("target") or "")
    selected_enemy = enemies.get(selected_target) or {}
    return bool(selected_enemy.get("attacking") and _is_lethal(action, enemies))


def _rows_from_trace(trace_path: Path, *, system_prompt: str) -> list[dict[str, Any]]:
    """Mine deterministic better-action candidates from a step_trace.

    For each step where ``_candidate_from_trace_row`` finds a clearly
    better action_index (visible-lethal / attacking-target priority /
    cheap-lethal-over-overkill), emit one combat SFT row whose assistant
    content is just ``{action_index, confidence}`` — no reason field.
    """
    out: list[dict[str, Any]] = []
    for row in _read_jsonl(trace_path):
        user = str(row.get("user_message") or "")
        candidate = _candidate_from_trace_row(row)
        if not candidate:
            continue
        ok, action = _validate_label(user, int(candidate["action_index"]))
        if not ok:
            continue
        out.append(_sample(
            user_message=user,
            action_index=int(candidate["action_index"]),
            meta={
                "source": "trace_rule",
                "source_trace": str(trace_path),
                "source_rule": candidate["rule"],
                "episode_id": row.get("episode_id"),
                "step": row.get("episode_step", row.get("step")),
                "original_action_index": _original_index(row, _legal_actions({"user_message": user})),
                "teacher_action": action,
                "action_quality_flags": list(row.get("quality_flags") or []),
            },
            system_prompt=system_prompt,
        ))
    return out


def _episode_decision_by_step(episode: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for turn in episode.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        for decision in turn.get("decisions") or []:
            if isinstance(decision, dict) and isinstance(decision.get("step"), int):
                out[int(decision["step"])] = decision
    return out


def _rows_from_review(
    review_path: Path,
    episode_path: Path,
    *,
    system_prompt: str,
    min_confidence: float,
    use_kimi_reasons: bool = True,  # kept for API compat; ignored on combat path
) -> tuple[list[dict[str, Any]], list[ExperienceEntry]]:
    """Convert teacher (deepseek/kimi) turn-order review into combat SFT rows
    plus reusable lesson entries for the experience replay buffer.

    Combat policy SFT samples carry only ``action_index`` (and confidence);
    the teacher-supplied reason text is kept on the row metadata for
    diagnostics but never appears in the assistant message — strategic
    reasoning belongs to the planner LoRA, taught via a separate
    ``build_planner_hint_dataset`` pipeline that *does* train on the
    teacher's reasoning text. ``use_kimi_reasons`` is retained as a
    no-op argument so legacy callers don't break.
    """
    _ = use_kimi_reasons  # combat SFT no longer carries reason text either way
    review = _read_json(review_path)
    episode = _read_json(episode_path)
    decisions = _episode_decision_by_step(episode)
    rows: list[dict[str, Any]] = []
    lessons: list[ExperienceEntry] = []

    for label in review.get("usable_training_labels") or []:
        if not isinstance(label, dict):
            continue
        try:
            step = int(label.get("step"))
            action_index = int(label.get("best_action_index"))
        except (TypeError, ValueError):
            continue
        confidence = _coerce_confidence(label.get("confidence"))
        if confidence < min_confidence:
            continue
        decision = decisions.get(step)
        if not decision:
            continue
        user = str(decision.get("pre_decision_state") or "")
        ok, action = _validate_label(user, action_index)
        if not ok:
            continue
        if not _preserves_visible_attacking_lethal(user, action):
            continue
        teacher_reason = str(label.get("reason_en") or label.get("reason") or "")
        # Teacher prose is allowed to stay on the row metadata (for
        # debug + planner SFT extraction) even when it's been validated
        # only loosely — ``_teacher_reason_matches_action`` continues to
        # reject labels where the prose clearly contradicts the chosen
        # action so we don't import lethal-claim hallucinations into
        # downstream tooling.
        if teacher_reason and not _teacher_reason_matches_action(user, action, teacher_reason):
            continue
        rows.append(_sample(
            user_message=user,
            action_index=action_index,
            meta={
                "source": "turn_order_review",
                "source_review": str(review_path),
                "source_episode_input": str(episode_path),
                "episode_id": episode.get("episode_id") or review.get("episode_id"),
                "step": step,
                "confidence": confidence,
                "original_action_index": decision.get("chosen_action_index"),
                "original_reason": decision.get("reason"),
                "teacher_reason_en": teacher_reason,
                "teacher_action": action,
            },
            system_prompt=system_prompt,
        ))

    for lesson in review.get("key_lessons") or []:
        if not isinstance(lesson, dict):
            continue
        advice = str(lesson.get("training_reason_en") or lesson.get("lesson_en") or lesson.get("lesson_zh") or "")
        # tags 防御：LLM 偶尔回单 string（如 "lethal"），不能直接当 iterable，
        # 否则会按字符拆成 ["l","e","t",...]，导致下游 set 比较全部失败。
        tags = _coerce_tag_list(lesson.get("tags"))
        lessons.append(ExperienceEntry(
            tags=tags,
            applies_when=_lesson_condition(tags, advice),
            advice=advice,
            avoid="",
            source=f"teacher_review:{review_path.name}",
            confidence=0.75,
        ))
    return rows, lessons


def _lesson_condition(tags: list[str], fallback: str) -> str:
    tagset = set(tags)
    if {"lethal", "incoming_damage"} <= tagset:
        return "visible lethal is available on an attacking enemy"
    if {"forgotten_ritual", "exhaust"} <= tagset:
        return "a card was Exhausted this turn and Forgotten Ritual is legal"
    if {"overkill", "draw"} <= tagset:
        return "a low-HP enemy can be killed by a cheaper draw attack"
    if {"target_priority", "enemy_intent"} <= tagset:
        return "multiple enemies are killable and only some are attacking"
    if {"hand_drill", "block_break"} <= tagset:
        return "Hand Drill is owned and an enemy has Block"
    return fallback


def _rows_from_reselect(path: Path, *, system_prompt: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        if not row.get("improved"):
            continue
        old = row.get("old") if isinstance(row.get("old"), dict) else {}
        new = row.get("new") if isinstance(row.get("new"), dict) else {}
        if new.get("status") not in {"ok", "dry_run"}:
            continue
        action_index = new.get("action_index")
        if not isinstance(action_index, int):
            continue
        # reselect_results does not store the user prompt unless dry-run, so use
        # only rows that include prompt_messages.
        messages = row.get("prompt_messages") if isinstance(row.get("prompt_messages"), list) else None
        user = ""
        if messages:
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    user = str(message.get("content") or "")
                    break
        if not user:
            continue
        ok, action = _validate_label(user, action_index)
        if not ok:
            continue
        if not _preserves_visible_attacking_lethal(user, action):
            continue
        out.append(_sample(
            user_message=user,
            action_index=action_index,
            meta={
                "source": "review_reselect",
                "source_results": str(path),
                "episode_id": row.get("episode_id"),
                "step": row.get("step"),
                "original_action_index": old.get("action_index"),
                "score_delta": row.get("score_delta"),
                "teacher_reason_en": str(new.get("reason") or ""),
                "teacher_action": action,
            },
            system_prompt=system_prompt,
        ))
    return out


# NOTE: ``_canonical_reason_from_action`` was deleted. Combat policy SFT
# samples no longer carry a reason field, so the per-action template is
# obsolete. Teacher prose (deepseek / kimi review) still ships on the
# ``meta`` blob for diagnostics and will be consumed by the planner SFT
# pipeline (``build_planner_hint_dataset``) — never written into the
# combat assistant message.


def _teacher_reason_matches_action(user: str, action: dict[str, Any] | None, reason: str) -> bool:
    """Reject labels where Kimi's prose clearly points at a different action."""
    if not action:
        return False
    text = reason.lower()
    if not text:
        return True
    mentions_end_turn = "end turn" in text or "end_turn" in text
    if mentions_end_turn and not _is_end_turn(action):
        return False
    if _is_end_turn(action) and any(token in text for token in ("play ", "attack", "damage", "lethal", "kill")):
        return False
    if "lethal" in text and not _is_lethal(action, _enemies(user)):
        return False
    if any(token in text for token in ("block", "defend")):
        block = action.get("block")
        if not isinstance(block, int) or block <= 0:
            return False
    if any(token in text for token in ("damage", "attack", "hit ", "kill")):
        damage = action.get("damage")
        if not isinstance(damage, int) or damage <= 0:
            return False
    return True


def _rows_from_kimi_labels(
    path: Path,
    *,
    system_prompt: str,
    min_confidence: float,
    keep_kimi_reasons: bool = False,  # legacy no-op; combat samples never carry reason
) -> list[dict[str, Any]]:
    """Convert filtered teacher hard-case labels into combat SFT rows.

    Combat assistant message contains only ``action_index`` + confidence;
    the teacher's prose reason ships on the row metadata (``teacher_reason_en``)
    so planner-side training and trace inspection can still see it, but
    the combat policy never learns to reproduce that text.
    """
    _ = keep_kimi_reasons  # legacy flag; kept for API compat
    out: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        user = str(row.get("user_message") or "")
        action_index = row.get("best_action_index")
        confidence = _coerce_confidence(row.get("confidence"))
        if not user or isinstance(action_index, bool) or not isinstance(action_index, int) or confidence < min_confidence:
            continue
        ok, action = _validate_label(user, int(action_index))
        if not ok:
            continue
        if not _preserves_visible_attacking_lethal(user, action):
            continue
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        teacher_reason = str(row.get("reason_en") or "")
        out.append(_sample(
            user_message=user,
            action_index=int(action_index),
            meta={
                "source": "kimi_teacher_label",
                "source_labels": str(path),
                "candidate_id": row.get("candidate_id"),
                "teacher_reason_en": teacher_reason,
                "episode_id": source.get("episode_id"),
                "step": source.get("episode_step", source.get("step")),
                "confidence": confidence,
                "judgement": row.get("judgement"),
                "mechanism_tags": row.get("mechanism_tags") if isinstance(row.get("mechanism_tags"), list) else [],
                "kimi_reason_en": row.get("reason_en"),
                "reason_source": "kimi" if keep_kimi_reasons else "canonical_verified",
                "original_action_index": row.get("original_action_index"),
                "teacher_action": action,
            },
            system_prompt=system_prompt,
        ))
    return out


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        user = ""
        for message in row.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "user":
                user = str(message.get("content") or "")
                break
        assistant = ""
        for message in row.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant = str(message.get("content") or "")
                break
        try:
            action_index = int(json.loads(assistant).get("action_index"))
        except Exception:  # noqa: BLE001
            continue
        key = (_user_hash(user), action_index)
        current = best.get(key)
        confidence = float((row.get("meta") or {}).get("confidence") or 0.75)
        if current is None or confidence > float((current.get("meta") or {}).get("confidence") or 0.75):
            best[key] = row
    return list(best.values())


def _append_experience(entries: list[ExperienceEntry], path: Path) -> int:
    if not entries:
        return 0
    existing = {
        (entry.applies_when, entry.advice, entry.avoid)
        for entry in load_experience(path)
    }
    new_entries = [
        entry for entry in entries
        if (entry.applies_when, entry.advice, entry.avoid) not in existing
    ]
    if new_entries:
        append_experience(new_entries, path)
    return len(new_entries)


def main() -> int:
    args = parse_args()
    ensure_dirs()
    out_dir = Path(args.out_dir).resolve()
    system_prompt = load_system_prompt()
    rows: list[dict[str, Any]] = []
    lessons: list[ExperienceEntry] = []
    counters: Counter[str] = Counter()

    for trace in args.trace:
        trace_rows = _rows_from_trace(Path(trace).resolve(), system_prompt=system_prompt)
        rows.extend(trace_rows)
        counters["trace_rule_rows"] += len(trace_rows)

    review_inputs = [(Path(review).resolve(), Path(episode).resolve()) for review, episode in zip(args.review, args.episode_input)]
    review_inputs.extend(_review_pairs_from_roots(args.review_root))

    if len(args.review) != len(args.episode_input):
        raise SystemExit("--review and --episode-input counts must match")
    for review_path, episode_path in review_inputs:
        review_rows, review_lessons = _rows_from_review(
            review_path,
            episode_path,
            system_prompt=system_prompt,
            min_confidence=args.min_confidence,
            use_kimi_reasons=not args.no_kimi_reasons_in_review,
        )
        rows.extend(review_rows)
        lessons.extend(review_lessons)
        counters["review_rows"] += len(review_rows)
        counters["review_lessons"] += len(review_lessons)

    for path in args.reselect_results:
        reselect_rows = _rows_from_reselect(Path(path).resolve(), system_prompt=system_prompt)
        rows.extend(reselect_rows)
        counters["reselect_rows"] += len(reselect_rows)

    for path in args.kimi_labels:
        kimi_rows = _rows_from_kimi_labels(
            Path(path).resolve(),
            system_prompt=system_prompt,
            min_confidence=args.min_confidence,
            keep_kimi_reasons=bool(args.keep_kimi_reasons),
        )
        rows.extend(kimi_rows)
        counters["kimi_label_rows"] += len(kimi_rows)

    deduped = _dedupe_rows(rows)
    rng = random.Random(args.seed)
    rng.shuffle(deduped)
    eval_n = max(1, int(len(deduped) * args.eval_ratio)) if len(deduped) >= 8 else 0
    eval_rows = deduped[:eval_n]
    train_rows = deduped[eval_n:]

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    _write_jsonl(out_dir / "lessons.jsonl", [entry.to_json() for entry in lessons])

    appended = _append_experience(lessons, Path(args.experience_path)) if args.append_experience else 0
    source_counts = Counter(str((row.get("meta") or {}).get("source") or "") for row in deduped)
    rule_counts = Counter(str((row.get("meta") or {}).get("source_rule") or "") for row in deduped if (row.get("meta") or {}).get("source_rule"))
    summary = {
        "kind": "teacher_turn_repair_dataset",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "raw_rows": len(rows),
        "rows": len(deduped),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "experience_appended": appended,
        "counters": {key: int(value) for key, value in counters.most_common()},
        "source_counts": {key: int(value) for key, value in source_counts.most_common()},
        "rule_counts": {key: int(value) for key, value in rule_counts.most_common()},
        "inputs": {
            "traces": [str(Path(path).resolve()) for path in args.trace],
            "review_roots": [str(Path(path).resolve()) for path in args.review_root],
            "reviews": [str(path) for path, _episode_path in review_inputs],
            "episode_inputs": [str(episode_path) for _path, episode_path in review_inputs],
            "reselect_results": [str(Path(path).resolve()) for path in args.reselect_results],
            "kimi_labels": [str(Path(path).resolve()) for path in args.kimi_labels],
            "keep_kimi_reasons": bool(args.keep_kimi_reasons),
        },
        "outputs": {
            "train": str(out_dir / "train.jsonl"),
            "eval": str(out_dir / "eval.jsonl"),
            "lessons": str(out_dir / "lessons.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
