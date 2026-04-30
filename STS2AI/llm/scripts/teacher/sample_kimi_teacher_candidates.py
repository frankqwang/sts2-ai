"""Sample diverse decision states for Kimi teacher review.

This script does not call Kimi. It prepares:

- candidates.jsonl: one row per selected decision state
- openai_batch_request.jsonl: Chat Completions JSONL suitable for a Batch job
- preview.md / summary.json: quick audit artifacts

The goal is to avoid spending teacher budget on near-duplicate states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.paths import ARTIFACTS_ROOT, DATASETS_ROOT, ensure_dirs  # noqa: E402
from llm.scripts.analysis.analyze_action_ordering import (  # noqa: E402
    _chosen,
    _energy,
    _legal_actions,
    _round,
)
from llm.scripts.teacher.teacher_review_turn_order import DEFAULT_MODEL  # noqa: E402


_HAND_RE = re.compile(r"^\s*\[(\d+)]\s+([A-Z0-9_+]+)\s+cost=([0-9]+|X|\?)\s+type=([a-zA-Z_]+)(.*)$", re.MULTILINE)
_ENEMY_RE = re.compile(
    r"^\s*(enemy\d+):\s+(\S+)\s+hp=(\d+)/(\d+)\s+block=(\d+)\s+intent=([^\s]+(?:\([^)]+\))?)\s+powers=(.*)$",
    re.MULTILINE,
)
_SECTION_RE = re.compile(r"^[A-Za-z_]+:", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", default=[], help="step_trace.jsonl. Defaults to all traces.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--items-per-request", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--include-forced", action="store_true", help="Include states where end_turn is the only legal action.")
    parser.add_argument("--max-state-chars", type=int, default=6500)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payload.setdefault("source_file", str(path))
                payload.setdefault("source_line", line_no)
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


def _all_traces() -> list[Path]:
    return sorted(
        [path for path in DATASETS_ROOT.rglob("step_trace.jsonl") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _sha(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def _hand_cards(user_message: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for match in _HAND_RE.finditer(user_message):
        hand, card, cost, card_type, rest = match.groups()
        cards.append({
            "hand": int(hand),
            "card_id": card.replace("+", ""),
            "cost": cost,
            "type": card_type.lower(),
            "text": rest.strip(),
            "draw": "Draw" in rest,
            "exhaust": "Exhaust" in rest,
            "vulnerable": "Vulnerable" in rest or "VULNERABLE" in rest,
            "block": "Block" in rest,
        })
    return cards


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


def _is_end_turn(action: dict[str, Any] | None) -> bool:
    if not action:
        return False
    return str(action.get("card_id") or "").lower() == "end_turn" or "end_turn" in str(action.get("raw") or "").lower()


def _effective_hp(enemy: dict[str, Any] | None) -> int:
    if not enemy:
        return 0
    return int(enemy.get("hp") or 0) + int(enemy.get("block") or 0)


def _is_lethal(action: dict[str, Any], enemies: dict[str, dict[str, Any]]) -> bool:
    target = str(action.get("target") or "")
    damage = action.get("damage")
    return target in enemies and isinstance(damage, int) and damage >= _effective_hp(enemies[target]) > 0


def _quality_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = [str(flag) for flag in (row.get("quality_flags") or [])]
    report = row.get("quality_report") if isinstance(row.get("quality_report"), dict) else {}
    for flag in report.get("flags") or []:
        text = str(flag)
        if text not in flags:
            flags.append(text)
    return flags


def _quality_opportunities(row: dict[str, Any]) -> dict[str, Any]:
    report = row.get("quality_report") if isinstance(row.get("quality_report"), dict) else {}
    value = report.get("opportunities")
    return value if isinstance(value, dict) else {}


def _quality_misses(row: dict[str, Any]) -> dict[str, Any]:
    report = row.get("quality_report") if isinstance(row.get("quality_report"), dict) else {}
    value = report.get("misses")
    return value if isinstance(value, dict) else {}


def _has_bash_followup(actions: list[dict[str, Any]], user_message: str) -> bool:
    energy = _energy(user_message)
    if energy is not None and energy < 3:
        return False
    has_bash = any(str(action.get("card_id") or "") == "BASH" and isinstance(action.get("damage"), int) for action in actions)
    has_other_attack = any(
        str(action.get("card_id") or "") not in {"", "BASH", "end_turn"}
        and isinstance(action.get("damage"), int)
        and int(action.get("damage") or 0) > 0
        for action in actions
    )
    return has_bash and has_other_attack


def _strip_strategy_context(user_message: str) -> str:
    lines = user_message.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        if line.startswith("strategy_context:"):
            skip = True
            continue
        if skip and _SECTION_RE.match(line):
            skip = False
        if not skip and not line.startswith("Return one JSON line:"):
            out.append(line)
    return "\n".join(out).strip()


def _trim(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...<truncated>"


def _original_index(row: dict[str, Any], actions: list[dict[str, Any]]) -> int | None:
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    raw = decoded.get("action_index")
    if isinstance(raw, int):
        return int(raw)
    chosen = _chosen(row, actions)
    raw = chosen.get("index")
    return int(raw) if isinstance(raw, int) else None


def _original_reason(row: dict[str, Any]) -> str:
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    if isinstance(decoded.get("reason"), str):
        return str(decoded["reason"])
    return ""


def _feature_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    user = str(row.get("user_message") or "")
    if not user:
        return None
    actions = _legal_actions({"user_message": user})
    if not actions:
        return None
    chosen = _chosen(row, actions)
    original_index = _original_index(row, actions)
    hand = _hand_cards(user)
    enemies = _enemies(user)
    flags = _quality_flags(row)
    opportunities = _quality_opportunities(row)
    misses = _quality_misses(row)
    play_actions = [action for action in actions if not _is_end_turn(action)]
    attack_actions = [
        action for action in actions
        if isinstance(action.get("damage"), int) and int(action.get("damage") or 0) > 0
    ]
    lethal_actions = [action for action in attack_actions if _is_lethal(action, enemies)]
    tags: set[str] = set()
    if lethal_actions:
        tags.add("visible_lethal")
    if opportunities.get("dangerous_turn") or any(enemy.get("attacking") for enemy in enemies.values()):
        tags.add("enemy_intent")
    if "dangerous_end_turn" in flags:
        tags.add("dangerous_end_turn")
    if _has_bash_followup(actions, user):
        tags.add("bash_vulnerable")
    if any(card.get("draw") for card in hand):
        tags.add("draw_order")
    if any(card.get("exhaust") for card in hand):
        tags.add("exhaust")
    if "FORGOTTEN_RITUAL" in {card.get("card_id") for card in hand}:
        tags.add("forgotten_ritual")
    if any(card.get("block") for card in hand) and any(enemy.get("attacking") for enemy in enemies.values()):
        tags.add("block_vs_attack")
    if any(enemy.get("block") for enemy in enemies.values()):
        tags.add("enemy_block")
    if "HAND_DRILL" in user and any(enemy.get("block") for enemy in enemies.values()):
        tags.add("hand_drill")
    if len(enemies) >= 2:
        tags.add("multi_enemy")
    if len(actions) >= 10:
        tags.add("many_actions")
    if not play_actions:
        tags.add("forced_end_turn")

    hand_ids = sorted(card["card_id"] for card in hand)
    enemy_shape = sorted(
        (
            enemy["name"],
            "atk" if enemy["attacking"] else "nonatk",
            min(5, int((enemy["hp"] + enemy["block"]) / max(1, enemy["max_hp"]) * 5)),
            "vuln" if enemy["vulnerable"] else "plain",
            "block" if enemy["block"] else "noblock",
        )
        for enemy in enemies.values()
    )
    chosen_card = str(chosen.get("card_id") or "")
    chosen_target = str(chosen.get("target") or "")
    scene_parts = {
        "encounter": row.get("encounter_id"),
        "tag": row.get("encounter_tag"),
        "round": _round(user),
        "energy": _energy(user),
        "hand": hand_ids,
        "enemies": enemy_shape,
        "tags": sorted(tags),
        "chosen": [chosen_card, chosen_target],
    }
    exact_state = _strip_strategy_context(user)
    return {
        "actions": actions,
        "chosen": chosen,
        "original_index": original_index,
        "original_reason": _original_reason(row),
        "hand": hand,
        "enemies": enemies,
        "flags": flags,
        "opportunities": opportunities,
        "misses": misses,
        "tags": sorted(tags),
        "play_action_count": len(play_actions),
        "attack_action_count": len(attack_actions),
        "visible_lethal_count": len(lethal_actions),
        "scene_signature": _sha(json.dumps(scene_parts, ensure_ascii=False, sort_keys=True), 20),
        "exact_hash": _sha(exact_state, 20),
    }


def _score(features: dict[str, Any]) -> float:
    flags = set(features["flags"])
    tags = set(features["tags"])
    score = 1.0
    if "missed_visible_lethal" in flags:
        score += 35
    if "dangerous_end_turn" in flags:
        score += 18
    if "invalid_output" in flags:
        score += 20
    score += min(4, int(features["visible_lethal_count"])) * 8
    score += min(3, int(features["attack_action_count"])) * 1.5
    weights = {
        "bash_vulnerable": 9,
        "forgotten_ritual": 9,
        "hand_drill": 8,
        "enemy_block": 5,
        "draw_order": 6,
        "exhaust": 5,
        "block_vs_attack": 5,
        "multi_enemy": 3,
        "many_actions": 2,
        "enemy_intent": 2,
    }
    for tag, weight in weights.items():
        if tag in tags:
            score += weight
    if "forced_end_turn" in tags:
        score -= 30
    return round(score, 4)


def _primary_bucket(row: dict[str, Any], features: dict[str, Any]) -> str:
    tags = list(features["tags"])
    priority = [
        "visible_lethal",
        "forgotten_ritual",
        "bash_vulnerable",
        "hand_drill",
        "draw_order",
        "block_vs_attack",
        "dangerous_end_turn",
        "enemy_block",
        "multi_enemy",
    ]
    tag = next((item for item in priority if item in tags), tags[0] if tags else "plain")
    return f"{row.get('encounter_id') or '?'}|{row.get('encounter_tag') or '?'}|{tag}"


def _messages(candidate: dict[str, Any], *, max_state_chars: int) -> list[dict[str, str]]:
    state = _trim(_strip_strategy_context(str(candidate["source"]["user_message"])), max_state_chars)
    features = candidate["features"]
    prompt = (
        "Review one Slay the Spire 2 decision state.\n"
        "Choose the best legal action_index for the current state only.\n"
        "Use only action indices listed under legal_actions. Do not invent cards, targets, future draws, or rollout results.\n"
        "If the original action is already best, keep the same action_index.\n"
        "If uncertain, keep the original action and set confidence <= 0.6.\n\n"
        "Return one valid JSON object with this schema:\n"
        "{"
        "\"judgement\":\"keep|change|uncertain\","
        "\"best_action_index\":0,"
        "\"confidence\":0.0,"
        "\"reason_en\":\"short concrete reason\","
        "\"reason_zh\":\"short Chinese review\","
        "\"mechanism_tags\":[\"short_tag\"]"
        "}\n\n"
        f"candidate_id={candidate['candidate_id']}\n"
        f"original_action_index={features['original_index']}\n"
        f"original_reason={features['original_reason']}\n"
        f"detected_flags={features['flags']}\n"
        f"detected_tags={features['tags']}\n"
        f"visible_lethal_count={features['visible_lethal_count']}\n\n"
        "current_state:\n"
        f"{state}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Slay the Spire 2 teacher. "
                "Review one decision and return JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _state_block(candidate: dict[str, Any], *, max_state_chars: int) -> str:
    state = _trim(_strip_strategy_context(str(candidate["source"]["user_message"])), max_state_chars)
    features = candidate["features"]
    return (
        f"candidate_id={candidate['candidate_id']}\n"
        f"original_action_index={features['original_index']}\n"
        f"original_reason={features['original_reason']}\n"
        f"detected_flags={features['flags']}\n"
        f"detected_tags={features['tags']}\n"
        f"visible_lethal_count={features['visible_lethal_count']}\n"
        "current_state:\n"
        f"{state}"
    )


def _single_batch_request(candidate: dict[str, Any], *, model: str, thinking: str, max_tokens: int) -> dict[str, Any]:
    return {
        "custom_id": candidate["candidate_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": candidate["messages"],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_tokens,
            "thinking": {"type": thinking},
            "stream": False,
        },
    }


def _group_messages(candidates: list[dict[str, Any]], *, max_state_chars: int) -> list[dict[str, str]]:
    blocks = [
        f"## decision {idx + 1}\n{_state_block(candidate, max_state_chars=max_state_chars)}"
        for idx, candidate in enumerate(candidates)
    ]
    decision_states = "\n\n".join(blocks)
    prompt = (
        "Review these independent Slay the Spire 2 decision states.\n"
        "For each candidate, choose the best legal action_index for that candidate's current_state only.\n"
        "Use only action indices listed under that candidate's legal_actions. Do not invent cards, targets, future draws, or rollout results.\n"
        "If the original action is already best, keep the same action_index.\n"
        "If uncertain, keep the original action and set confidence <= 0.6.\n\n"
        "Return one valid JSON object with this schema:\n"
        "{"
        "\"reviews\":[{"
        "\"candidate_id\":\"id\","
        "\"judgement\":\"keep|change|uncertain\","
        "\"best_action_index\":0,"
        "\"confidence\":0.0,"
        "\"reason_en\":\"short concrete reason\","
        "\"reason_zh\":\"short Chinese review\","
        "\"mechanism_tags\":[\"short_tag\"]"
        "}]"
        "}\n\n"
        "decision_states:\n"
        f"{decision_states}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Slay the Spire 2 teacher. "
                "Review decision states and return JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _grouped_batch_requests(
    selected: list[dict[str, Any]],
    *,
    model: str,
    thinking: str,
    max_tokens: int,
    items_per_request: int,
    max_state_chars: int,
) -> list[dict[str, Any]]:
    group_size = max(1, int(items_per_request))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(selected), group_size):
        group = selected[start:start + group_size]
        group_id = f"kimi-teacher-group-{start // group_size:04d}"
        rows.append({
            "custom_id": group_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": _group_messages(group, max_state_chars=max_state_chars),
                "response_format": {"type": "json_object"},
                "max_completion_tokens": max_tokens,
                "thinking": {"type": thinking},
                "stream": False,
            },
        })
    return rows


def _batch_groups(selected: list[dict[str, Any]], *, items_per_request: int) -> list[dict[str, Any]]:
    group_size = max(1, int(items_per_request))
    groups: list[dict[str, Any]] = []
    for start in range(0, len(selected), group_size):
        group = selected[start:start + group_size]
        groups.append({
            "custom_id": f"kimi-teacher-group-{start // group_size:04d}",
            "candidate_ids": [str(candidate["candidate_id"]) for candidate in group],
        })
    return groups


def _select_diverse(candidates: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    exact_seen: set[str] = set()
    scene_seen: set[str] = set()
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_bucket[str(candidate["bucket"])].append(candidate)
    for bucket_rows in by_bucket.values():
        rng.shuffle(bucket_rows)
        bucket_rows.sort(key=lambda item: (-float(item["score"]), str(item["candidate_id"])))

    selected: list[dict[str, Any]] = []
    # First pass: strict scene diversity, round-robin by bucket.
    while len(selected) < limit:
        added = False
        for bucket in sorted(by_bucket, key=lambda key: (-max(float(item["score"]) for item in by_bucket[key]), key)):
            for candidate in by_bucket[bucket]:
                exact = str(candidate["features"]["exact_hash"])
                scene = str(candidate["features"]["scene_signature"])
                if exact in exact_seen or scene in scene_seen:
                    continue
                selected.append(candidate)
                exact_seen.add(exact)
                scene_seen.add(scene)
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break

    # Second pass: allow similar scenes but never exact duplicates.
    if len(selected) < limit:
        leftovers = sorted(candidates, key=lambda item: (-float(item["score"]), str(item["candidate_id"])))
        for candidate in leftovers:
            exact = str(candidate["features"]["exact_hash"])
            if exact in exact_seen:
                continue
            selected.append(candidate)
            exact_seen.add(exact)
            if len(selected) >= limit:
                break
    return selected


def _preview(selected: list[dict[str, Any]], *, count: int) -> str:
    lines = [
        "# Kimi Teacher Candidate Preview",
        "",
        f"selected={len(selected)}",
        "",
    ]
    for candidate in selected[: max(0, count)]:
        source = candidate["source"]
        features = candidate["features"]
        lines.extend([
            f"## {candidate['candidate_id']}",
            "",
            f"- encounter: {source.get('encounter_id')} / {source.get('encounter_tag')}",
            f"- episode_step: {source.get('episode_step', source.get('step'))}",
            f"- score: {candidate['score']}",
            f"- bucket: {candidate['bucket']}",
            f"- tags: {', '.join(features['tags']) or 'none'}",
            f"- flags: {', '.join(features['flags']) or 'none'}",
            f"- original_action_index: {features['original_index']}",
            f"- original_reason: {features['original_reason']}",
            "",
            "```text",
            _trim(_strip_strategy_context(str(source.get("user_message") or "")), 2600),
            "```",
            "",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ensure_dirs()
    trace_paths = [Path(path).resolve() for path in args.trace] if args.trace else _all_traces()
    if not trace_paths:
        raise FileNotFoundError(f"No step_trace.jsonl found under {DATASETS_ROOT}")

    rows: list[dict[str, Any]] = []
    for path in trace_paths:
        rows.extend(_read_jsonl(path))

    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for row_index, row in enumerate(rows):
        features = _feature_payload(row)
        if not features:
            skipped["unparseable"] += 1
            continue
        if not args.include_forced and "forced_end_turn" in set(features["tags"]):
            skipped["forced_end_turn"] += 1
            continue
        score = _score(features)
        candidate_id = f"kimi-teacher-{len(candidates):05d}-{features['exact_hash'][:8]}"
        source = {
            key: row.get(key)
            for key in (
                "episode_id",
                "episode_step",
                "step",
                "encounter_id",
                "encounter_tag",
                "encounter_key",
                "encounter_label",
                "seed",
                "outcome",
                "episode_reward",
                "source_file",
                "source_line",
                "user_message",
            )
        }
        candidate = {
            "candidate_id": candidate_id,
            "score": score,
            "bucket": _primary_bucket(row, features),
            "features": {
                key: value
                for key, value in features.items()
                if key not in {"actions", "chosen", "hand", "enemies"}
            },
            "source": source,
            "selection_meta": {
                "row_index": row_index,
                "scene_signature": features["scene_signature"],
                "exact_hash": features["exact_hash"],
            },
        }
        candidate["messages"] = _messages(candidate, max_state_chars=args.max_state_chars)
        candidates.append(candidate)

    selected = _select_diverse(candidates, limit=max(0, int(args.limit)), seed=args.seed)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        ARTIFACTS_ROOT / "reviews" / f"kimi_teacher_candidates_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_rows = _grouped_batch_requests(
        selected,
        model=args.model,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        items_per_request=args.items_per_request,
        max_state_chars=args.max_state_chars,
    )
    individual_rows = [
        _single_batch_request(candidate, model=args.model, thinking=args.thinking, max_tokens=512)
        for candidate in selected
    ]
    _write_jsonl(out_dir / "candidates.jsonl", selected)
    _write_jsonl(out_dir / "openai_batch_request.jsonl", batch_rows)
    _write_jsonl(out_dir / "openai_batch_request_individual.jsonl", individual_rows)
    _write_json(out_dir / "batch_groups.json", {"groups": _batch_groups(selected, items_per_request=args.items_per_request)})
    (out_dir / "preview.md").write_text(_preview(selected, count=args.preview_count), encoding="utf-8")

    summary = {
        "kind": "kimi_teacher_candidates",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "trace_paths": [str(path) for path in trace_paths],
        "out_dir": str(out_dir),
        "input_rows": len(rows),
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "limit": args.limit,
        "include_forced": args.include_forced,
        "items_per_request": args.items_per_request,
        "batch_request_rows": len(batch_rows),
        "individual_request_rows": len(individual_rows),
        "skipped": dict(skipped),
        "score": {
            "min": min((float(item["score"]) for item in selected), default=0.0),
            "max": max((float(item["score"]) for item in selected), default=0.0),
            "avg": round(sum(float(item["score"]) for item in selected) / len(selected), 4) if selected else 0.0,
        },
        "bucket_counts": dict(Counter(str(item["bucket"]) for item in selected).most_common()),
        "encounter_counts": dict(Counter(str(item["source"].get("encounter_id") or "?") for item in selected).most_common()),
        "tag_counts": dict(Counter(tag for item in selected for tag in item["features"]["tags"]).most_common()),
        "flag_counts": dict(Counter(flag for item in selected for flag in item["features"]["flags"]).most_common()),
        "outputs": {
            "candidates": str(out_dir / "candidates.jsonl"),
            "batch_request": str(out_dir / "openai_batch_request.jsonl"),
            "individual_batch_request": str(out_dir / "openai_batch_request_individual.jsonl"),
            "batch_groups": str(out_dir / "batch_groups.json"),
            "preview": str(out_dir / "preview.md"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
