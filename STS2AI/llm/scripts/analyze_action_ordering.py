"""Analyze action-ordering behavior from LLM step traces.

This is intentionally diagnostic. It does not declare a full strategy rule
engine; it looks for concrete sequencing opportunities in actual model traces,
then emits counts plus examples for review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


_ROUND_RE = re.compile(r"^run: .* round=(\d+)\b", re.MULTILINE)
_ENERGY_RE = re.compile(r"^player: .* energy=(\d+)/(\d+)\b", re.MULTILINE)
_ENEMY_RE = re.compile(
    r"^\s*(enemy\d+):\s+(\S+)\s+hp=(\d+)/(\d+)\s+block=(\d+).* powers=(.*)$",
    re.MULTILINE,
)
_GROUP_RE = re.compile(r"^\s*([A-Z0-9_+]+)\s+hand\[(\d+)\]:\s*$")
_DIRECT_RE = re.compile(r"^\s*\[(\d+)\]\s+([A-Za-z0-9_+]+)?(?:\s+hand\[(\d+)\])?(.*)$")
_TARGET_RE = re.compile(r"\btarget=(enemy\d+|self)\b")
_DAMAGE_RE = re.compile(r"\bdamage=(-?\d+)\b")
_BLOCK_RE = re.compile(r"\bblock=(-?\d+)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True, help="step_trace.jsonl")
    parser.add_argument("--out", type=str, default="", help="optional JSON output path")
    parser.add_argument("--examples", type=int, default=12)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _round(user_message: str) -> int | None:
    match = _ROUND_RE.search(user_message)
    return int(match.group(1)) if match else None


def _energy(user_message: str) -> int | None:
    match = _ENERGY_RE.search(user_message)
    return int(match.group(1)) if match else None


def _enemies(user_message: str) -> dict[str, dict[str, Any]]:
    enemies: dict[str, dict[str, Any]] = {}
    for match in _ENEMY_RE.finditer(user_message):
        label, name, hp, max_hp, block, powers = match.groups()
        enemies[label] = {
            "label": label,
            "name": name,
            "hp": int(hp),
            "max_hp": int(max_hp),
            "block": int(block),
            "powers": powers.strip(),
            "vulnerable": "VULNERABLE_POWER" in powers,
        }
    return enemies


def _card_from_action(action: dict[str, Any]) -> str:
    return str(action.get("card_id") or action.get("card") or "").replace("+", "")


def _target_label_from_action(action: dict[str, Any]) -> str:
    target = action.get("target_label") or action.get("target")
    if target:
        return str(target)
    raw = action.get("target_id")
    if isinstance(raw, int) and raw >= 0:
        return f"enemy{raw}"
    if isinstance(raw, str) and raw.isdigit():
        return f"enemy{raw}"
    return str(raw or "")


def _legal_from_structured(row: dict[str, Any]) -> list[dict[str, Any]]:
    actions = row.get("legal_actions")
    if not isinstance(actions, list):
        return []
    result: list[dict[str, Any]] = []
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        card_id = _card_from_action(action)
        if not card_id:
            continue
        result.append({
            "index": idx,
            "card_id": card_id,
            "target": _target_label_from_action(action),
            "damage": action.get("preview_damage") or action.get("damage"),
            "block": action.get("preview_block") or action.get("block"),
        })
    return result


def _legal_from_text(user_message: str) -> list[dict[str, Any]]:
    in_legal = False
    current_card = ""
    current_hand = ""
    actions: list[dict[str, Any]] = []
    for raw_line in user_message.splitlines():
        line = raw_line.rstrip()
        if line.strip() == "legal_actions:":
            in_legal = True
            continue
        if in_legal and line and not line.startswith(" ") and not line.startswith("\t"):
            break
        if not in_legal:
            continue
        group = _GROUP_RE.match(line)
        if group:
            current_card = group.group(1).replace("+", "")
            current_hand = group.group(2)
            continue
        direct = _DIRECT_RE.match(line)
        if not direct:
            continue
        idx, card, hand, rest = direct.groups()
        if card and card.lower() in {"target", "damage", "block"}:
            rest = f"{card}{rest}"
            card = ""
        card_id = (card or current_card or "").replace("+", "")
        target = ""
        target_match = _TARGET_RE.search(rest)
        if target_match:
            target = target_match.group(1)
        damage_match = _DAMAGE_RE.search(rest)
        block_match = _BLOCK_RE.search(rest)
        actions.append({
            "index": int(idx),
            "card_id": card_id,
            "hand": hand or current_hand,
            "target": target,
            "damage": int(damage_match.group(1)) if damage_match else None,
            "block": int(block_match.group(1)) if block_match else None,
            "raw": line.strip(),
        })
    return actions


def _legal_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    structured = _legal_from_structured(row)
    return structured if structured else _legal_from_text(str(row.get("user_message") or ""))


def _chosen(row: dict[str, Any], legal_actions: list[dict[str, Any]]) -> dict[str, Any]:
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    raw_idx = decoded.get("action_index")
    if isinstance(raw_idx, int):
        for action in legal_actions:
            if action.get("index") == raw_idx:
                return action
    chosen_action = row.get("chosen_action") if isinstance(row.get("chosen_action"), dict) else {}
    return {
        "index": raw_idx,
        "card_id": _card_from_action(chosen_action),
        "target": _target_label_from_action(chosen_action),
    }


def _attack_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        action for action in actions
        if isinstance(action.get("damage"), int) and int(action["damage"]) > 0
    ]


def analyze(rows: list[dict[str, Any]], *, example_limit: int) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bash_events: list[dict[str, Any]] = []

    for row in rows:
        by_episode[str(row.get("episode_id") or "")].append(row)
        user_message = str(row.get("user_message") or "")
        actions = _legal_actions(row)
        chosen = _chosen(row, actions)
        enemies = _enemies(user_message)
        energy = _energy(user_message)
        round_no = _round(user_message)
        attacks = _attack_actions(actions)
        bash_actions = [action for action in attacks if str(action.get("card_id")) == "BASH"]
        non_bash_attacks = [action for action in attacks if str(action.get("card_id")) != "BASH"]

        if bash_actions and non_bash_attacks and (energy is None or energy >= 3):
            counters["bash_with_followup_attack_opportunities"] += 1
            chosen_card = str(chosen.get("card_id") or "")
            chosen_target = str(chosen.get("target") or "")
            target_enemy = enemies.get(chosen_target)
            if chosen_card == "BASH":
                counters["bash_first"] += 1
                bash_events.append({
                    "episode_id": row.get("episode_id"),
                    "round": round_no,
                    "target": chosen_target,
                    "step": row.get("episode_step", row.get("step")),
                })
                bucket = "bash_first_examples"
            elif chosen_card and chosen_card != "BASH" and chosen_target.startswith("enemy") and not (target_enemy or {}).get("vulnerable"):
                counters["attack_before_available_bash"] += 1
                bucket = "attack_before_available_bash_examples"
            else:
                counters["other_choice_with_available_bash"] += 1
                bucket = "other_choice_with_available_bash_examples"
            if len(examples[bucket]) < example_limit:
                examples[bucket].append({
                    "episode_id": row.get("episode_id"),
                    "step": row.get("episode_step", row.get("step")),
                    "round": round_no,
                    "chosen": chosen,
                    "reason": (row.get("decoded") or {}).get("reason") if isinstance(row.get("decoded"), dict) else "",
                    "enemies": enemies,
                })

        flags = row.get("quality_flags") if isinstance(row.get("quality_flags"), list) else []
        for flag in flags:
            counters[f"quality:{flag}"] += 1

    for episode_id, episode_rows in by_episode.items():
        sorted_rows = sorted(episode_rows, key=lambda item: int(item.get("episode_step") or item.get("step") or 0))
        for i, row in enumerate(sorted_rows[:-1]):
            actions = _legal_actions(row)
            chosen = _chosen(row, actions)
            if str(chosen.get("card_id") or "") != "BASH":
                continue
            target = str(chosen.get("target") or "")
            round_no = _round(str(row.get("user_message") or ""))
            for nxt in sorted_rows[i + 1:]:
                if _round(str(nxt.get("user_message") or "")) != round_no:
                    break
                next_actions = _legal_actions(nxt)
                next_chosen = _chosen(nxt, next_actions)
                if str(next_chosen.get("target") or "") != target:
                    continue
                if str(next_chosen.get("card_id") or "") != "BASH":
                    counters["bash_then_same_target_attack"] += 1
                    if len(examples["bash_then_same_target_attack_examples"]) < example_limit:
                        examples["bash_then_same_target_attack_examples"].append({
                            "episode_id": episode_id,
                            "bash_step": row.get("episode_step", row.get("step")),
                            "attack_step": nxt.get("episode_step", nxt.get("step")),
                            "target": target,
                            "attack": next_chosen,
                            "attack_reason": (nxt.get("decoded") or {}).get("reason") if isinstance(nxt.get("decoded"), dict) else "",
                        })
                    break

    opportunities = counters.get("bash_with_followup_attack_opportunities", 0)
    bash_first = counters.get("bash_first", 0)
    attack_before = counters.get("attack_before_available_bash", 0)
    return {
        "steps": len(rows),
        "episodes": len(by_episode),
        "counts": {key: int(value) for key, value in counters.most_common()},
        "rates": {
            "bash_first_rate": round(bash_first / opportunities, 4) if opportunities else None,
            "attack_before_available_bash_rate": round(attack_before / opportunities, 4) if opportunities else None,
        },
        "examples": examples,
    }


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace)
    rows = _read_jsonl(trace_path)
    result = analyze(rows, example_limit=max(0, args.examples))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
