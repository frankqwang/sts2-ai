"""Mine hard cases and preference pairs from rollout datasets without API calls.

This is the cheap first pass before using external teacher models. It turns
existing rollout artifacts into reusable training assets:

- hard_cases.jsonl: prompts/actions that deserve inspection or teacher labeling.
- preference_pairs.jsonl: same-prompt chosen/rejected pairs from rollout scores.
- repair_sft.jsonl: conservative rule repairs, currently only visible lethal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.metrics import read_jsonl, write_json  # noqa: E402
from llm.paths import ARTIFACTS_ROOT  # noqa: E402


_ACTION_INDEX_RE = re.compile(r'"?action_index"?\s*:\s*(\d+)')
_LETHAL_HINT_RE = re.compile(r"hand\[(\d+)]\s+\S+\s+kills\s+enemy(\d+)", re.IGNORECASE)
_LEGAL_SINGLE_RE = re.compile(r"^\s*\[(\d+)]\s+.*?hand\[(\d+)]\s+target=enemy(\d+)\b", re.MULTILINE)
_LEGAL_GROUP_RE = re.compile(r"^\s*(\S+)\s+hand\[(\d+)]:\s*$", re.MULTILINE)
_LEGAL_GROUP_TARGET_RE = re.compile(r"^\s*\[(\d+)]\s+target=enemy(\d+)\b", re.MULTILINE)

_FLAG_PENALTIES = {
    "missed_visible_lethal": 2.0,
    "dangerous_end_turn": 1.0,
    "end_turn_with_playable_cards": 0.5,
    "invalid_chosen_index": 3.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Rollout dataset directory containing train/eval JSONL.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to dataset/offline_mining.")
    parser.add_argument("--include-eval", action="store_true", help="Also mine eval.jsonl rows.")
    parser.add_argument("--min-score-gap", type=float, default=0.75)
    parser.add_argument("--hard-case-limit", type=int, default=500)
    return parser.parse_args()


def _message(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def _messages_without_assistant(row: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "assistant":
            continue
        if role in {"system", "user"}:
            out.append({"role": role, "content": str(message.get("content") or "")})
    return out


def _assistant_action_index(text: str) -> int | None:
    match = _ACTION_INDEX_RE.search(text or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _json_action(action_index: int, reason: str) -> str:
    return json.dumps({"action_index": int(action_index), "reason": reason}, ensure_ascii=False)


def _user_hash(user_text: str) -> str:
    return hashlib.sha1(user_text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _score_row(row: dict[str, Any]) -> float:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    try:
        score = float(meta.get("advantage") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        score += 0.05 * float(meta.get("episode_reward") or 0.0)
    except (TypeError, ValueError):
        pass
    for flag in meta.get("action_quality_flags") or []:
        score -= _FLAG_PENALTIES.get(str(flag), 0.25)
    if meta.get("policy_invalid_output"):
        score -= 3.0
    return score


def _row_payload(row: dict[str, Any], *, source_file: str, line_no: int) -> dict[str, Any]:
    user = _message(row, "user")
    assistant = _message(row, "assistant")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return {
        "source_file": source_file,
        "line_no": line_no,
        "user_hash": _user_hash(user),
        "score": round(_score_row(row), 4),
        "action_index": _assistant_action_index(assistant),
        "flags": list(meta.get("action_quality_flags") or []),
        "advantage": meta.get("advantage"),
        "episode_reward": meta.get("episode_reward"),
        "outcome": meta.get("outcome"),
        "encounter_key": meta.get("encounter_key"),
        "encounter_label": meta.get("encounter_label"),
        "messages": row.get("messages") or [],
    }


def _legal_action_section(user_text: str) -> str:
    lines = user_text.splitlines()
    out: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == "legal_actions:":
            in_section = True
            continue
        if in_section and line and not line.startswith((" ", "\t")):
            break
        if in_section:
            out.append(line)
    return "\n".join(out)


def _find_legal_index_for_hand_target(legal_text: str, hand_index: int, enemy_id: int) -> int | None:
    for match in _LEGAL_SINGLE_RE.finditer(legal_text):
        action_index, hand, target = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if hand == hand_index and target == enemy_id:
            return action_index

    lines = legal_text.splitlines()
    active_hand: int | None = None
    for line in lines:
        group = _LEGAL_GROUP_RE.match(line)
        if group:
            active_hand = int(group.group(2))
            continue
        target = _LEGAL_GROUP_TARGET_RE.match(line)
        if target and active_hand == hand_index and int(target.group(2)) == enemy_id:
            return int(target.group(1))
    return None


def _visible_lethal_repair(user_text: str, bad_action_index: int | None) -> tuple[int, str] | None:
    legal_text = _legal_action_section(user_text)
    for hint in _LETHAL_HINT_RE.finditer(user_text):
        hand_index = int(hint.group(1))
        enemy_id = int(hint.group(2))
        action_index = _find_legal_index_for_hand_target(legal_text, hand_index, enemy_id)
        if action_index is not None and action_index != bad_action_index:
            return action_index, f"take visible lethal on enemy{enemy_id}"
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_rows(dataset_dir: Path, *, include_eval: bool) -> list[tuple[str, int, dict[str, Any]]]:
    files = [dataset_dir / "train.jsonl"]
    if include_eval:
        files.append(dataset_dir / "eval.jsonl")
    rows: list[tuple[str, int, dict[str, Any]]] = []
    for path in files:
        for index, row in enumerate(read_jsonl(path), start=1):
            rows.append((path.name, index, row))
    return rows


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"dataset not found: {dataset_dir}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else dataset_dir / "offline_mining"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_rows(dataset_dir, include_eval=args.include_eval)
    hard_cases: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()

    for source_file, line_no, row in raw_rows:
        payload = _row_payload(row, source_file=source_file, line_no=line_no)
        user = _message(row, "user")
        assistant = _message(row, "assistant")
        grouped[payload["user_hash"]].append(payload)

        flags = payload["flags"]
        if flags or (payload["advantage"] is not None and float(payload["advantage"]) < -0.5):
            hard_cases.append(payload)
            for flag in flags:
                counters[f"hard_flag:{flag}"] += 1

        if "missed_visible_lethal" in flags:
            repair = _visible_lethal_repair(user, _assistant_action_index(assistant))
            if repair is not None:
                action_index, reason = repair
                chosen = _json_action(action_index, reason)
                repairs.append({
                    "messages": [
                        *_messages_without_assistant(row),
                        {"role": "assistant", "content": chosen},
                    ],
                    "meta": {
                        **(row.get("meta") if isinstance(row.get("meta"), dict) else {}),
                        "repair_source": "visible_lethal",
                        "original_assistant": assistant,
                        "original_action_index": _assistant_action_index(assistant),
                        "repaired_action_index": action_index,
                    },
                })
                counters["repairs_visible_lethal"] += 1

    preference_pairs: list[dict[str, Any]] = []
    for user_hash, rows in grouped.items():
        if len(rows) < 2:
            continue
        unique = {}
        for row in rows:
            action_index = row.get("action_index")
            if action_index is None:
                continue
            existing = unique.get(action_index)
            if existing is None or row["score"] > existing["score"]:
                unique[action_index] = row
        ranked = sorted(unique.values(), key=lambda item: item["score"], reverse=True)
        if len(ranked) < 2:
            continue
        chosen = ranked[0]
        rejected = ranked[-1]
        if chosen["score"] - rejected["score"] < args.min_score_gap:
            continue
        preference_pairs.append({
            "messages": _messages_without_assistant({"messages": chosen["messages"]}),
            "chosen": _message({"messages": chosen["messages"]}, "assistant"),
            "rejected": _message({"messages": rejected["messages"]}, "assistant"),
            "meta": {
                "pair_source": "same_prompt_rollout_score",
                "user_hash": user_hash,
                "score_gap": round(chosen["score"] - rejected["score"], 4),
                "chosen_score": chosen["score"],
                "rejected_score": rejected["score"],
                "chosen_action_index": chosen.get("action_index"),
                "rejected_action_index": rejected.get("action_index"),
                "chosen_flags": chosen.get("flags"),
                "rejected_flags": rejected.get("flags"),
                "encounter_key": chosen.get("encounter_key"),
                "encounter_label": chosen.get("encounter_label"),
            },
        })
        counters["preference_pairs_same_prompt"] += 1

    # Rule repairs are also valid preference pairs when the original action was bad.
    for row in repairs:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        original = str(meta.get("original_assistant") or "")
        chosen = _message(row, "assistant")
        if original and chosen:
            preference_pairs.append({
                "messages": _messages_without_assistant(row),
                "chosen": chosen,
                "rejected": original,
                "meta": {
                    "pair_source": "rule_visible_lethal_repair",
                    "score_gap": None,
                    "chosen_action_index": meta.get("repaired_action_index"),
                    "rejected_action_index": meta.get("original_action_index"),
                    "encounter_key": meta.get("encounter_key"),
                    "encounter_label": meta.get("encounter_label"),
                },
            })
            counters["preference_pairs_rule_repair"] += 1

    hard_cases.sort(key=lambda row: (len(row.get("flags") or []), -row["score"]), reverse=True)
    if args.hard_case_limit > 0:
        hard_cases = hard_cases[: args.hard_case_limit]

    _write_jsonl(out_dir / "hard_cases.jsonl", hard_cases)
    _write_jsonl(out_dir / "preference_pairs.jsonl", preference_pairs)
    _write_jsonl(out_dir / "repair_sft.jsonl", repairs)
    summary = {
        "kind": "offline_preference_mining",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "rows_scanned": len(raw_rows),
        "hard_cases": len(hard_cases),
        "preference_pairs": len(preference_pairs),
        "repair_sft_rows": len(repairs),
        "counters": {key: int(value) for key, value in counters.most_common()},
        "outputs": {
            "hard_cases": str(out_dir / "hard_cases.jsonl"),
            "preference_pairs": str(out_dir / "preference_pairs.jsonl"),
            "repair_sft": str(out_dir / "repair_sft.jsonl"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    # Also copy a short pointer under Artifacts for easy discovery.
    pointer_dir = ARTIFACTS_ROOT / "offline_mining"
    pointer_dir.mkdir(parents=True, exist_ok=True)
    write_json(pointer_dir / f"{dataset_dir.name}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
