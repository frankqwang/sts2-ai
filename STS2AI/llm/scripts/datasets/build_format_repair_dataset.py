"""Build a strict-JSON repair dataset from clean rows and rollout traces."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.paths import DATASETS_ROOT, ensure_dirs  # noqa: E402
from llm.prompts import load_system_prompt  # noqa: E402
from llm.scripts.analysis.analyze_action_ordering import _legal_actions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", action="append", default=[], help="Existing train/eval dataset dir")
    parser.add_argument("--trace", action="append", default=[], help="step_trace.jsonl to normalize")
    parser.add_argument("--out-dir", default=str(DATASETS_ROOT / f"format_repair_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    parser.add_argument("--max-source-rows", type=int, default=2500)
    parser.add_argument("--max-trace-rows", type=int, default=1500)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument(
        "--compact-format-prompt",
        action="store_true",
        help="Use short format-repair prompts instead of full combat state prompts.",
    )
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


def _strict_json_assistant(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    assistant = messages[-1] if isinstance(messages[-1], dict) else {}
    if assistant.get("role") != "assistant":
        return False
    try:
        payload = json.loads(str(assistant.get("content") or ""))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("action_index"), int)


def _source_rows(dataset_dir: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = _read_jsonl(dataset_dir / "train.jsonl") + _read_jsonl(dataset_dir / "eval.jsonl")
    rows = [row for row in rows if _strict_json_assistant(row)]
    rng.shuffle(rows)
    out: list[dict[str, Any]] = []
    for row in rows[: max(0, limit)]:
        meta = dict(row.get("meta") if isinstance(row.get("meta"), dict) else {})
        meta["source"] = meta.get("source") or "strict_source_dataset"
        meta["source_dataset"] = str(dataset_dir)
        meta["advantage"] = max(1.0, float(meta.get("advantage") or 1.0))
        out.append({"messages": row["messages"], "meta": meta})
    return out


_FORMAT_SYSTEM_PROMPT = (
    "You are a strict JSON action formatter. Return exactly one JSON object and no extra text. "
    "The JSON schema is {\"action_index\":N,\"confidence\":0.0,\"reason\":\"...\"}. "
    "Use only listed action_index values."
)


def _assistant_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    assistant = messages[-1] if isinstance(messages[-1], dict) else {}
    try:
        payload = json.loads(str(assistant.get("content") or ""))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or isinstance(payload.get("action_index"), bool) or not isinstance(payload.get("action_index"), int):
        return None
    return _normalize_payload(payload, int(payload["action_index"]))


def _normalize_payload(payload: dict[str, Any], action_index: int) -> dict[str, Any]:
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = 0.9
    return {
        "action_index": int(action_index),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "reason": str(payload.get("reason") or "selected legal action").strip()[:120],
    }


def _compact_action_label(action: dict[str, Any]) -> str:
    index = action.get("index")
    card = str(action.get("card_id") or action.get("label") or "action").strip() or "action"
    target = str(action.get("target") or action.get("target_id") or "").strip()
    parts = [f"[{index}] {card}"]
    if target and target not in {"None", "-1"}:
        parts.append(f"target={target}")
    for key in ("damage", "block"):
        value = action.get(key)
        if isinstance(value, int):
            parts.append(f"{key}={value}")
    raw = str(action.get("raw") or "").strip()
    if raw and len(parts) == 1:
        parts.append(raw[:80])
    return " ".join(parts)


def _compact_user_message(row: dict[str, Any], user_message: str) -> str:
    actions = _legal_actions({"user_message": user_message, "legal_actions": row.get("legal_actions")})
    lines = [
        "Choose one legal action and return strict JSON only.",
        "legal_actions:",
    ]
    for action in actions[:24]:
        if isinstance(action.get("index"), int):
            lines.append(f"  {_compact_action_label(action)}")
    if len(actions) > 24:
        lines.append(f"  ... {len(actions) - 24} more legal actions omitted")
    return "\n".join(lines)


def _compact_row(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = _assistant_payload(row)
    if payload is None:
        return None
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    user = str(messages[1].get("content") if len(messages) > 1 and isinstance(messages[1], dict) else "")
    if not user:
        return None
    compact = {
        "messages": [
            {"role": "system", "content": _FORMAT_SYSTEM_PROMPT},
            {"role": "user", "content": _compact_user_message(row, user)},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "meta": dict(row.get("meta") if isinstance(row.get("meta"), dict) else {}),
    }
    compact["meta"]["compact_format_prompt"] = True
    return compact


def _legal_count(row: dict[str, Any]) -> int:
    legal = row.get("legal_actions")
    return len(legal) if isinstance(legal, list) else 0


def _valid_trace_row(row: dict[str, Any]) -> bool:
    if row.get("outcome") != "victory":
        return False
    if row.get("invalid_output"):
        return False
    if row.get("quality_flags"):
        return False
    user = str(row.get("user_message") or "")
    if not user or _legal_count(row) <= 1:
        return False
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    action_index = decoded.get("action_index")
    return isinstance(action_index, int) and 0 <= action_index < _legal_count(row)


def _trace_assistant_payload(row: dict[str, Any]) -> dict[str, Any]:
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    action_index = int(decoded.get("action_index"))
    confidence = decoded.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = 0.9
    reason = str(decoded.get("reason") or "").strip() or "executed legal action"
    return _normalize_payload(
        {
            "action_index": action_index,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "reason": reason[:160],
        },
        action_index,
    )


def _trace_rows(trace_path: Path, limit: int, rng: random.Random, *, compact_format_prompt: bool = False) -> list[dict[str, Any]]:
    rows = [row for row in _read_jsonl(trace_path) if _valid_trace_row(row)]
    rng.shuffle(rows)
    system_prompt = load_system_prompt()
    out: list[dict[str, Any]] = []
    for row in rows[: max(0, limit)]:
        payload = _trace_assistant_payload(row)
        full_row = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(row.get("user_message") or "")},
                {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            "meta": {
                "source": "skada_trace_format_repair",
                "source_trace": str(trace_path),
                "episode_id": row.get("episode_id"),
                "step": row.get("episode_step", row.get("step")),
                "outcome": row.get("outcome"),
                "case_metadata": row.get("case_metadata") if isinstance(row.get("case_metadata"), dict) else {},
                "floor": (row.get("case_metadata") or {}).get("floor") if isinstance(row.get("case_metadata"), dict) else None,
                "encounter_id": row.get("encounter_id"),
                "advantage": 1.0,
                "original_strict_json_ok": any(bool(a.get("strict_json_ok")) for a in (row.get("attempts") or []) if isinstance(a, dict)),
            },
            "legal_actions": row.get("legal_actions"),
        }
        if compact_format_prompt:
            compact = _compact_row(full_row)
            if compact is not None:
                out.append(compact)
        else:
            out.append(full_row)
    return out


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages") if isinstance(row.get("messages"), list) else []
        user = str(messages[1].get("content") if len(messages) > 1 and isinstance(messages[1], dict) else "")
        assistant = str(messages[-1].get("content") if messages and isinstance(messages[-1], dict) else "")
        key = (user, assistant)
        if not user or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> int:
    args = parse_args()
    ensure_dirs()
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for item in args.source_dataset:
        source = _source_rows(Path(item).resolve(), args.max_source_rows, rng)
        if args.compact_format_prompt:
            source = [row for row in (_compact_row(row) for row in source) if row is not None]
        rows.extend(source)
        counters[f"source:{item}"] = len(source)
    for item in args.trace:
        trace = _trace_rows(Path(item).resolve(), args.max_trace_rows, rng, compact_format_prompt=args.compact_format_prompt)
        rows.extend(trace)
        counters[f"trace:{item}"] = len(trace)

    rows = _dedupe(rows)
    rng.shuffle(rows)
    eval_n = max(1, int(len(rows) * max(0.0, min(0.5, args.eval_ratio)))) if len(rows) > 20 else 0
    eval_rows = rows[:eval_n]
    train_rows = rows[eval_n:]
    out_dir = Path(args.out_dir).resolve()
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    summary = {
        "kind": "format_repair_dataset",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "rows": len(rows),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "inputs": {
            "source_dataset": [str(Path(path).resolve()) for path in args.source_dataset],
            "trace": [str(Path(path).resolve()) for path in args.trace],
        },
        "counters": counters,
        "strict_json_verified": all(_strict_json_assistant(row) for row in rows),
        "compact_format_prompt": bool(args.compact_format_prompt),
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
