"""Hard-gate planner-hint datasets and rollout traces.

This script is intentionally strict. It accepts only the current v2 planner
schema and fails nonzero for legacy fields, action fields, missing retrieved
knowledge, or unparseable hints. It does not map old labels forward.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.planner_hint import (  # noqa: E402
    FORBIDDEN_KEYS,
    LIST_KEYS,
    TEXT_KEYS,
    parse_planner_hint_json,
)
from llm.paths import EVALS_ROOT, ensure_dirs  # noqa: E402


LEGACY_KEYS = {
    "combat_plan",
    "encounter_guide",
    "defense_policy",
    "resource_policy",
    "potion_policy",
}
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", default=[], help="planner train/eval jsonl or dataset dir")
    parser.add_argument("--trace", action="append", default=[], help="rollout step_trace.jsonl")
    parser.add_argument("--out-dir", default="", help="default: Artifacts/llm/evals/planner_hint_quality_<timestamp>")
    parser.add_argument("--require-knowledge", action="store_true")
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                rows.append({"_json_error": str(exc), "_line_no": line_no})
                continue
            if isinstance(payload, dict):
                payload.setdefault("_line_no", line_no)
                rows.append(payload)
            else:
                rows.append({"_json_error": "json_not_object", "_line_no": line_no})
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dataset_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in values:
        path = Path(raw).resolve()
        if path.is_dir():
            for name in ("train.jsonl", "eval.jsonl"):
                candidate = path / name
                if candidate.exists():
                    paths.append(candidate)
        elif path.exists():
            paths.append(path)
    return paths


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in (row.get("messages") or []) if isinstance(item, dict)]


def _last_message_content(row: dict[str, Any], role: str) -> str:
    for message in reversed(_messages(row)):
        if message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def _trace_hint_payload(raw_hint: Any) -> dict[str, Any] | None:
    text = str(raw_hint or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    if text.startswith("planner_hint:"):
        text = "\n".join(text.splitlines()[1:]).strip()
    payload: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in LIST_KEYS:
            parts = [part.strip() for part in re.split(r"\s*->\s*|;\s*", value) if part.strip()]
            payload[key] = parts
        else:
            payload[key] = value
    return payload or None


def _invalid_key_status(payload: dict[str, Any]) -> str:
    keys = {str(key).strip() for key in payload}
    legacy = sorted(keys & LEGACY_KEYS)
    if legacy:
        return f"legacy_field:{legacy[0]}"
    forbidden = sorted(key for key in keys if key.lower() in FORBIDDEN_KEYS)
    if forbidden:
        return f"forbidden_field:{forbidden[0]}"
    allowed = set(TEXT_KEYS) | set(LIST_KEYS)
    unknown = sorted(keys - allowed)
    if unknown:
        return f"unknown_field:{unknown[0]}"
    return ""


def _knowledge_present(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, dict):
        return bool(value)
    return bool(str(value or "").strip())


def _check_dataset_row(row: dict[str, Any], *, require_knowledge: bool) -> str:
    if row.get("_json_error"):
        return f"json_error:{row['_json_error']}"
    assistant = _last_message_content(row, "assistant")
    user = _last_message_content(row, "user")
    if not assistant:
        return "missing_assistant"
    hint, status = parse_planner_hint_json(assistant)
    if status != "ok" or hint is None:
        return status
    bad_key = _invalid_key_status(hint)
    if bad_key:
        return bad_key
    if require_knowledge and "retrieved_knowledge:" not in user:
        return "missing_retrieved_knowledge"
    if "legal_actions:" in user:
        return "planner_prompt_contains_legal_actions"
    return "ok"


def _check_trace_row(row: dict[str, Any], *, require_knowledge: bool) -> str:
    if row.get("_json_error"):
        return f"json_error:{row['_json_error']}"
    status = str(row.get("planner_hint_status") or "").strip().lower()
    if status == "disabled" and str(row.get("route") or "") == "heuristic_forced":
        return "skipped_forced"
    if status and status not in {"ok", "cache_hit"}:
        return f"planner_hint_status:{status}"
    raw_hint = row.get("planner_hint")
    payload = _trace_hint_payload(raw_hint)
    if payload is None:
        return "missing_planner_hint"
    bad_key = _invalid_key_status(payload)
    if bad_key:
        return bad_key
    hint, parse_status = parse_planner_hint_json(json.dumps(payload, ensure_ascii=False))
    if parse_status != "ok" or hint is None:
        return parse_status
    if _CJK_RE.search(str(raw_hint or "")):
        return "non_english_text"
    if require_knowledge and not _knowledge_present(row.get("retrieved_knowledge")):
        return "missing_retrieved_knowledge"
    return "ok"


def _sample_failure(path: Path, row: dict[str, Any], status: str, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "line_no": row.get("_line_no"),
        "status": status,
        "episode_id": row.get("episode_id"),
        "step": row.get("step") or row.get("episode_step"),
        "planner_hint_status": row.get("planner_hint_status"),
        "planner_hint": row.get("planner_hint"),
    }


def main() -> int:
    args = parse_args()
    ensure_dirs()
    dataset_paths = _dataset_paths(args.dataset)
    trace_paths = [Path(raw).resolve() for raw in args.trace if Path(raw).exists()]
    if not dataset_paths and not trace_paths:
        raise SystemExit("no dataset or trace inputs found")

    counters: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    for path in dataset_paths:
        for row in _read_jsonl(path):
            status = _check_dataset_row(row, require_knowledge=args.require_knowledge)
            counters[f"dataset:{status}"] += 1
            if status != "ok" and len(failures) < args.examples:
                failures.append(_sample_failure(path, row, status, "dataset"))

    for path in trace_paths:
        for row in _read_jsonl(path):
            status = _check_trace_row(row, require_knowledge=args.require_knowledge)
            counters[f"trace:{status}"] += 1
            if status not in {"ok", "skipped_forced"} and len(failures) < args.examples:
                failures.append(_sample_failure(path, row, status, "trace"))

    invalid = sum(
        count
        for key, count in counters.items()
        if not key.endswith(":ok") and not key.endswith(":skipped_forced")
    )
    out_dir = Path(args.out_dir).resolve() if args.out_dir else EVALS_ROOT / f"planner_hint_quality_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    summary = {
        "kind": "planner_hint_quality",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_paths": [str(path) for path in dataset_paths],
        "trace_paths": [str(path) for path in trace_paths],
        "require_knowledge": bool(args.require_knowledge),
        "status_counts": dict(counters),
        "invalid": invalid,
        "failures": failures,
    }
    _write_json(out_dir / "summary.json", summary)
    print(out_dir)
    if invalid:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
