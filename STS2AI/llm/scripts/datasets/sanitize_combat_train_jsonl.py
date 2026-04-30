"""Sanitize legacy combat train.jsonl files to the v4 schema.

Pre-iter07 combat SFT data has assistant content like
``{"action_index":N,"confidence":0.x,"reason":"<canonical or model-hallucinated>"}``.
The current combat policy schema (system_prompt v4) expects only
``{"action_index":N,"confidence":0.x}`` — reasoning is the planner LoRA's
job. Mixing the two schemas during training confuses the model
(sometimes outputs reason, sometimes doesn't).

This script reads one or more legacy ``train.jsonl`` files and emits a
cleaned copy where:

1. Every assistant JSON payload is re-serialized without the ``reason``
   field. ``action_index`` and ``confidence`` are preserved verbatim.
2. The user-message ``Return strict JSON only:`` line is rewritten to
   the v4 instruction so the model never sees a request to emit ``reason``.
3. Rows whose ``meta.action_quality_flags`` (or ``meta.quality_flags``)
   intersects ``TRAINING_BLOCKLIST_FLAGS`` are dropped — these are
   dangerous_end_turn / dangerous_self_damage / missed_visible_lethal /
   reason_math_contradiction etc. that we don't want as positive
   training targets.
4. Rows where assistant.action_index is negative (invalid_output
   placeholder from old rollouts) are dropped.

Usage::

    python -m llm.scripts.datasets.sanitize_combat_train_jsonl \\
        --input STS2AI/Artifacts/llm/datasets/iter05c_rollout/train.jsonl \\
        --input STS2AI/Artifacts/llm/datasets/iter06_rollout/train.jsonl \\
        --out-dir STS2AI/Artifacts/llm/datasets/combined_clean_train

Outputs ``train.jsonl`` (cleaned), ``dropped.jsonl`` (one row per
discarded sample with the reason), and ``summary.json`` (counts by
discard reason).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.action_quality import TRAINING_BLOCKLIST_FLAGS  # noqa: E402


# v4 instruction line that combat user messages should end with.
_V4_RETURN_LINE = (
    'Return strict JSON only: {"action_index":N,"confidence":0.0} '
    "using one listed action_index. Do not output multiple objects or candidates. "
    "Do not include a reason / plan / extra keys — strategy text belongs to the planner model."
)
# Match any legacy "Return strict JSON only ..." or "Return one JSON line ..." line.
_RETURN_RE = re.compile(r"^Return (?:one JSON line|strict JSON only): .*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        action="append",
        default=[],
        help="Legacy train.jsonl path. Repeat to merge several iterations.",
    )
    p.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help="Glob pattern (relative or absolute) — every match is treated as --input.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory; will hold train.jsonl + dropped.jsonl + summary.json.",
    )
    p.add_argument(
        "--keep-blocked-flags",
        action="store_true",
        help="Skip the TRAINING_BLOCKLIST_FLAGS filter (debug only — these flags are "
             "dangerous as positive training targets).",
    )
    p.add_argument(
        "--keep-action-index-negative",
        action="store_true",
        help="Skip the action_index >= 0 filter (debug only).",
    )
    return p.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in args.input:
        path = Path(raw).resolve()
        if path.exists() and path not in seen:
            out.append(path)
            seen.add(path)
    for pattern in args.input_glob:
        # Use Path.glob if pattern is relative, otherwise treat as fully-qualified glob.
        p = Path(pattern)
        if p.is_absolute():
            base = p.anchor
            rel = str(p.relative_to(p.anchor))
        else:
            base = "."
            rel = pattern
        for match in Path(base).glob(rel):
            mp = match.resolve()
            if mp.exists() and mp.is_file() and mp not in seen:
                out.append(mp)
                seen.add(mp)
    return out


def _strip_reason_from_assistant(content: str) -> tuple[str, bool]:
    """Re-serialize an assistant JSON payload without ``reason``.

    Returns ``(new_content, dropped_reason)`` where ``dropped_reason`` is
    True when the original had a reason field. Falls through to the
    raw content unchanged when the JSON cannot be parsed (keeps
    permissive behaviour for any odd legacy rows).
    """
    text = (content or "").strip()
    if not text:
        return content, False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return content, False
    if not isinstance(payload, dict):
        return content, False
    had_reason = "reason" in payload
    payload.pop("reason", None)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), had_reason


def _normalize_user_return_line(content: str) -> str:
    """Rewrite the trailing 'Return strict JSON ...' instruction line to the
    v4 form (no reason field). Adds it if the user message was missing one.
    """
    text = content or ""
    if _RETURN_RE.search(text):
        return _RETURN_RE.sub(_V4_RETURN_LINE, text)
    # Older traces may have the line on the very last logical line; if not,
    # appending the v4 instruction is harmless.
    return f"{text.rstrip()}\n{_V4_RETURN_LINE}"


def _row_quality_flags(row: dict[str, Any]) -> set[str]:
    """Pull both the per-step ``quality_flags`` and any merged
    ``action_quality_flags`` blob out of meta for blocklist matching."""
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    flags: set[str] = set()
    for key in ("quality_flags", "action_quality_flags", "step_quality_flags"):
        raw = meta.get(key)
        if isinstance(raw, list):
            flags.update(str(f) for f in raw)
        elif isinstance(raw, str) and raw:
            flags.add(raw)
    # rollout/teacher rows sometimes carry the per-row flag list at the top
    raw = row.get("quality_flags")
    if isinstance(raw, list):
        flags.update(str(f) for f in raw)
    return flags


def _assistant_action_index(row: dict[str, Any]) -> int | None:
    msgs = row.get("messages") or []
    if not isinstance(msgs, list):
        return None
    for msg in reversed(msgs):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    return None
                value = payload.get("action_index") if isinstance(payload, dict) else None
                if isinstance(value, bool):
                    return None
                return int(value) if isinstance(value, int) else None
    return None


def sanitize_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Return ``(cleaned_row_or_None, drop_reason_or_'kept')``."""
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        return None, "no_messages"

    new_msgs: list[dict[str, Any]] = []
    had_reason = False
    user_seen = False
    for msg in msgs:
        if not isinstance(msg, dict):
            new_msgs.append(msg)
            continue
        role = str(msg.get("role") or "")
        content = msg.get("content") or ""
        if role == "assistant":
            new_content, dropped_reason = _strip_reason_from_assistant(content)
            had_reason = had_reason or dropped_reason
            new_msgs.append({**msg, "content": new_content})
        elif role == "user":
            user_seen = True
            new_msgs.append({**msg, "content": _normalize_user_return_line(content)})
        else:
            new_msgs.append({**msg, "content": content})

    if not user_seen:
        return None, "no_user_message"

    new_row = {**row, "messages": new_msgs}
    meta = dict(new_row.get("meta") or {})
    meta["sanitized"] = True
    if had_reason:
        meta["legacy_reason_stripped"] = True
    new_row["meta"] = meta
    return new_row, "kept"


def main() -> int:
    args = parse_args()
    inputs = _resolve_inputs(args)
    if not inputs:
        raise SystemExit("no input files matched")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    dropped_path = out_dir / "dropped.jsonl"

    counters: Counter[str] = Counter()
    legacy_reason_count = 0
    kept = 0
    with train_path.open("w", encoding="utf-8") as fout, dropped_path.open("w", encoding="utf-8") as fdrop:
        for src in inputs:
            with src.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.rstrip("\r\n")
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        counters["json_decode_error"] += 1
                        fdrop.write(json.dumps({
                            "source": str(src), "line": line_no, "drop_reason": "json_decode_error",
                            "error": str(exc),
                        }, ensure_ascii=False) + "\n")
                        continue

                    flags = _row_quality_flags(row)
                    if not args.keep_blocked_flags and flags & TRAINING_BLOCKLIST_FLAGS:
                        bad = sorted(flags & TRAINING_BLOCKLIST_FLAGS)
                        counters[f"blocklist:{bad[0]}"] += 1
                        counters["dropped_blocked_flag"] += 1
                        fdrop.write(json.dumps({
                            "source": str(src), "line": line_no, "drop_reason": "blocklist_flag",
                            "flags": bad,
                        }, ensure_ascii=False) + "\n")
                        continue

                    if not args.keep_action_index_negative:
                        action_index = _assistant_action_index(row)
                        if action_index is None or action_index < 0:
                            counters["dropped_negative_action_index"] += 1
                            fdrop.write(json.dumps({
                                "source": str(src), "line": line_no,
                                "drop_reason": "non_positive_action_index",
                                "action_index": action_index,
                            }, ensure_ascii=False) + "\n")
                            continue

                    cleaned, status = sanitize_row(row)
                    if cleaned is None:
                        counters[f"dropped_{status}"] += 1
                        fdrop.write(json.dumps({
                            "source": str(src), "line": line_no, "drop_reason": status,
                        }, ensure_ascii=False) + "\n")
                        continue
                    if cleaned["meta"].get("legacy_reason_stripped"):
                        legacy_reason_count += 1
                    fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    kept += 1

    summary = {
        "kind": "combat_train_sanitized",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": [str(p) for p in inputs],
        "out_dir": str(out_dir),
        "train_path": str(train_path),
        "dropped_path": str(dropped_path),
        "kept_rows": kept,
        "legacy_reason_stripped": legacy_reason_count,
        "drop_counters": dict(counters),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
