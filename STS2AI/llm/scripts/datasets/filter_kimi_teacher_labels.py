"""Filter Kimi teacher labels with local semantic sanity checks."""

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

from llm.paths import ARTIFACTS_ROOT, ensure_dirs  # noqa: E402
from llm.scripts.analysis.analyze_action_ordering import _legal_actions  # noqa: E402
from llm.scripts.teacher.sample_kimi_teacher_candidates import _enemies, _is_lethal  # noqa: E402


_KILL_CLAIM_RE = re.compile(r"\b(kill|kills|lethal|visible lethal|eliminate|eliminates)\b", re.IGNORECASE)
_SELF_CONTRADICT_RE = re.compile(r"\b(no,|actually|however)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="valid_labels.jsonl from run_kimi_teacher_candidate_reviews.py")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _action_by_index(user_message: str, action_index: int | None) -> dict[str, Any] | None:
    if action_index is None:
        return None
    for action in _legal_actions({"user_message": user_message}):
        if action.get("index") == action_index:
            return action
    return None


def reject_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    confidence = float(row.get("confidence") or 0.0)
    if confidence < 0.7:
        reasons.append("low_confidence")
    user = str(row.get("user_message") or "")
    action_index = row.get("best_action_index")
    if isinstance(action_index, bool) or not isinstance(action_index, int):
        reasons.append("best_action_index_not_int")
        return reasons
    action = _action_by_index(user, action_index)
    if action is None:
        reasons.append("best_action_index_not_legal")
        return reasons

    enemies = _enemies(user)
    is_lethal = _is_lethal(action, enemies)
    reason = f"{row.get('reason_en') or ''} {row.get('reason_zh') or ''}"
    tags = [str(tag).lower() for tag in (row.get("mechanism_tags") or [])]
    if _KILL_CLAIM_RE.search(reason) and not is_lethal:
        reasons.append("claims_lethal_but_action_not_lethal")
    if "visible_lethal" in tags and not is_lethal:
        reasons.append("visible_lethal_tag_but_action_not_lethal")
    if confidence >= 0.9 and _SELF_CONTRADICT_RE.search(reason):
        reasons.append("self_contradictory_high_confidence")
    return reasons


def main() -> int:
    args = parse_args()
    ensure_dirs()
    labels_path = Path(args.labels).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        ARTIFACTS_ROOT / "reviews" / f"kimi_teacher_filtered_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    rows = _read_jsonl(labels_path)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reasons = reject_reasons(row)
        reason_counts.update(reasons)
        if reasons:
            rejected.append({**row, "filter_reject_reasons": reasons})
        else:
            kept.append(row)

    _write_jsonl(out_dir / "kept_labels.jsonl", kept)
    _write_jsonl(out_dir / "rejected_labels.jsonl", rejected)
    summary = {
        "kind": "kimi_teacher_label_filter",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "labels": str(labels_path),
        "out_dir": str(out_dir),
        "input_labels": len(rows),
        "kept_labels": len(kept),
        "rejected_labels": len(rejected),
        "changed_input": sum(1 for row in rows if row.get("best_action_index") != row.get("original_action_index")),
        "changed_kept": sum(1 for row in kept if row.get("best_action_index") != row.get("original_action_index")),
        "reject_reason_counts": dict(reason_counts.most_common()),
        "outputs": {
            "kept_labels": str(out_dir / "kept_labels.jsonl"),
            "rejected_labels": str(out_dir / "rejected_labels.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
