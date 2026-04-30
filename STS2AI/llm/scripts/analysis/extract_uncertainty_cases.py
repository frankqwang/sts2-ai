"""Extract low/high action-score margin cases from LLM step traces."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.action_decoder import action_score_margin  # noqa: E402
from llm.data_pipeline.action_quality import assess_action_quality_report  # noqa: E402
from llm.paths import ARTIFACTS_ROOT, ensure_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-trace", required=True, help="step_trace.jsonl from rollout/eval/spectate.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit-per-bucket", type=int, default=80)
    parser.add_argument("--low-margin", type=float, default=1.0)
    parser.add_argument("--high-margin", type=float, default=5.0)
    parser.add_argument("--low-confidence", type=float, default=0.55)
    parser.add_argument(
        "--recompute-quality",
        action="store_true",
        help="Recompute quality_flags from state/legal_actions with the current diagnostic code.",
    )
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


def _decoded(row: dict[str, Any]) -> dict[str, Any]:
    decoded = row.get("decoded")
    if isinstance(decoded, dict):
        return decoded
    if any(key in row for key in ("confidence", "action_scores", "generated_action_index")):
        return {
            "action_index": row.get("generated_action_index", row.get("action_index")),
            "confidence": row.get("confidence"),
            "action_scores": row.get("action_scores") or [],
            "reason": row.get("reason"),
        }
    assistant = row.get("assistant")
    if isinstance(assistant, str) and assistant.strip():
        try:
            payload = json.loads(assistant)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    attempts = row.get("attempts")
    if isinstance(attempts, list) and attempts:
        last = attempts[-1]
        if isinstance(last, dict) and isinstance(last.get("decoded"), dict):
            return dict(last["decoded"])
    return {}


def _as_confidence(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def _scores(decoded: dict[str, Any]) -> list[dict[str, Any]]:
    raw = decoded.get("action_scores")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("action_index"))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        out.append({"action_index": index, "score": round(score, 4), "note": str(item.get("note") or "")[:80]})
    return sorted(out, key=lambda item: float(item["score"]), reverse=True)


def _with_recomputed_quality(row: dict[str, Any]) -> dict[str, Any]:
    decoded = _decoded(row)
    if not isinstance(row.get("state"), dict) or not isinstance(row.get("legal_actions"), list):
        return row
    action_index = decoded.get("action_index")
    if not isinstance(action_index, int):
        return row
    scores = decoded.get("action_scores")
    report = assess_action_quality_report(
        row["state"],
        row["legal_actions"],
        action_index,
        reason=str(decoded.get("reason") or ""),
        action_scores=scores if isinstance(scores, list) else None,
    ).as_dict()
    out = dict(row)
    out["quality_report"] = report
    out["quality_flags"] = list(report.get("flags") or [])
    return out


def _project(row: dict[str, Any]) -> dict[str, Any]:
    decoded = _decoded(row)
    scores = _scores(decoded)
    margin = action_score_margin(scores)
    user = str(row.get("user_message") or "")
    parse_status = row.get("parse_status") or row.get("strict_json_status")
    invalid_output = bool(row.get("invalid_output")) or (
        isinstance(parse_status, str) and parse_status not in ("", "ok")
    )
    return {
        "row_index": row.get("row_index"),
        "episode_id": row.get("episode_id"),
        "step": row.get("step", row.get("episode_step")),
        "encounter_id": row.get("encounter_id"),
        "outcome": row.get("outcome"),
        "action_index": decoded.get("action_index"),
        "target_action_index": row.get("target_action_index"),
        "action_valid": row.get("action_valid"),
        "action_exact": row.get("action_exact"),
        "parse_status": parse_status,
        "strict_json_status": row.get("strict_json_status"),
        "confidence": _as_confidence(decoded.get("confidence")),
        "score_margin": margin,
        "action_scores": scores,
        "reason": decoded.get("reason"),
        "quality_flags": row.get("quality_flags") or [],
        "invalid_output": invalid_output,
        "chosen_action": row.get("chosen_action"),
        "raw_generation": row.get("raw_generation"),
        "user_excerpt": user[:1800],
    }


def bucket_cases(
    rows: list[dict[str, Any]],
    *,
    low_margin: float,
    high_margin: float,
    low_confidence: float,
    recompute_quality: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "low_margin": [],
        "high_margin": [],
        "low_confidence": [],
        "high_confidence_low_margin": [],
        "low_confidence_high_margin": [],
        "high_margin_with_flags": [],
        "invalid_output": [],
    }
    for row in rows:
        if recompute_quality:
            row = _with_recomputed_quality(row)
        item = _project(row)
        margin = item.get("score_margin")
        confidence = item.get("confidence")
        flags = item.get("quality_flags") or []
        if item.get("invalid_output"):
            buckets["invalid_output"].append(item)
        if isinstance(confidence, float) and confidence <= low_confidence:
            buckets["low_confidence"].append(item)
        if isinstance(margin, float):
            if margin <= low_margin:
                buckets["low_margin"].append(item)
                if isinstance(confidence, float) and confidence >= 0.85:
                    buckets["high_confidence_low_margin"].append(item)
            if margin >= high_margin:
                buckets["high_margin"].append(item)
                if isinstance(confidence, float) and confidence <= low_confidence:
                    buckets["low_confidence_high_margin"].append(item)
                if flags:
                    buckets["high_margin_with_flags"].append(item)
    return buckets


def _default_out_dir(step_trace: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ARTIFACTS_ROOT / "diagnostics" / f"uncertainty_{step_trace.stem}_{stamp}"


def _write_examples(path: Path, buckets: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Uncertainty Cases", ""]
    for name, rows in buckets.items():
        lines.append(f"## {name}")
        lines.append("")
        for item in rows[:5]:
            lines.append(
                f"- step=`{item.get('step')}` action=`{item.get('action_index')}` "
                f"confidence=`{item.get('confidence')}` margin=`{item.get('score_margin')}` "
                f"flags=`{','.join(map(str, item.get('quality_flags') or [])) or '-'}`"
            )
            if item.get("reason"):
                lines.append(f"  reason: {item.get('reason')}")
            if item.get("action_scores"):
                lines.append(f"  scores: `{json.dumps(item.get('action_scores'), ensure_ascii=False)}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    step_trace = Path(args.step_trace).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_out_dir(step_trace)
    rows = _read_jsonl(step_trace)
    buckets = bucket_cases(
        rows,
        low_margin=args.low_margin,
        high_margin=args.high_margin,
        low_confidence=args.low_confidence,
        recompute_quality=args.recompute_quality,
    )
    if args.recompute_quality:
        rows = [_with_recomputed_quality(row) for row in rows]

    limited: dict[str, list[dict[str, Any]]] = {}
    for name, items in buckets.items():
        if name == "low_margin":
            selected = sorted(items, key=lambda item: float(item.get("score_margin") or 0.0))
        elif name == "low_confidence":
            selected = sorted(items, key=lambda item: float(item.get("confidence") or 0.0))
        else:
            selected = sorted(items, key=lambda item: float(item.get("score_margin") or 0.0), reverse=True)
        limited[name] = selected[: max(0, args.limit_per_bucket)]
        _write_jsonl(out_dir / f"{name}.jsonl", limited[name])

    summary = {
        "kind": "uncertainty_cases",
        "step_trace": str(step_trace),
        "out_dir": str(out_dir),
        "rows": len(rows),
        "thresholds": {
            "low_margin": args.low_margin,
            "high_margin": args.high_margin,
            "low_confidence": args.low_confidence,
        },
        "bucket_counts": {name: len(items) for name, items in buckets.items()},
        "limited_counts": {name: len(items) for name, items in limited.items()},
        "quality_flags": dict(Counter(flag for row in rows for flag in (row.get("quality_flags") or []))),
        "recompute_quality": bool(args.recompute_quality),
    }
    _write_json(out_dir / "summary.json", summary)
    _write_examples(out_dir / "examples.md", limited)
    print(f"[uncertainty] output -> {out_dir}")


if __name__ == "__main__":
    main()
