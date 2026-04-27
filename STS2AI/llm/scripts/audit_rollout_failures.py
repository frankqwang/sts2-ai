"""Audit rollout failures and hard cases from episode/step traces.

This pass is deliberately conservative: it does not fix or train anything.
It turns every invalid / abnormal / HP-loss case into a durable artifact so the
self-training loop cannot silently move past bad rollouts.
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

from llm.paths import ARTIFACTS_ROOT, ensure_dirs  # noqa: E402
from llm.scripts.analyze_action_ordering import (  # noqa: E402
    _chosen,
    _energy,
    _legal_actions,
    _round,
)


_FLOOR_RE = re.compile(r"\bfloor=(\d+|\?)\b")
_PLAYER_HP_RE = re.compile(r"\bplayer:\s+hp=(\d+)/(\d+)\b")
_ERROR_MARKERS = ("Traceback", "Exception", "Error", "UnicodeDecodeError", "RuntimeError")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="", help="Directory containing episode_trace.jsonl and step_trace.jsonl.")
    parser.add_argument("--episode-trace", default="", help="Explicit episode_trace.jsonl.")
    parser.add_argument("--step-trace", default="", help="Explicit step_trace.jsonl.")
    parser.add_argument("--log", action="append", default=[], help="stderr/stdout log to scan for exceptions; repeatable.")
    parser.add_argument("--out-dir", default="", help="Default: Artifacts/llm/reviews/rollout_audit_<name>_<timestamp>")
    parser.add_argument("--top", type=int, default=200)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                value.setdefault("source_line", line_no)
                rows.append(value)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _floor_from_user(user_message: str) -> int | None:
    match = _FLOOR_RE.search(user_message)
    if not match or match.group(1) == "?":
        return None
    return int(match.group(1))


def _hp_from_user(user_message: str) -> tuple[int, int] | None:
    match = _PLAYER_HP_RE.search(user_message)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _case_meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("case_metadata")
    return meta if isinstance(meta, dict) else {}


def _floor(row: dict[str, Any], steps_by_episode: dict[str, list[dict[str, Any]]]) -> int | None:
    meta = _case_meta(row)
    raw = meta.get("floor")
    if isinstance(raw, int):
        return raw
    for step in steps_by_episode.get(str(row.get("episode_id") or ""), []):
        floor = _floor_from_user(str(step.get("user_message") or ""))
        if floor is not None:
            return floor
    return None


def _step_no(row: dict[str, Any]) -> int:
    try:
        return int(row.get("episode_step") if row.get("episode_step") is not None else row.get("step") or 0)
    except (TypeError, ValueError):
        return 0


def _step_action(row: dict[str, Any]) -> dict[str, Any]:
    actions = _legal_actions(row)
    chosen = _chosen(row, actions)
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    return {
        "step": row.get("episode_step", row.get("step")),
        "round": _round(str(row.get("user_message") or "")),
        "energy": _energy(str(row.get("user_message") or "")),
        "action_index": decoded.get("action_index"),
        "chosen": chosen,
        "reason": decoded.get("reason"),
        "invalid_output": bool(row.get("invalid_output")),
        "fallback_reason": decoded.get("fallback_reason"),
        "quality_flags": row.get("quality_flags") or [],
    }


def _first_problem_step(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in steps:
        if step.get("invalid_output") or step.get("quality_flags"):
            return _step_action(step)
    return _step_action(steps[-1]) if steps else None


def _turn_key(row: dict[str, Any]) -> tuple[str, int | None]:
    return str(row.get("episode_id") or ""), _round(str(row.get("user_message") or ""))


def _damage_turn_cases(steps: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ordered_steps = sorted(steps, key=lambda item: (str(item.get("episode_id") or ""), _step_no(item)))
    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    episode_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered_steps:
        grouped[_turn_key(row)].append(row)
        episode_steps[str(row.get("episode_id") or "")].append(row)

    cases: list[dict[str, Any]] = []
    for (episode_id, round_no), rows in grouped.items():
        if round_no is None:
            continue
        start_hp = _hp_from_user(str(rows[0].get("user_message") or ""))
        end_hp: tuple[int, int] | None = None
        ordered_episode = episode_steps.get(episode_id) or []
        last_index = ordered_episode.index(rows[-1]) if rows[-1] in ordered_episode else -1
        if 0 <= last_index < len(ordered_episode) - 1:
            end_hp = _hp_from_user(str(ordered_episode[last_index + 1].get("user_message") or ""))
        end_hp = end_hp or _hp_from_user(str(rows[-1].get("user_message") or ""))
        if not start_hp or not end_hp:
            continue
        hp_loss = max(0, start_hp[0] - end_hp[0])
        if hp_loss <= 0:
            continue
        flags = Counter(flag for row in rows for flag in (row.get("quality_flags") or []))
        cases.append({
            "episode_id": episode_id,
            "round": round_no,
            "observed_hp_loss": hp_loss,
            "start_hp": start_hp[0],
            "end_hp": end_hp[0],
            "steps": [_step_action(row) for row in rows],
            "flag_counts": dict(flags),
        })
    cases.sort(key=lambda row: (float(row.get("observed_hp_loss") or 0), len(row.get("steps") or [])), reverse=True)
    return cases[:limit]


def _cause_for_episode(ep: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    outcome = str(ep.get("outcome") or "")
    invalid_reason = str(ep.get("invalid_reason") or "")
    flags = Counter()
    for step in steps:
        flags.update(str(flag) for flag in (step.get("quality_flags") or []))
    if not invalid_reason and outcome.startswith("invalid_output:"):
        invalid_reason = outcome.split(":", 1)[1]

    if invalid_reason == "dangerous_end_turn" or flags.get("dangerous_end_turn", 0) > 0:
        return {
            "category": "unsafe_end_turn",
            "reason": "ended the turn while incoming damage or useful legal actions made end_turn unsafe",
            "next_action": "review the whole turn plan and prefer kill/block/mitigation before end_turn",
        }
    if "self_damage" in invalid_reason or flags.get("dangerous_self_damage", 0) > 0 or flags.get("low_hp_self_damage", 0) > 0:
        return {
            "category": "unsafe_self_damage",
            "reason": "selected self-damage when HP/safety gate considered it dangerous",
            "next_action": "add/label alternatives that avoid self-damage or prove the self-damage wins immediately",
        }
    if "json" in invalid_reason.lower() or "parse" in invalid_reason.lower() or "format" in invalid_reason.lower():
        return {
            "category": "protocol_format",
            "reason": "model output was not accepted by the strict JSON/action decoder",
            "next_action": "keep this out of training labels and add it to format-repair evaluation",
        }
    if outcome == "left_combat":
        return {
            "category": "left_combat",
            "reason": "bridge reported leaving combat before a victory/defeat outcome was recorded",
            "next_action": "inspect final settlement events to distinguish real escape, reset artifact, or bridge state mismatch",
        }
    if outcome in {"defeat", "loss"}:
        return {
            "category": "combat_loss",
            "reason": "policy lost the combat from this reset state",
            "next_action": "send the full combat plus high-loss turns to teacher review",
        }
    if outcome in {"max_steps", "timeout"} or "max" in outcome.lower():
        return {
            "category": "stall_or_loop",
            "reason": "episode reached a step/time cap instead of terminating cleanly",
            "next_action": "inspect repeated actions and add an anti-loop hardcase",
        }
    if flags:
        return {
            "category": "quality_flags",
            "reason": "episode won or continued but contained flagged decision quality issues",
            "next_action": "rank the flagged step as hardcase; train only verified corrected labels",
        }
    return {
        "category": "other_abnormal",
        "reason": "non-victory or suspicious episode did not match a known cause bucket",
        "next_action": "manual review required before this case is ignored",
    }


def _episode_case(
    ep: dict[str, Any],
    steps_by_episode: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    episode_id = str(ep.get("episode_id") or "")
    steps = steps_by_episode.get(episode_id, [])
    meta = _case_meta(ep)
    cause = _cause_for_episode(ep, steps)
    quality_summary = ep.get("quality_summary") if isinstance(ep.get("quality_summary"), dict) else {}
    return {
        "episode_id": episode_id,
        "encounter_id": ep.get("encounter_id"),
        "encounter_label": ep.get("encounter_label"),
        "encounter_tag": ep.get("encounter_tag"),
        "case_id": meta.get("case_id"),
        "run_id": meta.get("run_id"),
        "floor": _floor(ep, steps_by_episode),
        "outcome": ep.get("outcome"),
        "invalid_output": bool(ep.get("invalid_output")),
        "invalid_reason": ep.get("invalid_reason"),
        "steps": ep.get("steps"),
        "duration_s": ep.get("duration_s"),
        "reward": ep.get("reward"),
        "quality_flags": ep.get("quality_flags") or {},
        "quality_summary": {
            "hp_lost": quality_summary.get("hp_lost"),
            "turns": quality_summary.get("turns"),
            "defense_score": quality_summary.get("defense_score"),
            "mechanism_score": quality_summary.get("mechanism_score"),
        },
        "cause": cause,
        "first_problem_step": _first_problem_step(steps),
    }


def _severity(case: dict[str, Any]) -> float:
    outcome = str(case.get("outcome") or "")
    cause = case.get("cause") if isinstance(case.get("cause"), dict) else {}
    summary = case.get("quality_summary") if isinstance(case.get("quality_summary"), dict) else {}
    flags = case.get("quality_flags") if isinstance(case.get("quality_flags"), dict) else {}
    score = 0.0
    if case.get("invalid_output") or outcome.startswith("invalid_output:"):
        score += 100.0
    if outcome in {"defeat", "loss"}:
        score += 90.0
    if outcome == "left_combat":
        score += 70.0
    if cause.get("category") == "unsafe_end_turn":
        score += 35.0
    try:
        score += float(summary.get("hp_lost") or 0) * 3.0
    except (TypeError, ValueError):
        pass
    score += sum(float(v or 0) for v in flags.values() if isinstance(v, (int, float))) * 4.0
    return score


def _scan_logs(paths: list[Path]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            errors.append({"path": str(path), "kind": "read_error", "message": str(exc)})
            continue
        for idx, line in enumerate(lines):
            if any(marker in line for marker in _ERROR_MARKERS):
                start = max(0, idx - 2)
                end = min(len(lines), idx + 8)
                errors.append({
                    "path": str(path),
                    "line": idx + 1,
                    "marker": next((marker for marker in _ERROR_MARKERS if marker in line), "error"),
                    "context": lines[start:end],
                })
    return errors


def _default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else Path()
    episode_trace = Path(args.episode_trace).resolve() if args.episode_trace else dataset_dir / "episode_trace.jsonl"
    step_trace = Path(args.step_trace).resolve() if args.step_trace else dataset_dir / "step_trace.jsonl"
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        name = dataset_dir.name if str(dataset_dir) not in {"", "."} else (episode_trace.parent.name or "trace")
        out_dir = ARTIFACTS_ROOT / "reviews" / f"rollout_audit_{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return episode_trace, step_trace, out_dir


def main() -> int:
    args = parse_args()
    ensure_dirs()
    episode_trace, step_trace, out_dir = _default_paths(args)
    episodes = _read_jsonl(episode_trace)
    steps = _read_jsonl(step_trace)
    steps_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(steps, key=lambda item: (str(item.get("episode_id") or ""), _step_no(item))):
        steps_by_episode[str(row.get("episode_id") or "")].append(row)

    invalid_cases: list[dict[str, Any]] = []
    abnormal_cases: list[dict[str, Any]] = []
    suspicious_cases: list[dict[str, Any]] = []
    for ep in episodes:
        case = _episode_case(ep, steps_by_episode)
        outcome = str(case.get("outcome") or "")
        flags = case.get("quality_flags") if isinstance(case.get("quality_flags"), dict) else {}
        if case.get("invalid_output") or outcome.startswith("invalid_output:"):
            invalid_cases.append(case)
        elif outcome != "victory":
            abnormal_cases.append(case)
        elif flags:
            suspicious_cases.append(case)

    all_cases = [*invalid_cases, *abnormal_cases, *suspicious_cases]
    all_cases.sort(key=_severity, reverse=True)
    damage_turn_cases = _damage_turn_cases(steps, limit=max(0, args.top))
    log_errors = _scan_logs([Path(p).resolve() for p in args.log])

    outcome_counts = Counter(str(ep.get("outcome") or "") for ep in episodes)
    invalid_reasons = Counter(str(case.get("invalid_reason") or case.get("outcome") or "") for case in invalid_cases)
    cause_counts = Counter(str((case.get("cause") or {}).get("category") or "") for case in all_cases)
    quality_counts: Counter[str] = Counter()
    for ep in episodes:
        flags = ep.get("quality_flags")
        if isinstance(flags, dict):
            quality_counts.update({str(k): int(v) for k, v in flags.items()})

    _write_jsonl(out_dir / "invalid_cases.jsonl", invalid_cases[: args.top])
    _write_jsonl(out_dir / "abnormal_cases.jsonl", abnormal_cases[: args.top])
    _write_jsonl(out_dir / "suspicious_cases.jsonl", suspicious_cases[: args.top])
    _write_jsonl(out_dir / "damage_turn_cases.jsonl", damage_turn_cases)
    _write_jsonl(out_dir / "failure_rank.jsonl", all_cases[: args.top])
    _write_json(out_dir / "stderr_errors.json", {"errors": log_errors})

    summary = {
        "kind": "rollout_failure_audit",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "episode_trace": str(episode_trace),
        "step_trace": str(step_trace),
        "out_dir": str(out_dir),
        "episodes": len(episodes),
        "steps": len(steps),
        "outcome_counts": dict(outcome_counts.most_common()),
        "invalid_cases": len(invalid_cases),
        "abnormal_cases": len(abnormal_cases),
        "suspicious_cases": len(suspicious_cases),
        "damage_turn_cases": len(damage_turn_cases),
        "log_error_events": len(log_errors),
        "invalid_reason_counts": dict(invalid_reasons.most_common()),
        "cause_counts": dict(cause_counts.most_common()),
        "quality_flag_counts": dict(quality_counts.most_common()),
        "top_failures": all_cases[: min(10, args.top)],
        "outputs": {
            "invalid_cases": str(out_dir / "invalid_cases.jsonl"),
            "abnormal_cases": str(out_dir / "abnormal_cases.jsonl"),
            "suspicious_cases": str(out_dir / "suspicious_cases.jsonl"),
            "damage_turn_cases": str(out_dir / "damage_turn_cases.jsonl"),
            "failure_rank": str(out_dir / "failure_rank.jsonl"),
            "stderr_errors": str(out_dir / "stderr_errors.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
