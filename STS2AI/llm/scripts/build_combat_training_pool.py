"""Build a strict, stratified combat training pool.

This is intentionally conservative:
- source assistant content must already be valid JSON
- action_index must be legal in the prompt
- self-rollout rows with invalid episodes or known bad quality flags are dropped
- output assistant messages are re-serialized as canonical compact JSON
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.paths import DATASETS_ROOT, ensure_dirs  # noqa: E402
from llm.data_pipeline.action_quality import TRAINING_BLOCKLIST_FLAGS  # noqa: E402
from llm.prompts import load_system_prompt  # noqa: E402
from llm.scripts.analyze_action_ordering import _legal_actions  # noqa: E402
from llm.scripts.build_teacher_dataset import (  # noqa: E402
    _action_by_index,
    _canonical_reason_from_action,
    _json_action,
    _normalize_user_message_schema,
    _rows_from_kimi_labels,
)


_RUN_RE = re.compile(r"^run:.*?\bact=(?P<act>-?\d+).*?\bfloor=(?P<floor>-?\d+)", re.MULTILINE)
_DECK_RE = re.compile(r"^deck:\s*(?P<deck>.+)$", re.MULTILINE)
_HAND_RE = re.compile(r"^\s+\[(?P<idx>\d+)]\s+(?P<card>[A-Z0-9_+]+)\s+cost=", re.MULTILINE)
_ENEMY_RE = re.compile(r"^\s+enemy\d+:\s+(?P<enemy>\S+)\s+hp=.*?\s+intent=(?P<intent>[^\s]+)", re.MULTILINE)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="append", default=[], help="Dataset directory with train/eval JSONL.")
    parser.add_argument("--trace", action="append", default=[], help="Full-run step_trace.jsonl with real floor metadata.")
    parser.add_argument("--kimi-labels", action="append", default=[], help="Filtered Kimi kept_labels/valid_labels JSONL.")
    parser.add_argument("--out-dir", default=str(DATASETS_ROOT / f"combat_pool_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    parser.add_argument("--target-size", type=int, default=2000)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--min-advantage", type=float, default=0.0)
    parser.add_argument("--min-kimi-confidence", type=float, default=0.75)
    parser.add_argument("--include-losses", action="store_true")
    parser.add_argument("--include-forced", action="store_true", help="Keep one-legal-action combat states.")
    parser.add_argument(
        "--discover-rollouts",
        action="store_true",
        help="When no --dataset-dir is given, scan Artifacts/llm/datasets/*_rollout.",
    )
    parser.add_argument(
        "--discover-fullrun-traces",
        action="store_true",
        help="Scan Artifacts/llm/spectate_llm/*/step_trace.jsonl for floor-aware full-run samples.",
    )
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
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payload.setdefault("_source_file", str(path))
                payload.setdefault("_source_line", line_no)
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _messages(row: dict[str, Any]) -> tuple[str, str, str] | None:
    system = user = assistant = ""
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            system = str(message.get("content") or "")
        elif role == "user":
            user = str(message.get("content") or "")
        elif role == "assistant":
            assistant = str(message.get("content") or "")
    if not user or not assistant:
        return None
    return system, user, assistant


def _source_kind(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    source_kind = str(meta.get("source_kind") or "")
    if source_kind:
        return source_kind
    source = str(meta.get("source") or "")
    if source:
        return source
    return "self_rollout"


def _strict_assistant_payload(assistant: str) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(assistant.strip())
    except json.JSONDecodeError:
        return None, "assistant_not_strict_json"
    if not isinstance(payload, dict):
        return None, "assistant_not_object"
    action_index = payload.get("action_index")
    if isinstance(action_index, bool) or not isinstance(action_index, int):
        return None, "action_index_not_int"
    return payload, "ok"


def _has_combat_sections(user: str) -> bool:
    return bool(_HAND_RE.search(user)) and bool(_ENEMY_RE.search(user))


def _floor_bucket(floor: int | None) -> str:
    if floor is None or floor <= 0:
        return "floor_unknown"
    if floor <= 5:
        return "floor_01_05"
    if floor <= 10:
        return "floor_06_10"
    if floor <= 16:
        return "floor_11_16"
    return "floor_17_plus"


def _deck_cards(user: str) -> list[str]:
    match = _DECK_RE.search(user)
    if not match:
        return []
    cards: list[str] = []
    for chunk in match.group("deck").split(","):
        token = chunk.strip().split("x", 1)[0].strip()
        if token:
            cards.append(token)
    return cards


def _deck_bucket(user: str) -> str:
    cards = set(_deck_cards(user))
    if any("EXHAUST" in card or card in {"HARDENED_BLADE", "FORGOTTEN_RITUAL"} for card in cards):
        return "deck_exhaust"
    if any(card in cards for card in {"TWIN_STRIKE", "POMMEL_STRIKE", "ANGER"}):
        return "deck_attack_dense"
    if any(card in cards for card in {"SHRUG_IT_OFF", "ARMAMENTS", "TRUE_GRIT"}):
        return "deck_defense"
    if cards - {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"}:
        return "deck_mixed"
    return "deck_starter"


def _features(user: str, row: dict[str, Any]) -> dict[str, Any]:
    run_match = _RUN_RE.search(user)
    floor = None
    act = None
    if run_match:
        try:
            act = int(run_match.group("act"))
            floor = int(run_match.group("floor"))
        except (TypeError, ValueError):
            pass
    enemies = [match.group("enemy") for match in _ENEMY_RE.finditer(user)]
    intents = [match.group("intent") for match in _ENEMY_RE.finditer(user)]
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    quality_report = meta.get("action_quality_report") if isinstance(meta.get("action_quality_report"), dict) else {}
    opportunities = quality_report.get("opportunities") if isinstance(quality_report.get("opportunities"), dict) else {}
    mechanism = "normal"
    if opportunities.get("visible_lethal"):
        mechanism = "visible_lethal"
    elif opportunities.get("dangerous_turn"):
        mechanism = "dangerous_turn"
    elif any("Attack" in intent for intent in intents) and any("Buff" in intent or "Debuff" in intent for intent in intents):
        mechanism = "mixed_intents"
    return {
        "act": act,
        "floor": floor,
        "floor_bucket": _floor_bucket(floor),
        "deck_bucket": _deck_bucket(user),
        "enemy_bucket": "+".join(sorted(set(enemies))) if enemies else "enemy_unknown",
        "mechanism": mechanism,
        "encounter_id": str(meta.get("encounter_id") or "unknown"),
        "encounter_tag": str(meta.get("encounter_tag") or "unknown"),
    }


def _reject_self_rollout(row: dict[str, Any], *, include_losses: bool, min_advantage: float) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if meta.get("policy_invalid_output"):
        return "policy_invalid_output"
    if not include_losses and str(meta.get("outcome") or "").lower() not in {"victory", "win", "won"}:
        return "non_victory_rollout"
    try:
        advantage = float(meta.get("advantage") or 0.0)
    except (TypeError, ValueError):
        advantage = 0.0
    if advantage < min_advantage:
        return "low_advantage"
    flags = {str(flag) for flag in (meta.get("action_quality_flags") or [])}
    if flags & TRAINING_BLOCKLIST_FLAGS:
        return "bad_quality_flags"
    return ""


def _reject_trace_row(row: dict[str, Any]) -> str:
    if row.get("invalid_output"):
        return "policy_invalid_output"
    flags = {str(flag) for flag in (row.get("quality_flags") or [])}
    if flags & TRAINING_BLOCKLIST_FLAGS:
        return "bad_quality_flags"
    attempts = row.get("attempts") if isinstance(row.get("attempts"), list) else []
    if attempts:
        final_attempt = attempts[-1] if isinstance(attempts[-1], dict) else {}
        if final_attempt.get("strict_json_status") not in {"ok", "not_applicable"}:
            return "trace_final_not_strict_json"
    return ""


def _stable_key(user: str, action_index: int) -> str:
    digest = hashlib.sha1(f"{user}\n{action_index}".encode("utf-8", errors="replace")).hexdigest()
    return digest[:20]


def _canonical_row(row: dict[str, Any], *, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    message_parts = _messages(row)
    if message_parts is None:
        return None, "missing_messages"
    _system, user, assistant = message_parts
    if not _has_combat_sections(user):
        return None, "not_combat_prompt"
    payload, status = _strict_assistant_payload(assistant)
    if payload is None:
        return None, status
    action_index = int(payload["action_index"])
    actions = _legal_actions({"user_message": user})
    action = _action_by_index(actions, action_index)
    if action is None:
        return None, "action_index_not_legal"
    if not args.include_forced and len(actions) <= 1:
        return None, "forced_action"
    source_kind = _source_kind(row)
    if source_kind == "self_rollout":
        reason = _reject_self_rollout(row, include_losses=args.include_losses, min_advantage=args.min_advantage)
        if reason:
            return None, reason
    elif source_kind == "fullrun_trace":
        reason = _reject_trace_row(row)
        if reason:
            return None, reason

    user = _normalize_user_message_schema(user)
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        reason = _canonical_reason_from_action(user, action)
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = 0.9 if source_kind != "self_rollout" else 0.75
    features = _features(user, row)
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    canonical = {
        "messages": [
            {"role": "system", "content": load_system_prompt()},
            {"role": "user", "content": user},
            {"role": "assistant", "content": _json_action(action_index, reason, user_message=user, confidence=float(confidence))},
        ],
        "meta": {
            **meta,
            "source_kind": source_kind,
            "source_file": row.get("_source_file") or meta.get("source_file"),
            "source_line": row.get("_source_line") or meta.get("source_line"),
            "strict_pool": True,
            "canonicalized_response": True,
            "coverage": features,
            "pool_key": _stable_key(user, action_index),
        },
    }
    return canonical, "ok"


def _load_dataset_rows(dataset_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        for name in ("train.jsonl", "eval.jsonl"):
            path = dataset_dir / name
            for row in _read_jsonl(path):
                row.setdefault("_source_dataset_dir", str(dataset_dir))
                rows.append(row)
    return rows


def _row_from_trace(row: dict[str, Any], *, trace_path: Path, line_no: int, system_prompt: str) -> dict[str, Any] | None:
    user = str(row.get("user_message") or "")
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    action_index = decoded.get("action_index")
    if isinstance(action_index, bool) or not isinstance(action_index, int):
        return None
    reason = str(decoded.get("reason") or "")
    confidence = decoded.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = 0.75
    assistant = json.dumps(
        {
            "action_index": int(action_index),
            "confidence": float(confidence),
            "action_scores": decoded.get("action_scores") if isinstance(decoded.get("action_scores"), list) else [],
            "reason": reason,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "source_kind": "fullrun_trace",
            "source_trace": str(trace_path),
            "source_line": line_no,
            "route": row.get("route"),
            "step": row.get("step"),
            "quality_flags": row.get("quality_flags") if isinstance(row.get("quality_flags"), list) else [],
        },
        "invalid_output": row.get("invalid_output"),
        "attempts": row.get("attempts"),
        "quality_flags": row.get("quality_flags"),
    }


def _load_trace_rows(trace_paths: list[Path], *, system_prompt: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        if not trace_path.exists():
            continue
        with trace_path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                row = _row_from_trace(payload, trace_path=trace_path, line_no=line_no, system_prompt=system_prompt)
                if row is not None:
                    rows.append(row)
    return rows


def _discover_rollout_dirs() -> list[Path]:
    return sorted(
        [path for path in DATASETS_ROOT.glob("*_rollout") if (path / "train.jsonl").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _discover_fullrun_traces() -> list[Path]:
    spectate_root = DATASETS_ROOT.parent / "spectate_llm"
    if not spectate_root.exists():
        return []
    return sorted(
        [path for path in spectate_root.glob("*/step_trace.jsonl") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _select_stratified(rows: list[dict[str, Any]], *, target: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        key = str(meta.get("pool_key") or "")
        if not key:
            continue
        current = deduped.get(key)
        if current is None:
            deduped[key] = row
            continue
        cur_source = str((current.get("meta") or {}).get("source_kind") or "")
        new_source = str(meta.get("source_kind") or "")
        if _source_priority(new_source) > _source_priority(cur_source):
            deduped[key] = row

    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in deduped.values():
        cov = ((row.get("meta") or {}).get("coverage") or {})
        bucket = (
            str(cov.get("encounter_id") or "unknown"),
            str(cov.get("floor_bucket") or "floor_unknown"),
            str(cov.get("deck_bucket") or "deck_unknown"),
            str(cov.get("mechanism") or "normal"),
        )
        buckets[bucket].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
        bucket_rows.sort(key=lambda row: _source_priority(str((row.get("meta") or {}).get("source_kind") or "")), reverse=True)

    selected: list[dict[str, Any]] = []
    active = sorted(buckets)
    while active and len(selected) < target:
        next_active: list[tuple[str, str, str, str]] = []
        for bucket in active:
            bucket_rows = buckets[bucket]
            if bucket_rows and len(selected) < target:
                selected.append(bucket_rows.pop(0))
            if bucket_rows:
                next_active.append(bucket)
        active = next_active
    return selected


def _source_priority(source: str) -> int:
    if source == "kimi_teacher_label":
        return 4
    if source in {"turn_order_review", "review_reselect", "trace_rule"}:
        return 3
    return 1


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = {
        "source_kind": Counter(),
        "encounter_id": Counter(),
        "encounter_tag": Counter(),
        "floor_bucket": Counter(),
        "deck_bucket": Counter(),
        "enemy_bucket": Counter(),
        "mechanism": Counter(),
    }
    for row in rows:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        cov = meta.get("coverage") if isinstance(meta.get("coverage"), dict) else {}
        counters["source_kind"].update([str(meta.get("source_kind") or "")])
        for key in counters:
            if key == "source_kind":
                continue
            counters[key].update([str(cov.get(key) or "unknown")])
    return {key: dict(counter.most_common()) for key, counter in counters.items()}


def main() -> int:
    args = parse_args()
    ensure_dirs()
    dataset_dirs = [Path(path).resolve() for path in args.dataset_dir]
    if not dataset_dirs and args.discover_rollouts:
        dataset_dirs = _discover_rollout_dirs()
    trace_paths = [Path(path).resolve() for path in args.trace]
    if args.discover_fullrun_traces:
        trace_paths.extend(path for path in _discover_fullrun_traces() if path not in trace_paths)

    system_prompt = load_system_prompt()
    source_rows = _load_dataset_rows(dataset_dirs)
    source_rows.extend(_load_trace_rows(trace_paths, system_prompt=system_prompt))
    for labels_path in args.kimi_labels:
        source_rows.extend(_rows_from_kimi_labels(
            Path(labels_path).resolve(),
            system_prompt=system_prompt,
            min_confidence=args.min_kimi_confidence,
            keep_kimi_reasons=False,
        ))

    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in source_rows:
        canonical, status = _canonical_row(row, args=args)
        if canonical is None:
            rejected[status] += 1
            continue
        accepted.append(canonical)

    selected = _select_stratified(accepted, target=max(0, args.target_size), seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(selected)
    eval_n = max(1, int(len(selected) * args.eval_ratio)) if len(selected) >= 20 else 0
    eval_rows = selected[:eval_n]
    train_rows = selected[eval_n:]

    out_dir = Path(args.out_dir).resolve()
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    _write_jsonl(out_dir / "selected.jsonl", selected)
    summary = {
        "kind": "combat_training_pool",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "target_size": args.target_size,
        "input_rows": len(source_rows),
        "accepted_rows": len(accepted),
        "selected_rows": len(selected),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "rejected": dict(rejected.most_common()),
        "coverage": _coverage_summary(selected),
        "inputs": {
            "dataset_dirs": [str(path) for path in dataset_dirs],
            "traces": [str(path) for path in trace_paths],
            "kimi_labels": [str(Path(path).resolve()) for path in args.kimi_labels],
            "min_advantage": args.min_advantage,
            "include_losses": bool(args.include_losses),
        },
        "outputs": {
            "train": str(out_dir / "train.jsonl"),
            "eval": str(out_dir / "eval.jsonl"),
            "selected": str(out_dir / "selected.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
