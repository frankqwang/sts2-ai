"""Manage the long-lived dataset pool for the training flywheel.

The pool separates data by role:
- gold: teacher / verified repair labels, highest priority for training.
- silver: clean positive rollout samples, useful for scale.
- hardcase: failures and high-loss turns that need teacher/manual review.
- quarantine: invalid or unsafe samples that must not enter training directly.

It intentionally writes plain JSONL artifacts so every promotion decision can be
audited and replayed without a database service.
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

from llm.data_pipeline.action_quality import TRAINING_BLOCKLIST_FLAGS  # noqa: E402
from llm.paths import ARTIFACTS_ROOT, ensure_dirs  # noqa: E402


DEFAULT_POOL_ROOT = ARTIFACTS_ROOT / "dataset_pool"
TEACHER_SOURCES = {
    "kimi_teacher_label",
    "turn_order_review",
    "review_reselect",
    "trace_rule",
    "teacher_repair",
    "manual_teacher",
}
NON_BOSS_SILVER_HP_LOSS_MAX = 4.0
NON_BOSS_FAILURE_HIGH_PROGRESS_MIN = 0.65
_RUN_RE = re.compile(r"^run:.*?\bfloor=(?P<floor>-?\d+|\?)", re.MULTILINE)
_ENEMY_RE = re.compile(r"^\s+enemy\d+:\s+(?P<enemy>\S+)\s+hp=.*?\s+intent=(?P<intent>[^\s]+)", re.MULTILINE)
_DECK_RE = re.compile(r"^deck:\s*(?P<deck>.+)$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest-dataset", help="Promote train/eval rows from a dataset directory into the long-lived pool.")
    ingest.add_argument("--dataset-dir", required=True)
    ingest.add_argument("--source-name", default="")

    audit = sub.add_parser("ingest-audit", help="Register rollout audit failures as hardcases/quarantine cases.")
    audit.add_argument("--audit-dir", required=True)
    audit.add_argument("--dataset-dir", default="")
    audit.add_argument("--source-name", default="")

    materialize = sub.add_parser("materialize", help="Build a train/eval dataset from active gold/silver pool rows.")
    materialize.add_argument("--out-dir", required=True)
    materialize.add_argument("--target-size", type=int, default=5000)
    materialize.add_argument("--eval-ratio", type=float, default=0.05)
    materialize.add_argument("--seed", type=int, default=20260428)
    materialize.add_argument("--gold-min-ratio", type=float, default=0.15)
    materialize.add_argument("--include-quarantine", action="store_true")

    report = sub.add_parser("report", help="Summarize current pool contents.")
    report.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
                payload.setdefault("_source_line", line_no)
                rows.append(payload)
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pool_paths(root: Path) -> dict[str, Path]:
    return {
        "registry": root / "registry.jsonl",
        "gold": root / "samples" / "gold.jsonl",
        "silver": root / "samples" / "silver.jsonl",
        "quarantine": root / "samples" / "quarantine.jsonl",
        "hardcase": root / "hardcases" / "hardcases.jsonl",
        "manifests": root / "manifests",
    }


def _existing_ids(path: Path, key: str) -> set[str]:
    ids: set[str] = set()
    for row in _read_jsonl(path):
        raw = row.get(key)
        if raw:
            ids.add(str(raw))
    return ids


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


def _assistant_payload(assistant: str) -> tuple[dict[str, Any] | None, str]:
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


def _sample_id(user: str, action_index: int) -> str:
    return hashlib.sha1(f"{user}\n{action_index}".encode("utf-8", errors="replace")).hexdigest()


def _hardcase_id(payload: dict[str, Any], source: str) -> str:
    basis = json.dumps({
        "source": source,
        "episode_id": payload.get("episode_id"),
        "step": payload.get("step"),
        "round": payload.get("round"),
        "case_id": payload.get("case_id"),
        "cause": (payload.get("cause") or {}).get("category") if isinstance(payload.get("cause"), dict) else "",
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(basis.encode("utf-8", errors="replace")).hexdigest()


def _source_kind(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(meta.get("source_kind") or meta.get("source") or "self_rollout")


def _quality_flags(row: dict[str, Any]) -> set[str]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    flags: set[str] = set()
    for raw in meta.get("action_quality_flags") or meta.get("quality_flags") or row.get("quality_flags") or []:
        flags.add(str(raw))
    return flags


def _meta_number(meta: dict[str, Any], key: str) -> float | None:
    value = meta.get(key)
    if value is None:
        summary = meta.get("episode_quality_summary") if isinstance(meta.get("episode_quality_summary"), dict) else {}
        value = summary.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_boss_like(meta: dict[str, Any]) -> bool:
    text = " ".join(
        str(meta.get(key) or "")
        for key in ("encounter_id", "encounter_key", "encounter_tag", "encounter_type")
    ).lower()
    return "boss" in text


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


def _floor_from_user(user: str) -> int | None:
    match = _RUN_RE.search(user)
    if not match or match.group("floor") == "?":
        return None
    try:
        return int(match.group("floor"))
    except ValueError:
        return None


def _deck_bucket(user: str) -> str:
    match = _DECK_RE.search(user)
    if not match:
        return "deck_unknown"
    cards = {chunk.strip().split("x", 1)[0].strip() for chunk in match.group("deck").split(",")}
    if cards <= {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"}:
        return "deck_starter"
    if any(card in cards for card in {"SHRUG_IT_OFF", "ARMAMENTS", "TRUE_GRIT"}):
        return "deck_defense"
    if any(card in cards for card in {"POMMEL_STRIKE", "ANGER", "TWIN_STRIKE"}):
        return "deck_attack_dense"
    return "deck_mixed"


def _coverage(row: dict[str, Any], user: str) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    existing = meta.get("coverage") if isinstance(meta.get("coverage"), dict) else {}
    floor = existing.get("floor")
    if not isinstance(floor, int):
        raw_floor = meta.get("floor")
        floor = raw_floor if isinstance(raw_floor, int) else _floor_from_user(user)
    enemies = [m.group("enemy") for m in _ENEMY_RE.finditer(user)]
    intents = [m.group("intent") for m in _ENEMY_RE.finditer(user)]
    mechanism = str(existing.get("mechanism") or "normal")
    if mechanism == "normal" and any("Attack" in intent for intent in intents):
        mechanism = "incoming_damage"
    return {
        "floor": floor,
        "floor_bucket": str(existing.get("floor_bucket") or _floor_bucket(floor)),
        "deck_bucket": str(existing.get("deck_bucket") or _deck_bucket(user)),
        "enemy_bucket": str(existing.get("enemy_bucket") or ("+".join(sorted(set(enemies))) if enemies else "enemy_unknown")),
        "mechanism": mechanism,
        "encounter_id": str(existing.get("encounter_id") or meta.get("encounter_id") or "unknown"),
        "encounter_tag": str(existing.get("encounter_tag") or meta.get("encounter_tag") or "unknown"),
    }


def _classify_sample(row: dict[str, Any], payload: dict[str, Any]) -> tuple[str, str]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    source = _source_kind(row)
    flags = _quality_flags(row)
    if flags & TRAINING_BLOCKLIST_FLAGS:
        return "quarantine", "bad_quality_flags"
    if meta.get("policy_invalid_output"):
        return "quarantine", "policy_invalid_output"
    if source in TEACHER_SOURCES or "teacher" in source or meta.get("teacher_action"):
        return "gold", "teacher_verified"
    if _is_boss_like(meta):
        return "quarantine", "boss_rollout_needs_separate_gate"
    outcome = str(meta.get("outcome") or "").lower()
    hp_lost = _meta_number(meta, "hp_lost")
    enemy_progress = _meta_number(meta, "enemy_damage_progress")
    if outcome and outcome not in {"victory", "win", "won"}:
        if enemy_progress is None:
            return "quarantine", "missing_enemy_damage_progress"
        if enemy_progress >= NON_BOSS_FAILURE_HIGH_PROGRESS_MIN:
            return "quarantine", "incomplete_high_enemy_progress"
        return "quarantine", "incomplete_low_enemy_progress"
    if hp_lost is None:
        return "quarantine", "missing_hp_lost_metric"
    if hp_lost > NON_BOSS_SILVER_HP_LOSS_MAX:
        return "quarantine", "high_hp_loss_rollout"
    try:
        advantage = float(meta.get("advantage") or 0.0)
    except (TypeError, ValueError):
        advantage = 0.0
    if source == "self_rollout" and advantage <= 0:
        return "quarantine", "non_positive_advantage"
    return "silver", "clean_low_hp_loss_rollout"


def _canonical_sample(
    row: dict[str, Any],
    *,
    dataset_dir: Path,
    split: str,
    source_name: str,
) -> tuple[dict[str, Any] | None, str, str]:
    unpacked = _messages(row)
    if unpacked is None:
        return None, "quarantine", "missing_messages"
    system, user, assistant = unpacked
    payload, status = _assistant_payload(assistant)
    if payload is None:
        return None, "quarantine", status
    action_index = int(payload["action_index"])
    sample_id = _sample_id(user, action_index)
    tier, reason = _classify_sample(row, payload)
    canonical_assistant = json.dumps({
        "action_index": action_index,
        "confidence": float(payload.get("confidence") or 0.75),
        "reason": str(payload.get("reason") or "selected legal action")[:240],
    }, ensure_ascii=False, separators=(",", ":"))
    meta = dict(row.get("meta") if isinstance(row.get("meta"), dict) else {})
    meta["pool"] = {
        "sample_id": sample_id,
        "tier": tier,
        "promotion_reason": reason,
        "source_dataset": str(dataset_dir),
        "source_name": source_name or dataset_dir.name,
        "source_split": split,
        "source_line": row.get("_source_line"),
        "ingested_at": _now(),
    }
    meta["coverage"] = _coverage(row, user)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": canonical_assistant},
        ],
        "meta": meta,
    }, tier, reason


def _load_dataset_rows(dataset_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for split in ("train", "eval"):
        for row in _read_jsonl(dataset_dir / f"{split}.jsonl"):
            rows.append((split, row))
    return rows


def ingest_dataset(args: argparse.Namespace) -> int:
    root = Path(args.pool_root).resolve()
    paths = _pool_paths(root)
    dataset_dir = Path(args.dataset_dir).resolve()
    existing = _existing_ids(paths["registry"], "sample_id")
    registry_rows: list[dict[str, Any]] = []
    sample_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()

    for split, row in _load_dataset_rows(dataset_dir):
        sample, tier, reason = _canonical_sample(
            row,
            dataset_dir=dataset_dir,
            split=split,
            source_name=str(args.source_name or dataset_dir.name),
        )
        if sample is None:
            rejected[reason] += 1
            continue
        sample_id = str(((sample.get("meta") or {}).get("pool") or {}).get("sample_id") or "")
        if sample_id in existing:
            rejected["duplicate"] += 1
            continue
        existing.add(sample_id)
        tier_counts[tier] += 1
        sample_rows[tier].append(sample)
        registry_rows.append({
            "kind": "sample",
            "sample_id": sample_id,
            "tier": tier,
            "status": "active" if tier in {"gold", "silver"} else "quarantine",
            "promotion_reason": reason,
            "source_dataset": str(dataset_dir),
            "source_name": str(args.source_name or dataset_dir.name),
            "coverage": (sample.get("meta") or {}).get("coverage") or {},
            "ingested_at": ((sample.get("meta") or {}).get("pool") or {}).get("ingested_at"),
        })

    for tier, rows in sample_rows.items():
        _append_jsonl(paths[tier], rows)
    _append_jsonl(paths["registry"], registry_rows)

    summary = {
        "kind": "dataset_pool_ingest_dataset",
        "built_at": _now(),
        "pool_root": str(root),
        "dataset_dir": str(dataset_dir),
        "input_rows": len(_load_dataset_rows(dataset_dir)),
        "ingested": int(sum(tier_counts.values())),
        "tier_counts": dict(tier_counts.most_common()),
        "rejected": dict(rejected.most_common()),
        "outputs": {key: str(value) for key, value in paths.items() if key != "manifests"},
    }
    manifest = paths["manifests"] / f"ingest_dataset_{dataset_dir.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    _write_json(manifest, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _severity(case: dict[str, Any]) -> float:
    cause = case.get("cause") if isinstance(case.get("cause"), dict) else {}
    category = str(cause.get("category") or "")
    outcome = str(case.get("outcome") or "")
    score = 0.0
    if outcome.startswith("invalid_output:"):
        score += 100
    if category in {"unsafe_end_turn", "unsafe_self_damage"}:
        score += 80
    if category == "combat_loss" or outcome == "defeat":
        score += 70
    if category == "left_combat":
        score += 50
    for key in ("observed_hp_loss",):
        try:
            score += float(case.get(key) or 0) * 3
        except (TypeError, ValueError):
            pass
    summary = case.get("quality_summary") if isinstance(case.get("quality_summary"), dict) else {}
    try:
        score += float(summary.get("hp_lost") or 0) * 2
    except (TypeError, ValueError):
        pass
    return score


def ingest_audit(args: argparse.Namespace) -> int:
    root = Path(args.pool_root).resolve()
    paths = _pool_paths(root)
    audit_dir = Path(args.audit_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else None
    source = str(args.source_name or audit_dir.name)
    existing = _existing_ids(paths["registry"], "hardcase_id")
    hardcases: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for name in ("invalid_cases", "abnormal_cases", "suspicious_cases", "damage_turn_cases", "failure_rank"):
        for raw in _read_jsonl(audit_dir / f"{name}.jsonl"):
            hardcase_id = _hardcase_id(raw, f"{source}:{name}")
            if hardcase_id in existing:
                counts["duplicate"] += 1
                continue
            existing.add(hardcase_id)
            cause = raw.get("cause") if isinstance(raw.get("cause"), dict) else {}
            category = str(cause.get("category") or name)
            status = "quarantine" if category in {"protocol_format", "unsafe_end_turn", "unsafe_self_damage"} else "needs_teacher"
            payload = {
                "hardcase_id": hardcase_id,
                "source_audit": str(audit_dir),
                "source_dataset": str(dataset_dir) if dataset_dir else "",
                "source_name": source,
                "source_file": name,
                "status": status,
                "category": category,
                "severity": round(_severity(raw), 4),
                "ingested_at": _now(),
                "case": raw,
            }
            hardcases.append(payload)
            counts[status] += 1
            counts[f"category:{category}"] += 1
            registry_rows.append({
                "kind": "hardcase",
                "hardcase_id": hardcase_id,
                "tier": "hardcase",
                "status": status,
                "category": category,
                "severity": payload["severity"],
                "source_audit": str(audit_dir),
                "source_dataset": str(dataset_dir) if dataset_dir else "",
                "ingested_at": payload["ingested_at"],
            })

    hardcases.sort(key=lambda row: float(row.get("severity") or 0), reverse=True)
    _append_jsonl(paths["hardcase"], hardcases)
    _append_jsonl(paths["registry"], registry_rows)
    summary = {
        "kind": "dataset_pool_ingest_audit",
        "built_at": _now(),
        "pool_root": str(root),
        "audit_dir": str(audit_dir),
        "dataset_dir": str(dataset_dir) if dataset_dir else "",
        "ingested_hardcases": len(hardcases),
        "counts": dict(counts.most_common()),
        "outputs": {
            "hardcases": str(paths["hardcase"]),
            "registry": str(paths["registry"]),
        },
    }
    manifest = paths["manifests"] / f"ingest_audit_{audit_dir.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    _write_json(manifest, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _bucket(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    cov = meta.get("coverage") if isinstance(meta.get("coverage"), dict) else {}
    pool = meta.get("pool") if isinstance(meta.get("pool"), dict) else {}
    return (
        str(pool.get("tier") or "unknown"),
        str(cov.get("encounter_id") or "unknown"),
        str(cov.get("floor_bucket") or "floor_unknown"),
        str(cov.get("deck_bucket") or "deck_unknown"),
        str(cov.get("mechanism") or "normal"),
    )


def _select(rows: list[dict[str, Any]], *, target: int, seed: int, gold_min_ratio: float) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    gold = [row for row in rows if (((row.get("meta") or {}).get("pool") or {}).get("tier") == "gold")]
    silver = [row for row in rows if (((row.get("meta") or {}).get("pool") or {}).get("tier") == "silver")]

    def stratified(pool_rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in pool_rows:
            buckets[_bucket(row)].append(row)
        for values in buckets.values():
            rng.shuffle(values)
        selected: list[dict[str, Any]] = []
        active = sorted(buckets)
        while active and len(selected) < n:
            next_active: list[tuple[str, str, str, str, str]] = []
            for bucket in active:
                values = buckets[bucket]
                if values and len(selected) < n:
                    selected.append(values.pop())
                if values:
                    next_active.append(bucket)
            active = next_active
        return selected

    gold_target = min(len(gold), int(round(target * max(0.0, min(1.0, gold_min_ratio)))))
    selected = stratified(gold, gold_target)
    selected_ids = {(((row.get("meta") or {}).get("pool") or {}).get("sample_id")) for row in selected}
    remaining = [row for row in [*gold, *silver] if (((row.get("meta") or {}).get("pool") or {}).get("sample_id")) not in selected_ids]
    selected.extend(stratified(remaining, max(0, target - len(selected))))
    rng.shuffle(selected)
    return selected[:target]


def materialize(args: argparse.Namespace) -> int:
    root = Path(args.pool_root).resolve()
    paths = _pool_paths(root)
    rows = _read_jsonl(paths["gold"]) + _read_jsonl(paths["silver"])
    if args.include_quarantine:
        rows.extend(_read_jsonl(paths["quarantine"]))
    selected = _select(
        rows,
        target=max(0, args.target_size),
        seed=args.seed,
        gold_min_ratio=args.gold_min_ratio,
    )
    eval_n = max(1, int(len(selected) * args.eval_ratio)) if len(selected) >= 20 else 0
    eval_rows = selected[:eval_n]
    train_rows = selected[eval_n:]
    out_dir = Path(args.out_dir).resolve()
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    _write_jsonl(out_dir / "selected.jsonl", selected)
    tier_counts = Counter(str((((row.get("meta") or {}).get("pool") or {}).get("tier") or "")) for row in selected)
    coverage = Counter("|".join(_bucket(row)) for row in selected)
    summary = {
        "kind": "dataset_pool_materialized_dataset",
        "built_at": _now(),
        "pool_root": str(root),
        "out_dir": str(out_dir),
        "target_size": args.target_size,
        "selected_rows": len(selected),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "tier_counts": dict(tier_counts.most_common()),
        "coverage_top": dict(coverage.most_common(50)),
        "args": vars(args),
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


def report(args: argparse.Namespace) -> int:
    root = Path(args.pool_root).resolve()
    paths = _pool_paths(root)
    registry = _read_jsonl(paths["registry"])
    hardcases = _read_jsonl(paths["hardcase"])
    samples = _read_jsonl(paths["gold"]) + _read_jsonl(paths["silver"]) + _read_jsonl(paths["quarantine"])
    kinds = Counter(str(row.get("kind") or "") for row in registry)
    tiers = Counter(str(row.get("tier") or "") for row in registry)
    statuses = Counter(str(row.get("status") or "") for row in registry)
    categories = Counter(str(row.get("category") or "") for row in hardcases)
    coverage = Counter("|".join(_bucket(row)) for row in samples if (((row.get("meta") or {}).get("pool") or {}).get("tier") in {"gold", "silver"}))
    payload = {
        "kind": "dataset_pool_report",
        "built_at": _now(),
        "pool_root": str(root),
        "registry_rows": len(registry),
        "sample_rows": len(samples),
        "hardcase_rows": len(hardcases),
        "kinds": dict(kinds.most_common()),
        "tiers": dict(tiers.most_common()),
        "statuses": dict(statuses.most_common()),
        "hardcase_categories": dict(categories.most_common(args.limit)),
        "coverage_top": dict(coverage.most_common(args.limit)),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    _write_json(root / "report.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    ensure_dirs()
    Path(args.pool_root).mkdir(parents=True, exist_ok=True)
    if args.cmd == "ingest-dataset":
        return ingest_dataset(args)
    if args.cmd == "ingest-audit":
        return ingest_audit(args)
    if args.cmd == "materialize":
        return materialize(args)
    if args.cmd == "report":
        return report(args)
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
