"""Unified metrics helpers for LLM data, training, and runtime runs."""

from __future__ import annotations

import ast
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


_ACTION_INDEX_RE = re.compile(r'"?action_index"?\s*:\s*(\d+)')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _number_stats(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "avg": round(float(sum(ordered)) / len(ordered), 4),
        "max": ordered[-1],
    }


def _counter_payload(counter: Counter[Any], *, top: int | None = None) -> dict[str, int]:
    items = counter.most_common(top) if top else counter.most_common()
    return {str(key): int(value) for key, value in items}


def _message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def _indented_section(text: str, header: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_section = False
    marker = f"{header}:"
    for line in lines:
        if not in_section:
            if line.strip() == marker:
                in_section = True
            continue
        if line and not line.startswith((" ", "\t")):
            break
        out.append(line)
    return "\n".join(out)


def _assistant_action_index(text: str) -> int | None:
    match = _ACTION_INDEX_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _legal_action_line(legal_section: str, action_index: int) -> str:
    pattern = re.compile(rf"^\s*\[{re.escape(str(action_index))}\]\s+([^\n]+)", re.MULTILINE)
    match = pattern.search(legal_section)
    return match.group(1).strip() if match else ""


def _classify_legal_action(line: str) -> str:
    lower = line.lower()
    if lower.startswith("end_turn") or " end_turn" in lower:
        return "end_turn"
    if "target=self" in lower:
        return "self_card"
    if "target=enemy" in lower:
        return "target_card"
    if "play_card" in lower or "hand[" in lower:
        return "card"
    return "other"


def summarize_sft_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_chars: list[int] = []
    assistant_chars: list[int] = []
    legal_counts: list[int] = []
    action_indices: Counter[int] = Counter()
    chosen_action_types: Counter[str] = Counter()
    encounters: Counter[str] = Counter()
    state_types: Counter[str] = Counter()
    action_modes: Counter[str] = Counter()
    encounter_keys: Counter[str] = Counter()
    action_quality: Counter[str] = Counter()
    mechanism_scores: list[float] = []
    parse_failures = 0

    for row in rows:
        user_text = _message_content(row, "user")
        assistant_text = _message_content(row, "assistant")
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        encounter_id = str(meta.get("encounter_id") or "")
        encounter_key = str(meta.get("encounter_key") or "")
        state_type = str(meta.get("state_type") or "")
        action_mode = str(meta.get("action_mode") or "")
        if encounter_id:
            encounters[encounter_id] += 1
        if encounter_key:
            encounter_keys[encounter_key] += 1
        if state_type:
            state_types[state_type] += 1
        if action_mode:
            action_modes[action_mode] += 1
        for flag in meta.get("action_quality_flags") or []:
            action_quality[str(flag)] += 1
        report = meta.get("action_quality_report") if isinstance(meta.get("action_quality_report"), dict) else {}
        score = report.get("mechanism_score")
        if isinstance(score, (int, float)):
            mechanism_scores.append(float(score))

        legal_section = _indented_section(user_text, "legal_actions")
        legal_counts.append(len(re.findall(r"^\s*\[\d+\]", legal_section, flags=re.MULTILINE)))
        prompt_chars.append(len(user_text))
        assistant_chars.append(len(assistant_text))

        action_index = _assistant_action_index(assistant_text)
        if action_index is None:
            parse_failures += 1
            continue
        action_indices[action_index] += 1
        chosen_action_types[_classify_legal_action(_legal_action_line(legal_section, action_index))] += 1

    return {
        "rows": len(rows),
        "assistant_parse_failures": parse_failures,
        "assistant_action_index_counts": _counter_payload(action_indices, top=20),
        "chosen_action_type_counts": _counter_payload(chosen_action_types),
        "legal_action_count": _number_stats(legal_counts),
        "user_prompt_chars": _number_stats(prompt_chars),
        "assistant_chars": _number_stats(assistant_chars),
        "encounter_counts": _counter_payload(encounters),
        "encounter_key_counts": _counter_payload(encounter_keys, top=30),
        "state_type_counts": _counter_payload(state_types),
        "action_mode_counts": _counter_payload(action_modes),
        "action_quality_counts": _counter_payload(action_quality),
        "mechanism_score": _number_stats(mechanism_scores),
    }


def summarize_dataset_dir(dataset_dir: Path) -> dict[str, Any]:
    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    eval_rows = read_jsonl(dataset_dir / "eval.jsonl")
    meta_path = dataset_dir / "meta.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    payload = {
        "dataset_dir": str(dataset_dir),
        "train": summarize_sft_rows(train_rows),
        "eval": summarize_sft_rows(eval_rows),
    }
    if meta:
        payload["rollout"] = summarize_rollout_meta(meta)
    return payload


def summarize_rollout_meta(meta: dict[str, Any]) -> dict[str, Any]:
    episodes = [ep for ep in (meta.get("episodes") or []) if isinstance(ep, dict)]
    outcomes = Counter(str(ep.get("outcome") or "") for ep in episodes)
    steps = [int(ep.get("steps") or 0) for ep in episodes]
    durations = [float(ep.get("duration_s") or 0.0) for ep in episodes]
    kept = [int(ep.get("kept_samples") or 0) for ep in episodes]
    discarded = [int(ep.get("discarded_samples") or 0) for ep in episodes]
    invalid_outputs = sum(1 for ep in episodes if bool(ep.get("invalid_output")))
    action_quality = Counter()
    mechanism_scores: list[float] = []
    hp_lost: list[float] = []
    sequence_scores: list[float] = []
    defense_scores: list[float] = []
    turns: list[float] = []
    for ep in episodes:
        if isinstance(ep.get("quality_flags"), dict):
            action_quality.update(ep["quality_flags"])
        summary = ep.get("quality_summary") if isinstance(ep.get("quality_summary"), dict) else {}
        if isinstance(summary.get("mechanism_score"), (int, float)):
            mechanism_scores.append(float(summary["mechanism_score"]))
        if isinstance(summary.get("hp_lost"), (int, float)):
            hp_lost.append(float(summary["hp_lost"]))
        if isinstance(summary.get("sequence_score"), (int, float)):
            sequence_scores.append(float(summary["sequence_score"]))
        if isinstance(summary.get("defense_score"), (int, float)):
            defense_scores.append(float(summary["defense_score"]))
        if isinstance(summary.get("turns"), (int, float)):
            turns.append(float(summary["turns"]))
    victories = outcomes.get("victory", 0)
    total = len(episodes)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        key = str(ep.get("encounter_key") or ep.get("encounter_id") or "")
        if key:
            grouped.setdefault(key, []).append(ep)

    by_encounter: dict[str, Any] = {}
    for key, values in grouped.items():
        group_outcomes = Counter(str(ep.get("outcome") or "") for ep in values)
        group_rewards = [
            float(((ep.get("reward") or {}).get("total")) or 0.0)
            for ep in values
        ]
        group_invalid = sum(1 for ep in values if bool(ep.get("invalid_output")))
        group_total = len(values)
        by_encounter[key] = {
            "encounter_label": values[0].get("encounter_label") or key,
            "episodes": group_total,
            "victories": group_outcomes.get("victory", 0),
            "win_rate": round(group_outcomes.get("victory", 0) / group_total, 4) if group_total else None,
            "invalid_output_episode_rate": round(group_invalid / group_total, 4) if group_total else None,
            "reward": _number_stats(group_rewards),
            "outcome_counts": _counter_payload(group_outcomes),
        }
    return {
        "pool_name": meta.get("pool_name"),
        "action_mode": meta.get("action_mode"),
        "episodes": total,
        "victories": victories,
        "win_rate": round(victories / total, 4) if total else None,
        "total_samples": meta.get("total_samples"),
        "train_size": meta.get("train_size"),
        "eval_size": meta.get("eval_size"),
        "discarded_samples": meta.get("discarded_samples"),
        "invalid_output_episodes": invalid_outputs,
        "invalid_output_episode_rate": round(invalid_outputs / total, 4) if total else None,
        "policy_stats": meta.get("policy_stats") if isinstance(meta.get("policy_stats"), dict) else {},
        "action_quality": (
            meta.get("action_quality")
            if isinstance(meta.get("action_quality"), dict)
            else _counter_payload(action_quality)
        ),
        "mechanism_score": _number_stats(mechanism_scores),
        "sequence_score": _number_stats(sequence_scores),
        "defense_score": _number_stats(defense_scores),
        "hp_lost": _number_stats(hp_lost),
        "turns": _number_stats(turns),
        "outcome_counts": _counter_payload(outcomes),
        "steps": _number_stats(steps),
        "duration_s": _number_stats(durations),
        "kept_samples_per_episode": _number_stats(kept),
        "discarded_samples_per_episode": _number_stats(discarded),
        "by_encounter": by_encounter,
    }


def summarize_trainer_history(
    log_history: list[dict[str, Any]],
    result_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_metrics = result_metrics or {}
    eval_history: list[dict[str, Any]] = []
    train_losses: list[float] = []
    grad_norms: list[float] = []
    for entry in log_history:
        if "eval_loss" in entry:
            eval_loss = float(entry["eval_loss"])
            eval_history.append({
                "epoch": entry.get("epoch"),
                "step": entry.get("step"),
                "eval_loss": eval_loss,
                "perplexity": round(math.exp(eval_loss), 6),
                "eval_runtime": entry.get("eval_runtime"),
                "eval_samples_per_second": entry.get("eval_samples_per_second"),
                "eval_steps_per_second": entry.get("eval_steps_per_second"),
            })
        if "loss" in entry:
            train_losses.append(float(entry["loss"]))
        if "grad_norm" in entry:
            grad_norms.append(float(entry["grad_norm"]))

    final_eval = eval_history[-1] if eval_history else None
    return {
        "final": {
            **result_metrics,
            "final_eval_loss": final_eval.get("eval_loss") if final_eval else None,
            "final_eval_perplexity": final_eval.get("perplexity") if final_eval else None,
        },
        "eval_history": eval_history,
        "train_loss": _number_stats(train_losses),
        "grad_norm": _number_stats(grad_norms),
    }


def latest_trainer_state(run_root: Path) -> dict[str, Any]:
    states = list((run_root / "trainer").glob("checkpoint-*/trainer_state.json"))
    if not states:
        return {}

    def _key(path: Path) -> tuple[int, float]:
        try:
            step = int(path.parent.name.split("-", 1)[1])
        except Exception:
            step = -1
        return step, path.stat().st_mtime

    latest = max(states, key=_key)
    return _read_json(latest)


def summarize_sft_run(
    run_root: Path,
    *,
    dataset_dir: Path | None = None,
    result_metrics: dict[str, Any] | None = None,
    log_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_meta_path = run_root / "run_meta.json"
    run_meta = _read_json(run_meta_path) if run_meta_path.exists() else {}
    if dataset_dir is None:
        raw_dataset_dir = run_meta.get("dataset_dir")
        dataset_dir = Path(raw_dataset_dir) if raw_dataset_dir else None
    if log_history is None:
        trainer_state = latest_trainer_state(run_root)
        log_history = trainer_state.get("log_history") or []
    if result_metrics is None:
        result_metrics = run_meta.get("metrics") or {}

    payload: dict[str, Any] = {
        "kind": "sft",
        "run_root": str(run_root),
        "run_name": run_meta.get("run_name") or run_root.name,
        "base_model": run_meta.get("base_model"),
        "adapter_dir": str(run_root / "adapter"),
        "training": summarize_trainer_history(log_history or [], result_metrics),
    }
    if dataset_dir is not None:
        payload["dataset"] = summarize_dataset_dir(dataset_dir)
    return payload


def summarize_trace(trace_path: Path) -> dict[str, Any]:
    rows = read_jsonl(trace_path)
    routes: Counter[str] = Counter()
    action_modes: Counter[str] = Counter()
    final_fallback_reasons: Counter[str] = Counter()
    attempt_fallback_reasons: Counter[str] = Counter()
    strict_json_statuses: Counter[str] = Counter()
    chosen_action_types: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    quality_opportunities: Counter[str] = Counter()
    quality_misses: Counter[str] = Counter()
    gen_ms: list[float] = []
    enabled_counts: list[int] = []
    retries = 0
    invalid = 0
    generated_attempts = 0
    generated_decisions = 0
    valid_attempts = 0
    first_attempt_invalid = 0
    retry_recovered = 0
    final_stats: dict[str, Any] = {}
    for row in rows:
        routes[str(row.get("route") or "")] += 1
        action_modes[str(row.get("action_mode") or "")] += 1
        enabled_counts.append(int(row.get("enabled_count") or 0))
        if row.get("invalid_output"):
            invalid += 1
        quality_flags.update(str(flag) for flag in (row.get("quality_flags") or []))
        report = row.get("quality_report") if isinstance(row.get("quality_report"), dict) else {}
        quality_opportunities.update(report.get("opportunities") or {})
        quality_misses.update(report.get("misses") or {})
        attempts = [a for a in (row.get("attempts") or []) if isinstance(a, dict)]
        retries += max(0, len(attempts) - 1)
        if attempts:
            generated_decisions += 1
            first_decoded = attempts[0].get("decoded") if isinstance(attempts[0].get("decoded"), dict) else {}
            first_invalid = bool(first_decoded.get("used_fallback"))
            if first_invalid:
                first_attempt_invalid += 1
            if first_invalid and not row.get("invalid_output"):
                retry_recovered += 1
        for attempt in attempts:
            generated_attempts += 1
            status = str(attempt.get("strict_json_status") or "")
            if status:
                strict_json_statuses[status] += 1
            attempt_decoded = attempt.get("decoded") if isinstance(attempt.get("decoded"), dict) else {}
            if not attempt_decoded.get("used_fallback"):
                valid_attempts += 1
            attempt_fallback_reason = str(attempt_decoded.get("fallback_reason") or "")
            if attempt_fallback_reason:
                attempt_fallback_reasons[attempt_fallback_reason] += 1
        decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
        fallback_reason = str(decoded.get("fallback_reason") or "")
        if fallback_reason:
            final_fallback_reasons[fallback_reason] += 1
        value = row.get("gen_ms")
        if isinstance(value, (int, float)) and value > 0:
            gen_ms.append(float(value))
        chosen = row.get("chosen_action") if isinstance(row.get("chosen_action"), dict) else {}
        action = str(chosen.get("action") or chosen.get("type") or "")
        if action:
            chosen_action_types[action] += 1
        if isinstance(row.get("stats"), dict):
            final_stats = row["stats"]

    strict_json_measured = sum(strict_json_statuses.values())
    return {
        "trace_path": str(trace_path),
        "steps": len(rows),
        "invalid_outputs": invalid,
        "invalid_output_rate": round(invalid / len(rows), 4) if rows else None,
        "retry_attempts": retries,
        "first_attempt_invalid": first_attempt_invalid,
        "first_attempt_invalid_rate": round(first_attempt_invalid / generated_decisions, 4) if generated_decisions else None,
        "retry_recovered": retry_recovered,
        "generated_decisions": generated_decisions,
        "generated_attempts": generated_attempts,
        "decoder_valid_attempts": valid_attempts,
        "decoder_valid_attempt_rate": round(valid_attempts / generated_attempts, 4) if generated_attempts else None,
        "strict_json_status_counts": _counter_payload(strict_json_statuses),
        "strict_json_ok_attempts": int(strict_json_statuses.get("ok", 0)),
        "strict_json_ok_attempt_rate": round(strict_json_statuses.get("ok", 0) / strict_json_measured, 4) if strict_json_measured else None,
        "routes": _counter_payload(routes),
        "action_modes": _counter_payload(action_modes),
        "final_fallback_reasons": _counter_payload(final_fallback_reasons),
        "attempt_fallback_reasons": _counter_payload(attempt_fallback_reasons),
        "fallback_reasons": _counter_payload(final_fallback_reasons),
        "parse_failed_attempts": int(attempt_fallback_reasons.get("parse_failed", 0)),
        "action_index_not_int_attempts": int(attempt_fallback_reasons.get("action_index_not_int", 0)),
        "action_index_out_of_range_attempts": int(attempt_fallback_reasons.get("action_index_out_of_range", 0)),
        "chosen_action_counts": _counter_payload(chosen_action_types),
        "quality_flags": _counter_payload(quality_flags),
        "quality_opportunities": _counter_payload(quality_opportunities),
        "quality_misses": _counter_payload(quality_misses),
        "gen_ms": _number_stats(gen_ms),
        "enabled_action_count": _number_stats(enabled_counts),
        "final_policy_stats": final_stats,
    }


def summarize_spectate_stdout(stdout_path: Path) -> dict[str, Any]:
    if not stdout_path.exists():
        return {}
    result: dict[str, Any] = {}
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                value = ast.literal_eval(stripped)
            except Exception:
                continue
            if isinstance(value, dict) and ("steps" in value or "run_outcome" in value):
                result = value
    return result


def summarize_spectate_run(
    run_root: Path,
    *,
    trace_path: Path | None = None,
    stdout_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if trace_path is None:
        trace_path = run_root / "step_trace.jsonl"
    if stdout_path is None:
        stdout_path = run_root / "logs" / "spectate.stdout.log"
    if manifest_path is None:
        manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    result = summarize_spectate_stdout(stdout_path)
    trace = summarize_trace(trace_path) if trace_path.exists() else {}
    return {
        "kind": "spectate",
        "run_root": str(run_root),
        "manifest": manifest,
        "episode_result": result,
        "policy_trace": trace,
    }
