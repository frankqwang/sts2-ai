"""Batch Kimi combat reviews for Skada reset rollouts.

Each selected episode is reviewed as a whole combat, but the prompt focuses on
the first two turns, two middle turns, last two turns, and high-HP-loss turns.
The output labels are still verified later by build_teacher_dataset.py before
they become trainable samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.paths import ARTIFACTS_ROOT, ensure_dirs  # noqa: E402
from llm.scripts.kimi_review_turn_order import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_USAGE_PATH,
    append_usage_record,
    build_chat_request,
    build_episode_payload,
    build_messages,
    call_kimi,
    count_recorded_api_calls,
    parse_review_json,
    response_content,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="step_trace.jsonl from Skada combat rollout")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--max-api-calls", type=int, default=200)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="MOONSHOT_API_KEY")
    parser.add_argument("--usage-path", default=str(DEFAULT_USAGE_PATH))
    parser.add_argument("--max-tokens", "--max-completion-tokens", dest="max_tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--max-decision-state-chars", type=int, default=7000)
    parser.add_argument("--damage-turns", type=int, default=2)
    parser.add_argument("--skip-episode-id", action="append", default=[], help="episode_id to skip, for continuing a reviewed batch.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
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


def _group_episodes(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if episode_id:
            grouped[episode_id].append(row)
    for episode_id in list(grouped):
        grouped[episode_id].sort(key=lambda row: int(row.get("episode_step") or row.get("step") or 0))
    return grouped


def _episode_score(rows: list[dict[str, Any]]) -> tuple[float, int, str]:
    flags = Counter(flag for row in rows for flag in (row.get("quality_flags") or []))
    weighted = (
        flags.get("missed_visible_lethal", 0) * 8
        + flags.get("dangerous_end_turn", 0) * 5
        + flags.get("floating_energy_end_turn", 0) * 2
        + sum(value for key, value in flags.items() if key not in {"missed_visible_lethal", "dangerous_end_turn"})
    )
    try:
        payload = build_episode_payload(rows, focus_policy="all", max_decision_state_chars=1000)
        hp_loss = sum(float(turn.get("observed_hp_loss") or 0) for turn in payload.get("turns") or [])
    except Exception:
        hp_loss = 0.0
    return weighted + hp_loss, len(rows), str(rows[0].get("episode_id") or "")


def _api_key(api_key_env: str) -> tuple[str, str]:
    key = os.environ.get(api_key_env, "")
    if key:
        return api_key_env, key
    fallback = "KIMI_API_KEY"
    if api_key_env != fallback and os.environ.get(fallback):
        return fallback, str(os.environ[fallback])
    return api_key_env, ""


def _safe_name(value: str, limit: int = 80) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return text[:limit].strip("._") or "episode"


def _labels_from_review(review: dict[str, Any] | None) -> list[dict[str, Any]]:
    labels = review.get("usable_training_labels") if isinstance(review, dict) else None
    if not isinstance(labels, list):
        return []
    return [label for label in labels if isinstance(label, dict)]


def main() -> int:
    args = parse_args()
    ensure_dirs()
    trace_path = Path(args.trace).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        ARTIFACTS_ROOT / "reviews" / f"kimi_combat_batch_{trace_path.parent.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(trace_path)
    grouped = _group_episodes(rows)
    skip_episode_ids = {str(value) for value in args.skip_episode_id}
    episode_ids = [
        key for key in sorted(grouped, key=lambda key: _episode_score(grouped[key]), reverse=True)
        if key not in skip_episode_ids
    ]
    if args.limit_episodes > 0:
        episode_ids = episode_ids[: args.limit_episodes]

    key_env, key = _api_key(args.api_key_env)
    usage_path = Path(args.usage_path).resolve()
    calls_before = count_recorded_api_calls(usage_path)
    calls_after = calls_before
    manifest = {
        "kind": "kimi_combat_review_batch",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "trace_path": str(trace_path),
        "out_dir": str(out_dir),
        "episode_count": len(episode_ids),
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": key_env,
        "has_api_key": bool(key),
        "usage_path": str(usage_path),
        "max_api_calls": args.max_api_calls,
        "dry_run": bool(args.dry_run),
        "focus_policy": "milestone",
        "damage_turns": args.damage_turns,
        "skip_episode_ids": sorted(skip_episode_ids),
    }
    _write_json(out_dir / "manifest.json", manifest)

    labels_all: list[dict[str, Any]] = []
    review_paths: list[str] = []
    episode_input_paths: list[str] = []
    summaries: list[dict[str, Any]] = []
    parse_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    request_args = SimpleNamespace(
        model=args.model,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
    )

    for index, episode_id in enumerate(episode_ids):
        episode_dir = out_dir / f"{index:04d}_{_safe_name(episode_id)}"
        episode = build_episode_payload(
            grouped[episode_id],
            max_decision_state_chars=args.max_decision_state_chars,
            focus_policy="milestone",
            damage_turns=args.damage_turns,
        )
        messages = build_messages(episode, prompt_style="compact", thinking=args.thinking)
        request_body = build_chat_request(request_args, messages)
        _write_json(episode_dir / "episode_input.json", episode)
        _write_json(episode_dir / "prompt_messages.json", {"messages": messages})
        _write_jsonl(episode_dir / "openai_batch_request.jsonl", [{
            "custom_id": f"kimi-combat-{episode_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_body,
        }])

        status = "dry_run" if args.dry_run else "no_api_key"
        parse_status = "not_run"
        latency_ms = 0.0
        error = ""
        review: dict[str, Any] | None = None
        if not args.dry_run and key:
            calls_used = max(0, calls_after - calls_before)
            if args.max_api_calls >= 0 and calls_used >= args.max_api_calls:
                status = "api_budget_exceeded"
                error = f"Kimi API budget exceeded for this run: {calls_used}/{args.max_api_calls} calls"
            else:
                calls_after += 1
                try:
                    response, latency_ms = call_kimi(
                        base_url=args.base_url,
                        api_key=key,
                        body=request_body,
                        timeout_s=args.timeout_s,
                    )
                    _write_json(episode_dir / "kimi_raw_response.json", response)
                    review, parse_status = parse_review_json(response_content(response))
                    if review is not None:
                        _write_json(episode_dir / "turn_order_review.json", review)
                        labels = _labels_from_review(review)
                        _write_jsonl(episode_dir / "teacher_turn_labels.jsonl", labels)
                        labels_all.extend(labels)
                        review_paths.append(str(episode_dir / "turn_order_review.json"))
                        episode_input_paths.append(str(episode_dir / "episode_input.json"))
                    status = "ok" if review is not None else "parse_failed"
                except Exception as exc:  # noqa: BLE001 - keep batch moving and record failure
                    error = str(exc)
                    status = "api_error"
                finally:
                    append_usage_record(usage_path, {
                        "provider": "kimi",
                        "model": args.model,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "run_dir": str(episode_dir),
                        "episode_id": episode_id,
                        "trace_path": str(trace_path),
                        "status": status,
                        "parse_status": parse_status,
                        "latency_ms": round(latency_ms, 1),
                        "call_count": 1,
                        "dry_run": False,
                        "error": error[:500],
                    })
                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)

        parse_counts[parse_status] += 1
        status_counts[status] += 1
        summaries.append({
            "episode_id": episode_id,
            "episode_dir": str(episode_dir),
            "status": status,
            "parse_status": parse_status,
            "latency_ms": round(latency_ms, 1),
            "labels": len(_labels_from_review(review)),
            "focused_rounds": (episode.get("focus") or {}).get("rounds") or [],
            "error": error[:500],
        })

    _write_jsonl(out_dir / "teacher_turn_labels_all.jsonl", labels_all)
    _write_jsonl(out_dir / "episode_summaries.jsonl", summaries)
    summary = {
        **manifest,
        "api_calls_before": calls_before,
        "api_calls_after": calls_after,
        "api_calls_used": max(0, calls_after - calls_before),
        "api_calls_remaining": (
            max(0, args.max_api_calls - max(0, calls_after - calls_before))
            if args.max_api_calls >= 0 else None
        ),
        "labels": len(labels_all),
        "reviews_ok": status_counts.get("ok", 0),
        "status_counts": {key: int(value) for key, value in status_counts.items()},
        "parse_counts": {key: int(value) for key, value in parse_counts.items()},
        "review_paths": review_paths,
        "episode_input_paths": episode_input_paths,
        "outputs": {
            "labels": str(out_dir / "teacher_turn_labels_all.jsonl"),
            "episode_summaries": str(out_dir / "episode_summaries.jsonl"),
            "manifest": str(out_dir / "manifest.json"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
