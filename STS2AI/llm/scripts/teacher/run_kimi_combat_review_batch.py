"""Batch teacher combat reviews for Skada reset rollouts.

Each selected episode is reviewed as a whole combat, but the prompt focuses on
the first two turns, two middle turns, last two turns, and high-HP-loss turns.
The output labels are still verified later by build_teacher_dataset.py before
they become trainable samples.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.paths import ARTIFACTS_ROOT, ensure_dirs  # noqa: E402
from llm.scripts.teacher.teacher_review_turn_order import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_CLAUDE_COMMAND,
    DEFAULT_CLAUDE_PROXY,
    DEFAULT_MODEL,
    DEFAULT_TEACHER_PROVIDER,
    DEFAULT_USAGE_PATH,
    append_usage_record,
    build_chat_request,
    build_episode_payload,
    build_messages,
    call_teacher_model,
    count_recorded_api_calls,
    normalize_provider,
    parse_review_json,
    raw_response_filename,
    response_content,
    resolve_provider_api_key_env,
    resolve_provider_base_url,
    resolve_provider_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, help="step_trace.jsonl from Skada combat rollout")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--max-api-calls", type=int, default=200)
    parser.add_argument("--provider", default=DEFAULT_TEACHER_PROVIDER, help="deepseek / kimi / claude_cli")
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL") or os.environ.get("KIMI_BASE_URL") or "")
    parser.add_argument("--api-key-env", default=os.environ.get("TEACHER_API_KEY_ENV") or "")
    parser.add_argument("--claude-command", default=os.environ.get("CLAUDE_CLI_COMMAND", DEFAULT_CLAUDE_COMMAND))
    parser.add_argument("--claude-proxy", default=os.environ.get("CLAUDE_PROXY", DEFAULT_CLAUDE_PROXY))
    parser.add_argument("--usage-path", default=str(DEFAULT_USAGE_PATH))
    parser.add_argument("--max-tokens", "--max-completion-tokens", dest="max_tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--max-decision-state-chars", type=int, default=7000)
    parser.add_argument("--damage-turns", type=int, default=2)
    parser.add_argument("--skip-episode-id", action="append", default=[], help="episode_id to skip, for continuing a reviewed batch.")
    parser.add_argument("--skip-episode-id-file", action="append", default=[], help="UTF-8 text/JSON file with episode_id values to skip.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip episodes already present in out-dir with a raw response or parsed review.")
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent normal API calls. Keep <= provider QPS/concurrency limit.")
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


def _read_skip_episode_ids(paths: list[str]) -> set[str]:
    ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            ids.update(str(value) for value in payload if str(value).strip())
        elif isinstance(payload, dict):
            values = payload.get("episode_ids") or payload.get("skip_episode_ids") or []
            if isinstance(values, list):
                ids.update(str(value) for value in values if str(value).strip())
        else:
            ids.update(line.strip() for line in text.splitlines() if line.strip())
    return ids


def _existing_episode_ids(out_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not out_dir.exists():
        return ids
    for episode_dir in out_dir.iterdir():
        if not episode_dir.is_dir():
            continue
        if not any(
            (episode_dir / name).exists()
            for name in ("turn_order_review.json", "teacher_raw_response.json", "kimi_raw_response.json", "claude_cli_raw_response.json")
        ):
            continue
        episode_path = episode_dir / "episode_input.json"
        if not episode_path.exists():
            continue
        try:
            episode = json.loads(episode_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        episode_id = str(episode.get("episode_id") or "")
        if episode_id:
            ids.add(episode_id)
    return ids


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
    """选送 Kimi 复盘的 episode 优先级。

    历史只看 quality_flags 加权 + 总损血。新增：
      - **boss 战 outcome=defeat 给 +200 大 bias**：act1/2/3 boss 是模型的真实瓶颈，必须强制选进。
      - **invalid_output / 0 win + high incoming**：边角失败模式
      - **特殊 power 战（SLIPPERY/INFESTED 等）出现失败**：让 Kimi 教这些 mechanic 应对

    返回元组：(score_desc, len, episode_id)
    """
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

    # 强 bias：boss/elite 战失败的 episode 优先送 Kimi（这是模型的核心弱点）
    bias = 0.0
    first = rows[0] if rows else {}
    encounter_id = str(first.get("encounter_id") or "").upper()
    encounter_tag = str(first.get("encounter_tag") or "").lower()
    outcome = str(first.get("outcome") or "")
    is_boss = "BOSS" in encounter_id or "boss" in encounter_tag
    is_elite = "ELITE" in encounter_id or "elite" in encounter_tag
    is_failure = outcome != "victory"
    if is_boss and is_failure:
        bias += 200.0  # boss 战失败永远在最前
    elif is_boss:
        bias += 80.0   # boss 战胜也优先（少见，让 Kimi 总结成功 pattern）
    elif is_elite and is_failure:
        bias += 40.0
    elif is_elite:
        bias += 10.0

    # 含特殊 power 的失败 episode（SLIPPERY/INFESTED/INTANGIBLE/BARRICADE/RITUAL）加权
    special_powers = {"SLIPPERY_POWER", "INFESTED_POWER", "INTANGIBLE_POWER", "BARRICADE_POWER", "RITUAL_POWER", "ARTIFACT_POWER", "SHARP_HIDE_POWER"}
    has_special = False
    for row in rows[:3]:  # 只看前 3 步避免遍历整 episode
        state = row.get("state") or {}
        for en in (state.get("enemies") or []):
            for power in (en.get("powers") or []):
                pid = str(power.get("id") or "").upper()
                if pid in special_powers:
                    has_special = True
                    break
            if has_special:
                break
        if has_special:
            break
    if has_special and is_failure:
        bias += 50.0

    return weighted + hp_loss + bias, len(rows), str(rows[0].get("episode_id") or "")


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
    args.provider = normalize_provider(args.provider)
    args.model = resolve_provider_model(args.provider, args.model)
    args.base_url = resolve_provider_base_url(args.provider, args.base_url)
    args.api_key_env = resolve_provider_api_key_env(args.provider, args.api_key_env)
    ensure_dirs()
    trace_path = Path(args.trace).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        ARTIFACTS_ROOT / "reviews" / f"kimi_combat_batch_{trace_path.parent.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(trace_path)
    grouped = _group_episodes(rows)
    skip_episode_ids = {str(value) for value in args.skip_episode_id}
    skip_episode_ids.update(_read_skip_episode_ids(args.skip_episode_id_file))
    existing_skip_ids = _existing_episode_ids(out_dir) if args.skip_existing else set()
    skip_episode_ids.update(existing_skip_ids)
    episode_ids = [
        key for key in sorted(grouped, key=lambda key: _episode_score(grouped[key]), reverse=True)
        if key not in skip_episode_ids
    ]
    if args.limit_episodes > 0:
        episode_ids = episode_ids[: args.limit_episodes]

    # OpenAI-compatible providers (kimi / deepseek / kimi_code) 都从 env 取 api key；
    # claude_cli 不需要 key, 改用 PATH 上的 claude 二进制。
    OPENAI_COMPATIBLE = {"kimi", "deepseek", "kimi_code"}
    if args.provider == "claude_cli":
        key_env, key = (args.api_key_env, "")
    else:
        key_env, key = _api_key(args.api_key_env)
    has_claude_cli = shutil.which(args.claude_command) is not None if args.provider == "claude_cli" else False
    usage_path = Path(args.usage_path).resolve()
    calls_before = count_recorded_api_calls(usage_path, provider=args.provider)
    calls_after = calls_before
    manifest = {
        "kind": "kimi_combat_review_batch",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "trace_path": str(trace_path),
        "out_dir": str(out_dir),
        "episode_count": len(episode_ids),
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": key_env,
        "has_api_key": bool(key),
        "claude_command": args.claude_command,
        "claude_proxy": args.claude_proxy if args.provider == "claude_cli" else "",
        "has_claude_cli": has_claude_cli,
        "usage_path": str(usage_path),
        "max_api_calls": args.max_api_calls,
        "dry_run": bool(args.dry_run),
        "focus_policy": "milestone",
        "damage_turns": args.damage_turns,
        "skip_episode_ids": sorted(skip_episode_ids),
        "skip_existing": bool(args.skip_existing),
        "existing_skip_count": len(existing_skip_ids),
        "max_workers": max(1, int(args.max_workers)),
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
    call_lock = Lock()
    usage_lock = Lock()

    def process_episode(index: int, episode_id: str) -> dict[str, Any]:
        nonlocal calls_after
        episode_dir = out_dir / f"{index:04d}_{_safe_name(episode_id)}"
        labels: list[dict[str, Any]] = []
        review_path = ""
        episode_input_path = str(episode_dir / "episode_input.json")
        status = "worker_error"
        parse_status = "not_run"
        latency_ms = 0.0
        error = ""
        review: dict[str, Any] | None = None
        focused_rounds: list[Any] = []
        try:
            episode = build_episode_payload(
                grouped[episode_id],
                max_decision_state_chars=args.max_decision_state_chars,
                focus_policy="milestone",
                damage_turns=args.damage_turns,
            )
            focused_rounds = (episode.get("focus") or {}).get("rounds") or []
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

            # provider gating: OpenAI-compatible (kimi/deepseek/kimi_code) needs an
            # API key; claude_cli needs the binary on PATH. Anything else → mark
            # "not_run" via the matching status so the manifest tells us why.
            if args.provider in OPENAI_COMPATIBLE:
                status = "dry_run" if args.dry_run else ("no_api_key" if not key else "ok")
                provider_ready = bool(key)
            elif args.provider == "claude_cli":
                status = "dry_run" if args.dry_run else ("no_claude_cli" if not has_claude_cli else "ok")
                provider_ready = has_claude_cli
            else:
                status = "unknown_provider"
                provider_ready = False
            should_call = False
            budget_error = ""
            if not args.dry_run and provider_ready:
                with call_lock:
                    calls_used = max(0, calls_after - calls_before)
                    if args.max_api_calls >= 0 and calls_used >= args.max_api_calls:
                        budget_error = f"Kimi API budget exceeded for this run: {calls_used}/{args.max_api_calls} calls"
                    else:
                        calls_after += 1
                        should_call = True
                if budget_error:
                    status = "api_budget_exceeded"
                    error = budget_error
                elif should_call:
                    try:
                        response, latency_ms = call_teacher_model(
                            provider=args.provider,
                            base_url=args.base_url,
                            api_key=key,
                            claude_command=args.claude_command,
                            claude_proxy=args.claude_proxy,
                            body=request_body,
                            messages=messages,
                            timeout_s=args.timeout_s,
                        )
                        _write_json(episode_dir / raw_response_filename(args.provider), response)
                        _write_json(episode_dir / "teacher_raw_response.json", response)
                        review, parse_status = parse_review_json(response_content(response))
                        if review is not None:
                            review_path = str(episode_dir / "turn_order_review.json")
                            _write_json(episode_dir / "turn_order_review.json", review)
                            labels = _labels_from_review(review)
                            _write_jsonl(episode_dir / "teacher_turn_labels.jsonl", labels)
                        status = "ok" if review is not None else "parse_failed"
                    except Exception as exc:  # noqa: BLE001 - keep batch moving and record failure
                        error = str(exc)
                        status = "api_error"
                    finally:
                        with usage_lock:
                            append_usage_record(usage_path, {
                                "provider": args.provider,
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
        except Exception as exc:  # noqa: BLE001 - record worker failure and continue
            error = str(exc)
            status = "worker_error"

        return {
            "labels": labels,
            "review_path": review_path,
            "episode_input_path": episode_input_path,
            "summary": {
                "index": index,
                "episode_id": episode_id,
                "episode_dir": str(episode_dir),
                "status": status,
                "parse_status": parse_status,
                "latency_ms": round(latency_ms, 1),
                "labels": len(labels),
                "focused_rounds": focused_rounds,
                "error": error[:500],
            },
            "status": status,
            "parse_status": parse_status,
        }

    results: list[dict[str, Any]] = []
    if args.max_workers <= 1 or len(episode_ids) <= 1:
        results = [process_episode(index, episode_id) for index, episode_id in enumerate(episode_ids)]
    else:
        worker_count = min(max(1, int(args.max_workers)), len(episode_ids))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(process_episode, index, episode_id)
                for index, episode_id in enumerate(episode_ids)
            ]
            for future in as_completed(futures):
                results.append(future.result())

    for result in sorted(results, key=lambda item: int((item.get("summary") or {}).get("index") or 0)):
        labels_all.extend(result.get("labels") or [])
        if result.get("review_path"):
            review_paths.append(str(result["review_path"]))
            episode_input_paths.append(str(result["episode_input_path"]))
        parse_counts[str(result.get("parse_status") or "not_run")] += 1
        status_counts[str(result.get("status") or "unknown")] += 1
        summaries.append(result.get("summary") or {})

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
