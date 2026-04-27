"""Review one rollout episode's turn-level play order with Kimi.

This is an offline teacher pass. It sends one complete combat trace to Kimi,
asks for turn-by-turn play-order review, and writes the raw response plus
training-friendly labels. It does not affect live policy decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.paths import ARTIFACTS_ROOT, DATASETS_ROOT, ensure_dirs  # noqa: E402
from llm.scripts.analyze_action_ordering import (  # noqa: E402
    _chosen,
    _energy,
    _legal_actions,
    _round,
)


DEFAULT_MODEL = "kimi-k2.6"
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_USAGE_PATH = ARTIFACTS_ROOT / "kimi_usage" / "usage.jsonl"
_PLAYER_HP_RE = re.compile(r"\bplayer:\s+hp=(\d+)/(\d+)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="", help="step_trace.jsonl. Defaults to latest dataset trace.")
    parser.add_argument("--episode-id", default="", help="Optional episode_id. Defaults to highest-signal episode.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="MOONSHOT_API_KEY")
    parser.add_argument("--max-tokens", "--max-completion-tokens", dest="max_tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-decision-state-chars", type=int, default=7000)
    parser.add_argument("--prompt-style", choices=["full", "compact"], default="compact")
    parser.add_argument("--focus-policy", choices=["all", "milestone"], default="all")
    parser.add_argument("--damage-turns", type=int, default=2)
    parser.add_argument("--usage-path", default=str(DEFAULT_USAGE_PATH))
    parser.add_argument("--max-api-calls", type=int, default=100)
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


def count_recorded_api_calls(path: Path) -> int:
    if not path.exists():
        return 0
    calls = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("provider") == "kimi" and row.get("dry_run") is not True:
                calls += int(row.get("call_count") or 1)
    return calls


def append_usage_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def latest_step_trace() -> Path:
    traces = [path for path in DATASETS_ROOT.rglob("step_trace.jsonl") if path.is_file()]
    if not traces:
        raise FileNotFoundError(f"No step_trace.jsonl found under {DATASETS_ROOT}")
    return max(traces, key=lambda path: path.stat().st_mtime)


def _reason(row: dict[str, Any]) -> str:
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    return str(decoded.get("reason") or "")


def _player_hp(user_message: str) -> tuple[int, int] | None:
    match = _PLAYER_HP_RE.search(user_message)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _flags(row: dict[str, Any]) -> list[str]:
    raw = row.get("quality_flags")
    return [str(flag) for flag in raw] if isinstance(raw, list) else []


def _chosen_label(row: dict[str, Any]) -> str:
    actions = _legal_actions(row)
    chosen = _chosen(row, actions)
    idx = chosen.get("index")
    card = str(chosen.get("card_id") or "")
    target = str(chosen.get("target") or "")
    if card.lower() == "end_turn":
        return f"[{idx}] end_turn"
    suffix = f" -> {target}" if target else ""
    return f"[{idx}] {card}{suffix}"


def _episode_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if episode_id:
            grouped[episode_id].append(row)
    return grouped


def _episode_signal_score(rows: list[dict[str, Any]]) -> tuple[int, int, int, str]:
    flags = Counter(flag for row in rows for flag in _flags(row))
    weighted = (
        flags.get("missed_visible_lethal", 0) * 8
        + flags.get("dangerous_end_turn", 0) * 3
        + sum(value for key, value in flags.items() if key not in {"missed_visible_lethal", "dangerous_end_turn"}) * 2
    )
    turns = len({(_round(str(row.get("user_message") or "")) or -1) for row in rows})
    return weighted, len(rows), turns, str(rows[0].get("episode_id") or "")


def select_episode_rows(rows: list[dict[str, Any]], episode_id: str = "") -> list[dict[str, Any]]:
    grouped = _episode_groups(rows)
    if episode_id:
        selected = grouped.get(episode_id)
        if not selected:
            raise ValueError(f"episode_id not found: {episode_id}")
        return sorted(selected, key=lambda row: int(row.get("episode_step") or row.get("step") or 0))
    if not grouped:
        raise ValueError("No episode_id values found in trace")
    selected_id = max(grouped, key=lambda key: _episode_signal_score(grouped[key]))
    return sorted(grouped[selected_id], key=lambda row: int(row.get("episode_step") or row.get("step") or 0))


def _trim(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...<truncated>"


def _section(user_message: str, name: str) -> str:
    lines = user_message.splitlines()
    out: list[str] = []
    in_section = False
    prefix = f"{name}:"
    for line in lines:
        if line.strip() == prefix:
            in_section = True
            out.append(line)
            continue
        if in_section and line and not line.startswith((" ", "\t")):
            break
        if in_section:
            out.append(line)
    return "\n".join(out).strip()


def _run_line(user_message: str) -> str:
    for line in user_message.splitlines():
        if line.startswith("run:"):
            return line.strip()
    return ""


def _compact_decision_state(user_message: str) -> str:
    keep = [
        _run_line(user_message),
        _section(user_message, "player"),
        _section(user_message, "enemies"),
        _section(user_message, "hand"),
        _section(user_message, "legal_actions"),
    ]
    return "\n".join(part for part in keep if part).strip()


def _static_context_from_first_state(user_message: str) -> dict[str, str]:
    return {
        "run": _run_line(user_message),
        "relics": _section(user_message, "relics"),
        "deck": _section(user_message, "deck"),
        "glossary": _section(user_message, "glossary"),
    }


def _focus_turns(turns: list[dict[str, Any]], *, damage_turns: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not turns:
        return [], {"rounds": [], "reasons": {}}
    selected: set[int] = set()
    reasons: dict[int, list[str]] = defaultdict(list)
    n = len(turns)
    for idx in range(min(2, n)):
        selected.add(idx)
        reasons[idx].append("first_2_turns")
    for idx in range(max(0, n - 2), n):
        selected.add(idx)
        reasons[idx].append("last_2_turns")
    if n > 4:
        mid_start = max(0, (n // 2) - 1)
        for idx in range(mid_start, min(n, mid_start + 2)):
            selected.add(idx)
            reasons[idx].append("middle_2_turns")
    damage_ranked = sorted(
        range(n),
        key=lambda idx: (float(turns[idx].get("observed_hp_loss") or 0), len(turns[idx].get("decisions") or [])),
        reverse=True,
    )
    for idx in damage_ranked[: max(0, damage_turns)]:
        if float(turns[idx].get("observed_hp_loss") or 0) > 0:
            selected.add(idx)
            reasons[idx].append("high_hp_loss_turn")
    focused = [turns[idx] for idx in sorted(selected)]
    return focused, {
        "rounds": [turn.get("round") for turn in focused],
        "reasons": {
            str(turns[idx].get("round")): reasons[idx]
            for idx in sorted(selected)
        },
        "whole_combat_turn_count": n,
    }


def build_episode_payload(
    rows: list[dict[str, Any]],
    *,
    max_decision_state_chars: int = 7000,
    focus_policy: str = "all",
    damage_turns: int = 2,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    first = rows[0]
    reward = first.get("episode_reward") if isinstance(first.get("episode_reward"), dict) else {}
    turn_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        round_no = _round(str(row.get("user_message") or ""))
        turn_rows[int(round_no or -1)].append(row)

    first_state = str(rows[0].get("user_message") or "")
    turns: list[dict[str, Any]] = []
    ordered_turn_items = [(round_no, items) for round_no, items in sorted(turn_rows.items())]
    for turn_index, (round_no, items) in enumerate(ordered_turn_items):
        decisions: list[dict[str, Any]] = []
        for row in items:
            actions = _legal_actions(row)
            chosen = _chosen(row, actions)
            quality = row.get("quality_report") if isinstance(row.get("quality_report"), dict) else {}
            decisions.append({
                "step": row.get("episode_step", row.get("step")),
                "energy_before": _energy(str(row.get("user_message") or "")),
                "chosen_action": _chosen_label(row),
                "chosen_action_index": chosen.get("index"),
                "reason": _reason(row),
                "quality_flags": _flags(row),
                "quality_opportunities": quality.get("opportunities") or {},
                "quality_misses": quality.get("misses") or {},
                "pre_decision_state": _trim(str(row.get("user_message") or ""), max_decision_state_chars),
                "compact_state": _trim(_compact_decision_state(str(row.get("user_message") or "")), max_decision_state_chars),
            })
        start_hp = _player_hp(str(items[0].get("user_message") or "")) if items else None
        next_hp = None
        if turn_index + 1 < len(ordered_turn_items):
            next_items = ordered_turn_items[turn_index + 1][1]
            if next_items:
                next_hp = _player_hp(str(next_items[0].get("user_message") or ""))
        observed_hp_loss = 0
        if isinstance(start_hp, tuple) and isinstance(next_hp, tuple):
            observed_hp_loss = max(0, int(start_hp[0]) - int(next_hp[0]))
        turns.append({
            "round": round_no if round_no >= 0 else None,
            "start_hp": start_hp[0] if isinstance(start_hp, tuple) else None,
            "next_turn_start_hp": next_hp[0] if isinstance(next_hp, tuple) else None,
            "observed_hp_loss": observed_hp_loss,
            "actions_played_in_order": [decision["chosen_action"] for decision in decisions],
            "decisions": decisions,
        })

    turn_summary = [
        {
            "round": turn.get("round"),
            "start_hp": turn.get("start_hp"),
            "next_turn_start_hp": turn.get("next_turn_start_hp"),
            "observed_hp_loss": turn.get("observed_hp_loss"),
            "actions_played_in_order": turn.get("actions_played_in_order") or [],
            "decision_steps": [decision.get("step") for decision in (turn.get("decisions") or [])],
        }
        for turn in turns
    ]
    focus_meta = {"rounds": [turn.get("round") for turn in turns], "reasons": {}, "whole_combat_turn_count": len(turns)}
    if focus_policy == "milestone":
        turns, focus_meta = _focus_turns(turns, damage_turns=damage_turns)

    return {
        "episode_id": first.get("episode_id"),
        "encounter_id": first.get("encounter_id"),
        "encounter_label": first.get("encounter_label"),
        "seed": first.get("seed"),
        "outcome": first.get("outcome"),
        "reward": reward,
        "steps": len(rows),
        "static_context": _static_context_from_first_state(first_state),
        "focus_policy": focus_policy,
        "focus": focus_meta,
        "whole_combat_turn_summary": turn_summary,
        "turns": turns,
        "all_quality_flags": dict(Counter(flag for row in rows for flag in _flags(row))),
    }


def compact_episode_for_prompt(episode: dict[str, Any]) -> dict[str, Any]:
    compact_turns: list[dict[str, Any]] = []
    for turn in episode.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        compact_decisions: list[dict[str, Any]] = []
        for decision in turn.get("decisions") or []:
            if not isinstance(decision, dict):
                continue
            compact_decisions.append({
                "step": decision.get("step"),
                "energy_before": decision.get("energy_before"),
                "chosen_action": decision.get("chosen_action"),
                "chosen_action_index": decision.get("chosen_action_index"),
                "reason": decision.get("reason"),
                "quality_flags": decision.get("quality_flags") or [],
                "quality_opportunities": decision.get("quality_opportunities") or {},
                "quality_misses": decision.get("quality_misses") or {},
                "state": decision.get("compact_state") or decision.get("pre_decision_state"),
            })
        compact_turns.append({
            "round": turn.get("round"),
            "actions_played_in_order": turn.get("actions_played_in_order") or [],
            "decisions": compact_decisions,
        })
    return {
        "episode_id": episode.get("episode_id"),
        "encounter_id": episode.get("encounter_id"),
        "seed": episode.get("seed"),
        "outcome": episode.get("outcome"),
        "reward": episode.get("reward"),
        "steps": episode.get("steps"),
        "static_context": episode.get("static_context") or {},
        "focus_policy": episode.get("focus_policy") or "all",
        "focus": episode.get("focus") or {},
        "whole_combat_turn_summary": episode.get("whole_combat_turn_summary") or [],
        "turns": compact_turns,
        "all_quality_flags": episode.get("all_quality_flags") or {},
    }


def build_messages(episode: dict[str, Any], *, prompt_style: str = "compact", thinking: str = "disabled") -> list[dict[str, str]]:
    schema = {
        "episode_judgement": "good|mixed|bad",
        "overall_score": "0-10 number",
        "summary_zh": "short Chinese summary",
        "turn_reviews": [
            {
                "round": "integer",
                "score": "0-10 number",
                "verdict": "good|mixed|bad",
                "played_sequence": ["actions as played"],
                "good_points": ["what was reasonable"],
                "issues": [
                    {
                        "step": "integer or null",
                        "severity": "minor|major|critical",
                        "problem_zh": "what was wrong",
                        "better_action_index": "integer or null",
                        "better_action": "legal action text or null",
                        "why_zh": "mechanism explanation",
                    }
                ],
                "ideal_sequence_zh": "best practical sequence for this turn",
            }
        ],
        "key_lessons": [
            {
                "tags": ["short tags"],
                "lesson_zh": "compact reusable lesson",
                "training_reason_en": "short English reason suitable for SFT labels",
            }
        ],
        "usable_training_labels": [
            {
                "step": "integer",
                "best_action_index": "integer",
                "reason_en": "short concrete reason",
                "confidence": "0-1 number",
            }
        ],
    }
    prompt_episode = compact_episode_for_prompt(episode) if prompt_style == "compact" else episode
    final_instruction = (
        "Return one valid JSON object matching this schema. Use Chinese for review text and English for training_reason_en/reason_en."
        if thinking == "disabled"
        else (
            "Think internally if needed. At the very end, output exactly one JSON object between "
            "<FINAL_JSON> and </FINAL_JSON>. The final JSON must match this schema. "
            "Use Chinese for review text and English for training_reason_en/reason_en."
        )
    )
    user = (
        "Review this Slay the Spire 2 combat trace at the turn level.\n"
        "Focus on whether the played card order within each turn is reasonable.\n"
        "Use only the listed legal_actions in each state; do not invent actions.\n"
        "If the original sequence is reasonable, say so. If not, identify the exact step and better legal action_index.\n"
        "Pay special attention to visible lethal, Vulnerable/BASH before follow-up attacks, energy spending, block vs incoming damage, and draw/setup order.\n"
        "Keep the output compact: at most 2 issues per reviewed turn, at most 8 usable_training_labels, "
        "and each explanation string under 80 Chinese characters or 25 English words. "
        "Do not copy legal_actions or combat_trace text into the output.\n"
        f"{final_instruction}\n\n"
        "output_schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "combat_trace:\n"
        f"{json.dumps(prompt_episode, ensure_ascii=False, indent=2)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Slay the Spire 2 teacher reviewing an AI policy. "
                "You judge play order, not just single actions. Return JSON only."
            ),
        },
        {"role": "user", "content": user},
    ]


def build_chat_request(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": args.max_tokens,
        "thinking": {"type": args.thinking},
        "stream": False,
    }


def _api_key(args: argparse.Namespace) -> tuple[str, str]:
    key = os.environ.get(args.api_key_env, "")
    if key:
        return args.api_key_env, key
    fallback = "KIMI_API_KEY"
    if args.api_key_env != fallback and os.environ.get(fallback):
        return fallback, str(os.environ[fallback])
    return args.api_key_env, ""


def call_kimi(*, base_url: str, api_key: str, body: dict[str, Any], timeout_s: float) -> tuple[dict[str, Any], float]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw), (time.monotonic() - started) * 1000.0
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kimi HTTP {exc.code}: {raw}") from exc


def response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def parse_review_json(content: str) -> tuple[dict[str, Any] | None, str]:
    if "<FINAL_JSON>" in content and "</FINAL_JSON>" in content:
        content = content.split("<FINAL_JSON>", 1)[1].split("</FINAL_JSON>", 1)[0].strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"json_parse_failed: {exc}"
    if not isinstance(payload, dict):
        return None, "not_json_object"
    return payload, "ok"


def _training_labels(review: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not review:
        return []
    labels = review.get("usable_training_labels")
    if not isinstance(labels, list):
        return []
    return [label for label in labels if isinstance(label, dict)]


def _default_out_dir(trace_path: Path, episode_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_episode = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in episode_id)[:96]
    return ARTIFACTS_ROOT / "reviews" / f"kimi_turn_order_{trace_path.parent.name}_{safe_episode}_{stamp}"


def main() -> int:
    args = parse_args()
    ensure_dirs()
    trace_path = Path(args.trace).resolve() if args.trace else latest_step_trace()
    rows = _read_jsonl(trace_path)
    episode_rows = select_episode_rows(rows, args.episode_id)
    episode = build_episode_payload(
        episode_rows,
        max_decision_state_chars=args.max_decision_state_chars,
        focus_policy=args.focus_policy,
        damage_turns=args.damage_turns,
    )
    episode_id = str(episode.get("episode_id") or "episode")
    messages = build_messages(episode, prompt_style=args.prompt_style, thinking=args.thinking)
    request_body = build_chat_request(args, messages)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_out_dir(trace_path, episode_id)
    key_env, key = _api_key(args)

    manifest = {
        "kind": "kimi_turn_order_review",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "trace_path": str(trace_path),
        "out_dir": str(out_dir),
        "episode_id": episode_id,
        "model": args.model,
        "thinking": args.thinking,
        "prompt_style": args.prompt_style,
        "base_url": args.base_url,
        "api_key_env": key_env,
        "usage_path": str(Path(args.usage_path).resolve()),
        "max_api_calls": args.max_api_calls,
        "dry_run": bool(args.dry_run),
        "has_api_key": bool(key),
        "episode_steps": len(episode_rows),
        "episode_turns": len(episode.get("turns") or []),
        "quality_flags": episode.get("all_quality_flags") or {},
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "episode_input.json", episode)
    _write_json(out_dir / "prompt_messages.json", {"messages": messages})
    _write_jsonl(
        out_dir / "openai_batch_request.jsonl",
        [{
            "custom_id": f"kimi-turn-order-{episode_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_body,
        }],
    )

    status = "dry_run" if args.dry_run else "no_api_key"
    response: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    parse_status = "not_run"
    latency_ms = 0.0
    error = ""
    usage_path = Path(args.usage_path).resolve()
    calls_before = count_recorded_api_calls(usage_path)
    calls_after = calls_before
    if not args.dry_run and key:
        calls_used = max(0, calls_after - calls_before)
        if args.max_api_calls >= 0 and calls_used >= args.max_api_calls:
            status = "api_budget_exceeded"
            error = f"Kimi API budget exceeded for this run: {calls_used}/{args.max_api_calls} calls"
        else:
            attempted = False
            usage_status = "not_started"
            usage_error = ""
            usage_parse_status = "not_run"
            usage_latency_ms = 0.0
            attempted = True
            calls_after = calls_before + 1
            try:
                response, latency_ms = call_kimi(
                    base_url=args.base_url,
                    api_key=key,
                    body=request_body,
                    timeout_s=args.timeout_s,
                )
                usage_latency_ms = latency_ms
                _write_json(out_dir / "kimi_raw_response.json", response)
                review, parse_status = parse_review_json(response_content(response))
                usage_parse_status = parse_status
                if review is not None:
                    _write_json(out_dir / "turn_order_review.json", review)
                    _write_jsonl(out_dir / "teacher_turn_labels.jsonl", _training_labels(review))
                status = "ok" if review is not None else "parse_failed"
                usage_status = status
            except Exception as exc:  # noqa: BLE001 - record API failure in artifact
                error = str(exc)
                usage_error = error
                status = "api_error"
                usage_status = status
            finally:
                if attempted:
                    append_usage_record(usage_path, {
                        "provider": "kimi",
                        "model": args.model,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "run_dir": str(out_dir),
                        "episode_id": episode_id,
                        "trace_path": str(trace_path),
                        "status": usage_status,
                        "parse_status": usage_parse_status,
                        "latency_ms": round(usage_latency_ms, 1),
                        "call_count": 1,
                        "dry_run": False,
                        "error": usage_error[:500],
                    })

    summary = {
        **manifest,
        "status": status,
        "api_calls_before": calls_before,
        "api_calls_after": calls_after,
        "api_calls_used": max(0, calls_after - calls_before),
        "api_calls_remaining": (
            max(0, args.max_api_calls - max(0, calls_after - calls_before))
            if args.max_api_calls >= 0 else None
        ),
        "latency_ms": round(latency_ms, 1),
        "parse_status": parse_status,
        "error": error,
        "outputs": {
            "manifest": str(out_dir / "manifest.json"),
            "episode_input": str(out_dir / "episode_input.json"),
            "prompt_messages": str(out_dir / "prompt_messages.json"),
            "batch_request": str(out_dir / "openai_batch_request.jsonl"),
            "raw_response": str(out_dir / "kimi_raw_response.json"),
            "review": str(out_dir / "turn_order_review.json"),
            "teacher_turn_labels": str(out_dir / "teacher_turn_labels.jsonl"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
