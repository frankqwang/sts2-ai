#!/usr/bin/env python3
"""Label Skada card reward SFT rows with Kimi teacher reviews.

The script reads an existing non-combat SFT dataset, samples card_reward rows,
asks Kimi to review the full prompt context, and writes:

- labels.jsonl: raw validated teacher labels
- train.jsonl/eval.jsonl: SFT rows with assistant targets replaced by Kimi labels
- summary.json: counts and artifact paths
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

_STS2AI_ROOT = Path(__file__).resolve().parents[2]
if str(_STS2AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_STS2AI_ROOT))

from llm.paths import ARTIFACTS_ROOT, DATASETS_ROOT, ensure_dirs  # noqa: E402


DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k2.6"
ACTION_RE = re.compile(r"^\s*\[(?P<idx>\d+)\]\s+(?P<label>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--model", default=os.environ.get("KIMI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="MOONSHOT_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--max-state-chars", type=int, default=7000)
    parser.add_argument("--min-confidence", type=float, default=0.65)
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


def _assistant(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            try:
                payload = json.loads(str(message.get("content") or ""))
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


def _user(row: dict[str, Any]) -> str:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _legal_actions(user: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    in_actions = False
    for line in user.splitlines():
        if line.strip() == "legal_actions:":
            in_actions = True
            continue
        if in_actions and line and not line.startswith((" ", "\t")):
            break
        if not in_actions:
            continue
        match = ACTION_RE.match(line)
        if match:
            actions.append({"action_index": int(match.group("idx")), "label": match.group("label").strip()})
    return actions


def _floor(row: dict[str, Any]) -> int:
    try:
        return int((row.get("meta") or {}).get("floor") or 0)
    except (TypeError, ValueError):
        return 0


def _selected(row: dict[str, Any]) -> str:
    return str((row.get("meta") or {}).get("selected") or "")


def _score_candidate(row: dict[str, Any]) -> float:
    user = _user(row)
    floor = _floor(row)
    selected = _selected(row)
    score = 1.0
    if 1 <= floor <= 17:
        score += 3.0
    if selected != "SKIP":
        score += 2.0
    if any(token in user for token in ("Elite", "Boss", "elite_before_rest=true")):
        score += 1.5
    if "hp_delta=-" in user or "dmg_taken=" in user:
        score += 1.0
    if any(card in user for card in ("FIEND_FIRE", "BURNING_PACT", "FEEL_NO_PAIN", "DARK_EMBRACE", "DEMON_FORM", "OFFERING")):
        score += 1.0
    return score


def _sample(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    card_rows = [
        row for row in rows
        if (row.get("meta") or {}).get("decision_type") == "card_reward" and _legal_actions(_user(row))
    ]
    rng.shuffle(card_rows)
    card_rows.sort(key=lambda row: (-_score_candidate(row), str((row.get("meta") or {}).get("run_id")), _floor(row)))
    seen: set[tuple[int, int, str]] = set()
    selected: list[dict[str, Any]] = []
    for row in card_rows:
        meta = row.get("meta") or {}
        key = (int(meta.get("run_id") or 0), _floor(row), _selected(row))
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def _candidate_id(index: int, row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    return f"cardreward-{index:05d}-run{meta.get('run_id')}-f{meta.get('floor')}"


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "\n...<truncated>"


def _messages(group: list[dict[str, Any]], *, max_state_chars: int) -> list[dict[str, str]]:
    blocks: list[str] = []
    for item in group:
        row = item["row"]
        user = _trim(_user(row), max_state_chars)
        original = _assistant(row).get("action_index")
        blocks.append(
            f"## candidate {item['candidate_id']}\n"
            f"human_selected={_selected(row)} original_action_index={original}\n"
            f"{user}"
        )
    prompt = (
        "你是 Slay the Spire 2 选卡老师。请基于每个 candidate 的完整上下文判断 card_reward。\n"
        "必须综合当前卡组/遗物、boss、未来路线、近期掉血、下一层风险、楼层阶段和所有候选牌；不要只看三张卡。\n"
        "Skada 人类选择来自胜利局，通常应尊重；只有明显更优时才改选。不能发明未列出的动作。\n"
        "plan_zh 和 reason_zh 都必须简短，每个不超过 200 个中文字，但要包含关键因果。\n"
        "action_scores 必须覆盖该 candidate 下所有 legal_actions 的 action_index；负例 note_zh 要说明为什么较差，不要写泛泛的 lower priority。\n\n"
        "只返回合法 JSON 对象，schema:\n"
        "{\"reviews\":[{\"candidate_id\":\"...\",\"best_action_index\":0,\"confidence\":0.0,"
        "\"plan_zh\":\"<=200字\",\"reason_zh\":\"<=200字\","
        "\"action_scores\":[{\"action_index\":0,\"score\":0.0,\"note_zh\":\"简短负例/正例原因\"}],"
        "\"tags\":[\"boss\",\"route\",\"deck\"]}]}\n\n"
        "candidates:\n" + "\n\n".join(blocks)
    )
    return [
        {"role": "system", "content": "你是严谨的 STS2 选卡标注老师。只输出 JSON。"},
        {"role": "user", "content": prompt},
    ]


def _call_chat(*, base_url: str, api_key: str, body: dict[str, Any], timeout_s: float) -> tuple[dict[str, Any], float]:
    endpoint = base_url.rstrip() + "/chat/completions"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8")), (time.monotonic() - started) * 1000.0
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kimi HTTP {exc.code}: {raw}") from exc


def _response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    text = text.strip()
    try:
        payload = json.loads(text)
        return (payload, "ok") if isinstance(payload, dict) else (None, "not_object")
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start:end + 1])
                return (payload, "ok_extracted") if isinstance(payload, dict) else (None, "not_object")
            except json.JSONDecodeError:
                pass
    return None, "json_parse_failed"


def _clip(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit]


def _validate_review(review: dict[str, Any], item: dict[str, Any], *, min_confidence: float) -> tuple[bool, str]:
    legal = {action["action_index"] for action in item["legal_actions"]}
    best = review.get("best_action_index")
    if isinstance(best, bool) or not isinstance(best, int) or best not in legal:
        return False, "invalid_best_action_index"
    confidence = review.get("confidence")
    if not isinstance(confidence, (int, float)) or float(confidence) < min_confidence:
        return False, "low_confidence"
    score_items = review.get("action_scores")
    if not isinstance(score_items, list):
        return False, "missing_action_scores"
    score_indices = {
        item.get("action_index")
        for item in score_items
        if isinstance(item, dict) and isinstance(item.get("action_index"), int)
    }
    if not legal.issubset(score_indices):
        return False, "incomplete_action_scores"
    return True, "ok"


def _assistant_from_label(label: dict[str, Any]) -> str:
    scores: list[dict[str, Any]] = []
    for item in label["action_scores"]:
        if not isinstance(item, dict):
            continue
        scores.append({
            "action_index": int(item.get("action_index")),
            "score": round(float(item.get("score") or 0.0), 2),
            "note": _clip(item.get("note_zh"), 80),
        })
    payload = {
        "action_index": int(label["best_action_index"]),
        "confidence": round(float(label["confidence"]), 2),
        "action_scores": scores,
        "plan": _clip(label.get("plan_zh"), 200),
        "reason": _clip(label.get("reason_zh"), 200),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _replace_assistant(row: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    for message in out.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            message["content"] = _assistant_from_label(label)
    meta = out.setdefault("meta", {})
    meta["teacher"] = "kimi_card_reward"
    meta["teacher_confidence"] = round(float(label.get("confidence") or 0.0), 3)
    meta["original_selected"] = meta.get("selected")
    meta["teacher_action_index"] = int(label["best_action_index"])
    meta["teacher_tags"] = label.get("tags") if isinstance(label.get("tags"), list) else []
    return out


def main() -> int:
    args = parse_args()
    ensure_dirs()
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        ARTIFACTS_ROOT / "llm" / "datasets" / f"skada_card_reward_kimi_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(dataset_dir / "train.jsonl") + _read_jsonl(dataset_dir / "eval.jsonl")
    selected_rows = _sample(rows, limit=args.limit, seed=args.seed)
    items = [
        {
            "candidate_id": _candidate_id(index, row),
            "row": row,
            "legal_actions": _legal_actions(_user(row)),
        }
        for index, row in enumerate(selected_rows)
    ]
    item_map = {item["candidate_id"]: item for item in items}
    key = os.environ.get(args.api_key_env) or os.environ.get("KIMI_API_KEY") or ""
    manifest = {
        "kind": "kimi_card_reward_labels",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "selected": len(items),
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "has_api_key": bool(key),
        "dry_run": args.dry_run,
        "group_size": args.group_size,
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_jsonl(out_dir / "candidates.jsonl", [
        {
            "candidate_id": item["candidate_id"],
            "meta": item["row"].get("meta"),
            "legal_actions": item["legal_actions"],
            "user_message": _user(item["row"]),
        }
        for item in items
    ])

    labels: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    latency_ms: list[float] = []
    calls = 0

    if args.dry_run or not key:
        status_counts["dry_run" if args.dry_run else "no_api_key"] = len(items)
    else:
        for start in range(0, len(items), max(1, args.group_size)):
            if 0 <= args.max_api_calls <= calls:
                status_counts["max_api_calls_reached"] += len(items) - start
                break
            group = items[start:start + max(1, args.group_size)]
            body = {
                "model": args.model,
                "messages": _messages(group, max_state_chars=args.max_state_chars),
                "response_format": {"type": "json_object"},
                "max_completion_tokens": args.max_tokens,
                "thinking": {"type": args.thinking},
                "stream": False,
            }
            try:
                response, elapsed = _call_chat(
                    base_url=args.base_url,
                    api_key=key,
                    body=body,
                    timeout_s=args.timeout_s,
                )
                calls += 1
                latency_ms.append(elapsed)
                _write_json(raw_dir / f"group_{start // max(1, args.group_size):04d}.json", response)
                payload, parse_status = _parse_json_object(_response_content(response))
                status_counts[parse_status] += 1
                review_items = payload.get("reviews") if isinstance(payload, dict) and isinstance(payload.get("reviews"), list) else []
                for review in review_items:
                    if not isinstance(review, dict):
                        continue
                    cid = str(review.get("candidate_id") or "")
                    item = item_map.get(cid)
                    if not item:
                        invalid.append({"review": review, "status": "unknown_candidate_id"})
                        continue
                    ok, status = _validate_review(review, item, min_confidence=args.min_confidence)
                    status_counts[status] += 1
                    if not ok:
                        invalid.append({"candidate_id": cid, "review": review, "status": status})
                        continue
                    label = {
                        "candidate_id": cid,
                        "best_action_index": int(review["best_action_index"]),
                        "confidence": float(review["confidence"]),
                        "plan_zh": _clip(review.get("plan_zh"), 200),
                        "reason_zh": _clip(review.get("reason_zh"), 200),
                        "action_scores": review.get("action_scores"),
                        "tags": review.get("tags") if isinstance(review.get("tags"), list) else [],
                        "meta": item["row"].get("meta"),
                        "legal_actions": item["legal_actions"],
                    }
                    labels.append(label)
            except Exception as exc:
                status_counts["api_error"] += len(group)
                invalid.append({"group_start": start, "status": "api_error", "error": str(exc)})
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

    label_by_id = {str(label["candidate_id"]): label for label in labels}
    annotated_rows = [
        _replace_assistant(item["row"], label_by_id[item["candidate_id"]])
        for item in items
        if item["candidate_id"] in label_by_id
    ]
    rng = random.Random(args.seed)
    rng.shuffle(annotated_rows)
    eval_size = int(len(annotated_rows) * max(0.0, min(0.5, args.eval_ratio)))
    eval_rows = annotated_rows[:eval_size]
    train_rows = annotated_rows[eval_size:]
    _write_jsonl(out_dir / "labels.jsonl", labels)
    _write_jsonl(out_dir / "invalid.jsonl", invalid)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)

    summary = {
        **manifest,
        "api_calls": calls,
        "labels": len(labels),
        "invalid": len(invalid),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "status_counts": dict(status_counts),
        "latency_ms_avg": round(sum(latency_ms) / len(latency_ms), 1) if latency_ms else None,
        "outputs": {
            "candidates": str(out_dir / "candidates.jsonl"),
            "labels": str(out_dir / "labels.jsonl"),
            "invalid": str(out_dir / "invalid.jsonl"),
            "train": str(out_dir / "train.jsonl"),
            "eval": str(out_dir / "eval.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
