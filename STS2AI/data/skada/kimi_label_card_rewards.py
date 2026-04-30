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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

_STS2AI_ROOT = Path(__file__).resolve().parents[2]
if str(_STS2AI_ROOT) not in sys.path:
    sys.path.insert(0, str(_STS2AI_ROOT))

from llm.paths import ARTIFACTS_ROOT, DATASETS_ROOT, ensure_dirs  # noqa: E402


DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k2.6"
ACTION_RE = re.compile(r"^\s*\[(?P<idx>\d+)\]\s+(?P<label>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["realtime", "batch-prepare", "batch-submit", "batch-status", "batch-collect"],
        default="realtime",
    )
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-s", type=float, default=5.0)
    parser.add_argument("--resume-raw", dest="resume_raw", action="store_true", default=True)
    parser.add_argument("--no-resume-raw", dest="resume_raw", action="store_false")
    parser.add_argument("--completion-window", default="12h")
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--batch-input", default="")
    parser.add_argument("--batch-output-jsonl", default="")
    parser.add_argument("--batch-error-jsonl", default="")
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


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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


def _api_json(
    *,
    base_url: str,
    api_key: str,
    path: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip() + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kimi HTTP {exc.code}: {raw}") from exc
    return payload if isinstance(payload, dict) else {}


def _upload_file(
    *,
    base_url: str,
    api_key: str,
    path: Path,
    purpose: str,
    timeout_s: float,
) -> dict[str, Any]:
    boundary = f"----sts2ai-{uuid4().hex}"
    file_bytes = path.read_bytes()
    parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            f"{purpose}\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            "Content-Type: application/jsonl\r\n\r\n"
        ).encode("utf-8"),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    request = urllib.request.Request(
        base_url.rstrip() + "/files",
        data=b"".join(parts),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kimi HTTP {exc.code}: {raw}") from exc
    return payload if isinstance(payload, dict) else {}


def _download_file_content(*, base_url: str, api_key: str, file_id: str, timeout_s: float) -> str:
    request = urllib.request.Request(
        base_url.rstrip() + f"/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read().decode("utf-8")
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


def _labels_from_response(
    response: dict[str, Any],
    item_map: dict[str, dict[str, Any]],
    *,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    labels: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
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
        ok, status = _validate_review(review, item, min_confidence=min_confidence)
        status_counts[status] += 1
        if not ok:
            invalid.append({"candidate_id": cid, "review": review, "status": status})
            continue
        labels.append({
            "candidate_id": cid,
            "best_action_index": int(review["best_action_index"]),
            "confidence": float(review["confidence"]),
            "plan_zh": _clip(review.get("plan_zh"), 200),
            "reason_zh": _clip(review.get("reason_zh"), 200),
            "action_scores": review.get("action_scores"),
            "tags": review.get("tags") if isinstance(review.get("tags"), list) else [],
            "meta": item["row"].get("meta"),
            "legal_actions": item["legal_actions"],
        })
    return labels, invalid, status_counts


def _chat_body(group: list[dict[str, Any]], args: argparse.Namespace, *, stream: bool) -> dict[str, Any]:
    body = {
        "model": args.model,
        "messages": _messages(group, max_state_chars=args.max_state_chars),
        "response_format": {"type": "json_object"},
        "max_completion_tokens": args.max_tokens,
        "thinking": {"type": args.thinking},
    }
    if stream:
        body["stream"] = False
    return body


def _call_group(
    *,
    group_index: int,
    group: list[dict[str, Any]],
    raw_path: Path,
    args: argparse.Namespace,
    key: str,
) -> tuple[int, dict[str, Any], float]:
    body = _chat_body(group, args, stream=True)
    attempt = 0
    while True:
        try:
            response, elapsed = _call_chat(
                base_url=args.base_url,
                api_key=key,
                body=body,
                timeout_s=args.timeout_s,
            )
            _write_json(raw_path, response)
            return group_index, response, elapsed
        except Exception:
            if attempt >= max(0, args.max_retries):
                raise
            sleep_s = max(0.0, args.retry_sleep_s) * (2 ** attempt)
            attempt += 1
            if sleep_s > 0:
                time.sleep(sleep_s)


def _batch_input_rows(
    groups: list[tuple[int, list[dict[str, Any]]]],
    raw_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_existing = 0
    for group_index, group in groups:
        raw_path = raw_dir / f"group_{group_index:04d}.json"
        if args.resume_raw and raw_path.exists():
            skipped_existing += 1
            continue
        rows.append({
            "custom_id": f"group_{group_index:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _chat_body(group, args, stream=False),
        })
    return rows, skipped_existing


def _write_batch_input(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _group_index_from_custom_id(custom_id: str) -> int | None:
    match = re.match(r"^group_(\d+)$", custom_id)
    return int(match.group(1)) if match else None


def _load_batch_id(out_dir: Path, explicit: str) -> str:
    if explicit:
        return explicit
    payload = _read_json(out_dir / "batch_job.json")
    if isinstance(payload, dict):
        return str(payload.get("id") or "")
    return ""


def _parse_batch_output_jsonl(text: str, raw_dir: Path) -> tuple[int, list[dict[str, Any]], Counter[str]]:
    written = 0
    invalid: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid.append({"line": line_no, "status": "batch_output_json_error", "error": str(exc)})
            status_counts["batch_output_json_error"] += 1
            continue
        if not isinstance(row, dict):
            status_counts["batch_output_not_object"] += 1
            invalid.append({"line": line_no, "status": "batch_output_not_object"})
            continue
        custom_id = str(row.get("custom_id") or "")
        group_index = _group_index_from_custom_id(custom_id)
        if group_index is None:
            status_counts["batch_unknown_custom_id"] += 1
            invalid.append({"line": line_no, "status": "batch_unknown_custom_id", "custom_id": custom_id})
            continue
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        body = response.get("body") if isinstance(response.get("body"), dict) else None
        status_code = response.get("status_code")
        if not isinstance(body, dict) or (status_code is not None and int(status_code) >= 400):
            status_counts["batch_response_error"] += 1
            invalid.append({"custom_id": custom_id, "status": "batch_response_error", "response": response, "error": row.get("error")})
            continue
        _write_json(raw_dir / f"group_{group_index:04d}.json", body)
        written += 1
    return written, invalid, status_counts


def _parse_batch_error_jsonl(text: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    invalid: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            payload = {"raw": line, "error": str(exc)}
        invalid.append({"line": line_no, "status": "batch_error_file", "payload": payload})
        status_counts["batch_error_file"] += 1
    return invalid, status_counts


def _usage_summary(responses: list[dict[str, Any]]) -> dict[str, Any]:
    models: Counter[str] = Counter()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    missing_usage = 0
    for response in responses:
        model = str(response.get("model") or "unknown")
        models[model] += 1
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        if not usage:
            missing_usage += 1
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
    return {
        "responses": len(responses),
        "models": dict(models),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "missing_usage": missing_usage,
        "web_search_calls": 0,
    }


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
    key = os.environ.get(args.api_key_env) or os.environ.get("KIMI_API_KEY") or ""
    if args.mode in {"batch-submit", "batch-status"} and not args.dry_run and not key:
        raise SystemExit(
            f"missing API key: set ${args.api_key_env} or $KIMI_API_KEY, or pass --dry-run"
        )
    if args.mode == "batch-collect" and not args.dry_run and not key and not args.batch_output_jsonl:
        raise SystemExit(
            f"missing API key: set ${args.api_key_env} or $KIMI_API_KEY, or pass --dry-run"
        )

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
        "mode": args.mode,
        "dry_run": args.dry_run,
        "group_size": args.group_size,
        "workers": args.workers,
        "resume_raw": args.resume_raw,
        "completion_window": args.completion_window,
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
    raw_reused = 0
    group_size = max(1, args.group_size)
    groups = [
        (start // group_size, items[start:start + group_size])
        for start in range(0, len(items), group_size)
    ]

    if args.mode in {"batch-prepare", "batch-submit"}:
        batch_input_path = Path(args.batch_input).resolve() if args.batch_input else out_dir / "batch_input.jsonl"
        batch_rows, batch_skipped_existing = _batch_input_rows(groups, raw_dir, args)
        _write_batch_input(batch_input_path, batch_rows)
        summary = {
            **manifest,
            "batch_input": str(batch_input_path),
            "batch_requests": len(batch_rows),
            "batch_skipped_existing_groups": batch_skipped_existing,
            "outputs": {
                "candidates": str(out_dir / "candidates.jsonl"),
                "batch_input": str(batch_input_path),
                "manifest": str(out_dir / "manifest.json"),
            },
        }
        if args.mode == "batch-submit" and not args.dry_run:
            if not batch_rows:
                raise SystemExit("no pending batch rows to submit")
            uploaded = _upload_file(
                base_url=args.base_url,
                api_key=key,
                path=batch_input_path,
                purpose="batch",
                timeout_s=args.timeout_s,
            )
            input_file_id = str(uploaded.get("id") or "")
            if not input_file_id:
                raise SystemExit(f"file upload did not return id: {uploaded}")
            batch_job = _api_json(
                base_url=args.base_url,
                api_key=key,
                path="/batches",
                method="POST",
                body={
                    "input_file_id": input_file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": args.completion_window,
                },
                timeout_s=args.timeout_s,
            )
            _write_json(out_dir / "batch_upload.json", uploaded)
            _write_json(out_dir / "batch_job.json", batch_job)
            summary["input_file_id"] = input_file_id
            summary["batch_id"] = batch_job.get("id")
            summary["batch_status"] = batch_job.get("status")
            summary["outputs"]["batch_upload"] = str(out_dir / "batch_upload.json")
            summary["outputs"]["batch_job"] = str(out_dir / "batch_job.json")
        _write_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "batch-status":
        batch_id = _load_batch_id(out_dir, args.batch_id)
        if not batch_id:
            raise SystemExit("provide --batch-id or keep batch_job.json in --out-dir")
        batch_job = _api_json(
            base_url=args.base_url,
            api_key=key,
            path=f"/batches/{batch_id}",
            method="GET",
            timeout_s=args.timeout_s,
        )
        _write_json(out_dir / "batch_status.json", batch_job)
        print(json.dumps(batch_job, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "batch-collect":
        batch_id = _load_batch_id(out_dir, args.batch_id)
        output_jsonl = ""
        error_jsonl = ""
        batch_job: dict[str, Any] = {}
        if args.batch_output_jsonl:
            output_jsonl = Path(args.batch_output_jsonl).read_text(encoding="utf-8-sig")
        if args.batch_error_jsonl:
            error_jsonl = Path(args.batch_error_jsonl).read_text(encoding="utf-8-sig")
        if not output_jsonl and not batch_id:
            raise SystemExit("provide --batch-id, batch_job.json, or --batch-output-jsonl")
        if batch_id:
            batch_job = _api_json(
                base_url=args.base_url,
                api_key=key,
                path=f"/batches/{batch_id}",
                method="GET",
                timeout_s=args.timeout_s,
            )
            _write_json(out_dir / "batch_status.json", batch_job)
            status = str(batch_job.get("status") or "")
            if status != "completed" and not args.batch_output_jsonl:
                raise SystemExit(f"batch is not completed: status={status}")
            output_file_id = str(batch_job.get("output_file_id") or "")
            error_file_id = str(batch_job.get("error_file_id") or "")
            if output_file_id and not output_jsonl:
                output_jsonl = _download_file_content(
                    base_url=args.base_url,
                    api_key=key,
                    file_id=output_file_id,
                    timeout_s=args.timeout_s,
                )
                (out_dir / "batch_output.jsonl").write_text(output_jsonl, encoding="utf-8")
            if error_file_id and not error_jsonl:
                error_jsonl = _download_file_content(
                    base_url=args.base_url,
                    api_key=key,
                    file_id=error_file_id,
                    timeout_s=args.timeout_s,
                )
                (out_dir / "batch_error.jsonl").write_text(error_jsonl, encoding="utf-8")
        written, batch_invalid, batch_counts = _parse_batch_output_jsonl(output_jsonl, raw_dir) if output_jsonl else (0, [], Counter())
        error_invalid, error_counts = _parse_batch_error_jsonl(error_jsonl) if error_jsonl else ([], Counter())
        invalid.extend(batch_invalid)
        invalid.extend(error_invalid)
        status_counts.update(batch_counts)
        status_counts.update(error_counts)
        status_counts["batch_output_groups_written"] = written

    if args.dry_run:
        status_counts["dry_run"] = len(items)
    else:
        raw_responses: list[tuple[int, dict[str, Any]]] = []
        pending: list[tuple[int, list[dict[str, Any]], Path]] = []
        for group_index, group in groups:
            raw_path = raw_dir / f"group_{group_index:04d}.json"
            response = _read_json(raw_path) if args.resume_raw else None
            if response is not None:
                raw_responses.append((group_index, response))
                raw_reused += 1
            else:
                pending.append((group_index, group, raw_path))

        if args.mode == "batch-collect" and pending:
            status_counts["batch_missing_raw_groups"] += sum(len(group) for _, group, _ in pending)
            pending = []

        if 0 <= args.max_api_calls < len(pending):
            skipped = pending[args.max_api_calls:]
            status_counts["max_api_calls_reached"] += sum(len(group) for _, group, _ in skipped)
            pending = pending[:args.max_api_calls]

        if pending and not key:
            raise SystemExit(
                f"missing API key: set ${args.api_key_env} or $KIMI_API_KEY; "
                f"{len(pending)} group(s) are not available in raw/"
            )

        workers = max(1, args.workers)
        if workers == 1:
            for group_index, group, raw_path in pending:
                try:
                    _, response, elapsed = _call_group(
                        group_index=group_index,
                        group=group,
                        raw_path=raw_path,
                        args=args,
                        key=key,
                    )
                    calls += 1
                    latency_ms.append(elapsed)
                    raw_responses.append((group_index, response))
                except Exception as exc:
                    status_counts["api_error"] += len(group)
                    invalid.append({"group_index": group_index, "status": "api_error", "error": str(exc)})
                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for group_index, group, raw_path in pending:
                    future = executor.submit(
                        _call_group,
                        group_index=group_index,
                        group=group,
                        raw_path=raw_path,
                        args=args,
                        key=key,
                    )
                    futures[future] = (group_index, group)
                    if args.sleep_s > 0:
                        time.sleep(args.sleep_s)
                for future in as_completed(futures):
                    group_index, group = futures[future]
                    try:
                        _, response, elapsed = future.result()
                        calls += 1
                        latency_ms.append(elapsed)
                        raw_responses.append((group_index, response))
                    except Exception as exc:
                        status_counts["api_error"] += len(group)
                        invalid.append({"group_index": group_index, "status": "api_error", "error": str(exc)})

        for _, response in sorted(raw_responses, key=lambda item: item[0]):
            group_labels, group_invalid, group_counts = _labels_from_response(
                response,
                item_map,
                min_confidence=args.min_confidence,
            )
            labels.extend(group_labels)
            invalid.extend(group_invalid)
            status_counts.update(group_counts)

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
        "raw_reused_groups": raw_reused,
        "successful_responses": len(raw_responses),
        "token_usage": _usage_summary([response for _, response in sorted(raw_responses, key=lambda item: item[0])]),
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
