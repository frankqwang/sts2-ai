"""Run Kimi reviews for sampled teacher candidate states.

The input is produced by sample_kimi_teacher_candidates.py. This runner uses
normal chat completions, not the provider Batch API, so it can validate and
record results immediately. It never writes API keys to artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.paths import ARTIFACTS_ROOT, ensure_dirs  # noqa: E402
from llm.scripts.analyze_action_ordering import _legal_actions  # noqa: E402
from llm.scripts.kimi_review_turn_order import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_USAGE_PATH,
    append_usage_record,
    count_recorded_api_calls,
    parse_review_json,
    response_content,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="candidates.jsonl from sample_kimi_teacher_candidates.py")
    parser.add_argument("--batch-request", required=True, help="grouped openai_batch_request.jsonl")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="MOONSHOT_API_KEY")
    parser.add_argument("--usage-path", default=str(DEFAULT_USAGE_PATH))
    parser.add_argument("--max-api-calls", type=int, default=100)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--sleep-s", type=float, default=0.0)
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


def _api_key(api_key_env: str) -> tuple[str, str]:
    key = os.environ.get(api_key_env, "")
    if key:
        return api_key_env, key
    fallback = "KIMI_API_KEY"
    if api_key_env != fallback and os.environ.get(fallback):
        return fallback, str(os.environ[fallback])
    return api_key_env, ""


def _call_chat(*, base_url: str, api_key: str, body: dict[str, Any], timeout_s: float) -> tuple[dict[str, Any], float]:
    endpoint = base_url.rstrip() + "/chat/completions"
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


def _review_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("reviews")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if "candidate_id" in payload:
        return [payload]
    return []


def _as_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _legal_indices(candidate: dict[str, Any]) -> set[int]:
    user = str((candidate.get("source") or {}).get("user_message") or "")
    return {
        int(action["index"])
        for action in _legal_actions({"user_message": user})
        if isinstance(action.get("index"), int)
    }


def _validate_review(
    review: dict[str, Any],
    *,
    candidates: dict[str, dict[str, Any]],
    min_confidence: float,
) -> tuple[bool, str, dict[str, Any]]:
    candidate_id = str(review.get("candidate_id") or "")
    if candidate_id not in candidates:
        return False, "unknown_candidate_id", {}
    action = review.get("best_action_index")
    if isinstance(action, bool) or not isinstance(action, int):
        return False, "best_action_index_not_int", {}
    legal = _legal_indices(candidates[candidate_id])
    if int(action) not in legal:
        return False, "best_action_index_not_legal", {"legal_indices": sorted(legal)}
    confidence = _as_confidence(review.get("confidence"))
    if confidence < min_confidence:
        return False, "low_confidence", {"confidence": confidence}
    return True, "ok", {"confidence": confidence}


def _label_from_review(review: dict[str, Any], candidate: dict[str, Any], confidence: float) -> dict[str, Any]:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    features = candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
    return {
        "candidate_id": review.get("candidate_id"),
        "best_action_index": int(review["best_action_index"]),
        "confidence": confidence,
        "judgement": str(review.get("judgement") or ""),
        "reason_en": str(review.get("reason_en") or "")[:240],
        "reason_zh": str(review.get("reason_zh") or "")[:240],
        "mechanism_tags": review.get("mechanism_tags") if isinstance(review.get("mechanism_tags"), list) else [],
        "original_action_index": features.get("original_index"),
        "source": {
            key: source.get(key)
            for key in (
                "episode_id",
                "episode_step",
                "step",
                "encounter_id",
                "encounter_tag",
                "encounter_key",
                "seed",
                "outcome",
                "source_file",
                "source_line",
            )
        },
        "user_message": source.get("user_message"),
    }


def _usage_payload(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
        raw = usage.get(key)
        if isinstance(raw, int):
            out[key] = raw
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        out["prompt_cached_tokens"] = int(details["cached_tokens"])
    return out


def main() -> int:
    args = parse_args()
    ensure_dirs()
    candidates_path = Path(args.candidates).resolve()
    batch_path = Path(args.batch_request).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        ARTIFACTS_ROOT / "reviews" / f"kimi_teacher_review_run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates_rows = _read_jsonl(candidates_path)
    candidate_map = {str(row.get("candidate_id") or ""): row for row in candidates_rows if row.get("candidate_id")}
    batch_rows = _read_jsonl(batch_path)
    if args.limit_groups > 0:
        batch_rows = batch_rows[: args.limit_groups]

    key_env, key = _api_key(args.api_key_env)
    usage_path = Path(args.usage_path).resolve()
    calls_before = count_recorded_api_calls(usage_path)
    calls_after = calls_before
    manifest = {
        "kind": "kimi_teacher_candidate_review_run",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "candidates": str(candidates_path),
        "batch_request": str(batch_path),
        "out_dir": str(out_dir),
        "base_url": args.base_url,
        "api_key_env": key_env,
        "has_api_key": bool(key),
        "usage_path": str(usage_path),
        "max_api_calls": args.max_api_calls,
        "min_confidence": args.min_confidence,
        "candidate_count": len(candidate_map),
        "group_count": len(batch_rows),
        "dry_run": args.dry_run,
    }
    _write_json(out_dir / "manifest.json", manifest)

    all_reviews: list[dict[str, Any]] = []
    valid_labels: list[dict[str, Any]] = []
    invalid_reviews: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    parse_status_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()

    if args.dry_run or not key:
        status = "dry_run" if args.dry_run else "no_api_key"
    else:
        status = "running"
        for group_index, group in enumerate(batch_rows):
            custom_id = str(group.get("custom_id") or f"group-{group_index:04d}")
            if args.max_api_calls >= 0 and calls_after >= args.max_api_calls:
                status = "api_budget_exceeded"
                group_summaries.append({"custom_id": custom_id, "status": status})
                break
            body = group.get("body") if isinstance(group.get("body"), dict) else {}
            usage_status = "not_started"
            usage_error = ""
            parse_status = "not_run"
            latency_ms = 0.0
            try:
                response, latency_ms = _call_chat(
                    base_url=args.base_url,
                    api_key=key,
                    body=body,
                    timeout_s=args.timeout_s,
                )
                _write_json(raw_dir / f"{custom_id}.json", response)
                review_payload, parse_status = parse_review_json(response_content(response))
                parse_status_counts[parse_status] += 1
                usage = _usage_payload(response)
                usage_totals.update(usage)
                if review_payload is None:
                    usage_status = "parse_failed"
                    group_summaries.append({
                        "custom_id": custom_id,
                        "status": "parse_failed",
                        "parse_status": parse_status,
                        "latency_ms": round(latency_ms, 1),
                        "usage": usage,
                    })
                else:
                    if "reviews" not in review_payload and "candidate_id" not in review_payload and custom_id in candidate_map:
                        review_payload = {"candidate_id": custom_id, **review_payload}
                    items = _review_items(review_payload)
                    group_valid = 0
                    group_invalid = 0
                    for item in items:
                        item = {**item, "_group_id": custom_id}
                        ok, validation_status, extra = _validate_review(
                            item,
                            candidates=candidate_map,
                            min_confidence=args.min_confidence,
                        )
                        validation_counts[validation_status] += 1
                        all_reviews.append(item)
                        if ok:
                            confidence = float(extra["confidence"])
                            candidate = candidate_map[str(item.get("candidate_id"))]
                            valid_labels.append(_label_from_review(item, candidate, confidence))
                            group_valid += 1
                        else:
                            invalid_reviews.append({
                                "review": item,
                                "validation_status": validation_status,
                                **extra,
                            })
                            group_invalid += 1
                    usage_status = "ok"
                    group_summaries.append({
                        "custom_id": custom_id,
                        "status": "ok",
                        "parse_status": parse_status,
                        "reviews": len(items),
                        "valid": group_valid,
                        "invalid": group_invalid,
                        "latency_ms": round(latency_ms, 1),
                        "usage": usage,
                    })
            except Exception as exc:  # noqa: BLE001 - persist run diagnostics
                usage_error = str(exc)
                usage_status = "api_error"
                parse_status_counts["api_error"] += 1
                group_summaries.append({
                    "custom_id": custom_id,
                    "status": "api_error",
                    "error": usage_error[:500],
                })
            finally:
                append_usage_record(usage_path, {
                    "provider": "kimi",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_dir": str(out_dir),
                    "custom_id": custom_id,
                    "status": usage_status,
                    "parse_status": parse_status,
                    "latency_ms": round(latency_ms, 1),
                    "call_count": 1,
                    "dry_run": False,
                    "error": usage_error[:500],
                })
                calls_after = count_recorded_api_calls(usage_path)
            print(
                json.dumps(
                    {
                        "group": group_index + 1,
                        "groups": len(batch_rows),
                        "custom_id": custom_id,
                        "calls_after": calls_after,
                        "status": group_summaries[-1].get("status"),
                        "valid_total": len(valid_labels),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
        if status == "running":
            status = "ok"

    _write_jsonl(out_dir / "reviews.jsonl", all_reviews)
    _write_jsonl(out_dir / "valid_labels.jsonl", valid_labels)
    _write_jsonl(out_dir / "invalid_reviews.jsonl", invalid_reviews)
    _write_jsonl(out_dir / "group_summaries.jsonl", group_summaries)
    summary = {
        **manifest,
        "status": status,
        "api_calls_before": calls_before,
        "api_calls_after": calls_after,
        "api_calls_used": max(0, calls_after - calls_before),
        "api_calls_remaining": args.max_api_calls - calls_after if args.max_api_calls >= 0 else None,
        "review_count": len(all_reviews),
        "valid_label_count": len(valid_labels),
        "invalid_review_count": len(invalid_reviews),
        "parse_status_counts": dict(parse_status_counts),
        "validation_counts": dict(validation_counts),
        "usage_totals": dict(usage_totals),
        "outputs": {
            "manifest": str(out_dir / "manifest.json"),
            "summary": str(out_dir / "summary.json"),
            "reviews": str(out_dir / "reviews.jsonl"),
            "valid_labels": str(out_dir / "valid_labels.jsonl"),
            "invalid_reviews": str(out_dir / "invalid_reviews.jsonl"),
            "group_summaries": str(out_dir / "group_summaries.jsonl"),
            "raw_dir": str(raw_dir),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if status in {"ok", "dry_run", "no_api_key"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
