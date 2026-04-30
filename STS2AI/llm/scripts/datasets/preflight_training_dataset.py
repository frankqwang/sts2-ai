"""Preflight token and response-mask checks before launching training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tokenizer", required=True, help="Adapter/model path used for chat_template tokenization")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--min-response-retention", type=float, default=0.95)
    parser.add_argument("--max-p95-tokens", type=int, default=0)
    parser.add_argument("--max-p95-assistant-start", type=int, default=0)
    parser.add_argument("--out", default="")
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
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row["_source_line"] = line_no
                rows.append(row)
    return rows


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int(len(sorted_values) * q)))
    return int(sorted_values[idx])


def _stats(values: list[int]) -> dict[str, int]:
    values = sorted(values)
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "min": values[0],
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": values[-1],
    }


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer

    dataset_dir = Path(args.dataset_dir).resolve()
    rows = _read_jsonl(dataset_dir / "train.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.tokenizer).resolve()), trust_remote_code=True)
    marker = "<|im_start|>assistant\n"
    token_lengths: list[int] = []
    assistant_starts: list[int] = []
    retained = 0
    missing_marker = 0
    source_counts: Counter[str] = Counter()
    longest: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text = text.replace("<|im_start|>assistant\n<think>\n\n</think>\n\n", marker)
        pos = text.rfind(marker)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_lengths.append(len(ids))
        if pos < 0:
            missing_marker += 1
        else:
            start = len(tokenizer(text[:pos], add_special_tokens=False)["input_ids"])
            assistant_starts.append(start)
            truncated = tokenizer.decode(ids[: args.max_seq_length], skip_special_tokens=False)
            if marker in truncated:
                retained += 1
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        source_counts[str(meta.get("source") or "unknown")] += 1
        longest.append({
            "row": idx,
            "source_line": row.get("_source_line"),
            "tokens": len(ids),
            "assistant_start": assistant_starts[-1] if pos >= 0 and assistant_starts else None,
            "source": str(meta.get("source") or "unknown"),
            "floor": meta.get("floor"),
            "encounter_id": meta.get("encounter_id"),
        })

    retention = retained / max(1, len(rows))
    summary = {
        "kind": "training_dataset_preflight",
        "dataset_dir": str(dataset_dir),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "rows": len(rows),
        "max_seq_length": args.max_seq_length,
        "assistant_marker_missing": missing_marker,
        "assistant_retained_after_truncation": retained,
        "assistant_retention_rate": round(retention, 6),
        "token_lengths": _stats(token_lengths),
        "assistant_start_tokens": _stats(assistant_starts),
        "source_counts": {key: int(value) for key, value in source_counts.most_common()},
        "longest_rows": sorted(longest, key=lambda item: int(item.get("tokens") or 0), reverse=True)[:10],
        "thresholds": {
            "min_response_retention": args.min_response_retention,
            "max_p95_tokens": args.max_p95_tokens,
            "max_p95_assistant_start": args.max_p95_assistant_start,
        },
        "passed": True,
        "reasons": [],
    }
    if retention + 1e-12 < args.min_response_retention:
        summary["passed"] = False
        summary["reasons"].append(
            f"assistant retention {retention:.4f} < {args.min_response_retention:.4f}"
        )
    if args.max_p95_tokens > 0 and summary["token_lengths"]["p95"] > args.max_p95_tokens:
        summary["passed"] = False
        summary["reasons"].append(
            f"p95 tokens {summary['token_lengths']['p95']} > {args.max_p95_tokens}"
        )
    if args.max_p95_assistant_start > 0 and summary["assistant_start_tokens"]["p95"] > args.max_p95_assistant_start:
        summary["passed"] = False
        summary["reasons"].append(
            f"p95 assistant start {summary['assistant_start_tokens']['p95']} > {args.max_p95_assistant_start}"
        )

    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
