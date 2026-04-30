"""Greedy exact-match eval for chat SFT adapters.

This is a lightweight offline check for JSON action_index style datasets. It
does not run the game; it only compares model output against the assistant
label in an existing eval.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.paths import BASE_MODEL_ID, EVALS_ROOT, ensure_dirs  # noqa: E402


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="eval.jsonl or train.jsonl")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--response-prefix", default="{")
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _json_object(raw_text: str) -> dict[str, Any] | None:
    text = _THINK_BLOCK_RE.sub("", raw_text or "").strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _action_index(raw_text: str) -> int | None:
    payload = _json_object(raw_text)
    if not isinstance(payload, dict):
        return None
    value = payload.get("action_index")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _decision_type(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if meta.get("decision_type"):
        return str(meta["decision_type"])
    user = ""
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            user = str(message.get("content") or "")
            break
    match = re.search(r"decision_type:\s*([A-Za-z0-9_\\-]+)", user)
    return match.group(1) if match else "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_dirs()
    dataset_path = Path(args.dataset).resolve()
    rows = _read_jsonl(dataset_path, max(0, int(args.limit)))
    if not rows:
        raise SystemExit(f"no rows: {dataset_path}")

    from unsloth import FastLanguageModel
    from peft import PeftModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    FastLanguageModel.for_inference(model)

    correct = 0
    generated = 0
    parse_failed = 0
    too_long = 0
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for idx, row in enumerate(rows):
        messages = [message for message in (row.get("messages") or []) if isinstance(message, dict)]
        if len(messages) < 2:
            continue
        gold = _action_index(str(messages[-1].get("content") or ""))
        prompt_messages = messages[:-1]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if args.enable_thinking:
            kwargs["enable_thinking"] = True
        prompt = tokenizer.apply_chat_template(prompt_messages, **kwargs)
        prompt_with_prefix = prompt + args.response_prefix
        encoded = tokenizer(prompt_with_prefix, add_special_tokens=False)
        if len(encoded.get("input_ids") or []) > args.max_seq_length:
            too_long += 1
            by_type[_decision_type(row)]["too_long"] += 1
            continue
        inputs = tokenizer(prompt_with_prefix, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        raw = (args.response_prefix + tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
        pred = _action_index(raw)
        dtype = _decision_type(row)
        generated += 1
        by_type[dtype]["total"] += 1
        if pred is None:
            parse_failed += 1
            by_type[dtype]["parse_failed"] += 1
        elif pred == gold:
            correct += 1
            by_type[dtype]["correct"] += 1
        else:
            by_type[dtype]["wrong"] += 1
        if len(samples) < 20:
            samples.append({
                "row": idx,
                "decision_type": dtype,
                "gold": gold,
                "pred": pred,
                "raw": raw[:500],
            })

    elapsed = time.monotonic() - t0
    run_name = args.run_name or f"chat_adapter_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = EVALS_ROOT / run_name
    payload = {
        "kind": "chat_adapter_exact_match_eval",
        "run_name": run_name,
        "dataset": str(dataset_path),
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "base_model": BASE_MODEL_ID,
        "rows": len(rows),
        "generated": generated,
        "too_long": too_long,
        "correct": correct,
        "parse_failed": parse_failed,
        "accuracy": round(correct / generated, 6) if generated else None,
        "parse_failed_rate": round(parse_failed / generated, 6) if generated else None,
        "elapsed_s": round(elapsed, 3),
        "rows_per_second": round(generated / elapsed, 4) if elapsed > 0 else None,
        "by_decision_type": {
            key: dict(counter)
            for key, counter in sorted(by_type.items())
        },
        "samples": samples,
    }
    _write_json(out_dir / "metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
