"""Sample and optionally self-rerank multiple actions for the same prompt.

This does not run real rollouts and does not require save/load. It asks the
model to reason about the same state multiple times, then can ask the model to
judge those candidate reasons and choose the most coherent one.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.metrics import read_jsonl, write_json  # noqa: E402
from llm.paths import BASE_MODEL_ID, RUNS_ROOT, ensure_dirs  # noqa: E402
from llm.data_pipeline.experience_library import (  # noqa: E402
    DEFAULT_EXPERIENCE_PATH,
    load_experience,
    render_experience_block,
    retrieve_experience,
)


_THINK_END_RE = re.compile(r"</think>\s*", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_ACTION_INDEX_RE = re.compile(r'"?action_index"?\s*:\s*(-?\d+)')
_BEST_SAMPLE_RE = re.compile(r'"?best_sample_index"?\s*:\s*(-?\d+)')
_LEGAL_INDEX_RE = re.compile(r"^\s*\[(\d+)]", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="hard_cases.jsonl or another JSONL with messages.")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--samples-per-state", type=int, default=8)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--self-rerank", action="store_true", help="Ask the model to judge its sampled candidates.")
    parser.add_argument("--experience-path", default=str(DEFAULT_EXPERIENCE_PATH))
    parser.add_argument("--experience-limit", type=int, default=4)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-new-tokens", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _strip_thinking(text: str) -> str:
    if not text:
        return ""
    if _THINK_END_RE.search(text):
        return _THINK_END_RE.split(text)[-1].strip()
    return _THINK_BLOCK_RE.sub("", text).strip()


def parse_action_index(raw_text: str) -> tuple[int | None, str]:
    stripped = _strip_thinking(raw_text).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = _ACTION_INDEX_RE.search(stripped)
        if not match:
            return None, "json_parse_failed"
        try:
            return int(match.group(1)), "json_parse_failed_but_index_found"
        except ValueError:
            return None, "action_index_not_int"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    value = payload.get("action_index")
    if isinstance(value, bool) or not isinstance(value, int):
        return None, "action_index_not_int"
    return int(value), "ok"


def parse_judge_selection(raw_text: str) -> tuple[int | None, int | None, str, str]:
    """Return (best_sample_index, action_index, reason, status)."""
    stripped = _strip_thinking(raw_text).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        sample_match = _BEST_SAMPLE_RE.search(stripped)
        action_match = _ACTION_INDEX_RE.search(stripped)
        best_sample = int(sample_match.group(1)) if sample_match else None
        action_index = int(action_match.group(1)) if action_match else None
        if best_sample is None and action_index is None:
            return None, None, "", "json_parse_failed"
        return best_sample, action_index, "", "json_parse_failed_but_selection_found"
    if not isinstance(payload, dict):
        return None, None, "", "json_not_object"
    best_raw = payload.get("best_sample_index")
    action_raw = payload.get("action_index")
    best_sample = best_raw if isinstance(best_raw, int) and not isinstance(best_raw, bool) else None
    action_index = action_raw if isinstance(action_raw, int) and not isinstance(action_raw, bool) else None
    reason = str(payload.get("reason") or "")
    if best_sample is None and action_index is None:
        return None, None, reason, "selection_missing"
    return best_sample, action_index, reason, "ok"


def _candidate_reason(raw_text: str) -> str:
    stripped = _strip_thinking(raw_text).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped[:180]
    if not isinstance(payload, dict):
        return stripped[:180]
    return str(payload.get("reason") or "")[:180]


def _messages_without_assistant(row: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role in {"system", "user"}:
            out.append({"role": role, "content": str(message.get("content") or "")})
    return out


def _user_text(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def listed_action_indices(user_text: str) -> set[int]:
    return {int(match.group(1)) for match in _LEGAL_INDEX_RE.finditer(user_text or "")}


def entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("user_hash") or f"row_{index:05d}")


class CandidateSampler:
    def __init__(self, args: argparse.Namespace) -> None:
        from unsloth import FastLanguageModel

        self.args = args
        self.experience_entries = load_experience(Path(args.experience_path))
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model_id,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir)
        FastLanguageModel.for_inference(model)
        self.model = model
        self.tokenizer = tokenizer

    def prompt(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.args.enable_thinking:
            kwargs["enable_thinking"] = True
        return self.tokenizer.apply_chat_template(messages, **kwargs) + "{"

    def sample_once(
        self,
        prompt_text: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> tuple[str, float]:
        import torch

        temperature = self.args.temperature if temperature is None else temperature
        max_new_tokens = self.args.max_new_tokens if max_new_tokens is None else max_new_tokens
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        t0 = time.monotonic()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = "{" + self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return raw.strip(), (time.monotonic() - t0) * 1000.0

    def judge_prompt(self, messages: list[dict[str, str]], samples: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        user_state = _user_text(messages)
        experience = retrieve_experience(
            user_state,
            self.experience_entries,
            limit=max(0, int(self.args.experience_limit)),
        )
        experience_block = render_experience_block(experience)
        candidate_lines: list[str] = []
        for sample in samples:
            action_index = sample.get("action_index")
            status = sample.get("status")
            reason = _candidate_reason(str(sample.get("raw_generation") or ""))
            candidate_lines.append(
                f"sample[{sample.get('sample_index')}]: action_index={action_index} status={status} reason={reason}"
            )
        judge_messages = [
            {
                "role": "system",
                "content": (
                    "You are judging Slay the Spire 2 candidate decisions for one state. "
                    "Do not assume any rollout or hidden future information. Compare only the current state, "
                    "legal actions, card text, enemy intent, energy, and candidate reasons. "
                    "Return exactly one JSON object."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{user_state}\n\n"
                    + (experience_block + "\n\n" if experience_block else "")
                    + "candidate_decisions:\n"
                    + "\n".join(candidate_lines)
                    + "\n\nReturn JSON: "
                    + '{"best_sample_index": I, "action_index": N, "reason": "why this candidate is most coherent"}'
                ),
            },
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            judge_messages,
            tokenize=False,
            add_generation_prompt=True,
        ) + "{"
        return prompt_text, [entry.to_json() for entry in experience]

    def judge(self, messages: list[dict[str, str]], samples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_text, experience = self.judge_prompt(messages, samples)
        raw, gen_ms = self.sample_once(
            prompt_text,
            temperature=self.args.judge_temperature,
            max_new_tokens=self.args.judge_max_new_tokens,
        )
        best_sample, action_index, reason, status = parse_judge_selection(raw)
        return {
            "raw_generation": raw,
            "best_sample_index": best_sample,
            "action_index": action_index,
            "reason": reason,
            "status": status,
            "gen_ms": round(gen_ms, 1),
            "experience": experience,
        }


def _summarize_samples(samples: list[dict[str, Any]], legal_indices: set[int]) -> dict[str, Any]:
    valid = [
        sample
        for sample in samples
        if isinstance(sample.get("action_index"), int)
        and (not legal_indices or int(sample["action_index"]) in legal_indices)
    ]
    invalid = len(samples) - len(valid)
    counts = Counter(int(sample["action_index"]) for sample in valid)
    total = len(samples)
    return {
        "valid_samples": len(valid),
        "invalid_samples": invalid,
        "valid_rate": round(len(valid) / total, 4) if total else None,
        "action_counts": {str(key): int(value) for key, value in counts.most_common()},
        "unique_valid_actions": len(counts),
        "entropy": entropy_from_counts(counts),
        "majority_action_index": counts.most_common(1)[0][0] if counts else None,
        "majority_count": counts.most_common(1)[0][1] if counts else 0,
    }


def _selected_action(summary: dict[str, Any], judge: dict[str, Any] | None) -> dict[str, Any]:
    if judge and isinstance(judge.get("action_index"), int):
        return {
            "source": "self_rerank",
            "action_index": int(judge["action_index"]),
            "reason": str(judge.get("reason") or ""),
            "best_sample_index": judge.get("best_sample_index"),
            "status": judge.get("status"),
        }
    if isinstance(summary.get("majority_action_index"), int):
        return {
            "source": "majority",
            "action_index": int(summary["majority_action_index"]),
            "reason": "majority sampled action",
            "best_sample_index": None,
            "status": "ok",
        }
    return {
        "source": "none",
        "action_index": None,
        "reason": "",
        "best_sample_index": None,
        "status": "no_valid_candidate",
    }


def main() -> int:
    args = parse_args()
    ensure_dirs()
    input_path = Path(args.input_jsonl).resolve()
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")
    run_name = args.run_name or f"candidate_probe_{time.strftime('%Y%m%d-%H%M%S')}"
    run_root = RUNS_ROOT / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    manifest = {
        "kind": "state_candidate_probe",
        "input_jsonl": str(input_path),
        "run_root": str(run_root),
        "adapter_dir": args.adapter_dir,
        "samples_per_state": args.samples_per_state,
        "limit": args.limit,
        "temperature": args.temperature,
        "self_rerank": args.self_rerank,
        "experience_path": args.experience_path,
        "experience_limit": args.experience_limit,
        "judge_temperature": args.judge_temperature,
        "dry_run": args.dry_run,
        "outputs": {
            "samples": str(run_root / "candidate_samples.jsonl"),
            "disagreements": str(run_root / "disagreement_states.jsonl"),
            "summary": str(run_root / "summary.json"),
        },
    }
    write_json(run_root / "manifest.json", manifest)

    if args.dry_run:
        write_json(run_root / "summary.json", {**manifest, "states": len(rows)})
        print(json.dumps({**manifest, "states": len(rows)}, ensure_ascii=False, indent=2))
        return 0

    sampler = CandidateSampler(args)
    all_payloads: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        messages = _messages_without_assistant(row)
        user_text = _user_text(messages)
        legal_indices = listed_action_indices(user_text)
        prompt_text = sampler.prompt(messages)
        samples: list[dict[str, Any]] = []
        for sample_index in range(max(1, args.samples_per_state)):
            raw, gen_ms = sampler.sample_once(prompt_text)
            action_index, status = parse_action_index(raw)
            in_legal = action_index in legal_indices if action_index is not None and legal_indices else action_index is not None
            samples.append({
                "sample_index": sample_index,
                "raw_generation": raw,
                "action_index": action_index,
                "status": status,
                "in_legal_actions": bool(in_legal),
                "gen_ms": round(gen_ms, 1),
            })
        summary = _summarize_samples(samples, legal_indices)
        judge = sampler.judge(messages, samples) if args.self_rerank else None
        payload = {
            "row_id": _row_id(row, row_index),
            "source_meta": {key: row.get(key) for key in ("source_file", "line_no", "score", "flags", "encounter_key", "encounter_label")},
            "original_action_index": row.get("action_index"),
            "legal_action_indices": sorted(legal_indices),
            "summary": summary,
            "judge": judge,
            "selected_action": _selected_action(summary, judge),
            "messages": messages,
            "samples": samples,
        }
        all_payloads.append(payload)
        if summary["unique_valid_actions"] >= 2 or summary["invalid_samples"] > 0:
            disagreements.append(payload)

    with (run_root / "candidate_samples.jsonl").open("w", encoding="utf-8") as handle:
        for payload in all_payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    with (run_root / "disagreement_states.jsonl").open("w", encoding="utf-8") as handle:
        for payload in disagreements:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    entropies = [float((payload.get("summary") or {}).get("entropy") or 0.0) for payload in all_payloads]
    valid_rates = [float((payload.get("summary") or {}).get("valid_rate") or 0.0) for payload in all_payloads]
    summary = {
        **manifest,
        "states": len(all_payloads),
        "disagreement_states": len(disagreements),
        "avg_entropy": round(sum(entropies) / len(entropies), 4) if entropies else 0.0,
        "avg_valid_rate": round(sum(valid_rates) / len(valid_rates), 4) if valid_rates else 0.0,
        "self_rerank_states": sum(1 for payload in all_payloads if payload.get("judge")),
    }
    write_json(run_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
