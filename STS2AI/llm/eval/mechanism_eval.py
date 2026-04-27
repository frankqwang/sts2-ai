"""Offline mechanism-understanding eval for action-index adapters.

This does not execute a rollout. It asks the model on already collected states
and scores three local properties:

- whether the output is strict JSON with an integer action_index;
- whether the action_index is legal and matches the engine-supervised target;
- whether the reason mentions the concrete effects observed from engine
  settlement_events, such as damage, block, energy, powers, and card moves.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.data_pipeline.action_decoder import action_score_margin  # noqa: E402
from llm.paths import BASE_MODEL_ID, EVALS_ROOT, ensure_dirs  # noqa: E402
from llm.scripts.sample_state_candidates import parse_action_index  # noqa: E402


_LEGAL_INDEX_RE = re.compile(r"^\s*\[(?P<index>-?\d+)\]\s+", re.MULTILINE)
_THINK_END_RE = re.compile(r"</think>\s*", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ExpectedEffect:
    kind: str
    amount: str = ""
    target: str = ""
    card_id: str = ""
    power_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Dataset eval.jsonl/train.jsonl with messages + meta.")
    parser.add_argument("--adapter-dir", default="", help="LoRA adapter dir. Empty means base model.")
    parser.add_argument("--base-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
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


def _strip_thinking(text: str) -> str:
    if not text:
        return ""
    if _THINK_END_RE.search(text):
        return _THINK_END_RE.split(text)[-1].strip()
    return _THINK_BLOCK_RE.sub("", text).strip()


def _strict_json_payload(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    stripped = _strip_thinking(raw_text).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None, "json_parse_failed"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    if "action_index" not in payload:
        return payload, "action_index_missing"
    if isinstance(payload.get("action_index"), bool) or not isinstance(payload.get("action_index"), int):
        return payload, "action_index_not_int"
    if "reason" in payload and not isinstance(payload.get("reason"), str):
        return payload, "reason_not_string"
    return payload, "ok"


def _messages(row: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role in {"system", "user"}:
            out.append({"role": role, "content": str(message.get("content") or "")})
    return out[:2]


def _user_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _target_payload(row: dict[str, Any]) -> dict[str, Any]:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            try:
                payload = json.loads(str(message.get("content") or ""))
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


def _target_action_index(row: dict[str, Any]) -> int | None:
    raw = _target_payload(row).get("action_index")
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _target_reason(row: dict[str, Any]) -> str:
    raw = _target_payload(row).get("reason")
    return str(raw or "")


def legal_indices_from_user(user_text: str) -> set[int]:
    return {int(match.group("index")) for match in _LEGAL_INDEX_RE.finditer(user_text or "")}


def _as_confidence(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def _action_scores(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        try:
            action_index = int(item.get("action_index", item.get("index")))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        payload: dict[str, Any] = {
            "action_index": action_index,
            "score": round(score, 4),
        }
        note = str(item.get("note") or item.get("reason") or "").strip()
        if note:
            payload["note"] = note[:80]
        out.append(payload)
    return sorted(out, key=lambda row: float(row["score"]), reverse=True)


def _amount(value: Any) -> str:
    try:
        as_float = float(value)
    except Exception:
        return str(value or "")
    if as_float.is_integer():
        return str(int(as_float))
    return f"{as_float:g}"


def _target(event: dict[str, Any]) -> str:
    raw = event.get("target_id") or event.get("actor_id") or ""
    combat_id = event.get("target_combat_id")
    if combat_id:
        return f"{raw}#{combat_id}"
    return str(raw or "")


def expected_effects_from_events(events: list[dict[str, Any]]) -> list[ExpectedEffect]:
    effects: list[ExpectedEffect] = []

    total_energy = sum(
        int(event.get("energy_spent") or 0)
        for event in events
        if event.get("type") == "energy_spent"
    )
    if total_energy:
        effects.append(ExpectedEffect(kind="energy", amount=str(total_energy)))

    for event in events:
        if event.get("type") != "damage_received":
            continue
        damage = int(event.get("unblocked_damage") or event.get("total_damage") or 0)
        if damage > 0:
            effects.append(ExpectedEffect(kind="damage", amount=str(damage), target=_target(event)))

    total_block = sum(int(event.get("amount_int") or 0) for event in events if event.get("type") == "block_gained")
    if total_block:
        effects.append(ExpectedEffect(kind="block", amount=str(total_block)))

    for event in events:
        if event.get("type") != "power_received":
            continue
        power_id = str(event.get("power_id") or "")
        amount = _amount(event.get("amount_value", event.get("amount_int", "")))
        effects.append(ExpectedEffect(kind="power", amount=amount, target=_target(event), power_id=power_id))

    for event_type, kind in (
        ("card_drawn", "draw"),
        ("card_discarded", "discard"),
        ("card_exhausted", "exhaust"),
        ("card_generated", "generate"),
    ):
        seen: set[str] = set()
        for event in events:
            if event.get("type") != event_type:
                continue
            card_id = str(event.get("card_id") or "")
            if card_id and card_id not in seen:
                seen.add(card_id)
                effects.append(ExpectedEffect(kind=kind, card_id=card_id))
                if len(seen) >= 3:
                    break

    return effects[:8]


def _contains_amount(text: str, amount: str) -> bool:
    if not amount:
        return True
    return re.search(rf"(?<!\d){re.escape(amount)}(?!\d)", text) is not None


def _power_aliases(power_id: str) -> list[str]:
    upper = power_id.upper()
    aliases = [power_id.lower()]
    if "VULNERABLE" in upper:
        aliases.extend(["vulnerable", "vulnerability"])
    if "WEAK" in upper:
        aliases.append("weak")
    if "STRENGTH" in upper:
        aliases.append("strength")
    if "POISON" in upper:
        aliases.append("poison")
    return [alias for alias in aliases if alias]


def effect_matched(effect: ExpectedEffect, reason: str) -> bool:
    text = (reason or "").lower()
    card = effect.card_id.lower()
    if effect.kind == "energy":
        return "energy" in text and _contains_amount(text, effect.amount)
    if effect.kind == "damage":
        return ("damage" in text or "deal" in text or "deals" in text) and _contains_amount(text, effect.amount)
    if effect.kind == "block":
        return "block" in text and _contains_amount(text, effect.amount)
    if effect.kind == "power":
        has_power = any(alias in text for alias in _power_aliases(effect.power_id))
        return has_power and _contains_amount(text, effect.amount)
    if effect.kind == "draw":
        return ("draw" in text or "draws" in text) and (not card or card in text)
    if effect.kind == "discard":
        return ("discard" in text or "discards" in text) and (not card or card in text)
    if effect.kind == "exhaust":
        return ("exhaust" in text or "exhausts" in text) and (not card or card in text)
    if effect.kind == "generate":
        return ("generate" in text or "generates" in text or "add" in text or "adds" in text) and (not card or card in text)
    return False


def score_reason_effects(reason: str, effects: list[ExpectedEffect]) -> dict[str, Any]:
    if not effects:
        return {"expected": 0, "matched": 0, "recall": None, "by_kind": {}}
    by_kind: dict[str, dict[str, int]] = {}
    matched = 0
    for effect in effects:
        ok = effect_matched(effect, reason)
        bucket = by_kind.setdefault(effect.kind, {"expected": 0, "matched": 0})
        bucket["expected"] += 1
        if ok:
            matched += 1
            bucket["matched"] += 1
    return {
        "expected": len(effects),
        "matched": matched,
        "recall": round(matched / len(effects), 4),
        "by_kind": by_kind,
    }


class EvalModel:
    def __init__(self, args: argparse.Namespace) -> None:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model_id,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        if args.adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter_dir)
        FastLanguageModel.for_inference(model)
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def run(self, messages: list[dict[str, str]]) -> tuple[str, float]:
        import torch

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "{"
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        t0 = time.monotonic()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=self.args.temperature > 0,
                temperature=self.args.temperature if self.args.temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return "{" + self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), (time.monotonic() - t0) * 1000.0


def _number_stats(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "p50": round(ordered[len(ordered) // 2], 4),
        "avg": round(sum(ordered) / len(ordered), 4),
        "max": round(ordered[-1], 4),
    }


def _default_run_root(run_name: str) -> Path:
    name = run_name or f"mechanism_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return EVALS_ROOT / name


def _write_examples(path: Path, results: list[dict[str, Any]]) -> None:
    selected = sorted(
        results,
        key=lambda row: (
            0 if not row.get("action_exact") else 1,
            float(((row.get("reason_score") or {}).get("recall") if (row.get("reason_score") or {}).get("recall") is not None else -1)),
        ),
    )[:8]
    lines = ["# Mechanism Eval Examples", ""]
    for i, row in enumerate(selected, 1):
        lines.append(f"## Example {i}")
        lines.append("")
        lines.append(f"- row_index: `{row.get('row_index')}`")
        lines.append(f"- parse_status: `{row.get('parse_status')}`")
        lines.append(f"- target_action_index: `{row.get('target_action_index')}`")
        lines.append(f"- generated_action_index: `{row.get('generated_action_index')}`")
        lines.append(f"- action_exact: `{row.get('action_exact')}`")
        lines.append(f"- confidence: `{row.get('confidence')}`")
        lines.append(f"- score_margin: `{row.get('score_margin')}`")
        lines.append(f"- reason_recall: `{(row.get('reason_score') or {}).get('recall')}`")
        lines.append("")
        lines.append("### Expected effects")
        lines.append("```json")
        lines.append(json.dumps(row.get("expected_effects") or [], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### Target")
        lines.append("```json")
        lines.append(json.dumps(row.get("target") or {}, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### Generated")
        lines.append("```text")
        lines.append(str(row.get("raw_generation") or "")[:1200])
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    input_path = Path(args.input_jsonl).resolve()
    run_root = Path(args.out_dir).resolve() if args.out_dir else _default_run_root(args.run_name)
    run_root.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(input_path)
    input_rows = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    model = EvalModel(args)
    results: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    effect_kind_expected: Counter[str] = Counter()
    effect_kind_matched: Counter[str] = Counter()
    latencies: list[float] = []
    confidences: list[float] = []
    margins: list[float] = []
    score_list_lengths: list[int] = []

    for row_index, row in enumerate(rows):
        messages = _messages(row)
        user_text = _user_text(row)
        legal_indices = legal_indices_from_user(user_text)
        target = _target_payload(row)
        target_action = _target_action_index(row)
        events = [
            dict(event)
            for event in ((row.get("meta") or {}).get("settlement_events") or [])
            if isinstance(event, dict)
        ]
        effects = expected_effects_from_events(events)
        raw, gen_ms = model.run(messages)
        latencies.append(gen_ms)
        payload, strict_status = _strict_json_payload(raw)
        parsed_index, parse_status = parse_action_index(raw)
        status = strict_status if strict_status != "ok" else parse_status
        status_counts[status] += 1
        confidence = _as_confidence((payload or {}).get("confidence"))
        if confidence is not None:
            confidences.append(confidence)
        scores = _action_scores((payload or {}).get("action_scores", (payload or {}).get("scores")))
        if scores:
            score_list_lengths.append(len(scores))
        margin = action_score_margin(scores)
        if margin is not None:
            margins.append(float(margin))
        reason = str((payload or {}).get("reason") or "")
        reason_score = score_reason_effects(reason, effects)
        for kind, bucket in (reason_score.get("by_kind") or {}).items():
            effect_kind_expected[str(kind)] += int(bucket.get("expected") or 0)
            effect_kind_matched[str(kind)] += int(bucket.get("matched") or 0)
        action_valid = parsed_index in legal_indices if parsed_index is not None else False
        action_exact = parsed_index == target_action if parsed_index is not None and target_action is not None else False
        results.append({
            "row_index": row_index,
            "parse_status": status,
            "strict_json_status": strict_status,
            "target_action_index": target_action,
            "generated_action_index": parsed_index,
            "action_valid": action_valid,
            "action_exact": action_exact,
            "gen_ms": round(gen_ms, 1),
            "target": target,
            "raw_generation": raw,
            "reason": reason,
            "confidence": confidence,
            "action_scores": scores,
            "score_margin": margin,
            "target_reason": _target_reason(row),
            "reason_score": reason_score,
            "expected_effects": [effect.__dict__ for effect in effects],
            "meta": row.get("meta") or {},
        })

    with (run_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    recalls = [
        float(score["recall"])
        for score in (row.get("reason_score") or {} for row in results)
        if score.get("recall") is not None
    ]
    valid = sum(1 for row in results if row.get("action_valid"))
    exact = sum(1 for row in results if row.get("action_exact"))
    strict_ok = sum(1 for row in results if row.get("strict_json_status") == "ok")
    with_confidence = sum(1 for row in results if row.get("confidence") is not None)
    with_scores = sum(1 for row in results if row.get("action_scores"))
    with_margin = sum(1 for row in results if row.get("score_margin") is not None)
    kind_payload = {
        kind: {
            "expected": int(effect_kind_expected[kind]),
            "matched": int(effect_kind_matched[kind]),
            "recall": round(effect_kind_matched[kind] / effect_kind_expected[kind], 4)
            if effect_kind_expected[kind]
            else None,
        }
        for kind in sorted(effect_kind_expected)
    }
    metrics = {
        "kind": "mechanism_eval",
        "run_root": str(run_root),
        "input_jsonl": str(input_path),
        "input_rows": input_rows,
        "cases": len(results),
        "adapter_dir": args.adapter_dir or None,
        "base_model": args.base_model_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strict_json_ok": strict_ok,
        "strict_json_ok_rate": round(strict_ok / len(results), 4) if results else None,
        "action_valid": valid,
        "action_valid_rate": round(valid / len(results), 4) if results else None,
        "action_exact": exact,
        "action_exact_rate": round(exact / len(results), 4) if results else None,
        "reason_effect_recall": _number_stats(recalls),
        "confidence": _number_stats(confidences),
        "confidence_present_rate": round(with_confidence / len(results), 4) if results else None,
        "action_scores_present_rate": round(with_scores / len(results), 4) if results else None,
        "score_margin_present_rate": round(with_margin / len(results), 4) if results else None,
        "score_margin": _number_stats(margins),
        "action_score_count": _number_stats(score_list_lengths),
        "low_margin_cases": sum(1 for value in margins if value <= 1.0),
        "high_margin_cases": sum(1 for value in margins if value >= 5.0),
        "effect_recall_by_kind": kind_payload,
        "parse_status_counts": {key: int(value) for key, value in status_counts.most_common()},
        "latency_ms": _number_stats(latencies),
        "results": str(run_root / "results.jsonl"),
        "examples": str(run_root / "examples.md"),
        "args": vars(args),
    }
    _write_json(run_root / "metrics.json", metrics)
    _write_json(run_root / "manifest.json", {
        "metrics": str(run_root / "metrics.json"),
        "results": str(run_root / "results.jsonl"),
        "examples": str(run_root / "examples.md"),
    })
    _write_examples(run_root / "examples.md", results)
    return metrics


def main() -> None:
    metrics = evaluate(parse_args())
    print(f"[mechanism-eval] metrics -> {metrics['run_root']}\\metrics.json")
    print(
        "[mechanism-eval] "
        f"strict_json={metrics['strict_json_ok_rate']} "
        f"action_exact={metrics['action_exact_rate']} "
        f"reason_recall_avg={(metrics['reason_effect_recall'] or {}).get('avg')}"
    )


if __name__ == "__main__":
    main()
