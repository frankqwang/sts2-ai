"""Ask the model to reselect actions after review, without rollout.

This probes whether review/experience changes the model's choice on the same
state. It does not execute the action and therefore cannot prove win-rate
improvement; it measures local improvements such as fixing visible lethal,
avoiding bad end_turn, Bash-before-attack opportunities, and reason consistency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.data_pipeline.experience_library import (  # noqa: E402
    DEFAULT_EXPERIENCE_PATH,
    load_experience,
    render_experience_block,
    retrieve_experience,
)
from llm.paths import ARTIFACTS_ROOT, BASE_MODEL_ID, ensure_dirs  # noqa: E402
from llm.scripts.analyze_action_ordering import (  # noqa: E402
    _chosen,
    _energy,
    _enemies,
    _legal_actions,
)
from llm.scripts.sample_state_candidates import parse_action_index  # noqa: E402


_THINK_END_RE = re.compile(r"</think>\s*", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_MATH_GE_RE = re.compile(r"\b(-?\d+(?:\.\d+)?)\s*>=\s*(-?\d+(?:\.\d+)?)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="hard_cases.jsonl, step_trace.jsonl, or JSONL with messages")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--experience-path", default=str(DEFAULT_EXPERIENCE_PATH))
    parser.add_argument("--experience-limit", type=int, default=4)
    parser.add_argument(
        "--only-actionable",
        action="store_true",
        help="Skip forced/no-alternative rows that cannot improve through local reselect.",
    )
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


def _strip_thinking(text: str) -> str:
    if not text:
        return ""
    if _THINK_END_RE.search(text):
        return _THINK_END_RE.split(text)[-1].strip()
    return _THINK_BLOCK_RE.sub("", text).strip()


def _parse_reason(raw_text: str) -> str:
    stripped = _strip_thinking(raw_text).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("reason") or "")


def _user_message(row: dict[str, Any]) -> str:
    if isinstance(row.get("user_message"), str):
        return str(row["user_message"])
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _original_index(row: dict[str, Any], actions: list[dict[str, Any]]) -> int | None:
    chosen = row.get("chosen") if isinstance(row.get("chosen"), dict) else {}
    raw = chosen.get("index")
    if isinstance(raw, int):
        return raw
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    raw = decoded.get("action_index")
    if isinstance(raw, int):
        return raw
    raw = row.get("action_index")
    if isinstance(raw, int):
        return raw
    if actions:
        chosen_action = _chosen(row, actions)
        raw = chosen_action.get("index")
        return raw if isinstance(raw, int) else None
    return None


def _original_reason(row: dict[str, Any]) -> str:
    if isinstance(row.get("reason"), str):
        return str(row["reason"])
    decoded = row.get("decoded") if isinstance(row.get("decoded"), dict) else {}
    if isinstance(decoded.get("reason"), str):
        return str(decoded["reason"])
    return _parse_reason(str(row.get("raw_generation") or ""))


def _flags(row: dict[str, Any]) -> list[str]:
    raw = row.get("flags")
    if isinstance(raw, list):
        return [str(flag) for flag in raw]
    raw = row.get("quality_flags")
    if isinstance(raw, list):
        return [str(flag) for flag in raw]
    return []


def _action_by_index(actions: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    for action in actions:
        if action.get("index") == index:
            return action
    return None


def _is_end_turn(action: dict[str, Any] | None) -> bool:
    if not action:
        return False
    return str(action.get("card_id") or "").lower() == "end_turn" or "end_turn" in str(action.get("raw") or "").lower()


def _enemy_hp_plus_block(user_message: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, enemy in _enemies(user_message).items():
        out[key] = float(enemy.get("hp") or 0) + float(enemy.get("block") or 0)
    return out


def _is_lethal(action: dict[str, Any] | None, hp_block: dict[str, float]) -> bool:
    if not action:
        return False
    target = str(action.get("target") or "")
    damage = action.get("damage")
    return target in hp_block and isinstance(damage, int) and float(damage) >= hp_block[target] > 0


def _visible_lethal_actions(actions: list[dict[str, Any]], user_message: str) -> list[int]:
    hp_block = _enemy_hp_plus_block(user_message)
    return [
        int(action["index"])
        for action in actions
        if isinstance(action.get("index"), int) and _is_lethal(action, hp_block)
    ]


def _reason_math_ok(reason: str) -> bool:
    for match in _MATH_GE_RE.finditer(reason or ""):
        try:
            if float(match.group(1)) + 1e-9 < float(match.group(2)):
                return False
        except ValueError:
            continue
    return True


def _bash_opportunity(actions: list[dict[str, Any]], user_message: str) -> bool:
    energy = _energy(user_message)
    if energy is not None and energy < 3:
        return False
    has_bash = any(str(action.get("card_id") or "") == "BASH" and isinstance(action.get("damage"), int) for action in actions)
    has_other_attack = any(
        str(action.get("card_id") or "") not in {"", "BASH", "end_turn"}
        and isinstance(action.get("damage"), int)
        and int(action.get("damage") or 0) > 0
        for action in actions
    )
    return has_bash and has_other_attack


def _bash_score(action: dict[str, Any] | None, user_message: str) -> float:
    if not action:
        return 0.0
    if str(action.get("card_id") or "") == "BASH":
        return 1.0
    target = str(action.get("target") or "")
    target_enemy = _enemies(user_message).get(target)
    if isinstance(action.get("damage"), int) and target_enemy and not target_enemy.get("vulnerable"):
        return -1.0
    return 0.0


def score_choice(
    *,
    row: dict[str, Any],
    action_index: int | None,
    reason: str,
    actions: list[dict[str, Any]],
    user_message: str,
) -> dict[str, Any]:
    action = _action_by_index(actions, action_index)
    flags = _flags(row)
    score = 0.0
    notes: list[str] = []
    if action is None:
        return {"score": -5.0, "notes": ["invalid_action_index"], "action": None}
    score += 1.0
    hp_block = _enemy_hp_plus_block(user_message)
    lethal_actions = _visible_lethal_actions(actions, user_message)
    if lethal_actions:
        if _is_lethal(action, hp_block):
            score += 2.0
            notes.append("takes_visible_lethal")
        else:
            score -= 2.0
            notes.append("misses_visible_lethal")
    play_actions = [a for a in actions if not _is_end_turn(a)]
    if "dangerous_end_turn" in flags and play_actions:
        if _is_end_turn(action):
            score -= 1.0
            notes.append("keeps_dangerous_end_turn")
        else:
            score += 1.0
            notes.append("avoids_dangerous_end_turn")
    if _bash_opportunity(actions, user_message):
        delta = _bash_score(action, user_message)
        score += delta
        if delta > 0:
            notes.append("takes_bash_first")
        elif delta < 0:
            notes.append("attacks_before_bash")
    if "reason_math_contradiction" in flags:
        if _reason_math_ok(reason):
            score += 0.5
            notes.append("fixes_reason_math")
        else:
            score -= 0.5
            notes.append("keeps_reason_math_contradiction")
    return {
        "score": round(score, 4),
        "notes": notes,
        "action": action,
        "visible_lethal_actions": lethal_actions,
    }


def is_actionable_review_row(row: dict[str, Any]) -> bool:
    """Whether the row can plausibly improve without executing a rollout."""

    user_message = _user_message(row)
    actions = _legal_actions({"user_message": user_message})
    if not actions:
        return False

    old_index = _original_index(row, actions)
    old_action = _action_by_index(actions, old_index)
    flags = set(_flags(row))
    play_actions = [action for action in actions if not _is_end_turn(action)]

    if old_action is None and actions:
        return True
    if "reason_math_contradiction" in flags:
        return True
    if "missed_visible_lethal" in flags and _visible_lethal_actions(actions, user_message):
        return True
    if "dangerous_end_turn" in flags and _is_end_turn(old_action) and play_actions:
        return True
    if _bash_opportunity(actions, user_message) and str((old_action or {}).get("card_id") or "") != "BASH":
        return True
    return False


def _review_hints(
    *,
    row: dict[str, Any],
    actions: list[dict[str, Any]],
    user_message: str,
    old_score: dict[str, Any],
) -> str:
    flags = set(_flags(row))
    hints: list[str] = []
    lethal_actions = old_score.get("visible_lethal_actions") or []
    if lethal_actions:
        lethal_lines = []
        for index in lethal_actions:
            action = _action_by_index(actions, int(index))
            if action:
                lethal_lines.append(str(action.get("raw") or f"[{index}]"))
        hints.append("  visible_lethal_actions: " + "; ".join(lethal_lines))
    if "missed_visible_lethal" in flags:
        hints.append("  issue: original action missed a currently visible lethal action.")
    if "reason_math_contradiction" in flags:
        hints.append("  issue: original reason contains false arithmetic; recompute damage, block, and HP before answering.")
    if "dangerous_end_turn" in flags:
        play_actions = [action for action in actions if not _is_end_turn(action)]
        if play_actions:
            hints.append("  issue: original ended turn while useful play_card actions were still legal.")
        else:
            hints.append("  note: end_turn is forced here; keep it if no legal play_card action exists.")
    if _bash_opportunity(actions, user_message):
        hints.append("  tactic: BASH applies Vulnerable; prefer BASH before nonlethal follow-up attacks when energy supports it.")
    if not hints:
        return ""
    return "review_hints:\n" + "\n".join(hints) + "\n"


def _prompt(
    row: dict[str, Any],
    experience_block: str,
    *,
    actions: list[dict[str, Any]],
    old_score: dict[str, Any],
) -> list[dict[str, str]]:
    flags = ", ".join(_flags(row)) or "none"
    original_index = row.get("chosen", {}).get("index") if isinstance(row.get("chosen"), dict) else row.get("action_index")
    original_reason = _original_reason(row) or str(row.get("raw_generation") or "")[:180]
    user = _user_message(row)
    review = (
        "review_task:\n"
        "  Reconsider the exact same state. Do not assume any hidden rollout or future draw.\n"
        "  Choose a better legal action only from listed action_index values.\n"
        "  If the original action is still best, keep it but fix the reason.\n"
        f"  original_action_index={original_index}\n"
        f"  original_reason={original_reason}\n"
        f"  detected_flags={flags}\n"
    )
    if experience_block:
        review += experience_block + "\n"
    review_hints = _review_hints(row=row, actions=actions, user_message=user, old_score=old_score)
    if review_hints:
        review += review_hints
    review += (
        "\ncurrent_state:\n"
        f"{user}\n\n"
        'Return one JSON object: {"action_index": N, "reason": "short concrete reason"}'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a Slay the Spire 2 post-turn reviewer. "
                "Your job is to fix the local decision for one already observed state. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": review},
    ]


class ReselectModel:
    def __init__(self, args: argparse.Namespace) -> None:
        from unsloth import FastLanguageModel

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


def _default_out_dir(input_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ARTIFACTS_ROOT / "reviews" / f"reselect_{input_path.stem}_{stamp}"


def main() -> int:
    args = parse_args()
    ensure_dirs()
    input_path = Path(args.input_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_out_dir(input_path)
    rows = _read_jsonl(input_path)
    input_rows = len(rows)
    if args.only_actionable:
        rows = [row for row in rows if is_actionable_review_row(row)]
    filtered_rows = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    experience_entries = load_experience(Path(args.experience_path))

    manifest = {
        "kind": "review_reselect_actions",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "input_jsonl": str(input_path),
        "out_dir": str(out_dir),
        "adapter_dir": args.adapter_dir,
        "limit": args.limit,
        "temperature": args.temperature,
        "experience_path": args.experience_path,
        "experience_limit": args.experience_limit,
        "only_actionable": args.only_actionable,
        "input_rows": input_rows,
        "filtered_rows": filtered_rows,
        "dry_run": args.dry_run,
    }
    _write_json(out_dir / "manifest.json", manifest)

    model = None if args.dry_run else ReselectModel(args)
    results: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        user = _user_message(row)
        actions = _legal_actions({"user_message": user})
        old_index = _original_index(row, actions)
        old_reason = _original_reason(row)
        old_score = score_choice(row=row, action_index=old_index, reason=old_reason, actions=actions, user_message=user)
        experience = retrieve_experience(user, experience_entries, limit=max(0, args.experience_limit))
        messages = _prompt(row, render_experience_block(experience), actions=actions, old_score=old_score)
        if args.dry_run:
            raw = ""
            new_index = old_index
            status = "dry_run"
            new_reason = old_reason
            gen_ms = 0.0
        else:
            assert model is not None
            raw, gen_ms = model.run(messages)
            new_index, status = parse_action_index(raw)
            new_reason = _parse_reason(raw)
        new_score = score_choice(row=row, action_index=new_index, reason=new_reason, actions=actions, user_message=user)
        delta = float(new_score["score"]) - float(old_score["score"])
        results.append({
            "row_index": row_index,
            "episode_id": row.get("episode_id"),
            "step": row.get("step", row.get("episode_step")),
            "flags": _flags(row),
            "old": {
                "action_index": old_index,
                "reason": old_reason,
                "score": old_score,
            },
            "new": {
                "action_index": new_index,
                "status": status,
                "reason": new_reason,
                "raw_generation": raw,
                "gen_ms": round(gen_ms, 1),
                "score": new_score,
            },
            "score_delta": round(delta, 4),
            "improved": delta > 1e-9,
            "same_action": old_index == new_index,
            "experience": [entry.to_json() for entry in experience],
            "prompt_messages": messages if args.dry_run else None,
        })

    with (out_dir / "reselect_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    status_counts = Counter(str((result.get("new") or {}).get("status") or "") for result in results)
    note_counts = Counter()
    for result in results:
        note_counts.update(((result.get("new") or {}).get("score") or {}).get("notes") or [])
    valid = sum(1 for result in results if ((result.get("new") or {}).get("score") or {}).get("action") is not None)
    improved = sum(1 for result in results if result.get("improved"))
    worsened = sum(1 for result in results if float(result.get("score_delta") or 0.0) < -1e-9)
    changed = sum(1 for result in results if not result.get("same_action"))
    summary = {
        **manifest,
        "cases": len(results),
        "valid_new_actions": valid,
        "valid_new_action_rate": round(valid / len(results), 4) if results else None,
        "changed_actions": changed,
        "changed_action_rate": round(changed / len(results), 4) if results else None,
        "improved_cases": improved,
        "improved_rate": round(improved / len(results), 4) if results else None,
        "worsened_cases": worsened,
        "worsened_rate": round(worsened / len(results), 4) if results else None,
        "avg_score_delta": round(sum(float(result.get("score_delta") or 0.0) for result in results) / len(results), 4) if results else 0.0,
        "status_counts": {key: int(value) for key, value in status_counts.most_common()},
        "new_note_counts": {key: int(value) for key, value in note_counts.most_common()},
        "outputs": {
            "results": str(out_dir / "reselect_results.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
