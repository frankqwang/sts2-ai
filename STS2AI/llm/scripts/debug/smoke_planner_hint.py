"""Smoke-test a planner-hint adapter on one seed state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.planner_hint import (  # noqa: E402
    format_planner_hint,
    parse_planner_hint_json,
    render_planner_hint_user_message,
)
from llm.paths import BASE_MODEL_ID  # noqa: E402
from llm.prompts import load_system_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def _state() -> dict[str, Any]:
    return {
        "run": {"act": 1, "floor": 4, "gold": 99},
        "player": {
            "character": "IRONCLAD",
            "hp": 62,
            "max_hp": 80,
            "deck": [{"id": "STRIKE_IRONCLAD"}] * 5 + [{"id": "DEFEND_IRONCLAD"}] * 4 + [{"id": "BASH"}],
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
        "battle": {
            "encounter_id": "CULTISTS_NORMAL",
            "round_number_raw": 1,
            "energy": 3,
            "player": {"block": 0},
            "hand": [
                {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "attack", "description": "Deal 6 damage.", "preview_damage_per_target": {"1": 6, "2": 6}},
                {"id": "DEFEND_IRONCLAD", "cost": 1, "type": "skill", "description": "Gain 5 Block."},
                {"id": "BASH", "cost": 2, "type": "attack", "description": "Deal 8 damage. Apply 2 Vulnerable.", "preview_damage_per_target": {"1": 8, "2": 8}},
            ],
        },
        "enemies": [
            {"target_id": 1, "monster_id": "CALCIFIED_CULTIST", "hp": 39, "max_hp": 39, "block": 0, "intent_type": "Buff", "is_alive": True},
            {"target_id": 2, "monster_id": "DAMP_CULTIST", "hp": 51, "max_hp": 51, "block": 0, "intent_type": "Buff", "is_alive": True},
        ],
    }


def main() -> int:
    args = parse_args()
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    FastLanguageModel.for_inference(model)

    user_msg = render_planner_hint_user_message(_state(), require_knowledge=True)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": load_system_prompt("planner_hint")},
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    import torch

    inputs = tokenizer(prompt + "{", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = ("{" + tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
    hint, status = parse_planner_hint_json(raw)
    print(json.dumps({
        "status": status,
        "raw": raw,
        "formatted": format_planner_hint(hint or {}),
        "user_message": user_msg,
    }, ensure_ascii=False, indent=2))
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
