"""手工捏的 toy 数据集，用来让训练管道先跑通。

真正的数据来自 `rollout_teacher.py`（下一阶段）。这里只是占位，
确保 `sft_lora.py` 能读到格式正确的 messages jsonl。

约定：
- 输出到 STS2AI/Artifacts/llm/datasets/toy/{train,eval}.jsonl
- messages 格式 = [{role, content}...]
- assistant 必须严格是一行 JSON {"action_index": int, "reason": str}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.data_pipeline.state_renderer import render_state_text
from llm.paths import DATASETS_ROOT, ensure_dirs
from llm.prompts import load_system_prompt


def _state_a() -> tuple[dict[str, Any], list[dict[str, Any]], int, str]:
    """Act1 遇到 Cultist，3 能量，有 BASH —— 答案是先 BASH 上脆弱。"""
    state = {
        "state_type": "monster",
        "run": {"character": "IRONCLAD", "floor_reached": 3, "gold": 135},
        "player": {"hp": 68, "max_hp": 80, "gold": 135},
        "battle": {
            "encounter_id": "CULTIST",
            "turn": "player",
            "is_play_phase": True,
            "energy": 3,
            "max_energy": 3,
            "player": {"block": 0, "powers": []},
            "hand": [
                {"card_id": "STRIKE_RED", "cost_now": 1, "damage_now": 6, "can_play": True},
                {"card_id": "DEFEND_RED", "cost_now": 1, "block_now": 5, "can_play": True},
                {"card_id": "BASH", "cost_now": 2, "damage_now": 8, "can_play": True},
                {"card_id": "STRIKE_RED", "cost_now": 1, "damage_now": 6, "can_play": True},
                {"card_id": "STRIKE_RED", "cost_now": 1, "damage_now": 6, "can_play": True},
            ],
            "draw_pile_cards": [{}] * 5,
            "discard_pile_cards": [],
            "exhaust_pile_cards": [],
        },
        "enemies": [
            {"monster_id": "CULTIST", "hp": 50, "max_hp": 50, "block": 0, "intent_type": "ritual"}
        ],
    }
    legal = [
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 0, "target_id": "CULTIST_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "DEFEND_RED", "hand_index": 1, "is_enabled": True},
        {"action_type": "play_card", "card_id": "BASH",       "hand_index": 2, "target_id": "CULTIST_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 3, "target_id": "CULTIST_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 4, "target_id": "CULTIST_0", "is_enabled": True},
        {"action_type": "end_turn",  "is_enabled": True},
    ]
    return state, legal, 2, "先用 BASH 给 Cultist 挂脆弱，后手 Strike 伤害放大 50%"


def _state_b() -> tuple[dict[str, Any], list[dict[str, Any]], int, str]:
    """低血量要挡，没 Bash，只能 Defend + Strike。"""
    state = {
        "state_type": "monster",
        "run": {"character": "IRONCLAD", "floor_reached": 5, "gold": 60},
        "player": {"hp": 22, "max_hp": 80},
        "battle": {
            "encounter_id": "JAW_WORM",
            "turn": "player",
            "is_play_phase": True,
            "energy": 3,
            "max_energy": 3,
            "player": {"block": 0, "powers": []},
            "hand": [
                {"card_id": "STRIKE_RED", "cost_now": 1, "damage_now": 6, "can_play": True},
                {"card_id": "DEFEND_RED", "cost_now": 1, "block_now": 5, "can_play": True},
                {"card_id": "DEFEND_RED", "cost_now": 1, "block_now": 5, "can_play": True},
                {"card_id": "STRIKE_RED", "cost_now": 1, "damage_now": 6, "can_play": True},
                {"card_id": "DEFEND_RED", "cost_now": 1, "block_now": 5, "can_play": True},
            ],
            "draw_pile_cards": [{}] * 5,
            "discard_pile_cards": [],
            "exhaust_pile_cards": [],
        },
        "enemies": [
            {"monster_id": "JAW_WORM", "hp": 40, "max_hp": 42, "block": 0, "intent_type": "attack", "move_base_damage": 11}
        ],
    }
    legal = [
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 0, "target_id": "JAW_WORM_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "DEFEND_RED", "hand_index": 1, "is_enabled": True},
        {"action_type": "play_card", "card_id": "DEFEND_RED", "hand_index": 2, "is_enabled": True},
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 3, "target_id": "JAW_WORM_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "DEFEND_RED", "hand_index": 4, "is_enabled": True},
        {"action_type": "end_turn",  "is_enabled": True},
    ]
    return state, legal, 1, "血线 22 扛不住 11 攻击，优先堆挡活下来"


def _state_c() -> tuple[dict[str, Any], list[dict[str, Any]], int, str]:
    """满血打小史莱姆，直接打出去省能耗。"""
    state = {
        "state_type": "monster",
        "run": {"character": "IRONCLAD", "floor_reached": 2, "gold": 99},
        "player": {"hp": 80, "max_hp": 80},
        "battle": {
            "encounter_id": "SMALL_SLIMES",
            "turn": "player",
            "is_play_phase": True,
            "energy": 3,
            "max_energy": 3,
            "player": {"block": 0, "powers": []},
            "hand": [
                {"card_id": "STRIKE_RED", "cost_now": 1, "damage_now": 6, "can_play": True},
                {"card_id": "STRIKE_RED", "cost_now": 1, "damage_now": 6, "can_play": True},
                {"card_id": "BASH",       "cost_now": 2, "damage_now": 8, "can_play": True},
                {"card_id": "DEFEND_RED", "cost_now": 1, "block_now": 5, "can_play": True},
                {"card_id": "DEFEND_RED", "cost_now": 1, "block_now": 5, "can_play": True},
            ],
            "draw_pile_cards": [{}] * 5,
            "discard_pile_cards": [],
            "exhaust_pile_cards": [],
        },
        "enemies": [
            {"monster_id": "ACID_SLIME_S", "hp": 8, "max_hp": 12, "intent_type": "attack"}
        ],
    }
    legal = [
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 0, "target_id": "ACID_SLIME_S_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 1, "target_id": "ACID_SLIME_S_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "BASH",       "hand_index": 2, "target_id": "ACID_SLIME_S_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "DEFEND_RED", "hand_index": 3, "is_enabled": True},
        {"action_type": "play_card", "card_id": "DEFEND_RED", "hand_index": 4, "is_enabled": True},
        {"action_type": "end_turn",  "is_enabled": True},
    ]
    return state, legal, 0, "Strike 6 伤就能秒小史莱姆，BASH 过量浪费 1 能量"


def _state_d() -> tuple[dict[str, Any], list[dict[str, Any]], int, str]:
    """能量已空，只能 end_turn。"""
    state = {
        "state_type": "monster",
        "run": {"character": "IRONCLAD", "floor_reached": 4, "gold": 120},
        "player": {"hp": 55, "max_hp": 80},
        "battle": {
            "encounter_id": "LOUSE_PAIR",
            "turn": "player",
            "is_play_phase": True,
            "energy": 0,
            "max_energy": 3,
            "player": {"block": 10, "powers": []},
            "hand": [
                {"card_id": "STRIKE_RED", "cost_now": 1, "damage_now": 6, "can_play": False, "unplayable_reason": "NotEnoughEnergy"},
                {"card_id": "DEFEND_RED", "cost_now": 1, "block_now": 5, "can_play": False, "unplayable_reason": "NotEnoughEnergy"},
            ],
            "draw_pile_cards": [{}] * 3,
            "discard_pile_cards": [{}] * 5,
            "exhaust_pile_cards": [],
        },
        "enemies": [
            {"monster_id": "LOUSE_RED", "hp": 11, "max_hp": 14, "block": 0, "intent_type": "attack"}
        ],
    }
    legal = [
        {"action_type": "end_turn", "is_enabled": True},
    ]
    return state, legal, 0, "手牌都点不起，只能结束回合"


def _state_e() -> tuple[dict[str, Any], list[dict[str, Any]], int, str]:
    """打精英 Sentries，前期需要挂脆弱堆 block。"""
    state = {
        "state_type": "elite",
        "run": {"character": "IRONCLAD", "floor_reached": 7, "gold": 200},
        "player": {"hp": 45, "max_hp": 80},
        "battle": {
            "encounter_id": "SENTRIES",
            "turn": "player",
            "is_play_phase": True,
            "energy": 3,
            "max_energy": 3,
            "player": {"block": 0, "powers": []},
            "hand": [
                {"card_id": "BASH",         "cost_now": 2, "damage_now": 8, "can_play": True},
                {"card_id": "DEFEND_RED",   "cost_now": 1, "block_now": 5, "can_play": True},
                {"card_id": "STRIKE_RED",   "cost_now": 1, "damage_now": 6, "can_play": True},
                {"card_id": "IRON_WAVE",    "cost_now": 1, "damage_now": 5, "block_now": 5, "can_play": True},
                {"card_id": "ANGER",        "cost_now": 0, "damage_now": 6, "can_play": True},
            ],
            "draw_pile_cards": [{}] * 5,
            "discard_pile_cards": [],
            "exhaust_pile_cards": [],
        },
        "enemies": [
            {"monster_id": "SENTRY", "hp": 38, "max_hp": 38, "intent_type": "attack", "move_base_damage": 9},
            {"monster_id": "SENTRY", "hp": 38, "max_hp": 38, "intent_type": "debuff"},
            {"monster_id": "SENTRY", "hp": 38, "max_hp": 38, "intent_type": "attack", "move_base_damage": 9},
        ],
    }
    legal = [
        {"action_type": "play_card", "card_id": "BASH",       "hand_index": 0, "target_id": "SENTRY_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "BASH",       "hand_index": 0, "target_id": "SENTRY_1", "is_enabled": True},
        {"action_type": "play_card", "card_id": "BASH",       "hand_index": 0, "target_id": "SENTRY_2", "is_enabled": True},
        {"action_type": "play_card", "card_id": "DEFEND_RED", "hand_index": 1, "is_enabled": True},
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 2, "target_id": "SENTRY_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 2, "target_id": "SENTRY_1", "is_enabled": True},
        {"action_type": "play_card", "card_id": "STRIKE_RED", "hand_index": 2, "target_id": "SENTRY_2", "is_enabled": True},
        {"action_type": "play_card", "card_id": "IRON_WAVE",  "hand_index": 3, "target_id": "SENTRY_0", "is_enabled": True},
        {"action_type": "play_card", "card_id": "ANGER",      "hand_index": 4, "target_id": "SENTRY_0", "is_enabled": True},
        {"action_type": "end_turn",  "is_enabled": True},
    ]
    return state, legal, 0, "BASH 打攻击型 Sentry_0，配合 Anger/Strike 一回合点杀拿 tempo"


_CASES = [_state_a, _state_b, _state_c, _state_d, _state_e]


def build_sample(case_fn) -> dict[str, Any]:
    state, legal, chosen, reason = case_fn()
    user_msg = render_state_text(state, legal)
    assistant_msg = json.dumps({"action_index": chosen, "reason": reason}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": load_system_prompt()},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "meta": {
            "source": "toy_hand_crafted",
            "chosen_action": legal[chosen],
        },
    }


def main(out_dir: Path | None = None) -> None:
    ensure_dirs()
    target = out_dir or (DATASETS_ROOT / "toy")
    target.mkdir(parents=True, exist_ok=True)

    train = [build_sample(case) for case in _CASES]
    # eval 用训练集里的子集 + 一个外部变体，占位
    eval_set = [build_sample(_CASES[0]), build_sample(_CASES[2])]

    train_path = target / "train.jsonl"
    eval_path = target / "eval.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as f:
        for item in eval_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[toy] wrote {len(train)} train -> {train_path}")
    print(f"[toy] wrote {len(eval_set)} eval  -> {eval_path}")
    print("\n[toy] sample[0] user message:")
    print("-" * 60)
    print(train[0]["messages"][1]["content"])
    print("-" * 60)
    print("[toy] sample[0] assistant message:")
    print(train[0]["messages"][2]["content"])


if __name__ == "__main__":
    main()
