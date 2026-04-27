from __future__ import annotations

import json

from llm.prompts import load_system_prompt
from llm.scripts.build_teacher_dataset import (
    _candidate_from_trace_row,
    _dedupe_rows,
    _reason_repair_from_trace_row,
    _rows_from_kimi_labels,
    _rows_from_review,
    _sample,
)


def test_trace_rule_prefers_attacking_lethal_target() -> None:
    row = {
        "user_message": (
            "run: char=IRONCLAD round=3\n"
            "player: hp=33/80 block=0 energy=3/3 powers=-\n"
            "enemies:\n"
            "  enemy2: BOWLBUG_SILK hp=42/42 block=0 intent=Debuff powers=-\n"
            "  enemy3: BOWLBUG_EGG hp=21/21 block=7 intent=Attack(7) powers=-\n"
            "hand:\n"
            "  [3] BLUDGEON cost=3 type=attack upgraded=true | Deal 45 damage.\n"
            "legal_actions:\n"
            "  BLUDGEON hand[3]:\n"
            "    [4] target=enemy2 damage=45\n"
            "    [5] target=enemy3 damage=45\n"
            "  [8] end_turn\n"
        ),
        "decoded": {"action_index": 4, "reason": "lethal BOWLBUG_SILK(45>=42)"},
    }

    candidate = _candidate_from_trace_row(row)

    assert candidate is not None
    assert candidate["action_index"] == 5
    assert candidate["rule"] == "attacking_lethal_priority"


def test_trace_rule_prefers_cheap_draw_lethal_over_bludgeon() -> None:
    row = {
        "user_message": (
            "run: char=IRONCLAD round=2\n"
            "player: hp=48/80 block=0 energy=3/3 powers=WEAK_POWER=1\n"
            "enemies:\n"
            "  enemy1: BOWLBUG_ROCK hp=7/46 block=0 intent=Attack(15) powers=-\n"
            "hand:\n"
            "  [0] POMMEL_STRIKE cost=1 type=attack upgraded=true | Deal 9 damage. Draw 2 cards.\n"
            "  [3] BLUDGEON cost=3 type=attack upgraded=true | Deal 33 damage.\n"
            "legal_actions:\n"
            "  POMMEL_STRIKE hand[0]:\n"
            "    [0] target=enemy1 damage=9\n"
            "  BLUDGEON hand[3]:\n"
            "    [7] target=enemy1 damage=33\n"
            "  [13] end_turn\n"
        ),
        "decoded": {"action_index": 7, "reason": "lethal BOWLBUG_ROCK(33>=7)"},
    }

    candidate = _candidate_from_trace_row(row)

    assert candidate is not None
    assert candidate["action_index"] == 0
    assert candidate["rule"] == "cheap_lethal_over_overkill"


def test_reason_repair_keeps_action_but_rewrites_bad_lethal_reason() -> None:
    row = {
        "user_message": (
            "run: char=IRONCLAD round=3\n"
            "player: hp=69/80 block=0 energy=3/3 powers=-\n"
            "enemies:\n"
            "  enemy2: DAMP_CULTIST hp=51/51 block=0 intent=Attack(6) powers=-\n"
            "hand:\n"
            "  [3] BLUDGEON cost=3 type=attack upgraded=true | Deal 45 damage.\n"
            "legal_actions:\n"
            "  BLUDGEON hand[3]:\n"
            "    [3] target=enemy2 damage=45\n"
            "  [5] end_turn\n"
        ),
        "attempts": [
            {
                "decoded": {
                    "action_index": 3,
                    "reason": "lethal DAMP_CULTIST(45>=51)",
                    "used_fallback": True,
                    "fallback_reason": "reason_claims_lethal_but_action_not_lethal,reason_math_contradiction",
                }
            }
        ],
    }

    repair = _reason_repair_from_trace_row(row)

    assert repair is not None
    assert repair["action_index"] == 3
    assert repair["reason"] == "deal 45 damage to enemy2"
    assert repair["rule"] == "reason_consistency_repair"


def test_reason_repair_catches_action_score_math_contradiction_without_invalid_attempt() -> None:
    row = {
        "user_message": (
            "run: char=IRONCLAD round=2\n"
            "player: hp=42/80 block=0 energy=3/3 powers=-\n"
            "enemies:\n"
            "  enemy1: BOWLBUG_ROCK hp=46/46 block=0 intent=Attack(15) powers=-\n"
            "hand:\n"
            "  [3] BLUDGEON cost=3 type=attack upgraded=true | Deal 45 damage.\n"
            "legal_actions:\n"
            "  BLUDGEON hand[3]:\n"
            "    [2] target=enemy1 damage=45\n"
            "  [5] end_turn\n"
        ),
        "state": {
            "battle": {
                "energy": 3,
                "player": {"block": 0},
                "hand": [{"id": "X"}, {"id": "Y"}, {"id": "Z"}, {"id": "BLUDGEON", "preview_damage_per_target": {"1": 45}}],
            },
            "enemies": [
                {
                    "target_id": 1,
                    "hp": 46,
                    "block": 0,
                    "intent_type": "Attack",
                    "intent_damage": 15,
                    "intent_hits": 1,
                    "is_alive": True,
                }
            ],
        },
        "legal_actions": [
            {"action": "play_card", "card_index": 0, "target_id": 1},
            {"action": "play_card", "card_index": 1, "target_id": 1},
            {"action": "play_card", "card_index": 3, "target_id": 1},
        ],
        "decoded": {
            "action_index": 2,
            "reason": "BLUDGEON deals damage.",
            "action_scores": [
                {"action_index": 2, "score": 9.0, "note": "lethal BOWLBUG_ROCK: damage=45 target_hp=46"}
            ],
        },
    }

    repair = _reason_repair_from_trace_row(row)

    assert repair is not None
    assert repair["rule"] == "explanation_consistency_repair"
    assert "action_score_lethal_math_contradiction" in repair["fallback_reason"]


def test_rows_from_review_validates_teacher_labels(tmp_path) -> None:
    episode = {
        "episode_id": "ep",
        "turns": [
            {
                "round": 1,
                "decisions": [
                    {
                        "step": 0,
                        "chosen_action_index": 1,
                        "reason": "end turn",
                        "pre_decision_state": (
                            "run: char=IRONCLAD round=1\n"
                            "player: hp=80/80 block=0 energy=1/3 powers=-\n"
                            "enemies:\n"
                            "  enemy1: CULTIST hp=3/30 block=0 intent=Attack(6) powers=-\n"
                            "hand:\n"
                            "  [0] STRIKE_IRONCLAD cost=1 type=attack | Deal 6 damage.\n"
                            "legal_actions:\n"
                            "  STRIKE_IRONCLAD hand[0]:\n"
                            "    [0] target=enemy1 damage=6\n"
                            "  [1] end_turn\n"
                        ),
                    }
                ],
            }
        ],
    }
    review = {
        "episode_id": "ep",
        "usable_training_labels": [
            {"step": 0, "best_action_index": 0, "reason_en": "take lethal", "confidence": 0.9},
            {"step": 0, "best_action_index": 99, "reason_en": "bad index", "confidence": 1.0},
        ],
        "key_lessons": [{"tags": ["lethal"], "lesson_zh": "kill", "training_reason_en": "take lethal"}],
    }
    episode_path = tmp_path / "episode_input.json"
    review_path = tmp_path / "review.json"
    episode_path.write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
    review_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")

    rows, lessons = _rows_from_review(review_path, episode_path, system_prompt=load_system_prompt(), min_confidence=0.7)

    assert len(rows) == 1
    assert json.loads(rows[0]["messages"][-1]["content"])["action_index"] == 0
    assert len(lessons) == 1


def test_dedupe_rows_keeps_single_user_action_pair() -> None:
    first = _sample(
        user_message="legal_actions:\n  [0] A\n",
        action_index=0,
        reason="x",
        meta={"confidence": 0.7},
        system_prompt="sys",
    )
    second = _sample(
        user_message="legal_actions:\n  [0] A\n",
        action_index=0,
        reason="y",
        meta={"confidence": 0.9},
        system_prompt="sys",
    )

    rows = _dedupe_rows([first, second])

    assert len(rows) == 1
    assert rows[0]["meta"]["teacher_reason"] == "y"


def test_sample_writes_current_scored_output_schema() -> None:
    row = _sample(
        user_message=(
            "run: char=IRONCLAD round=1\n"
            "player: hp=80/80 block=0 energy=1/3 powers=-\n"
            "enemies:\n"
            "  enemy1: CULTIST hp=3/30 block=0 intent=Attack(6) powers=-\n"
            "hand:\n"
            "  [0] STRIKE_IRONCLAD cost=1 type=attack | Deal 6 damage.\n"
            "legal_actions:\n"
            "  STRIKE_IRONCLAD hand[0]:\n"
            "    [0] target=enemy1 damage=6\n"
            "  [1] end_turn\n"
        ),
        action_index=0,
        reason="verified lethal on enemy1",
        meta={"confidence": 0.8},
        system_prompt="sys",
    )

    payload = json.loads(row["messages"][-1]["content"])

    assert payload["action_index"] == 0
    assert payload["confidence"] == 0.8
    assert "action_scores" not in payload
    user = row["messages"][1]["content"]
    assert '"confidence":0.0' in user
    assert '"action_scores"' not in user


def test_rows_from_kimi_labels_writes_trainable_rows(tmp_path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({
        "candidate_id": "c1",
        "best_action_index": 0,
        "confidence": 0.9,
        "reason_en": "take lethal",
        "mechanism_tags": ["visible_lethal"],
        "original_action_index": 1,
        "source": {"episode_id": "ep", "episode_step": 0},
        "user_message": (
            "run: char=IRONCLAD round=1\n"
            "player: hp=80/80 block=0 energy=1/3 powers=-\n"
            "enemies:\n"
            "  enemy1: CULTIST hp=3/30 block=0 intent=Attack(6) powers=-\n"
            "hand:\n"
            "  [0] STRIKE_IRONCLAD cost=1 type=attack | Deal 6 damage.\n"
            "legal_actions:\n"
            "  STRIKE_IRONCLAD hand[0]:\n"
            "    [0] target=enemy1 damage=6\n"
            "  [1] end_turn\n"
        ),
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = _rows_from_kimi_labels(
        labels,
        system_prompt="sys",
        min_confidence=0.7,
        keep_kimi_reasons=False,
    )

    assert len(rows) == 1
    assert rows[0]["meta"]["source"] == "kimi_teacher_label"
    assistant = json.loads(rows[0]["messages"][-1]["content"])
    assert assistant["action_index"] == 0
    assert assistant["reason"] == "verified lethal on enemy1"
    assert rows[0]["meta"]["reason_source"] == "canonical_verified"
