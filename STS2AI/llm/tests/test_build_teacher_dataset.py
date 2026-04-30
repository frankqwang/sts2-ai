from __future__ import annotations

import json

from llm.prompts import load_system_prompt
from llm.scripts.datasets.build_teacher_dataset import (
    _candidate_from_trace_row,
    _coerce_confidence,
    _coerce_tag_list,
    _dedupe_rows,
    _reason_repair_from_trace_row,
    _review_pairs_from_roots,
    _rows_from_kimi_labels,
    _rows_from_review,
    _sample,
)


def test_coerce_confidence_rejects_bool():
    """LLM 偶尔回 True/False；bool 是 int 子类，float(True)=1.0 会被错当成 100% 置信。"""
    assert _coerce_confidence(True) == 0.0
    assert _coerce_confidence(False) == 0.0


def test_coerce_confidence_handles_str_and_none():
    assert _coerce_confidence(None) == 0.0
    assert _coerce_confidence("0.8") == 0.8
    assert _coerce_confidence("not_a_number") == 0.0
    assert _coerce_confidence(0.75) == 0.75
    assert _coerce_confidence(1) == 1.0


def test_coerce_tag_list_handles_string_not_iterating_chars():
    """tags 字段防御：LLM 偶尔回 'lethal'（不是 list），不能按字符拆开。"""
    assert _coerce_tag_list("lethal") == ["lethal"]
    assert _coerce_tag_list(["lethal", "incoming_damage"]) == ["lethal", "incoming_damage"]
    assert _coerce_tag_list(None) == []
    assert _coerce_tag_list("") == []
    assert _coerce_tag_list([1, "x", None, ""]) == ["1", "x"]


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

    # use_kimi_reasons=False 显式声明：测试 canonical reason 路径（向后兼容）。
    rows, lessons = _rows_from_review(
        review_path, episode_path,
        system_prompt=load_system_prompt(), min_confidence=0.7,
        use_kimi_reasons=False,
    )

    assert len(rows) == 1
    assert json.loads(rows[0]["messages"][-1]["content"])["action_index"] == 0
    assert json.loads(rows[0]["messages"][-1]["content"])["reason"] == "verified lethal on enemy1"
    assert len(lessons) == 1


def test_rows_from_review_uses_kimi_reason_by_default(tmp_path) -> None:
    """新默认行为：use_kimi_reasons=True，SFT row 的 reason 来自 Kimi 教师 reason_en，
    替代 canonical 模板，避免 model 学到模板化 reason hallucination。"""
    # 复用 test_rows_from_review_validates_teacher_labels 的 fixture，但 default flag
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
            {"step": 0, "best_action_index": 0, "reason_en": "lethal: enemy1 hp=3 attack=6 finishes", "confidence": 0.9},
        ],
        "key_lessons": [],
    }
    episode_path = tmp_path / "ep_kimi.json"
    review_path = tmp_path / "review_kimi.json"
    episode_path.write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
    review_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")

    rows, _ = _rows_from_review(
        review_path, episode_path,
        system_prompt=load_system_prompt(), min_confidence=0.7,
    )

    assert len(rows) == 1
    parsed = json.loads(rows[0]["messages"][-1]["content"])
    assert parsed["action_index"] == 0
    assert parsed["reason"] == "lethal: enemy1 hp=3 attack=6 finishes"
    assert rows[0]["meta"]["reason_source"] == "kimi_review"
    assert rows[0]["meta"]["canonical_reason"] == "verified lethal on enemy1"


def test_rows_from_review_rejects_reason_action_mismatch(tmp_path) -> None:
    episode = {
        "episode_id": "ep",
        "turns": [{
            "decisions": [{
                "step": 0,
                "chosen_action_index": 1,
                "pre_decision_state": (
                    "run: char=IRONCLAD round=1\n"
                    "player: hp=80/80 block=99 energy=1/3 powers=-\n"
                    "enemies:\n"
                    "  enemy1: CULTIST hp=30/30 block=0 intent=Attack(6) powers=-\n"
                    "hand:\n"
                    "  [0] BLOODLETTING cost=0 type=skill | Lose 3 HP. Gain 2 Energy.\n"
                    "legal_actions:\n"
                    "  [0] BLOODLETTING hand[0] target=self self_hp_loss=3 self_hp_after=77\n"
                    "  [1] end_turn\n"
                ),
            }]
        }],
    }
    review = {
        "episode_id": "ep",
        "usable_training_labels": [
            {"step": 0, "best_action_index": 0, "reason_en": "End turn; already sufficient block", "confidence": 0.9},
        ],
    }
    episode_path = tmp_path / "episode_input.json"
    review_path = tmp_path / "review.json"
    episode_path.write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
    review_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")

    rows, _lessons = _rows_from_review(review_path, episode_path, system_prompt=load_system_prompt(), min_confidence=0.7)

    assert rows == []


def test_rows_from_review_rejects_label_that_skips_attacking_lethal(tmp_path) -> None:
    episode = {
        "episode_id": "ep",
        "turns": [{
            "decisions": [{
                "step": 65,
                "chosen_action_index": 4,
                "pre_decision_state": (
                    "run: char=IRONCLAD round=4\n"
                    "player: hp=29/80 block=0 energy=3/3 powers=-\n"
                    "enemies:\n"
                    "  enemy4: TWO_TAILED_RAT hp=21/21 block=0 intent=Debuff powers=-\n"
                    "  enemy2: TWO_TAILED_RAT hp=2/20 block=0 intent=Attack(6) powers=-\n"
                    "hand:\n"
                    "  [3] SETUP_STRIKE cost=1 type=attack | Deal 7 damage. Gain 2 Strength this turn.\n"
                    "  [4] BASH cost=2 type=attack | Deal 8 damage. Apply 2 Vulnerable.\n"
                    "legal_actions:\n"
                    "  SETUP_STRIKE hand[3]:\n"
                    "    [4] target=enemy2 damage=7 hp=2 lethal=true\n"
                    "  BASH hand[4]:\n"
                    "    [6] target=enemy4 damage=8 hp=21 lethal=false\n"
                    "  [9] end_turn\n"
                ),
            }]
        }],
    }
    review = {
        "episode_id": "ep",
        "usable_training_labels": [
            {
                "step": 65,
                "best_action_index": 6,
                "reason_en": "Bash before Setup Strike wastes temporary strength",
                "confidence": 0.9,
            },
        ],
    }
    episode_path = tmp_path / "episode_input.json"
    review_path = tmp_path / "review.json"
    episode_path.write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
    review_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")

    rows, _lessons = _rows_from_review(review_path, episode_path, system_prompt=load_system_prompt(), min_confidence=0.7)

    assert rows == []


def test_review_pairs_from_root_finds_episode_siblings(tmp_path) -> None:
    case_dir = tmp_path / "0000_case"
    case_dir.mkdir()
    review_path = case_dir / "turn_order_review.json"
    episode_path = case_dir / "episode_input.json"
    review_path.write_text("{}", encoding="utf-8")
    episode_path.write_text("{}", encoding="utf-8")

    pairs = _review_pairs_from_roots([str(tmp_path)])

    assert pairs == [(review_path.resolve(), episode_path.resolve())]


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
