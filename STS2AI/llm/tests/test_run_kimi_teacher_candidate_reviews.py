from __future__ import annotations

from llm.scripts.run_kimi_teacher_candidate_reviews import (
    _review_items,
    _validate_review,
)


def _candidate() -> dict:
    return {
        "candidate_id": "c1",
        "source": {
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
            )
        },
    }


def test_review_items_accepts_grouped_reviews() -> None:
    assert _review_items({"reviews": [{"candidate_id": "c1"}]}) == [{"candidate_id": "c1"}]


def test_review_items_accepts_single_review() -> None:
    assert _review_items({"candidate_id": "c1", "best_action_index": 0}) == [
        {"candidate_id": "c1", "best_action_index": 0}
    ]


def test_validate_review_rejects_illegal_index() -> None:
    ok, status, extra = _validate_review(
        {"candidate_id": "c1", "best_action_index": 99, "confidence": 1.0},
        candidates={"c1": _candidate()},
        min_confidence=0.7,
    )

    assert ok is False
    assert status == "best_action_index_not_legal"
    assert extra["legal_indices"] == [0, 1]


def test_validate_review_accepts_legal_high_confidence() -> None:
    ok, status, extra = _validate_review(
        {"candidate_id": "c1", "best_action_index": 0, "confidence": 0.8},
        candidates={"c1": _candidate()},
        min_confidence=0.7,
    )

    assert ok is True
    assert status == "ok"
    assert extra["confidence"] == 0.8
