from __future__ import annotations

from llm.scripts.datasets.filter_kimi_teacher_labels import reject_reasons


def _label(action_index: int, reason: str) -> dict:
    return {
        "best_action_index": action_index,
        "confidence": 0.9,
        "reason_en": reason,
        "reason_zh": "",
        "mechanism_tags": [],
        "user_message": (
            "run: char=IRONCLAD round=1\n"
            "player: hp=80/80 block=0 energy=1/3 powers=-\n"
            "enemies:\n"
            "  enemy1: CULTIST hp=9/30 block=0 intent=Attack(6) powers=-\n"
            "hand:\n"
            "  [0] STRIKE_IRONCLAD cost=1 type=attack | Deal 6 damage.\n"
            "legal_actions:\n"
            "  STRIKE_IRONCLAD hand[0]:\n"
            "    [0] target=enemy1 damage=6\n"
            "  [1] end_turn\n"
        ),
    }


def test_rejects_kill_claim_when_action_is_not_lethal() -> None:
    reasons = reject_reasons(_label(0, "Strike kills enemy1."))

    assert "claims_lethal_but_action_not_lethal" in reasons


def test_keeps_nonlethal_reason_without_kill_claim() -> None:
    assert reject_reasons(_label(0, "Strike damages the attacking enemy.")) == []
