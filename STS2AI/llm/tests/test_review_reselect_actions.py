from __future__ import annotations

from llm.scripts.analyze_action_ordering import _legal_actions
from llm.scripts.review_reselect_actions import is_actionable_review_row, score_choice


def test_score_choice_rewards_fixing_visible_lethal() -> None:
    user = (
        "run: char=IRONCLAD round=1\n"
        "player: hp=80/80 block=0 energy=1/3 powers=-\n"
        "enemies:\n"
        "  enemy1: CULTIST hp=3/30 block=0 intent=Attack(6) powers=-\n"
        "legal_actions:\n"
        "  STRIKE_IRONCLAD hand[0]:\n"
        "    [0] target=enemy1 damage=6\n"
        "  [1] end_turn\n"
    )
    actions = _legal_actions({"user_message": user})
    row = {"flags": ["missed_visible_lethal", "dangerous_end_turn"]}

    old = score_choice(row=row, action_index=1, reason="end turn", actions=actions, user_message=user)
    new = score_choice(row=row, action_index=0, reason="kill enemy1", actions=actions, user_message=user)

    assert new["score"] > old["score"]
    assert "takes_visible_lethal" in new["notes"]
    assert "avoids_dangerous_end_turn" in new["notes"]


def test_score_choice_does_not_penalize_forced_end_turn_as_avoidable() -> None:
    user = (
        "run: char=IRONCLAD round=1\n"
        "player: hp=80/80 block=0 energy=0/3 powers=-\n"
        "enemies:\n"
        "  enemy1: CULTIST hp=30/30 block=0 intent=Attack(6) powers=-\n"
        "legal_actions:\n"
        "  [0] end_turn\n"
    )
    actions = _legal_actions({"user_message": user})
    row = {"flags": ["dangerous_end_turn"]}

    scored = score_choice(row=row, action_index=0, reason="forced", actions=actions, user_message=user)

    assert scored["score"] >= 1.0
    assert "keeps_dangerous_end_turn" not in scored["notes"]


def test_actionable_filter_skips_forced_end_turn_but_keeps_lethal_miss() -> None:
    forced = {
        "flags": ["dangerous_end_turn"],
        "chosen": {"index": 0, "card_id": "end_turn"},
        "user_message": (
            "run: char=IRONCLAD round=1\n"
            "player: hp=80/80 block=0 energy=0/3 powers=-\n"
            "enemies:\n"
            "  enemy1: CULTIST hp=30/30 block=0 intent=Attack(6) powers=-\n"
            "legal_actions:\n"
            "  [0] end_turn\n"
        ),
    }
    lethal_miss = {
        "flags": ["missed_visible_lethal"],
        "chosen": {"index": 1, "card_id": "end_turn"},
        "user_message": (
            "run: char=IRONCLAD round=1\n"
            "player: hp=80/80 block=0 energy=1/3 powers=-\n"
            "enemies:\n"
            "  enemy1: CULTIST hp=3/30 block=0 intent=Attack(6) powers=-\n"
            "legal_actions:\n"
            "  STRIKE_IRONCLAD hand[0]:\n"
            "    [0] target=enemy1 damage=6\n"
            "  [1] end_turn\n"
        ),
    }

    assert not is_actionable_review_row(forced)
    assert is_actionable_review_row(lethal_miss)
