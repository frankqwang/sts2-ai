from __future__ import annotations

from llm.scripts.analysis.analyze_action_ordering import analyze


def test_analyze_detects_bash_before_strike_sequence() -> None:
    rows = [
        {
            "episode_id": "ep1",
            "episode_step": 0,
            "user_message": (
                "run: char=IRONCLAD round=1\n"
                "player: hp=80/80 block=0 energy=3/3 powers=-\n"
                "enemies:\n"
                "  enemy1: CULTIST hp=30/30 block=0 intent=Attack(6) powers=-\n"
                "legal_actions:\n"
                "  BASH hand[0]:\n"
                "    [0] target=enemy1 damage=8\n"
                "  STRIKE_IRONCLAD hand[1]:\n"
                "    [1] target=enemy1 damage=6\n"
            ),
            "decoded": {"action_index": 0, "reason": "apply vulnerable"},
            "chosen_action": {"card_id": "BASH", "target_id": 1},
        },
        {
            "episode_id": "ep1",
            "episode_step": 1,
            "user_message": (
                "run: char=IRONCLAD round=1\n"
                "player: hp=80/80 block=0 energy=1/3 powers=-\n"
                "enemies:\n"
                "  enemy1: CULTIST hp=22/30 block=0 intent=Attack(6) powers=VULNERABLE_POWER=2\n"
                "legal_actions:\n"
                "  STRIKE_IRONCLAD hand[0]:\n"
                "    [0] target=enemy1 damage=9\n"
            ),
            "decoded": {"action_index": 0, "reason": "hit vulnerable target"},
            "chosen_action": {"card_id": "STRIKE_IRONCLAD", "target_id": 1},
        },
    ]

    result = analyze(rows, example_limit=3)

    assert result["counts"]["bash_with_followup_attack_opportunities"] == 1
    assert result["counts"]["bash_first"] == 1
    assert result["counts"]["bash_then_same_target_attack"] == 1
    assert result["rates"]["bash_first_rate"] == 1.0


def test_analyze_detects_attack_before_available_bash() -> None:
    rows = [
        {
            "episode_id": "ep1",
            "episode_step": 0,
            "user_message": (
                "run: char=IRONCLAD round=1\n"
                "player: hp=80/80 block=0 energy=3/3 powers=-\n"
                "enemies:\n"
                "  enemy1: CULTIST hp=30/30 block=0 intent=Attack(6) powers=-\n"
                "legal_actions:\n"
                "  BASH hand[0]:\n"
                "    [0] target=enemy1 damage=8\n"
                "  STRIKE_IRONCLAD hand[1]:\n"
                "    [1] target=enemy1 damage=6\n"
            ),
            "decoded": {"action_index": 1, "reason": "strike"},
            "chosen_action": {"card_id": "STRIKE_IRONCLAD", "target_id": 1},
        }
    ]

    result = analyze(rows, example_limit=3)

    assert result["counts"]["bash_with_followup_attack_opportunities"] == 1
    assert result["counts"]["attack_before_available_bash"] == 1
    assert result["rates"]["attack_before_available_bash_rate"] == 1.0
