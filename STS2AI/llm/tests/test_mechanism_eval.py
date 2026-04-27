from __future__ import annotations

from llm.eval.mechanism_eval import (
    _action_scores,
    _as_confidence,
    effect_matched,
    expected_effects_from_events,
    legal_indices_from_user,
    score_reason_effects,
)


def test_legal_indices_from_grouped_actions() -> None:
    text = """legal_actions:
  BASH hand[0]:
    [0] target=enemy1 damage=8
    [1] target=enemy2 damage=8

  [2] DEFEND_IRONCLAD hand[1] target=self block=5
  [3] end_turn
"""
    assert legal_indices_from_user(text) == {0, 1, 2, 3}


def test_expected_effects_use_only_energy_spent_event_for_energy() -> None:
    effects = expected_effects_from_events([
        {"type": "energy_spent", "energy_spent": 2},
        {"type": "card_play_started", "energy_spent": 2},
        {"type": "damage_received", "unblocked_damage": 8, "target_id": "CULTIST", "target_combat_id": 1},
        {"type": "power_received", "power_id": "VULNERABLE_POWER", "amount_int": 2, "target_id": "CULTIST", "target_combat_id": 1},
    ])
    assert effects[0].kind == "energy"
    assert effects[0].amount == "2"
    assert [effect.kind for effect in effects] == ["energy", "damage", "power"]


def test_effect_match_power_alias() -> None:
    effects = expected_effects_from_events([
        {"type": "power_received", "power_id": "VULNERABLE_POWER", "amount_int": 2, "target_id": "CULTIST"},
    ])
    assert effect_matched(effects[0], "applies 2 Vulnerable to CULTIST")


def test_score_reason_effects_counts_recall() -> None:
    effects = expected_effects_from_events([
        {"type": "energy_spent", "energy_spent": 1},
        {"type": "block_gained", "amount_int": 5},
    ])
    score = score_reason_effects("spends 1 energy and gains 5 block", effects)
    assert score["expected"] == 2
    assert score["matched"] == 2
    assert score["recall"] == 1.0


def test_confidence_and_scores_are_normalized() -> None:
    assert _as_confidence(82) == 0.82
    assert _as_confidence("0.6") == 0.6
    assert _as_confidence(True) is None
    assert _action_scores([
        {"action_index": "1", "score": "3.5", "note": "block"},
        {"index": 0, "score": 5},
        {"action_index": "bad", "score": 1},
    ]) == [
        {"action_index": 0, "score": 5.0},
        {"action_index": 1, "score": 3.5, "note": "block"},
    ]
