from __future__ import annotations

from llm.data_pipeline.heuristic_teacher import score_actions


def test_score_action_notes_use_explicit_damage_and_hp_fields() -> None:
    state = {
        "player": {"hp": 70, "max_hp": 80},
        "battle": {
            "player": {"block": 0, "powers": []},
            "hand": [
                {
                    "card_id": "STRIKE_IRONCLAD",
                    "preview_damage_per_target": {"1": 6},
                }
            ],
        },
        "enemies": [
            {
                "target_id": 1,
                "monster_id": "CULTIST",
                "hp": 6,
                "block": 0,
                "intent_type": "Attack",
                "intent_damage": 0,
                "is_alive": True,
            }
        ],
    }
    legal_actions = [
        {"type": "play_card", "card_id": "STRIKE_IRONCLAD", "card_index": 0, "target_id": 1},
    ]

    scores = score_actions(state, legal_actions)

    assert scores[0].reason == "lethal CULTIST: damage=6 target_hp=6"


def test_forgotten_ritual_is_scored_as_energy_skill_not_power() -> None:
    state = {
        "player": {"hp": 70, "max_hp": 80, "exhaust_pile_count": 1},
        "battle": {
            "energy": 1,
            "player": {"block": 0, "powers": [], "exhaust_pile_count": 1},
            "hand": [
                {
                    "card_id": "FORGOTTEN_RITUAL",
                    "type": "skill",
                }
            ],
        },
        "enemies": [],
    }
    legal_actions = [
        {"type": "play_card", "card_id": "FORGOTTEN_RITUAL", "card_index": 0},
    ]

    scores = score_actions(state, legal_actions)

    assert scores[0].reason == "gain energy=3 after Exhaust"
