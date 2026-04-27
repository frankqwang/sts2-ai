from __future__ import annotations

from llm.eval.policy_eval import _hard_cases


def test_hard_cases_include_high_hp_loss_even_when_winning() -> None:
    cases = _hard_cases({
        "clean": {
            "encounter_label": "clean",
            "win_rate": 1.0,
            "invalid_output_episode_rate": 0.0,
            "reward": {"avg": 8.0},
            "hp_lost": {"avg": 0.0, "max": 0.0},
            "defense_score": {"avg": 1.0},
        },
        "bloody": {
            "encounter_label": "bloody",
            "win_rate": 1.0,
            "invalid_output_episode_rate": 0.0,
            "reward": {"avg": 7.9},
            "hp_lost": {"avg": 9.0, "max": 14.0},
            "defense_score": {"avg": 0.86},
        },
    })

    assert cases[0]["encounter_key"] == "bloody"
    assert cases[0]["reason"] == "high_hp_loss"
    assert cases[0]["hp_lost_avg"] == 9.0
