from __future__ import annotations

from llm.eval.policy_eval import _final_player_hp, _hard_cases


def test_final_player_hp_preserves_zero_not_falling_back():
    """玩家死亡时 hp=0 必须保留为 0；不能因 `or` 短路被 fallback 到 battle.player.hp。"""
    final_state = {
        "player": {"hp": 0, "max_hp": 80},
        "battle": {"player": {"hp": 80, "max_hp": 80}},
    }
    assert _final_player_hp(final_state) == 0


def test_final_player_hp_falls_back_when_top_player_missing():
    final_state = {
        "battle": {"player": {"hp": 50, "max_hp": 80}},
    }
    assert _final_player_hp(final_state) == 50


def test_final_player_hp_returns_none_when_both_missing():
    assert _final_player_hp({}) is None
    assert _final_player_hp(None) is None
    assert _final_player_hp({"player": {}}) is None


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
