from __future__ import annotations

from types import SimpleNamespace

from llm.data_pipeline.action_quality import (
    assess_action_quality,
    assess_action_quality_report,
    summarize_quality_reports,
)


def _state() -> dict:
    return {
        "player": {"hp": 30, "max_hp": 80},
        "battle": {
            "energy": 1,
            "player": {"block": 0},
            "hand": [
                {
                    "id": "STRIKE_IRONCLAD",
                    "preview_damage_per_target": {"1": 6},
                },
                {
                    "id": "DEFEND_IRONCLAD",
                    "preview_block": 5,
                },
            ],
        },
        "enemies": [
            {
                "target_id": 1,
                "hp": 6,
                "max_hp": 40,
                "block": 0,
                "intent_type": "Attack",
                "intent_damage": 8,
                "intent_hits": 1,
                "is_alive": True,
            }
        ],
    }


def test_action_quality_flags_missed_lethal_and_dangerous_end_turn() -> None:
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": 1},
        {"action": "play_card", "card_index": 1, "target_id": -1},
        {"action": "end_turn"},
    ]

    flags = assess_action_quality(_state(), legal, 2)

    assert "missed_visible_lethal" in flags
    assert "end_turn_with_playable_cards" in flags
    assert "dangerous_end_turn" in flags

    report = assess_action_quality_report(_state(), legal, 2).as_dict()
    assert report["opportunities"]["visible_lethal"] == 1
    assert report["misses"]["visible_lethal"] == 1
    assert report["misses"]["dangerous_turn"] == 1
    assert report["mechanism_score"] < 1.0


def test_action_quality_allows_visible_lethal() -> None:
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": 1},
        {"action": "end_turn"},
    ]

    assert assess_action_quality(_state(), legal, 0) == []


def test_action_quality_does_not_flag_forced_end_turn_as_dangerous() -> None:
    legal = [{"action": "end_turn"}]

    report = assess_action_quality_report(_state(), legal, 0).as_dict()

    assert "dangerous_end_turn" not in report["flags"]
    assert "dangerous_turn" not in report["opportunities"]


def test_action_quality_flags_reason_math_contradiction() -> None:
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": 1},
        {"action": "end_turn"},
    ]
    state = _state()
    state["enemies"][0]["hp"] = 9

    report = assess_action_quality_report(
        state,
        legal,
        0,
        reason="lethal CULTIST(6>=9)",
    ).as_dict()

    assert "reason_math_contradiction" in report["flags"]
    assert "reason_claims_lethal_but_action_not_lethal" in report["flags"]
    assert report["misses"]["reason_consistency"] >= 1


def test_action_quality_flags_proceed_with_unclaimed_rewards() -> None:
    report = assess_action_quality_report(
        {"state_type": "combat_rewards"},
        [
            {"action": "claim_reward", "label": "11 Gold"},
            {"action": "claim_reward", "label": "Add a card to your deck."},
            {"action": "proceed"},
        ],
        2,
    ).as_dict()

    assert "proceed_with_unclaimed_rewards" in report["flags"]
    assert report["opportunities"]["unclaimed_reward"] == 1
    assert report["misses"]["unclaimed_reward"] == 1


def test_action_quality_flags_unnecessary_potion_use() -> None:
    report = assess_action_quality_report(
        {"battle": {"player": {"block": 0}}, "enemies": []},
        [
            {"action": "use_potion", "slot": 0, "label": "POWER_POTION"},
            {"action": "end_turn"},
        ],
        0,
    ).as_dict()

    assert "unnecessary_potion_use" in report["flags"]
    assert report["opportunities"]["potion_conservation"] == 1
    assert report["misses"]["potion_conservation"] == 1


def test_action_quality_counts_constrict_as_end_turn_hp_loss() -> None:
    state = _state()
    state["player"]["hp"] = 6
    state["battle"]["player"]["block"] = 10
    state["battle"]["player"]["powers"] = [{"id": "CONSTRICT_POWER", "amount": 6}]
    state["enemies"][0]["hp"] = 20
    state["enemies"][0]["intent_damage"] = 12
    legal = [
        {"action": "play_card", "card_index": 1, "target_id": -1},
        {"action": "end_turn"},
    ]

    report = assess_action_quality_report(state, legal, 1).as_dict()

    assert report["metrics"]["end_turn_hp_loss"] == 6
    assert report["metrics"]["hp_after_current_threat"] <= 0
    assert "dangerous_end_turn" in report["flags"]


def test_action_quality_flags_dangerous_self_damage() -> None:
    state = _state()
    state["player"]["hp"] = 9
    state["battle"]["player"]["powers"] = [{"id": "CONSTRICT_POWER", "amount": 6}]
    state["battle"]["hand"].append(
        {"id": "BLOODLETTING", "description": "Lose 3 HP. Gain 3 Energy."}
    )
    state["enemies"][0]["hp"] = 20
    state["enemies"][0]["intent_damage"] = 12
    legal = [
        {"action": "play_card", "card_index": 2, "target_id": -1},
        {"action": "end_turn"},
    ]

    report = assess_action_quality_report(state, legal, 0).as_dict()

    assert "dangerous_self_damage" in report["flags"]
    assert report["misses"]["self_damage_safety"] == 1


def test_action_quality_flags_low_hp_self_damage_with_incoming() -> None:
    state = _state()
    state["player"]["hp"] = 9
    state["battle"]["hand"].append(
        {"id": "BLOODLETTING", "description": "Lose 3 HP. Gain 3 Energy."}
    )
    state["enemies"][0]["hp"] = 20
    state["enemies"][0]["intent_damage"] = 5
    legal = [
        {"action": "play_card", "card_index": 2, "target_id": -1},
        {"action": "end_turn"},
    ]

    report = assess_action_quality_report(state, legal, 0).as_dict()

    assert "low_hp_self_damage" in report["flags"]
    assert report["misses"]["self_damage_safety"] == 1


def test_action_quality_flags_action_score_lethal_math_contradiction() -> None:
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": 1},
        {"action": "end_turn"},
    ]
    state = _state()
    state["enemies"][0]["hp"] = 9

    report = assess_action_quality_report(
        state,
        legal,
        0,
        reason="deal damage",
        action_scores=[
            {
                "action_index": 0,
                "score": 8.0,
                "note": "lethal CULTIST: damage=6 target_hp=9",
            }
        ],
    ).as_dict()

    assert "action_score_lethal_math_contradiction" in report["flags"]
    assert report["misses"]["score_consistency"] == 1


def test_quality_summary_tracks_hp_and_speed_metrics() -> None:
    class Step:
        pass

    step = Step()
    step.state = _state()
    step.legal_actions = [{"action": "end_turn"}]
    step.chosen_index = 0
    step.quality_report = {
        "flags": ["dangerous_end_turn"],
        "opportunities": {"dangerous_turn": 1},
        "misses": {"dangerous_turn": 1},
        "metrics": {"visible_damage": 6},
        "mechanism_score": 0.0,
    }
    final_state = _state()
    final_state["player"]["hp"] = 24

    summary = summarize_quality_reports([step], final_state=final_state)

    assert summary["mechanism_score"] == 0.0
    assert summary["sequence_score"] == 1.0
    assert summary["defense_score"] == 0.0
    assert summary["hp_lost"] == 6
    assert summary["turns"] == 1
    assert summary["visible_damage_per_step"] == 6


def test_summarize_quality_reports_ignores_reset_final_state_hp_rebound():
    """sim 在败场可能把 final_state.player.hp 反弹到初始值；hp_lost 不能因此被算成 0。"""
    # step 1: hp=30 (第一个决策步)
    step_first = SimpleNamespace()
    s1 = _state()
    s1["player"]["hp"] = 30
    step_first.state = s1
    step_first.legal_actions = [{"action": "end_turn"}]
    step_first.chosen_index = 0
    step_first.quality_report = {"flags": [], "opportunities": {}, "misses": {}, "metrics": {}, "mechanism_score": 1.0}

    # step 2: hp=12 (玩家已被打到残血)
    step_last = SimpleNamespace()
    s2 = _state()
    s2["player"]["hp"] = 12
    step_last.state = s2
    step_last.legal_actions = [{"action": "end_turn"}]
    step_last.chosen_index = 0
    step_last.quality_report = {"flags": [], "opportunities": {}, "misses": {}, "metrics": {}, "mechanism_score": 1.0}

    # final_state 报的 hp 反而比 step 末尾还高（典型 sim reset 信号）
    rebound_final = _state()
    rebound_final["player"]["hp"] = 80

    summary = summarize_quality_reports([step_first, step_last], final_state=rebound_final)
    # 关键：不能因为 final_state 反弹（80 > 12）就把 hp_lost 算成 0
    # hp_start=30, hp_end 应该取 step_last 的 12（不是 final_state 的 80），hp_lost = 30 - 12 = 18
    assert summary["hp_lost"] == 18


def test_summarize_quality_reports_ignores_negative_chosen_index_for_turn_count():
    """chosen_index=-1 (invalid_output) 不能被当作 end_turn 误数 turn_count。"""
    step = SimpleNamespace()
    step.state = _state()
    # legal_actions 末尾恰好是 end_turn；负索引会取末尾
    step.legal_actions = [{"action": "play_card"}, {"action": "end_turn"}]
    step.chosen_index = -1  # invalid output
    step.quality_report = {"flags": [], "opportunities": {}, "misses": {}, "metrics": {}, "mechanism_score": 1.0}

    summary = summarize_quality_reports([step], final_state=None)
    assert summary["turns"] == 0  # 不能误数为 1
