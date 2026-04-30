from __future__ import annotations

import pytest

from llm.data_pipeline.state_renderer import render_state_text
from llm.data_pipeline.strategy_context import StrategyMemory, build_strategy_context


def _state(round_number: int = 1) -> dict:
    return {
        "run": {"act": 1, "floor": 3, "gold": 99, "seed": "test-seed"},
        "player": {
            "character": "IRONCLAD",
            "hp": 30,
            "max_hp": 80,
            "gold": 99,
            "deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}],
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [{"id": "FORTIFIER"}],
        },
        "battle": {
            "round_number_raw": round_number,
            "energy": 2,
            "player": {"block": 0},
            "hand": [
                {
                    "id": "STRIKE_IRONCLAD",
                    "cost": 1,
                    "type": "attack",
                    "description": "Deal 6 damage.",
                    "preview_damage_per_target": {"1": 6},
                }
            ],
        },
        "enemies": [
            {
                "target_id": 1,
                "monster_id": "CULTIST",
                "hp": 6,
                "max_hp": 48,
                "block": 0,
                "intent_type": "Attack",
                "intent_damage": 7,
                "intent_hits": 1,
                "is_alive": True,
            }
        ],
        "legal_actions": [
            {"action": "play_card", "card_id": "STRIKE_IRONCLAD", "card_index": 0, "target_id": 1},
            {"action": "end_turn"},
        ],
    }


def test_strategy_context_contains_game_combat_turn_and_lethal() -> None:
    state = _state()
    context = build_strategy_context(state, state["legal_actions"])

    assert "strategy_context:" in context
    assert "agent_memory:" not in context
    assert "short_term:" in context
    assert "long_term: none" in context
    assert "run_memory:" not in context
    assert "plan:" not in context
    assert "threat:" not in context
    assert "target:" not in context
    assert "avoid=" not in context


def test_strategy_context_can_replace_rule_plan_with_planner_hint() -> None:
    state = _state()
    context = build_strategy_context(
        state,
        state["legal_actions"],
        planner_hint="battle_objective: kill CULTIST before it scales\nkill_order: enemy1",
    )

    assert "planner_hint:" in context
    assert "battle_objective: kill CULTIST before it scales" in context
    assert "  plan:" not in context
    assert "threat:" not in context
    assert "legal_actions override context" not in context


def test_strategy_context_rejects_legacy_planner_hint_fields() -> None:
    state = _state()

    with pytest.raises(ValueError, match="legacy_planner_hint_field:combat_plan"):
        build_strategy_context(
            state,
            state["legal_actions"],
            planner_hint="combat_plan: old schema",
        )


def test_strategy_context_rejects_extra_memory_wrapper() -> None:
    state = _state()

    with pytest.raises(ValueError, match="strategy_context_memory_must_start_with_short_term"):
        build_strategy_context(
            state,
            state["legal_actions"],
            memory="memory:\n  old: yes",
        )


def test_strategy_context_does_not_generate_threat_rules() -> None:
    state = _state()
    state["player"]["hp"] = 9
    state["battle"]["player"] = {"block": 0, "powers": [{"id": "CONSTRICT_POWER", "amount": 6}]}
    state["enemies"][0]["hp"] = 40
    state["enemies"][0]["intent_damage"] = 12

    context = build_strategy_context(state, state["legal_actions"])

    assert "incoming_damage=12" not in context
    assert "end_turn_hp_loss=6" not in context
    assert "dying to current attack/end-turn HP loss" not in context


def test_strategy_memory_tracks_turn_history_and_resets_on_new_round() -> None:
    memory = StrategyMemory()
    state = _state(round_number=1)
    legal = state["legal_actions"]

    first = memory.context_text(state, legal)
    memory.record_action(legal[0])
    second = memory.context_text(state, legal)
    third = memory.context_text(_state(round_number=2), legal)

    assert "recent_actions: none" in first
    assert "played STRIKE_IRONCLAD hand[0] -> enemy1" in second
    assert "recent_actions: none" in third


def test_strategy_memory_does_not_reset_when_dead_enemy_disappears() -> None:
    memory = StrategyMemory()
    state = _state(round_number=1)
    state["enemies"].append({
        "target_id": 2,
        "monster_id": "DAMP_CULTIST",
        "hp": 40,
        "max_hp": 40,
        "is_alive": True,
    })
    legal = state["legal_actions"]

    memory.context_text(state, legal)
    memory.record_action(legal[0])
    after_kill = _state(round_number=1)
    carried = memory.context_text(after_kill, legal)

    assert "played STRIKE_IRONCLAD hand[0] -> enemy1" in carried


def test_strategy_memory_keeps_concise_run_and_combat_memory() -> None:
    memory = StrategyMemory()
    state = _state(round_number=1)
    legal = state["legal_actions"]

    memory.context_text(state, legal)
    memory.record_action(legal[0])
    hurt_state = _state(round_number=2)
    hurt_state["player"]["hp"] = 24
    same_combat = memory.context_text(hurt_state, legal)
    next_combat = _state(round_number=1)
    next_combat["battle"]["encounter_id"] = "NEXT"
    carried = memory.context_text(next_combat, legal)

    assert "lost_hp=6" not in same_combat
    assert "prev_combat lost_hp=6" not in carried
    assert "agent_memory:" not in carried
    assert "short_term:" in carried


def test_strategy_memory_combat_reset_clears_short_term_memory() -> None:
    memory = StrategyMemory()
    legal = _state()["legal_actions"]
    memory.context_text(_state(round_number=1), legal)
    memory.record_action(legal[0])

    memory.reset_combat()
    context = memory.context_text(_state(round_number=1), legal)

    assert "recent_actions: none" in context
    assert "run_prev_combats" not in context


def test_planner_memory_text_is_memory_not_state_summary() -> None:
    memory = StrategyMemory()
    text = memory.planner_memory_text(_state())

    assert "agent_memory:" not in text
    assert "short_term:" in text
    assert "long_term: none" in text
    assert "BURNING_BLOOD" not in text


def test_strategy_memory_does_not_repeat_turn_actions_in_combat_prior_actions() -> None:
    memory = StrategyMemory()
    state = _state(round_number=1)
    legal = state["legal_actions"]

    memory.context_text(state, legal)
    memory.record_action(legal[0])
    context = memory.context_text(state, legal)

    assert "prior_actions" not in context
    assert "recent_actions: played STRIKE_IRONCLAD hand[0] -> enemy1" in context


def test_render_state_text_injects_strategy_context_after_run_line() -> None:
    state = _state()
    text = render_state_text(state, state["legal_actions"], strategy_context="strategy_context:\n  test: yes")
    lines = text.splitlines()

    assert lines[0].startswith("run:")
    assert lines[1] == "strategy_context:"
    assert lines[2] == "  test: yes"
    assert lines[3] == ""
    assert lines[4].startswith("player:")


def test_render_state_text_separates_main_sections_with_blank_lines() -> None:
    state = _state()
    text = render_state_text(state, state["legal_actions"])
    lines = text.splitlines()
    player_index = lines.index(next(line for line in lines if line.startswith("player:")))
    actions_index = lines.index("legal_actions:")

    assert lines[player_index - 1] == ""
    assert lines[actions_index - 1] == ""
