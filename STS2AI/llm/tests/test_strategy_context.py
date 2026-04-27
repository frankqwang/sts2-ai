from __future__ import annotations

from llm.data_pipeline.state_renderer import render_state_text
from llm.data_pipeline.strategy_context import StrategyMemory, build_strategy_context


def _state(round_number: int = 1) -> dict:
    return {
        "player": {
            "hp": 30,
            "max_hp": 80,
            "deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}],
            "relics": [{"id": "BURNING_BLOOD"}],
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
    assert "memory:" in context
    assert "plan:" in context
    assert "turn:" in context
    assert "hand[0] STRIKE_IRONCLAD kills enemy1" in context
    assert "legal_actions override context" in context


def test_strategy_context_counts_constrict_as_total_hp_threat() -> None:
    state = _state()
    state["player"]["hp"] = 9
    state["battle"]["player"] = {"block": 0, "powers": [{"id": "CONSTRICT_POWER", "amount": 6}]}
    state["enemies"][0]["hp"] = 40
    state["enemies"][0]["intent_damage"] = 12

    context = build_strategy_context(state, state["legal_actions"])

    assert "incoming_damage=12" in context
    assert "end_turn_hp_loss=6" in context
    assert "total_hp_loss=18" in context
    assert "dying to current attack/end-turn HP loss" in context


def test_strategy_memory_tracks_turn_history_and_resets_on_new_round() -> None:
    memory = StrategyMemory()
    state = _state(round_number=1)
    legal = state["legal_actions"]

    first = memory.context_text(state, legal)
    memory.record_action(legal[0])
    second = memory.context_text(state, legal)
    third = memory.context_text(_state(round_number=2), legal)

    assert "turn: actions=none" in first
    assert "played STRIKE_IRONCLAD hand[0] -> enemy1" in second
    assert "turn: actions=none" in third


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

    assert "combat: lost_hp=6" in same_combat
    assert "prev_combat lost_hp=6" in carried


def test_strategy_memory_does_not_repeat_turn_actions_in_combat_prior_actions() -> None:
    memory = StrategyMemory()
    state = _state(round_number=1)
    legal = state["legal_actions"]

    memory.context_text(state, legal)
    memory.record_action(legal[0])
    context = memory.context_text(state, legal)

    assert "combat: lost_hp=0; prior_actions=none" in context
    assert "turn: actions=played STRIKE_IRONCLAD hand[0] -> enemy1" in context


def test_render_state_text_injects_strategy_context_after_run_line() -> None:
    state = _state()
    text = render_state_text(state, state["legal_actions"], strategy_context="strategy_context:\n  test: yes")
    lines = text.splitlines()

    assert lines[0].startswith("run:")
    assert lines[1] == "strategy_context:"
    assert lines[2] == "  test: yes"
