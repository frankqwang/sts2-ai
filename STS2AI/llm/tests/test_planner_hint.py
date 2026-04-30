from __future__ import annotations

from llm.data_pipeline.planner_hint import (
    format_planner_hint,
    parse_planner_hint_json,
    planner_hint_cache_key,
    render_planner_hint_user_message,
)


def _state(round_number: int = 1) -> dict:
    return {
        "run": {"act": 1, "floor": 4},
        "player": {
            "hp": 30,
            "max_hp": 80,
            "deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}],
            "relics": [{"id": "BURNING_BLOOD"}],
        },
        "battle": {
            "encounter_id": "CULTISTS_NORMAL",
            "round_number_raw": round_number,
            "energy": 2,
            "player": {"block": 0},
            "hand": [
                {
                    "id": "BASH",
                    "cost": 2,
                    "type": "attack",
                    "description": "Deal 8 damage. Apply 2 Vulnerable.",
                    "preview_damage_per_target": {"1": 8},
                }
            ],
        },
        "enemies": [
            {
                "target_id": 1,
                "monster_id": "CULTIST",
                "hp": 30,
                "max_hp": 48,
                "block": 0,
                "intent_type": "Attack",
                "intent_damage": 7,
                "intent_hits": 1,
                "is_alive": True,
            }
        ],
    }


def test_parse_planner_hint_rejects_action_fields() -> None:
    hint, status = parse_planner_hint_json('{"combat_plan":"ok","action_index":0}')

    assert hint is None
    assert status == "forbidden_action_fields"


def test_parse_planner_hint_rejects_cjk_values() -> None:
    hint, status = parse_planner_hint_json('{"battle_objective":"先防御再输出"}')

    assert hint is None
    assert status == "non_english_text"


def test_parse_and_format_planner_hint() -> None:
    raw = (
        '{"battle_objective":"Use BASH to create a vulnerable damage window.",'
        '"enemy_focus":"focus enemy1",'
        '"kill_order":["enemy1"],'
        '"danger_notes":["Do not end turn into avoidable attack."]}'
    )

    hint, status = parse_planner_hint_json(raw)

    assert status == "ok"
    assert hint is not None
    rendered = format_planner_hint(hint)
    assert "battle_objective: Use BASH" in rendered
    assert "enemy_focus: focus enemy1" in rendered
    assert "kill_order: enemy1" in rendered
    assert "danger_notes: Do not end turn" in rendered


def test_parse_legacy_planner_hint_keys_are_invalid() -> None:
    hint, status = parse_planner_hint_json(
        '{"combat_plan":"Kill the scaling enemy.","resource_policy":"Use BASH for Vulnerable windows."}'
    )

    assert status == "legacy_planner_hint_fields"
    assert hint is None


def test_parse_unknown_planner_hint_keys_are_invalid() -> None:
    hint, status = parse_planner_hint_json(
        '{"battle_objective":"Kill the scaling enemy.","deck_plan":"Use BASH before attacks."}'
    )

    assert status == "unknown_planner_hint_fields"
    assert hint is None


def test_render_planner_hint_user_message_has_no_legal_actions() -> None:
    message = render_planner_hint_user_message(
        _state(),
        [{"action": "play_card", "card_index": 0, "target_id": 1}],
    )

    assert "legal_actions:" not in message
    assert "retrieved_knowledge:" in message
    assert "Task: write a short battle-level planner_hint" in message
    assert "BASH" in message


def test_planner_hint_cache_key_can_refresh_per_turn() -> None:
    first = planner_hint_cache_key(_state(round_number=1), refresh="turn")
    second = planner_hint_cache_key(_state(round_number=2), refresh="turn")
    combat_first = planner_hint_cache_key(_state(round_number=1), refresh="combat")
    combat_second = planner_hint_cache_key(_state(round_number=2), refresh="combat")

    assert first != second
    assert combat_first == combat_second


def test_planner_hint_combat_cache_key_ignores_dead_enemy_removed_from_state() -> None:
    state = _state()
    state["enemies"].append({
        "target_id": 2,
        "monster_id": "DAMP_CULTIST",
        "hp": 40,
        "max_hp": 40,
        "is_alive": True,
    })
    after_kill = _state()
    after_kill["enemies"] = [state["enemies"][0]]

    assert planner_hint_cache_key(state, refresh="combat") == planner_hint_cache_key(after_kill, refresh="combat")
