from __future__ import annotations

from llm.data_pipeline.catalog_loader import render_card_description
from llm.data_pipeline.state_renderer import render_state_text


def test_render_card_description_resolves_source_dynamic_vars() -> None:
    assert render_card_description("BLUDGEON") == "造成32点伤害。"
    assert render_card_description("BLUDGEON", is_upgraded=True) == "造成42点伤害。"
    assert render_card_description("DEFEND_IRONCLAD") == "获得5点格挡。"
    assert render_card_description("BASH", is_upgraded=True) == "造成10点伤害。；给予3层易伤。"
    assert render_card_description("FORGOTTEN_RITUAL", is_upgraded=True) == (
        "如果你在本回合消耗过卡牌，则获得4能量。"
    )
    assert (
        render_card_description(
            "STRIKE_IRONCLAD",
            runtime_values={"preview_damage_per_target": {1: 9}},
        )
        == "造成9点伤害。"
    )


def test_render_state_text_resolves_hand_description_placeholders() -> None:
    state = {
        "run": {"act": 0, "floor": 0},
        "player": {"character": "IRONCLAD", "hp": 70, "max_hp": 80, "gold": 125, "deck": []},
        "battle": {
            "energy": 3,
            "max_energy": 3,
            "round_number": 1,
            "hand": [
                {
                    "id": "BLUDGEON",
                    "cost": 3,
                    "type": "Attack",
                    "is_upgraded": True,
                    "can_play": True,
                    "description": "BLUDGEON.description",
                },
                {
                    "id": "STRIKE_IRONCLAD",
                    "cost": 1,
                    "type": "Attack",
                    "can_play": True,
                    "description": "造成{Damage:diff()}点伤害。",
                },
                {
                    "id": "FORGOTTEN_RITUAL",
                    "cost": 1,
                    "type": "Skill",
                    "is_upgraded": True,
                    "can_play": True,
                    "description": (
                        "如果你在本回合[gold]消耗过[/gold]卡牌，则获得4"
                        "[img]res://images/packed/sprite_fonts/ironclad_energy_icon.png[/img]。\n"
                        "[gold]消耗[/gold]。"
                    ),
                },
                {"id": "DEFEND_IRONCLAD", "cost": 1, "type": "Skill", "can_play": True},
            ],
            "enemies": [],
            "draw_pile_cards": [],
            "discard_pile_cards": [],
            "exhaust_pile_cards": [],
        },
    }

    rendered = render_state_text(state, [{"action": "end_turn"}])

    assert "{Damage:diff()}" not in rendered
    assert "{Block:diff()}" not in rendered
    assert "BLUDGEON cost=3 tags=attack,upg | 造成42点伤害。" in rendered
    assert "STRIKE_IRONCLAD cost=1 tags=attack | 造成6点伤害。" in rendered
    assert "FORGOTTEN_RITUAL cost=1 tags=skill,upg | 如果你在本回合消耗过卡牌，则获得4能量。；消耗。" in rendered
    assert "DEFEND_IRONCLAD cost=1 tags=skill | 获得5点格挡。" in rendered
