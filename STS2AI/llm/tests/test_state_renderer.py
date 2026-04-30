from __future__ import annotations

from llm.data_pipeline.catalog_loader import render_card_description
from llm.data_pipeline.state_renderer import (
    can_render_structured_actions,
    inject_experience_context,
    render_state_text,
    render_structured_action_state_text,
)


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
        "player": {
            "character": "IRONCLAD",
            "hp": 70,
            "max_hp": 80,
            "gold": 125,
            "deck": [],
            "relics": [{"id": "BURNING_BLOOD"}],
        },
        "battle": {
            "energy": 3,
            "max_energy": 3,
            "round_number": 1,
            "player": {"powers": [{"id": "VULNERABLE_POWER", "amount": 1}]},
            "hand": [
                {
                    "id": "BLUDGEON",
                    "cost": 3,
                    "type": "Attack",
                    "is_upgraded": True,
                    "can_play": True,
                    "description": "Deal 42 damage.",
                    "preview_damage_per_target": {1: 42},
                },
                {
                    "id": "STRIKE_IRONCLAD",
                    "cost": 1,
                    "type": "Attack",
                    "can_play": True,
                    "description": "Deal 6 damage.",
                    "preview_damage_per_target": {1: 6},
                },
                {
                    "id": "FORGOTTEN_RITUAL",
                    "cost": 1,
                    "type": "Skill",
                    "is_upgraded": True,
                    "can_play": True,
                    "description": "If you [gold]Exhausted[/gold] a card this turn, gain 4 Energy.\nExhaust.",
                },
                {
                    "id": "DEFEND_IRONCLAD",
                    "cost": 1,
                    "type": "Skill",
                    "can_play": True,
                    "description": "Gain 5 Block.",
                    "preview_block": 5,
                },
            ],
            "enemies": [
                {
                    "id": "CHOMPER",
                    "combat_id": 1,
                    "hp": 64,
                    "max_hp": 64,
                    "block": 0,
                    "next_move_id": "Attack",
                    "powers": [{"id": "ARTIFACT_POWER", "amount": 2}],
                }
            ],
            "draw_pile_cards": [],
            "discard_pile_cards": [],
            "exhaust_pile_cards": [],
        },
    }

    rendered = render_state_text(
        state,
        [
            {"action": "play_card", "card_index": 0, "card_id": "BLUDGEON", "target_id": 1},
            {"action": "play_card", "card_index": 3, "card_id": "DEFEND_IRONCLAD", "target_id": -1},
            {"action": "end_turn", "card_index": 0, "target_id": 0},
        ],
    )

    assert "{Damage:diff()}" not in rendered
    assert "{Block:diff()}" not in rendered
    assert "BLUDGEON cost=3 type=Attack upgraded=true | Deal 42 damage." in rendered
    assert "STRIKE_IRONCLAD cost=1 type=Attack | Deal 6 damage." in rendered
    assert "damage_preview=" not in rendered
    assert "FORGOTTEN_RITUAL cost=1 type=Skill upgraded=true | If you Exhausted a card this turn, gain 4 Energy. Exhaust." in rendered
    assert "DEFEND_IRONCLAD cost=1 type=Skill | Gain 5 Block." in rendered
    assert "block_preview=" not in rendered
    # tooltip 来源改为 localization/eng/*.json（STS2 官方 i18n 文件，含占位符 {Heal}/{Amount}）
    assert "BURNING_BLOOD | At the end of combat, heal" in rendered
    # ARTIFACT_POWER 是 enemy power, 现在在 enemy 行下 inline 渲染（带 (amount) 标注），
    # glossary 不再重复——这是去重优化, prompt 长度可省 ~19%
    assert "ARTIFACT_POWER(2): Negates" in rendered
    # VULNERABLE_POWER 是 player power, 仍在 glossary（player 行不 inline 描述）
    assert "VULNERABLE_POWER: Receive" in rendered
    assert "Exhaust: Removed until the end of combat." in rendered
    assert "[0] BLUDGEON hand[0] target=enemy1 damage=42" in rendered
    assert "hp=64 lethal=false" not in rendered
    assert "[1] DEFEND_IRONCLAD hand[3] target=self block=5" in rendered
    assert "[2] end_turn" in rendered
    assert "end_turn hand_idx=" not in rendered
    assert '"confidence":0.0' in rendered
    assert '"action_scores"' not in rendered


def test_render_legal_actions_shows_blocked_target_lethality() -> None:
    state = {
        "run": {"act": 1, "floor": 4},
        "player": {"character": "IRONCLAD", "hp": 36, "max_hp": 80, "deck": []},
        "battle": {
            "energy": 3,
            "max_energy": 3,
            "round_number": 3,
            "hand": [
                {
                    "id": "STRIKE_IRONCLAD",
                    "cost": 1,
                    "type": "Attack",
                    "description": "Deal 6 damage.",
                    "preview_damage_per_target": {1: 6},
                },
            ],
            "enemies": [
                {
                    "id": "NIBBIT",
                    "combat_id": 1,
                    "hp": 4,
                    "max_hp": 43,
                    "block": 5,
                    "next_move_id": "Buff",
                }
            ],
        },
    }

    rendered = render_state_text(
        state,
        [{"action": "play_card", "card_index": 0, "card_id": "STRIKE_IRONCLAD", "target_id": 1}],
    )

    assert "damage=6 block=5 hp_damage=1" in rendered
    assert "lethal=false" not in rendered


def test_render_state_text_lists_all_pile_cards() -> None:
    state = {
        "run": {"act": 1, "floor": 6},
        "player": {
            "character": "IRONCLAD",
            "hp": 70,
            "max_hp": 80,
            "gold": 125,
            "deck": [],
        },
        "battle": {
            "energy": 3,
            "max_energy": 3,
            "round_number": 2,
            "hand": [],
            "enemies": [],
            "draw_pile_cards": ["BASH", "STRIKE_IRONCLAD"],
            "discard_pile_cards": ["DEFEND_IRONCLAD", {"id": "POMMEL_STRIKE", "is_upgraded": True}],
            "exhaust_pile_cards": ["SLIMED"],
        },
    }

    rendered = render_state_text(state, [{"action": "end_turn"}])

    assert "piles:" in rendered
    assert "draw=2 discard=2 exhaust=1" in rendered
    assert "draw_cards: BASH, STRIKE_IRONCLAD" in rendered
    assert "discard_cards: DEFEND_IRONCLAD, POMMEL_STRIKE+" in rendered
    assert "exhaust_cards: SLIMED" in rendered


def test_render_state_text_prefers_runtime_relic_and_potion_descriptions() -> None:
    state = {
        "run": {"act": 1, "floor": 9},
        "player": {
            "character": "IRONCLAD",
            "hp": 5,
            "max_hp": 80,
            "relics": [
                {
                    "id": "WHETSTONE",
                    "description": "Upon pickup, upgrade 2 random Attacks.",
                }
            ],
            "potions": [
                {
                    "id": "HEART_OF_IRON",
                    "description": "Gain 6 Metallicize.",
                }
            ],
            "deck": [],
        },
        "battle": {"energy": 3, "max_energy": 3, "hand": [], "enemies": []},
    }

    rendered = render_state_text(state, [{"action": "end_turn"}])

    assert "WHETSTONE | Upon pickup, upgrade 2 random Attacks." in rendered
    assert "HEART_OF_IRON | Gain 6 Metallicize." in rendered


def test_render_state_text_includes_runtime_power_and_intent_tooltips() -> None:
    state = {
        "run": {"act": 1, "floor": 17},
        "player": {"character": "IRONCLAD", "hp": 4, "max_hp": 80, "deck": []},
        "battle": {
            "energy": 3,
            "max_energy": 3,
            "round_number": 17,
            "hand": [],
            "enemies": [
                {
                    "id": "WATERFALL_GIANT",
                    "combat_id": 1,
                    "hp": 999999971,
                    "max_hp": 999999999,
                    "block": 0,
                    "next_move_id": "DeathBlow",
                    "powers": [
                        {
                            "id": "STEAM_ERUPTION_POWER",
                            "amount": 57,
                            "description": "When this enemy dies, it explodes next turn for this much damage.",
                        }
                    ],
                    "intents": [
                        {
                            "type": "DeathBlow",
                            "label": "DeathBlow",
                            "damage": 58,
                            "hits": 1,
                            "title": "Death Blow",
                            "description": "Deal 58 damage at end of turn.",
                        }
                    ],
                }
            ],
        },
    }

    rendered = render_state_text(state, [{"action": "end_turn"}])

    assert "Death Blow: Deal 58 damage at end of turn." in rendered
    # STEAM_ERUPTION_POWER 是 enemy power, glossary 去重后不再重复, 现在在 enemy 行下 inline 渲染
    assert "STEAM_ERUPTION_POWER" in rendered
    assert "When this enemy dies, it explodes next turn for this much damage." in rendered


def test_render_event_includes_body_and_option_text() -> None:
    state = {
        "run": {"act": 1, "floor": 1},
        "player": {"character": "IRONCLAD", "hp": 80, "max_hp": 80, "deck": []},
        "event": {
            "event_id": "NEOW",
            "event_name": "Neow",
            "body": "Choose a starting bonus.",
            "options": [
                {"index": 0, "label": "Booming Conch", "text": "Gain Booming Conch."},
                {"index": 1, "label": "Lava Rock", "text": "Gain Lava Rock."},
            ],
        },
    }

    rendered = render_state_text(
        state,
        [{"action": "choose_event_option", "index": 0, "label": "Booming Conch"}],
    )

    assert "event:" in rendered
    assert "body=Choose a starting bonus." in rendered
    assert '[0] label="Booming Conch" text="Gain Booming Conch."' in rendered
    assert 'option_text="Gain Booming Conch."' in rendered


def test_render_legal_actions_shows_self_hp_loss_preview() -> None:
    state = {
        "run": {"act": 1, "floor": 8},
        "player": {
            "character": "IRONCLAD",
            "hp": 9,
            "max_hp": 80,
            "deck": [],
            "potions": [{"id": "FORTIFIER"}],
        },
        "battle": {
            "energy": 1,
            "max_energy": 3,
            "round_number": 4,
            "player": {"powers": [{"id": "CONSTRICT_POWER", "amount": 6}]},
            "hand": [
                {
                    "id": "BLOODLETTING",
                    "cost": 0,
                    "type": "Skill",
                    "description": "Lose 3 HP. Gain 3 Energy.",
                },
            ],
            "enemies": [],
        },
    }

    rendered = render_state_text(
        state,
        [{"action": "play_card", "card_index": 0, "card_id": "BLOODLETTING", "target_id": -1}],
    )

    # localization/eng/powers.json CONSTRICT_POWER.smartDescription
    assert "CONSTRICT_POWER:" in rendered
    assert "end of your turn" in rendered or "end of turn" in rendered
    # localization/eng/potions.json: FORTIFIER.description = "Triple your Block."
    assert "FORTIFIER | Triple your Block." in rendered
    assert "[0] BLOODLETTING hand[0] target=self self_hp_loss=3 self_hp_after=6" in rendered


def test_render_card_removal_selection_marks_purpose_and_priority() -> None:
    state = {
        "run": {"act": 1, "floor": 1},
        "player": {
            "character": "IRONCLAD",
            "hp": 80,
            "max_hp": 80,
            "deck": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "BASH"},
            ],
        },
        "card_select": {
            "screen_type": "DeckGeneric",
            "selected_cards": [],
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "BASH"},
            ],
        },
    }

    rendered = render_state_text(
        state,
        [
            {
                "action": "select_card",
                "index": 0,
                "card_index": 0,
                "card_id": "STRIKE_IRONCLAD",
                "label": "remove_card STRIKE_IRONCLAD",
            },
            {
                "action": "select_card",
                "index": 1,
                "card_index": 1,
                "card_id": "BASH",
                "label": "remove_card BASH",
            },
        ],
    )

    assert "[0] select_card purpose=remove_card card=STRIKE_IRONCLAD choice_idx=0 priority=starter_basic" in rendered
    assert "[1] select_card purpose=remove_card card=BASH choice_idx=1 priority=protected_starter_key" in rendered
    assert "hand_idx=0" not in rendered


def test_render_structured_action_text_groups_targets_by_hand_card() -> None:
    state = {
        "run": {"act": 0, "floor": 0},
        "player": {"character": "IRONCLAD", "hp": 70, "max_hp": 80, "deck": []},
        "battle": {
            "energy": 3,
            "max_energy": 3,
            "round_number": 1,
            "hand": [
                {"id": "DEFEND_IRONCLAD", "cost": 1, "type": "Skill", "description": "Gain 5 Block."},
                {"id": "POMMEL_STRIKE", "cost": 1, "type": "Attack", "description": "Deal 10 damage."},
            ],
            "enemies": [
                {"id": "A", "combat_id": 1, "hp": 10, "max_hp": 10},
                {"id": "B", "combat_id": 2, "hp": 10, "max_hp": 10},
            ],
        },
    }
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": -1},
        {"action": "play_card", "card_index": 1, "target_id": 1},
        {"action": "play_card", "card_index": 1, "target_id": 2},
        {"action": "end_turn"},
    ]

    rendered = render_structured_action_state_text(state, legal)

    assert can_render_structured_actions(legal)
    assert "commands:" in rendered
    assert "play_card: use a hand index with legal_target or legal_targets" in rendered
    assert "DEFEND_IRONCLAD cost=1 type=Skill legal_target=self" in rendered
    assert "POMMEL_STRIKE cost=1 type=Attack legal_targets=enemy1,enemy2" in rendered
    assert "play_card hand[0] DEFEND_IRONCLAD" not in rendered
    assert "  [1] play hand[1] POMMEL_STRIKE -> enemy1" not in rendered
    assert '{"action":"play_card","hand_index":HAND,"target_id":ENEMY,"reason":"..."}' in rendered
    assert "Do not output `action_index`" not in rendered


def test_render_card_reward_actions_include_candidate_details() -> None:
    state = {
        "run": {"act": 1, "floor": 2},
        "player": {
            "character": "IRONCLAD",
            "hp": 75,
            "max_hp": 80,
            "gold": 112,
            "deck": [],
            "relics": [{"id": "BURNING_BLOOD"}],
        },
        "card_reward": {
            "can_skip": True,
            "cards": [
                {
                    "index": 0,
                    "id": "BULLY",
                    "cost": 1,
                    "type": "attack",
                    "rarity": "common",
                    "description": "Deal 9 damage. Apply 1 Vulnerable.",
                    "keywords": ["Vulnerable"],
                },
                {
                    "index": 1,
                    "id": "BLOOD_WALL",
                    "cost": 2,
                    "type": "skill",
                    "rarity": "uncommon",
                    "description": "Gain 14 Block.",
                    "keywords": ["Block"],
                },
            ],
        },
    }

    rendered = render_state_text(
        state,
        [
            {"action": "select_card_reward", "index": 0, "card_index": 0, "card_id": "BULLY"},
            {"action": "select_card_reward", "index": 1, "card_index": 1, "card_id": "BLOOD_WALL"},
            {"action": "skip_card_reward"},
        ],
    )

    assert '[0] select_card_reward card=BULLY cost=1 type=attack rarity=common keywords=Vulnerable | Deal 9 damage. Apply 1 Vulnerable.' in rendered
    assert '[1] select_card_reward card=BLOOD_WALL cost=2 type=skill rarity=uncommon keywords=Block | Gain 14 Block.' in rendered
    assert "select_card_reward card=BULLY hand_idx=0" not in rendered
    # tooltip 来源：Vulnerable -> powers.json VULNERABLE_POWER；Block -> static_hover_tips.json BLOCK
    assert "Vulnerable: Receive" in rendered
    assert "Block: Until next turn, prevents damage." in rendered


def test_render_state_text_uses_choice_index_for_combat_card_select() -> None:
    state = {
        "run": {"act": 1, "floor": 11},
        "player": {
            "character": "IRONCLAD",
            "hp": 21,
            "max_hp": 84,
            "block": 9,
            "energy": 1,
            "max_energy": 3,
        },
        "state_type": "card_select",
        "card_select": {
            "screen_type": "combat_select",
            "cards": [
                {"index": 0, "id": "HAVOC", "cost": 1, "type": "skill"},
                {"index": 1, "id": "STRIKE_IRONCLAD", "cost": 1, "type": "attack"},
            ],
        },
    }

    rendered = render_state_text(
        state,
        [
            {"action": "combat_select_card", "index": 0, "card_index": 0, "label": "Havoc"},
            {"action": "combat_select_card", "index": 1, "card_index": 2, "label": "Strike"},
        ],
    )

    assert "[1] combat_select_card card=STRIKE_IRONCLAD choice_idx=1 source_hand_idx=2" in rendered
    assert "[1] combat_select_card card=Strike choice_idx=2" not in rendered


def test_inject_experience_context_uses_review_lessons(tmp_path, monkeypatch) -> None:
    lessons = tmp_path / "lessons.jsonl"
    lessons.write_text(
        '{"tags":["vulnerable","attack"],'
        '"applies_when":"hand can apply Vulnerable and attack",'
        '"advice":"apply Vulnerable before the largest attack",'
        '"avoid":"do not attack before the debuff",'
        '"confidence":0.9}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("STS2_LLM_EXPERIENCE_PATH", str(lessons))

    rendered = inject_experience_context(
        "run: char=IRONCLAD\n"
        "strategy_context:\n"
        "  rule: current state and legal_actions override this context.\n"
        "player: hp=70/80 block=0 energy=3/3 powers=-\n"
        "hand:\n"
        "  [0] BASH cost=2 type=attack | Deal 8 damage. Apply 2 Vulnerable.\n"
        "  [1] STRIKE_IRONCLAD cost=1 type=attack | Deal 6 damage.\n",
        limit=2,
    )

    assert "experience:" in rendered
    assert "apply Vulnerable before the largest attack" in rendered
    assert rendered.index("experience:") < rendered.index("player:")
