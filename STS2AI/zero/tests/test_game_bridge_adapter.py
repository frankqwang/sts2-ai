from __future__ import annotations

import unittest

from zero.adapters.game_bridge import convert_game_bridge_state


class GameBridgeAdapterTests(unittest.TestCase):
    def test_convert_combat_state_uses_battle_and_run_subtrees(self) -> None:
        raw = {
            "state_type": "monster",
            "terminal": False,
            "run_outcome": None,
            "run": {"act": 1, "floor": 6},
            "player": {
                "hp": 70,
                "current_hp": 70,
                "max_hp": 80,
                "block": 2,
                "gold": 99,
                "energy": 3,
                "draw_pile_count": 7,
                "discard_pile_count": 2,
                "exhaust_pile_count": 1,
                "open_potion_slots": 2,
                "deck": [{"id": "BASH", "upgrades": 1}, {"id": "STRIKE_IRONCLAD"}],
                "relics": [{"id": "burning_blood"}],
                "potions": [{"id": "weak_potion"}],
            },
            "enemies": [
                {
                    "monster_id": "SLIME",
                    "target_id": 1,
                    "hp": 20,
                    "current_hp": 20,
                    "max_hp": 40,
                    "block": 0,
                    "is_alive": True,
                    "intent_type": "attack",
                    "buffs": [{"id": "StrengthPower", "amount": 2}],
                }
            ],
            "battle": {
                "energy": 3,
                "max_energy": 3,
                "draw_pile_cards": ["a", "b", "c", "d"],
                "discard_pile_cards": ["x", "y"],
                "exhaust_pile_cards": ["z"],
                "player": {
                    "block": 4,
                    "powers": [{"id": "StrengthPower", "amount": 2}],
                    "stars": 1,
                },
                "hand": [
                    {
                        "index": 0,
                        "id": "STRIKE_IRONCLAD",
                        "cost": 1,
                        "type": "attack",
                        "is_upgraded": False,
                        "can_play": True,
                        "requires_target": True,
                    }
                ],
            },
            "legal_actions": [
                {
                    "action": "play_card",
                    "index": 0,
                    "card_index": 0,
                    "target_id": 1,
                    "is_enabled": True,
                    "label": "STRIKE_IRONCLAD",
                }
            ],
        }

        state = convert_game_bridge_state(
            raw,
            fallback_encounter_id="CHOMPERS_NORMAL",
            fallback_seed="seed-1",
            fallback_encounter_class="elite",
        )

        self.assertEqual(state.context.act, 1)
        self.assertEqual(state.context.floor, 6)
        self.assertEqual(state.context.encounter_id, "CHOMPERS_NORMAL")
        self.assertEqual(state.context.encounter_class, "elite")
        self.assertEqual(len(state.hand), 1)
        self.assertEqual(state.hand[0].card_id, "STRIKE_IRONCLAD")
        self.assertEqual(state.piles.draw_pile_size, 4)
        self.assertEqual(state.piles.draw_cards, ["a", "b", "c", "d"])
        self.assertEqual(state.piles.discard_cards, ["x", "y"])
        self.assertEqual(state.piles.exhaust_cards, ["z"])
        self.assertEqual(state.context.deck_cards, ["BASH+1", "STRIKE_IRONCLAD"])
        self.assertEqual(state.player.buffs["StrengthPower"], 2.0)
        self.assertEqual(state.legal_actions[0].card_id, "STRIKE_IRONCLAD")
        self.assertIn("0", state.legal_actions[0].action_id)

    def test_convert_combat_state_infers_basic_card_numbers_and_keeps_end_turn_clean(self) -> None:
        raw = {
            "state_type": "monster",
            "terminal": False,
            "run_outcome": None,
            "run": {"act": 1, "floor": 6},
            "player": {
                "hp": 70,
                "current_hp": 70,
                "max_hp": 80,
                "gold": 99,
                "energy": 3,
            },
            "battle": {
                "energy": 3,
                "max_energy": 3,
                "player": {"block": 0},
                "hand": [
                    {
                        "index": 0,
                        "id": "STRIKE_IRONCLAD",
                        "cost": 1,
                        "type": "attack",
                        "can_play": True,
                        "requires_target": True,
                    },
                    {
                        "index": 1,
                        "id": "DEFEND_IRONCLAD",
                        "cost": 1,
                        "type": "skill",
                        "can_play": True,
                    },
                ],
            },
            "enemies": [
                {
                    "monster_id": "SLIME",
                    "target_id": 1,
                    "hp": 20,
                    "current_hp": 20,
                    "max_hp": 40,
                    "block": 0,
                    "is_alive": True,
                    "intent_type": "attack",
                }
            ],
            "legal_actions": [
                {
                    "action": "play_card",
                    "index": 0,
                    "card_index": 0,
                    "target_id": 1,
                    "is_enabled": True,
                    "label": "STRIKE_IRONCLAD",
                },
                {
                    "action": "end_turn",
                    "index": 1,
                    "card_index": 0,
                    "is_enabled": True,
                },
            ],
        }

        state = convert_game_bridge_state(raw, fallback_encounter_id="SLIME", fallback_seed="seed-1")

        self.assertEqual(state.hand[0].damage_now, 6.0)
        self.assertEqual(state.hand[1].block_now, 5.0)
        self.assertEqual(state.legal_actions[0].damage_now, 6.0)
        self.assertEqual(state.legal_actions[1].card_id, "")
        self.assertEqual(state.legal_actions[1].cost_now, 0.0)
        self.assertEqual(state.legal_actions[1].damage_now, 0.0)
        self.assertEqual(state.legal_actions[1].block_now, 0.0)
        self.assertEqual(state.legal_actions[1].tags, ["end_turn", "index:1"])

    def test_convert_visible_combat_state_uses_top_level_player_hand_and_battle_round(self) -> None:
        raw = {
            "state_type": "monster",
            "terminal": False,
            "run_outcome": None,
            "run": {"act": 1, "floor": 25},
            "player": {
                "hp": 60,
                "current_hp": 60,
                "max_hp": 60,
                "block": 0,
                "gold": 99,
                "energy": 3,
                "hand": [
                    {
                        "index": 0,
                        "id": "TRUE_GRIT",
                        "cost": 1,
                        "type": "skill",
                        "can_play": True,
                    },
                    {
                        "index": 1,
                        "id": "BLUDGEON",
                        "cost": 3,
                        "type": "attack",
                        "can_play": True,
                        "requires_target": True,
                    },
                ],
            },
            "battle": {
                "round": 2,
                "turn": "player",
                "is_play_phase": True,
                "energy": 3,
                "max_energy": 3,
                "player": {"block": 0},
                "enemies": [
                    {
                        "monster_id": "CHOMPER_A",
                        "target_id": 1,
                        "hp": 61,
                        "current_hp": 61,
                        "max_hp": 61,
                        "block": 0,
                        "is_alive": True,
                        "intent_type": "attack",
                    }
                ],
            },
            "legal_actions": [
                {
                    "action": "play_card",
                    "card_index": 1,
                    "target_id": 1,
                    "is_enabled": True,
                },
                {
                    "action": "end_turn",
                    "is_enabled": True,
                },
            ],
        }

        state = convert_game_bridge_state(raw, fallback_encounter_id="CHOMPERS_NORMAL", fallback_seed="seed-1")

        self.assertEqual(state.context.metadata["round_number_raw"], 2)
        self.assertEqual([card.card_id for card in state.hand], ["TRUE_GRIT", "BLUDGEON"])
        self.assertEqual(state.legal_actions[0].card_id, "BLUDGEON")
        self.assertEqual(state.legal_actions[0].target_id, "1")

    def test_convert_hand_select_uses_selection_cards_for_combat_select_actions(self) -> None:
        raw = {
            "state_type": "hand_select",
            "terminal": False,
            "run_outcome": None,
            "run": {"act": 1, "floor": 25},
            "player": {
                "hp": 60,
                "current_hp": 60,
                "max_hp": 60,
                "energy": 1,
                "hand": [
                    {"index": 0, "id": "TREMBLE", "cost": 1, "type": "skill"},
                    {"index": 1, "id": "DARK_EMBRACE", "cost": 2, "type": "power"},
                ],
            },
            "battle": {
                "round": 1,
                "turn": "player",
                "is_play_phase": True,
                "player": {"block": 0},
                "enemies": [],
            },
            "hand_select": {
                "selected_count": 1,
                "max_select": 3,
                "can_confirm": True,
                "cards": [
                    {"index": 0, "id": "TREMBLE", "type": "skill", "cost": 1},
                    {"index": 1, "id": "DARK_EMBRACE", "type": "power", "cost": 2},
                ],
                "selected_cards": [
                    {"index": 0, "id": "TREMBLE", "type": "skill", "cost": 1},
                ],
            },
            "legal_actions": [
                {"action": "combat_select_card", "card_index": 1, "is_enabled": True},
                {"action": "combat_confirm_selection", "is_enabled": True},
            ],
        }

        state = convert_game_bridge_state(raw, fallback_encounter_id="CHOMPERS_NORMAL", fallback_seed="seed-1")

        self.assertEqual(state.context.metadata["submenu_selected_count"], 1)
        self.assertEqual(state.context.metadata["submenu_max_select"], 3)
        self.assertEqual(state.context.metadata["submenu_remaining_slots"], 2)
        self.assertTrue(state.context.metadata["submenu_can_confirm"])
        self.assertEqual(state.legal_actions[0].card_id, "DARK_EMBRACE")


if __name__ == "__main__":
    unittest.main()
