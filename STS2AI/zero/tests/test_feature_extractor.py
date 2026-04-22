from __future__ import annotations

import unittest

from zero.config import EncoderConfig
from zero.domain import BattleState, EnemyState, HandCardState, LegalAction, PileSummary, PlayerState, StaticContext
from zero.features.extractor import FeatureExtractor, STATIC_NUMERIC_DIM


class FeatureExtractorTests(unittest.TestCase):
    def test_static_numeric_does_not_include_future_target_metadata(self) -> None:
        state = BattleState(
            player=PlayerState(hp=60.0, max_hp=80.0, block=0.0, energy=3.0, buffs={"STRENGTH_POWER": 2.0}),
            enemies=[EnemyState(enemy_id="SLIME", hp=20.0, max_hp=40.0, block=0.0, intent_id="attack")],
            hand=[HandCardState(card_id="STRIKE_IRONCLAD", cost_now=1.0, damage_now=6.0)],
            piles=PileSummary(
                draw_pile_size=4,
                discard_pile_size=2,
                exhaust_pile_size=1,
                draw_cards=["BASH", "STRIKE_IRONCLAD"],
                discard_cards=["DEFEND_IRONCLAD"],
                exhaust_cards=["BURNING_PACT"],
            ),
            context=StaticContext(
                character_id="IRONCLAD",
                act=2,
                floor=25,
                encounter_class="normal",
                encounter_id="CHOMPERS_NORMAL",
                deck_cards=["BASH+1", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
                relics=["BURNING_BLOOD"],
                metadata={
                    "round_number_raw": 3,
                    "combat_start_hp": 60.0,
                    "combat_target_hp_after": 58.0,
                    "combat_target_hp_loss_ratio": 0.0333,
                },
            ),
            legal_actions=[LegalAction(action_id="end_turn", action_type="end_turn")],
        )

        encoded = FeatureExtractor(EncoderConfig()).encode_inference(state, [], state.legal_actions)

        self.assertEqual(len(encoded.static_numeric), STATIC_NUMERIC_DIM)
        self.assertEqual(
            encoded.static_numeric,
            [
                2.0,
                25.0,
                1.0,
                1.0,
                3.0,
                4.0,
                2.0,
                1.0,
                1.0,
                1.0,
            ],
        )


if __name__ == "__main__":
    unittest.main()
