from __future__ import annotations

import unittest

import torch

from zero.config import EncoderConfig
from zero.domain import (
    BattleState,
    EnemyState,
    FightLabel,
    HandCardState,
    LegalAction,
    PileSummary,
    PlayerState,
    StaticContext,
    TrainingSample,
    TransitionDelta,
)
from zero.features import BatchCollator
from zero.model import FlatPolicyOutput, HierarchicalPolicyOutput, ZeroNet


def make_sample() -> TrainingSample:
    state = BattleState(
        player=PlayerState(hp=70.0, max_hp=80.0, block=4.0, energy=3.0),
        enemies=[
            EnemyState(enemy_id="slime", hp=25.0, max_hp=40.0, block=0.0, intent_id="attack"),
            EnemyState(enemy_id="louse", hp=10.0, max_hp=10.0, block=2.0, intent_id="defend"),
        ],
        hand=[
            HandCardState(card_id="strike", cost_now=1.0, damage_now=6.0),
            HandCardState(card_id="defend", cost_now=1.0, block_now=5.0),
        ],
        piles=PileSummary(draw_pile_size=4, discard_pile_size=2),
        context=StaticContext(character_id="IRONCLAD", act=1, floor=6, encounter_class="elite"),
        legal_actions=[
            LegalAction(action_id="play_strike", action_type="play_card", card_id="strike"),
            LegalAction(action_id="play_defend", action_type="play_card", card_id="defend"),
            LegalAction(action_id="end_turn", action_type="end_turn"),
        ],
    )
    return TrainingSample(
        sample_id="sample1",
        run_id="run1",
        fight_id="fight1",
        step_idx=0,
        state=state,
        history=[],
        legal_actions=state.legal_actions,
        behavior_action_index=0,
        delta=TransitionDelta(),
        fight_label=FightLabel(fight_win=1.0, enemy_hp_fraction_dealt=0.8, self_hp_fraction_remaining=0.7),
    )


class ModelForwardTests(unittest.TestCase):
    def test_network_forward_shapes(self) -> None:
        for variant in ("stateless", "history_transformer", "recurrent_gru", "hierarchical_intent"):
            with self.subTest(variant=variant):
                config = EncoderConfig(model_variant=variant)
                collator = BatchCollator(config)
                batch = collator.collate([make_sample(), make_sample()])
                model = ZeroNet(config)
                output = model(batch)

                if variant == "hierarchical_intent":
                    self.assertIsInstance(output, HierarchicalPolicyOutput)
                    self.assertEqual(tuple(output.intent_logits.shape), (2, config.intent_vocab_size))
                    self.assertEqual(tuple(output.intent_value.shape), (2,))
                    self.assertEqual(tuple(output.action_logits.shape), (2, config.intent_vocab_size, 3))
                    self.assertEqual(tuple(output.action_value.shape), (2, config.intent_vocab_size, 3))
                    self.assertEqual(tuple(output.death_risk_2t.shape), (2, config.intent_vocab_size))
                    self.assertEqual(tuple(output.next_turn_power.shape), (2, config.intent_vocab_size))
                    self.assertEqual(tuple(output.setup_value.shape), (2, config.intent_vocab_size))
                    self.assertEqual(tuple(output.confirm_now_logit.shape), (2, config.intent_vocab_size))
                    self.assertTrue(torch.isfinite(output.intent_logits).all().item())
                    self.assertTrue(torch.isfinite(output.intent_value).all().item())
                else:
                    self.assertIsInstance(output, FlatPolicyOutput)
                    self.assertEqual(tuple(output.state_value.shape), (2,))
                    self.assertEqual(tuple(output.action_logits.shape), (2, 3))
                    self.assertEqual(tuple(output.action_value.shape), (2, 3))
                    self.assertEqual(tuple(output.death_risk_2t.shape), (2,))
                    self.assertEqual(tuple(output.next_turn_power.shape), (2,))
                    self.assertEqual(tuple(output.setup_value.shape), (2,))
                    self.assertEqual(tuple(output.confirm_now_logit.shape), (2,))
                    self.assertTrue(torch.isfinite(output.state_value).all().item())
                self.assertTrue(torch.isfinite(output.action_value).all().item())
                self.assertTrue(torch.isfinite(output.death_risk_2t).all().item())
                self.assertTrue(torch.isfinite(output.next_turn_power).all().item())
                self.assertTrue(torch.isfinite(output.setup_value).all().item())


if __name__ == "__main__":
    unittest.main()
