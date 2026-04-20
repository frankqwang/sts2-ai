from __future__ import annotations

import unittest

from zero.config import EncoderConfig
from zero.domain import (
    BattleState,
    EnemyState,
    HandCardState,
    LegalAction,
    PileSummary,
    PlayerState,
    RawTransition,
    StaticContext,
)
from zero.orchestration.sample_builder import SampleBuilder


def make_state(*, hp: float, enemy_hp: float, step: int, terminal: bool = False, outcome: str = "") -> BattleState:
    return BattleState(
        player=PlayerState(hp=hp, max_hp=80.0, block=5.0, energy=3.0),
        enemies=[EnemyState(enemy_id="slime", hp=enemy_hp, max_hp=40.0, block=0.0, intent_id="attack")],
        hand=[HandCardState(card_id="strike", cost_now=1.0, damage_now=6.0)],
        piles=PileSummary(draw_pile_size=5 - step, discard_pile_size=step),
        context=StaticContext(character_id="IRONCLAD", act=1, floor=step + 1, encounter_class="normal"),
        legal_actions=[
            LegalAction(action_id="play_strike", action_type="play_card", card_id="strike"),
            LegalAction(action_id="end_turn", action_type="end_turn"),
        ],
        terminal=terminal,
        run_outcome=outcome,
    )


class SampleBuilderTests(unittest.TestCase):
    def test_build_samples_with_history_and_labels(self) -> None:
        builder = SampleBuilder(EncoderConfig(history_steps=4))
        state0 = make_state(hp=80.0, enemy_hp=40.0, step=0)
        state1 = make_state(hp=78.0, enemy_hp=34.0, step=1)
        state2 = make_state(hp=78.0, enemy_hp=0.0, step=2, terminal=True, outcome="victory")
        transitions = [
            RawTransition(
                run_id="run1",
                fight_id="fight1",
                step_idx=0,
                seed="seed",
                action_index=0,
                state=state0,
                action=state0.legal_actions[0],
                next_state=state1,
                done=False,
                fight_outcome="",
                run_outcome="",
                metadata={"top2_gap": 0.1},
            ),
            RawTransition(
                run_id="run1",
                fight_id="fight1",
                step_idx=1,
                seed="seed",
                action_index=0,
                state=state1,
                action=state1.legal_actions[0],
                next_state=state2,
                done=True,
                fight_outcome="victory",
                run_outcome="victory",
                metadata={"top2_gap": 0.1},
            ),
        ]

        samples = builder.build(transitions)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].fight_label.fight_win, 1.0)
        self.assertAlmostEqual(samples[1].history[0].delta.self_hp, -0.025)
        self.assertTrue(samples[0].bucket_key.startswith("combat|A1_"))
        self.assertEqual(samples[1].behavior_action_index, 0)
        self.assertGreater(samples[0].keep_score, 0.0)
        self.assertAlmostEqual(samples[0].metadata["uncertainty_target"], 0.45)

    def test_filters_empty_legal_actions(self) -> None:
        builder = SampleBuilder(EncoderConfig(history_steps=4))
        state0 = make_state(hp=80.0, enemy_hp=40.0, step=0)
        state0.legal_actions = []
        state1 = make_state(hp=80.0, enemy_hp=40.0, step=1)

        samples = builder.build(
            [
                RawTransition(
                    run_id="run1",
                    fight_id="fight1",
                    step_idx=0,
                    seed="seed",
                    action_index=0,
                    state=state0,
                    action=LegalAction(action_id="noop", action_type="end_turn"),
                    next_state=state1,
                    done=False,
                    fight_outcome="",
                    run_outcome="",
                )
            ]
        )

        self.assertEqual(samples, [])


if __name__ == "__main__":
    unittest.main()
