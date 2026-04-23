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
        self.assertGreater(samples[0].fight_score, 0.0)
        self.assertGreater(samples[0].sample_weight, 0.1)
        self.assertIn("score_band", samples[0].metadata)

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

    def test_good_fight_samples_are_weighted_above_bad_fight_samples(self) -> None:
        builder = SampleBuilder(EncoderConfig(history_steps=4))
        good_state0 = make_state(hp=80.0, enemy_hp=40.0, step=0)
        good_state1 = make_state(hp=80.0, enemy_hp=28.0, step=1)
        good_state2 = make_state(hp=78.0, enemy_hp=0.0, step=2, terminal=True, outcome="victory")
        bad_state0 = make_state(hp=80.0, enemy_hp=40.0, step=0)
        bad_state1 = make_state(hp=68.0, enemy_hp=40.0, step=1)
        bad_state2 = make_state(hp=52.0, enemy_hp=40.0, step=2, terminal=True, outcome="defeat")
        transitions = [
            RawTransition(
                run_id="run-good",
                fight_id="fight-good",
                step_idx=0,
                seed="seed",
                action_index=0,
                state=good_state0,
                action=good_state0.legal_actions[0],
                next_state=good_state1,
                done=False,
                fight_outcome="",
                run_outcome="",
                metadata={"top2_gap": 0.1, "made_progress": True},
            ),
            RawTransition(
                run_id="run-good",
                fight_id="fight-good",
                step_idx=1,
                seed="seed",
                action_index=0,
                state=good_state1,
                action=good_state1.legal_actions[0],
                next_state=good_state2,
                done=True,
                fight_outcome="victory",
                run_outcome="victory",
                metadata={"top2_gap": 0.1, "made_progress": True},
            ),
            RawTransition(
                run_id="run-bad",
                fight_id="fight-bad",
                step_idx=0,
                seed="seed",
                action_index=1,
                state=bad_state0,
                action=bad_state0.legal_actions[1],
                next_state=bad_state1,
                done=False,
                fight_outcome="",
                run_outcome="",
                metadata={"top2_gap": 0.1, "made_progress": False},
            ),
            RawTransition(
                run_id="run-bad",
                fight_id="fight-bad",
                step_idx=1,
                seed="seed",
                action_index=1,
                state=bad_state1,
                action=bad_state1.legal_actions[1],
                next_state=bad_state2,
                done=True,
                fight_outcome="defeat",
                run_outcome="defeat",
                metadata={"top2_gap": 0.1, "made_progress": False},
            ),
        ]

        samples = builder.build(transitions)
        good_weights = [sample.sample_weight for sample in samples if sample.fight_id == "fight-good"]
        bad_weights = [sample.sample_weight for sample in samples if sample.fight_id == "fight-bad"]
        self.assertGreater(sum(good_weights) / len(good_weights), sum(bad_weights) / len(bad_weights))

    def test_behavior_index_mismatch_drops_sample_instead_of_falling_back_to_zero(self) -> None:
        builder = SampleBuilder(EncoderConfig(history_steps=4))
        state0 = make_state(hp=80.0, enemy_hp=40.0, step=0)
        state1 = make_state(hp=78.0, enemy_hp=34.0, step=1)
        missing_action = LegalAction(action_id="missing_action", action_type="play_card", card_id="ghost")
        samples = builder.build(
            [
                RawTransition(
                    run_id="run1",
                    fight_id="fight1",
                    step_idx=0,
                    seed="seed",
                    action_index=99,
                    state=state0,
                    action=missing_action,
                    next_state=state1,
                    done=False,
                    fight_outcome="",
                    run_outcome="",
                )
            ]
        )
        self.assertEqual(samples, [])

    def test_submenu_confirm_target_marks_explicit_confirm_choice(self) -> None:
        builder = SampleBuilder(EncoderConfig(history_steps=4))
        state0 = BattleState(
            player=PlayerState(hp=60.0, max_hp=80.0, block=0.0, energy=1.0),
            enemies=[EnemyState(enemy_id="slime", hp=30.0, max_hp=40.0, block=0.0, intent_id="attack")],
            hand=[
                HandCardState(card_id="TREMBLE", cost_now=1.0, tags=["skill"]),
                HandCardState(card_id="DARK_EMBRACE", cost_now=2.0, tags=["power"]),
            ],
            piles=PileSummary(draw_pile_size=5, discard_pile_size=1),
            context=StaticContext(
                character_id="IRONCLAD",
                act=1,
                floor=25,
                encounter_class="normal",
                metadata={
                    "state_type": "hand_select",
                    "submenu_selected_count": 1,
                    "submenu_max_select": 3,
                    "submenu_remaining_slots": 2,
                    "submenu_can_confirm": True,
                },
            ),
            legal_actions=[
                LegalAction(action_id="select_dark", action_type="combat_select_card", card_id="DARK_EMBRACE", tags=["power"]),
                LegalAction(action_id="confirm", action_type="combat_confirm_selection"),
            ],
        )
        state1 = make_state(hp=60.0, enemy_hp=30.0, step=1)
        samples = builder.build(
            [
                RawTransition(
                    run_id="run1",
                    fight_id="fight1",
                    step_idx=0,
                    seed="seed",
                    action_index=1,
                    state=state0,
                    action=state0.legal_actions[1],
                    next_state=state1,
                    done=False,
                    fight_outcome="",
                    run_outcome="",
                )
            ]
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].submenu_has_confirm, 1.0)
        self.assertEqual(samples[0].submenu_confirm_target, 1.0)


if __name__ == "__main__":
    unittest.main()
