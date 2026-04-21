from __future__ import annotations

import unittest

from zero.config import EncoderConfig, SearchConfig
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
from zero.orchestration.search import SearchQueueBuilder
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

    def test_timeout_like_samples_are_prioritized_for_search(self) -> None:
        builder = SampleBuilder(EncoderConfig(history_steps=4))
        state0 = make_state(hp=60.0, enemy_hp=40.0, step=0)
        state0.context.encounter_class = "elite"
        state1 = make_state(hp=48.0, enemy_hp=40.0, step=1, terminal=True, outcome="defeat")
        transition = RawTransition(
            run_id="run1",
            fight_id="fight1",
            step_idx=0,
            seed="seed",
            action_index=1,
            state=state0,
            action=state0.legal_actions[1],
            next_state=state1,
            done=True,
            fight_outcome="defeat",
            run_outcome="defeat",
            metadata={"top2_gap": 0.0, "made_progress": False},
        )
        sample = builder.build([transition])[0]
        sample.metadata["fight_timeout"] = True
        sample.metadata["fight_no_progress_ratio"] = 1.0
        queue = SearchQueueBuilder(SearchConfig(max_requests_per_iteration=8))
        requests = queue.select([sample])
        self.assertEqual(len(requests), 1)
        self.assertIn("fight_timeout", requests[0].reason_tags)
        self.assertIn("high_no_progress", requests[0].reason_tags)

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

    def test_search_collected_transition_builds_search_label_and_disables_behavior_ce(self) -> None:
        builder = SampleBuilder(EncoderConfig(history_steps=4))
        state0 = make_state(hp=80.0, enemy_hp=40.0, step=0)
        state1 = make_state(hp=78.0, enemy_hp=28.0, step=1, terminal=True, outcome="victory")
        samples = builder.build(
            [
                RawTransition(
                    run_id="run1",
                    fight_id="fight1",
                    step_idx=0,
                    seed="seed",
                    action_index=0,
                    state=state0,
                    action=state0.legal_actions[0],
                    next_state=state1,
                    done=True,
                    fight_outcome="victory",
                    run_outcome="victory",
                    metadata={
                        "top2_gap": 0.1,
                        "made_progress": True,
                        "search_collected": True,
                        "search_source": "search_self_play",
                        "search_policy": [0.2, 0.8],
                        "search_topk": [1, 0],
                        "search_best_action_index": 1,
                        "search_ranking_margin": 0.35,
                        "search_value": 0.91,
                        "search_trace": [{"action_index": 1, "visits": 8}],
                    },
                )
            ]
        )
        self.assertEqual(len(samples), 1)
        self.assertIsNotNone(samples[0].search_label)
        self.assertEqual(samples[0].search_label.best_action_index, 1)
        self.assertEqual(samples[0].metadata["behavior_ce_scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
