from __future__ import annotations

import unittest

from zero.config import EncoderConfig
from zero.buffers.pools import _allocate_counts
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
from zero.orchestration.admission import SampleAdmissionPlanner


def make_sample(*, encounter_class: str = "normal", alive_enemy: bool = True) -> TrainingSample:
    state = BattleState(
        player=PlayerState(hp=20.0, max_hp=80.0, block=3.0, energy=3.0),
        enemies=[EnemyState(enemy_id="slime", hp=0.0 if not alive_enemy else 10.0, max_hp=20.0, block=0.0, intent_id="attack", alive=alive_enemy)],
        hand=[HandCardState(card_id="strike", cost_now=1.0, damage_now=6.0)],
        piles=PileSummary(draw_pile_size=4, discard_pile_size=1),
        context=StaticContext(character_id="IRONCLAD", act=1, floor=6, encounter_class=encounter_class),
        legal_actions=[LegalAction(action_id="a0", action_type="play_card", card_id="strike")],
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
        fight_label=FightLabel(fight_win=0.0, enemy_hp_fraction_dealt=0.5, self_hp_fraction_remaining=0.25),
        rare_cohort_tags=["elite"] if encounter_class == "elite" else [],
        keep_score=1.2,
        metadata={"uncertainty_target": 0.9},
    )


class AdmissionAndBatchingTests(unittest.TestCase):
    def test_admission_clones_samples_for_distinct_pools(self) -> None:
        planner = SampleAdmissionPlanner()
        sample = make_sample(encounter_class="elite")
        online_entries = planner.build_online_entries([sample])
        teacher_entries = planner.build_teacher_entries([sample.clone_for_pool(pool_name="recent_online")])

        self.assertEqual([item.pool_name for item in online_entries], ["recent_online", "rare"])
        self.assertEqual([item.pool_name for item in teacher_entries], ["teacher", "rare"])
        self.assertIsNot(online_entries[0], online_entries[1])
        self.assertIsNot(teacher_entries[0], teacher_entries[1])

    def test_teacher_admission_does_not_back_mutate_online_entry(self) -> None:
        planner = SampleAdmissionPlanner()
        sample = make_sample(encounter_class="elite")
        online_entries = planner.build_online_entries([sample])
        labeled_sample = sample.clone_for_pool(
            pool_name="recent_online",
            metadata={"teacher_priority": 1.5},
        )
        teacher_entries = planner.build_teacher_entries([labeled_sample])

        self.assertNotIn("teacher_priority", online_entries[0].metadata)
        self.assertEqual(teacher_entries[0].metadata["teacher_priority"], 1.5)

    def test_dead_enemy_is_masked_out(self) -> None:
        collator = BatchCollator(EncoderConfig())
        batch = collator.collate([make_sample(alive_enemy=False)])
        self.assertEqual(float(batch.enemy_mask[0, 0].item()), 0.0)

    def test_allocate_counts_preserves_small_batch_weights(self) -> None:
        counts = _allocate_counts(
            3,
            [
                ("recent_online", 0.35, 10),
                ("teacher", 0.25, 10),
                ("rare", 0.20, 10),
                ("reanalyse", 0.10, 10),
                ("legacy", 0.10, 10),
            ],
        )

        self.assertEqual(sum(counts.values()), 3)
        self.assertGreaterEqual(counts["recent_online"], 1)


if __name__ == "__main__":
    unittest.main()
