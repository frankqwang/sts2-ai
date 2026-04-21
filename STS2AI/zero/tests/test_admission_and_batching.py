from __future__ import annotations

import unittest

from zero.buffers.pools import SamplePoolSet, _allocate_capacities, _allocate_counts
from zero.config import EncoderConfig, PoolConfig
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
        search_entries = planner.build_search_entries([sample.clone_for_pool(pool_name="recent_online")])

        self.assertEqual([item.pool_name for item in online_entries], ["recent_online", "rare"])
        self.assertEqual([item.pool_name for item in search_entries], ["search", "rare"])
        self.assertIsNot(online_entries[0], online_entries[1])
        self.assertIsNot(search_entries[0], search_entries[1])

    def test_search_admission_does_not_back_mutate_online_entry(self) -> None:
        planner = SampleAdmissionPlanner()
        sample = make_sample(encounter_class="elite")
        online_entries = planner.build_online_entries([sample])
        labeled_sample = sample.clone_for_pool(
            pool_name="recent_online",
            metadata={"search_priority": 1.5},
        )
        search_entries = planner.build_search_entries([labeled_sample])

        self.assertNotIn("search_priority", online_entries[0].metadata)
        self.assertEqual(search_entries[0].metadata["search_priority"], 1.5)

    def test_online_admission_thins_redundant_no_progress_samples(self) -> None:
        planner = SampleAdmissionPlanner()
        samples = []
        for step_idx in range(12):
            sample = make_sample()
            sample.sample_id = f"sample-{step_idx}"
            sample.step_idx = step_idx
            sample.fight_id = "fight-timeout"
            sample.behavior_action_id = "end_turn" if step_idx % 2 == 0 else "play_strike"
            sample.step_progress_score = 0.0
            sample.metadata["fight_timeout"] = True
            sample.metadata["fight_no_progress_ratio"] = 1.0
            sample.metadata["score_band"] = "normal"
            samples.append(sample)

        entries = planner.build_online_entries(samples)
        self.assertLess(len(entries), len(samples))
        kept_ids = {item.sample_id for item in entries}
        self.assertIn("sample-0", kept_ids)

    def test_dead_enemy_is_masked_out(self) -> None:
        collator = BatchCollator(EncoderConfig())
        batch = collator.collate([make_sample(alive_enemy=False)])
        self.assertEqual(float(batch.enemy_mask[0, 0].item()), 0.0)

    def test_allocate_counts_preserves_small_batch_weights(self) -> None:
        counts = _allocate_counts(
            3,
            [
                ("recent_online", 0.35, 10),
                ("search", 0.25, 10),
                ("rare", 0.20, 10),
                ("reanalyse", 0.10, 10),
                ("legacy", 0.10, 10),
            ],
        )

        self.assertEqual(sum(counts.values()), 3)
        self.assertGreaterEqual(counts["recent_online"], 1)

    def test_allocate_capacities_expand_with_recent_two_iterations(self) -> None:
        capacities = _allocate_capacities(
            target_total=12000,
            weighted_bases=[
                ("recent_online", 0.35, 2048),
                ("search", 0.25, 1024),
                ("rare", 0.20, 256),
                ("reanalyse", 0.10, 1024),
                ("legacy", 0.10, 2048),
            ],
        )

        self.assertEqual(sum(capacities.values()), 12000)
        self.assertGreater(capacities["recent_online"], 2048)
        self.assertGreater(capacities["search"], 1024)

    def test_sample_pool_set_expands_capacity_and_prefers_higher_keep_score(self) -> None:
        pools = SamplePoolSet(PoolConfig())
        capacities = pools.update_capacity_plan(logical_samples=7000)

        self.assertGreaterEqual(sum(capacities.values()), 7000)
        self.assertGreater(capacities["recent_online"], 2048)

        low = make_sample()
        low.sample_id = "low"
        low.keep_score = 0.1
        high = make_sample()
        high.sample_id = "high"
        high.keep_score = 1.5

        pools.add(low.clone_for_pool(pool_name="recent_online"))
        pools.add(high.clone_for_pool(pool_name="recent_online"))

        recent_items = {item.sample_id for item in pools._pools["recent_online"].items()}
        self.assertIn("high", recent_items)

    def test_pool_counters_report_add_reject_and_sample(self) -> None:
        pools = SamplePoolSet(PoolConfig(bucket_capacity=1, search_bucket_capacity=1, rare_bucket_capacity=1))
        low = make_sample()
        low.sample_id = "low"
        low.keep_score = 0.1
        high = make_sample()
        high.sample_id = "high"
        high.keep_score = 1.5

        pools.reset_iteration_counters()
        pools.add(low.clone_for_pool(pool_name="recent_online"))
        pools.add(high.clone_for_pool(pool_name="recent_online"))
        pools.mixed_sample(1)

        counters = pools.iteration_counters()["recent_online"]
        self.assertEqual(counters["attempted_adds"], 2)
        self.assertEqual(counters["accepted_adds"], 2)
        self.assertEqual(counters["replaced_adds"], 1)
        self.assertEqual(counters["evicted_items"], 1)
        self.assertGreaterEqual(counters["sampled_items"], 1)

    def test_sample_serialization_skips_runtime_raw_payloads(self) -> None:
        sample = make_sample()
        sample.state.raw = {"battle": {"hand": [{"id": "strike"}]}}
        sample.legal_actions[0].raw = {"action": "play_card", "index": 0}
        payload = sample.to_dict()

        self.assertNotIn("raw", payload["state"])
        self.assertNotIn("raw", payload["legal_actions"][0])


if __name__ == "__main__":
    unittest.main()
