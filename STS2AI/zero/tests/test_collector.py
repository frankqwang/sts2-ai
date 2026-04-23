from __future__ import annotations

import math
import unittest

from zero.domain import BattleState, EnemyState, LegalAction, PileSummary, PlayerState, StaticContext
from zero.orchestration.collector import TrajectoryCollector


class _SingleDecisionRuntime:
    def __init__(self) -> None:
        self._state = self.reset()
        self._last_reset_timing = {}
        self._last_step_timing = {}

    def reset(self):
        self._state = BattleState(
            player=PlayerState(hp=50.0, max_hp=50.0, block=0.0, energy=3.0),
            enemies=[EnemyState(enemy_id="enemy", hp=10.0, max_hp=10.0, block=0.0, intent_id="attack")],
            hand=[],
            piles=PileSummary(),
            context=StaticContext(character_id="IRONCLAD", encounter_id="TEST_CASE"),
            legal_actions=[
                LegalAction(action_id="play_a", action_type="play_card", card_id="STRIKE_A"),
                LegalAction(action_id="play_b", action_type="play_card", card_id="STRIKE_B"),
            ],
        )
        self._last_reset_timing = {
            "session_call_duration_s": 0.010,
            "transport_duration_s": 0.008,
            "transport_write_duration_s": 0.001,
            "transport_read_duration_s": 0.006,
            "transport_decode_duration_s": 0.001,
            "state_convert_duration_s": 0.002,
        }
        return self._state

    def step(self, action_index: int):
        self._state = BattleState(
            player=PlayerState(hp=50.0, max_hp=50.0, block=0.0, energy=3.0),
            enemies=[EnemyState(enemy_id="enemy", hp=0.0, max_hp=10.0, block=0.0, intent_id="attack", alive=False)],
            hand=[],
            piles=PileSummary(),
            context=StaticContext(character_id="IRONCLAD", encounter_id="TEST_CASE"),
            legal_actions=[],
            terminal=True,
            run_outcome="victory",
        )
        self._last_step_timing = {
            "session_call_duration_s": 0.020,
            "transport_duration_s": 0.015,
            "transport_write_duration_s": 0.002,
            "transport_read_duration_s": 0.010,
            "transport_decode_duration_s": 0.003,
            "state_convert_duration_s": 0.005,
        }
        return self._state

    def close(self) -> None:
        return None

    def get_last_reset_timing(self):
        return dict(self._last_reset_timing)

    def get_last_step_timing(self):
        return dict(self._last_step_timing)


class _FixedPolicy:
    def infer(self, state: BattleState) -> dict[str, object]:
        return {
            "scores": [2.0, 1.0],
            "action_index": 0,
            "fight_win_prob": 0.0,
            "enemy_hp_fraction_dealt": 0.0,
            "self_hp_fraction_remaining": 0.0,
            "ppo_value": 0.0,
            "policy_collate_duration_s": 0.004,
            "policy_forward_duration_s": 0.012,
            "policy_postprocess_duration_s": 0.003,
        }


class CollectorTests(unittest.TestCase):
    def test_old_logprob_tracks_model_distribution_when_collect_is_greedy(self) -> None:
        collector = TrajectoryCollector()
        episode_events: list[dict[str, object]] = []
        transitions = collector.collect(
            runtime_factory=_SingleDecisionRuntime,
            policy=_FixedPolicy(),
            episodes=1,
            max_steps=1,
            epsilon_greedy=0.0,
            temperature=0.0,
            seed=7,
            on_episode_end=episode_events.append,
        )

        self.assertEqual(len(transitions), 1)
        metadata = transitions[0].metadata
        expected_old_logprob = math.log(math.exp(2.0) / (math.exp(2.0) + math.exp(1.0)))
        self.assertAlmostEqual(float(metadata["old_logprob"]), expected_old_logprob, places=6)
        self.assertAlmostEqual(float(metadata["behavior_logprob"]), 0.0, places=6)
        self.assertEqual(len(episode_events), 1)
        event = episode_events[0]
        self.assertAlmostEqual(float(event["policy_collate_duration_s"]), 0.004, places=6)
        self.assertAlmostEqual(float(event["policy_forward_duration_s"]), 0.012, places=6)
        self.assertAlmostEqual(float(event["policy_postprocess_duration_s"]), 0.003, places=6)
        self.assertAlmostEqual(float(event["runtime_reset_transport_duration_s"]), 0.008, places=6)
        self.assertAlmostEqual(float(event["runtime_step_transport_duration_s"]), 0.015, places=6)
        self.assertAlmostEqual(float(event["runtime_step_state_convert_duration_s"]), 0.005, places=6)


if __name__ == "__main__":
    unittest.main()
