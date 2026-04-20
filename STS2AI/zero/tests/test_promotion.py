from __future__ import annotations

import unittest

from zero.config import EvalConfig
from zero.domain import EvalSummary
from zero.orchestration.promotion import PromotionJudge


class PromotionJudgeTests(unittest.TestCase):
    def test_timeout_rate_blocks_promotion(self) -> None:
        judge = PromotionJudge(
            EvalConfig(
                promote_min_win_rate_gain=-1.0,
                max_timeout_rate=0.0,
                max_no_progress_ratio=1.0,
                max_no_progress_streak=999.0,
            )
        )
        baseline = [
            EvalSummary(
                cohort_name="case_a",
                fight_win_rate=0.0,
                enemy_hp_fraction_dealt=0.1,
                self_hp_fraction_remaining=0.0,
                metadata={"num_episodes": 1, "timeout_rate": 0.0, "avg_no_progress_ratio": 0.1, "avg_max_no_progress_streak": 10.0},
            )
        ]
        current = [
            EvalSummary(
                cohort_name="case_a",
                fight_win_rate=0.0,
                enemy_hp_fraction_dealt=0.2,
                self_hp_fraction_remaining=0.0,
                metadata={"num_episodes": 1, "timeout_rate": 1.0, "avg_no_progress_ratio": 1.0, "avg_max_no_progress_streak": 200.0},
            )
        ]
        decision = judge.decide(candidate_version="student_v0002", current=current, baseline=baseline)
        self.assertFalse(decision.promoted)
        self.assertIn("timeout", decision.reason)

    def test_low_no_progress_can_still_promote(self) -> None:
        judge = PromotionJudge(
            EvalConfig(
                promote_min_win_rate_gain=-1.0,
                max_timeout_rate=0.0,
                max_no_progress_ratio=0.95,
                max_no_progress_streak=128.0,
            )
        )
        baseline = [
            EvalSummary(
                cohort_name="case_a",
                fight_win_rate=0.0,
                enemy_hp_fraction_dealt=0.1,
                self_hp_fraction_remaining=0.0,
                metadata={"num_episodes": 1, "timeout_rate": 0.0, "avg_no_progress_ratio": 0.2, "avg_max_no_progress_streak": 12.0},
            )
        ]
        current = [
            EvalSummary(
                cohort_name="case_a",
                fight_win_rate=1.0,
                enemy_hp_fraction_dealt=1.0,
                self_hp_fraction_remaining=0.5,
                teacher_agreement_at_1=0.5,
                metadata={"num_episodes": 1, "timeout_rate": 0.0, "avg_no_progress_ratio": 0.3, "avg_max_no_progress_streak": 20.0},
            )
        ]
        decision = judge.decide(candidate_version="student_v0002", current=current, baseline=baseline)
        self.assertTrue(decision.promoted)


if __name__ == "__main__":
    unittest.main()
