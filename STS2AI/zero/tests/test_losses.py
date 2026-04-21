from __future__ import annotations

import unittest

import torch

from zero.config import LossWeights
from zero.model.losses import compute_losses
from zero.model.network import ZeroNetOutput


class LossesTests(unittest.TestCase):
    def test_teacher_policy_loss_ignores_masked_invalid_actions(self) -> None:
        batch = type(
            "Batch",
            (),
            {
                "teacher_policy_mask": torch.tensor([1.0], dtype=torch.float32),
                "teacher_policy": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                "action_mask": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
                "behavior_action_index": torch.tensor([0], dtype=torch.long),
                "sample_weight": torch.tensor([1.0], dtype=torch.float32),
                "fight_quality_score": torch.tensor([0.8], dtype=torch.float32),
                "behavior_ce_scale": torch.tensor([1.0], dtype=torch.float32),
                "fight_targets": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                "delta_targets": torch.zeros((1, 14), dtype=torch.float32),
                "uncertainty_target": torch.tensor([0.0], dtype=torch.float32),
                "teacher_best_action_index": torch.tensor([0], dtype=torch.long),
                "teacher_ranking_margin": torch.tensor([0.2], dtype=torch.float32),
            },
        )()
        output = ZeroNetOutput(
            policy_logits=torch.tensor([[0.2, -0.3, float("-inf")]], dtype=torch.float32),
            fight_win=torch.tensor([0.0], dtype=torch.float32),
            enemy_hp_fraction_dealt=torch.tensor([0.0], dtype=torch.float32),
            self_hp_fraction_remaining=torch.tensor([0.0], dtype=torch.float32),
            delta_pred=torch.zeros((1, 14), dtype=torch.float32),
            uncertainty=torch.tensor([0.0], dtype=torch.float32),
        )

        losses = compute_losses(output, batch, LossWeights())

        self.assertTrue(torch.isfinite(losses.policy))
        self.assertTrue(torch.isfinite(losses.total))

    def test_teacher_samples_do_not_use_behavior_cross_entropy(self) -> None:
        batch = type(
            "Batch",
            (),
            {
                "teacher_policy_mask": torch.tensor([1.0], dtype=torch.float32),
                "teacher_policy": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
                "action_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                "behavior_action_index": torch.tensor([0], dtype=torch.long),
                "sample_weight": torch.tensor([1.0], dtype=torch.float32),
                "fight_quality_score": torch.tensor([0.8], dtype=torch.float32),
                "behavior_ce_scale": torch.tensor([1.0], dtype=torch.float32),
                "fight_targets": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                "delta_targets": torch.zeros((1, 14), dtype=torch.float32),
                "uncertainty_target": torch.tensor([1.0], dtype=torch.float32),
                "teacher_best_action_index": torch.tensor([1], dtype=torch.long),
                "teacher_ranking_margin": torch.tensor([0.3], dtype=torch.float32),
            },
        )()
        output = ZeroNetOutput(
            policy_logits=torch.tensor([[0.2, 0.9]], dtype=torch.float32),
            fight_win=torch.tensor([0.0], dtype=torch.float32),
            enemy_hp_fraction_dealt=torch.tensor([0.0], dtype=torch.float32),
            self_hp_fraction_remaining=torch.tensor([0.0], dtype=torch.float32),
            delta_pred=torch.zeros((1, 14), dtype=torch.float32),
            uncertainty=torch.tensor([4.0], dtype=torch.float32),
        )

        losses = compute_losses(output, batch, LossWeights())
        self.assertLess(float(losses.policy), 0.7)
        self.assertTrue(torch.isfinite(losses.uncertainty))

    def test_low_quality_rollout_downweights_behavior_ce(self) -> None:
        batch = type(
            "Batch",
            (),
            {
                "teacher_policy_mask": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "teacher_policy": torch.zeros((2, 2), dtype=torch.float32),
                "action_mask": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
                "behavior_action_index": torch.tensor([0, 0], dtype=torch.long),
                "sample_weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
                "fight_quality_score": torch.tensor([0.9, 0.2], dtype=torch.float32),
                "behavior_ce_scale": torch.tensor([1.0, 1.0], dtype=torch.float32),
                "fight_targets": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
                "delta_targets": torch.zeros((2, 14), dtype=torch.float32),
                "uncertainty_target": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "teacher_best_action_index": torch.tensor([-1, -1], dtype=torch.long),
                "teacher_ranking_margin": torch.tensor([0.0, 0.0], dtype=torch.float32),
            },
        )()
        output = ZeroNetOutput(
            policy_logits=torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
            fight_win=torch.tensor([0.0, 0.0], dtype=torch.float32),
            enemy_hp_fraction_dealt=torch.tensor([0.0, 0.0], dtype=torch.float32),
            self_hp_fraction_remaining=torch.tensor([0.0, 0.0], dtype=torch.float32),
            delta_pred=torch.zeros((2, 14), dtype=torch.float32),
            uncertainty=torch.tensor([0.0, 0.0], dtype=torch.float32),
        )
        baseline = compute_losses(output, batch, LossWeights(policy_bad_rollout_ce_scale=1.0))
        downweighted = compute_losses(output, batch, LossWeights(policy_bad_rollout_ce_scale=0.1))
        self.assertLess(float(downweighted.policy), float(baseline.policy))


if __name__ == "__main__":
    unittest.main()
