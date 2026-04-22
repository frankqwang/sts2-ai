from __future__ import annotations

import unittest

import torch

from zero.config import LossWeights
from zero.model.losses import compute_losses
from zero.model.network import ZeroNetOutput


def _make_batch(*, fight_quality_score: float = 0.8) -> object:
    return type(
        "Batch",
        (),
        {
            "action_mask": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
            "behavior_action_index": torch.tensor([0], dtype=torch.long),
            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
            "fight_quality_score": torch.tensor([fight_quality_score], dtype=torch.float32),
            "behavior_ce_scale": torch.tensor([1.0], dtype=torch.float32),
            "fight_targets": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            "delta_targets": torch.zeros((1, 14), dtype=torch.float32),
            "uncertainty_target": torch.tensor([0.0], dtype=torch.float32),
        },
    )()


def _make_output() -> ZeroNetOutput:
    return ZeroNetOutput(
        policy_logits=torch.tensor([[0.2, -0.3, float("-inf")]], dtype=torch.float32),
        fight_win=torch.tensor([0.0], dtype=torch.float32),
        enemy_hp_fraction_dealt=torch.tensor([0.0], dtype=torch.float32),
        self_hp_fraction_remaining=torch.tensor([0.0], dtype=torch.float32),
        ppo_value=torch.tensor([0.0], dtype=torch.float32),
        action_value=torch.zeros((1, 3), dtype=torch.float32),
        delta_pred=torch.zeros((1, 14), dtype=torch.float32),
        uncertainty=torch.tensor([0.0], dtype=torch.float32),
    )


class LossesTests(unittest.TestCase):
    def test_policy_loss_ignores_masked_invalid_actions(self) -> None:
        batch = _make_batch()
        output = _make_output()

        losses = compute_losses(output, batch, LossWeights())

        self.assertTrue(torch.isfinite(losses.policy))
        self.assertTrue(torch.isfinite(losses.total))

    def test_policy_loss_scales_with_behavior_ce_weight(self) -> None:
        batch = _make_batch()
        output = _make_output()
        baseline = compute_losses(output, batch, LossWeights(policy_behavior_ce_weight=1.0))
        zeroed = compute_losses(output, batch, LossWeights(policy_behavior_ce_weight=0.0))
        self.assertGreater(float(baseline.policy), float(zeroed.policy))
        self.assertEqual(float(zeroed.policy), 0.0)


if __name__ == "__main__":
    unittest.main()
