from __future__ import annotations

import unittest

import torch

from zero.config import LossWeights
from zero.model.losses import compute_losses
from zero.model.network import FlatPolicyOutput, HierarchicalPolicyOutput


def _make_batch(*, fight_quality_score: float = 0.8) -> object:
    return type(
        "Batch",
        (),
        {
            "action_mask": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
            "behavior_action_index": torch.tensor([0], dtype=torch.long),
            "active_intent": torch.tensor([0], dtype=torch.long),
            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
            "fight_quality_score": torch.tensor([fight_quality_score], dtype=torch.float32),
            "behavior_ce_scale": torch.tensor([1.0], dtype=torch.float32),
            "fight_targets": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            "ppo_return": torch.tensor([0.0], dtype=torch.float32),
            "ppo_advantage": torch.tensor([0.0], dtype=torch.float32),
            "turn_start_mask": torch.tensor([1.0], dtype=torch.float32),
            "old_intent_logprob": torch.tensor([0.0], dtype=torch.float32),
            "turn_return": torch.tensor([0.0], dtype=torch.float32),
            "turn_advantage": torch.tensor([0.0], dtype=torch.float32),
            "chosen_action_future_targets": torch.zeros((1, 3), dtype=torch.float32),
            "submenu_confirm_target": torch.tensor([0.0], dtype=torch.float32),
            "submenu_has_confirm": torch.tensor([0.0], dtype=torch.float32),
        },
    )()


def _make_hier_output() -> HierarchicalPolicyOutput:
    return HierarchicalPolicyOutput(
        state_value=torch.tensor([0.0], dtype=torch.float32),
        intent_logits=torch.tensor([[0.2, -0.3, -0.5, -0.7]], dtype=torch.float32),
        intent_value=torch.tensor([0.0], dtype=torch.float32),
        action_logits=torch.tensor([[[0.2, -0.3, float("-inf")], [0.0, 0.0, float("-inf")], [0.0, 0.0, float("-inf")], [0.0, 0.0, float("-inf")]]], dtype=torch.float32),
        action_value=torch.zeros((1, 4, 3), dtype=torch.float32),
        death_risk_2t=torch.zeros((1, 4), dtype=torch.float32),
        next_turn_power=torch.zeros((1, 4), dtype=torch.float32),
        setup_value=torch.zeros((1, 4), dtype=torch.float32),
        confirm_now_logit=torch.zeros((1, 4), dtype=torch.float32),
    )


def _make_flat_output() -> FlatPolicyOutput:
    return FlatPolicyOutput(
        state_value=torch.tensor([0.0], dtype=torch.float32),
        action_logits=torch.tensor([[2.0, -1.0, float("-inf")]], dtype=torch.float32),
        action_value=torch.tensor([[3.0, -2.0, 0.0]], dtype=torch.float32),
        death_risk_2t=torch.zeros((1,), dtype=torch.float32),
        next_turn_power=torch.zeros((1,), dtype=torch.float32),
        setup_value=torch.zeros((1,), dtype=torch.float32),
        confirm_now_logit=torch.zeros((1,), dtype=torch.float32),
    )


class LossesTests(unittest.TestCase):
    def test_policy_loss_ignores_masked_invalid_actions(self) -> None:
        batch = _make_batch()
        output = _make_hier_output()

        losses = compute_losses(output, batch, LossWeights())

        self.assertTrue(torch.isfinite(losses.policy))
        self.assertTrue(torch.isfinite(losses.policy_align))
        self.assertTrue(torch.isfinite(losses.total))

    def test_policy_loss_scales_with_behavior_ce_weight(self) -> None:
        batch = _make_batch()
        output = _make_hier_output()
        baseline = compute_losses(output, batch, LossWeights(policy_behavior_ce_weight=1.0))
        zeroed = compute_losses(output, batch, LossWeights(policy_behavior_ce_weight=0.0))
        self.assertGreater(float(baseline.policy), float(zeroed.policy))
        self.assertEqual(float(zeroed.policy), 0.0)

    def test_policy_align_penalizes_policy_when_it_disagrees_with_action_value_teacher(self) -> None:
        batch = _make_batch()
        aligned_output = _make_flat_output()
        misaligned_output = FlatPolicyOutput(
            state_value=torch.tensor([0.0], dtype=torch.float32),
            action_logits=torch.tensor([[-1.0, 2.0, float("-inf")]], dtype=torch.float32),
            action_value=torch.tensor([[3.0, -2.0, 0.0]], dtype=torch.float32),
            death_risk_2t=torch.zeros((1,), dtype=torch.float32),
            next_turn_power=torch.zeros((1,), dtype=torch.float32),
            setup_value=torch.zeros((1,), dtype=torch.float32),
            confirm_now_logit=torch.zeros((1,), dtype=torch.float32),
        )

        weights = LossWeights(policy=0.0, value=0.0, delta=0.0, future_summary=0.0)
        aligned = compute_losses(aligned_output, batch, weights)
        misaligned = compute_losses(misaligned_output, batch, weights)

        self.assertLess(float(aligned.policy_align), float(misaligned.policy_align))


if __name__ == "__main__":
    unittest.main()
