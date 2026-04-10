"""Semi-MDP Segment Buffer for non-combat PPO training.

A "segment" spans from one non-combat decision point to the next,
potentially crossing one or more combat encounters. This naturally
propagates combat outcomes back to the non-combat decision that led to them.

GAE uses gamma^seg_len instead of gamma per step, which compresses
the effective horizon and improves credit assignment.
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from rl_encoder_v2 import MAX_ACTIONS, StructuredActions, StructuredState


@dataclass
class Segment:
    """One non-combat macro-transition.

    Spans from a non-combat decision to the next non-combat decision point
    (or episode terminal), accumulating all intermediate rewards.
    """
    state: dict[str, np.ndarray]        # StructuredState at decision point
    actions: dict[str, np.ndarray]      # StructuredActions at decision point
    action_idx: int                      # chosen action index
    log_prob: float                      # log probability of chosen action
    value: float                         # V(s) at decision point
    reward_sum: float                    # accumulated rewards (PBRS + fight_summary + milestone + counterfactual)
    seg_len: int                         # number of environment steps in this segment
    done: bool                           # episode terminated during this segment
    screen_type_idx: int                 # screen type at decision point
    floor_target: float = 0.0            # deck quality target (set after episode)
    teacher_logits: np.ndarray | None = None  # heuristic teacher distribution (Phase 4)


class SegmentRolloutBuffer:
    """Rollout buffer for semi-MDP non-combat segments.

    Same interface as StructuredRolloutBuffer for PPO training compatibility,
    but uses segment-level data with variable-length discount.
    """

    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self.advantages: list[float] = []
        self.returns: list[float] = []

    def add(self, segment: Segment) -> None:
        self.segments.append(segment)

    def set_floor_targets(self, final_floor: float) -> None:
        """Set deck quality target for all segments in the last episode."""
        target = min(final_floor / 20.0, 1.0)
        for i in range(len(self.segments) - 1, -1, -1):
            self.segments[i].floor_target = target
            if i > 0 and self.segments[i - 1].done:
                break

    def compute_gae(
        self,
        gamma: float = 0.999,
        lam: float = 0.95,
        max_discount_steps: int = 32,
    ) -> None:
        """Compute GAE with segment-level discount.

        Uses gamma^min(seg_len, max_discount_steps) instead of gamma per step.
        This compresses the horizon for long segments (e.g., those containing combat).
        """
        n = len(self.segments)
        self.advantages = [0.0] * n
        self.returns = [0.0] * n
        last_gae = 0.0

        for t in reversed(range(n)):
            seg = self.segments[t]
            eff_steps = min(seg.seg_len, max_discount_steps)
            seg_discount = gamma ** eff_steps

            if seg.done:
                next_value = 0.0
                last_gae = 0.0
            elif t + 1 < n:
                next_value = self.segments[t + 1].value
            else:
                next_value = 0.0

            delta = seg.reward_sum + seg_discount * next_value - seg.value
            last_gae = delta + seg_discount * lam * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + seg.value

    def to_tensors(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Convert buffer to tensors for training (same format as StructuredRolloutBuffer)."""
        n = len(self.segments)

        # Stack state tensors
        state_tensors = {}
        if n > 0:
            keys = self.segments[0].state.keys()
            for key in keys:
                arrays = [s.state[key] for s in self.segments]
                if arrays[0].dtype in (np.int64, np.int32):
                    state_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.long)
                elif arrays[0].dtype == bool:
                    state_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.bool)
                else:
                    state_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.float32)

        # Stack action tensors
        action_tensors = {}
        if n > 0:
            keys = self.segments[0].actions.keys()
            for key in keys:
                arrays = [s.actions[key] for s in self.segments]
                if arrays[0].dtype in (np.int64, np.int32):
                    action_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.long)
                elif arrays[0].dtype == bool:
                    action_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.bool)
                else:
                    action_tensors[key] = torch.tensor(np.stack(arrays), dtype=torch.float32)

        # Teacher logits (Phase 4)
        teacher_list = []
        has_any_teacher = False
        for s in self.segments:
            if s.teacher_logits is not None:
                teacher_list.append(s.teacher_logits)
                has_any_teacher = True
            else:
                teacher_list.append(np.zeros(MAX_ACTIONS, dtype=np.float32))

        return {
            "state_tensors": state_tensors,
            "action_tensors": action_tensors,
            "actions": torch.tensor([s.action_idx for s in self.segments], dtype=torch.long),
            "old_log_probs": torch.tensor([s.log_prob for s in self.segments], dtype=torch.float32),
            "advantages": torch.tensor(self.advantages, dtype=torch.float32),
            "returns": torch.tensor(self.returns, dtype=torch.float32),
            "floor_targets": torch.tensor([s.floor_target for s in self.segments], dtype=torch.float32),
            "teacher_logits": torch.tensor(np.stack(teacher_list), dtype=torch.float32) if has_any_teacher else None,
        }

    def clear(self) -> None:
        self.segments.clear()
        self.advantages.clear()
        self.returns.clear()

    def __len__(self) -> int:
        return len(self.segments)

    def get_segment_stats(self) -> dict[str, float]:
        """Get diagnostic statistics about segments."""
        if not self.segments:
            return {}
        seg_lens = [s.seg_len for s in self.segments]
        rewards = [s.reward_sum for s in self.segments]
        screen_counts: dict[int, int] = {}
        for s in self.segments:
            screen_counts[s.screen_type_idx] = screen_counts.get(s.screen_type_idx, 0) + 1
        return {
            "seg_len_mean": float(np.mean(seg_lens)),
            "seg_len_max": max(seg_lens),
            "seg_len_min": min(seg_lens),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "num_segments": len(self.segments),
            "screen_distribution": screen_counts,
        }
