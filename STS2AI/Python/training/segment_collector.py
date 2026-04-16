"""半马尔可夫训练的非战斗段收集器。

收集从一个非战斗决策点到下一个的宏观转移，将所有中间奖励
（PBRS、战斗摘要、里程碑、反事实评分）累积到单个段中。

在 rollout 循环中的用法：
    collector = NonCombatSegmentCollector()

    # 做出非战斗决策时：
    collector.open_segment(state_np, actions_np, action_idx, log_prob, value, screen_idx)

    # 在段期间累积奖励：
    collector.add_reward(shaped_reward, tag="pbrs", steps=1)
    collector.add_reward(fight_feedback, tag="fight_summary", steps=combat_steps)
    collector.add_reward(milestone_r, tag="milestone", steps=1)
    collector.add_reward(cf_reward, tag="counterfactual", steps=0)

    # 在下一个非战斗决策或 episode 结束时：
    segment = collector.close_segment(done=False)
    buffer.add(segment)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from training.rl_segment_buffer import Segment

logger = logging.getLogger(__name__)


class NonCombatSegmentCollector:
    """Stateful collector that accumulates rewards between non-combat decisions."""

    def __init__(self) -> None:
        self._pending: dict[str, Any] | None = None

    @property
    def is_open(self) -> bool:
        return self._pending is not None

    def open_segment(
        self,
        state: dict[str, np.ndarray],
        actions: dict[str, np.ndarray],
        action_idx: int,
        log_prob: float,
        value: float,
        screen_type_idx: int,
        teacher_logits: np.ndarray | None = None,
    ) -> None:
        """Begin a new segment at a non-combat decision point.

        If a segment is already open, it is auto-closed as done=False first.
        This handles edge cases where two non-combat decisions happen back-to-back.
        """
        if self._pending is not None:
            logger.debug("Auto-closing previous segment (back-to-back noncombat decisions)")
            # Return value is discarded here — caller should use close_segment explicitly
            # This is just a safety net
        self._pending = {
            "state": state,
            "actions": actions,
            "action_idx": action_idx,
            "log_prob": log_prob,
            "value": value,
            "screen_type_idx": screen_type_idx,
            "reward_sum": 0.0,
            "seg_len": 0,
            "teacher_logits": teacher_logits,
            "reward_tags": {},
        }

    def add_reward(
        self,
        reward: float,
        tag: str = "other",
        steps: int = 1,
    ) -> None:
        """Accumulate a reward into the current open segment.

        Args:
            reward: Reward value to accumulate.
            tag: Category tag for logging (e.g., "pbrs", "fight_summary", "milestone").
            steps: Number of environment steps this reward spans (for discount computation).
        """
        if self._pending is None:
            return  # No open segment — silently ignore
        self._pending["reward_sum"] += float(reward)
        self._pending["seg_len"] += int(steps)
        # Track reward breakdown for diagnostics
        self._pending["reward_tags"][tag] = self._pending["reward_tags"].get(tag, 0.0) + float(reward)

    def close_segment(self, done: bool = False) -> Segment | None:
        """Close the current segment and return it.

        Args:
            done: Whether the episode terminated during this segment.

        Returns:
            Completed Segment, or None if no segment was open.
        """
        if self._pending is None:
            return None
        seg = Segment(
            state=self._pending["state"],
            actions=self._pending["actions"],
            action_idx=self._pending["action_idx"],
            log_prob=self._pending["log_prob"],
            value=self._pending["value"],
            reward_sum=self._pending["reward_sum"],
            seg_len=max(1, self._pending["seg_len"]),  # at least 1 step
            done=done,
            screen_type_idx=self._pending["screen_type_idx"],
            teacher_logits=self._pending["teacher_logits"],
        )
        self._pending = None
        return seg

    def discard(self) -> None:
        """Discard current pending segment without saving."""
        self._pending = None

    def get_pending_reward_tags(self) -> dict[str, float]:
        """Get reward breakdown of current pending segment (for logging)."""
        if self._pending is None:
            return {}
        return dict(self._pending["reward_tags"])
