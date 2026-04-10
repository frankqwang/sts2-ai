"""Save high-quality episode data for offline RL / behavioral cloning.

When an episode reaches a threshold floor (default >=14, i.e. boss fight),
the full trajectory of (state_tensor, action_tensor, action_idx, reward,
screen_type) is saved as a .pt file.

Usage:
    saver = EpisodeDataSaver(output_dir="artifacts/offline_data", min_floor=14)
    # During episode collection, call per-step:
    saver.add_step(state_dict, actions_dict, action_idx, reward, screen_type, log_prob, value)
    # At episode end:
    saver.finish_episode(floor, outcome, combats_won, stats)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeDataSaver:
    """Accumulates episode transitions and saves high-quality ones to disk."""

    def __init__(
        self,
        output_dir: str | Path = "artifacts/offline_data",
        min_floor: int = 14,
        save_victories: bool = True,
        max_files: int = 50000,
    ):
        self.output_dir = Path(output_dir)
        self.min_floor = min_floor
        self.save_victories = save_victories
        self.max_files = max_files
        self._steps: list[dict] = []
        self._combat_steps: list[dict] = []
        self._save_count = 0

    def add_step(
        self,
        state: dict[str, np.ndarray],
        actions: dict[str, np.ndarray],
        action_idx: int,
        reward: float,
        screen_type: str,
        log_prob: float = 0.0,
        value: float = 0.0,
    ) -> None:
        """Record one non-combat decision step."""
        self._steps.append({
            "state": {k: v.copy() for k, v in state.items()},
            "actions": {k: v.copy() for k, v in actions.items()},
            "action_idx": action_idx,
            "reward": reward,
            "screen_type": screen_type,
            "log_prob": log_prob,
            "value": value,
        })

    def add_combat_step(
        self,
        state_tensor: np.ndarray | torch.Tensor,
        action_idx: int,
        reward: float,
        log_prob: float = 0.0,
        value: float = 0.0,
    ) -> None:
        """Record one combat decision step."""
        if isinstance(state_tensor, torch.Tensor):
            state_tensor = state_tensor.cpu().numpy()
        self._combat_steps.append({
            "state": state_tensor.copy(),
            "action_idx": action_idx,
            "reward": reward,
            "log_prob": log_prob,
            "value": value,
        })

    def finish_episode(
        self,
        floor: int,
        outcome: str | None,
        combats_won: int = 0,
        extra_stats: dict[str, Any] | None = None,
    ) -> bool:
        """Decide whether to save this episode and write to disk.

        Returns True if episode was saved.
        """
        is_victory = outcome == "victory"
        should_save = (
            (floor >= self.min_floor)
            or (is_victory and self.save_victories)
        )

        if should_save and self._steps and self._save_count < self.max_files:
            self._write(floor, outcome, combats_won, extra_stats)
            saved = True
        else:
            saved = False

        # Always clear buffer
        self._steps.clear()
        self._combat_steps.clear()
        return saved

    def _write(
        self,
        floor: int,
        outcome: str | None,
        combats_won: int,
        extra_stats: dict[str, Any] | None,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        tag = "VIC" if outcome == "victory" else f"f{floor}"
        fname = f"ep_{ts}_{tag}_{self._save_count:05d}.pt"

        data = {
            "version": 1,
            "floor": floor,
            "outcome": outcome,
            "combats_won": combats_won,
            "num_noncombat_steps": len(self._steps),
            "num_combat_steps": len(self._combat_steps),
            # Non-combat trajectory
            "noncombat_states": [s["state"] for s in self._steps],
            "noncombat_actions": [s["actions"] for s in self._steps],
            "noncombat_action_indices": [s["action_idx"] for s in self._steps],
            "noncombat_rewards": [s["reward"] for s in self._steps],
            "noncombat_screen_types": [s["screen_type"] for s in self._steps],
            "noncombat_log_probs": [s["log_prob"] for s in self._steps],
            "noncombat_values": [s["value"] for s in self._steps],
            # Combat trajectory
            "combat_states": [s["state"] for s in self._combat_steps],
            "combat_action_indices": [s["action_idx"] for s in self._combat_steps],
            "combat_rewards": [s["reward"] for s in self._combat_steps],
            "combat_log_probs": [s["log_prob"] for s in self._combat_steps],
            "combat_values": [s["value"] for s in self._combat_steps],
        }
        if extra_stats:
            data["extra_stats"] = {
                k: v for k, v in extra_stats.items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool, list))
            }

        try:
            torch.save(data, self.output_dir / fname)
            self._save_count += 1
            if self._save_count % 100 == 1:
                logger.info("Saved offline episode %d: %s (%d nc + %d c steps)",
                            self._save_count, fname,
                            len(self._steps), len(self._combat_steps))
        except Exception as e:
            logger.warning("Failed to save episode data: %s", e)

    def clear(self) -> None:
        """Discard accumulated data without saving."""
        self._steps.clear()
        self._combat_steps.clear()
