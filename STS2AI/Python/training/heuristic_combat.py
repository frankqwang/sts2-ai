"""Rule-based combat fallback used for debugging and baseline runs."""

from __future__ import annotations

from training.combat_safety import choose_heuristic_combat_action


def heuristic_combat_action(legal: list[dict], state: dict) -> tuple[int, dict]:
    """Pick a deterministic safety-aware combat action."""
    return choose_heuristic_combat_action(legal, state)
