"""结局词表：胜利/死亡/放弃等结局类型的标准化。"""
from __future__ import annotations

from typing import Any


RUN_OUTCOME_VICTORY = "victory"
RUN_OUTCOME_DEATH = "death"
RUN_OUTCOME_TIMEOUT = "timeout"
RUN_OUTCOME_ERROR = "error"
RUN_OUTCOME_FLOOR_CAP = "floor_cap"


def normalize_run_outcome(value: Any, *, default: str = RUN_OUTCOME_DEATH) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return str(default)
    if text in {"victory", "win", "won"}:
        return RUN_OUTCOME_VICTORY
    if text in {"defeat", "loss", "lose", "death", "dead", "game_over"}:
        return RUN_OUTCOME_DEATH
    if text in {RUN_OUTCOME_TIMEOUT, RUN_OUTCOME_ERROR, RUN_OUTCOME_FLOOR_CAP}:
        return text
    return text


def is_victory_outcome(value: Any) -> bool:
    return normalize_run_outcome(value, default="") == RUN_OUTCOME_VICTORY


def is_failure_outcome(value: Any) -> bool:
    return normalize_run_outcome(value, default="") == RUN_OUTCOME_DEATH
