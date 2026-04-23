from __future__ import annotations

from .battle import (
    BattleState,
    EnemyState,
    HandCardState,
    HistoryStep,
    LegalAction,
    PileSummary,
    PlayerState,
    StaticContext,
    TargetSummary,
    TransitionDelta,
)
from .labels import FightLabel
from .manifest import EvalSummary, IterationManifest, PromotionDecision, TrainingSummary
from .progress import ProgressSignal, assess_transition_progress
from .scoring import (
    compute_episode_score_proxy,
    compute_fight_score,
    compute_hp_quality_score,
    compute_step_progress_score,
)
from .samples import (
    RawTransition,
    TrainingSample,
    compact_battle_state,
    compact_legal_action,
    compact_raw_transition,
)

__all__ = [
    "BattleState",
    "EnemyState",
    "EvalSummary",
    "FightLabel",
    "HandCardState",
    "HistoryStep",
    "IterationManifest",
    "LegalAction",
    "PileSummary",
    "PlayerState",
    "PromotionDecision",
    "ProgressSignal",
    "RawTransition",
    "StaticContext",
    "TargetSummary",
    "TrainingSample",
    "TrainingSummary",
    "TransitionDelta",
    "assess_transition_progress",
    "compact_battle_state",
    "compact_legal_action",
    "compact_raw_transition",
    "compute_episode_score_proxy",
    "compute_fight_score",
    "compute_hp_quality_score",
    "compute_step_progress_score",
]
