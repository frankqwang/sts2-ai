from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class TrainingSummary:
    steps: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    ranking_loss: float = 0.0
    delta_loss: float = 0.0
    uncertainty_loss: float = 0.0
    total_loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    teacher_sample_ratio: float = 0.0
    skipped_non_finite_steps: int = 0
    zero_step: bool = False
    pool_usage: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class EvalSummary:
    cohort_name: str
    fight_win_rate: float
    enemy_hp_fraction_dealt: float
    self_hp_fraction_remaining: float
    teacher_agreement_at_1: float = 0.0
    teacher_topk_overlap: float = 0.0
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class PromotionDecision:
    promoted: bool
    reason: str
    new_version: str = ""


@dataclass(slots=True)
class IterationManifest:
    iteration: int
    collector_version: str
    teacher_version: str
    sample_counts: dict[str, int] = field(default_factory=dict)
    admission_stats: dict[str, object] = field(default_factory=dict)
    pool_sizes: dict[str, int] = field(default_factory=dict)
    pool_capacities: dict[str, int] = field(default_factory=dict)
    pool_stats: dict[str, object] = field(default_factory=dict)
    training: TrainingSummary = field(default_factory=TrainingSummary)
    evaluations: list[EvalSummary] = field(default_factory=list)
    promotion: PromotionDecision = field(
        default_factory=lambda: PromotionDecision(promoted=False, reason="pending")
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
