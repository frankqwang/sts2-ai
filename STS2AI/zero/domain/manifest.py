from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class TrainingSummary:
    steps: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    delta_loss: float = 0.0
    uncertainty_loss: float = 0.0
    total_loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    skipped_non_finite_steps: int = 0
    zero_step: bool = False
    pool_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class EvalSummary:
    cohort_name: str
    fight_win_rate: float
    enemy_hp_fraction_dealt: float
    self_hp_fraction_remaining: float
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PromotionDecision:
    promoted: bool
    reason: str
    new_version: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class IterationManifest:
    iteration: int
    collector_version: str
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
        return {
            "iteration": self.iteration,
            "collector_version": self.collector_version,
            "sample_counts": dict(self.sample_counts),
            "admission_stats": dict(self.admission_stats),
            "pool_sizes": dict(self.pool_sizes),
            "pool_capacities": dict(self.pool_capacities),
            "pool_stats": dict(self.pool_stats),
            "training": self.training.to_dict(),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "promotion": self.promotion.to_dict(),
        }
