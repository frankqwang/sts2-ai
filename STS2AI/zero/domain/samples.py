from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from .battle import BattleState, HistoryStep, LegalAction, TransitionDelta
from .labels import FightLabel, TeacherLabel


@dataclass(slots=True)
class RawTransition:
    """One on-policy combat decision as collected from the runtime.

    Important invariants:
    - `action_index` is only meaningful relative to `state.legal_actions`.
    - `action` is the concrete chosen action snapshot at collection time.
    - We keep both because the trainer needs a categorical behavior label
      (`action_index`) and the sample pipeline also needs the chosen action's
      explicit semantics (`action_id`, card/target payload, etc.).
    """

    run_id: str
    fight_id: str
    step_idx: int
    seed: str
    action_index: int
    state: BattleState
    action: LegalAction
    next_state: BattleState
    done: bool
    fight_outcome: str
    run_outcome: str
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def action_id(self) -> str:
        return self.action.action_id


@dataclass(slots=True)
class TrainingSample:
    """One immutable-ish training example snapshot.

    Important invariant:
    - A pool entry must own its own `TrainingSample` instance.
    - If the same logical decision point is admitted to multiple pools
      (`recent_online`, `teacher`, `rare`), callers must clone via
      `clone_for_pool()` instead of mutating and reusing the same object.
    """

    sample_id: str
    run_id: str
    fight_id: str
    step_idx: int
    state: BattleState
    history: list[HistoryStep]
    legal_actions: list[LegalAction]
    # Index into `legal_actions`, used as the categorical behavior label.
    behavior_action_index: int
    delta: TransitionDelta
    fight_label: FightLabel
    teacher_label: TeacherLabel | None = None
    # Stable action identity carried alongside the categorical index so callers
    # do not need to mentally reconstruct "which action was chosen".
    behavior_action_id: str = ""
    bucket_key: str = ""
    pool_name: str = "recent_online"
    main_card_id: str = ""
    risk_band: str = "normal"
    archetype_tags: list[str] = field(default_factory=list)
    rare_cohort_tags: list[str] = field(default_factory=list)
    student_disagreement: float = 0.0
    teacher_budget: float = 0.0
    keep_score: float = 0.0
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def behavior_action(self) -> LegalAction:
        if 0 <= self.behavior_action_index < len(self.legal_actions):
            return self.legal_actions[self.behavior_action_index]
        if self.legal_actions:
            return self.legal_actions[0]
        raise IndexError("TrainingSample has no legal_actions to resolve behavior_action.")

    def clone_for_pool(self, *, pool_name: str, keep_score: float | None = None, metadata: dict[str, str | float | int | bool] | None = None, teacher_label: TeacherLabel | None = None) -> "TrainingSample":
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return replace(
            self,
            pool_name=pool_name,
            keep_score=self.keep_score if keep_score is None else keep_score,
            metadata=merged_metadata,
            teacher_label=self.teacher_label if teacher_label is None else teacher_label,
        )


@dataclass(slots=True)
class TeacherRequest:
    request_id: str
    sample: TrainingSample
    priority: float
    reason_tags: list[str] = field(default_factory=list)
