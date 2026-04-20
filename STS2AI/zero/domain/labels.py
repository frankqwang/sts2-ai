from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FightLabel:
    fight_win: float
    enemy_hp_fraction_dealt: float
    self_hp_fraction_remaining: float
    potion_cost: float = 0.0

    @property
    def fight_score(self) -> float:
        return (
            1.0 * self.fight_win
            + 0.6 * self.enemy_hp_fraction_dealt
            + 0.3 * self.self_hp_fraction_remaining
            - 0.1 * self.potion_cost
        )


@dataclass(slots=True)
class TeacherLabel:
    policy: list[float] = field(default_factory=list)
    topk_indices: list[int] = field(default_factory=list)
    best_action_index: int = -1
    ranking_margin: float = 0.0
    teacher_value: float = 0.0
    metadata: dict[str, float | int | str] = field(default_factory=dict)
