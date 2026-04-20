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
        """返回轻量级的归一化战斗质量分。

        这里保留一个 0~1 附近的摘要分，方便弱 teacher / 摘要逻辑使用；
        训练主链里真正用于样本加权、保留和晋级的分数，统一走
        `zero.domain.scoring.compute_fight_score(...)`。

        之所以不再直接把不同量纲线性相加，是因为：
        - `fight_win` 是离散胜负信号
        - `enemy_hp_fraction_dealt` / `self_hp_fraction_remaining` 是比例信号
        - `potion_cost` 是消耗信号

        这里先把所有分量压到统一的 0~1 区间，再做加权平均。
        """
        win_component = 1.0 if self.fight_win >= 0.5 else 0.0
        enemy_component = max(0.0, min(1.0, float(self.enemy_hp_fraction_dealt)))
        hp_component = max(0.0, min(1.0, float(self.self_hp_fraction_remaining)))
        potion_penalty = max(0.0, min(1.0, float(self.potion_cost)))

        score = (
            0.55 * win_component
            + 0.30 * enemy_component
            + 0.15 * hp_component
            - 0.10 * potion_penalty
        )
        return max(0.0, min(1.0, float(score)))


@dataclass(slots=True)
class TeacherLabel:
    policy: list[float] = field(default_factory=list)
    topk_indices: list[int] = field(default_factory=list)
    best_action_index: int = -1
    ranking_margin: float = 0.0
    teacher_value: float = 0.0
    metadata: dict[str, float | int | str] = field(default_factory=dict)
