from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FightLabel:
    fight_win: float
    enemy_hp_fraction_dealt: float
    self_hp_fraction_remaining: float
    player_hp: float = 0.0
    player_max_hp: float = 0.0
    potion_cost: float = 0.0

    @property
    def fight_score(self) -> float:
        """返回轻量级的归一化战斗质量分。

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
