from __future__ import annotations

from ..config import EvalConfig
from ..domain import EvalSummary, PromotionDecision


class PromotionJudge:
    def __init__(self, config: EvalConfig):
        self._config = config

    def decide(
        self,
        *,
        candidate_version: str,
        current: list[EvalSummary],
        baseline: list[EvalSummary] | None,
    ) -> PromotionDecision:
        if not baseline:
            return PromotionDecision(promoted=True, reason="首次评估通过", new_version=candidate_version)

        current_map = {item.cohort_name: item for item in current}
        baseline_map = {item.cohort_name: item for item in baseline}

        for cohort_name, base in baseline_map.items():
            cand = current_map.get(cohort_name)
            if cand is None:
                return PromotionDecision(promoted=False, reason=f"缺少 cohort: {cohort_name}")
            if cand.self_hp_fraction_remaining + self._config.allow_hp_remaining_drop < base.self_hp_fraction_remaining:
                return PromotionDecision(promoted=False, reason=f"{cohort_name} 剩余血量退化")

        main_current = current_map.get("main")
        main_baseline = baseline_map.get("main")
        if main_current and main_baseline:
            gain = main_current.fight_win_rate - main_baseline.fight_win_rate
            if gain < self._config.promote_min_win_rate_gain:
                return PromotionDecision(promoted=False, reason="主 cohort 胜率提升不足")

        return PromotionDecision(promoted=True, reason="评估通过", new_version=candidate_version)
