from __future__ import annotations

import math
from collections import defaultdict

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
            cand_hp_quality = _hp_quality(cand)
            base_hp_quality = _hp_quality(base)
            cand_fight_quality = _fight_quality(cand)
            base_fight_quality = _fight_quality(base)
            if cand_hp_quality + self._config.allow_hp_quality_drop < base_hp_quality:
                return PromotionDecision(promoted=False, reason=f"{cohort_name} HP 质量退化")
            if cand_fight_quality - base_fight_quality < self._config.promote_min_fight_quality_gain:
                return PromotionDecision(promoted=False, reason=f"{cohort_name} 战斗质量提升不足")
            timeout_rate = float(cand.metadata.get("timeout_rate", 0.0) or 0.0)
            no_progress_ratio = float(cand.metadata.get("avg_no_progress_ratio", 0.0) or 0.0)
            no_progress_streak = float(cand.metadata.get("avg_max_no_progress_streak", 0.0) or 0.0)
            if timeout_rate > self._config.max_timeout_rate:
                return PromotionDecision(promoted=False, reason=f"{cohort_name} timeout 率过高")
            if no_progress_ratio > self._config.max_no_progress_ratio:
                return PromotionDecision(promoted=False, reason=f"{cohort_name} 无进展比例过高")
            if no_progress_streak > self._config.max_no_progress_streak:
                return PromotionDecision(promoted=False, reason=f"{cohort_name} 无进展连续回合过长")

        current_agg = _aggregate(current)
        baseline_agg = _aggregate(baseline)
        if current_agg["timeout_rate"] > self._config.max_timeout_rate:
            return PromotionDecision(promoted=False, reason="整体 timeout 率过高")
        if current_agg["avg_no_progress_ratio"] > self._config.max_no_progress_ratio:
            return PromotionDecision(promoted=False, reason="整体无进展比例过高")
        if current_agg["avg_max_no_progress_streak"] > self._config.max_no_progress_streak:
            return PromotionDecision(promoted=False, reason="整体无进展连续回合过长")
        if current_agg["hp_quality_score"] + self._config.allow_hp_quality_drop < baseline_agg["hp_quality_score"]:
            return PromotionDecision(promoted=False, reason="整体 HP 质量退化")
        if current_agg["fight_quality_score"] - baseline_agg["fight_quality_score"] < self._config.promote_min_fight_quality_gain:
            return PromotionDecision(promoted=False, reason="整体战斗质量提升不足")
        if current_agg["fight_win_rate"] - baseline_agg["fight_win_rate"] < self._config.promote_min_win_rate_gain:
            return PromotionDecision(promoted=False, reason="整体胜率提升不足")
        if current_agg["enemy_hp_fraction_dealt"] - baseline_agg["enemy_hp_fraction_dealt"] < self._config.promote_min_enemy_hp_gain:
            return PromotionDecision(promoted=False, reason="整体敌方掉血提升不足")
        if current_agg["teacher_agreement_at_1"] - baseline_agg["teacher_agreement_at_1"] < self._config.promote_min_teacher_agreement_gain:
            return PromotionDecision(promoted=False, reason="teacher agreement 提升不足")

        z_threshold = float(self._config.significance_z)
        if z_threshold > 0.0:
            win_z = _two_proportion_z(
                current_agg["fight_win_rate"],
                int(current_agg["num_episodes"]),
                baseline_agg["fight_win_rate"],
                int(baseline_agg["num_episodes"]),
            )
            if win_z < z_threshold:
                return PromotionDecision(promoted=False, reason="整体胜率提升未达统计显著")
            teacher_z = _two_proportion_z(
                current_agg["teacher_agreement_at_1"],
                int(current_agg["num_episodes"]),
                baseline_agg["teacher_agreement_at_1"],
                int(baseline_agg["num_episodes"]),
            )
            if teacher_z < z_threshold and self._config.promote_min_teacher_agreement_gain > 0.0:
                return PromotionDecision(promoted=False, reason="teacher agreement 提升未达统计显著")

        for bucket, base_bucket in _bucketed(baseline).items():
            cand_bucket = _bucketed(current).get(bucket)
            if cand_bucket is None:
                return PromotionDecision(promoted=False, reason=f"缺少 bucket: {bucket}")
            if cand_bucket["hp_quality_score"] + self._config.allow_hp_quality_drop < base_bucket["hp_quality_score"]:
                return PromotionDecision(promoted=False, reason=f"{bucket} bucket HP 质量退化")
            if cand_bucket["fight_quality_score"] - base_bucket["fight_quality_score"] < self._config.promote_min_fight_quality_gain:
                return PromotionDecision(promoted=False, reason=f"{bucket} bucket 战斗质量提升不足")
            if cand_bucket["timeout_rate"] > self._config.max_timeout_rate:
                return PromotionDecision(promoted=False, reason=f"{bucket} bucket timeout 率过高")
            if cand_bucket["avg_no_progress_ratio"] > self._config.max_no_progress_ratio:
                return PromotionDecision(promoted=False, reason=f"{bucket} bucket 无进展比例过高")
            if cand_bucket["avg_max_no_progress_streak"] > self._config.max_no_progress_streak:
                return PromotionDecision(promoted=False, reason=f"{bucket} bucket 无进展连续回合过长")

        return PromotionDecision(promoted=True, reason="评估通过", new_version=candidate_version)


def _aggregate(rows: list[EvalSummary]) -> dict[str, float]:
    if not rows:
        return {
            "fight_win_rate": 0.0,
            "enemy_hp_fraction_dealt": 0.0,
            "self_hp_fraction_remaining": 0.0,
            "teacher_agreement_at_1": 0.0,
            "timeout_rate": 0.0,
            "avg_no_progress_ratio": 0.0,
            "avg_max_no_progress_streak": 0.0,
            "fight_quality_score": 0.0,
            "hp_quality_score": 0.0,
            "num_episodes": 0.0,
        }
    total_weight = 0.0
    totals = defaultdict(float)
    for row in rows:
        weight = float(row.metadata.get("num_episodes", 1) or 1)
        total_weight += weight
        totals["fight_win_rate"] += row.fight_win_rate * weight
        totals["enemy_hp_fraction_dealt"] += row.enemy_hp_fraction_dealt * weight
        totals["self_hp_fraction_remaining"] += row.self_hp_fraction_remaining * weight
        totals["teacher_agreement_at_1"] += row.teacher_agreement_at_1 * weight
        totals["timeout_rate"] += float(row.metadata.get("timeout_rate", 0.0) or 0.0) * weight
        totals["avg_no_progress_ratio"] += float(row.metadata.get("avg_no_progress_ratio", 0.0) or 0.0) * weight
        totals["avg_max_no_progress_streak"] += float(row.metadata.get("avg_max_no_progress_streak", 0.0) or 0.0) * weight
        totals["fight_quality_score"] += _fight_quality(row) * weight
        totals["hp_quality_score"] += _hp_quality(row) * weight
    return {
        "fight_win_rate": totals["fight_win_rate"] / max(total_weight, 1.0),
        "enemy_hp_fraction_dealt": totals["enemy_hp_fraction_dealt"] / max(total_weight, 1.0),
        "self_hp_fraction_remaining": totals["self_hp_fraction_remaining"] / max(total_weight, 1.0),
        "teacher_agreement_at_1": totals["teacher_agreement_at_1"] / max(total_weight, 1.0),
        "timeout_rate": totals["timeout_rate"] / max(total_weight, 1.0),
        "avg_no_progress_ratio": totals["avg_no_progress_ratio"] / max(total_weight, 1.0),
        "avg_max_no_progress_streak": totals["avg_max_no_progress_streak"] / max(total_weight, 1.0),
        "fight_quality_score": totals["fight_quality_score"] / max(total_weight, 1.0),
        "hp_quality_score": totals["hp_quality_score"] / max(total_weight, 1.0),
        "num_episodes": total_weight,
    }


def _bucketed(rows: list[EvalSummary]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[EvalSummary]] = defaultdict(list)
    for row in rows:
        bucket = str(row.metadata.get("eval_bucket", row.metadata.get("encounter_type", "default")) or "default")
        buckets[bucket].append(row)
    return {bucket: _aggregate(items) for bucket, items in buckets.items()}


def _hp_quality(row: EvalSummary) -> float:
    value = row.metadata.get("hp_quality_score")
    if value is not None:
        return float(value or 0.0)
    return max(0.0, min(1.0, float(row.self_hp_fraction_remaining or 0.0)))


def _fight_quality(row: EvalSummary) -> float:
    value = row.metadata.get("fight_quality_score")
    if value is not None:
        return float(value or 0.0)
    return (
        0.55 * max(0.0, min(1.0, float(row.fight_win_rate or 0.0)))
        + 0.30 * max(0.0, min(1.0, float(row.enemy_hp_fraction_dealt or 0.0)))
        + 0.15 * _hp_quality(row)
    )


def _two_proportion_z(p1: float, n1: int, p0: float, n0: int) -> float:
    if n1 <= 0 or n0 <= 0:
        return 0.0
    pooled = ((p1 * n1) + (p0 * n0)) / float(n1 + n0)
    variance = pooled * (1.0 - pooled) * ((1.0 / float(n1)) + (1.0 / float(n0)))
    if variance <= 1e-9:
        return 0.0
    return (p1 - p0) / math.sqrt(variance)
