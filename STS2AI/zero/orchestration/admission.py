from __future__ import annotations

"""Admission rules from logical samples to concrete pool entries.

This module centralizes pool duplication so a logical decision point can be
represented by separate immutable pool entries:
- online/base sample -> `recent_online`
- teacher-labeled copy -> `teacher`
- rare-copy with dedicated capacity -> `rare`
"""

from ..domain import TrainingSample


class SampleAdmissionPlanner:
    def build_online_entries(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        entries: list[TrainingSample] = []
        no_progress_kept: dict[str, int] = {}
        for sample in samples:
            if not _should_keep_online_sample(sample, no_progress_kept):
                continue
            entries.append(sample.clone_for_pool(pool_name="recent_online"))
            if sample.rare_cohort_tags:
                entries.append(
                    sample.clone_for_pool(
                        pool_name="rare",
                        keep_score=max(sample.keep_score, 1.0),
                        metadata={"admission_reason": "rare_cohort"},
                    )
                )
        return entries

    def build_teacher_entries(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        entries: list[TrainingSample] = []
        for sample in samples:
            entries.append(
                sample.clone_for_pool(
                    pool_name="teacher",
                    keep_score=max(sample.keep_score, 1.0),
                    metadata={"admission_reason": "teacher_label"},
                    teacher_label=sample.teacher_label,
                )
            )
            if sample.rare_cohort_tags:
                entries.append(
                    sample.clone_for_pool(
                        pool_name="rare",
                        keep_score=max(sample.keep_score, 1.0),
                        metadata={"admission_reason": "teacher_rare"},
                        teacher_label=sample.teacher_label,
                    )
                )
        return entries


def _should_keep_online_sample(sample: TrainingSample, no_progress_kept: dict[str, int]) -> bool:
    """对明显重复的拖回合在线样本做温和裁剪，减少噪声和池内膨胀。"""
    fight_timeout = bool(sample.metadata.get("fight_timeout", False))
    no_progress_ratio = float(sample.metadata.get("fight_no_progress_ratio", 0.0) or 0.0)
    if not fight_timeout and no_progress_ratio < 0.85:
        return True
    if sample.step_progress_score > 0.0:
        return True
    score_band = str(sample.metadata.get("score_band", "normal") or "normal")
    if score_band == "boost" or sample.step_idx == 0:
        return True
    if str(sample.behavior_action_id or "").startswith("end_turn"):
        return False
    seen = no_progress_kept.get(sample.fight_id, 0) + 1
    no_progress_kept[sample.fight_id] = seen
    return seen % 6 == 0
