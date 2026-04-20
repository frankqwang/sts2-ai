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
        for sample in samples:
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
