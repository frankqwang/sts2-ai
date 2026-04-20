from __future__ import annotations

"""Teacher queue construction and label materialization."""

import heapq

from ..config import TeacherConfig
from ..domain import TeacherRequest, TrainingSample
from ..ports import TeacherOracle


class TeacherQueueBuilder:
    def __init__(self, config: TeacherConfig):
        self._config = config

    def select(self, samples: list[TrainingSample]) -> list[TeacherRequest]:
        heap: list[tuple[float, int, TeacherRequest]] = []
        for index, sample in enumerate(samples):
            score, tags = self._priority(sample)
            if score <= 0:
                continue
            request = TeacherRequest(
                request_id=sample.sample_id,
                sample=sample,
                priority=score,
                reason_tags=tags,
            )
            heapq.heappush(heap, (-score, index, request))

        result: list[TeacherRequest] = []
        limit = self._config.max_requests_per_iteration
        while heap and len(result) < limit:
            result.append(heapq.heappop(heap)[2])
        return result

    def _priority(self, sample: TrainingSample) -> tuple[float, list[str]]:
        score = 0.0
        tags: list[str] = []
        if sample.state.context.encounter_class in {"elite", "boss"}:
            score += 1.0
            tags.append(sample.state.context.encounter_class)

        hp_ratio = sample.state.player.hp / max(1.0, sample.state.player.max_hp)
        if hp_ratio <= self._config.near_lethal_hp_ratio:
            score += 1.0
            tags.append("near_lethal")

        # Prefer the derived target used for supervision so queueing does not
        # overfit to the online policy's own uncertainty head calibration.
        uncertainty = float(
            sample.metadata.get("uncertainty_target", sample.metadata.get("uncertainty", 0.0)) or 0.0
        )
        if uncertainty >= self._config.uncertainty_threshold:
            score += uncertainty
            tags.append("high_uncertainty")

        top2_gap = float(sample.metadata.get("top2_gap", 1.0) or 1.0)
        if top2_gap <= self._config.top2_gap_threshold:
            score += 1.0 - top2_gap
            tags.append("small_top2_gap")

        if sample.rare_cohort_tags:
            score += 0.5
            tags.append("rare_cohort")

        return score, tags


class TeacherQueueProcessor:
    def label(self, requests: list[TeacherRequest], teacher: TeacherOracle, runtime_factory=None) -> list[TrainingSample]:
        labeled: list[TrainingSample] = []
        for request in requests:
            labeled_sample = request.sample.clone_for_pool(
                pool_name=request.sample.pool_name,
                keep_score=max(request.sample.keep_score, request.priority),
                metadata={"teacher_priority": request.priority},
                teacher_label=teacher.label_request(request, runtime_factory=runtime_factory),
            )
            labeled.append(labeled_sample)
        return labeled
