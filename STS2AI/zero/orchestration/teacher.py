from __future__ import annotations

"""Teacher queue construction and label materialization."""

import heapq
from collections import defaultdict, deque

from ..config import TeacherConfig
from ..domain import TeacherRequest, TrainingSample
from ..ports import TeacherOracle


class TeacherQueueBuilder:
    def __init__(self, config: TeacherConfig):
        self._config = config

    def select(self, samples: list[TrainingSample]) -> list[TeacherRequest]:
        bucketed: dict[str, list[tuple[float, int, TeacherRequest]]] = defaultdict(list)
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
            bucket = tags[0] if tags else sample.state.context.encounter_class or "default"
            heapq.heappush(bucketed[bucket], (-score, index, request))

        result: list[TeacherRequest] = []
        limit = self._config.max_requests_per_iteration
        bucket_keys = deque(sorted(bucketed.keys()))
        while bucket_keys and len(result) < limit:
            bucket = bucket_keys.popleft()
            heap = bucketed[bucket]
            if heap:
                result.append(heapq.heappop(heap)[2])
            if heap:
                bucket_keys.append(bucket)
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

        if bool(sample.metadata.get("fight_timeout", False)):
            score += 1.0
            tags.append("fight_timeout")

        no_progress_ratio = float(sample.metadata.get("fight_no_progress_ratio", 0.0) or 0.0)
        if no_progress_ratio >= 0.70:
            score += 0.75 * no_progress_ratio
            tags.append("high_no_progress")

        if sample.step_progress_score < 0.0:
            score += min(0.5, abs(sample.step_progress_score))
            tags.append("negative_step_progress")

        if sample.fight_score <= 0.45:
            score += 0.5
            tags.append("low_fight_score")

        return score, tags


class TeacherQueueProcessor:
    def label(self, requests: list[TeacherRequest], teacher: TeacherOracle, runtime_factory=None) -> list[TrainingSample]:
        if not requests:
            return []

        labels = None
        batch_hook = getattr(teacher, "label_requests", None)
        if callable(batch_hook):
            labels = batch_hook(
                requests,
                runtime_factory=runtime_factory,
            )
        if labels is None:
            labels = [
                teacher.label_request(
                    request,
                    runtime_factory=runtime_factory,
                    seed=str(request.sample.state.context.metadata.get("seed", "")),
                )
                for request in requests
            ]

        labeled: list[TrainingSample] = []
        for request, label in zip(requests, labels, strict=False):
            labeled_sample = request.sample.clone_for_pool(
                pool_name=request.sample.pool_name,
                keep_score=max(request.sample.keep_score, request.priority),
                metadata={"teacher_priority": request.priority},
                teacher_label=label,
            )
            labeled.append(labeled_sample)
        return labeled
