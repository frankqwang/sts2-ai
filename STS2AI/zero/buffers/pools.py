from __future__ import annotations

"""Pool storage and hierarchical sampling for zero-style training data."""

import heapq
import random
from collections import Counter
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..config import PoolConfig
from ..domain import TrainingSample


@dataclass(slots=True)
class BucketedSamplePool:
    name: str
    bucket_capacity: int
    retention_mode: str = "fifo"
    _fifo_buckets: dict[str, deque[TrainingSample]] = field(default_factory=lambda: defaultdict(deque))
    _score_buckets: dict[str, list[tuple[float, int, TrainingSample]]] = field(default_factory=lambda: defaultdict(list))
    _sequence: int = 0

    def add(self, sample: TrainingSample) -> None:
        bucket = sample.bucket_key or "default"
        if self.retention_mode == "score":
            heap = self._score_buckets[bucket]
            if len(heap) < self.bucket_capacity:
                heapq.heappush(heap, (sample.keep_score, self._sequence, sample))
            elif heap and sample.keep_score > heap[0][0]:
                heapq.heapreplace(heap, (sample.keep_score, self._sequence, sample))
            self._sequence += 1
            return

        queue = self._fifo_buckets[bucket]
        queue.append(sample)
        while len(queue) > self.bucket_capacity:
            queue.popleft()

    def sample(self, count: int) -> list[TrainingSample]:
        if count <= 0:
            return []
        buckets = [bucket for bucket, items in self._iter_buckets() if items]
        if not buckets:
            return []
        result: list[TrainingSample] = []
        for _ in range(count):
            bucket_key = random.choice(buckets)
            items = self._bucket_items(bucket_key)
            if not items:
                continue
            result.append(_sample_with_card_diversity(items))
        return result

    def items(self) -> list[TrainingSample]:
        if self.retention_mode == "score":
            return [entry[2] for bucket in self._score_buckets.values() for entry in bucket]
        return [item for bucket in self._fifo_buckets.values() for item in bucket]

    def size(self) -> int:
        return len(self.items())

    def _iter_buckets(self):
        if self.retention_mode == "score":
            for bucket, entries in self._score_buckets.items():
                yield bucket, [entry[2] for entry in entries]
            return
        for bucket, items in self._fifo_buckets.items():
            yield bucket, list(items)

    def _bucket_items(self, bucket: str) -> list[TrainingSample]:
        if self.retention_mode == "score":
            return [entry[2] for entry in self._score_buckets.get(bucket, [])]
        return list(self._fifo_buckets.get(bucket, []))


class SamplePoolSet:
    def __init__(self, config: PoolConfig):
        self._config = config
        self._pools = {
            "recent_online": BucketedSamplePool("recent_online", config.bucket_capacity, retention_mode="fifo"),
            "teacher": BucketedSamplePool("teacher", config.teacher_bucket_capacity, retention_mode="score"),
            "rare": BucketedSamplePool("rare", config.rare_bucket_capacity, retention_mode="score"),
            "reanalyse": BucketedSamplePool("reanalyse", config.bucket_capacity // 2, retention_mode="score"),
            "legacy": BucketedSamplePool("legacy", config.bucket_capacity, retention_mode="fifo"),
        }

    def add(self, sample: TrainingSample) -> None:
        pool = self._pools.get(sample.pool_name, self._pools["recent_online"])
        pool.add(sample)

    def add_many(self, samples: list[TrainingSample]) -> None:
        for sample in samples:
            self.add(sample)

    def mixed_sample(self, batch_size: int) -> list[TrainingSample]:
        weighted = [
            ("recent_online", self._config.recent_online_weight),
            ("teacher", self._config.teacher_weight),
            ("rare", self._config.rare_weight),
            ("reanalyse", self._config.reanalyse_weight),
            ("legacy", self._config.legacy_weight),
        ]
        counts = _allocate_counts(
            batch_size,
            [(name, weight, self._pools[name].size()) for name, weight in weighted],
        )
        result: list[TrainingSample] = []
        for pool_name, _weight in weighted:
            count = counts.get(pool_name, 0)
            result.extend(self._pools[pool_name].sample(count))
        while len(result) < batch_size:
            result.extend(self._pools["recent_online"].sample(1))
            if not self._pools["recent_online"].items():
                break
        return result[:batch_size]

    def size_by_pool(self) -> dict[str, int]:
        return {name: pool.size() for name, pool in self._pools.items()}


def _sample_with_card_diversity(items: list[TrainingSample]) -> TrainingSample:
    card_counts = Counter(sample.main_card_id or "__none__" for sample in items)
    weights = []
    for sample in items:
        key = sample.main_card_id or "__none__"
        weights.append(1.0 / max(1, card_counts[key]))
    return random.choices(items, weights=weights, k=1)[0]


def _allocate_counts(batch_size: int, weighted_sizes: list[tuple[str, float, int]]) -> dict[str, int]:
    if batch_size <= 0:
        return {name: 0 for name, _, _ in weighted_sizes}

    active = [(name, weight, size) for name, weight, size in weighted_sizes if size > 0 and weight > 0]
    if not active:
        return {name: 0 for name, _, _ in weighted_sizes}

    raw = [(name, batch_size * weight) for name, weight, _ in active]
    counts = {name: int(value) for name, value in raw}
    remainders = sorted(
        ((value - int(value), name) for name, value in raw),
        reverse=True,
    )
    assigned = sum(counts.values())
    for _remainder, name in remainders:
        if assigned >= batch_size:
            break
        counts[name] += 1
        assigned += 1

    if assigned == 0:
        top_name = max(active, key=lambda item: item[1])[0]
        counts[top_name] = 1
        assigned = 1

    while assigned < batch_size:
        for name, _weight, _size in sorted(active, key=lambda item: item[1], reverse=True):
            if assigned >= batch_size:
                break
            counts[name] = counts.get(name, 0) + 1
            assigned += 1

    return {name: counts.get(name, 0) for name, _, _ in weighted_sizes}
