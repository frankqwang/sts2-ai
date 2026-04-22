from __future__ import annotations

"""Pool storage and hierarchical sampling for zero-style training data."""

import heapq
import random
from collections import Counter
from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import floor

from ..config import PoolConfig
from ..domain import TrainingSample


@dataclass(slots=True)
class BucketedSamplePool:
    name: str
    bucket_capacity: int
    retention_mode: str = "fifo"
    _fifo_buckets: dict[str, deque[TrainingSample]] = field(default_factory=lambda: defaultdict(deque))
    _score_buckets: dict[str, list[tuple[float, int, TrainingSample]]] = field(default_factory=lambda: defaultdict(list))
    _bucket_card_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    _sequence: int = 0
    _size: int = 0
    _attempted_adds: int = 0
    _accepted_adds: int = 0
    _replaced_adds: int = 0
    _rejected_adds: int = 0
    _evicted_items: int = 0
    _sampled_items: int = 0

    def add(self, sample: TrainingSample) -> None:
        bucket = sample.bucket_key or "default"
        self._attempted_adds += 1
        if self.retention_mode == "score":
            heap = self._score_buckets[bucket]
            score_entry = (sample.keep_score, self._sequence, sample)
            if len(heap) < self.bucket_capacity:
                heapq.heappush(heap, score_entry)
                self._increment_card_count(bucket, sample)
                self._accepted_adds += 1
                self._size += 1
            elif heap and score_entry[:2] > heap[0][:2]:
                removed = heapq.heapreplace(heap, score_entry)[2]
                self._decrement_card_count(bucket, removed)
                self._increment_card_count(bucket, sample)
                self._accepted_adds += 1
                self._replaced_adds += 1
                self._evicted_items += 1
            else:
                self._rejected_adds += 1
            self._sequence += 1
            return

        queue = self._fifo_buckets[bucket]
        queue.append(sample)
        self._increment_card_count(bucket, sample)
        self._accepted_adds += 1
        self._size += 1
        while len(queue) > self.bucket_capacity:
            removed = queue.popleft()
            self._decrement_card_count(bucket, removed)
            self._evicted_items += 1
            self._size -= 1

    def set_bucket_capacity(self, bucket_capacity: int) -> None:
        self.bucket_capacity = max(1, int(bucket_capacity))
        if self.retention_mode == "score":
            for bucket, heap in self._score_buckets.items():
                while len(heap) > self.bucket_capacity:
                    removed = heapq.heappop(heap)[2]
                    self._decrement_card_count(bucket, removed)
                    self._evicted_items += 1
                    self._size -= 1
            return
        for bucket, queue in self._fifo_buckets.items():
            while len(queue) > self.bucket_capacity:
                removed = queue.popleft()
                self._decrement_card_count(bucket, removed)
                self._evicted_items += 1
                self._size -= 1

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
            result.append(_sample_with_card_diversity(items, self._bucket_card_counts.get(bucket_key)))
        self._sampled_items += len(result)
        return result

    def items(self) -> list[TrainingSample]:
        if self.retention_mode == "score":
            return [entry[2] for bucket in self._score_buckets.values() for entry in bucket]
        return [item for bucket in self._fifo_buckets.values() for item in bucket]

    def size(self) -> int:
        return self._size

    def reset_iteration_counters(self) -> None:
        self._attempted_adds = 0
        self._accepted_adds = 0
        self._replaced_adds = 0
        self._rejected_adds = 0
        self._evicted_items = 0
        self._sampled_items = 0

    def counters(self) -> dict[str, int]:
        return {
            "attempted_adds": self._attempted_adds,
            "accepted_adds": self._accepted_adds,
            "replaced_adds": self._replaced_adds,
            "rejected_adds": self._rejected_adds,
            "evicted_items": self._evicted_items,
            "sampled_items": self._sampled_items,
        }

    def describe(self) -> dict[str, object]:
        items = self.items()
        keep_scores = [float(item.keep_score) for item in items]
        sample_weights = [float(item.sample_weight) for item in items]
        encounter_counts: Counter[str] = Counter(str(item.state.context.encounter_id or "unknown") for item in items)
        score_band_counts: Counter[str] = Counter(str(item.metadata.get("score_band", "unknown") or "unknown") for item in items)
        if self.retention_mode == "score":
            top_buckets = sorted(((bucket, len(entries)) for bucket, entries in self._score_buckets.items()), key=lambda item: item[1], reverse=True)[:5]
            bucket_count = len(self._score_buckets)
        else:
            top_buckets = sorted(((bucket, len(entries)) for bucket, entries in self._fifo_buckets.items()), key=lambda item: item[1], reverse=True)[:5]
            bucket_count = len(self._fifo_buckets)
        return {
            "size": self._size,
            "bucket_count": bucket_count,
            "keep_score_min": min(keep_scores) if keep_scores else 0.0,
            "keep_score_avg": (sum(keep_scores) / len(keep_scores)) if keep_scores else 0.0,
            "keep_score_max": max(keep_scores) if keep_scores else 0.0,
            "sample_weight_avg": (sum(sample_weights) / len(sample_weights)) if sample_weights else 0.0,
            "top_encounters": encounter_counts.most_common(5),
            "score_band_counts": dict(score_band_counts),
            "top_buckets": top_buckets,
        }

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

    def _increment_card_count(self, bucket: str, sample: TrainingSample) -> None:
        key = sample.main_card_id or "__none__"
        self._bucket_card_counts[bucket][key] += 1

    def _decrement_card_count(self, bucket: str, sample: TrainingSample) -> None:
        key = sample.main_card_id or "__none__"
        counts = self._bucket_card_counts[bucket]
        counts[key] -= 1
        if counts[key] <= 0:
            counts.pop(key, None)
        if not counts:
            self._bucket_card_counts.pop(bucket, None)

    def clear(self) -> None:
        self._fifo_buckets.clear()
        self._score_buckets.clear()
        self._bucket_card_counts.clear()
        self._size = 0


class SamplePoolSet:
    def __init__(self, config: PoolConfig):
        self._config = config
        self._recent_iteration_samples: deque[int] = deque(maxlen=max(1, config.dynamic_capacity_recent_iterations))
        self._base_capacities = {
            "recent_online": config.bucket_capacity,
            "search": config.search_bucket_capacity,
            "rare": config.rare_bucket_capacity,
            "reanalyse": max(1, config.bucket_capacity // 2),
            "legacy": config.bucket_capacity,
        }
        self._pools = {
            "recent_online": BucketedSamplePool("recent_online", config.bucket_capacity, retention_mode="score"),
            "search": BucketedSamplePool("search", config.search_bucket_capacity, retention_mode="score"),
            "rare": BucketedSamplePool("rare", config.rare_bucket_capacity, retention_mode="score"),
            "reanalyse": BucketedSamplePool("reanalyse", max(1, config.bucket_capacity // 2), retention_mode="score"),
            "legacy": BucketedSamplePool("legacy", config.bucket_capacity, retention_mode="score"),
        }
        self._current_capacities = dict(self._base_capacities)

    def add(self, sample: TrainingSample) -> None:
        pool = self._pools.get(sample.pool_name, self._pools["recent_online"])
        pool.add(sample)

    def add_many(self, samples: list[TrainingSample]) -> None:
        for sample in samples:
            self.add(sample)

    def replace_pool(self, pool_name: str, samples: list[TrainingSample]) -> None:
        pool = self._pools.get(pool_name)
        if pool is None:
            return
        if samples and len(samples) > pool.bucket_capacity:
            pool.set_bucket_capacity(len(samples))
            self._current_capacities[pool_name] = int(len(samples))
        pool.clear()
        for sample in samples:
            pool.add(sample)

    def pool_items(self, pool_name: str) -> list[TrainingSample]:
        pool = self._pools.get(pool_name)
        if pool is None:
            return []
        return pool.items()

    def mixed_sample(self, batch_size: int) -> list[TrainingSample]:
        weighted = [
            ("recent_online", self._config.recent_online_weight),
            ("search", self._config.search_weight),
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

    def reset_iteration_counters(self) -> None:
        for pool in self._pools.values():
            pool.reset_iteration_counters()

    def size_by_pool(self) -> dict[str, int]:
        return {name: pool.size() for name, pool in self._pools.items()}

    def capacity_by_pool(self) -> dict[str, int]:
        return dict(self._current_capacities)

    def iteration_counters(self) -> dict[str, dict[str, int]]:
        return {name: pool.counters() for name, pool in self._pools.items()}

    def describe(self) -> dict[str, dict[str, object]]:
        return {name: pool.describe() for name, pool in self._pools.items()}

    def update_capacity_plan(self, *, logical_samples: int) -> dict[str, int]:
        if logical_samples > 0:
            self._recent_iteration_samples.append(int(logical_samples))
        if not self._config.dynamic_capacity_enabled:
            return self.capacity_by_pool()

        target_total = max(sum(self._base_capacities.values()), sum(self._recent_iteration_samples))
        capacities = _allocate_capacities(
            target_total=target_total,
            weighted_bases=[
                ("recent_online", self._config.recent_online_weight, self._base_capacities["recent_online"]),
                ("search", self._config.search_weight, self._base_capacities["search"]),
                ("rare", self._config.rare_weight, self._base_capacities["rare"]),
                ("reanalyse", self._config.reanalyse_weight, self._base_capacities["reanalyse"]),
                ("legacy", self._config.legacy_weight, self._base_capacities["legacy"]),
            ],
        )
        for name, capacity in capacities.items():
            self._pools[name].set_bucket_capacity(capacity)
        self._current_capacities = capacities
        return self.capacity_by_pool()


def _sample_with_card_diversity(items: list[TrainingSample], card_counts: Counter[str] | None = None) -> TrainingSample:
    if not card_counts:
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


def _allocate_capacities(target_total: int, weighted_bases: list[tuple[str, float, int]]) -> dict[str, int]:
    if target_total <= 0:
        return {name: base for name, _weight, base in weighted_bases}

    positive = [(name, weight, base) for name, weight, base in weighted_bases if weight > 0.0]
    if not positive:
        return {name: base for name, _weight, base in weighted_bases}

    base_total = sum(base for _name, _weight, base in weighted_bases)
    target_total = max(target_total, base_total)
    extra_total = target_total - base_total
    total_weight = sum(weight for _name, weight, _base in positive)
    extra_by_name = {name: 0 for name, _weight, _base in weighted_bases}

    raw = []
    assigned_extra = 0
    for name, weight, _base in positive:
        value = extra_total * (weight / total_weight)
        integer = floor(value)
        extra_by_name[name] = integer
        assigned_extra += integer
        raw.append((value - integer, name))

    for _remainder, name in sorted(raw, reverse=True):
        if assigned_extra >= extra_total:
            break
        extra_by_name[name] += 1
        assigned_extra += 1

    return {
        name: max(1, base + extra_by_name.get(name, 0))
        for name, _weight, base in weighted_bases
    }
