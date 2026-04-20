"""关键 combat step 的在线重加权 / 重采样 helper。"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from networkV2.s6_training.batch import TrainingSample


_COMBAT_ROOM_TYPES = {"monster", "elite", "boss"}
_CRITICAL_SCORE_THRESHOLD = 0.8


def is_combat_sample(sample: TrainingSample) -> bool:
    return str(sample.room_type or "").strip().lower() in _COMBAT_ROOM_TYPES


def _terminal_swing_indices(samples: list[TrainingSample]) -> set[int]:
    terminal_indices: set[int] = set()
    seg_start: int | None = None
    for idx, sample in enumerate(samples):
        if is_combat_sample(sample):
            if seg_start is None:
                seg_start = idx
            continue
        if seg_start is not None:
            seg_end = idx
            if seg_end - seg_start > 0:
                terminal_indices.update(range(max(seg_end - 2, seg_start), seg_end))
            seg_start = None
    if seg_start is not None:
        seg_end = len(samples)
        if seg_end - seg_start > 0:
            terminal_indices.update(range(max(seg_end - 2, seg_start), seg_end))
    return terminal_indices


def annotate_critical_steps(samples: list[TrainingSample]) -> dict[str, float]:
    """给 rollout 样本补关键步标签，并更新最终 sample_weight。"""
    if not samples:
        return {}

    combat_indices = [idx for idx, sample in enumerate(samples) if is_combat_sample(sample)]
    adv_threshold = float("inf")
    if combat_indices:
        abs_advs = sorted((abs(float(samples[idx].advantage)) for idx in combat_indices), reverse=True)
        top_n = max(1, int(len(abs_advs) * 0.1))
        adv_threshold = abs_advs[top_n - 1]
    terminal_indices = _terminal_swing_indices(samples)

    boss_hits = 0
    elite_hits = 0
    high_adv_hits = 0
    turn_swing_hits = 0
    terminal_hits = 0
    critical_count = 0

    for idx, sample in enumerate(samples):
        base_weight = float(sample.base_sample_weight or sample.sample_weight or 1.0)
        if not is_combat_sample(sample):
            samples[idx] = replace(
                sample,
                critical_tags=(),
                critical_score=0.0,
                sample_weight=base_weight,
                base_sample_weight=base_weight,
            )
            continue

        tags: list[str] = []
        score = 0.0
        room_type = str(sample.room_type or "").strip().lower()
        if room_type == "boss":
            tags.append("boss_room")
            score += 1.0
            boss_hits += 1
        elif room_type == "elite":
            tags.append("elite_room")
            score += 0.6
            elite_hits += 1

        if abs(float(sample.advantage)) >= adv_threshold:
            tags.append("high_adv")
            score += 0.8
            high_adv_hits += 1

        if float(sample.turn_damage_target) >= 12.0 or float(sample.turn_block_target) >= 12.0:
            tags.append("turn_swing")
            score += 0.8
            turn_swing_hits += 1

        if idx in terminal_indices and float(sample.fight_win_target) >= 0.0:
            tags.append("terminal_swing")
            score += 0.6
            terminal_hits += 1

        multiplier = 1.0
        if score >= 1.6:
            multiplier = 2.5
        elif score >= _CRITICAL_SCORE_THRESHOLD:
            multiplier = 1.75

        if score >= _CRITICAL_SCORE_THRESHOLD:
            critical_count += 1

        samples[idx] = replace(
            sample,
            critical_tags=tuple(tags),
            critical_score=score,
            sample_weight=base_weight * multiplier,
            base_sample_weight=base_weight,
        )

    return {
        "critical_combat_count": float(critical_count),
        "critical_boss_hits": float(boss_hits),
        "critical_elite_hits": float(elite_hits),
        "critical_high_adv_hits": float(high_adv_hits),
        "critical_turn_swing_hits": float(turn_swing_hits),
        "critical_terminal_hits": float(terminal_hits),
        "critical_adv_threshold": 0.0 if adv_threshold == float("inf") else float(adv_threshold),
    }


def rebalance_training_samples(
    samples: list[TrainingSample],
    *,
    rng: random.Random,
) -> tuple[list[TrainingSample], dict[str, float]]:
    """按固定配额重采样，防止普通战斗 step 淹没关键样本。"""
    if not samples:
        return [], {
            "rebalance_total": 0.0,
            "rebalance_critical_combat": 0.0,
            "rebalance_regular_combat": 0.0,
            "rebalance_noncombat": 0.0,
        }

    critical_combat = [s for s in samples if is_combat_sample(s) and float(s.critical_score) >= _CRITICAL_SCORE_THRESHOLD]
    regular_combat = [s for s in samples if is_combat_sample(s) and float(s.critical_score) < _CRITICAL_SCORE_THRESHOLD]
    noncombat = [s for s in samples if not is_combat_sample(s)]

    total = len(samples)
    critical_target = int(round(total * 0.35))
    regular_target = int(round(total * 0.45))
    noncombat_target = max(total - critical_target - regular_target, 0)

    def _draw(bucket: list[TrainingSample], count: int) -> list[TrainingSample]:
        if count <= 0 or not bucket:
            return []
        return list(rng.choices(bucket, k=count))

    out = []
    out.extend(_draw(critical_combat, critical_target))
    out.extend(_draw(regular_combat, regular_target))
    out.extend(_draw(noncombat, noncombat_target))
    if len(out) < total:
        out.extend(_draw(samples, total - len(out)))
    rng.shuffle(out)
    out = out[:total]
    return out, {
        "rebalance_total": float(total),
        "rebalance_critical_combat": float(len(critical_combat)),
        "rebalance_regular_combat": float(len(regular_combat)),
        "rebalance_noncombat": float(len(noncombat)),
        "rebalance_target_critical": float(critical_target),
        "rebalance_target_regular": float(regular_target),
        "rebalance_target_noncombat": float(noncombat_target),
        "rebalance_output_critical": float(sum(1 for sample in out if is_combat_sample(sample) and float(sample.critical_score) >= _CRITICAL_SCORE_THRESHOLD)),
        "rebalance_output_regular": float(sum(1 for sample in out if is_combat_sample(sample) and float(sample.critical_score) < _CRITICAL_SCORE_THRESHOLD)),
        "rebalance_output_noncombat": float(sum(1 for sample in out if not is_combat_sample(sample))),
    }


def sort_capture_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    room_priority = {"boss": 2, "elite": 1, "monster": 0}
    return sorted(
        records,
        key=lambda item: (
            float(item.get("critical_score", 0.0)),
            abs(float(item.get("advantage", 0.0))),
            room_priority.get(str(item.get("room_type", "")).lower(), -1),
        ),
        reverse=True,
    )
