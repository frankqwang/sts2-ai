"""Act1 Ironclad 启发式 rollout 用的 encounter + build 池。

- encounter_id 必须是 sim 能识别的大写形式（GAME_CATALOG 里的是小写，reset 接口用大写）
- 每个 encounter 的难度大致递增
- 默认训练用 ACT1_WINNABLE_POOL，只放启发式老师能稳定打赢的 encounter/build 组合
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_IRONCLAD_STARTER_BUILD: dict[str, Any] = {
    "deck": [
        {"id": "STRIKE_IRONCLAD"},
        {"id": "STRIKE_IRONCLAD"},
        {"id": "STRIKE_IRONCLAD"},
        {"id": "STRIKE_IRONCLAD"},
        {"id": "STRIKE_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "BASH"},
    ],
    "relics": [
        {"id": "BURNING_BLOOD"},
    ],
    "current_hp": 80,
    "max_hp": 80,
    "max_energy": 3,
    "gold": 99,
}


# 来自 game_bridge_smoke.py 的已验证 build
_IRONCLAD_MIDRUN_BUILD: dict[str, Any] = {
    "deck": [
        {"id": "STRIKE_IRONCLAD"},
        {"id": "STRIKE_IRONCLAD"},
        {"id": "STRIKE_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "BASH"},
        {"id": "POMMEL_STRIKE", "upgrade_level": 1},
        {"id": "SETUP_STRIKE", "upgrade_level": 1},
        {"id": "FORGOTTEN_RITUAL"},
        {"id": "BLUDGEON", "upgrade_level": 1},
        {"id": "CINDER", "upgrade_level": 1},
    ],
    "relics": [
        {"id": "BURNING_BLOOD"},
        {"id": "HAND_DRILL"},
        {"id": "MINIATURE_CANNON"},
        {"id": "SILVER_CRUCIBLE"},
    ],
    "current_hp": 70,
    "max_hp": 80,
    "max_energy": 3,
    "gold": 125,
}


@dataclass(frozen=True)
class EncounterSpec:
    encounter_id: str
    build: dict[str, Any] = field(default_factory=dict)
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def build_fingerprint(build: dict[str, Any]) -> str:
    """Stable short id for a combat build."""
    payload = json.dumps(build or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def encounter_key(spec: EncounterSpec) -> str:
    """Unique key for reward grouping and eval parity.

    The same encounter with different deck/relic builds is a different task.
    Grouping only by encounter_id gives noisy or wrong advantage signals.
    """
    tag = spec.tag or "untagged"
    return f"{spec.encounter_id}::{tag}::{build_fingerprint(spec.build)}"


def encounter_label(spec: EncounterSpec) -> str:
    tag = spec.tag or "untagged"
    return f"{spec.encounter_id}[{tag}:{build_fingerprint(spec.build)}]"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _copy_build_with_floor(case: dict[str, Any]) -> dict[str, Any]:
    build = dict(case.get("build") if isinstance(case.get("build"), dict) else {})
    floor = _safe_int(case.get("floor"), 0)
    if floor > 0:
        build["floor"] = floor
    return build


def skada_case_to_spec(case: dict[str, Any]) -> EncounterSpec:
    """Convert one Skada single-combat case to an EncounterSpec.

    The sim combat_reset schema does not carry floor separately, so floor is
    stored in build/metadata and later injected into rendered rollout states.
    """
    floor = _safe_int(case.get("floor"), 0)
    encounter_type = str(case.get("encounter_type") or "unknown").lower()
    tag = f"skada_floor_{floor:02d}_{encounter_type}" if floor > 0 else f"skada_{encounter_type}"
    return EncounterSpec(
        str(case.get("encounter_id") or "").upper(),
        _copy_build_with_floor(case),
        tag,
        {
            "source": "skada_case",
            "case_id": case.get("case_id"),
            "source_path": case.get("source_path"),
            "source_line": case.get("source_line"),
            "run_id": case.get("run_id"),
            "seed": case.get("seed"),
            "character_id": case.get("character_id"),
            "ascension": case.get("ascension"),
            "floor": floor,
            "encounter_type": case.get("encounter_type"),
            "won": case.get("won"),
            "floor_state": case.get("floor_state") if isinstance(case.get("floor_state"), dict) else {},
            "case_metadata": case.get("metadata") if isinstance(case.get("metadata"), dict) else {},
        },
    )


def load_skada_case_pool(
    case_index: str | Path,
    *,
    character_id: str = "IRONCLAD",
    floor_min: int = 1,
    floor_max: int = 17,
    won_only: bool = True,
    limit: int = 0,
    sample_seed: int = 0,
    sample_mode: str = "stratified",
) -> list[EncounterSpec]:
    """Load floor-aware Skada combat reset cases for rollout training."""
    path = Path(case_index)
    specs: list[EncounterSpec] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(case, dict):
                continue
            if str(case.get("character_id") or "").upper() != character_id.upper():
                continue
            floor = _safe_int(case.get("floor"), 0)
            if floor < floor_min or floor > floor_max:
                continue
            if won_only and case.get("won") is not True:
                continue
            encounter_id = str(case.get("encounter_id") or "").strip()
            build = case.get("build")
            if not encounter_id or not isinstance(build, dict):
                continue
            specs.append(skada_case_to_spec(case))
    if limit <= 0 or len(specs) <= limit:
        return specs
    mode = (sample_mode or "file").strip().lower()
    if mode == "file":
        return specs[:limit]
    rng = random.Random(sample_seed)
    if mode == "random":
        shuffled = list(specs)
        rng.shuffle(shuffled)
        return shuffled[:limit]
    if mode != "stratified":
        raise ValueError(f"unsupported Skada case sample_mode: {sample_mode}")
    return _stratified_case_sample(specs, limit=limit, rng=rng)


def _stratified_case_sample(specs: list[EncounterSpec], *, limit: int, rng: random.Random) -> list[EncounterSpec]:
    groups: dict[tuple[int, str], list[EncounterSpec]] = defaultdict(list)
    for spec in specs:
        meta = spec.metadata if isinstance(spec.metadata, dict) else {}
        floor = _safe_int(meta.get("floor"), 0)
        encounter_type = str(meta.get("encounter_type") or "unknown").lower()
        groups[(floor, encounter_type)].append(spec)

    buckets = sorted(groups)
    for bucket in buckets:
        rng.shuffle(groups[bucket])

    selected: list[EncounterSpec] = []
    cursor = 0
    while len(selected) < limit and buckets:
        bucket = buckets[cursor % len(buckets)]
        items = groups[bucket]
        if items:
            selected.append(items.pop())
        if not items:
            buckets.remove(bucket)
            if not buckets:
                break
            cursor %= len(buckets)
        else:
            cursor += 1
    return selected


def filter_encounter_pool(pool: list[EncounterSpec], query: str = "") -> list[EncounterSpec]:
    """Filter by comma-separated encounter id, tag, key, or label fragments."""
    raw = (query or "").strip()
    if not raw:
        return list(pool)
    tokens = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not tokens:
        return list(pool)

    selected: list[EncounterSpec] = []
    for spec in pool:
        haystack = " ".join(
            [
                spec.encounter_id,
                spec.tag,
                encounter_key(spec),
                encounter_label(spec),
            ]
        ).lower()
        if any(token in haystack for token in tokens):
            selected.append(spec)
    return selected


# Act1 常见战斗（sim reset 接口要大写）
ACT1_POOL: list[EncounterSpec] = [
    EncounterSpec("CHOMPERS_NORMAL",    _IRONCLAD_STARTER_BUILD, "act1_normal"),
    EncounterSpec("SLIMES_NORMAL",      _IRONCLAD_STARTER_BUILD, "act1_normal"),
    EncounterSpec("CULTISTS_NORMAL",    _IRONCLAD_STARTER_BUILD, "act1_normal"),
    EncounterSpec("EXOSKELETONS_NORMAL", _IRONCLAD_STARTER_BUILD, "act1_normal"),
    EncounterSpec("BOWLBUGS_NORMAL",    _IRONCLAD_STARTER_BUILD, "act1_normal"),
    # 中段一点的编队用带遗物 build
    EncounterSpec("CHOMPERS_NORMAL",    _IRONCLAD_MIDRUN_BUILD, "act1_midrun"),
    EncounterSpec("GREMLIN_MERC_NORMAL", _IRONCLAD_MIDRUN_BUILD, "act1_midrun"),
]


# 默认 SFT rollout 池：starter deck 只打较容易的战斗；硬一点的敌人用
# midrun build。避免把“老师自己都打不赢”的失败轨迹当成主要模仿数据。
ACT1_WINNABLE_POOL: list[EncounterSpec] = [
    EncounterSpec("SLIMES_NORMAL",       _IRONCLAD_STARTER_BUILD, "act1_normal"),
    EncounterSpec("CULTISTS_NORMAL",     _IRONCLAD_STARTER_BUILD, "act1_normal"),
    EncounterSpec("SLIMES_NORMAL",       _IRONCLAD_MIDRUN_BUILD, "act1_midrun"),
    EncounterSpec("CULTISTS_NORMAL",     _IRONCLAD_MIDRUN_BUILD, "act1_midrun"),
    EncounterSpec("CHOMPERS_NORMAL",     _IRONCLAD_MIDRUN_BUILD, "act1_midrun"),
    EncounterSpec("BOWLBUGS_NORMAL",     _IRONCLAD_MIDRUN_BUILD, "act1_midrun"),
    EncounterSpec("GREMLIN_MERC_NORMAL", _IRONCLAD_MIDRUN_BUILD, "act1_midrun"),
]


__all__ = [
    "ACT1_POOL",
    "ACT1_WINNABLE_POOL",
    "EncounterSpec",
    "build_fingerprint",
    "encounter_key",
    "encounter_label",
    "filter_encounter_pool",
    "load_skada_case_pool",
    "skada_case_to_spec",
]
