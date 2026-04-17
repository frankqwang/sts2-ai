"""Encounter → integer index 映射（Conditional Policy 用）。

用途：
  UnifiedNet 的 encounter conditioning 需要把 encounter_id 映射成 int index，
  然后查 `nn.Embedding(n_encounters, d_model)` 得到 boss-specific 向量，
  再注入 decision_repr。

规范（SCHEMA_CONVENTION.md）：
  vocab 从 GAME_CATALOG 派生，**不硬写**任何 encounter_id。
  单次启动 cache，保证不同 worker / attach_sim 前后一致。

索引约定：
  0 = UNKNOWN / 非战斗 / 未在 catalog 中（safe fallback，embedding 0 初始化）
  1..N = 按字典序排列的 encounter_id (lowercase)
  超过 n_encounters 的 id 自动 map 到 0
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


UNKNOWN_INDEX = 0


@lru_cache(maxsize=1)
def _build_vocab() -> tuple[list[str], dict[str, int]]:
    """从 GAME_CATALOG 派生 encounter vocab。

    返回 (sorted_ids, id→index dict)。index 0 保留给 UNKNOWN。
    """
    from networkV2.s1_schema.sim_catalog import GAME_CATALOG
    ids = set()
    for enc in GAME_CATALOG.encounters():
        eid = str(enc.get("encounter_id", "")).lower().strip()
        if eid:
            ids.add(eid)
    sorted_ids = sorted(ids)
    # index 0 = UNKNOWN，真实 encounter 从 1 开始
    id_to_idx = {eid: i + 1 for i, eid in enumerate(sorted_ids)}
    return sorted_ids, id_to_idx


def encounter_to_index(encounter_id: str | None, *, max_index: int | None = None) -> int:
    """Encounter id → int index。

    未知 / 空 → UNKNOWN_INDEX (0)。
    超过 max_index（若指定）的 → 也映射 0。
    """
    if not encounter_id:
        return UNKNOWN_INDEX
    eid = str(encounter_id).lower().strip()
    if not eid:
        return UNKNOWN_INDEX
    try:
        _, id_to_idx = _build_vocab()
    except Exception:
        return UNKNOWN_INDEX
    idx = id_to_idx.get(eid, UNKNOWN_INDEX)
    if max_index is not None and idx >= max_index:
        return UNKNOWN_INDEX
    return idx


def vocab_size() -> int:
    """当前 catalog 里 encounter 数量 + 1 (UNKNOWN)。"""
    try:
        sorted_ids, _ = _build_vocab()
        return len(sorted_ids) + 1
    except Exception:
        return 1


def index_to_encounter(idx: int) -> str | None:
    """反查（诊断用）。idx=0 → None (UNKNOWN)。"""
    if idx <= 0:
        return None
    try:
        sorted_ids, _ = _build_vocab()
    except Exception:
        return None
    i = idx - 1
    if 0 <= i < len(sorted_ids):
        return sorted_ids[i]
    return None
