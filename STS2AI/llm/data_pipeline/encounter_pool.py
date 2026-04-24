"""Act1 Ironclad 启发式 rollout 用的 encounter + build 池。

- encounter_id 必须是 sim 能识别的大写形式（GAME_CATALOG 里的是小写，reset 接口用大写）
- 每个 encounter 的难度大致递增
- 只挑 Ironclad 容易打得赢的基础编队，避免在未调教的启发式老师下全军覆没
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


__all__ = [
    "ACT1_POOL",
    "EncounterSpec",
]
