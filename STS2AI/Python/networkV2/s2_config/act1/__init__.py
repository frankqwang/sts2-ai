"""Act 1 encounter 机制配置。

**已废弃手写配置**：见 SCHEMA_CONVENTION.md，所有 encounter 机制现在由
`mechanism_registry._auto_derive_configs` 从 `GAME_CATALOG` 运行时派生。
本模块保留仅为 backward-compat（外部代码仍 import register_act1）。
"""

from __future__ import annotations

from networkV2.s2_config.mechanism_registry import MechanismRegistry


def register_act1(registry: MechanismRegistry) -> None:
    """已废弃。Auto-derived now. No-op to stay backward-compat."""
    # 手写 bosses/elites 已删除；primitive 从 GAME_CATALOG 派生
    pass
