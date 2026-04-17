"""[DEPRECATED] Act 1 Boss 机制配置。

⚠️ **本文件已废弃**。所有手写 encounter_id / 机制配置都是硬编码，
违反 `docs/design/SCHEMA_CONVENTION.md`。

现在机制 primitive **运行时从 GAME_CATALOG 派生**，见
`networkV2/s2_config/mechanism_registry.py::_auto_derive_configs`。

本文件保留为空壳仅为 backward-compat（避免其他模块 import 失败）。
"""

from __future__ import annotations

from networkV2.s2_config.mechanism_registry import MechanismRegistry


def register_act1_bosses(registry: MechanismRegistry) -> None:
    """已废弃；保留 no-op。机制现由 mechanism_registry 自动派生。"""
    pass
