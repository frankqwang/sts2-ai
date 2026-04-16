"""Act 1 encounter 机制配置。"""

from __future__ import annotations

from networkV2.s2_config.mechanism_registry import MechanismRegistry


def register_act1(registry: MechanismRegistry) -> None:
    """注册 Act 1 所有需要特殊机制配置的 encounter。"""
    from networkV2.s2_config.act1.elites import register_act1_elites
    from networkV2.s2_config.act1.bosses import register_act1_bosses
    register_act1_elites(registry)
    register_act1_bosses(registry)
