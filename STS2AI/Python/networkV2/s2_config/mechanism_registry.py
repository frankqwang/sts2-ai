"""Mechanism 全局注册表。

每个 encounter 的机制配置由 EncounterMechanismConfig 描述，
包含该 encounter 拥有的所有 mechanism primitive 和 modifier primitive。

使用方式:
    registry = get_registry()
    config = registry.get("hexaghost")
    if config:
        # 用 config.phases / config.damage_caps 等做运行时推断
        ...

添加新 boss:
    在 config/act1/bosses.py 中调用 registry.register(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from networkV2.s1_schema.primitives import (
    PhaseTransition,
    Window,
    SummonCycle,
    ThresholdGate,
    ShieldProgress,
    DamageCap,
    TargetRestriction,
    EffectScaling,
    OnPlayTrigger,
    OnHitTrigger,
    DrawModifier,
    ExhaustModifier,
    PhaseTransitionEffect,
    MechanismPrimitive,
    ModifierPrimitive,
)


@dataclass
class EncounterMechanismConfig:
    """一个 encounter 的完整机制配置。

    每个 encounter 是多个 primitive 的组合，不是一个独立类。
    没有复杂机制的普通怪可以用空列表注册（或不注册，registry 会返回 None）。
    """
    encounter_id: str
    # Mechanism primitives
    phases: list[PhaseTransition] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    summon_cycles: list[SummonCycle] = field(default_factory=list)
    threshold_gates: list[ThresholdGate] = field(default_factory=list)
    shield_progress: list[ShieldProgress] = field(default_factory=list)
    # Modifier primitives (Level 3: 手工配置的 modifier，补充 auto_modifier_rules)
    damage_caps: list[DamageCap] = field(default_factory=list)
    target_restrictions: list[TargetRestriction] = field(default_factory=list)
    effect_scalings: list[EffectScaling] = field(default_factory=list)
    on_play_triggers: list[OnPlayTrigger] = field(default_factory=list)
    on_hit_triggers: list[OnHitTrigger] = field(default_factory=list)
    draw_modifiers: list[DrawModifier] = field(default_factory=list)
    exhaust_modifiers: list[ExhaustModifier] = field(default_factory=list)
    phase_transition_effects: list[PhaseTransitionEffect] = field(default_factory=list)
    # 元信息
    room_type: str = "monster"   # "monster" / "elite" / "boss"
    description: str = ""

    @property
    def all_mechanisms(self) -> list[MechanismPrimitive]:
        result: list[MechanismPrimitive] = []
        result.extend(self.phases)
        result.extend(self.windows)
        result.extend(self.summon_cycles)
        result.extend(self.threshold_gates)
        result.extend(self.shield_progress)
        return result

    @property
    def all_config_modifiers(self) -> list[ModifierPrimitive]:
        """只返回 Level 3 手工配置的 modifier。Level 2 auto modifier 由 auto_modifier_rules 处理。"""
        result: list[ModifierPrimitive] = []
        result.extend(self.damage_caps)
        result.extend(self.target_restrictions)
        result.extend(self.effect_scalings)
        result.extend(self.on_play_triggers)
        result.extend(self.on_hit_triggers)
        result.extend(self.draw_modifiers)
        result.extend(self.exhaust_modifiers)
        result.extend(self.phase_transition_effects)
        return result

    @property
    def has_mechanisms(self) -> bool:
        return len(self.all_mechanisms) > 0

    @property
    def has_config_modifiers(self) -> bool:
        return len(self.all_config_modifiers) > 0


class MechanismRegistry:
    """全局 encounter 机制注册表。"""

    def __init__(self) -> None:
        self._configs: dict[str, EncounterMechanismConfig] = {}

    def register(self, config: EncounterMechanismConfig) -> None:
        """注册一个 encounter 的机制配置。"""
        key = config.encounter_id.lower().strip()
        self._configs[key] = config

    def get(self, encounter_id: str) -> EncounterMechanismConfig | None:
        """查询 encounter 的机制配置。返回 None 表示无特殊机制。"""
        return self._configs.get(encounter_id.lower().strip())

    def has(self, encounter_id: str) -> bool:
        return encounter_id.lower().strip() in self._configs

    @property
    def registered_ids(self) -> list[str]:
        return list(self._configs.keys())

    def __len__(self) -> int:
        return len(self._configs)


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_GLOBAL_REGISTRY: MechanismRegistry | None = None


def get_registry() -> MechanismRegistry:
    """获取全局 mechanism registry（懒加载，首次调用时注册所有配置）。"""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = MechanismRegistry()
        _load_all_configs(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def _load_all_configs(registry: MechanismRegistry) -> None:
    """加载所有 act 的 encounter 配置。"""
    from networkV2.s2_config.act1 import register_act1
    register_act1(registry)
    # 后续 act 在此追加:
    # from networkV2.s2_config.act2 import register_act2
    # register_act2(registry)
