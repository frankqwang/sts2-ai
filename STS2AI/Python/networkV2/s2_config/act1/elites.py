"""Act 1 Elite 机制配置。

STS2 Act 1 Elite 列表（基于 source_knowledge.sqlite）。
部分 elite 没有复杂机制（Level 2 auto modifier 已足够），此处只配置
需要 Level 3 mechanism primitive 的 elite。

注意: STS2 还在 Early Access，boss/elite 列表可能随版本变化。
未注册的 encounter 会使用默认行为（只有 Level 1 + Level 2）。
"""

from __future__ import annotations

from networkV2.s2_config.mechanism_registry import (
    EncounterMechanismConfig,
    MechanismRegistry,
)
from networkV2.s1_schema.primitives import (
    PhaseTransition,
    ThresholdGate,
    OnPlayTrigger,
    OnHitTrigger,
    SourceKind,
    Scope,
)


def register_act1_elites(registry: MechanismRegistry) -> None:
    """注册 Act 1 elite encounter 的机制配置。"""

    # --- Ceremonial Beast ---
    # plow power: 每回合力量递增，是一个 scaling 怪
    # auto_modifier_rules 里 plow 没有，需要手动配
    registry.register(EncounterMechanismConfig(
        encounter_id="ceremonial_beast",
        room_type="elite",
        on_play_triggers=[
            OnPlayTrigger(
                trigger_card_type="any",
                effect="scaling_strength",
                description="仪兽：力量持续递增",
                source_kind=SourceKind.CONFIG,
            ),
        ],
        description="仪兽：力量递增的 scaling elite",
    ))

    # --- Crusher ---
    # 高伤害单体 elite，可能有 threshold 式行为
    registry.register(EncounterMechanismConfig(
        encounter_id="crusher",
        room_type="elite",
        threshold_gates=[
            ThresholdGate(
                threshold_type="hp_percent",
                threshold_value=0.5,
                detect_triggered=lambda e: e.hp_ratio <= 0.5,
                description="Crusher: HP<50% 时行为变化",
                source_kind=SourceKind.CONFIG,
            ),
        ],
        description="Crusher: 高伤害 elite，低血时可能变更行为模式",
    ))

    # --- Byrdonis (鸟群 elite) ---
    # flight power → auto_modifier_rules 已覆盖 (DamageCap)
    # 但有多个 byrdpip 小鸟作为 minion
    registry.register(EncounterMechanismConfig(
        encounter_id="byrdonis",
        room_type="elite",
        description="Byrdonis: 带飞行的鸟群 elite，flight 由 auto_modifier 处理",
    ))

    # --- Fogmog ---
    registry.register(EncounterMechanismConfig(
        encounter_id="fogmog",
        room_type="elite",
        description="Fogmog: Act 1 elite",
    ))
