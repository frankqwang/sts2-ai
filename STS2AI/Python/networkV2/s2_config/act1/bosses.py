"""Act 1 Boss 机制配置。

STS2 Act 1 Boss（基于 source_knowledge.sqlite + 游戏实际机制）。
每个 boss 是多个 primitive 的组合。

注意: STS2 EA 版本机制可能变化，配置需要根据实际版本校验。
"""

from __future__ import annotations

from networkV2.s2_config.mechanism_registry import (
    EncounterMechanismConfig,
    MechanismRegistry,
)
from networkV2.s1_schema.primitives import (
    PhaseTransition,
    Window,
    SummonCycle,
    ThresholdGate,
    ShieldProgress,
    DamageCap,
    TargetRestriction,
    OnPlayTrigger,
    OnHitTrigger,
    PhaseTransitionEffect,
    SourceKind,
    Scope,
)


def register_act1_bosses(registry: MechanismRegistry) -> None:
    """注册 Act 1 boss encounter 的机制配置。"""

    # --- Haunted Ship (幽灵船) ---
    # 多阶段 boss，有护盾机制和召唤
    registry.register(EncounterMechanismConfig(
        encounter_id="haunted_ship",
        room_type="boss",
        phases=[
            PhaseTransition(
                phase_id="phase_1",
                trigger=lambda e: e.hp_ratio > 0.5,
                priority=0,
                description="幽灵船：第一阶段",
            ),
            PhaseTransition(
                phase_id="phase_2",
                trigger=lambda e: e.hp_ratio <= 0.5,
                priority=1,
                description="幽灵船：第二阶段（HP<50%）",
            ),
        ],
        description="幽灵船：多阶段 boss",
    ))

    # --- Doormaker / Door ---
    # 两个实体的 boss 战: Doormaker 召唤 Door
    # Door 存活时 Doormaker 可能有 target_restriction
    registry.register(EncounterMechanismConfig(
        encounter_id="doormaker",
        room_type="boss",
        summon_cycles=[
            SummonCycle(
                summon_id="door",
                interval_turns=0,  # 开场就有
                detect_active=lambda e: True,
                description="Doormaker: 持续维护 Door 实体",
                source_kind=SourceKind.CONFIG,
            ),
        ],
        target_restrictions=[
            TargetRestriction(
                restriction_type="must_clear_adds",
                detect_restricted=lambda e: not e.is_hittable,
                description="Doormaker: Door 存活时可能不可直接攻击",
                source_kind=SourceKind.CONFIG,
            ),
        ],
        description="Doormaker: 召唤 Door 的双实体 boss",
    ))

    # --- Devoted Sculptor (虔诚雕塑家) ---
    registry.register(EncounterMechanismConfig(
        encounter_id="devoted_sculptor",
        room_type="boss",
        phases=[
            PhaseTransition(
                phase_id="sculpting",
                trigger=lambda e: e.has_buff("Barricade") or e.block > 0,
                priority=0,
                description="雕塑家：堆叠格挡阶段",
            ),
            PhaseTransition(
                phase_id="attacking",
                trigger=lambda e: not e.has_buff("Barricade") and e.block == 0,
                priority=1,
                description="雕塑家：攻击阶段",
            ),
        ],
        description="虔诚雕塑家：交替堆甲和攻击的 boss",
    ))

    # --- Entomancer (虫术师) ---
    # 召唤虫子的 boss
    registry.register(EncounterMechanismConfig(
        encounter_id="entomancer",
        room_type="boss",
        summon_cycles=[
            SummonCycle(
                summon_id="exoskeleton",
                interval_turns=2,
                detect_active=lambda e: True,
                description="虫术师: 周期性召唤甲壳虫",
                source_kind=SourceKind.CONFIG,
            ),
        ],
        description="虫术师：召唤甲壳虫的 boss",
    ))

    # --- Infested Prism (寄生棱镜) ---
    registry.register(EncounterMechanismConfig(
        encounter_id="infested_prism",
        room_type="boss",
        phases=[
            PhaseTransition(
                phase_id="shielded",
                trigger=lambda e: e.has_buff("Plating") or e.has_buff("HardenedShell"),
                priority=0,
                description="棱镜：护盾阶段",
            ),
            PhaseTransition(
                phase_id="exposed",
                trigger=lambda e: not e.has_buff("Plating") and not e.has_buff("HardenedShell"),
                priority=1,
                description="棱镜：暴露阶段",
            ),
        ],
        windows=[
            Window(
                window_type="vulnerable",
                detect_open=lambda e: not e.has_buff("Plating") and not e.has_buff("HardenedShell"),
                description="棱镜：护盾破碎后的易伤窗口",
                source_kind=SourceKind.CONFIG,
            ),
        ],
        damage_caps=[
            DamageCap(
                cap_value=1,
                scope=Scope.PER_HIT,
                active_when=lambda e: e.has_buff("HardenedShell"),
                description="棱镜: HardenedShell 限制每次伤害为 1",
                source_kind=SourceKind.CONFIG,
            ),
        ],
        description="寄生棱镜：有护盾和易伤窗口的 boss",
    ))

    # --- Fabricator (制造者) ---
    registry.register(EncounterMechanismConfig(
        encounter_id="fabricator",
        room_type="boss",
        summon_cycles=[
            SummonCycle(
                summon_id="cubex_construct",
                interval_turns=3,
                detect_active=lambda e: True,
                description="制造者: 周期性召唤方块构造体",
                source_kind=SourceKind.CONFIG,
            ),
        ],
        description="制造者：召唤构造体的 boss",
    ))
