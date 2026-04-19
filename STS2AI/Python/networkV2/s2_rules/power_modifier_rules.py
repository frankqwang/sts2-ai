"""Level 2: Power → RuleModifier 规则表。

这些规则覆盖所有怪物（不限于 boss）身上的"特殊行为 power"。
普通数值 power（strength/weak/vulnerable 等）不在此处——它们直接
存在 RuntimeInstances.powers 里作为 Level 1 数值字段。

运行时 modifier inferer 会扫描每个敌人的 power 列表，
对匹配的 power 自动生成 RuleModifier token。

添加新规则: 在 POWER_MODIFIER_RULES dict 中加一条即可。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from networkV2.s1_schema.primitives import (
    ModifierPrimitive,
    ModifierType,
    SourceKind,
    Scope,
    OnHitTrigger,
    OnPlayTrigger,
    DamageCap,
    TargetRestriction,
    ThresholdGate,
)
from networkV2.s1_schema.entities import EnemyRuntime


# ---------------------------------------------------------------------------
# 自动映射表: power_id (小写) → ModifierPrimitive 模板
#
# 运行时 inferer 会用 enemy 的 power stacks 填充 current_value
# ---------------------------------------------------------------------------

POWER_MODIFIER_RULES: dict[str, ModifierPrimitive] = {
    # ---- 受击触发 ----
    "thorns": OnHitTrigger(
        effect="reflect_damage",
        description="荆棘：受到攻击时对攻击者造成等量伤害",
        source_kind=SourceKind.AUTO,
    ),
    "curl_up": OnHitTrigger(
        effect="gain_block_once",
        triggers_once=True,
        description="蜷缩：首次受击时获得等量格挡",
        source_kind=SourceKind.AUTO,
    ),
    "angry": OnHitTrigger(
        effect="gain_strength",
        description="愤怒：受到攻击时获得力量",
        source_kind=SourceKind.AUTO,
    ),
    "plated_armor": OnHitTrigger(
        effect="lose_plating",
        description="镀层护甲：受击时减少 1 层",
        source_kind=SourceKind.AUTO,
    ),
    "spore_cloud": OnHitTrigger(
        effect="apply_debuff_on_death",
        description="孢子云：死亡时对玩家施加 debuff",
        source_kind=SourceKind.AUTO,
    ),

    # ---- 出牌触发 ----
    "enrage": OnPlayTrigger(
        trigger_card_type="skill",
        effect="gain_strength",
        description="激怒：玩家打 skill 时 boss 获得力量",
        source_kind=SourceKind.AUTO,
    ),

    # ---- 伤害上限 ----
    "intangible": DamageCap(
        cap_value=1,
        scope=Scope.PER_HIT,
        description="无实体：所有受到的伤害降为 1",
        source_kind=SourceKind.AUTO,
    ),
    "slippery": DamageCap(
        cap_value=1,
        scope=Scope.PER_HIT,
        description="滑溜：每次受击伤害上限 1，每次受击减 1 层",
        source_kind=SourceKind.AUTO,
    ),
    "hardtokill": DamageCap(
        cap_value=1,
        scope=Scope.PER_HIT,
        description="难以击杀：伤害上限等于层数",
        source_kind=SourceKind.AUTO,
    ),

    # ---- 其他特殊行为 ----
    "flight": DamageCap(
        cap_value=1,
        scope=Scope.PER_HIT,
        description="飞行：未受到攻击的回合恢复飞行层数",
        source_kind=SourceKind.AUTO,
    ),
    "mode_shift": OnHitTrigger(
        effect="phase_shift",
        description="形态转换：累计受伤达到阈值后切换形态",
        source_kind=SourceKind.AUTO,
    ),
    "ritual": OnPlayTrigger(
        trigger_card_type="any",
        effect="gain_strength_eot",
        description="仪式：每回合结束时获得力量",
        source_kind=SourceKind.AUTO,
    ),
}


# ---------------------------------------------------------------------------
# Level 1: 通用数值 power，直接存 RuntimeInstances.powers，不生成 modifier token
# ---------------------------------------------------------------------------

LEVEL1_NUMERIC_POWERS = frozenset({
    "strength", "dexterity", "weak", "vulnerable", "frail",
    "poison", "regen", "metallicize", "artifact", "vigor",
    "barricade", "buffer", "entangled", "lockon",
    "rage", "calamity",
})


def is_level1_power(power_name: str) -> bool:
    """判断是否为 Level 1 通用数值 power。"""
    return power_name.lower() in LEVEL1_NUMERIC_POWERS


def build_power_modifiers(enemy: EnemyRuntime) -> list[ModifierPrimitive]:
    """根据敌人的 power 列表构建 Level 2 modifier。

    返回的每个 modifier 已填充 active=True, current_value=stacks, owner_id。
    """
    modifiers: list[ModifierPrimitive] = []
    for power_name, stacks in enemy.powers.items():
        if stacks <= 0:
            continue
        lower = power_name.lower()
        if lower in LEVEL1_NUMERIC_POWERS:
            continue
        template = POWER_MODIFIER_RULES.get(lower)
        if template is None:
            continue
        # 复制模板并填充运行时值
        mod = replace(
            template,
            active=True,
            current_value=float(stacks),
            owner_id=enemy.entity_id,
        )
        # 对于 DamageCap，hardtokill 的 cap_value = stacks
        if isinstance(mod, DamageCap) and lower == "hardtokill":
            mod = replace(mod, cap_value=stacks)
        modifiers.append(mod)
    return modifiers
