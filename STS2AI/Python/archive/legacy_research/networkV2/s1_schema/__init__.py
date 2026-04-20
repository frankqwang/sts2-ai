"""Canonical schema 定义：9 类对象 + 7 组 token bank。"""

from networkV2.s1_schema.primitives import (
    MechanismPrimitive,
    ModifierPrimitive,
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
)
from networkV2.s1_schema.entities import (
    CardSemantics,
    RelicSemantics,
    PotionSemantics,
    EnemySemantics,
    PlayerRuntime,
    HandCardRuntime,
    EnemyRuntime,
    IntentInfo,
    PileSummary,
)
from networkV2.s1_schema.memory import (
    PlayedAction,
    TurnPrefixMemory,
    CombatMemory,
    RunBuildMemory,
)
from networkV2.s1_schema.actions import ActionCandidate
from networkV2.s1_schema.token_banks import (
    Token, TokenBank, SharedWorldBanks, CombatBanks, UnifiedTokenBanks,
    CombatTokenBanks,  # 向后兼容别名
)
