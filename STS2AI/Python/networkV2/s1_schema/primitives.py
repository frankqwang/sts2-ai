"""Mechanism 和 Modifier 的 primitive 类型定义。

Mechanism primitive (5 种): 描述"战斗进程走到哪了"
  - PhaseTransition: 阶段切换 (HP 阈值、回合数等)
  - Window: 时间窗口 (易伤、攻击、护盾破碎等)
  - SummonCycle: 召唤循环
  - ThresholdGate: 阈值门控 (累计伤害触发)
  - ShieldProgress: 护盾进度 (多层击破)

Modifier primitive (8 种): 描述"当前哪些规则变了"
  - DamageCap: 伤害上限
  - TargetRestriction: 目标限制
  - EffectScaling: 效果缩放
  - OnPlayTrigger: 出牌触发
  - OnHitTrigger: 受击触发
  - DrawModifier: 抽牌修改
  - ExhaustModifier: 消耗修改
  - PhaseTransitionEffect: 阶段切换效果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from networkV2.s1_schema.entities import EnemyRuntime


# ---------------------------------------------------------------------------
# 枚举类型
# ---------------------------------------------------------------------------

class MechanismType(Enum):
    PHASE_TRANSITION = auto()
    WINDOW = auto()
    SUMMON_CYCLE = auto()
    THRESHOLD_GATE = auto()
    SHIELD_PROGRESS = auto()


class ModifierType(Enum):
    DAMAGE_CAP = auto()
    TARGET_RESTRICTION = auto()
    EFFECT_SCALING = auto()
    ON_PLAY_TRIGGER = auto()
    ON_HIT_TRIGGER = auto()
    DRAW_MODIFIER = auto()
    EXHAUST_MODIFIER = auto()
    PHASE_TRANSITION_EFFECT = auto()


class SourceKind(Enum):
    """数据来源类型。"""
    CONFIG = auto()      # Level 3: 手工 mechanism_config
    AUTO = auto()        # Level 2: POWER_MODIFIER_RULES 自动映射
    INFERRED = auto()    # 运行时推断


class Scope(Enum):
    """作用范围。"""
    PER_HIT = auto()
    PER_TURN = auto()
    PER_CARD = auto()
    GLOBAL = auto()
    OWNER = auto()


# ---------------------------------------------------------------------------
# Mechanism Primitives - 战斗进程节点
# ---------------------------------------------------------------------------

@dataclass
class MechanismPrimitive:
    """所有 mechanism primitive 的基类。"""
    mechanism_type: MechanismType
    owner_id: str = ""
    source_kind: SourceKind = SourceKind.CONFIG
    description: str = ""


@dataclass
class PhaseTransition(MechanismPrimitive):
    """阶段切换。

    示例: boss HP < 50% 进入 phase 2
    """
    mechanism_type: MechanismType = field(default=MechanismType.PHASE_TRANSITION, init=False)
    phase_id: str = ""
    trigger: Callable[[EnemyRuntime], bool] = field(default=lambda _: False, repr=False)
    priority: int = 0  # 高优先级的 phase 先检测


@dataclass
class Window(MechanismPrimitive):
    """时间窗口（易伤窗口、攻击窗口等）。

    示例: 护盾破碎后 2 回合内可造成额外伤害
    """
    mechanism_type: MechanismType = field(default=MechanismType.WINDOW, init=False)
    window_type: str = ""  # "vulnerable" / "attack" / "shield_down" / ...
    detect_open: Callable[[EnemyRuntime], bool] = field(default=lambda _: False, repr=False)
    max_duration: int | None = None  # 最长持续回合数, None = 无限制


@dataclass
class SummonCycle(MechanismPrimitive):
    """召唤循环。

    示例: 每 3 回合召唤一个 add
    """
    mechanism_type: MechanismType = field(default=MechanismType.SUMMON_CYCLE, init=False)
    summon_id: str = ""
    interval_turns: int = 0
    detect_active: Callable[[EnemyRuntime], bool] = field(default=lambda _: True, repr=False)


@dataclass
class ThresholdGate(MechanismPrimitive):
    """阈值门控。

    示例: 累计 X 伤害后触发效果 / HP 降到某值触发
    """
    mechanism_type: MechanismType = field(default=MechanismType.THRESHOLD_GATE, init=False)
    threshold_type: str = ""  # "damage_taken" / "hp_percent" / "turn_count"
    threshold_value: float = 0.0
    detect_triggered: Callable[[EnemyRuntime], bool] = field(default=lambda _: False, repr=False)


@dataclass
class ShieldProgress(MechanismPrimitive):
    """护盾进度（多层护盾逐层击破）。

    示例: 3 层护盾，每次攻击破一层
    """
    mechanism_type: MechanismType = field(default=MechanismType.SHIELD_PROGRESS, init=False)
    total_layers: int = 0
    detect_current_layers: Callable[[EnemyRuntime], int] = field(default=lambda _: 0, repr=False)
    detect_broken: Callable[[EnemyRuntime], bool] = field(default=lambda _: False, repr=False)


# ---------------------------------------------------------------------------
# Modifier Primitives - 规则改写
# ---------------------------------------------------------------------------

@dataclass
class ModifierPrimitive:
    """所有 modifier primitive 的基类。"""
    modifier_type: ModifierType
    scope: Scope = Scope.GLOBAL
    owner_id: str = ""
    source_kind: SourceKind = SourceKind.AUTO
    description: str = ""
    # 运行时值：由 compiler 填充
    active: bool = False
    current_value: float = 0.0


@dataclass
class DamageCap(ModifierPrimitive):
    """伤害上限。

    示例: 每次只受 1 点伤害 (intangible / slippery / hardtokill)
    """
    modifier_type: ModifierType = field(default=ModifierType.DAMAGE_CAP, init=False)
    cap_value: int = 1
    scope: Scope = Scope.PER_HIT
    active_when: Callable[[EnemyRuntime], bool] = field(default=lambda _: False, repr=False)


@dataclass
class TargetRestriction(ModifierPrimitive):
    """目标限制。

    示例: adds 存活时不可选中本体
    """
    modifier_type: ModifierType = field(default=ModifierType.TARGET_RESTRICTION, init=False)
    restriction_type: str = ""  # "must_clear_adds" / "untargetable" / "only_aoe"
    detect_restricted: Callable[[EnemyRuntime], bool] = field(default=lambda _: False, repr=False)


@dataclass
class EffectScaling(ModifierPrimitive):
    """效果缩放。

    示例: 按格挡值造成伤害
    """
    modifier_type: ModifierType = field(default=ModifierType.EFFECT_SCALING, init=False)
    scaling_source: str = ""  # "block" / "hp_lost" / "cards_played" / ...
    scaling_factor: float = 1.0


@dataclass
class OnPlayTrigger(ModifierPrimitive):
    """出牌触发。

    示例: 打 skill 时 boss 加 strength (Angry/Enrage)
    """
    modifier_type: ModifierType = field(default=ModifierType.ON_PLAY_TRIGGER, init=False)
    trigger_card_type: str = ""  # "skill" / "attack" / "any"
    effect: str = ""  # "gain_strength" / "gain_block" / ...
    effect_value: float = 0.0


@dataclass
class OnHitTrigger(ModifierPrimitive):
    """受击触发。

    示例: 荆棘 (thorns), 弹甲 (curl_up), 怒意 (angry)
    """
    modifier_type: ModifierType = field(default=ModifierType.ON_HIT_TRIGGER, init=False)
    effect: str = ""  # "reflect_damage" / "gain_block_once" / "gain_strength"
    effect_value: float = 0.0
    triggers_once: bool = False


@dataclass
class DrawModifier(ModifierPrimitive):
    """抽牌修改。

    示例: 本回合抽牌减少 2 / 增加 1
    """
    modifier_type: ModifierType = field(default=ModifierType.DRAW_MODIFIER, init=False)
    draw_delta: int = 0


@dataclass
class ExhaustModifier(ModifierPrimitive):
    """消耗修改。

    示例: 所有卡打出后被消耗
    """
    modifier_type: ModifierType = field(default=ModifierType.EXHAUST_MODIFIER, init=False)
    exhaust_on_play: bool = False
    affected_card_type: str = "any"  # "attack" / "skill" / "any"


@dataclass
class PhaseTransitionEffect(ModifierPrimitive):
    """阶段切换效果。

    示例: 切换时清除所有 debuff / 切换时恢复 HP
    """
    modifier_type: ModifierType = field(default=ModifierType.PHASE_TRANSITION_EFFECT, init=False)
    trigger_phase: str = ""  # 触发的 phase_id
    effect: str = ""  # "clear_debuffs" / "heal" / "gain_strength" / ...
    effect_value: float = 0.0
