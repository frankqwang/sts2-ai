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


def normalize_monster_id(mid: str) -> str:
    """Monster id 归一化：去下划线/连字符、转小写。

    GAME_CATALOG 的 monster id 是紧凑格式（"frogknight"、"lagavulinmatriarch"），
    而 sim runtime obs 里 enemies[].id 常带下划线（"frog_knight"）。为了反查匹配，
    两边都 normalize 到同一形式。

    公共 helper：供 compiler 层 resolve primitive.owner_id → runtime EnemyRuntime。
    """
    return str(mid or "").lower().replace("_", "").replace("-", "").strip()


# 内部 alias 保留
_normalize_monster_id = normalize_monster_id


class MechanismRegistry:
    """全局 encounter 机制注册表。"""

    def __init__(self) -> None:
        self._configs: dict[str, EncounterMechanismConfig] = {}
        # encounter_id → frozenset(normalized monster_id)，供 find_encounter_id 反查。
        # normalized = 去下划线/连字符、lowercase
        self._encounter_monsters: dict[str, frozenset[str]] = {}
        # encounter_id → room_type（normalized lowercase）
        self._encounter_room_type: dict[str, str] = {}

    def register(
        self,
        config: EncounterMechanismConfig,
        monsters: list[str] | None = None,
    ) -> None:
        """注册一个 encounter 的机制配置。

        Args:
            config: 机制配置。
            monsters: encounter 里的 monster id 列表（可选）。传入后 find_encounter_id
                能用 monster 组合反查 encounter_id，解决 sim 不返回 encounter_id 时
                fallback 拼错 key 的问题（P1-1）。
        """
        key = config.encounter_id.lower().strip()
        self._configs[key] = config
        if monsters:
            self._encounter_monsters[key] = frozenset(
                _normalize_monster_id(m) for m in monsters if m
            )
        self._encounter_room_type[key] = (config.room_type or "").lower().strip()

    def get(self, encounter_id: str) -> EncounterMechanismConfig | None:
        """查询 encounter 的机制配置。返回 None 表示无特殊机制。"""
        return self._configs.get(encounter_id.lower().strip())

    def has(self, encounter_id: str) -> bool:
        return encounter_id.lower().strip() in self._configs

    def find_encounter_id(
        self,
        monsters: list[str],
        room_type: str = "",
    ) -> str | None:
        """根据 monster 集合 + room_type 反查正式 encounter_id。

        用于 sim 不返回 encounter_id 时的 fallback。注册的 key 形如 "frog_knight_normal"
        / "queen_boss"，而 sim 的 enemies[].id 只有 monster 名，拼法对不上。

        匹配顺序：
          1. monster set 完全一致 且 room_type 一致 → exact match
          2. monster set 完全一致（忽略 room_type）→ 放宽 fallback
          3. None（compiler 会返回空 mechanism_bank）
        """
        target = frozenset(
            _normalize_monster_id(m) for m in monsters if m
        )
        if not target:
            return None
        target_rt = (room_type or "").lower().strip()
        # exact: monsters + room_type
        if target_rt:
            for eid, mset in self._encounter_monsters.items():
                if mset == target and self._encounter_room_type.get(eid) == target_rt:
                    return eid
        # monster-only
        for eid, mset in self._encounter_monsters.items():
            if mset == target:
                return eid
        return None

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
    """从 GAME_CATALOG（sqlite / sim API）派生所有 encounter 机制配置。

    **规范**（SCHEMA_CONVENTION.md）：不手写 encounter_id / power name 列表。
    所有 primitive 从真实 monster powers_json 自动派生。

    派生规则（power class name → primitive）：
      - PlatingPower           → ShieldProgress (block-regenerating shield)
      - BarricadePower         → PhaseTransition (block 保留状态)
      - RitualPower/EnragePower → ThresholdGate (scaling threat)
      - MinionPower            → SummonCycle (has minions)
      - IntangiblePower        → DamageCap (所有伤害降 1)
      - HardToKillPower        → DamageCap (按层数)
      - FlightPower            → DamageCap (未受攻击回合恢复)
      - SlipperyPower          → DamageCap
      - ThornsPower/CurlUpPower → OnHitTrigger
    """
    # Deprecated: 不再从 act1/bosses.py 和 act1/elites.py 加载手写配置
    # 所有配置由 _auto_derive_configs 从 GAME_CATALOG 派生
    _auto_derive_configs(registry)


def _power_stem(class_name: str) -> str:
    """PowerClass 运行时 key：strip "Power"/"power" 后 lowercase。

    对齐 `runtime_compiler._normalize_power_id`：
      PlatingPower   → plating
      HardenedShell  → hardenedshell  (runtime 已转 lower，无 Power 后缀)
      HardToKillPower → hardtokill
    """
    s = str(class_name or "")
    for suffix in ("Power", "power"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s.lower()


# #8: base_class → 通用 primitive factory 映射
# 覆盖没进 power_to_primitive_gens 手写 5 条的所有 power，只要它们有明确父类。
# 精细 effect 值从 class name stem 推（统一命名 "on_attacked_{stem}" / "cap_{stem}" 等），
# 网络通过 owner_id + class_name metadata 仍能区分同组不同子类 power。

def _factory_on_hit_trigger(class_name: str, owner_id: str):
    from networkV2.s1_schema.primitives import OnHitTrigger, SourceKind
    stem = _power_stem(class_name)
    return OnHitTrigger(
        effect=f"on_attacked_{stem}",
        triggers_once=False,
        owner_id=owner_id,
        description=f"{owner_id}: {class_name} (auto, TriggerOnAttackedPower)",
        source_kind=SourceKind.CONFIG,
    )


def _factory_on_play_trigger(class_name: str, owner_id: str):
    from networkV2.s1_schema.primitives import OnPlayTrigger, SourceKind
    stem = _power_stem(class_name)
    return OnPlayTrigger(
        trigger_card_type="any",
        effect=f"on_card_played_{stem}",
        owner_id=owner_id,
        description=f"{owner_id}: {class_name} (auto, TriggerOnCardPlayedPower)",
        source_kind=SourceKind.CONFIG,
    )


def _factory_threshold_gate_turn(class_name: str, owner_id: str):
    """回合末/回合初触发的 power → ThresholdGate (threshold_type='turn_count')。"""
    from networkV2.s1_schema.primitives import ThresholdGate, SourceKind
    stem = _power_stem(class_name)
    return ThresholdGate(
        threshold_type="turn_count",
        threshold_value=1.0,
        detect_triggered=lambda e, _s=stem: int(e.powers.get(_s, 0) or 0) > 0,
        owner_id=owner_id,
        description=f"{owner_id}: {class_name} (auto, TriggerEndOfTurnPower)",
        source_kind=SourceKind.CONFIG,
    )


def _factory_damage_cap(class_name: str, owner_id: str):
    """DamageReductionPower 子类 → DamageCap(cap_value=1, per-hit)。"""
    from networkV2.s1_schema.primitives import DamageCap, SourceKind, Scope
    stem = _power_stem(class_name)
    # HardToKill 的 cap 按 stacks 动态；其他默认 1
    cap_value = 1
    return DamageCap(
        cap_value=cap_value,
        scope=Scope.PER_HIT,
        owner_id=owner_id,
        active_when=lambda e, _s=stem: int(e.powers.get(_s, 0) or 0) > 0,
        description=f"{owner_id}: {class_name} (auto, DamageReductionPower, cap={cap_value})",
        source_kind=SourceKind.CONFIG,
    )


def _factory_phase_transition(class_name: str, owner_id: str):
    from networkV2.s1_schema.primitives import PhaseTransition, SourceKind
    stem = _power_stem(class_name)
    return PhaseTransition(
        phase_id=stem,
        trigger=lambda e, _s=stem: int(e.powers.get(_s, 0) or 0) > 0,
        owner_id=owner_id,
        priority=1,
        description=f"{owner_id}: {class_name} (auto, PhaseTransitionPower)",
        source_kind=SourceKind.CONFIG,
    )


def _factory_summon_cycle(class_name: str, owner_id: str):
    from networkV2.s1_schema.primitives import SummonCycle, SourceKind
    stem = _power_stem(class_name)
    return SummonCycle(
        summon_id=stem,
        interval_turns=0,
        detect_active=lambda e: True,
        owner_id=owner_id,
        description=f"{owner_id}: {class_name} (auto, MinionSpawnerPower)",
        source_kind=SourceKind.CONFIG,
    )


# base_class（按 game_catalog 的继承链返回）→ factory
# 命名和 game_vocab._BASE_CLASS_TO_GROUP 一致，只是这里返回 factory（不是 group 标签）。
BASE_CLASS_TO_PRIMITIVE_FACTORY: dict[str, callable] = {
    "TriggerOnAttackedPower":    _factory_on_hit_trigger,
    "TriggerOnHitPower":         _factory_on_hit_trigger,
    "TriggerOnCardPlayedPower":  _factory_on_play_trigger,
    "TriggerOnCardPlayPower":    _factory_on_play_trigger,
    "TriggerEndOfTurnPower":     _factory_threshold_gate_turn,
    "TriggerStartOfTurnPower":   _factory_threshold_gate_turn,
    "DamageReductionPower":      _factory_damage_cap,
    "MinionSpawnerPower":        _factory_summon_cycle,
    "PhaseTransitionPower":      _factory_phase_transition,
}


def _auto_derive_configs(registry: MechanismRegistry) -> None:
    """遍历 GAME_CATALOG 所有 encounter，为每个 encounter 生成 mechanism config。

    派生优先级（#8 优化）：
      1. `power_to_primitive_gens`（手写 5 条精细 primitive，带具体 detect/trigger）
      2. `BASE_CLASS_TO_PRIMITIVE_FACTORY`（按 game_catalog base_classes 批量派生）
      3. 跳过（无匹配规则）

    启用 sim game_catalog 后，大多数 power 都能通过 base_classes 命中 factory，
    mechanism_bank 覆盖率从 14/88 encounter → 接近 100% 有 power 的 encounter。
    """
    from networkV2.s1_schema.sim_catalog import GAME_CATALOG
    from networkV2.s1_schema.primitives import SourceKind

    # Power class → primitive 生成器（手写精细版，优先级最高）
    # 每个条目 (power_class, primitive_factory)
    power_to_primitive_gens: dict[str, callable] = {
        "PlatingPower": lambda owner_id: ShieldProgress(
            total_layers=5,  # 实际 layers 运行时从 state 读
            detect_current_layers=lambda e, _oid=owner_id: int(e.powers.get("plating", 0) or 0),
            detect_broken=lambda e, _oid=owner_id: int(e.powers.get("plating", 0) or 0) <= 0,
            owner_id=owner_id,
            description=f"{owner_id}: Plating（回合末加 block，回合初减 1）",
            source_kind=SourceKind.CONFIG,
        ),
        "BarricadePower": lambda owner_id: PhaseTransition(
            phase_id="barricaded",
            trigger=lambda e: int(e.powers.get("barricade", 0) or 0) > 0,
            owner_id=owner_id,
            priority=1,
            description=f"{owner_id}: Barricade（block 保留）",
        ),
        "RitualPower": lambda owner_id: ThresholdGate(
            threshold_type="turn_count",
            threshold_value=1.0,
            detect_triggered=lambda e: int(e.powers.get("ritual", 0) or 0) > 0,
            owner_id=owner_id,
            description=f"{owner_id}: Ritual（每回合 +strength）",
            source_kind=SourceKind.CONFIG,
        ),
        "EnragePower": lambda owner_id: OnPlayTrigger(
            trigger_card_type="skill",
            effect="gain_strength",
            owner_id=owner_id,
            description=f"{owner_id}: Enrage（打 skill 时获得 strength）",
            source_kind=SourceKind.CONFIG,
        ),
        "MinionPower": lambda owner_id: SummonCycle(
            summon_id="minion",
            interval_turns=0,
            detect_active=lambda e: True,
            owner_id=owner_id,
            description=f"{owner_id}: Minion（非主体敌人）",
            source_kind=SourceKind.CONFIG,
        ),
    }

    for enc in GAME_CATALOG.encounters():
        eid = enc["encounter_id"]
        rt = enc["room_type"]
        config = EncounterMechanismConfig(
            encounter_id=eid.lower(),
            room_type=rt,
            description=f"Auto-derived from monster powers for {eid}",
        )
        # 记录 encounter 里所有 monster id，供 MechanismRegistry.find_encounter_id
        # 反查（训练 fallback 时 sim 可能不返回 encounter_id，用 monster 集合反查）。
        monsters_in_enc: list[str] = []
        has_any = False
        for mid in GAME_CATALOG.encounter_monsters(eid):
            monsters_in_enc.append(mid)
            powers = GAME_CATALOG.monster_powers(mid)
            for p in powers:
                # #8 派生优先级：
                #   1. 手写 power_to_primitive_gens (精细 detect/trigger)
                #   2. base_classes → BASE_CLASS_TO_PRIMITIVE_FACTORY 批量派生
                gen = power_to_primitive_gens.get(p)
                prim = None
                if gen is not None:
                    prim = gen(mid)
                else:
                    # fallback: 查 base_classes
                    base_classes = GAME_CATALOG.power_base_classes(p)
                    for base in base_classes:
                        factory = BASE_CLASS_TO_PRIMITIVE_FACTORY.get(base)
                        if factory is not None:
                            prim = factory(p, mid)
                            break
                if prim is None:
                    continue
                # 按 primitive 类型放到对应字段
                if isinstance(prim, PhaseTransition):
                    config.phases.append(prim)
                elif isinstance(prim, Window):
                    config.windows.append(prim)
                elif isinstance(prim, SummonCycle):
                    config.summon_cycles.append(prim)
                elif isinstance(prim, ThresholdGate):
                    config.threshold_gates.append(prim)
                elif isinstance(prim, ShieldProgress):
                    config.shield_progress.append(prim)
                elif isinstance(prim, OnPlayTrigger):
                    config.on_play_triggers.append(prim)
                elif isinstance(prim, OnHitTrigger):
                    config.on_hit_triggers.append(prim)
                elif isinstance(prim, DamageCap):
                    config.damage_caps.append(prim)
                has_any = True
        # 仅注册有 primitive 的 encounter（空 config 没意义）
        if has_any:
            registry.register(config, monsters=monsters_in_enc)
