"""MechanismStates 编译器：从 mechanism_config + 运行时状态推断当前机制状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from networkV2.s1_schema.primitives import (
    MechanismPrimitive,
    MechanismType,
    PhaseTransition,
    Window,
    SummonCycle,
    ThresholdGate,
    ShieldProgress,
)
from networkV2.s1_schema.entities import EnemyRuntime
from networkV2.s2_config.mechanism_registry import EncounterMechanismConfig


@dataclass
class ActiveMechanism:
    """运行时的活跃机制状态。"""
    primitive: MechanismPrimitive
    is_active: bool = False
    owner_enemy_id: str = ""
    # Phase 专用
    current_phase_id: str = ""
    # Window 专用
    window_open: bool = False
    # Summon 专用
    summon_active: bool = False
    # Threshold 专用
    triggered: bool = False
    # Shield 专用
    current_layers: int = 0
    broken: bool = False


class MechanismCompiler:
    """从 mechanism_config + 运行时敌人状态编译 MechanismStates。"""

    def compile(
        self,
        enemies: list[EnemyRuntime],
        config: EncounterMechanismConfig | None,
    ) -> list[ActiveMechanism]:
        """编译当前活跃的机制状态。

        如果 config 为 None（未注册的 encounter），返回空列表。
        """
        if config is None:
            return []

        results: list[ActiveMechanism] = []

        # 找到主要敌人（通常是 boss/elite 本体）
        # 优先取 HP 最高的，或者第一个
        primary_enemy = self._find_primary_enemy(enemies, config)

        if primary_enemy is None:
            return results

        # 编译 phases
        results.extend(self._compile_phases(config.phases, primary_enemy))
        # 编译 windows
        results.extend(self._compile_windows(config.windows, primary_enemy))
        # 编译 summon_cycles
        results.extend(self._compile_summons(config.summon_cycles, primary_enemy))
        # 编译 threshold_gates
        results.extend(self._compile_thresholds(config.threshold_gates, primary_enemy))
        # 编译 shield_progress
        results.extend(self._compile_shields(config.shield_progress, primary_enemy))

        return results

    def _find_primary_enemy(
        self,
        enemies: list[EnemyRuntime],
        config: EncounterMechanismConfig,
    ) -> EnemyRuntime | None:
        """找到 mechanism config 对应的主要敌人实体。"""
        encounter_id = config.encounter_id.lower()
        # 精确匹配 enemy_id
        for e in enemies:
            if e.enemy_id == encounter_id:
                return e
        # 如果没有精确匹配，取 max_hp 最大的（通常是 boss 本体）
        if enemies:
            return max(enemies, key=lambda e: e.max_hp)
        return None

    def _compile_phases(
        self,
        phases: list[PhaseTransition],
        enemy: EnemyRuntime,
    ) -> list[ActiveMechanism]:
        results: list[ActiveMechanism] = []
        current_phase = ""
        # 按 priority 排序，取最后一个匹配的 phase
        sorted_phases = sorted(phases, key=lambda p: p.priority)
        for phase in sorted_phases:
            is_active = phase.trigger(enemy)
            if is_active:
                current_phase = phase.phase_id
            results.append(ActiveMechanism(
                primitive=phase,
                is_active=is_active,
                owner_enemy_id=enemy.entity_id,
                current_phase_id=phase.phase_id if is_active else "",
            ))
        return results

    def _compile_windows(
        self,
        windows: list[Window],
        enemy: EnemyRuntime,
    ) -> list[ActiveMechanism]:
        return [
            ActiveMechanism(
                primitive=w,
                is_active=True,
                owner_enemy_id=enemy.entity_id,
                window_open=w.detect_open(enemy),
            )
            for w in windows
        ]

    def _compile_summons(
        self,
        summons: list[SummonCycle],
        enemy: EnemyRuntime,
    ) -> list[ActiveMechanism]:
        return [
            ActiveMechanism(
                primitive=s,
                is_active=True,
                owner_enemy_id=enemy.entity_id,
                summon_active=s.detect_active(enemy),
            )
            for s in summons
        ]

    def _compile_thresholds(
        self,
        thresholds: list[ThresholdGate],
        enemy: EnemyRuntime,
    ) -> list[ActiveMechanism]:
        return [
            ActiveMechanism(
                primitive=t,
                is_active=True,
                owner_enemy_id=enemy.entity_id,
                triggered=t.detect_triggered(enemy),
            )
            for t in thresholds
        ]

    def _compile_shields(
        self,
        shields: list[ShieldProgress],
        enemy: EnemyRuntime,
    ) -> list[ActiveMechanism]:
        return [
            ActiveMechanism(
                primitive=s,
                is_active=True,
                owner_enemy_id=enemy.entity_id,
                current_layers=s.detect_current_layers(enemy),
                broken=s.detect_broken(enemy),
            )
            for s in shields
        ]
