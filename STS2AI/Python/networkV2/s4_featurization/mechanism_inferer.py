"""Mechanism 状态推断器：从 encounter ruleset + 运行时状态推断当前机制状态。"""

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
from networkV2.s2_rules.encounter_registry import (
    EncounterRuleset, normalize_monster_id,
)


def _resolve_owner_enemy(
    owner_id: str,
    enemies: list[EnemyRuntime],
    fallback: EnemyRuntime | None,
) -> EnemyRuntime | None:
    """按 primitive.owner_id 找对应 runtime EnemyRuntime。

    primitive.owner_id 来自 GAME_CATALOG（紧凑格式 "frogknight"），runtime enemy 的
    enemy_id/entity_id 可能是带下划线的 sim 格式（"frog_knight"）。normalize 后匹配。
    查不到才返回 fallback（通常是 primary_enemy）—— 这里保留 fallback 是为了兼容
    registry 里 owner_id 缺失或拼写不一致的边缘 case，**不是**正常路径。

    P1-2 修复：原先所有 primitive 都硬绑 primary_enemy，boss+adds / 多怪 encounter
    的 minion/add primitive 会全挂到 HP 最大的那只怪身上。改成按 owner_id 查真正的
    目标 enemy。
    """
    if not owner_id:
        return fallback
    normalized = normalize_monster_id(owner_id)
    for e in enemies:
        for candidate in (getattr(e, "enemy_id", ""), getattr(e, "entity_id", "")):
            if candidate and normalize_monster_id(candidate) == normalized:
                return e
    return fallback


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


class MechanismInferer:
    """从 encounter ruleset + 运行时敌人状态推断 MechanismStates。"""

    def infer(
        self,
        enemies: list[EnemyRuntime],
        ruleset: EncounterRuleset | None,
    ) -> list[ActiveMechanism]:
        """编译当前活跃的机制状态。

        如果 ruleset 为 None（未注册的 encounter），返回空列表。
        """
        if ruleset is None:
            return []

        results: list[ActiveMechanism] = []

        # 找到主要敌人（通常是 boss/elite 本体）作为 fallback
        # owner_id 缺失或拼错时才用它，正常路径走 _resolve_owner_enemy。
        primary_enemy = self._find_primary_enemy(enemies, ruleset)

        if primary_enemy is None and not enemies:
            return results

        # 编译各类 primitive：每个 primitive 用自己的 owner_id 查 enemy，
        # 查不到才 fallback 到 primary。
        results.extend(self._compile_phases(ruleset.phases, enemies, primary_enemy))
        results.extend(self._compile_windows(ruleset.windows, enemies, primary_enemy))
        results.extend(self._compile_summons(ruleset.summon_cycles, enemies, primary_enemy))
        results.extend(self._compile_thresholds(ruleset.threshold_gates, enemies, primary_enemy))
        results.extend(self._compile_shields(ruleset.shield_progress, enemies, primary_enemy))

        return results

    def _find_primary_enemy(
        self,
        enemies: list[EnemyRuntime],
        ruleset: EncounterRuleset,
    ) -> EnemyRuntime | None:
        """找到 ruleset 对应的主要敌人实体。"""
        encounter_id = ruleset.encounter_id.lower()
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
        enemies: list[EnemyRuntime],
        primary: EnemyRuntime | None,
    ) -> list[ActiveMechanism]:
        results: list[ActiveMechanism] = []
        # 按 priority 排序，保留原语义
        sorted_phases = sorted(phases, key=lambda p: p.priority)
        for phase in sorted_phases:
            enemy = _resolve_owner_enemy(phase.owner_id, enemies, primary)
            if enemy is None:
                continue
            is_active = phase.trigger(enemy)
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
        enemies: list[EnemyRuntime],
        primary: EnemyRuntime | None,
    ) -> list[ActiveMechanism]:
        results: list[ActiveMechanism] = []
        for w in windows:
            enemy = _resolve_owner_enemy(w.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(ActiveMechanism(
                primitive=w,
                is_active=True,
                owner_enemy_id=enemy.entity_id,
                window_open=w.detect_open(enemy),
            ))
        return results

    def _compile_summons(
        self,
        summons: list[SummonCycle],
        enemies: list[EnemyRuntime],
        primary: EnemyRuntime | None,
    ) -> list[ActiveMechanism]:
        results: list[ActiveMechanism] = []
        for s in summons:
            enemy = _resolve_owner_enemy(s.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(ActiveMechanism(
                primitive=s,
                is_active=True,
                owner_enemy_id=enemy.entity_id,
                summon_active=s.detect_active(enemy),
            ))
        return results

    def _compile_thresholds(
        self,
        thresholds: list[ThresholdGate],
        enemies: list[EnemyRuntime],
        primary: EnemyRuntime | None,
    ) -> list[ActiveMechanism]:
        results: list[ActiveMechanism] = []
        for t in thresholds:
            enemy = _resolve_owner_enemy(t.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(ActiveMechanism(
                primitive=t,
                is_active=True,
                owner_enemy_id=enemy.entity_id,
                triggered=t.detect_triggered(enemy),
            ))
        return results

    def _compile_shields(
        self,
        shields: list[ShieldProgress],
        enemies: list[EnemyRuntime],
        primary: EnemyRuntime | None,
    ) -> list[ActiveMechanism]:
        results: list[ActiveMechanism] = []
        for s in shields:
            enemy = _resolve_owner_enemy(s.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(ActiveMechanism(
                primitive=s,
                is_active=True,
                owner_enemy_id=enemy.entity_id,
                current_layers=s.detect_current_layers(enemy),
                broken=s.detect_broken(enemy),
            ))
        return results
