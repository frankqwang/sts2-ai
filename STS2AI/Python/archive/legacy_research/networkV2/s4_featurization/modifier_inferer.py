"""RuleModifier 推断器：Level 2 power rules + Level 3 encounter ruleset 合并。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from networkV2.s1_schema.primitives import (
    ModifierPrimitive,
    DamageCap,
    TargetRestriction,
)
from networkV2.s1_schema.entities import EnemyRuntime
from networkV2.s2_rules.encounter_registry import (
    EncounterRuleset, normalize_monster_id,
)
from networkV2.s2_rules.power_modifier_rules import build_power_modifiers


def _resolve_owner_enemy(
    owner_id: str,
    enemies: list[EnemyRuntime],
    fallback: EnemyRuntime | None,
) -> EnemyRuntime | None:
    """见 mechanism_inferer._resolve_owner_enemy —— 同语义，复制过来避免跨文件
    import。modifier 侧也用 normalize 匹配，保证多怪 encounter 的 ruleset modifier
    不被错绑到 primary_enemy。"""
    if not owner_id:
        return fallback
    normalized = normalize_monster_id(owner_id)
    for e in enemies:
        for candidate in (getattr(e, "enemy_id", ""), getattr(e, "entity_id", "")):
            if candidate and normalize_monster_id(candidate) == normalized:
                return e
    return fallback


class ModifierInferer:
    """推断当前活跃的 RuleModifiers。

    合并两个来源：
      Level 2: 从 enemy.powers 自动映射 (power_modifier_rules)
      Level 3: 从 encounter ruleset 手工规则
    """

    def infer(
        self,
        enemies: list[EnemyRuntime],
        ruleset: EncounterRuleset | None,
    ) -> list[ModifierPrimitive]:
        modifiers: list[ModifierPrimitive] = []

        # Level 2: 对每个敌人，自动映射 power → modifier
        for enemy in enemies:
            auto_mods = build_power_modifiers(enemy)
            modifiers.extend(auto_mods)

        # Level 3: 从 ruleset 获取手工规则 modifier。
        # P1-2 修复：每个 modifier 用它自己的 owner_id 查对应 enemy，不再全绑 primary。
        if ruleset is not None and enemies:
            primary = self._find_primary(enemies, ruleset)
            ruleset_mods = self._infer_ruleset_modifiers(ruleset, enemies, primary)
            modifiers.extend(ruleset_mods)

        return modifiers

    def _find_primary(
        self,
        enemies: list[EnemyRuntime],
        ruleset: EncounterRuleset,
    ) -> EnemyRuntime | None:
        encounter_id = ruleset.encounter_id.lower()
        for e in enemies:
            if e.enemy_id == encounter_id:
                return e
        if enemies:
            return max(enemies, key=lambda e: e.max_hp)
        return None

    def _infer_ruleset_modifiers(
        self,
        ruleset: EncounterRuleset,
        enemies: list[EnemyRuntime],
        primary: EnemyRuntime | None,
    ) -> list[ModifierPrimitive]:
        """按 primitive.owner_id 找对应 enemy，不再所有 ruleset modifier 都挂 primary。"""
        results: list[ModifierPrimitive] = []

        for cap in ruleset.damage_caps:
            enemy = _resolve_owner_enemy(cap.owner_id, enemies, primary)
            if enemy is None:
                continue
            active = cap.active_when(enemy)
            results.append(replace(
                cap, active=active, owner_id=enemy.entity_id,
                current_value=float(cap.cap_value) if active else 0.0,
            ))

        for tr in ruleset.target_restrictions:
            enemy = _resolve_owner_enemy(tr.owner_id, enemies, primary)
            if enemy is None:
                continue
            active = tr.detect_restricted(enemy)
            results.append(replace(
                tr, active=active, owner_id=enemy.entity_id,
            ))

        # 其他 ruleset modifier 类型：per-primitive owner_id 查 enemy
        for mod in ruleset.effect_scalings:
            enemy = _resolve_owner_enemy(mod.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in ruleset.on_play_triggers:
            enemy = _resolve_owner_enemy(mod.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in ruleset.on_hit_triggers:
            enemy = _resolve_owner_enemy(mod.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in ruleset.draw_modifiers:
            enemy = _resolve_owner_enemy(mod.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in ruleset.exhaust_modifiers:
            enemy = _resolve_owner_enemy(mod.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in ruleset.phase_transition_effects:
            enemy = _resolve_owner_enemy(mod.owner_id, enemies, primary)
            if enemy is None:
                continue
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))

        return results
