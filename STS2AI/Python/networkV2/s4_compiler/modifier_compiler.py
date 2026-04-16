"""RuleModifiers 编译器：Level 2 auto + Level 3 config 合并。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from networkV2.s1_schema.primitives import (
    ModifierPrimitive,
    DamageCap,
    TargetRestriction,
)
from networkV2.s1_schema.entities import EnemyRuntime
from networkV2.s2_config.mechanism_registry import EncounterMechanismConfig
from networkV2.s2_config.auto_modifier_rules import compile_auto_modifiers


class ModifierCompiler:
    """编译当前活跃的 RuleModifiers。

    合并两个来源：
      Level 2: 从 enemy.powers 自动映射 (auto_modifier_rules)
      Level 3: 从 mechanism_config 手工配置
    """

    def compile(
        self,
        enemies: list[EnemyRuntime],
        config: EncounterMechanismConfig | None,
    ) -> list[ModifierPrimitive]:
        modifiers: list[ModifierPrimitive] = []

        # Level 2: 对每个敌人，自动映射 power → modifier
        for enemy in enemies:
            auto_mods = compile_auto_modifiers(enemy)
            modifiers.extend(auto_mods)

        # Level 3: 从 config 获取手工配置的 modifier，检测激活状态
        if config is not None:
            primary = self._find_primary(enemies, config)
            if primary is not None:
                config_mods = self._compile_config_modifiers(config, primary)
                modifiers.extend(config_mods)

        return modifiers

    def _find_primary(
        self,
        enemies: list[EnemyRuntime],
        config: EncounterMechanismConfig,
    ) -> EnemyRuntime | None:
        encounter_id = config.encounter_id.lower()
        for e in enemies:
            if e.enemy_id == encounter_id:
                return e
        if enemies:
            return max(enemies, key=lambda e: e.max_hp)
        return None

    def _compile_config_modifiers(
        self,
        config: EncounterMechanismConfig,
        enemy: EnemyRuntime,
    ) -> list[ModifierPrimitive]:
        results: list[ModifierPrimitive] = []

        for cap in config.damage_caps:
            active = cap.active_when(enemy)
            results.append(replace(
                cap, active=active, owner_id=enemy.entity_id,
                current_value=float(cap.cap_value) if active else 0.0,
            ))

        for tr in config.target_restrictions:
            active = tr.detect_restricted(enemy)
            results.append(replace(
                tr, active=active, owner_id=enemy.entity_id,
            ))

        # 其他 config modifier 类型
        for mod in config.effect_scalings:
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in config.on_play_triggers:
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in config.on_hit_triggers:
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in config.draw_modifiers:
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in config.exhaust_modifiers:
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))
        for mod in config.phase_transition_effects:
            results.append(replace(mod, active=True, owner_id=enemy.entity_id))

        return results
