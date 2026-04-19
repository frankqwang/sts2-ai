# 2026-0419 networkV2 分层重命名

## 目的

本次重构只做一件事：把 `networkV2` 里误导性的层名改成职责名，不保留兼容别名。

重点是 3 层：

- `s2_config` 实际不是“配置”，而是 encounter 规则与 power 规则来源
- `s3_state_tracker` 实际不是泛泛“tracker 包”，而是时序状态累计层
- `s4_compiler` 实际不是普通“compiler 包”，而是决策特征化流水线

## 新目录

- `networkV2/s2_rules`
- `networkV2/s3_temporal_state`
- `networkV2/s4_featurization`

## 关键命名映射

- `EncounterMechanismConfig` -> `EncounterRuleset`
- `MechanismRegistry` -> `EncounterRuleRegistry`
- `get_registry()` -> `get_encounter_registry()`
- `AUTO_MODIFIER_RULES` -> `POWER_MODIFIER_RULES`
- `compile_auto_modifiers()` -> `build_power_modifiers()`
- `combat_env_wrapper.py` -> `combat_state_tracker.py`
- `CombatFeatureCompiler` -> `DecisionFeaturizer`
- `RuntimeCompiler` -> `RuntimeExtractor`
- `MechanismCompiler` -> `MechanismInferer`
- `ModifierCompiler` -> `ModifierInferer`
- `MemoryCompiler` -> `MemoryEncoder`
- `ActionCompiler` -> `ActionExtractor`
- `BankAssembler` -> `TokenBankBuilder`

## 非战斗子层

`s4_featurization/noncombat` 统一改成 option builder 语义：

- `CardRewardOptionBuilder`
- `ShopOptionBuilder`
- `RouteOptionBuilder`
- `RestOptionBuilder`
- `EventOptionBuilder`

它们都负责把 domain-specific 选项构造成 `ActionCandidate`，再交给 `DecisionFeaturizer` 汇总。

## 当前边界

- `s2_rules`：静态规则与运行时派生规则源
- `s3_temporal_state`：跨 step / turn / combat 的状态累计
- `s4_featurization`：把 runtime + memory + rules 转成 `UnifiedTokenBanks`

## 说明

- 本次不保留旧 import 路径
- 已删除 `s2_config/act1` 这类废弃兼容残留
- README 与主要训练/观战入口已同步到新命名
