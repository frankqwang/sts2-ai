下面这份可以直接交给 Claude 施工。
它不是“网络实现说明”，而是**STS2 特征工程 vNext 设计方案**：先把世界表示、schema、编译流程、模块边界定清楚，再去接网络和训练。之所以要这么做，是因为 STS2 官方明确还会在 Early Access 期间持续加入并调整 cards、relics、potions、enemies、events，以及 enchantments、afflictions、Ancients、alternate acts 等新内容和新机制，所以方案必须优先服务“持续扩展”，不能围绕当前 boss 名单做临时补丁。([Mega Crit][1])

---

# STS2 特征工程 vNext 设计方案（Claude 施工指引）

## 1. 设计目标

新特征工程的目标不是“把更多东西塞进 state vector”，而是构建一套**统一、分层、可扩展的战斗世界表示系统**。它必须满足四件事：

第一，能清楚表达：

* build 差异
* 当前战场
* boss/怪物复杂机制
* 当前规则改写
* 回合内牌序
* 战斗内长程态势
* 动作在当前规则世界下的真实含义

第二，新机制接入时，尽量通过：

* 新增 primitive
* 新增 schema
* 新增编译规则
  而不是大改主干网络。

第三，特征工程必须同时服务：

* 直推 policy
* value 估值
* search prior
* leaf evaluator

第四，它要能吸收你们当前系统里已经成熟的输入表面和语义 scaffold，而不是全部推倒。当前系统已经有 `world/query/local` 三路输入、`semantic_action`、`run_memory`、`build_profile`、`enemy_phase_rule`、`target_reaction`、pile binding、cycle plan、banked world cross-attention 这些成熟资产，这些都应该被复用，但要**重新归入更清晰的上层抽象**。

---

## 2. 不可妥协的设计原则

### 2.1 schema-first，不是 feature-first

先定义世界里有哪些对象、状态、规则，再决定提什么特征。以后任何新特征都必须先回答：

* 它属于哪一层
* 时间尺度是什么
* 作用对象是谁
* 它想帮助哪个决策问题

### 2.2 慢变量、快变量、局部历史分开

至少分三层时间尺度：

* `Run/BuildMemory`：整局慢变量
* `CombatMemory`：本战斗长程摘要
* `TurnPrefixMemory`：本回合高分辨率历史

不能把这三者混成一个 history。

### 2.3 机制状态和规则改写分开

* `MechanismState`：当前战斗进程走到哪里
* `RuleModifier`：当前规则说明书被怎么改了

它们在游戏实现里可能都表现为 buff，但在 AI 表达里绝不能混为一谈。

### 2.4 动作先被上下文化，再参与比较

模型最终比较的不是 raw action，而是：

* 这个动作在当前 board 下意味着什么
* 在当前 mechanism 下意味着什么
* 在当前 modifier 下怎么算
* 在当前 prefix/combat/run context 下值不值得做

也就是比较 `ActionHypothesis`，不是比较“动作名字”。

### 2.5 模拟器负责规则真值，网络负责策略泛化

* **模拟器/规则系统**负责：合法动作、preview、状态更新、primitive 编译、运行时求值
* **网络**负责：泛化、组合、排序、价值判断、搜索接口

---

## 3. 统一 canonical schema

Claude 先实现 schema，不要先实现网络。

## 3.1 EntitySemantics

表示实体“本体是什么”。

对象：

* card
* relic
* potion
* enemy
* summon / special entity

字段建议：

* `entity_type`
* `entity_id`
* `static_attrs`
* `rule_template_tags`
* `symbol/tags`
* `text_semantics`（辅助）

来源：

* 源码 / 数据配置
* 文本描述只做辅助

不包含：

* 当前 cost
* 当前 hp
* 当前 damage preview

---

## 3.2 RuntimeInstances

表示实体“当前实例状态是什么”。

对象：

* hand card instance
* enemy runtime instance
* player runtime
* relic/potion runtime
* pile summary

字段建议：

* card：`current_cost`, `current_damage_est`, `current_block_est`, `retain`, `ethereal`, `discount_state`, `buffed_state`
* enemy：`hp`, `max_hp`, `block`, `intent`, `stacks`, `charges`
* player：`hp`, `block`, `energy`, `buffs`, `debuffs`
* pile：数量、组成比例、关键牌剩余比、reshuffle 接近度

来源：

* 运行时状态
* action preview
* 规则求值器

---

## 3.3 MechanismStates

表示“当前处于哪个机制节点 / 状态机阶段”。

第一批 primitive：

* `phase_transition`
* `window`
* `summon_cycle`
* `threshold_gate`
* `shield_progress`

字段建议：

* `mechanism_type`
* `owner`
* `phase_id`
* `window_state`
* `window_turns_left`
* `threshold_pending`
* `threshold_remaining`
* `next_phase_type`
* `summon_cycle_id`
* `summon_countdown`
* `shield_layers`
* `break_imminent`

来源：

* 源码/配置
* 运行时状态
* 小量 tracked state

定义标准：
它回答的是：**战斗进程走到哪了**。

---

## 3.4 RuleModifiers

表示“当前哪些规则正在改写基础逻辑”。

第一批 primitive：

* `damage_cap`
* `target_restriction`
* `effect_scaling`
* `on_play_trigger`
* `on_hit_trigger`
* `draw_modifier`
* `exhaust_modifier`
* `phase_transition_effect`

字段建议：

* `modifier_type`
* `scope`
* `activation`
* `termination`
* `param_1`
* `param_2`
* `source_entity`
* `current_runtime_value`
* `confidence/source_kind`

例子：

* 每次只受 1 点伤害 → `damage_cap(per_hit, 1)`
* 本体不可选中直到 adds 死 → `target_restriction(must_clear_adds_first)`
* 按当前格挡值造成伤害 → `effect_scaling(by_block)`

定义标准：
它回答的是：**当前规则怎么变了**。

---

## 3.5 TurnPrefixMemory

表示“本回合高分辨率历史”。

字段建议：

* 已执行动作序列
* source / target
* 每步资源变化
* 每步 pile 变化
* 当前已建立的 buff chain / draw chain / exhaust chain / combo chain

用途：

* 牌序学习
* end turn 判断
* 本回合内的连段与前后手依赖

---

## 3.6 CombatMemory

表示“本战斗长程摘要”。

字段建议：

* `turn_index`
* `combat_elapsed`
* `cumulative_hp_loss`
* `recent_hp_loss_window`
* `potions_used`
* `burst_resources_spent`
* `phase_history`
* `transition_count`
* `thresholds_triggered`
* `summon_cycle_round`
* `window_open_count`
* `reshuffle_count`
* `exhaust_total`
* `thinning_progress`
* `current_fight_mode`（race / stabilize / attrition / burst_prep）

用途：

* 跨回合策略
* 阶段切换后续判断
* leaf evaluator 的长程战斗上下文

---

## 3.7 RunBuildMemory

表示“整局慢变量背景”。

字段建议：

* `build_identity`
* `deck_style`
* `frontload / block / draw / scaling / aoe / heal`
* `curse_density`
* `high_cost_density / zero_cost_density / x_cost_density`
* `consistency`
* `objective_context`
* `survival_priority`
* `resource_priority`
* `preserve_hp_bias`
* `boss_pressure / elite_pressure`

这部分可以大量复用现有 `run_memory / build_profile / objective_context` 的提炼结果，但它们现在应被视为**正式 slow memory 输入**，不只是 patch 特征。

---

## 3.8 RawActionCandidates

表示当前合法动作集合。

字段建议：

* `action_type`
* `source`
* `target`
* `legality`
* `preview_values`
* `semantic_action_signature`
* `target_scope`
* `action_roles`

这部分可以直接复用现有 `semantic_action` 输出作为原始动作语义基础。

---

## 3.9 ActionHypotheses

这是最关键的新对象。

它不是原始动作，而是：

> 这个动作在当前 runtime + mechanism + modifier + prefix + combat + build context 下，实际意味着什么。

字段建议：

* `immediate_damage_est`
* `immediate_block_est`
* `draw_delta_est`
* `energy_delta_est`
* `pile_delta_est`
* `target_validity_under_modifiers`
* `modifier_inefficiency`
* `phase_trigger_risk`
* `window_utilization_score`
* `followup_action_space_delta`
* `combo_role`（starter / extender / finisher / reset / waste）
* `resource_retention_impact`

---

## 4. Primitive 设计方案

Claude 不要为每个 boss 写独立逻辑。
先实现一个**primitive registry**。

每个 primitive 必须统一包含：

* `type`
* `scope`
* `activation`
* `termination`
* `params`
* `runtime_outputs`
* `source_kind`
* `confidence`

### 4.1 Mechanism primitive registry

第一版就做这 5 个：

* `phase_transition`
* `window`
* `summon_cycle`
* `threshold_gate`
* `shield_progress`

### 4.2 Modifier primitive registry

第一版就做这 8 个：

* `damage_cap`
* `target_restriction`
* `effect_scaling`
* `on_play_trigger`
* `on_hit_trigger`
* `draw_modifier`
* `exhaust_modifier`
* `phase_transition_effect`

### 4.3 原则

* 先做**类型少、参数多**
* 先粗分，再细化
* 每个 boss 是多个 primitive 的组合，不是一个 boss 类

---

## 5. 数据来源与编译策略

这部分是 Claude 施工重点。

## 5.1 来源优先级

### 一级：源码 / 配置 /数据表

最好来源。
如果规则是数据驱动的，直接编译成 primitive。

### 二级：运行时状态 / preview / legal actions

适合生成：

* RuntimeInstances
* MechanismStates 的当前参数
* RuleModifiers 的当前激活态
* ActionHypotheses 的即时估计

### 三级：文本 / wiki / buff 描述

只做：

* 语义辅助
* 冷启动分类
* primitive 猜测
* 近义机制对齐

**不能做唯一真值来源。**

---

## 5.2 编译流程

Claude 先实现一个 `CombatFeatureCompiler`，按顺序做：

1. 读取 bridge/raw obs + legal_actions + preview + planner_context
2. 编译 `EntitySemantics`
3. 编译 `RuntimeInstances`
4. 编译 `MechanismStates`
5. 编译 `RuleModifiers`
6. 编译 `TurnPrefixMemory`
7. 编译 `CombatMemory`
8. 编译 `RunBuildMemory`
9. 编译 `RawActionCandidates`
10. 基于上面对象，构建 `ActionHypotheses` 初级字段

---

## 5.3 自动与手工的边界

### 手工做一次的

* primitive 类型系统
* schema
* 编译规则框架
* 自动映射规则

### 自动做大部分的

* 从源码/运行时提字段
* 把机制映射到 primitive
* 填参数
* 生成 runtime outputs

### 少量人工补的

* 隐式 phase
* 特别绕的多条件状态机
* 无法直接从当前 bridge 恢复的隐藏逻辑

---

## 6. token bank 输出规范

为了兼容后续网络，特征工程最后不要只输出一个拼接向量。
应输出 bank 化对象。

第一版建议输出 7 组：

1. `build_bank`
2. `board_bank`
3. `mechanism_bank`
4. `modifier_bank`
5. `turn_prefix_bank`
6. `combat_memory_bank`
7. `action_bank`

现有的 `world/query/local` 可以先作为过渡 transport surface，但语义上要重新归位：

* `world` 主要承载 build/board/mechanism/modifier/combat
* `query` 承载 raw action
* `local` 改成 action-specific compiled local context，也就是 ActionHypothesis 的局部支持表面

你们当前的 bank-aware world memory 和 query-local bridge 思想值得保留，但应以上述 canonical schema 为上层语义，而不是继续无限堆 surface。

---

## 7. 网络消费接口要求

虽然这份方案重点是特征工程，但 Claude 在落地时必须留好网络消费接口。

### 7.1 必须支持的读取关系

* action 直接读取 `turn_prefix_bank`
* action 直接读取 `mechanism_bank`
* action 直接读取 `modifier_bank`
* action 直接读取 `build_bank`
* leaf evaluator 直接读取 `combat_memory_bank`

### 7.2 明确区分

* `action token`：动作描述
* `action hypothesis`：当前世界里的动作意义

### 7.3 禁止回退

* 不允许把 prefix 又塞回 state summary
* 不允许把 modifier 又混回 enemy scalar
* 不允许只靠 raw buff text 或 enemy id 让网络自己猜

---

## 8. 搜索与 leaf evaluator 的特征要求

这部分必须在特征工程阶段就预留，不要事后补。

### 8.1 Leaf evaluator 需要的输入

至少包括：

* `hp_utility_features`
* `survival_margin_features`
* `resource_retention_features`
* `transition_risk_features`
* `modifier_inefficiency_features`
* `window_opportunity_features`
* `fight_mode_features`

### 8.2 非线性 HP 特征

特征工程要直接提供一些非线性生存表面，例如：

* 当前 hp ratio
* near-lethal margin
* next_turn_death_risk summary
* recent_hp_loss trend

不能只给原始 hp。

### 8.3 资源保留特征

至少提供：

* 药水是否已用
* 关键一次性资源是否已交
* 当前 burst 资源剩余
* 当前 setup 是否已完成

---

## 9. 训练接口需要的标签与分层字段

Claude 在 feature compiler 里就把这些标签打出来，后面训练直接用。

### 9.1 分层标签

* `room_type`: normal / elite / boss
* `mechanism_family`
* `risk_level`
* `build_family`
* `transition_pending`
* `window_state`
* `resource_pressure_level`

### 9.2 作用

这些标签用于：

* PPO 分层采样
* boss/high-risk 状态加权
* imitation / ranking 数据筛选
* search distillation 的关键状态采集

---

## 10. 对当前系统的迁移建议

Claude 不要推翻现有特征工程资产，而是做**重新编排**。

### 可直接复用的

* `semantic_action`
* `run_memory`
* `build_profile`
* `objective_context`
* `enemy_phase_rule`
* `target_reaction`
* pile context / pile binding / cycle plan
* world/query/local 运输层
* banked world cross-attention 的输入组织经验

### 需要“升维重定义”的

* `enemy_phase_rule` → 拆进 `MechanismStates`
* `target_reaction` → 拆进 `RuleModifiers + ActionHypotheses`
* `run_memory/build_profile/objective_context` → 正式归到 `RunBuildMemory`
* pile/cycle/binding → 一部分归 `RuntimeInstances`，一部分归 `ActionHypotheses`
* 各种 preview / heuristic-like numeric → 归到 `RuntimeInstances` 或 `ActionHypotheses`

---

## 11. Claude 施工顺序

### Phase 1：先交 schema 文档

必须先输出：

* canonical schema 定义
* primitive registry 定义
* 数据来源表
* feature 分类表

### Phase 2：实现 compiler 骨架

实现 `CombatFeatureCompiler`，能输出：

* 9 类 canonical objects
* 7 组 token banks

### Phase 3：实现 primitive 自动映射

从源码/运行时自动编译：

* MechanismStates
* RuleModifiers

### Phase 4：实现 ActionHypothesis 编译

至少先做：

* immediate effect
* legality under modifiers
* phase/window/targetability related fields
* followup action space delta 的摘要版

### Phase 5：接网络和训练

再把这些 bank 接到：

* policy
* value
* leaf evaluator
* search

---

## 12. Claude 的交付标准

这部分你可以直接拿来验收。

### 必须达到

1. 新机制接入时，优先新增 primitive/schema，而不是加 boss 专属分支
2. 每个 feature 都能回答：属于哪层、时间尺度是什么、服务什么决策问题
3. `TurnPrefixMemory` 与 `CombatMemory` 明确分开
4. `MechanismStates` 与 `RuleModifiers` 明确分开
5. 每个合法动作都能生成 `ActionHypothesis`
6. Leaf evaluator 所需特征在 feature compiler 阶段就能拿到
7. 当前系统已有的 world/query/local 与 semantic scaffold 被复用，而不是浪费掉 

### 明确不接受

* 再搞一个更大的 state vector
* 把复杂机制继续塞进 raw buff list
* 只靠文本描述或 enemy id 表示机制
* 只靠 value head 做叶子评估
* 继续 patch / gate / delta 累加式扩展

---

## 最后一句定性

这套方案的本质不是“换一版特征”，而是：

> **把 STS2 战斗从“状态拼接问题”提升成“规则世界编译问题”。**

一旦 Claude 先把这层编译清楚，后面的网络、PPO、搜索、leaf evaluator 才会真正有共同语言。

如果你要，我下一条可以直接把这份方案再压缩成一版 **Claude 执行 prompt**，就是那种“请按以下阶段完成并产出哪些文件”的交付格式。

[1]: https://www.megacrit.com/faq/?utm_source=chatgpt.com "Faq"
