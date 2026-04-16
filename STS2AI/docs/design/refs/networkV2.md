对，你说得对。
前面更多是在定 **世界表示和 schema**，还没有把 **“网络到底长什么样”** 讲到位。

下面我直接给你一版 **新的完整网络架构**。
这版是可以交给 Claude 直接按模块开工的。

---

# 新网络架构：CombatNet vNext

## 0. 一句话定义

这不是：

> `state_repr + action_repr -> scorer`

而是：

> **多记忆体分层世界模型 + 动作上下文化 + 动作比较器 + 独立叶子评估器**

也就是：

**FeatureCompiler**
→ **多 bank 表示层**
→ **Action Contextualizer**
→ **Decision Core**
→ **Policy / Value / LeafEvaluator**

---

# 1. 总体结构图

## 整体流水线

```text
Raw game state / simulator / legal actions
    ↓
FeatureCompiler
    ↓
[build_bank, board_bank, mechanism_bank, modifier_bank,
 turn_prefix_bank, combat_memory_bank, action_bank]
    ↓
Action Contextualizer
    ↓
Action Hypotheses
    ↓
Decision Core
    ↓
PolicyHead + ValueHeads + LeafEvaluator
```

---

# 2. 网络分 5 大块

---

## A. Tokenization & Bank Projection

这是神经网络的第一层，不是特征工程本身。

输入已经是你们编译好的 canonical schema：

* `EntitySemantics`
* `RuntimeInstances`
* `MechanismStates`
* `RuleModifiers`
* `TurnPrefixMemory`
* `CombatMemory`
* `RunBuildMemory`
* `RawActionCandidates`

每一类先变成 token，并加上：

* type embedding
* role embedding
* owner embedding
* scope embedding
* zone embedding
* time-scale embedding

### 输出

7 组 token bank：

* `build_bank`
* `board_bank`
* `mechanism_bank`
* `modifier_bank`
* `turn_prefix_bank`
* `combat_memory_bank`
* `action_bank`

### 作用

把不同类型对象放进同一维度空间，但保留类别边界。

---

## B. Memory Encoders

这是第二层，负责把每个 bank 先内部整理好。
不是所有 token 一上来混一起。

---

### B1. Build Memory Encoder

**职责**：表示“这套牌怎么赢”。

### 输入

* deck card tokens
* relic tokens
* potion tokens
* build summary tokens

### 结构

* `N_build_slots = 8~12` 个 learnable latent slots
* 2~3 层 slot-to-build cross-attention block

### 输出

* `build_memory_slots`

### 为什么这样设计

build 是慢变量，不该每次和 board 一起搅成一锅。
它应该像“长期策略内存”。

---

### B2. Board Encoder

**职责**：表示“当前战场是什么”。

### 输入

* player runtime token
* hand card runtime tokens
* enemy runtime tokens
* pile summary tokens
* global combat scalar token

### 结构

* 3~4 层 self-attention blocks
* relation bias：

  * card↔player
  * card↔enemy
  * enemy↔enemy
  * pile↔card

### 输出

* `board_tokens`

### 为什么这样设计

board 是快变量，要表达当前局面，但不应该承载机制、长期 build 和记忆职责。

---

### B3. Mechanism Encoder

**职责**：表示“机制状态机当前走到哪”。

### 输入

* mechanism tokens（phase/window/threshold/summon/shield）

### 结构

* 2 层 self-attention
* ownership bias：机制 token 强绑定所属 enemy / owner

### 输出

* `mechanism_tokens_refined`

### 为什么这样设计

mechanism 不是普通 buff，它是“战斗进程节点”。

---

### B4. Modifier Encoder

**职责**：表示“当前规则被怎么改了”。

### 输入

* modifier tokens（damage cap / target restriction / scaling / triggers）

### 结构

* 2 层 self-attention
* scope bias：modifier 强绑定作用对象

### 输出

* `modifier_tokens_refined`

### 为什么这样设计

modifier 改的是“动作怎么算”，必须独立存在。

---

### B5. Turn Prefix Encoder

**职责**：表示“本回合已经怎么打了”。

### 输入

* action history tokens
* source/target tokens
* step resource delta tokens

### 结构

* 2~3 层 causal/ordered transformer

### 输出

* `turn_prefix_memory`

### 为什么这样设计

牌序是强顺序问题，不能只是一个 prefix summary。

---

### B6. Combat Memory Encoder

**职责**：表示“这场战斗到现在的长程态势”。

### 输入

* combat summary tokens
* phase history
* potion usage
* cumulative hp loss
* reshuffle/exhaust progress
* fight mode summary

### 结构

* 1~2 层小 transformer 或 gated summary block

### 输出

* `combat_memory_tokens`

### 为什么这样设计

CombatMemory 不是 prefix；它是跨回合长程摘要，主要给动作与 leaf evaluator 用。

---

# 3. Action Contextualizer

这层是整个架构的核心。

## 输入

* `action_bank`
* `board_tokens`
* `modifier_tokens_refined`
* `mechanism_tokens_refined`
* `turn_prefix_memory`
* `combat_memory_tokens`
* `build_memory_slots`

## 输出

* `action_hypothesis_tokens`

---

## 这层到底做什么

它不再把动作当成“一个 ID + target”。
而是把每个动作变成：

> **这个动作在当前规则世界里，实际意味着什么**

例如：

* 当前真实伤害/格挡/抽牌收益
* 是否被 damage cap 压制
* 是否会提前触发 phase
* 是否浪费窗口
* 是否违反当前 target restriction
* 是否是 combo extender / finisher / waste
* 是否会破坏资源保留
* 是否符合这套 build 的长期赢法

---

## 具体结构

我建议用 **6 段交叉读取块**：

### Block 1: action → board

让动作先知道当前局面和对象状态。

### Block 2: action → modifier

让动作知道当前规则怎么改了。

### Block 3: action → mechanism

让动作知道当前阶段、窗口、threshold、召唤循环。

### Block 4: action → turn_prefix

让动作知道当前回合前面已经怎么打了。

### Block 5: action → combat_memory

让动作知道这场战斗的长期态势。

### Block 6: action → build

让动作受到 build 慢变量的长期调制。

---

## 为什么要这个顺序

因为这最符合决策逻辑：

1. 先看当前局面
2. 再看当前规则
3. 再看当前机制节奏
4. 再看本回合牌序
5. 再看本战斗长程态势
6. 最后用 build 决定长期价值权重

---

# 4. Decision Core

Action Contextualizer 之后，才开始真正“比较动作”。

## 输入

* `action_hypothesis_tokens`
* 一个 learnable `decision_token`

## 结构

* 把 `decision_token` prepend 到所有 action hypotheses 前面
* 做 2~4 层 self-attention reasoning blocks

## 输出

* `decision_token_refined`
* `action_hypotheses_refined`

---

## 作用

### `decision_token`

表示当前全局决策摘要，供：

* ValueHeads
* LeafEvaluator
* 全局策略判断

### `action_hypotheses_refined`

表示每个动作在全局比较后的最终语义，供：

* PolicyHead
* 搜索 prior

---

## 为什么不能省

因为 action contextualizer 是“理解动作”，
而 decision core 是“把所有动作放在一起比”。

这两个不是一回事。

---

# 5. Heads 层

---

## 5.1 Policy Head

### 输入

* `decision_token_refined`
* `action_hypotheses_refined`

### 输出

* 每个动作的 logit
* top-k prior

### 设计

不是简单线性层，建议用：

* action-specific scorer
* decision-conditioned scorer

也就是动作分数要同时看：

* 自己是什么动作
* 当前全局决策上下文是什么

---

## 5.2 Value Heads

至少 3 个头：

### 1. `fight_win_value`

这局最终有多大赢面

### 2. `expected_hp_loss`

这场战斗期望掉多少血

### 3. `survival_2turn`

未来 1~2 回合的生存概率

建议再加 1 个：

### 4. `tempo_or_phase_value`

当前节奏/阶段位置是否占优

---

## 5.3 Leaf Evaluator

这是**独立模块**，不要混进普通 value head。

### 输入

* `decision_token_refined`
* `combat_memory_tokens`
* `mechanism summary`
* `modifier summary`
* `resource summary`
* `survival summary`

### 输出

* `leaf_score`
* `transition_risk`
* `mechanism_penalty`
* `resource_retention_value`
* `survival_margin`

---

## 为什么必须独立

因为搜索叶子停在中间局面时，普通 value 不够。

它需要特别评估：

* 低血是否危险到非线性爆炸
* 当前是否快转阶段
* 当前资源是否该留
* 当前 modifier 下动作效率是否严重失真
* 当前窗口是否即将开/关

---

# 6. 这套网络跟传统结构的根本区别

---

## 传统结构

```text
state_repr + action_repr -> score
```

问题：

* build 不是真 slow memory
* prefix 不是独立 memory
* mechanism/modifier 混在 state
* 动作没被上下文化
* leaf evaluation 依赖裸 value

---

## 新结构

```text
world representation
-> contextualized actions
-> action comparison
-> policy/value/leaf evaluation
```

区别在于：

### 1. 先建“规则世界”

不是先建 state vector

### 2. 决策对象变成了 `ActionHypothesis`

不是 raw action

### 3. 有多种 memory

* build
* turn prefix
* combat memory

### 4. 有 mechanism / modifier 双层

不是全叫 buff

### 5. leaf evaluator 独立存在

不是 value 顺带做

---

# 7. 推荐参数规模

---

## 主力 dense 版（推荐）

* `d_model = 512`
* `n_heads = 8~12`
* total blocks ≈ `10~14`

  * build 2
  * board 3~4
  * mechanism 2
  * modifier 2
  * prefix 2~3
  * combat memory 1~2
  * action contextualizer 6 读块
  * decision core 2~4
* FFN `2048~3072`

大概：

* **60M ~ 90M**

---

## 轻量验证版

* `d_model = 384`
* blocks 略缩
* 参数量：
* **35M ~ 55M**

适合先验证抽象和信息流。

---

## 暂时不建议

一开始就上 MoE。
先把 canonical schema 和 memory 分层做对，再谈专家模块。

---

# 8. Claude 的施工顺序

---

## Phase 1

先把 **canonical schema** 和 **primitive registry** 做完

## Phase 2

实现 FeatureCompiler，产出：

* build_bank
* board_bank
* mechanism_bank
* modifier_bank
* turn_prefix_bank
* combat_memory_bank
* action_bank

## Phase 3

实现各个 Memory Encoders

## Phase 4

实现 `ActionContextualizer`

## Phase 5

实现 `DecisionCore + PolicyHead`

## Phase 6

实现 `ValueHeads + LeafEvaluator`

## Phase 7

接搜索和训练

---

# 9. 给 Claude 的最短版架构要求

你可以直接发这段：

```text
请实现一个全新的 STS2 CombatNet，不兼容旧状态向量架构。

目标结构：
FeatureCompiler
-> build_bank / board_bank / mechanism_bank / modifier_bank / turn_prefix_bank / combat_memory_bank / action_bank
-> BuildMemoryEncoder / BoardEncoder / MechanismEncoder / ModifierEncoder / TurnPrefixEncoder / CombatMemoryEncoder
-> ActionContextualizer（action 依次读取 board, modifier, mechanism, turn_prefix, combat_memory, build）
-> DecisionCore（decision token + action hypotheses 比较）
-> PolicyHead + ValueHeads + LeafEvaluator

关键要求：
1. 动作比较对象必须是 ActionHypothesis，不是 raw action
2. Build 是 slow memory
3. TurnPrefixMemory 与 CombatMemory 必须独立
4. MechanismStates 与 RuleModifiers 必须独立
5. LeafEvaluator 必须独立存在
6. 搜索使用 prior + ActionHypothesis + LeafEvaluator
```

---

如果你愿意，我下一条可以继续把这套网络结构再压成一个 **“架构图 + 模块职责表”**，方便直接发给团队。
