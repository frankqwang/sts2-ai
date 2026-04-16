对，我前面那版**主要是在补战斗主干**，如果只停在那一步，整体 agent 还是不完整。
STS2 的非战斗部分不能当附属，因为官方已经明确这代会持续扩充 cards、relics、potions、enemies、events，以及 enchantments、afflictions、Ancients、alternate acts 等新内容；而且 Early Access 期间还会继续新增和重平衡。这意味着“选牌、商店、路线、休息、事件”不是静态菜单问题，而是长期、分布会持续变化的策略层。([Mega Crit][1])

## 先给结论

完整的 STS2 agent 不该是一套统一 head 硬吃所有决策，也不该是“战斗一个模型，非战斗随便糊个 MLP”。
更合理的是：

> **共享一个 run/world 编译层 + 共享 slow memory + 按决策家族拆分的 option contextualizer / reasoner / evaluator。**

也就是：

* 战斗决策：`ActionHypothesis`
* 非战斗决策：`OptionHypothesis`

两边共享世界观，但**不共享最终决策语义**。

---

# 一、为什么非战斗更吃设计

战斗动作的好坏，多数还能在短期里看出来。
非战斗动作的问题是：**收益滞后、影响间接、需要跨多个未来战斗才兑现**。

比如：

* 选一张牌，可能 8 场战斗后才知道是不是好 pick
* 商店里买 relic、remove、买 potion，机会成本很高
* 路线不是“下一格好不好”，而是整段路径的风险-收益结构
* 休息是 heal、upgrade、remove、特殊行动之间的长期 tradeoff
* 事件选项经常带有强波动、强条件、强 build 依赖

所以非战斗决策最怕两件事：

1. 只看即时收益
2. 没有“这个选项会把 run 变成什么样”的表示

这就是为什么我说，非战斗部分需要从“动作评分”升级成“**选项假设比较**”。

---

# 二、整体应该改成“双层架构”

## 第 1 层：共享 world/run compiler

先把全局 run 状态统一编译出来。

这层输出的不是单个 state vector，而是几组长期 bank：

* `build_bank`
* `economy_bank`
* `route_bank`
* `run_objective_bank`
* `combat_forecast_bank`
* `inventory_bank`

这层负责回答：

* 当前这套牌是什么风格
* 当前金币/药水/遗物/删牌机会的经济位置
* 当前 act/路线结构与风险是什么
* 当前 run 的目标偏向是什么（保命、冲精英、贪成长、保药等）
* 未来几场典型战斗/精英/boss 对当前 build 的压力是什么

你们现有系统其实已经有一些雏形，比如 `run_memory`、`build_profile`、`objective_context`、`route summary`、`selection surface`、`build candidate`、`route risk/value local`。这些说明你们已经知道非战斗不能只看当前一帧，只是现在还没升成正式架构。

## 第 2 层：按决策家族拆分的 option modules

非战斗不应该共用战斗的 `Action Contextualizer`。
它应该有一层更泛化的：

> **Option Contextualizer**

然后再按家族拆成：

* `CardChoice / Reward Contextualizer`
* `Shop Contextualizer`
* `Route Contextualizer`
* `Rest Contextualizer`
* `Event Contextualizer`

共享上层 world/run memory，但每类选项有自己的 option schema 和 reasoning head。

---

# 三、非战斗的核心对象不是 action，而是 OptionHypothesis

这和战斗很像，只是时间尺度不同。

## Raw Option

只是一个菜单项：

* 拿这张牌
* 买这个 relic
* 去这条路
* 在火堆 upgrade
* 事件里选 A

## OptionHypothesis

是“这个选项会把 run 变成什么样”。

它至少要表达三件事：

1. **立即变化**

   * hp / gold / relic / potion / card / curse / remove / upgrade 的变化
2. **结构变化**

   * 牌组曲线、抽牌质量、费用分布、解怪能力、AOE、成长、稳定性怎么变
3. **下游变化**

   * 接下来几场战斗/精英/boss 的可承受性怎么变
   * 路线可达性和资源压力怎么变

也就是说，非战斗决策不该比较“选项名字”，而要比较：

> **这个选项让 run 的未来分布怎样变化**

---

# 四、非战斗的网络应该怎么拆

## A. Shared Run Backbone

这一层全模式共享，负责长期背景。

建议 bank：

* `build_bank`
* `economy_bank`
* `route_bank`
* `inventory_bank`
* `run_objective_bank`
* `combat_forecast_bank`

### 作用

* `build_bank`：现在这套牌靠什么赢
* `economy_bank`：金币、删牌、买 relic、买 potion、商店机会成本
* `route_bank`：未来节点分布、到休息/商店/精英/boss 的距离和组合
* `inventory_bank`：当前 relic/potion/关键一次性资源
* `run_objective_bank`：当前保命还是贪成长
* `combat_forecast_bank`：未来几类典型战斗对当前 build 的压力

这一层建议和战斗共享，因为战斗也需要 build、inventory、run objective。

---

## B. Domain-specific Option Encoders

### 1. CardChoice / Reward Encoder

负责：

* 选牌
* 三选一
* 发现牌
* transform / remove / upgrade / card selection

输入不只是卡本体，而是：

* 候选项本身
* 候选项对 build 的 delta
* 候选项对 curve/consistency/aoe/scaling 的 delta
* 候选项与当前 relic/potion/已拥有牌的联动
* 候选项对未来 boss/elite 的针对性改进

核心思想：

> 不是“这张牌强不强”，而是“这张牌加入这套 build 后，run 结构怎么变”

---

### 2. Shop Encoder

负责：

* 买牌
* 买 relic
* 买 potion
* remove
* skip

输入至少要表达：

* 当前金币与未来金币压力
* 本店与未来商店的机会成本
* 当前 item 的 immediate value
* 当前 item 对 build 的结构增益
* remove 对 deck thinning / consistency 的提升
* potion 的短期战斗价值 vs 长期占槽成本

核心思想：

> 商店不是比较物品，而是比较“当前花这笔钱最值得把 run 改向哪里”。

---

### 3. Route Encoder

负责：

* 地图分支选择
* node risk/value 组合判断

输入至少要表达：

* 未来 2~4 步结构，不只是下一格
* 精英、休息、商店、事件、问号、宝箱的组合关系
* 当前 build 对精英/boss 的 readiness
* 当前 hp / potion / gold / remove pressure
* 走这条路后的“恢复窗口”和“高风险窗口”

核心思想：

> 路线不是局部节点评分，而是路径段落的风险-收益配置。

---

### 4. Rest Encoder

负责：

* heal
* upgrade
* remove
* 特殊火堆行动

输入至少要表达：

* 当前 survival margin
* 下一关键战斗前的 hp buffer
* upgrade 的边际收益
* remove 的长期收益
* 当前 build 是更缺立即生存还是更缺质量跃迁

核心思想：

> 火堆决策本质是“当前 run 更缺恢复，还是更缺结构升级”。

---

### 5. Event Encoder

负责：

* 事件选项比较

输入至少要表达：

* 每个选项的 immediate delta
* 风险/波动大小
* 对 build / economy / hp / relic / potion / curse 的结构影响
* 这个事件选项是否引入新的机制、后续战斗、隐藏成本

核心思想：

> 事件不是“按描述选一个”，而是“在当前 objective 下接受哪种分布性收益/惩罚”。

---

# 五、怎么保证这些非战斗选项能被“有效表示、有效学习”

这是最关键的问题。

## 1. 不要直接把非战斗项混进同一个 action space

如果把：

* 选牌
* 商店
* 路线
* 休息
* 事件
  全都当成“动作列表”统一打分，网络会学得很差，因为它们的语义、时间尺度、目标函数都不同。

所以必须：

* 共享 backbone
* 拆 domain-specific option heads

## 2. 每个选项都要编译成 “delta + downstream forecast”

非战斗最怕只看表面文字。

所以每个 option 至少要有两部分：

* `Immediate Delta`
* `Downstream Forecast`

例如拿一张牌，不只是“得到卡 X”，而是：

* curve 变化
* draw quality 变化
* elite readiness 变化
* boss matchup 改善/恶化
* consistency 变化

## 3. 加一个独立的 Run Evaluator

战斗里我们说要有 Leaf Evaluator。
非战斗里也需要一个更高层的：

> **Run Evaluator / Option Evaluator**

它不判断“这回合能不能活”，而判断：

* 这个 run 的 expected survival / expected boss readiness / expected resource pressure / expected deck quality

否则非战斗 heads 会非常短视。

## 4. 用“局部反事实”构建监督

非战斗最好学的方式，不是光靠 PPO。

要尽量有：

* 同一 seed / 同一状态下不同选项的 counterfactual rollout
* 选项后的短中期模拟
* 对同一 node 的 preference / ranking supervision

例如：

* 同一张地图分叉，走左 vs 走右
* 同一火堆，heal vs smith
* 同一商店，buy relic vs remove vs save gold

这类训练数据对非战斗特别有用。

## 5. 把未来 combat pressure 显式带进来

非战斗之所以难，是因为收益在未来 combat 才兑现。
所以共享 backbone 里必须有 `combat_forecast_bank`，回答：

* 未来典型小怪压力
* 未来 elite 压力
* 未来 boss 压力
* 当前 build 对这些压力的 readiness

这样选牌/路线/商店/火堆才不是“离线决策”，而是“围绕未来战斗做的结构投资”。

---

# 六、推荐的完整全-run 架构

如果把 combat 和 non-combat 一起讲完整，我推荐这样：

## Shared World Compiler

输出：

* `build_bank`
* `board_bank`（仅战斗时有）
* `mechanism_bank`（仅战斗时有）
* `modifier_bank`（仅战斗时有）
* `turn_prefix_bank`（仅战斗时有）
* `combat_memory_bank`
* `economy_bank`
* `route_bank`
* `inventory_bank`
* `run_objective_bank`
* `combat_forecast_bank`

## Domain Routers

先判定当前决策域：

* combat
* reward/card choice
* shop
* route
* rest
* event/selection

## Domain Modules

### Combat

* `Action Contextualizer`
* `Decision Core`
* `Policy / Value / LeafEvaluator`

### Non-combat

统一有一个：

* `Option Contextualizer`
* `Option Reasoner`
* `Option Policy Head`
* `Run Evaluator`

然后按 domain 细分出：

* card/reward
* shop
* route
* rest
* event

---

# 七、和你们当前系统的关系

你们当前系统其实已经有非战斗特征工程的雏形了：

* `BUILD_CANDIDATE`
* `SELECTION_CANDIDATE`
* `ROUTE_RISK_LOCAL`
* `ROUTE_VALUE_LOCAL`
* `BUILD_STATE_LOCAL`
* `DECK_SYNERGY_LOCAL`
* `SHOP_ECON_LOCAL`
* `SELECTION_OPERATOR_LOCAL`
* `SELECTION_SEMANTICS_LOCAL`
  这些都说明方向是对的。

真正的问题不是“有没有做非战斗”，而是：

> 这些东西现在还更像 feature surfaces，
> 还没有提升成一个完整的 **non-combat architecture**。

也就是说，你们已经有了材料，但还没把它们升格成：

* shared run backbone
* option hypotheses
* domain-specific reasoners
* run evaluator

---

# 八、最后给你一个最清晰的定性

是的，我前面的战斗架构不够完整。
真正完整的 STS2 agent 应该是：

> **共享 run/world backbone + combat branch + non-combat branches**

其中：

* 战斗分支比较的是 `ActionHypothesis`
* 非战斗分支比较的是 `OptionHypothesis`

战斗更看短中期动作链；
非战斗更看长程结构变化。
这两者必须共用一套世界观，但不能共用一个简单 head。

如果你愿意，我下一条可以直接把这个整理成一版 **“全 run 网络架构图 + 模块职责表”**，一页就能发给团队。

[1]: https://www.megacrit.com/faq/?utm_source=chatgpt.com "Faq"
