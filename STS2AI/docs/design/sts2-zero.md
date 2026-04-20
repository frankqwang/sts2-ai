下面是一版 **STS2 V1 设计文档**。目标很明确：

**先做出一版能跑、能自动迭代、能验证方向的系统。**
范围先收窄到 **战斗**，不先做全局 build、地图、商店。
搜索器先当 **oracle teacher**，学生网络是最终要用的主体。

---

# 1. 目标与边界

## 1.1 V1 目标

做出一套可闭环系统，能够：

* 用当前学生策略跑对局
* 自动抽取关键战斗状态
* 用同 seed oracle 搜索给这些状态打老师标签
* 用这些样本训练学生网络
* 固定评估集上比较新旧学生
* 自动晋级更好的学生版本
* 迭代多轮后，观察学生是否越来越接近老师

## 1.2 明确不做

V1 不做这些：

* 不做 PPO 主线
* 不做公平 chance MCTS
* 不做大规模游戏源码重构
* 不做 per-buff / per-mechanic 白盒规则重写
* 不做完整世界模型 / MuZero
* 不做全局整局策略统一训练

## 1.3 当前前提

V1 假设你有：

* `step(state, action) -> next_state`
* `clone/save/load`
* 当前合法动作枚举
* 能拿到完整战斗状态快照
* 搜索时可以固定/复用 seed

---

# 2. V1 核心思路

整套系统的主线不是 reward shaping，也不是 policy gradient。

主线是：

**学生采样分布 → 老师在关键状态上给更强标签 → 学生蒸馏老师并学长期价值**

也就是：

* 学生负责“跑世界”
* 老师负责“纠偏”
* 样本系统负责“保新鲜度和覆盖”
* 评估器负责“决定谁晋级”

---

# 3. V1 范围

## 3.1 只做战斗

先只做：

* 战斗状态编码
* 战斗历史编码
* 战斗合法动作打分
* 战斗价值预测
* 战斗关键状态老师标注

## 3.2 角色范围

先只做一个角色，例如 Ironclad。

## 3.3 内容范围

建议先聚焦：

* Act1 normal / elite / boss
* 之后再扩 A2/A3

这样最容易先跑通。

---

# 4. 数据流总览

一轮迭代的完整流程：

1. 用 `student_vK` 跑一批局，记录原始轨迹
2. 从轨迹中筛关键状态，加入 `teacher_queue`
3. 老师对关键状态做同-seed oracle 搜索，生成标签
4. 从原始轨迹构造训练样本
5. 样本入池、分桶、淘汰
6. 从样本池混采训练 `student_v(K+1)`
7. 用固定 cohort 评估
8. 决定是否晋级
9. 用晋级后的学生继续下一轮采样

---

# 5. 样本定义

## 5.1 原始 transition

原始轨迹层保存：

* `run_id`
* `fight_id`
* `step_idx`
* `seed`
* `state_raw = s_t`
* `action_raw = a_t`
* `next_state_raw = s_{t+1}`
* `done`
* 战斗最终结果
* 整局最终结果
* meta 信息

这层永远尽量保真，不做复杂处理。

## 5.2 训练样本

每条训练样本对应一个决策点：

* 当前状态特征 `state_feat_t`
* 最近 K 步历史 `history_feat_t`
* 合法动作集合特征 `legal_action_feat_t`
* 行为动作 `a_t`
* `delta_t = diff(s_t, s_{t+1})`
* 战斗结果标签
* 教师标签（如有）
* 样本 meta

---

# 6. 状态表示设计

V1 不走“全平铺大向量”，也不做白盒事件流。
走 **对象化状态表示**。

## 6.1 当前状态主干

### 玩家对象

包含：

* hp
* max hp
* block
* energy
* 药水状态
* 关键 buff 向量
* 其他关键资源

### 敌人对象

每个敌人一份：

* hp
* block
* intent
* buff 向量
* 是否存活
* 特殊状态摘要

### 手牌对象

每张手牌一个对象：

* `card_id`
* `cost_now`
* `damage_now`
* `block_now`
* `magic_now`
* `is_upgraded`
* `retain`
* `exhaust`
* `ethereal`
* 其他实例状态

### 牌堆摘要

V1 不把整副牌堆全 token 化。
先做摘要：

* draw pile 数量
* discard pile 数量
* exhaust pile 数量
* 攻击/技能/能力数量
* 关键牌剩余计数
* archetype 相关统计

### 静态上下文

不进时序，直接当前状态给：

* 角色
* act
* floor
* 战斗类型（normal/elite/boss）
* relic 集合
* 关键固定 power

---

# 7. 历史表示设计

我们只有 `step` 接口，所以历史不做事件流，只做：

[
(s_{t-k}, a_{t-k}, \Delta_{t-k}), \dots, (s_{t-1}, a_{t-1}, \Delta_{t-1})
]

其中：

[
\Delta_t = diff(s_t, s_{t+1})
]

## 7.1 delta 内容

只做通用差分，不做机制重写。

建议至少有：

* `delta_self_hp`
* `delta_self_block`
* `delta_self_energy`
* `delta_enemy_hp[j]`
* `delta_enemy_block[j]`
* `delta_buff[self][id]`
* `delta_buff[enemy_j][id]`
* `delta_hand_size`
* `delta_draw_pile_size`
* `delta_discard_pile_size`

## 7.2 历史长度

V1 建议：

* 最近 8 步先试
* 不够再扩到 12 或 16

目的不是记整场战斗，而是学：

* 牌序
* buff 演化
* 动作导致的短时状态变化

---

# 8. 合法动作表示设计

每个合法动作一个对象，不做全局动作表 softmax。

## 8.1 每个动作包含

* `action_type`
* `card_id / potion_id / special_id`
* `target_id/type`
* `cost_now`
* `damage_now`
* `block_now`
* `magic_now`
* `tags`
* 可否执行

## 8.2 目标相关信息

如果动作有目标，再加目标摘要：

* target hp
* target block
* target intent
* target buff 摘要

## 8.3 关键点

动作对象表示的是：

**当前这一步的动作实例**

不是“抽象卡牌名字”。

---

# 9. 网络设计

V1 网络建议做成 4 块：

## 9.1 CurrentStateEncoder

输入当前状态对象集合，输出 `h_state`

### 子模块

* `PlayerEncoder`
* `EnemyEncoder`
* `HandCardEncoder`
* `PileSummaryEncoder`
* `StaticContextEncoder`

### 聚合方式

* 玩家直接 MLP
* 敌人对象做 attention pooling
* 手牌对象做 attention pooling
* 全部拼接后再过 MLP

输出一个全局当前状态向量：
[
h_{state} \in \mathbb{R}^{256}
]

---

## 9.2 HistoryEncoder

输入最近 K 步 transition token，输出 `h_hist`

### 每个 token 包含

* 状态摘要
* 动作摘要
* delta 摘要

### 模型

一个小 Transformer：

* hidden dim = 256
* layers = 4
* heads = 4 或 8
* causal 或普通 self-attention 都可以，V1 用普通即可

输出：
[
h_{hist} \in \mathbb{R}^{256}
]

---

## 9.3 ActionEncoder

每个合法动作编码成向量：

[
h_{action_i} \in \mathbb{R}^{128 \sim 256}
]

结构：

* `card_id embedding`
* `action_type embedding`
* 数值特征 MLP
* 目标特征 MLP
* 拼接后过一层融合 MLP

---

## 9.4 Fusion + Heads

融合当前状态和历史：

[
h_{ctx} = MLP([h_{state}, h_{hist}])
]

然后对每个动作打分：

[
score_i = PolicyMLP([h_{ctx}, h_{action_i}])
]

并输出若干头。

---

# 10. 输出头设计

## 10.1 Policy Head

对所有合法动作输出 logit。

训练来源：

* teacher policy
* teacher ranking
* 高质量行为动作

---

## 10.2 Fight Value Heads

不要只用一个 win/loss。

建议至少预测：

* `fight_win_prob`
* `enemy_hp_fraction_dealt`
* `self_hp_fraction_remaining`

这三个能区分：

* 输得很惨
* 输得接近赢
* 惊险赢
* 健康赢

---

## 10.3 Delta Head

预测一步局部变化摘要：

* `delta_self_hp`
* `delta_self_block`
* `delta_enemy_hp`
* `delta_buff`

这个头不直接决定 policy，但能让共享表示更懂：

**动作会怎样改变状态**

---

## 10.4 Uncertainty Head

预测当前状态的不确定性或难度，例如：

* 当前 top1 与 top2 是否接近
* 当前是否需要 teacher 介入

这个头主要给 teacher 调度服务。

---

# 11. 标签设计

## 11.1 战斗结果标签

对一场战斗定义：

* `fight_win ∈ {0,1}`
* `enemy_hp_fraction_dealt`
* `self_hp_fraction_remaining`

可额外定义一个连续分数：

[
fight_score =
1.0 \cdot fight_win +
0.6 \cdot enemy_hp_fraction_dealt +
0.3 \cdot self_hp_fraction_remaining -
0.1 \cdot potion_cost
]

这个用于 value 学习和行为样本加权。

## 11.2 Teacher 标签

关键状态老师输出：

* `teacher_policy`
* `teacher_topk`
* `teacher_best_action`
* `teacher_ranking_margin`
* `teacher_value`

## 11.3 Delta 标签

直接由 `(s_t, s_{t+1})` 做差得到。

---

# 12. 样本池与分桶

## 12.1 数据池

至少分：

* `recent_online`
* `teacher`
* `rare`
* `reanalyse`
* `legacy`

## 12.2 Combat 主桶

主桶先只用：

* `act_floor_stage`
* `encounter_class`
* `build_maturity`

例如：

* `combat|A1_early|normal|base`
* `combat|A2_mid|elite|formed`

## 12.3 标签

额外保存：

* `main_card_id`
* `risk_band`
* `archetype_tags`
* `rare_cohort_tags`
* `student_disagreement`
* `teacher_budget`

---

# 13. 样本保留与淘汰

## 13.1 Recent Online

* 每个主桶单独 cap
* FIFO 淘汰最老
* 可加近重复抑制

## 13.2 Teacher

* 单独池
* 不用 FIFO 直接删
* 按 `keep_score` 淘汰最低分

[
keep_score =
0.40 \cdot disagreement +
0.20 \cdot rarity +
0.20 \cdot hardness +
0.10 \cdot teacher_budget +
0.10 \cdot freshness

* 0.20 \cdot duplicate
* 0.10 \cdot seen
  ]

## 13.3 Rare

* 每个 rare cohort 单独 cap
* 永不和普通样本竞争容量

## 13.4 每张卡约束

在每个 combat 主桶里，对 `main_card_id` 设：

* `min_keep`
* `max_frac`

目标：

* 高频普通牌不能塞满
* 稀有牌不能消失

---

# 14. 训练采样策略

不要全局随机抽。
训练时三层抽样：

## 第一层：按池抽

建议初始：

* 35% `recent_online`
* 25% `teacher`
* 20% `rare`
* 10% `reanalyse`
* 10% `legacy`

## 第二层：按桶抽

Combat 内尽量保证：

* A1/A2/A3 覆盖
* normal/elite/boss 覆盖
* build maturity 覆盖

## 第三层：按卡牌和价值重权重

* 关键 card_id 不缺样本
* teacher 大分歧样本更常见
* 稀有样本有保底出镜率

---

# 15. 损失函数

总损失：

[
L =
\lambda_p L_{policy}
+
\lambda_v L_{value}
+
\lambda_r L_{ranking}
+
\lambda_d L_{delta}
+
\lambda_u L_{uncertainty}
]

建议 V1 先用：

* `λ_p = 1.0`
* `λ_v = 0.5`
* `λ_r = 0.5`
* `λ_d = 0.2`
* `λ_u = 0.1`

## 各项说明

* `L_policy`：teacher 样本拟合 teacher policy，非 teacher 样本做 outcome-weighted imitation
* `L_value`：拟合 `fight_win / hp_fraction_dealt / hp_remaining`
* `L_ranking`：pairwise ranking，老师好动作分数要更高
* `L_delta`：预测一步变化
* `L_uncertainty`：预测难度/不确定性

---

# 16. 训练流程

一轮迭代如下：

## Step 1：采样

用 `student_vK` 跑 N 场战斗，保存原始轨迹。

## Step 2：teacher 队列筛选

把这些状态里满足条件的放入 `teacher_queue`：

* elite / boss
* near-lethal
* 高不确定
* 稀有 build
* 学生 top1 / top2 差距小

## Step 3：老师标注

对 `teacher_queue` 做同-seed oracle 搜索，输出 teacher 标签。

## Step 4：构造训练样本

从原始轨迹构造：

* 当前状态
* 历史
* 合法动作
* delta
* value 标签
* teacher 标签

## Step 5：入池与淘汰

各池各桶执行接纳/淘汰。

## Step 6：训练新学生

从样本池混采训练 `student_v(K+1)`。

## Step 7：评估

在固定 cohort 上评估：

* 战斗胜率
* enemy hp dealt fraction
* self hp remaining
* 老师一致率
* 稀有桶表现

## Step 8：晋级

若新学生整体提升且关键 cohort 不退化，则晋级。

---

# 17. 评估设计

固定三套 cohort：

## Main Cohort

常规战斗代表集

## Rare Cohort

稀有 build / 稀有卡 / 稀有 boss 机制

## Hard Cohort

boss / elite / near-lethal / 高分歧状态

### 核心指标

* `fight_win_rate`
* `enemy_hp_fraction_dealt`
* `self_hp_fraction_remaining`
* `teacher_agreement@1`
* `teacher_topk_overlap`
* 分桶表现

---

# 18. 最小可行版本（你们先跑这一版）

## 范围

* 只做 Ironclad
* 只做战斗
* 先做 Act1
* elite / boss 重点标注

## 网络

* hidden dim 256
* history transformer 4 层
* 当前状态主干 + 历史 + 合法动作打分
* policy + fight value + delta + uncertainty

## 样本池

* recent_online
* teacher
* rare

## 老师

* 同 seed root-focused oracle search
* 只标关键状态

## 训练目标

* teacher 蒸馏为主
* value 多头
* delta 辅助

---

# 19. 工程目录建议

```text
project/
  raw_runs/
  teacher_labels/
  dataset_shards/
  checkpoints/
  eval/
  manifests/
  index.sqlite
```

每一轮生成一份 manifest，记录：

* collector 版本
* teacher 版本
* 数据量
* 池大小
* 训练损失
* 评估结果
* 是否晋级

---

# 20. 这版方案的优点与缺点

## 优点

* 不依赖源码重构
* 能直接利用你现有 `step` 接口
* 搜索老师可以马上开始提供价值
* 动态动作集合可以自然处理
* 历史序列能学牌序和 buff 演化
* 可自动闭环

## 缺点

* 搜索器不是公平正式 agent
* delta 和价值仍然是近似学习
* 某些复杂随机机制泛化仍依赖覆盖
* 需要比较强的样本系统和离线处理

---

# 最终结论

V1 最佳路线不是：

* PPO 主训
* 公平 chance MCTS
* 世界模型
* 全局整局统一

而是：

**战斗先行，学生网络为主体，同-seed oracle 搜索做老师，基于 `(state, action, next_state)` 构造历史序列与多头监督，样本分池分桶管理，自动采样-标注-训练-评估-晋级闭环。**

这版已经足够“先跑一版试试”。
