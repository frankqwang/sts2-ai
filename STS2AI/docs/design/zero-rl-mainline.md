# Zero 主线：纯 policy-only RL，已完全移除搜索 / MCTS

## 1. 结论

当前 `STS2AI/zero` 主线专注纯 RL 闭环，不做搜索 / MCTS。

从这一版开始：

- collect 动作完全由当前 policy 自己给出（带 `epsilon_greedy` / `temperature` 探索）
- 不再有任何 `SearchBackend` / `SearchQueueBuilder` / same-seed root search / snapshot 回溯链路
- 代码仓库里已经把搜索相关的模块（`orchestration/search.py`、`replay/search_backend.py`、`NoopSearchBackend`、`AggregateCardUsageSearchBackend`、`MultiCaseSearchBackend` 等）整体删除
- 训练样本和 loss 中和搜索相关的字段（`search_label`、`search_policy`、`search_trace`、`search_agreement_at_1`、`policy_search_kl_weight` 等）全部移除

## 2. 为什么不再做搜索

核心理由是：**我们要训练的是模型自身的思考能力，而不是把外部 save/load + 回溯能力当成“作弊”途径。**

继续把 MCTS 放进主线会带来 3 类问题：

1. 训练问题和搜索问题耦合  
   训练不收敛时，很难判断问题来自 policy/value 学习本身，还是搜索评分、搜索标签、snapshot 一致性。

2. 对外部 save/load 接口的强依赖  
   搜索链路深度依赖 `GameBridgeCombatRuntime.save_state / load_state`，这让模型的“决策能力”相当程度上来自外部回溯，而不是网络参数本身。

3. 默认路径心智模型过重  
   之前默认路径会带上 `search_only_collect` / `MultiCaseSearchBackend` / same-seed search label / search trace / root cache / snapshot restore，全链路排查成本高。

所以现在的原则是：

- 模型的推理能力完全来自 `ZeroNet` 的 policy / value / delta / uncertainty
- 不给模型提供任何形式的 rollout 回溯能力
- 如果要再实验搜索，应该在独立分支上单独做，不要挂到主线

## 3. 主线默认行为

### 3.1 collect

- 动作直接来自当前 policy
- 支持 `epsilon_greedy` / `temperature` 做采样探索
- 不再有 `policy_only_collect` / `search_guided_collect` 的模式开关 —— 因为只剩一种模式

### 3.2 训练信号

`ZeroNet` 输出四类头：

- `policy`（行为动作加权 CE，落在 `losses.policy_behavior_ce_weight`）
- `value`（`fight_win / enemy_hp_fraction_dealt / self_hp_fraction_remaining`）
- `delta`（一步想象的局部状态变化）
- `uncertainty`（难度/不确定性）

PPO-lite 算法走自己的分支，产出 `ppo_value / ppo_return / ppo_advantage` 并做 clip ratio 更新。

### 3.3 评估与晋级

`PromotionJudge` 基于：

- cohort 胜率 / HP 剩余 / 战斗质量分
- timeout 率 / 无进展比例 / 无进展连续回合
- 整体胜率 / 敌方掉血提升

不再检查 `search_agreement_at_1` 等搜索指标。

## 4. 代码层面的不变量

- 主训练入口：[/C:/dev/sts2-ai/STS2AI/zero/replay/train.py](/C:/dev/sts2-ai/STS2AI/zero/replay/train.py)
- 核心循环：[/C:/dev/sts2-ai/STS2AI/zero/orchestration/loop.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/loop.py)
- 网络模型：[/C:/dev/sts2-ai/STS2AI/zero/model/network.py](/C:/dev/sts2-ai/STS2AI/zero/model/network.py)
- 损失函数：[/C:/dev/sts2-ai/STS2AI/zero/model/losses.py](/C:/dev/sts2-ai/STS2AI/zero/model/losses.py)

这几个文件里不应再出现 `Search*` 类型、`search_*` 字段或任何调用 `save_state / load_state` 做回溯的逻辑。

## 5. 当前阶段的成功标准

1. 纯 `policy` 采样下，训练是否稳定
2. evaluator 能稳定区分版本优劣
3. policy / value / delta / uncertainty 各头都有明确学习信号
4. 样本池和 loss 在无搜索标签时仍然合理

## 6. 如果以后重新考虑搜索

必须满足：

1. 不复用这条主训练入口
2. 单独走实验分支，避免再次和主线耦合
3. 先证明搜索能给主线带来稳定收益，而不是反过来成为模型能力的“替代”
