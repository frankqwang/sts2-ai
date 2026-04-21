现在这条 `zero` 链路，可以理解成 8 步：

先说当前主线决策：

- 默认主线是 `policy_only_collect`，先做不带搜索的 RL 闭环
- `search / MCTS` 相关链路先保留代码，但降级为显式实验能力
- 主训练入口如果收到非 `disabled` 的 `search_mode`，会直接报错退出
- 也就是说，下面第 5 步在默认配置下通常不会执行

1. `sim state -> BattleState`
   入口是 [game_bridge.py](/C:/dev/sts2-ai/STS2AI/zero/adapters/game_bridge.py:19)。它把 bridge 返回的原始 combat payload 归一成 zero 自己的领域对象：
   - `battle/run/player/enemies/hand/piles` 变成 `BattleState`
   - `legal_actions` 变成带实例唯一 `action_id` 的 `LegalAction`
   - `GameBridgeCombatRuntime.reset/get_state/step` 把 bridge session 包成统一 runtime 端口，见 [game_bridge.py](/C:/dev/sts2-ai/STS2AI/zero/adapters/game_bridge.py:128)

2. `当前策略采样轨迹`
   `TrajectoryCollector` 用某个 policy 去跑战斗，每一步记下：
   - 当前状态 `state`
   - 选择的 `action_index`
   - 执行动作后的 `next_state`
   - 辅助 meta，比如 `uncertainty`、`top2_gap`
   
   代码在 [collector.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/collector.py:10)。输出是 `RawTransition` 序列。

3. `原始轨迹 -> 训练样本`
   `SampleBuilder` 把同一场 fight 的 transition 串起来，生成 `TrainingSample`，见 [sample_builder.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/sample_builder.py:20)。
   这里会补齐：
   - `history`：最近 K 步 `(state, action, delta)`
   - `delta`：`s_t -> s_{t+1}` 的状态差分
   - `fight_label`：战斗胜负、敌方掉血比例、我方剩余血量
   - `bucket_key` / `rare_cohort_tags`
   - `keep_score`
   - `uncertainty_target`
   
   也就是说，`RawTransition` 是“保真日志”，`TrainingSample` 是“可训练样本”。

4. `样本分流入池`
   这层现在是单独的 `SampleAdmissionPlanner`，见 [admission.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/admission.py:15)。
   它的作用是把“一个逻辑样本” clone 成不同池里的独立 entry：
   - 在线样本进 `recent_online`
   - rare 样本额外进 `rare`
   - search 标注后的样本进 `search`
   
   这样避免了“同一个 sample 对象多池共享、后续再被改坏”的问题。

5. `search_queue -> 搜索打标（实验链路，默认关闭）`
   `SearchQueueBuilder` 先从在线样本里挑“值得搜索介入”的状态，规则在 [search.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/search.py:12)：
   - elite / boss
   - near-lethal
   - 高 uncertainty
   - top1/top2 很接近
   - rare cohort
   
   然后 `SearchQueueProcessor` 调 `search_backend.label_request(...)` 真正产出 `SearchLabel`，见 [search.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/search.py:69)。
   
   但当前主线默认不做这一步：
   - `search_mode = disabled`
   - `SearchConfig.max_requests_per_iteration = 0`
   - 只有显式打开实验搜索时，才会真正进入 search backend

6. `池内保留 + 混采 batch`
   `SamplePoolSet` 管 5 个池：`recent_online / search / rare / reanalyse / legacy`，见 [pools.py](/C:/dev/sts2-ai/STS2AI/zero/buffers/pools.py:77)。
   当前策略是：
   - `recent_online/search/rare/reanalyse/legacy` 都用 `keep_score` 保留高价值样本
   - 训练时先按池权重分配 batch，再在池内按 bucket 抽，再做一点 `main_card_id` 多样性倾斜，见 [pools.py](/C:/dev/sts2-ai/STS2AI/zero/buffers/pools.py:96)

7. `样本编码 -> 模型训练`
   训练前先把对象样本编码成 tensor：
   - `FeatureExtractor` 负责把 player/enemy/hand/history/action 编成定长数值，见 [extractor.py](/C:/dev/sts2-ai/STS2AI/zero/features/extractor.py:37)
   - `BatchCollator` 负责 padding 和 mask，见 [batching.py](/C:/dev/sts2-ai/STS2AI/zero/features/batching.py:42)
   
   模型本体是 [network.py](/C:/dev/sts2-ai/STS2AI/zero/model/network.py:22)：
   - `CurrentStateEncoder`
   - `HistoryEncoder`
   - `ActionEncoder`
   - 多头输出：`policy / value / delta / uncertainty`
   
   损失在 [losses.py](/C:/dev/sts2-ai/STS2AI/zero/model/losses.py:22)：
   - `policy`
   - `value`
   - `ranking`
   - `delta`
   - `uncertainty`
   
   `ZeroTrainer` 负责真正训练，见 [trainer.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/trainer.py:17)。

8. `评估 -> 晋级 -> 下一轮继续采样`
   总控是 [loop.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/loop.py:22)。
   一轮 `run_iteration()` 的顺序就是：
   - collect transitions
   - build samples
   - 选 search queue
   - 搜索打标
   - 入池
   - 从当前 active checkpoint 加载模型继续训练
   - 保存 `policy_vXXXX`
   - evaluator 评估
   - promotion judge 决定是否晋级
   
   如果晋级成功，会更新：
   - `_active_version`
   - `_active_policy`
   - `_baseline_eval`
   
   下一轮就不再从零开始，而是沿着上轮 policy 继续采样和训练，见 [loop.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/loop.py:79) 和 [loop.py](/C:/dev/sts2-ai/STS2AI/zero/orchestration/loop.py:91)。

一句话概括就是：

`bridge/sim` 提供战斗状态和合法动作  
-> `collector` 采当前 policy 轨迹  
-> `sample_builder` 变训练样本  
-> `pools` 做样本保留和混采  
-> `trainer` 训练新 policy  
-> `loop` 评估、晋级，并把晋级模型变成下一轮 collector

如果显式打开搜索实验，才会在 `sample_builder` 和 `pools` 之间插入一层 `search label`。

现在主线先聚焦 RL，本阶段最关心的是：
- 不带搜索时，策略和 value 能不能稳定学起来
- evaluator cohort 是否足够稳定

搜索/MCTS 相关能力暂时不作为主线目标。
