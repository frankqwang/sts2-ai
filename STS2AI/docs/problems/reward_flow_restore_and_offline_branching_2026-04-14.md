# reward flow restore 与离线分支回根问题

## 问题概述

本问题发生在离线非战斗排序数据生成链路，影响入口主要是：

- [generate_offline_noncombat_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_offline_noncombat_ranking_data.py:1)
- 其底层实现 [generate_card_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_card_ranking_data.py:1)

症状一开始表现为：

- `save_state/load_state` 在 `card_reward` 或 `combat_rewards` 周边 restore 失败
- `reward_tree` 直接报 `search_error`
- Python 侧后续在非 `card_reward` 状态发 `select_card_reward`
- 再进一步会在非 `map` 状态发 `choose_map_node`

这不是训练主线 live rollout 的问题，而是离线分支搜索这条链路特有的问题。主训练按当前状态一步一步往前走，不依赖 `save/load/export/import` 做分支评估；离线 ranking 生成必须依赖这些能力，所以会先踩中这里。

## 影响范围

直接受影响：

- `reward_tree`
- `map_route_tree`
- `export_state/import_state`
- `card_reward` 附近的 exact restore

不直接受影响：

- [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:1) 主训练在线采样

## 根因拆解

### 1. reward flow restore 语义不一致

旧的 reward flow 修复后，runtime 已经把 `combat_pending` 拆成了更细的过渡态，但 `card_reward` restore 这条链还有一个专门问题：

- [FullRunSimulatorRuntimeFacade.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulatorRuntimeFacade.cs:2317) 原来在恢复 `CardRewardSelection` 时，调用的是 `reward.OnSelectWrapper()`
- 但 full-run overlay 自己真正接管卡奖流程的代码是 [ResolveCardRewardAsync(...)](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulatorRuntimeFacade.cs:2441)

结果是：

- 保存时明明在 `card_reward`
- 恢复后却回到 `combat_rewards`
- 进而触发 `restore_signature_mismatch`

### 2. export/import 自己缺序列化元数据

`reward_tree` 不是只用 `save_state/load_state`，还会用 `export_state/import_state` 做根节点和子节点快照：

- [card_reward_tree.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/card_reward_tree.py:77)
- [map_route_tree.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/map_route_tree.py:72)

之前这条链直接报：

- `FullRunExportedRunSnapshot` 没有 `JsonSerializerContext` 元数据

于是 `reward_tree` 会在 `client.export_state(...)` 这里直接异常，被上层吞掉以后表现成 `search_error`，最后退回 `single_step`。

### 3. Python 分支执行器没有把 backend restore 回根状态

即使 `reward_tree` 本身跑通，如果函数退出时没有把 backend restore 回最初的 `card_reward`，调用者就会拿着“旧的 Python 变量 state”继续发动作，但 backend 已经跑到了别的 screen。

具体问题点在：

- [_resolve_best_card_reward_choice(...)](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_card_ranking_data.py:767)

原实现会：

- 导出 root snapshot
- 在分支树里反复 `import_state -> apply`
- 算完分数后直接返回 best action

但不会在 return 前 restore root snapshot。于是上层继续对“以为还是 `card_reward`”的状态发 `select_card_reward`，backend 实际却已经到 `map` 或别的后续 screen，于是报：

- `select_card_reward is only valid from card_reward`
- `choose_map_node is only valid from map`

## 修复内容

### 1. reward flow restore 改成走 overlay 自己的卡奖解析链

修改：

- [FullRunSimulatorRuntimeFacade.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulatorRuntimeFacade.cs:2317)

修法：

- 恢复 `CardRewardSelection` 时，不再调用 `reward.OnSelectWrapper()`
- 改为直接启动 [ResolveCardRewardAsync(...)](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulatorRuntimeFacade.cs:2441)
- 等待条件也从泛化的 `IsSelectionActive` 收紧到 `IsCardRewardSelectionActive`

效果：

- `card_reward` 页面上的 `save_state -> load_state` 可以稳定回到 `card_reward`

### 2. 给 exported snapshot 单独补 serializer context

新增：

- [FullRunSimulationSerializerContext.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulationSerializerContext.cs:1)

修改：

- [FullRunSimulatorRuntimeFacade.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulatorRuntimeFacade.cs:40)
- [FullRunSimulatorRuntimeFacade.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulatorRuntimeFacade.cs:1842)
- [FullRunSimulatorRuntimeFacade.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulatorRuntimeFacade.cs:2152)

修法：

- 不再去改全局的 `JsonSerializationUtility.Options`
- 为 `FullRunExportedRunSnapshot` 单独构造一份 `JsonSerializerOptions`
- 在这份 options 上组合 `FullRunSimulationSerializerContext.Default`

效果：

- `export_state/import_state` 不再因为元数据缺项直接崩掉

### 3. `_resolve_best_card_reward_choice` 结束前强制 restore root

修改：

- [generate_card_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_card_ranking_data.py:854)

修法：

- 在 `_resolve_best_card_reward_choice(...)` 返回 best action 之前
- 强制 `client.import_state(snapshot_path)`
- 并校验 restore 后必须回到 `card_reward`

效果：

- 共享分支求值函数在退出时不再把 backend 留在脏状态
- 上层 `map_route_tree` / 继续 rollout 可以安全复用它

## 验证结果

### 最小验证 1：`card_reward` 页直接 save/load

固定 seed：

- `ONCR2290C_00046`

修复后，`card_reward` 页上的：

- `save_state`
- `load_state`

已经能直接回到 `card_reward`，不再跳回 `combat_rewards`。

### 最小验证 2：单独 `reward_tree`

同一个 seed 下，单独跑 `evaluate_card_reward_tree(...)`：

- 能返回非零 spread 分数
- 不再在 `export_state(...)` 处抛序列化异常

### 最小验证 3：完整 `generate_from_episode(...)`

同一个 seed 下，完整 episode 生成结果从原来的：

- `status='post_card_reward_restore_failed'`
- `search_error=1`
- `card_reward` 样本退回 `single_step`

变成：

- `status='floor_cap'`
- `search_error=0`
- `sample_count=6`
- `map_samples_recorded=4`
- `card_reward_samples_recorded=2`
- `card_reward` 样本 `label_source='reward_tree'`

一次实际复现样本：

- `card_reward reward_tree [0.9527, 0.8411, 0.9121, 0.9886]`

说明 reward tree 和 map route tree 已经都能产出有效 spread。

## 当前结论

这次问题不是单点 bug，而是一条离线分支链路上连续叠了 3 层：

1. reward flow restore 回错 screen
2. export/import 缺 serializer metadata
3. 分支求值函数退出时没 restore root

三层都修完后，固定 seed 已经能正常生成 `reward_tree` 样本，且不会再出现“非 `card_reward` 发 `select_card_reward`”和“非 `map` 发 `choose_map_node`”这一类 backend / Python 脱节错误。

## 后续约束

后面如果继续扩离线数据生成链路，必须遵守：

- 共享动作推进语义继续复用 [full_run_action_semantics.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/runtime/full_run_action_semantics.py:1)
- 分支搜索函数如果会改 backend live state，返回前必须 restore 回根状态
- `export_state/import_state` 的自定义快照类型，要么挂入专用 serializer context，要么显式走单独 serializer options，不要隐式依赖全局 save serializer
