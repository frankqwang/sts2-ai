# Zero 主线切换：先做 RL，默认关闭搜索 / MCTS

## 1. 结论

当前 `STS2AI/zero` 主线先不继续推进搜索 / MCTS。

从这一版开始，主线默认切到：

- `collect_mode = policy_only_collect`
- `search_mode = disabled`
- `SearchConfig.max_requests_per_iteration = 0`

也就是：

- 默认不做 `search_queue`
- 默认不做搜索打标
- 默认不在 collect 阶段依赖搜索动作

并且主训练入口会直接拦截：

- 只要 `search_mode != disabled`，立即报错退出
- 只要 `collect_mode` 试图切到搜索 collect，立即报错退出

搜索 / MCTS 相关代码先保留，但统一视为**显式实验能力**，不是主线。

## 2. 为什么先停 MCTS

这次决策不是否定搜索本身，而是为了先把主线问题收敛。

当前阶段，继续把 MCTS 放在主线里，会同时带来 3 类干扰：

1. 训练问题和搜索问题耦合在一起  
   一旦训练效果不好，很难快速判断问题到底来自：
   - policy/value 学习本身
   - 搜索评分逻辑
   - 搜索标签质量
   - snapshot / replay / sidecar 一致性

2. 默认路径的心智模型过重  
   现在一提 `zero`，默认就会联想到：
   - `search_only_collect`
   - `MultiCaseSearchBackend`
   - same-seed search label
   - search trace / root cache / snapshot restore  
   这会让主线调试成本非常高。

3. 当前最需要验证的是“纯 RL 能不能先学起来”  
   主线先回答这个更基础的问题：
   - 不带搜索标签时，policy 能不能稳定改进
   - value/head 是否有正常学习信号
   - evaluator cohort 是否稳定

所以当前阶段更合理的顺序是：

`先做纯 RL 主线`
`-> 先把 baseline 跑稳`
`-> 再决定是否重新引入搜索`

## 3. 这次切换后的主线定义

### 3.1 默认 collect

默认使用：

- `policy_only_collect`

含义是：

- 动作直接来自当前 policy
- 可保留 epsilon / temperature 探索
- 但不再让搜索器接管动作选择

### 3.2 默认搜索状态

默认使用：

- `search_mode = disabled`

对应行为：

- `build_search_backend(...)` 返回 `NoopSearchBackend`
- `SearchQueueBuilder` 默认请求数为 `0`
- `ZeroLoopRunner` 仍保留统一接口，但不会产出搜索标签

### 3.3 默认训练信号

主线默认主要依赖：

- online 行为样本
- fight/value/delta/uncertainty 这些已有监督

也就是先观察：

- 不带搜索标签时，loss 是否稳定
- 纯 online 样本是否能带来行为改进

## 4. 代码层面的明确约定

### 4.1 主入口

主入口是：

- [/C:/dev/sts2-ai/STS2AI/zero/replay/train.py](/C:/dev/sts2-ai/STS2AI/zero/replay/train.py)

从这里开始，默认值应明确表达：

- 当前主线不是 MCTS
- 搜索 collect 不是默认行为
- 搜索模式若打开，不是“实验继续跑”，而是直接报错

### 4.2 空搜索 backend

为了不把 `loop` / `trainer` / `collector` 的接口全部打散，保留：

- `SearchBackend` 端口

但主线默认接：

- `NoopSearchBackend`

它的意义不是“一个更弱的搜索器”，而是：

- 明确表达“这里没有搜索”
- 避免到处用 `None` 做特判
- 让主线和实验线的切换更清楚

同时参数层也会额外保护：

- `--search-mode` 当前只允许 `disabled`
- `--collect-mode` 当前只允许 `policy_only_collect`

这样即便后面有人误传参数，也不会把 MCTS 悄悄重新打开。

### 4.3 搜索 collect 当前禁止开启

下面这些模式不再允许作为默认路径：

- `search_only_collect`
- `search_guided_collect`
- `search_root_sweep`
- `search_branching`

当前主训练入口里，这些模式都不会进入“实验继续跑”的状态，而是直接失败退出。

## 5. 为什么先不动搜索代码

当前不删搜索代码，是为了保留后续实验能力。

这包括：

- `MultiCaseSearchBackend`
- `CombatSearchBackend`
- search trace / root cache / snapshot restore
- sim-native / sidecar / attached 这些后续方案讨论

但这些能力当前都不属于主线目标。

也就是说：

- 可以保留
- 可以继续单独实验
- 但不能再让默认入口自动走进去

## 6. 和后续 MCTS 方案的关系

之前已经讨论过一版更长期的搜索方向：

- 统一搜索核心
- 支持 `attached` / `sidecar`
- 删除 `MultiCaseSearchBackend`
- 改成围绕当前 session / snapshot 搜索

这套方向当前**不否定**，但统一延后。

当前阶段先不展开实现，原因是：

- 还没有证明纯 RL 主线本身已经稳定
- 过早重构搜索架构，会同时动到 runtime / session / planner / spectate
- 容易再次把主线问题和搜索问题搅在一起

所以当前策略是：

- 搜索架构设计保留为 backlog
- 主线先做 RL baseline

## 7. 当前阶段的成功标准

当前主线先关注这些问题：

1. 纯 `policy_only_collect` 下，训练是否稳定  
2. evaluator 是否能稳定区分版本优劣  
3. policy/value 是否出现明确提升  
4. 不带搜索时，样本池和 loss 是否仍然合理  

只有这几件事先回答清楚，后面再讨论“要不要把搜索重新接回来”才有意义。

## 8. 后续再开搜索时的要求

以后如果重新启用搜索 / MCTS，至少要满足：

1. 不是默认路径  
2. 日志和 manifest 能明确区分 RL 主线 vs 搜索实验  
3. snapshot / sidecar 一致性有持续校验  
4. 搜索标签质量和训练收益有独立评估  

在满足这些条件之前，搜索都不应重新回到主线。
