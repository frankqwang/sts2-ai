# MCTS 正确性验证

## 什么时候才允许开 MCTS 训练

只有下面几条都成立，MCTS 才能当训练老师：

1. `save_state -> load_state` 后根状态一致。
2. 搜索结束后的 `cleanup_and_restore()` 能回到原始根状态。
3. 对同一个根动作，`forward_model.step(action)` 和真实 `client.act(action)` 的子状态一致。
4. 分支里不会去执行当前 `legal_actions` 之外的动作。
5. 小规模固定 seed 回归是 `0 error`、`0 timeout`。

只要其中一条不成立，MCTS 监督就不够干净，不能拿来训练。

## 当前推荐的验证流程

### 1. 单测

先跑 [test_training_smoke.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/test_training_smoke.py:5770)。

重点要锁住三类问题：

- `play_card` 的严格 target 匹配
- 卡牌/药水动作不能走宽松 fallback
- 相邻 legal action 之间不能串卡牌元数据

### 2. 管线审计

再跑 [mcts_pipe_audit.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/diagnostics/mcts_pipe_audit.py:1)。

它主要检查：

- root 的 save/load 是否一致
- 搜索后的 restore 是否回到 root
- `forward_model.step(action)` 和真实环境执行后的 child state 是否一致

这是查 save/load、restore、分支推进是否正确的主工具。

### 3. 决策探针

如果 legality 和 parity 都过了，但 MCTS 看起来还是比 plain 差，就跑 [mcts_decision_probe.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/diagnostics/mcts_decision_probe.py:1)。

它会并排看：

- plain policy logits
- MCTS root prior
- root Q
- root visit count
- 最终 chosen action

这一步是为了判断：

- 是不是 root 最终选动作规则在带偏
- 是不是 prior 太强
- 是不是 value/backup 口径还不理想

### 4. 小规模固定 seed 门禁

在更大 benchmark 或训练之前，先跑一组固定小 seed，至少要求：

- `0 error`
- `0 timeout`
- 没有新的 `Pipe step failed`

这轮调试里常用的是 `EVAL_001..EVAL_007`。

## 这轮踩出来的几个硬规则

### 1. 卡牌动作必须严格匹配

`play_card`、`use_potion` 这类动作，不能在 strict key 失配后退化成：

- 只按 action name 匹配
- 只按 label 匹配
- 或者“差不多像同一张牌”就继续执行

这类宽松匹配是下面这些错误的主要来源：

- `EnergyCostTooHigh`
- `requires a target`
- restore 后继续执行旧分支动作

### 2. `card_index` 不能当稳定身份

跨 save/load 后，单靠 `card_index` 不可靠。

更稳的是：

- `card_id`
- `label`
- `cost`
- `target_id`

然后再用当前 server legal action 重建最终 payload。

### 3. save/load 之后要重灌 RNG

`RunState.FromSerializable(...)` 会恢复序列化出来的 RNG，但房间 bootstrap 本身也可能先吃掉一部分随机数，比如：

- shop roll
- combat setup
- treasure 初始化

所以 restore 完房间后，必须再把保存下来的 run/player RNG 灌回去。对应代码在：

- [FullRunSimulatorRuntimeFacade.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulatorRuntimeFacade.cs:1704)

否则即使表面状态一样，后续随机分支也会漂。

## 当前训练建议

在下面三条没有同时满足前，不建议开大规模 MCTS 训练：

- correctness 已经签收
- 固定 seed 上，MCTS 至少不明显弱于 plain
- 对应 `mcts_sims` 的 wallclock 成本可接受

当前阶段更适合：

- 小规模研究跑
- 固定 benchmark 对照
- 先验证老师值不值得，再考虑扩量训练
