# 2026-0419 Async Rollout Engine

## 目标

把原先训练期默认的“每个 worker 线程各自 `compile -> forward -> env step`”同步收集，改成：

- 多进程 actor 独占 simulator session
- 主进程集中 batched inference
- 按 batch bucket 的 CUDA graph cache

当前默认覆盖的在线训练入口：

- `networkV2.s6_training.combat_cotrainer`
- `networkV2.s6_training.train_full_run_v2`
- `networkV2.s6_training.train_combat_v2`

旧线程收集路径仍保留，但只作为隐藏 debug fallback：`--legacy-thread-rollout`。

## 架构

### Actor

- 每个 actor 是独立进程，独占一个 sim port。
- actor 只负责：
  - `reset / step / get_state`
  - `CombatStateTracker`
  - `CombatFeatureCompiler`
  - `TrainingSample` 回填
- actor 不持有模型，不直接访问 GPU。

### IPC

- 推理请求通过共享内存 slot 传递，slot 形状由 `BankMaxSpec` 固定。
- 每个 slot 预分配：
  - 各 bank 的 `numeric / type_ids / ts_ids / mask`
  - `decision_domain`
  - `encounter_idx`
  - `legal_len`
  - `greedy`
  - `request_id`
  - `active_task_id`

### InferenceCoordinator

- 主进程后台线程，消费所有 actor 的请求队列。
- 按 `decision_domain` 聚合请求，使用 `rollout_infer_max_wait_ms` 小时间窗凑 batch。
- unified 网络走：
  - `batched_banks`
  - `GraphBatchBucketCache`
  - bucket 未命中时 eager batched forward
- `combat_v2` 目前走 batched eager forward。

### 故障恢复

- actor dead：自动重启并重排队当前 task。
- actor stuck：若超出 `rollout_actor_reply_timeout_s`，会 terminate + respawn。
- 队列是有界的，避免无限积压。

## 统一参数

三条入口都支持：

- `--rollout-num-actors`
- `--rollout-infer-batch-size`
- `--rollout-infer-max-wait-ms`
- `--rollout-queue-depth`
- `--rollout-graph-batch-buckets`
- `--no-rollout-graph`

隐藏调试参数：

- `--legacy-thread-rollout`
- `--rollout-actor-reply-timeout-s`
- `--rollout-result-poll-timeout-s`
- `--rollout-max-actor-restarts`

兼容规则：

- 若未显式传 `--rollout-num-actors`，默认继承旧的 `--num-workers`。

## 代码位置

- 异步引擎：`STS2AI/Python/networkV2/s6_training/rollout_async_engine.py`
- batched graph bucket：`STS2AI/Python/networkV2/s5_net/graph_runner.py`
- static padded buffer 写入：`STS2AI/Python/networkV2/s5_net/tokenizer.py`
- 三个训练入口已经默认接入 async runtime。

## 验证

### 单测

已通过：

- `python -m pytest STS2AI/Python/tests/test_graph_runner.py STS2AI/Python/tests/test_rollout_async_engine.py -q`

覆盖：

- shared slot roundtrip
- batched inference reply
- dead actor restart
- stuck actor timeout restart
- graph runner padding / shape 回归

### smoke

已验证：

- `combat_cotrainer` async 路径可启动、可采样、可回主线程
- `train_combat_v2` async 路径可启动、可采样、可训练循环收尾

当前环境限制：

- `train_full_run_v2` async 路径已能启动 actor 和集中推理，但本地环境底库仍报
  `no such column: possible_monsters_json`
  这属于 simulator/catalog sqlite schema 问题，不是 async runtime 本身的问题。

## benchmark

新版 benchmark：

- `STS2AI/Python/tools/cuda_graph_rollout_bench.py`

支持：

- `--engines legacy,async`
- `--modes eager,graph`
- `--workers 2,4,...`

产出默认建议写到：

- `STS2AI/Artifacts/benchmarks`

## 已知问题

1. `combat_cotrainer` 的主进程目录查询不能用 proto `CombatSession.combat_catalog()`，因为该路径当前返回空 catalog。
   现在已改成：
   - actor 训练仍走 proto `CombatSession`
   - 主进程 helper/query 走 JSON `env.combat_training_env.PipeBackedCombatTrainingClient`

2. non-combat graph bucket 之前在 `OptionContextualizer` 里每步构造 `torch.tensor(type_id, device=...)`，
   会破坏 CUDA graph capture。该问题已改成预注册 `_bank_type_ids` buffer。

3. `train_full_run_v2` 的 full-run schema 仍受环境底库影响；若要继续压测 full-run 吞吐，需要先修底库字段缺失。

## TODO

- `TODO(pb-align)`: 补齐 proto/pb 路径的静态 catalog 接口，对齐至少以下能力：
  - `game_catalog`
  - `combat_catalog`
  - `power metadata`（`base_classes / is_debuff_hint`）
  - 视需要补 `perf_stats / reset_perf_stats`
- 目标是让主力训练链在全 proto 下不再依赖 JSON helper 或 sqlite fallback。

## 模块边界

- `STS2AI/Python/networkV2/s6_training/rollout_async_engine.py`
  - 放通用并发引擎。
  - 负责 shared slot、请求队列、batch inference、graph bucket、actor restart、指标汇总。
  - 不放具体 trainer 的任务结构、deck/build 采样策略、reward 逻辑。

- `STS2AI/Python/networkV2/s6_training/rollout_workers.py`
  - 放训练入口到 async 引擎的适配层。
  - 负责：
    - 每条训练线对应的 worker factory
    - task payload -> rollout 函数调用
    - helper catalog client 的统一创建
    - `create_*_runtime()` 工厂函数
  - 目标是让上层 trainer 只依赖稳定工厂，不再自己管理 actor entry、reply queue、连接生命周期。

- `combat_cotrainer.py / train_full_run_v2.py / train_combat_v2.py`
  - 只保留任务生成、训练循环、指标/ckpt、domain 业务逻辑。
  - 不再直接 new `QueueInferenceClient`、`PersistentActorRuntime`，也不再各自复制 actor 主循环。
