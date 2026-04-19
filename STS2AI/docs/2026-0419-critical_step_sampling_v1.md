# 2026-0419 critical_step_sampling_v1 首版实现记录

## 目标

本次改动在 `networkV2` 主线上落地关键 combat step 分布修正与原生 combat teacher 首版，覆盖三条链路：

1. full-run PPO 样本的关键步打标、重加权、重采样
2. 关键步 snapshot 队列、branch rollout 原始数据、teacher 数据生成
3. 基于 `critical_step_teacher_v1.jsonl` 的轻量 offline combat teacher 更新

## 代码入口

- 训练入口：[train_full_run_v2.py](/C:/dev/sts2-ai/STS2AI/Python/networkV2/s6_training/train_full_run_v2.py:1066)
- 关键步打标与重采样：[critical_step_pipeline.py](/C:/dev/sts2-ai/STS2AI/Python/networkV2/s6_training/critical_step_pipeline.py:1)
- branch / teacher 生成与加载：[combat_teacher_v1.py](/C:/dev/sts2-ai/STS2AI/Python/networkV2/s6_training/combat_teacher_v1.py:1)
- batch 元数据扩展：[batch.py](/C:/dev/sts2-ai/STS2AI/Python/networkV2/s6_training/batch.py:102)

## 本次落地点

### 1. 在线关键步标注

- `TrainingSample` 新增：
  - `floor`
  - `action_name`
  - `critical_tags`
  - `critical_score`
  - `base_sample_weight`
- `run_full_episode(...)` 在 combat sample 上写入：
  - 楼层
  - 动作名
  - 原始基础权重
- `annotate_critical_steps(samples)` 统一在 GAE / backfill 之后执行，规则包括：
  - `boss_room`
  - `elite_room`
  - `high_adv`
  - `turn_swing`
  - `terminal_swing`

### 2. 分布修正

- `rebalance_training_samples(...)` 采用固定配额：
  - `35% critical_combat`
  - `45% regular_combat`
  - `20% noncombat`
- 桶内不足时允许带放回抽样，保持总样本数不变。
- 指标里额外记录了目标配额和实际输出配额，便于 A/B 对比。

### 3. snapshot / branch / teacher 数据

- `--critical-step-capture` 开启后，combat sample 会导出 snapshot 到：
  - `STS2AI/Artifacts/combat_teacher/<run_name>/snapshots/iter_xxxx/`
- 每 iter 结束后按：
  - `critical_score`
  - `abs(advantage)`
  - `boss > elite > monster`
 生成 `critical_step_queue.jsonl`
- branch generator 会写出：
  - `raw/raw_branch_rollout.jsonl`
  - `raw/raw_manifest.json`
  - `critical_step_teacher_v1.jsonl`

### 4. offline combat teacher

- 支持从 `critical_step_teacher_v1.jsonl` 读取原生 teacher 样本。
- 样本权重按 `best_score - mean(other_scores)` 计算，并裁剪到 `[1.0, 3.0]`。
- 默认 teacher loss 采用：
  - `rank loss`
  - `continuity aux`
- 默认不启用 hard CE。

## 新增参数

训练入口新增以下开关：

- `--critical-step-rebalance`
- `--critical-step-capture`
- `--critical-step-queue-topk`
- `--offline-combat-teacher-data`
- `--offline-combat-teacher-updates-per-iter`
- `--offline-combat-teacher-batch-size`

## 测试

新增测试文件：[test_critical_step_pipeline.py](/C:/dev/sts2-ai/STS2AI/Python/tests/test_critical_step_pipeline.py:1)

覆盖点：

- 关键步规则打标
- 重采样固定配额
- teacher loader 权重与无效 target 哨值
- branch writer / manifest / teacher 数据产出

## 当前边界

- branch rollout 仍是短 horizon greedy rollout，首版优先保证数据链路可跑通。
- offline combat teacher 更新沿用 PPO optimizer，不额外分离 optimizer / scheduler。
- `turn_block_target` 目前只在 teacher 记录显式提供时才会写入；没有 terminal summary 时继续使用无效哨值跳过。
