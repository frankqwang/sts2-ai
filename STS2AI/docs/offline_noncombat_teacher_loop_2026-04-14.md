# offline_noncombat_teacher_loop_2026-04-14

## 目标

把 `card_reward` 短路线搜索从“手工分析工具”收成一条异步慢 teacher 链路：

1. `4env` 主训练继续做便宜采样
2. 每个训练窗口结束后，从 `replays/*.summary.json` 挑一批高价值 seed
3. `2env` 后台只对这些 seed 跑 `card_reward + route_search`
4. 只接受“同 seed 下优于 baseline”的离线样本
5. 把 accepted 样本物化成新的 `offline_noncombat_ranking` 数据集
6. 下一训练窗口再小权重回灌

这条链路是窗口式异步回灌，不要求离线速度和训练速度严格对齐。

## 新增脚本

### 1. 构建窗口 seed 队列

脚本：
[build_offline_noncombat_teacher_queue.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/build_offline_noncombat_teacher_queue.py:1)

作用：

- 读取一个训练窗口的 `replays/*.summary.json`
- 只挑高价值局进入慢 teacher 队列
- 当前默认桶：
  - `boss_reached_defeat`
  - `preboss_death`
  - `act1_clear_anchor`

最小用法：

```powershell
python STS2AI/Python/search/build_offline_noncombat_teacher_queue.py ^
  --run-dir STS2AI/Artifacts/hybrid_training_main_attention/<run_dir> ^
  --output STS2AI/Artifacts/offline_noncombat_teacher_loop/window_2291_2295_queue.json ^
  --seed-list-out STS2AI/Artifacts/offline_noncombat_teacher_loop/window_2291_2295_seeds.txt
```

## 2. 运行 teacher refresh

脚本：
[refresh_offline_noncombat_teacher.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/refresh_offline_noncombat_teacher.py:1)

作用：

- 读取 queue JSON
- 调用离线生成器对这些 seed 跑 `card_reward route_search`
- 用 baseline 门控比较 route 结果
- 只保留更优 seed 的样本
- 产出 `accepted_dataset`

默认生成配置：
[offline_noncombat_teacher_route_default.toml](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/offline_noncombat_teacher_route_default.toml:1)

默认口径：

- `sample_types = "card_reward"`
- `label_mode = "reward_tree"`
- `tree_route_search = true`
- `rollout_goal = "terminal"`
- `tree_max_reward_depth = 2`
- `tree_beam_width = 2`
- `num_envs = 2`
- `auto_launch = true`

最小用法：

```powershell
python STS2AI/Python/search/refresh_offline_noncombat_teacher.py ^
  --queue STS2AI/Artifacts/offline_noncombat_teacher_loop/window_2291_2295_queue.json ^
  --output-dir STS2AI/Artifacts/offline_noncombat_teacher_loop/window_2291_2295_refresh ^
  --checkpoint STS2AI/Artifacts/hybrid_training_main_attention/<run_dir>/hybrid_02295.pt ^
  --auto-launch
```

输出目录里关键产物：

- `comparison_report.json`
- `accepted_seeds.txt`
- `accepted_dataset/raw/raw_branch_rollout.jsonl`
- `accepted_dataset/derived/rl/ranking_sample.jsonl`
- `accepted_dataset/derived/rl/tensors/*.npz`

## 3. 离线生成器新增显式 seed 输入

脚本：
[generate_card_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_card_ranking_data.py:2662)

新增支持：

- `--seed`
- `--seed-file`
- `--num-seeds`

这允许慢 teacher 不再扫 `seed_prefix + episodes`，而是只重标注窗口里挑出来的 seed。

## 4. 训练 summary 的 seed 记录

训练主线：
[train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:3121)

修复点：

- episode summary 在 reset 后会优先从 `state.run.seed` 回填 `seed`
- 这让 `replays/*.summary.json` 可以直接作为 teacher queue 的种子来源

## 门控口径

当前 `refresh_offline_noncombat_teacher.py` 的 baseline 门控是：

1. `route_act1_clear && !baseline_act1_clear`
2. `route_boss_reached && !baseline_boss_reached`
3. `route_end_floor >= baseline_end_floor + min_floor_gain`
4. 同层级下 `boss_hp_fraction_dealt_mean` 高出一个 margin

默认参数：

- `min_floor_gain = 2`
- `boss_damage_margin = 0.10`

## 当前边界

### 老训练窗口可能筛不出 seed

原因不是 queue 脚本坏了，而是老窗口的 `summary.json` 里 `seed` 还是空的。

例如：

- 旧窗口测试时，`200` 个 summary 全部 `missing_seed`
- queue 文件会正常落盘，但 `entries = []`

这意味着：

- 新代码之后产生的训练窗口可以进入 teacher loop
- 老窗口如果没有 seed，需要额外从别的轨迹数据恢复，不在这次最小闭环范围内

### 这条链路是慢 teacher，不直接进在线采样

短路线搜索仍然很贵，所以定位不变：

- 在线训练：便宜采样
- 离线 teacher：异步 route_search 提纯
- 版本化回灌：下一窗口再吃

不要把 `route_search` 直接塞进主训练每一步决策。

## 已做 smoke

最小 smoke 已经验证：

- queue -> generated_run_dir -> comparison -> accepted_dataset

其中 smoke refresh 目录在：
[smoke_refresh](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/offline_noncombat_teacher_loop/smoke_refresh:1)

它已经成功产出：

- `accepted_seeds = 1`
- `accepted_samples = 12`

说明这条闭环从脚本层面已经能跑通。
