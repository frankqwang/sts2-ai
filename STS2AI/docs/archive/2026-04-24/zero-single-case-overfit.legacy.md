# Zero 单 Case 过拟合实验

## 目的

先固定一场战斗 case，不做多样性样本管理。当前 zero 主线已完全移除搜索 / MCTS，
单 case 过拟合实验自然也走纯 policy 路线。

目标只有一个：

- 对比不同模型结构，观察谁更容易把这一个 case 拟合下来
- 先验证模型是否能学会牌序、铺垫、combo

## 当前实现

代码入口：

- `STS2AI/zero/experiments/single_case_overfit.py`

当前支持的模型结构：

- `stateless`
- `history_transformer`
- `recurrent_gru`

训练入口仍复用：

- `STS2AI/zero/replay/train.py`

只是额外加了：

- `--model-variant`

## 产物位置

实验代码放在：

- `STS2AI/zero/experiments`

运行产物默认放在：

- `STS2AI/Artifacts/zero/experiments`

单次批量实验目录里会包含：

- `experiment_config.json`
- `summary.json`
- `runs.csv`
- `logs/*.log`
- `runs/*` 下每个实际训练 run 的完整产物

## 用法

先确认目标 case id，然后运行：

```powershell
cd C:\Users\Administrator\Desktop\sts2Zero\STS2AI
python -m zero.experiments.single_case_overfit `
  --target-case-id run_1312734_floor_2_shrinker_beetle_weak `
  --variants stateless history_transformer recurrent_gru `
  --seeds 20260420 20260421 20260422 `
  --iterations 12 `
  --collect-episodes 64 `
  --train-steps 256 `
  --eval-episodes 16
```

如果只想先试一个模型：

```powershell
cd C:\Users\Administrator\Desktop\sts2Zero\STS2AI
python -m zero.experiments.single_case_overfit `
  --target-case-id run_1312734_floor_2_shrinker_beetle_weak `
  --variants recurrent_gru `
  --seeds 20260420 `
  --iterations 4 `
  --collect-episodes 16 `
  --train-steps 64 `
  --eval-episodes 4
```

## 结果怎么看

先看：

- `summary.json`
- `runs.csv`

重点指标：

- `final_fight_win_rate`
- `final_self_hp_fraction_remaining`
- `best_fight_win_rate`
- `final_avg_step_count`
- `mean_total_loss`

如果一个模型在多个 seed 上都能快速把 `fight_win_rate` 拉高，并且剩余血量稳定更高，它更适合作为后续纯 RL 主线的候选骨架。
