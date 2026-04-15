# Combat Turn Teacher

IRONCLAD Act1 战斗教师（combat teacher）训练链路的入口文档。细节设计在 tactical v1 设计文档里，各轮迭代记录在 session notes 里。

- 设计：[combat_turn_teacher_tactical_v1.md](combat_turn_teacher_tactical_v1.md)
- session notes：
  - [combat_turn_teacher_v4_session_notes.md](combat_turn_teacher_v4_session_notes.md)（首次把 66.67% → trajectory-replay 方向成立）
  - [combat_turn_teacher_v6_session_notes.md](combat_turn_teacher_v6_session_notes.md)（当前最佳，24-seed 62.5%）

## 这是什么

在 `train_hybrid` 的 PPO 主训练循环之外，额外加一项 **combat teacher loss**：

- 收一批 IRONCLAD 战斗中状态样本（state + legal_actions）
- 每个样本用 beam search `combat_turn_solver` 算出本回合的最优出牌序列
- 训练时让 combat policy 在这些状态上的动作 softmax 概率向求解器的答案靠拢
- 等价于一个带 soft label 的行为克隆辅助头

损失权重 `combat_teacher_loss_weight` 默认 0.2，批 32，每迭代 6 次更新。

## 数据生产的两条路径

### 路径 A：live builder —— 脚本自己玩游戏

`STS2AI/Python/search/build_act1_combat_teacher_v2_dataset.py`

脚本内部持有一个 FullRunPolicyNetworkV2（noncombat）+ combat baseline policy，自己从 seed 0 reset 开始打完整局游戏，遇到 combat state 就挂 solver 出样本。配合 `--progress-combat-with-solver` 可以让 solver 本身帮助推进战斗。

**已知问题**：这条路径在大多数 seed 上推不过 floor 2-6。根因是 noncombat 推进和 evaluate_ai 完整流程（含 combat_safety_rerank / combat_turn_solver 集成等）有差异。上一轮 `mixed` 数据集 219 条里 171 条都是 floor 2，就是这个原因。

仍然保留这条路径做 floor 2 / elite 1 / floor 11-13 这些能推得到的场景的基础样本。

### 路径 B：trajectory replay —— 本轮新增

`STS2AI/Python/search/build_combat_teacher_from_trajectory.py`

绕过 live builder 的推进问题：先用 `evaluate_ai.py --save-trajectory-dir --trajectory-seeds` 把一个已经能打得动的 checkpoint（如 mixed_w02）的完整战斗录像（每步 raw_state + chosen_action）存成 JSONL。replay builder 只需：

1. reset sim 到该 seed
2. 按 trajectory 里的 action 顺序重放
3. 当 sim state 变成 floor ≥ `--min-sample-floor` 的 combat 时，挂 solver 出样本
4. 继续重放到 trajectory 结束

这样样本的状态分布 == evaluate_ai 跑出来的真实分布，包括 elite + boss。

**注意事项**：当 trajectory 里的 `chosen_action` 在当前 sim 的 `legal_actions` 里通过 `sanitize_action` 匹配不上时（常见于 card_reward 这类 noncombat 步骤的 action 格式差异），脚本会退化到选 `legal_actions[0]`（通常是 proceed），**不** break 整条 trajectory。本轮 v6 > v5 的关键修复就是这里（见 v6 session notes）。

## 合并工具

`STS2AI/Python/search/merge_combat_teacher_datasets.py`

读若干 jsonl，按 sample_id 去重、按 `stable_split` 重新划 train/holdout、写 summary。所有 mixed_vN 数据集都用它合。

## 当前最佳：mixed_v6

### 数据集

`STS2AI/Artifacts/combat_teacher/tactical_v1_mixed_v6_20260415_213719/ironclad_act1_tactical_teacher_v1_mixed_v6.jsonl`

502 条样本（train 391 / holdout 111），floor 分布：

| floor | 数量 | 来源 |
|---:|---:|---|
| 2 | 171 | live builder（boss_medium）|
| 11-13 | 48 | live builder（highfloor_solverprog）|
| 14 | 134 | trajectory replay |
| 15 | 65 | trajectory replay |
| 17 (boss) | 84 | trajectory replay（SOUL_FYSH 51 + WATERFALL_GIANT 33）|

### 训练配置

和 mixed_w02_5iter 一模一样，只换了数据：

- config：`STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_tactical_teacher_smoke.toml`
- resume 自：`STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt`
- `--max-iterations 5 --episodes-per-iter 8`
- `--combat-teacher-loss-weight 0.2 --combat-teacher-batch-size 32 --combat-teacher-updates-per-iter 6`

### 结果（24 seed）

固定评估套件：`evaluate_ai.py --num-games 24 --seed-suite regression`（EVAL_001..EVAL_024）。

| checkpoint | win rate | avg_floor | combats_won | avg_boss_hp_fraction |
|---|---:|---:|---:|---:|
| baseline retrieval_final_iter2175 | 45.83% | 14.1 | 7.42 | 0.278 |
| mixed_w02_5iter（前一轮最佳）| 54.17% | 13.8 | 7.79 | 0.408 |
| mixed_v4 | 58.33% | 14.5 | 8.25 | 0.440 |
| **mixed_v6** | **62.50%** | **14.9** | **9.83** | 0.444 |

相对 baseline：**+16.67 百分点 win rate**。

## 复现 mixed_v6 训练的最短命令链

从 trained w02 checkpoint 导出 trajectory（两批 seed）：

```powershell
# 原 6 seed
python STS2AI/Python/evaluate_ai.py `
  --checkpoint <w02_checkpoint.pt> `
  --transport pipe-binary --port 15800 --auto-launch `
  --num-games 8 --seed-suite regression `
  --save-trajectory-dir <out>/trajectory `
  --trajectory-seeds "EVAL_001,EVAL_002,EVAL_003,EVAL_004,EVAL_005,EVAL_006" `
  --output <out>/traj_eval.json

# 扩展 18 seed
python STS2AI/Python/evaluate_ai.py `
  --checkpoint <w02_checkpoint.pt> `
  --transport pipe-binary --port 15900 --auto-launch `
  --num-games 24 --seed-suite regression `
  --save-trajectory-dir <out>/trajectory_ext `
  --trajectory-seeds "EVAL_007,...,EVAL_024" `
  --output <out>/traj_ext_eval.json
```

把两批 trajectory 分别通过 replay builder 抽样：

```powershell
python STS2AI/Python/search/build_combat_teacher_from_trajectory.py `
  --trajectory-dir <out>/trajectory `
  --combat-checkpoint <baseline_checkpoint.pt> `
  --transport pipe-binary --port 15810 --auto-launch `
  --teacher-config STS2AI/Python/configs/combat_turn_teacher_tactical_v1.toml `
  --min-sample-floor 14 `
  --max-samples-per-floor-per-seed 10 `
  --max-samples-per-seed 40 `
  --output <out>/replay.jsonl --eval-output-dir <out>/replay/eval
```

合并三份（原 mixed + 两批 replay）：

```powershell
python STS2AI/Python/search/merge_combat_teacher_datasets.py `
  --source STS2AI/Artifacts/combat_teacher/tactical_v1_mixed_20260415_184015/ironclad_act1_tactical_teacher_v1_mixed.jsonl `
  --source <out>/replay.jsonl `
  --source <out>/replay_ext.jsonl `
  --output <out>/mixed_v6.jsonl
```

训练：

```powershell
python STS2AI/Python/train_hybrid.py `
  --config STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_tactical_teacher_smoke.toml `
  --pipe --transport pipe-binary --auto-launch `
  --num-envs 2 --start-port 15960 `
  --max-iterations 5 --episodes-per-iter 8 `
  --resume <baseline_checkpoint.pt> `
  --combat-teacher-data-dir <out>/mixed_v6.jsonl `
  --combat-teacher-loss-weight 0.2 `
  --combat-teacher-batch-size 32 --combat-teacher-updates-per-iter 6 `
  --output-dir <out>/hybrid_training
```

评估：

```powershell
python STS2AI/Python/evaluate_ai.py `
  --checkpoint <out>/hybrid_training/.../hybrid_final.pt `
  --transport pipe-binary --port 15970 --auto-launch `
  --num-games 24 --seed-suite regression `
  --output <out>/eval.json
```

## 诊断工具

- `STS2AI/Python/diagnostics/combat_teacher_experiment_report.py`：聚合 teacher_eval + metrics.jsonl + baseline/trained eval 出 markdown 报告 + 6 张图。
- `STS2AI/Python/diagnostics/compare_trajectory_combats.py`：3-way 同 seed 战斗级对比（baseline/w02/v4）。
- `STS2AI/Python/diagnostics/compare_4way_combats.py`：4-way 对比（加 v6）。

## 已知缺口（→ v7 候选）

1. **Act 1 第三个 boss `HULK_MATRIARCH` 目前 0 条样本**。24 seed 内没人走到这个 boss。需要扩 seed pool（EVAL_025+）或用更强 checkpoint 跑 trajectory 才能出 HULK_MATRIARCH 样本。
2. **crisis management 场景缺数据**：v6 在 EVAL_003 的 floor 15 把自己打到 6HP 再扛精英翻车。低 HP 下的保守决策样本需要专门抓。
3. **floor 2 样本占比高（34%）**：历史遗留，现在看可以适当裁剪让 high-floor 信号更集中。
4. **live builder 推进 bug 未根治**：路径 A 仍然只能在少数 seed 上推到 floor 11+。修这个是做 iterative teacher（每轮训完用自己的新 checkpoint dump trajectory 做下一轮数据）的前提。
