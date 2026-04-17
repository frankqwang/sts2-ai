# 训练产物 / 诊断产物目录规范

> 所有训练产物（rollout dump / checkpoint / 分析图表）**必须**统一落在
> `STS2AI/Artifacts/` 下。**不得**写到 `STS2AI/Python/runs/`、`STS2AI/Python/checkpoints/`、
> `/tmp/`、桌面、项目根、用户主目录。

---

## 1. 标准目录结构

项目根下**唯一**的训练产物目录就是 `STS2AI/Artifacts/`：

```
STS2AI/Artifacts/
├── runs/                                   # ★ rollout dump（训练过程产物）
│   └── <experiment>/
│       ├── run_meta.json                   # 训练开始时的超参 snapshot
│       ├── iter0001_samples.jsonl          # raw dump
│       ├── iter0001_metrics.json
│       ├── iter0001_advantages.npz
│       ├── iter0001_episodes.jsonl
│       ├── iter0001_trajectories.jsonl     # record_trajectory 开时
│       ├── ...
│       └── analysis/                       # ★ 所有派生产物
│           ├── win_rate_curves.png
│           ├── win_rate_breakdown.json
│           ├── policy_evolution.txt
│           ├── stuck_loop_report_iterNN.txt
│           ├── encounter_action_hist.png
│           ├── advantage_sign_over_time.png
│           ├── summary.md
│           └── ...
│
├── checkpoints/                            # ★ 模型权重（按 experiment 分组）
│   └── <experiment>/
│       ├── cotrainer_iter0020.pt
│       ├── cotrainer_iter0040.pt
│       └── ...
│
├── combat_teacher/                         # (既有) 战斗教师 replay
├── combat_trace/                           # (既有) combat trace
├── checkpoint_merges/                      # (既有) merge 输出
├── skada/                                  # (既有) human_victory_builds
├── combat_training/                        # (既有) v1 combat trainer 输出
├── eval/                                   # (既有) evaluation 产物
├── recording/                              # (既有) spectator 录制
├── verification/                           # (既有) 一致性审计
└── offline_data/                           # (既有) 离线数据集
```

**好处**：
- 一个 run 的所有信息都在一起（rollout + analysis）
- 训练产物和**源码**严格隔离：`STS2AI/Python/` 下**不得**有 `runs/` 或 `checkpoints/` 目录
- 打包/归档/分享时一起走
- 跨 run 对比 `diff Artifacts/runs/a/analysis/summary.md Artifacts/runs/b/analysis/summary.md` 很清爽

---

## 2. CLI 传参约定

### 2.1 训练命令

```bash
cd STS2AI/Python

# Combat 专项训练
python -u -m networkV2.s6_training.combat_cotrainer \
  --preset slim \
  --checkpoint ../Artifacts/checkpoints/co8/cotrainer_iter120.pt \
  --dump-dir ../Artifacts/runs/co13 \
  --output-dir ../Artifacts/checkpoints/co13 \
  ...
```

### 2.2 诊断工具

所有 `networkV2/s7_diagnostics/*.py` 工具：
- CLI 接受 `dump_dir` 作位置参数（要分析的 run 目录）
- `--out` 若未显式指定，**默认输出到 `<dump_dir>/analysis/<工具特征名>.{png|txt|json}`**
- 工具内部自动 `mkdir -p <dump_dir>/analysis/`
- 输出文件命名：`<主题>[_<scope>].<ext>`，如 `win_rate_curves.png`、`stuck_loops_iter0012.txt`

```bash
python -m networkV2.s7_diagnostics.live_monitor ../Artifacts/runs/co13 --once
python -m networkV2.s7_diagnostics.plot_win_rates ../Artifacts/runs/co13
python -m networkV2.s7_diagnostics.trajectory_analyzer ../Artifacts/runs/co13 --save
```

### 2.3 live_monitor 特殊情形

`live_monitor.py` 是**实时屏幕刷新**工具，不落盘 → 不受此约定约束。
但它的 `--once` 模式如果加 `--save-report` 选项，应写到 `<dump_dir>/analysis/live_snapshot_<timestamp>.txt`。

---

## 3. Do / Don't

| Don't | Do |
|-------|-----|
| 写 `STS2AI/Python/runs/co13/` | 写 `STS2AI/Artifacts/runs/co13/` |
| 写 `STS2AI/Python/checkpoints/co13/` | 写 `STS2AI/Artifacts/checkpoints/co13/` |
| 写 `/tmp/plot.png` | 写 `Artifacts/runs/co13/analysis/win_rate_curves.png` |
| 写桌面 / 用户主目录 | 同上 |
| 混进 `runs/exp/` 根目录和 raw dump 文件平级 | 放进 `analysis/` 子目录 |
| 每次手动传 `--out /tmp/xxx` | 默认即正确路径，不用显式传 |
| 不同工具写同名文件 | 每个工具有自己的 primary output 文件名 |

---

## 4. 何时允许 `/tmp`

只有"临时 debug、一次性、不需保存"的场景才写 `/tmp`：
- 训练日志（`/tmp/co13.log` 这种 nohup 的 stdout）✓
- ad-hoc 调试 `python -c '...'` 的小输出 ✓
- 分析 run 的产出 ✗ —— **必须** `STS2AI/Artifacts/runs/<exp>/analysis/`

---

## 5. 历史迁移

### 5.1 遗留位置（已淘汰）

- `STS2AI/Python/runs/` —— long1-long5 / co6-co12 旧 run 残留
- `STS2AI/Python/checkpoints/` —— 旧 checkpoint 残留

这两个目录**不再接受新产物**。迁移策略：
- **活动中 run**（如 co12 正在跑）：跑完后整体 `mv STS2AI/Python/runs/co12 STS2AI/Artifacts/runs/co12`
- **历史 run**：按需迁移，长期不用的可直接归档到 `STS2AI/Artifacts/runs_legacy/`

### 5.2 一次性迁移命令

```bash
# 所有历史 run 归档
mkdir -p STS2AI/Artifacts/runs_legacy STS2AI/Artifacts/checkpoints
mv STS2AI/Python/runs/* STS2AI/Artifacts/runs_legacy/  2>/dev/null
mv STS2AI/Python/checkpoints/* STS2AI/Artifacts/checkpoints/  2>/dev/null
rmdir STS2AI/Python/runs STS2AI/Python/checkpoints  2>/dev/null
```

迁移完后，文档所有路径按 `STS2AI/Artifacts/...` 统一。

### 5.3 临时 `/tmp` 产物清理

`/tmp/co6_wr.png`、`/tmp/curves.png` 这种遗留产物直接删除或迁移到正确位置。
