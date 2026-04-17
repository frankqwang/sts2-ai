# CLAUDE.md — STS2 AI 项目 Claude Code 指引

> 本文件会在 Claude Code 进入本项目时**自动加载**。用于强制工程规范。
> 中文对话（用户英文看不懂）。

---

## 🔴 硬性规范（违反即返工）

### 1. Schema / 特征工程必须**数据驱动**

**见 `STS2AI/docs/design/SCHEMA_CONVENTION.md`**

- ❌ 严禁根据 STS1 记忆或经验硬写 power/card/relic 名字（如 `metallicize`、`barricade`）
- ✅ 必须从 `STS2AI/Python/data/source_knowledge.sqlite` 提取
- ✅ 统一入口：`STS2AI/Python/networkV2/s1_schema/game_vocab.py`

加新游戏字段前**必须**：
```python
from networkV2.s1_schema.game_vocab import MONSTER_POWER_VOCAB
# 查名字是否真实存在：
assert "YourGuess" in MONSTER_POWER_VOCAB, "查 sqlite 确认实际命名"
```

**历史坑**：`metallicize` 是 STS1 名字，STS2 改叫 `PlatingPower`。硬编码导致 FROG_KNIGHT 15 层 plating 对网络完全不可见、13 iter 零胜率。

### 2. 训练产物 / 诊断产物目录

**见 `STS2AI/docs/design/DIAGNOSTICS_CONVENTION.md`**

- ❌ 不得写 `/tmp/`、桌面、项目根、`STS2AI/Python/runs/`、`STS2AI/Python/checkpoints/`
- ✅ **所有训练产物统一落在 `STS2AI/Artifacts/` 下**
  - rollout dump: `STS2AI/Artifacts/runs/<experiment>/iter*.jsonl+.npz+metrics`
  - checkpoint:  `STS2AI/Artifacts/checkpoints/<experiment>/cotrainer_iter*.pt`
  - 分析产物:    `STS2AI/Artifacts/runs/<experiment>/analysis/`
- ✅ `s7_diagnostics/*.py` 默认 `--out` 已设到 `<dump_dir>/analysis/`，直接跑即可

**历史坑**：`STS2AI/Python/runs/` 和 `STS2AI/Python/checkpoints/` 是 networkV2 早期遗留目录
（long1-long5 / co6-co12），会让训练产物**混进 Python 源码树**。新 run 必须用 `Artifacts/`。

### 3. 交接文档

- 交接文档统一写到 `STS2AI/docs/handoff/` 下
- 文件名格式：`handoff-日期-内容关键词.md`

### 4. 禁止破坏性操作

- 不私自 `git reset --hard` / `git push --force`
- 不删 `STS2AI/Artifacts/checkpoints/`、`STS2AI/Artifacts/runs/` 目录（历史实验要留）
- 不直接杀 training job 前先给用户报告

---

## 🗂️ 项目结构速查

```
STS2AI/
├── ENV/
│   ├── Sim/                 # 无头 C# sim（headless_sim_host_0991.exe）
│   └── Spectator/           # Godot 观战后端
├── Python/
│   ├── networkV2/           # V2 网络（当前主力）
│   │   ├── s1_schema/       # 数据结构（含 game_vocab.py）
│   │   ├── s2_config/       # mechanism_registry + auto_modifier_rules
│   │   ├── s3_state_tracker/
│   │   ├── s4_compiler/     # feature_compiler / bank_assembler
│   │   ├── s5_net/          # UnifiedNet
│   │   ├── s6_training/     # train_full_run_v2 / combat_cotrainer / deck_eval
│   │   └── s7_diagnostics/  # live_monitor / plot_win_rates / trajectory_analyzer
│   ├── core/                # 老 V1 基础设施（rl_reward_shaping 等）
│   ├── env/                 # bridge clients (BinaryBackedFullRunClient 等)
│   └── data/source_knowledge.sqlite  # 游戏真值（权威）
├── Artifacts/               # ★ 所有训练产物统一目录
│   ├── runs/<exp>/          # rollout dump（iter*_*.jsonl + analysis/）
│   ├── checkpoints/<exp>/   # 模型权重
│   ├── combat_teacher/      # 战斗教师 replay
│   └── ...                  # 其它历史产物（skada / eval / recording 等）
├── docs/design/             # 设计文档（含各种 CONVENTION.md）
├── docs/handoff/            # 交接文档（handoff-日期-关键词.md）
└── src/                     # 反编译游戏源码（只读参考）
```

---

## 🛠️ 常用命令

```bash
cd STS2AI/Python

# 长 run 训练
python -u -m networkV2.s6_training.train_full_run_v2 \
  --preset slim --num-workers 8 --max-iterations 200 \
  --dump-dir ../Artifacts/runs/<exp> \
  --output-dir ../Artifacts/checkpoints/<exp>

# Combat 专项训练（硬战斗）
python -u -m networkV2.s6_training.combat_cotrainer \
  --preset slim \
  --checkpoint ../Artifacts/checkpoints/<prev>/cotrainer_iter120.pt \
  --dump-dir ../Artifacts/runs/<exp> \
  --output-dir ../Artifacts/checkpoints/<exp>

# 监控
python -m networkV2.s7_diagnostics.live_monitor ../Artifacts/runs/<exp> --once
python -m networkV2.s7_diagnostics.plot_win_rates ../Artifacts/runs/<exp>   # 输出到 analysis/
python -m networkV2.s7_diagnostics.trajectory_analyzer ../Artifacts/runs/<exp> --save

# 评测 checkpoint
python -m networkV2.s6_training.deck_eval_cli \
  --checkpoint ../Artifacts/checkpoints/<exp>/cotrainer_iter60.pt \
  --preset slim --n-trials 3
```

---

## 🎯 当前状态（2026-04-17）

- 训练历程：long1-long5（full-run）+ co6（combat 专项）+ 正在重启 co7
- 主要 bug 历史：policy_loss=0 修、end_turn collapse 修、map 特征丢失修
- 当前瓶颈：power vocab 硬编码（schema bug）已修；FROG_KNIGHT 等 block-heavy encounter
- 下一步：co7（data-driven enemy token + balanced shaping + dynamic curriculum）

详见 `docs/design/HANDOFF.md`。
