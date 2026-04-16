# STS2AI

杀戮尖塔 2 AI 训练项目。作为子目录放到反编译的游戏工程根目录下使用。

## 当前主线状态（2026-04-14）

当前主线已经切到 `main_attention + multi_process + no-MCTS` 路线，训练/诊断/日志链路都围绕这条线展开。最新接手说明放在：

- [STS2AI/docs/当前训练主线与接手说明_2026-04-14.md](STS2AI/docs/当前训练主线与接手说明_2026-04-14.md)
- [STS2AI/docs/review_summary_2026-04-14_teacherloop_phase1.md](STS2AI/docs/review_summary_2026-04-14_teacherloop_phase1.md)
- [STS2AI/docs/training_process_and_params_2026-04-14.md](STS2AI/docs/training_process_and_params_2026-04-14.md)

如果只需要一个可直接复现的恢复点，优先使用：

- **2026-04-16 当前 10-iter 冠军**（act1% 3.50%, boss_reach 56.02%）：
  `STS2AI/Assets/checkpoints/act1/planb_iter2303_selfplay_teacher.pt`
- 保守稳定起点（作为 fallback）：
  `STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt`（2026-04-14 frozen）

两者 SHA256、训练链路、关联 run dir / teacher data / config 都在
`STS2AI/Assets/checkpoints/act1/manifest.json` 里追溯。`README` 这里只保留入口。

## Python 代码目录结构

```
STS2AI/Python/
│
├── network/                      ← 【核心】网络架构 + 特征工程
│   ├── combat_network.py          战斗策略-价值网络 (CombatPolicyValueNetwork)
│   ├── fullrun_policy.py          全局策略网络 (FullRunPolicyNetworkV2 + PPOTrainer)
│   ├── shared_encoders.py         共享 NN 模块 (EntityEmbeddings, SetEncoder, BilinearActionScorer)
│   ├── combat_features.py         战斗状态/动作特征构建
│   └── state_features.py          全局状态特征构建 (StructuredState, build_structured_state)
│
├── env/                          ← 游戏环境接口
│   ├── full_run_env.py            HTTP/Pipe 两种后端的统一客户端
│   ├── full_run_backend.py        后端适配层（屏蔽 HTTP/Pipe 差异）
│   ├── binary_pipe_client.py      二进制管道通信（高性能）
│   ├── pipe_client.py             JSON 管道通信（调试用）
│   ├── headless_sim_runner.py     无头模拟器进程管理
│   ├── sim_host_lifecycle.py      多进程环境的进程池和端口分配
│   ├── inference_server.py        GPU 批量推理服务器
│   ├── action_semantics.py        动作语义和自动推进规则
│   └── ...                        combat_training_env, sts2_singleplayer_env 等
│
├── training/                     ← 训练/评估基础设施
│   ├── combat_ppo.py              战斗 PPO buffer + trainer + mcts_train_step
│   ├── combat_diagnostics.py      战斗诊断/trace/中文日志
│   ├── game_decisions.py          地图路线/卡牌奖励/商店决策逻辑
│   ├── eval_action_selection.py   推理动作选择（NN/Teacher/MCTS 路由）
│   ├── eval_game_state.py         游戏状态追踪/循环检测/自动推进
│   ├── combat_safety.py           战斗安全遮罩 (R1+R2 规则)
│   ├── training_health.py         训练异常检测（loss 爆炸、KL 过大等）
│   └── ...                        episode_data_saver, segment_collector, vectorized_collector 等
│
├── core/                         ← 基础工具
│   ├── vocab.py                   词表管理（卡牌/遗物/药水/怪物 ID ↔ 索引）
│   ├── rl_reward_shaping.py       奖励塑形（PBRS + 里程碑 + 战斗局部奖励）
│   ├── symbolic_features_head.py  符号特征头（sqlite cross-attention 零样本先验）
│   ├── card_tags.py               卡牌 32 维功能标签
│   ├── card_base_stats.py         卡牌伤害/格挡查找表
│   └── ...                        relic_tags, checkpoint_compat, full_run_agent 等
│
├── search/                       ← MCTS / 求解器 / Teacher
│   ├── mcts_core.py               蒙特卡洛树搜索核心
│   ├── combat_mcts_agent.py       战斗 MCTS 代理
│   ├── combat_turn_solver.py      回合穷举求解器
│   ├── train_combat_teacher.py    离线 Teacher 训练
│   └── ...                        turn_solver_planner, counterfactual_scoring 等
│
├── data/                         ← 数据
│   ├── vocab.json / card_tags.json / relic_tags.json  词表和标签数据
│   ├── skada/                     人类玩家数据采集/清洗/模型
│   └── ...                        source_knowledge, derived 等
│
├── configs/                      ← 训练配置 TOML
├── diagnostics/                  ← 分析和审计脚本
├── tools/                        ← 非主线工具（ONNX 导出、审计、演示等）
├── archive/                      ← 归档的旧实验代码
│
├── train_hybrid.py               ← 【主入口】统一训练循环
├── evaluate_ai.py                ← 【主入口】评估/benchmark
├── demo_play.py                  ← 【主入口】实时可视化演示
├── train_combat_only.py          ← 战斗专项训练
├── test_training_smoke.py        ← 回归测试（185 tests）
└── constants.py                  ← 全局路径常量
```

**规范：根目录只放主入口脚本和测试。所有新代码按职责放入对应子目录，不要在根目录写脚本。**

## 前置准备
所有环境、ai相关代码都在STS2AI里。
src下面以及最外层，都是反编译的源码，本项目中反编译部分只包含了游戏逻辑，sim模式只依赖这部分游戏逻辑源码,剔除了godot相关游戏资源。

### 1. 排除编译冲突
理论上如果你只跑sim的话，本项目已经完全包含了你需要的内容，如果你想完整跑godot进行一致性对比、观战等，需要把反编译的源码直接覆盖到当前仓库。然后执行当前步骤

在游戏工程的 `sts2.csproj` 中 `<Project>` 下添加：

```xml
<ItemGroup>
  <Compile Remove="STS2AI\**" />
</ItemGroup>
```

如果git有lf/crlf问题，直接用 git add --renormalize . 让 git 重新规范化索引，不需要重写文件。

### 2. 构建 HeadlessSim（无头模拟器）

```powershell
dotnet build STS2AI/ENV/Sim/Host/headless_sim_host_0991.csproj -c Debug
```

构建产物在 `STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe`。

### 3. 构建 Spectator Mod（观战用）

```powershell
dotnet build STS2AI/ENV/Spectator/SpectatorBridgeMod/sts2_mcp_spectator.csproj -c Debug
```

将产物复制到 Godot 引擎的 `mods/sts2_mcp_spectator/` 目录下。

### 4. Python 依赖

需要 Python 3.11+ 和 PyTorch：

```powershell
pip install torch numpy
```

### 5. Smoke Test

```powershell
python -m pytest STS2AI/Python/test_training_smoke.py -q
```

## 训练

从当前工作 checkpoint 继续训练：

```powershell
python STS2AI/Python/train_hybrid.py `
  --config STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml `
  --output-dir STS2AI/Artifacts/hybrid_training_main_attention `
  --run-tag acttransitionfix_resume2275 `
  --resume STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt `
  --max-iterations 5 `
  --save-interval 5 `
  --act1-no-elite-routes `
  --combat-pending-stall-threshold 30 `
  --boss-entry-quality-weight 0.15 `
  --boss-conditioned-card-guidance-weight 0.8 `
  --combat-safety-rerank-weight 1.0
```

参数说明：
- 默认环境参数仍来自 `STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml`
- 目前实验节奏用 `--max-iterations 5 --save-interval 5` 做短窗口复盘
- `--act1-no-elite-routes`、boss 条件化选卡、combat safety rerank 是当前主线的一部分
- `--output-dir` 现在只表示实验族根目录，例如 `STS2AI/Artifacts/hybrid_training_main_attention`
- `--run-tag` 用来放这次改造标签，例如 `acttransitionfix_resume2275`
- 每次实际 run 子目录统一为 `时间_环境数_标签`，例如 `20260414-110316_4env_acttransitionfix_resume2275`

训练产物默认输出到 `STS2AI/Artifacts/` 下对应 run 目录。当前主线的详细参数、输出目录和分析脚本请看接手说明。

## 评估

```powershell
python STS2AI/Python/evaluate_ai.py `
  --checkpoint STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt `
  --transport pipe-binary `
  --auto-launch `
  --headless-dll STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe `
  --num-games 50
```

## 观战（可见窗口）(依赖反编译源码，或者godot dll)

```powershell
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/spectate.ps1 `
  -StopExistingGodot `
  -Episodes 1 `
  -StepDelay 0.60 `
  -CombatDelay 0.25
```

- 自动启动 Godot 游戏窗口，AI 实时操控
- 窗口默认居中，存档自动隔离（不影响 Steam 存档）
- 右上角显示 AI 决策 overlay（需要 Spectator Mod）
- 输出写到 `STS2AI/Artifacts/recording/`

多实例观战用不同端口：

```powershell
# 实例 2（另一个终端，不加 -StopExistingGodot）
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/spectate.ps1 `
  -McpPort 15601 -Episodes 1 -StepDelay 0.60 -CombatDelay 0.25
```

## Skada 社区数据

Skada 提供 549 张卡牌评分、290 个遗物、卡牌协同、Boss 攻略等社区统计数据。

```powershell
# 查看总览
python STS2AI/Python/skada/query_skada.py overview

# 卡牌排名
python STS2AI/Python/skada/query_skada.py card-tier IRONCLAD

# 重新抓取
python STS2AI/Python/skada/scrape_skada.py --skip-runs
```

数据位于 `STS2AI/Assets/datasets/skada/skada_analytics.sqlite`，训练时通过 `--skada-prior-weight` 自动加载。

## 目录结构

```
STS2AI/
  Assets/        稳定资产：checkpoint、数据集
  Artifacts/     临时输出：训练结果、评估结果、录屏
  docs/          文档用中文。sts2ai/docs下面放文档，文档上面开头用2026-0416日期开头，好判断时效性
  ENV/           HeadlessSim、Spectator Mod 等 C# 代码
  Python/        训练、评估、数据工具
    core/        NN 模型、编码器、奖励塑形
    search/      MCTS、反事实评分、排名损失
    ipc/         模拟器通信（pipe/HTTP）
    skada/       Skada 社区数据加载
    data/        数据生成
    scripts/     启动脚本
```

## 当前 Checkpoint

当前推荐恢复点（按 act1% 排序，路径都在规范目录 `STS2AI/Assets/checkpoints/act1/`）：

1. **Plan B 10-iter 冠军**（2026-04-16，act1% 3.50%，boss_reach 56.02%）：
   `STS2AI/Assets/checkpoints/act1/planb_iter2303_selfplay_teacher.pt`
2. 保守 baseline（2026-04-14 frozen）：
   `STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt`

两者都包含：
- PPO 非战斗脑（选卡/商店/路径/休息）
- 战斗脑（出牌/药水/目标）
- SymbolicFeaturesHead（符号特征交叉注意力）
- main combat rollout `light_attention`

详细追溯见：
- `STS2AI/docs/session_2026-04-15_skada_vs_selfplay_teacher.md`（本场 teacher 实验完整结果）
- `STS2AI/Assets/checkpoints/act1/manifest.json`（SHA256 + 历史 champion）
