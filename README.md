# STS2AI

杀戮尖塔 2 AI 训练项目。作为子目录放到反编译的游戏工程根目录下使用。

## 当前主线（networkV2）

当前唯一训练主线是 **networkV2** —— 分层 schema + 三层时间尺度 memory + combat/non-combat 统一路由。V1（`train_hybrid.py` / `main_attention` 系列）已于 commit `084fd20` 整体下线。

**核心特点**：
- 三层时间记忆：TurnPrefix（本回合）/ CombatMemory（本战斗）/ RunBuildMemory（整局）
- UnifiedPPOTrainer 按 `decision_domain` 拆子批路由，combat / non-combat 独立 loss
- 多 head 全监督（value / leaf_evaluator / run_evaluator 各 head），无饥饿 head
- 特征工程**数据驱动**：power/card/relic/monster vocab 从 `Python/data/source_knowledge.sqlite` 派生,统一入口 `networkV2/s1_schema/game_vocab.py`

## 项目结构

```
STS2AI/
├── Artifacts/                       临时输出：训练 run、checkpoint、评测、录屏、审计
│   ├── runs/<exp>/                  rollout dump + analysis/
│   └── checkpoints/<exp>/           模型权重
├── Assets/                          稳定资产
│   ├── builds/                      手工 / sandbox build 池
│   ├── checkpoints/                 promoted checkpoint
│   ├── datasets/                    离线数据集
│   └── seeds/                       评测种子
├── ENV/                             C# 侧
│   ├── Sim/                         HeadlessSim（无头模拟器，二进制 pipe 协议）
│   └── Spectator/                   Godot 观战 Mod
├── Python/                          训练 / 评估 / 数据
│   ├── networkV2/                   ★ 当前网络与训练主线
│   │   ├── s0_bridge/               sim pipe client / proto codec
│   │   ├── s1_schema/               数据结构（含 game_vocab.py）
│   │   ├── s2_config/               mechanism_registry + auto_modifier_rules
│   │   ├── s3_state_tracker/        状态追踪
│   │   ├── s4_compiler/             feature_compiler / bank_assembler
│   │   ├── s5_net/                  UnifiedNet
│   │   ├── s6_training/             train_full_run_v2 / combat_cotrainer / deck_eval
│   │   ├── s7_diagnostics/          live_monitor / plot_win_rates / trajectory_analyzer
│   │   └── s8_spectate/             V2 demo 播放
│   ├── core/                        V1 遗留基础设施（rl_reward_shaping 等,仍被部分模块复用）
│   ├── data/                        数据
│   │   └── source_knowledge.sqlite  ★ 游戏真值 snapshot（权威）
│   ├── env/                         环境接口（full_run_env, combat_training_env, ...）
│   ├── configs/                     训练 TOML
│   ├── diagnostics/                 跨 V1/V2 审计脚本
│   ├── scripts/                     PowerShell wrapper（spectate / sim_vs_godot_audit）
│   └── tests/                       测试
├── docs/
│   ├── design/                      架构规范（CONVENTION / HANDOFF / networkV2Final ...）
│   └── handoff/                     交接文档（handoff-日期-关键词.md）
└── src/                             反编译游戏源码（只读参考）
```

## 前置准备

所有环境、AI 相关代码都在 `STS2AI/` 里。`src/` 以及最外层是反编译源码，项目只依赖其中游戏逻辑部分（剔除 Godot 资源）。

### 1. 排除编译冲突

理论上如果只跑 sim，项目已自包含。若要跑 Godot 做一致性对比或观战，需要把反编译源码覆盖到本仓库，然后在 `sts2.csproj` 的 `<Project>` 下添加：

```xml
<ItemGroup>
  <Compile Remove="STS2AI\**" />
</ItemGroup>
```

如果 git 有 lf/crlf 问题，`git add --renormalize .` 让 git 重新规范化索引。

### 2. 构建 HeadlessSim

```powershell
dotnet build STS2AI/ENV/Sim/Host/headless_sim_host_0991.csproj -c Debug
```

产物：`STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe`。

### 3. 构建 Spectator Mod（观战用）

```powershell
dotnet build STS2AI/ENV/Spectator/SpectatorBridgeMod/sts2_mcp_spectator.csproj -c Debug
```

将产物复制到 Godot 的 `mods/sts2_mcp_spectator/`。

### 4. Python 依赖

Python 3.11+：

```powershell
pip install -r STS2AI/Python/requirements.txt
```

## 训练

### 整局训练（长 run）

```powershell
cd STS2AI/Python
python -u -m networkV2.s6_training.train_full_run_v2 `
  --preset slim --num-workers 8 --max-iterations 200 `
  --dump-dir ../Artifacts/runs/<exp> `
  --output-dir ../Artifacts/checkpoints/<exp>
```

### 战斗专项（从 full-run checkpoint resume）

```powershell
cd STS2AI/Python
python -u -m networkV2.s6_training.combat_cotrainer `
  --preset slim `
  --checkpoint ../Artifacts/checkpoints/<prev>/cotrainer_iter120.pt `
  --dump-dir ../Artifacts/runs/<exp> `
  --output-dir ../Artifacts/checkpoints/<exp>
```

训练产物统一写 `STS2AI/Artifacts/` 下,规范见 [DIAGNOSTICS_CONVENTION.md](STS2AI/docs/design/DIAGNOSTICS_CONVENTION.md)。

## 监控与诊断

```powershell
cd STS2AI/Python
python -m networkV2.s7_diagnostics.live_monitor ../Artifacts/runs/<exp> --once
python -m networkV2.s7_diagnostics.plot_win_rates ../Artifacts/runs/<exp>
python -m networkV2.s7_diagnostics.trajectory_analyzer ../Artifacts/runs/<exp> --save
```

所有 `s7_diagnostics/*.py` 默认输出到 `<dump_dir>/analysis/`。

## 评测

```powershell
cd STS2AI/Python
python -m networkV2.s6_training.deck_eval_cli `
  --checkpoint ../Artifacts/checkpoints/<exp>/cotrainer_iter60.pt `
  --preset slim --n-trials 3
```

## 观战（可见窗口，依赖反编译源码 / Godot）

```powershell
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/spectate.ps1 `
  -StopExistingGodot `
  -Episodes 1 `
  -StepDelay 0.60 `
  -CombatDelay 0.25
```

- 自动启动 Godot + AI 实时操控
- 窗口默认居中,存档自动隔离（不影响 Steam 存档）
- 右上角显示 AI 决策 overlay（需要 Spectator Mod）
- 输出写到 `STS2AI/Artifacts/recording/`

## 文档入口

| 文档 | 用途 |
|---|---|
| [docs/design/networkV2Final.md](STS2AI/docs/design/networkV2Final.md) | 架构设计：数据流、schema、token bank、网络层次 |
| [docs/design/HANDOFF.md](STS2AI/docs/design/HANDOFF.md) | 接手指引：项目状态快照、已知问题、下一步计划 |
| [docs/design/SCHEMA_CONVENTION.md](STS2AI/docs/design/SCHEMA_CONVENTION.md) | schema / vocab 数据驱动规范（严禁硬编码卡名/power 名） |
| [docs/design/DIAGNOSTICS_CONVENTION.md](STS2AI/docs/design/DIAGNOSTICS_CONVENTION.md) | 训练产物 / 诊断产物目录规范 |
| [docs/design/nonCombat.md](STS2AI/docs/design/nonCombat.md) | 非战斗 domain（shop/rest/event/map）的特征/网络设计 |
| [docs/design/proto_bridge_usage.md](STS2AI/docs/design/proto_bridge_usage.md) | proto-pipe 通信协议与 bridge 用法 |
| [docs/networkV2_guide.md](STS2AI/docs/networkV2_guide.md) | V2 使用指南：训练命令、参数、日志解读、FAQ |
| [docs/网络与训练概览.md](STS2AI/docs/网络与训练概览.md) | 现网架构与训练流程中文速览 |
