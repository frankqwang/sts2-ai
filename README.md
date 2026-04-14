# STS2AI

杀戮尖塔 2 AI 训练项目。作为子目录放到反编译的游戏工程根目录下使用。

## 当前主线状态（2026-04-14）

当前主线已经切到 `main_attention + multi_process + no-MCTS` 路线，训练/诊断/日志链路都围绕这条线展开。最新接手说明放在：

- [STS2AI/docs/当前训练主线与接手说明_2026-04-14.md](STS2AI/docs/当前训练主线与接手说明_2026-04-14.md)

如果只需要一个可直接复现的恢复点，优先使用：

- `STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt`

当前推荐恢复命令见上面的接手说明文档；`README` 这里只保留入口和基础环境说明。

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
  ENV/           HeadlessSim、Spectator Mod 等 C# 代码
  Python/        训练、评估、数据工具
    core/        NN 模型、编码器、奖励塑形
    search/      MCTS、反事实评分、排名损失
    ipc/         模拟器通信（pipe/HTTP）
    skada/       Skada 社区数据加载
    data/        source_knowledge 知识库
    scripts/     启动脚本
```

## 当前 Checkpoint

当前推荐恢复点：

- `STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt`

历史 champion 仍保留在：

- `STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt`

包含：
- PPO 非战斗脑（选卡/商店/路径/休息）
- 战斗脑（出牌/药水/目标）
- SymbolicFeaturesHead（符号特征交叉注意力）
- main combat rollout `light_attention`

checkpoint 说明和 SHA256 见：

- `STS2AI/Assets/checkpoints/act1/manifest.json`
