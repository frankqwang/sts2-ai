# STS2AI

杀戮尖塔 2 AI 训练项目。作为子目录放到反编译的游戏工程根目录下使用。

## 前置准备

### 1. 排除编译冲突

在游戏工程的 `sts2.csproj` 中 `<Project>` 下添加：

```xml
<ItemGroup>
  <Compile Remove="STS2AI\**" />
</ItemGroup>
```

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

从当前 champion checkpoint 继续训练：

```powershell
python STS2AI/Python/train_hybrid.py `
  --pipe --auto-launch `
  --headless-dll STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe `
  --num-envs 4 `
  --start-port 15527 `
  --max-iterations 500 `
  --skada-prior-weight 0.15 `
  --skada-boss-weights
```

参数说明：
- `--num-envs` 并行模拟器数量（推荐 4-8）
- `--max-iterations` 本次训练跑多少轮
- `--skada-prior-weight` Skada 社区数据混合权重（0=关闭，0.15=推荐）
- `--skada-boss-weights` 用 Skada Boss 团灭率缩放奖励

checkpoint 自动从 `STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt` 加载。
训练产物输出到 `STS2AI/Artifacts/hybrid_training/`。

## 评估

```powershell
python STS2AI/Python/evaluate_ai.py `
  --checkpoint STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt `
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

`STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt`

包含：
- PPO 非战斗脑（选卡/商店/路径/休息）
- 战斗脑（出牌/药水/目标）
- SymbolicFeaturesHead（符号特征交叉注意力）
