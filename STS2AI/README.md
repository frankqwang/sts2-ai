# STS2AI 使用说明

### first
sts2ai，本项目，放到反编译的项目根目录下。
原项目sts2.csproj 中Project下面加如下代码，让sts2排除编译本项目。
```


  <ItemGroup>
    <Compile Remove="STS2AI\**" />
  </ItemGroup>
```
然后就能开心的训练啦
### then

`STS2AI` 是当前主链的自包含工作目录。

目标不是单独脱离游戏工程运行，而是作为一整个子目录，复制到新的反编译工程根目录下继续开发。只要它位于新的工程根目录下，里面的 Python 和 PowerShell 入口就会自动按下面的结构找路径：

```text
<new-upstream-root>/
  project.godot
  src/
  MegaCrit/
  STS2AI/
```

也就是说：

1. 复制整个 `STS2AI/` 到新的反编译项目根目录。
2. 不要把它放到更深层子目录。
3. 训练、评估、录屏、审计都默认从工程根目录执行。

## 目录职责

```text
STS2AI/
  Assets/      稳定资产：checkpoint、dataset、seed
  Artifacts/   临时输出：训练结果、评估结果、轨迹、录屏、审计
  ENV/         pureSim host / overlay / spectator 相关 C#
  Python/      主链训练、评估、数据生成、审计、工具脚本
```

规则：

- 稳定可复用的内容放 `Assets/`
- 临时运行产物放 `Artifacts/`
- 不要再把旧根目录 `tools/python`、`checkpoints`、`datasets` 当主入口维护

## 先看哪几个入口

主链最常用的是这些：

- `STS2AI/Python/train_hybrid.py`
  - 主线训练入口
- `STS2AI/Python/evaluate_ai.py`
  - 主线评估入口
- `STS2AI/Python/verify_save_load.py`
  - save/load 验证入口
- `STS2AI/Python/test_training_smoke.py`
  - 轻量 smoke tests
- `STS2AI/Python/scripts/start-hybrid-training.ps1`
  - `train_hybrid.py` 的薄包装
- `STS2AI/Python/scripts/canonical-eval.ps1`
  - `evaluate_ai.py` 的薄包装
- `STS2AI/Python/scripts/run_full_run_recording.ps1`
  - 可见窗口 demo / 录屏入口
- `STS2AI/Python/scripts/run_sim_vs_godot_audit.ps1`
  - sim-vs-Godot 审计入口

## 当前主链资产

当前 Act 1 主链 champion：

- `STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt`

这一个 checkpoint 里已经同时包含：

- 非战斗 PPO 脑
- 战斗头
- retrieval head

默认配套数据：

- `STS2AI/Assets/datasets/card_ranking_post_wizardly`
- `STS2AI/Assets/datasets/combat_teacher_post_wizardly/teacher.jsonl`
- `STS2AI/Assets/seeds`

## 快速开始

### 1. Python smoke test

先确认 Python 入口没坏：

```powershell
python -m pytest STS2AI/Python/test_training_smoke.py -q
```

### 2. 构建 pureSim host

```powershell
dotnet build STS2AI/ENV/Sim/Host/headless_sim_host_0991.csproj -c Debug
```

如果默认输出目录被正在运行的 host 占用，可以改到临时目录验证：

```powershell
dotnet build STS2AI/ENV/Sim/Host/headless_sim_host_0991.csproj -c Debug -o STS2AI/Artifacts/temp_build/headless_host_verify
```

### 3. 跑一轮评估

```powershell
python STS2AI/Python/evaluate_ai.py `
  --checkpoint STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt `
  --transport pipe-binary `
  --seeds-file STS2AI/Assets/seeds/full_run_benchmark_seeds_200.json `
  --seed-suite benchmark `
  --num-games 50
```

或者：

```powershell
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/canonical-eval.ps1 `
  --checkpoint STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt
```

### 4. 从 champion 继续训练

```powershell
python STS2AI/Python/train_hybrid.py `
  --resume STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt `
  --retrieval-head `
  --retrieval-proj-dim 16 `
  --matchup-data-dir STS2AI/Assets/datasets/card_ranking_post_wizardly `
  --combat-teacher-data-dir STS2AI/Assets/datasets/combat_teacher_post_wizardly/teacher.jsonl `
  --pipe `
  --transport pipe-binary `
  --num-envs 4 `
  --start-port 17120 `
  --multi-process
```

或者：

```powershell
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/start-hybrid-training.ps1 `
  --resume STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt `
  --retrieval-head `
  --retrieval-proj-dim 16 `
  --matchup-data-dir STS2AI/Assets/datasets/card_ranking_post_wizardly `
  --combat-teacher-data-dir STS2AI/Assets/datasets/combat_teacher_post_wizardly/teacher.jsonl
```

## 录屏 / 演示

可见窗口录屏用：

```powershell
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/run_full_run_recording.ps1 `
  -StopExistingGodot `
  -Seed DEMO_VISIBLE_MASTER_001 `
  -Episodes 1 `
  -StepDelay 0.60 `
  -CombatDelay 0.25
```

默认行为：

- checkpoint 默认指向 `STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt`
- 输出写到 `STS2AI/Artifacts/recording`
- 会自动解析真实工程根目录，所以复制到新反编译工程后还能继续用

## 审计 / 回归

sim-vs-Godot 统一入口：

```powershell
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/run_sim_vs_godot_audit.ps1 -Mode Quick
```

完整模式：

```powershell
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/run_sim_vs_godot_audit.ps1 -Mode Both
```

输出默认写到：

- `STS2AI/Artifacts/verification`

## 提交前最低验证

任何训练链、编码器、状态构建、reward shaping、simulator、评估逻辑改动，在 push 前至少做：

1. `python -m pytest STS2AI/Python/test_training_smoke.py -q`
2. pureSim canary 或真机短训 `20-50 iter`
3. 检查 replay
4. 至少一轮 fixed-seed eval
5. 检查 entropy、warning、reward/card_reward 暴露

当前健康门槛：

- `ppo_ent > 0.3`
- `cbt_ent > 0.5`
- replay 以 `DEA` 为主
- `MAX / UNK / ERR` 必须能解释

## pureSim 约定

- 训练和评估都用 fresh ports + fresh hosts
- 不要复用刚跑完 eval 的 host 直接开训练
- Godot 主要作为确认 / 回归后端，不是默认短训主线

## Skada 数据怎么用

`Skada` 代码在：

- `STS2AI/Python/skada`

稳定数据在：

- `STS2AI/Assets/datasets/skada/skada_analytics.sqlite`

常用命令：

```powershell
python STS2AI/Python/skada/query_skada.py overview
```

```powershell
python STS2AI/Python/skada/query_skada.py card-tier IRONCLAD
```

如果要重新抓取覆盖本地库：

```powershell
python STS2AI/Python/skada/scrape_skada.py --skip-runs
```

## 复制到新工程后的检查清单

复制完 `STS2AI/` 到新的反编译工程根目录后，建议立刻跑：

```powershell
python -m pytest STS2AI/Python/test_training_smoke.py -q
```

```powershell
dotnet build STS2AI/ENV/Sim/Host/headless_sim_host_0991.csproj -c Debug
```

```powershell
python STS2AI/Python/evaluate_ai.py `
  --checkpoint STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt `
  --transport pipe-binary `
  --num-games 5
```

如果这三步都通了，说明新的工作区基本已经接上当前主链。
