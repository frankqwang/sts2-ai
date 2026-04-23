# Visible 观战模式使用指引

更新时间：2026-04-22

本文约束 `STS2AI` 当前这条可见观战链路的标准用法，目标是：

- 固定唯一入口，不再临时手搓启动方式
- 把训练 / visible / debug 的端口和职责分开
- 把这次已经踩过的坑写死，后面按清单排查

## 1. 结论先看

日常使用时，visible 观战只走：

- [spectate_zero_checkpoint.ps1](/C:/dev/sts2-ai/STS2AI/bridge/scripts/spectate_zero_checkpoint.ps1:1)

不要直接手工：

- 启 Godot
- 手工拷 mod
- 手工拼 Python spectator 参数
- 用全新空 `APPDATA` 直接试

上面这些动作只有在“排障模式”下才允许做。

## 2. 三条链路分别是什么

### 2.1 训练

- 入口：`python -m zero.replay.train`
- 环境：`HeadlessSim.dll`
- 默认端口：`15527`
- 用途：收集 episode、更新 checkpoint

### 2.2 Visible 观战

- 入口：[spectate_zero_checkpoint.ps1](/C:/dev/sts2-ai/STS2AI/bridge/scripts/spectate_zero_checkpoint.ps1:1)
- 环境：Godot + `sts2_mcp_spectator` mod
- 默认端口：`15526`
- 用途：加载一个已有 checkpoint，在可见战斗里逐步出牌

### 2.3 Visible 调试

- 只在排障时使用
- 建议改到 `15528+`
- 目的：和正式观战隔离，避免把正在看的实例打断

结论：

- 训练和 visible 可以并行
- 前提是端口错开
- 推荐固定约定：
  - `15526`：visible spectator
  - `15527`：HeadlessSim 训练
  - `15528+`：visible debug

## 3. 标准使用方式

### 3.1 先训练，拿到一个固定 checkpoint

建议只拿“已经训练完成的 checkpoint”去观战，不要直接盯着正在写入的最新文件。

原因：

- 正在训练的目录还会继续写 `active.json` 和新 checkpoint
- 观战更适合用一个冻结快照

### 3.2 用 wrapper 启动 visible

示例：

```powershell
cd C:\dev\sts2-ai\STS2AI

powershell -ExecutionPolicy Bypass -File .\bridge\scripts\spectate_zero_checkpoint.ps1 `
  -CheckpointPath C:\dev\sts2-ai\STS2AI\Artifacts\zero\manual_runs\visible_align_smoke_20260422_154500\04-22-11-49-targeted_cases_recurrent_gru_cases_1_eval_0_iters_4_seed_20260420\checkpoints\policy_v0004.pt `
  -ModelVariant recurrent_gru `
  -EncounterId CHOMPERS_NORMAL `
  -BuildFile C:\path\to\build.json `
  -GodotExe C:\dev\game\Godot_v4.5.1-stable_mono_win64\Godot_v4.5.1-stable_mono_win64.exe
```

运行后主要产物会落到：

- `STS2AI/Artifacts/recording/<run>/manifest.json`
- `STS2AI/Artifacts/recording/<run>/logs/spectate.stdout.log`
- `STS2AI/Artifacts/recording/<run>/logs/spectate.stderr.log`
- `STS2AI/Artifacts/recording/<run>/live_overlay.json`

### 3.3 现在脚本会自动做什么

这次之后，wrapper 启动前会自动把最新 spectator mod 同步到 Godot 实际加载的目录：

- 源目录：`STS2AI/ENV/SpectatorBridgeMod/bin/Debug/net9.0`
- 目标目录：`<GodotExe 同级目录>/mods/sts2_mcp_spectator`

同步规则：

- 会复制：
  - `sts2_mcp_spectator.dll`
  - `sts2_mcp_spectator.json`
  - `README.md`
- 会主动清理：
  - `sts2_mcp_spectator.deps.json`
  - `sts2_mcp_spectator.pdb`
  - `sts2_mcp_spectator.runtimeconfig.json`

原因见下面的坑点说明。

## 4. Build 文件要求

`--BuildFile` 对应的是战斗构筑 JSON。

要求：

- 编码必须是 `UTF-8 without BOM`
- 内容必须是合法 JSON

不要用会默认写 BOM 的方式直接输出文件，否则 Python 侧
[cli.py](/C:/dev/sts2-ai/STS2AI/bridge/game_bridge/spectate/cli.py:61)
这里会在 `json.loads(...)` 前就炸掉。

PowerShell 推荐写法：

```powershell
$json = Get-Content $src -Raw -Encoding utf8
[System.IO.File]::WriteAllText(
  $dst,
  $json,
  (New-Object System.Text.UTF8Encoding($false))
)
```

## 5. 本次已经确认的高频坑

### 5.1 改了 `SpectatorBridgeMod` 源码，但 Godot 还在吃旧 DLL

现象：

- 代码明明改了
- `dotnet build` 也过了
- visible 表现还是旧逻辑

根因：

- Godot 实际加载的是自己安装目录下 `mods/sts2_mcp_spectator/` 里的 DLL
- 不是工作区 `bin/Debug/net9.0/` 里的那份

现在的规避方式：

- wrapper 启动前自动同步

人工排查时重点看：

- Godot mod 目录里的 `sts2_mcp_spectator.dll` 时间戳

### 5.2 不要把 `sts2_mcp_spectator.deps.json` 拷进 mod 目录

现象：

- Godot 启动正常
- 但 spectator API 根本没起来

根因：

- 游戏 mod loader 会把 `*.json` 当 manifest 读
- `sts2_mcp_spectator.deps.json` 不是 mod manifest
- 它没有 `id` 字段，会导致：
  - `Mod manifest ... deps.json is missing the 'id' field`
  - 整个 mod 不加载

结论：

- mod 目录只放 `dll + manifest json`
- 不要额外塞 `.deps.json`

### 5.3 全新空 `APPDATA` 下，mod 可能被“mods warning”拦掉

现象：

- 日志里能看到 mod manifest 被发现
- 但又出现：

```text
Skipping loading mod sts2_mcp_spectator, user has not yet seen the mods warning
```

根因：

- 这是游戏首次加载 mod 的安全确认逻辑
- 如果直接拿一个全新的空 `APPDATA` 启 visible，mod 会被跳过

结论：

- 正式观战优先走 wrapper
- wrapper 会先从现有 editor 配置复制一份隔离数据根，再起新实例

排障时如果你手工起 Godot，也必须先准备好带已有设置的 `APPDATA`，不要空跑。

### 5.4 `SentryInit.gd` parse error 不是这条链路的主因

现象：

- stderr 里会有：
  - `SentryInit.gd` parse error
  - `OneTimeInitialization not declared`

当前结论：

- 这是噪声
- 它不自动等于 spectator mod 失败

真正该优先看的，是下面几类日志：

- 是否找到 `sts2_mcp_spectator.json`
- 是否出现 `Skipping loading mod ...`
- 是否出现 `server started on http://localhost:<port>/`

## 6. 推荐排障顺序

如果 visible 没起来，不要乱试，按下面顺序查。

### 第一步：看端口是不是对

正常约定：

- `15526`：visible
- `15527`：训练 HeadlessSim

如果你在 debug：

- 换到 `15528+`

### 第二步：看 Godot mod 目录里实际加载的 DLL

检查：

- `C:\dev\game\Godot_v4.5.1-stable_mono_win64\mods\sts2_mcp_spectator\sts2_mcp_spectator.dll`

确认：

- 时间戳是最新的
- 目录里没有 `.deps.json`

### 第三步：看 API 是否起来

检查：

```powershell
Invoke-WebRequest -UseBasicParsing `
  -Uri http://127.0.0.1:15526/api/v2/full_run_env/state
```

### 第四步：看日志里 mod 到底有没有加载

重点搜这些关键词：

- `Found mod manifest file`
- `Skipping loading mod`
- `STS2 MCP Spectator`
- `server started`

### 第五步：再查 reset 逻辑本身

只有前四步都正常后，才值得查：

- `encounter_id` 有没有透传
- reset 是不是走到了 single-combat 路径
- `/api/v2/full_run_env/reset` 为什么会 `500`

## 7. 并行使用建议

训练跑长任务时，可以并行修 / 看 visible，但遵守两条：

- 端口不要冲突
- 观战尽量用冻结 checkpoint，不要直接盯正在更新的文件

推荐做法：

1. 后台跑训练
2. 选一个旧 checkpoint 或已完成 run 的 checkpoint
3. 在 `15526` 开 visible
4. 如果需要排障，另起 `15528+` 的 debug 实例

## 8. 当前唯一推荐入口

### 单 checkpoint 可见观战

- [spectate_zero_checkpoint.ps1](/C:/dev/sts2-ai/STS2AI/bridge/scripts/spectate_zero_checkpoint.ps1:1)

### 混合 demo / overlay 录制

- [spectate.ps1](/C:/dev/sts2-ai/STS2AI/bridge/scripts/spectate.ps1:1)

### 公共 helper

- [trainer_common.ps1](/C:/dev/sts2-ai/STS2AI/bridge/scripts/trainer_common.ps1:1)

---

后续如果 visible 再出问题，先更新本文档，再改代码；不要先回到“手工启动 + 临时试端口 + 盲拷文件”的方式。
