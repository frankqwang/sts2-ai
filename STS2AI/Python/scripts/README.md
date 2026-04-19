# STS2AI 启动脚本（PowerShell Wrapper）

`STS2AI` 目录独立可移植到新反编译工程时，这些 wrapper 也跟着走。

> 训练主入口现在全部在 `networkV2/s6_training/` 下（V1 入口已下线）。
> 直接用 `python -m networkV2.s6_training.<entry>` 调就行,无需 wrapper。
> 本目录只保留那些"非 python 主训练"的工具 wrapper。

## 当前 wrapper

- `spectate.ps1`
  - 一键开 Godot 游戏窗口 + AI 实时操控 + overlay 观战
  - 依赖 Spectator Mod 和 V2 checkpoint
  - 录屏/日志落到 `STS2AI/Artifacts/recording/`
- `trainer_common.ps1`
  - 公共 PowerShell 函数（`Resolve-CommandOrPath` 等），被 `spectate.ps1` source

## V2 训练命令速查（不通过 wrapper）

```powershell
cd STS2AI/Python

# 整局训练
python -u -m networkV2.s6_training.train_full_run_v2 `
  --preset slim --num-workers 8 --max-iterations 200 `
  --dump-dir ../Artifacts/runs/<exp> `
  --output-dir ../Artifacts/checkpoints/<exp>

# 战斗专项训练（硬战斗）
python -u -m networkV2.s6_training.combat_cotrainer `
  --preset slim `
  --checkpoint ../Artifacts/checkpoints/<prev>/cotrainer_iter120.pt `
  --dump-dir ../Artifacts/runs/<exp> `
  --output-dir ../Artifacts/checkpoints/<exp>

# checkpoint 评测
python -m networkV2.s6_training.deck_eval_cli `
  --checkpoint ../Artifacts/checkpoints/<exp>/cotrainer_iter60.pt `
  --preset slim --n-trials 3
```

## 诊断工具

所有 `networkV2/s7_diagnostics/*.py` 诊断脚本都可 `python -m` 直接跑，默认输出到
`<dump_dir>/analysis/`，规范详见 `STS2AI/docs/design/DIAGNOSTICS_CONVENTION.md`。
