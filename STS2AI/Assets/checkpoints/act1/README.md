# STS2AI Stable Checkpoints (Act 1)

这个目录保存可以直接复现当前主线工作的稳定 checkpoint 包。

## 当前可复现恢复点

- `mainline_iter2270_carddebug.pt`
  - role: 当前主线恢复点
  - frozen_at: `2026-04-14`
  - source_run: `STS2AI/Artifacts/hybrid_training_main_attention_iter5_carddebug_cont/hybrid_4env_20260414-094343`
  - note:
    - 已包含多进程 worker 权重同步修复
    - 已包含 Act1 路线规划、boss 条件化卡奖引导、combat safety rerank
    - 当前仓库代码还额外修了 Act1 boss -> Act2 过渡 bug；该修复会在下一次启动 host 后生效

## 历史 Champion

- `retrieval_final_iter2175.pt`
  - role: 历史 champion / 检索头阶段基线
  - frozen_at: `2026-04-09`

## 推荐恢复命令

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

## 推荐评估命令

```powershell
python STS2AI/Python/evaluate_ai.py `
  --checkpoint STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt `
  --transport pipe-binary `
  --auto-launch `
  --headless-dll STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe `
  --num-games 50
```

## 说明

- `manifest.json` 里保存了 checkpoint 的 SHA256、来源和推荐命令。
- 当前主线的详细交接、训练参数、分析脚本和下一步计划请看：
  - `STS2AI/docs/当前训练主线与接手说明_2026-04-14.md`
