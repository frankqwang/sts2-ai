# STS2AI Stable Checkpoints (Act 1)

这个目录保存可以直接复现当前主线工作的稳定 checkpoint 包。

## 当前冠军（2026-04-16）

- **`planb_iter2303_selfplay_teacher.pt`**
  - role: **10-iter 冠军** (act1% 3.50%, boss_reach 56.02%，2500 ep apples-to-apples vs B.1)
  - frozen_at: `2026-04-16`
  - iteration: 2303
  - source_run: `STS2AI/Artifacts/hybrid_training_main_attention/20260415-195934_4env_planb_merged_teacher_5iter_resume2298/hybrid_02303.pt`
  - 训练链路: `hybrid_02293.pt` 起点 → Plan B 5 iter 到 02298 → resume 再 5 iter 到 02303
  - teacher: 2024 条合并 self-play (`from_hybrid_02293_merged_20260415/`, Plan B 实验)
  - config: `hybrid_train_ironclad_bigbatch_500ep_offline_noncombat_002_merged_teacher.toml`
  - 详细追溯: `STS2AI/docs/session_2026-04-15_skada_vs_selfplay_teacher.md`
  - deep-dive: `Artifacts/.../analysis/planb_vs_b1_build_deepdive_2026-04-15/`
  - 注意: boss_reach +5.78pp 的机制是"skip rate 17%→14%、多拿杂食卡磨过中段"，不是"选好卡"。ANGER(S) 在 clear 子集 -14.9%。WATERFALL boss 胜率仍 3% (act1% 瓶颈在 combat)

## 保守 baseline（2026-04-14）

- `mainline_iter2270_carddebug.pt`
  - role: 保守稳定起点 (frozen 前 act1% ≈ 2-3%)
  - frozen_at: `2026-04-14`
  - source_run: `STS2AI/Artifacts/hybrid_training_main_attention_iter5_carddebug_cont/hybrid_4env_20260414-094343`
  - note:
    - 已包含多进程 worker 权重同步修复
    - 已包含 Act1 路线规划、boss 条件化卡奖引导、combat safety rerank
    - 当前仓库代码还额外修了 Act1 boss -> Act2 过渡 bug；该修复会在下一次启动 host 后生效
    - 本场实验 (2026-04-15/16) 的 B.1 baseline 是从这里后续 23 iter 到 `hybrid_02293.pt` 起点再开始的

## 历史 Champion

- `retrieval_final_iter2175.pt`
  - role: 历史 champion / 检索头阶段基线
  - frozen_at: `2026-04-09`

## 推荐恢复命令（用冠军起点）

```powershell
python STS2AI/Python/train_hybrid.py `
  --config STS2AI/Python/configs/hybrid_train_ironclad_bigbatch_500ep_offline_noncombat_002_merged_teacher.toml `
  --output-dir STS2AI/Artifacts/hybrid_training_main_attention `
  --run-tag my_run_from_champion `
  --resume STS2AI/Assets/checkpoints/act1/planb_iter2303_selfplay_teacher.pt `
  --headless-dll STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe `
  --max-iterations 5 `
  --save-interval 1
```

## 推荐评估命令

```powershell
python STS2AI/Python/evaluate_ai.py `
  --checkpoint STS2AI/Assets/checkpoints/act1/planb_iter2303_selfplay_teacher.pt `
  --transport pipe-binary `
  --auto-launch `
  --headless-dll STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe `
  --num-games 50
```

## 保守起点的恢复命令（用 2270）

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

## 说明

- `manifest.json` 里保存了 checkpoint 的 SHA256、来源和推荐命令。
- 当前主线的详细交接、训练参数、分析脚本和下一步计划请看：
  - `STS2AI/docs/当前训练主线与接手说明_2026-04-14.md`
