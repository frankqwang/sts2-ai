# Combat-Only Sim 训练模式

本次新增的是一条独立于 full-run 的 `combat-only` 训练链，目标是：

- 直接指定 `build`（当前重点支持 `deck + relics`）
- 从 build 池中采样
- 按 `monster / elite / boss = 7 / 2 / 1` 这类权重采样战斗
- 用纯 simulator 反复重置并训练 combat policy/value 网络

## 设计原则

- **不复写战斗特征与模型**：继续复用 [combat_nn.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/core/combat_nn.py:162) 的 `build_combat_features(...)`、`build_combat_action_features(...)` 和 `CombatPolicyValueNetwork`
- **不另造 PPO 更新器**：继续复用 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:2350) 的 `CombatRolloutBuffer` 与 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:2490) 的 `CombatPPOTrainer`
- **不把 combat-only 逻辑塞回 full-run 状态机**：新入口单独放在 [train_combat_only.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_combat_only.py:1)

## 新增组件

- host pipe 新方法：
  - `combat_catalog`
  - `combat_reset`
  - `combat_state`
  - `combat_step`
  - 入口在 [Program.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Runtime/HeadlessSim/Program.cs:1160)
- Python env 适配层：
  - [combat_training_env.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/ipc/combat_training_env.py:1)
  - 负责：
    - 启动 `--combat-sim-server`
    - 读取 encounter catalog
    - 把 combat snapshot 适配成 combat NN 现有输入格式
    - 根据 hand/selection snapshot 生成 `legal_actions`
- 训练入口：
  - [train_combat_only.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_combat_only.py:1)

## 当前边界

- 已支持：
  - build 池 JSON
  - `deck + relics`
  - weighted encounter curriculum
  - combat PPO 更新
- 暂未开放：
  - 药水动作训练
    - 当前 combat snapshot 没有 potion 明细，所以第一版不把 `use_potion` 放进 legal actions
  - 可视化观战脚本
  - 多 env 并行 collector

## 最小命令

```powershell
python STS2AI/Python/train_combat_only.py `
  --builds-path STS2AI/Artifacts/skada/human_victory_builds_2026-04-14.json `
  --auto-launch `
  --episodes-per-iter 10 `
  --max-iterations 5 `
  --monster-weight 7 `
  --elite-weight 2 `
  --boss-weight 1
```

输出会写到：

- `STS2AI/Artifacts/combat_training/sim_build_curriculum_<timestamp>/run_manifest.json`
- `STS2AI/Artifacts/combat_training/sim_build_curriculum_<timestamp>/metrics.jsonl`
- `STS2AI/Artifacts/combat_training/sim_build_curriculum_<timestamp>/combat_only_last.pt`
