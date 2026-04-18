# networkV2 使用指南

> networkV2 是与 `network/` 并行的下一代网络架构实验线。三层时间尺度记忆 + 统一 combat/non-combat 路由 + 全监督多 head。
>
> 完整架构设计见 [docs/design/networkV2Final.md](design/networkV2Final.md)。
> 接手开发见 [docs/design/HANDOFF.md](design/HANDOFF.md)。

---

## 1. 环境准备

### 1.1 构建 bridge（C# sim 端）

networkV2 的 proto-pipe 通信需要 HeadlessSim。参考顶层 [README](../../README.md#2-构建-headlesssim无头模拟器)：

```powershell
dotnet build STS2AI/ENV/Sim/HeadlessSim/HeadlessSim.csproj -c Debug
```

产物：`STS2AI/ENV/Sim/HeadlessSim/bin/Debug/net9.0/HeadlessSim.exe`。

### 1.2 Python 依赖

```powershell
pip install torch numpy protobuf grpcio
```

### 1.3 proto 代码生成（如果改过 .proto）

```powershell
python -m grpc_tools.protoc `
  -I STS2AI/proto `
  --python_out=STS2AI/Python/networkV2/s0_bridge/generated `
  --grpc_python_out=STS2AI/Python/networkV2/s0_bridge/generated `
  STS2AI/proto/game_state.proto
```

---

## 2. 训练

networkV2 提供两个训练入口，分别对应两种训练目标：

| 入口 | 用途 | 环境 |
|---|---|---|
| `s6_training.train_combat_v2` | 纯战斗专项训练（沙盒 encounter 池）| `PipeBackedCombatTrainingClient` |
| `s6_training.train_full_run_v2` | 整局 full-run（战斗 + map + shop + rest + event）| `BinaryBackedFullRunClient` |

### 2.1 Combat-only 训练

```powershell
python -m networkV2.s6_training.train_combat_v2 `
  --builds STS2AI/Assets/builds/combat_sandbox_builds.json `
  --d-model 384 --n-heads 8 `
  --episodes-per-iter 20 `
  --max-iterations 500 `
  --monster-weight 1.0 --elite-weight 0.5 --boss-weight 0.1 `
  --output-dir STS2AI/Python/checkpoints/networkV2_combat
```

**Rollout 流程**：随机抽 build → 按 room 权重采 encounter → 跑一场战斗 → GAE → PPO 更新。

### 2.2 Full-run 训练（推荐）

```powershell
python -m networkV2.s6_training.train_full_run_v2 `
  --d-model 384 --n-heads 8 `
  --episodes-per-iter 10 `
  --max-iterations 500 `
  --num-workers 4 `
  --value-warmup-iters 3 `
  --target-kl 0.02 `
  --output-dir STS2AI/Python/checkpoints/networkV2_fullrun
```

**Rollout 流程**：`client.reset()` → 整局走到终局（victory/defeat）→ 混合 combat/non-combat 样本 → `UnifiedPPOTrainer` 按 domain 拆子批训练。

### 2.3 核心参数

**网络**
- `--d-model 384` —— Transformer 隐藏维度（小规模测试用 128）
- `--n-heads 8` —— attention 头数（必须能整除 d-model）
- `--n-build-slots 8` —— BuildMemory 的 slot 数
- `--max-numeric-dim 32` —— token numeric 向量上限（tokenizer 会截断）
- `--preset slim / full / tiny` —— 预设配置（空字符串用散参数）

**训练**
- `--lr 1e-4` —— PPO 学习率（比常见 3e-4 小，适配 slim 网络）
- `--ppo-epochs 4` —— 每轮 PPO 更新 epoch 数
- `--mini-batch-size 64` —— PPO minibatch 大小
- `--value-warmup-iters 3` —— 前 N 轮 policy_coef=0，让 value head 先分化
- `--target-kl 0.02` —— approx_kl 早停阈值；一个 epoch 内平均超过就终止剩余 epoch
- `--max-episode-steps 200 / --max-steps 800` —— 战斗/整局 step 上限

**采样**
- `--episodes-per-iter 10` —— 每 iter 采多少 episode
- `--min-update-samples 128` —— 样本数低于此跳过 PPO 更新
- `--num-workers 4` —— 并行 sim 数（需要匹配端口）
- `--seed 42` —— 随机种子

### 2.4 从 checkpoint 续训

```powershell
python -m networkV2.s6_training.train_full_run_v2 `
  --checkpoint STS2AI/Python/checkpoints/networkV2_fullrun/unified_v2_iter50.pt `
  --max-iterations 100
```

---

## 3. 日志指标解读

训练时每 iter 输出一行类似：

```
Iter  12W |  10 |  1234 | 3/7 | 28.5% |  6.23 | pl=0.01234 vl=0.543 hp=0.210 kl=0.0087 ep=4 | 18.3s
```

| 字段 | 含义 |
|---|---|
| `12W` | 第 12 轮；`W` = value warmup 中，空 = 正常 |
| `10` | 本轮 episode 数 |
| `1234` | 本轮总 step 数 |
| `3/7` | 胜/败数 |
| `28.5%` | 累计胜率 |
| `6.23` | 平均抵达楼层 |
| `pl` | policy loss（PPO clipped）|
| `vl` | value loss（fight_win/hp/survival/tempo 加权和）|
| `hp` | hp_loss head 单独损失 |
| `kl` | 本轮平均 approx_kl |
| `ep` | 实际完成的 PPO epoch 数（可能因 KL 早停 < ppo_epochs）|

**健康指标参考**
- `pl` 在 1e-5 ~ 1e-3：正常；长期 =0：advantage 退化（查 reward shaping）
- `vl` 稳定下降后趋于平稳：正常
- `kl` < target_kl：policy 更新幅度合理；频繁接近阈值说明 lr 或 clip_eps 太大
- `warmup` 阶段 `pl=0` 是预期行为

---

## 4. 诊断

### 4.1 Rollout dump

`networkV2/s7_diagnostics/rollout_dumper.py` 可以把每条样本的 banks + action + targets 保存成 jsonl，用于事后分析：

```python
from networkV2.s7_diagnostics.rollout_dumper import RolloutDumper
dumper = RolloutDumper("runs/dump_iter100.jsonl")
dumper.dump(samples)
```

### 4.2 分析脚本

- `s7_diagnostics/analyze_rollout.py` —— 统计 action 分布、value 误差等
- `s7_diagnostics/plot_training.py` —— 训练曲线可视化
- `s7_diagnostics/live_monitor.py` —— 实时监控训练 log

---

## 5. 代码目录（s0 → s7）

```
networkV2/
├── s0_bridge              sim proto-pipe 通信（ProtoPipeClient / generated/ pb2）
├── s1_schema              数据类：entities / actions / memory / token_banks / primitives
├── s2_config              encounter registry + mechanism 配置（act1 boss/elite）
├── s3_state_tracker       CombatStateTracker：跨步维护三层记忆
├── s4_compiler            obs → token banks
│   ├── runtime_compiler       玩家/敌人/手牌/牌堆
│   ├── action_compiler        legal_actions → ActionCandidate
│   ├── mechanism_compiler     boss 机制激活
│   ├── modifier_compiler      规则改写（buff/debuff 触发）
│   ├── memory_compiler        memory → numeric
│   ├── bank_assembler         组装成 UnifiedTokenBanks
│   ├── feature_compiler       顶层入口，按 domain 分派
│   └── noncombat/             shop / rest / event / map / card_reward
├── s5_net                 网络
│   ├── tokenizer              numeric + type_embed + time_scale_embed → d_model
│   ├── encoders/              board/build/mechanism/modifier/prefix/combat_memory
│   ├── action_contextualizer  战斗 action banks 上下文融合
│   ├── option_contextualizer  非战斗 option banks 上下文融合
│   ├── decision_core          共享 transformer encoder
│   ├── heads/                 policy / value / leaf_evaluator / run_evaluator
│   ├── combat_net             只战斗的小网络
│   └── unified_net            combat + non-combat 统一网络（训练用）
├── s6_training            训练
│   ├── batch                  TrainingSample / BatchedBanks / collate
│   ├── losses                 CombatLoss / NonCombatLoss
│   ├── ppo                    CombatPPOTrainerV2 / UnifiedPPOTrainer
│   ├── train_combat_v2        纯战斗训练入口
│   └── train_full_run_v2      整局训练入口
└── s7_diagnostics         诊断工具
```

---

## 6. 设计要点快览

- **三层时间尺度记忆**：TurnPrefix（本回合）/ CombatMemory（本战斗）/ RunBuildMemory（整局）
- **UnifiedPPOTrainer**：按 `decision_domain` 拆 combat/non-combat 子批路由，不混训
- **fight_win 防自蒸馏**：非终局用 GAE returns 做监督，终局用 0/1 硬标签
- **4 head 全监督**：leaf_evaluator 和 run_evaluator 所有 head 都接 loss，无饥饿 head
- **PlayedAction 效果差分**：tracker 接 prev/next 两帧算 damage/block/drawn/energy delta
- **Relic/Potion 语义**：`core/relic_rules.py` 静态规则表，覆盖 Ironclad 核心 + 通用 potion
- **behavior_history**：敌人 `next_move_id` 变化检测，代理 phase 切换
- **reshuffle_count**：pile 状态机检测洗牌事件

详见 [networkV2Final.md](design/networkV2Final.md)。

---

## 7. 常见问题

**Q: 训练时 `policy_loss` 长期为 0？**
A: 多半是 advantage 同质化。检查：① reward shaping 是否有效（非战斗步数太多 reward 几乎一致）；② advantage 是否被错误归一化（advantage 在 `train_step` 入口全局归一化，loss 内不再重算）；③ value warmup 是否还没结束。

**Q: `approx_kl` 爆炸？**
A: 降 lr 或 clip_eps；启用 `--target-kl`（默认 0.02）做早停。

**Q: sim 启动超时？**
A: 检查 `STS2AI/ENV/Sim/HeadlessSim/bin/Debug/net9.0/HeadlessSim.exe` 是否编译好；`BinaryBackedFullRunClient` 的 `auto_launch=True` 会自动起；也可以在独立终端起 sim 后用 `auto_launch=False`。

**Q: `max_numeric_dim` 是什么？**
A: Token numeric 向量最大长度（默认 32）。超出会被 tokenizer 截断。扩 schema 时要算好各 token 的 numeric 长度不超过这个值。

**Q: 与老 `network/` 架构的关系？**
A: 独立并行。老架构还在 main_attention 主线跑 checkpoint/evaluate；networkV2 作为实验线迭代。两边不共享 checkpoint，也不互相依赖代码。
