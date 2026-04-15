# 2026-04-15 session 稳定性补丁与 B.1 实验记录

日期：2026-04-15

本 session 主要做了两件事：

1. 给 `train_hybrid.py` 打三个稳定性补丁，把之前隐性的 KL 污染、内存泄漏、硬规则未生效问题都修了
2. 跑了一轮 `offline_noncombat_ranking` teacher 实验（B.1），验证补丁效果并拿到首批可用数据

## 1. 三个落代码的补丁

都已写进 `STS2AI/Python/train_hybrid.py`，注释也补齐。

### 1.1 iter 末尾显式释放 CUDA caching allocator

**位置**：`train_hybrid.py:main()` iter loop 末尾（metrics log 之后，health check 之前）。

**现象**：`episodes_per_iter=2000` 的 bigbatch 配置下，iter 2 开始主进程 RSS 从 876 MB 爬到 21 GB，GPU 显存占用 94%（15.3 / 16.3 GB），`ep/s` 从 4.2 跌到 1.2，再跑一轮就要 CUDA OOM。

**根因**：项目原来从不调用 `torch.cuda.empty_cache()` 和 `gc.collect()`。PPO update 阶段分配的大量 cuda tensor，Python 引用归零后 torch caching allocator 会保留在池子里等复用。bigbatch 下样本规模 × 张量形状分布剧烈扩大，池子持续膨胀直到打满显存。

**修法**：

```python
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**验证**：iter 2293 补丁版 `ep/s = 4.44`（比污染期 iter 2292 的 1.22 快 3.5 倍），GPU 显存回到 2.12 GB（-86%），主 RSS 回到 9.87 GB。

### 1.2 shop force_remove 硬规则的 schema 匹配修复

**位置**：`train_hybrid.py:_choose_shop_remove_purchase_action()`。

**现象**：文档里明确写了"shop 只要能删牌就强制删牌"，但实测 iter 2293 下 2380 次 shop session 里**只有 325 次（13.7%）真的触发 remove**，其余 86% 有 remove option 的机会被白白跳过。

**根因**：原代码找 `action.get("action") == "remove_card"`。但 binary 协议里所有 shop legal action 的 `action` 字段都是 `"shop_purchase"`，真正的类型区分在 `state.shop.items[index].category` 这个 metadata 里。判断字段错位，条件**永不命中**。13.7% 的 remove 是 PPO 自己随机撞出来的，不是硬规则。

**修法**：改为从 `state.shop.items` 找 `category == "remove_card"` 且 `can_afford && is_stocked` 的 item，拿到它的 `index`，再到 legal 里按 `action="shop_purchase"` + 同 index 匹配。代码里加了完整的 docstring 说明这段历史。

**验证**：iter 2294 修补版 2380 次 shop session 里 remove 触发 714 次（**115%**，因为同一 shop 有多轮 remove 机会），`deck_size_at_boss_mean` 从 17.5 降到 15.3，每 shop 平均多删 1.15 张基础牌。

### 1.3 PPO buffer-skip：force_remove 样本不进 PPO buffer

**位置**：`train_hybrid.py:collect_unified_episode()` 中 `_ppo_pending` 赋值处。

**现象**：修完 1.2 之后 force_remove 生效了，但 `ppo_approx_kl` 立刻飙到 0.3 ~ 0.87（target 是 0.02），`early_stop` 每 iter 都触发，`avg_floor` 和 `boss_reach_rate` 连续 3-4 iter 持续下跌，combat 行为也被牵连退化。

**根因**：force_remove 作为硬规则 override 了 PPO 决策，但这条样本仍然被写进 `ppo_buffer`，写时用 `log_prob=0.0` 作为 "old_policy 100% 选这个动作"的占位。实际 PPO actor 从没想过选 `remove_card`，真实 `log P_new(remove_card)` 很小（比如 -4.6）。update 时 ratio 和 approx_kl 单样本就被拉到天文数字：

```
ratio = P_new(remove) / P_old(remove) = 0.01 / 1.00 = 0.01
approx_kl = |log_prob_old - log_prob_new| = |0 - (-4.6)| = 4.6
```

平均进整个 iter 的 KL 就被这一类样本拉到 0.3+。然后 PPO 为了"对齐" force_remove 行为，把 `P_new(remove)` 拉高 —— 但 policy 参数是共享的，combat 和其他决策的概率分布一起被带歪，floor / boss% 一起掉。

换句话说，**硬规则执行 + 同时把样本交给 PPO 学，这两件事不能同时做**。

**修法**：在 `_ppo_pending` 赋值处分支，force_remove 触发时直接 `_ppo_pending = None`，后续 `if _ppo_pending is not None: ppo_buffer.add(...)` 会自动跳过。也就是让 force_remove 只产生环境 action，**完全不进 PPO 的训练信号**。

**验证**：iter 2294 补丁版 KL 从 0.87 降到 **0.006**（-99%），`early_stop` 从每 iter 触发改为偶发，`act1_clear_rate` 首次达到 3.8%（vs 基线 2.1% 接近翻倍）。

## 2. 本 session 的 B.1 实验：offline_noncombat_ranking + 上面三个补丁

### 2.1 数据生成

用 `generate_offline_noncombat_ranking_data.py` 从 `hybrid_02293.pt` 派生 teacher 数据：

- `--seed-prefix RANK --episodes 50 --num-envs 4`
- 耗时 61 分钟
- 产出 **274 ranking samples**（过滤 zero-spread 后 236 条可训练）
- 目录：`STS2AI/Artifacts/offline_noncombat_ranking/from_hybrid_02293_20260415-031531/`

### 2.2 训练配置

新建 toml：`STS2AI/Python/configs/hybrid_train_ironclad_bigbatch_500ep_offline_noncombat_002.toml`

关键参数（相对主线 toml 的差异）：

- `episodes_per_iter = 500`（小批快验证）
- `ppo_epochs = 1` / `combat_ppo_epochs = 1`（批量放大后配合小 epoch）
- `target_kl = 0.02` / `combat_target_kl = 0.02`
- `offline_noncombat_ranking_data_dir` 指向派生数据
- `offline_noncombat_ranking_loss_weight = 0.02`（reuse_rules 文档推荐第一档）
- `save_interval = 1`

### 2.3 10 iter 结果（iter 2294 - iter 2303）

对比基线主线 iter 2286-2290（5 iter 均）：

| 指标 | 基线均 | B.1 10-iter 均 | Δ |
|---|---|---|---|
| avg_floor | 14.12 | 13.87 | -0.25（噪声级） |
| boss_reach_rate | 53.0% | 49.9% | -3pp |
| **act1_clear_rate** | **2.12%** | **2.90%** | **+0.78pp / +37%** |
| boss_hp_fraction_dealt | 0.619 | 0.629 | +0.010 |
| deck_size_at_boss | 16.76 | 15.13 | -1.63 张 |
| card_reward_skip_rate | 24.9% | 19.2% | -5.7pp |
| PPO approx_kl | 0.0035 | 0.0073 | 2x 但安全 |
| PPO early_stop 触发比例 | 5/5 | 4/10 | 大幅改善 |

### 2.4 观察到的衰减模式

`skip%` 10 iter 轨迹：`17.1 → 14.6 → 16.9 → 18.4 → 18.9 → 23.1 → 23.6 → 22.3 → 18.4 → 18.7`。

前 2 iter 看到最强的"少 skip + act1% 峰值 4.8%"，之后 PPO 主体逐渐把 skip 拉回基线水平（24.9%）。同时 `offline_noncombat_ranking_loss` 从 0.04 降到 0.02，说明 **274 条 teacher 数据被学干了**，没有新的梯度信号。

诊断：teacher 数据只教 agent"拒绝坏牌"（SHRUG_IT_OFF、THUNDERCLAP、PERFECTED_STRIKE 等拿得少），但 A+ 强牌的 exposure 不够（ANGER 每 50 ep 大约遇到 2-3 次），所以 agent 没学到"必选强牌"。短期红利 + 中期均值回归的典型形态。

## 3. 新增 / 修改的文件

### 3.1 新增配置文件

- `STS2AI/Python/configs/hybrid_train_ironclad_bigbatch_2000ep.toml`
  - 2000 ep/iter bigbatch 主实验档，记录本 session 第一轮尝试
- `STS2AI/Python/configs/hybrid_train_ironclad_bigbatch_500ep_offline_noncombat_002.toml`
  - B.1 实验配置（500 ep + offline_noncombat_ranking 第一档）

### 3.2 修改的代码

- `STS2AI/Python/train_hybrid.py`
  - 顶部 `import gc`
  - iter 末尾 cleanup（见 1.1）
  - `_choose_shop_remove_purchase_action` 修复（见 1.2）
  - `_ppo_pending` force_remove 分支（见 1.3）

### 3.3 新增数据（不 push）

- `STS2AI/Artifacts/offline_noncombat_ranking/from_hybrid_02293_20260415-031531/`（274 samples）
- 多个训练 run 目录（hybrid_training_main_attention/20260415-*）

## 4. 当前 checkpoint 状态

- `hybrid_02293.pt` —— 纯净起点（force_remove 补丁前的最后一个正常 checkpoint）
- `hybrid_02303.pt` —— B.1 补丁版 10 iter 训完的末态，已吃到 offline teacher 信号

路径：
- `STS2AI/Artifacts/hybrid_training_main_attention/20260415-021853_4env_bigbatch_resume2292_18iter_plana_mempatch/hybrid_02293.pt`
- `STS2AI/Artifacts/hybrid_training_main_attention/20260415-043941_4env_b1_bufferskip_offline_noncombat_loss002_resume2293/hybrid_02303.pt`

## 5. 没做完的事 / 下一步候选

### 5.1 更多 teacher 数据

当前 274 条被学干了。RANKB2 大版 generator（300 ep）已经起在后台，预计产出 ~1650 条，完成后可以和旧 274 条 merge 跑 B.1v2。

### 5.2 skada 清洗的潜在改进（挖掘中发现）

`build_matchup_ranking_from_skada.py` 的 `_normalize_scores` 有两个保守点：

1. 只用 `context_score` 单字段，丢弃了 `win_rate_delta` / `pick_rate` / `hold_rate` 等硬信号
2. 强行保证 `chosen_index` 的 score > 其他候选 + 0.05，等价于"人类永远对"的硬假设

这两点可以分别改善（用 `0.6×context + 0.4×win_rate_delta`、过滤 low-ascension 玩家、去掉 chosen 强拉最高约束），代价是派生一份新数据集做 A/B。

### 5.3 其他硬规则的 buffer 污染风险

本 session 只修了 `force_remove` 这一处。代码里其他 deterministic override（`act1_route_plan_keep`、`shop_remove_target` 等）没挨个审过，它们如果也写 `log_prob=0` 就会有同样的 KL 污染问题。但目前数据没显示它们在出问题，暂缓。

### 5.4 card_reward 硬规则（方案 C）

挖掘发现 `ANGER` 是赢 boss 的关键分水岭（赢 0.92/ep vs 输 0.44/ep）。可以写一个类似 `force_remove` 的硬规则：`deck 里 ANGER < 某阈值 → 候选里有 ANGER 就强制选`。做法和 1.3 的补丁模式一样，`log_prob=None` + buffer-skip 就能避坑。

维护 tier S 表从 `STS2AI/Artifacts/skada/ironclad_card_reward_3000.jsonl` 的 `win_rate_delta` 挖。

## 6. 最重要的一句话（2026-04-15 更新）

2026-04-14 交接时写的"最该啃的是策略本体"仍然成立，但具体下钻后发现：**最该修的不是 PPO 本身，而是"硬规则 + PPO 学习"的耦合方式**。本 session 的三个补丁把这层解耦了，现在 offline teacher 能真正干净地训练 policy。下一步应该继续扩 teacher 数据 + 加更多瞄准 build 的硬规则（尤其是 card_reward keep），而不是在 PPO 超参上打转。
