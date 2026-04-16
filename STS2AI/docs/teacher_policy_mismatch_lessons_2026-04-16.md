# Teacher-Policy Mismatch 教训总结（2026-04-16）

**读者**：未来想接 offline teacher loss 到 PPO policy 头的人。
**背景**：2026-04-15/16 session 三次 teacher 实验里有两次崩（Plan A 后期退化 / Plan C 单调下降），共同 root cause 是 **teacher 数据分布 ≠ 当前 policy 的 on-policy 分布**。这份 doc 锁教训，避免下次再踩。

## 症状模板

如果你看到以下信号**同时出现**，大概率是 teacher-policy mismatch：

1. **teacher loss 下降**（学进去了）但 **act1% / 主 metric 退化**
2. 学习曲线**单调向下**（不是 plateau，是跌）
3. 前 3-5 iter 指标还 OK，中后期突然崩
4. boss_reach / survival 指标比 act1% 先出问题
5. `combat_ppo_approx_kl` 经常 > `target_kl`，early_stop = 1.0 常态化

## 两个已知 failure case

### Case 1: Plan A 纯 skada teacher 后期退化

- Config: `hybrid_train_ironclad_bigbatch_500ep_offline_noncombat_002_skada_pure.toml`
- 起点: `hybrid_02293.pt`，teacher: 1830 条 skada 人类 card_reward 排序
- 前 5 iter act1% 3.84%（看似最强），后 5 iter 2.80%
- 现象: skada 把 build 推向**人类偏好**（ANGER-heavy, 强牌）但 combat head 还按老 02293 的策略打 → build 改变了但 combat 跟不上 → 后期 act1% 崩

### Case 2: Plan C combat_teacher 从冠军 resume 崩盘

- Config: `hybrid_train_ironclad_bigbatch_500ep_combat_teacher_2000balanced.toml`
- 起点: `planb_iter2303_selfplay_teacher.pt`（Plan B 10-iter 冠军 3.50%）
- Teacher: 2000_balanced combat teacher, **但 baseline 是 `retrieval_final_iter2175.pt` (2026-04-09 历史 champion)**
- 40 iter 后 act1% 从 3.50% → 1.14%（跌 67%）
- 现象: teacher 的"最优 action"是针对 2175 状态空间算的，02303 早已不在那些 state（Plan B 推出来的"多拿杂食卡"build 改了 state distribution）→ combat policy 被 teacher 硬拉去做 2175-era 的 correction → 跟当前 build 失配

## 根本原因

**On-policy RL 的 teacher 不能随便用离线数据**。teacher 信号要求：
- 对 agent 实际 rollout 时**会遇到的** state，提供监督信号
- 如果 teacher 的 state distribution 跟 agent 的 rollout distribution 不重合，强拉 agent 去学 teacher = 让 policy 飘离 on-policy 分布
- PPO 的 trust region（clip / KL 限制）会试图 pull back，但如果 teacher loss weight 大，直接把 trust region 撑爆

**具体的数据层错配**：
- skada 数据：来自**真人玩家**，胜率远高于 02293 agent → skada 看的 state 是"好 build 已经成型后"的，agent 还在早期 build 阶段根本到不了那些 state
- 2175 combat teacher：baseline agent 是 04-09 的版本，**不包含 Plan B 的"多拿杂食卡"build pattern** → teacher 样本集中在 2175 的 state，02303 rollout 不会访问

## 避坑规则

### 规则 1：teacher 来源必须跟 agent 同代

**不要** 用老 checkpoint 生成 teacher 喂新 agent。**要做** 的话：每次大幅更新 policy 都重新跑 `build_act1_combat_teacher_v2_dataset.py --combat-checkpoint <当前 agent>` 生成新的 teacher。

### 规则 2：loss_weight 起点要保守

- **noncombat ranking teacher**: 0.01-0.02 OK（本场实验过）
- **combat teacher**: **0.05 以下起步**（0.2 是本场证明会崩的值）
- 大于 PPO policy loss 一个量级的 teacher weight 容易盖过 on-policy gradient

### 规则 3：iter 数要控制

- teacher overfit 通常在 Q2-Q3（10-30 iter）开始显现
- **10 iter 是 teacher 实验的标准长度**
- 如果 10 iter 已经有退化信号（act1% 下降 2 iter 连续），**不要再 resume scale**

### 规则 4：监控指标

每 iter 看这些**早期退化信号**：

| 指标 | 警戒阈值 | 含义 |
|---|---|---|
| `combat_ppo_approx_kl` | > 2 × target_kl（即 > 0.04） | trust region 被 teacher 挤爆 |
| `combat_ppo_early_stop` | == 1.0 连续多 iter | PPO update 被截断 |
| `combat_teacher_ce` ↓ + act1% ↓ | 共现 | **teacher 学进去但学错**（典型 mismatch） |
| boss_reach 跟 act1% 同向跌 | 两个都跌 | 不只是 combat 问题，policy 整体飘走 |

### 规则 5：设计 apples-to-apples 对照

做 teacher 实验时：
1. 总有一个 **不含你测试 teacher** 的 baseline 起同 iter
2. 起点 checkpoint 对齐
3. 至少跑 10 iter（5 iter 容易被 seed 噪声干扰，参考 Plan A 前 5 vs 后 5 反转）

## 推荐模板（你下次起 teacher 实验的 checklist）

1. [ ] teacher 生成 baseline 是什么 agent？跟起点 diff 多少 iter？
2. [ ] teacher loss_weight 设多少？是不是从 0.01 开始 scan？
3. [ ] 准备好 apples-to-apples baseline 起同 iter 对比？
4. [ ] 先跑 10 iter，看 Q1 末（iter 5-10）有无退化信号
5. [ ] 如果 Q1 就崩，**立即停**，不要 resume

## 本场冠军保持

- `STS2AI/Assets/checkpoints/act1/planb_iter2303_selfplay_teacher.pt`
- Plan B 10-iter act1% 3.50%, boss_reach 56.02%
- **所有 teacher 实验失败都不改变冠军位置**

## 相关文档

- `session_2026-04-15_skada_vs_selfplay_teacher.md` — 3 次 teacher 实验完整对比
- `当前训练主线与接手说明_2026-04-14.md` §10.3 — 本场产物索引
- `Assets/checkpoints/act1/manifest.json` — champion + 6 步 SOP
- `Artifacts/hybrid_training_main_attention/.../analysis/planb_vs_b1_build_deepdive_2026-04-15/` — Plan B 深挖分析
