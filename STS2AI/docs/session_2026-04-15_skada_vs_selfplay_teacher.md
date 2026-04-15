# 2026-04-15/16 session 总结：skada 人类 teacher vs self-play teacher

接续 `session_2026-04-15_stability_patches.md` 和 `session_2026-04-15_deep_analysis.md`。
本 session 的主线实验：在 `hybrid_02293.pt` 同起点下，apples-to-apples 对比
**teacher data 来源**（self-play vs skada 真人）对 act1% 的影响。

## TL;DR（2026-04-16 修订：10-iter 反转前述结论）

所有实验都从 `hybrid_02293.pt` resume、10-iter、500 ep/iter。

- **B.1 baseline**（274 self-play）: act1% 2.90%, boss_reach 50.24%
- **Plan B**（2024 self-play teacher, 274+RANKB2）: act1% **3.50%**, boss_reach **56.02%** 🏆
- **Plan A**（1830 skada 人类 teacher）: act1% 3.32%, boss_reach 54.04%

**Plan A 前 5 iter (2294-2298) act1% 3.84% 看似最强，但后 5 iter (2299-2303) 跌到 2.80%**（iter 2299→2.6% / 2300→2.2% / 回升）。10-iter 均值 3.32% 反而低于 Plan B 3.50%。

**修正结论**：
1. **5-iter 会被幸存者偏差误导**。Plan A 前 5 iter"每 iter 稳 ≥4%"是短期方差。
2. **Plan B 10-iter 最强**（就 act1% 看），虽然机制是"多拿杂食卡磨过中段"不是"选好卡"。
3. **skada 人类 teacher 有潜力但过 aggressive**：前期推 build 朝人类方向快速切换，但 combat policy 跟不上（teacher-policy mismatch），后期 act1% 塌回 B.1 水平。
4. **下一步候选**：loss_weight 0.02→0.01 降激进度；或加 warmup 让 agent 逐步吸收 skada 方向；或保留 Plan B 基线 + skada 作软 prior。

## 4 个实验

| # | 名称 | teacher | iter | act1% avg | boss_reach avg | 结论 |
|---|---|---|---|---|---|---|
| 1 | B.1 baseline | 274 self-play (02293) | 10 | 2.90% | 50.24% | 参照线 |
| 2 | Tier 1 unsafe mask | (同 B.1) | 5+5 | 2.40% / 2.00% | 42.40% / 38.20% | **负收益**（-8 ~ -12pp boss_reach），代码保留 flag 关 |
| 3 | **Plan B** | 274 + RANKB2 1750 (全 self-play) | 5+5 | **3.50%** (10-iter) | **56.02%** (10-iter) | **act1% 冠军**，但机制是"多拿杂食卡磨过中段"（ANGER -14.9% 等） |
| 4 | Plan A | 1830 skada (纯真人) | 5+5 | 3.32% (10-iter) <br>前 5: 3.84% / 后 5: 2.80% | 54.04% (10-iter) | **前期爆发后期退化**，5-iter 看起来最强是幸存者偏差 |

详细分析：
- `planb_vs_b1_5iter_2026-04-15/` — 6 PNG + summary
- `planb_vs_b1_build_deepdive_2026-04-15/` — 6 PNG + 12 维度挖掘
- （待补）Plan A 10-iter 完成后的 deep-dive

## 关键发现

### F1. Combat hard-safety mask 是负向的（Tier 1 失败）

R1 (hp≤sd mask 自损牌) + R2 (Phase 2 max_hp>10000 mask attack) + R5 (HEMOKINESIS 扩)。
两次 5-iter 测试都让 boss_reach 下降 -8 ~ -12pp，尝试用 buffer-skip fix 反而更差。
"KL 爆 early-stop" 假设被数据否定（B.1 本身 `combat_ppo_early_stop=1.0`）。
真因不明，候选：seed 过读 / R2 改 action 让 Waterfall Phase 2 trajectory off-policy。
代码保留，`_COMBAT_UNSAFE_MASK_ENABLED = False` 默认关。

### F2. Self-play teacher 放大 agent 已有偏见

Plan B 的 RANKB2 teacher 从 02293 自对弈生成 → 深挖发现：
- ANGER（S 牌）copies/clear run **-14.9%**（1.092 → 0.929）
- THUNDERCLAP（F 牌）**+15%**
- **WATERFALL_GIANT clear 率反降 -1pp**（所有 +9 clear 来自 SOUL_FYSH）
- smith 升级使用 **-11.5%**
- +5.78pp boss_reach 的真实机制是 skip rate 17.18% → 14.12%，**多拿杂食卡磨过中段**而不是"筛好卡"

### F3. skada 真人 teacher 前期爆发后期退化（teacher-policy mismatch）

Plan A 前 5 iter 看起来非常好：act1% 每 iter ≥3.6% 稳在 ~4%，显著优于 B.1 的 1.6~4.8% 震荡。信息密度也更高：1830/1830 条 full spread（Plan B 有 14.1% zero-spread 被 filter）。context_score 基于 skada 统计（0.7 × skada_score_norm + 0.3 × floor_val + synergy）。

**但后 5 iter（2299-2303）act1% 跌到 2.80%**（iter 2299 跌到 2.6%，iter 2300 最低 2.2%）。boss_reach 依然 53%（>B.1 50%），说明 **build 仍然"更会活"但 combat policy 跟不上**——skada 把 build 推向人类偏好方向，但 agent 的 combat 打法还是老的，teacher 推得越快偏差越大。这就是 **teacher-policy mismatch**。

Plan A 的 +0.94pp 优势在 10-iter 视角下缩到 **+0.42pp**（3.32% vs B.1 2.90%），反而被 Plan B 的 +0.60pp 超越。

### F4. act1% 瓶颈在 combat，不在 iter / 数据量

- B.1 10 iter 内部 act1% 无上升趋势（已饱和）
- Plan B 扩 teacher 7.6x 只换 +0.50pp → 量的边际极低
- **WATERFALL boss 胜率一直 3%**（随机水平）→ 战斗层才是真瓶颈

## 代码改动（push 了哪些 commit）

| commit | 内容 |
|---|---|
| `534480d` | sentinel feature + combat room_type weighting（前半 session） |
| `6a85916` | skada soft-mode 清洗 |
| `2db2aed` | §11 分析产物落盘规范 + session 深度分析 doc |
| `4b974a5` | 复现产物（02293/02303 ckpt + 274 teacher + softened bridge + requirements.txt） |
| `a29076f` | combat hard safety mask Tier 1（lab feature，默认关） |
| `e6b4505` | `merge_offline_noncombat_ranking.py` 合并工具 |
| 本 commit | plan B / plan A toml + 两份 analysis + 本 doc |

## 冠军 checkpoint（按 10-iter act1% 定）

**Plan B resume run 末态 `hybrid_02303.pt`**（act1% 3.50% 10-iter avg）：
```
STS2AI/Artifacts/hybrid_training_main_attention/
  20260415-195934_4env_planb_merged_teacher_5iter_resume2298/hybrid_02303.pt
```
（这是 Plan B 从 02293 跑 5 iter 到 02298，然后 resume 再跑 5 iter 到 02303 的最终态）

Plan A 末态 `hybrid_02303.pt` (act1% 3.32% 10-iter) 作为**对照组**参考，位置：
```
STS2AI/Artifacts/hybrid_training_main_attention/
  20260416-001940_4env_plana_pure_skada_5iter_resume2298/hybrid_02303.pt
```

## 下一步建议（ROI 排序，10-iter 数据定稿后）

1. **act1% 真瓶颈在 combat**：teacher（noncombat）已经试了 self-play 和 skada 两条路，增量都 +0.5pp 左右就饱和。WATERFALL boss 胜率 3% = 随机水平，说明 combat policy 根本没学会打 boss。下一步必须**动 combat 侧**。
2. **Combat-side 三个方向**：
   - a) 扩 combat_teacher 数据（当前只 259 条 `ironclad_act1_solver_v2_dataset_320.jsonl`）
   - b) MCTS 推理时叠加（业界 30%+ bot 都走这条路，项目代码里 `combat_mcts_backend` 结构已在但关着）
   - c) boss-specific combat teacher（针对 WATERFALL Phase 2、SOUL_FYSH 等单独出）
3. **skada 想继续用，考虑三个改法**：
   - 降 `offline_noncombat_ranking_loss_weight` 0.02 → 0.01（减 aggressive 度）
   - 加 warmup（前 N iter 不加 teacher loss，让 combat 先跟上再加）
   - 硬编码进 `_BOSS_CARD_PREFS`（直接读 skada `boss_best_cards` 表，与 teacher loss 可叠加）
4. **Plan B 10-iter 冠军继续扩**：从 `hybrid_02303.pt` Plan B 末态再跑 10 iter，看 boss_reach 是否继续爬。但 act1% 大概率依然卡在 combat 瓶颈。

## 架构备忘（新人容易问的）

`ckpt['ppo_model']` 1.44M params = **noncombat head**（选卡/地图/商店/事件/休息）
`ckpt['mcts_model']` 1.70M params = **combat head**（出牌/目标/药水），历史命名，实际是 PPO 不是 MCTS
两者共享 `deck_encoder` / `symbolic_features_head` / 部分 `action_proj`（58 个同名 tensor）
总共 3.14M params，embed_dim=48, combat_hidden_dim=192。
