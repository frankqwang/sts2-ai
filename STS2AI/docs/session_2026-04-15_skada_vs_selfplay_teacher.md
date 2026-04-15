# 2026-04-15/16 session 总结：skada 人类 teacher vs self-play teacher

接续 `session_2026-04-15_stability_patches.md` 和 `session_2026-04-15_deep_analysis.md`。
本 session 的主线实验：在 `hybrid_02293.pt` 同起点下，apples-to-apples 对比
**teacher data 来源**（self-play vs skada 真人）对 act1% 的影响。

## TL;DR

**teacher 来源 > teacher 数量**。相同起点下：
- 274 条 self-play teacher → act1% 2.90% (B.1 baseline, 10-iter)
- 2024 条 self-play teacher → act1% 3.40% (Plan B, 10-iter；+0.50pp，机制是"多拿杂食卡磨过中段"，ANGER 反降)
- **1830 条 skada 人类 teacher → act1% 3.84% (Plan A, 5-iter；+0.94pp，每 iter 稳 ≥4%)**

Plan A 5 iter 已超 Plan B 10 iter，**teacher source 是比数量更重要的变量**。

## 4 个实验

| # | 名称 | teacher | iter | act1% avg | boss_reach avg | 结论 |
|---|---|---|---|---|---|---|
| 1 | B.1 baseline | 274 self-play (02293) | 10 | 2.90% | 50.24% | 参照线 |
| 2 | Tier 1 unsafe mask | (同 B.1) | 5+5 | 2.40% / 2.00% | 42.40% / 38.20% | **负收益**（-8 ~ -12pp boss_reach），代码保留 flag 关 |
| 3 | Plan B | 274 + RANKB2 1750 (全 self-play) | 5+5 | 3.20% / 3.60% | 54.00% / 58.04% | **+0.50pp act1 / +5.78pp boss_reach**，但机制有问题（ANGER -14.9% 等） |
| 4 | **Plan A** | 1830 skada (纯真人) | 5 | **3.84%** | 54.56% | **当前最强**，每 iter 稳 4% |

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

### F3. skada 真人 teacher 消除了这个偏见

Plan A 5-iter 结果：
- act1% 每 iter 都是 4%（不像 B.1 震荡 1.6% ~ 4.8%）
- 信息密度更高：1830/1830 条 full spread（Plan B 有 14.1% zero-spread 被 filter）
- context_score 基于 skada 统计（0.7 × skada_score_norm + 0.3 × floor_val + synergy）

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

## 冠军 checkpoint

**Plan A `hybrid_02298.pt`**:
```
STS2AI/Artifacts/hybrid_training_main_attention/
  20260415-222149_4env_plana_pure_skada_5iter_resume2293/hybrid_final.pt
```

act1% 3.84% 5-iter avg，每 iter ≥4%。

## 下一步建议（ROI 排序）

1. **Plan A 跑完 10-iter** + deep-dive（正在跑，看 act1% 稳定性 + build 是否真变"精"）
2. **Plan A + combat 侧改进**：act1% 瓶颈在 combat，teacher 只改 noncombat；研究 combat teacher 扩充（当前只 259 条）或 MCTS 推理时叠加
3. **扩 skada 覆盖**：当前 soft-mode 只覆盖 card_reward，`run_floor_shop_actions` / `run_floor_relic_choices` / boss_best_cards 都还没用
4. **硬编码 prior**（之前讨论的方案 D）：直接读 skada `boss_best_cards` 改 `_BOSS_CARD_PREFS`，与 Plan A teacher loss 叠加不冲突

## 架构备忘（新人容易问的）

`ckpt['ppo_model']` 1.44M params = **noncombat head**（选卡/地图/商店/事件/休息）
`ckpt['mcts_model']` 1.70M params = **combat head**（出牌/目标/药水），历史命名，实际是 PPO 不是 MCTS
两者共享 `deck_encoder` / `symbolic_features_head` / 部分 `action_proj`（58 个同名 tensor）
总共 3.14M params，embed_dim=48, combat_hidden_dim=192。
