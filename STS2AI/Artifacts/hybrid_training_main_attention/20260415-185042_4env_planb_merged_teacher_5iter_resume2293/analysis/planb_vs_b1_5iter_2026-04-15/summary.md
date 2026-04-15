# Plan B (7.6x teacher) vs B.1 baseline, 5-iter (2294-2298) 对比分析

- Run dirs: B.1 = `20260415-043941_4env_b1_bufferskip_offline_noncombat_loss002_resume2293`, Plan B = `20260415-185042_4env_planb_merged_teacher_5iter_resume2293`
- Replay scope: iter 2294-2298, B.1 episodes=2500, Plan B episodes=2500

## TL;DR

**+3.56pp boss_reach 的机制是"在本该 skip 的 floor 拿了一张平庸卡"，不是"挑更好的卡"**。Plan B 在 floor 6-14（尤其 floor 13 -9.1pp / floor 14 -6.5pp）skip rate 显著下降，每局 cards_taken 多 0.25 张，boss deck +0.33，但 HP 到 boss 时相同（62.4/67），额外那张卡把他们带到 boss（+89 次）却没改善 boss 胜率（clear 仅 +0.36pp）。ANGER 张数 -0.103 是反例，THUNDERCLAP（tier-F）反而 +0.031 — teacher 数据扩展**没**产生"挑 S-tier"效应。

## Key Metrics (5-iter mean)

| 指标 | B.1 | Plan B | Δ |
|---|---|---|---|
| act1_clear_rate | 0.0304 | 0.0340 | +0.0036 |
| boss_reach_rate | 0.5044 | 0.5400 | +0.0356 |
| avg_floor | 13.9376 | 14.1452 | +0.2076 |
| offline_noncombat_ranking_loss | 0.0341 | 0.0328 | -0.0013 |
| combat_teacher_ce | 0.9319 | 0.9582 | +0.0263 |
| combat_teacher_rank | 0.0591 | 0.0537 | -0.0055 |
| combat_ppo_approx_kl | 0.0897 | 0.0880 | -0.0017 |
| combat_ppo_clip_fraction | 0.2473 | 0.2477 | +0.0004 |
| ppo_approx_kl | 0.0068 | 0.0073 | +0.0005 |
| card_reward_skip_rate | 0.1718 | 0.1417 | -0.0301 |
| deck_size_at_boss_mean | 15.1300 | 15.4420 | +0.3120 |
| boss_hp_fraction_dealt_mean | 0.6348 | 0.6421 | +0.0073 |

## Summary.json level (实际 replay 统计)

| | B.1 | Plan B |
|---|---|---|
| total_episodes | 2500 | 2500 |
| boss_reached | 1261 (50.44%) | 1350 (54.00%) |
| act1_cleared | 76 (3.04%) | 85 (3.40%) |
| card_reward skip rate | 17.18% | 14.12% |
| avg cards_taken / ep | 4.63 | 4.88 |
| avg cards_taken (boss runs) | 5.49 | 5.74 |
| avg boss deck size | 15.18 | 15.51 |
| avg combat steps / ep | 146.57 | 151.30 |

## Key finding 1: Build 选择 — FOCUS 卡每局 boss 到达时的平均张数

| 卡 | B.1 | Plan B | Δ |
|---|---|---|---|
| SWORD_BOOMERANG | 0.148 | 0.223 | +0.075 |
| BLOODLETTING | 0.280 | 0.318 | +0.038 |
| BODY_SLAM | 0.102 | 0.133 | +0.032 |
| THUNDERCLAP | 0.199 | 0.230 | +0.031 |
| HEADBUTT | 0.151 | 0.176 | +0.026 |
| SECOND_WIND | 0.017 | 0.033 | +0.017 |
| HEMOKINESIS | 0.077 | 0.092 | +0.015 |
| HAVOC | 0.091 | 0.103 | +0.012 |
| SHRUG_IT_OFF | 0.109 | 0.115 | +0.005 |
| LIMIT_BREAK | 0.000 | 0.000 | +0.000 |
| CLASH | 0.000 | 0.000 | +0.000 |
| CARNAGE | 0.000 | 0.000 | +0.000 |
| FLEX | 0.000 | 0.000 | +0.000 |
| WARCRY | 0.000 | 0.000 | +0.000 |
| PUMMEL | 0.000 | 0.000 | +0.000 |
| CRUELTY | 0.012 | 0.011 | -0.001 |
| DEMON_FORM | 0.008 | 0.004 | -0.003 |
| POMMEL_STRIKE | 0.150 | 0.144 | -0.006 |
| OFFERING | 0.023 | 0.012 | -0.011 |
| PERFECTED_STRIKE | 0.095 | 0.079 | -0.017 |
| ANGER | 0.586 | 0.483 | -0.103 |

## Key finding 2: Combat-layer 行为

- P1a (hp<=3 BLOODLETTING 打出): B.1 61/297 (20.5%), Plan B 67/311 (21.5%)
- P3a (Phase-2 对面, 选攻击): B.1 461/869 (53.0%), Plan B 448/863 (51.9%)
- Total combat steps: B.1 366419, Plan B 378256

## Key finding 3: Death floor 分布

| floor | B.1 deaths | B.1 % | Plan B deaths | Plan B % | Δ% |
|---|---|---|---|---|---|
| 2 | 1 | 0.04% | 1 | 0.04% | +0.00% |
| 3 | 0 | 0.00% | 1 | 0.04% | +0.04% |
| 4 | 3 | 0.12% | 0 | 0.00% | -0.12% |
| 5 | 26 | 1.04% | 24 | 0.96% | -0.08% |
| 6 | 66 | 2.64% | 62 | 2.48% | -0.16% |
| 7 | 131 | 5.24% | 127 | 5.08% | -0.16% |
| 8 | 166 | 6.64% | 152 | 6.08% | -0.56% |
| 9 | 150 | 6.00% | 131 | 5.24% | -0.76% |
| 11 | 128 | 5.12% | 141 | 5.64% | +0.52% |
| 12 | 135 | 5.40% | 117 | 4.68% | -0.72% |
| 13 | 119 | 4.76% | 106 | 4.24% | -0.52% |
| 14 | 155 | 6.20% | 140 | 5.60% | -0.60% |
| 15 | 159 | 6.36% | 148 | 5.92% | -0.44% |
| 17 | 1185 | 47.40% | 1265 | 50.60% | +3.20% |
| 18 | 51 | 2.04% | 66 | 2.64% | +0.60% |
| 19 | 10 | 0.40% | 9 | 0.36% | -0.04% |
| 20 | 5 | 0.20% | 6 | 0.24% | +0.04% |
| 21 | 2 | 0.08% | 1 | 0.04% | -0.04% |
| 22 | 2 | 0.08% | 2 | 0.08% | +0.00% |
| 23 | 2 | 0.08% | 0 | 0.00% | -0.08% |
| 24 | 2 | 0.08% | 0 | 0.00% | -0.08% |
| 27 | 1 | 0.04% | 0 | 0.00% | -0.04% |

## 反证 (Plan B 更差的地方)

- floor 17: B.1 47.40% → Plan B 50.60% (Δ +3.20pp)
- tier-F 卡 THUNDERCLAP: Plan B 选的更多 (B.1 0.199 → Plan B 0.230, Δ +0.031)
- Plan B skip rate 反而更低 (14.12% vs B.1 17.18%), 这与 teacher 教 skip 反例不一致。

## 关键补充图: floor-level skip rate (`floor_breakdown.png`)

| floor | B.1 skip/total | Plan B skip/total | Δpp |
|---|---|---|---|
| 2 | 323/2500 (12.9%) | 307/2501 (12.3%) | -0.6 |
| 3 | 161/1012 (15.9%) | 138/1025 (13.5%) | -2.4 |
| 6 | 169/967 (17.5%) | 150/953 (15.7%) | -1.7 |
| 7 | 195/1055 (18.5%) | 152/1097 (13.9%) | **-4.6** |
| 8 | 180/1002 (18.0%) | 138/997 (13.8%) | **-4.1** |
| 9 | 150/833 (18.0%) | 124/831 (14.9%) | -3.1 |
| 11 | 131/719 (18.2%) | 105/716 (14.7%) | -3.6 |
| 12 | 171/825 (20.7%) | 121/813 (14.9%) | **-5.8** |
| 13 | 162/688 (23.5%) | 104/719 (14.5%) | **-9.1** |
| 14 | 205/880 (23.3%) | 164/976 (16.8%) | **-6.5** |
| 15 | 135/776 (17.4%) | 115/817 (14.1%) | -3.3 |

HP at boss arrival: B.1 mean 62.37, Plan B mean 62.48 (几乎相同) — Plan B 不是"拿卡换 HP 收益"，而是纯多一张卡的问题。

## 结论 & 待验证

1. **+3.56pp boss_reach 的机制**：floor 6-14 skip rate 下降（尤其 13-14 的 -6/-9pp）→ 每局多 0.25 张卡 → 过 elite/burning 更稳，但 boss 打不穿。
2. **没有"挑 S-tier"效应**：ANGER copies/run -0.103 是最强反向信号；正向 delta 最多的是 SWORD_BOOMERANG / BODY_SLAM / BLOODLETTING，连 tier-F 的 THUNDERCLAP 都 +0.031。
3. **Combat policy 未被 teacher 改变**：P1a 21% vs 20.5%、P3a 52% vs 53%、combat_teacher_ce 反升 +0.026，证实 +3.56pp 全来自 noncombat (选卡) head。
4. **反证 — floor 17 死亡率 +3.20pp 是分母效应**：到 boss 的 run 中，boss 胜率 B.1 6.03% vs Plan B 6.30%，略好但接近噪声。
5. **Boss-reach iter-by-iter 方差大**：B.1 [0.442,0.498,0.528,0.498,0.556] vs Plan B [0.48,0.478,0.542,0.622,0.578]，±0.07 量级，+0.036 mean delta 不算稳健，**需重复 run + bootstrap CI** 才能支撑"显著改善"。
6. **待验证**：
   - a) teacher 2024 条里 THUNDERCLAP / SWORD_BOOMERANG 的正例比例；直接检查数据集 card 分布。
   - b) teacher 集合里 skip 作为正例的比例是否偏低，导致 skip logit 整体下降。
   - c) 再跑 5-10 iter 看 deck size 扩张是否继续 — 若继续扩大，可能是在过拟合"多拿就对"。
