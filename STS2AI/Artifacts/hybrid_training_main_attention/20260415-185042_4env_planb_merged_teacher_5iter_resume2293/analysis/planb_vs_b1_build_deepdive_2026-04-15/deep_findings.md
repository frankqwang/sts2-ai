# Plan B vs B.1 Build & 决策 深度挖掘（iter 2294-2298, 各 2500 ep）

资源参考：`stats.json` / `extra_stats.json` / `rest_event_fixed.json`（本目录）；输入 replays 目录见 **项目背景**。两 run 均 resume 自 `hybrid_02293.pt`，唯一差异 = teacher 数据（B.1 用 274 条，Plan B 用 274+1750 = 2024 条 RANK-expanded teacher）。B.1 n=2500 clear=76 / reach=1185 / other=1239；Plan B n=2500 clear=85 / reach=1265 / other=1150（clear +1.18pp，reach-or-clear 由 50.44% → 54.00%，+3.56pp，复现上轮结论）。

---

## D1. Outcome 分层 build 对比（clear / reach / other）

| run | outcome | n | deck_size | remove/run | relics(end) | potions(end) | skip/run | shop_card | shop_potion |
|---|---|---|---|---|---|---|---|---|---|
| B.1 | clear | 76 | **14.46** | **2.38** | 2.88 | 0.46 | **1.57** | 0.342 | 0.132 |
| B.1 | reach | 1185 | 13.65 | 1.75 | 2.88 | 0.53 | 1.15 | 0.402 | 0.133 |
| B.1 | other | 1239 | 12.66 | 1.10 | 2.63 | 0.44 | 0.74 | 0.212 | 0.078 |
| Plan B | clear | 85 | **14.59** | **1.85** | 2.93 | 0.39 | **1.13** | 0.400 | 0.176 |
| Plan B | reach | 1265 | 14.00 | 1.70 | 2.86 | 0.59 | 0.88 | 0.389 | 0.129 |
| Plan B | other | 1150 | 12.75 | 1.12 | 2.54 | 0.42 | 0.69 | 0.198 | 0.086 |

**关键反直觉**：
1. **clear 的 B.1 比 Plan B 删卡多 28.8%（2.38 vs 1.85）、skip 多 38.7%（1.57 vs 1.13）**——B.1 的 clear run build 质量更"精"。Plan B clear 虽然净卡数近似，但是"多拿多带"策略（deck +0.13 而 remove -0.53），**这是上轮"skip rate ↓、磨过中段"结论在 clear 子集上的精确化**。
2. reach / other 两 run 的 end-state relics & potions 差距都 < 0.05 一瓶药水，属 seed 噪声。Plan B 只在 **reach 子集 potions 0.59 vs 0.53 (+0.06)** 有 subtle 提升，说明 Plan B 中期活得更久 → 多拿了一瓶药。
3. `shop_relic` 几乎为 0（B.1 4/2500, Plan B 0/2500）—— 两 run 都**几乎从不买 relic**，目前 Ironclad shop 遗物几乎没被学到。

## D2. Boss-specific clear

`final_floor=17 + boss_reached=True` 记录到的 boss 只有 2 种：`WATERFALL_GIANT` 与 `SOUL_FYSH`，**没一个 run 遇到 LAGAVULIN**（0/2500 × 2）。这是 act1 路线生成 / 过滤时的已知约束，不是本次差异原因。

| boss | B.1 clear/reach | Plan B clear/reach | 备注 |
|---|---|---|---|
| WATERFALL_GIANT | 16/417 = 3.8% | 13/457 = 2.8% | **Plan B 退步** |
| SOUL_FYSH | 50/448 = 11.2% | 58/447 = 13.0% | Plan B 胜率 +1.8pp |
| LAGAVULIN | 0/0 | 0/0 | 未遭遇 |

SOUL_FYSH 是 Plan B 所有 +9 clear 的来源——且 PlanB 击败 WATERFALL_GIANT 能力**反而下降 1pp**（16→13 虽是小样本，但 reach 从 417→457 +40 场仍只多 -3 次 clear，说明 B.1 对 WATERFALL_GIANT 更懂）。Pre-boss HP 两 run 基本持平（WG: 80.56 vs 78；SF: 76.16 vs 78.69）。relics 在 WG 子集 B.1 2.63 → Plan B 3.23，Plan B 多半瓶遗物但没转成 clear，证实"多资源 ≠ 会用"。

## D3. Shop + 删卡

- 全集 shop-card 购买：B.1=765, Plan B=754（基本持平）。TOP 购卡均为 ANGER / BLOODLETTING / CINDER / BREAKTHROUGH / TWIN_STRIKE。**最大差异：B.1 买 ANGER 98 次，Plan B 81 次（-17, -17%）**；Plan B 多买 BREAKTHROUGH（34→53 +56%）与 IRON_WAVE（21→32 +52%）。
- Potion 购买：两 run 次数 265 / 277 近似，**偏好显著不同**。B.1 TOP3 = SPEED(26) / STRENGTH(21) / SKILL(20)；Plan B TOP3 = VULNERABLE(20) / SKILL(20) / FIRE(20)。**Plan B 偏向 debuff-on-enemy 药（VULNERABLE/WEAK），B.1 偏自强/位移（SPEED/STRENGTH）**。这个 profile 改变可能是 RANK-B2 teacher 在 shop_potion 场景的投票倾向被学进来的信号。
- 删卡（remove）总数 B.1=3614, Plan B=3589，几乎一致。clear 子集 remove/run 差异（2.38 vs 1.85）是**卡池长度决定**——B.1 clear 的 deck 更紧，所以有更多"能删/愿删"的机会。

## D4. 路线偏好

两 run 几乎**完全一致**（这本身是 sanity check：avoid_elite 与 force_rest 等硬规则都在生效）：

| node | B.1 % | Plan B % |
|---|---|---|
| monster | 41.0 | 40.7 |
| unknown (event) | 25.3 | 25.3 |
| rest_site | 15.6 | 15.8 |
| shop | 7.9 | 7.6 |
| treasure | 6.1 | 6.1 |
| boss | 3.9 | 4.1 |
| **elite** | **0.4** | **0.3** |

elite 硬规则 100% 生效（<1%，几乎全是"绕不开"的终点 elite）。差异全在 ±0.3pp，属于地图生成噪声。

## D5. Event 决策

共 9404（B.1）/ 9373（Plan B）条 event_choice。TOP5 event 与出现次数两 run 几乎持平：NEOW / ABYSSAL_BATHS / PUNCH_OFF / ROOM_FULL_OF_CHEESE / TRASH_HEAP。

**关键差异：NEOW 礼物偏好发生显著漂移**。
- B.1 TOP4: SCROLL_BOXES(224) / CURSED_PEARL(210) / PRECARIOUS_SHEARS(209) / LARGE_CAPSULE(208)
- Plan B TOP4: POMANDER(173) / ARCANE_SCROLL(171) / BOOMING_CONCH(166) / SILVER_CRUCIBLE(164)

这是两个完全不同的 reward bundle 组合——NEOW 在 teacher 数据里出现频率极高，RANK-B2 扩大后其"最优选项"的学习目标变了。没有哪一组明显更强（都是 Act1 不同流派），但**说明 teacher 数据扩大的主要作用点在 non-combat 选项层，影响符合预期**。

其他 event（ABYSSAL_BATHS / PUNCH_OFF / ROOM_FULL_OF_CHEESE / TRASH_HEAP）两 run 选项分布差异 < 10%。ABYSSAL_BATHS 的 LINGER 选项 B.1=154 → Plan B=187 (+21%)，是本组少数有方向性差异的事件。

## D6. Rest site

| label | B.1 | Plan B |
|---|---|---|
| rest | 4857 | 5027 |
| smith（升级） | 191 | 169 |
| dig/lift | 1/0 | 1/1 |
| proceed | 5049 | 5198 |

rest:smith ≈ 25:1 两 run 基本一致。**Plan B smith 次数 -22（-11.5%）**——reach+clear 场次多但升级行为反而略减，和 D1 观察"Plan B 倾向拿更多卡不精修"一致。

## D7. Combat

- mean_combats B.1 = 6.58, Plan B = 6.67 (+1.4%)；won 5.59 → 5.68 (+1.6%)，与 reach 率提升对应。
- `death_enemy` 字段在 summary.json **恒为 null**（环境 bug），用 `last combat enemy_group` 代替。非 clear 死亡敌人分布两 run 无实质差异。
- **死亡楼层分布**（n≈2420/run）：early(f<7) 4.1% vs 3.7%，mid(f7-13) 34.2% vs 32.1%（Plan B **-2.1pp，最大改善处**），pre-boss(f14-16) 12.8% vs 11.8%，boss(f17) 48.9% vs 52.4%（Plan B **+3.5pp**）。**mid 段存活率提升直接转化为 boss 层死亡增多** —— Plan B 把 "中段死掉" 的 run 推到 boss 前，但 clear 转化没跟上，印证"skip↓多拿过关"机制。

## D8. 其他发现

1. **card_rewards source** 100% 为 `boss_card_guidance_keep`（B.1: 13984；Plan B: 14207）——guidance_weight=0.8 的软偏置全局生效，teacher 数据没改变 decision pipeline 结构。
2. **ANGER(S) 继续退化**：clear 子集 1.092 → 0.929 (-14.9%)；reach-only 0.554 → 0.557 持平；全集 shop-bought 98 → 81。**ANGER 是 B.1 胜利 build 的核心 strike 引擎，Plan B 整体淡化它**。
3. **Plan B clear 子集**：INFERNO +0.075 (0.066→0.141), ASHEN_STRIKE +0.102 (0.039→0.141), CASCADE +0.068 (0.026→0.094), ONE_TWO_PUNCH +0.071 (0→0.071), COLOSSUS +0.059 (0→0.059)——**火/残烬流派浮现**。同时 OFFERING -0.121 / HEADBUTT -0.067 / MOLTEN_FIST -0.048 / BLOOD_WALL -0.044。
4. **首卡**（cards_taken[0]）：B.1 clear 中 ANGER 22/76 (28.9%)；Plan B clear 中 ANGER 只 18/85 (21.2%)，首卡池分散化。
5. **clear 里高频双卡组合**：B.1 TOP1 = `ANGER+CINDER (16)`；Plan B TOP1 = `ANGER+SETUP_STRIKE (16)`。两 run 的"clear 联合"都以 ANGER 为锚，但 Plan B 的"ANGER+THUNDERCLAP" 上升到 12（B.1=10）——THUNDERCLAP(F) +14% 的来源。
6. **Iter trend（clear%）**：B.1 iter 2294-2298 = 3.8 / 4.8 / 2.4 / 2.6 / 1.6（**单调下降**）；Plan B = 2.4 / 4.2 / 4.2 / 3.0 / 3.2（更稳定）。**B.1 后期 clear 掉到 1.6%**，是不是 batch seed 噪声需要再看；但确实提示 Plan B 的 clear 分布更平坦，不是 cherry-picked。

---

## 结论速览

- Plan B 的 clear +1.18pp / reach +3.56pp 是**实在的存活改善，但质量下滑**：clear build 平均删卡少、skip 少、ANGER 少、火流派替代，说明 teacher 扩数据鼓励了"多拿卡"偏好而非"筛好卡"。
- 唯一的结构性胜利：mid 段(f7-13) 死亡率 -2.1pp，可归因 Plan B 的 pot 使用增加（reach-only subset +0.06）+ 更杂食的 build 减少了 "早期 miss 关键卡导致暴死"。
- 反直觉信号：(a) WATERFALL_GIANT clear 从 16→13 下降；(b) smith 使用 -11.5%；(c) ANGER 全方位下降；(d) NEOW 首选完全漂移。**下一步若要 scale RANK teacher，应对 NEOW 与 ANGER 选卡类 sample 做 quality filter**，避免数据扩大反稀释核心强项。
