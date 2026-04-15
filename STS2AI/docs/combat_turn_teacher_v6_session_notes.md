# Combat Turn Teacher —— v6 session 总结（2026-04-15 夜）

## 最终结果（24-seed regression）

| 版本 | win rate (24-seed) | avg_floor | combats_won | boss_hp% | 备注 |
|---|---:|---:|---:|---:|---|
| baseline (retrieval_final_iter2175.pt) | 45.83% (11/24) | 14.1 | 7.42 | 0.278 | 原始起点 |
| mixed_w02_5iter | 54.17% (13/24) | 13.8 | 7.79 | 0.408 | 上一个 session 最佳 |
| mixed_v2 / v3 | 58.33% → 58.33% (12-seed 口径) / **24-seed 未跑** | — | — | — | 失败方向（live builder 数据污染） |
| **mixed_v4** | 58.33% (14/24) | 14.5 | 8.25 | 0.440 | 第一次突破 w02 |
| mixed_v5 | 54.17% (13/24) | 14.3 | 8.71 | 0.382 | replay builder bug → boss 样本丢失 → 退化 |
| **mixed_v6** | **62.50% (15/24)** | **14.9** | **9.83** | 0.444 | **本 session 新纪录** |

相对 baseline：**+16.7 百分点 win**；相对上一轮 w02：**+8.3 百分点**。

分段看更有意思（因为 w02 和 v4 都有"seed 区间偏科"现象）：

| 区间 | baseline | w02 | v4 | v5 | **v6** |
|---|---:|---:|---:|---:|---:|
| EVAL_001-012 | 5/12 | 8/12 | 8/12 | 6/12 | 7/12 |
| EVAL_013-024 | 6/12 | 5/12 | 6/12 | 7/12 | **8/12** |

`w02` 在 13-24 区间比 baseline 还差，是 **floor-2 样本过拟合**；
`v6` 在 13-24 区间回到 8/12，说明扩 WATERFALL_GIANT 样本有效。

## 关键 bug 与修复：replay builder 提前 break

`build_combat_teacher_from_trajectory.py` 原实现：

```python
idx = _match_action_index(legal_actions, chosen_action)
if idx < 0:
    replay_errors += 1
    break   # <-- 一次 match 失败就放弃整条 trajectory
```

后果：EVAL_018/021/022/024 的 trajectory 里各有数十个 WATERFALL_GIANT boss
combat state，但只要在前面某个 noncombat 步（比如 card_reward 选牌）
`sanitize_action` 后对不上，就 break，boss 样本全部丢失。v5 数据里
WATERFALL_GIANT boss 样本 **0 条**，结果和 v4 没有差别甚至更差。

修复：

```python
if idx < 0:
    replay_errors += 1
    idx = 0  # 退化为选 legal[0]（通常是 proceed / first_legal），不 break
```

修复后 EVAL_018 从 6 samples → 21 samples，EVAL_022 从 19 → 31，EVAL_024 从
13 → 25。最关键的是**抓到了 33 条 WATERFALL_GIANT boss 样本**（此前是 0）。

## 数据集演化路径

| 版本 | 样本数 | floor 构成 |
|---|---:|---|
| mixed (w02 用的) | 219 | floor 2: 171 / 11-13: 48 / **high floor: 0** |
| mixed_v4 | 321 | + replay (EVAL_001-006，6 seed)：floor 14-17 高 floor 102 条 |
| mixed_v5 | 467 | + replay_ext v1（EVAL_007-024，但 boss 漏采）：boss 11 条，仍 SOUL_FYSH only |
| **mixed_v6** | **502** | + replay_ext **修复后**：boss 84 条，覆盖 SOUL_FYSH + **WATERFALL_GIANT** 两个 boss |

v6 dataset: `STS2AI/Artifacts/combat_teacher/tactical_v1_mixed_v6_20260415_213719/ironclad_act1_tactical_teacher_v1_mixed_v6.jsonl`

Boss 样本分 boss：

| boss | 样本数 | 覆盖 seed |
|---|---:|---|
| SOUL_FYSH | 51 | EVAL_001, 002, 003, 005, 009, 015 |
| WATERFALL_GIANT | 33 | EVAL_018, 022, 024 |
| HULK_MATRIARCH | **0** | — 仍缺 |

## 具体战斗差异挖掘（方法：抓 baseline / w02 / v4 / v6 同一 seed 的 trajectory，按 floor×enemy 对齐）

### 例 1：EVAL_008（baseline/w02/v4 都死在 floor 9，v6 走到 boss）

```
floor 5 CORPSE_SLUG×2:   base 14dmg, w02 16, v4 16, v6 9    ← v6 省 5-7 HP
floor 8 TWO_TAILED_RAT×3: base 29dmg, w02 25, v4 19, v6 16  ← v6 省 3-13 HP
floor 9 GREMLIN_MERC:
  base 进场 6HP → 死
  w02  进场 2HP → 死
  v4   进场 15HP → 死
  v6   进场 41HP → 赢（10dmg）       ← 前面省 HP 直接决定存活
```

v6 在 floor 5-8 普通战斗里每场少挨 3-7 HP，累积到 floor 9 就多 26 HP
buffer，足以过 elite。

### 例 2：EVAL_013（v4 到 floor 15 死，v6 到 boss）

```
floor 12 CALCIFIED_CULTIST,TOADPOLE:
  base 19dmg → 死
  w02  [没到]
  v4   9dmg 剩 39HP → 赢
  v6   2dmg 剩 31HP → 赢              ← v6 最省 HP

floor 15 CORPSE_SLUG×3:
  v4   8HP 进 → 死
  v6   14HP 进 → 赢 (6dmg)             ← v6 策略更保守，留血到 boss

floor 17 boss WATERFALL_GIANT:
  v4   [没到]
  v6   32HP 进 → 挨 32dmg 死           ← 仍不够，但至少到 boss
```

### 例 3：EVAL_022（v4 到 boss 但死，v6 boss 赢）

```
floor 17 boss WATERFALL_GIANT[250HP]:
  w02  51HP 进 → 39dmg 剩 12HP 赢（4 张牌收）
  v4    9HP 进 → 9dmg 死
  v6   51HP 进 → 27dmg 剩 24HP 赢（5 张牌收）   ← v6 和 w02 同水平
```

### 例 4：EVAL_018（所有版本都到 boss 都死，v6 boss 战斗更持久）

```
floor 17 boss WATERFALL_GIANT:
  base 13HP 进，打 2 张牌死
  w02   7HP 进，打 1 张牌死
  v4    2HP 进，打 6 张牌死
  v6   88HP 进，打 43 张牌死          ← v6 扛了 21 轮才倒
```

这 case 说明 v6 的 WATERFALL_GIANT 对战是**质变**，之前都是开局即死。

### 反例：EVAL_003（v4 到 floor 20，v6 退到 15）

v4 能深入 Act 2 但 v6 反而退回。原因追到 floor 15 CALCIFIED+DAMP：
- v4 进场 54HP → 回血 4HP，轻松过
- v6 进场 6HP → 死

v6 在 floor 12-14 的 GREMLIN_MERC / PUNCH_CONSTRUCT 上用了太激进的打法，
HP 从 41→32→19→25→6，最后在 floor 15 暴毙。这是 **v6 的小倒退**：
牺牲了一些对已经拿到 boss 的 seed 的稳定性，换来了对更多 seed 到 boss 的改进。

## Session 主线回顾

1. **先做了扩大评估（12→24 seed）**：发现 12 seed 样本太小（1 个 seed = 8 pt），
   扩到 24 seed 确认 v4 58.3% > w02 54.2%。
2. **sanity check 挖 combat 细节**：抓 baseline/w02/v4 6 seed 的 trajectory，
   按 floor 对齐看 dmg_in，确认 v4 是**通过"floor 8-15 前期保血"间接加强 boss 战斗**
   （进 boss 时 HP 高 40+）；反之 v4 在 HP 低时有时做激进决策（BLOODLETTING）导致
   EVAL_006 退化。
3. **尝试 v5 失败**：直接扩数据但 replay builder 有 break bug，WATERFALL_GIANT 0 条
   boss 样本被漏采，中段 monster 样本稀释 boss 信号，v5 掉回 54.2%。
4. **定位 bug + 修 replay builder**：把 "match 失败→break" 改成 "match 失败→选 legal[0]
   继续"。重跑：EVAL_018/022/024 终于有 WATERFALL_GIANT boss 样本各 10-12 条。
5. **v6 训练 + 24-seed 评估 → 62.5% 新高**。

## 下一步假设

基于本 session 的数据，有几个可以继续尝试的方向：

1. **补 HULK_MATRIARCH 样本**：扩 trajectory dump 到 EVAL_025..EVAL_050，
   筛走 HULK_MATRIARCH boss 的 seed（每次 dump ~2 分钟/seed，跑背景就行）。
   Act1 三 boss 各一个样本池，预期能进一步提升。
2. **补 crisis-management 样本**：v4 的 EVAL_006 / v6 的 EVAL_003 都暴露了"低 HP
   时做激进决策"的问题。solver 的评分里应该已经考虑，但数据分布里缺低 HP 场景。
   可以专门跑 baseline 或弱 checkpoint 的 trajectory（baseline 会被打得更惨、留更多
   低 HP state），然后只挑 start_hp ≤ 30% 的 combat state 挂 solver。
3. **训练时 iter 延长 + 数据权重**：v6 数据量 502 条比 v4 的 321 大 56%，
   5 iter × 8 episodes × 8 updates 可能还没收敛。试 10 iter 看是否再提升。
4. **把 floor 2 样本降权或裁剪**：mixed 原始 171 条 floor 2 占 34%，即使在 v6 也是
   最大单一 bucket。做一版 "mixed_trim" 砍到 80 条 floor 2 + 全部 high-floor，看是
   否能让 boss 数据信号更强。

## 产物清单

- 代码：
  - `STS2AI/Python/search/build_combat_teacher_from_trajectory.py`（本 session fix 了 break bug）
  - `STS2AI/Python/search/merge_combat_teacher_datasets.py`
  - `STS2AI/Python/diagnostics/compare_trajectory_combats.py`（3-way combat diff）
  - `STS2AI/Python/diagnostics/compare_4way_combats.py`（本 session 新增，4-way diff）
- 24-seed regression eval：`STS2AI/Artifacts/combat_teacher/eval24_20260415_211034/`
  - baseline/w02/v4/v5/v6 eval json 都在里面
- trajectory 深挖：`STS2AI/Artifacts/combat_teacher/dive_20260415_211920/`
  - `combat_diff.txt`（baseline/w02/v4，6 seed）
  - `v6_combat_diff.txt`（baseline/w02/v4/v6，6 seed）
  - `key_findings.md`（v4 vs w02 文字分析）
- 数据集：
  - mixed_v4: `STS2AI/Artifacts/combat_teacher/tactical_v1_mixed_v4_20260415_203858/`
  - mixed_v5: `STS2AI/Artifacts/combat_teacher/tactical_v1_mixed_v5_20260415_212650/`
  - **mixed_v6**: `STS2AI/Artifacts/combat_teacher/tactical_v1_mixed_v6_20260415_213719/`
- trajectory 原始 dump：
  - `STS2AI/Artifacts/combat_teacher/trajectory_20260415_203343/`（EVAL_001-006，w02）
  - `STS2AI/Artifacts/combat_teacher/trajectory_ext_20260415_211023/`（EVAL_007-024，w02）
- replay 中间结果：
  - `tactical_v1_replay_20260415_203730`（EVAL_001-006）
  - `tactical_v1_replay_ext_20260415_211633`（bug 版，v5 用的）
  - `tactical_v1_replay_ext_v2_20260415_213324`（修复版，v6 用的）
- checkpoint：
  - v4: `STS2AI/Artifacts/hybrid_training_tactical_teacher_mixed_v4_20260415_203914/*/hybrid_final.pt`
  - v5: `STS2AI/Artifacts/hybrid_training_tactical_teacher_mixed_v5_20260415_212726/*/hybrid_final.pt`
  - **v6**: `STS2AI/Artifacts/hybrid_training_tactical_teacher_mixed_v6_20260415_213737/*/hybrid_final.pt`
