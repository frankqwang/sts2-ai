# ONCRVIS_00000 短路线搜索分析

## 背景

- 数据目录：[20260414_cardreward_route_smoke_vis](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/offline_noncombat_ranking/20260414_cardreward_route_smoke_vis:1)
- 汇总进度：[progress.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/offline_noncombat_ranking/20260414_cardreward_route_smoke_vis/progress.json:1)
- 可视化报告：[branch_report_ONCRVIS_00000.md](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/offline_noncombat_ranking/20260414_cardreward_route_smoke_vis/branch_report_ONCRVIS_00000.md:1)
- 原始分支日志：[raw_branch_rollout.jsonl](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/offline_noncombat_ranking/20260414_cardreward_route_smoke_vis/raw/raw_branch_rollout.jsonl:1)

本次配置口径：

- 只采 `card_reward`
- `label_mode = reward_tree`
- `rollout_goal = terminal`
- `tree_route_search = true`
- `tree_max_reward_depth = 3`
- `tree_beam_width = 2`

最终整局结果：

- `boss_reached = true`
- `end_floor = 20`
- `outcome = defeat`

这表示该路线已经通过 `Act1 boss` 并进入 `Act2`，最后在 `floor 20` 失败。

## 搜索路径

短路线搜索最终实际选择链是：

1. `floor 2` 选 `VICIOUS`
2. `floor 3` 选 `HAVOC`
3. `floor 4` 选 `BLUDGEON`
4. `floor 8` 选 `BLOODLETTING`
5. `floor 11` 选 `SKIP`
6. `floor 13` 选 `SKIP`
7. `floor 15` 选 `STOMP`
8. `floor 17` 选 `TEAR_ASUNDER`
9. `floor 18` 选 `INFERNAL_BLADE`
10. `floor 19` 选 `HOWL_FROM_BEYOND`

## 阶段判断

可以把这条路线分成 4 段看。

### 1. 早期爆发构筑

`floor 2 -> 4` 连续选择：

- `VICIOUS`
- `HAVOC`
- `BLUDGEON`

这段搜索明显偏向“尽快把终局 ceiling 拉高”，而不是保守保血。对应分数里：

- `floor 2` 的 `VICIOUS = 1.3583`，高于 `SKIP = 1.3470`
- `floor 3` 的 `HAVOC = 1.3791`，高于 `THUNDERCLAP = 1.3676`
- `floor 4` 的 `BLUDGEON = 1.3900`，显著高于 `RAMPAGE = 1.2986`

这说明 route search 在早期更愿意为后续 boss reach 去拿高 swing 牌。

### 2. 中段资源再压榨

`floor 8` 选 `BLOODLETTING`，比：

- `PERFECTED_STRIKE = 1.3839`
- `RAMPAGE = 1.3771`
- `SKIP = 1.3827`

都略高，最终拿到了 `1.3905`。

这一步的信号很弱，是典型“多个分支都能到 boss，但其中一条终局稍高”的情况。它不是压倒性最优，而是 route search 在细小终局差异里偏向了更激进的一条。

### 3. Boss 前两次主动 `SKIP`

这是这条 seed 里最有价值的点。

`floor 11`：

- `INFLAME = 1.3467`
- `TRUE_GRIT = 1.3075`
- `BLOODLETTING = 1.2617`
- `SKIP = 1.3756`

`floor 13`：

- `TWIN_STRIKE = 1.3075`
- `TREMBLE = 1.1550`
- `THUNDERCLAP = 1.1550`
- `SKIP = 1.3497`

这里 route search 没有继续无脑加牌，而是在 boss 前两次都判断 `SKIP` 最优。

这说明短路线搜索不是“永远更贪拿牌”，而是在 build 到一定形态后，会主动压缩牌组厚度。

### 4. 进 Act2 后的信号塌缩

`floor 15 / 17 / 18 / 19` 的分数大量并列：

- `floor 15` 四个选项全部 `1.2762`
- `floor 17` 四个选项全部 `1.4467`
- `floor 19` 四个选项全部 `1.4467`

这说明进入 `Act2` 后，当前评分函数的分辨率已经明显不够。

也就是说：

- route search 在 `Act1` 前中段有明显决策价值
- 但进入 `Act2` 后，很多分支被打成“都一样差”或“都一样好”
- 所以后半段选择虽然被记录了，但不应过度解读

## 这条 seed 的真正收获

这条 seed 说明两件事。

### 1. 短路线搜索确实比单点贪心更像 build 规划器

它不是每次都拿当前分最高的一张功能牌，而是在：

- 前期连续堆高 ceiling
- 中段补资源转换
- boss 前主动 `SKIP`

这已经是“看组合”的行为，而不是单点贪心。

### 2. 当前终局评分在 `Act2` 仍然不够细

虽然 `rollout_goal = terminal` 已经修正了“只看两场战斗”的错误，但 `Act2` 后大量并列分数说明：

- 评分函数还不足以稳定区分“都能过 boss 但后续谁更强”
- route search 在深后期会退化成弱排序

因此这条链现在最适合用来指导：

- `Act1` 的 card reward build
- boss 前构筑决策

但还不适合作为“全程通关最优搜索器”。

## 下一步建议

1. 用当前同一配置跑 4 个 seed，对照 `end_floor / boss_reached / outcome`
2. 如果 4-seed 里有稳定增益，再扩大到更大的 seed 批次
3. 如果增益只出现在 `Act1`，后续应把 route search 的用途明确收敛为 `Act1 card_reward planner`
4. 若要继续提高后段质量，需要补 `Act2` 终局 tie-break，而不是继续盲目加深搜索
