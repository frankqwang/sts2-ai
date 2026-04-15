# Full-Turn Tactical+Mechanism Combat Teacher v1

## 目标

v1 先不改网络结构，只把“完整回合牌序老师”闭环打通：

- 用 beam full-turn solver 产出当前回合的最佳 action line。
- 用可配置 `tactical + mechanism + continuation + rule` 分数替代不稳定的 NN leaf value。
- 把完整牌序拆成 prefix samples，让模型学习 `S0 -> A1`、`S1 -> A2` 这种逐步出牌决策。
- 每次生成数据输出 `teacher_eval.json` 和 `teacher_eval.md`，方便后续按权重和 motif 调参。

## 运行入口

Teacher 配置：

```powershell
STS2AI/Python/configs/combat_turn_teacher_tactical_v1.toml
```

生成数据示例：

```powershell
python STS2AI/Python/search/build_act1_combat_teacher_v2_dataset.py `
  --teacher-config STS2AI/Python/configs/combat_turn_teacher_tactical_v1.toml `
  --output STS2AI/Artifacts/combat_teacher/ironclad_act1_tactical_teacher_v1.jsonl `
  --eval-output-dir STS2AI/Artifacts/combat_teacher/ironclad_act1_tactical_teacher_v1_eval
```

短训 smoke 配置：

```powershell
python STS2AI/Python/train_hybrid.py `
  --config STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_tactical_teacher_smoke.toml
```

## Scoring 结构

最终路线分：

```text
route_score = tactical_score + mechanism_score + continuation_score + rule_bonus
```

`tactical_score` 直接看当前回合短期收益：

```text
damage_progress_weight * enemy_damage_progress
- hp_loss_weight * expected_hp_loss_ratio
```

`mechanism_score` 用规则代理跨回合机制价值，例如 boss/elite 前几回合的 power、易伤前置、X-cost 先打、Body Slam 与格挡顺序等。低血、高敌方压力、敌人快死时会通过 safety gate 压低长期收益。

`continuation_score` v1 不引入新 head，先用规则代理：玩家 HP 安全度、敌方本回合压力、已打出的长期机制价值。

所有分项都会写进样本的 `leaf_breakdown`，评估报告会聚合均值和分布。

## Prefix 样本

默认 `emit_prefix_samples = true` 且 `rerun_solver_per_prefix = true`。也就是说：

1. root 局面搜索出完整最佳 line。
2. 写入 root teacher sample。
3. 按第一张牌真实 `act` 到下一个 prefix 局面。
4. 在新局面重新 solve，再写一条 prefix sample。
5. 直到 end_turn、战斗结束、unsupported、或达到动作上限。

这样模型学到的是每一步真实看到的新手牌、新能量、新抽牌结果之后应该怎么选，而不是只背 root line。

## 评估报告

builder 默认输出：

- `teacher_eval.json`
- `teacher_eval.md`

当前 v1 报告包含：

- solver 支持率；
- root / prefix 样本数量；
- 每个 root 平均 prefix 数；
- baseline best 与 teacher best 分歧率；
- baseline regret 均值；
- 平均 line 长度；
- 平均搜索节点、leaf 数、cache 命中；
- score breakdown 聚合；
- motif 覆盖。

训练后效果评估仍走现有 `train_hybrid.py` offline combat teacher loss。重点观察：

- `combat_teacher_ce` 是否下降；
- prefix action accuracy 是否上升；
- ranking/pairwise loss 是否下降；
- 固定 seed 胜率、平均掉血是否不变差。

## v1 边界

- v1 不新增 tactical/mechanism 网络 head。
- v1 不改 `src`。
- 机制牌感知先走可解释规则和配置权重，确认 teacher 信号有效后再进入 v2 head 训练。
- 当前 solver 仍遵守已有 supported action / unsupported card tag 约束，复杂抽牌、随机、弃牌等机制不会被强行纳入搜索。
