# 训练下一步交接

更新时间：2026-04-28

## 当前判断

当前最短路径不是先训练 planner LoRA，而是先把非战斗选卡和战斗 prompt 信息补齐：

1. 非战斗选卡直接影响 boss 前卡组质量，当前线上 adapter 仍是旧 v2b，`card_reward` 只有 2000 条。
2. 新数据集已把 `card_reward` 扩到 10000 条，并补入 boss、未来路线、近期掉血、下一层风险、胜利终局参考。
3. `reason/plan/action_scores` 仍需要 Kimi 标注一批高质量样本，替换规则合成理由。
4. 战斗侧需要先有稳定 strategy context 和怪物/卡组机制说明，再考虑 planner LoRA。

## 当前可用数据

非战斗基线数据集：

```text
STS2AI/Artifacts/llm/datasets/skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428
```

分布：

```text
total: 18000
card_reward: 10000
map_choice: 2000
relic_select: 2000
rest_site_choice: 2000
shop_choice: 2000
```

说明：

- 保留游戏原始占位符文案，例如 `{MaxHp}`、`{Energy:energyIcons()}`。
- `current_plan` 只基于当前卡组。
- `winning_outcome_reference` 单独提供胜利局最终构筑方向，作为参考，不混入当前计划。
- `boss`、`route_ahead`、`recent_combats`、`next_risk` 已加入 prompt。

当前线上非战斗 adapter 仍是旧版：

```text
STS2AI/Artifacts/llm/sft/non_combat_skada_ironclad_v01032_2k_v2b_20260426/adapter
```

不要把旧 v2d/v2e/v2f/v2g 当正式训练输入；当前应使用 `v2h_fulltext_placeholders`。

## Kimi 选卡标注

脚本：

```text
STS2AI/data/skada/kimi_label_card_rewards.py
```

用途：

- 从 `v2h_fulltext_placeholders` 抽高价值 `card_reward` 样本。
- 让 Kimi 基于完整上下文标注 `best_action_index`。
- 输出简短 `plan_zh`、`reason_zh`，每个不超过 200 中文字。
- `action_scores` 覆盖全部候选动作，每个负例要写具体原因。

先干跑：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python STS2AI\data\skada\kimi_label_card_rewards.py `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428 `
  --out-dir STS2AI\Artifacts\llm\datasets\skada_card_reward_kimi_dryrun_v2h_fulltext_placeholders_20260428 `
  --limit 4 `
  --group-size 2 `
  --dry-run
```

真实调用前设置环境变量，不要把 key 写入命令或日志：

```powershell
$env:MOONSHOT_API_KEY="<secret>"
```

建议第一批：

```powershell
python STS2AI\data\skada\kimi_label_card_rewards.py `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428 `
  --out-dir STS2AI\Artifacts\llm\datasets\skada_card_reward_kimi_200_v2h_20260428 `
  --limit 200 `
  --group-size 4 `
  --max-api-calls 60 `
  --min-confidence 0.65
```

检查输出：

```text
summary.json
labels.jsonl
invalid.jsonl
train.jsonl
eval.jsonl
```

如果 `invalid` 很多，先抽样看 `invalid.jsonl`，不要直接训练。

## 非战斗训练优先级

优先级 P0：

- 用 `v2h_fulltext_placeholders` 训练新 non-combat LoRA。
- 如果 Kimi 200 条已完成，则把 Kimi `train/eval.jsonl` 作为 gold 混入训练集，比例先控制在 10% 到 20%。
- 训练后先做离线抽样和 fullrun 小评估，再替换线上 non-combat adapter。

建议训练参数：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
C:\Users\Administrator\Desktop\sts2Zero\STS2AI\llm\.venv311\Scripts\python.exe `
  STS2AI\llm\training\sft_lora.py `
  --run-name non_combat_skada_ironclad_v01032_card10k_v2h_20260428 `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428 `
  --max-seq-length 1536 `
  --batch-size 1 `
  --grad-accum 8 `
  --num-epochs 1 `
  --load-in-4bit
```

注意：Windows 下 Unsloth 输出可能有编码问题，必须设置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。

## 战斗侧优先级

优先级 P0：补 strategy context，不先训 planner LoRA。

目标 prompt 字段：

```text
combat_strategy:
  deck_plan
  encounter_guide
  kill_order
  turn_policy
  potion_policy
  danger_notes
```

原则：

- 人类能在游戏内看到或从机制理解的信息必须给模型。
- 怪物 intent、power、relic、potion、card 的可见说明必须完整。
- 对复杂 boss 和机制怪写短攻略，优先覆盖当前 fullrun 常见失败点。

优先级 P1：Kimi 复盘整场战斗和掉血回合。

- 按“整回合掉血”抽 hardcase，不只看单动作。
- 输出 turn plan、错误原因、可训练动作。
- 进入 dataset pool 的 `gold` 或 `hardcase`。

优先级 P2：planner LoRA。

planner LoRA 不应第一步做。等 strategy context 和 Kimi turn labels 稳定后，再训练 `turn_plan`。

planner 输出示例：

```json
{
  "sequence": [
    {"action":"play_card","card_id":"BASH","target_id":0},
    {"action":"play_card","card_id":"PERFECTED_STRIKE","target_id":0},
    {"action":"end_turn"}
  ],
  "reason": "先易伤再爆发，当前回合不需要额外防御。"
}
```

执行约束：

- 不能批量盲执行旧 `action_index`。
- 每一步都必须重新取状态并映射当前 `legal_actions`。
- 任何一步进入选择 UI、目标死亡、能量异常、敌人阶段变化，都停止 sequence 并重新问模型。

## 推荐顺序

1. 跑 Kimi 选卡 200 条，检查 `labels/invalid`。
2. 训练新 non-combat LoRA，至少跑一把 fullrun 小评估。
3. 如果选卡改善明显，扩大 Kimi card_reward 到 1000 条。
4. 同时补战斗 `combat_strategy` builder 和怪物机制 guide。
5. 用新 prompt 跑 combat rollout，继续收集掉血回合 hardcase。
6. Kimi 复盘高损回合，进入 dataset pool gold。
7. 当整回合 turn labels 足够稳定，再启动 planner LoRA 实验。

## 接手注意

- 不要使用 `v2d/v2e/v2f/v2g` 做正式训练。
- `v2h_fulltext_placeholders` 是当前非战斗基线。
- 当前 Kimi 脚本不会把 API key 写入产物。
- 未完成真实 Kimi 调用，因为当前环境没有 `MOONSHOT_API_KEY`。
- 训练产物和数据都在 `STS2AI/Artifacts/llm`，不要把临时交接文件放仓库根目录。
